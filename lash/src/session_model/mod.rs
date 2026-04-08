pub(crate) mod context;
pub use lash_sansio::session_model::message;
pub use lash_sansio::session_model::prompt;

use tokio::sync::mpsc;

use crate::ContextStrategy;
use crate::ExecutionMode;
use crate::PromptContext;
use crate::ToolDefinition;
use crate::llm::factory::adapter_for;
use crate::llm::types::{LlmEventSender, LlmStreamEvent, LlmToolSpec};
use crate::plugin::PromptContribution;
use crate::provider::Provider;
use crate::session::Session;

pub use lash_sansio::session_model::{
    DefaultPromptRenderer, DurableTurnSnapshot, ErrorEnvelope, LLM_MAX_RETRIES, LLM_RETRY_DELAYS,
    Message, MessageRole, Part, PartKind, PromptOverrideMode, PromptRenderer, PromptSectionName,
    PromptSectionOverride, PruneState, SessionEvent, TokenUsage, TurnTerminationPolicyState,
    append_line_segment, build_assistant_parts, default_prompt_renderer,
    format_tool_result_content, is_malformed_assistant_output, make_error_envelope,
    make_error_event, parse_fence_line, render_prompt, render_transcript_prompt,
    truncate_raw_error,
};

/// Send an event to the channel if it's still open.
pub(crate) async fn send_event(tx: &mpsc::Sender<SessionEvent>, event: SessionEvent) {
    if !tx.is_closed() {
        let _ = tx.send(event).await;
    }
}

/// Resolved session policy for a running session.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionPolicy {
    pub model: String,
    pub provider: Provider,
    pub max_context_tokens: Option<usize>,
    pub model_variant: Option<String>,
    pub session_id: Option<String>,
    pub max_turns: Option<usize>,
    pub execution_mode: ExecutionMode,
    pub context_strategy: ContextStrategy,
}

impl Default for SessionPolicy {
    fn default() -> Self {
        Self {
            model: "anthropic/claude-sonnet-4.6".to_string(),
            provider: Provider::OpenAiGeneric {
                api_key: String::new(),
                base_url: String::new(),
                options: crate::provider::ProviderOptions::default(),
            },
            max_context_tokens: None,
            model_variant: None,
            session_id: None,
            max_turns: None,
            execution_mode: crate::default_execution_mode(),
            context_strategy: crate::default_context_strategy(),
        }
    }
}

pub(crate) struct ExecutionPreamble {
    pub(crate) model: String,
    pub(crate) tool_specs: Vec<LlmToolSpec>,
    pub(crate) prompt: PromptContext,
}

pub(crate) fn transport_stream_events(
    provider: &Provider,
    requested: Option<tokio::sync::mpsc::UnboundedSender<LlmStreamEvent>>,
) -> Option<LlmEventSender> {
    if let Some(requested) = requested {
        return Some(make_stream_event_sender(requested));
    }

    let llm = adapter_for(provider);
    if llm.requires_streaming() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
        drop(rx);
        Some(make_stream_event_sender(tx))
    } else {
        None
    }
}

fn make_stream_event_sender(
    tx: tokio::sync::mpsc::UnboundedSender<LlmStreamEvent>,
) -> LlmEventSender {
    LlmEventSender::new(move |event| {
        let _ = tx.send(event);
    })
}

pub(crate) fn build_execution_preamble(
    session: &Session,
    policy: &SessionPolicy,
    mode: ExecutionMode,
    model: String,
) -> ExecutionPreamble {
    let session_id = policy.session_id.as_deref().unwrap_or("root");
    let surface = session.execution_surface(session_id, mode);
    let enabled_tools = surface.enabled_tools();
    let (tool_list, omitted_tool_count) = if matches!(mode, ExecutionMode::Repl) {
        let prompt_tools = surface.prompt_tools();
        let mut tool_list = ToolDefinition::format_tool_docs(&prompt_tools);
        let omitted_tool_count = count_prompt_omitted_tools(&enabled_tools);
        for note in &surface.tool_list_notes {
            tool_list.push_str("\n\n");
            tool_list.push_str(note);
        }
        (tool_list, omitted_tool_count)
    } else {
        (String::new(), 0)
    };
    let tool_specs = if enabled_tools.is_empty() {
        Vec::new()
    } else if matches!(mode, ExecutionMode::Standard) {
        lash_sansio::session_model::model_tool_specs(&enabled_tools)
    } else {
        Vec::new()
    };
    let tool_names: Vec<String> = enabled_tools.iter().map(|t| t.name.clone()).collect();
    let prompt = PromptContext {
        mode,
        tool_list,
        tool_names,
        omitted_tool_count,
        contributions: Vec::new(),
    };

    ExecutionPreamble {
        model,
        tool_specs,
        prompt,
    }
}

pub(crate) fn finalize_prompt_context(
    mut prompt: PromptContext,
    plugin_prompt_contributions: Vec<PromptContribution>,
) -> PromptContext {
    prompt.contributions.extend(plugin_prompt_contributions);
    prompt
}

fn count_prompt_omitted_tools(all_tools: &[ToolDefinition]) -> usize {
    all_tools
        .iter()
        .filter(|t| t.enabled)
        .filter(|t| !t.injected)
        .count()
}
