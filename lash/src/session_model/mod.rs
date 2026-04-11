pub mod context;
pub use lash_sansio::session_model::message;
pub use lash_sansio::session_model::prompt;

use tokio::sync::mpsc;

use crate::ExecutionMode;
use crate::PromptContext;
use crate::ToolDefinition;
use crate::llm::factory::adapter_for;
use crate::llm::types::{LlmEventSender, LlmStreamEvent, LlmToolSpec};
use crate::plugin::{PluginMessage, PromptContribution};
use crate::provider::Provider;
use crate::session::Session;

pub use lash_sansio::session_model::{
    DefaultPromptRenderer, DurableTurnSnapshot, ErrorEnvelope, LLM_MAX_RETRIES, LLM_RETRY_DELAYS,
    Message, MessageRole, Part, PartKind, PromptOverrideMode, PromptRenderer, PromptSectionName,
    PromptSectionOverride, PruneState, SessionEvent, TokenUsage, TurnTerminationPolicyState,
    default_prompt_renderer, format_tool_result_content, fresh_message_id, make_error_envelope,
    make_error_event, reassign_part_ids, render_prompt, render_transcript_prompt,
};

/// Send an event to the channel if it's still open.
pub(crate) async fn send_event(tx: &mpsc::Sender<SessionEvent>, event: SessionEvent) {
    if !tx.is_closed() {
        let _ = tx.send(event).await;
    }
}

pub(crate) fn plugin_message_to_message(
    plugin_message: &PluginMessage,
    user_input: Option<crate::UserInputProvenance>,
) -> Message {
    let message_id = fresh_message_id();
    let mut parts = if plugin_message.parts.is_empty() {
        vec![Part {
            id: format!("{message_id}.p0"),
            kind: PartKind::Text,
            content: plugin_message.content.clone(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }]
    } else {
        plugin_message.parts.clone()
    };
    let has_image_parts = parts
        .iter()
        .any(|part| matches!(part.kind, PartKind::Image));
    if matches!(plugin_message.role, MessageRole::User) && !has_image_parts {
        parts.extend(plugin_message.images.iter().map(|bytes| Part {
            id: String::new(),
            kind: PartKind::Image,
            content: String::new(),
            attachment: Some(message::PartAttachment {
                mime: "image/png".to_string(),
                url: message::data_url_for_bytes("image/png", bytes),
                filename: None,
            }),
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }));
    }
    reassign_part_ids(&message_id, &mut parts);
    Message {
        id: message_id,
        role: plugin_message.role,
        parts,
        user_input,
        origin: Some(crate::MessageOrigin::Plugin {
            plugin_id: "plugin".to_string(),
        }),
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
        }
    }
}

pub(crate) struct ExecutionPreamble {
    pub(crate) model: String,
    pub(crate) tool_specs: Vec<LlmToolSpec>,
    pub(crate) prompt: PromptContext,
}

fn repl_model_tool_specs() -> Vec<LlmToolSpec> {
    let tool = crate::tools::repl_execute_lashlang_tool_definition().model_tool();
    vec![LlmToolSpec {
        name: tool.name,
        description: tool.description,
        input_schema: tool.input_schema,
        output_schema: tool.output_schema,
    }]
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
    let prompt_tools = surface.prompt_tools();
    let (tool_list, omitted_tool_count) = if matches!(mode, ExecutionMode::Repl) {
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
    let tool_specs = match mode {
        ExecutionMode::Standard if !enabled_tools.is_empty() => {
            lash_sansio::session_model::model_tool_specs(&enabled_tools)
        }
        ExecutionMode::Repl => repl_model_tool_specs(),
        _ => Vec::new(),
    };
    let tool_names: Vec<String> = enabled_tools.iter().map(|t| t.name.clone()).collect();
    let prompt = PromptContext {
        mode,
        tool_list,
        tool_names,
        omitted_tool_count,
        contributions: Vec::new(),
    };

    tracing::debug!(
        session_id,
        ?mode,
        model,
        enabled_tool_count = enabled_tools.len(),
        prompt_tool_count = prompt_tools.len(),
        omitted_tool_count,
        tool_names = ?prompt.tool_names,
        tool_list_preview = %prompt.tool_list.chars().take(400).collect::<String>(),
        "built execution preamble"
    );

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
