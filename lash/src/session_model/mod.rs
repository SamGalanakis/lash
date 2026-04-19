pub mod context;
pub use lash_sansio::session_model::message;
pub use lash_sansio::session_model::prompt;

use tokio::sync::mpsc;

use crate::llm::factory::adapter_for;
use crate::llm::types::{LlmEventSender, LlmStreamEvent};
use crate::plugin::PluginMessage;
use crate::provider::Provider;
use crate::{ContextApproach, ExecutionMode};

pub use lash_sansio::session_model::{
    CORE_GUIDANCE_SECTION, ErrorEnvelope, LLM_MAX_RETRIES, LLM_RETRY_DELAYS, MAIN_AGENT_INTRO,
    Message, MessageRole, Part, PartKind, PromptBuiltin, PromptSlot, PromptTemplate,
    PromptTemplateEntry, PromptTemplateSection, PruneState, SessionEvent, TokenUsage,
    TurnTerminationPolicyState, default_prompt_template, format_tool_result_content,
    fresh_message_id, make_error_envelope, make_error_event, reassign_part_ids, render_prompt,
    render_transcript_prompt,
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
            reasoning_meta: None,
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
            reasoning_meta: None,
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
            transient: false,
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
    #[serde(default)]
    pub context_approach: ContextApproach,
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
            context_approach: ContextApproach::default(),
        }
    }
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
