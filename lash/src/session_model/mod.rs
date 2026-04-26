pub mod context;
pub use lash_sansio::session_model::message;
pub use lash_sansio::session_model::prompt;

use tokio::sync::mpsc;

use crate::llm::types::{LlmEventSender, LlmStreamEvent};
use crate::plugin::PluginMessage;
use crate::provider::ProviderHandle;
use crate::{ContextApproach, ExecutionMode};

pub use lash_sansio::session_model::{
    CORE_GUIDANCE_SECTION, ConversationRecord, ErrorEnvelope, LLM_MAX_RETRIES, LLM_RETRY_DELAYS,
    MAIN_AGENT_INTRO, Message, MessageRole, Part, PartKind, PromptBuiltin, PromptSlot,
    PromptTemplate, PromptTemplateEntry, PromptTemplateSection, PruneState, SessionEvent,
    StateSnapshotEvent, TokenUsage, ToolEvent, TurnTerminationPolicyState, default_prompt_template,
    format_tool_result_content, fresh_message_id, make_error_envelope, make_error_event,
    reassign_part_ids, render_prompt, render_transcript_prompt,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ModeEvent {
    pub mode_id: ExecutionMode,
    pub payload: serde_json::Value,
}

impl ModeEvent {
    pub fn typed<T>(mode_id: ExecutionMode, event: T) -> Result<Self, serde_json::Error>
    where
        T: serde::Serialize,
    {
        Ok(Self {
            mode_id,
            payload: serde_json::to_value(event)?,
        })
    }

    pub fn decode<T>(&self, expected_mode: &ExecutionMode) -> Result<Option<T>, serde_json::Error>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        if &self.mode_id != expected_mode {
            return Ok(None);
        }
        serde_json::from_value(self.payload.clone()).map(Some)
    }

    pub fn rlm(event: lash_rlm_types::RlmModeEvent) -> Self {
        Self::typed(ExecutionMode::new("rlm"), event).expect("RLM mode events serialize")
    }

    pub fn rlm_event(&self) -> Option<lash_rlm_types::RlmModeEvent> {
        self.decode(&ExecutionMode::new("rlm")).ok().flatten()
    }
}

impl serde::Serialize for ModeEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde::Serialize)]
        struct Tagged<'a> {
            mode_id: &'a ExecutionMode,
            payload: &'a serde_json::Value,
        }
        Tagged {
            mode_id: &self.mode_id,
            payload: &self.payload,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ModeEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(object) = value.as_object()
            && let (Some(mode_id), Some(payload)) = (object.get("mode_id"), object.get("payload"))
        {
            let mode_id =
                ExecutionMode::deserialize(mode_id.clone()).map_err(serde::de::Error::custom)?;
            return Ok(Self {
                mode_id,
                payload: payload.clone(),
            });
        }
        let _: lash_rlm_types::RlmModeEvent =
            serde_json::from_value(value.clone()).map_err(serde::de::Error::custom)?;
        Ok(Self {
            mode_id: ExecutionMode::new("rlm"),
            payload: value,
        })
    }
}

pub type SessionEventRecord = lash_sansio::session_model::SessionEventRecord<ModeEvent>;

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
            tool_item_id: None,
            tool_signature: None,
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
            tool_item_id: None,
            tool_signature: None,
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
///
/// `provider` is a [`ProviderHandle`] — serializes through
/// [`crate::provider::ProviderSpec`], rebuilt via the global
/// [`crate::provider::ProviderRegistry`] on load. Hosts register the
/// concrete provider types they support at startup.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionPolicy {
    pub model: String,
    pub provider: ProviderHandle,
    pub max_context_tokens: Option<usize>,
    pub model_variant: Option<String>,
    pub session_id: Option<String>,
    #[serde(default)]
    pub autonomous: bool,
    pub max_turns: Option<usize>,
    pub execution_mode: ExecutionMode,
    #[serde(default)]
    pub context_approach: ContextApproach,
}

pub(crate) fn transport_stream_events(
    provider: &ProviderHandle,
    requested: Option<tokio::sync::mpsc::UnboundedSender<LlmStreamEvent>>,
) -> Option<LlmEventSender> {
    if let Some(requested) = requested {
        return Some(make_stream_event_sender(requested));
    }

    if provider.requires_streaming() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_event_reads_legacy_rlm_event_and_writes_tagged_payload() {
        let legacy = serde_json::json!({
            "RlmGlobalsPatch": {
                "set": { "answer": 42 },
                "unset": []
            }
        });
        let event: ModeEvent = serde_json::from_value(legacy).expect("legacy event");
        assert_eq!(event.mode_id, ExecutionMode::new("rlm"));
        assert!(matches!(
            event.rlm_event(),
            Some(lash_rlm_types::RlmModeEvent::RlmGlobalsPatch(_))
        ));

        let serialized = serde_json::to_value(event).expect("serialize");
        assert_eq!(serialized["mode_id"], "rlm");
        assert!(serialized.get("payload").is_some());
    }
}
