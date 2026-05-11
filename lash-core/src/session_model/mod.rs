pub mod context;
pub use lash_sansio::session_model::message;
pub use lash_sansio::session_model::prompt;

use std::sync::Arc;
use tokio::sync::mpsc;

use crate::llm::types::{LlmEventSender, LlmStreamEvent};
use crate::plugin::PluginMessage;
use crate::provider::ProviderHandle;
use crate::{ExecutionMode, StandardContextApproach};

pub use lash_sansio::session_model::{
    CORE_GUIDANCE_SECTION, ConversationRecord, ErrorEnvelope, MAIN_AGENT_INTRO, Message,
    MessageRole, Part, PartKind, PromptBuiltin, PromptSlot, PromptTemplate, PromptTemplateEntry,
    PromptTemplateSection, PruneState, SessionEvent, StateSnapshotEvent, TokenUsage, ToolEvent,
    TurnTerminationPolicyState, default_prompt_template, format_tool_result_content,
    fresh_message_id, make_error_envelope, make_error_event, reassign_part_ids, render_prompt,
    render_transcript_prompt, shared_parts,
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
        Err(serde::de::Error::custom(
            "mode events must be tagged with mode_id and payload",
        ))
    }
}

pub type SessionEventRecord = lash_sansio::session_model::SessionEventRecord<ModeEvent>;

/// Send an event to the channel if it's still open.
pub(crate) async fn send_event(tx: &mpsc::Sender<SessionEvent>, event: SessionEvent) {
    if !tx.is_closed() {
        let _ = tx.send(event).await;
    }
}

pub(crate) fn plugin_message_to_message(plugin_message: &PluginMessage) -> Message {
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
            response_meta: None,
        }]
    } else {
        plugin_message.parts.clone()
    };
    reassign_part_ids(&message_id, &mut parts);
    Message {
        id: message_id,
        role: plugin_message.role,
        parts: Arc::new(parts),
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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_context_approach: Option<StandardContextApproach>,
    #[serde(default, skip_serializing_if = "crate::PromptLayer::is_empty")]
    pub prompt: crate::PromptLayer,
}

impl Default for SessionPolicy {
    fn default() -> Self {
        Self {
            model: String::new(),
            provider: ProviderHandle::default(),
            max_context_tokens: None,
            model_variant: None,
            session_id: None,
            autonomous: false,
            max_turns: None,
            execution_mode: ExecutionMode::standard(),
            standard_context_approach: Some(StandardContextApproach::default()),
            prompt: crate::PromptLayer::default(),
        }
    }
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
    fn mode_event_writes_tagged_payload() {
        let event = ModeEvent::typed(
            ExecutionMode::new("test"),
            serde_json::json!({ "value": 42 }),
        )
        .expect("typed event");
        let serialized = serde_json::to_value(event).expect("serialize");
        assert_eq!(serialized["mode_id"], "test");
        assert!(serialized.get("payload").is_some());
    }
}
