pub mod context;
pub use lash_sansio::session_model::message;
pub use lash_sansio::session_model::prompt;

use std::sync::Arc;
use tokio::sync::mpsc;

use crate::ModelSpec;
use crate::llm::types::{LlmEventSender, LlmStreamEvent};
use crate::plugin::PluginMessage;
use crate::provider::ProviderHandle;

pub use lash_sansio::format_tool_output_content;
pub use lash_sansio::session_model::{
    ConversationRecord, ErrorEnvelope, MAIN_AGENT_INTRO, Message, MessageRole, Part, PartKind,
    PromptBuiltin, PromptSlot, PromptTemplate, PromptTemplateEntry, PromptTemplateSection,
    PruneState, SessionEvent, TokenUsage, ToolEvent, TurnTerminationPolicyState,
    default_prompt_template, make_error_envelope, make_error_event, reassign_part_ids,
    render_prompt, render_transcript_prompt, shared_parts,
};

pub fn fresh_message_id() -> String {
    format!("m{}", uuid::Uuid::new_v4().simple())
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProtocolEvent {
    pub plugin_id: String,
    pub payload: serde_json::Value,
}

impl ProtocolEvent {
    pub fn typed<T>(plugin_id: impl Into<String>, event: T) -> Result<Self, serde_json::Error>
    where
        T: serde::Serialize,
    {
        Ok(Self {
            plugin_id: plugin_id.into(),
            payload: serde_json::to_value(event)?,
        })
    }

    pub fn decode<T>(&self, expected_plugin_id: &str) -> Result<Option<T>, serde_json::Error>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        if self.plugin_id != expected_plugin_id {
            return Ok(None);
        }
        serde_json::from_value(self.payload.clone()).map(Some)
    }
}

impl serde::Serialize for ProtocolEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde::Serialize)]
        struct Tagged<'a> {
            plugin_id: &'a str,
            payload: &'a serde_json::Value,
        }
        Tagged {
            plugin_id: &self.plugin_id,
            payload: &self.payload,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ProtocolEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(object) = value.as_object()
            && let (Some(plugin_id), Some(payload)) =
                (object.get("plugin_id"), object.get("payload"))
        {
            return Ok(Self {
                plugin_id: plugin_id
                    .as_str()
                    .ok_or_else(|| serde::de::Error::custom("plugin_id must be a string"))?
                    .to_string(),
                payload: payload.clone(),
            });
        }
        Err(serde::de::Error::custom(
            "protocol events must be tagged with plugin_id and payload",
        ))
    }
}

pub type SessionEventRecord = lash_sansio::session_model::SessionEventRecord<ProtocolEvent>;

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
            tool_replay: None,
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
        origin: plugin_message.origin.clone().or_else(|| {
            Some(crate::MessageOrigin::Plugin {
                plugin_id: "plugin".to_string(),
                transient: false,
            })
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
    pub model: ModelSpec,
    pub provider: ProviderHandle,
    pub session_id: Option<String>,
    #[serde(default)]
    pub autonomous: bool,
    pub max_turns: Option<usize>,
    #[serde(default, skip_serializing_if = "crate::PromptLayer::is_empty")]
    pub prompt: crate::PromptLayer,
}

impl SessionPolicy {
    pub fn model_id(&self) -> &str {
        &self.model.id
    }

    pub fn model_variant(&self) -> Option<&str> {
        self.model.variant.as_deref()
    }

    pub fn context_window_tokens(&self) -> usize {
        self.model.context_window_tokens()
    }
}

/// Reusable session configuration overlay.
///
/// `SessionSpec` is the public configuration shape for callers that want to
/// describe either a root session or a child session without constructing the
/// resolved runtime-only [`SessionPolicy`] directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionSpec {
    inherit: bool,
    pub provider: Option<ProviderHandle>,
    pub model: Option<ModelSpec>,
    pub max_turns: Option<Option<usize>>,
    pub prompt: Option<crate::PromptLayer>,
}

impl SessionSpec {
    /// Create an explicit root-style spec. Unset fields resolve from the
    /// runtime's core defaults.
    pub fn new() -> Self {
        Self {
            inherit: false,
            provider: None,
            model: None,
            max_turns: None,
            prompt: None,
        }
    }

    /// Create a parent-relative spec. Unset fields inherit from the live
    /// parent policy at resolution time.
    pub fn inherit() -> Self {
        Self {
            inherit: true,
            ..Self::new()
        }
    }

    pub fn inherits(&self) -> bool {
        self.inherit
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn model(mut self, model: ModelSpec) -> Self {
        self.model = Some(model);
        self
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(Some(max_turns));
        self
    }

    pub fn clear_max_turns(mut self) -> Self {
        self.max_turns = Some(None);
        self
    }

    pub fn prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.prompt = Some(prompt);
        self
    }

    pub fn resolve_against(&self, base: &SessionPolicy) -> SessionPolicy {
        let mut policy = base.clone();
        if let Some(provider) = self.provider.as_ref() {
            policy.provider = provider.clone();
        }
        if let Some(model) = self.model.as_ref() {
            policy.model = model.clone();
        }
        if let Some(max_turns) = self.max_turns {
            policy.max_turns = max_turns;
        }
        if let Some(prompt) = self.prompt.as_ref() {
            policy.prompt = prompt.clone();
        }
        policy
    }
}

impl Default for SessionSpec {
    fn default() -> Self {
        Self::new()
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
    fn protocol_event_writes_tagged_payload() {
        let event = ProtocolEvent::typed("test_protocol", serde_json::json!({ "value": 42 }))
            .expect("typed event");
        let serialized = serde_json::to_value(event).expect("serialize");
        assert_eq!(serialized["plugin_id"], "test_protocol");
        assert!(serialized.get("payload").is_some());
    }
}
