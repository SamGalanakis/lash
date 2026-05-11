use serde::{Deserialize, Serialize};

use lash_core::Message;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ActiveMemoryNode {
    pub(crate) observed_through_message_id: String,
    pub(crate) observations: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) current_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) suggested_response: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BufferedObservationNode {
    pub(crate) observed_through_message_id: String,
    pub(crate) observations: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) current_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) suggested_response: Option<String>,
    #[serde(default)]
    pub(crate) observation_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct BufferedReflectionNode {
    pub(crate) source_state_node_id: String,
    pub(crate) observed_through_message_id: String,
    pub(crate) observations: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) current_task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) suggested_response: Option<String>,
    #[serde(default)]
    pub(crate) observation_tokens: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ActiveMemoryState {
    pub(crate) state_node_id: String,
    pub(crate) observed_through_message_id: Option<String>,
    pub(crate) observations: String,
    pub(crate) current_task: Option<String>,
    pub(crate) suggested_response: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct BufferedObservationState {
    pub(crate) observed_through_message_id: String,
    pub(crate) observations: String,
    pub(crate) current_task: Option<String>,
    pub(crate) suggested_response: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct BufferedReflectionState {
    pub(crate) source_state_node_id: String,
    pub(crate) observed_through_message_id: String,
    pub(crate) observations: String,
    pub(crate) current_task: Option<String>,
    pub(crate) suggested_response: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OmGraphState {
    pub(crate) active: Option<ActiveMemoryState>,
    pub(crate) buffered_observations: Vec<BufferedObservationState>,
    pub(crate) buffered_reflection: Option<BufferedReflectionState>,
}

#[derive(Clone, Debug)]
pub(crate) struct MessageNode {
    pub(crate) timestamp: String,
    pub(crate) message: Message,
}

pub(crate) trait ObservedMessageNode {
    fn timestamp(&self) -> &str;
    fn message(&self) -> &Message;
}

impl ObservedMessageNode for MessageNode {
    fn timestamp(&self) -> &str {
        &self.timestamp
    }

    fn message(&self) -> &Message {
        &self.message
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ParsedMemoryOutput {
    pub(crate) observations: String,
    pub(crate) current_task: Option<String>,
    pub(crate) suggested_response: Option<String>,
}
