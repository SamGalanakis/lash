use crate::{CheckpointKind, PluginMessage, TurnCause, TurnInput};

pub const TURN_INPUT_CLAIM_TTL_MS: u64 = 30 * 1000;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum TurnInputIngress {
    ActiveTurn {
        turn_id: String,
        #[serde(default)]
        min_boundary: TurnInputCheckpointBoundary,
    },
    NextTurn,
}

impl TurnInputIngress {
    pub fn active_turn(
        turn_id: impl Into<String>,
        min_boundary: TurnInputCheckpointBoundary,
    ) -> Self {
        Self::ActiveTurn {
            turn_id: turn_id.into(),
            min_boundary,
        }
    }

    pub fn next_turn() -> Self {
        Self::NextTurn
    }

    pub fn active_turn_id(&self) -> Option<&str> {
        match self {
            Self::ActiveTurn { turn_id, .. } => Some(turn_id),
            Self::NextTurn => None,
        }
    }

    pub fn admits_checkpoint(&self, checkpoint: CheckpointKind) -> bool {
        match self {
            Self::ActiveTurn { min_boundary, .. } => min_boundary.admits(checkpoint),
            Self::NextTurn => false,
        }
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TurnInputCheckpointBoundary {
    #[default]
    AfterWork,
    BeforeCompletion,
}

impl TurnInputCheckpointBoundary {
    pub fn admits(self, checkpoint: CheckpointKind) -> bool {
        match self {
            Self::AfterWork => true,
            Self::BeforeCompletion => checkpoint == CheckpointKind::BeforeCompletion,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnInputState {
    PendingActive,
    DeferredNextTurn,
    Accepted,
    Cancelled,
    Completed,
}

impl TurnInputState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PendingActive => "pending_active",
            Self::DeferredNextTurn => "deferred_next_turn",
            Self::Accepted => "accepted",
            Self::Cancelled => "cancelled",
            Self::Completed => "completed",
        }
    }

    pub fn from_wire_str(value: &str) -> Option<Self> {
        match value {
            "pending_active" => Some(Self::PendingActive),
            "deferred_next_turn" => Some(Self::DeferredNextTurn),
            "accepted" => Some(Self::Accepted),
            "cancelled" => Some(Self::Cancelled),
            "completed" => Some(Self::Completed),
            _ => None,
        }
    }

    pub fn is_next_turn_pending(self) -> bool {
        matches!(self, Self::DeferredNextTurn)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PendingTurnInputDraft {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    pub ingress: TurnInputIngress,
    pub input: TurnInput,
}

impl PendingTurnInputDraft {
    pub fn new(session_id: impl Into<String>, ingress: TurnInputIngress, input: TurnInput) -> Self {
        Self {
            session_id: session_id.into(),
            input_id: None,
            source_key: None,
            ingress,
            input,
        }
    }

    pub fn with_input_id(mut self, input_id: impl Into<String>) -> Self {
        self.input_id = Some(input_id.into());
        self
    }

    pub fn with_source_key(mut self, source_key: impl Into<String>) -> Self {
        self.source_key = Some(source_key.into());
        self
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PendingTurnInput {
    pub input_id: String,
    pub session_id: String,
    pub enqueue_seq: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    pub ingress: TurnInputIngress,
    pub state: TurnInputState,
    pub enqueued_at_ms: u64,
    pub input: TurnInput,
}

impl PendingTurnInput {
    pub fn source_or_id(&self) -> &str {
        self.source_key.as_deref().unwrap_or(&self.input_id)
    }

    pub fn accepted_input(&self) -> Option<crate::AcceptedInjectedTurnInput> {
        plugin_message_from_turn_input(&self.input).map(|message| {
            crate::AcceptedInjectedTurnInput {
                id: self
                    .source_key
                    .as_deref()
                    .map(source_key_display_id)
                    .or_else(|| Some(self.input_id.clone())),
                message,
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnInputClaimMode {
    ActiveTurn {
        turn_id: String,
        checkpoint: CheckpointKind,
    },
    NextTurn,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TurnInputCompletion {
    pub session_id: String,
    pub claim_id: String,
    pub lease_token: String,
    pub input_ids: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnInputClaim {
    pub session_id: String,
    pub claim_id: String,
    pub owner: crate::LeaseOwnerIdentity,
    pub lease_token: String,
    pub fencing_token: u64,
    pub claimed_at_epoch_ms: u64,
    pub expires_at_epoch_ms: u64,
    pub mode: TurnInputClaimMode,
    pub inputs: Vec<PendingTurnInput>,
}

impl TurnInputClaim {
    pub fn completion(&self) -> TurnInputCompletion {
        TurnInputCompletion {
            session_id: self.session_id.clone(),
            claim_id: self.claim_id.clone(),
            lease_token: self.lease_token.clone(),
            input_ids: self
                .inputs
                .iter()
                .map(|input| input.input_id.clone())
                .collect(),
        }
    }

    pub fn accepted_turn_inputs(&self) -> Vec<crate::AcceptedInjectedTurnInput> {
        self.inputs
            .iter()
            .filter_map(PendingTurnInput::accepted_input)
            .collect()
    }

    pub async fn materialize_for_checkpoint(
        &self,
        attachment_store: &dyn crate::AttachmentStore,
    ) -> Result<QueuedCheckpointTurnInput, String> {
        let mut transient_messages = Vec::new();
        for input in &self.inputs {
            if let Some(message) =
                plugin_message_from_turn_input_with_attachments(&input.input, attachment_store)
                    .await?
            {
                transient_messages.push(message);
            }
        }
        Ok(QueuedCheckpointTurnInput {
            transient_messages,
            turn_causes: Vec::new(),
        })
    }

    pub fn materialize_for_turn(&self) -> TurnInput {
        let mut input_items = Vec::new();
        let mut image_blobs = std::collections::HashMap::new();
        let mut protocol_turn_options = None;
        let mut trace_turn_id = None;
        for pending in &self.inputs {
            input_items.extend(pending.input.items.clone());
            image_blobs.extend(pending.input.image_blobs.clone());
            if protocol_turn_options.is_none() {
                protocol_turn_options = pending.input.protocol_turn_options.clone();
            }
            if trace_turn_id.is_none() {
                trace_turn_id = pending.input.trace_turn_id.clone();
            }
        }
        TurnInput {
            items: input_items,
            image_blobs,
            protocol_turn_options,
            trace_turn_id,
            protocol_extension: None,
            turn_context: crate::TurnContext::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct QueuedCheckpointTurnInput {
    pub transient_messages: Vec<PluginMessage>,
    pub turn_causes: Vec<TurnCause>,
}

pub(crate) fn source_key_display_id(source: &str) -> String {
    source
        .strip_prefix("host:")
        .or_else(|| source.strip_prefix("injection:"))
        .unwrap_or(source)
        .to_string()
}

pub(crate) fn plugin_message_from_turn_input(input: &TurnInput) -> Option<PluginMessage> {
    let mut text = Vec::new();
    let mut images = Vec::new();
    for item in &input.items {
        match item {
            crate::InputItem::Text { text: item_text } if !item_text.is_empty() => {
                text.push(item_text.clone());
            }
            crate::InputItem::Text { .. } => {}
            crate::InputItem::ImageRef { id } => {
                if let Some(bytes) = input.image_blobs.get(id).cloned() {
                    images.push(bytes);
                }
            }
        }
    }
    if text.is_empty() && images.is_empty() {
        return None;
    }
    Some(PluginMessage {
        role: crate::MessageRole::User,
        content: text.join("\n"),
        origin: None,
        parts: Vec::new(),
        images,
    })
}

pub(crate) async fn plugin_message_from_turn_input_with_attachments(
    input: &TurnInput,
    attachment_store: &dyn crate::AttachmentStore,
) -> Result<Option<PluginMessage>, String> {
    let normalized =
        super::io::normalize_input_items(&input.items, &input.image_blobs, attachment_store)
            .await?;
    let has_image = normalized
        .iter()
        .any(|item| matches!(item, super::NormalizedItem::Image(_)));
    if !has_image {
        return Ok(plugin_message_from_turn_input(input));
    }

    let mut content = Vec::new();
    let mut parts = Vec::new();
    for item in normalized {
        match item {
            super::NormalizedItem::Text(text) if !text.is_empty() => {
                let part_id = format!("pending.p{}", parts.len());
                content.push(text.clone());
                parts.push(crate::Part {
                    id: part_id,
                    kind: crate::PartKind::Text,
                    content: text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: crate::PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                });
            }
            super::NormalizedItem::Text(_) => {}
            super::NormalizedItem::Image(reference) => {
                let part_id = format!("pending.p{}", parts.len());
                parts.push(crate::Part {
                    id: part_id,
                    kind: crate::PartKind::Image,
                    content: String::new(),
                    attachment: Some(crate::session_model::message::PartAttachment { reference }),
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: crate::PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                });
            }
        }
    }
    if parts.is_empty() {
        return Ok(None);
    }
    Ok(Some(PluginMessage {
        role: crate::MessageRole::User,
        content: content.join("\n"),
        origin: None,
        parts,
        images: Vec::new(),
    }))
}
