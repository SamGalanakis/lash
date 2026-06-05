use super::process::ProcessWakeDelivery;
use crate::{PluginMessage, TurnCause, TurnInput};

pub const QUEUED_WORK_CLAIM_TTL_MS: u64 = 15 * 60 * 1000;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionCommand {
    RefreshToolSurface {
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected_generation: Option<u64>,
    },
    EmitHostEvent {
        resource_type: String,
        alias: String,
        event: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    ResetSession {
        reason: String,
    },
}

impl SessionCommand {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::RefreshToolSurface { .. } => "refresh_tool_surface",
            Self::EmitHostEvent { .. } => "emit_host_event",
            Self::ResetSession { .. } => "reset_session",
        }
    }

    pub fn source_key(&self, idempotency_key: impl AsRef<str>) -> String {
        format!("command:{}:{}", self.kind(), idempotency_key.as_ref())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionCommandReceipt {
    pub session_id: String,
    pub batch_id: String,
    pub source_key: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryPolicy {
    EarliestSafeBoundary,
    AfterCurrentTurnCommit,
}

impl DeliveryPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EarliestSafeBoundary => "earliest_safe_boundary",
            Self::AfterCurrentTurnCommit => "after_current_turn_commit",
        }
    }

    pub fn from_wire_str(value: &str) -> Option<Self> {
        match value {
            "earliest_safe_boundary" => Some(Self::EarliestSafeBoundary),
            "after_current_turn_commit" => Some(Self::AfterCurrentTurnCommit),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotPolicy {
    Join,
    Exclusive,
}

impl SlotPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Join => "join",
            Self::Exclusive => "exclusive",
        }
    }

    pub fn from_wire_str(value: &str) -> Option<Self> {
        match value {
            "join" => Some(Self::Join),
            "exclusive" => Some(Self::Exclusive),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeKey {
    Never,
    PayloadDefault,
    Group(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueuedWorkPayload {
    TurnInput {
        input: Box<TurnInput>,
    },
    ProcessWake {
        wake: Box<ProcessWakeDelivery>,
    },
    HostEvent {
        name: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    Timer {
        name: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    SessionCommand {
        command: Box<SessionCommand>,
    },
}

impl QueuedWorkPayload {
    pub fn turn_input(input: TurnInput) -> Self {
        Self::TurnInput {
            input: Box::new(input),
        }
    }

    pub fn process_wake(wake: ProcessWakeDelivery) -> Self {
        Self::ProcessWake {
            wake: Box::new(wake),
        }
    }

    pub fn session_command(command: SessionCommand) -> Self {
        Self::SessionCommand {
            command: Box::new(command),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueuedWorkItem {
    pub item_id: String,
    pub payload: QueuedWorkPayload,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueuedWorkBatch {
    pub batch_id: String,
    pub session_id: String,
    pub enqueue_seq: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    pub delivery_policy: DeliveryPolicy,
    pub slot_policy: SlotPolicy,
    pub merge_key: MergeKey,
    pub available_at_ms: u64,
    pub enqueued_at_ms: u64,
    pub items: Vec<QueuedWorkItem>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueuedWorkBatchDraft {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    pub delivery_policy: DeliveryPolicy,
    pub slot_policy: SlotPolicy,
    pub merge_key: MergeKey,
    pub available_at_ms: u64,
    pub payloads: Vec<QueuedWorkPayload>,
}

impl QueuedWorkBatchDraft {
    pub fn new(
        session_id: impl Into<String>,
        delivery_policy: DeliveryPolicy,
        slot_policy: SlotPolicy,
        payloads: impl Into<Vec<QueuedWorkPayload>>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            source_key: None,
            delivery_policy,
            slot_policy,
            merge_key: MergeKey::Never,
            available_at_ms: 0,
            payloads: payloads.into(),
        }
    }

    pub fn with_source_key(mut self, source_key: impl Into<String>) -> Self {
        self.source_key = Some(source_key.into());
        self
    }

    pub fn with_available_at_ms(mut self, available_at_ms: u64) -> Self {
        self.available_at_ms = available_at_ms;
        self
    }

    pub fn with_merge_key(mut self, merge_key: MergeKey) -> Self {
        self.merge_key = merge_key;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueuedWorkClaimBoundary {
    ActiveTurnCheckpoint,
    Idle,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QueuedWorkCompletion {
    pub session_id: String,
    pub claim_id: String,
    pub lease_token: String,
    pub batch_ids: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueuedWorkClaim {
    pub session_id: String,
    pub claim_id: String,
    pub owner_id: String,
    pub lease_token: String,
    pub fencing_token: u64,
    pub claimed_at_epoch_ms: u64,
    pub expires_at_epoch_ms: u64,
    pub batches: Vec<QueuedWorkBatch>,
}

impl QueuedWorkClaim {
    pub fn completion(&self) -> QueuedWorkCompletion {
        QueuedWorkCompletion {
            session_id: self.session_id.clone(),
            claim_id: self.claim_id.clone(),
            lease_token: self.lease_token.clone(),
            batch_ids: self
                .batches
                .iter()
                .map(|batch| batch.batch_id.clone())
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.batches.iter().all(|batch| batch.items.is_empty())
    }

    pub fn materialize_for_checkpoint(&self) -> QueuedCheckpointWork {
        let messages = Vec::new();
        let mut transient_messages = Vec::new();
        let mut turn_causes = Vec::new();
        for batch in &self.batches {
            for item in &batch.items {
                match &item.payload {
                    QueuedWorkPayload::TurnInput { input } => {
                        if let Some(message) = plugin_message_from_turn_input(input) {
                            transient_messages.push(message);
                        }
                    }
                    QueuedWorkPayload::ProcessWake { wake } => {
                        turn_causes.push(crate::process_wake_turn_cause(wake));
                    }
                    QueuedWorkPayload::HostEvent { name, payload }
                    | QueuedWorkPayload::Timer { name, payload } => {
                        turn_causes.push(host_event_cause(
                            &item.item_id,
                            name,
                            payload,
                            matches!(&item.payload, QueuedWorkPayload::Timer { .. }),
                        ));
                    }
                    QueuedWorkPayload::SessionCommand { .. } => {}
                }
            }
        }
        QueuedCheckpointWork {
            messages,
            transient_messages,
            turn_causes,
        }
    }

    pub fn materialize_for_checkpoint_with_attachments(
        &self,
        attachment_store: &dyn crate::AttachmentStore,
    ) -> Result<QueuedCheckpointWork, String> {
        let messages = Vec::new();
        let mut transient_messages = Vec::new();
        let mut turn_causes = Vec::new();
        for batch in &self.batches {
            for item in &batch.items {
                match &item.payload {
                    QueuedWorkPayload::TurnInput { input } => {
                        if let Some(message) = plugin_message_from_turn_input_with_attachments(
                            input,
                            attachment_store,
                        )? {
                            transient_messages.push(message);
                        }
                    }
                    QueuedWorkPayload::ProcessWake { wake } => {
                        turn_causes.push(crate::process_wake_turn_cause(wake));
                    }
                    QueuedWorkPayload::HostEvent { name, payload }
                    | QueuedWorkPayload::Timer { name, payload } => {
                        turn_causes.push(host_event_cause(
                            &item.item_id,
                            name,
                            payload,
                            matches!(&item.payload, QueuedWorkPayload::Timer { .. }),
                        ));
                    }
                    QueuedWorkPayload::SessionCommand { .. } => {}
                }
            }
        }
        Ok(QueuedCheckpointWork {
            messages,
            transient_messages,
            turn_causes,
        })
    }

    pub fn accepted_turn_inputs(&self) -> Vec<crate::AcceptedInjectedTurnInput> {
        let mut accepted = Vec::new();
        for batch in &self.batches {
            let id = batch.source_key.as_deref().map(|source| {
                source
                    .strip_prefix("host:")
                    .or_else(|| source.strip_prefix("injection:"))
                    .unwrap_or(source)
                    .to_string()
            });
            for item in &batch.items {
                if let QueuedWorkPayload::TurnInput { input } = &item.payload
                    && let Some(message) = plugin_message_from_turn_input(input)
                {
                    accepted.push(crate::AcceptedInjectedTurnInput {
                        id: id.clone(),
                        message,
                    });
                }
            }
        }
        accepted
    }

    pub fn exclusive_session_command(&self) -> Option<(&QueuedWorkBatch, &SessionCommand)> {
        if self.batches.len() != 1 {
            return None;
        }
        let batch = self.batches.first()?;
        if batch.slot_policy != SlotPolicy::Exclusive || batch.items.len() != 1 {
            return None;
        }
        let item = batch.items.first()?;
        match &item.payload {
            QueuedWorkPayload::SessionCommand { command } => Some((batch, command.as_ref())),
            _ => None,
        }
    }

    pub fn materialize_for_turn(&self) -> QueuedTurnWork {
        let checkpoint = self.materialize_for_checkpoint();
        let mut input_items = Vec::new();
        let mut image_blobs = std::collections::HashMap::new();
        let mut protocol_turn_options = None;
        let mut trace_turn_id = None;
        for batch in &self.batches {
            for item in &batch.items {
                if let QueuedWorkPayload::TurnInput { input } = &item.payload {
                    input_items.extend(input.items.clone());
                    image_blobs.extend(input.image_blobs.clone());
                    if protocol_turn_options.is_none() {
                        protocol_turn_options = input.protocol_turn_options.clone();
                    }
                    if trace_turn_id.is_none() {
                        trace_turn_id = input.trace_turn_id.clone();
                    }
                }
            }
        }
        QueuedTurnWork {
            input: TurnInput {
                items: input_items,
                image_blobs,
                protocol_turn_options,
                trace_turn_id,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            messages: checkpoint.messages,
            turn_causes: checkpoint.turn_causes,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct QueuedCheckpointWork {
    pub messages: Vec<PluginMessage>,
    pub transient_messages: Vec<PluginMessage>,
    pub turn_causes: Vec<TurnCause>,
}

#[derive(Clone, Debug)]
pub struct QueuedTurnWork {
    pub input: TurnInput,
    pub messages: Vec<PluginMessage>,
    pub turn_causes: Vec<TurnCause>,
}

pub fn process_wake_batch_draft(wake: ProcessWakeDelivery) -> QueuedWorkBatchDraft {
    let source_key = format!("process:{}:event:{}:wake", wake.process_id, wake.sequence);
    QueuedWorkBatchDraft::new(
        wake.target_session_id.clone(),
        DeliveryPolicy::EarliestSafeBoundary,
        SlotPolicy::Exclusive,
        vec![QueuedWorkPayload::process_wake(wake)],
    )
    .with_source_key(source_key)
}

fn plugin_message_from_turn_input(input: &TurnInput) -> Option<PluginMessage> {
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

fn plugin_message_from_turn_input_with_attachments(
    input: &TurnInput,
    attachment_store: &dyn crate::AttachmentStore,
) -> Result<Option<PluginMessage>, String> {
    let normalized =
        super::io::normalize_input_items(&input.items, &input.image_blobs, attachment_store)?;
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
                let part_id = format!("queued.p{}", parts.len());
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
                let part_id = format!("queued.p{}", parts.len());
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

fn host_event_cause(
    item_id: &str,
    name: &str,
    payload: &serde_json::Value,
    timer: bool,
) -> TurnCause {
    let event_type = if timer { "timer" } else { "host_event" };
    TurnCause {
        id: item_id.to_string(),
        event_type: name.to_string(),
        origin: crate::MessageOrigin::Plugin {
            plugin_id: event_type.to_string(),
            transient: false,
        },
        text: if payload.is_null() {
            name.to_string()
        } else {
            format!("{name}\n{payload}")
        },
    }
}
