use std::collections::HashSet;

use lash::{ChronologicalPayload, ExecutionMode, Message, MessageRole, PartKind, ToolCallRecord};
use lash_rlm_types::{
    RlmAttachmentRef, RlmHistoryItem, RlmHistoryRole, RlmImageRef, RlmModeEvent, RlmTrajectoryEntry,
};

pub fn rlm_mode_event(event: RlmModeEvent) -> lash::ModeEvent {
    lash::ModeEvent::typed(ExecutionMode::new("rlm"), event).expect("RLM mode events serialize")
}

pub fn decode_rlm_mode_event(event: &lash::ModeEvent) -> Option<RlmModeEvent> {
    event.decode(&ExecutionMode::new("rlm")).ok().flatten()
}

pub fn project_rlm_globals_from_events<'a>(
    events: impl IntoIterator<Item = &'a lash::SessionEventRecord>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut globals = serde_json::Map::new();
    for event in events {
        if let lash::SessionEventRecord::Mode(event) = event
            && let Some(RlmModeEvent::RlmGlobalsPatch(patch)) = decode_rlm_mode_event(event)
        {
            lash_rlm_types::apply_globals_patch(&mut globals, &patch);
        }
    }
    globals
}

#[derive(Clone, Debug)]
pub struct RlmHistoryProjection {
    history: Vec<RlmHistoryItem>,
}

impl RlmHistoryProjection {
    pub fn from_chronological(projection: &lash::ChronologicalProjection) -> Self {
        let mut history = Vec::with_capacity(projection.entries().len());
        let mut seen_tool_calls = HashSet::new();
        for entry in projection.entries() {
            match &entry.payload {
                ChronologicalPayload::Message(message) => {
                    if let Some(item) = history_item_from_message(message) {
                        history.push(item);
                    }
                }
                ChronologicalPayload::ToolCall(record) => {
                    if seen_tool_calls.insert(tool_call_record_key(record)) {
                        history.push(history_item_from_tool_call(record));
                    }
                }
                ChronologicalPayload::ModeEvent(event) => {
                    if let Some(RlmModeEvent::RlmTrajectoryEntry(mut step)) =
                        decode_rlm_mode_event(event)
                    {
                        step.tool_calls
                            .retain(|record| seen_tool_calls.insert(tool_call_record_key(record)));
                        history.push(history_item_from_rlm_step(&step));
                    }
                }
            }
        }
        Self { history }
    }

    pub fn history(&self) -> &[RlmHistoryItem] {
        self.history.as_slice()
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    pub fn item(&self, index: usize) -> Option<RlmHistoryItem> {
        self.history.get(index).cloned()
    }

    pub fn value(&self) -> serde_json::Value {
        serde_json::to_value(&self.history).unwrap_or_else(|_| serde_json::Value::Array(vec![]))
    }
}

pub fn rlm_history_projection(projection: &lash::ChronologicalProjection) -> RlmHistoryProjection {
    RlmHistoryProjection::from_chronological(projection)
}

fn history_item_from_message(message: &Message) -> Option<RlmHistoryItem> {
    Some(RlmHistoryItem::Message {
        id: message.id.clone(),
        role: history_role(message.role),
        content: message_history_text(message),
        attachments: message
            .parts
            .iter()
            .filter_map(|part| {
                let attachment = part.attachment.as_ref()?;
                Some(RlmAttachmentRef {
                    id: part.id.clone(),
                    media_type: attachment.reference.media_type,
                    label: attachment.reference.label.clone(),
                    reference: attachment.reference.id.to_string(),
                })
            })
            .collect(),
    })
}

fn history_item_from_rlm_step(entry: &RlmTrajectoryEntry) -> RlmHistoryItem {
    RlmHistoryItem::RlmStep {
        id: entry.id.clone(),
        mode_iteration: entry.mode_iteration,
        reasoning: entry.reasoning.clone(),
        code: entry.code.clone(),
        output: entry.output.clone(),
        tool_calls: entry.tool_calls.clone(),
        images: entry.images.iter().map(image_ref).collect(),
        error: entry.error.clone(),
        final_output: entry.final_output.clone(),
    }
}

fn history_item_from_tool_call(record: &ToolCallRecord) -> RlmHistoryItem {
    RlmHistoryItem::ToolCall {
        id: record
            .call_id
            .clone()
            .unwrap_or_else(|| tool_call_record_key(record)),
        tool: record.tool.clone(),
        args: record.args.clone(),
        result: record.result.clone(),
        success: record.success,
        duration_ms: record.duration_ms,
    }
}

fn message_history_text(message: &Message) -> String {
    let chunks = message
        .parts
        .iter()
        .filter(|part| matches!(part.kind, PartKind::Text | PartKind::Prose))
        .map(|part| part.content.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    chunks.join("\n\n")
}

fn history_role(role: MessageRole) -> RlmHistoryRole {
    match role {
        MessageRole::User => RlmHistoryRole::User,
        MessageRole::System => RlmHistoryRole::System,
        MessageRole::Assistant => RlmHistoryRole::Assistant,
    }
}

fn image_ref(image: &lash::AttachmentRef) -> RlmImageRef {
    RlmImageRef {
        id: image.id.to_string(),
        media_type: image.media_type,
        width: image.width,
        height: image.height,
        bytes: image.byte_len as usize,
        label: image.label.clone(),
    }
}

fn tool_call_record_key(record: &ToolCallRecord) -> String {
    if let Some(call_id) = record
        .call_id
        .as_ref()
        .filter(|call_id| !call_id.is_empty())
    {
        return format!("call_id:{call_id}");
    }
    serde_json::to_string(record).unwrap_or_else(|_| format!("tool:{}", record.tool))
}
