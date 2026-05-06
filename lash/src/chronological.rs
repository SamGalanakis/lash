use std::collections::{HashMap, HashSet};

use crate::session_model::message::PartAttachment;
use crate::session_model::{SessionEventRecord, ToolEvent};
use crate::{Message, MessageRole, PartKind, ToolCallRecord};
use lash_rlm_types::{
    RlmAttachmentRef, RlmHistoryItem, RlmHistoryRole, RlmImageRef, RlmModeEvent, RlmTrajectoryEntry,
};

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ChronologicalProjection {
    entries: Vec<ChronologicalEntry>,
    #[serde(skip)]
    tool_call_by_call_id: HashMap<String, usize>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChronologicalEntry {
    pub index: usize,
    pub payload: ChronologicalPayload,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChronologicalPayload {
    Message(Message),
    ToolCall(ToolCallRecord),
    RlmStep(RlmTrajectoryEntry),
}

impl ChronologicalProjection {
    pub(crate) fn from_read_model(read_model: &crate::session_graph::SessionReadModel) -> Self {
        if read_model.active_events.is_empty() {
            return Self::from_transcript(
                read_model.messages.as_slice(),
                read_model.tool_calls.as_slice(),
            );
        }

        let mut projection = Self::from_events_with_capacity(
            read_model.active_events.iter(),
            read_model.active_events.len()
                + read_model.messages.len()
                + read_model.tool_calls.len(),
        );
        let mut seen_messages = projection
            .entries
            .iter()
            .filter_map(|entry| match &entry.payload {
                ChronologicalPayload::Message(message) => Some(message.id.clone()),
                ChronologicalPayload::ToolCall(_) | ChronologicalPayload::RlmStep(_) => None,
            })
            .collect::<HashSet<_>>();
        seen_messages.reserve(read_model.messages.len());
        let mut seen_tool_calls =
            HashSet::with_capacity(projection.entries.len() + read_model.tool_calls.len());
        for entry in &projection.entries {
            match &entry.payload {
                ChronologicalPayload::ToolCall(record) => {
                    seen_tool_calls.insert(tool_call_record_key(record));
                }
                ChronologicalPayload::RlmStep(entry) => {
                    seen_tool_calls.extend(entry.tool_calls.iter().map(tool_call_record_key));
                }
                ChronologicalPayload::Message(_) => {}
            }
        }

        for message in read_model.messages.iter() {
            if !message.is_transient() && seen_messages.insert(message.id.clone()) {
                projection.push(ChronologicalPayload::Message(message.clone()));
            }
        }
        for record in read_model.tool_calls.iter() {
            if seen_tool_calls.insert(tool_call_record_key(record)) {
                projection.push(ChronologicalPayload::ToolCall(record.clone()));
            }
        }
        projection
    }

    fn from_events_with_capacity<'a>(
        events: impl IntoIterator<Item = &'a SessionEventRecord>,
        capacity: usize,
    ) -> Self {
        let mut projection = Self::default();
        projection.entries.reserve(capacity);
        let mut seen_messages = HashSet::new();
        let mut seen_tool_calls = HashSet::new();

        for event in events {
            match event {
                SessionEventRecord::Conversation(record) => {
                    let message = record.to_message();
                    if !message.is_transient() && seen_messages.insert(message.id.clone()) {
                        projection.push(ChronologicalPayload::Message(message));
                    }
                }
                SessionEventRecord::Tool(ToolEvent::Invocation { record, .. }) => {
                    if seen_tool_calls.insert(tool_call_record_key(record)) {
                        projection.push(ChronologicalPayload::ToolCall(record.clone()));
                    }
                }
                SessionEventRecord::Mode(event) => {
                    if let Some(RlmModeEvent::RlmTrajectoryEntry(mut entry)) = event.rlm_event() {
                        entry
                            .tool_calls
                            .retain(|record| seen_tool_calls.insert(tool_call_record_key(record)));
                        projection.push(ChronologicalPayload::RlmStep(entry));
                    }
                }
                SessionEventRecord::StateSnapshot(_) => {}
            }
        }

        projection
    }

    fn from_transcript(messages: &[Message], tool_calls: &[ToolCallRecord]) -> Self {
        let active_messages = messages
            .iter()
            .filter(|message| !message.is_transient())
            .collect::<Vec<_>>();
        let mut first_message_for_call = HashMap::<String, usize>::new();
        for (idx, message) in active_messages.iter().enumerate() {
            for part in message.parts.iter() {
                if let Some(call_id) = &part.tool_call_id {
                    first_message_for_call.entry(call_id.clone()).or_insert(idx);
                }
            }
        }

        let mut anchored = HashMap::<usize, Vec<&ToolCallRecord>>::new();
        for record in tool_calls {
            let anchor = record
                .call_id
                .as_ref()
                .and_then(|call_id| first_message_for_call.get(call_id).copied())
                .unwrap_or_else(|| active_messages.len().saturating_sub(1));
            anchored.entry(anchor).or_default().push(record);
        }

        let mut projection = Self::default();
        for (idx, message) in active_messages.into_iter().enumerate() {
            projection.push(ChronologicalPayload::Message(message.clone()));
            if let Some(records) = anchored.remove(&idx) {
                for record in records {
                    projection.push(ChronologicalPayload::ToolCall(record.clone()));
                }
            }
        }
        projection
    }

    fn push(&mut self, payload: ChronologicalPayload) {
        let index = self.entries.len();
        if let ChronologicalPayload::ToolCall(record) = &payload
            && let Some(call_id) = record
                .call_id
                .as_ref()
                .filter(|call_id| !call_id.is_empty())
        {
            self.tool_call_by_call_id.insert(call_id.clone(), index);
        }
        self.entries.push(ChronologicalEntry { index, payload });
    }

    pub fn entries(&self) -> &[ChronologicalEntry] {
        self.entries.as_slice()
    }

    pub fn into_entries(self) -> Vec<ChronologicalEntry> {
        self.entries
    }

    pub fn tool_call_by_call_id(&self, call_id: &str) -> Option<&ToolCallRecord> {
        self.tool_call_by_call_id
            .get(call_id)
            .and_then(|idx| self.entries.get(*idx))
            .and_then(|entry| match &entry.payload {
                ChronologicalPayload::ToolCall(record) => Some(record),
                ChronologicalPayload::Message(_) | ChronologicalPayload::RlmStep(_) => None,
            })
    }

    pub fn rlm_history(&self) -> Vec<RlmHistoryItem> {
        let mut history = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            match &entry.payload {
                ChronologicalPayload::Message(message) => {
                    if let Some(item) = history_item_from_message(message) {
                        history.push(item);
                    }
                }
                ChronologicalPayload::ToolCall(record) => {
                    history.push(history_item_from_tool_call(record));
                }
                ChronologicalPayload::RlmStep(entry) => {
                    history.push(history_item_from_rlm_step(entry))
                }
            }
        }
        history
    }

    pub fn rlm_history_len(&self) -> usize {
        self.entries.len()
    }

    pub fn rlm_history_item(&self, index: usize) -> Option<RlmHistoryItem> {
        self.entries
            .get(index)
            .and_then(|entry| match &entry.payload {
                ChronologicalPayload::Message(message) => history_item_from_message(message),
                ChronologicalPayload::ToolCall(record) => Some(history_item_from_tool_call(record)),
                ChronologicalPayload::RlmStep(entry) => Some(history_item_from_rlm_step(entry)),
            })
    }

    pub fn rlm_history_value(&self) -> serde_json::Value {
        serde_json::to_value(self.rlm_history())
            .unwrap_or_else(|_| serde_json::Value::Array(vec![]))
    }

    pub(crate) fn rlm_history_len_from_read_model(
        read_model: &crate::session_graph::SessionReadModel,
    ) -> usize {
        if read_model.active_events.is_empty() {
            let active_message_count = read_model
                .messages
                .iter()
                .filter(|message| !message.is_transient())
                .count();
            return if active_message_count == 0 {
                0
            } else {
                active_message_count + read_model.tool_calls.len()
            };
        }

        let mut count = 0usize;
        let mut seen_messages = HashSet::new();
        let mut seen_tool_calls = HashSet::new();
        for event in read_model.active_events.iter() {
            match event {
                SessionEventRecord::Conversation(record) => {
                    let transient = matches!(
                        record.origin,
                        Some(crate::MessageOrigin::Plugin {
                            transient: true,
                            ..
                        })
                    );
                    if !transient && seen_messages.insert(record.id.clone()) {
                        count += 1;
                    }
                }
                SessionEventRecord::Tool(ToolEvent::Invocation { record, .. }) => {
                    if seen_tool_calls.insert(tool_call_record_key(record)) {
                        count += 1;
                    }
                }
                SessionEventRecord::Mode(event) => {
                    if let Some(RlmModeEvent::RlmTrajectoryEntry(entry)) = event.rlm_event() {
                        seen_tool_calls.extend(entry.tool_calls.iter().map(tool_call_record_key));
                        count += 1;
                    }
                }
                SessionEventRecord::StateSnapshot(_) => {}
            }
        }

        for message in read_model.messages.iter() {
            if !message.is_transient() && seen_messages.insert(message.id.clone()) {
                count += 1;
            }
        }
        for record in read_model.tool_calls.iter() {
            if seen_tool_calls.insert(tool_call_record_key(record)) {
                count += 1;
            }
        }
        count
    }
}

pub(crate) fn project_rlm_globals_from_events<'a>(
    events: impl IntoIterator<Item = &'a SessionEventRecord>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut globals = serde_json::Map::new();
    for event in events {
        if let SessionEventRecord::Mode(event) = event
            && let Some(RlmModeEvent::RlmGlobalsPatch(patch)) = event.rlm_event()
        {
            lash_rlm_types::apply_globals_patch(&mut globals, &patch);
        }
    }
    globals
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

fn history_item_from_message(message: &Message) -> Option<RlmHistoryItem> {
    Some(RlmHistoryItem::Message {
        id: message.id.clone(),
        role: history_role(message.role),
        content: message_history_text(message),
        attachments: message
            .parts
            .iter()
            .filter_map(|part| attachment_ref(&part.id, part.attachment.as_ref()))
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
    if let Some(user_input) = message.effective_user_text()
        && !user_input.trim().is_empty()
    {
        return user_input.to_string();
    }
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

fn attachment_ref(id: &str, attachment: Option<&PartAttachment>) -> Option<RlmAttachmentRef> {
    let attachment = attachment?;
    Some(RlmAttachmentRef {
        id: id.to_string(),
        media_type: attachment.reference.media_type,
        label: attachment.reference.label.clone(),
        reference: attachment.reference.id.to_string(),
    })
}

fn image_ref(image: &crate::AttachmentRef) -> RlmImageRef {
    RlmImageRef {
        id: image.id.to_string(),
        media_type: image.media_type,
        width: image.width,
        height: image.height,
        bytes: image.byte_len as usize,
        label: image.label.clone(),
    }
}
