use std::collections::{HashMap, HashSet};

use crate::session_model::{SessionEventRecord, ToolEvent};
use crate::{Message, ToolCallRecord};

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
    ModeEvent(crate::session_model::ModeEvent),
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
                ChronologicalPayload::ToolCall(_) | ChronologicalPayload::ModeEvent(_) => None,
            })
            .collect::<HashSet<_>>();
        seen_messages.reserve(read_model.messages.len());
        let mut seen_tool_calls =
            HashSet::with_capacity(projection.entries.len() + read_model.tool_calls.len());
        for entry in &projection.entries {
            if let ChronologicalPayload::ToolCall(record) = &entry.payload {
                seen_tool_calls.insert(tool_call_record_key(record));
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
                    projection.push(ChronologicalPayload::ModeEvent(event.clone()));
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
                ChronologicalPayload::Message(_) | ChronologicalPayload::ModeEvent(_) => None,
            })
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
