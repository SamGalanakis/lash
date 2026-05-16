use std::collections::{HashMap, HashSet};

use crate::session_model::{ConversationRecord, ModeEvent, SessionEventRecord, ToolEvent};
use crate::{Message, MessageOrigin, MessageRole, MessageSequence, Part, ToolCallRecord};

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
#[allow(clippy::large_enum_variant)]
pub enum ChronologicalPayload {
    Message(Message),
    ToolCall(ToolCallRecord),
    ModeEvent(crate::session_model::ModeEvent),
}

#[derive(Clone, Copy, Debug)]
pub struct BorrowedChronologicalEntry<'a> {
    pub index: usize,
    pub payload: BorrowedChronologicalPayload<'a>,
}

#[derive(Clone, Copy, Debug)]
pub enum BorrowedChronologicalPayload<'a> {
    Message(BorrowedChronologicalMessage<'a>),
    ToolCall(&'a ToolCallRecord),
    ModeEvent(&'a ModeEvent),
}

#[derive(Clone, Copy, Debug)]
pub struct BorrowedChronologicalMessage<'a> {
    pub id: &'a str,
    pub role: MessageRole,
    pub parts: &'a [Part],
    pub origin: Option<&'a MessageOrigin>,
}

impl<'a> BorrowedChronologicalMessage<'a> {
    fn from_message(message: &'a Message) -> Self {
        Self {
            id: &message.id,
            role: message.role,
            parts: message.parts.as_slice(),
            origin: message.origin.as_ref(),
        }
    }

    fn from_record(record: &'a ConversationRecord) -> Self {
        Self {
            id: &record.id,
            role: record.role,
            parts: record.parts.as_slice(),
            origin: record.origin.as_ref(),
        }
    }

    pub fn is_transient(&self) -> bool {
        matches!(
            self.origin,
            Some(MessageOrigin::Plugin {
                transient: true,
                ..
            })
        )
    }

    fn to_owned(self) -> Message {
        Message {
            id: self.id.to_string(),
            role: self.role,
            parts: std::sync::Arc::new(self.parts.to_vec()),
            origin: self.origin.cloned(),
        }
    }
}

impl ChronologicalProjection {
    pub(crate) fn from_read_model(read_model: &crate::session_graph::SessionReadModel) -> Self {
        Self::from_active_read(
            read_model.active_events.as_slice(),
            read_model.messages.as_slice(),
            read_model.tool_calls.as_slice(),
        )
    }

    pub fn from_turn_view(
        events: &[SessionEventRecord],
        messages: &MessageSequence,
        tool_calls: &[ToolCallRecord],
    ) -> Self {
        Self::from_active_read(events, messages.as_slice(), tool_calls)
    }

    fn from_active_read(
        active_events: &[SessionEventRecord],
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) -> Self {
        let mut projection = Self::default();
        projection.entries.reserve(
            active_events
                .len()
                .saturating_add(messages.len())
                .saturating_add(tool_calls.len()),
        );
        visit_active_read(active_events, messages, tool_calls, |entry| {
            projection.push(match entry.payload {
                BorrowedChronologicalPayload::Message(message) => {
                    ChronologicalPayload::Message(message.to_owned())
                }
                BorrowedChronologicalPayload::ToolCall(record) => {
                    ChronologicalPayload::ToolCall(record.clone())
                }
                BorrowedChronologicalPayload::ModeEvent(event) => {
                    ChronologicalPayload::ModeEvent(event.clone())
                }
            });
        });
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

pub fn visit_turn_view<'a>(
    events: &'a [SessionEventRecord],
    messages: &'a MessageSequence,
    tool_calls: &'a [ToolCallRecord],
    visit: impl FnMut(BorrowedChronologicalEntry<'a>),
) {
    visit_active_read(events, messages.as_slice(), tool_calls, visit);
}

fn visit_active_read<'a>(
    active_events: &'a [SessionEventRecord],
    messages: &'a [Message],
    tool_calls: &'a [ToolCallRecord],
    mut visit: impl FnMut(BorrowedChronologicalEntry<'a>),
) {
    if active_events.is_empty() {
        visit_transcript(messages, tool_calls, visit);
        return;
    }

    let mut index = 0;
    let mut seen_messages = HashSet::new();
    let mut seen_tool_calls = HashSet::new();

    for event in active_events {
        match event {
            SessionEventRecord::Conversation(record) => {
                let message = BorrowedChronologicalMessage::from_record(record);
                if !message.is_transient() && seen_messages.insert(message.id.to_string()) {
                    visit(BorrowedChronologicalEntry {
                        index,
                        payload: BorrowedChronologicalPayload::Message(message),
                    });
                    index += 1;
                }
            }
            SessionEventRecord::Tool(ToolEvent::Invocation { record, .. }) => {
                if seen_tool_calls.insert(tool_call_record_key(record)) {
                    visit(BorrowedChronologicalEntry {
                        index,
                        payload: BorrowedChronologicalPayload::ToolCall(record),
                    });
                    index += 1;
                }
            }
            SessionEventRecord::Mode(event) => {
                visit(BorrowedChronologicalEntry {
                    index,
                    payload: BorrowedChronologicalPayload::ModeEvent(event),
                });
                index += 1;
            }
            SessionEventRecord::StateSnapshot(_) => {}
        }
    }

    seen_messages.reserve(messages.len());
    for message in messages {
        let message = BorrowedChronologicalMessage::from_message(message);
        if !message.is_transient() && seen_messages.insert(message.id.to_string()) {
            visit(BorrowedChronologicalEntry {
                index,
                payload: BorrowedChronologicalPayload::Message(message),
            });
            index += 1;
        }
    }
    for record in tool_calls {
        if seen_tool_calls.insert(tool_call_record_key(record)) {
            visit(BorrowedChronologicalEntry {
                index,
                payload: BorrowedChronologicalPayload::ToolCall(record),
            });
            index += 1;
        }
    }
}

fn visit_transcript<'a>(
    messages: &'a [Message],
    tool_calls: &'a [ToolCallRecord],
    mut visit: impl FnMut(BorrowedChronologicalEntry<'a>),
) {
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

    let mut index = 0;
    for (message_idx, message) in active_messages.into_iter().enumerate() {
        visit(BorrowedChronologicalEntry {
            index,
            payload: BorrowedChronologicalPayload::Message(
                BorrowedChronologicalMessage::from_message(message),
            ),
        });
        index += 1;
        if let Some(records) = anchored.remove(&message_idx) {
            for record in records {
                visit(BorrowedChronologicalEntry {
                    index,
                    payload: BorrowedChronologicalPayload::ToolCall(record),
                });
                index += 1;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_model::ConversationRecord;
    use crate::{ExecutionMode, PartKind, PruneState, shared_parts};

    fn text_message(id: &str, role: MessageRole, text: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: shared_parts(vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: text.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]),
            origin: None,
        }
    }

    fn tool_result_message(id: &str, call_id: &str) -> Message {
        let mut message = text_message(id, MessageRole::User, "tool result");
        std::sync::Arc::make_mut(&mut message.parts)[0].tool_call_id = Some(call_id.to_string());
        message
    }

    fn transient_message(id: &str) -> Message {
        let mut message = text_message(id, MessageRole::System, "transient");
        message.origin = Some(MessageOrigin::Plugin {
            plugin_id: "test".to_string(),
            transient: true,
        });
        message
    }

    fn tool_call(call_id: Option<&str>, tool: &str) -> ToolCallRecord {
        ToolCallRecord {
            call_id: call_id.map(str::to_string),
            tool: tool.to_string(),
            args: serde_json::json!({ "tool": tool }),
            output: crate::ToolCallOutput::success(serde_json::json!({ "ok": true })),
            duration_ms: 7,
        }
    }

    fn mode_event(value: &str) -> ModeEvent {
        ModeEvent {
            mode_id: ExecutionMode::new("rlm"),
            payload: serde_json::json!({ "value": value }),
        }
    }

    fn owned_summary(projection: &ChronologicalProjection) -> Vec<String> {
        projection
            .entries()
            .iter()
            .map(|entry| match &entry.payload {
                ChronologicalPayload::Message(message) => {
                    format!("{}:message:{}", entry.index, message.id)
                }
                ChronologicalPayload::ToolCall(record) => {
                    format!("{}:tool:{}", entry.index, tool_call_record_key(record))
                }
                ChronologicalPayload::ModeEvent(event) => {
                    format!("{}:mode:{}", entry.index, event.payload)
                }
            })
            .collect()
    }

    fn borrowed_summary(
        events: &[SessionEventRecord],
        messages: &MessageSequence,
        tool_calls: &[ToolCallRecord],
    ) -> Vec<String> {
        let mut summary = Vec::new();
        visit_turn_view(events, messages, tool_calls, |entry| {
            summary.push(match entry.payload {
                BorrowedChronologicalPayload::Message(message) => {
                    format!("{}:message:{}", entry.index, message.id)
                }
                BorrowedChronologicalPayload::ToolCall(record) => {
                    format!("{}:tool:{}", entry.index, tool_call_record_key(record))
                }
                BorrowedChronologicalPayload::ModeEvent(event) => {
                    format!("{}:mode:{}", entry.index, event.payload)
                }
            });
        });
        summary
    }

    #[test]
    fn borrowed_turn_view_matches_owned_active_event_projection() {
        let m1 = text_message("m1", MessageRole::User, "first");
        let m2 = text_message("m2", MessageRole::Assistant, "second");
        let t1 = tool_call(Some("call-1"), "search");
        let t2 = tool_call(Some("call-2"), "read_file");
        let events = vec![
            SessionEventRecord::Conversation(ConversationRecord::from_message(m1.clone())),
            SessionEventRecord::Tool(ToolEvent::Invocation {
                stable_key: "tool-1".to_string(),
                record: t1.clone(),
            }),
            SessionEventRecord::Mode(mode_event("step")),
            SessionEventRecord::Conversation(ConversationRecord::from_message(m1.clone())),
            SessionEventRecord::Tool(ToolEvent::Invocation {
                stable_key: "tool-1-duplicate".to_string(),
                record: t1.clone(),
            }),
        ];
        let messages = MessageSequence::from_owned(vec![m1, m2]);
        let tool_calls = vec![t1, t2];
        let projection = ChronologicalProjection::from_turn_view(&events, &messages, &tool_calls);

        assert_eq!(
            borrowed_summary(&events, &messages, &tool_calls),
            owned_summary(&projection)
        );
    }

    #[test]
    fn borrowed_turn_view_matches_owned_transcript_fallback_projection() {
        let messages = MessageSequence::from_owned(vec![
            tool_result_message("m1", "call-1"),
            transient_message("transient"),
            text_message("m2", MessageRole::Assistant, "second"),
        ]);
        let tool_calls = vec![
            tool_call(Some("call-1"), "search"),
            tool_call(Some("call-2"), "read_file"),
        ];
        let events = Vec::new();
        let projection = ChronologicalProjection::from_turn_view(&events, &messages, &tool_calls);

        assert_eq!(
            borrowed_summary(&events, &messages, &tool_calls),
            owned_summary(&projection)
        );
    }
}
