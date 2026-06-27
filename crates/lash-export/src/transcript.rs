use std::collections::HashSet;

use lash_core::session_model::{Message, MessageRole, PartKind, ProtocolEvent};
use lash_core::{ChronologicalEntry, ChronologicalPayload, ChronologicalProjection};

#[derive(Clone, Debug, PartialEq)]
pub struct LashlangTranscriptStep {
    pub id: String,
    pub protocol_iteration: usize,
    pub code: String,
    pub output: Vec<String>,
    pub error: Option<String>,
    pub final_output: Option<serde_json::Value>,
}

impl LashlangTranscriptStep {
    pub fn output_chars(&self) -> usize {
        self.output.iter().map(|item| item.chars().count()).sum()
    }
}

#[derive(Clone, Debug)]
pub struct TranscriptEntry<'a> {
    pub index: usize,
    pub kind: TranscriptEntryKind<'a>,
}

#[derive(Clone, Debug)]
pub enum TranscriptEntryKind<'a> {
    Message(&'a Message),
    AssistantReasoning(String),
    AssistantText(String),
    LashlangStep(LashlangTranscriptStep),
}

pub fn projection_transcript_entries<'a>(
    projection: &'a ChronologicalProjection,
) -> Vec<TranscriptEntry<'a>> {
    chronological_transcript_entries(projection.entries())
}

pub fn chronological_transcript_entries<'a>(
    chronological: &'a [ChronologicalEntry],
) -> Vec<TranscriptEntry<'a>> {
    let suppressed = suppressed_rlm_final_output_message_ids(chronological);
    chronological
        .iter()
        .filter(|entry| match &entry.payload {
            ChronologicalPayload::Message(message) => !suppressed.contains(&message.id),
            ChronologicalPayload::ProtocolEvent(_) => true,
        })
        .flat_map(project_chronological_entries)
        .collect()
}

pub fn project_chronological_entries(entry: &ChronologicalEntry) -> Vec<TranscriptEntry<'_>> {
    match &entry.payload {
        ChronologicalPayload::Message(message) => project_message_entries(entry.index, message),
        ChronologicalPayload::ProtocolEvent(event) => {
            rlm_transcript_entries_from_event(entry.index, event)
        }
    }
}

fn project_message_entries<'a>(index: usize, message: &'a Message) -> Vec<TranscriptEntry<'a>> {
    if !matches!(message.role, MessageRole::Assistant) {
        return vec![TranscriptEntry {
            index,
            kind: TranscriptEntryKind::Message(message),
        }];
    }

    let simple_assistant_parts = message.parts.iter().all(|part| {
        matches!(
            part.kind,
            PartKind::Reasoning | PartKind::Text | PartKind::Prose
        ) && part.attachment.is_none()
            && part.tool_call_id.is_none()
            && part.tool_name.is_none()
    });
    if !simple_assistant_parts {
        return vec![TranscriptEntry {
            index,
            kind: TranscriptEntryKind::Message(message),
        }];
    }

    message
        .parts
        .iter()
        .filter_map(|part| match part.kind {
            PartKind::Reasoning => display_text(&part.content).map(|text| TranscriptEntry {
                index,
                kind: TranscriptEntryKind::AssistantReasoning(text),
            }),
            PartKind::Text | PartKind::Prose => {
                display_text(&part.content).map(|text| TranscriptEntry {
                    index,
                    kind: TranscriptEntryKind::AssistantText(text),
                })
            }
            _ => None,
        })
        .collect()
}

fn rlm_transcript_entries_from_event(
    index: usize,
    event: &ProtocolEvent,
) -> Vec<TranscriptEntry<'_>> {
    let step = match lash_protocol_rlm::decode_rlm_protocol_event(event) {
        Some(lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(step)) => step,
        Some(_) | None => return Vec::new(),
    };
    vec![TranscriptEntry {
        index,
        kind: TranscriptEntryKind::LashlangStep(LashlangTranscriptStep {
            id: step.id,
            protocol_iteration: step.protocol_iteration,
            code: step.code,
            output: step.output,
            error: step.error,
            final_output: step.final_output,
        }),
    }]
}

pub fn lashlang_transcript_step_from_event(
    event: &ProtocolEvent,
) -> Option<LashlangTranscriptStep> {
    let step = match lash_protocol_rlm::decode_rlm_protocol_event(event)? {
        lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(step) => step,
        _ => return None,
    };
    Some(LashlangTranscriptStep {
        id: step.id,
        protocol_iteration: step.protocol_iteration,
        code: step.code,
        output: step.output,
        error: step.error,
        final_output: step.final_output,
    })
}

pub fn display_text(text: &str) -> Option<String> {
    let cleaned = text.trim().to_string();
    (!cleaned.is_empty()).then_some(cleaned)
}

pub fn suppressed_rlm_final_output_message_ids(
    chronological: &[ChronologicalEntry],
) -> HashSet<String> {
    let mut suppressed = HashSet::new();
    let mut last_final_output: Option<String> = None;
    for entry in chronological {
        match &entry.payload {
            ChronologicalPayload::ProtocolEvent(event) => {
                last_final_output = lashlang_transcript_step_from_event(event)
                    .and_then(|step| step.final_output)
                    .map(|value| final_value_match_text(&value));
            }
            ChronologicalPayload::Message(message) => {
                if matches!(message.role, MessageRole::Assistant)
                    && let Some(prev) = last_final_output.as_deref()
                    && message_matches_text(message, prev)
                {
                    suppressed.insert(message.id.clone());
                }
                last_final_output = None;
            }
        }
    }
    suppressed
}

pub fn message_matches_text(message: &Message, expected: &str) -> bool {
    let expected = expected.trim();
    if expected.is_empty() {
        return false;
    }
    let collected: String = message
        .parts
        .iter()
        .filter(|p| matches!(p.kind, PartKind::Text | PartKind::Prose))
        .map(|p| p.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    collected.trim() == expected
}

pub fn final_value_display_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(text) => text.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

pub fn final_value_match_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

pub fn format_count(n: u64) -> String {
    if n < 1024 {
        format!("{n}b")
    } else if n < 1024 * 1024 {
        format!("{:.1}kb", n as f64 / 1024.0)
    } else if n < 1024 * 1024 * 1024 {
        format!("{:.1}mb", n as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2}gb", n as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

pub fn format_tokens(n: i64) -> String {
    let n = n.max(0) as f64;
    if n < 1_000.0 {
        format!("{}", n as u64)
    } else if n < 1_000_000.0 {
        format!("{:.1}k", n / 1_000.0)
    } else {
        format!("{:.2}m", n / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lash_core::session_model::{Part, PruneState};

    use super::*;

    fn text_message(id: &str, role: MessageRole, text: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: Arc::new(vec![Part {
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

    fn assistant_message(id: &str, reasoning: &str, text: &str) -> Message {
        Message {
            id: id.to_string(),
            role: MessageRole::Assistant,
            parts: Arc::new(vec![
                Part {
                    id: format!("{id}.r"),
                    kind: PartKind::Reasoning,
                    content: reasoning.to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                },
                Part {
                    id: format!("{id}.t"),
                    kind: PartKind::Text,
                    content: text.to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                },
            ]),
            origin: None,
        }
    }

    fn rlm_payload(step: lash_rlm_types::RlmTrajectoryEntry) -> ChronologicalPayload {
        ChronologicalPayload::ProtocolEvent(lash_protocol_rlm::rlm_protocol_event(
            lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(step),
        ))
    }

    fn lashlang_step(
        final_output: Option<serde_json::Value>,
    ) -> lash_rlm_types::RlmTrajectoryEntry {
        lash_rlm_types::RlmTrajectoryEntry {
            id: "step-1".to_string(),
            protocol_iteration: 3,
            code: "answer = 1".to_string(),
            output: vec!["1".to_string()],
            images: Vec::new(),
            error: None,
            final_output,
        }
    }

    #[test]
    fn projects_rlm_trajectory_as_generic_assistant_entries_then_lashlang_step() {
        let chronological = vec![
            ChronologicalEntry {
                index: 0,
                payload: ChronologicalPayload::Message(assistant_message(
                    "a1",
                    "think",
                    "visible note",
                )),
            },
            ChronologicalEntry {
                index: 1,
                payload: rlm_payload(lashlang_step(None)),
            },
        ];

        let entries = chronological_transcript_entries(&chronological);

        assert_eq!(entries.len(), 3);
        assert!(matches!(
            &entries[0].kind,
            TranscriptEntryKind::AssistantReasoning(text) if text == "think"
        ));
        assert!(matches!(
            &entries[1].kind,
            TranscriptEntryKind::AssistantText(text) if text == "visible note"
        ));
        let TranscriptEntryKind::LashlangStep(step) = &entries[2].kind else {
            panic!("expected lashlang step");
        };
        assert_eq!(step.protocol_iteration, 3);
        assert_eq!(step.code, "answer = 1");
    }

    #[test]
    fn assistant_entries_from_rlm_do_not_duplicate_lashlang_code() {
        let mut step = lashlang_step(None);
        step.code = "secret = await tools.hidden({})?\nprint secret".to_string();
        let chronological = vec![
            ChronologicalEntry {
                index: 0,
                payload: ChronologicalPayload::Message(assistant_message(
                    "a1",
                    "think",
                    "visible before\n\nvisible after",
                )),
            },
            ChronologicalEntry {
                index: 1,
                payload: rlm_payload(step),
            },
        ];

        let entries = chronological_transcript_entries(&chronological);

        for entry in &entries {
            match &entry.kind {
                TranscriptEntryKind::AssistantReasoning(text)
                | TranscriptEntryKind::AssistantText(text) => {
                    assert!(!text.contains("secret = await"));
                    assert!(!text.contains("<lashlang>"));
                }
                TranscriptEntryKind::LashlangStep(step) => {
                    assert!(step.code.contains("secret = await"));
                }
                TranscriptEntryKind::Message(_) => {}
            }
        }
    }

    #[test]
    fn suppresses_assistant_echo_after_rlm_final_output() {
        let chronological = vec![
            ChronologicalEntry {
                index: 0,
                payload: rlm_payload(lashlang_step(Some(serde_json::json!("done")))),
            },
            ChronologicalEntry {
                index: 1,
                payload: ChronologicalPayload::Message(text_message(
                    "assistant-echo",
                    MessageRole::Assistant,
                    "done",
                )),
            },
        ];

        let suppressed = suppressed_rlm_final_output_message_ids(&chronological);

        assert!(suppressed.contains("assistant-echo"));
    }
}
