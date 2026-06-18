use std::collections::HashSet;

use lash_core::session_model::{Message, MessageRole, PartKind, ProtocolEvent};
use lash_core::{ChronologicalEntry, ChronologicalPayload, ChronologicalProjection};

#[derive(Clone, Debug, PartialEq)]
pub struct RlmTranscriptStep {
    pub id: String,
    pub protocol_iteration: usize,
    pub reasoning: Option<String>,
    pub code: String,
    pub output: Vec<String>,
    pub error: Option<String>,
    pub final_output: Option<serde_json::Value>,
}

impl RlmTranscriptStep {
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
    RlmStep(RlmTranscriptStep),
}

pub fn projection_transcript_entries<'a>(
    projection: &'a ChronologicalProjection,
) -> Vec<TranscriptEntry<'a>> {
    chronological_transcript_entries(projection.entries())
}

pub fn chronological_transcript_entries<'a>(
    chronological: &'a [ChronologicalEntry],
) -> Vec<TranscriptEntry<'a>> {
    chronological
        .iter()
        .filter_map(project_chronological_entry)
        .collect()
}

pub fn project_chronological_entry(entry: &ChronologicalEntry) -> Option<TranscriptEntry<'_>> {
    let kind = match &entry.payload {
        ChronologicalPayload::Message(message) => TranscriptEntryKind::Message(message),
        ChronologicalPayload::ProtocolEvent(event) => {
            TranscriptEntryKind::RlmStep(rlm_transcript_step_from_event(event)?)
        }
    };
    Some(TranscriptEntry {
        index: entry.index,
        kind,
    })
}

pub fn rlm_transcript_step_from_event(event: &ProtocolEvent) -> Option<RlmTranscriptStep> {
    let step = match lash_protocol_rlm::decode_rlm_protocol_event(event)? {
        lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(step) => step,
        _ => return None,
    };
    let reasoning = rlm_reasoning_display_text(&step.reasoning);
    Some(RlmTranscriptStep {
        id: step.id,
        protocol_iteration: step.protocol_iteration,
        reasoning,
        code: step.code,
        output: step.output,
        error: step.error,
        final_output: step.final_output,
    })
}

pub fn rlm_reasoning_display_text(text: &str) -> Option<String> {
    let cleaned = strip_first_lashlang_fence(text).trim().to_string();
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
                last_final_output = rlm_transcript_step_from_event(event)
                    .and_then(|step| step.final_output)
                    .map(|value| submitted_value_match_text(&value));
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

pub fn submitted_value_display_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(text) => text.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

pub fn submitted_value_match_text(value: &serde_json::Value) -> String {
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

pub fn strip_first_lashlang_fence(text: &str) -> String {
    let Some(open_rel) = text.find("```") else {
        return text.to_string();
    };
    let opener_len = text.as_bytes()[open_rel..]
        .iter()
        .take_while(|&&b| b == b'`')
        .count();
    let after_open = open_rel + opener_len;
    let rest = &text[after_open..];
    let Some(lang_end_rel) = rest.find('\n') else {
        return text[..open_rel].to_string();
    };
    let lang = rest[..lang_end_rel].trim();
    if !matches!(lang, "lashlang" | "rlm" | "lash") {
        return text.to_string();
    }
    let body_start = after_open + lang_end_rel + 1;
    let body_bytes = &text.as_bytes()[body_start..];
    let mut close = text.len();
    let mut consumed = 0usize;
    let mut i = 0;
    while i < body_bytes.len() {
        if body_bytes[i] == b'`' {
            let start = i;
            while i < body_bytes.len() && body_bytes[i] == b'`' {
                i += 1;
            }
            if i - start >= opener_len {
                close = body_start + start;
                consumed = opener_len;
                break;
            }
        } else {
            i += 1;
        }
    }
    let after_close = (close + consumed).min(text.len());
    let mut out = String::new();
    out.push_str(text[..open_rel].trim_end());
    let tail = text[after_close..].trim_start();
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(tail);
    }
    out
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

    fn rlm_payload(step: lash_rlm_types::RlmTrajectoryEntry) -> ChronologicalPayload {
        ChronologicalPayload::ProtocolEvent(lash_protocol_rlm::rlm_protocol_event(
            lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(step),
        ))
    }

    fn rlm_step(final_output: Option<serde_json::Value>) -> lash_rlm_types::RlmTrajectoryEntry {
        lash_rlm_types::RlmTrajectoryEntry {
            id: "step-1".to_string(),
            protocol_iteration: 3,
            reasoning: "think\n```lashlang\nanswer = 1\n```\nthen submit".to_string(),
            code: "answer = 1".to_string(),
            output: vec!["1".to_string()],
            images: Vec::new(),
            error: None,
            final_output,
        }
    }

    #[test]
    fn strips_first_lashlang_fence_without_losing_surrounding_reasoning() {
        assert_eq!(
            strip_first_lashlang_fence("before\n````lashlang\nx = `1`\n````\nafter"),
            "before\n\nafter"
        );
        assert_eq!(
            strip_first_lashlang_fence("before\n```rust\nx\n```\nafter"),
            "before\n```rust\nx\n```\nafter"
        );
    }

    #[test]
    fn projects_rlm_steps_with_display_reasoning() {
        let chronological = vec![ChronologicalEntry {
            index: 0,
            payload: rlm_payload(rlm_step(None)),
        }];

        let entries = chronological_transcript_entries(&chronological);

        assert_eq!(entries.len(), 1);
        let TranscriptEntryKind::RlmStep(step) = &entries[0].kind else {
            panic!("expected rlm step");
        };
        assert_eq!(step.protocol_iteration, 3);
        assert_eq!(step.reasoning.as_deref(), Some("think\n\nthen submit"));
        assert_eq!(step.code, "answer = 1");
    }

    #[test]
    fn suppresses_assistant_echo_after_rlm_final_output() {
        let chronological = vec![
            ChronologicalEntry {
                index: 0,
                payload: rlm_payload(rlm_step(Some(serde_json::json!("done")))),
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
