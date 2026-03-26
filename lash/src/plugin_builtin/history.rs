use serde_json::json;

use crate::search::truncate_preview;
use crate::store::{HistoryTurnRecord, Store};
use crate::{AssembledTurn, Message, MessageRole, PartKind, ToolResult};

fn history_message_text(msg: &Message) -> String {
    msg.parts
        .iter()
        .filter_map(|part| match part.kind {
            PartKind::Text | PartKind::Prose | PartKind::Code => Some(part.content.as_str()),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn latest_turn_history_payload(turn: &AssembledTurn) -> serde_json::Value {
    let messages = &turn.state.messages;
    let turn_index = messages
        .iter()
        .filter(|msg| matches!(msg.role, MessageRole::User))
        .count() as i64;
    let last_user_idx = messages
        .iter()
        .rposition(|msg| matches!(msg.role, MessageRole::User));

    let user_message = last_user_idx
        .and_then(|idx| messages.get(idx))
        .map(history_message_text)
        .unwrap_or_default();

    let mut prose_parts = Vec::new();
    let mut code_parts = Vec::new();
    if let Some(idx) = last_user_idx {
        for msg in messages.iter().skip(idx + 1) {
            if !matches!(msg.role, MessageRole::Assistant) {
                continue;
            }
            for part in &msg.parts {
                match part.kind {
                    PartKind::Text | PartKind::Prose => {
                        if !part.content.trim().is_empty() {
                            prose_parts.push(part.content.clone());
                        }
                    }
                    PartKind::Code => {
                        if !part.content.trim().is_empty() {
                            code_parts.push(part.content.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let prose = if prose_parts.is_empty() {
        turn.assistant_output.raw_text.clone()
    } else {
        prose_parts.join("\n\n")
    };
    let code = code_parts.join("\n\n");
    let output = turn
        .code_outputs
        .iter()
        .map(|record| match (&record.output, &record.error) {
            (output, Some(error)) if !output.is_empty() && !error.is_empty() => {
                format!("{output}\n{error}")
            }
            (output, _) if !output.is_empty() => output.clone(),
            (_, Some(error)) => error.clone(),
            _ => String::new(),
        })
        .filter(|chunk| !chunk.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let error = turn.errors.first().map(|issue| issue.message.clone());

    serde_json::json!({
        "index": turn_index,
        "user_message": user_message,
        "prose": prose,
        "code": code,
        "output": output,
        "error": error,
        "tool_calls": turn.tool_calls,
    })
}

pub(crate) fn final_history_record(turn: &AssembledTurn) -> HistoryTurnRecord {
    let payload = latest_turn_history_payload(turn);
    HistoryTurnRecord {
        index: payload.get("index").and_then(|v| v.as_i64()).unwrap_or(0),
        user_message: payload
            .get("user_message")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        prose: payload
            .get("prose")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        code: payload
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        output: payload
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        error: payload
            .get("error")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        tool_calls: payload
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
        files_read: Vec::new(),
        files_written: Vec::new(),
    }
}

pub(crate) fn history_summary(store: &Store, session_id: &str, limit: usize) -> ToolResult {
    let turns = store.history_export(session_id);
    let latest_user_message = turns.iter().rev().find_map(|turn| {
        (!turn.user_message.trim().is_empty()).then_some(turn.user_message.clone())
    });
    let recent_turns = turns
        .iter()
        .rev()
        .take(limit.clamp(1, 20))
        .map(|turn| {
            let preview_source = if !turn.user_message.trim().is_empty() {
                &turn.user_message
            } else if !turn.prose.trim().is_empty() {
                &turn.prose
            } else {
                &turn.output
            };
            json!({
                "turn": turn.index,
                "preview": truncate_preview(preview_source, 180),
                "tool_calls": turn.tool_calls.len(),
            })
        })
        .collect::<Vec<_>>();
    ToolResult::ok(json!({
        "session_id": session_id,
        "turn_count": turns.len(),
        "latest_user_message": latest_user_message,
        "recent_turns": recent_turns,
    }))
}
