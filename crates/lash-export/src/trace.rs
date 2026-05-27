//! Read per-LLM-call system prompts from a full provider trace file.
//!
//! The session `.db` stores the durable conversation thread but not the
//! fully-assembled system prompt sent to the provider on each call —
//! that prompt is rendered just-in-time from the preamble + plugin
//! contributions + chronological history. The required trace JSONL captures
//! every `llm_call_started` event with the full request, including the system
//! block. We pull provider request snapshots out so the HTML exporter can show
//! what the model was actually told on each call.

use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Clone, Debug, serde::Serialize)]
pub struct LlmPromptSnapshot {
    /// Session this call belongs to. Populated from `context.session_id`
    /// in the trace record. Critical for multi-session exports — one trace
    /// file can span root + handoff successors + spawned subagents.
    pub session_id: Option<String>,
    pub turn_index: Option<u64>,
    pub protocol_iteration: Option<u64>,
    pub llm_call_id: Option<String>,
    /// When this LLM call was issued from inside a tool's
    /// `direct_completion`, carries the originating tool's call id.
    /// The renderer uses this to fold fan-out reranks (tournament_rerank,
    /// llm_query batches, etc.) under their parent tool call.
    pub originating_tool_call_id: Option<String>,
    pub timestamp: Option<String>,
    pub model: Option<String>,
    pub model_variant: Option<String>,
    pub system_text: String,
    pub system_chars: usize,
    pub system_hash: String,
    pub message_count: usize,
    /// Total characters across every message (system + user + history)
    /// in the request — the actual payload size sent to the model.
    pub total_chars: usize,
    /// Each non-system message in the request, in order. In RLM mode
    /// the runtime synthesises a single growing user message containing
    /// the original task plus serialised history of prior iterations;
    /// the export uses this list to render one block per message rather
    /// than concatenating into a wall of text.
    pub request_messages: Vec<RequestMessage>,
    pub request_chars: usize,
    pub request_hash: String,
    pub usage: Option<LlmCallUsage>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LlmCallUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_tokens: i64,
    pub duration_ms: Option<u64>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct RequestMessage {
    pub role: String,
    pub text: String,
    pub chars: usize,
}

/// Parse the trace and extract one snapshot per `llm_call_started` event,
/// preserving file order.
pub fn load_prompts_from_trace(trace_path: &Path) -> Result<Vec<LlmPromptSnapshot>> {
    let file = File::open(trace_path)
        .with_context(|| format!("opening provider trace at {}", trace_path.display()))?;
    let reader = BufReader::new(file);
    let mut snapshots = Vec::new();
    for line in reader.lines().map_while(std::result::Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        match record.get("type").and_then(Value::as_str) {
            Some("llm_call_started") => {
                if let Some(snapshot) = snapshot_from_record(&record) {
                    snapshots.push(snapshot);
                }
            }
            Some("llm_call_completed") => {
                if let Some(usage) = usage_from_completed_record(&record) {
                    attach_usage_to_latest_matching_snapshot(&mut snapshots, &record, usage);
                }
            }
            _ => {}
        }
    }
    Ok(snapshots)
}

fn snapshot_from_record(record: &Value) -> Option<LlmPromptSnapshot> {
    let request = record.get("request")?;
    let messages = request.get("messages").and_then(Value::as_array)?;
    let system_text = collect_system_text(messages);
    if system_text.is_empty() {
        return None;
    }
    let system_chars = system_text.chars().count();
    let system_hash = short_hash(&system_text);
    let total_chars = total_message_chars(messages);
    let request_messages = collect_non_system_messages(messages);
    let request_chars: usize = request_messages.iter().map(|m| m.chars).sum();
    let request_concat = request_messages
        .iter()
        .map(|m| m.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let request_hash = short_hash(&request_concat);
    let context = record.get("context");
    Some(LlmPromptSnapshot {
        session_id: context
            .and_then(|c| c.get("session_id"))
            .and_then(Value::as_str)
            .map(str::to_string),
        turn_index: context
            .and_then(|c| c.get("turn_index"))
            .and_then(Value::as_u64),
        protocol_iteration: context
            .and_then(|c| c.get("protocol_iteration"))
            .and_then(Value::as_u64),
        llm_call_id: context
            .and_then(|c| c.get("llm_call_id"))
            .and_then(Value::as_str)
            .map(str::to_string),
        originating_tool_call_id: context
            .and_then(|c| c.get("originating_tool_call_id"))
            .and_then(Value::as_str)
            .map(str::to_string),
        timestamp: record
            .get("timestamp")
            .and_then(Value::as_str)
            .map(str::to_string),
        model: request
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        model_variant: request
            .get("model_variant")
            .and_then(Value::as_str)
            .map(str::to_string),
        system_text,
        system_chars,
        system_hash,
        message_count: messages.len(),
        total_chars,
        request_messages,
        request_chars,
        request_hash,
        usage: None,
    })
}

fn usage_from_completed_record(record: &Value) -> Option<LlmCallUsage> {
    let usage = record.get("usage")?;
    let parsed = LlmCallUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        output_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        cached_input_tokens: usage
            .get("cached_input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        reasoning_tokens: usage
            .get("reasoning_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        duration_ms: record
            .get("response")
            .and_then(|response| response.get("duration_ms"))
            .and_then(Value::as_u64),
    };
    if parsed.input_tokens == 0
        && parsed.output_tokens == 0
        && parsed.cached_input_tokens == 0
        && parsed.reasoning_tokens == 0
        && parsed.duration_ms.is_none()
    {
        None
    } else {
        Some(parsed)
    }
}

fn attach_usage_to_latest_matching_snapshot(
    snapshots: &mut [LlmPromptSnapshot],
    record: &Value,
    usage: LlmCallUsage,
) {
    let context = record.get("context");
    let call_id = context
        .and_then(|c| c.get("llm_call_id"))
        .and_then(Value::as_str);
    if let Some(snapshot) = snapshots.iter_mut().rev().find(|snapshot| {
        snapshot.usage.is_none() && call_id.is_some() && snapshot.llm_call_id.as_deref() == call_id
    }) {
        snapshot.usage = Some(usage);
    }
}

fn total_message_chars(messages: &[Value]) -> usize {
    let mut total = 0usize;
    for message in messages {
        if let Some(blocks) = message.get("blocks").and_then(Value::as_array) {
            for block in blocks {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    total = total.saturating_add(text.chars().count());
                }
            }
        } else if let Some(text) = message.get("text").and_then(Value::as_str) {
            total = total.saturating_add(text.chars().count());
        }
    }
    total
}

fn collect_system_text(messages: &[Value]) -> String {
    let mut out = String::new();
    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("system") {
            continue;
        }
        append_message_text(message, &mut out);
    }
    out
}

fn collect_non_system_messages(messages: &[Value]) -> Vec<RequestMessage> {
    let mut out = Vec::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if role == "system" {
            continue;
        }
        let mut text = String::new();
        append_message_text(message, &mut text);
        if text.is_empty() {
            continue;
        }
        let chars = text.chars().count();
        out.push(RequestMessage { role, text, chars });
    }
    out
}

fn append_message_text(message: &Value, out: &mut String) {
    if let Some(blocks) = message.get("blocks").and_then(Value::as_array) {
        for block in blocks {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push_str("\n\n");
                }
                out.push_str(text);
            }
        }
    } else if let Some(text) = message.get("text").and_then(Value::as_str) {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push_str("\n\n");
        }
        out.push_str(text);
    }
}

fn short_hash(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_system_text_and_metadata() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        writeln!(
            tmp,
            r#"{{"type":"turn_started","context":{{"turn_index":1,"turn_id":"turn-1"}}}}"#
        )
        .unwrap();
        writeln!(
            tmp,
            r#"{{"type":"llm_call_started","timestamp":"2026-05-04T00:00:00Z","context":{{"turn_index":1,"protocol_iteration":0,"turn_id":"turn-1","llm_call_id":"abc:1:0:0"}},"request":{{"model":"gpt-5.5","model_variant":"high","messages":[{{"role":"system","blocks":[{{"kind":"text","text":"You are lash."}}]}},{{"role":"user","blocks":[{{"kind":"text","text":"hi"}}]}}]}}}}"#
        )
        .unwrap();
        writeln!(
            tmp,
            r#"{{"type":"llm_call_completed","context":{{"turn_index":1,"protocol_iteration":0,"turn_id":"turn-1","llm_call_id":"abc:1:0:0"}},"response":{{"text":"ok","duration_ms":1234}},"usage":{{"input_tokens":100,"output_tokens":12,"cached_input_tokens":80,"reasoning_tokens":4}}}}"#
        )
        .unwrap();
        tmp.flush().unwrap();

        let prompts = load_prompts_from_trace(tmp.path()).unwrap();
        assert_eq!(prompts.len(), 1);
        let p = &prompts[0];
        assert_eq!(p.turn_index, Some(1));
        assert_eq!(p.protocol_iteration, Some(0));
        assert_eq!(p.llm_call_id.as_deref(), Some("abc:1:0:0"));
        assert_eq!(p.model.as_deref(), Some("gpt-5.5"));
        assert_eq!(p.model_variant.as_deref(), Some("high"));
        assert_eq!(p.system_text, "You are lash.");
        assert_eq!(p.system_chars, 13);
        assert_eq!(p.message_count, 2);
        assert_eq!(p.total_chars, 15); // "You are lash." (13) + "hi" (2)
        assert_eq!(p.system_hash.len(), 16);
        assert_eq!(p.request_messages.len(), 1);
        assert_eq!(p.request_messages[0].role, "user");
        assert_eq!(p.request_messages[0].text, "hi");
        assert_eq!(p.request_messages[0].chars, 2);
        assert_eq!(p.request_chars, 2);
        let usage = p.usage.as_ref().expect("usage");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 12);
        assert_eq!(usage.cached_input_tokens, 80);
        assert_eq!(usage.reasoning_tokens, 4);
        assert_eq!(usage.duration_ms, Some(1234));
    }

    #[test]
    fn missing_trace_is_an_error() {
        let err = load_prompts_from_trace(Path::new("/nonexistent/path.trace.jsonl"))
            .expect_err("missing trace should fail");
        assert!(
            err.to_string().contains("opening provider trace"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn skips_records_without_system_block() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        writeln!(
            tmp,
            r#"{{"type":"llm_call_started","request":{{"messages":[{{"role":"user","blocks":[{{"text":"hi"}}]}}]}}}}"#
        )
        .unwrap();
        tmp.flush().unwrap();

        let prompts = load_prompts_from_trace(tmp.path()).unwrap();
        assert!(prompts.is_empty());
    }

    #[test]
    fn repeated_llm_call_ids_attach_usage_in_trace_order() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        for input in [11, 22] {
            writeln!(
                tmp,
                r#"{{"type":"llm_call_started","context":{{"turn_index":1,"protocol_iteration":0,"llm_call_id":"root:1:0:{input}"}},"request":{{"messages":[{{"role":"system","blocks":[{{"text":"sys {input}"}}]}},{{"role":"user","blocks":[{{"text":"hi"}}]}}]}}}}"#
            )
            .unwrap();
            writeln!(
                tmp,
                r#"{{"type":"llm_call_completed","context":{{"turn_index":1,"protocol_iteration":0,"llm_call_id":"root:1:0:{input}"}},"response":{{"duration_ms":1,"text":"ok"}},"usage":{{"input_tokens":{input},"output_tokens":1,"cached_input_tokens":0,"reasoning_tokens":0}}}}"#
            )
            .unwrap();
        }
        tmp.flush().unwrap();

        let prompts = load_prompts_from_trace(tmp.path()).unwrap();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0].usage.as_ref().unwrap().input_tokens, 11);
        assert_eq!(prompts[1].usage.as_ref().unwrap().input_tokens, 22);
    }
}
