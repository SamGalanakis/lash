//! Read per-LLM-call system prompts from a full provider trace file.
//!
//! The session `.db` stores the durable conversation thread but not the
//! fully-assembled system prompt sent to the provider on each call —
//! that prompt is rendered just-in-time from the preamble + plugin
//! contributions + chronological history. The required trace JSONL captures
//! every `llm_call_started` event with the full request, including the system
//! block. We pull provider request snapshots out so the HTML exporter can show
//! what the model was actually told on each call.
//!
//! Each JSONL line is a typed [`lash_trace::TraceRecord`]. We deserialize into
//! that schema and pattern-match the typed [`lash_trace::TraceEvent`] variants
//! rather than string-matching raw JSON keys. A line that fails to parse (a
//! malformed record, or a future/unknown event kind) is skipped, not fatal.

use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use lash_trace::{
    TraceContentBlock, TraceContext, TraceEvent, TraceLlmMessage, TraceLlmRequest,
    TraceLlmResponse, TraceRecord, TraceTokenUsage,
};

#[derive(Clone, Debug, serde::Serialize)]
pub struct LlmPromptSnapshot {
    /// Session this call belongs to. Populated from `context.session_id`
    /// in the trace record. Critical for multi-session exports — one trace
    /// file can span root + AgentFrame switch AgentFrames + spawned subagents.
    pub session_id: Option<String>,
    pub turn_index: Option<u64>,
    pub protocol_iteration: Option<u64>,
    pub llm_call_id: Option<String>,
    /// Typed causal source from `context.metadata.caused_by`.
    pub caused_by: Option<lash_core::CausalRef>,
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
    pub cache_read_input_tokens: i64,
    pub cache_write_input_tokens: i64,
    pub reasoning_output_tokens: i64,
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
        // A record that does not deserialize into the typed schema (malformed,
        // or a future/unknown event kind) is silently skipped.
        let Ok(record) = serde_json::from_str::<TraceRecord>(trimmed) else {
            continue;
        };
        match &record.event {
            TraceEvent::LlmCallStarted { request } => {
                if let Some(snapshot) =
                    snapshot_from_started(&record.context, &record.timestamp, request)
                {
                    snapshots.push(snapshot);
                }
            }
            TraceEvent::LlmCallCompleted {
                response, usage, ..
            } => {
                if let Some(usage) = usage_from_completed(response, usage.as_ref()) {
                    attach_usage_to_latest_matching_snapshot(
                        &mut snapshots,
                        record.context.llm_call_id.as_deref(),
                        usage,
                    );
                }
            }
            _ => {}
        }
    }
    Ok(snapshots)
}

fn snapshot_from_started(
    context: &TraceContext,
    timestamp: &str,
    request: &TraceLlmRequest,
) -> Option<LlmPromptSnapshot> {
    let system_text = collect_system_text(&request.messages);
    if system_text.is_empty() {
        return None;
    }
    let system_chars = system_text.chars().count();
    let system_hash = short_hash(&system_text);
    let total_chars = total_message_chars(&request.messages);
    let request_messages = collect_non_system_messages(&request.messages);
    let request_chars: usize = request_messages.iter().map(|m| m.chars).sum();
    let request_concat = request_messages
        .iter()
        .map(|m| m.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let request_hash = short_hash(&request_concat);
    Some(LlmPromptSnapshot {
        session_id: context.session_id.clone(),
        turn_index: context.turn_index.map(|v| v as u64),
        protocol_iteration: context.protocol_iteration.map(|v| v as u64),
        llm_call_id: context.llm_call_id.clone(),
        caused_by: context
            .metadata
            .get("caused_by")
            .and_then(|value| serde_json::from_value(value.clone()).ok()),
        timestamp: Some(timestamp.to_string()),
        model: Some(request.model.clone()),
        model_variant: request.model_variant.clone(),
        system_text,
        system_chars,
        system_hash,
        message_count: request.messages.len(),
        total_chars,
        request_messages,
        request_chars,
        request_hash,
        usage: None,
    })
}

fn usage_from_completed(
    response: &TraceLlmResponse,
    usage: Option<&TraceTokenUsage>,
) -> Option<LlmCallUsage> {
    let usage = usage?;
    let parsed = LlmCallUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_input_tokens: usage.cache_read_input_tokens,
        cache_write_input_tokens: usage.cache_write_input_tokens,
        reasoning_output_tokens: usage.reasoning_output_tokens,
        duration_ms: Some(response.duration_ms),
    };
    if parsed.input_tokens == 0
        && parsed.output_tokens == 0
        && parsed.cache_read_input_tokens == 0
        && parsed.cache_write_input_tokens == 0
        && parsed.reasoning_output_tokens == 0
        && parsed.duration_ms.is_none()
    {
        None
    } else {
        Some(parsed)
    }
}

fn attach_usage_to_latest_matching_snapshot(
    snapshots: &mut [LlmPromptSnapshot],
    call_id: Option<&str>,
    usage: LlmCallUsage,
) {
    if let Some(snapshot) = snapshots.iter_mut().rev().find(|snapshot| {
        snapshot.usage.is_none() && call_id.is_some() && snapshot.llm_call_id.as_deref() == call_id
    }) {
        snapshot.usage = Some(usage);
    }
}

/// Text carried by a content block. Matches the historical extraction, which
/// pulled any block exposing a `text` field — that is `Text` and `Reasoning`
/// blocks (a `ToolResult`'s prose lives under `content`, not `text`).
fn block_text(block: &TraceContentBlock) -> Option<&str> {
    match block {
        TraceContentBlock::Text { text, .. } | TraceContentBlock::Reasoning { text, .. } => {
            Some(text)
        }
        _ => None,
    }
}

fn total_message_chars(messages: &[TraceLlmMessage]) -> usize {
    let mut total = 0usize;
    for message in messages {
        for block in &message.blocks {
            if let Some(text) = block_text(block) {
                total = total.saturating_add(text.chars().count());
            }
        }
    }
    total
}

fn collect_system_text(messages: &[TraceLlmMessage]) -> String {
    let mut out = String::new();
    for message in messages {
        if message.role != "system" {
            continue;
        }
        append_message_text(&message.blocks, &mut out);
    }
    out
}

fn collect_non_system_messages(messages: &[TraceLlmMessage]) -> Vec<RequestMessage> {
    let mut out = Vec::new();
    for message in messages {
        if message.role == "system" {
            continue;
        }
        let mut text = String::new();
        append_message_text(&message.blocks, &mut text);
        if text.is_empty() {
            continue;
        }
        let chars = text.chars().count();
        out.push(RequestMessage {
            role: message.role.clone(),
            text,
            chars,
        });
    }
    out
}

fn append_message_text(blocks: &[TraceContentBlock], out: &mut String) {
    for block in blocks {
        if let Some(text) = block_text(block) {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push_str("\n\n");
            }
            out.push_str(text);
        }
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
    use lash_trace::{TraceContext, TraceLlmMessage, TraceRecord};
    use std::io::Write;

    fn text_block(text: &str) -> TraceContentBlock {
        TraceContentBlock::Text {
            text: text.to_string(),
            cache_breakpoint: false,
        }
    }

    fn message(role: &str, text: &str) -> TraceLlmMessage {
        TraceLlmMessage {
            role: role.to_string(),
            blocks: vec![text_block(text)],
        }
    }

    fn request(messages: Vec<TraceLlmMessage>) -> TraceLlmRequest {
        TraceLlmRequest {
            model: "gpt-5.5".to_string(),
            model_variant: Some("high".to_string()),
            messages,
            attachments: Vec::new(),
            tools: Vec::new(),
            tool_choice: "auto".to_string(),
            output_spec: None,
            stream: true,
        }
    }

    fn write_records(tmp: &mut tempfile::NamedTempFile, records: &[TraceRecord]) {
        for record in records {
            writeln!(
                tmp,
                "{}",
                serde_json::to_string(record).expect("serialize record")
            )
            .unwrap();
        }
        tmp.flush().unwrap();
    }

    #[test]
    fn parses_system_text_and_metadata() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        let started = TraceRecord::new(
            TraceContext {
                turn_index: Some(1),
                protocol_iteration: Some(0),
                turn_id: Some("turn-1".to_string()),
                llm_call_id: Some("abc:1:0:0".to_string()),
                ..Default::default()
            },
            TraceEvent::LlmCallStarted {
                request: request(vec![
                    message("system", "You are lash."),
                    message("user", "hi"),
                ]),
            },
        );
        let completed = TraceRecord::new(
            TraceContext {
                turn_index: Some(1),
                protocol_iteration: Some(0),
                turn_id: Some("turn-1".to_string()),
                llm_call_id: Some("abc:1:0:0".to_string()),
                ..Default::default()
            },
            TraceEvent::LlmCallCompleted {
                response: TraceLlmResponse {
                    text: "ok".to_string(),
                    duration_ms: 1234,
                    terminal_reason: None,
                    parts: None,
                },
                usage: Some(TraceTokenUsage {
                    input_tokens: 100,
                    output_tokens: 12,
                    cache_read_input_tokens: 80,
                    cache_write_input_tokens: 0,
                    reasoning_output_tokens: 4,
                }),
                provider_usage: None,
                stream_summary: None,
            },
        );
        write_records(&mut tmp, &[started, completed]);

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
        assert_eq!(usage.cache_read_input_tokens, 80);
        assert_eq!(usage.cache_write_input_tokens, 0);
        assert_eq!(usage.reasoning_output_tokens, 4);
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
        let started = TraceRecord::new(
            TraceContext::default(),
            TraceEvent::LlmCallStarted {
                request: request(vec![message("user", "hi")]),
            },
        );
        write_records(&mut tmp, &[started]);

        let prompts = load_prompts_from_trace(tmp.path()).unwrap();
        assert!(prompts.is_empty());
    }

    #[test]
    fn skips_malformed_and_unknown_lines() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        // Not JSON at all, a future/unknown event kind, and a valid but
        // non-llm record — all skipped without a snapshot or a hard error.
        writeln!(tmp, "not json").unwrap();
        writeln!(
            tmp,
            r#"{{"schema_version":2,"id":"x","timestamp":"t","context":{{}},"type":"future_event","payload":{{}}}}"#
        )
        .unwrap();
        let started = TraceRecord::new(
            TraceContext::default(),
            TraceEvent::LlmCallStarted {
                request: request(vec![message("system", "sys"), message("user", "hi")]),
            },
        );
        write_records(&mut tmp, &[started]);

        let prompts = load_prompts_from_trace(tmp.path()).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].system_text, "sys");
    }

    #[test]
    fn repeated_llm_call_ids_attach_usage_in_trace_order() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        let mut records = Vec::new();
        for input in [11i64, 22] {
            let call_id = format!("root:1:0:{input}");
            records.push(TraceRecord::new(
                TraceContext {
                    turn_index: Some(1),
                    protocol_iteration: Some(0),
                    llm_call_id: Some(call_id.clone()),
                    ..Default::default()
                },
                TraceEvent::LlmCallStarted {
                    request: request(vec![
                        message("system", &format!("sys {input}")),
                        message("user", "hi"),
                    ]),
                },
            ));
            records.push(TraceRecord::new(
                TraceContext {
                    turn_index: Some(1),
                    protocol_iteration: Some(0),
                    llm_call_id: Some(call_id),
                    ..Default::default()
                },
                TraceEvent::LlmCallCompleted {
                    response: TraceLlmResponse {
                        text: "ok".to_string(),
                        duration_ms: 1,
                        terminal_reason: None,
                        parts: None,
                    },
                    usage: Some(TraceTokenUsage {
                        input_tokens: input,
                        output_tokens: 1,
                        cache_read_input_tokens: 0,
                        cache_write_input_tokens: 0,
                        reasoning_output_tokens: 0,
                    }),
                    provider_usage: None,
                    stream_summary: None,
                },
            ));
        }
        write_records(&mut tmp, &records);

        let prompts = load_prompts_from_trace(tmp.path()).unwrap();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0].usage.as_ref().unwrap().input_tokens, 11);
        assert_eq!(prompts[1].usage.as_ref().unwrap().input_tokens, 22);
    }
}
