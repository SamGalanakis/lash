#![allow(clippy::result_large_err)]

//! Provider-neutral response-normalization primitives shared by the lash
//! provider crates. Each of these existed in 2–4 near-identical copies across
//! the OpenAI, Anthropic, Google, and Codex crates; this is the single
//! canonical implementation every provider now calls.
//!
//! These are deliberately small, stateless helpers. The cross-provider
//! conformance suite ([`crate::conformance`]) pins the normalized-output
//! contract, so there is no normalizer trait here — providers wire their own
//! wire-format parsing to these primitives directly.

use lash_core::LlmTransportError;
use lash_core::provider::ProviderOptions;
use lash_sansio::llm::types::{LlmOutputPart, LlmTerminalReason, LlmUsage};

/// Merge incremental streaming usage: overwrite each `dst` field with the
/// corresponding `next` field only when `next` reports a non-zero value,
/// keeping prior non-zero counters when a later event reports only a subset.
///
/// Streaming usage arrives incrementally and later events may carry only some
/// counters; this keeps the earlier ones rather than clobbering them with
/// zeros.
pub fn merge_usage(dst: &mut LlmUsage, next: &LlmUsage) {
    if next.input_tokens > 0 {
        dst.input_tokens = next.input_tokens;
    }
    if next.output_tokens > 0 {
        dst.output_tokens = next.output_tokens;
    }
    if next.cached_input_tokens > 0 {
        dst.cached_input_tokens = next.cached_input_tokens;
    }
    if next.reasoning_tokens > 0 {
        dst.reasoning_tokens = next.reasoning_tokens;
    }
}

/// Terminal reason derived from assembled parts alone: `ToolUse` when any tool
/// call is present, otherwise `Stop`. Used as the fallback when a provider's
/// wire response carries no recognizable finish reason.
pub fn terminal_reason_from_parts(parts: &[LlmOutputPart]) -> LlmTerminalReason {
    if parts
        .iter()
        .any(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
    {
        LlmTerminalReason::ToolUse
    } else {
        LlmTerminalReason::Stop
    }
}

/// Frame a buffered SSE payload — possibly several `data:`/`event:` lines
/// separated by blank lines — into individual event strings, invoking
/// `on_event` once per complete event. Each event's `data:` lines are joined
/// with `\n`; `event:` lines are ignored; a blank line terminates the current
/// event. A trailing event without a closing blank line is still flushed.
///
/// This frames an already-buffered SSE blob (e.g. a non-streaming response
/// body that nonetheless carries SSE text), as opposed to
/// [`crate::streaming::drive_sse_response`], which frames a live byte stream.
pub fn frame_sse_payload<F>(payload: &str, mut on_event: F) -> Result<(), LlmTransportError>
where
    F: FnMut(&str) -> Result<(), LlmTransportError>,
{
    let mut event_lines: Vec<String> = Vec::new();
    for mut line in payload.lines().map(str::to_string) {
        if line.ends_with('\r') {
            line.pop();
        }
        if let Some(data) = line.strip_prefix("data:") {
            event_lines.push(data.trim().to_string());
            continue;
        }
        if line.starts_with("event:") {
            continue;
        }
        if line.trim().is_empty() && !event_lines.is_empty() {
            let raw = event_lines.join("\n");
            on_event(&raw)?;
            event_lines.clear();
        }
    }
    if !event_lines.is_empty() {
        let raw = event_lines.join("\n");
        on_event(&raw)?;
    }
    Ok(())
}

/// Build the canonical HTTP-error envelope every provider returns for a non-2xx
/// response: an [`LlmTransportError`] carrying the status code, the response
/// headers, and the raw body, plus the originating request body when available.
///
/// `message` is the provider-specific human-readable summary (e.g.
/// `"Anthropic request failed with 429: rate limited"`); retryability is left
/// to the central provider failure classifier, which reads the attached status.
pub fn http_error_envelope(
    message: impl Into<String>,
    status: u16,
    headers: &reqwest::header::HeaderMap,
    raw_body: impl Into<String>,
    request_body: Option<String>,
) -> LlmTransportError {
    let mut err = LlmTransportError::new(message)
        .with_status(status)
        .with_headers(crate::timeouts::header_pairs(headers))
        .with_raw(raw_body);
    if let Some(request_body) = request_body {
        err = err.with_request_body(request_body);
    }
    err
}

/// Append the `options` tail to a provider's `serialize_config` map: emit the
/// `options` key only when the options differ from the default, matching what
/// each provider's `serialize_config` hand-rolled.
pub fn serialize_options_tail(
    map: &mut serde_json::Map<String, serde_json::Value>,
    options: &ProviderOptions,
) {
    if !options.is_default() {
        map.insert(
            "options".to_string(),
            serde_json::to_value(options).unwrap_or(serde_json::Value::Null),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_usage_keeps_prior_nonzero_when_next_is_zero() {
        let mut dst = LlmUsage {
            input_tokens: 100,
            ..LlmUsage::default()
        };
        merge_usage(
            &mut dst,
            &LlmUsage {
                output_tokens: 20,
                ..LlmUsage::default()
            },
        );
        assert_eq!(dst.input_tokens, 100);
        assert_eq!(dst.output_tokens, 20);
    }

    #[test]
    fn terminal_reason_from_parts_prefers_tool_use() {
        let tool = vec![LlmOutputPart::ToolCall {
            call_id: "c".into(),
            tool_name: "t".into(),
            input_json: "{}".into(),
            replay: None,
        }];
        assert_eq!(
            terminal_reason_from_parts(&tool),
            LlmTerminalReason::ToolUse
        );
        assert_eq!(terminal_reason_from_parts(&[]), LlmTerminalReason::Stop);
    }

    #[test]
    fn frame_sse_payload_splits_events_and_joins_data_lines() {
        let mut events = Vec::new();
        frame_sse_payload("event: x\ndata: a\ndata: b\n\ndata: c\n", |e| {
            events.push(e.to_string());
            Ok(())
        })
        .unwrap();
        assert_eq!(events, vec!["a\nb".to_string(), "c".to_string()]);
    }

    #[test]
    fn serialize_options_tail_omits_default() {
        let mut map = serde_json::Map::new();
        serialize_options_tail(&mut map, &ProviderOptions::default());
        assert!(map.is_empty());
    }
}
