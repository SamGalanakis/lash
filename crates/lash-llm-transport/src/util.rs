//! Small shared utilities used across the lash provider crates. These were
//! previously copy-pasted into each provider; they live here because every
//! provider already depends on `lash-llm-transport`.

use lash_sansio::llm::types::{LlmProviderTraceEvent, LlmProviderTraceSender};
use serde_json::Value;

/// Image MIME types accepted by the OpenAI/Anthropic-flavored providers
/// (OpenAI Responses, OpenAI Chat Completions, and Anthropic Messages).
/// Pass to [`lash_core::llm::transport::validate_image_attachments`].
pub const OPENAI_IMAGE_MIMES: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];

/// Forward a raw provider event to the trace sink, deriving an event name from
/// the JSON `type` (or `event`) field when present.
pub fn emit_provider_trace(tx: Option<&LlmProviderTraceSender>, provider: &'static str, raw: &str) {
    let Some(tx) = tx else {
        return;
    };
    let event_name = serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .or_else(|| value.get("event"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "provider_event".to_string());
    tx.send(LlmProviderTraceEvent {
        provider,
        event_name,
        raw: raw.to_string(),
    });
}

/// Coerce a JSON value into an `i64`, accepting both numbers and numeric
/// strings (some providers report token counts as strings). Missing or
/// non-numeric values yield `0`.
pub fn parse_i64(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
        Some(Value::String(s)) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

/// Extract a human-readable error detail from a provider error body. Prefers
/// the `error.message` field of a JSON body; otherwise falls back to the first
/// 200 characters of the raw text. Returns `None` for empty bodies.
pub fn extract_error_detail(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<Value>(trimmed)
        && let Some(msg) = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
    {
        return Some(msg.to_string());
    }
    Some(trimmed.chars().take(200).collect())
}
