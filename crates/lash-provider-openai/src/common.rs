use serde_json::{Value, json};
use std::sync::LazyLock;

use lash_llm_transport::timeouts::build_http_client;
use lash_openai_schema::emit_provider_trace;

pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub(crate) const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 32_768;

pub(crate) const OPENROUTER_REASONING_VARIANTS: &[&str] =
    &["none", "minimal", "low", "medium", "high", "xhigh"];

pub(crate) static DEFAULT_HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(build_http_client);

pub(crate) fn base_url_is_openrouter(base_url: &str) -> bool {
    base_url.trim_end_matches('/') == OPENROUTER_BASE_URL
}

pub(crate) fn sanitize_surrogates(s: &str) -> &str {
    s
}

pub(crate) fn emit_provider_request_trace(
    provider_trace: Option<&lash_core::llm::types::LlmProviderTraceSender>,
    endpoint: &str,
    body: &Value,
) {
    let raw = json!({
        "type": "openai_compatible.request_body",
        "endpoint": endpoint,
        "body": body,
    });
    if let Ok(raw) = serde_json::to_string(&raw) {
        emit_provider_trace(provider_trace, "openai_compatible", &raw);
    }
}

pub(crate) fn extract_error_detail(raw: &str) -> Option<String> {
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

pub(crate) fn parse_i64(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
        Some(Value::String(s)) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

pub(crate) fn usage_from_response_value(value: &Value) -> lash_core::llm::types::LlmUsage {
    usage_from_usage_value(value.get("usage").unwrap_or(&Value::Null))
}

pub(crate) fn usage_from_usage_value(usage: &Value) -> lash_core::llm::types::LlmUsage {
    let cached_tokens = parse_i64(
        usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .or_else(|| {
                usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
            })
            .or_else(|| usage.get("prompt_cache_hit_tokens")),
    );
    let cache_write_tokens = parse_i64(
        usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cache_write_tokens"))
            .or_else(|| {
                usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cache_write_tokens"))
            }),
    );
    lash_core::llm::types::LlmUsage {
        input_tokens: parse_i64(
            usage
                .get("input_tokens")
                .or_else(|| usage.get("prompt_tokens")),
        ),
        output_tokens: parse_i64(
            usage
                .get("output_tokens")
                .or_else(|| usage.get("completion_tokens")),
        ),
        cached_input_tokens: if cache_write_tokens > 0 {
            cached_tokens.saturating_sub(cache_write_tokens).max(0)
        } else {
            cached_tokens
        },
        reasoning_tokens: parse_i64(
            usage
                .get("output_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
                .or_else(|| {
                    usage
                        .get("completion_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                }),
        ),
    }
}

pub(crate) fn merge_usage(
    dst: &mut lash_core::llm::types::LlmUsage,
    src: &lash_core::llm::types::LlmUsage,
) {
    if src != &lash_core::llm::types::LlmUsage::default() {
        *dst = src.clone();
    }
}

pub(crate) fn has_response_content(parts: &[lash_core::llm::types::LlmOutputPart]) -> bool {
    parts.iter().any(|part| match part {
        lash_core::llm::types::LlmOutputPart::Text { text, .. } => !text.is_empty(),
        lash_core::llm::types::LlmOutputPart::Reasoning { .. } => true,
        lash_core::llm::types::LlmOutputPart::ToolCall { .. } => true,
    })
}

pub(crate) fn terminal_reason_from_parts(
    parts: &[lash_core::llm::types::LlmOutputPart],
) -> lash_core::llm::types::LlmTerminalReason {
    if parts
        .iter()
        .any(|part| matches!(part, lash_core::llm::types::LlmOutputPart::ToolCall { .. }))
    {
        lash_core::llm::types::LlmTerminalReason::ToolUse
    } else {
        lash_core::llm::types::LlmTerminalReason::Stop
    }
}

pub(crate) fn terminal_reason_from_chat_value(
    value: &Value,
    parts: &[lash_core::llm::types::LlmOutputPart],
) -> lash_core::llm::types::LlmTerminalReason {
    let finish = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(Value::as_str)
        .unwrap_or("");
    match finish {
        "stop" => lash_core::llm::types::LlmTerminalReason::Stop,
        "tool_calls" | "function_call" => lash_core::llm::types::LlmTerminalReason::ToolUse,
        "length" | "max_tokens" => lash_core::llm::types::LlmTerminalReason::OutputLimit,
        "content_filter" | "safety" => lash_core::llm::types::LlmTerminalReason::ContentFilter,
        "" => terminal_reason_from_parts(parts),
        _ => lash_core::llm::types::LlmTerminalReason::ProviderError,
    }
}

pub(crate) fn terminal_reason_from_responses_value(
    value: &Value,
    parts: &[lash_core::llm::types::LlmOutputPart],
) -> lash_core::llm::types::LlmTerminalReason {
    let incomplete_details = value
        .get("incomplete_details")
        .or_else(|| value.get("incompleteDetails"));
    if incomplete_details
        .and_then(|details| details.get("reason").and_then(Value::as_str))
        .is_some_and(|reason| matches!(reason, "content_filter" | "safety"))
    {
        return lash_core::llm::types::LlmTerminalReason::ContentFilter;
    }
    if value.get("status").and_then(Value::as_str) == Some("incomplete")
        || incomplete_details.is_some_and(|details| !details.is_null())
    {
        return lash_core::llm::types::LlmTerminalReason::OutputLimit;
    }
    if value.get("status").and_then(Value::as_str) == Some("cancelled") {
        return lash_core::llm::types::LlmTerminalReason::Cancelled;
    }
    if value.get("status").and_then(Value::as_str) == Some("failed") {
        return lash_core::llm::types::LlmTerminalReason::ProviderError;
    }
    terminal_reason_from_parts(parts)
}

pub(crate) fn empty_response_error(raw: String) -> lash_core::llm::transport::LlmTransportError {
    lash_core::llm::transport::LlmTransportError::new("OpenAI-compatible empty_response")
        .retryable(true)
        .with_code("empty_response")
        .with_raw(raw)
}

pub(crate) fn retryable_error_code(value: &Value) -> Option<i64> {
    value
        .get("code")
        .or_else(|| value.get("status"))
        .and_then(|v| match v {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        })
}

pub(crate) fn responses_error_is_retryable(value: &Value) -> bool {
    matches!(retryable_error_code(value), Some(429))
        || matches!(retryable_error_code(value), Some(status) if status >= 500)
        || value
            .get("code")
            .or_else(|| value.get("type"))
            .or_else(|| value.get("status"))
            .and_then(|v| v.as_str())
            .is_some_and(|code| {
                matches!(
                    code,
                    "server_error"
                        | "internal_server_error"
                        | "service_unavailable"
                        | "temporarily_unavailable"
                        | "overloaded"
                        | "rate_limit_exceeded"
                )
            })
}
