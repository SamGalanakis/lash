use serde_json::{Value, json};
use std::sync::LazyLock;

use lash_llm_transport::timeouts::build_http_client;
use lash_llm_transport::util::emit_provider_trace;

pub(crate) use crate::responses_shared::{
    merge_usage, usage_from_response_value, usage_from_usage_value,
};

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
    if provider_trace.is_none() {
        return;
    }
    let raw = json!({
        "type": "openai_compatible.request_body",
        "endpoint": endpoint,
        "body": body,
    });
    if let Ok(raw) = serde_json::to_string(&raw) {
        emit_provider_trace(provider_trace, "openai_compatible", &raw);
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
