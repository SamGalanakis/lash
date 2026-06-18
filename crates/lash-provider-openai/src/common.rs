use serde_json::{Value, json};
use std::sync::LazyLock;

use lash_llm_transport::timeouts::build_http_client;
use lash_llm_transport::util::emit_provider_trace;

pub(crate) use lash_llm_transport::{
    merge_usage,
    openai_terminal_reason_from_chat_finish_reason as terminal_reason_from_chat_finish_reason,
    openai_terminal_reason_from_chat_value as terminal_reason_from_chat_value,
    openai_terminal_reason_from_response_value as terminal_reason_from_responses_value,
    openai_usage_from_response_value as usage_from_response_value,
    openai_usage_from_usage_value as usage_from_usage_value, terminal_reason_from_parts,
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

pub(crate) fn empty_response_error(raw: String) -> lash_core::llm::transport::LlmTransportError {
    lash_core::llm::transport::LlmTransportError::new("OpenAI-compatible empty_response")
        .retryable(true)
        .with_code("empty_response")
        .with_raw(raw)
}
