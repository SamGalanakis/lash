use serde_json::{Value, json};
use std::sync::LazyLock;

use lash_llm_transport::timeouts::build_http_client;
use lash_openai_schema::emit_provider_trace;

pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

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
    provider_trace: Option<&lash::llm::types::LlmProviderTraceSender>,
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
