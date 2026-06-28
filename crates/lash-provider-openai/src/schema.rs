use serde_json::Value;

/// Strip a leading `vendor/` prefix from a model id (e.g. `openai/gpt-4o` to
/// `gpt-4o`). Used by the OpenAI-flavored providers when matching model
/// families.
pub fn model_id(model: &str) -> &str {
    model
        .rsplit_once('/')
        .map(|(_, tail)| tail)
        .unwrap_or(model)
}

/// Decide whether an error object embedded in a Responses SSE event (or a
/// non-2xx Responses body) is retryable.
pub fn responses_error_is_retryable(value: &Value) -> bool {
    let numeric_code = value
        .get("code")
        .or_else(|| value.get("status"))
        .and_then(|v| match v {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        });
    matches!(numeric_code, Some(429))
        || matches!(numeric_code, Some(status) if status >= 500)
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
