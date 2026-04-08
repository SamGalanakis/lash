use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};

use crate::llm::adapters::streaming::{drive_sse_response, emit_progress};
use crate::llm::timeouts::{
    LlmTimeouts, build_http_client, read_response_text, response_start_timeout, send_request,
};
use crate::llm::transport::{LlmTransport, LlmTransportError};
use crate::llm::types::{
    LlmMessage, LlmOutputPart, LlmOutputSpec, LlmReplayChunk, LlmRequest, LlmResponse, LlmRole,
    LlmStreamEvent, LlmUsage, ModelSelection, coalesce_replay_messages,
};
use crate::model_variant::VariantRequestConfig;
use crate::provider::Provider;

// ─── Provider compatibility ───

/// Per-provider compatibility overrides for OpenAI-compatible APIs.
/// Different providers diverge from the standard in subtle ways;
/// these flags let us adapt the request format at runtime.
#[derive(Clone, Debug)]
struct OpenAiCompat {
    /// Use `max_tokens` instead of `max_completion_tokens`.
    use_max_tokens_field: bool,
    /// Provider supports the `strict` field on tool definitions.
    supports_strict_mode: bool,
    /// Provider supports `stream_options: { include_usage: true }`.
    supports_usage_in_streaming: bool,
}

impl Default for OpenAiCompat {
    fn default() -> Self {
        Self {
            use_max_tokens_field: false,
            supports_strict_mode: true,
            supports_usage_in_streaming: true,
        }
    }
}

fn detect_compat(base_url: &str) -> OpenAiCompat {
    let normalized = base_url.trim().trim_end_matches('/').to_ascii_lowercase();

    let use_max_tokens_field = normalized.contains("chutes.ai");

    let is_non_standard = normalized.contains("cerebras.ai")
        || normalized.contains("api.x.ai")
        || normalized.contains("chutes.ai")
        || normalized.contains("deepseek.com")
        || normalized.contains("api.z.ai")
        || normalized.contains("opencode.ai");

    OpenAiCompat {
        use_max_tokens_field,
        supports_strict_mode: !is_non_standard,
        supports_usage_in_streaming: true,
    }
}

/// Sanitize surrogates and other problematic Unicode from text content
/// to avoid 400 errors from providers that reject lone surrogates.
fn sanitize_surrogates(s: &str) -> String {
    s.chars()
        .map(|c| if c == '\u{FFFD}' { '\u{FFFD}' } else { c })
        .collect()
}

/// Extract a human-readable error detail from a JSON error response body.
/// Tries `error.message`, then `error.metadata.raw`, then falls back to the
/// first 200 chars of the raw text.
fn extract_error_detail(raw: &str) -> Option<String> {
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
        let mut detail = msg.to_string();
        // Some providers (OpenRouter) include additional info in metadata.raw.
        if let Some(raw_meta) = v
            .get("error")
            .and_then(|e| e.get("metadata"))
            .and_then(|m| m.get("raw"))
            .and_then(|r| r.as_str())
        {
            detail.push_str(" — ");
            detail.push_str(raw_meta);
        }
        return Some(detail);
    }
    // Fallback: first 200 chars of raw body.
    Some(trimmed.chars().take(200).collect())
}

/// Normalize a tool call ID for cross-provider compatibility.
/// OpenAI Responses API generates IDs that are 450+ chars with special
/// characters (`|`, `+`, `/`, `=`). Many providers (especially Anthropic
/// via proxy) require IDs matching `^[a-zA-Z0-9_-]+$` (max 40-64 chars).
fn normalize_tool_call_id(id: &str) -> String {
    // Handle pipe-separated IDs from OpenAI Responses API
    let base = if let Some(idx) = id.find('|') {
        &id[..idx]
    } else {
        id
    };
    // Sanitize to allowed chars and truncate
    let sanitized: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(40)
        .collect();
    if sanitized.is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        sanitized
    }
}

pub struct OpenAiGenericAdapter {
    client: reqwest::Client,
    request_timeout: Option<std::time::Duration>,
    chunk_timeout: std::time::Duration,
}

#[derive(Clone, Debug, Default)]
struct StreamingToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl Default for OpenAiGenericAdapter {
    fn default() -> Self {
        Self::new(LlmTimeouts::default())
    }
}

impl OpenAiGenericAdapter {
    pub fn new(timeouts: LlmTimeouts) -> Self {
        Self {
            client: build_http_client(),
            request_timeout: timeouts.request_timeout,
            chunk_timeout: timeouts.chunk_timeout,
        }
    }

    fn content_json_for_message(req: &LlmRequest, msg: &LlmMessage) -> Value {
        match msg.kind.as_str() {
            "image" => {
                let idx = msg.image_idx.max(0) as usize;
                if let Some(att) = req.attachments.get(idx) {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
                    let data_url = format!("data:{};base64,{}", att.mime, b64);
                    json!([{
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }])
                } else {
                    json!("[Image attached]")
                }
            }
            _ => json!(msg.content),
        }
    }

    fn role_name(role: &LlmRole) -> &'static str {
        match role {
            LlmRole::User => "user",
            LlmRole::Assistant => "assistant",
            LlmRole::System => "system",
        }
    }

    /// Build a single content part for a message. Images become `image_url` objects;
    /// text becomes a `{"type":"text","text":"..."}` object suitable for multipart arrays.
    fn content_part_for_message(req: &LlmRequest, msg: &LlmMessage) -> Value {
        match msg.kind.as_str() {
            "image" => {
                let idx = msg.image_idx.max(0) as usize;
                if let Some(att) = req.attachments.get(idx) {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
                    let data_url = format!("data:{};base64,{}", att.mime, b64);
                    json!({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                } else {
                    json!({"type": "text", "text": "[Image attached]"})
                }
            }
            _ => json!({"type": "text", "text": msg.content}),
        }
    }

    fn build_messages(&self, req: &LlmRequest) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        let mut seen_first_system = false;
        for chunk in coalesce_replay_messages(&req.messages) {
            match chunk {
                LlmReplayChunk::Message(msg) => {
                    let role = if msg.kind == "tool_result" {
                        "tool"
                    } else if matches!(msg.role, LlmRole::System) {
                        // The first system message is the system prompt;
                        // subsequent system messages are runtime feedback
                        // (execution output, errors) which must be sent as
                        // "user" so the conversation can end with a user
                        // message — required by providers like Claude via
                        // OpenRouter that reject assistant-message prefill.
                        if seen_first_system {
                            "user"
                        } else {
                            seen_first_system = true;
                            "system"
                        }
                    } else {
                        Self::role_name(&msg.role)
                    };

                    // Merge consecutive user messages into a single multipart content
                    // array so text + images land in one API message (required by
                    // OpenAI-compatible vision APIs).
                    if role == "user"
                        && msg.kind != "tool_result"
                        && let Some(prev) = out.last_mut()
                        && prev.get("role").and_then(|r| r.as_str()) == Some("user")
                        && prev.get("tool_call_id").is_none()
                    {
                        let part = Self::content_part_for_message(req, &msg);
                        let content = &mut prev["content"];
                        if content.is_array() {
                            content.as_array_mut().unwrap().push(part);
                        } else {
                            // Previous message had a plain string content; promote
                            // it to a multipart array.
                            let prev_text =
                                content.as_str().map(|s| s.to_string()).unwrap_or_default();
                            *content = json!([
                                {"type": "text", "text": prev_text},
                                part,
                            ]);
                        }
                        continue;
                    }

                    let content = Self::content_json_for_message(req, &msg);
                    let mut item = json!({
                        "role": role,
                        "content": sanitize_surrogates(
                            &content.as_str().map(String::from).unwrap_or_else(|| content.to_string()),
                        ),
                    });
                    // Restore non-string content (arrays for images etc.)
                    if !content.is_string() {
                        item["content"] = content;
                    }
                    if role == "tool" {
                        let raw_id = msg.tool_call_id.clone().unwrap_or_default();
                        item["tool_call_id"] = json!(normalize_tool_call_id(&raw_id));
                    }
                    out.push(item);
                }
                LlmReplayChunk::AssistantToolCalls { text, tool_calls } => {
                    let content_text = text.unwrap_or_default();
                    // Skip empty assistant messages with no tool calls (some
                    // providers reject these).
                    if content_text.trim().is_empty() && tool_calls.is_empty() {
                        continue;
                    }
                    let mut msg = json!({
                        "role": "assistant",
                        "content": sanitize_surrogates(&content_text),
                    });
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = json!(
                            tool_calls
                                .into_iter()
                                .map(|call| json!({
                                    "id": normalize_tool_call_id(&call.call_id),
                                    "type": "function",
                                    "function": {
                                        "name": call.tool_name,
                                        "arguments": call.input_json,
                                    }
                                }))
                                .collect::<Vec<_>>()
                        );
                    }
                    out.push(msg);
                }
                LlmReplayChunk::ToolResults { results } => {
                    out.extend(results.into_iter().map(|msg| {
                        let raw_id = msg.tool_call_id.unwrap_or_default();
                        json!({
                            "role": "tool",
                            "tool_call_id": normalize_tool_call_id(&raw_id),
                            "content": sanitize_surrogates(&msg.content),
                        })
                    }));
                }
            }
        }
        out
    }

    fn has_tool_history(messages: &[LlmMessage]) -> bool {
        messages.iter().any(|msg| {
            msg.kind == "tool_result"
                || (matches!(msg.role, LlmRole::Assistant) && msg.kind == "tool_call")
        })
    }

    fn maybe_add_openrouter_anthropic_cache_control(
        messages: &mut [Value],
        model: &str,
        base_url: &str,
    ) {
        if !Self::supports_prompt_cache_key(base_url) || !model.starts_with("anthropic/") {
            return;
        }

        for message in messages.iter_mut().rev() {
            let role = message.get("role").and_then(|value| value.as_str());
            if !matches!(role, Some("user" | "assistant")) {
                continue;
            }

            let Some(content) = message.get_mut("content") else {
                continue;
            };

            if let Some(text) = content.as_str() {
                *content = json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": { "type": "ephemeral" }
                }]);
                return;
            }

            let Some(parts) = content.as_array_mut() else {
                continue;
            };

            for part in parts.iter_mut().rev() {
                if part.get("type").and_then(|value| value.as_str()) != Some("text") {
                    continue;
                }
                part["cache_control"] = json!({ "type": "ephemeral" });
                return;
            }
        }
    }

    fn is_openrouter(base_url: &str) -> bool {
        base_url
            .trim()
            .trim_end_matches('/')
            .to_ascii_lowercase()
            .contains("openrouter.ai")
    }

    fn supports_prompt_cache_key(base_url: &str) -> bool {
        Self::is_openrouter(base_url)
    }

    fn build_request_body(
        &self,
        provider: &Provider,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<(Value, String), LlmTransportError> {
        let (_, base_url) = match provider {
            Provider::OpenAiGeneric {
                api_key, base_url, ..
            } => (api_key.clone(), base_url.clone()),
            _ => {
                return Err(LlmTransportError::new(
                    "OpenAI-compatible adapter received non-OpenAI-compatible provider",
                ));
            }
        };

        let compat = detect_compat(&base_url);

        let mut messages = self.build_messages(req);
        Self::maybe_add_openrouter_anthropic_cache_control(&mut messages, &req.model, &base_url);

        let mut body = json!({
            "model": req.model,
            "messages": messages,
            "temperature": 0,
            "stream": stream,
        });

        // Use the correct max-tokens field name per provider.
        if compat.use_max_tokens_field {
            body["max_tokens"] = json!(32768);
        } else {
            body["max_completion_tokens"] = json!(32768);
        }

        if let Some(variant) = req.model_variant.as_deref()
            && let Some(VariantRequestConfig::ReasoningEffort(effort)) =
                crate::model_variant::request_config(provider, &req.model, variant)
        {
            if Self::is_openrouter(&base_url) {
                // OpenRouter normalizes reasoning via a nested reasoning object.
                body["reasoning"] = json!({ "effort": effort });
            } else {
                body["reasoning_effort"] = json!(effort);
            }
        }
        if stream && compat.supports_usage_in_streaming {
            body["stream_options"] = json!({ "include_usage": true });
        }
        if let Some(session_id) = req.session_id.as_deref()
            && Self::supports_prompt_cache_key(&base_url)
        {
            body["prompt_cache_key"] = json!(session_id);
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(
                req.tools
                    .iter()
                    .map(|tool| {
                        let mut func = json!({
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        });
                        // Only include strict if provider supports it. Some
                        // reject unknown fields in the function object.
                        if compat.supports_strict_mode {
                            func["strict"] = json!(false);
                        }
                        json!({
                            "type": "function",
                            "function": func,
                        })
                    })
                    .collect::<Vec<_>>()
            );
            body["tool_choice"] = match req.tool_choice {
                crate::llm::types::LlmToolChoice::Auto => json!("auto"),
                crate::llm::types::LlmToolChoice::None => json!("none"),
                crate::llm::types::LlmToolChoice::Required => json!("required"),
            };
        } else if Self::has_tool_history(&req.messages) {
            // Anthropic-compatible backends can require an explicit tools field
            // when replay history already contains tool calls/results.
            body["tools"] = json!([]);
        }
        if let Some(output_spec) = &req.output_spec {
            body["response_format"] = match output_spec {
                LlmOutputSpec::JsonObject => json!({ "type": "json_object" }),
                LlmOutputSpec::JsonSchema(schema) => json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.name,
                        "schema": schema.schema,
                        "strict": schema.strict,
                    }
                }),
            };
        }

        Ok((body, base_url))
    }

    fn parse_i64(v: Option<&Value>) -> i64 {
        match v {
            Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
            Some(Value::String(s)) => s.parse::<i64>().unwrap_or(0),
            _ => 0,
        }
    }

    fn usage_from_value(value: &Value) -> Option<LlmUsage> {
        let usage = value.get("usage")?;
        Some(LlmUsage {
            input_tokens: Self::parse_i64(usage.get("prompt_tokens")),
            output_tokens: Self::parse_i64(usage.get("completion_tokens")),
            cached_input_tokens: Self::parse_i64(
                usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .or_else(|| usage.get("cached_prompt_tokens"))
                    .or_else(|| usage.get("cached_tokens")),
            ),
            reasoning_tokens: Self::parse_i64(usage.get("reasoning_tokens").or_else(|| {
                usage
                    .get("completion_tokens_details")
                    .and_then(|d| d.get("reasoning_tokens"))
            })),
        })
    }

    fn extract_text_parts(value: &Value) -> Vec<String> {
        let mut out = Vec::new();
        let Some(choices) = value.get("choices").and_then(|v| v.as_array()) else {
            return out;
        };
        for choice in choices {
            if let Some(text) = choice
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
            {
                if !text.is_empty() {
                    out.push(text.to_string());
                }
                continue;
            }
            if let Some(parts) = choice
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_array())
            {
                for p in parts {
                    if let Some(text) = p.get("text").and_then(|t| t.as_str())
                        && !text.is_empty()
                    {
                        out.push(text.to_string());
                    }
                }
                continue;
            }
            if let Some(text) = choice
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
            {
                if !text.is_empty() {
                    out.push(text.to_string());
                }
                continue;
            }
            if let Some(parts) = choice
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_array())
            {
                for p in parts {
                    if let Some(text) = p.get("text").and_then(|t| t.as_str())
                        && !text.is_empty()
                    {
                        out.push(text.to_string());
                    }
                }
            }
        }
        out
    }

    fn apply_stream_piece(full: &mut String, deltas: &mut Vec<String>, piece: &str) {
        if piece.is_empty() {
            return;
        }
        if piece.starts_with(full.as_str()) {
            let delta = &piece[full.len()..];
            if !delta.is_empty() {
                full.push_str(delta);
                deltas.push(delta.to_string());
            }
            return;
        }
        full.push_str(piece);
        deltas.push(piece.to_string());
    }

    #[cfg(test)]
    fn process_sse_event(
        raw: &str,
        full: &mut String,
        deltas: &mut Vec<String>,
        usage: &mut LlmUsage,
    ) -> Result<(), LlmTransportError> {
        Self::process_sse_event_with_tools(raw, full, deltas, usage, None)
    }

    fn process_sse_event_with_tools(
        raw: &str,
        full: &mut String,
        deltas: &mut Vec<String>,
        usage: &mut LlmUsage,
        tool_calls: Option<&mut Vec<StreamingToolCall>>,
    ) -> Result<(), LlmTransportError> {
        let raw = raw.trim();
        if raw.is_empty() || raw == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(raw).map_err(|e| {
            LlmTransportError::new(format!("Invalid SSE payload: {e}")).with_raw(raw)
        })?;
        if let Some(err) = event.get("error") {
            return Err(
                LlmTransportError::new("OpenAI-compatible stream error").with_raw(err.to_string())
            );
        }
        if let Some(new_usage) = Self::usage_from_value(&event)
            && (new_usage.input_tokens > 0
                || new_usage.output_tokens > 0
                || new_usage.cached_input_tokens > 0
                || new_usage.reasoning_tokens > 0)
        {
            *usage = new_usage;
        }
        for piece in Self::extract_text_parts(&event) {
            Self::apply_stream_piece(full, deltas, &piece);
        }
        // Accumulate streaming tool call deltas.
        if let Some(tool_calls) = tool_calls
            && let Some(choices) = event.get("choices").and_then(|v| v.as_array())
        {
            for choice in choices {
                let Some(tcs) = choice
                    .get("delta")
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|t| t.as_array())
                else {
                    continue;
                };
                for tc in tcs {
                    let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    while tool_calls.len() <= index {
                        tool_calls.push(StreamingToolCall::default());
                    }
                    if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                        tool_calls[index].id = id.to_string();
                    }
                    if let Some(f) = tc.get("function") {
                        if let Some(name) = f.get("name").and_then(|n| n.as_str()) {
                            tool_calls[index].name = name.to_string();
                        }
                        if let Some(args) = f.get("arguments").and_then(|a| a.as_str()) {
                            tool_calls[index].arguments.push_str(args);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn parse_non_stream_response(raw: &str) -> Result<(String, LlmUsage), LlmTransportError> {
        let value: Value = serde_json::from_str(raw).map_err(|e| {
            LlmTransportError::new(format!("Invalid response JSON: {e}")).with_raw(raw.to_string())
        })?;
        let mut full = String::new();
        for piece in Self::extract_text_parts(&value) {
            Self::apply_stream_piece(&mut full, &mut Vec::new(), &piece);
        }
        let usage = Self::usage_from_value(&value).unwrap_or_default();
        Ok((full, usage))
    }

    fn response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
        let mut parts = Vec::new();
        let Some(choice) = value
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
        else {
            return parts;
        };
        let Some(message) = choice.get("message") else {
            return parts;
        };

        if let Some(text) = message.get("content").and_then(|c| c.as_str())
            && !text.is_empty()
        {
            parts.push(LlmOutputPart::Text {
                text: text.to_string(),
            });
        } else if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
            for item in content {
                if let Some(text) = item.get("text").and_then(|t| t.as_str())
                    && !text.is_empty()
                {
                    parts.push(LlmOutputPart::Text {
                        text: text.to_string(),
                    });
                }
            }
        }

        if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for tool_call in tool_calls {
                let Some(name) = tool_call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                else {
                    continue;
                };
                let arguments = tool_call
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .map(|v| {
                        v.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| v.to_string())
                    })
                    .unwrap_or_else(|| "{}".to_string());
                parts.push(LlmOutputPart::ToolCall {
                    call_id: tool_call
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    tool_name: name.to_string(),
                    input_json: arguments,
                });
            }
        }

        parts
    }
}

#[async_trait]
impl LlmTransport for OpenAiGenericAdapter {
    fn default_root_model(&self) -> &'static str {
        "anthropic/claude-sonnet-4.6"
    }

    fn default_agent_model(&self, tier: &str) -> Option<ModelSelection> {
        match tier {
            "low" => Some(ModelSelection {
                model: "minimax/minimax-m2.5",
                variant: None,
            }),
            "medium" => Some(ModelSelection {
                model: "z-ai/glm-5",
                variant: None,
            }),
            "high" => Some(ModelSelection {
                model: "anthropic/claude-sonnet-4.6",
                variant: Some("high"),
            }),
            _ => None,
        }
    }

    fn normalize_model(&self, model: &str) -> String {
        model.to_string()
    }

    fn context_lookup_model(&self, model: &str) -> String {
        if model.starts_with("openrouter/") {
            model.to_string()
        } else {
            format!("openrouter/{model}")
        }
    }

    async fn ensure_ready(&self, _provider: &mut Provider) -> Result<bool, LlmTransportError> {
        Ok(false)
    }

    async fn complete(
        &self,
        provider: &mut Provider,
        req: LlmRequest,
    ) -> Result<LlmResponse, LlmTransportError> {
        let stream_events = req.stream_events.clone();
        let (api_key, _) = match provider {
            Provider::OpenAiGeneric {
                api_key, base_url, ..
            } => (api_key.clone(), base_url.clone()),
            _ => {
                return Err(LlmTransportError::new(
                    "OpenAI-compatible adapter received non-OpenAI-compatible provider",
                ));
            }
        };

        let (body, base_url) = self.build_request_body(provider, &req, stream_events.is_some())?;

        let request_body = serde_json::to_string(&body).ok();
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let request = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body);
        let resp = send_request(
            request,
            response_start_timeout(
                self.request_timeout,
                self.chunk_timeout,
                stream_events.is_some(),
            ),
            "OpenAI-compatible response start timed out",
        )
        .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = read_response_text(
                resp,
                self.request_timeout,
                "OpenAI-compatible response body timed out",
            )
            .await
            .unwrap_or_default();
            // Include the raw error body in the user-facing message so the
            // actual provider rejection reason is visible (e.g. schema
            // validation failures, unsupported fields).
            let detail = extract_error_detail(&text);
            let message = if let Some(detail) = detail {
                format!(
                    "OpenAI-compatible request failed with {}: {}",
                    status.as_u16(),
                    detail,
                )
            } else {
                format!("OpenAI-compatible request failed with {}", status.as_u16())
            };
            return Err(LlmTransportError {
                message,
                retryable: status.as_u16() == 429 || status.as_u16() >= 500,
                raw: Some(text),
                code: Some(status.as_u16().to_string()),
            });
        }

        let is_sse = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);

        if !is_sse {
            let text = read_response_text(
                resp,
                self.request_timeout,
                "OpenAI-compatible response body timed out",
            )
            .await?;
            let (content, usage) = Self::parse_non_stream_response(&text)?;
            let value: Value = serde_json::from_str(&text).map_err(|e| {
                LlmTransportError::new(format!("Invalid response JSON: {e}")).with_raw(text.clone())
            })?;
            let mut parts = Self::response_parts_from_value(&value);
            if parts.is_empty() && !content.is_empty() {
                parts.push(LlmOutputPart::Text {
                    text: content.clone(),
                });
            }
            if let Some(tx) = &stream_events {
                if usage != LlmUsage::default() {
                    tx.send(LlmStreamEvent::Usage(usage.clone()));
                }
                if !content.is_empty() {
                    tx.send(LlmStreamEvent::Delta(content.clone()));
                }
            }
            return Ok(LlmResponse {
                deltas: vec![content.clone()],
                full_text: content,
                parts,
                usage,
                request_body,
                http_summary: Some(format!("HTTP POST {}", url)),
            });
        }

        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut streaming_tool_calls: Vec<StreamingToolCall> = Vec::new();
        drive_sse_response(
            resp,
            self.chunk_timeout,
            "OpenAI-compatible stream chunk timed out",
            |raw| {
                let prev_len = deltas.len();
                let prev_usage = usage.clone();
                Self::process_sse_event_with_tools(
                    &raw,
                    &mut full,
                    &mut deltas,
                    &mut usage,
                    Some(&mut streaming_tool_calls),
                )?;
                emit_progress(
                    stream_events.as_ref(),
                    &deltas,
                    prev_len,
                    &usage,
                    &prev_usage,
                );
                Ok(())
            },
        )
        .await?;

        let mut parts = Vec::new();
        if !full.is_empty() {
            parts.push(LlmOutputPart::Text { text: full.clone() });
        }
        for tc in &streaming_tool_calls {
            if !tc.id.is_empty() && !tc.name.is_empty() {
                parts.push(LlmOutputPart::ToolCall {
                    call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    input_json: tc.arguments.clone(),
                });
            }
        }

        Ok(LlmResponse {
            deltas,
            full_text: full,
            parts,
            usage,
            request_body,
            http_summary: Some(format!("HTTP POST {} (stream)", url)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(messages: Vec<LlmMessage>) -> LlmRequest {
        LlmRequest {
            model: "gpt-5.4".to_string(),
            messages,
            attachments: vec![],
            tools: vec![],
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        }
    }

    #[test]
    fn parses_openai_generic_sse_deltas_and_usage() {
        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();

        OpenAiGenericAdapter::process_sse_event(
            r#"{"choices":[{"delta":{"content":"Hel"}}]}"#,
            &mut full,
            &mut deltas,
            &mut usage,
        )
        .unwrap();
        OpenAiGenericAdapter::process_sse_event(
            r#"{"choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":10,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":4}}}"#,
            &mut full,
            &mut deltas,
            &mut usage,
        )
        .unwrap();

        assert_eq!(full, "Hello");
        assert_eq!(deltas, vec!["Hel".to_string(), "lo".to_string()]);
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(usage.cached_input_tokens, 4);
    }

    #[test]
    fn streaming_accumulates_tool_calls() {
        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut tool_calls = Vec::new();

        // First SSE event: tool call start with id and name
        OpenAiGenericAdapter::process_sse_event_with_tools(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"read_file","arguments":""}}]}}]}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            Some(&mut tool_calls),
        )
        .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].name, "read_file");

        // Second SSE event: argument chunk
        OpenAiGenericAdapter::process_sse_event_with_tools(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            Some(&mut tool_calls),
        )
        .unwrap();

        // Third SSE event: argument continuation
        OpenAiGenericAdapter::process_sse_event_with_tools(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.rs\"}"}}]}}]}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            Some(&mut tool_calls),
        )
        .unwrap();

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].arguments, r#"{"path":"a.rs"}"#);
        assert!(full.is_empty());
    }

    #[test]
    fn build_messages_preserve_system_and_user_order() {
        let adapter = OpenAiGenericAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        let req = req(vec![
            LlmMessage {
                role: LlmRole::System,
                content: "sys".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
            LlmMessage {
                role: LlmRole::User,
                content: "history".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
        ]);

        let messages = adapter.build_messages(&req);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "history");
    }

    #[test]
    fn build_messages_uses_structured_replay_for_standard_mode() {
        let adapter = OpenAiGenericAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        let req = req(vec![
            LlmMessage {
                role: LlmRole::System,
                content: "sys".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
            LlmMessage {
                role: LlmRole::User,
                content: "question".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
            LlmMessage {
                role: LlmRole::Assistant,
                content: "{\"path\":\"README.md\"}".to_string(),
                kind: "tool_call".to_string(),
                image_idx: -1,
                tool_call_id: Some("call_1".to_string()),
                tool_name: Some("read_file".to_string()),
            },
            LlmMessage {
                role: LlmRole::User,
                content: "ok".to_string(),
                kind: "tool_result".to_string(),
                image_idx: -1,
                tool_call_id: Some("call_1".to_string()),
                tool_name: Some("read_file".to_string()),
            },
        ]);

        let messages = adapter.build_messages(&req);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call_1");
    }

    #[test]
    fn build_request_body_sets_prompt_cache_key_for_openrouter() {
        let adapter = OpenAiGenericAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        let provider = Provider::OpenAiGeneric {
            api_key: "tok".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            options: crate::provider::ProviderOptions::default(),
        };
        let mut req = req(vec![
            LlmMessage {
                role: LlmRole::System,
                content: "sys".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
            LlmMessage {
                role: LlmRole::User,
                content: "hi".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
        ]);
        req.model = "openai/gpt-5".to_string();
        req.tool_choice = crate::llm::types::LlmToolChoice::None;
        req.session_id = Some("sess-123".to_string());

        let (body, _) = adapter
            .build_request_body(&provider, &req, false)
            .expect("request body");
        assert_eq!(body["prompt_cache_key"], "sess-123");
    }

    #[test]
    fn build_request_body_keeps_tools_field_for_tool_history_without_current_tools() {
        let adapter = OpenAiGenericAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        let provider = Provider::OpenAiGeneric {
            api_key: "tok".to_string(),
            base_url: "https://example.com/v1".to_string(),
            options: crate::provider::ProviderOptions::default(),
        };
        let req = req(vec![
            LlmMessage {
                role: LlmRole::User,
                content: "question".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
            LlmMessage {
                role: LlmRole::Assistant,
                content: "{\"path\":\"README.md\"}".to_string(),
                kind: "tool_call".to_string(),
                image_idx: -1,
                tool_call_id: Some("call_1".to_string()),
                tool_name: Some("read_file".to_string()),
            },
            LlmMessage {
                role: LlmRole::User,
                content: "done".to_string(),
                kind: "tool_result".to_string(),
                image_idx: -1,
                tool_call_id: Some("call_1".to_string()),
                tool_name: Some("read_file".to_string()),
            },
        ]);

        let (body, _) = adapter
            .build_request_body(&provider, &req, false)
            .expect("request body");

        assert_eq!(body["tools"], json!([]));
    }

    #[test]
    fn build_request_body_omits_prompt_cache_key_for_non_openrouter() {
        let adapter = OpenAiGenericAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        let provider = Provider::OpenAiGeneric {
            api_key: "tok".to_string(),
            base_url: "https://example.com/v1".to_string(),
            options: crate::provider::ProviderOptions::default(),
        };
        let mut req = req(vec![
            LlmMessage {
                role: LlmRole::System,
                content: "sys".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
            LlmMessage {
                role: LlmRole::User,
                content: "hi".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
        ]);
        req.model = "model".to_string();
        req.tool_choice = crate::llm::types::LlmToolChoice::None;
        req.session_id = Some("sess-123".to_string());

        let (body, _) = adapter
            .build_request_body(&provider, &req, false)
            .expect("request body");
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn build_request_body_adds_cache_control_for_openrouter_anthropic_models() {
        let adapter = OpenAiGenericAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        let provider = Provider::OpenAiGeneric {
            api_key: "tok".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            options: crate::provider::ProviderOptions::default(),
        };
        let mut req = req(vec![LlmMessage {
            role: LlmRole::User,
            content: "hi".to_string(),
            kind: "text".to_string(),
            image_idx: -1,
            tool_call_id: None,
            tool_name: None,
        }]);
        req.model = "anthropic/claude-sonnet-4.6".to_string();

        let (body, _) = adapter
            .build_request_body(&provider, &req, false)
            .expect("request body");

        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn build_request_body_adds_json_schema_response_format() {
        let adapter = OpenAiGenericAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        let provider = Provider::OpenAiGeneric {
            api_key: "tok".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            options: crate::provider::ProviderOptions::default(),
        };
        let mut req = req(vec![
            LlmMessage {
                role: LlmRole::System,
                content: "sys".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
            LlmMessage {
                role: LlmRole::User,
                content: "hi".to_string(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            },
        ]);
        req.model = "openai/gpt-5".to_string();
        req.tool_choice = crate::llm::types::LlmToolChoice::None;
        req.output_spec = Some(crate::llm::types::LlmOutputSpec::JsonSchema(
            crate::llm::types::LlmJsonSchema {
                name: "answer".to_string(),
                schema: json!({"type": "object", "properties": {"ok": {"type": "boolean"}}}),
                strict: true,
            },
        ));

        let (body, _) = adapter
            .build_request_body(&provider, &req, false)
            .expect("request body");
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["name"], "answer");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    }
}
