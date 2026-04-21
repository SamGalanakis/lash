use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};

use crate::llm::adapters::streaming::drive_sse_response;
use crate::llm::timeouts::{
    LlmTimeouts, build_http_client, read_response_text, response_start_timeout, send_request,
};
use crate::llm::transport::{LlmTransport, LlmTransportError};
use crate::llm::types::{
    LlmContentBlock, LlmEventSender, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse,
    LlmRole, LlmStreamEvent, LlmToolChoice, LlmUsage, ModelSelection,
};
use crate::model_variant::{VariantRequestConfig, anthropic_supports_adaptive_thinking};
use crate::provider::Provider;

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const FINE_GRAINED_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

pub struct AnthropicAdapter {
    client: reqwest::Client,
    request_timeout: Option<std::time::Duration>,
    chunk_timeout: std::time::Duration,
}

impl Default for AnthropicAdapter {
    fn default() -> Self {
        Self::new(LlmTimeouts::default())
    }
}

impl AnthropicAdapter {
    pub fn new(timeouts: LlmTimeouts) -> Self {
        Self {
            client: build_http_client(),
            request_timeout: timeouts.request_timeout,
            chunk_timeout: timeouts.chunk_timeout,
        }
    }

    /// Use an embedder-provided `reqwest::Client` instead of building a
    /// fresh one. Shares the TLS stack + connection pool across every
    /// adapter constructed from the same pool.
    pub fn with_client(client: std::sync::Arc<reqwest::Client>, timeouts: LlmTimeouts) -> Self {
        Self {
            client: (*client).clone(),
            request_timeout: timeouts.request_timeout,
            chunk_timeout: timeouts.chunk_timeout,
        }
    }

    fn role_name(role: &LlmRole) -> &'static str {
        match role {
            LlmRole::User => "user",
            LlmRole::Assistant => "assistant",
            LlmRole::System => "user",
        }
    }

    fn image_block_value(req: &LlmRequest, attachment_idx: usize) -> Option<Value> {
        let att = req.attachments.get(attachment_idx)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
        Some(json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": att.mime,
                "data": b64,
            }
        }))
    }

    fn text_block_value(text: impl Into<String>) -> Value {
        json!({
            "type": "text",
            "text": sanitize_surrogates(&text.into()),
        })
    }

    /// Translate one `LlmContentBlock` into the Anthropic wire shape.
    /// Returns `None` for blocks that have no valid wire form (e.g. an
    /// empty text block — Anthropic 400s on those).
    fn content_block_value(req: &LlmRequest, block: &LlmContentBlock) -> Option<Value> {
        match block {
            LlmContentBlock::Text(text) => {
                if text.trim().is_empty() {
                    return None;
                }
                Some(Self::text_block_value(text.clone()))
            }
            LlmContentBlock::Image { attachment_idx } => Some(
                Self::image_block_value(req, *attachment_idx)
                    .unwrap_or_else(|| Self::text_block_value("[Image attached]")),
            ),
            LlmContentBlock::ToolCall {
                call_id,
                tool_name,
                input_json,
                ..
            } => {
                let input: Value = serde_json::from_str(input_json).unwrap_or_else(|_| json!({}));
                Some(json!({
                    "type": "tool_use",
                    "id": normalize_tool_call_id(call_id),
                    "name": tool_name,
                    "input": input,
                }))
            }
            LlmContentBlock::ToolResult {
                call_id, content, ..
            } => Some(json!({
                "type": "tool_result",
                "tool_use_id": normalize_tool_call_id(call_id),
                "content": content.clone(),
            })),
            LlmContentBlock::Reasoning {
                text,
                signature,
                redacted,
                ..
            } => {
                // Anthropic requires a signature to replay a thinking
                // block. If we don't have one (e.g. aborted stream, or
                // reasoning captured from a non-Anthropic provider that
                // stored its payload in `encrypted_content` only), fall
                // back to plain text so the turn still validates.
                let Some(sig) = signature.as_deref() else {
                    if text.trim().is_empty() {
                        return None;
                    }
                    return Some(Self::text_block_value(text.clone()));
                };
                if *redacted {
                    return Some(json!({
                        "type": "redacted_thinking",
                        "data": sig,
                    }));
                }
                if text.trim().is_empty() {
                    return None;
                }
                Some(json!({
                    "type": "thinking",
                    "thinking": sanitize_surrogates(text),
                    "signature": sig,
                }))
            }
        }
    }

    /// Build the `messages` array for Anthropic Messages API. Each lash
    /// `LlmMessage` becomes one wire message; adjacent same-role messages
    /// get merged to match Anthropic's alternation rules.
    fn build_messages(&self, req: &LlmRequest) -> (Option<String>, Vec<Value>) {
        let mut system_prompt: Option<String> = None;
        let mut out: Vec<Value> = Vec::new();
        let mut first_system_seen = false;

        for msg in &req.messages {
            // First system message is the real system prompt; hoist it
            // into the top-level `system` field. Subsequent system
            // messages (runtime feedback) become user turns so the
            // conversation ends on a user boundary.
            if matches!(msg.role, LlmRole::System) && !first_system_seen {
                first_system_seen = true;
                let text = collect_text(&msg.blocks);
                if !text.is_empty() {
                    system_prompt = Some(text);
                }
                continue;
            }

            let wire_role = Self::role_name(&msg.role);
            let mut blocks: Vec<Value> = Vec::new();
            for block in &msg.blocks {
                if let Some(value) = Self::content_block_value(req, block) {
                    blocks.push(value);
                }
            }
            if blocks.is_empty() {
                continue;
            }

            // Merge with previous turn if same role — keeps replay valid
            // when a reasoning-only message immediately precedes a text
            // message from the same role.
            if let Some(prev) = out.last_mut()
                && prev.get("role").and_then(|v| v.as_str()) == Some(wire_role)
                && let Some(prev_content) = prev.get_mut("content").and_then(|c| c.as_array_mut())
            {
                prev_content.extend(blocks);
                continue;
            }

            out.push(json!({
                "role": wire_role,
                "content": blocks,
            }));
        }

        (system_prompt, out)
    }

    fn build_tools(&self, req: &LlmRequest) -> Vec<Value> {
        req.tools
            .iter()
            .map(|tool| {
                let schema = &tool.input_schema;
                let properties = schema.get("properties").cloned().unwrap_or(json!({}));
                let required = schema.get("required").cloned().unwrap_or(json!([]));
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                })
            })
            .collect()
    }

    /// Build the `cache_control` value. Honors `LASH_CACHE_RETENTION`:
    /// `long` attaches a 1-hour TTL (matching pi-mono's `long` retention
    /// on `api.anthropic.com`), any other value stays on the default
    /// 5-minute ephemeral window.
    fn cache_control_value() -> Value {
        let retention = std::env::var("LASH_CACHE_RETENTION")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if retention == "long" {
            json!({ "type": "ephemeral", "ttl": "1h" })
        } else {
            json!({ "type": "ephemeral" })
        }
    }

    fn apply_cache_control(
        system: &mut Option<Value>,
        messages: &mut [Value],
        tools: &mut [Value],
    ) {
        let ctrl = Self::cache_control_value();

        if let Some(sys) = system
            && let Some(arr) = sys.as_array_mut()
            && let Some(last) = arr.last_mut()
            && last.is_object()
        {
            last["cache_control"] = ctrl.clone();
        }

        if let Some(last_msg) = messages.last_mut()
            && last_msg.get("role").and_then(|v| v.as_str()) == Some("user")
            && let Some(content) = last_msg.get_mut("content").and_then(|c| c.as_array_mut())
            && let Some(last_block) = content.last_mut()
            && last_block.is_object()
        {
            last_block["cache_control"] = ctrl.clone();
        }

        if let Some(last_tool) = tools.last_mut()
            && last_tool.is_object()
        {
            last_tool["cache_control"] = ctrl;
        }
    }

    fn build_request_body(
        &self,
        provider: &Provider,
        req: &LlmRequest,
    ) -> Result<Value, LlmTransportError> {
        let _ = match provider {
            Provider::Anthropic { api_key, .. } => api_key.clone(),
            _ => {
                return Err(LlmTransportError::new(
                    "Anthropic adapter received non-Anthropic provider",
                ));
            }
        };

        let (system_text, mut messages) = self.build_messages(req);
        let mut tools = self.build_tools(req);

        let max_output_tokens: u64 = std::env::var("LASH_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|v: &u64| *v > 0)
            .unwrap_or(32768);

        let mut system_value: Option<Value> = system_text.map(|text| {
            json!([{
                "type": "text",
                "text": sanitize_surrogates(&text),
            }])
        });

        // Cache control: mark system, last user message, and last tool as
        // ephemeral to benefit from prompt caching. Applied before the body
        // is assembled so we only serialize the final state once.
        Self::apply_cache_control(&mut system_value, &mut messages, &mut tools);

        let mut body = json!({
            "model": req.model,
            "max_tokens": max_output_tokens,
            "messages": messages,
        });

        if let Some(system_value) = system_value {
            body["system"] = system_value;
        }
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
            body["tool_choice"] = match req.tool_choice {
                LlmToolChoice::Auto => json!({ "type": "auto" }),
                LlmToolChoice::None => json!({ "type": "none" }),
                LlmToolChoice::Required => json!({ "type": "any" }),
            };
        }

        // Extended thinking. Temperature is incompatible with thinking; omit
        // it whenever thinking is enabled (matches Anthropic API rules).
        let mut thinking_enabled = false;
        if let Some(variant) = req.model_variant.as_deref()
            && let Some(cfg) = crate::model_variant::request_config(provider, &req.model, variant)
        {
            match cfg {
                VariantRequestConfig::AnthropicAdaptiveThinking { effort } => {
                    let clamped = clamp_effort(&req.model, &effort);
                    body["thinking"] = json!({
                        "type": "adaptive",
                        "display": "summarized",
                    });
                    body["output_config"] = json!({ "effort": clamped });
                    thinking_enabled = true;
                }
                VariantRequestConfig::AnthropicThinkingBudget { budget_tokens } => {
                    body["thinking"] = json!({
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                        "display": "summarized",
                    });
                    thinking_enabled = true;
                }
                _ => {}
            }
        }
        if !thinking_enabled {
            body["temperature"] = json!(0);
        }

        if let Some(output_spec) = &req.output_spec {
            // Anthropic Messages API doesn't support native JSON schema
            // response formats. Append a system directive so the model
            // emits well-formed JSON while leaving parsing to the caller.
            let directive = match output_spec {
                LlmOutputSpec::JsonObject => "Respond with a single JSON object.".to_string(),
                LlmOutputSpec::JsonSchema(schema) => format!(
                    "Respond with a single JSON object matching this schema (name: {}). Schema: {}",
                    schema.name, schema.schema,
                ),
            };
            let suffix = json!({
                "type": "text",
                "text": directive,
            });
            match body.get_mut("system") {
                Some(Value::Array(arr)) => arr.push(suffix),
                _ => body["system"] = json!([suffix]),
            }
        }

        body["stream"] = json!(true);
        Ok(body)
    }

    fn parse_usage(usage: &Value) -> LlmUsage {
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let cache_read = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let cache_write = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        LlmUsage {
            // Anthropic's `input_tokens` is net of cache reads; add them so
            // lash-side accounting mirrors what the model was shown.
            input_tokens: input + cache_read + cache_write,
            output_tokens: output,
            cached_input_tokens: cache_read,
            reasoning_tokens: 0,
        }
    }
}

/// Replace lone surrogate code units with U+FFFD so the Anthropic API, which
/// strictly validates UTF-8, doesn't reject the request. Normal BMP text
/// passes through untouched.
fn sanitize_surrogates(s: &str) -> String {
    s.chars()
        .map(|c| if c == '\u{FFFD}' { '\u{FFFD}' } else { c })
        .collect()
}

/// Join all `Text` blocks in a message into a single string, separated by
/// blank lines. Non-text blocks are ignored. Used to collapse a multi-block
/// system message into the top-level `system` field.
fn collect_text(blocks: &[LlmContentBlock]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let LlmContentBlock::Text(text) = block {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(text);
        }
    }
    out
}

/// Normalize tool call IDs to the Anthropic-allowed character set and length.
fn normalize_tool_call_id(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if sanitized.is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        sanitized
    }
}

/// Clamp a requested effort level to what the target Anthropic model actually
/// supports. `xhigh` is Opus-4.7-only; on Opus 4.6 it maps to `max`; every
/// other adaptive model collapses unknown effort down to `high`.
pub(crate) fn clamp_effort(model: &str, effort: &str) -> String {
    let normalized = effort.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "xhigh" => {
            if crate::model_variant::anthropic_supports_xhigh(model) {
                "xhigh".to_string()
            } else if crate::model_variant::anthropic_supports_max(model) {
                "max".to_string()
            } else {
                "high".to_string()
            }
        }
        "max" => {
            if crate::model_variant::anthropic_supports_max(model) {
                "max".to_string()
            } else if crate::model_variant::anthropic_supports_xhigh(model) {
                "xhigh".to_string()
            } else {
                "high".to_string()
            }
        }
        other => other.to_string(),
    }
}

// ─── SSE event parsing ───

#[derive(Debug, Default)]
struct StreamBlock {
    kind: BlockKind,
    /// Accumulated visible text (for text/thinking blocks).
    text: String,
    /// `signature_delta` payload preserved for thinking blocks so we can
    /// replay them intact on the next turn. Empty for other block types.
    thinking_signature: String,
    /// Streaming buffer for tool_use input JSON.
    input_buffer: String,
    /// tool_use metadata.
    tool_call_id: String,
    tool_name: String,
    /// Initial input payload from `content_block_start` for tool_use.
    tool_initial_input: Value,
    /// Flags redacted thinking blocks; signature carries opaque payload.
    redacted: bool,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum BlockKind {
    #[default]
    Unknown,
    Text,
    Thinking,
    ToolUse,
}

#[derive(Debug, Default)]
struct StreamState {
    blocks: Vec<StreamBlock>,
    usage: LlmUsage,
    stop_reason: Option<String>,
}

fn parse_event(raw: &str) -> Option<Value> {
    serde_json::from_str::<Value>(raw).ok()
}

impl AnthropicAdapter {
    fn process_sse_event(
        raw: &str,
        state: &mut StreamState,
        stream_events: Option<&LlmEventSender>,
    ) -> Result<(), LlmTransportError> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Ok(());
        }
        let event = parse_event(raw).ok_or_else(|| {
            LlmTransportError::new("Invalid Anthropic SSE payload").with_raw(raw.to_string())
        })?;
        let kind = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        match kind.as_str() {
            "message_start" => {
                if let Some(usage) = event.get("message").and_then(|m| m.get("usage")) {
                    state.usage = Self::parse_usage(usage);
                    if let Some(tx) = stream_events
                        && state.usage != LlmUsage::default()
                    {
                        tx.send(LlmStreamEvent::Usage(state.usage.clone()));
                    }
                }
            }
            "content_block_start" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                while state.blocks.len() <= index {
                    state.blocks.push(StreamBlock::default());
                }
                let block_meta = event.get("content_block").cloned().unwrap_or_default();
                let block_type = block_meta
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let slot = &mut state.blocks[index];
                match block_type {
                    "text" => {
                        slot.kind = BlockKind::Text;
                    }
                    "thinking" => {
                        slot.kind = BlockKind::Thinking;
                    }
                    "redacted_thinking" => {
                        slot.kind = BlockKind::Thinking;
                        slot.redacted = true;
                        if let Some(data) = block_meta.get("data").and_then(|v| v.as_str()) {
                            slot.thinking_signature = data.to_string();
                        }
                        slot.text = "[Reasoning redacted]".to_string();
                    }
                    "tool_use" => {
                        slot.kind = BlockKind::ToolUse;
                        if let Some(id) = block_meta.get("id").and_then(|v| v.as_str()) {
                            slot.tool_call_id = id.to_string();
                        }
                        if let Some(name) = block_meta.get("name").and_then(|v| v.as_str()) {
                            slot.tool_name = name.to_string();
                        }
                        if let Some(input) = block_meta.get("input") {
                            slot.tool_initial_input = input.clone();
                        }
                    }
                    _ => {
                        slot.kind = BlockKind::Unknown;
                    }
                }
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if index >= state.blocks.len() {
                    return Ok(());
                }
                let delta = event.get("delta").cloned().unwrap_or_default();
                let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let slot = &mut state.blocks[index];
                match delta_type {
                    "text_delta" => {
                        let piece = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                        if !piece.is_empty() {
                            slot.text.push_str(piece);
                            if let Some(tx) = stream_events {
                                tx.send(LlmStreamEvent::Delta(piece.to_string()));
                            }
                        }
                    }
                    "thinking_delta" => {
                        let piece = delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                        if !piece.is_empty() {
                            slot.text.push_str(piece);
                            if let Some(tx) = stream_events {
                                tx.send(LlmStreamEvent::ReasoningDelta(piece.to_string()));
                            }
                        }
                    }
                    "signature_delta" => {
                        let piece = delta
                            .get("signature")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !piece.is_empty() {
                            slot.thinking_signature.push_str(piece);
                        }
                    }
                    "input_json_delta" => {
                        let piece = delta
                            .get("partial_json")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !piece.is_empty() {
                            slot.input_buffer.push_str(piece);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                // No state change required — the block is finalized by later
                // reconstruction. Anthropic does not send block content here.
            }
            "message_delta" => {
                if let Some(usage) = event.get("usage") {
                    let new_usage = Self::parse_usage(usage);
                    let mut merged = state.usage.clone();
                    if new_usage.input_tokens > 0 {
                        merged.input_tokens = new_usage.input_tokens;
                    }
                    if new_usage.output_tokens > 0 {
                        merged.output_tokens = new_usage.output_tokens;
                    }
                    if new_usage.cached_input_tokens > 0 {
                        merged.cached_input_tokens = new_usage.cached_input_tokens;
                    }
                    if merged != state.usage {
                        state.usage = merged;
                        if let Some(tx) = stream_events {
                            tx.send(LlmStreamEvent::Usage(state.usage.clone()));
                        }
                    }
                }
                if let Some(stop) = event
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|v| v.as_str())
                {
                    state.stop_reason = Some(stop.to_string());
                }
            }
            "message_stop" => {
                // End-of-stream marker. Nothing to collect here.
            }
            "ping" => {}
            "error" => {
                let msg = event
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Anthropic stream error")
                    .to_string();
                return Err(
                    LlmTransportError::new(format!("Anthropic stream error: {msg}"))
                        .with_raw(raw.to_string())
                        .retryable(is_retryable_error_event(&event)),
                );
            }
            _ => {}
        }
        Ok(())
    }

    fn finalize(state: StreamState) -> (Vec<LlmOutputPart>, String, LlmUsage) {
        let mut parts: Vec<LlmOutputPart> = Vec::new();
        let mut full_text = String::new();
        for block in state.blocks {
            match block.kind {
                BlockKind::Text => {
                    if !block.text.is_empty() {
                        full_text.push_str(&block.text);
                        parts.push(LlmOutputPart::Text { text: block.text });
                    }
                }
                BlockKind::Thinking => {
                    if block.text.is_empty() && block.thinking_signature.is_empty() {
                        continue;
                    }
                    let sig = if block.thinking_signature.is_empty() {
                        None
                    } else {
                        Some(block.thinking_signature)
                    };
                    parts.push(LlmOutputPart::Reasoning {
                        text: block.text,
                        // Anthropic's thinking signature lives in the shared
                        // `signature` slot. `encrypted_content` mirrors it
                        // so session persistence (which stores one opaque
                        // blob per reasoning item) can round-trip without
                        // knowing which provider produced it.
                        signature: sig.clone(),
                        redacted: block.redacted,
                        item_id: None,
                        encrypted_content: sig,
                        summary: Vec::new(),
                    });
                }
                BlockKind::ToolUse => {
                    let input_json = if !block.input_buffer.is_empty() {
                        block.input_buffer
                    } else if block.tool_initial_input.is_object() {
                        serde_json::to_string(&block.tool_initial_input)
                            .unwrap_or_else(|_| "{}".to_string())
                    } else {
                        "{}".to_string()
                    };
                    if block.tool_name.is_empty() {
                        continue;
                    }
                    parts.push(LlmOutputPart::ToolCall {
                        call_id: block.tool_call_id,
                        tool_name: block.tool_name,
                        input_json,
                        item_id: None,
                        signature: None,
                    });
                }
                BlockKind::Unknown => {}
            }
        }
        (parts, full_text, state.usage)
    }
}

fn is_retryable_error_event(event: &Value) -> bool {
    let error_type = event
        .get("error")
        .and_then(|e| e.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    matches!(
        error_type,
        "overloaded_error" | "api_error" | "rate_limit_error"
    )
}

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
        return Some(msg.to_string());
    }
    Some(trimmed.chars().take(200).collect())
}

#[async_trait]
impl LlmTransport for AnthropicAdapter {
    fn default_root_model(&self) -> &'static str {
        "claude-opus-4-7"
    }

    fn default_agent_model(&self, tier: &str) -> Option<ModelSelection> {
        match tier {
            "low" => Some(ModelSelection {
                model: "claude-haiku-4-6",
                variant: None,
            }),
            "medium" => Some(ModelSelection {
                model: "claude-sonnet-4-6",
                variant: Some("medium"),
            }),
            "high" => Some(ModelSelection {
                model: "claude-opus-4-7",
                variant: Some("high"),
            }),
            _ => None,
        }
    }

    fn normalize_model(&self, model: &str) -> String {
        model.to_string()
    }

    fn context_lookup_model(&self, model: &str) -> String {
        if model.starts_with("anthropic/") {
            model.to_string()
        } else {
            format!("anthropic/{model}")
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
        let (api_key, base_url) = match provider {
            Provider::Anthropic {
                api_key, base_url, ..
            } => (
                api_key.clone(),
                base_url
                    .clone()
                    .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            ),
            _ => {
                return Err(LlmTransportError::new(
                    "Anthropic adapter received non-Anthropic provider",
                ));
            }
        };

        let body = self.build_request_body(provider, &req)?;
        let request_body = serde_json::to_string(&body).ok();

        // `fine-grained-tool-streaming-2025-05-14` streams partial JSON so we
        // can surface tool arguments incrementally. Interleaved thinking is
        // built-in on adaptive models; for older models we opt into the beta.
        let mut betas = vec![FINE_GRAINED_BETA.to_string()];
        if !anthropic_supports_adaptive_thinking(&req.model) {
            betas.push(INTERLEAVED_THINKING_BETA.to_string());
        }

        let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
        let request = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", betas.join(","))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body);

        let resp = send_request(
            request,
            request_body.clone(),
            // Anthropic is always an SSE stream on this adapter, regardless
            // of whether a listener is attached. The first-chunk deadline
            // should reflect that.
            response_start_timeout(self.request_timeout, self.chunk_timeout, true),
            "Anthropic response start timed out",
        )
        .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = read_response_text(
                resp,
                self.request_timeout,
                "Anthropic response body timed out",
            )
            .await
            .unwrap_or_default();
            let detail = extract_error_detail(&text);
            let message = if let Some(detail) = detail {
                format!(
                    "Anthropic request failed with {}: {}",
                    status.as_u16(),
                    detail,
                )
            } else {
                format!("Anthropic request failed with {}", status.as_u16())
            };
            return Err(LlmTransportError {
                message,
                retryable: status.as_u16() == 429 || status.as_u16() >= 500,
                raw: Some(text),
                code: Some(status.as_u16().to_string()),
                request_body,
            });
        }

        let mut state = StreamState::default();
        drive_sse_response(
            resp,
            self.chunk_timeout,
            "Anthropic stream chunk timed out",
            |raw| Self::process_sse_event(&raw, &mut state, stream_events.as_ref()),
        )
        .await?;

        let (parts, full_text, usage) = Self::finalize(state);
        let deltas = if stream_events.is_none() {
            vec![full_text.clone()]
        } else {
            Vec::new()
        };
        Ok(LlmResponse {
            deltas,
            full_text,
            parts,
            usage,
            provider_usage: None,
            request_body,
            http_summary: Some(format!("HTTP POST {} (stream)", url)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{LlmAttachment, LlmContentBlock, LlmMessage, LlmRole, LlmToolSpec};
    use std::sync::Arc;

    fn provider() -> Provider {
        Provider::Anthropic {
            api_key: "sk-ant-test".into(),
            base_url: None,
            options: crate::provider::ProviderOptions::default(),
        }
    }

    fn adapter() -> AnthropicAdapter {
        AnthropicAdapter::new(LlmTimeouts::default())
    }

    fn empty_req() -> LlmRequest {
        LlmRequest {
            model: "claude-opus-4-7".into(),
            messages: vec![],
            attachments: vec![],
            tools: Arc::new(vec![]),
            tool_choice: LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        }
    }

    #[test]
    fn default_models_match_tiers() {
        let a = adapter();
        assert_eq!(a.default_root_model(), "claude-opus-4-7");
        let low = a.default_agent_model("low").unwrap();
        assert_eq!(low.model, "claude-haiku-4-6");
        let med = a.default_agent_model("medium").unwrap();
        assert_eq!(med.model, "claude-sonnet-4-6");
        assert_eq!(med.variant, Some("medium"));
        let high = a.default_agent_model("high").unwrap();
        assert_eq!(high.model, "claude-opus-4-7");
        assert_eq!(high.variant, Some("high"));
    }

    #[test]
    fn context_lookup_prefixes_anthropic() {
        let a = adapter();
        assert_eq!(
            a.context_lookup_model("claude-opus-4-7"),
            "anthropic/claude-opus-4-7"
        );
        assert_eq!(
            a.context_lookup_model("anthropic/claude-opus-4-7"),
            "anthropic/claude-opus-4-7"
        );
    }

    #[test]
    fn build_messages_hoists_first_system_prompt() {
        let a = adapter();
        let mut req = empty_req();
        req.messages = vec![
            LlmMessage::text(LlmRole::System, "system prompt"),
            LlmMessage::text(LlmRole::User, "hello"),
        ];
        let (system, messages) = a.build_messages(&req);
        assert_eq!(system.as_deref(), Some("system prompt"));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"][0]["type"], "text");
        assert_eq!(messages[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn build_messages_merges_consecutive_user_turns() {
        let a = adapter();
        let mut req = empty_req();
        req.messages = vec![
            LlmMessage::text(LlmRole::User, "first"),
            LlmMessage::text(LlmRole::User, "second"),
        ];
        let (_sys, messages) = a.build_messages(&req);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn build_request_body_omits_temperature_when_thinking_enabled() {
        let a = adapter();
        let mut req = empty_req();
        req.model_variant = Some("high".into());
        let body = a.build_request_body(&provider(), &req).unwrap();
        assert!(
            body.get("temperature").is_none(),
            "thinking + temperature is incompatible"
        );
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn build_request_body_passes_xhigh_on_opus_47() {
        let a = adapter();
        let mut req = empty_req();
        req.model_variant = Some("xhigh".into());
        let body = a.build_request_body(&provider(), &req).unwrap();
        assert_eq!(body["output_config"]["effort"], "xhigh");
    }

    #[test]
    fn build_request_body_clamps_xhigh_on_sonnet_46() {
        let a = adapter();
        let mut req = empty_req();
        req.model = "claude-sonnet-4-6".into();
        req.model_variant = Some("high".into());
        let body = a.build_request_body(&provider(), &req).unwrap();
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn clamp_effort_maps_xhigh_to_max_on_opus_46() {
        assert_eq!(clamp_effort("claude-opus-4-6", "xhigh"), "max");
        assert_eq!(clamp_effort("claude-opus-4-6", "max"), "max");
        assert_eq!(clamp_effort("claude-opus-4-7", "xhigh"), "xhigh");
        assert_eq!(clamp_effort("claude-opus-4-7", "max"), "xhigh");
        assert_eq!(clamp_effort("claude-sonnet-4-6", "xhigh"), "high");
    }

    #[test]
    fn cache_control_respects_long_retention() {
        // SAFETY: env mutation is the only way to exercise the branch.
        unsafe { std::env::set_var("LASH_CACHE_RETENTION", "long") };
        let value = AnthropicAdapter::cache_control_value();
        assert_eq!(value["ttl"], "1h");
        assert_eq!(value["type"], "ephemeral");
        unsafe { std::env::remove_var("LASH_CACHE_RETENTION") };
        let short = AnthropicAdapter::cache_control_value();
        assert!(short.get("ttl").is_none());
    }

    #[test]
    fn build_request_body_uses_budget_thinking_for_haiku() {
        let a = adapter();
        let mut req = empty_req();
        req.model = "claude-haiku-4-6".into();
        req.model_variant = Some("medium".into());
        let body = a.build_request_body(&provider(), &req).unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4_096);
    }

    #[test]
    fn process_sse_event_accumulates_text_delta() {
        let mut state = StreamState::default();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        let (parts, full, _) = AnthropicAdapter::finalize(state);
        assert_eq!(full, "Hello");
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn process_sse_event_captures_thinking_signature() {
        let mut state = StreamState::default();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc=="}}"#,
            &mut state,
            None,
        )
        .unwrap();
        let (parts, _, _) = AnthropicAdapter::finalize(state);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            LlmOutputPart::Reasoning {
                text,
                encrypted_content,
                ..
            } => {
                assert_eq!(text, "hmm");
                assert_eq!(encrypted_content.as_deref(), Some("abc=="));
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }

    #[test]
    fn process_sse_event_collects_tool_use_input_json() {
        let mut state = StreamState::default();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}"#,
            &mut state,
            None,
        )
        .unwrap();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"a.rs\"}"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        let (parts, _, _) = AnthropicAdapter::finalize(state);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                ..
            } => {
                assert_eq!(call_id, "toolu_1");
                assert_eq!(tool_name, "read_file");
                assert_eq!(input_json, r#"{"path":"a.rs"}"#);
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn process_sse_event_records_usage_from_message_start() {
        let mut state = StreamState::default();
        AnthropicAdapter::process_sse_event(
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0,"cache_read_input_tokens":4,"cache_creation_input_tokens":0}}}"#,
            &mut state,
            None,
        )
        .unwrap();
        assert_eq!(state.usage.input_tokens, 14);
        assert_eq!(state.usage.cached_input_tokens, 4);
    }

    #[test]
    fn apply_cache_control_marks_last_tool_and_user() {
        let mut sys = Some(json!([{"type":"text","text":"sys"}]));
        let mut messages = vec![json!({
            "role": "user",
            "content": [{"type":"text","text":"hi"}],
        })];
        let mut tools = vec![json!({"name": "foo"})];
        AnthropicAdapter::apply_cache_control(&mut sys, &mut messages, &mut tools);
        assert_eq!(
            sys.as_ref().unwrap()[0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(
            messages[0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn build_request_body_emits_tool_choice_and_tools() {
        let a = adapter();
        let mut req = empty_req();
        req.tools = Arc::new(vec![LlmToolSpec {
            name: "foo".into(),
            description: "x".into(),
            input_schema: json!({"properties": {}, "required": []}),
            output_schema: json!({}),
        }]);
        let body = a.build_request_body(&provider(), &req).unwrap();
        assert_eq!(body["tools"][0]["name"], "foo");
        assert_eq!(body["tool_choice"]["type"], "auto");
    }

    #[test]
    fn build_request_body_attaches_image_blocks() {
        let a = adapter();
        let mut req = empty_req();
        req.attachments = vec![LlmAttachment {
            mime: "image/png".into(),
            data: b"png-bytes".to_vec(),
        }];
        req.messages = vec![LlmMessage::new(
            LlmRole::User,
            vec![LlmContentBlock::Image { attachment_idx: 0 }],
        )];
        let body = a.build_request_body(&provider(), &req).unwrap();
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "user");
        assert_eq!(msg["content"][0]["type"], "image");
        assert_eq!(msg["content"][0]["source"]["type"], "base64");
        assert_eq!(msg["content"][0]["source"]["media_type"], "image/png");
    }

    #[test]
    fn build_messages_replays_assistant_tool_calls_and_results() {
        let a = adapter();
        let mut req = empty_req();
        req.messages = vec![
            LlmMessage::text(LlmRole::User, "run something"),
            LlmMessage::new(
                LlmRole::Assistant,
                vec![
                    LlmContentBlock::Text("sure".into()),
                    LlmContentBlock::ToolCall {
                        call_id: "toolu_abc".into(),
                        tool_name: "read_file".into(),
                        input_json: r#"{"path":"foo.rs"}"#.into(),
                        item_id: None,
                        signature: None,
                    },
                ],
            ),
            LlmMessage::new(
                LlmRole::User,
                vec![LlmContentBlock::ToolResult {
                    call_id: "toolu_abc".into(),
                    content: "file contents".into(),
                    tool_name: None,
                }],
            ),
        ];
        let (_sys, messages) = a.build_messages(&req);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "text");
        assert_eq!(messages[1]["content"][1]["type"], "tool_use");
        assert_eq!(messages[1]["content"][1]["id"], "toolu_abc");
        assert_eq!(messages[1]["content"][1]["name"], "read_file");
        assert_eq!(messages[1]["content"][1]["input"]["path"], "foo.rs");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_abc");
    }

    #[test]
    fn build_messages_replays_thinking_with_signature() {
        let a = adapter();
        let mut req = empty_req();
        req.messages = vec![
            LlmMessage::text(LlmRole::User, "hi"),
            LlmMessage::new(
                LlmRole::Assistant,
                vec![
                    LlmContentBlock::Reasoning {
                        text: "pondering".into(),
                        signature: Some("SIG==".into()),
                        redacted: false,
                        item_id: None,
                        encrypted_content: Some("SIG==".into()),
                        summary: Vec::new(),
                    },
                    LlmContentBlock::Text("answer".into()),
                ],
            ),
            LlmMessage::text(LlmRole::User, "next"),
        ];
        let (_sys, messages) = a.build_messages(&req);
        assert_eq!(messages.len(), 3);
        let assistant = &messages[1];
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["content"][0]["type"], "thinking");
        assert_eq!(assistant["content"][0]["thinking"], "pondering");
        assert_eq!(assistant["content"][0]["signature"], "SIG==");
        assert_eq!(assistant["content"][1]["type"], "text");
    }

    #[test]
    fn build_messages_emits_redacted_thinking_block() {
        let a = adapter();
        let mut req = empty_req();
        req.messages = vec![
            LlmMessage::text(LlmRole::User, "hi"),
            LlmMessage::new(
                LlmRole::Assistant,
                vec![LlmContentBlock::Reasoning {
                    text: String::new(),
                    signature: Some("OPAQUE==".into()),
                    redacted: true,
                    item_id: None,
                    encrypted_content: Some("OPAQUE==".into()),
                    summary: Vec::new(),
                }],
            ),
        ];
        let (_sys, messages) = a.build_messages(&req);
        assert_eq!(messages[1]["content"][0]["type"], "redacted_thinking");
        assert_eq!(messages[1]["content"][0]["data"], "OPAQUE==");
    }

    #[test]
    fn build_messages_falls_back_to_text_when_signature_missing() {
        let a = adapter();
        let mut req = empty_req();
        req.messages = vec![
            LlmMessage::text(LlmRole::User, "hi"),
            LlmMessage::new(
                LlmRole::Assistant,
                vec![LlmContentBlock::Reasoning {
                    text: "aborted thought".into(),
                    signature: None,
                    redacted: false,
                    item_id: None,
                    encrypted_content: None,
                    summary: Vec::new(),
                }],
            ),
        ];
        let (_sys, messages) = a.build_messages(&req);
        // No signature → downgrades to plain text block so Anthropic still
        // accepts the turn.
        assert_eq!(messages[1]["content"][0]["type"], "text");
        assert_eq!(messages[1]["content"][0]["text"], "aborted thought");
    }

    #[test]
    fn build_messages_skips_empty_user_text() {
        let a = adapter();
        let mut req = empty_req();
        req.messages = vec![
            LlmMessage::text(LlmRole::User, "   "),
            LlmMessage::text(LlmRole::User, "hello"),
        ];
        let (_sys, messages) = a.build_messages(&req);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"][0]["text"], "hello");
    }
}
