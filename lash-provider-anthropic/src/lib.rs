use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};

use lash::llm::streaming::drive_sse_response;
use lash::llm::timeouts::{
    build_http_client, read_response_text, request_body_snapshot, response_start_timeout,
    send_request,
};
use lash::llm::transport::LlmTransportError;
use lash::llm::types::{
    LlmContentBlock, LlmEventSender, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse,
    LlmRole, LlmStreamEvent, LlmToolChoice, LlmUsage,
};
use lash::provider::{
    AgentModelSelection, Provider, ProviderFactory, ProviderOptions, VariantRequestConfig,
};

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const FINE_GRAINED_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

const CLAUDE_ADAPTIVE_XHIGH_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];
const CLAUDE_ADAPTIVE_MAX_VARIANTS: &[&str] = &["low", "medium", "high", "max"];
const CLAUDE_ADAPTIVE_VARIANTS: &[&str] = &["low", "medium", "high"];
const CLAUDE_BUDGET_VARIANTS: &[&str] = &["none", "low", "medium", "high"];

pub(crate) fn anthropic_supports_adaptive_thinking(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-6")
        || lower.contains("opus-4.6")
        || lower.contains("opus-4-7")
        || lower.contains("opus-4.7")
        || lower.contains("sonnet-4-6")
        || lower.contains("sonnet-4.6")
}

pub(crate) fn anthropic_supports_xhigh(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-7") || lower.contains("opus-4.7")
}

pub(crate) fn anthropic_supports_max(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("opus-4-6") || lower.contains("opus-4.6")
}

/// Anthropic API (Claude) provider. Implements the `Provider` trait
/// directly — the old `AnthropicProvider` split has collapsed into one
/// type that owns both config and transport.
#[derive(Clone, Debug)]
pub struct AnthropicProvider {
    pub api_key: String,
    pub base_url: Option<String>,
    pub options: ProviderOptions,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            options: ProviderOptions::default(),
            client: build_http_client(),
        }
    }

    pub fn with_base_url(mut self, base_url: Option<String>) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
        self
    }

    /// Share an embedder-provided `reqwest::Client` instead of building
    /// a fresh one. Saves ~42 MB of TLS state per provider when the
    /// host pools connections across sessions.
    pub fn with_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.client = (*client).clone();
        self
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

    fn text_block_value(text: &str) -> Value {
        json!({
            "type": "text",
            "text": sanitize_surrogates(text),
        })
    }

    /// Translate one `LlmContentBlock` into the Anthropic wire shape.
    /// Returns `None` for blocks that have no valid wire form (e.g. an
    /// empty text block — Anthropic 400s on those).
    fn content_block_value(req: &LlmRequest, block: &LlmContentBlock) -> Option<Value> {
        match block {
            LlmContentBlock::Text { text, .. } => {
                if text.trim().is_empty() {
                    return None;
                }
                Some(Self::text_block_value(text))
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
                    return Some(Self::text_block_value(text));
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
            for block in msg.blocks.iter() {
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

    fn build_request_body(&self, req: &LlmRequest) -> Result<Value, LlmTransportError> {
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
            && let Some(cfg) = self.request_variant_config(&req.model, variant)
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
        if let LlmContentBlock::Text { text, .. } = block {
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
            if anthropic_supports_xhigh(model) {
                "xhigh".to_string()
            } else if anthropic_supports_max(model) {
                "max".to_string()
            } else {
                "high".to_string()
            }
        }
        "max" => {
            if anthropic_supports_max(model) {
                "max".to_string()
            } else if anthropic_supports_xhigh(model) {
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

impl AnthropicProvider {
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
                        parts.push(LlmOutputPart::Text {
                            text: block.text,
                            response_meta: None,
                        });
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
impl Provider for AnthropicProvider {
    fn kind(&self) -> &'static str {
        "anthropic"
    }

    fn default_model(&self) -> &str {
        "claude-opus-4-7"
    }

    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        let lower = model.to_ascii_lowercase();
        if anthropic_supports_adaptive_thinking(model) {
            if anthropic_supports_xhigh(model) {
                CLAUDE_ADAPTIVE_XHIGH_VARIANTS
            } else if anthropic_supports_max(model) {
                CLAUDE_ADAPTIVE_MAX_VARIANTS
            } else {
                CLAUDE_ADAPTIVE_VARIANTS
            }
        } else if lower.contains("haiku-4")
            || lower.contains("claude-opus-4")
            || lower.contains("claude-sonnet-4")
        {
            CLAUDE_BUDGET_VARIANTS
        } else {
            &[]
        }
    }

    fn default_model_variant(&self, model: &str) -> Option<&'static str> {
        let variants = self.supported_variants(model);
        if variants.is_empty() {
            return None;
        }
        if variants.contains(&"xhigh") {
            Some("xhigh")
        } else if variants.contains(&"max") {
            Some("max")
        } else {
            Some("high")
        }
    }

    fn request_variant_config(&self, model: &str, variant: &str) -> Option<VariantRequestConfig> {
        if self.validate_variant(model, variant).is_err() {
            return None;
        }
        if anthropic_supports_adaptive_thinking(model) {
            if variant == "none" {
                return None;
            }
            Some(VariantRequestConfig::AnthropicAdaptiveThinking {
                effort: variant.to_string(),
            })
        } else {
            let budget_tokens = match variant {
                "none" => return None,
                "low" => 1_024,
                "medium" => 4_096,
                "high" => 12_288,
                _ => return None,
            };
            Some(VariantRequestConfig::AnthropicThinkingBudget { budget_tokens })
        }
    }

    fn default_agent_model(&self, tier: &str) -> Option<AgentModelSelection> {
        match tier {
            "low" => Some(AgentModelSelection {
                model: "claude-haiku-4-6".to_string(),
                variant: None,
            }),
            "medium" => Some(AgentModelSelection {
                model: "claude-sonnet-4-6".to_string(),
                variant: Some("medium".to_string()),
            }),
            "high" => Some(AgentModelSelection {
                model: "claude-opus-4-7".to_string(),
                variant: Some("high".to_string()),
            }),
            _ => None,
        }
    }

    fn context_lookup_model(&self, model: &str) -> String {
        if model.starts_with("anthropic/") {
            model.to_string()
        } else {
            format!("anthropic/{model}")
        }
    }

    fn options(&self) -> &ProviderOptions {
        &self.options
    }

    fn options_mut(&mut self) -> &mut ProviderOptions {
        &mut self.options
    }

    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        let stream_events = req.stream_events.clone();
        let timeouts = self.options.llm_timeouts();
        let base_url = self
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

        let body = self.build_request_body(&req)?;
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
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", betas.join(","))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body);

        let resp = send_request(
            request,
            request_body.clone().map(request_body_snapshot),
            response_start_timeout(timeouts.request_timeout, timeouts.chunk_timeout, true),
            "Anthropic response start timed out",
        )
        .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = read_response_text(
                resp,
                timeouts.request_timeout,
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
            timeouts.chunk_timeout,
            "Anthropic stream chunk timed out",
            |raw| Self::process_sse_event(raw, &mut state, stream_events.as_ref()),
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

    fn serialize_config(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "api_key".to_string(),
            serde_json::Value::String(self.api_key.clone()),
        );
        if let Some(base_url) = &self.base_url {
            map.insert(
                "base_url".to_string(),
                serde_json::Value::String(base_url.clone()),
            );
        }
        if !self.options.is_default() {
            map.insert(
                "options".to_string(),
                serde_json::to_value(&self.options).unwrap_or(serde_json::Value::Null),
            );
        }
        serde_json::Value::Object(map)
    }

    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}

/// Deserialize payload for `ProviderSpec::config` when building an
/// `AnthropicProvider` from a stored [`lash::LashConfig`].
#[derive(Deserialize)]
struct AnthropicProviderConfig {
    api_key: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    options: ProviderOptions,
}

/// Factory that registers [`AnthropicProvider`] with lash's global
/// provider registry. Hosts call [`register`] once at startup.
pub struct AnthropicProviderFactory;

impl AnthropicProviderFactory {
    /// Convenience: install this factory into lash's global registry.
    pub fn register() {
        lash::register_provider_factory(std::sync::Arc::new(Self));
    }
}

impl ProviderFactory for AnthropicProviderFactory {
    fn kind(&self) -> &'static str {
        "anthropic"
    }
    fn cli_label(&self) -> &'static str {
        "Anthropic API (Claude)"
    }
    fn setup_name(&self) -> &'static str {
        "Anthropic API"
    }
    fn setup_description(&self) -> &'static str {
        "Claude via Anthropic API key"
    }
    fn default_base_url(&self) -> Option<&'static str> {
        Some("https://api.anthropic.com")
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<Box<dyn Provider>, String> {
        let cfg: AnthropicProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(Box::new(AnthropicProvider {
            api_key: cfg.api_key,
            base_url: cfg.base_url,
            options: cfg.options,
            client: build_http_client(),
        }))
    }
}
