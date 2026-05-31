#![allow(clippy::result_large_err)]

use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};

use lash_core::llm::transport::{LlmTransportError, validate_image_attachments};
use lash_core::llm::types::{
    LlmContentBlock, LlmEventSender, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse,
    LlmRole, LlmStreamEvent, LlmTerminalReason, LlmToolChoice, LlmUsage, ProviderReasoningReplay,
};
use lash_core::provider::{
    CacheRetention, Provider, ProviderComponents, ProviderFactory, ProviderModelPolicy,
    ProviderOptions, resolve_generation_policy,
};
use lash_llm_transport::streaming::drive_sse_response;
use lash_llm_transport::timeouts::{
    build_http_client, header_pairs, read_response_text, request_body_snapshot,
    response_start_timeout, send_request,
};
use lash_llm_transport::util::{OPENAI_IMAGE_MIMES, emit_provider_trace, extract_error_detail};

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const FINE_GRAINED_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 32_768;

#[derive(Clone, Debug, PartialEq, Eq)]
enum AnthropicThinkingConfig {
    Adaptive { effort: String },
    Budget { budget_tokens: i32 },
}

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

/// Anthropic API (Claude) provider state and transport.
#[derive(Clone, Debug)]
pub struct AnthropicProvider {
    pub api_key: String,
    pub base_url: Option<String>,
    pub options: ProviderOptions,
    client: reqwest::Client,
}

#[derive(Clone, Debug)]
struct AnthropicModelPolicy;

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

    fn text_block_value(text: &str, cache_breakpoint: bool) -> Value {
        let mut block = json!({
            "type": "text",
            "text": text,
        });
        if cache_breakpoint {
            block["__lash_cache_breakpoint"] = json!(true);
        }
        block
    }

    /// Translate one `LlmContentBlock` into the Anthropic wire shape.
    /// Returns `None` for blocks that have no valid wire form (e.g. an
    /// empty text block — Anthropic 400s on those).
    fn content_block_value(req: &LlmRequest, block: &LlmContentBlock) -> Option<Value> {
        match block {
            LlmContentBlock::Text {
                text,
                cache_breakpoint,
                ..
            } => {
                if text.trim().is_empty() {
                    return None;
                }
                Some(Self::text_block_value(text, *cache_breakpoint))
            }
            LlmContentBlock::Image { attachment_idx } => Some(
                Self::image_block_value(req, *attachment_idx)
                    .unwrap_or_else(|| Self::text_block_value("[Image attached]", false)),
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
            LlmContentBlock::Reasoning { text, replay, .. } => {
                // Anthropic requires a signature to replay a thinking
                // block. If we don't have one (e.g. aborted stream, or
                // reasoning captured from a non-Anthropic provider that
                // stored its payload in `encrypted_content` only), fall
                // back to plain text so the turn still validates.
                let Some(sig) = replay.as_ref().and_then(|meta| meta.signature.as_deref()) else {
                    if text.trim().is_empty() {
                        return None;
                    }
                    return Some(Self::text_block_value(text, false));
                };
                if replay.as_ref().is_some_and(|meta| meta.redacted) {
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
                    "thinking": text,
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

    fn cache_control_value(cache_retention: CacheRetention) -> Option<Value> {
        match cache_retention {
            CacheRetention::None => None,
            CacheRetention::Short => Some(json!({ "type": "ephemeral" })),
            CacheRetention::Long => Some(json!({ "type": "ephemeral", "ttl": "1h" })),
        }
    }

    fn apply_cache_control(
        &self,
        cache_retention: CacheRetention,
        system: &mut Option<Value>,
        messages: &mut [Value],
        tools: &mut [Value],
    ) {
        let Some(ctrl) = Self::cache_control_value(cache_retention) else {
            for msg in messages {
                if let Some(content) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
                    for block in content {
                        block
                            .as_object_mut()
                            .map(|obj| obj.remove("__lash_cache_breakpoint"));
                    }
                }
            }
            return;
        };

        if let Some(sys) = system
            && let Some(arr) = sys.as_array_mut()
            && let Some(last) = arr.last_mut()
            && last.is_object()
        {
            last["cache_control"] = ctrl.clone();
        }

        let mut applied_explicit_breakpoint = false;
        for msg in messages.iter_mut().rev() {
            if !matches!(
                msg.get("role").and_then(Value::as_str),
                Some("user" | "assistant")
            ) {
                continue;
            }
            let Some(content) = msg.get_mut("content").and_then(|c| c.as_array_mut()) else {
                continue;
            };
            for block in content.iter_mut().rev() {
                let is_marked = block
                    .get("__lash_cache_breakpoint")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if is_marked && block.is_object() {
                    block["cache_control"] = ctrl.clone();
                    block
                        .as_object_mut()
                        .map(|obj| obj.remove("__lash_cache_breakpoint"));
                    applied_explicit_breakpoint = true;
                    break;
                }
            }
            if applied_explicit_breakpoint {
                break;
            }
        }

        if !applied_explicit_breakpoint
            && let Some(last_msg) = messages.last_mut()
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

        for msg in messages {
            if let Some(content) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
                for block in content {
                    block
                        .as_object_mut()
                        .map(|obj| obj.remove("__lash_cache_breakpoint"));
                }
            }
        }
    }

    fn build_request_body(&self, req: &LlmRequest) -> Result<Value, LlmTransportError> {
        validate_image_attachments(req, OPENAI_IMAGE_MIMES, "Anthropic")?;
        let (system_text, mut messages) = self.build_messages(req);
        let mut tools = self.build_tools(req);

        let thinking_config = req
            .model_variant
            .as_deref()
            .and_then(|variant| AnthropicModelPolicy.thinking_config(&req.model, variant));
        let policy = resolve_generation_policy(
            &req.generation,
            &self.options,
            DEFAULT_MAX_OUTPUT_TOKENS,
            thinking_config,
        );

        let mut system_value: Option<Value> = system_text.map(|text| {
            json!([{
                "type": "text",
                "text": text,
            }])
        });

        // Cache control: mark system, last user message, and last tool as
        // ephemeral to benefit from prompt caching. Applied before the body
        // is assembled so we only serialize the final state once.
        self.apply_cache_control(
            policy.cache_retention,
            &mut system_value,
            &mut messages,
            &mut tools,
        );

        let mut body = json!({
            "model": req.model,
            "max_tokens": policy.max_output_tokens,
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
        if let Some(cfg) = policy.thinking {
            let display = if policy.expose_thinking {
                "summarized"
            } else {
                "omitted"
            };
            match cfg {
                AnthropicThinkingConfig::Adaptive { effort } => {
                    let clamped = clamp_effort(&req.model, &effort);
                    body["thinking"] = json!({
                        "type": "adaptive",
                        "display": display,
                    });
                    body["output_config"] = json!({ "effort": clamped });
                    thinking_enabled = true;
                }
                AnthropicThinkingConfig::Budget { budget_tokens } => {
                    body["thinking"] = json!({
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                        "display": display,
                    });
                    thinking_enabled = true;
                }
            }
        }
        if !thinking_enabled {
            body["temperature"] = json!(0);
        }

        if let Some(output_spec) = &req.output_spec {
            let format = match output_spec {
                LlmOutputSpec::JsonObject => json!({
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "additionalProperties": true,
                    },
                }),
                LlmOutputSpec::JsonSchema(schema) => json!({
                    "type": "json_schema",
                    "schema": schema.schema,
                }),
            };
            if !body.get("output_config").is_some_and(Value::is_object) {
                body["output_config"] = json!({});
            }
            body["output_config"]["format"] = format;
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
        expose_thinking: bool,
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
                            if let Some(tx) = stream_events
                                && expose_thinking
                            {
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

    fn finalize(state: StreamState) -> (Vec<LlmOutputPart>, String, LlmUsage, LlmTerminalReason) {
        let mut parts: Vec<LlmOutputPart> = Vec::new();
        let mut full_text = String::new();
        let stop_reason = state.stop_reason.clone();
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
                        replay: sig.map(|signature| ProviderReasoningReplay {
                            item_id: None,
                            encrypted_content: None,
                            signature: Some(signature),
                            redacted: block.redacted,
                            summary: Vec::new(),
                        }),
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
                        replay: None,
                    });
                }
                BlockKind::Unknown => {}
            }
        }
        let terminal_reason = match stop_reason.as_deref() {
            Some("end_turn" | "stop_sequence") => LlmTerminalReason::Stop,
            Some("tool_use") => LlmTerminalReason::ToolUse,
            Some("max_tokens") => LlmTerminalReason::OutputLimit,
            Some("pause_turn") => LlmTerminalReason::Stop,
            Some("refusal" | "safety" | "sensitive") => LlmTerminalReason::ContentFilter,
            Some(_) => LlmTerminalReason::ProviderError,
            None => {
                if parts
                    .iter()
                    .any(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
                {
                    LlmTerminalReason::ToolUse
                } else {
                    LlmTerminalReason::Stop
                }
            }
        };
        (parts, full_text, state.usage, terminal_reason)
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

impl AnthropicProvider {
    pub fn into_components(self) -> ProviderComponents {
        ProviderComponents::new(Box::new(self), std::sync::Arc::new(AnthropicModelPolicy))
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn kind(&self) -> &'static str {
        "anthropic"
    }

    fn options(&self) -> ProviderOptions {
        self.options.clone()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.options = options;
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

    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        let stream_events = req.stream_events.clone();
        let provider_trace = req.provider_trace.clone();
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
            let headers = resp.headers().clone();
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
            let mut err = LlmTransportError::new(message)
                .with_status(status.as_u16())
                .with_headers(header_pairs(&headers))
                .with_raw(text);
            if let Some(request_body) = request_body {
                err = err.with_request_body(request_body);
            }
            return Err(err);
        }

        let mut state = StreamState::default();
        let expose_thinking = self.options.thinking.expose;
        drive_sse_response(
            resp,
            timeouts.chunk_timeout,
            "Anthropic stream chunk timed out",
            |raw| {
                emit_provider_trace(provider_trace.as_ref(), "anthropic", raw);
                Self::process_sse_event(raw, &mut state, stream_events.as_ref(), expose_thinking)
            },
        )
        .await?;

        let (parts, full_text, usage, terminal_reason) = Self::finalize(state);
        Ok(LlmResponse {
            full_text,
            parts,
            usage,
            terminal_reason,
            terminal_diagnostic: None,
            provider_usage: None,
            request_body,
            http_summary: Some(format!("HTTP POST {} (stream)", url)),
        })
    }

    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}

impl ProviderModelPolicy for AnthropicModelPolicy {
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
}

impl AnthropicModelPolicy {
    fn thinking_config(&self, model: &str, variant: &str) -> Option<AnthropicThinkingConfig> {
        if !self.supported_variants(model).contains(&variant) {
            return None;
        }
        if anthropic_supports_adaptive_thinking(model) {
            if variant == "none" {
                return None;
            }
            Some(AnthropicThinkingConfig::Adaptive {
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
            Some(AnthropicThinkingConfig::Budget { budget_tokens })
        }
    }
}

/// Deserialize payload for `ProviderSpec::config` when building an
/// `AnthropicProvider` from a stored [`lash_core::LashConfig`].
#[derive(Deserialize)]
struct AnthropicProviderConfig {
    api_key: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    options: ProviderOptions,
}

/// Factory that materializes [`AnthropicProvider`] from a host-owned
/// [`ProviderSpec`](lash_core::ProviderSpec).
pub struct AnthropicProviderFactory;

impl ProviderFactory for AnthropicProviderFactory {
    fn kind(&self) -> &'static str {
        "anthropic"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String> {
        let cfg: AnthropicProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(AnthropicProvider {
            api_key: cfg.api_key,
            base_url: cfg.base_url,
            options: cfg.options,
            client: build_http_client(),
        }
        .into_components())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::llm::types::{
        LlmAttachment, LlmContentBlock, LlmJsonSchema, LlmMessage, LlmToolSpec,
    };
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    fn request(messages: Vec<LlmMessage>) -> LlmRequest {
        LlmRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages,
            attachments: Vec::new(),
            tools: Arc::new(Vec::<LlmToolSpec>::new()),
            tool_choice: LlmToolChoice::Auto,
            model_variant: None,
            session_id: Some("session-1".to_string()),
            output_spec: None,
            stream_events: None,
            generation: lash_core::GenerationOptions::default(),
            provider_trace: None,
        }
    }

    #[test]
    fn image_attachment_serializes_as_base64_image_block() {
        let provider = AnthropicProvider::new("key");
        let png_bytes = vec![0x89, 0x50, 0x4E, 0x47];
        let mut req = request(vec![LlmMessage::new(
            LlmRole::User,
            vec![
                LlmContentBlock::Text {
                    text: "look at this".into(),
                    response_meta: None,
                    cache_breakpoint: false,
                },
                LlmContentBlock::Image { attachment_idx: 0 },
            ],
        )]);
        req.attachments = vec![LlmAttachment::bytes("image/png", png_bytes.clone())];

        let body = provider.build_request_body(&req).expect("body");

        let messages = body["messages"].as_array().expect("messages array");
        let user_msg = messages.last().expect("user message");
        let content = user_msg["content"].as_array().expect("content array");
        let image_block = content
            .iter()
            .find(|b| b["type"] == "image")
            .expect("image block");
        assert_eq!(image_block["source"]["type"], "base64");
        assert_eq!(image_block["source"]["media_type"], "image/png");
        let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        assert_eq!(image_block["source"]["data"], expected_b64);
    }

    #[test]
    fn unsupported_image_mime_is_rejected_at_request_boundary() {
        let provider = AnthropicProvider::new("key");
        let mut req = request(vec![LlmMessage::new(
            LlmRole::User,
            vec![LlmContentBlock::Image { attachment_idx: 0 }],
        )]);
        req.attachments = vec![LlmAttachment::bytes("image/bmp", vec![0x42, 0x4D])];

        let err = provider
            .build_request_body(&req)
            .expect_err("bmp should be rejected before wire");

        assert_eq!(err.code.as_deref(), Some("unsupported_image_format"));
        assert!(err.message.contains("Anthropic"));
        assert!(err.message.contains("image/bmp"));
    }

    #[test]
    fn structured_output_uses_native_output_config_format() {
        let provider = AnthropicProvider::new("key");
        let mut req = request(vec![
            LlmMessage::text(LlmRole::System, "system prompt"),
            LlmMessage::text(LlmRole::User, "extract"),
        ]);
        req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
            name: "extract_result".to_string(),
            strict: true,
            schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["answer"],
                "properties": {
                    "answer": { "type": "string" }
                }
            }),
        }));

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(
            body["output_config"]["format"],
            json!({
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["answer"],
                    "properties": {
                        "answer": { "type": "string" }
                    }
                }
            })
        );
        let system_text = body["system"][0]["text"].as_str().unwrap_or_default();
        assert_eq!(system_text, "system prompt");
        assert!(!system_text.contains("Respond with a single JSON object"));
    }

    #[test]
    fn structured_output_preserves_adaptive_effort_config() {
        let provider = AnthropicProvider::new("key");
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "extract")]);
        req.model_variant = Some("medium".to_string());
        req.output_spec = Some(LlmOutputSpec::JsonObject);

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(body["output_config"]["effort"], json!("medium"));
        assert_eq!(
            body["output_config"]["format"],
            json!({
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "additionalProperties": true,
                }
            })
        );
    }

    #[test]
    fn pause_turn_maps_to_stop() {
        let state = StreamState {
            stop_reason: Some("pause_turn".to_string()),
            ..StreamState::default()
        };

        let (_, _, _, terminal_reason) = AnthropicProvider::finalize(state);

        assert_eq!(terminal_reason, LlmTerminalReason::Stop);
    }

    #[test]
    fn unknown_stop_reason_maps_to_provider_error() {
        let state = StreamState {
            stop_reason: Some("new_provider_reason".to_string()),
            ..StreamState::default()
        };

        let (_, _, _, terminal_reason) = AnthropicProvider::finalize(state);

        assert_eq!(terminal_reason, LlmTerminalReason::ProviderError);
    }

    #[test]
    fn thinking_display_is_omitted_unless_provider_exposes_thinking() {
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "extract")]);
        req.model_variant = Some("medium".to_string());

        let hidden = AnthropicProvider::new("key")
            .build_request_body(&req)
            .expect("body");
        assert_eq!(hidden["thinking"]["display"], "omitted");

        let exposed = AnthropicProvider::new("key")
            .with_options(ProviderOptions {
                thinking: lash_core::provider::ProviderThinkingPolicy { expose: true },
                ..ProviderOptions::default()
            })
            .build_request_body(&req)
            .expect("body");
        assert_eq!(exposed["thinking"]["display"], "summarized");
    }

    #[test]
    fn explicit_text_cache_breakpoint_beats_last_user_block() {
        let provider = AnthropicProvider::new("key");
        let req = request(vec![LlmMessage::new(
            LlmRole::User,
            vec![
                LlmContentBlock::Text {
                    text: "stable history".into(),
                    response_meta: None,
                    cache_breakpoint: true,
                },
                LlmContentBlock::Text {
                    text: "dynamic current iteration".into(),
                    response_meta: None,
                    cache_breakpoint: false,
                },
            ],
        )]);

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert!(
            body["messages"][0]["content"][1]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body["messages"][0]["content"][0]
                .get("__lash_cache_breakpoint")
                .is_none()
        );
    }

    #[test]
    fn cache_retention_none_removes_cache_control() {
        let provider = AnthropicProvider::new("key").with_options(ProviderOptions {
            cache_retention: CacheRetention::None,
            ..ProviderOptions::default()
        });
        let req = request(vec![
            LlmMessage::text(LlmRole::System, "stable system prompt"),
            LlmMessage::text(LlmRole::User, "dynamic tail"),
        ]);

        let body = provider.build_request_body(&req).expect("body");

        assert!(body["system"][0].get("cache_control").is_none());
        assert!(
            body["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[test]
    fn cache_retention_long_emits_ttl() {
        let provider = AnthropicProvider::new("key").with_options(ProviderOptions {
            cache_retention: CacheRetention::Long,
            ..ProviderOptions::default()
        });
        let req = request(vec![
            LlmMessage::text(LlmRole::System, "stable system prompt"),
            LlmMessage::text(LlmRole::User, "dynamic tail"),
        ]);

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(
            body["system"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
    }

    #[test]
    fn output_token_cap_maps_to_max_tokens() {
        let provider = AnthropicProvider::new("key").with_options(ProviderOptions {
            max_output_tokens: Some(9999),
            ..ProviderOptions::default()
        });
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        req.generation.output_token_cap = NonZeroUsize::new(2048);

        let body = provider.build_request_body(&req).expect("body");

        assert_eq!(body["max_tokens"], 2048);
        let provider_limited_body = provider
            .build_request_body(&request(vec![LlmMessage::text(LlmRole::User, "hello")]))
            .expect("body");
        assert_eq!(provider_limited_body["max_tokens"], 9999);
    }

    /// Cross-provider response-normalization conformance. Anthropic is
    /// streaming-first (no non-streaming `parts_from_value`), so each scenario's
    /// `body` carries the SSE event sequence as a JSON array of strings, and all
    /// three accessors replay it through `process_sse_event` + `finalize`.
    /// Anthropic reports no separate reasoning-token count (`parse_usage`
    /// hardcodes `reasoning_tokens: 0`), so it opts out of `UsageReasoning`.
    #[cfg(feature = "testing")]
    mod conformance {
        use super::*;
        use lash_llm_transport::conformance::{
            CanonicalUsage as U, ProviderNormalizer, ProviderWire, Scenario, StreamAssembly,
            provider_conformance,
        };
        use serde_json::{Value, json};

        struct AnthropicNormalizer;

        // Replay a `body` that encodes a JSON array of SSE event strings through
        // the streaming parser and finalize into normalized outputs.
        fn replay(body: &Value) -> (Vec<LlmOutputPart>, LlmUsage, LlmTerminalReason) {
            let mut state = StreamState::default();
            if let Some(events) = body.as_array() {
                for event in events {
                    let raw = event.as_str().expect("sse event is a string");
                    AnthropicProvider::process_sse_event(raw, &mut state, None, true)
                        .expect("anthropic sse event parses");
                }
            }
            let (parts, _text, usage, terminal) = AnthropicProvider::finalize(state);
            (parts, usage, terminal)
        }

        // Build the SSE event array for a plain single-text-block message with a
        // given stop_reason and message_start usage block.
        fn text_message(stop_reason: &str, text: &str, usage: Value) -> Value {
            json!([
                json!({ "type": "message_start", "message": { "usage": usage } }).to_string(),
                json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text" } }).to_string(),
                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }).to_string(),
                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
                json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason } }).to_string(),
            ])
        }

        impl ProviderNormalizer for AnthropicNormalizer {
            fn name(&self) -> &str {
                "anthropic"
            }

            fn wire_for(&self, scenario: Scenario) -> Option<ProviderWire> {
                let wire = match scenario {
                    Scenario::PlainTextStop => ProviderWire::body(text_message(
                        "end_turn",
                        "hello",
                        json!({ "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT }),
                    )),
                    Scenario::OutputCapped => ProviderWire::body(text_message(
                        "max_tokens",
                        "trunc",
                        json!({ "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT }),
                    )),
                    Scenario::ContentFilter => ProviderWire::body(text_message(
                        "refusal",
                        "",
                        json!({ "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT }),
                    )),
                    Scenario::ToolUse => {
                        let events = json!([
                            json!({ "type": "message_start", "message": { "usage": { "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT } } }).to_string(),
                            json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": "call_1", "name": "lookup", "input": {} } }).to_string(),
                            // arguments deliberately split across two delta events
                            json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": "{\"q\":" } }).to_string(),
                            json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": "\"x\"}" } }).to_string(),
                            json!({ "type": "content_block_stop", "index": 0 }).to_string(),
                            json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" } }).to_string(),
                        ]);
                        ProviderWire::body(events.clone()).with_tool_call_stream(
                            events
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_str().unwrap().to_string())
                                .collect(),
                            "lookup",
                            json!({ "q": "x" }),
                        )
                    }
                    Scenario::UsageCacheHit => ProviderWire::body(text_message(
                        "end_turn",
                        "ok",
                        // Anthropic's input_tokens is net of cache; the suite's
                        // canonical input is the gross total, so split it.
                        json!({
                            "input_tokens": U::BASE_INPUT - U::CACHED_INPUT,
                            "output_tokens": U::BASE_OUTPUT,
                            "cache_read_input_tokens": U::CACHED_INPUT
                        }),
                    )),
                    // Anthropic does not report a separate reasoning-token count.
                    Scenario::UsageReasoning => return None,
                    Scenario::ReasoningExtraction => ProviderWire::body(json!([
                        json!({ "type": "message_start", "message": { "usage": { "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT } } }).to_string(),
                        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "thinking" } }).to_string(),
                        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "thinking_delta", "thinking": "thinking about it" } }).to_string(),
                        json!({ "type": "content_block_stop", "index": 0 }).to_string(),
                        json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "text" } }).to_string(),
                        json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "text_delta", "text": "answer" } }).to_string(),
                        json!({ "type": "content_block_stop", "index": 1 }).to_string(),
                        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" } }).to_string(),
                    ]))
                    .with_reasoning_text("thinking about it"),
                    Scenario::StreamingUsageMerge => ProviderWire::body(Value::Null)
                        .with_usage_merge_stream(vec![
                            // input arrives in message_start
                            json!({ "type": "message_start", "message": { "usage": { "input_tokens": U::BASE_INPUT } } }).to_string(),
                            // output arrives later in message_delta; merge must keep input
                            json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": U::BASE_OUTPUT } }).to_string(),
                        ]),
                };
                Some(wire)
            }

            fn parts_from_wire(&self, body: &Value) -> Vec<LlmOutputPart> {
                replay(body).0
            }

            fn usage_from_wire(&self, body: &Value) -> LlmUsage {
                replay(body).1
            }

            fn terminal_from_wire(
                &self,
                body: &Value,
                _parts: &[LlmOutputPart],
            ) -> LlmTerminalReason {
                replay(body).2
            }

            fn assemble_stream(&self, sse_events: &[String]) -> StreamAssembly {
                let mut state = StreamState::default();
                for raw in sse_events {
                    AnthropicProvider::process_sse_event(raw, &mut state, None, true)
                        .expect("anthropic sse event parses");
                }
                let (parts, _text, usage, _terminal) = AnthropicProvider::finalize(state);
                StreamAssembly { parts, usage }
            }
        }

        #[test]
        fn anthropic_satisfies_provider_conformance() {
            provider_conformance(&AnthropicNormalizer);
        }
    }
}
