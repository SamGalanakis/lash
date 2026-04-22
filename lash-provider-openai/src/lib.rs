use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::LazyLock;

use lash::llm::streaming::drive_sse_response;
use lash::llm::timeouts::{
    build_http_client, read_response_text, request_body_snapshot_bytes, response_start_timeout,
    send_request,
};
use lash::llm::transport::LlmTransportError;
use lash::llm::types::{
    LlmContentBlock, LlmEventSender, LlmMessage, LlmOutputPart, LlmOutputSpec, LlmRequest,
    LlmResponse, LlmRole, LlmStreamEvent, LlmToolChoice, LlmUsage,
};
use lash::provider::{
    AgentModelSelection, Provider, ProviderFactory, ProviderOptions, VariantRequestConfig,
};

/// Well-known OpenRouter base URL; used at runtime to detect
/// OpenRouter-specific features (prompt caching, reasoning variants).
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

const OPENROUTER_REASONING_VARIANTS: &[&str] =
    &["none", "minimal", "low", "medium", "high", "xhigh"];

static DEFAULT_HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(build_http_client);

fn base_url_is_openrouter(base_url: &str) -> bool {
    base_url.trim_end_matches('/') == OPENROUTER_BASE_URL
}

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
///
/// Rust `&str` is guaranteed valid UTF-8, so unpaired surrogates cannot
/// appear here; the returned slice is the input unchanged. Kept as a
/// function so call sites can stay untouched if we ever need to restore
/// non-trivial sanitization.
fn sanitize_surrogates(s: &str) -> &str {
    s
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

/// OpenAI-compatible (API key) provider. Works with OpenRouter,
/// OpenAI, vLLM, and any other OpenAI-compatible backend. Set
/// `base_url` to point at the target endpoint; auto-detects
/// provider-specific quirks (max-tokens field, strict-mode support).
#[derive(Clone, Debug)]
pub struct OpenAiGenericProvider {
    pub api_key: String,
    pub base_url: String,
    pub options: ProviderOptions,
    client: reqwest::Client,
}

#[derive(Clone, Debug, Default)]
struct StreamingToolCall {
    id: String,
    name: String,
    arguments: String,
    /// Serialized `reasoning_details` entry keyed to this tool call's
    /// id. OpenRouter emits `{ type: "reasoning.encrypted", id, data }`
    /// on a separate channel, which we must echo back as the
    /// `reasoning_details` field on the assistant message to round-trip
    /// the encrypted chain-of-thought.
    signature: Option<String>,
}

/// Accumulator for reasoning summary text produced by providers that
/// expose it on the Chat Completions stream. Reasoning parts sit BEFORE
/// the assistant's visible text in the output list, matching the order
/// in which providers emit the deltas. A new reasoning part begins each
/// time reasoning resumes after a gap (e.g. after some text or a tool
/// call), so multi-segment reasoning stays addressable as separate
/// summary blocks.
#[derive(Debug, Default)]
struct ReasoningAccumulator {
    /// Finalized reasoning parts, in emission order.
    finished: Vec<String>,
    /// The in-progress reasoning segment. `None` when the last delta
    /// seen was not reasoning (so the next reasoning delta starts a
    /// fresh segment).
    current: Option<String>,
}

impl ReasoningAccumulator {
    fn push_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        match &mut self.current {
            Some(buf) => buf.push_str(piece),
            None => self.current = Some(piece.to_string()),
        }
    }

    /// Close the in-progress reasoning segment. Called whenever a
    /// non-reasoning signal arrives (text, tool-call arguments) so the
    /// next reasoning delta starts a new block rather than merging into
    /// the previous one.
    fn close_segment(&mut self) {
        if let Some(text) = self.current.take() {
            let trimmed = text.trim_end();
            if !trimmed.is_empty() {
                self.finished.push(trimmed.to_string());
            }
        }
    }

    /// Drain the accumulator into the finished list and return all
    /// reasoning parts in order. Consumes the accumulator.
    fn into_parts(mut self) -> Vec<String> {
        self.close_segment();
        self.finished
    }
}

struct SseEventState<'a> {
    full: &'a mut String,
    retained_deltas: Option<&'a mut Vec<String>>,
    usage: &'a mut LlmUsage,
    prev_usage: &'a LlmUsage,
    stream_events: Option<&'a LlmEventSender>,
    tool_calls: Option<&'a mut Vec<StreamingToolCall>>,
    provider_usage: Option<&'a mut Option<Value>>,
    reasoning: Option<&'a mut ReasoningAccumulator>,
}

impl<'a> SseEventState<'a> {
    fn new(full: &'a mut String, usage: &'a mut LlmUsage, prev_usage: &'a LlmUsage) -> Self {
        Self {
            full,
            retained_deltas: None,
            usage,
            prev_usage,
            stream_events: None,
            tool_calls: None,
            provider_usage: None,
            reasoning: None,
        }
    }

    fn with_retained_deltas_opt(mut self, retained_deltas: Option<&'a mut Vec<String>>) -> Self {
        self.retained_deltas = retained_deltas;
        self
    }

    fn with_stream_events(mut self, stream_events: Option<&'a LlmEventSender>) -> Self {
        self.stream_events = stream_events;
        self
    }

    fn with_tool_calls(mut self, tool_calls: &'a mut Vec<StreamingToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    fn with_provider_usage(mut self, provider_usage: &'a mut Option<Value>) -> Self {
        self.provider_usage = Some(provider_usage);
        self
    }

    fn with_reasoning(mut self, reasoning: &'a mut ReasoningAccumulator) -> Self {
        self.reasoning = Some(reasoning);
        self
    }
}

#[derive(Deserialize)]
struct OpenAiCompatStreamEvent {
    #[serde(default)]
    choices: Vec<OpenAiCompatChoice>,
    #[serde(default)]
    usage: Option<Value>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize)]
struct OpenAiCompatChoice {
    #[serde(default)]
    delta: Option<OpenAiCompatMessage>,
    #[serde(default)]
    message: Option<OpenAiCompatMessage>,
}

#[derive(Deserialize, Default)]
struct OpenAiCompatMessage {
    #[serde(default)]
    content: Option<OpenAiCompatContent>,
    #[serde(default)]
    tool_calls: Vec<OpenAiCompatToolCall>,
    // Reasoning summary text. Providers disagree on the field name:
    //  - `reasoning_content` (llama.cpp, DeepSeek, Chutes)
    //  - `reasoning` (OpenRouter-normalized, some Anthropic/xAI bridges)
    //  - `reasoning_text` (some OpenAI-compatible servers)
    // We accept all three. At most one is set per delta in practice;
    // the extraction helper picks the first non-empty one to avoid
    // duplicating text when a provider mirrors both fields.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_text: Option<String>,
    /// OpenRouter-only channel that carries encrypted reasoning. Each
    /// entry describes a discrete reasoning item (e.g. a Claude
    /// thinking block or OpenAI reasoning blob); when `type` is
    /// `reasoning.encrypted` and `id` is set, the `data` payload is the
    /// opaque blob we must round-trip on the next turn so the model
    /// accepts the previously-made tool call.
    #[serde(default)]
    reasoning_details: Vec<Value>,
}

impl OpenAiCompatMessage {
    /// Returns the first non-empty reasoning field present on this message.
    /// Providers sometimes populate multiple fields with the same content
    /// (e.g. chutes.ai sets both `reasoning_content` and `reasoning`); we
    /// pick the first match so callers don't double-accumulate.
    fn reasoning_text(&self) -> Option<&str> {
        for field in [
            self.reasoning_content.as_deref(),
            self.reasoning.as_deref(),
            self.reasoning_text.as_deref(),
        ] {
            if let Some(text) = field
                && !text.is_empty()
            {
                return Some(text);
            }
        }
        None
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAiCompatContent {
    Text(String),
    Parts(Vec<OpenAiCompatContentPart>),
}

#[derive(Deserialize)]
struct OpenAiCompatContentPart {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiCompatToolCall {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiCompatToolFunction>,
}

#[derive(Deserialize)]
struct OpenAiCompatToolFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Serialize)]
struct OpenAiTextOnlyBody<'a> {
    model: &'a str,
    messages: Vec<OpenAiTextOnlyMessage<'a>>,
    temperature: u8,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OpenAiStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiToolSpecWire<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

#[derive(Serialize)]
struct OpenAiTextOnlyMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Clone, Copy, Serialize)]
struct OpenAiStreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct OpenAiToolSpecWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiToolFunctionWire<'a>,
}

#[derive(Serialize)]
struct OpenAiToolFunctionWire<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

impl OpenAiGenericProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            options: ProviderOptions::default(),
            client: DEFAULT_HTTP_CLIENT.clone(),
        }
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.client = (*client).clone();
        self
    }

    fn role_name(role: &LlmRole) -> &'static str {
        match role {
            LlmRole::User => "user",
            LlmRole::Assistant => "assistant",
            LlmRole::System => "system",
        }
    }

    fn image_content_part(req: &LlmRequest, attachment_idx: usize) -> Value {
        if let Some(att) = req.attachments.get(attachment_idx) {
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

    fn build_messages(&self, req: &LlmRequest) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        let mut seen_first_system = false;

        for msg in &req.messages {
            // Split each lash message into:
            //  - an optional assistant/user/system textual message (with
            //    tool_calls attached for assistant turns), and
            //  - zero or more standalone `tool` messages for each
            //    ToolResult block (Chat Completions expects one tool
            //    message per tool_result).
            let mut text_parts: Vec<Value> = Vec::new();
            let mut tool_calls: Vec<Value> = Vec::new();
            let mut tool_results: Vec<Value> = Vec::new();
            let mut reasoning_details: Vec<Value> = Vec::new();
            for block in msg.blocks.iter() {
                match block {
                    LlmContentBlock::Text(text) => {
                        if text.is_empty() {
                            continue;
                        }
                        text_parts.push(json!({"type": "text", "text": sanitize_surrogates(text)}));
                    }
                    LlmContentBlock::Image { attachment_idx } => {
                        text_parts.push(Self::image_content_part(req, *attachment_idx));
                    }
                    LlmContentBlock::ToolCall {
                        call_id,
                        tool_name,
                        input_json,
                        signature,
                        ..
                    } => {
                        tool_calls.push(json!({
                            "id": normalize_tool_call_id(call_id),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": input_json,
                            }
                        }));
                        // Re-attach the encrypted reasoning detail so the
                        // upstream model (OR-routed Claude / GPT) accepts
                        // the stored tool call on replay.
                        if let Some(raw) = signature.as_deref()
                            && let Ok(detail) = serde_json::from_str::<Value>(raw)
                        {
                            reasoning_details.push(detail);
                        }
                    }
                    LlmContentBlock::ToolResult {
                        call_id, content, ..
                    } => {
                        tool_results.push(json!({
                            "role": "tool",
                            "tool_call_id": normalize_tool_call_id(call_id),
                            "content": sanitize_surrogates(content),
                        }));
                    }
                    LlmContentBlock::Reasoning { .. } => {
                        // Chat Completions has no canonical reasoning replay
                        // channel. Drop the block.
                    }
                }
            }

            // Collapse a single-text content array back to a bare string,
            // the common shape Chat Completions expects for plain turns.
            let content_value: Option<Value> = if text_parts.is_empty() {
                if !tool_calls.is_empty() {
                    Some(json!(""))
                } else {
                    None
                }
            } else if text_parts.len() == 1
                && text_parts[0].get("type").and_then(|v| v.as_str()) == Some("text")
            {
                Some(json!(
                    text_parts[0]
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                ))
            } else {
                Some(Value::Array(text_parts))
            };

            if let Some(content) = content_value {
                let role = match msg.role {
                    LlmRole::System => {
                        // The first system message is the real system
                        // prompt; later system messages (runtime feedback)
                        // become user turns so the conversation ends on a
                        // user boundary — required by providers like
                        // Claude via OpenRouter.
                        if seen_first_system {
                            "user"
                        } else {
                            seen_first_system = true;
                            "system"
                        }
                    }
                    _ => Self::role_name(&msg.role),
                };

                // Merge consecutive user-role turns into a single multipart
                // content array (OpenAI-compatible vision APIs prefer one
                // user message carrying all user content for the turn).
                if role == "user"
                    && let Some(prev) = out.last_mut()
                    && prev.get("role").and_then(|r| r.as_str()) == Some("user")
                    && prev.get("tool_call_id").is_none()
                {
                    let prev_content = prev["content"].clone();
                    let mut merged = match prev_content {
                        Value::Array(arr) => arr,
                        Value::String(s) => vec![json!({"type": "text", "text": s})],
                        _ => Vec::new(),
                    };
                    match content {
                        Value::Array(arr) => merged.extend(arr),
                        Value::String(s) => merged.push(json!({"type": "text", "text": s})),
                        other => merged.push(other),
                    }
                    prev["content"] = Value::Array(merged);
                } else {
                    let mut item = json!({
                        "role": role,
                        "content": content,
                    });
                    if !tool_calls.is_empty() {
                        item["tool_calls"] = Value::Array(tool_calls.clone());
                    }
                    if !reasoning_details.is_empty() {
                        item["reasoning_details"] = Value::Array(reasoning_details.clone());
                    }
                    out.push(item);
                }
            } else if !tool_calls.is_empty() {
                let mut item = json!({
                    "role": Self::role_name(&msg.role),
                    "content": "",
                    "tool_calls": tool_calls,
                });
                if !reasoning_details.is_empty() {
                    item["reasoning_details"] = Value::Array(reasoning_details);
                }
                out.push(item);
            }

            out.extend(tool_results);
        }

        out
    }

    fn has_tool_history(messages: &[LlmMessage]) -> bool {
        messages.iter().any(|msg| {
            msg.blocks.iter().any(|block| {
                matches!(
                    block,
                    LlmContentBlock::ToolCall { .. } | LlmContentBlock::ToolResult { .. }
                )
            })
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

    /// Clamp a requested reasoning-effort level to what the target model
    /// actually supports on the Chat Completions API. Mirrors pi's
    /// `clampReasoning` + `supportsXhigh` logic from
    /// `packages/ai/src/providers/openai-completions.ts`:
    ///
    /// * `xhigh` is only valid on GPT-5.2/5.3/5.4 families and
    ///   Opus 4.6/4.7. For everything else it collapses to `high`.
    /// * All other values (`minimal`, `low`, `medium`, `high`) pass
    ///   through untouched.
    ///
    /// The `model` string is matched case-insensitively against the raw
    /// provider-qualified id (e.g. `"openai/gpt-5.4"`,
    /// `"anthropic/claude-opus-4.7"`).
    pub(crate) fn clamp_reasoning_effort_chat(model: &str, effort: &str) -> String {
        let normalized = effort.trim().to_ascii_lowercase();
        if normalized != "xhigh" {
            return normalized;
        }
        let model_lc = model.to_ascii_lowercase();
        let supports_xhigh = model_lc.contains("gpt-5.2")
            || model_lc.contains("gpt-5.3")
            || model_lc.contains("gpt-5.4")
            || model_lc.contains("opus-4-6")
            || model_lc.contains("opus-4.6")
            || model_lc.contains("opus-4-7")
            || model_lc.contains("opus-4.7");
        if supports_xhigh {
            "xhigh".to_string()
        } else {
            "high".to_string()
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

    fn max_output_tokens() -> u64 {
        // The cap is 32k by default; benchmarks that need longer outputs
        // (e.g. long-horizon CoT) can raise it via `LASH_MAX_OUTPUT_TOKENS`.
        std::env::var("LASH_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|v: &u64| *v > 0)
            .unwrap_or(32768)
    }

    fn try_build_text_only_request_body(
        &self,
        req: &LlmRequest,
        stream: bool,
        base_url: &str,
        compat: &OpenAiCompat,
    ) -> Option<Result<Vec<u8>, LlmTransportError>> {
        if req.output_spec.is_some() || req.model_variant.is_some() {
            return None;
        }
        if Self::supports_prompt_cache_key(base_url) && req.model.starts_with("anthropic/") {
            return None;
        }

        let mut messages = Vec::with_capacity(req.messages.len());
        let mut seen_first_system = false;
        for msg in &req.messages {
            let mut text = None;
            for block in msg.blocks.iter() {
                match block {
                    LlmContentBlock::Text(value) if value.is_empty() => {}
                    LlmContentBlock::Text(value) if text.is_none() => {
                        text = Some(sanitize_surrogates(value));
                    }
                    LlmContentBlock::Text(_) => return None,
                    LlmContentBlock::Image { .. }
                    | LlmContentBlock::ToolCall { .. }
                    | LlmContentBlock::ToolResult { .. }
                    | LlmContentBlock::Reasoning { .. } => return None,
                }
            }
            let Some(content) = text else {
                continue;
            };
            let role = match msg.role {
                LlmRole::System => {
                    if seen_first_system {
                        "user"
                    } else {
                        seen_first_system = true;
                        "system"
                    }
                }
                _ => Self::role_name(&msg.role),
            };
            if role == "user"
                && messages
                    .last()
                    .is_some_and(|message: &OpenAiTextOnlyMessage<'_>| message.role == "user")
            {
                return None;
            }
            messages.push(OpenAiTextOnlyMessage { role, content });
        }

        let max_output_tokens = Self::max_output_tokens();
        let tools = (!req.tools.is_empty()).then(|| {
            req.tools
                .iter()
                .map(|tool| OpenAiToolSpecWire {
                    kind: "function",
                    function: OpenAiToolFunctionWire {
                        name: &tool.name,
                        description: &tool.description,
                        parameters: &tool.input_schema,
                        strict: compat.supports_strict_mode.then_some(false),
                    },
                })
                .collect::<Vec<_>>()
        });
        let tool_choice = tools.as_ref().map(|_| match req.tool_choice {
            LlmToolChoice::Auto => "auto",
            LlmToolChoice::None => "none",
            LlmToolChoice::Required => "required",
        });
        let body = OpenAiTextOnlyBody {
            model: &req.model,
            messages,
            temperature: 0,
            stream,
            max_tokens: compat.use_max_tokens_field.then_some(max_output_tokens),
            max_completion_tokens: (!compat.use_max_tokens_field).then_some(max_output_tokens),
            stream_options: (stream && compat.supports_usage_in_streaming).then_some(
                OpenAiStreamOptions {
                    include_usage: true,
                },
            ),
            prompt_cache_key: req
                .session_id
                .as_deref()
                .filter(|_| Self::supports_prompt_cache_key(base_url)),
            tools,
            tool_choice,
        };

        Some(serde_json::to_vec(&body).map_err(|e| {
            LlmTransportError::new(format!("Failed to serialize OpenAI-compatible body: {e}"))
        }))
    }

    fn build_request_body_bytes(
        &self,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<(Vec<u8>, String), LlmTransportError> {
        let base_url = self.base_url.clone();
        let compat = detect_compat(&base_url);
        if let Some(body) = self.try_build_text_only_request_body(req, stream, &base_url, &compat) {
            return body.map(|body| (body, base_url));
        }

        let (body, base_url) =
            self.build_request_body_with_compat(req, stream, base_url, compat)?;
        let body = serde_json::to_vec(&body).map_err(|e| {
            LlmTransportError::new(format!("Failed to serialize OpenAI-compatible body: {e}"))
        })?;
        Ok((body, base_url))
    }

    fn build_request_body_with_compat(
        &self,
        req: &LlmRequest,
        stream: bool,
        base_url: String,
        compat: OpenAiCompat,
    ) -> Result<(Value, String), LlmTransportError> {
        let mut messages = self.build_messages(req);
        Self::maybe_add_openrouter_anthropic_cache_control(&mut messages, &req.model, &base_url);

        let mut body = json!({
            "model": req.model,
            "messages": messages,
            "temperature": 0,
            "stream": stream,
        });

        // Use the correct max-tokens field name per provider.
        let max_output_tokens = Self::max_output_tokens();
        if compat.use_max_tokens_field {
            body["max_tokens"] = json!(max_output_tokens);
        } else {
            body["max_completion_tokens"] = json!(max_output_tokens);
        }

        if let Some(variant) = req.model_variant.as_deref()
            && let Some(VariantRequestConfig::ReasoningEffort(effort)) =
                self.request_variant_config(&req.model, variant)
        {
            let clamped = Self::clamp_reasoning_effort_chat(&req.model, &effort);
            if Self::is_openrouter(&base_url) {
                // OpenRouter normalizes reasoning via a nested reasoning object.
                body["reasoning"] = json!({ "effort": clamped });
            } else {
                body["reasoning_effort"] = json!(clamped);
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
                LlmToolChoice::Auto => json!("auto"),
                LlmToolChoice::None => json!("none"),
                LlmToolChoice::Required => json!("required"),
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
        Some(Self::usage_from_usage_value(value.get("usage")?))
    }

    fn usage_from_usage_value(usage: &Value) -> LlmUsage {
        LlmUsage {
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
        }
    }

    fn provider_usage_from_value(value: &Value) -> Option<Value> {
        value.get("usage").cloned()
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

    fn emit_added_delta(
        retained_deltas: &mut Option<&mut Vec<String>>,
        stream_events: Option<&LlmEventSender>,
        delta: String,
    ) {
        match (retained_deltas.as_mut(), stream_events) {
            (Some(dst), Some(tx)) => {
                (**dst).push(delta.clone());
                tx.send(LlmStreamEvent::Delta(delta));
            }
            (Some(dst), None) => (**dst).push(delta),
            (None, Some(tx)) => tx.send(LlmStreamEvent::Delta(delta)),
            (None, None) => {}
        }
    }

    fn append_added_delta_owned(
        full: &mut String,
        retained_deltas: &mut Option<&mut Vec<String>>,
        stream_events: Option<&LlmEventSender>,
        piece: String,
    ) {
        if piece.is_empty() {
            return;
        }
        if piece.starts_with(full.as_str()) {
            let delta_start = full.len();
            if delta_start < piece.len() {
                full.push_str(&piece[delta_start..]);
                Self::emit_added_delta(
                    retained_deltas,
                    stream_events,
                    piece[delta_start..].to_string(),
                );
            }
            return;
        }
        full.push_str(piece.as_str());
        Self::emit_added_delta(retained_deltas, stream_events, piece);
    }

    fn process_stream_content(
        full: &mut String,
        retained_deltas: &mut Option<&mut Vec<String>>,
        stream_events: Option<&LlmEventSender>,
        content: OpenAiCompatContent,
    ) {
        match content {
            OpenAiCompatContent::Text(text) => {
                Self::append_added_delta_owned(full, retained_deltas, stream_events, text);
            }
            OpenAiCompatContent::Parts(parts) => {
                for part in parts {
                    if let Some(text) = part.text {
                        Self::append_added_delta_owned(full, retained_deltas, stream_events, text);
                    }
                }
            }
        }
    }

    fn append_stream_piece(full: &mut String, piece: &str) -> Option<String> {
        if piece.is_empty() {
            return None;
        }
        if piece.starts_with(full.as_str()) {
            let delta = &piece[full.len()..];
            if !delta.is_empty() {
                full.push_str(delta);
                return Some(delta.to_string());
            }
            return None;
        }
        full.push_str(piece);
        Some(piece.to_string())
    }

    fn process_sse_event_with_tools(
        raw: &str,
        mut state: SseEventState<'_>,
    ) -> Result<(), LlmTransportError> {
        let raw = raw.trim();
        if raw.is_empty() || raw == "[DONE]" {
            return Ok(());
        }
        let event: OpenAiCompatStreamEvent = serde_json::from_str(raw).map_err(|e| {
            LlmTransportError::new(format!("Invalid SSE payload: {e}")).with_raw(raw)
        })?;
        if let Some(err) = event.error.as_ref() {
            let mut transport_error = LlmTransportError::new("OpenAI-compatible stream error")
                .with_raw(err.to_string())
                .retryable(Self::stream_error_is_retryable(err));
            if let Some(code) = Self::stream_error_code(err) {
                transport_error = transport_error.with_code(code);
            }
            return Err(transport_error);
        }
        if let Some(new_usage) = event.usage.as_ref().map(Self::usage_from_usage_value)
            && (new_usage.input_tokens > 0
                || new_usage.output_tokens > 0
                || new_usage.cached_input_tokens > 0
                || new_usage.reasoning_tokens > 0)
        {
            *state.usage = new_usage;
        }
        if let Some(dst) = state.provider_usage.as_deref_mut()
            && let Some(raw_usage) = event.usage.clone()
        {
            *dst = Some(raw_usage);
        }
        if let Some(tx) = state.stream_events
            && state.usage != state.prev_usage
            && state.usage != &LlmUsage::default()
        {
            tx.send(LlmStreamEvent::Usage(state.usage.clone()));
        }
        for mut choice in event.choices {
            // Reasoning arrives before text on providers that emit it;
            // apply it first so ordering in the final parts list matches.
            let mut handled_reasoning_from_delta = false;
            if let Some(delta) = choice.delta.as_mut()
                && let Some(piece) = delta.reasoning_text()
            {
                Self::handle_reasoning_piece(
                    state.reasoning.as_deref_mut(),
                    state.stream_events,
                    piece,
                );
                handled_reasoning_from_delta = true;
            }
            if !handled_reasoning_from_delta
                && let Some(message) = choice.message.as_mut()
                && let Some(piece) = message.reasoning_text()
            {
                Self::handle_reasoning_piece(
                    state.reasoning.as_deref_mut(),
                    state.stream_events,
                    piece,
                );
            }

            if let Some(delta) = choice.delta.as_mut()
                && let Some(content) = delta.content.take()
            {
                // A content delta means we're no longer streaming reasoning;
                // close the segment so later reasoning (if any) starts fresh.
                if let Some(acc) = state.reasoning.as_deref_mut() {
                    acc.close_segment();
                }
                Self::process_stream_content(
                    state.full,
                    &mut state.retained_deltas,
                    state.stream_events,
                    content,
                );
            }
            if let Some(message) = choice.message.as_mut()
                && let Some(content) = message.content.take()
            {
                if let Some(acc) = state.reasoning.as_deref_mut() {
                    acc.close_segment();
                }
                Self::process_stream_content(
                    state.full,
                    &mut state.retained_deltas,
                    state.stream_events,
                    content,
                );
            }

            // Accumulate streaming tool call deltas.
            if let Some(tool_calls) = state.tool_calls.as_deref_mut() {
                for tc in choice
                    .delta
                    .as_ref()
                    .into_iter()
                    .flat_map(|delta| delta.tool_calls.iter())
                {
                    // Tool-call activity also ends any in-progress
                    // reasoning segment so the next reasoning burst is
                    // its own part.
                    if let Some(acc) = state.reasoning.as_deref_mut() {
                        acc.close_segment();
                    }
                    let index = tc.index.unwrap_or(0) as usize;
                    while tool_calls.len() <= index {
                        tool_calls.push(StreamingToolCall::default());
                    }
                    if let Some(id) = tc.id.as_ref() {
                        tool_calls[index].id.clone_from(id);
                    }
                    if let Some(function) = tc.function.as_ref() {
                        if let Some(name) = function.name.as_ref() {
                            tool_calls[index].name.clone_from(name);
                        }
                        if let Some(args) = function.arguments.as_ref() {
                            tool_calls[index].arguments.push_str(args);
                        }
                    }
                }

                // OpenRouter emits encrypted reasoning alongside the
                // assistant's visible output. Pair each `reasoning.encrypted`
                // entry with the tool call whose id it references so we can
                // replay it as `reasoning_details` on the next request.
                for details in choice
                    .delta
                    .as_ref()
                    .map(|d| d.reasoning_details.as_slice())
                    .into_iter()
                    .chain(
                        choice
                            .message
                            .as_ref()
                            .map(|m| m.reasoning_details.as_slice()),
                    )
                {
                    for detail in details {
                        Self::attach_reasoning_detail(tool_calls, detail);
                    }
                }
            }
        }
        Ok(())
    }

    /// Store an OpenRouter `reasoning_details` entry on its matching
    /// tool call by id. The entry is kept as a full JSON string so we
    /// can splat it back into the next request unchanged.
    fn attach_reasoning_detail(tool_calls: &mut [StreamingToolCall], detail: &Value) {
        let detail_type = detail.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if detail_type != "reasoning.encrypted" {
            return;
        }
        let Some(id) = detail.get("id").and_then(|v| v.as_str()) else {
            return;
        };
        let Some(call) = tool_calls.iter_mut().find(|tc| tc.id == id) else {
            return;
        };
        if let Ok(encoded) = serde_json::to_string(detail) {
            call.signature = Some(encoded);
        }
    }

    fn handle_reasoning_piece(
        reasoning: Option<&mut ReasoningAccumulator>,
        stream_events: Option<&LlmEventSender>,
        piece: &str,
    ) {
        if piece.is_empty() {
            return;
        }
        if let Some(acc) = reasoning {
            acc.push_delta(piece);
        }
        if let Some(tx) = stream_events {
            tx.send(LlmStreamEvent::ReasoningDelta(piece.to_string()));
        }
    }

    fn stream_error_code(err: &Value) -> Option<String> {
        err.get("code")
            .or_else(|| err.get("status"))
            .and_then(|value| match value {
                Value::Number(number) => number.as_i64().map(|v| v.to_string()),
                Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
                _ => None,
            })
    }

    fn stream_error_is_retryable(err: &Value) -> bool {
        let code = err
            .get("code")
            .or_else(|| err.get("status"))
            .and_then(|value| match value {
                Value::Number(number) => number.as_i64(),
                Value::String(text) => text.trim().parse::<i64>().ok(),
                _ => None,
            });
        if matches!(code, Some(429)) || matches!(code, Some(status) if status >= 500) {
            return true;
        }
        matches!(
            err.get("metadata")
                .and_then(|meta| meta.get("error_type"))
                .and_then(|value| value.as_str()),
            Some("provider_unavailable")
        )
    }

    fn parse_non_stream_response(raw: &str) -> Result<(String, LlmUsage), LlmTransportError> {
        let value: Value = serde_json::from_str(raw).map_err(|e| {
            LlmTransportError::new(format!("Invalid response JSON: {e}")).with_raw(raw.to_string())
        })?;
        let mut full = String::new();
        for piece in Self::extract_text_parts(&value) {
            let _ = Self::append_stream_piece(&mut full, &piece);
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

        // Non-streaming path: some providers (e.g. llama.cpp, DeepSeek,
        // OpenRouter-normalized) include reasoning alongside the final
        // message. Check all three field names, matching pi's behavior.
        // `reasoning` is placed before the visible text so replay order
        // matches the streaming path.
        for field in ["reasoning_content", "reasoning", "reasoning_text"] {
            if let Some(text) = message.get(field).and_then(|v| v.as_str())
                && !text.is_empty()
            {
                parts.push(LlmOutputPart::Reasoning {
                    text: text.to_string(),
                    signature: None,
                    redacted: false,
                    item_id: None,
                    encrypted_content: None,
                    summary: Vec::new(),
                });
                break;
            }
        }

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
                    item_id: None,
                    signature: None,
                });
            }
        }

        parts
    }
}

#[async_trait]
impl Provider for OpenAiGenericProvider {
    fn kind(&self) -> &'static str {
        "openai-compatible"
    }

    fn default_model(&self) -> &str {
        "anthropic/claude-sonnet-4.6"
    }

    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        if !base_url_is_openrouter(&self.base_url) {
            return &[];
        }
        let lower = model.to_ascii_lowercase();
        if lower.contains("gpt") || lower.contains("claude") || lower.contains("gemini-3") {
            OPENROUTER_REASONING_VARIANTS
        } else {
            &[]
        }
    }

    fn default_model_variant(&self, model: &str) -> Option<&'static str> {
        let variants = self.supported_variants(model);
        if variants.is_empty() {
            return None;
        }
        let lower = model.to_ascii_lowercase();
        if lower.contains("gpt") {
            Some("medium")
        } else {
            Some("high")
        }
    }

    fn request_variant_config(&self, model: &str, variant: &str) -> Option<VariantRequestConfig> {
        if self.validate_variant(model, variant).is_err() {
            return None;
        }
        if base_url_is_openrouter(&self.base_url) {
            Some(VariantRequestConfig::ReasoningEffort(variant.to_string()))
        } else {
            None
        }
    }

    fn default_agent_model(&self, tier: &str) -> Option<AgentModelSelection> {
        match tier {
            "low" => Some(AgentModelSelection {
                model: "minimax/minimax-m2.5".to_string(),
                variant: None,
            }),
            "medium" => Some(AgentModelSelection {
                model: "z-ai/glm-5".to_string(),
                variant: None,
            }),
            "high" => Some(AgentModelSelection {
                model: "anthropic/claude-sonnet-4.6".to_string(),
                variant: Some("high".to_string()),
            }),
            _ => None,
        }
    }

    fn context_lookup_model(&self, model: &str) -> String {
        if model.starts_with("openrouter/") {
            model.to_string()
        } else {
            format!("openrouter/{model}")
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
        let api_key = self.api_key.clone();

        let (body_bytes, base_url) =
            self.build_request_body_bytes(&req, stream_events.is_some())?;

        // Serialize once. reqwest's `.json(&body)` would re-run
        // `serde_json::to_vec` internally, wasting a large allocation per
        // request on long histories. Keep the bytes cheaply cloneable for
        // transport errors, but do not attach a full request copy to
        // successful responses; the runtime fills a compact debug request
        // when LLM logging is enabled.
        let request_body = request_body_snapshot_bytes(body_bytes);
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let request = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .body(request_body.clone());
        let resp = send_request(
            request,
            Some(request_body.clone()),
            response_start_timeout(
                timeouts.request_timeout,
                timeouts.chunk_timeout,
                stream_events.is_some(),
            ),
            "OpenAI-compatible response start timed out",
        )
        .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = read_response_text(
                resp,
                timeouts.request_timeout,
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
                request_body: Some(String::from_utf8_lossy(&request_body).into_owned()),
            });
        }
        drop(request_body);

        let is_sse = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);

        if !is_sse {
            let text = read_response_text(
                resp,
                timeouts.request_timeout,
                "OpenAI-compatible response body timed out",
            )
            .await?;
            let (content, usage) = Self::parse_non_stream_response(&text)?;
            let value: Value = serde_json::from_str(&text).map_err(|e| {
                LlmTransportError::new(format!("Invalid response JSON: {e}")).with_raw(text.clone())
            })?;
            let provider_usage = Self::provider_usage_from_value(&value);
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
                // Replay any reasoning captured on the non-streaming
                // response so the UI still gets to render it. Emit as a
                // single delta per reasoning part (consumers accumulate
                // by convention).
                for part in &parts {
                    if let LlmOutputPart::Reasoning { text, .. } = part
                        && !text.is_empty()
                    {
                        tx.send(LlmStreamEvent::ReasoningDelta(text.clone()));
                    }
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
                provider_usage,
                request_body: None,
                http_summary: Some(format!("HTTP POST {}", url)),
            });
        }

        let mut full = String::new();
        let mut retained_deltas = stream_events.is_none().then(Vec::new);
        let mut usage = LlmUsage::default();
        let mut provider_usage = None;
        let mut streaming_tool_calls: Vec<StreamingToolCall> = Vec::new();
        let mut reasoning_acc = ReasoningAccumulator::default();
        drive_sse_response(
            resp,
            timeouts.chunk_timeout,
            "OpenAI-compatible stream chunk timed out",
            |raw| {
                let prev_usage = usage.clone();
                Self::process_sse_event_with_tools(
                    raw,
                    SseEventState::new(&mut full, &mut usage, &prev_usage)
                        .with_retained_deltas_opt(retained_deltas.as_mut())
                        .with_stream_events(stream_events.as_ref())
                        .with_tool_calls(&mut streaming_tool_calls)
                        .with_provider_usage(&mut provider_usage)
                        .with_reasoning(&mut reasoning_acc),
                )?;
                Ok(())
            },
        )
        .await?;

        let mut parts = Vec::new();
        // Reasoning precedes the assistant text in the output list: it
        // was emitted first over the wire, and consumers render it as
        // pre-answer "thinking". Filter empties so spurious frames don't
        // produce ghost reasoning parts.
        for text in reasoning_acc.into_parts() {
            if !text.is_empty() {
                parts.push(LlmOutputPart::Reasoning {
                    text,
                    signature: None,
                    redacted: false,
                    item_id: None,
                    encrypted_content: None,
                    summary: Vec::new(),
                });
            }
        }
        if !full.is_empty() {
            parts.push(LlmOutputPart::Text { text: full.clone() });
        }
        for tc in &streaming_tool_calls {
            if !tc.id.is_empty() && !tc.name.is_empty() {
                parts.push(LlmOutputPart::ToolCall {
                    call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    input_json: tc.arguments.clone(),
                    item_id: None,
                    signature: tc.signature.clone(),
                });
            }
        }

        Ok(LlmResponse {
            deltas: retained_deltas.unwrap_or_default(),
            full_text: full,
            parts,
            usage,
            provider_usage,
            request_body: None,
            http_summary: Some(format!("HTTP POST {} (stream)", url)),
        })
    }

    fn serialize_config(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "api_key".to_string(),
            serde_json::Value::String(self.api_key.clone()),
        );
        map.insert(
            "base_url".to_string(),
            serde_json::Value::String(self.base_url.clone()),
        );
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

#[derive(Deserialize)]
struct OpenAiProviderConfig {
    api_key: String,
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    options: ProviderOptions,
}

/// Factory that registers [`OpenAiGenericProvider`] with lash's global
/// provider registry. Hosts call [`Self::register`] once at startup.
pub struct OpenAiGenericProviderFactory;

impl OpenAiGenericProviderFactory {
    pub fn register() {
        lash::register_provider_factory(std::sync::Arc::new(Self));
    }
}

impl ProviderFactory for OpenAiGenericProviderFactory {
    fn kind(&self) -> &'static str {
        "openai-compatible"
    }
    fn cli_label(&self) -> &'static str {
        "OpenAI-compatible (API key)"
    }
    fn setup_name(&self) -> &'static str {
        "OpenAI-compatible"
    }
    fn setup_description(&self) -> &'static str {
        "Any OpenAI-compatible API endpoint"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<Box<dyn Provider>, String> {
        let cfg: OpenAiProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(Box::new(OpenAiGenericProvider {
            api_key: cfg.api_key,
            base_url: cfg.base_url,
            options: cfg.options,
            client: build_http_client(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use lash::llm::types::LlmToolSpec;

    fn text_request(messages: Vec<LlmMessage>) -> LlmRequest {
        LlmRequest {
            model: "mock-model".to_string(),
            messages,
            attachments: Vec::new(),
            tools: Arc::new(Vec::<LlmToolSpec>::new()),
            tool_choice: LlmToolChoice::Auto,
            model_variant: None,
            session_id: Some("session-1".to_string()),
            output_spec: None,
            stream_events: None,
        }
    }

    #[test]
    fn text_only_request_fast_path_matches_compat_builder() {
        let provider = OpenAiGenericProvider::new("test-key", "https://example.invalid/v1");
        let req = text_request(vec![
            LlmMessage::text(LlmRole::System, "system prompt"),
            LlmMessage::text(LlmRole::User, "hello"),
            LlmMessage::text(LlmRole::Assistant, "hi"),
        ]);
        let base_url = provider.base_url.clone();
        let compat = detect_compat(&base_url);

        let fast = provider
            .try_build_text_only_request_body(&req, true, &base_url, &compat)
            .expect("text request should use fast path")
            .expect("serialize fast request");
        let (fallback, _) = provider
            .build_request_body_with_compat(&req, true, base_url, compat)
            .expect("build fallback request");

        assert_eq!(
            serde_json::from_slice::<Value>(&fast).expect("parse fast request"),
            fallback
        );
    }

    #[test]
    fn text_only_request_fast_path_defers_for_consecutive_user_messages() {
        let provider = OpenAiGenericProvider::new("test-key", "https://example.invalid/v1");
        let req = text_request(vec![
            LlmMessage::text(LlmRole::User, "first"),
            LlmMessage::text(LlmRole::User, "second"),
        ]);
        let base_url = provider.base_url.clone();
        let compat = detect_compat(&base_url);

        assert!(
            provider
                .try_build_text_only_request_body(&req, true, &base_url, &compat)
                .is_none()
        );
    }

    #[test]
    fn stream_reasoning_prefers_delta_over_mirrored_message_field() {
        let raw = r#"{"choices":[{"delta":{"reasoning_content":"Thinking"},"message":{"reasoning_content":"Thinking"}}]}"#;
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let tx = LlmEventSender::new(move |event| {
            captured.lock().expect("events lock").push(event);
        });
        let mut full = String::new();
        let mut usage = LlmUsage::default();
        let prev_usage = usage.clone();
        let mut reasoning = ReasoningAccumulator::default();

        OpenAiGenericProvider::process_sse_event_with_tools(
            raw,
            SseEventState::new(&mut full, &mut usage, &prev_usage)
                .with_stream_events(Some(&tx))
                .with_reasoning(&mut reasoning),
        )
        .expect("process event");

        assert_eq!(reasoning.into_parts(), vec!["Thinking"]);
        let reasoning_events = events
            .lock()
            .expect("events lock")
            .iter()
            .filter(
                |event| matches!(event, LlmStreamEvent::ReasoningDelta(text) if text == "Thinking"),
            )
            .count();
        assert_eq!(reasoning_events, 1);
    }
}
