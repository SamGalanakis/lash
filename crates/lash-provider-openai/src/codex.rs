//! OpenAI Codex OAuth provider (ChatGPT Plus/Pro/Team via device-code flow).

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;

use crate::responses_shared as shared;
use crate::schema::{OpenAiSchemaProfile, model_id};
use lash_core::SchemaProjectionOverride;
use lash_core::llm::transport::{LlmTransportError, ProviderFailure, ProviderFailureKind};
use lash_core::llm::types::{LlmOutputSpec, LlmRequest, LlmResponse, LlmStreamEvent, LlmUsage};
use lash_core::provider::{
    CacheRetention, DefaultProviderFailureClassifier, Provider, ProviderComponents,
    ProviderFactory, ProviderFailureClassifier, ProviderModelPolicy, ProviderOptions,
    ProviderReliability, resolve_generation_policy,
};
use lash_llm_transport::streaming::{drive_sse_response, emit_stream_progress};
use lash_llm_transport::timeouts::{
    build_http_client, header_pairs, read_response_text, request_body_snapshot,
    response_start_timeout, send_request,
};
use lash_llm_transport::util::emit_provider_trace;
use lash_llm_transport::{
    LlmHttpBody, openai_terminal_reason_from_response_value, openai_usage_from_response_value,
};

pub mod oauth;

/// Provider name used in shared-machinery error messages and trace events.
const PROVIDER: &str = "Codex";

const OPENAI_GPT5_VARIANTS: &[&str] = &["minimal", "low", "medium", "high"];
const OPENAI_GPT5_XHIGH_VARIANTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh"];
const OPENAI_GPT55_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];
const CODEX_VARIANTS: &[&str] = &["low", "medium", "high"];
const CODEX_XHIGH_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 32_768;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodexTransport {
    #[default]
    Auto,
    Sse,
    Websocket,
    WebsocketCached,
}

#[derive(Clone, Debug, Default)]
struct CodexContinuation {
    previous_response_id: String,
    input: Vec<Value>,
    body_fingerprint: String,
}

#[derive(Clone, Debug, Default)]
struct CodexContinuationCache {
    by_session: HashMap<String, CodexContinuation>,
}

struct CodexWebSocketAttemptError {
    error: LlmTransportError,
    events_seen: bool,
}

fn has_xhigh_suffix(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("5.2") || lower.contains("5.3") || lower.contains("5.4")
}

/// OpenAI Codex OAuth provider (ChatGPT Plus/Pro/Team via device-code flow).
///
/// Codex speaks the OpenAI Responses streaming protocol, so the request/stream
/// machinery is shared verbatim from [`crate::responses_shared`].
/// This module owns only the Codex-specific surface: the
/// `chatgpt.com/backend-api/codex/responses` endpoint, the `codex_cli_rs`
/// originator/User-Agent headers, the system→`instructions` request shape with
/// tool-result image folding, `clamp_reasoning_effort`, and Codex error/quota
/// classification.
#[derive(Clone, Debug)]
pub struct CodexProvider {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
    pub account_id: Option<String>,
    pub options: ProviderOptions,
    pub transport: CodexTransport,
    continuation_cache: CodexContinuationCache,
    client: reqwest::Client,
}

#[derive(Clone, Debug)]
struct CodexModelPolicy;

impl CodexProvider {
    const CODEX_ORIGINATOR: &'static str = "codex_cli_rs";
    const CODEX_RESPONSES_URL: &'static str = "https://chatgpt.com/backend-api/codex/responses";
    const CODEX_RESPONSES_WS_URL: &'static str = "wss://chatgpt.com/backend-api/codex/responses";

    pub fn new(
        access_token: impl Into<String>,
        refresh_token: impl Into<String>,
        expires_at: u64,
    ) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: refresh_token.into(),
            expires_at,
            account_id: None,
            options: ProviderOptions {
                reliability: ProviderReliability::codex(),
                ..ProviderOptions::default()
            },
            transport: CodexTransport::Auto,
            continuation_cache: CodexContinuationCache::default(),
            client: build_http_client(),
        }
    }

    pub fn with_account_id(mut self, account_id: Option<String>) -> Self {
        self.account_id = account_id;
        self
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_transport(mut self, transport: CodexTransport) -> Self {
        self.transport = transport;
        self
    }

    pub fn with_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.client = (*client).clone();
        self
    }

    /// Translate a Codex error body into a user-friendly one-line message.
    /// Mirrors pi-mono's `openai-codex-responses.ts:880-904`: for a
    /// `usage_limit_reached`/`rate_limit_exceeded` code (or any 429),
    /// parse the `plan_type` and `resets_at` epoch and render
    /// `"You have hit your ChatGPT usage limit (plus plan). Try again in
    /// ~12 min."`. Returns `None` when the body isn't parseable or the
    /// status doesn't match the pattern, so the caller falls back to the
    /// raw status.
    fn codex_error_summary(status: u16, body_text: &str) -> Option<String> {
        let parsed: Value = serde_json::from_str(body_text).ok()?;
        if let Some(detail) = parsed.get("detail").and_then(|v| v.as_str()) {
            return Some(format!("Codex request failed with {status}: {detail}"));
        }
        let err = parsed.get("error")?;
        let code = err
            .get("code")
            .and_then(|v| v.as_str())
            .or_else(|| err.get("type").and_then(|v| v.as_str()))
            .unwrap_or("");
        let code_matches = {
            let lc = code.to_ascii_lowercase();
            lc.contains("usage_limit_reached")
                || lc.contains("usage_not_included")
                || lc.contains("rate_limit_exceeded")
        };
        if !code_matches && status != 429 {
            // Prefer the raw `error.message` if the server gave us one —
            // useful for refusals, invalid-request errors, etc.
            let msg = err.get("message").and_then(|v| v.as_str())?;
            return Some(format!("Codex request failed with {status}: {msg}"));
        }

        let plan = err
            .get("plan_type")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|p| format!(" ({} plan)", p.to_ascii_lowercase()))
            .unwrap_or_default();
        let resets_at_secs = err.get("resets_at").and_then(|v| v.as_i64());
        let mins = resets_at_secs.and_then(|ts| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs() as i64;
            let delta_secs = ts - now;
            if delta_secs <= 0 {
                Some(0)
            } else {
                Some(((delta_secs + 30) / 60).max(0))
            }
        });
        let when = match mins {
            Some(m) => format!(" Try again in ~{m} min."),
            None => String::new(),
        };
        Some(format!(
            "You have hit your ChatGPT usage limit{plan}.{when}"
        ))
    }

    fn should_parse_stream(stream_requested: bool, content_type: Option<&str>) -> bool {
        stream_requested
            || content_type
                .map(|ct| ct.contains("text/event-stream"))
                .unwrap_or(false)
    }

    fn non_sse_body_read_error(
        status: u16,
        content_type: Option<&str>,
        err: LlmTransportError,
    ) -> LlmTransportError {
        let content_type_detail = content_type
            .map(|ct| format!(" ({ct})"))
            .unwrap_or_default();
        let code = err
            .code
            .clone()
            .unwrap_or_else(|| "body_read_failed".to_string());
        LlmTransportError::new(format!(
            "Codex returned HTTP {status} with non-SSE body{content_type_detail} but it could not be read: {}",
            err.message
        ))
        .retryable(err.retryable)
        .with_code(code)
    }

    fn projected_schema(
        canonical: &Value,
        overrides: &[SchemaProjectionOverride],
        profile: OpenAiSchemaProfile,
    ) -> Result<Value, LlmTransportError> {
        shared::projected_schema(PROVIDER, canonical, overrides, profile)
    }

    fn build_tools(req: &LlmRequest) -> Result<Vec<Value>, LlmTransportError> {
        shared::build_tools(PROVIDER, req)
    }

    fn codex_user_agent() -> String {
        format!(
            "{}/{} ({}; {}) lash",
            Self::CODEX_ORIGINATOR,
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    }

    /// Clamp an incoming reasoning effort string to a value the Codex
    /// Responses API will accept for the given model. Mirrors the
    /// `clampReasoningEffort` helper in pi-mono's
    /// `openai-codex-responses.ts` so that invalid combinations like
    /// `minimal` on gpt-5.4 or `xhigh` on gpt-5.1-codex-mini don't 4xx.
    fn clamp_reasoning_effort(model: &str, effort: &str) -> String {
        // Strip any provider prefix (e.g. "openai/gpt-5.4").
        let id = model_id(model);

        if (id.starts_with("gpt-5.2") || id.starts_with("gpt-5.3") || id.starts_with("gpt-5.4"))
            && effort == "minimal"
        {
            return "low".to_string();
        }
        if id == "gpt-5.1" && effort == "xhigh" {
            return "high".to_string();
        }
        if id == "gpt-5.1-codex-mini" {
            return if effort == "high" || effort == "xhigh" {
                "high".to_string()
            } else {
                "medium".to_string()
            };
        }
        effort.to_string()
    }

    fn build_request_body(
        &self,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<Value, LlmTransportError> {
        let tools = Self::build_tools(req)?;
        let (instructions, input) =
            shared::build_responses_input(req, shared::ResponsesInputOptions::CODEX);
        let reasoning_effort = req
            .model_variant
            .as_deref()
            .and_then(|variant| CodexModelPolicy.reasoning_effort(&req.model, variant));
        let policy = resolve_generation_policy(
            &req.generation,
            &self.options,
            DEFAULT_MAX_OUTPUT_TOKENS,
            reasoning_effort,
        );
        let mut body = json!({
            "model": req.model,
            "instructions": instructions,
            "input": input,
            "tools": tools,
            "parallel_tool_calls": !req.tools.is_empty(),
            "stream": stream,
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "text": {
                "verbosity": "medium",
            },
        });
        // `tool_choice` is only meaningful when the request advertises tools.
        // In RLM mode we intentionally send `tools: []` because tools are
        // documented in the prompt body and invoked via `lashlang`, not the
        // native tool-call envelope. Sending `tool_choice: "none"` on top of
        // an empty tool list adds a second "definitely don't call any
        // function" signal that gpt-5.x reasoning models take literally,
        // causing them to refuse to emit `call` expressions in lashlang.
        if !req.tools.is_empty() {
            body["tool_choice"] = json!(shared::tool_choice_value(&req.tool_choice));
        }
        if let Some(effort) = policy.thinking {
            let mut reasoning = json!({
                "effort": Self::clamp_reasoning_effort(&req.model, &effort),
            });
            if policy.expose_thinking {
                reasoning["summary"] = json!("auto");
            }
            body["reasoning"] = reasoning;
        }
        if policy.cache_retention != CacheRetention::None
            && let Some(session_id) = req.session_id.as_deref()
        {
            body["prompt_cache_key"] = json!(session_id);
        }
        if let Some(output_spec) = &req.output_spec {
            body["text"]["format"] = match output_spec {
                LlmOutputSpec::JsonObject => json!({ "type": "json_object" }),
                LlmOutputSpec::JsonSchema(schema) => {
                    let projected = Self::projected_schema(
                        &schema.schema,
                        &[],
                        OpenAiSchemaProfile::StructuredOutput,
                    )?;
                    json!({
                        "type": "json_schema",
                        "name": schema.name,
                        "schema": projected,
                        "strict": schema.strict,
                    })
                }
            };
        }
        Ok(body)
    }

    fn body_input(body: &Value) -> Vec<Value> {
        body.get("input")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    }

    fn body_fingerprint(body: &Value) -> String {
        let mut comparable = body.clone();
        if let Some(obj) = comparable.as_object_mut() {
            obj.remove("input");
            obj.remove("previous_response_id");
        }
        comparable.to_string()
    }

    fn cached_websocket_body(&mut self, req: &LlmRequest, full_body: &Value) -> (Value, bool) {
        let Some(session_id) = req.session_id.as_deref() else {
            return (full_body.clone(), false);
        };
        let Some(cached) = self.continuation_cache.by_session.get(session_id).cloned() else {
            return (full_body.clone(), false);
        };
        let current_fingerprint = Self::body_fingerprint(full_body);
        let current_input = Self::body_input(full_body);
        let compatible = cached.body_fingerprint == current_fingerprint
            && current_input.len() >= cached.input.len()
            && current_input
                .iter()
                .take(cached.input.len())
                .eq(cached.input.iter());
        if !compatible {
            self.continuation_cache.by_session.remove(session_id);
            return (full_body.clone(), false);
        }
        let mut body = full_body.clone();
        body["previous_response_id"] = json!(cached.previous_response_id);
        body["input"] = Value::Array(current_input[cached.input.len()..].to_vec());
        (body, true)
    }

    fn record_continuation(&mut self, req: &LlmRequest, full_body: &Value, final_response: &Value) {
        let Some(session_id) = req.session_id.as_deref() else {
            return;
        };
        let Some(response_id) = final_response.get("id").and_then(Value::as_str) else {
            self.continuation_cache.by_session.remove(session_id);
            return;
        };
        self.continuation_cache.by_session.insert(
            session_id.to_string(),
            CodexContinuation {
                previous_response_id: response_id.to_string(),
                input: Self::body_input(full_body),
                body_fingerprint: Self::body_fingerprint(full_body),
            },
        );
    }

    fn clear_continuation(&mut self, req: &LlmRequest) {
        if let Some(session_id) = req.session_id.as_deref() {
            self.continuation_cache.by_session.remove(session_id);
        }
    }

    async fn complete_websocket(
        &mut self,
        req: LlmRequest,
    ) -> Result<LlmResponse, CodexWebSocketAttemptError> {
        let stream_events = req.stream_events.clone();
        let provider_trace = req.provider_trace.clone();
        let full_body =
            self.build_request_body(&req, true)
                .map_err(|error| CodexWebSocketAttemptError {
                    error,
                    events_seen: false,
                })?;
        let use_cached = matches!(
            self.transport,
            CodexTransport::Auto | CodexTransport::WebsocketCached
        );
        let (body, cached) = if use_cached {
            self.cached_websocket_body(&req, &full_body)
        } else {
            (full_body.clone(), false)
        };
        let request_body =
            serde_json::to_string(&body).map_err(|error| CodexWebSocketAttemptError {
                error: LlmTransportError::new(format!(
                    "Failed to serialize Codex WebSocket body: {error}"
                )),
                events_seen: false,
            })?;

        let mut ws_request =
            Self::CODEX_RESPONSES_WS_URL
                .into_client_request()
                .map_err(|error| CodexWebSocketAttemptError {
                    error: LlmTransportError::new(format!(
                        "Failed to build Codex WebSocket request: {error}"
                    )),
                    events_seen: false,
                })?;
        let headers = ws_request.headers_mut();
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.access_token)).map_err(|error| {
                CodexWebSocketAttemptError {
                    error: LlmTransportError::new(format!(
                        "Invalid Codex WebSocket authorization header: {error}"
                    )),
                    events_seen: false,
                }
            })?,
        );
        headers.insert(
            "OpenAI-Beta",
            HeaderValue::from_static("responses=experimental"),
        );
        headers.insert(
            "originator",
            HeaderValue::from_static(Self::CODEX_ORIGINATOR),
        );
        headers.insert(
            "User-Agent",
            HeaderValue::from_str(&Self::codex_user_agent()).map_err(|error| {
                CodexWebSocketAttemptError {
                    error: LlmTransportError::new(format!(
                        "Invalid Codex WebSocket user-agent header: {error}"
                    )),
                    events_seen: false,
                }
            })?,
        );
        if let Some(session_id) = req.session_id.as_deref() {
            let value =
                HeaderValue::from_str(session_id).map_err(|error| CodexWebSocketAttemptError {
                    error: LlmTransportError::new(format!(
                        "Invalid Codex WebSocket session header: {error}"
                    )),
                    events_seen: false,
                })?;
            headers.insert("session_id", value.clone());
            headers.insert("x-client-request-id", value);
        }
        if let Some(account_id) = self.account_id.as_deref() {
            headers.insert(
                "ChatGPT-Account-ID",
                HeaderValue::from_str(account_id).map_err(|error| CodexWebSocketAttemptError {
                    error: LlmTransportError::new(format!(
                        "Invalid Codex WebSocket account header: {error}"
                    )),
                    events_seen: false,
                })?,
            );
        }

        let mut events_seen = false;
        let (mut websocket, _) =
            connect_async(ws_request)
                .await
                .map_err(|error| CodexWebSocketAttemptError {
                    error: LlmTransportError::new(format!(
                        "Codex WebSocket connect failed: {error}"
                    ))
                    .retryable(true)
                    .with_code("websocket_connect"),
                    events_seen,
                })?;
        websocket
            .send(WsMessage::Text(request_body.clone().into()))
            .await
            .map_err(|error| CodexWebSocketAttemptError {
                error: LlmTransportError::new(format!("Codex WebSocket send failed: {error}"))
                    .retryable(true)
                    .with_code("websocket_send"),
                events_seen,
            })?;

        let mut state = shared::ResponsesStreamState::default();
        let expose_thinking = self.options.expose_thinking;
        while let Some(message) = websocket.next().await {
            let message = message.map_err(|error| CodexWebSocketAttemptError {
                error: LlmTransportError::new(format!("Codex WebSocket receive failed: {error}"))
                    .retryable(true)
                    .with_code("websocket_receive"),
                events_seen,
            })?;
            let raw = match message {
                WsMessage::Text(text) => text.to_string(),
                WsMessage::Binary(bytes) => String::from_utf8(bytes.to_vec()).map_err(|error| {
                    CodexWebSocketAttemptError {
                        error: LlmTransportError::new(format!(
                            "Codex WebSocket binary frame was not UTF-8: {error}"
                        ))
                        .with_code("websocket_protocol"),
                        events_seen,
                    }
                })?,
                WsMessage::Close(_) => break,
                WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Frame(_) => continue,
            };
            emit_provider_trace(provider_trace.as_ref(), "codex", &raw);
            events_seen = true;
            let prev_usage = state.usage.clone();
            let mut emitted_parts = Vec::new();
            if Self::looks_like_sse_payload(&raw) {
                shared::parse_sse_payload(PROVIDER, &raw, &mut state)
                    .map_err(|error| CodexWebSocketAttemptError { error, events_seen })?;
            } else {
                shared::process_sse_event(PROVIDER, &raw, &mut state, Some(&mut emitted_parts))
                    .map_err(|error| CodexWebSocketAttemptError { error, events_seen })?;
            }
            emit_stream_progress(
                stream_events.as_ref(),
                state.take_text_deltas(),
                &state.usage,
                &prev_usage,
            );
            if let Some(tx) = &stream_events {
                for piece in state.take_reasoning_deltas() {
                    if expose_thinking {
                        tx.send(LlmStreamEvent::ReasoningDelta(piece));
                    }
                }
                for part in emitted_parts {
                    if matches!(part, lash_core::llm::types::LlmOutputPart::Reasoning { .. })
                        && !expose_thinking
                    {
                        continue;
                    }
                    tx.send(LlmStreamEvent::Part(part));
                }
            } else {
                state.take_reasoning_deltas();
            }
            if state
                .final_response
                .as_ref()
                .and_then(|response| response.get("status").and_then(Value::as_str))
                .is_some_and(|status| matches!(status, "completed" | "failed" | "incomplete"))
            {
                break;
            }
        }

        if state.final_response.is_none() && state.parts.is_empty() {
            return Err(CodexWebSocketAttemptError {
                error: LlmTransportError::new("Codex WebSocket ended without response events")
                    .retryable(true)
                    .with_code("empty_websocket_stream"),
                events_seen,
            });
        }

        if let Some(final_response) = state.final_response.clone() {
            self.record_continuation(&req, &full_body, &final_response);
        } else {
            self.clear_continuation(&req);
        }
        let mut response = shared::response_from_stream_state(
            state,
            Some(request_body),
            format!(
                "WS {}{}",
                Self::CODEX_RESPONSES_WS_URL,
                if cached { " (cached)" } else { "" }
            ),
        );
        response.http_summary = Some(format!(
            "WS {}{}",
            Self::CODEX_RESPONSES_WS_URL,
            if cached { " (cached)" } else { "" }
        ));
        Ok(response)
    }

    fn looks_like_sse_payload(payload: &str) -> bool {
        let trimmed = payload.trim_start();
        trimmed.starts_with("event:")
            || trimmed.starts_with("data:")
            || payload.contains("\nevent:")
            || payload.contains("\ndata:")
    }

    #[cfg(test)]
    fn process_sse_event(
        raw: &str,
        state: &mut shared::ResponsesStreamState,
        emitted_parts: Option<&mut Vec<lash_core::llm::types::LlmOutputPart>>,
    ) -> Result<(), LlmTransportError> {
        shared::process_sse_event(PROVIDER, raw, state, emitted_parts)
    }
}

impl CodexProvider {
    pub fn into_components(self) -> ProviderComponents {
        ProviderComponents::new(Box::new(self), std::sync::Arc::new(CodexModelPolicy))
            .with_failure_classifier(std::sync::Arc::new(CodexFailureClassifier))
    }
}

#[derive(Debug)]
struct CodexFailureClassifier;

impl ProviderFailureClassifier for CodexFailureClassifier {
    fn classify(&self, failure: ProviderFailure) -> ProviderFailure {
        // The default classifier already covers everything Codex needs from a
        // status/text standpoint: HTTP-status → kind/retryability, the
        // usage-limit/quota and content-filter text markers, and context
        // overflow. Codex's only genuine delta is rewriting the user-facing
        // message into a friendly "you hit your ChatGPT usage limit" form.
        let status = failure
            .status
            .or_else(|| failure.code.as_deref().and_then(|code| code.parse().ok()));
        let summary = status.and_then(|status| {
            CodexProvider::codex_error_summary(status, failure.raw.as_deref().unwrap_or_default())
        });
        let mut failure = DefaultProviderFailureClassifier.classify(failure);
        if let Some(summary) = summary {
            failure.message = summary;
        }
        failure
    }
}

impl ProviderModelPolicy for CodexModelPolicy {
    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        let lower = model.to_ascii_lowercase();
        if !lower.contains("gpt-5") {
            return &[];
        }
        if model_id(&lower) == "gpt-5.5" {
            return OPENAI_GPT55_VARIANTS;
        }
        if lower.contains("codex") {
            if has_xhigh_suffix(&lower) {
                CODEX_XHIGH_VARIANTS
            } else {
                CODEX_VARIANTS
            }
        } else if has_xhigh_suffix(&lower) {
            OPENAI_GPT5_XHIGH_VARIANTS
        } else {
            OPENAI_GPT5_VARIANTS
        }
    }
}

impl CodexModelPolicy {
    fn reasoning_effort(&self, model: &str, variant: &str) -> Option<String> {
        self.supported_variants(model)
            .contains(&variant)
            .then(|| variant.to_string())
    }
}

#[async_trait]
impl Provider for CodexProvider {
    fn kind(&self) -> &'static str {
        "codex"
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
            "access_token".to_string(),
            serde_json::Value::String(self.access_token.clone()),
        );
        map.insert(
            "refresh_token".to_string(),
            serde_json::Value::String(self.refresh_token.clone()),
        );
        map.insert(
            "expires_at".to_string(),
            serde_json::Value::Number(self.expires_at.into()),
        );
        if let Some(account_id) = &self.account_id {
            map.insert(
                "account_id".to_string(),
                serde_json::Value::String(account_id.clone()),
            );
        } else {
            map.insert("account_id".to_string(), serde_json::Value::Null);
        }
        if !self.options.is_default() {
            map.insert(
                "options".to_string(),
                serde_json::to_value(&self.options).unwrap_or(serde_json::Value::Null),
            );
        }
        if self.transport != CodexTransport::Auto {
            map.insert(
                "transport".to_string(),
                serde_json::to_value(self.transport).unwrap_or(serde_json::Value::Null),
            );
        }
        serde_json::Value::Object(map)
    }

    fn requires_streaming(&self) -> bool {
        true
    }

    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        if !matches!(self.transport, CodexTransport::Sse) {
            match self.complete_websocket(req.clone()).await {
                Ok(response) => return Ok(response),
                Err(err) if matches!(self.transport, CodexTransport::Auto) && !err.events_seen => {
                    self.clear_continuation(&req);
                    tracing::debug!(
                        target: "lash_core::llm::codex_oauth",
                        error = %err.error.message,
                        "Codex WebSocket failed before stream start; falling back to SSE"
                    );
                }
                Err(err) => {
                    self.clear_continuation(&req);
                    return Err(err.error);
                }
            }
        }
        let stream_events = req.stream_events.clone();
        let provider_trace = req.provider_trace.clone();
        let access_token = self.access_token.clone();
        let account_id = self.account_id.clone();
        let timeouts = self.options.llm_timeouts();

        let body = self.build_request_body(&req, stream_events.is_some())?;

        let request_body = serde_json::to_string(&body).ok();

        let mut http = self
            .client
            .post(Self::CODEX_RESPONSES_URL)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", Self::CODEX_ORIGINATOR)
            .header("User-Agent", Self::codex_user_agent())
            .json(&body);
        if let Some(session_id) = req.session_id.as_deref() {
            http = http
                .header("session_id", session_id)
                .header("x-client-request-id", session_id);
        }
        if let Some(id) = account_id.as_deref() {
            http = http.header("ChatGPT-Account-ID", id);
        }
        let resp = send_request(
            http,
            request_body.clone().map(request_body_snapshot),
            response_start_timeout(
                timeouts.request_timeout,
                timeouts.chunk_timeout,
                stream_events.is_some(),
            ),
            "Codex response start timed out",
        )
        .await?;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let headers = resp.headers().clone();
        if !status.is_success() {
            let text = read_response_text(
                resp,
                timeouts.request_timeout,
                "Codex response body timed out",
            )
            .await
            .unwrap_or_default();
            let message = Self::codex_error_summary(status.as_u16(), &text).unwrap_or_else(|| {
                format!(
                    "Codex request failed with {}{}",
                    status.as_u16(),
                    content_type
                        .as_deref()
                        .map(|ct| format!(" ({ct})"))
                        .unwrap_or_default()
                )
            });
            // Retryability is decided centrally by `CodexFailureClassifier`
            // from the attached HTTP status; no inline override here.
            let mut err = LlmTransportError::new(message)
                .with_kind(ProviderFailureKind::Http)
                .with_status(status.as_u16())
                .with_headers(header_pairs(&headers))
                .with_raw(text);
            if let Some(request_body) = request_body.clone() {
                err = err.with_request_body(request_body);
            }
            return Err(err);
        }

        let is_sse = content_type
            .as_deref()
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);
        let parse_stream =
            Self::should_parse_stream(stream_events.is_some(), content_type.as_deref());

        if !parse_stream {
            let text = read_response_text(
                resp,
                timeouts.request_timeout,
                "Codex response body timed out",
            )
            .await
            .map_err(|err| {
                Self::non_sse_body_read_error(status.as_u16(), content_type.as_deref(), err)
            })?;
            emit_provider_trace(provider_trace.as_ref(), "codex", &text);
            if Self::looks_like_sse_payload(&text) {
                let mut state = shared::ResponsesStreamState::default();
                shared::parse_sse_payload(PROVIDER, &text, &mut state)?;
                if let Some(final_response) = state.final_response.clone() {
                    self.record_continuation(&req, &body, &final_response);
                }
                let response = shared::response_from_stream_state(
                    state,
                    request_body,
                    format!("HTTP POST {} (stream/fallback)", Self::CODEX_RESPONSES_URL),
                );
                if let Some(tx) = &stream_events {
                    if response.usage != LlmUsage::default() {
                        tx.send(LlmStreamEvent::Usage(response.usage.clone()));
                    }
                    for part in &response.parts {
                        if let lash_core::llm::types::LlmOutputPart::Text { text, .. } = part
                            && !text.is_empty()
                        {
                            tx.send(LlmStreamEvent::Delta(text.clone()));
                        }
                    }
                    for part in &response.parts {
                        match part {
                            lash_core::llm::types::LlmOutputPart::ToolCall { .. } => {
                                tx.send(LlmStreamEvent::Part(part.clone()));
                            }
                            lash_core::llm::types::LlmOutputPart::Reasoning { text, .. }
                                if !text.is_empty() && self.options.expose_thinking =>
                            {
                                tx.send(LlmStreamEvent::ReasoningDelta(text.clone()));
                            }
                            _ => {}
                        }
                    }
                }
                return Ok(response);
            }
            let value: Value = serde_json::from_str(&text).map_err(|e| {
                LlmTransportError::new(format!("Invalid Codex response JSON: {e}"))
                    .with_raw(text.clone())
            })?;
            self.record_continuation(&req, &body, &value);
            let content = shared::extract_text(&value);
            let usage = openai_usage_from_response_value(&value);
            let mut parts = shared::response_parts_from_value(&value);
            if parts.is_empty() && !content.is_empty() {
                parts.push(lash_core::llm::types::LlmOutputPart::Text {
                    text: content.clone(),
                    response_meta: None,
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
            let terminal_reason = openai_terminal_reason_from_response_value(&value, &parts);
            return Ok(LlmResponse {
                full_text: content,
                parts,
                usage,
                terminal_reason,
                terminal_diagnostic: None,
                provider_usage: None,
                request_body,
                http_summary: Some(format!("HTTP POST {}", Self::CODEX_RESPONSES_URL)),
            });
        }

        if stream_events.is_some() && !is_sse {
            tracing::debug!(
                target: "lash_core::llm::codex_oauth",
                status = status.as_u16(),
                content_type = content_type.as_deref().unwrap_or("<missing>"),
                "Codex streaming response did not advertise SSE; parsing as stream because stream=true was requested"
            );
        }

        let mut state = shared::ResponsesStreamState::default();
        let expose_thinking = self.options.expose_thinking;
        drive_sse_response(
            LlmHttpBody::from_reqwest_response(resp),
            timeouts.chunk_timeout,
            "Codex stream chunk timed out",
            |raw| {
                emit_provider_trace(provider_trace.as_ref(), "codex", raw);
                let prev_usage = state.usage.clone();
                let mut emitted_parts = Vec::new();
                shared::process_sse_event(PROVIDER, raw, &mut state, Some(&mut emitted_parts))?;
                emit_stream_progress(
                    stream_events.as_ref(),
                    state.take_text_deltas(),
                    &state.usage,
                    &prev_usage,
                );
                if let Some(tx) = &stream_events {
                    for piece in state.take_reasoning_deltas() {
                        if expose_thinking {
                            tx.send(LlmStreamEvent::ReasoningDelta(piece));
                        }
                    }
                    for part in emitted_parts {
                        if matches!(part, lash_core::llm::types::LlmOutputPart::Reasoning { .. })
                            && !expose_thinking
                        {
                            continue;
                        }
                        tx.send(LlmStreamEvent::Part(part));
                    }
                }
                Ok(())
            },
        )
        .await?;

        if state.final_response.is_none()
            && state.parts.is_empty()
            && state.pending_text_deltas.is_empty()
        {
            return Err(LlmTransportError::new(format!(
                "Codex stream ended without SSE events (HTTP {}{})",
                status.as_u16(),
                content_type
                    .as_deref()
                    .map(|ct| format!(", content-type {ct}"))
                    .unwrap_or_else(|| ", missing content-type".to_string())
            ))
            .retryable(true)
            .with_code("empty_stream"));
        }

        if let Some(final_response) = state.final_response.clone() {
            self.record_continuation(&req, &body, &final_response);
        }

        Ok(shared::response_from_stream_state(
            state,
            request_body,
            format!("HTTP POST {} (stream)", Self::CODEX_RESPONSES_URL),
        ))
    }

    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}

#[derive(Deserialize)]
struct CodexProviderConfig {
    access_token: String,
    refresh_token: String,
    expires_at: u64,
    #[serde(default)]
    account_id: Option<String>,
    #[serde(default = "default_codex_options")]
    options: ProviderOptions,
    #[serde(default)]
    transport: CodexTransport,
}

fn default_codex_options() -> ProviderOptions {
    ProviderOptions {
        reliability: ProviderReliability::codex(),
        ..ProviderOptions::default()
    }
}

/// Factory that materializes [`CodexProvider`] from a host-owned
/// [`ProviderSpec`](lash_core::ProviderSpec).
pub struct CodexProviderFactory;

impl ProviderFactory for CodexProviderFactory {
    fn kind(&self) -> &'static str {
        "codex"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String> {
        let cfg: CodexProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(CodexProvider {
            access_token: cfg.access_token,
            refresh_token: cfg.refresh_token,
            expires_at: cfg.expires_at,
            account_id: cfg.account_id,
            options: cfg.options,
            transport: cfg.transport,
            continuation_cache: CodexContinuationCache::default(),
            client: build_http_client(),
        }
        .into_components())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::llm::types::{
        LlmJsonSchema, LlmMessage, LlmOutputPart, LlmRole, LlmTerminalReason, LlmToolChoice,
        LlmToolSpec, ResponseTextMeta,
    };
    use lash_core::provider::ProviderModelPolicy;
    use shared::ResponsesStreamState as CodexStreamState;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    fn process_event(state: &mut CodexStreamState, event: Value) {
        CodexProvider::process_sse_event(&event.to_string(), state, None).unwrap();
    }

    fn process_event_with_parts(
        state: &mut CodexStreamState,
        event: Value,
        emitted_parts: &mut Vec<LlmOutputPart>,
    ) {
        CodexProvider::process_sse_event(&event.to_string(), state, Some(emitted_parts)).unwrap();
    }

    fn response_from_state(state: CodexStreamState) -> LlmResponse {
        shared::response_from_stream_state(state, None, "test".to_string())
    }

    fn request(messages: Vec<LlmMessage>) -> LlmRequest {
        LlmRequest {
            model: "gpt-5.4".to_string(),
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
    fn gpt_55_variants_match_codex_catalog() {
        let provider = CodexModelPolicy;

        assert_eq!(
            provider.supported_variants("gpt-5.5"),
            ["low", "medium", "high", "xhigh"]
        );
        assert_eq!(
            provider.reasoning_effort("gpt-5.5", "xhigh").as_deref(),
            Some("xhigh")
        );
        assert_eq!(provider.reasoning_effort("gpt-5.5", "minimal"), None);
    }

    #[test]
    fn codex_null_incomplete_details_does_not_map_to_output_limit() {
        let terminal_reason = openai_terminal_reason_from_response_value(
            &json!({"status":"completed","incomplete_details":null}),
            &[LlmOutputPart::Text {
                text: "Hi".to_string(),
                response_meta: None,
            }],
        );

        assert_eq!(terminal_reason, LlmTerminalReason::Stop);
    }

    #[test]
    fn codex_content_filter_incomplete_maps_to_content_filter() {
        let terminal_reason = openai_terminal_reason_from_response_value(
            &json!({"status":"incomplete","incomplete_details":{"reason":"content_filter"}}),
            &[],
        );

        assert_eq!(terminal_reason, LlmTerminalReason::ContentFilter);
    }

    #[test]
    fn codex_request_body_exposes_reasoning_summary_only_when_configured() {
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        req.model_variant = Some("medium".to_string());

        let hidden = CodexProvider::new("access", "refresh", 0)
            .build_request_body(&req, true)
            .unwrap();
        assert_eq!(hidden["reasoning"], json!({ "effort": "medium" }));

        let exposed = CodexProvider::new("access", "refresh", 0)
            .with_options(ProviderOptions {
                expose_thinking: true,
                ..ProviderOptions::default()
            })
            .build_request_body(&req, true)
            .unwrap();
        assert_eq!(exposed["reasoning"]["summary"], "auto");
    }

    #[test]
    fn codex_request_omits_output_token_cap() {
        let provider = CodexProvider::new("access", "refresh", 0).with_options(ProviderOptions {
            max_output_tokens: Some(9_999),
            ..ProviderOptions::default()
        });
        let provider_limited = provider
            .build_request_body(
                &request(vec![LlmMessage::text(LlmRole::User, "hello")]),
                false,
            )
            .unwrap();
        assert!(provider_limited.get("max_output_tokens").is_none());

        let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        req.generation.output_token_cap = NonZeroUsize::new(2_048);
        let request_limited = provider.build_request_body(&req, false).unwrap();
        assert!(request_limited.get("max_output_tokens").is_none());
    }

    #[test]
    fn codex_error_summary_uses_top_level_detail() {
        let summary =
            CodexProvider::codex_error_summary(400, r#"{"detail":"Unsupported parameter: foo"}"#);
        assert_eq!(
            summary.as_deref(),
            Some("Codex request failed with 400: Unsupported parameter: foo")
        );
    }

    #[test]
    fn response_failed_server_error_is_retryable() {
        let mut state = CodexStreamState::default();
        let err = CodexProvider::process_sse_event(
            r#"{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"internal stream ended unexpectedly"}}}"#,
            &mut state,
            None,
        )
        .unwrap_err();

        assert!(err.retryable);
        assert_eq!(err.message, "internal stream ended unexpectedly");
    }

    #[test]
    fn gpt_55_variant_match_ignores_provider_prefix() {
        let provider = CodexModelPolicy;

        assert_eq!(
            provider.supported_variants("openai/gpt-5.5"),
            ["low", "medium", "high", "xhigh"]
        );
    }

    #[test]
    fn codex_request_uses_openai_schema_projection() {
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        req.tools = Arc::new(vec![LlmToolSpec {
            name: "empty".to_string(),
            description: "Empty".to_string(),
            input_schema: json!({"type": "object"}),
            output_schema: json!({}),
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
        }]);
        req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
            name: "result".to_string(),
            schema: json!({
                "type": "object",
                "properties": { "summary": { "type": "string" } }
            }),
            strict: true,
        }));

        let body = CodexProvider::new("access", "refresh", 0)
            .build_request_body(&req, false)
            .unwrap();
        assert_eq!(body["tools"][0]["parameters"]["properties"], json!({}));
        assert_eq!(
            body["text"]["format"]["schema"]["required"],
            json!(["summary"])
        );
        assert_eq!(
            body["text"]["format"]["schema"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn codex_request_history_preserves_assistant_message_metadata() {
        let req = request(vec![LlmMessage::new(
            LlmRole::Assistant,
            vec![lash_core::llm::types::LlmContentBlock::Text {
                text: "final".into(),
                response_meta: Some(ResponseTextMeta {
                    id: Some("msg_1".to_string()),
                    status: Some("completed".to_string()),
                    phase: Some("final_answer".to_string()),
                    ..ResponseTextMeta::default()
                }),
                cache_breakpoint: false,
            }],
        )]);

        let body = CodexProvider::new("access", "refresh", 0)
            .build_request_body(&req, false)
            .unwrap();

        assert_eq!(body["input"][0]["type"], "message");
        assert_eq!(body["input"][0]["id"], "msg_1");
        assert_eq!(body["input"][0]["status"], "completed");
        assert_eq!(body["input"][0]["phase"], "final_answer");
        assert_eq!(body["input"][0]["content"][0]["type"], "output_text");
        assert!(body["input"][0]["content"][0]["annotations"].is_array());
    }

    #[test]
    fn codex_cached_continuation_sends_delta_after_full_input() {
        let mut provider = CodexProvider::new("access", "refresh", 0)
            .with_transport(CodexTransport::WebsocketCached);
        let first = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        let first_body = provider.build_request_body(&first, true).unwrap();
        provider.record_continuation(&first, &first_body, &json!({"id":"resp_1"}));

        let second = request(vec![
            LlmMessage::text(LlmRole::User, "hello"),
            LlmMessage::new(
                LlmRole::Assistant,
                vec![lash_core::llm::types::LlmContentBlock::Text {
                    text: "answer".into(),
                    response_meta: Some(ResponseTextMeta {
                        id: Some("msg_1".to_string()),
                        status: Some("completed".to_string()),
                        phase: Some("final_answer".to_string()),
                        ..ResponseTextMeta::default()
                    }),
                    cache_breakpoint: false,
                }],
            ),
            LlmMessage::text(LlmRole::User, "next"),
        ]);
        let second_body = provider.build_request_body(&second, true).unwrap();
        let (cached_body, cached) = provider.cached_websocket_body(&second, &second_body);

        assert!(cached);
        assert_eq!(cached_body["previous_response_id"], "resp_1");
        assert_eq!(
            cached_body["input"].as_array().expect("delta input").len(),
            second_body["input"].as_array().unwrap().len()
                - first_body["input"].as_array().unwrap().len()
        );
        assert_eq!(cached_body["input"][0]["type"], "message");
    }

    #[test]
    fn codex_schema_projection_failure_is_local_validation_error() {
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
            name: "bad".to_string(),
            schema: json!({"type": "object", "allOf": []}),
            strict: true,
        }));

        let err = CodexProvider::new("access", "refresh", 0)
            .build_request_body(&req, false)
            .unwrap_err();
        assert_eq!(err.kind, ProviderFailureKind::Validation);
        assert!(err.message.contains("allOf"));
    }

    #[test]
    fn codex_stream_assembles_single_message_item_once() {
        let mut state = CodexStreamState::default();

        process_event(
            &mut state,
            json!({"type":"response.output_item.added","item":{"type":"message","id":"msg_1","status":"in_progress","phase":"commentary"}}),
        );
        process_event(
            &mut state,
            json!({"type":"response.output_text.delta","item_id":"msg_1","delta":"Hel"}),
        );
        process_event(
            &mut state,
            json!({"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"Hello"}]}}),
        );

        let response = response_from_state(state);
        assert_eq!(response.full_text, "Hello");
        assert_eq!(response.parts.len(), 1);
        assert_eq!(
            response.parts[0],
            LlmOutputPart::Text {
                text: "Hello".to_string(),
                response_meta: Some(ResponseTextMeta {
                    id: Some("msg_1".to_string()),
                    status: Some("completed".to_string()),
                    phase: Some("commentary".to_string()),
                    ..ResponseTextMeta::default()
                }),
            }
        );
    }

    #[test]
    fn codex_stream_replayed_message_item_does_not_duplicate_text() {
        let mut state = CodexStreamState::default();

        for event in [
            json!({"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}),
            json!({"type":"response.output_text.delta","item_id":"msg_1","delta":"The sentence."}),
            json!({"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"The sentence."}]}}),
            json!({"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}),
            json!({"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"The sentence."}]}}),
        ] {
            process_event(&mut state, event);
        }

        let response = response_from_state(state);
        assert_eq!(response.full_text, "The sentence.");
        assert_eq!(
            response
                .parts
                .iter()
                .filter(|part| matches!(part, LlmOutputPart::Text { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn codex_stream_completed_response_merges_existing_message_by_id() {
        let mut state = CodexStreamState::default();

        for event in [
            json!({"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}),
            json!({"type":"response.output_text.delta","item_id":"msg_1","delta":"Final answer."}),
            json!({"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"Final answer."}]}}),
            json!({"type":"response.completed","response":{"id":"resp_1","output_text":"Final answer.","output":[{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"Final answer."}]}]}}),
        ] {
            process_event(&mut state, event);
        }

        let response = response_from_state(state);
        assert_eq!(response.full_text, "Final answer.");
        assert_eq!(response.parts.len(), 1);
    }

    #[test]
    fn codex_stream_distinct_message_ids_stay_separate_without_inserted_separator() {
        let mut state = CodexStreamState::default();

        for event in [
            json!({"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}),
            json!({"type":"response.output_text.delta","item_id":"msg_1","delta":"One."}),
            json!({"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"One."}]}}),
            json!({"type":"response.output_item.added","item":{"type":"message","id":"msg_2"}}),
            json!({"type":"response.output_text.delta","item_id":"msg_2","delta":"Two."}),
            json!({"type":"response.output_item.done","item":{"type":"message","id":"msg_2","status":"completed","content":[{"type":"output_text","text":"Two."}]}}),
        ] {
            process_event(&mut state, event);
        }

        let response = response_from_state(state);
        assert_eq!(response.full_text, "One.Two.");
        assert_eq!(response.parts.len(), 2);
        assert_eq!(
            response.parts,
            vec![
                LlmOutputPart::Text {
                    text: "One.".to_string(),
                    response_meta: Some(ResponseTextMeta {
                        id: Some("msg_1".to_string()),
                        status: Some("completed".to_string()),
                        phase: None,
                        ..ResponseTextMeta::default()
                    }),
                },
                LlmOutputPart::Text {
                    text: "Two.".to_string(),
                    response_meta: Some(ResponseTextMeta {
                        id: Some("msg_2".to_string()),
                        status: Some("completed".to_string()),
                        phase: None,
                        ..ResponseTextMeta::default()
                    }),
                },
            ]
        );
    }

    #[test]
    fn codex_stream_preserves_reasoning_message_and_tool_call_once() {
        let mut state = CodexStreamState::default();
        let mut emitted_parts = Vec::new();

        for event in [
            json!({"type":"response.reasoning_summary_part.added"}),
            json!({"type":"response.reasoning_summary_text.delta","delta":"Think"}),
            json!({"type":"response.reasoning_summary_part.done"}),
            json!({"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think"}],"encrypted_content":"enc"}}),
            json!({"type":"response.output_item.added","item":{"type":"message","id":"msg_1","phase":"final_answer"}}),
            json!({"type":"response.output_text.delta","item_id":"msg_1","delta":"Hi"}),
            json!({"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"Hi"}]}}),
            json!({"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":""}}),
            json!({"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"x\""}),
            json!({"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"x\":1}"}),
            json!({"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":"{\"x\":1}","status":"completed"}}),
        ] {
            process_event_with_parts(&mut state, event, &mut emitted_parts);
        }
        process_event_with_parts(
            &mut state,
            json!({"type":"response.completed","response":{"id":"resp_1","output":[
                {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think"}],"encrypted_content":"enc"},
                {"type":"message","id":"msg_1","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"Hi"}]},
                {"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":"{\"x\":1}","status":"completed"}
            ],"output_text":"Hi"}}),
            &mut emitted_parts,
        );

        let response = response_from_state(state);
        assert_eq!(response.full_text, "Hi");
        assert_eq!(emitted_parts.len(), 1);
        assert_eq!(
            response
                .parts
                .iter()
                .filter(|part| matches!(part, LlmOutputPart::Reasoning { .. }))
                .count(),
            1
        );
        assert_eq!(
            response
                .parts
                .iter()
                .filter(|part| matches!(part, LlmOutputPart::Text { .. }))
                .count(),
            1
        );
        assert_eq!(
            response
                .parts
                .iter()
                .filter(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
                .count(),
            1
        );
    }

    /// Cross-provider response-normalization conformance. Codex shares OpenAI's
    /// Responses-API normalizers (`shared::*`), so this wires those into the
    /// shared suite with Responses-API wire fixtures.
    #[cfg(feature = "testing")]
    mod conformance {
        use super::super::{PROVIDER, shared};
        use lash_core::llm::types::{LlmOutputPart, LlmTerminalReason, LlmUsage};
        use lash_llm_transport::conformance::{
            CanonicalUsage as U, ProviderNormalizer, ProviderWire, Scenario, StreamAssembly,
            provider_conformance,
        };
        use lash_llm_transport::{
            openai_terminal_reason_from_response_value, openai_usage_from_response_value,
        };
        use serde_json::{Value, json};

        struct CodexNormalizer;

        impl ProviderNormalizer for CodexNormalizer {
            fn name(&self) -> &str {
                "codex-responses"
            }

            fn wire_for(&self, scenario: Scenario) -> Option<ProviderWire> {
                let wire = match scenario {
                    Scenario::PlainTextStop => ProviderWire::body(json!({
                        "status": "completed",
                        "output": [{
                            "type": "message", "id": "msg_1", "status": "completed",
                            "content": [{ "type": "output_text", "text": "hello" }]
                        }],
                        "usage": { "input_tokens": U::BASE_INPUT, "output_tokens": U::BASE_OUTPUT }
                    })),
                    Scenario::OutputCapped => ProviderWire::body(json!({
                        "status": "incomplete",
                        "incomplete_details": { "reason": "max_output_tokens" },
                        "output": [{
                            "type": "message", "id": "msg_1", "status": "incomplete",
                            "content": [{ "type": "output_text", "text": "trunc" }]
                        }]
                    })),
                    Scenario::ContentFilter => ProviderWire::body(json!({
                        "status": "incomplete",
                        "incomplete_details": { "reason": "content_filter" },
                        "output": []
                    })),
                    Scenario::NonStreamingToolUse => ProviderWire::body(json!({
                        "status": "completed",
                        "output": [{
                            "type": "function_call", "id": "fc_1", "call_id": "call_1",
                            "name": "lookup", "arguments": "{\"q\":\"x\"}", "status": "completed"
                        }]
                    })),
                    Scenario::StreamingTextAssembly => {
                        ProviderWire::body(json!({})).with_text_stream(
                            vec![
                                r#"{"type":"response.output_text.delta","item_id":"msg_1","delta":"hello "}"#.to_string(),
                                r#"{"type":"response.output_text.delta","item_id":"msg_1","delta":"world"}"#.to_string(),
                            ],
                            "hello world",
                        )
                    }
                    Scenario::StreamingToolArgumentMerge => {
                        ProviderWire::body(json!({})).with_tool_call_stream(
                            vec![
                                r#"{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#.to_string(),
                                // arguments deliberately split across two delta events
                                r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"q\":"}"#.to_string(),
                                r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"\"x\"}"}"#.to_string(),
                                r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}","status":"completed"}}"#.to_string(),
                            ],
                            "lookup",
                            json!({ "q": "x" }),
                        )
                    }
                    Scenario::UsageCacheHit => ProviderWire::body(json!({
                        "status": "completed",
                        "output": [{
                            "type": "message", "id": "msg_1", "status": "completed",
                            "content": [{ "type": "output_text", "text": "ok" }]
                        }],
                        "usage": {
                            "input_tokens": U::BASE_INPUT,
                            "output_tokens": U::BASE_OUTPUT,
                            "input_tokens_details": { "cached_tokens": U::CACHED_INPUT }
                        }
                    })),
                    Scenario::UsageReasoning => ProviderWire::body(json!({
                        "status": "completed",
                        "output": [{
                            "type": "message", "id": "msg_1", "status": "completed",
                            "content": [{ "type": "output_text", "text": "ok" }]
                        }],
                        "usage": {
                            "input_tokens": U::BASE_INPUT,
                            "output_tokens": U::BASE_OUTPUT,
                            "output_tokens_details": { "reasoning_tokens": U::REASONING }
                        }
                    })),
                    Scenario::ReasoningExtraction => ProviderWire::body(json!({
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning", "id": "rs_1",
                                "summary": [{ "type": "summary_text", "text": "thinking about it" }]
                            },
                            {
                                "type": "message", "id": "msg_1", "status": "completed",
                                "content": [{ "type": "output_text", "text": "answer" }]
                            }
                        ]
                    }))
                    .with_reasoning_text("thinking about it"),
                    Scenario::StreamingUsageMerge => {
                        ProviderWire::body(json!({})).with_usage_merge_stream(vec![
                            // input arrives on an early event
                            format!(
                                r#"{{"type":"response.output_text.delta","delta":"hi","usage":{{"input_tokens":{}}}}}"#,
                                U::BASE_INPUT
                            ),
                            // output arrives on a later event; merge must keep input
                            format!(
                                r#"{{"type":"response.output_text.delta","delta":"!","usage":{{"output_tokens":{}}}}}"#,
                                U::BASE_OUTPUT
                            ),
                        ])
                    }
                };
                Some(wire)
            }

            fn parts_from_wire(&self, body: &Value) -> Vec<LlmOutputPart> {
                shared::response_parts_from_value(body)
            }

            fn usage_from_wire(&self, body: &Value) -> LlmUsage {
                openai_usage_from_response_value(body)
            }

            fn terminal_from_wire(
                &self,
                body: &Value,
                parts: &[LlmOutputPart],
            ) -> LlmTerminalReason {
                openai_terminal_reason_from_response_value(body, parts)
            }

            fn assemble_stream(&self, sse_events: &[String]) -> StreamAssembly {
                let mut state = shared::ResponsesStreamState::default();
                for raw in sse_events {
                    shared::process_sse_event(PROVIDER, raw, &mut state, None)
                        .expect("responses sse event parses");
                }
                StreamAssembly {
                    parts: state.response_parts(),
                    usage: state.usage.clone(),
                }
            }
        }

        #[test]
        fn codex_satisfies_provider_conformance() {
            provider_conformance(&CodexNormalizer);
        }
    }
}
