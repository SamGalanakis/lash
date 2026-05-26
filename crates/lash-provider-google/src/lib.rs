#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::OnceLock;

use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use lash_core::llm::transport::{LlmTransportError, validate_image_attachments};
use lash_core::llm::types::{
    LlmAttachment, LlmContentBlock, LlmOutputPart, LlmOutputSpec, LlmProviderTraceEvent,
    LlmRequest, LlmResponse, LlmRole, LlmTerminalReason, LlmToolChoice, LlmUsage,
    ProviderReasoningReplay, ProviderReplayMeta,
};
use lash_core::provider::{
    ProviderComponents, ProviderFactory, ProviderModelPolicy, ProviderOptions, ProviderState,
    ProviderTransport, resolve_generation_policy,
};
use lash_llm_transport::streaming::{drive_sse_response, emit_stream_progress};
use lash_llm_transport::timeouts::{
    build_http_client, header_pairs, read_response_text, request_body_snapshot,
    response_start_timeout, send_request,
};

#[derive(Clone, Debug, PartialEq, Eq)]
enum GoogleThinkingConfig {
    Level { level: String },
    Budget { budget_tokens: i32 },
}

pub mod oauth;

fn emit_provider_trace(
    tx: Option<&lash_core::llm::types::LlmProviderTraceSender>,
    provider: &'static str,
    raw: &str,
) {
    let Some(tx) = tx else {
        return;
    };
    let event_name = serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .or_else(|| value.get("event"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "provider_event".to_string());
    tx.send(LlmProviderTraceEvent {
        provider,
        event_name,
        raw: raw.to_string(),
    });
}

const CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const CODE_ASSIST_API_VERSION: &str = "v1internal";
const GEMINI_FILES_UPLOAD_URL: &str =
    "https://generativelanguage.googleapis.com/upload/v1beta/files";

const GEMINI_31_VARIANTS: &[&str] = &["low", "medium", "high"];
const GEMINI_3_VARIANTS: &[&str] = &["low", "high"];
const GEMINI_25_VARIANTS: &[&str] = &["high", "max"];

/// Pi-mono sentinel: Gemini 3 refuses to run when a function_call is
/// replayed without a thoughtSignature. The server recognises this magic
/// string and skips signature validation for the item, so lash can round-
/// trip tool calls captured from non-Gemini models without crashing the
/// turn. Matches `google-shared.ts:51`.
const SKIP_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct UploadedAttachmentCacheKey {
    project_id: String,
    mime: String,
    hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct UploadedAttachmentRef {
    uri: String,
}

/// Google OAuth (Gemini via Code Assist) provider.
#[derive(Clone, Debug)]
pub struct GoogleOAuthProvider {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
    pub project_id: Option<String>,
    pub options: ProviderOptions,
    client: reqwest::Client,
}

#[derive(Clone, Debug)]
struct GoogleModelPolicy;

impl GoogleOAuthProvider {
    fn uploaded_attachment_cache()
    -> &'static tokio::sync::Mutex<HashMap<UploadedAttachmentCacheKey, UploadedAttachmentRef>> {
        static CACHE: OnceLock<
            tokio::sync::Mutex<HashMap<UploadedAttachmentCacheKey, UploadedAttachmentRef>>,
        > = OnceLock::new();
        CACHE.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()))
    }

    pub fn new(
        access_token: impl Into<String>,
        refresh_token: impl Into<String>,
        expires_at: u64,
    ) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: refresh_token.into(),
            expires_at,
            project_id: None,
            options: ProviderOptions::default(),
            client: build_http_client(),
        }
    }

    pub fn with_project_id(mut self, project_id: Option<String>) -> Self {
        self.project_id = project_id;
        self
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.client = (*client).clone();
        self
    }

    fn endpoint_base_url() -> String {
        let endpoint = std::env::var("CODE_ASSIST_ENDPOINT")
            .unwrap_or_else(|_| CODE_ASSIST_ENDPOINT.to_string());
        let version = std::env::var("CODE_ASSIST_API_VERSION")
            .unwrap_or_else(|_| CODE_ASSIST_API_VERSION.to_string());
        format!("{endpoint}/{version}")
    }

    fn method_url(method: &str) -> String {
        format!("{}:{}", Self::endpoint_base_url(), method)
    }

    fn inline_attachment_part(att: &LlmAttachment) -> Value {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
        json!({
            "inlineData": {
                "mimeType": att.mime,
                "data": b64,
            }
        })
    }

    fn attachment_part_for_index(attachment_parts: &[Value], idx: usize) -> Value {
        attachment_parts
            .get(idx)
            .cloned()
            .unwrap_or_else(|| json!({ "text": "[Image attached]" }))
    }

    fn build_contents_with_attachment_parts(
        req: &LlmRequest,
        attachment_parts: &[Value],
    ) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        let is_gemini_3 = req.model.to_ascii_lowercase().contains("gemini-3");

        for msg in &req.messages {
            if matches!(msg.role, LlmRole::System) {
                // System content is hoisted into `systemInstruction` on the
                // Gemini request, not the `contents` list.
                continue;
            }
            let role = match msg.role {
                LlmRole::Assistant => "model",
                LlmRole::User | LlmRole::System => "user",
            };

            let mut parts: Vec<Value> = Vec::new();
            for block in msg.blocks.iter() {
                match block {
                    LlmContentBlock::Text { text, .. } => {
                        if text.is_empty() {
                            continue;
                        }
                        parts.push(json!({ "text": text }));
                    }
                    LlmContentBlock::Image { attachment_idx } => {
                        if matches!(msg.role, LlmRole::User) {
                            parts.push(Self::attachment_part_for_index(
                                attachment_parts,
                                *attachment_idx,
                            ));
                        }
                    }
                    LlmContentBlock::ToolCall {
                        call_id,
                        tool_name,
                        input_json,
                        replay,
                        ..
                    } => {
                        let mut part = json!({
                            "functionCall": {
                                "id": call_id,
                                "name": tool_name,
                                "args": serde_json::from_str::<Value>(input_json)
                                    .unwrap_or_else(|_| json!({"_raw": input_json})),
                            }
                        });
                        // Gemini 3 rejects turns where a function_call
                        // from a thinking-enabled run is replayed without
                        // its original thoughtSignature. When we don't
                        // have the real signature (cross-model hop, older
                        // session), drop in the pi sentinel.
                        let effective = replay
                            .as_ref()
                            .and_then(|meta| meta.opaque.clone())
                            .or_else(|| is_gemini_3.then(|| SKIP_THOUGHT_SIGNATURE.to_string()));
                        if let Some(sig) = effective {
                            part["thoughtSignature"] = Value::String(sig);
                        }
                        parts.push(part);
                    }
                    LlmContentBlock::ToolResult {
                        call_id,
                        content,
                        tool_name,
                    } => {
                        parts.push(json!({
                            "functionResponse": {
                                "id": call_id,
                                "name": tool_name.clone().unwrap_or_else(|| "tool".to_string()),
                                "response": { "output": content },
                            }
                        }));
                    }
                    LlmContentBlock::Reasoning { text, replay, .. } => {
                        // Gemini replays reasoning as a `thought:true`
                        // text part carrying the thoughtSignature.
                        let sig = replay.as_ref().and_then(|meta| meta.signature.clone());
                        if sig.is_none() && text.trim().is_empty() {
                            continue;
                        }
                        let mut part = json!({
                            "text": if text.is_empty() { String::from(" ") } else { text.clone() },
                            "thought": true,
                        });
                        if let Some(s) = sig {
                            part["thoughtSignature"] = Value::String(s);
                        }
                        parts.push(part);
                    }
                }
            }

            if parts.is_empty() {
                continue;
            }

            // Merge with previous same-role turn so text + images + tool
            // calls land as a single `contents` entry (matches the old
            // behavior expected by Gemini clients).
            if let Some(prev) = out.last_mut()
                && prev.get("role").and_then(|r| r.as_str()) == Some(role)
                && prev.get("parts").is_some_and(|p| p.is_array())
            {
                prev["parts"].as_array_mut().unwrap().extend(parts);
            } else {
                out.push(json!({
                    "role": role,
                    "parts": parts,
                }));
            }
        }
        out
    }

    fn uses_legacy_tool_parameters(model: &str) -> bool {
        model.starts_with("claude-")
    }

    fn google_tool_choice(choice: &LlmToolChoice) -> &'static str {
        match choice {
            LlmToolChoice::Auto => "AUTO",
            LlmToolChoice::None => "NONE",
            LlmToolChoice::Required => "ANY",
        }
    }

    fn system_instruction(req: &LlmRequest) -> Option<Value> {
        let mut parts: Vec<String> = Vec::new();
        for msg in &req.messages {
            if !matches!(msg.role, LlmRole::System) {
                continue;
            }
            for block in msg.blocks.iter() {
                if let LlmContentBlock::Text { text, .. } = block
                    && !text.is_empty()
                {
                    parts.push(text.to_string());
                }
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(json!({
                "parts": [{ "text": parts.join("\n\n") }],
            }))
        }
    }

    fn parse_i64(v: Option<&Value>) -> i64 {
        match v {
            Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
            Some(Value::String(s)) => s.parse::<i64>().unwrap_or(0),
            _ => 0,
        }
    }

    fn usage_from_event(event: &Value) -> LlmUsage {
        let meta = event
            .get("response")
            .and_then(|r| r.get("usageMetadata"))
            .unwrap_or(&Value::Null);
        LlmUsage {
            input_tokens: Self::parse_i64(
                meta.get("promptTokenCount")
                    .or_else(|| meta.get("inputTokenCount"))
                    .or_else(|| meta.get("inputTokens")),
            ),
            output_tokens: Self::parse_i64(
                meta.get("candidatesTokenCount")
                    .or_else(|| meta.get("outputTokenCount"))
                    .or_else(|| meta.get("outputTokens")),
            ),
            cached_input_tokens: Self::parse_i64(
                meta.get("cachedContentTokenCount")
                    .or_else(|| meta.get("cachedPromptTokenCount"))
                    .or_else(|| meta.get("cachedInputTokenCount")),
            ),
            reasoning_tokens: Self::parse_i64(
                meta.get("thoughtsTokenCount")
                    .or_else(|| meta.get("reasoningTokenCount"))
                    .or_else(|| meta.get("reasoningTokens")),
            ),
        }
    }

    fn text_parts_from_event(event: &Value) -> Vec<String> {
        let mut out = Vec::new();
        let Some(candidates) = event
            .get("response")
            .and_then(|r| r.get("candidates"))
            .and_then(|c| c.as_array())
        else {
            return out;
        };

        for candidate in candidates {
            let Some(parts) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            else {
                continue;
            };
            for part in parts {
                // Skip reasoning text (Gemini marks it with `thought:true`);
                // it isn't assistant-visible output and shouldn't accumulate
                // into `full`. The signature ride-along is captured when
                // we finalize the response.
                if part.get("thought").and_then(|v| v.as_bool()) == Some(true) {
                    continue;
                }
                if let Some(text) = part.get("text").and_then(|t| t.as_str())
                    && !text.is_empty()
                {
                    out.push(text.to_string());
                }
            }
        }

        out
    }

    fn tool_call_parts_from_event(event: &Value) -> Vec<LlmOutputPart> {
        let mut out = Vec::new();
        let Some(candidates) = event
            .get("response")
            .and_then(|r| r.get("candidates"))
            .and_then(|c| c.as_array())
        else {
            return out;
        };
        for candidate in candidates {
            let Some(parts) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            else {
                continue;
            };
            for part in parts {
                if let Some(function_call) = part.get("functionCall") {
                    let Some(name) = function_call.get("name").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let input_json = function_call
                        .get("args")
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string());
                    // Capture `thoughtSignature` (if present) alongside
                    // the functionCall. Gemini 3 will reject the next
                    // turn if we don't echo it back.
                    let signature = part
                        .get("thoughtSignature")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(str::to_string);
                    out.push(LlmOutputPart::ToolCall {
                        call_id: function_call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        tool_name: name.to_string(),
                        input_json,
                        replay: signature.map(|opaque| ProviderReplayMeta {
                            item_id: None,
                            opaque: Some(opaque),
                        }),
                    });
                }
            }
        }
        out
    }

    fn apply_stream_piece(full: &mut String, text_deltas: &mut Vec<String>, piece: &str) {
        if piece.is_empty() {
            return;
        }
        if piece.starts_with(full.as_str()) {
            let delta = &piece[full.len()..];
            if !delta.is_empty() {
                full.push_str(delta);
                text_deltas.push(delta.to_string());
            }
            return;
        }
        full.push_str(piece);
        text_deltas.push(piece.to_string());
    }

    fn process_sse_event(
        raw: &str,
        full: &mut String,
        text_deltas: &mut Vec<String>,
        usage: &mut LlmUsage,
        tool_call_parts: Option<&mut Vec<LlmOutputPart>>,
    ) -> Result<(), LlmTransportError> {
        if raw.trim().is_empty() || raw.trim() == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(raw)
            .map_err(|e| LlmTransportError::new(format!("Invalid Cloud Code SSE payload: {e}")))?;
        let new_usage = Self::usage_from_event(&event);
        if new_usage.input_tokens > 0
            || new_usage.output_tokens > 0
            || new_usage.cached_input_tokens > 0
            || new_usage.reasoning_tokens > 0
        {
            *usage = new_usage;
        }
        for piece in Self::text_parts_from_event(&event) {
            Self::apply_stream_piece(full, text_deltas, &piece);
        }
        if let Some(parts) = tool_call_parts {
            parts.extend(Self::tool_call_parts_from_event(&event));
        }
        Ok(())
    }

    fn response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
        let mut parts = Vec::new();
        let Some(candidates) = value.get("candidates").and_then(|c| c.as_array()) else {
            return parts;
        };
        for candidate in candidates {
            let Some(items) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            else {
                continue;
            };
            for item in items {
                let signature = item
                    .get("thoughtSignature")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                let is_thought = item
                    .get("thought")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                if let Some(text) = item.get("text").and_then(|t| t.as_str())
                    && !text.is_empty()
                {
                    if is_thought {
                        // Gemini flags reasoning text with `thought: true`.
                        // Route those into Reasoning so downstream code
                        // doesn't show them as assistant prose. Signature
                        // lives on the same part.
                        parts.push(LlmOutputPart::Reasoning {
                            text: text.to_string(),
                            replay: signature.clone().map(|signature| ProviderReasoningReplay {
                                item_id: None,
                                encrypted_content: None,
                                signature: Some(signature),
                                redacted: false,
                                summary: Vec::new(),
                            }),
                        });
                    } else {
                        parts.push(LlmOutputPart::Text {
                            text: text.to_string(),
                            response_meta: None,
                        });
                    }
                }
                if let Some(function_call) = item.get("functionCall") {
                    let Some(name) = function_call.get("name").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let input_json = function_call
                        .get("args")
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string());
                    parts.push(LlmOutputPart::ToolCall {
                        call_id: function_call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        tool_name: name.to_string(),
                        input_json,
                        replay: signature.clone().map(|opaque| ProviderReplayMeta {
                            item_id: None,
                            opaque: Some(opaque),
                        }),
                    });
                }
            }
        }
        parts
    }

    fn terminal_reason_from_value(value: &Value, parts: &[LlmOutputPart]) -> LlmTerminalReason {
        let finish = value
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(|candidate| candidate.get("finishReason"))
            .or_else(|| {
                value
                    .get("response")
                    .and_then(|response| response.get("candidates"))
                    .and_then(Value::as_array)
                    .and_then(|candidates| candidates.first())
                    .and_then(|candidate| candidate.get("finishReason"))
            })
            .and_then(Value::as_str)
            .unwrap_or("");
        match finish {
            "STOP" => LlmTerminalReason::Stop,
            "MAX_TOKENS" => LlmTerminalReason::OutputLimit,
            "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "IMAGE_RECITATION"
            | "IMAGE_OTHER"
            | "LANGUAGE" => LlmTerminalReason::ContentFilter,
            "MALFORMED_FUNCTION_CALL"
            | "UNEXPECTED_TOOL_CALL"
            | "FINISH_REASON_UNSPECIFIED"
            | "OTHER"
            | "NO_IMAGE" => LlmTerminalReason::ProviderError,
            "" => {
                if parts
                    .iter()
                    .any(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
                {
                    LlmTerminalReason::ToolUse
                } else {
                    LlmTerminalReason::Stop
                }
            }
            _ => LlmTerminalReason::ProviderError,
        }
    }

    fn upload_cache_key(
        project_id: Option<&str>,
        att: &LlmAttachment,
    ) -> UploadedAttachmentCacheKey {
        UploadedAttachmentCacheKey {
            project_id: project_id.unwrap_or_default().to_string(),
            mime: att.mime.clone(),
            hash: format!("{:x}", Sha256::digest(&att.data)),
        }
    }

    fn uploaded_attachment_filename(key: &UploadedAttachmentCacheKey) -> String {
        let ext = match key.mime.as_str() {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/jpg" => "jpg",
            "image/webp" => "webp",
            "image/gif" => "gif",
            "image/heic" => "heic",
            "image/heif" => "heif",
            "image/bmp" => "bmp",
            "image/tiff" => "tiff",
            _ => "bin",
        };
        format!("lash-{}.{}", &key.hash[..12], ext)
    }

    async fn upload_attachment_cached(
        &self,
        access_token: &str,
        project_id: Option<&str>,
        att: &LlmAttachment,
    ) -> Result<UploadedAttachmentRef, LlmTransportError> {
        let key = Self::upload_cache_key(project_id, att);
        if let Some(existing) = Self::uploaded_attachment_cache()
            .lock()
            .await
            .get(&key)
            .cloned()
        {
            return Ok(existing);
        }

        let uploaded = self
            .upload_attachment(
                access_token,
                project_id,
                att,
                &Self::uploaded_attachment_filename(&key),
            )
            .await?;
        Self::uploaded_attachment_cache()
            .lock()
            .await
            .insert(key, uploaded.clone());
        Ok(uploaded)
    }

    async fn upload_attachment(
        &self,
        access_token: &str,
        project_id: Option<&str>,
        att: &LlmAttachment,
        filename: &str,
    ) -> Result<UploadedAttachmentRef, LlmTransportError> {
        let mut start = self
            .client
            .post(GEMINI_FILES_UPLOAD_URL)
            .bearer_auth(access_token)
            .header("Content-Type", "application/json")
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header(
                "X-Goog-Upload-Header-Content-Length",
                att.data.len().to_string(),
            )
            .header("X-Goog-Upload-Header-Content-Type", att.mime.as_str())
            .header("X-Goog-Upload-File-Name", filename)
            .json(&json!({
                "file": {
                    "displayName": filename,
                    "mimeType": att.mime,
                    "sizeBytes": att.data.len().to_string(),
                }
            }));
        if let Some(project_id) = project_id.filter(|project_id| !project_id.trim().is_empty()) {
            start = start.header("x-goog-user-project", project_id);
        }

        let start_resp = send_request(
            start,
            None,
            self.options.llm_timeouts().request_timeout,
            "Gemini Files upload start timed out",
        )
        .await?;
        if !start_resp.status().is_success() {
            let status = start_resp.status().as_u16();
            let headers = start_resp.headers().clone();
            let body = read_response_text(
                start_resp,
                self.options.llm_timeouts().request_timeout,
                "Gemini Files upload start body timed out",
            )
            .await
            .unwrap_or_default();
            return Err(LlmTransportError::new(format!(
                "Gemini Files upload start failed with {}",
                status
            ))
            .with_status(status)
            .with_headers(header_pairs(&headers))
            .with_raw(body)
            .retryable(status == 429 || status >= 500));
        }

        let upload_url = start_resp
            .headers()
            .get("x-goog-upload-url")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                LlmTransportError::new(
                    "Gemini Files upload start response missing x-goog-upload-url header",
                )
            })?
            .to_string();

        let mut finalize = self
            .client
            .post(upload_url)
            .bearer_auth(access_token)
            .header("X-Goog-Upload-Command", "upload, finalize")
            .header("X-Goog-Upload-Offset", "0")
            .header("Content-Length", att.data.len().to_string())
            .body(att.data.clone());
        if let Some(project_id) = project_id.filter(|project_id| !project_id.trim().is_empty()) {
            finalize = finalize.header("x-goog-user-project", project_id);
        }

        let finalize_resp = send_request(
            finalize,
            None,
            self.options.llm_timeouts().request_timeout,
            "Gemini Files upload finalize timed out",
        )
        .await?;
        if !finalize_resp.status().is_success() {
            let status = finalize_resp.status().as_u16();
            let headers = finalize_resp.headers().clone();
            let body = read_response_text(
                finalize_resp,
                self.options.llm_timeouts().request_timeout,
                "Gemini Files upload finalize body timed out",
            )
            .await
            .unwrap_or_default();
            return Err(LlmTransportError::new(format!(
                "Gemini Files upload finalize failed with {}",
                status
            ))
            .with_status(status)
            .with_headers(header_pairs(&headers))
            .with_raw(body)
            .retryable(status == 429 || status >= 500));
        }

        let upload_status = finalize_resp
            .headers()
            .get("x-goog-upload-status")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = read_response_text(
            finalize_resp,
            self.options.llm_timeouts().request_timeout,
            "Gemini Files upload finalize body timed out",
        )
        .await?;
        if upload_status
            .as_deref()
            .is_some_and(|status| status != "final")
        {
            return Err(LlmTransportError::new(format!(
                "Gemini Files upload finalize returned unexpected status `{}`",
                upload_status.unwrap_or_default()
            ))
            .with_raw(body));
        }

        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LlmTransportError::new(format!("Invalid Gemini Files upload JSON: {err}"))
                .with_raw(body.clone())
        })?;
        let file = value.get("file").unwrap_or(&value);
        let uri = if let Some(uri) = file.get("uri").and_then(|value| value.as_str()) {
            uri.to_string()
        } else if let Some(name) = file.get("name").and_then(|value| value.as_str()) {
            format!("https://generativelanguage.googleapis.com/v1beta/{name}")
        } else {
            return Err(
                LlmTransportError::new("Gemini Files upload response missing file uri")
                    .with_raw(body.clone()),
            );
        };

        Ok(UploadedAttachmentRef { uri })
    }

    async fn prepare_attachment_parts(
        &self,
        access_token: &str,
        project_id: Option<&str>,
        attachments: &[LlmAttachment],
    ) -> (Vec<Value>, bool) {
        let mut parts = Vec::with_capacity(attachments.len());
        let mut used_uploaded_files = false;

        for att in attachments {
            if !att.mime.starts_with("image/") {
                parts.push(Self::inline_attachment_part(att));
                continue;
            }

            match self
                .upload_attachment_cached(access_token, project_id, att)
                .await
            {
                Ok(uploaded) => {
                    used_uploaded_files = true;
                    parts.push(json!({
                        "fileData": {
                            "mimeType": att.mime,
                            "fileUri": uploaded.uri,
                        }
                    }));
                }
                Err(_) => parts.push(Self::inline_attachment_part(att)),
            }
        }

        (parts, used_uploaded_files)
    }

    fn build_request(
        provider: &GoogleOAuthProvider,
        req: &LlmRequest,
        contents: Vec<Value>,
        project_id: Option<&str>,
    ) -> Value {
        let thinking_config = req
            .model_variant
            .as_deref()
            .and_then(|variant| GoogleModelPolicy.thinking_config(&req.model, variant));
        let policy =
            resolve_generation_policy(&req.generation, &provider.options, 32_768, thinking_config);
        let mut request = json!({
            "model": req.model,
            "user_prompt_id": uuid::Uuid::new_v4().to_string(),
            "request": {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": policy.max_output_tokens,
                }
            }
        });
        if let Some(system_instruction) = Self::system_instruction(req) {
            request["request"]["systemInstruction"] = system_instruction;
        }
        if let Some(session_id) = req.session_id.as_deref() {
            request["request"]["sessionId"] = json!(session_id);
        }
        if let Some(config) = policy.thinking {
            match config {
                GoogleThinkingConfig::Level { level } => {
                    let mut thinking_config = json!({
                        "thinkingLevel": level,
                    });
                    if policy.expose_thinking {
                        thinking_config["includeThoughts"] = json!(true);
                    }
                    request["request"]["generationConfig"]["thinkingConfig"] = thinking_config;
                }
                GoogleThinkingConfig::Budget { budget_tokens } => {
                    let mut thinking_config = json!({
                        "thinkingBudget": budget_tokens,
                    });
                    if policy.expose_thinking {
                        thinking_config["includeThoughts"] = json!(true);
                    }
                    request["request"]["generationConfig"]["thinkingConfig"] = thinking_config;
                }
            }
        }
        if !req.tools.is_empty() {
            let use_legacy_parameters = Self::uses_legacy_tool_parameters(&req.model);
            request["request"]["tools"] = json!([{
                "functionDeclarations": req
                    .tools
                    .iter()
                    .map(|tool| {
                        let mut declaration = json!({
                            "name": tool.name.clone(),
                            "description": tool.description.clone(),
                        });
                        if use_legacy_parameters {
                            declaration["parameters"] = tool.input_schema.clone();
                        } else {
                            declaration["parametersJsonSchema"] = tool.input_schema.clone();
                        }
                        declaration
                    })
                    .collect::<Vec<_>>()
            }]);
            request["request"]["toolConfig"] = json!({
                "functionCallingConfig": {
                    "mode": Self::google_tool_choice(&req.tool_choice),
                }
            });
        }
        if let Some(output_spec) = &req.output_spec {
            request["request"]["generationConfig"]["responseMimeType"] = json!("application/json");
            if let LlmOutputSpec::JsonSchema(schema) = output_spec {
                request["request"]["generationConfig"]["responseSchema"] = schema.schema.clone();
            }
        }
        if let Some(project) = project_id.filter(|p| !p.trim().is_empty()) {
            request["project"] = json!(project);
        }
        request
    }

    fn should_retry_inline(err: &LlmTransportError) -> bool {
        matches!(err.code.as_deref(), Some("400" | "404"))
            || err.raw.as_deref().is_some_and(|raw| {
                raw.contains("fileData") || raw.contains("fileUri") || raw.contains("file_uri")
            })
    }

    async fn execute_request(
        &self,
        access_token: &str,
        request: Value,
        stream_events: Option<lash_core::llm::types::LlmEventSender>,
        provider_trace: Option<lash_core::llm::types::LlmProviderTraceSender>,
    ) -> Result<LlmResponse, LlmTransportError> {
        let request_body = serde_json::to_string(&request).ok();
        let method = if stream_events.is_some() {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let url = Self::method_url(method);
        let mut http = self
            .client
            .post(&url)
            .bearer_auth(access_token)
            .json(&request);
        if stream_events.is_some() {
            http = http.query(&[("alt", "sse")]);
        }
        let resp = send_request(
            http,
            request_body.clone().map(request_body_snapshot),
            response_start_timeout(
                self.options.llm_timeouts().request_timeout,
                self.options.llm_timeouts().chunk_timeout,
                stream_events.is_some(),
            ),
            "Cloud Code response start timed out",
        )
        .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let body = read_response_text(
                resp,
                self.options.llm_timeouts().request_timeout,
                "Cloud Code response body timed out",
            )
            .await
            .unwrap_or_default();
            let mut err =
                LlmTransportError::new(format!("Cloud Code request failed with {}", status))
                    .with_status(status)
                    .with_headers(header_pairs(&headers))
                    .with_raw(body)
                    .retryable(status == 429 || status >= 500);
            if let Some(request_body) = request_body {
                err = err.with_request_body(request_body);
            }
            return Err(err);
        }

        if stream_events.is_none() {
            let text = read_response_text(
                resp,
                self.options.llm_timeouts().request_timeout,
                "Cloud Code response body timed out",
            )
            .await?;
            emit_provider_trace(provider_trace.as_ref(), "google", &text);
            let value: Value = serde_json::from_str(&text).map_err(|e| {
                LlmTransportError::new(format!("Invalid Cloud Code response JSON: {e}"))
                    .with_raw(text.clone())
            })?;
            let parts = Self::response_parts_from_value(&value);
            let full_text = parts
                .iter()
                .filter_map(|part| match part {
                    LlmOutputPart::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            let usage = value
                .get("usageMetadata")
                .map(|meta| {
                    Self::usage_from_event(&json!({
                        "response": {
                            "usageMetadata": meta
                        }
                    }))
                })
                .unwrap_or_default();
            let terminal_reason = Self::terminal_reason_from_value(&value, &parts);
            return Ok(LlmResponse {
                full_text,
                parts,
                usage,
                terminal_reason,
                terminal_diagnostic: None,
                provider_usage: None,
                request_body,
                http_summary: Some(format!("HTTP POST {}", url)),
            });
        }

        let mut full = String::new();
        let mut usage = LlmUsage::default();
        let mut tool_call_parts: Vec<LlmOutputPart> = Vec::new();
        drive_sse_response(
            resp,
            self.options.llm_timeouts().chunk_timeout,
            "Cloud Code stream chunk timed out",
            |raw| {
                emit_provider_trace(provider_trace.as_ref(), "google", raw);
                let mut text_deltas = Vec::new();
                let prev_usage = usage.clone();
                Self::process_sse_event(
                    raw,
                    &mut full,
                    &mut text_deltas,
                    &mut usage,
                    Some(&mut tool_call_parts),
                )?;
                emit_stream_progress(stream_events.as_ref(), text_deltas, &usage, &prev_usage);
                Ok(())
            },
        )
        .await?;

        let mut parts = Vec::new();
        if !full.is_empty() {
            parts.push(LlmOutputPart::Text {
                text: full.clone(),
                response_meta: None,
            });
        }
        parts.extend(tool_call_parts);

        Ok(LlmResponse {
            full_text: full,
            parts,
            usage,
            terminal_reason: LlmTerminalReason::Stop,
            terminal_diagnostic: None,
            provider_usage: None,
            request_body,
            http_summary: Some(format!("HTTP POST {}?alt=sse", url)),
        })
    }

    async fn resolve_project_id(
        &self,
        access_token: &str,
        project_hint: Option<&str>,
    ) -> Result<Option<String>, LlmTransportError> {
        let mut metadata = json!({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        });
        if let Some(project) = project_hint.filter(|p| !p.trim().is_empty()) {
            metadata["duetProject"] = json!(project);
        }

        let req = json!({
            "cloudaicompanionProject": project_hint,
            "metadata": metadata,
        });
        let request_body = serde_json::to_string(&req).ok();

        let resp = self
            .client
            .post(Self::method_url("loadCodeAssist"))
            .bearer_auth(access_token)
            .json(&req)
            .send()
            .await
            .map_err(|e| {
                let error = LlmTransportError::new(format!("HTTP request failed: {e}"));
                if let Some(request_body) = request_body.clone() {
                    error.with_request_body(request_body)
                } else {
                    error
                }
            })?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let body = resp.text().await.unwrap_or_default();
            let mut err =
                LlmTransportError::new(format!("Cloud Code loadCodeAssist failed with {}", status))
                    .with_status(status)
                    .with_headers(header_pairs(&headers))
                    .with_raw(body)
                    .retryable(status == 429 || status >= 500);
            if let Some(request_body) = request_body {
                err = err.with_request_body(request_body);
            }
            return Err(err);
        }
        let body: Value = resp.json().await.map_err(|e| {
            LlmTransportError::new(format!("Invalid Cloud Code loadCodeAssist JSON: {e}"))
        })?;
        Ok(body
            .get("cloudaicompanionProject")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| project_hint.map(|s| s.to_string())))
    }
}

impl GoogleOAuthProvider {
    pub fn into_components(self) -> ProviderComponents {
        ProviderComponents::shared(self, std::sync::Arc::new(GoogleModelPolicy))
    }
}

impl ProviderState for GoogleOAuthProvider {
    fn kind(&self) -> &'static str {
        "google_oauth"
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
        if let Some(project_id) = &self.project_id {
            map.insert(
                "project_id".to_string(),
                serde_json::Value::String(project_id.clone()),
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

    fn clone_boxed(&self) -> Box<dyn ProviderState> {
        Box::new(self.clone())
    }
}

impl ProviderModelPolicy for GoogleModelPolicy {
    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        let lower = model.to_ascii_lowercase();
        if lower.contains("gemini-2.5") {
            GEMINI_25_VARIANTS
        } else if lower.contains("gemini-3.1") {
            GEMINI_31_VARIANTS
        } else if lower.contains("gemini-3") {
            GEMINI_3_VARIANTS
        } else {
            &[]
        }
    }
}

impl GoogleModelPolicy {
    fn thinking_config(&self, model: &str, variant: &str) -> Option<GoogleThinkingConfig> {
        if !self.supported_variants(model).contains(&variant) {
            return None;
        }
        let lower = model.to_ascii_lowercase();
        if lower.contains("gemini-2.5") {
            let budget_tokens = match variant {
                "high" => 16_000,
                "max" => 24_576,
                _ => return None,
            };
            Some(GoogleThinkingConfig::Budget { budget_tokens })
        } else {
            Some(GoogleThinkingConfig::Level {
                level: variant.to_string(),
            })
        }
    }
}

#[async_trait]
impl ProviderTransport for GoogleOAuthProvider {
    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        validate_image_attachments(
            &req,
            &[
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/heic",
                "image/heif",
            ],
            "Google Gemini",
        )?;
        let stream_events = req.stream_events.clone();
        let provider_trace = req.provider_trace.clone();
        let access_token = self.access_token.clone();
        if self.project_id.is_none() {
            let hint = std::env::var("GOOGLE_CLOUD_PROJECT")
                .ok()
                .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT_ID").ok());
            self.project_id = self
                .resolve_project_id(&access_token, hint.as_deref())
                .await?;
        }
        let project_id = self.project_id.clone();

        let inline_attachment_parts = req
            .attachments
            .iter()
            .map(Self::inline_attachment_part)
            .collect::<Vec<_>>();
        let inline_contents =
            Self::build_contents_with_attachment_parts(&req, &inline_attachment_parts);

        let (attachment_parts, used_uploaded_files) = self
            .prepare_attachment_parts(&access_token, project_id.as_deref(), &req.attachments)
            .await;
        let contents = if used_uploaded_files {
            Self::build_contents_with_attachment_parts(&req, &attachment_parts)
        } else {
            inline_contents.clone()
        };

        let request = Self::build_request(self, &req, contents, project_id.as_deref());

        match self
            .execute_request(
                &access_token,
                request,
                stream_events.clone(),
                provider_trace.clone(),
            )
            .await
        {
            Ok(response) => Ok(response),
            Err(err) if used_uploaded_files && Self::should_retry_inline(&err) => {
                let inline_request =
                    Self::build_request(self, &req, inline_contents, project_id.as_deref());
                self.execute_request(&access_token, inline_request, stream_events, provider_trace)
                    .await
            }
            Err(err) => Err(err),
        }
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
        Box::new(self.clone())
    }
}

#[derive(Deserialize)]
struct GoogleProviderConfig {
    access_token: String,
    refresh_token: String,
    expires_at: u64,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    options: ProviderOptions,
}

pub struct GoogleOAuthProviderFactory;

impl GoogleOAuthProviderFactory {
    pub fn register() {
        lash_core::register_provider_factory(std::sync::Arc::new(Self));
    }
}

impl ProviderFactory for GoogleOAuthProviderFactory {
    fn kind(&self) -> &'static str {
        "google_oauth"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String> {
        let cfg: GoogleProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(GoogleOAuthProvider {
            access_token: cfg.access_token,
            refresh_token: cfg.refresh_token,
            expires_at: cfg.expires_at,
            project_id: cfg.project_id,
            options: cfg.options,
            client: build_http_client(),
        }
        .into_components())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use super::*;
    use lash_core::llm::types::{LlmEventSender, LlmMessage, LlmToolSpec};

    fn request(model_variant: Option<&str>) -> LlmRequest {
        LlmRequest {
            model: "gemini-3.1-pro-preview".to_string(),
            messages: vec![LlmMessage::text(LlmRole::User, "hello")],
            attachments: Vec::new(),
            tools: Arc::new(Vec::<LlmToolSpec>::new()),
            tool_choice: LlmToolChoice::Auto,
            model_variant: model_variant.map(str::to_string),
            session_id: None,
            output_spec: None,
            stream_events: None::<LlmEventSender>,
            generation: lash_core::GenerationOptions::default(),
            provider_trace: None,
        }
    }

    #[test]
    fn google_image_attachment_serializes_as_inline_data_part() {
        let png_bytes = vec![0x89, 0x50, 0x4E, 0x47];
        let attachment =
            lash_core::llm::types::LlmAttachment::bytes("image/png", png_bytes.clone());

        let part = GoogleOAuthProvider::inline_attachment_part(&attachment);

        let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        assert_eq!(part["inlineData"]["mimeType"], "image/png");
        assert_eq!(part["inlineData"]["data"], expected_b64);
    }

    #[test]
    fn google_rejects_gif_attachment_at_request_boundary() {
        let mut req = request(None);
        req.attachments = vec![lash_core::llm::types::LlmAttachment::bytes(
            "image/gif",
            vec![0x47, 0x49, 0x46],
        )];

        let err = validate_image_attachments(
            &req,
            &[
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/heic",
                "image/heif",
            ],
            "Google Gemini",
        )
        .expect_err("gif should be rejected for Gemini");

        assert_eq!(err.code.as_deref(), Some("unsupported_image_format"));
        assert!(err.message.contains("Google Gemini"));
        assert!(err.message.contains("image/gif"));
    }

    #[test]
    fn google_accepts_webp_attachment_through_validation() {
        let mut req = request(None);
        req.attachments = vec![lash_core::llm::types::LlmAttachment::bytes(
            "image/webp",
            vec![0],
        )];

        validate_image_attachments(
            &req,
            &[
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/heic",
                "image/heif",
            ],
            "Google Gemini",
        )
        .expect("webp is supported");
    }

    #[test]
    fn google_unknown_finish_reason_maps_to_provider_error() {
        let terminal_reason = GoogleOAuthProvider::terminal_reason_from_value(
            &json!({"candidates":[{"finishReason":"NEW_REASON"}]}),
            &[],
        );

        assert_eq!(terminal_reason, LlmTerminalReason::ProviderError);
    }

    #[test]
    fn google_image_safety_finish_reason_maps_to_content_filter() {
        let terminal_reason = GoogleOAuthProvider::terminal_reason_from_value(
            &json!({"candidates":[{"finishReason":"IMAGE_SAFETY"}]}),
            &[],
        );

        assert_eq!(terminal_reason, LlmTerminalReason::ContentFilter);
    }

    #[test]
    fn thinking_config_omits_thoughts_unless_provider_exposes_thinking() {
        let hidden_provider = GoogleOAuthProvider::new("access", "refresh", 0);
        let hidden = GoogleOAuthProvider::build_request(
            &hidden_provider,
            &request(Some("medium")),
            Vec::new(),
            None,
        );
        assert_eq!(
            hidden["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "medium"
        );
        assert!(
            hidden["request"]["generationConfig"]["thinkingConfig"]
                .get("includeThoughts")
                .is_none()
        );

        let exposed_provider =
            GoogleOAuthProvider::new("access", "refresh", 0).with_options(ProviderOptions {
                thinking: lash_core::ProviderThinkingPolicy { expose: true },
                ..ProviderOptions::default()
            });
        let exposed = GoogleOAuthProvider::build_request(
            &exposed_provider,
            &request(Some("medium")),
            Vec::new(),
            None,
        );
        assert_eq!(
            exposed["request"]["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
    }

    #[test]
    fn output_token_cap_maps_to_max_output_tokens() {
        let provider =
            GoogleOAuthProvider::new("access", "refresh", 0).with_options(ProviderOptions {
                max_output_tokens: Some(9999),
                ..ProviderOptions::default()
            });

        let mut req = request(None);
        req.generation.output_token_cap = NonZeroUsize::new(4096);
        let body = GoogleOAuthProvider::build_request(&provider, &req, Vec::new(), None);

        assert_eq!(body["request"]["generationConfig"]["maxOutputTokens"], 4096);
        let provider_limited =
            GoogleOAuthProvider::build_request(&provider, &request(None), Vec::new(), None);
        assert_eq!(
            provider_limited["request"]["generationConfig"]["maxOutputTokens"],
            9999
        );
    }
}
