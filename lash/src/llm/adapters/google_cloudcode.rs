use std::collections::HashMap;
use std::sync::OnceLock;

use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::llm::adapters::streaming::{drive_sse_response, emit_progress};
use crate::llm::timeouts::{
    LlmTimeouts, build_http_client, read_response_text, response_start_timeout, send_request,
};
use crate::llm::transport::{LlmTransport, LlmTransportError};
use crate::llm::types::{
    LlmMessage, LlmOutputPart, LlmOutputSpec, LlmReplayChunk, LlmRequest, LlmResponse, LlmRole,
    LlmUsage, ModelSelection, coalesce_replay_messages,
};
use crate::model_variant::VariantRequestConfig;
use crate::provider::Provider;

const CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const CODE_ASSIST_API_VERSION: &str = "v1internal";
const GEMINI_FILES_UPLOAD_URL: &str =
    "https://generativelanguage.googleapis.com/upload/v1beta/files";

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

pub struct GoogleCloudCodeAdapter {
    client: reqwest::Client,
    request_timeout: Option<std::time::Duration>,
    chunk_timeout: std::time::Duration,
}

impl Default for GoogleCloudCodeAdapter {
    fn default() -> Self {
        Self::new(LlmTimeouts::default())
    }
}

impl GoogleCloudCodeAdapter {
    fn uploaded_attachment_cache()
    -> &'static tokio::sync::Mutex<HashMap<UploadedAttachmentCacheKey, UploadedAttachmentRef>> {
        static CACHE: OnceLock<
            tokio::sync::Mutex<HashMap<UploadedAttachmentCacheKey, UploadedAttachmentRef>>,
        > = OnceLock::new();
        CACHE.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()))
    }

    pub fn new(timeouts: LlmTimeouts) -> Self {
        Self {
            client: build_http_client(),
            request_timeout: timeouts.request_timeout,
            chunk_timeout: timeouts.chunk_timeout,
        }
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

    fn inline_attachment_part(att: &crate::llm::types::LlmAttachment) -> Value {
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
        for chunk in coalesce_replay_messages(&req.messages) {
            match chunk {
                LlmReplayChunk::Message(msg) => {
                    if matches!(msg.role, LlmRole::System) {
                        continue;
                    }
                    let role = match msg.role {
                        LlmRole::Assistant => "model",
                        LlmRole::User => "user",
                        LlmRole::System => "user",
                    };
                    let part = Self::content_part_for_message_with_attachment_parts(
                        attachment_parts,
                        &msg,
                    );
                    // Merge consecutive same-role messages into a single parts array
                    // so text + images land in one API message.
                    if let Some(prev) = out.last_mut()
                        && prev.get("role").and_then(|r| r.as_str()) == Some(role)
                        && prev.get("parts").is_some_and(|p| p.is_array())
                    {
                        prev["parts"].as_array_mut().unwrap().push(part);
                    } else {
                        out.push(json!({
                            "role": role,
                            "parts": [part],
                        }));
                    }
                }
                LlmReplayChunk::AssistantToolCalls { text, tool_calls } => {
                    let mut parts = Vec::new();
                    if let Some(text) = text.filter(|text| !text.is_empty()) {
                        parts.push(json!({ "text": text }));
                    }
                    parts.extend(tool_calls.into_iter().map(|call| {
                        json!({
                            "functionCall": {
                                "id": call.call_id,
                                "name": call.tool_name,
                                "args": serde_json::from_str::<Value>(&call.input_json)
                                    .unwrap_or_else(|_| json!({"_raw": call.input_json})),
                            }
                        })
                    }));
                    out.push(json!({
                        "role": "model",
                        "parts": parts,
                    }));
                }
                LlmReplayChunk::ToolResults { results } => {
                    let parts = results
                        .into_iter()
                        .map(|msg| {
                            json!({
                                "functionResponse": {
                                    "id": msg.tool_call_id.clone().unwrap_or_default(),
                                    "name": msg.tool_name.clone().unwrap_or_else(|| "tool".to_string()),
                                    "response": { "output": msg.content }
                                }
                            })
                        })
                        .collect::<Vec<_>>();
                    out.push(json!({
                        "role": "user",
                        "parts": parts,
                    }));
                }
            }
        }
        out
    }

    #[cfg(test)]
    fn build_contents(req: &LlmRequest) -> Vec<Value> {
        let attachment_parts = req
            .attachments
            .iter()
            .map(Self::inline_attachment_part)
            .collect::<Vec<_>>();
        Self::build_contents_with_attachment_parts(req, &attachment_parts)
    }

    fn content_part_for_message_with_attachment_parts(
        attachment_parts: &[Value],
        msg: &LlmMessage,
    ) -> Value {
        match msg.kind.as_str() {
            "image" if matches!(msg.role, LlmRole::User) => {
                let idx = msg.image_idx.max(0) as usize;
                Self::attachment_part_for_index(attachment_parts, idx)
            }
            _ => json!({ "text": msg.content.clone() }),
        }
    }

    fn uses_legacy_tool_parameters(model: &str) -> bool {
        model.starts_with("claude-")
    }

    fn google_tool_choice(choice: &crate::llm::types::LlmToolChoice) -> &'static str {
        match choice {
            crate::llm::types::LlmToolChoice::Auto => "AUTO",
            crate::llm::types::LlmToolChoice::None => "NONE",
            crate::llm::types::LlmToolChoice::Required => "ANY",
        }
    }

    fn system_instruction(req: &LlmRequest) -> Option<Value> {
        let text = req
            .messages
            .iter()
            .filter(|msg| matches!(msg.role, LlmRole::System) && msg.kind != "tool_result")
            .map(|msg| msg.content.as_str())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if text.is_empty() {
            None
        } else {
            Some(json!({
                "parts": [{ "text": text }],
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
                    out.push(LlmOutputPart::ToolCall {
                        call_id: function_call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        tool_name: name.to_string(),
                        input_json,
                    });
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

    fn process_sse_event(
        raw: &str,
        full: &mut String,
        deltas: &mut Vec<String>,
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
            Self::apply_stream_piece(full, deltas, &piece);
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
                if let Some(text) = item.get("text").and_then(|t| t.as_str())
                    && !text.is_empty()
                {
                    parts.push(LlmOutputPart::Text {
                        text: text.to_string(),
                    });
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
                    });
                }
            }
        }
        parts
    }

    fn upload_cache_key(
        project_id: Option<&str>,
        att: &crate::llm::types::LlmAttachment,
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
        att: &crate::llm::types::LlmAttachment,
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
        att: &crate::llm::types::LlmAttachment,
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
            self.request_timeout,
            "Gemini Files upload start timed out",
        )
        .await?;
        if !start_resp.status().is_success() {
            let status = start_resp.status().as_u16();
            let body = read_response_text(
                start_resp,
                self.request_timeout,
                "Gemini Files upload start body timed out",
            )
            .await
            .unwrap_or_default();
            return Err(LlmTransportError {
                message: format!("Gemini Files upload start failed with {}", status),
                retryable: status == 429 || status >= 500,
                raw: Some(body),
                code: Some(status.to_string()),
            });
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
            self.request_timeout,
            "Gemini Files upload finalize timed out",
        )
        .await?;
        if !finalize_resp.status().is_success() {
            let status = finalize_resp.status().as_u16();
            let body = read_response_text(
                finalize_resp,
                self.request_timeout,
                "Gemini Files upload finalize body timed out",
            )
            .await
            .unwrap_or_default();
            return Err(LlmTransportError {
                message: format!("Gemini Files upload finalize failed with {}", status),
                retryable: status == 429 || status >= 500,
                raw: Some(body),
                code: Some(status.to_string()),
            });
        }

        let upload_status = finalize_resp
            .headers()
            .get("x-goog-upload-status")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = read_response_text(
            finalize_resp,
            self.request_timeout,
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
        attachments: &[crate::llm::types::LlmAttachment],
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
        provider: &Provider,
        req: &LlmRequest,
        contents: Vec<Value>,
        project_id: Option<&str>,
    ) -> Value {
        let mut request = json!({
            "model": req.model,
            "user_prompt_id": uuid::Uuid::new_v4().to_string(),
            "request": {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 32768,
                }
            }
        });
        if let Some(system_instruction) = Self::system_instruction(req) {
            request["request"]["systemInstruction"] = system_instruction;
        }
        if let Some(session_id) = req.session_id.as_deref() {
            request["request"]["sessionId"] = json!(session_id);
        }
        if let Some(variant) = req.model_variant.as_deref()
            && let Some(config) =
                crate::model_variant::request_config(provider, &req.model, variant)
        {
            match config {
                VariantRequestConfig::GoogleThinkingLevel { level } => {
                    request["request"]["generationConfig"]["thinkingConfig"] = json!({
                        "includeThoughts": true,
                        "thinkingLevel": level,
                    });
                }
                VariantRequestConfig::GoogleThinkingBudget { budget_tokens } => {
                    request["request"]["generationConfig"]["thinkingConfig"] = json!({
                        "includeThoughts": true,
                        "thinkingBudget": budget_tokens,
                    });
                }
                _ => {}
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
        stream_events: Option<crate::llm::types::LlmEventSender>,
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
            response_start_timeout(
                self.request_timeout,
                self.chunk_timeout,
                stream_events.is_some(),
            ),
            "Cloud Code response start timed out",
        )
        .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_response_text(
                resp,
                self.request_timeout,
                "Cloud Code response body timed out",
            )
            .await
            .unwrap_or_default();
            return Err(LlmTransportError {
                message: format!("Cloud Code request failed with {}", status),
                retryable: status == 429 || status >= 500,
                raw: Some(body),
                code: Some(status.to_string()),
            });
        }

        if stream_events.is_none() {
            let text = read_response_text(
                resp,
                self.request_timeout,
                "Cloud Code response body timed out",
            )
            .await?;
            let value: Value = serde_json::from_str(&text).map_err(|e| {
                LlmTransportError::new(format!("Invalid Cloud Code response JSON: {e}"))
                    .with_raw(text.clone())
            })?;
            let parts = Self::response_parts_from_value(&value);
            let full_text = parts
                .iter()
                .filter_map(|part| match part {
                    LlmOutputPart::Text { text } => Some(text.as_str()),
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
            return Ok(LlmResponse {
                full_text,
                deltas: Vec::new(),
                parts,
                usage,
                request_body,
                http_summary: Some(format!("HTTP POST {}", url)),
            });
        }

        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut tool_call_parts: Vec<LlmOutputPart> = Vec::new();
        drive_sse_response(
            resp,
            self.chunk_timeout,
            "Cloud Code stream chunk timed out",
            |raw| {
                let prev_len = deltas.len();
                let prev_usage = usage.clone();
                Self::process_sse_event(
                    &raw,
                    &mut full,
                    &mut deltas,
                    &mut usage,
                    Some(&mut tool_call_parts),
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
        parts.extend(tool_call_parts);

        Ok(LlmResponse {
            full_text: full,
            deltas,
            parts,
            usage,
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

        let resp = self
            .client
            .post(Self::method_url("loadCodeAssist"))
            .bearer_auth(access_token)
            .json(&req)
            .send()
            .await
            .map_err(|e| LlmTransportError::new(format!("HTTP request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmTransportError {
                message: format!("Cloud Code loadCodeAssist failed with {}", status),
                retryable: status == 429 || status >= 500,
                raw: Some(body),
                code: Some(status.to_string()),
            });
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

#[async_trait]
impl LlmTransport for GoogleCloudCodeAdapter {
    fn default_root_model(&self) -> &'static str {
        "gemini-3.1-pro-preview"
    }

    fn default_agent_model(&self, tier: &str) -> Option<ModelSelection> {
        match tier {
            "low" => Some(ModelSelection {
                model: "gemini-3-flash-preview",
                variant: Some("low"),
            }),
            "medium" | "high" => Some(ModelSelection {
                model: "gemini-3.1-pro-preview",
                variant: Some(if tier == "medium" { "medium" } else { "high" }),
            }),
            _ => None,
        }
    }

    fn normalize_model(&self, model: &str) -> String {
        model.strip_prefix("google/").unwrap_or(model).to_string()
    }

    fn context_lookup_model(&self, model: &str) -> String {
        if model.contains('/') {
            model.to_string()
        } else {
            format!("google/{model}")
        }
    }

    async fn ensure_ready(&self, provider: &mut Provider) -> Result<bool, LlmTransportError> {
        let Provider::GoogleOAuth {
            access_token,
            project_id,
            ..
        } = provider
        else {
            return Err(LlmTransportError::new(
                "Google Cloud Code adapter received non-Google provider",
            ));
        };

        if project_id.is_none() {
            let hint = std::env::var("GOOGLE_CLOUD_PROJECT")
                .ok()
                .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT_ID").ok());
            let resolved = self
                .resolve_project_id(access_token, hint.as_deref())
                .await?;
            *project_id = resolved;
            return Ok(true);
        }

        Ok(false)
    }

    async fn complete(
        &self,
        provider: &mut Provider,
        req: LlmRequest,
    ) -> Result<LlmResponse, LlmTransportError> {
        let stream_events = req.stream_events.clone();
        let (access_token, project_id) = match provider {
            Provider::GoogleOAuth {
                access_token,
                project_id,
                ..
            } => (access_token.clone(), project_id.clone()),
            _ => {
                return Err(LlmTransportError::new(
                    "Google Cloud Code adapter received non-Google provider",
                ));
            }
        };

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

        let request = Self::build_request(provider, &req, contents, project_id.as_deref());

        match self
            .execute_request(&access_token, request, stream_events.clone())
            .await
        {
            Ok(response) => Ok(response),
            Err(err) if used_uploaded_files && Self::should_retry_inline(&err) => {
                let inline_request =
                    Self::build_request(provider, &req, inline_contents, project_id.as_deref());
                self.execute_request(&access_token, inline_request, stream_events)
                    .await
            }
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: LlmRole, kind: &str, content: &str) -> LlmMessage {
        LlmMessage {
            role,
            content: content.to_string(),
            kind: kind.to_string(),
            image_idx: -1,
            tool_call_id: None,
            tool_name: None,
        }
    }

    #[test]
    fn build_contents_uses_structured_replay_for_standard_mode() {
        let req = LlmRequest {
            model: "gemini-3.1-pro-preview".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "question"),
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
            ],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let contents = GoogleCloudCodeAdapter::build_contents(&req);
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["functionCall"]["name"], "read_file");
        assert_eq!(contents[2]["parts"][0]["functionResponse"]["id"], "call_1");
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"]["output"],
            "ok"
        );
    }

    #[test]
    fn streaming_extracts_function_calls() {
        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut tool_parts = Vec::new();

        GoogleCloudCodeAdapter::process_sse_event(
            r#"{"response":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read_file","args":{"path":"a.rs"}}}]}}]}}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            Some(&mut tool_parts),
        )
        .unwrap();

        assert!(full.is_empty());
        assert_eq!(tool_parts.len(), 1);
        match &tool_parts[0] {
            LlmOutputPart::ToolCall {
                tool_name,
                input_json,
                ..
            } => {
                assert_eq!(tool_name, "read_file");
                let args: Value = serde_json::from_str(input_json).unwrap();
                assert_eq!(args["path"], "a.rs");
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn streaming_text_and_function_calls() {
        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut tool_parts = Vec::new();

        GoogleCloudCodeAdapter::process_sse_event(
            r#"{"response":{"candidates":[{"content":{"parts":[{"text":"Let me check."}]}}]}}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            Some(&mut tool_parts),
        )
        .unwrap();

        GoogleCloudCodeAdapter::process_sse_event(
            r#"{"response":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read_file","args":{"path":"b.rs"}}}]}}]}}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            Some(&mut tool_parts),
        )
        .unwrap();

        assert_eq!(full, "Let me check.");
        assert_eq!(tool_parts.len(), 1);
        match &tool_parts[0] {
            LlmOutputPart::ToolCall { tool_name, .. } => assert_eq!(tool_name, "read_file"),
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn build_contents_uses_single_user_message() {
        let req = LlmRequest {
            model: "gemini".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "history"),
            ],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let contents = GoogleCloudCodeAdapter::build_contents(&req);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"].as_array().map(Vec::len), Some(1));
        assert_eq!(contents[0]["parts"][0]["text"], "history");
    }

    #[test]
    fn build_contents_can_use_uploaded_file_data_for_prompt_images() {
        let req = LlmRequest {
            model: "gemini".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "look"),
                LlmMessage {
                    role: LlmRole::User,
                    content: String::new(),
                    kind: "image".to_string(),
                    image_idx: 0,
                    tool_call_id: None,
                    tool_name: None,
                },
            ],
            attachments: vec![crate::llm::types::LlmAttachment {
                mime: "image/png".to_string(),
                data: vec![1, 2, 3],
            }],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let contents = GoogleCloudCodeAdapter::build_contents_with_attachment_parts(
            &req,
            &[json!({
                "fileData": {
                    "mimeType": "image/png",
                    "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/abc"
                }
            })],
        );

        // Text + image should be merged into a single user content object
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["parts"][0]["text"], "look");
        assert_eq!(
            contents[0]["parts"][1]["fileData"]["fileUri"],
            "https://generativelanguage.googleapis.com/v1beta/files/abc"
        );
    }

    #[test]
    fn build_contents_can_use_uploaded_file_data_for_replay_images() {
        let req = LlmRequest {
            model: "gemini".to_string(),
            messages: vec![LlmMessage {
                role: LlmRole::User,
                content: String::new(),
                kind: "image".to_string(),
                image_idx: 0,
                tool_call_id: None,
                tool_name: None,
            }],
            attachments: vec![crate::llm::types::LlmAttachment {
                mime: "image/png".to_string(),
                data: vec![1, 2, 3],
            }],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let contents = GoogleCloudCodeAdapter::build_contents_with_attachment_parts(
            &req,
            &[json!({
                "fileData": {
                    "mimeType": "image/png",
                    "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/replay"
                }
            })],
        );

        assert_eq!(
            contents[0]["parts"][0]["fileData"]["fileUri"],
            "https://generativelanguage.googleapis.com/v1beta/files/replay"
        );
    }

    #[test]
    fn build_request_adds_response_schema_for_structured_output() {
        let provider = Provider::GoogleOAuth {
            access_token: "tok".to_string(),
            refresh_token: "refresh".to_string(),
            expires_at: u64::MAX,
            project_id: Some("proj".to_string()),
            options: crate::provider::ProviderOptions::default(),
        };
        let req = LlmRequest {
            model: "gemini".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "hi"),
            ],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: Some(crate::llm::types::LlmOutputSpec::JsonSchema(
                crate::llm::types::LlmJsonSchema {
                    name: "ignored_by_google".to_string(),
                    schema: json!({"type": "OBJECT", "properties": {"ok": {"type": "BOOLEAN"}}}),
                    strict: true,
                },
            )),
            stream_events: None,
        };

        let request = GoogleCloudCodeAdapter::build_request(
            &provider,
            &req,
            GoogleCloudCodeAdapter::build_contents(&req),
            Some("proj"),
        );

        assert_eq!(
            request["request"]["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(
            request["request"]["generationConfig"]["responseSchema"]["type"],
            "OBJECT"
        );
    }

    #[test]
    fn build_request_uses_google_tool_config_and_json_schema_tools_for_gemini() {
        let provider = Provider::GoogleOAuth {
            access_token: "tok".to_string(),
            refresh_token: "refresh".to_string(),
            expires_at: u64::MAX,
            project_id: Some("proj".to_string()),
            options: crate::provider::ProviderOptions::default(),
        };
        let req = LlmRequest {
            model: "gemini-3.1-pro-preview".to_string(),
            messages: vec![message(LlmRole::User, "text", "hi")],
            attachments: vec![],
            tools: vec![crate::llm::types::LlmToolSpec {
                name: "find".to_string(),
                description: "Locate code".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
                output_schema: serde_json::Value::Null,
            }]
            .into(),
            tool_choice: crate::llm::types::LlmToolChoice::Required,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let request = GoogleCloudCodeAdapter::build_request(
            &provider,
            &req,
            GoogleCloudCodeAdapter::build_contents(&req),
            Some("proj"),
        );

        assert_eq!(
            request["request"]["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["type"],
            "object"
        );
        assert!(
            request["request"]["tools"][0]["functionDeclarations"][0]
                .get("parameters")
                .is_none()
        );
        assert_eq!(
            request["request"]["toolConfig"]["functionCallingConfig"]["mode"],
            "ANY"
        );
    }

    #[test]
    fn build_request_uses_legacy_tool_parameters_for_claude_models() {
        let provider = Provider::GoogleOAuth {
            access_token: "tok".to_string(),
            refresh_token: "refresh".to_string(),
            expires_at: u64::MAX,
            project_id: Some("proj".to_string()),
            options: crate::provider::ProviderOptions::default(),
        };
        let req = LlmRequest {
            model: "claude-sonnet-4-5".to_string(),
            messages: vec![message(LlmRole::User, "text", "hi")],
            attachments: vec![],
            tools: vec![crate::llm::types::LlmToolSpec {
                name: "find".to_string(),
                description: "Locate code".to_string(),
                input_schema: json!({"type": "object"}),
                output_schema: serde_json::Value::Null,
            }]
            .into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let request = GoogleCloudCodeAdapter::build_request(
            &provider,
            &req,
            GoogleCloudCodeAdapter::build_contents(&req),
            Some("proj"),
        );

        assert_eq!(
            request["request"]["tools"][0]["functionDeclarations"][0]["parameters"]["type"],
            "object"
        );
        assert!(
            request["request"]["tools"][0]["functionDeclarations"][0]
                .get("parametersJsonSchema")
                .is_none()
        );
        assert_eq!(
            request["request"]["toolConfig"]["functionCallingConfig"]["mode"],
            "NONE"
        );
    }

    #[test]
    fn build_request_places_session_and_thinking_config_in_generation_config() {
        let provider = Provider::GoogleOAuth {
            access_token: "tok".to_string(),
            refresh_token: "refresh".to_string(),
            expires_at: u64::MAX,
            project_id: Some("proj".to_string()),
            options: crate::provider::ProviderOptions::default(),
        };
        let req = LlmRequest {
            model: "gemini-3.1-pro-preview".to_string(),
            messages: vec![message(LlmRole::User, "text", "hi")],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: Some("medium".to_string()),
            session_id: Some("sess-123".to_string()),
            output_spec: None,
            stream_events: None,
        };

        let request = GoogleCloudCodeAdapter::build_request(
            &provider,
            &req,
            GoogleCloudCodeAdapter::build_contents(&req),
            Some("proj"),
        );

        assert_eq!(request["request"]["sessionId"], "sess-123");
        assert_eq!(
            request["request"]["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
        assert_eq!(
            request["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "medium"
        );
        assert!(request["request"].get("thinkingConfig").is_none());
    }
}
