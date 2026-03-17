use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};

use crate::llm::adapters::streaming::{drive_sse_response, emit_progress};
use crate::llm::timeouts::{
    LlmTimeouts, build_http_client, read_response_text, response_start_timeout, send_request,
};
use crate::llm::transport::{LlmTransport, LlmTransportError};
use crate::llm::types::{
    LlmMessage, LlmOutputPart, LlmPromptPart, LlmReplayChunk, LlmRequest, LlmResponse, LlmRole,
    LlmUsage, ModelSelection, coalesce_replay_messages,
};
use crate::model_variant::VariantRequestConfig;
use crate::provider::Provider;

const CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const CODE_ASSIST_API_VERSION: &str = "v1internal";

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

    fn build_contents(req: &LlmRequest) -> Vec<Value> {
        if !req.messages.is_empty() {
            let mut out = Vec::new();
            for chunk in coalesce_replay_messages(&req.messages) {
                match chunk {
                    LlmReplayChunk::Message(msg) => {
                        let role = match msg.role {
                            LlmRole::Assistant => "model",
                            LlmRole::User | LlmRole::System => "user",
                        };
                        let part = Self::content_part_for_message(req, &msg);
                        out.push(json!({
                            "role": role,
                            "parts": [part],
                        }));
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
                                        "response": { "content": msg.content }
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
            return out;
        }

        let mut parts = Vec::new();
        for part in &req.user_prompt {
            match part {
                LlmPromptPart::Text(text) => {
                    if !text.is_empty() {
                        parts.push(json!({ "text": text }));
                    }
                }
                LlmPromptPart::Image(idx) => {
                    if let Some(att) = req.attachments.get(*idx) {
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
                        parts.push(json!({
                            "inlineData": {
                                "mimeType": att.mime,
                                "data": b64,
                            }
                        }));
                    }
                }
            }
        }
        vec![json!({
            "role": "user",
            "parts": parts,
        })]
    }

    fn content_part_for_message(req: &LlmRequest, msg: &LlmMessage) -> Value {
        match msg.kind.as_str() {
            "image" if matches!(msg.role, LlmRole::User) => {
                let idx = msg.image_idx.max(0) as usize;
                if let Some(att) = req.attachments.get(idx) {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
                    json!({
                        "inlineData": {
                            "mimeType": att.mime,
                            "data": b64,
                        }
                    })
                } else {
                    json!({ "text": "[Image attached]" })
                }
            }
            _ => {
                let text = if matches!(msg.role, LlmRole::System) {
                    format!("Runtime note:\n{}", msg.content)
                } else {
                    msg.content.clone()
                };
                json!({ "text": text })
            }
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

        let contents = Self::build_contents(&req);

        let mut request = json!({
            "model": req.model,
            "user_prompt_id": uuid::Uuid::new_v4().to_string(),
            "request": {
                "contents": contents,
                "systemInstruction": {
                    "parts": [{ "text": req.system_prompt }],
                },
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 32768,
                }
            }
        });
        if let Some(variant) = req.model_variant.as_deref()
            && let Some(config) =
                crate::model_variant::request_config(provider, &req.model, variant)
        {
            match config {
                VariantRequestConfig::GoogleThinkingLevel { level } => {
                    request["request"]["thinkingConfig"] = json!({
                        "includeThoughts": true,
                        "thinkingLevel": level,
                    });
                }
                VariantRequestConfig::GoogleThinkingBudget { budget_tokens } => {
                    request["request"]["thinkingConfig"] = json!({
                        "includeThoughts": true,
                        "thinkingBudget": budget_tokens,
                    });
                }
                _ => {}
            }
        }
        if !req.tools.is_empty() {
            request["request"]["tools"] = json!([{
                "functionDeclarations": req
                    .tools
                    .iter()
                    .map(|tool| json!({
                        "name": tool.name.clone(),
                        "description": tool.description.clone(),
                        "parameters": tool.input_schema.clone(),
                    }))
                    .collect::<Vec<_>>()
            }]);
        }
        if let Some(project) = project_id.filter(|p| !p.trim().is_empty()) {
            request["project"] = json!(project);
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_contents_uses_structured_replay_for_standard_mode() {
        let req = LlmRequest {
            model: "gemini-3.1-pro-preview".to_string(),
            system_prompt: "sys".to_string(),
            user_prompt: vec![],
            messages: vec![
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
            ],
            attachments: vec![],
            tools: vec![],
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            stream_events: None,
        };

        let contents = GoogleCloudCodeAdapter::build_contents(&req);
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["functionCall"]["name"], "read_file");
        assert_eq!(contents[2]["parts"][0]["functionResponse"]["id"], "call_1");
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

        // Text event
        GoogleCloudCodeAdapter::process_sse_event(
            r#"{"response":{"candidates":[{"content":{"parts":[{"text":"Let me check."}]}}]}}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            Some(&mut tool_parts),
        )
        .unwrap();

        // Function call event
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
    fn build_contents_uses_single_user_prompt() {
        let req = LlmRequest {
            model: "gemini".to_string(),
            system_prompt: "sys".to_string(),
            user_prompt: vec![LlmPromptPart::Text("history".to_string())],
            messages: vec![],
            attachments: vec![],
            tools: vec![],
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            stream_events: None,
        };

        let contents = GoogleCloudCodeAdapter::build_contents(&req);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"].as_array().map(Vec::len), Some(1));
        assert_eq!(contents[0]["parts"][0]["text"], "history");
    }
}
