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
    LlmStreamEvent, LlmUsage, ModelSelection, coalesce_replay_messages,
};
use crate::provider::Provider;

pub struct CodexOAuthAdapter {
    client: reqwest::Client,
    request_timeout: Option<std::time::Duration>,
    chunk_timeout: std::time::Duration,
}

impl Default for CodexOAuthAdapter {
    fn default() -> Self {
        Self::new(LlmTimeouts::default())
    }
}

impl CodexOAuthAdapter {
    pub fn new(timeouts: LlmTimeouts) -> Self {
        Self {
            client: build_http_client(),
            request_timeout: timeouts.request_timeout,
            chunk_timeout: timeouts.chunk_timeout,
        }
    }

    fn user_input_item(req: &LlmRequest) -> Value {
        let mut content = Vec::new();
        for part in &req.user_prompt {
            match part {
                LlmPromptPart::Text(text) => {
                    if !text.is_empty() {
                        content.push(json!({"type": "input_text", "text": text}));
                    }
                }
                LlmPromptPart::Image(idx) => {
                    if let Some(att) = req.attachments.get(*idx) {
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
                        content.push(json!({
                            "type": "input_image",
                            "image_base64": b64,
                            "mime_type": att.mime,
                        }));
                    }
                }
            }
        }
        json!({
            "role": "user",
            "content": content,
        })
    }

    fn role_name(role: &LlmRole) -> &'static str {
        match role {
            LlmRole::User => "user",
            LlmRole::Assistant => "assistant",
            LlmRole::System => "system",
        }
    }

    fn content_part_for_message(req: &LlmRequest, msg: &LlmMessage) -> Value {
        match msg.kind.as_str() {
            "image" if matches!(msg.role, LlmRole::User) => {
                let idx = msg.image_idx.max(0) as usize;
                if let Some(att) = req.attachments.get(idx) {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
                    json!({
                        "type": "input_image",
                        "image_base64": b64,
                        "mime_type": att.mime,
                    })
                } else {
                    json!({"type": "input_text", "text": "[Image attached]"})
                }
            }
            _ => {
                if matches!(msg.role, LlmRole::Assistant) {
                    json!({"type": "output_text", "text": msg.content})
                } else {
                    json!({"type": "input_text", "text": msg.content})
                }
            }
        }
    }

    fn build_input(req: &LlmRequest) -> Vec<Value> {
        if req.messages.is_empty() {
            return vec![
                json!({
                    "role": "system",
                    "content": [{"type": "input_text", "text": req.system_prompt}],
                }),
                Self::user_input_item(req),
            ];
        }

        let mut input = vec![json!({
            "role": "system",
            "content": [{"type": "input_text", "text": req.system_prompt}],
        })];
        for chunk in coalesce_replay_messages(&req.messages) {
            match chunk {
                LlmReplayChunk::Message(msg) => {
                    if msg.kind == "tool_result" {
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": msg.tool_call_id.unwrap_or_default(),
                            "output": msg.content,
                        }));
                    } else {
                        input.push(json!({
                            "role": Self::role_name(&msg.role),
                            "content": [Self::content_part_for_message(req, &msg)],
                        }));
                    }
                }
                LlmReplayChunk::AssistantToolCalls { text, tool_calls } => {
                    if let Some(text) = text.filter(|text| !text.is_empty()) {
                        input.push(json!({
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }));
                    }
                    input.extend(tool_calls.into_iter().map(|call| {
                        json!({
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.tool_name,
                            "arguments": call.input_json,
                        })
                    }));
                }
                LlmReplayChunk::ToolResults { results } => {
                    input.extend(results.into_iter().map(|msg| {
                        json!({
                            "type": "function_call_output",
                            "call_id": msg.tool_call_id.unwrap_or_default(),
                            "output": msg.content,
                        })
                    }));
                }
            }
        }
        input
    }

    fn build_request_body(req: &LlmRequest, stream: bool) -> Value {
        let mut body = json!({
            "model": req.model,
            "input": Self::build_input(req),
            "stream": stream,
            "store": false,
            "instructions": "",
        });
        if let Some(effort) = req.model_variant.as_deref() {
            body["reasoning"] = json!({"effort": effort});
        }
        if let Some(session_id) = req.session_id.as_deref() {
            body["prompt_cache_key"] = json!(session_id);
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(
                req.tools
                    .iter()
                    .map(|tool| json!({
                        "type": "function",
                        "name": tool.name.clone(),
                        "description": tool.description.clone(),
                        "parameters": tool.input_schema.clone(),
                    }))
                    .collect::<Vec<_>>()
            );
            body["tool_choice"] = match req.tool_choice {
                crate::llm::types::LlmToolChoice::Auto => json!("auto"),
                crate::llm::types::LlmToolChoice::None => json!("none"),
                crate::llm::types::LlmToolChoice::Required => json!("required"),
            };
            body["parallel_tool_calls"] = json!(true);
        }
        body
    }

    fn extract_text(value: &Value) -> String {
        if let Some(s) = value.get("output_text").and_then(|v| v.as_str()) {
            return s.to_string();
        }
        if let Some(arr) = value.get("output").and_then(|v| v.as_array()) {
            let mut out = String::new();
            for item in arr {
                if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                    for part in content {
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            out.push_str(text);
                        }
                    }
                }
            }
            return out;
        }
        String::new()
    }

    fn parse_i64(v: Option<&Value>) -> i64 {
        match v {
            Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
            Some(Value::String(s)) => s.parse::<i64>().unwrap_or(0),
            _ => 0,
        }
    }

    fn extract_usage(value: &Value) -> LlmUsage {
        let usage = value.get("usage").unwrap_or(&Value::Null);
        LlmUsage {
            input_tokens: Self::parse_i64(usage.get("input_tokens")),
            output_tokens: Self::parse_i64(usage.get("output_tokens")),
            cached_input_tokens: Self::parse_i64(
                usage
                    .get("input_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .or_else(|| usage.get("cached_input_tokens"))
                    .or_else(|| usage.get("cached_tokens")),
            ),
            reasoning_tokens: Self::parse_i64(usage.get("reasoning_tokens").or_else(|| {
                usage
                    .get("output_tokens_details")
                    .and_then(|d| d.get("reasoning_tokens"))
            })),
        }
    }

    fn merge_usage(dst: &mut LlmUsage, next: &LlmUsage) {
        if next.input_tokens > 0 {
            dst.input_tokens = next.input_tokens;
        }
        if next.output_tokens > 0 {
            dst.output_tokens = next.output_tokens;
        }
        if next.cached_input_tokens > 0 {
            dst.cached_input_tokens = next.cached_input_tokens;
        }
        if next.reasoning_tokens > 0 {
            dst.reasoning_tokens = next.reasoning_tokens;
        }
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

    fn log_sse_event(
        event_type: &str,
        raw: &str,
        added_deltas: &[String],
        full_len: usize,
        usage: &LlmUsage,
        has_final_response: bool,
    ) {
        tracing::debug!(
            target: "lash::llm::codex_oauth",
            event_type,
            raw_len = raw.len(),
            raw_preview = %raw.chars().take(240).collect::<String>(),
            delta_count = added_deltas.len(),
            delta_lens = ?added_deltas.iter().map(|d| d.len()).collect::<Vec<_>>(),
            full_len,
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            cached_input_tokens = usage.cached_input_tokens,
            reasoning_tokens = usage.reasoning_tokens,
            has_final_response,
            "codex sse event"
        );
    }

    fn process_sse_event(
        raw: &str,
        full: &mut String,
        deltas: &mut Vec<String>,
        usage: &mut LlmUsage,
        final_response: &mut Option<Value>,
    ) -> Result<(), LlmTransportError> {
        let raw = raw.trim();
        if raw.is_empty() || raw == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(raw).map_err(|e| {
            LlmTransportError::new(format!("Invalid Codex SSE payload: {e}")).with_raw(raw)
        })?;
        let event_type = event
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        if event_type == "error" {
            let msg = event
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Codex stream error");
            return Err(LlmTransportError::new(msg).with_raw(event.to_string()));
        }

        let had_final_response = event.get("response").is_some();
        let prev_delta_len = deltas.len();

        if let Some(resp_value) = event.get("response") {
            *final_response = Some(resp_value.clone());
            let u = Self::extract_usage(resp_value);
            Self::merge_usage(usage, &u);
        } else {
            let u = Self::extract_usage(&event);
            Self::merge_usage(usage, &u);
        }

        match event_type.as_str() {
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(|d| d.as_str()) {
                    Self::apply_stream_piece(full, deltas, delta);
                }
            }
            "response.output_text.done" => {
                if let Some(text) = event.get("text").and_then(|t| t.as_str()) {
                    Self::apply_stream_piece(full, deltas, text);
                }
            }
            "response.completed" => {
                if let Some(resp_value) = event.get("response") {
                    let final_text = Self::extract_text(resp_value);
                    Self::apply_stream_piece(full, deltas, &final_text);
                }
            }
            "response.failed" => {
                let msg = event
                    .get("response")
                    .and_then(|r| r.get("error"))
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("Codex response failed");
                return Err(LlmTransportError::new(msg).with_raw(event.to_string()));
            }
            _ => {}
        }

        Self::log_sse_event(
            &event_type,
            raw,
            &deltas[prev_delta_len..],
            full.len(),
            usage,
            had_final_response,
        );
        Ok(())
    }

    fn parse_sse_payload(
        payload: &str,
        full: &mut String,
        deltas: &mut Vec<String>,
        usage: &mut LlmUsage,
        final_response: &mut Option<Value>,
    ) -> Result<(), LlmTransportError> {
        let mut event_lines: Vec<String> = Vec::new();
        for mut line in payload.lines().map(|l| l.to_string()) {
            if line.ends_with('\r') {
                line.pop();
            }
            if let Some(data) = line.strip_prefix("data:") {
                event_lines.push(data.trim().to_string());
                continue;
            }
            if line.starts_with("event:") {
                continue;
            }
            if line.trim().is_empty() {
                if !event_lines.is_empty() {
                    let raw = event_lines.join("\n");
                    Self::process_sse_event(&raw, full, deltas, usage, final_response)?;
                    event_lines.clear();
                }
                continue;
            }
        }
        if !event_lines.is_empty() {
            let raw = event_lines.join("\n");
            Self::process_sse_event(&raw, full, deltas, usage, final_response)?;
        }
        Ok(())
    }

    fn looks_like_sse_payload(payload: &str) -> bool {
        let trimmed = payload.trim_start();
        trimmed.starts_with("event:")
            || trimmed.starts_with("data:")
            || payload.contains("\nevent:")
            || payload.contains("\ndata:")
    }

    fn response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
        let mut parts = Vec::new();
        if let Some(output) = value.get("output").and_then(|v| v.as_array()) {
            for item in output {
                match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "message" => {
                        if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                            for block in content {
                                if let Some(text) = block.get("text").and_then(|v| v.as_str())
                                    && !text.is_empty()
                                {
                                    parts.push(LlmOutputPart::Text {
                                        text: text.to_string(),
                                    });
                                }
                            }
                        }
                    }
                    "function_call" => {
                        let Some(name) = item.get("name").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let input_json = item
                            .get("arguments")
                            .map(|v| {
                                v.as_str()
                                    .map(str::to_string)
                                    .unwrap_or_else(|| v.to_string())
                            })
                            .unwrap_or_else(|| "{}".to_string());
                        parts.push(LlmOutputPart::ToolCall {
                            call_id: item
                                .get("call_id")
                                .and_then(|v| v.as_str())
                                .map(str::to_string)
                                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                            tool_name: name.to_string(),
                            input_json,
                        });
                    }
                    _ => {}
                }
            }
        }
        if parts.is_empty()
            && let Some(text) = value.get("output_text").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            parts.push(LlmOutputPart::Text {
                text: text.to_string(),
            });
        }
        parts
    }
}

#[async_trait]
impl LlmTransport for CodexOAuthAdapter {
    fn default_root_model(&self) -> &'static str {
        "gpt-5.4"
    }

    fn default_agent_model(&self, tier: &str) -> Option<ModelSelection> {
        match tier {
            "low" => Some(ModelSelection {
                model: "gpt-5.3-codex-spark",
                variant: Some("low"),
            }),
            "medium" => Some(ModelSelection {
                model: "gpt-5.4",
                variant: Some("medium"),
            }),
            "high" => Some(ModelSelection {
                model: "gpt-5.4",
                variant: Some("high"),
            }),
            _ => None,
        }
    }

    fn normalize_model(&self, model: &str) -> String {
        model.to_string()
    }

    fn context_lookup_model(&self, model: &str) -> String {
        if model.contains('/') {
            model.to_string()
        } else {
            format!("openai/{model}")
        }
    }

    fn requires_streaming(&self) -> bool {
        true
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
        let (access_token, account_id) = match provider {
            Provider::Codex {
                access_token,
                account_id,
                ..
            } => (access_token.clone(), account_id.clone()),
            _ => {
                return Err(LlmTransportError::new(
                    "Codex adapter received non-Codex provider",
                ));
            }
        };

        let body = Self::build_request_body(&req, stream_events.is_some());

        let request_body = serde_json::to_string(&body).ok();
        let url = "https://chatgpt.com/backend-api/codex/responses".to_string();
        let mut http = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("originator", "lash")
            .header(
                "User-Agent",
                format!(
                    "lash/{} ({}; {})",
                    env!("CARGO_PKG_VERSION"),
                    std::env::consts::OS,
                    std::env::consts::ARCH
                ),
            )
            .json(&body);
        if let Some(session_id) = req.session_id.as_deref() {
            http = http.header("session_id", session_id);
        }
        if let Some(id) = account_id {
            http = http.header("ChatGPT-Account-Id", id);
        }

        let resp = send_request(
            http,
            response_start_timeout(
                self.request_timeout,
                self.chunk_timeout,
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
        if !status.is_success() {
            let text = match read_response_text(
                resp,
                self.request_timeout,
                "Codex response body timed out",
            )
            .await
            {
                Ok(text) => text,
                Err(err) => {
                    return Err(LlmTransportError {
                        message: format!(
                            "Codex request failed with {} and unreadable body: {}",
                            status.as_u16(),
                            err.message
                        ),
                        retryable: err.retryable
                            || status.as_u16() == 429
                            || status.as_u16() >= 500,
                        raw: err.raw,
                        code: Some(status.as_u16().to_string()),
                    });
                }
            };
            return Err(LlmTransportError {
                message: format!(
                    "Codex request failed with {}{}",
                    status.as_u16(),
                    content_type
                        .as_deref()
                        .map(|ct| format!(" ({ct})"))
                        .unwrap_or_default()
                ),
                retryable: status.as_u16() == 429 || status.as_u16() >= 500,
                raw: Some(text),
                code: Some(status.as_u16().to_string()),
            });
        }

        let is_sse = content_type
            .as_deref()
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);

        if !is_sse {
            let text =
                read_response_text(resp, self.request_timeout, "Codex response body timed out")
                    .await
                    .map_err(|err| {
                        LlmTransportError::new(format!(
                            "Codex returned non-SSE body{} but it could not be read: {}",
                            content_type
                                .as_deref()
                                .map(|ct| format!(" ({ct})"))
                                .unwrap_or_default(),
                            err.message
                        ))
                        .retryable(err.retryable)
                        .with_code(
                            err.code
                                .clone()
                                .unwrap_or_else(|| "body_read_failed".to_string()),
                        )
                    })?;
            if Self::looks_like_sse_payload(&text) {
                let mut full = String::new();
                let mut deltas = Vec::new();
                let mut usage = LlmUsage::default();
                let mut final_response = None;
                Self::parse_sse_payload(
                    &text,
                    &mut full,
                    &mut deltas,
                    &mut usage,
                    &mut final_response,
                )?;
                let parts = final_response
                    .as_ref()
                    .map(Self::response_parts_from_value)
                    .filter(|parts| !parts.is_empty())
                    .unwrap_or_else(|| {
                        if full.is_empty() {
                            Vec::new()
                        } else {
                            vec![LlmOutputPart::Text { text: full.clone() }]
                        }
                    });
                if let Some(tx) = &stream_events {
                    if usage != LlmUsage::default() {
                        let _ = tx.send(LlmStreamEvent::Usage(usage.clone()));
                    }
                    for piece in &deltas {
                        let _ = tx.send(LlmStreamEvent::Delta(piece.clone()));
                    }
                }
                return Ok(LlmResponse {
                    deltas,
                    full_text: full,
                    parts,
                    usage,
                    request_body,
                    http_summary: Some(format!("HTTP POST {} (stream/fallback)", url)),
                });
            }
            let value: Value = serde_json::from_str(&text).map_err(|e| {
                LlmTransportError::new(format!("Invalid Codex response JSON: {e}"))
                    .with_raw(text.clone())
            })?;
            let content = Self::extract_text(&value);
            let usage = Self::extract_usage(&value);
            let mut parts = Self::response_parts_from_value(&value);
            if parts.is_empty() && !content.is_empty() {
                parts.push(LlmOutputPart::Text {
                    text: content.clone(),
                });
            }
            if let Some(tx) = &stream_events {
                if usage != LlmUsage::default() {
                    let _ = tx.send(LlmStreamEvent::Usage(usage.clone()));
                }
                if !content.is_empty() {
                    let _ = tx.send(LlmStreamEvent::Delta(content.clone()));
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
        let mut final_response = None;
        drive_sse_response(
            resp,
            self.chunk_timeout,
            "Codex stream chunk timed out",
            |raw| {
                let prev_len = deltas.len();
                let prev_usage = usage.clone();
                Self::process_sse_event(
                    &raw,
                    &mut full,
                    &mut deltas,
                    &mut usage,
                    &mut final_response,
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

        let parts = final_response
            .as_ref()
            .map(Self::response_parts_from_value)
            .filter(|parts| !parts.is_empty())
            .unwrap_or_else(|| {
                if full.is_empty() {
                    Vec::new()
                } else {
                    vec![LlmOutputPart::Text { text: full.clone() }]
                }
            });

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

    #[test]
    fn parses_codex_sse_delta_and_completed_usage() {
        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut final_response = None;

        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_text.delta","delta":"Hi "}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            &mut final_response,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.completed","response":{"output_text":"Hi there","usage":{"input_tokens":30,"output_tokens":8,"input_tokens_details":{"cached_tokens":10}}}}"#,
            &mut full,
            &mut deltas,
            &mut usage,
            &mut final_response,
        )
        .unwrap();

        assert_eq!(full, "Hi there");
        assert_eq!(usage.input_tokens, 30);
        assert_eq!(usage.output_tokens, 8);
        assert_eq!(usage.cached_input_tokens, 10);
    }

    #[test]
    fn parses_codex_sse_payload_when_header_missing() {
        let payload = r#"event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hey "}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"there"}

event: response.completed
data: {"type":"response.completed","response":{"output_text":"Hey there","usage":{"input_tokens":9,"output_tokens":2,"input_tokens_details":{"cached_tokens":3}}}}
"#;

        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut final_response = None;
        CodexOAuthAdapter::parse_sse_payload(
            payload,
            &mut full,
            &mut deltas,
            &mut usage,
            &mut final_response,
        )
        .unwrap();

        assert_eq!(full, "Hey there");
        assert_eq!(usage.input_tokens, 9);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.cached_input_tokens, 3);
    }

    #[test]
    fn extracts_tool_calls_from_completed_stream_response() {
        let payload = r#"event: response.completed
data: {"type":"response.completed","response":{"output":[{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"README.md\"}"}],"usage":{"input_tokens":12,"output_tokens":3}}}
"#;

        let mut full = String::new();
        let mut deltas = Vec::new();
        let mut usage = LlmUsage::default();
        let mut final_response = None;
        CodexOAuthAdapter::parse_sse_payload(
            payload,
            &mut full,
            &mut deltas,
            &mut usage,
            &mut final_response,
        )
        .unwrap();

        let parts = final_response
            .as_ref()
            .map(CodexOAuthAdapter::response_parts_from_value)
            .unwrap_or_default();
        assert_eq!(
            parts,
            vec![LlmOutputPart::ToolCall {
                call_id: "call_1".to_string(),
                tool_name: "read_file".to_string(),
                input_json: "{\"path\":\"README.md\"}".to_string(),
            }]
        );
    }

    #[test]
    fn structured_messages_build_responses_input_items() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
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

        let input = {
            let mut input = vec![json!({
                "role": "system",
                "content": [{"type": "input_text", "text": req.system_prompt}],
            })];
            for chunk in coalesce_replay_messages(&req.messages) {
                match chunk {
                    LlmReplayChunk::Message(msg) => {
                        if msg.kind == "tool_result" {
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": msg.tool_call_id.unwrap_or_default(),
                                "output": msg.content,
                            }));
                        } else {
                            input.push(json!({
                                "role": CodexOAuthAdapter::role_name(&msg.role),
                                "content": [CodexOAuthAdapter::content_part_for_message(&req, &msg)],
                            }));
                        }
                    }
                    LlmReplayChunk::AssistantToolCalls { text, tool_calls } => {
                        if let Some(text) = text.filter(|text| !text.is_empty()) {
                            input.push(json!({
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text}],
                            }));
                        }
                        input.extend(tool_calls.into_iter().map(|call| {
                            json!({
                                "type": "function_call",
                                "call_id": call.call_id,
                                "name": call.tool_name,
                                "arguments": call.input_json,
                            })
                        }));
                    }
                    LlmReplayChunk::ToolResults { results } => {
                        input.extend(results.into_iter().map(|msg| {
                            json!({
                                "type": "function_call_output",
                                "call_id": msg.tool_call_id.unwrap_or_default(),
                                "output": msg.content,
                            })
                        }));
                    }
                }
            }
            input
        };

        assert_eq!(input.len(), 4);
        assert_eq!(input[1]["role"], "user");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[3]["type"], "function_call_output");
    }

    #[test]
    fn build_request_body_sets_prompt_cache_key_from_session_id() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            system_prompt: "sys".to_string(),
            user_prompt: vec![LlmPromptPart::Text("hello".to_string())],
            messages: vec![],
            attachments: vec![],
            tools: vec![],
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: Some("sess-123".to_string()),
            stream_events: None,
        };

        let body = CodexOAuthAdapter::build_request_body(&req, false);
        assert_eq!(body["prompt_cache_key"], "sess-123");
        assert_eq!(body["store"], false);
    }

    #[test]
    fn codex_transport_requires_streaming() {
        let adapter = CodexOAuthAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        assert!(LlmTransport::requires_streaming(&adapter));
    }

    #[test]
    fn user_input_item_uses_input_text_content_type() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            system_prompt: "sys".to_string(),
            user_prompt: vec![LlmPromptPart::Text("hello".to_string())],
            messages: vec![],
            attachments: vec![],
            tools: vec![],
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            stream_events: None,
        };

        let item = CodexOAuthAdapter::user_input_item(&req);

        assert_eq!(item["role"], "user");
        assert_eq!(item["content"][0]["type"], "input_text");
        assert_eq!(item["content"][0]["text"], "hello");
    }
}
