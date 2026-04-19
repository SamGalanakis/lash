use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::llm::adapters::streaming::{drive_sse_response, emit_progress};
use crate::llm::timeouts::{
    LlmTimeouts, build_http_client, read_response_text, response_start_timeout, send_request,
};
use crate::llm::transport::{LlmTransport, LlmTransportError};
use crate::llm::types::{
    LlmMessage, LlmOutputPart, LlmOutputSpec, LlmReplayChunk, LlmRequest, LlmResponse, LlmRole,
    LlmStreamEvent, LlmUsage, ModelSelection, coalesce_replay_messages,
};
use crate::provider::Provider;

pub struct CodexOAuthAdapter {
    client: reqwest::Client,
    request_timeout: Option<std::time::Duration>,
    chunk_timeout: std::time::Duration,
}

#[derive(Clone, Debug, Default)]
struct CodexStreamingToolCall {
    call_id: String,
    tool_name: String,
    input_json: String,
}

#[derive(Clone, Debug, Default)]
struct CodexStreamState {
    full_text: String,
    deltas: Vec<String>,
    parts: Vec<LlmOutputPart>,
    usage: LlmUsage,
    final_response: Option<Value>,
    current_text_part: Option<usize>,
    /// Index of the reasoning-summary `LlmOutputPart` currently receiving
    /// deltas. Each `response.reasoning_summary_part.added` starts a new
    /// entry; the server groups reasoning output into multiple "parts"
    /// (paragraphs) so we keep one slot per part instead of merging them
    /// into a single blob.
    current_reasoning_part: Option<usize>,
    /// Streaming-time reasoning-summary deltas collected since the last
    /// flush. Fed to `LlmStreamEvent::ReasoningDelta` by the caller so
    /// the UI can render thinking incrementally.
    reasoning_deltas: Vec<String>,
    tool_calls: HashMap<String, CodexStreamingToolCall>,
}

impl CodexStreamState {
    fn begin_message(&mut self) {
        // When a new message item follows a previous one with existing text,
        // emit a paragraph break so the UI renders separate items as separate
        // paragraphs instead of concatenating their sentences (e.g. progress
        // summaries that land as "blocks it.Next I'm fetching...").
        let prev_has_text = self
            .parts
            .iter()
            .rev()
            .find_map(|part| match part {
                LlmOutputPart::Text { text } if !text.is_empty() => Some(true),
                LlmOutputPart::Text { .. } => None,
                _ => None,
            })
            .unwrap_or(false);
        if prev_has_text {
            if let Some(idx) = self
                .parts
                .iter()
                .rposition(|part| matches!(part, LlmOutputPart::Text { text } if !text.is_empty()))
                && let Some(LlmOutputPart::Text { text }) = self.parts.get_mut(idx)
                && !text.ends_with("\n\n")
            {
                text.push_str("\n\n");
            }
            self.deltas.push("\n\n".to_string());
            self.recompute_full_text();
        }
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Text {
            text: String::new(),
        });
        self.current_text_part = Some(index);
    }

    fn finish_message(&mut self, item: Option<&Value>) {
        if let Some(item) = item {
            let text = CodexOAuthAdapter::message_text_from_item(item);
            if !text.is_empty() {
                self.reconcile_current_message_text(&text);
            }
        }
        self.current_text_part = None;
    }

    fn push_text_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }

        let part_index = self.ensure_text_part_index();

        if let Some(LlmOutputPart::Text { text }) = self.parts.get_mut(part_index) {
            text.push_str(piece);
        }
        self.deltas.push(piece.to_string());
        self.recompute_full_text();
    }

    fn reconcile_current_message_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let part_index = self.ensure_text_part_index();
        let existing = self
            .parts
            .get(part_index)
            .and_then(|part| match part {
                LlmOutputPart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();

        if text == existing {
            return;
        }
        if let Some(suffix) = text.strip_prefix(existing.as_str()) {
            self.push_text_delta(suffix);
            return;
        }
        self.set_text_part(part_index, text.to_string());
    }

    fn reconcile_final_response_text(&mut self, text: &str) {
        if text.is_empty() || self.full_text == text {
            return;
        }
        if let Some(suffix) = text.strip_prefix(self.full_text.as_str()) {
            self.push_text_delta(suffix);
            return;
        }
        if self.full_text.is_empty() {
            let part_index = self.ensure_text_part_index();
            self.set_text_part(part_index, text.to_string());
            self.deltas.push(text.to_string());
            return;
        }
        self.replace_text_output(text);
    }

    fn ensure_text_part_index(&mut self) -> usize {
        if let Some(index) = self.current_text_part {
            return index;
        }
        if let Some(index) = self
            .parts
            .iter()
            .rposition(|part| matches!(part, LlmOutputPart::Text { .. }))
        {
            return index;
        }

        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Text {
            text: String::new(),
        });
        index
    }

    fn set_text_part(&mut self, part_index: usize, text: String) {
        if let Some(LlmOutputPart::Text { text: existing }) = self.parts.get_mut(part_index) {
            *existing = text;
        }
        self.recompute_full_text();
    }

    fn replace_text_output(&mut self, text: &str) {
        let mut replaced = false;
        let mut parts = Vec::with_capacity(self.parts.len().max(1));
        for part in self.parts.drain(..) {
            match part {
                LlmOutputPart::Text { .. } if !replaced => {
                    parts.push(LlmOutputPart::Text {
                        text: text.to_string(),
                    });
                    replaced = true;
                }
                LlmOutputPart::Text { .. } => {}
                other => parts.push(other),
            }
        }
        if !replaced {
            parts.push(LlmOutputPart::Text {
                text: text.to_string(),
            });
        }
        self.parts = parts;
        self.current_text_part = None;
        self.recompute_full_text();
    }

    fn recompute_full_text(&mut self) {
        self.full_text.clear();
        for part in &self.parts {
            if let LlmOutputPart::Text { text } = part {
                self.full_text.push_str(text);
            }
        }
    }

    fn begin_reasoning_part(&mut self) {
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Reasoning {
            text: String::new(),
        });
        self.current_reasoning_part = Some(index);
    }

    fn push_reasoning_delta(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let index = match self.current_reasoning_part {
            Some(index) => index,
            None => {
                // Some providers send a delta before the `part.added`
                // event. Create an implicit part so we don't drop text.
                self.begin_reasoning_part();
                self.current_reasoning_part
                    .expect("reasoning part just pushed")
            }
        };
        if let Some(LlmOutputPart::Reasoning { text }) = self.parts.get_mut(index) {
            text.push_str(delta);
        }
        self.reasoning_deltas.push(delta.to_string());
    }

    fn finish_reasoning_part(&mut self) {
        // Drop the cursor; the next `part.added` will open a fresh slot.
        // Trim trailing whitespace off the completed part so concatenated
        // paragraphs don't carry stray blanks.
        if let Some(index) = self.current_reasoning_part.take()
            && let Some(LlmOutputPart::Reasoning { text }) = self.parts.get_mut(index)
        {
            let trimmed = text.trim_end();
            if trimmed.len() != text.len() {
                *text = trimmed.to_string();
            }
        }
    }

    fn take_reasoning_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reasoning_deltas)
    }

    fn update_tool_call_from_item(&mut self, item: &Value) -> Option<String> {
        let item_id = item.get("id").and_then(|v| v.as_str())?.to_string();
        let tool_call = self.tool_calls.entry(item_id.clone()).or_default();
        if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
            tool_call.call_id = call_id.to_string();
        }
        if let Some(tool_name) = item.get("name").and_then(|v| v.as_str()) {
            tool_call.tool_name = tool_name.to_string();
        }
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str())
            && !arguments.is_empty()
        {
            tool_call.input_json = arguments.to_string();
        }
        Some(item_id)
    }

    fn push_tool_call_delta(&mut self, item_id: &str, delta: &str) {
        if item_id.is_empty() || delta.is_empty() {
            return;
        }
        self.tool_calls
            .entry(item_id.to_string())
            .or_default()
            .input_json
            .push_str(delta);
    }

    fn set_tool_call_arguments(&mut self, item_id: &str, arguments: &str) {
        if item_id.is_empty() {
            return;
        }
        let tool_call = self.tool_calls.entry(item_id.to_string()).or_default();
        tool_call.input_json = arguments.to_string();
    }

    fn finish_tool_call(&mut self, item: &Value) -> Option<LlmOutputPart> {
        let item_id = self.update_tool_call_from_item(item)?;
        let mut tool_call = self.tool_calls.remove(&item_id).unwrap_or_default();
        if tool_call.call_id.is_empty() {
            tool_call.call_id = uuid::Uuid::new_v4().to_string();
        }
        if tool_call.tool_name.is_empty() {
            return None;
        }
        if tool_call.input_json.is_empty() {
            tool_call.input_json = "{}".to_string();
        }

        let part = LlmOutputPart::ToolCall {
            call_id: tool_call.call_id,
            tool_name: tool_call.tool_name,
            input_json: tool_call.input_json,
        };
        if !self.parts.iter().any(|existing| existing == &part) {
            self.parts.push(part.clone());
            return Some(part);
        }
        None
    }

    fn response_parts(&self) -> Vec<LlmOutputPart> {
        let parts = self
            .parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text } if text.is_empty() => None,
                LlmOutputPart::Reasoning { text } if text.trim().is_empty() => None,
                _ => Some(part.clone()),
            })
            .collect::<Vec<_>>();
        if !parts.is_empty() {
            return parts;
        }

        if let Some(final_response) = &self.final_response {
            let parts = CodexOAuthAdapter::response_parts_from_value(final_response);
            if !parts.is_empty() {
                return parts;
            }
            let text = CodexOAuthAdapter::extract_text(final_response);
            if !text.is_empty() {
                return vec![LlmOutputPart::Text { text }];
            }
        }

        if !self.full_text.is_empty() {
            return vec![LlmOutputPart::Text {
                text: self.full_text.clone(),
            }];
        }

        Vec::new()
    }

    fn response_full_text(&self, parts: &[LlmOutputPart]) -> String {
        if !self.full_text.is_empty() {
            return self.full_text.clone();
        }
        parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text } => Some(text.as_str()),
                LlmOutputPart::ToolCall { .. } | LlmOutputPart::Reasoning { .. } => None,
            })
            .collect::<String>()
    }
}

impl Default for CodexOAuthAdapter {
    fn default() -> Self {
        Self::new(LlmTimeouts::default())
    }
}

impl CodexOAuthAdapter {
    const CODEX_ORIGINATOR: &'static str = "codex_cli_rs";
    const CODEX_RESPONSES_URL: &'static str = "https://chatgpt.com/backend-api/codex/responses";

    pub fn new(timeouts: LlmTimeouts) -> Self {
        Self {
            client: build_http_client(),
            request_timeout: timeouts.request_timeout,
            chunk_timeout: timeouts.chunk_timeout,
        }
    }

    fn input_image_part(att: &crate::llm::types::LlmAttachment) -> Value {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
        let data_url = format!("data:{};base64,{}", att.mime, b64);
        json!({
            "type": "input_image",
            "image_url": data_url,
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
                    Self::input_image_part(att)
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

    fn build_input(req: &LlmRequest) -> (String, Vec<Value>) {
        let mut input = Vec::new();
        let mut instructions = Vec::new();
        for chunk in coalesce_replay_messages(&req.messages) {
            match chunk {
                LlmReplayChunk::Message(msg) => {
                    if matches!(msg.role, LlmRole::System) && msg.kind != "tool_result" {
                        if !msg.content.is_empty() {
                            instructions.push(msg.content);
                        }
                    } else if msg.kind == "tool_result" {
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": msg.tool_call_id.unwrap_or_default(),
                            "output": msg.content,
                        }));
                    } else {
                        // Merge consecutive user messages into a single multipart
                        // content array so text + images land in one API message.
                        let role_str = Self::role_name(&msg.role);
                        let part = Self::content_part_for_message(req, &msg);
                        if role_str == "user"
                            && let Some(prev) = input.last_mut()
                            && prev.get("role").and_then(|r| r.as_str()) == Some("user")
                            && prev.get("content").is_some_and(|c| c.is_array())
                        {
                            prev["content"].as_array_mut().unwrap().push(part);
                            continue;
                        }
                        input.push(json!({
                            "role": role_str,
                            "content": [part],
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
        (instructions.join("\n\n"), input)
    }

    fn build_tools(req: &LlmRequest) -> Vec<Value> {
        req.tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name.clone(),
                    "description": tool.description.clone(),
                    "strict": false,
                    "parameters": tool.input_schema.clone(),
                })
            })
            .collect()
    }

    fn tool_choice_value(choice: &crate::llm::types::LlmToolChoice) -> &'static str {
        match choice {
            crate::llm::types::LlmToolChoice::Auto => "auto",
            crate::llm::types::LlmToolChoice::None => "none",
            crate::llm::types::LlmToolChoice::Required => "required",
        }
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

    fn build_request_body(req: &LlmRequest, stream: bool) -> Value {
        let tools = Self::build_tools(req);
        let (instructions, input) = Self::build_input(req);
        let mut body = json!({
            "model": req.model,
            "instructions": instructions,
            "input": input,
            "tools": tools,
            "tool_choice": Self::tool_choice_value(&req.tool_choice),
            "parallel_tool_calls": !req.tools.is_empty(),
            "stream": stream,
            "store": false,
            "include": [],
        });
        if let Some(effort) = req.model_variant.as_deref() {
            body["reasoning"] = json!({"effort": effort});
        }
        if let Some(session_id) = req.session_id.as_deref() {
            body["prompt_cache_key"] = json!(session_id);
        }
        if let Some(output_spec) = &req.output_spec {
            body["text"] = json!({
                "format": match output_spec {
                    LlmOutputSpec::JsonObject => json!({ "type": "json_object" }),
                    LlmOutputSpec::JsonSchema(schema) => json!({
                        "type": "json_schema",
                        "name": schema.name,
                        "schema": schema.schema,
                        "strict": schema.strict,
                    }),
                }
            });
        }
        body
    }

    fn extract_text(value: &Value) -> String {
        if let Some(s) = value.get("output_text").and_then(|v| v.as_str()) {
            return s.to_string();
        }
        if let Some(arr) = value.get("output").and_then(|v| v.as_array()) {
            // Concatenate items with a paragraph break so this stays in sync
            // with `begin_message`, which injects "\n\n" between streamed
            // message items. Mismatched separators would otherwise trigger
            // `reconcile_final_response_text` to replace the live buffer.
            let mut items_text: Vec<String> = Vec::new();
            for item in arr {
                if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                    let mut item_text = String::new();
                    for part in content {
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            item_text.push_str(text);
                        }
                    }
                    if !item_text.is_empty() {
                        items_text.push(item_text);
                    }
                }
            }
            return items_text.join("\n\n");
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

    fn message_text_from_item(item: &Value) -> String {
        item.get("content")
            .and_then(|v| v.as_array())
            .map(|content| {
                content
                    .iter()
                    .filter_map(|part| match part.get("type").and_then(|v| v.as_str()) {
                        Some("output_text") => part.get("text").and_then(|v| v.as_str()),
                        Some("refusal") => part
                            .get("refusal")
                            .and_then(|v| v.as_str())
                            .or_else(|| part.get("text").and_then(|v| v.as_str())),
                        _ => None,
                    })
                    .collect::<String>()
            })
            .unwrap_or_default()
    }

    fn response_from_stream_state(
        state: CodexStreamState,
        request_body: Option<String>,
        http_summary: String,
    ) -> LlmResponse {
        let parts = state.response_parts();
        let full_text = state.response_full_text(&parts);
        LlmResponse {
            deltas: state.deltas,
            full_text,
            parts,
            usage: state.usage,
            provider_usage: None,
            request_body,
            http_summary: Some(http_summary),
        }
    }

    fn process_sse_event(
        raw: &str,
        state: &mut CodexStreamState,
        emitted_parts: Option<&mut Vec<LlmOutputPart>>,
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
        let prev_delta_len = state.deltas.len();

        if let Some(resp_value) = event.get("response") {
            state.final_response = Some(resp_value.clone());
            let u = Self::extract_usage(resp_value);
            Self::merge_usage(&mut state.usage, &u);
        } else {
            let u = Self::extract_usage(&event);
            Self::merge_usage(&mut state.usage, &u);
        }

        match event_type.as_str() {
            "response.output_item.added" => {
                if let Some(item) = event.get("item") {
                    match item.get("type").and_then(|v| v.as_str()) {
                        Some("message") => state.begin_message(),
                        Some("function_call") => {
                            let _ = state.update_tool_call_from_item(item);
                        }
                        // For reasoning items we wait for
                        // `reasoning_summary_part.added` to open a slot —
                        // the outer item carries no text on its own.
                        _ => {}
                    }
                }
            }
            "response.reasoning_summary_part.added" => {
                state.begin_reasoning_part();
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(|d| d.as_str()) {
                    state.push_reasoning_delta(delta);
                }
            }
            "response.reasoning_summary_text.done" => {
                // The `text` field on this event is the full text for the
                // current part; if our accumulator already matches we do
                // nothing, otherwise reconcile by appending the missing
                // suffix (mirrors the logic for `output_text.done`).
                if let Some(text) = event.get("text").and_then(|v| v.as_str())
                    && let Some(index) = state.current_reasoning_part
                    && let Some(LlmOutputPart::Reasoning { text: existing }) =
                        state.parts.get(index)
                {
                    let existing = existing.clone();
                    if text != existing
                        && let Some(suffix) = text.strip_prefix(existing.as_str())
                    {
                        state.push_reasoning_delta(suffix);
                    }
                }
            }
            "response.reasoning_summary_part.done" => {
                state.finish_reasoning_part();
            }
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(|d| d.as_str()) {
                    state.push_text_delta(delta);
                }
            }
            "response.output_text.done" => {}
            "response.function_call_arguments.delta" => {
                if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str())
                    && let Some(delta) = event.get("delta").and_then(|v| v.as_str())
                {
                    state.push_tool_call_delta(item_id, delta);
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str())
                    && let Some(arguments) = event.get("arguments").and_then(|v| v.as_str())
                {
                    state.set_tool_call_arguments(item_id, arguments);
                }
            }
            "response.output_item.done" => {
                if let Some(item) = event.get("item") {
                    match item.get("type").and_then(|v| v.as_str()) {
                        Some("message") => state.finish_message(Some(item)),
                        Some("function_call") => {
                            if let Some(parts) = emitted_parts
                                && let Some(part) = state.finish_tool_call(item)
                            {
                                parts.push(part);
                            } else {
                                let _ = state.finish_tool_call(item);
                            }
                        }
                        Some("reasoning") => state.finish_reasoning_part(),
                        _ => {}
                    }
                }
            }
            "response.completed" => {
                if let Some(resp_value) = event.get("response") {
                    let final_text = Self::extract_text(resp_value);
                    state.reconcile_final_response_text(&final_text);
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
            &state.deltas[prev_delta_len..],
            state.full_text.len(),
            &state.usage,
            had_final_response,
        );
        Ok(())
    }

    fn parse_sse_payload(
        payload: &str,
        state: &mut CodexStreamState,
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
                    Self::process_sse_event(&raw, state, None)?;
                    event_lines.clear();
                }
                continue;
            }
        }
        if !event_lines.is_empty() {
            let raw = event_lines.join("\n");
            Self::process_sse_event(&raw, state, None)?;
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
                model: "gpt-5.4-mini",
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
        let mut http = self
            .client
            .post(Self::CODEX_RESPONSES_URL)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("originator", Self::CODEX_ORIGINATOR)
            .header("User-Agent", Self::codex_user_agent())
            .json(&body);
        if let Some(session_id) = req.session_id.as_deref() {
            http = http
                .header("session_id", session_id)
                .header("x-client-request-id", session_id);
        }
        if let Some(id) = account_id {
            http = http.header("ChatGPT-Account-ID", id);
        }

        let resp = send_request(
            http,
            request_body.clone(),
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
                        request_body: request_body.clone().or(err.request_body),
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
                request_body,
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
                let mut state = CodexStreamState::default();
                Self::parse_sse_payload(&text, &mut state)?;
                let response = Self::response_from_stream_state(
                    state,
                    request_body,
                    format!("HTTP POST {} (stream/fallback)", Self::CODEX_RESPONSES_URL),
                );
                if let Some(tx) = &stream_events {
                    if response.usage != LlmUsage::default() {
                        tx.send(LlmStreamEvent::Usage(response.usage.clone()));
                    }
                    for piece in &response.deltas {
                        tx.send(LlmStreamEvent::Delta(piece.clone()));
                    }
                    for part in &response.parts {
                        match part {
                            LlmOutputPart::ToolCall { .. } => {
                                tx.send(LlmStreamEvent::Part(part.clone()));
                            }
                            LlmOutputPart::Reasoning { text } if !text.is_empty() => {
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
                provider_usage: None,
                request_body,
                http_summary: Some(format!("HTTP POST {}", Self::CODEX_RESPONSES_URL)),
            });
        }

        let mut state = CodexStreamState::default();
        drive_sse_response(
            resp,
            self.chunk_timeout,
            "Codex stream chunk timed out",
            |raw| {
                let prev_len = state.deltas.len();
                let prev_usage = state.usage.clone();
                let mut emitted_parts = Vec::new();
                Self::process_sse_event(&raw, &mut state, Some(&mut emitted_parts))?;
                emit_progress(
                    stream_events.as_ref(),
                    &state.deltas,
                    prev_len,
                    &state.usage,
                    &prev_usage,
                );
                if let Some(tx) = &stream_events {
                    for piece in state.take_reasoning_deltas() {
                        tx.send(LlmStreamEvent::ReasoningDelta(piece));
                    }
                    for part in emitted_parts {
                        tx.send(LlmStreamEvent::Part(part));
                    }
                }
                Ok(())
            },
        )
        .await?;

        Ok(Self::response_from_stream_state(
            state,
            request_body,
            format!("HTTP POST {} (stream)", Self::CODEX_RESPONSES_URL),
        ))
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
    fn parses_codex_sse_delta_and_completed_usage() {
        let mut state = CodexStreamState::default();

        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_text.delta","delta":"Hi "}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.completed","response":{"output_text":"Hi there","usage":{"input_tokens":30,"output_tokens":8,"input_tokens_details":{"cached_tokens":10}}}}"#,
            &mut state,
            None,
        )
        .unwrap();

        assert_eq!(state.full_text, "Hi there");
        assert_eq!(state.usage.input_tokens, 30);
        assert_eq!(state.usage.output_tokens, 8);
        assert_eq!(state.usage.cached_input_tokens, 10);
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

        let mut state = CodexStreamState::default();
        CodexOAuthAdapter::parse_sse_payload(payload, &mut state).unwrap();

        assert_eq!(state.full_text, "Hey there");
        assert_eq!(state.usage.input_tokens, 9);
        assert_eq!(state.usage.output_tokens, 2);
        assert_eq!(state.usage.cached_input_tokens, 3);
    }

    #[test]
    fn extracts_tool_calls_from_completed_stream_response() {
        let payload = r#"event: response.completed
data: {"type":"response.completed","response":{"output":[{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"README.md\"}"}],"usage":{"input_tokens":12,"output_tokens":3}}}
"#;

        let mut state = CodexStreamState::default();
        CodexOAuthAdapter::parse_sse_payload(payload, &mut state).unwrap();

        let parts = state.response_parts();
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
    fn extracts_tool_calls_from_stream_events_when_completed_response_is_empty() {
        let payload = r#"event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"exec_command"},"output_index":0}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","delta":"{\"cmd\":\"date","item_id":"fc_1","output_index":0}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","arguments":"{\"cmd\":\"date -u\"}","item_id":"fc_1","output_index":0}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\"cmd\":\"date -u\"}","call_id":"call_1","name":"exec_command"},"output_index":0}

event: response.completed
data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":12,"output_tokens":3}}}
"#;

        let mut state = CodexStreamState::default();
        CodexOAuthAdapter::parse_sse_payload(payload, &mut state).unwrap();

        assert_eq!(
            state.response_parts(),
            vec![LlmOutputPart::ToolCall {
                call_id: "call_1".to_string(),
                tool_name: "exec_command".to_string(),
                input_json: "{\"cmd\":\"date -u\"}".to_string(),
            }]
        );
        assert_eq!(state.usage.input_tokens, 12);
        assert_eq!(state.usage.output_tokens, 3);
    }

    #[test]
    fn output_text_done_does_not_duplicate_a_repeated_tail() {
        let mut state = CodexStreamState::default();

        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_item.added","item":{"id":"msg_1","type":"message","status":"in_progress","content":[]}}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_text.delta","delta":"I’ve got the wiring. "}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_text.delta","delta":"I’m doing one direct read pass."}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_text.done","text":"I’m doing one direct read pass."}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_item.done","item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","text":"I’ve got the wiring. I’m doing one direct read pass."}]}}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.completed","response":{"output_text":"I’ve got the wiring. I’m doing one direct read pass.","usage":{"input_tokens":12,"output_tokens":9}}}"#,
            &mut state,
            None,
        )
        .unwrap();

        assert_eq!(
            state.deltas,
            vec![
                "I’ve got the wiring. ".to_string(),
                "I’m doing one direct read pass.".to_string(),
            ]
        );
        assert_eq!(
            state.full_text,
            "I’ve got the wiring. I’m doing one direct read pass."
        );
    }

    #[test]
    fn consecutive_message_items_are_separated_with_paragraph_break() {
        let mut state = CodexStreamState::default();

        for event in [
            r#"{"type":"response.output_item.added","item":{"id":"msg_1","type":"message","status":"in_progress","content":[]}}"#,
            r#"{"type":"response.output_text.delta","delta":"I'm checking the repo."}"#,
            r#"{"type":"response.output_item.done","item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","text":"I'm checking the repo."}]}}"#,
            r#"{"type":"response.output_item.added","item":{"id":"msg_2","type":"message","status":"in_progress","content":[]}}"#,
            r#"{"type":"response.output_text.delta","delta":"Next I'm fetching remote state."}"#,
            r#"{"type":"response.output_item.done","item":{"id":"msg_2","type":"message","status":"completed","content":[{"type":"output_text","text":"Next I'm fetching remote state."}]}}"#,
            r#"{"type":"response.completed","response":{"output":[{"content":[{"type":"output_text","text":"I'm checking the repo."}]},{"content":[{"type":"output_text","text":"Next I'm fetching remote state."}]}],"usage":{"input_tokens":10,"output_tokens":12}}}"#,
        ] {
            CodexOAuthAdapter::process_sse_event(event, &mut state, None).unwrap();
        }

        assert_eq!(
            state.deltas,
            vec![
                "I'm checking the repo.".to_string(),
                "\n\n".to_string(),
                "Next I'm fetching remote state.".to_string(),
            ]
        );
        assert_eq!(
            state.full_text,
            "I'm checking the repo.\n\nNext I'm fetching remote state."
        );
    }

    #[test]
    fn completed_response_appends_only_missing_suffix_once() {
        let mut state = CodexStreamState::default();

        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_item.added","item":{"id":"msg_1","type":"message","status":"in_progress","content":[]}}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.output_text.delta","delta":"Hi "}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.completed","response":{"output_text":"Hi there","usage":{"input_tokens":30,"output_tokens":8}}}"#,
            &mut state,
            None,
        )
        .unwrap();

        assert_eq!(state.deltas, vec!["Hi ".to_string(), "there".to_string()]);
        assert_eq!(state.full_text, "Hi there");
    }

    #[test]
    fn structured_messages_build_responses_input_items() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
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
                message(LlmRole::User, "text", "new turn"),
            ],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let (instructions, input) = CodexOAuthAdapter::build_input(&req);

        assert_eq!(instructions, "sys");
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[3]["role"], "user");
        assert_eq!(input[3]["content"][0]["text"], "new turn");
    }

    #[test]
    fn structured_messages_preserve_empty_function_call_output() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                LlmMessage {
                    role: LlmRole::Assistant,
                    content: "{\"question\":\"Pick one\"}".to_string(),
                    kind: "tool_call".to_string(),
                    image_idx: -1,
                    tool_call_id: Some("call_ask".to_string()),
                    tool_name: Some("ask".to_string()),
                },
                LlmMessage {
                    role: LlmRole::User,
                    content: String::new(),
                    kind: "tool_result".to_string(),
                    image_idx: -1,
                    tool_call_id: Some("call_ask".to_string()),
                    tool_name: Some("ask".to_string()),
                },
                message(LlmRole::User, "text", "continue"),
            ],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let (_, input) = CodexOAuthAdapter::build_input(&req);

        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_ask");
        assert_eq!(input[1]["output"], "");
        assert_eq!(input[2]["role"], "user");
    }

    #[test]
    fn build_request_body_sets_prompt_cache_key_from_session_id() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "hello"),
            ],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: Some("sess-123".to_string()),
            output_spec: None,
            stream_events: None,
        };

        let body = CodexOAuthAdapter::build_request_body(&req, false);
        assert_eq!(body["prompt_cache_key"], "sess-123");
        assert_eq!(body["store"], false);
        assert_eq!(body["instructions"], "sys");
        assert_eq!(body["include"], json!([]));
    }

    #[test]
    fn build_request_body_uses_top_level_instructions_instead_of_system_input() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "system guidance"),
                message(LlmRole::User, "text", "hello"),
            ],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let body = CodexOAuthAdapter::build_request_body(&req, false);

        assert_eq!(body["instructions"], "system guidance");
        assert_eq!(body["input"].as_array().map(Vec::len), Some(1));
        assert_eq!(body["input"][0]["role"], "user");
        assert!(body["input"][0].get("type").is_none());
    }

    #[test]
    fn build_request_body_emits_codex_style_function_tools() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "hello"),
            ],
            attachments: vec![],
            tools: vec![crate::llm::types::LlmToolSpec {
                name: "find".to_string(),
                description: "Locate code".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }),
                output_schema: serde_json::Value::Null,
            }]
            .into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let body = CodexOAuthAdapter::build_request_body(&req, true);

        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "find");
        assert_eq!(body["tools"][0]["strict"], false);
        assert!(body["tools"][0].get("output_schema").is_none());
    }

    #[test]
    fn build_request_body_adds_text_format_for_structured_output() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "hello"),
            ],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: Some(crate::llm::types::LlmOutputSpec::JsonSchema(
                crate::llm::types::LlmJsonSchema {
                    name: "shape".to_string(),
                    schema: json!({"type": "object", "properties": {"name": {"type": "string"}}}),
                    strict: true,
                },
            )),
            stream_events: None,
        };

        let body = CodexOAuthAdapter::build_request_body(&req, false);

        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["name"], "shape");
        assert_eq!(body["text"]["format"]["strict"], true);
    }

    #[test]
    fn codex_transport_requires_streaming() {
        let adapter = CodexOAuthAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        assert!(LlmTransport::requires_streaming(&adapter));
    }

    #[test]
    fn user_text_messages_use_input_text_content_type() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![message(LlmRole::User, "text", "hello")],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let (_, input) = CodexOAuthAdapter::build_input(&req);
        let item = &input[0];

        assert_eq!(item["role"], "user");
        assert_eq!(item["content"][0]["type"], "input_text");
        assert_eq!(item["content"][0]["text"], "hello");
    }

    #[test]
    fn user_image_messages_encode_images_as_data_urls() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
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
                data: vec![0, 1, 2, 3],
            }],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let (_, input) = CodexOAuthAdapter::build_input(&req);
        let item = &input[0];

        assert_eq!(item["role"], "user");
        assert_eq!(item["content"][0]["type"], "input_image");
        assert_eq!(
            item["content"][0]["image_url"],
            "data:image/png;base64,AAECAw=="
        );
        assert!(item["content"][0].get("image_base64").is_none());
        assert!(item["content"][0].get("mime_type").is_none());
    }

    #[test]
    fn structured_image_messages_use_input_image_data_urls() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
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
                data: vec![0, 1, 2, 3],
            }],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let part = CodexOAuthAdapter::content_part_for_message(&req, &req.messages[0]);

        assert_eq!(part["type"], "input_image");
        assert_eq!(part["image_url"], "data:image/png;base64,AAECAw==");
        assert!(part.get("image_base64").is_none());
        assert!(part.get("mime_type").is_none());
    }

    #[test]
    fn reasoning_summary_events_produce_reasoning_parts_and_deltas() {
        let payload = r#"event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"rs_1","type":"reasoning","summary":[]}}

event: response.reasoning_summary_part.added
data: {"type":"response.reasoning_summary_part.added","item_id":"rs_1","part":{"type":"summary_text","text":""}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":"Checking the "}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","delta":"codebase."}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","text":"Checking the codebase."}

event: response.reasoning_summary_part.done
data: {"type":"response.reasoning_summary_part.done","item_id":"rs_1"}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"Checking the codebase."}]}}

event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","status":"in_progress","content":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Done."}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","status":"completed","content":[{"type":"output_text","text":"Done."}]}}

event: response.completed
data: {"type":"response.completed","response":{"output_text":"Done.","usage":{"input_tokens":12,"output_tokens":3}}}
"#;

        let mut state = CodexStreamState::default();
        CodexOAuthAdapter::parse_sse_payload(payload, &mut state).unwrap();

        // Reasoning deltas should have been accumulated on the
        // `reasoning_deltas` channel so the UI can render incrementally.
        assert_eq!(
            state.reasoning_deltas,
            vec!["Checking the ".to_string(), "codebase.".to_string()]
        );

        // The finalized response parts should carry a `Reasoning` entry
        // before the assistant text, exposing the trace to consumers that
        // rehydrate the turn after the stream ends.
        let parts = state.response_parts();
        assert_eq!(parts.len(), 2);
        assert_eq!(
            parts[0],
            LlmOutputPart::Reasoning {
                text: "Checking the codebase.".to_string(),
            }
        );
        assert_eq!(
            parts[1],
            LlmOutputPart::Text {
                text: "Done.".to_string(),
            }
        );

        // `full_text` must still report only the assistant's answer so
        // downstream text-centric code paths (usage logging, etc.) stay
        // unaffected by the new reasoning signal.
        assert_eq!(state.full_text, "Done.");
    }

    #[test]
    fn multiple_reasoning_summary_parts_become_separate_parts() {
        let mut state = CodexStreamState::default();

        for event in [
            r#"{"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":""}}"#,
            r#"{"type":"response.reasoning_summary_text.delta","delta":"First paragraph."}"#,
            r#"{"type":"response.reasoning_summary_part.done"}"#,
            r#"{"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":""}}"#,
            r#"{"type":"response.reasoning_summary_text.delta","delta":"Second paragraph."}"#,
            r#"{"type":"response.reasoning_summary_part.done"}"#,
        ] {
            CodexOAuthAdapter::process_sse_event(event, &mut state, None).unwrap();
        }

        let parts = state.response_parts();
        assert_eq!(
            parts,
            vec![
                LlmOutputPart::Reasoning {
                    text: "First paragraph.".to_string(),
                },
                LlmOutputPart::Reasoning {
                    text: "Second paragraph.".to_string(),
                }
            ]
        );
    }

    #[test]
    fn reasoning_summary_done_reconciles_missing_suffix() {
        let mut state = CodexStreamState::default();

        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":""}}"#,
            &mut state,
            None,
        )
        .unwrap();
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.reasoning_summary_text.delta","delta":"Looking"}"#,
            &mut state,
            None,
        )
        .unwrap();
        // Server sends the `done` event with the full text; the
        // accumulator must pick up the missing suffix without duplicating
        // the prefix that already arrived.
        CodexOAuthAdapter::process_sse_event(
            r#"{"type":"response.reasoning_summary_text.done","text":"Looking at it."}"#,
            &mut state,
            None,
        )
        .unwrap();

        assert_eq!(
            state.reasoning_deltas,
            vec!["Looking".to_string(), " at it.".to_string()]
        );
        let parts = state.response_parts();
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0],
            LlmOutputPart::Reasoning {
                text: "Looking at it.".to_string(),
            }
        );
    }
}
