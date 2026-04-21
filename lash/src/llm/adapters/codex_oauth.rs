use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::Duration;

use crate::llm::adapters::streaming::{drive_sse_response, emit_progress};
use crate::llm::timeouts::{
    LlmTimeouts, build_http_client, read_response_text, response_start_timeout, send_request,
};
use crate::llm::transport::{LlmTransport, LlmTransportError};
#[cfg(test)]
use crate::llm::types::LlmMessage;
use crate::llm::types::{
    LlmContentBlock, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse, LlmRole,
    LlmStreamEvent, LlmUsage, ModelSelection,
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
    /// Codex Responses API item-id (e.g. `fc_...`). Preserved so we can
    /// re-emit it on the next request body alongside `call_id`; Codex uses
    /// it to pair the function_call with its sibling reasoning item.
    item_id: String,
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
            signature: None,
            redacted: false,
            item_id: None,
            encrypted_content: None,
            summary: Vec::new(),
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
        if let Some(LlmOutputPart::Reasoning { text, .. }) = self.parts.get_mut(index) {
            text.push_str(delta);
        }
        self.reasoning_deltas.push(delta.to_string());
    }

    fn finish_reasoning_part(&mut self) {
        // Drop the cursor; the next `part.added` will open a fresh slot.
        // Trim trailing whitespace off the completed part so concatenated
        // paragraphs don't carry stray blanks.
        if let Some(index) = self.current_reasoning_part.take()
            && let Some(LlmOutputPart::Reasoning { text, .. }) = self.parts.get_mut(index)
        {
            let trimmed = text.trim_end();
            if trimmed.len() != text.len() {
                *text = trimmed.to_string();
            }
        }
    }

    /// Populate the most recent reasoning part with the authoritative
    /// payload from `response.output_item.done`: the Codex `rs_...` id,
    /// the `summary[*].text` entries, and the `encrypted_content` blob
    /// that must be replayed on the next turn.
    fn finalize_reasoning_item(&mut self, item: &Value) {
        // Find the nearest Reasoning part without an id yet; server emits
        // items in order so the latest slot is the right one to populate.
        let Some((_, part)) = self
            .parts
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(_, p)| matches!(p, LlmOutputPart::Reasoning { .. }))
        else {
            return;
        };
        let LlmOutputPart::Reasoning {
            item_id,
            encrypted_content,
            summary,
            ..
        } = part
        else {
            return;
        };
        if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
            *item_id = Some(id.to_string());
        }
        if let Some(blob) = item.get("encrypted_content").and_then(|v| v.as_str()) {
            *encrypted_content = Some(blob.to_string());
        }
        if let Some(arr) = item.get("summary").and_then(|v| v.as_array()) {
            let texts: Vec<String> = arr
                .iter()
                .filter_map(|entry| entry.get("text").and_then(|v| v.as_str()).map(String::from))
                .collect();
            if !texts.is_empty() {
                *summary = texts;
            }
        }
    }

    fn take_reasoning_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reasoning_deltas)
    }

    fn update_tool_call_from_item(&mut self, item: &Value) -> Option<String> {
        let item_id = item.get("id").and_then(|v| v.as_str())?.to_string();
        let tool_call = self.tool_calls.entry(item_id.clone()).or_default();
        if tool_call.item_id.is_empty() {
            tool_call.item_id = item_id.clone();
        }
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
            item_id: if tool_call.item_id.is_empty() {
                None
            } else {
                Some(tool_call.item_id)
            },
            signature: None,
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
                LlmOutputPart::Reasoning { text, .. } if text.trim().is_empty() => None,
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
    /// Maximum number of submission attempts (1 initial + up to 3 retries).
    /// Mirrors pi's `MAX_RETRIES = 3` loop, which iterates `attempt <= MAX_RETRIES`.
    const MAX_ATTEMPTS: u32 = 4;
    /// Base delay for exponential backoff between retries (ms).
    /// Matches pi-mono `BASE_DELAY_MS` at `openai-codex-responses.ts:47`.
    const BASE_DELAY_MS: u64 = 1000;

    pub fn new(timeouts: LlmTimeouts) -> Self {
        Self {
            client: build_http_client(),
            request_timeout: timeouts.request_timeout,
            chunk_timeout: timeouts.chunk_timeout,
        }
    }

    /// Decide whether an HTTP failure should be retried.
    ///
    /// Mirrors the rules in pi-mono (`openai-codex-responses.ts:94-99`, `:229`):
    /// - 429 is retryable *except* when the body reports `"type": "usage_limit_reached"`
    ///   (the user has exhausted their quota and retrying won't help).
    /// - 5xx (500-599) is retryable.
    /// - All other statuses (2xx/3xx, and 4xx other than 429) are not retryable.
    /// - Once `attempt_number` reaches `MAX_ATTEMPTS - 1` (the final attempt), no more retries.
    fn should_retry(status: u16, body_text: &str, attempt_number: u32) -> bool {
        if attempt_number + 1 >= Self::MAX_ATTEMPTS {
            return false;
        }
        if status == 429 {
            return !Self::is_usage_limit_error(body_text);
        }
        (500..600).contains(&status)
    }

    /// Detect pi's "usage limit reached" marker in an error body.
    /// Pi matches the literal string `"type":"usage_limit_reached"` (see
    /// `openai-codex-responses.ts:891`, which tests `/usage_limit_reached/`).
    fn is_usage_limit_error(body_text: &str) -> bool {
        // JSON serialisers don't insert whitespace between `"type":` and the value,
        // but be lenient across a single space just in case.
        body_text.contains("\"type\":\"usage_limit_reached\"")
            || body_text.contains("\"type\": \"usage_limit_reached\"")
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

    /// Exponential backoff for retry attempt `attempt`.
    /// attempt=0 -> 1000ms, attempt=1 -> 2000ms, attempt=2 -> 4000ms (mirrors pi's
    /// `BASE_DELAY_MS * 2 ** attempt`).
    fn backoff_delay(attempt: u32) -> Duration {
        Duration::from_millis(Self::BASE_DELAY_MS.saturating_mul(1u64 << attempt.min(16)))
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

    fn build_input(req: &LlmRequest) -> (String, Vec<Value>) {
        let mut input = Vec::new();
        let mut instructions: Vec<String> = Vec::new();

        for msg in &req.messages {
            // System-role messages are hoisted into the Codex Responses
            // `instructions` top-level field rather than appearing in the
            // input array.
            if matches!(msg.role, LlmRole::System) {
                for block in &msg.blocks {
                    if let LlmContentBlock::Text(text) = block
                        && !text.is_empty()
                    {
                        instructions.push(text.clone());
                    }
                }
                continue;
            }

            // Per-message input items are emitted in block order. Text and
            // image blocks accumulate into a single role-typed content
            // message that gets flushed before the next tool/reasoning
            // item so the wire ordering matches pi's shape.
            let role_str = Self::role_name(&msg.role);
            let mut pending_content: Vec<Value> = Vec::new();

            // Pre-compute which blocks have already been folded into an
            // earlier ToolResult's `output` so we don't double-emit them.
            let mut consumed_after_tool_result: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for (idx, block) in msg.blocks.iter().enumerate() {
                let LlmContentBlock::ToolResult { .. } = block else {
                    continue;
                };
                for (j, sibling) in msg.blocks.iter().enumerate().skip(idx + 1) {
                    match sibling {
                        LlmContentBlock::Image { .. } => {
                            consumed_after_tool_result.insert(j);
                        }
                        LlmContentBlock::Text(t) if t.starts_with("[Tool image:") => {
                            consumed_after_tool_result.insert(j);
                        }
                        _ => break,
                    }
                }
            }

            for (block_idx, block) in msg.blocks.iter().enumerate() {
                if consumed_after_tool_result.contains(&block_idx) {
                    continue;
                }
                match block {
                    LlmContentBlock::Text(text) => {
                        if text.is_empty() {
                            continue;
                        }
                        let part_type = match msg.role {
                            LlmRole::Assistant => "output_text",
                            _ => "input_text",
                        };
                        pending_content.push(json!({
                            "type": part_type,
                            "text": text,
                        }));
                    }
                    LlmContentBlock::Image { attachment_idx } => {
                        let Some(att) = req.attachments.get(*attachment_idx) else {
                            continue;
                        };
                        if matches!(msg.role, LlmRole::User) {
                            pending_content.push(Self::input_image_part(att));
                        }
                    }
                    LlmContentBlock::Reasoning {
                        text,
                        encrypted_content,
                        signature,
                        item_id,
                        summary,
                        redacted: _,
                    } => {
                        Self::flush_pending_content(
                            &mut pending_content,
                            &mut input,
                            role_str,
                            matches!(msg.role, LlmRole::User),
                        );
                        let payload = encrypted_content.as_deref().or(signature.as_deref());
                        // Only replay reasoning items that actually carry a
                        // Codex-style encrypted blob. Display-only summaries
                        // (no blob) must not be fed back — the server will
                        // either ignore them or reject the turn.
                        let Some(blob) = payload else {
                            continue;
                        };
                        let summary_items: Vec<Value> = if summary.is_empty() {
                            if text.is_empty() {
                                Vec::new()
                            } else {
                                vec![json!({"type": "summary_text", "text": text})]
                            }
                        } else {
                            summary
                                .iter()
                                .map(|entry| json!({"type": "summary_text", "text": entry}))
                                .collect()
                        };
                        let mut item = json!({
                            "type": "reasoning",
                            "summary": summary_items,
                            "encrypted_content": blob,
                        });
                        if let Some(id) = item_id
                            && !id.is_empty()
                        {
                            item["id"] = json!(id);
                        }
                        input.push(item);
                    }
                    LlmContentBlock::ToolCall {
                        call_id,
                        tool_name,
                        input_json,
                        item_id,
                        signature: _,
                    } => {
                        Self::flush_pending_content(
                            &mut pending_content,
                            &mut input,
                            role_str,
                            matches!(msg.role, LlmRole::User),
                        );
                        let mut item = json!({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tool_name,
                            "arguments": input_json,
                        });
                        // Codex uses `id` (e.g. `fc_...`) to pair
                        // function_call items with their sibling reasoning
                        // items across turns. Omit when absent so we don't
                        // send a bogus id.
                        if let Some(id) = item_id {
                            item["id"] = json!(id);
                        }
                        input.push(item);
                    }
                    LlmContentBlock::ToolResult {
                        call_id, content, ..
                    } => {
                        Self::flush_pending_content(
                            &mut pending_content,
                            &mut input,
                            role_str,
                            matches!(msg.role, LlmRole::User),
                        );
                        // Look ahead in THIS message for sibling image
                        // blocks (plus optional placeholder text) so the
                        // image rides in the tool_result's `output` array.
                        // Matches pi's tool-result-with-image shape.
                        let mut image_parts: Vec<Value> = Vec::new();
                        for sibling in msg
                            .blocks
                            .iter()
                            .skip_while(|b| {
                                !matches!(
                                    b,
                                    LlmContentBlock::ToolResult { call_id: id, .. }
                                        if id == call_id
                                )
                            })
                            .skip(1)
                        {
                            match sibling {
                                LlmContentBlock::Image { attachment_idx } => {
                                    if let Some(att) = req.attachments.get(*attachment_idx) {
                                        image_parts.push(Self::input_image_part(att));
                                    }
                                }
                                LlmContentBlock::Text(t) if t.starts_with("[Tool image:") => {
                                    // placeholder — consume silently
                                }
                                _ => break,
                            }
                        }

                        if image_parts.is_empty() {
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": content,
                            }));
                        } else {
                            let mut parts: Vec<Value> = Vec::new();
                            if !content.is_empty() {
                                parts.push(json!({
                                    "type": "input_text",
                                    "text": content,
                                }));
                            }
                            parts.extend(image_parts);
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": parts,
                            }));
                        }
                    }
                }
            }
            Self::flush_pending_content(
                &mut pending_content,
                &mut input,
                role_str,
                matches!(msg.role, LlmRole::User),
            );

            // Fold any user-role Image blocks that shared a message with
            // tool_result blocks into the preceding function_call_output's
            // `output` as a structured {input_text, input_image} array —
            // matches pi's "tool_result images" shape.
            if matches!(msg.role, LlmRole::User) {
                Self::fold_tool_result_images(&mut input);
            }
        }
        (instructions.join("\n\n"), input)
    }

    fn flush_pending_content(
        pending: &mut Vec<Value>,
        input: &mut Vec<Value>,
        role_str: &'static str,
        is_user: bool,
    ) {
        if pending.is_empty() {
            return;
        }
        let content = std::mem::take(pending);
        if is_user
            && let Some(prev) = input.last_mut()
            && prev.get("role").and_then(|r| r.as_str()) == Some("user")
            && prev.get("content").is_some_and(|c| c.is_array())
        {
            prev["content"].as_array_mut().unwrap().extend(content);
        } else {
            input.push(json!({
                "role": role_str,
                "content": content,
            }));
        }
    }

    /// Walk backwards from the last input item: if the final entry is a
    /// user `content` message whose parts are all `input_image`, and the
    /// entry before it is a `function_call_output`, promote the image
    /// parts into the `output` of that function_call_output so the Codex
    /// server sees the image as the tool's result rather than as a
    /// standalone user turn.
    fn fold_tool_result_images(input: &mut Vec<Value>) {
        if input.len() < 2 {
            return;
        }
        let last_idx = input.len() - 1;
        let is_user_image_msg = input[last_idx].get("role").and_then(|v| v.as_str())
            == Some("user")
            && input[last_idx]
                .get("content")
                .and_then(|c| c.as_array())
                .is_some_and(|parts| {
                    parts
                        .iter()
                        .all(|p| p.get("type").and_then(|t| t.as_str()) == Some("input_image"))
                });
        if !is_user_image_msg {
            return;
        }
        let prev_is_call_output = input[last_idx - 1].get("type").and_then(|v| v.as_str())
            == Some("function_call_output");
        if !prev_is_call_output {
            return;
        }
        let last = input.remove(last_idx);
        let image_parts = last
            .get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        let prev = input.last_mut().expect("function_call_output present");
        if !prev["output"].is_array() {
            let existing_text = prev["output"]
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_default();
            let mut parts: Vec<Value> = Vec::new();
            if !existing_text.is_empty() {
                parts.push(json!({
                    "type": "input_text",
                    "text": existing_text,
                }));
            }
            prev["output"] = Value::Array(parts);
        }
        prev["output"].as_array_mut().unwrap().extend(image_parts);
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

    /// Clamp an incoming reasoning effort string to a value the Codex
    /// Responses API will accept for the given model. Mirrors the
    /// `clampReasoningEffort` helper in pi-mono's
    /// `openai-codex-responses.ts` so that invalid combinations like
    /// `minimal` on gpt-5.4 or `xhigh` on gpt-5.1-codex-mini don't 4xx.
    fn clamp_reasoning_effort(model: &str, effort: &str) -> String {
        // Strip any provider prefix (e.g. "openai/gpt-5.4").
        let id = match model.rsplit_once('/') {
            Some((_, tail)) => tail,
            None => model,
        };

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

    fn build_request_body(req: &LlmRequest, stream: bool) -> Value {
        let tools = Self::build_tools(req);
        let (instructions, input) = Self::build_input(req);
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
            body["tool_choice"] = json!(Self::tool_choice_value(&req.tool_choice));
        }
        if let Some(effort) = req.model_variant.as_deref() {
            body["reasoning"] = json!({
                "effort": Self::clamp_reasoning_effort(&req.model, effort),
                "summary": "auto",
            });
        }
        if let Some(session_id) = req.session_id.as_deref() {
            body["prompt_cache_key"] = json!(session_id);
        }
        if let Some(output_spec) = &req.output_spec {
            body["text"]["format"] = match output_spec {
                LlmOutputSpec::JsonObject => json!({ "type": "json_object" }),
                LlmOutputSpec::JsonSchema(schema) => json!({
                    "type": "json_schema",
                    "name": schema.name,
                    "schema": schema.schema,
                    "strict": schema.strict,
                }),
            };
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
                    && let Some(LlmOutputPart::Reasoning { text: existing, .. }) =
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
                        Some("reasoning") => {
                            state.finish_reasoning_part();
                            state.finalize_reasoning_item(item);
                        }
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
                            item_id: item.get("id").and_then(|v| v.as_str()).map(str::to_string),
                            signature: None,
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

        // Build the HTTP request fresh on each attempt — reqwest's RequestBuilder
        // doesn't universally implement Clone once `.json()` has been called, and
        // rebuilding keeps each attempt self-contained.
        let build_request = || {
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
            http
        };

        // Retry loop: short-circuit locally on 429/5xx and transient network
        // errors with exponential backoff. Mirrors pi-mono
        // `openai-codex-responses.ts:206-256`. The adapter still sets
        // `retryable: true` on the final error so upstream retry infrastructure
        // keeps working for non-Codex providers; this loop just cuts down
        // round-trips the user has to endure.
        let mut attempt: u32 = 0;
        let (resp, status, content_type) = loop {
            let http = build_request();
            let send_result = send_request(
                http,
                request_body.clone(),
                response_start_timeout(
                    self.request_timeout,
                    self.chunk_timeout,
                    stream_events.is_some(),
                ),
                "Codex response start timed out",
            )
            .await;

            match send_result {
                Ok(resp) => {
                    let status = resp.status();
                    let content_type = resp
                        .headers()
                        .get(reqwest::header::CONTENT_TYPE)
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_string);
                    if status.is_success() {
                        break (resp, status, content_type);
                    }

                    // Non-success: read body text, decide whether to retry.
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

                    if Self::should_retry(status.as_u16(), &text, attempt) {
                        let delay = Self::backoff_delay(attempt);
                        tracing::debug!(
                            target: "lash::llm::codex_oauth",
                            status = status.as_u16(),
                            attempt,
                            delay_ms = delay.as_millis() as u64,
                            "Codex request returned retryable status; sleeping before retry"
                        );
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue;
                    }

                    let friendly = Self::codex_error_summary(status.as_u16(), &text);
                    let message = friendly.unwrap_or_else(|| {
                        format!(
                            "Codex request failed with {}{}",
                            status.as_u16(),
                            content_type
                                .as_deref()
                                .map(|ct| format!(" ({ct})"))
                                .unwrap_or_default()
                        )
                    });
                    return Err(LlmTransportError {
                        message,
                        retryable: status.as_u16() == 429 || status.as_u16() >= 500,
                        raw: Some(text),
                        code: Some(status.as_u16().to_string()),
                        request_body: request_body.clone(),
                    });
                }
                Err(err) => {
                    // Transient network / body / decode errors: pi retries
                    // these the same way it retries HTTP 5xx. Respect the
                    // flag set by `send_request` (timeouts + reqwest
                    // is_connect / is_body / is_decode).
                    if err.retryable && attempt + 1 < Self::MAX_ATTEMPTS {
                        let delay = Self::backoff_delay(attempt);
                        tracing::debug!(
                            target: "lash::llm::codex_oauth",
                            attempt,
                            delay_ms = delay.as_millis() as u64,
                            err = %err.message,
                            "Codex request failed with transient network error; sleeping before retry"
                        );
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue;
                    }
                    return Err(err);
                }
            }
        };

        let is_sse = content_type
            .as_deref()
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);
        let parse_stream =
            Self::should_parse_stream(stream_events.is_some(), content_type.as_deref());

        if !parse_stream {
            let text =
                read_response_text(resp, self.request_timeout, "Codex response body timed out")
                    .await
                    .map_err(|err| {
                        Self::non_sse_body_read_error(status.as_u16(), content_type.as_deref(), err)
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
                            LlmOutputPart::Reasoning { text, .. } if !text.is_empty() => {
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

        if stream_events.is_some() && !is_sse {
            tracing::debug!(
                target: "lash::llm::codex_oauth",
                status = status.as_u16(),
                content_type = content_type.as_deref().unwrap_or("<missing>"),
                "Codex streaming response did not advertise SSE; parsing as stream because stream=true was requested"
            );
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

        if state.final_response.is_none() && state.parts.is_empty() && state.deltas.is_empty() {
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
        match kind {
            "text" => LlmMessage::text(role, content),
            "tool_result" => LlmMessage::new(
                role,
                vec![LlmContentBlock::ToolResult {
                    call_id: String::new(),
                    content: content.to_string(),
                    tool_name: None,
                }],
            ),
            "image" => LlmMessage::new(role, vec![LlmContentBlock::Image { attachment_idx: 0 }]),
            other => panic!("unknown test message kind: {other}"),
        }
    }

    /// Assistant turn that combines any pre-text with a tool_call block.
    fn assistant_tool_call(
        text: Option<&str>,
        call_id: &str,
        tool_name: &str,
        args: &str,
        item_id: Option<&str>,
    ) -> LlmMessage {
        let mut blocks: Vec<LlmContentBlock> = Vec::new();
        if let Some(text) = text
            && !text.is_empty()
        {
            blocks.push(LlmContentBlock::Text(text.to_string()));
        }
        blocks.push(LlmContentBlock::ToolCall {
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
            input_json: args.to_string(),
            item_id: item_id.map(str::to_string),
            signature: None,
        });
        LlmMessage::new(LlmRole::Assistant, blocks)
    }

    fn user_tool_result(call_id: &str, content: &str) -> LlmMessage {
        LlmMessage::new(
            LlmRole::User,
            vec![LlmContentBlock::ToolResult {
                call_id: call_id.to_string(),
                content: content.to_string(),
                tool_name: None,
            }],
        )
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
                item_id: None,
                signature: None,
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
                item_id: Some("fc_1".to_string()),
                signature: None,
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
                assistant_tool_call(None, "call_1", "read_file", r#"{"path":"README.md"}"#, None),
                user_tool_result("call_1", "ok"),
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
    fn build_input_emits_function_call_id_alongside_call_id() {
        // Simulates a replay where Codex previously returned a function_call
        // with both `call_id` ("call_1") and item-id ("fc_abc"). The adapter
        // must re-emit the item-id on the next turn so Codex can pair it with
        // its sibling reasoning item.
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "question"),
                assistant_tool_call(
                    None,
                    "call_1",
                    "read_file",
                    r#"{"path":"README.md"}"#,
                    Some("fc_abc"),
                ),
                user_tool_result("call_1", "ok"),
                message(LlmRole::User, "text", "next"),
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

        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["id"], "fc_abc");
    }

    #[test]
    fn build_input_omits_function_call_id_when_not_captured() {
        // Replay from a provider that didn't surface an item-id: the adapter
        // must not synthesize one or leave a bogus value on the request body.
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                message(LlmRole::System, "text", "sys"),
                message(LlmRole::User, "text", "question"),
                assistant_tool_call(None, "call_x", "noop", "{}", None),
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
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_x");
        assert!(input[1].get("id").is_none());
    }

    #[test]
    fn sse_stream_captures_function_call_item_id() {
        // The SSE parser should pull `fc_...` off the function_call item and
        // surface it on the resulting `LlmOutputPart::ToolCall::id`.
        let payload = r#"event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"fc_zzz","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"exec_command"},"output_index":0}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","arguments":"{\"cmd\":\"date\"}","item_id":"fc_zzz","output_index":0}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"id":"fc_zzz","type":"function_call","status":"completed","arguments":"{\"cmd\":\"date\"}","call_id":"call_1","name":"exec_command"},"output_index":0}

event: response.completed
data: {"type":"response.completed","response":{"output":[],"usage":{"input_tokens":1,"output_tokens":1}}}
"#;

        let mut state = CodexStreamState::default();
        CodexOAuthAdapter::parse_sse_payload(payload, &mut state).unwrap();

        assert_eq!(
            state.response_parts(),
            vec![LlmOutputPart::ToolCall {
                call_id: "call_1".to_string(),
                tool_name: "exec_command".to_string(),
                input_json: "{\"cmd\":\"date\"}".to_string(),
                item_id: Some("fc_zzz".to_string()),
                signature: None,
            }]
        );
    }

    #[test]
    fn structured_messages_preserve_empty_function_call_output() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                assistant_tool_call(None, "call_ask", "ask", r#"{"question":"Pick one"}"#, None),
                user_tool_result("call_ask", ""),
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
    fn tool_result_text_only_keeps_bare_string_output() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                assistant_tool_call(None, "call_1", "read_file", r#"{"path":"README.md"}"#, None),
                user_tool_result("call_1", "file contents"),
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

        assert_eq!(input.len(), 2);
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_1");
        // Text-only results stay a bare string (common case Codex accepts).
        assert_eq!(input[1]["output"], "file contents");
        assert!(input[1]["output"].is_string());
    }

    #[test]
    fn tool_result_with_image_emits_structured_output_array() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![
                assistant_tool_call(None, "call_img", "get_circle", "{}", None),
                // Session layer packs the tool_result, placeholder text,
                // and image blocks into a single user LlmMessage — the
                // adapter folds the image into the tool_result's output.
                LlmMessage::new(
                    LlmRole::User,
                    vec![
                        LlmContentBlock::ToolResult {
                            call_id: "call_img".to_string(),
                            content: "A red circle.".to_string(),
                            tool_name: None,
                        },
                        LlmContentBlock::Text("[Tool image: circle-0]".to_string()),
                        LlmContentBlock::Image { attachment_idx: 0 },
                    ],
                ),
                message(LlmRole::User, "text", "what color is it?"),
            ],
            attachments: vec![crate::llm::types::LlmAttachment {
                mime: "image/png".to_string(),
                data: vec![0x89, 0x50, 0x4E, 0x47],
            }],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::Auto,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let (_, input) = CodexOAuthAdapter::build_input(&req);

        // function_call + function_call_output(with image folded in) +
        // trailing user question. No stray trailing image/user placeholder
        // items should be emitted.
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_img");

        let output = &input[1]["output"];
        assert!(
            output.is_array(),
            "expected structured output array when tool result carries an image"
        );
        let items = output.as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "input_text");
        assert_eq!(items[0]["text"], "A red circle.");
        assert_eq!(items[1]["type"], "input_image");
        assert_eq!(items[1]["image_url"], "data:image/png;base64,iVBORw==");

        // Trailing user question should still be present as its own user
        // message, and must not contain the image.
        assert_eq!(input[2]["role"], "user");
        assert_eq!(input[2]["content"][0]["text"], "what color is it?");
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
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["text"]["verbosity"], "medium");
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
        assert_eq!(body["text"]["verbosity"], "medium");
    }

    #[test]
    fn codex_transport_requires_streaming() {
        let adapter = CodexOAuthAdapter::new(crate::llm::timeouts::LlmTimeouts::default());
        assert!(LlmTransport::requires_streaming(&adapter));
    }

    #[test]
    fn codex_streaming_requests_parse_stream_even_without_sse_content_type() {
        assert!(CodexOAuthAdapter::should_parse_stream(true, None));
        assert!(CodexOAuthAdapter::should_parse_stream(
            true,
            Some("application/octet-stream")
        ));
        assert!(CodexOAuthAdapter::should_parse_stream(
            false,
            Some("text/event-stream; charset=utf-8")
        ));
        assert!(!CodexOAuthAdapter::should_parse_stream(
            false,
            Some("application/json")
        ));
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
            messages: vec![LlmMessage::new(
                LlmRole::User,
                vec![LlmContentBlock::Image { attachment_idx: 0 }],
            )],
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
            messages: vec![LlmMessage::new(
                LlmRole::User,
                vec![LlmContentBlock::Image { attachment_idx: 0 }],
            )],
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
        let part = &input[0]["content"][0];

        assert_eq!(part["type"], "input_image");
        assert_eq!(part["image_url"], "data:image/png;base64,AAECAw==");
        assert!(part.get("image_base64").is_none());
        assert!(part.get("mime_type").is_none());
    }

    #[test]
    fn clamp_reasoning_effort_promotes_minimal_on_gpt_5_2_3_4() {
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.2", "minimal"),
            "low"
        );
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.3", "minimal"),
            "low"
        );
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.4", "minimal"),
            "low"
        );
        // Provider-prefixed ids are normalised before matching.
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("openai/gpt-5.4", "minimal"),
            "low"
        );
        // Non-minimal efforts pass through unchanged on these models.
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.4", "high"),
            "high"
        );
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
                signature: None,
                redacted: false,
                item_id: Some("rs_1".to_string()),
                encrypted_content: None,
                summary: vec!["Checking the codebase.".to_string()],
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
                    signature: None,
                    redacted: false,
                    item_id: None,
                    encrypted_content: None,
                    summary: Vec::new(),
                },
                LlmOutputPart::Reasoning {
                    text: "Second paragraph.".to_string(),
                    signature: None,
                    redacted: false,
                    item_id: None,
                    encrypted_content: None,
                    summary: Vec::new(),
                }
            ]
        );
    }

    #[test]
    fn clamp_reasoning_effort_downgrades_xhigh_on_gpt_5_1() {
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.1", "xhigh"),
            "high"
        );
        // Other efforts pass through.
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.1", "minimal"),
            "minimal"
        );
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.1", "medium"),
            "medium"
        );
    }

    #[test]
    fn clamp_reasoning_effort_codex_mini_only_high_or_medium() {
        // xhigh is capped at high.
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.1-codex-mini", "xhigh"),
            "high"
        );
        // high stays high.
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.1-codex-mini", "high"),
            "high"
        );
        // Everything else collapses to medium (codex-mini only supports
        // medium/high per pi-mono's clampReasoningEffort).
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.1-codex-mini", "minimal"),
            "medium"
        );
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.1-codex-mini", "low"),
            "medium"
        );
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.1-codex-mini", "medium"),
            "medium"
        );
    }

    #[test]
    fn clamp_reasoning_effort_passthrough_for_other_models() {
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("gpt-5.0", "minimal"),
            "minimal"
        );
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("o4-mini", "high"),
            "high"
        );
        assert_eq!(
            CodexOAuthAdapter::clamp_reasoning_effort("some-future-model", "xhigh"),
            "xhigh"
        );
    }

    #[test]
    fn reasoning_object_includes_summary_auto_and_clamped_effort() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![message(LlmRole::User, "text", "hi")],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: Some("minimal".to_string()),
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let body = CodexOAuthAdapter::build_request_body(&req, false);
        let reasoning = body.get("reasoning").expect("reasoning present");
        // minimal on gpt-5.4 clamps to low.
        assert_eq!(reasoning["effort"], "low");
        // summary: "auto" is always included.
        assert_eq!(reasoning["summary"], "auto");
    }

    #[test]
    fn reasoning_omitted_when_no_variant_provided() {
        let req = LlmRequest {
            model: "gpt-5.4".to_string(),
            messages: vec![message(LlmRole::User, "text", "hi")],
            attachments: vec![],
            tools: vec![].into(),
            tool_choice: crate::llm::types::LlmToolChoice::None,
            model_variant: None,
            session_id: None,
            output_spec: None,
            stream_events: None,
        };

        let body = CodexOAuthAdapter::build_request_body(&req, false);
        assert!(body.get("reasoning").is_none());
    }

    // ------------------------------------------------------------------
    // Retry / backoff (fix for audit finding 1.5)
    // ------------------------------------------------------------------

    #[test]
    fn should_retry_skips_429_usage_limit_reached() {
        // Pi explicitly does not retry usage_limit_reached errors — the user
        // has no quota, so retrying burns latency for a guaranteed failure.
        let body = r#"{"error":{"type":"usage_limit_reached","message":"You've hit your ChatGPT usage limit","plan_type":"plus"}}"#;
        assert!(!CodexOAuthAdapter::should_retry(429, body, 0));
        // Still false on subsequent attempts, just in case the loop somehow
        // gets there.
        assert!(!CodexOAuthAdapter::should_retry(429, body, 1));
    }

    #[test]
    fn should_retry_retries_normal_429() {
        let body = r#"{"error":{"type":"rate_limit_exceeded","message":"Too many requests"}}"#;
        assert!(CodexOAuthAdapter::should_retry(429, body, 0));
        assert!(CodexOAuthAdapter::should_retry(429, body, 1));
        assert!(CodexOAuthAdapter::should_retry(429, body, 2));
        // The final attempt must not retry (we've used all attempts).
        assert!(!CodexOAuthAdapter::should_retry(
            429,
            body,
            CodexOAuthAdapter::MAX_ATTEMPTS - 1
        ));
    }

    #[test]
    fn should_retry_retries_5xx() {
        assert!(CodexOAuthAdapter::should_retry(500, "internal error", 0));
        assert!(CodexOAuthAdapter::should_retry(502, "bad gateway", 0));
        assert!(CodexOAuthAdapter::should_retry(
            503,
            "service unavailable",
            1
        ));
        assert!(CodexOAuthAdapter::should_retry(504, "gateway timeout", 0));
        // 599 is still 5xx.
        assert!(CodexOAuthAdapter::should_retry(599, "", 0));
        // 600 is not 5xx.
        assert!(!CodexOAuthAdapter::should_retry(600, "", 0));
    }

    #[test]
    fn should_retry_does_not_retry_4xx_other_than_429() {
        assert!(!CodexOAuthAdapter::should_retry(400, "bad request", 0));
        assert!(!CodexOAuthAdapter::should_retry(401, "unauthorized", 0));
        assert!(!CodexOAuthAdapter::should_retry(403, "forbidden", 0));
        assert!(!CodexOAuthAdapter::should_retry(404, "not found", 0));
        assert!(!CodexOAuthAdapter::should_retry(418, "I'm a teapot", 0,));
    }

    #[test]
    fn backoff_delay_matches_pi_exponential_schedule() {
        // Pi's schedule is `BASE_DELAY_MS * 2 ** attempt`:
        //   attempt 0 -> 1000ms
        //   attempt 1 -> 2000ms
        //   attempt 2 -> 4000ms
        assert_eq!(
            CodexOAuthAdapter::backoff_delay(0),
            Duration::from_millis(1000)
        );
        assert_eq!(
            CodexOAuthAdapter::backoff_delay(1),
            Duration::from_millis(2000)
        );
        assert_eq!(
            CodexOAuthAdapter::backoff_delay(2),
            Duration::from_millis(4000)
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
                signature: None,
                redacted: false,
                item_id: None,
                encrypted_content: None,
                summary: Vec::new(),
            }
        );
    }

    #[test]
    fn codex_error_summary_renders_usage_limit_with_plan_and_reset() {
        // `resets_at` 2 minutes in the future → rounded up to "~2 min".
        let future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 150;
        let body = format!(
            r#"{{"error":{{"code":"usage_limit_reached","plan_type":"Plus","resets_at":{future},"message":"ignored"}}}}"#,
        );
        let summary = CodexOAuthAdapter::codex_error_summary(429, &body).expect("summary");
        assert!(summary.starts_with("You have hit your ChatGPT usage limit (plus plan)."));
        assert!(summary.contains("Try again in ~"));
        assert!(summary.contains(" min."));
    }

    #[test]
    fn codex_error_summary_falls_back_to_error_message_on_other_errors() {
        let body = r#"{"error":{"code":"invalid_request_error","message":"Bad thing"}}"#;
        let summary = CodexOAuthAdapter::codex_error_summary(400, body).expect("summary");
        assert_eq!(summary, "Codex request failed with 400: Bad thing");
    }

    #[test]
    fn codex_error_summary_returns_none_for_unparseable_body() {
        assert!(CodexOAuthAdapter::codex_error_summary(500, "oops").is_none());
    }
}
