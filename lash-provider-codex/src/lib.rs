use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::Duration;

use lash::llm::streaming::{drive_sse_response, emit_progress};
use lash::llm::timeouts::{
    build_http_client, read_response_text, request_body_snapshot, response_start_timeout,
    send_request,
};
use lash::llm::transport::LlmTransportError;
use lash::llm::types::{
    LlmAttachment, LlmContentBlock, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse, LlmRole,
    LlmStreamEvent, LlmToolChoice, LlmUsage,
};
use lash::provider::{
    AgentModelSelection, Provider, ProviderFactory, ProviderOptions, VariantRequestConfig,
};

pub mod oauth;

const OPENAI_GPT5_VARIANTS: &[&str] = &["minimal", "low", "medium", "high"];
const OPENAI_GPT5_XHIGH_VARIANTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh"];
const OPENAI_GPT55_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];
const CODEX_VARIANTS: &[&str] = &["low", "medium", "high"];
const CODEX_XHIGH_VARIANTS: &[&str] = &["low", "medium", "high", "xhigh"];

fn has_xhigh_suffix(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("5.2") || lower.contains("5.3") || lower.contains("5.4")
}

fn model_id(model: &str) -> &str {
    match model.rsplit_once('/') {
        Some((_, tail)) => tail,
        None => model,
    }
}

/// OpenAI Codex OAuth provider (ChatGPT Plus/Pro/Team via device-code flow).
#[derive(Clone, Debug)]
pub struct CodexProvider {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
    pub account_id: Option<String>,
    pub options: ProviderOptions,
    client: reqwest::Client,
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
            let text = CodexProvider::message_text_from_item(item);
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
            let parts = CodexProvider::response_parts_from_value(final_response);
            if !parts.is_empty() {
                return parts;
            }
            let text = CodexProvider::extract_text(final_response);
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

impl CodexProvider {
    const CODEX_ORIGINATOR: &'static str = "codex_cli_rs";
    const CODEX_RESPONSES_URL: &'static str = "https://chatgpt.com/backend-api/codex/responses";
    /// Maximum number of submission attempts (1 initial + up to 3 retries).
    /// Mirrors pi's `MAX_RETRIES = 3` loop, which iterates `attempt <= MAX_RETRIES`.
    const MAX_ATTEMPTS: u32 = 4;
    /// Base delay for exponential backoff between retries (ms).
    /// Matches pi-mono `BASE_DELAY_MS` at `openai-codex-responses.ts:47`.
    const BASE_DELAY_MS: u64 = 1000;

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
            options: ProviderOptions::default(),
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

    pub fn with_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.client = (*client).clone();
        self
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

    fn input_image_part(att: &LlmAttachment) -> Value {
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
                for block in msg.blocks.iter() {
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

    fn tool_choice_value(choice: &LlmToolChoice) -> &'static str {
        match choice {
            LlmToolChoice::Auto => "auto",
            LlmToolChoice::None => "none",
            LlmToolChoice::Required => "required",
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
impl Provider for CodexProvider {
    fn kind(&self) -> &'static str {
        "codex"
    }

    fn default_model(&self) -> &str {
        "gpt-5.5"
    }

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

    fn default_model_variant(&self, model: &str) -> Option<&'static str> {
        let variants = self.supported_variants(model);
        if variants.is_empty() {
            return None;
        }
        if model.eq_ignore_ascii_case("gpt-5.5") {
            return Some("medium");
        }
        if variants.contains(&"xhigh") {
            Some("xhigh")
        } else {
            Some("high")
        }
    }

    fn request_variant_config(&self, model: &str, variant: &str) -> Option<VariantRequestConfig> {
        if self.validate_variant(model, variant).is_err() {
            return None;
        }
        Some(VariantRequestConfig::ReasoningEffort(variant.to_string()))
    }

    fn default_agent_model(&self, tier: &str) -> Option<AgentModelSelection> {
        match tier {
            "low" => Some(AgentModelSelection {
                model: "gpt-5.4-mini".to_string(),
                variant: Some("low".to_string()),
            }),
            "medium" => Some(AgentModelSelection {
                model: "gpt-5.4".to_string(),
                variant: Some("medium".to_string()),
            }),
            "high" => Some(AgentModelSelection {
                model: "gpt-5.5".to_string(),
                variant: Some("medium".to_string()),
            }),
            _ => None,
        }
    }

    fn context_lookup_model(&self, model: &str) -> String {
        if model.contains('/') {
            model.to_string()
        } else {
            format!("openai/{model}")
        }
    }

    fn options(&self) -> &ProviderOptions {
        &self.options
    }

    fn options_mut(&mut self) -> &mut ProviderOptions {
        &mut self.options
    }

    fn requires_streaming(&self) -> bool {
        true
    }

    async fn ensure_fresh(&mut self) -> Result<bool, lash::oauth::OAuthError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        if now + 300 >= self.expires_at {
            let tokens = oauth::refresh_tokens(&self.refresh_token).await?;
            self.access_token = tokens.access_token;
            self.refresh_token = tokens.refresh_token;
            self.expires_at = tokens.expires_at;
            if let Some(new_account_id) = tokens.account_id {
                self.account_id = Some(new_account_id);
            }
            return Ok(true);
        }
        Ok(false)
    }

    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        let stream_events = req.stream_events.clone();
        let access_token = self.access_token.clone();
        let account_id = self.account_id.clone();
        let timeouts = self.options.llm_timeouts();

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
                request_body.clone().map(request_body_snapshot),
                response_start_timeout(
                    timeouts.request_timeout,
                    timeouts.chunk_timeout,
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
                        timeouts.request_timeout,
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
            let text = read_response_text(
                resp,
                timeouts.request_timeout,
                "Codex response body timed out",
            )
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
            timeouts.chunk_timeout,
            "Codex stream chunk timed out",
            |raw| {
                let prev_len = state.deltas.len();
                let prev_usage = state.usage.clone();
                let mut emitted_parts = Vec::new();
                Self::process_sse_event(raw, &mut state, Some(&mut emitted_parts))?;
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
        serde_json::Value::Object(map)
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
    #[serde(default)]
    options: ProviderOptions,
}

/// Factory that registers [`CodexProvider`] with lash's global
/// provider registry.
pub struct CodexProviderFactory;

impl CodexProviderFactory {
    pub fn register() {
        lash::register_provider_factory(std::sync::Arc::new(Self));
    }
}

impl ProviderFactory for CodexProviderFactory {
    fn kind(&self) -> &'static str {
        "codex"
    }
    fn cli_label(&self) -> &'static str {
        "OpenAI Codex OAuth"
    }
    fn setup_name(&self) -> &'static str {
        "Codex"
    }
    fn setup_description(&self) -> &'static str {
        "ChatGPT Plus/Pro/Team"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<Box<dyn Provider>, String> {
        let cfg: CodexProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(Box::new(CodexProvider {
            access_token: cfg.access_token,
            refresh_token: cfg.refresh_token,
            expires_at: cfg.expires_at,
            account_id: cfg.account_id,
            options: cfg.options,
            client: build_http_client(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::provider::{Provider, ProviderOptions};

    fn provider() -> CodexProvider {
        CodexProvider {
            access_token: String::new(),
            refresh_token: String::new(),
            expires_at: 0,
            account_id: None,
            options: ProviderOptions::default(),
            client: build_http_client(),
        }
    }

    #[test]
    fn gpt_55_variants_match_codex_catalog() {
        let provider = provider();

        assert_eq!(
            provider.supported_variants("gpt-5.5"),
            ["low", "medium", "high", "xhigh"]
        );
        assert_eq!(provider.default_model_variant("gpt-5.5"), Some("medium"));
        assert!(provider.validate_variant("gpt-5.5", "xhigh").is_ok());
        assert!(provider.validate_variant("gpt-5.5", "minimal").is_err());
    }

    #[test]
    fn gpt_55_variant_match_ignores_provider_prefix() {
        let provider = provider();

        assert_eq!(
            provider.supported_variants("openai/gpt-5.5"),
            ["low", "medium", "high", "xhigh"]
        );
    }
}
