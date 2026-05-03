use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::LazyLock;

use lash::llm::streaming::{drive_sse_response, emit_progress};
use lash::llm::timeouts::{
    build_http_client, read_response_text, request_body_snapshot_bytes, response_start_timeout,
    send_request,
};
use lash::llm::transport::LlmTransportError;
use lash::llm::types::{
    LlmAttachment, LlmContentBlock, LlmOutputPart, LlmOutputSpec, LlmProviderTraceEvent,
    LlmRequest, LlmResponse, LlmRole, LlmStreamEvent, LlmToolChoice, LlmUsage, ResponseTextMeta,
    ResponseTextPhase,
};
use lash::provider::{
    AgentModelSelection, Provider, ProviderFactory, ProviderOptions, VariantRequestConfig,
};

pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

const OPENROUTER_REASONING_VARIANTS: &[&str] =
    &["none", "minimal", "low", "medium", "high", "xhigh"];

fn emit_provider_trace(
    tx: Option<&lash::llm::types::LlmProviderTraceSender>,
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

static DEFAULT_HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(build_http_client);

fn base_url_is_openrouter(base_url: &str) -> bool {
    base_url.trim_end_matches('/') == OPENROUTER_BASE_URL
}

fn model_id(model: &str) -> &str {
    model
        .rsplit_once('/')
        .map(|(_, tail)| tail)
        .unwrap_or(model)
}

fn sanitize_surrogates(s: &str) -> &str {
    s
}

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
        return Some(msg.to_string());
    }
    Some(trimmed.chars().take(200).collect())
}

#[derive(Clone, Debug)]
pub struct OpenAiGenericProvider {
    pub api_key: String,
    pub base_url: String,
    pub options: ProviderOptions,
    client: reqwest::Client,
}

#[derive(Clone, Debug, Default)]
struct ResponsesStreamingToolCall {
    call_id: String,
    tool_name: String,
    input_json: String,
    item_id: String,
}

#[derive(Clone, Debug, Default)]
struct ResponsesStreamState {
    full_text: String,
    deltas: Vec<String>,
    parts: Vec<LlmOutputPart>,
    usage: LlmUsage,
    provider_usage: Option<Value>,
    current_text_part: Option<usize>,
    current_reasoning_part: Option<usize>,
    reasoning_deltas: Vec<String>,
    tool_calls: HashMap<String, ResponsesStreamingToolCall>,
    final_response: Option<Value>,
}

impl ResponsesStreamState {
    fn begin_message(&mut self, item: Option<&Value>) {
        let meta = item.map(OpenAiGenericProvider::response_text_meta_from_message_item);
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Text {
            text: String::new(),
            response_meta: meta,
        });
        self.current_text_part = Some(index);
    }

    fn finish_message(&mut self, item: Option<&Value>) {
        if let Some(item) = item {
            let text = OpenAiGenericProvider::message_text_from_item(item);
            let meta = OpenAiGenericProvider::response_text_meta_from_message_item(item);
            let index = self.ensure_text_part_index();
            if let Some(LlmOutputPart::Text {
                text: existing,
                response_meta,
            }) = self.parts.get_mut(index)
            {
                if !text.is_empty() {
                    *existing = text;
                }
                *response_meta = Some(meta);
            }
            self.recompute_full_text();
        }
        self.current_text_part = None;
    }

    fn push_text_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        let index = self.ensure_text_part_index();
        if let Some(LlmOutputPart::Text { text, .. }) = self.parts.get_mut(index) {
            text.push_str(piece);
        }
        self.deltas.push(piece.to_string());
        self.recompute_full_text();
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
            response_meta: None,
        });
        index
    }

    fn recompute_full_text(&mut self) {
        self.full_text.clear();
        for part in &self.parts {
            if let LlmOutputPart::Text { text, .. } = part {
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
        if let Some(index) = self.current_reasoning_part.take()
            && let Some(LlmOutputPart::Reasoning { text, .. }) = self.parts.get_mut(index)
        {
            let trimmed = text.trim_end();
            if trimmed.len() != text.len() {
                *text = trimmed.to_string();
            }
        }
    }

    fn finalize_reasoning_item(&mut self, item: &Value) {
        let Some((_, part)) = self
            .parts
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(_, part)| matches!(part, LlmOutputPart::Reasoning { .. }))
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
            let texts = arr
                .iter()
                .filter_map(|entry| entry.get("text").and_then(|v| v.as_str()).map(String::from))
                .collect::<Vec<_>>();
            if !texts.is_empty() {
                *summary = texts;
            }
        }
    }

    fn update_tool_call_from_item(&mut self, item: &Value) -> Option<String> {
        let item_id = item.get("id").and_then(|v| v.as_str())?.to_string();
        let tool_call = self.tool_calls.entry(item_id.clone()).or_default();
        tool_call.item_id.clone_from(&item_id);
        if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
            tool_call.call_id = call_id.to_string();
        }
        if let Some(name) = item.get("name").and_then(|v| v.as_str()) {
            tool_call.tool_name = name.to_string();
        }
        if let Some(args) = item.get("arguments").and_then(|v| v.as_str())
            && !args.is_empty()
        {
            tool_call.input_json = args.to_string();
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
        self.tool_calls
            .entry(item_id.to_string())
            .or_default()
            .input_json = arguments.to_string();
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
            item_id: (!tool_call.item_id.is_empty()).then_some(tool_call.item_id),
            signature: None,
        };
        self.parts.push(part.clone());
        Some(part)
    }

    fn take_reasoning_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reasoning_deltas)
    }

    fn response_parts(&self) -> Vec<LlmOutputPart> {
        let mut parts = self
            .parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text, .. } if text.is_empty() => None,
                LlmOutputPart::Reasoning { text, .. } if text.trim().is_empty() => None,
                other => Some(other.clone()),
            })
            .collect::<Vec<_>>();
        if parts.is_empty()
            && let Some(final_response) = &self.final_response
        {
            parts = OpenAiGenericProvider::response_parts_from_value(final_response);
        }
        parts
    }
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

    fn input_image_part(att: &LlmAttachment) -> Value {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
        json!({
            "type": "input_image",
            "image_url": format!("data:{};base64,{}", att.mime, b64),
        })
    }

    fn phase_value(phase: &ResponseTextPhase) -> &'static str {
        match phase {
            ResponseTextPhase::Commentary => "commentary",
            ResponseTextPhase::FinalAnswer => "final_answer",
        }
    }

    fn response_text_meta_from_message_item(item: &Value) -> ResponseTextMeta {
        ResponseTextMeta {
            id: item.get("id").and_then(|v| v.as_str()).map(str::to_string),
            status: item
                .get("status")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| Some("completed".to_string())),
            phase: item
                .get("phase")
                .and_then(|v| v.as_str())
                .and_then(|phase| match phase {
                    "commentary" => Some(ResponseTextPhase::Commentary),
                    "final_answer" => Some(ResponseTextPhase::FinalAnswer),
                    _ => None,
                }),
        }
    }

    fn flush_pending_content(
        pending: &mut Vec<Value>,
        input: &mut Vec<Value>,
        role: &'static str,
        is_user: bool,
        response_meta: Option<ResponseTextMeta>,
        message_index: usize,
        part_index: usize,
    ) {
        if pending.is_empty() {
            return;
        }
        let content = std::mem::take(pending);
        if role == "assistant" {
            let meta = response_meta.unwrap_or(ResponseTextMeta {
                id: Some(format!("msg_lash_{message_index}_{part_index}")),
                status: Some("completed".to_string()),
                phase: None,
            });
            let mut item = json!({
                "type": "message",
                "role": "assistant",
                "id": meta.id.unwrap_or_else(|| format!("msg_lash_{message_index}_{part_index}")),
                "status": meta.status.unwrap_or_else(|| "completed".to_string()),
                "content": content,
            });
            if let Some(phase) = meta.phase.as_ref() {
                item["phase"] = json!(Self::phase_value(phase));
            }
            input.push(item);
            return;
        }
        if is_user
            && let Some(prev) = input.last_mut()
            && prev.get("role").and_then(|v| v.as_str()) == Some("user")
            && prev.get("content").is_some_and(|v| v.is_array())
        {
            prev["content"].as_array_mut().unwrap().extend(content);
        } else {
            input.push(json!({
                "role": role,
                "content": content,
            }));
        }
    }

    fn build_input(req: &LlmRequest) -> (String, Vec<Value>) {
        let mut instructions = Vec::new();
        let mut input = Vec::new();

        for (message_index, msg) in req.messages.iter().enumerate() {
            if matches!(msg.role, LlmRole::System) {
                for block in msg.blocks.iter() {
                    if let LlmContentBlock::Text { text, .. } = block
                        && !text.is_empty()
                    {
                        instructions.push(sanitize_surrogates(text).to_string());
                    }
                }
                continue;
            }

            let role = Self::role_name(&msg.role);
            let mut pending_content = Vec::new();
            let mut pending_meta: Option<ResponseTextMeta> = None;
            let mut pending_part_index = 0usize;

            for (part_index, block) in msg.blocks.iter().enumerate() {
                match block {
                    LlmContentBlock::Text {
                        text,
                        response_meta,
                    } => {
                        if text.is_empty() {
                            continue;
                        }
                        if matches!(msg.role, LlmRole::Assistant)
                            && (!pending_content.is_empty() || response_meta.is_some())
                        {
                            Self::flush_pending_content(
                                &mut pending_content,
                                &mut input,
                                role,
                                false,
                                pending_meta.take(),
                                message_index,
                                pending_part_index,
                            );
                            pending_part_index = part_index;
                            pending_meta = response_meta.clone();
                        }
                        let part_type = if matches!(msg.role, LlmRole::Assistant) {
                            "output_text"
                        } else {
                            "input_text"
                        };
                        pending_content.push(json!({
                            "type": part_type,
                            "text": sanitize_surrogates(text),
                            "annotations": if part_type == "output_text" { json!([]) } else { Value::Null },
                        }));
                        if part_type == "input_text" {
                            pending_content
                                .last_mut()
                                .and_then(Value::as_object_mut)
                                .map(|obj| obj.remove("annotations"));
                        }
                    }
                    LlmContentBlock::Image { attachment_idx } => {
                        if matches!(msg.role, LlmRole::User)
                            && let Some(att) = req.attachments.get(*attachment_idx)
                        {
                            pending_content.push(Self::input_image_part(att));
                        }
                    }
                    LlmContentBlock::Reasoning {
                        text,
                        encrypted_content,
                        signature,
                        item_id,
                        summary,
                        ..
                    } => {
                        Self::flush_pending_content(
                            &mut pending_content,
                            &mut input,
                            role,
                            matches!(msg.role, LlmRole::User),
                            pending_meta.take(),
                            message_index,
                            pending_part_index,
                        );
                        let Some(blob) = encrypted_content.as_deref().or(signature.as_deref())
                        else {
                            continue;
                        };
                        let summary_items = if summary.is_empty() {
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
                        ..
                    } => {
                        Self::flush_pending_content(
                            &mut pending_content,
                            &mut input,
                            role,
                            matches!(msg.role, LlmRole::User),
                            pending_meta.take(),
                            message_index,
                            pending_part_index,
                        );
                        let mut item = json!({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tool_name,
                            "arguments": input_json,
                        });
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
                            role,
                            matches!(msg.role, LlmRole::User),
                            pending_meta.take(),
                            message_index,
                            pending_part_index,
                        );
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": sanitize_surrogates(content),
                        }));
                    }
                }
            }
            Self::flush_pending_content(
                &mut pending_content,
                &mut input,
                role,
                matches!(msg.role, LlmRole::User),
                pending_meta.take(),
                message_index,
                pending_part_index,
            );
        }
        (instructions.join("\n\n"), input)
    }

    fn build_tools(req: &LlmRequest) -> Vec<Value> {
        req.tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": false,
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

    fn local_style_base_url(base_url: &str) -> bool {
        let normalized = base_url.trim().to_ascii_lowercase();
        normalized.contains("localhost")
            || normalized.contains("127.0.0.1")
            || normalized.contains("0.0.0.0")
            || normalized.contains("ollama")
    }

    fn supports_openai_request_fields(base_url: &str) -> bool {
        !Self::local_style_base_url(base_url)
    }

    fn max_output_tokens() -> u64 {
        std::env::var("LASH_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|v: &u64| *v > 0)
            .unwrap_or(32768)
    }

    fn clamp_reasoning_effort(model: &str, effort: &str) -> String {
        let id = model_id(model).to_ascii_lowercase();
        if (id.starts_with("gpt-5.2") || id.starts_with("gpt-5.3") || id.starts_with("gpt-5.4"))
            && effort == "minimal"
        {
            return "low".to_string();
        }
        effort.to_string()
    }

    fn build_request_body(&self, req: &LlmRequest, stream: bool) -> Value {
        let tools = Self::build_tools(req);
        let (instructions, input) = Self::build_input(req);
        let mut body = json!({
            "model": req.model,
            "instructions": instructions,
            "input": input,
            "tools": tools,
            "stream": stream,
            "max_output_tokens": Self::max_output_tokens(),
        });
        if !req.tools.is_empty() {
            body["tool_choice"] = json!(Self::tool_choice_value(&req.tool_choice));
        }
        if Self::supports_openai_request_fields(&self.base_url) {
            body["include"] = json!(["reasoning.encrypted_content"]);
            body["store"] = json!(false);
            body["parallel_tool_calls"] = json!(!req.tools.is_empty());
            body["text"] = json!({"verbosity": "medium"});
        }
        if let Some(variant) = req.model_variant.as_deref()
            && let Some(VariantRequestConfig::ReasoningEffort(effort)) =
                self.request_variant_config(&req.model, variant)
            && effort != "none"
        {
            body["reasoning"] = json!({
                "effort": Self::clamp_reasoning_effort(&req.model, &effort),
            });
        }
        if let Some(output_spec) = &req.output_spec {
            let format = match output_spec {
                LlmOutputSpec::JsonObject => json!({ "type": "json_object" }),
                LlmOutputSpec::JsonSchema(schema) => json!({
                    "type": "json_schema",
                    "name": schema.name,
                    "schema": schema.schema,
                    "strict": schema.strict,
                }),
            };
            if body.get("text").is_none() {
                body["text"] = json!({});
            }
            body["text"]["format"] = format;
        }
        body
    }

    fn parse_i64(value: Option<&Value>) -> i64 {
        match value {
            Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
            Some(Value::String(s)) => s.parse().unwrap_or(0),
            _ => 0,
        }
    }

    fn usage_from_response_value(value: &Value) -> LlmUsage {
        let usage = value.get("usage").unwrap_or(&Value::Null);
        LlmUsage {
            input_tokens: Self::parse_i64(
                usage
                    .get("input_tokens")
                    .or_else(|| usage.get("prompt_tokens")),
            ),
            output_tokens: Self::parse_i64(
                usage
                    .get("output_tokens")
                    .or_else(|| usage.get("completion_tokens")),
            ),
            cached_input_tokens: Self::parse_i64(
                usage
                    .get("input_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .or_else(|| {
                        usage
                            .get("prompt_tokens_details")
                            .and_then(|d| d.get("cached_tokens"))
                    }),
            ),
            reasoning_tokens: Self::parse_i64(
                usage
                    .get("output_tokens_details")
                    .and_then(|d| d.get("reasoning_tokens"))
                    .or_else(|| {
                        usage
                            .get("completion_tokens_details")
                            .and_then(|d| d.get("reasoning_tokens"))
                    }),
            ),
        }
    }

    fn merge_usage(dst: &mut LlmUsage, src: &LlmUsage) {
        if src != &LlmUsage::default() {
            *dst = src.clone();
        }
    }

    fn message_text_from_item(item: &Value) -> String {
        item.get("content")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|part| match part.get("type").and_then(|v| v.as_str()) {
                Some("output_text") => part.get("text").and_then(|v| v.as_str()),
                Some("refusal") => part.get("refusal").and_then(|v| v.as_str()),
                _ => None,
            })
            .collect::<String>()
    }

    fn response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
        let mut parts = Vec::new();
        if let Some(output) = value.get("output").and_then(|v| v.as_array()) {
            for item in output {
                match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "reasoning" => {
                        let summary = item
                            .get("summary")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|entry| {
                                        entry.get("text").and_then(|v| v.as_str()).map(String::from)
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let text = summary.join("\n\n");
                        parts.push(LlmOutputPart::Reasoning {
                            text,
                            signature: None,
                            redacted: false,
                            item_id: item.get("id").and_then(|v| v.as_str()).map(str::to_string),
                            encrypted_content: item
                                .get("encrypted_content")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            summary,
                        });
                    }
                    "message" => {
                        let text = Self::message_text_from_item(item);
                        if !text.is_empty() {
                            parts.push(LlmOutputPart::Text {
                                text,
                                response_meta: Some(Self::response_text_meta_from_message_item(
                                    item,
                                )),
                            });
                        }
                    }
                    "function_call" => {
                        let Some(name) = item.get("name").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let arguments = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}")
                            .to_string();
                        parts.push(LlmOutputPart::ToolCall {
                            call_id: item
                                .get("call_id")
                                .and_then(|v| v.as_str())
                                .map(str::to_string)
                                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                            tool_name: name.to_string(),
                            input_json: arguments,
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
                response_meta: None,
            });
        }
        parts
    }

    fn has_response_content(parts: &[LlmOutputPart]) -> bool {
        parts.iter().any(|part| match part {
            LlmOutputPart::Text { text, .. } => !text.is_empty(),
            LlmOutputPart::Reasoning { .. } => true,
            LlmOutputPart::ToolCall { .. } => true,
        })
    }

    fn empty_response_error(raw: String) -> LlmTransportError {
        LlmTransportError::new("OpenAI-compatible empty_response")
            .retryable(true)
            .with_code("empty_response")
            .with_raw(raw)
    }

    fn retryable_error_code(value: &Value) -> Option<i64> {
        value
            .get("code")
            .or_else(|| value.get("status"))
            .and_then(|v| match v {
                Value::Number(n) => n.as_i64(),
                Value::String(s) => s.trim().parse().ok(),
                _ => None,
            })
    }

    fn responses_error_is_retryable(value: &Value) -> bool {
        matches!(Self::retryable_error_code(value), Some(429))
            || matches!(Self::retryable_error_code(value), Some(status) if status >= 500)
    }

    fn error_message_from_response_failed(event: &Value) -> String {
        event
            .get("response")
            .and_then(|r| r.get("error"))
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .or_else(|| {
                event
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
            })
            .unwrap_or("OpenAI-compatible response failed")
            .to_string()
    }

    fn process_sse_event(
        raw: &str,
        state: &mut ResponsesStreamState,
        emitted_parts: Option<&mut Vec<LlmOutputPart>>,
    ) -> Result<(), LlmTransportError> {
        let raw = raw.trim();
        if raw.is_empty() || raw == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(raw).map_err(|e| {
            LlmTransportError::new(format!("Invalid Responses SSE payload: {e}")).with_raw(raw)
        })?;
        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if event_type == "error" {
            let retryable = event
                .get("error")
                .map(Self::responses_error_is_retryable)
                .unwrap_or(false);
            let message = event
                .get("message")
                .and_then(|v| v.as_str())
                .or_else(|| {
                    event
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|v| v.as_str())
                })
                .unwrap_or("OpenAI-compatible stream error");
            return Err(LlmTransportError::new(message)
                .retryable(retryable)
                .with_raw(event.to_string()));
        }

        if let Some(resp) = event.get("response") {
            state.final_response = Some(resp.clone());
            state.provider_usage = resp.get("usage").cloned();
            Self::merge_usage(&mut state.usage, &Self::usage_from_response_value(resp));
        }

        match event_type {
            "response.output_item.added" => {
                if let Some(item) = event.get("item") {
                    match item.get("type").and_then(|v| v.as_str()) {
                        Some("message") => state.begin_message(Some(item)),
                        Some("function_call") => {
                            let _ = state.update_tool_call_from_item(item);
                        }
                        Some("reasoning") => state.begin_reasoning_part(),
                        _ => {}
                    }
                }
            }
            "response.reasoning_summary_part.added" => state.begin_reasoning_part(),
            "response.reasoning_summary_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                    state.push_reasoning_delta(delta);
                }
            }
            "response.reasoning_summary_part.done" => state.finish_reasoning_part(),
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                    state.push_text_delta(delta);
                }
            }
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
                        Some("reasoning") => {
                            state.finish_reasoning_part();
                            state.finalize_reasoning_item(item);
                        }
                        Some("function_call") => {
                            let part = state.finish_tool_call(item);
                            if let (Some(parts), Some(part)) = (emitted_parts, part) {
                                parts.push(part);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "response.completed" => {}
            "response.failed" => {
                let error_value = event
                    .get("response")
                    .and_then(|r| r.get("error"))
                    .or_else(|| event.get("error"))
                    .cloned()
                    .unwrap_or(Value::Null);
                return Err(
                    LlmTransportError::new(Self::error_message_from_response_failed(&event))
                        .retryable(Self::responses_error_is_retryable(&error_value))
                        .with_raw(event.to_string()),
                );
            }
            _ => {}
        }
        Ok(())
    }

    fn parse_sse_payload(
        payload: &str,
        state: &mut ResponsesStreamState,
    ) -> Result<(), LlmTransportError> {
        let mut event_lines = Vec::new();
        for mut line in payload.lines().map(str::to_string) {
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
            if line.trim().is_empty() && !event_lines.is_empty() {
                let raw = event_lines.join("\n");
                Self::process_sse_event(&raw, state, None)?;
                event_lines.clear();
            }
        }
        if !event_lines.is_empty() {
            let raw = event_lines.join("\n");
            Self::process_sse_event(&raw, state, None)?;
        }
        Ok(())
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
        if model.to_ascii_lowercase().contains("gpt") {
            Some("medium")
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
        let provider_trace = req.provider_trace.clone();
        let timeouts = self.options.llm_timeouts();
        let body = self.build_request_body(&req, stream_events.is_some());
        let body_bytes = serde_json::to_vec(&body).map_err(|e| {
            LlmTransportError::new(format!("Failed to serialize Responses body: {e}"))
        })?;
        let request_body = request_body_snapshot_bytes(body_bytes);
        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let request = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
            let detail = extract_error_detail(&text);
            let message = detail
                .map(|detail| {
                    format!(
                        "OpenAI-compatible request failed with {}: {}",
                        status.as_u16(),
                        detail
                    )
                })
                .unwrap_or_else(|| {
                    format!("OpenAI-compatible request failed with {}", status.as_u16())
                });
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
            emit_provider_trace(provider_trace.as_ref(), "openai_compatible", &text);
            let mut state = ResponsesStreamState::default();
            if text.trim_start().starts_with("data:") || text.contains("\ndata:") {
                Self::parse_sse_payload(&text, &mut state)?;
            } else {
                let value: Value = serde_json::from_str(&text).map_err(|e| {
                    LlmTransportError::new(format!("Invalid Responses JSON: {e}"))
                        .with_raw(text.clone())
                })?;
                state.provider_usage = value.get("usage").cloned();
                state.usage = Self::usage_from_response_value(&value);
                state.parts = Self::response_parts_from_value(&value);
                state.recompute_full_text();
                state.final_response = Some(value);
            }
            let parts = state.response_parts();
            if !Self::has_response_content(&parts) {
                return Err(Self::empty_response_error(text));
            }
            if let Some(tx) = &stream_events {
                if state.usage != LlmUsage::default() {
                    tx.send(LlmStreamEvent::Usage(state.usage.clone()));
                }
                for part in &parts {
                    if let LlmOutputPart::Reasoning { text, .. } = part
                        && !text.is_empty()
                    {
                        tx.send(LlmStreamEvent::ReasoningDelta(text.clone()));
                    }
                }
                if !state.full_text.is_empty() {
                    tx.send(LlmStreamEvent::Delta(state.full_text.clone()));
                }
            }
            return Ok(LlmResponse {
                deltas: (!state.full_text.is_empty())
                    .then_some(state.full_text.clone())
                    .into_iter()
                    .collect(),
                full_text: state.full_text,
                parts,
                usage: state.usage,
                provider_usage: state.provider_usage,
                request_body: None,
                http_summary: Some(format!("HTTP POST {}", url)),
            });
        }

        let mut state = ResponsesStreamState::default();
        let mut emitted_parts = Vec::new();
        drive_sse_response(
            resp,
            timeouts.chunk_timeout,
            "OpenAI-compatible stream chunk timed out",
            |raw| {
                emit_provider_trace(provider_trace.as_ref(), "openai_compatible", raw);
                let prev_len = state.deltas.len();
                let prev_usage = state.usage.clone();
                Self::process_sse_event(raw, &mut state, Some(&mut emitted_parts))?;
                emit_progress(
                    stream_events.as_ref(),
                    &state.deltas,
                    prev_len,
                    &state.usage,
                    &prev_usage,
                );
                if let Some(tx) = &stream_events {
                    for delta in state.take_reasoning_deltas() {
                        tx.send(LlmStreamEvent::ReasoningDelta(delta));
                    }
                    for part in emitted_parts.drain(..) {
                        tx.send(LlmStreamEvent::Part(part));
                    }
                } else {
                    emitted_parts.clear();
                    state.take_reasoning_deltas();
                }
                Ok(())
            },
        )
        .await?;

        let parts = state.response_parts();
        if !Self::has_response_content(&parts) {
            return Err(Self::empty_response_error(
                state
                    .final_response
                    .as_ref()
                    .map(Value::to_string)
                    .unwrap_or_default(),
            ));
        }
        Ok(LlmResponse {
            deltas: state.deltas,
            full_text: state.full_text,
            parts,
            usage: state.usage,
            provider_usage: state.provider_usage,
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
        "Any OpenAI-compatible Responses API endpoint"
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
    use super::*;
    use lash::llm::types::{LlmMessage, LlmToolSpec};
    use std::sync::Arc;

    fn request(messages: Vec<LlmMessage>) -> LlmRequest {
        LlmRequest {
            model: "openai/gpt-5.4".to_string(),
            messages,
            attachments: Vec::new(),
            tools: Arc::new(Vec::<LlmToolSpec>::new()),
            tool_choice: LlmToolChoice::Auto,
            model_variant: None,
            session_id: Some("session-1".to_string()),
            output_spec: None,
            stream_events: None,
            provider_trace: None,
        }
    }

    #[test]
    fn builds_responses_body_with_instructions_and_input() {
        let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
        let req = request(vec![
            LlmMessage::text(LlmRole::System, "system prompt"),
            LlmMessage::text(LlmRole::User, "hello"),
        ]);
        let body = provider.build_request_body(&req, true);
        assert_eq!(body["instructions"], "system prompt");
        assert_eq!(body["stream"], true);
        assert!(body.get("messages").is_none());
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    }

    #[test]
    fn assistant_text_preserves_response_meta() {
        let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
        let req = request(vec![LlmMessage::new(
            LlmRole::Assistant,
            vec![LlmContentBlock::Text {
                text: "done".into(),
                response_meta: Some(ResponseTextMeta {
                    id: Some("msg_1".to_string()),
                    status: Some("completed".to_string()),
                    phase: Some(ResponseTextPhase::FinalAnswer),
                }),
            }],
        )]);
        let body = provider.build_request_body(&req, false);
        assert_eq!(body["input"][0]["type"], "message");
        assert_eq!(body["input"][0]["id"], "msg_1");
        assert_eq!(body["input"][0]["phase"], "final_answer");
    }

    #[test]
    fn legacy_assistant_text_gets_deterministic_id() {
        let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
        let req = request(vec![LlmMessage::text(LlmRole::Assistant, "legacy")]);
        let body = provider.build_request_body(&req, false);
        assert_eq!(body["input"][0]["id"], "msg_lash_0_0");
        assert_eq!(body["input"][0]["status"], "completed");
        assert!(body["input"][0].get("phase").is_none());
    }

    #[test]
    fn local_base_url_omits_openai_only_fields() {
        let provider = OpenAiGenericProvider::new("key", "http://localhost:11434/v1");
        let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        let body = provider.build_request_body(&req, true);
        assert!(body.get("include").is_none());
        assert!(body.get("store").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
        assert!(body.get("text").is_none());
    }

    #[test]
    fn usage_parser_accepts_responses_and_chat_completion_shapes() {
        let responses_usage =
            OpenAiGenericProvider::usage_from_response_value(&serde_json::json!({
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "input_tokens_details": { "cached_tokens": 3 },
                    "output_tokens_details": { "reasoning_tokens": 5 }
                }
            }));
        assert_eq!(responses_usage.input_tokens, 11);
        assert_eq!(responses_usage.output_tokens, 7);
        assert_eq!(responses_usage.cached_input_tokens, 3);
        assert_eq!(responses_usage.reasoning_tokens, 5);

        let chat_usage = OpenAiGenericProvider::usage_from_response_value(&serde_json::json!({
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 17,
                "prompt_tokens_details": { "cached_tokens": 2 },
                "completion_tokens_details": { "reasoning_tokens": 4 }
            }
        }));
        assert_eq!(chat_usage.input_tokens, 13);
        assert_eq!(chat_usage.output_tokens, 17);
        assert_eq!(chat_usage.cached_input_tokens, 2);
        assert_eq!(chat_usage.reasoning_tokens, 4);
    }

    #[test]
    fn stream_parser_captures_text_reasoning_tool_and_phase() {
        let mut state = ResponsesStreamState::default();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.reasoning_summary_text.delta","delta":"Think"}"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think"}],"encrypted_content":"enc"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.output_text.delta","delta":"Hi"}"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","phase":"commentary","content":[{"type":"output_text","text":"Hi"}]}}"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":""}}"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"x\":"}"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"x\":1}" }"#,
            &mut state,
            None,
        )
        .unwrap();
        OpenAiGenericProvider::process_sse_event(
            r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"tool","arguments":"{\"x\":1}","status":"completed"}}"#,
            &mut state,
            None,
        )
        .unwrap();

        assert_eq!(state.full_text, "Hi");
        let parts = state.response_parts();
        assert!(matches!(
            &parts[0],
            LlmOutputPart::Reasoning {
                item_id: Some(id),
                encrypted_content: Some(blob),
                ..
            } if id == "rs_1" && blob == "enc"
        ));
        assert!(matches!(
            &parts[1],
            LlmOutputPart::Text {
                response_meta: Some(ResponseTextMeta {
                    phase: Some(ResponseTextPhase::Commentary),
                    ..
                }),
                ..
            }
        ));
        assert!(matches!(
            &parts[2],
            LlmOutputPart::ToolCall {
                item_id: Some(id),
                input_json,
                ..
            } if id == "fc_1" && input_json == "{\"x\":1}"
        ));
    }
}
