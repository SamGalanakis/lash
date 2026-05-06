#![allow(clippy::result_large_err)]

use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::LazyLock;

use lash::SchemaProjectionOverride;
use lash::llm::streaming::{drive_sse_response, emit_progress};
use lash::llm::timeouts::{
    build_http_client, read_response_text, request_body_snapshot_bytes, response_start_timeout,
    send_request,
};
use lash::llm::transport::{LlmTransportError, ProviderFailureKind};
use lash::llm::types::{
    LlmAttachment, LlmContentBlock, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse,
    LlmRole, LlmStreamEvent, LlmToolChoice, LlmUsage, ResponseTextMeta, ResponseTextPhase,
};
use lash::provider::{
    AgentModelSelection, NoopProviderAuth, NoopProviderReadiness, ProviderComponents,
    ProviderFactory, ProviderModelPolicy, ProviderOptions, ProviderState, ProviderTransport,
    VariantRequestConfig,
};
use lash_openai_schema::{
    OpenAiSchemaProfile, SchemaProjectionError, project_schema, project_structured_output,
    project_tool_parameters,
};

pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

const OPENROUTER_REASONING_VARIANTS: &[&str] =
    &["none", "minimal", "low", "medium", "high", "xhigh"];

use lash_openai_schema::{emit_provider_trace, model_id};

static DEFAULT_HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(build_http_client);

fn base_url_is_openrouter(base_url: &str) -> bool {
    base_url.trim_end_matches('/') == OPENROUTER_BASE_URL
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
    pub wire_api: OpenAiWireApi,
    pub cache_retention: OpenAiCacheRetention,
    client: reqwest::Client,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAiWireApi {
    #[default]
    Auto,
    Responses,
    ChatCompletions,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAiCacheRetention {
    None,
    #[default]
    Short,
    Long,
}

#[derive(Clone, Debug)]
struct OpenAiModelPolicy {
    base_url: String,
}

impl OpenAiModelPolicy {
    fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
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

#[derive(Clone, Debug, Default)]
struct ChatStreamingToolCall {
    call_id: String,
    tool_name: String,
    input_json: String,
    signature: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ChatStreamState {
    full_text: String,
    deltas: Vec<String>,
    reasoning_text: String,
    reasoning_deltas: Vec<String>,
    usage: LlmUsage,
    provider_usage: Option<Value>,
    tool_calls: HashMap<usize, ChatStreamingToolCall>,
    final_response: Option<Value>,
}

impl ChatStreamState {
    fn push_text_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        self.full_text.push_str(piece);
        self.deltas.push(piece.to_string());
    }

    fn push_reasoning_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        self.reasoning_text.push_str(piece);
        self.reasoning_deltas.push(piece.to_string());
    }

    fn update_tool_call_delta(&mut self, value: &Value) {
        let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        let tool_call = self.tool_calls.entry(index).or_default();
        if let Some(id) = value.get("id").and_then(Value::as_str)
            && !id.is_empty()
        {
            tool_call.call_id = id.to_string();
        }
        if let Some(function) = value.get("function") {
            if let Some(name) = function.get("name").and_then(Value::as_str)
                && !name.is_empty()
            {
                tool_call.tool_name = name.to_string();
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str)
                && !arguments.is_empty()
            {
                tool_call.input_json.push_str(arguments);
            }
        }
    }

    fn apply_reasoning_details(&mut self, details: &Value) {
        let Some(details) = details.as_array() else {
            return;
        };
        for detail in details {
            if detail.get("type").and_then(Value::as_str) != Some("reasoning.encrypted") {
                continue;
            }
            let Some(id) = detail.get("id").and_then(Value::as_str) else {
                continue;
            };
            if detail.get("data").and_then(Value::as_str).is_none() {
                continue;
            }
            for tool_call in self.tool_calls.values_mut() {
                if tool_call.call_id == id {
                    tool_call.signature = Some(detail.to_string());
                    break;
                }
            }
        }
    }

    fn take_reasoning_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reasoning_deltas)
    }

    fn parts(&self) -> Vec<LlmOutputPart> {
        let mut parts = Vec::new();
        if !self.reasoning_text.trim().is_empty() {
            parts.push(LlmOutputPart::Reasoning {
                text: self.reasoning_text.clone(),
                signature: None,
                redacted: false,
                item_id: None,
                encrypted_content: None,
                summary: Vec::new(),
            });
        }
        if !self.full_text.is_empty() {
            parts.push(LlmOutputPart::Text {
                text: self.full_text.clone(),
                response_meta: None,
            });
        }
        let mut tool_calls = self.tool_calls.iter().collect::<Vec<_>>();
        tool_calls.sort_by_key(|(index, _)| **index);
        for (_, tool_call) in tool_calls {
            if tool_call.tool_name.is_empty() {
                continue;
            }
            parts.push(LlmOutputPart::ToolCall {
                call_id: if tool_call.call_id.is_empty() {
                    uuid::Uuid::new_v4().to_string()
                } else {
                    tool_call.call_id.clone()
                },
                tool_name: tool_call.tool_name.clone(),
                input_json: if tool_call.input_json.is_empty() {
                    "{}".to_string()
                } else {
                    tool_call.input_json.clone()
                },
                item_id: None,
                signature: tool_call.signature.clone(),
            });
        }
        if parts.is_empty()
            && let Some(final_response) = &self.final_response
        {
            parts = OpenAiGenericProvider::chat_response_parts_from_value(final_response);
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
            wire_api: OpenAiWireApi::Auto,
            cache_retention: OpenAiCacheRetention::Short,
            client: DEFAULT_HTTP_CLIENT.clone(),
        }
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_wire_api(mut self, wire_api: OpenAiWireApi) -> Self {
        self.wire_api = wire_api;
        self
    }

    pub fn with_cache_retention(mut self, retention: OpenAiCacheRetention) -> Self {
        self.cache_retention = retention;
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
                        ..
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

    fn projection_error(err: SchemaProjectionError) -> LlmTransportError {
        LlmTransportError::new(format!(
            "OpenAI schema projection failed: {}",
            err.first_diagnostic()
        ))
        .with_kind(ProviderFailureKind::Validation)
        .with_raw(
            json!({
                "profile": format!("{:?}", err.profile),
                "diagnostics": err.diagnostics,
            })
            .to_string(),
        )
    }

    fn projected_schema(
        canonical: &Value,
        overrides: &[SchemaProjectionOverride],
        profile: OpenAiSchemaProfile,
    ) -> Result<Value, LlmTransportError> {
        if let Some(override_schema) = overrides
            .iter()
            .find(|projection| projection.profile == profile.projection_id())
            .map(|projection| projection.schema.clone())
        {
            return Ok(override_schema);
        }
        match profile {
            OpenAiSchemaProfile::ToolParameters => {
                project_tool_parameters(canonical).map(|projection| projection.schema)
            }
            OpenAiSchemaProfile::StructuredOutput => {
                project_structured_output(canonical).map(|projection| projection.schema)
            }
            OpenAiSchemaProfile::StrictToolParameters => {
                project_schema(canonical, profile).map(|projection| projection.schema)
            }
        }
        .map_err(Self::projection_error)
    }

    fn build_tools(req: &LlmRequest) -> Result<Vec<Value>, LlmTransportError> {
        req.tools
            .iter()
            .map(|tool| {
                let parameters = Self::projected_schema(
                    &tool.input_schema,
                    &tool.input_schema_projections,
                    OpenAiSchemaProfile::ToolParameters,
                )?;
                Ok(json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                    "strict": false,
                }))
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

    fn model_is_anthropic_claude(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        model.contains("claude") || model.contains("anthropic/")
    }

    fn openrouter_claude_request(&self, req: &LlmRequest) -> bool {
        base_url_is_openrouter(&self.base_url) && Self::model_is_anthropic_claude(&req.model)
    }

    fn resolved_wire_api(&self, req: &LlmRequest) -> OpenAiWireApi {
        match self.wire_api {
            OpenAiWireApi::Auto if self.openrouter_claude_request(req) => {
                OpenAiWireApi::ChatCompletions
            }
            OpenAiWireApi::Auto => OpenAiWireApi::Responses,
            forced => forced,
        }
    }

    fn chat_cache_control_value(&self) -> Option<Value> {
        match self.cache_retention {
            OpenAiCacheRetention::None => None,
            OpenAiCacheRetention::Short => Some(json!({ "type": "ephemeral" })),
            OpenAiCacheRetention::Long => Some(json!({ "type": "ephemeral", "ttl": "1h" })),
        }
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

    fn build_responses_request_body(
        &self,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<Value, LlmTransportError> {
        let tools = Self::build_tools(req)?;
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
                OpenAiModelPolicy::new(self.base_url.clone())
                    .request_variant_config(&req.model, variant)
            && effort != "none"
        {
            body["reasoning"] = json!({
                "effort": Self::clamp_reasoning_effort(&req.model, &effort),
            });
        }
        if let Some(output_spec) = &req.output_spec {
            let format = match output_spec {
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
            if body.get("text").is_none() {
                body["text"] = json!({});
            }
            body["text"]["format"] = format;
        }
        if self.cache_retention != OpenAiCacheRetention::None
            && let Some(session_id) = req.session_id.as_deref()
        {
            body["prompt_cache_key"] = json!(session_id);
        }
        if self.cache_retention == OpenAiCacheRetention::Long
            && Self::supports_openai_request_fields(&self.base_url)
        {
            body["prompt_cache_retention"] = json!("24h");
        }
        Ok(body)
    }

    fn chat_image_part(att: &LlmAttachment) -> Value {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
        json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:{};base64,{}", att.mime, b64),
            },
        })
    }

    fn chat_text_or_parts(mut parts: Vec<Value>) -> Value {
        if parts.len() == 1
            && parts[0].get("__lash_cache_breakpoint").is_none()
            && let Some(text) = parts[0].get("text").and_then(Value::as_str)
        {
            return json!(text);
        }
        Value::Array(std::mem::take(&mut parts))
    }

    fn build_chat_messages(req: &LlmRequest) -> Vec<Value> {
        let mut messages = Vec::new();
        for msg in &req.messages {
            let role = Self::role_name(&msg.role);
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut reasoning_details = Vec::new();

            for block in msg.blocks.iter() {
                match block {
                    LlmContentBlock::Text { text, .. } if !text.is_empty() => {
                        let mut part = json!({
                            "type": "text",
                            "text": sanitize_surrogates(text),
                        });
                        if let LlmContentBlock::Text {
                            cache_breakpoint: true,
                            ..
                        } = block
                        {
                            part["__lash_cache_breakpoint"] = json!(true);
                        }
                        text_parts.push(part);
                    }
                    LlmContentBlock::Image { attachment_idx }
                        if matches!(msg.role, LlmRole::User) =>
                    {
                        if let Some(att) = req.attachments.get(*attachment_idx) {
                            text_parts.push(Self::chat_image_part(att));
                        }
                    }
                    LlmContentBlock::ToolCall {
                        call_id,
                        tool_name,
                        input_json,
                        signature,
                        ..
                    } if matches!(msg.role, LlmRole::Assistant) => {
                        tool_calls.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": input_json,
                            },
                        }));
                        if let Some(signature) = signature.as_deref()
                            && let Ok(detail) = serde_json::from_str::<Value>(signature)
                            && detail.get("type").and_then(Value::as_str)
                                == Some("reasoning.encrypted")
                        {
                            reasoning_details.push(detail);
                        }
                    }
                    LlmContentBlock::ToolResult {
                        call_id,
                        content,
                        tool_name,
                    } => {
                        let mut tool_message = json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": sanitize_surrogates(content),
                        });
                        if let Some(name) = tool_name.as_deref()
                            && !name.is_empty()
                        {
                            tool_message["name"] = json!(name);
                        }
                        messages.push(tool_message);
                    }
                    LlmContentBlock::Reasoning { .. } | LlmContentBlock::Image { .. } => {}
                    LlmContentBlock::Text { .. } | LlmContentBlock::ToolCall { .. } => {}
                }
            }

            if matches!(
                msg.role,
                LlmRole::System | LlmRole::User | LlmRole::Assistant
            ) && (!text_parts.is_empty() || !tool_calls.is_empty())
            {
                let mut wire_message = json!({ "role": role });
                if !text_parts.is_empty() {
                    wire_message["content"] = Self::chat_text_or_parts(text_parts);
                } else if matches!(msg.role, LlmRole::Assistant) {
                    wire_message["content"] = Value::Null;
                }
                if !tool_calls.is_empty() {
                    wire_message["tool_calls"] = Value::Array(tool_calls);
                }
                if !reasoning_details.is_empty() {
                    wire_message["reasoning_details"] = Value::Array(reasoning_details);
                }
                messages.push(wire_message);
            }
        }
        messages
    }

    fn build_chat_tools(req: &LlmRequest) -> Result<Vec<Value>, LlmTransportError> {
        req.tools
            .iter()
            .map(|tool| {
                let parameters = Self::projected_schema(
                    &tool.input_schema,
                    &tool.input_schema_projections,
                    OpenAiSchemaProfile::ToolParameters,
                )?;
                Ok(json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters,
                        "strict": false,
                    },
                }))
            })
            .collect()
    }

    fn add_cache_control_to_text_content(message: &mut Value, cache_control: &Value) -> bool {
        let Some(content) = message.get_mut("content") else {
            return false;
        };
        if let Some(text) = content.as_str() {
            if text.is_empty() {
                return false;
            }
            *content = json!([{
                "type": "text",
                "text": text,
                "cache_control": cache_control,
            }]);
            return true;
        }
        let Some(parts) = content.as_array_mut() else {
            return false;
        };
        for part in parts.iter_mut().rev() {
            if part.get("type").and_then(Value::as_str) == Some("text") {
                part["cache_control"] = cache_control.clone();
                return true;
            }
        }
        false
    }

    fn add_cache_control_to_marked_text_content(
        message: &mut Value,
        cache_control: &Value,
    ) -> bool {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            return false;
        };
        for part in parts.iter_mut().rev() {
            let is_marked = part
                .get("__lash_cache_breakpoint")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if is_marked && part.get("type").and_then(Value::as_str) == Some("text") {
                part["cache_control"] = cache_control.clone();
                part.as_object_mut()
                    .map(|obj| obj.remove("__lash_cache_breakpoint"));
                return true;
            }
        }
        false
    }

    fn strip_internal_cache_markers(messages: &mut [Value]) {
        for message in messages {
            if let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) {
                for part in parts {
                    part.as_object_mut()
                        .map(|obj| obj.remove("__lash_cache_breakpoint"));
                }
            }
        }
    }

    fn apply_anthropic_cache_control(
        &self,
        req: &LlmRequest,
        messages: &mut [Value],
        tools: &mut [Value],
    ) {
        if !self.openrouter_claude_request(req) {
            Self::strip_internal_cache_markers(messages);
            return;
        }
        let Some(cache_control) = self.chat_cache_control_value() else {
            Self::strip_internal_cache_markers(messages);
            return;
        };

        for message in messages.iter_mut() {
            if matches!(
                message.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            ) {
                Self::add_cache_control_to_text_content(message, &cache_control);
                break;
            }
        }
        if let Some(last_tool) = tools.last_mut() {
            last_tool["cache_control"] = cache_control.clone();
        }
        let mut applied_explicit_breakpoint = false;
        for message in messages.iter_mut().rev() {
            if matches!(
                message.get("role").and_then(Value::as_str),
                Some("user" | "assistant")
            ) && Self::add_cache_control_to_marked_text_content(message, &cache_control)
            {
                applied_explicit_breakpoint = true;
                break;
            }
        }
        if !applied_explicit_breakpoint {
            for message in messages.iter_mut().rev() {
                if matches!(
                    message.get("role").and_then(Value::as_str),
                    Some("user" | "assistant")
                ) && Self::add_cache_control_to_text_content(message, &cache_control)
                {
                    break;
                }
            }
        }
        Self::strip_internal_cache_markers(messages);
    }

    fn build_chat_request_body(
        &self,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<Value, LlmTransportError> {
        let mut messages = Self::build_chat_messages(req);
        let mut tools = Self::build_chat_tools(req)?;
        self.apply_anthropic_cache_control(req, &mut messages, &mut tools);
        let mut body = json!({
            "model": req.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": Self::max_output_tokens(),
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
            body["tool_choice"] = json!(Self::tool_choice_value(&req.tool_choice));
            body["parallel_tool_calls"] = json!(true);
        }
        if stream {
            body["stream_options"] = json!({ "include_usage": true });
        }
        if let Some(variant) = req.model_variant.as_deref()
            && let Some(VariantRequestConfig::ReasoningEffort(effort)) =
                OpenAiModelPolicy::new(self.base_url.clone())
                    .request_variant_config(&req.model, variant)
            && effort != "none"
        {
            body["reasoning"] = json!({
                "effort": Self::clamp_reasoning_effort(&req.model, &effort),
            });
        }
        if let Some(output_spec) = &req.output_spec {
            body["response_format"] = match output_spec {
                LlmOutputSpec::JsonObject => json!({ "type": "json_object" }),
                LlmOutputSpec::JsonSchema(schema) => {
                    let projected = Self::projected_schema(
                        &schema.schema,
                        &[],
                        OpenAiSchemaProfile::StructuredOutput,
                    )?;
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.name,
                            "schema": projected,
                            "strict": schema.strict,
                        },
                    })
                }
            };
        }
        Ok(body)
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
        let cached_tokens = Self::parse_i64(
            usage
                .get("input_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .or_else(|| {
                    usage
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                })
                .or_else(|| usage.get("prompt_cache_hit_tokens")),
        );
        let cache_write_tokens = Self::parse_i64(
            usage
                .get("input_tokens_details")
                .and_then(|d| d.get("cache_write_tokens"))
                .or_else(|| {
                    usage
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cache_write_tokens"))
                }),
        );
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
            cached_input_tokens: if cache_write_tokens > 0 {
                cached_tokens.saturating_sub(cache_write_tokens).max(0)
            } else {
                cached_tokens
            },
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

    fn chat_message_text(message: &Value) -> String {
        match message.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                    Some("text") => part.get("text").and_then(Value::as_str),
                    _ => None,
                })
                .collect::<String>(),
            _ => String::new(),
        }
    }

    fn chat_response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
        let mut parts = Vec::new();
        let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return parts;
        };
        let Some(message) = choice.get("message") else {
            return parts;
        };
        for field in ["reasoning_content", "reasoning", "reasoning_text"] {
            if let Some(reasoning) = message.get(field).and_then(Value::as_str)
                && !reasoning.trim().is_empty()
            {
                parts.push(LlmOutputPart::Reasoning {
                    text: reasoning.to_string(),
                    signature: None,
                    redacted: false,
                    item_id: None,
                    encrypted_content: None,
                    summary: Vec::new(),
                });
                break;
            }
        }
        let text = Self::chat_message_text(message);
        if !text.is_empty() {
            parts.push(LlmOutputPart::Text {
                text,
                response_meta: None,
            });
        }
        let reasoning_details = message
            .get("reasoning_details")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|detail| {
                detail.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
                    && detail.get("id").and_then(Value::as_str).is_some()
                    && detail.get("data").and_then(Value::as_str).is_some()
            })
            .map(|detail| {
                (
                    detail
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    detail.to_string(),
                )
            })
            .collect::<HashMap<_, _>>();
        if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                let Some(function) = tool_call.get("function") else {
                    continue;
                };
                let Some(name) = function.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let arguments = function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}")
                    .to_string();
                parts.push(LlmOutputPart::ToolCall {
                    call_id: tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    signature: tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .and_then(|id| reasoning_details.get(id).cloned()),
                    tool_name: name.to_string(),
                    input_json: arguments,
                    item_id: None,
                });
            }
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

    fn process_chat_sse_event(
        raw: &str,
        state: &mut ChatStreamState,
    ) -> Result<(), LlmTransportError> {
        let raw = raw.trim();
        if raw.is_empty() || raw == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(raw).map_err(|e| {
            LlmTransportError::new(format!("Invalid Chat Completions SSE payload: {e}"))
                .with_raw(raw)
        })?;
        if event.get("error").is_some() {
            let retryable = event
                .get("error")
                .map(Self::responses_error_is_retryable)
                .unwrap_or(false);
            let message = event
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("OpenAI-compatible chat stream error");
            return Err(LlmTransportError::new(message)
                .retryable(retryable)
                .with_raw(event.to_string()));
        }
        if let Some(usage) = event.get("usage")
            && !usage.is_null()
        {
            state.provider_usage = Some(usage.clone());
            Self::merge_usage(&mut state.usage, &Self::usage_from_response_value(&event));
        }
        state.final_response = Some(event.clone());
        let Some(choices) = event.get("choices").and_then(Value::as_array) else {
            return Ok(());
        };
        for choice in choices {
            if let Some(usage) = choice.get("usage")
                && !usage.is_null()
            {
                state.provider_usage = Some(usage.clone());
                Self::merge_usage(
                    &mut state.usage,
                    &Self::usage_from_response_value(&json!({ "usage": usage })),
                );
            }
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                state.push_text_delta(content);
            }
            for field in ["reasoning_content", "reasoning", "reasoning_text"] {
                if let Some(reasoning) = delta.get(field).and_then(Value::as_str)
                    && !reasoning.is_empty()
                {
                    state.push_reasoning_delta(reasoning);
                    break;
                }
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    state.update_tool_call_delta(tool_call);
                }
            }
            if let Some(details) = delta.get("reasoning_details") {
                state.apply_reasoning_details(details);
            }
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

    fn parse_chat_sse_payload(
        payload: &str,
        state: &mut ChatStreamState,
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
                Self::process_chat_sse_event(&raw, state)?;
                event_lines.clear();
            }
        }
        if !event_lines.is_empty() {
            let raw = event_lines.join("\n");
            Self::process_chat_sse_event(&raw, state)?;
        }
        Ok(())
    }

    async fn complete_responses(
        &mut self,
        req: LlmRequest,
    ) -> Result<LlmResponse, LlmTransportError> {
        let stream_events = req.stream_events.clone();
        let provider_trace = req.provider_trace.clone();
        let timeouts = self.options.llm_timeouts();
        let body = self.build_responses_request_body(&req, stream_events.is_some())?;
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
            let headers = resp.headers().clone();
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
            return Err(LlmTransportError::new(message)
                .with_status(status.as_u16())
                .with_headers(&headers)
                .with_raw(text)
                .with_request_body(String::from_utf8_lossy(&request_body).into_owned())
                .retryable(status.as_u16() == 429 || status.as_u16() >= 500));
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

    async fn complete_chat_completions(
        &mut self,
        req: LlmRequest,
    ) -> Result<LlmResponse, LlmTransportError> {
        let stream_events = req.stream_events.clone();
        let provider_trace = req.provider_trace.clone();
        let timeouts = self.options.llm_timeouts();
        let body = self.build_chat_request_body(&req, stream_events.is_some())?;
        let body_bytes = serde_json::to_vec(&body).map_err(|e| {
            LlmTransportError::new(format!("Failed to serialize Chat Completions body: {e}"))
        })?;
        let request_body = request_body_snapshot_bytes(body_bytes);
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
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
            "OpenAI-compatible chat response start timed out",
        )
        .await?;

        let status = resp.status();
        if !status.is_success() {
            let headers = resp.headers().clone();
            let text = read_response_text(
                resp,
                timeouts.request_timeout,
                "OpenAI-compatible chat response body timed out",
            )
            .await
            .unwrap_or_default();
            let detail = extract_error_detail(&text);
            let message = detail
                .map(|detail| {
                    format!(
                        "OpenAI-compatible chat request failed with {}: {}",
                        status.as_u16(),
                        detail
                    )
                })
                .unwrap_or_else(|| {
                    format!(
                        "OpenAI-compatible chat request failed with {}",
                        status.as_u16()
                    )
                });
            return Err(LlmTransportError::new(message)
                .with_status(status.as_u16())
                .with_headers(&headers)
                .with_raw(text)
                .with_request_body(String::from_utf8_lossy(&request_body).into_owned())
                .retryable(status.as_u16() == 429 || status.as_u16() >= 500));
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
                "OpenAI-compatible chat response body timed out",
            )
            .await?;
            emit_provider_trace(provider_trace.as_ref(), "openai_compatible", &text);
            let mut state = ChatStreamState::default();
            let mut parsed_parts = None;
            if text.trim_start().starts_with("data:") || text.contains("\ndata:") {
                Self::parse_chat_sse_payload(&text, &mut state)?;
            } else {
                let value: Value = serde_json::from_str(&text).map_err(|e| {
                    LlmTransportError::new(format!("Invalid Chat Completions JSON: {e}"))
                        .with_raw(text.clone())
                })?;
                state.provider_usage = value.get("usage").cloned();
                state.usage = Self::usage_from_response_value(&value);
                let parts = Self::chat_response_parts_from_value(&value);
                state.full_text = parts
                    .iter()
                    .filter_map(|part| match part {
                        LlmOutputPart::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<String>();
                parsed_parts = Some(parts);
                state.final_response = Some(value);
            }
            let parts = parsed_parts.unwrap_or_else(|| state.parts());
            if !Self::has_response_content(&parts) {
                return Err(Self::empty_response_error(text));
            }
            if let Some(tx) = &stream_events {
                if state.usage != LlmUsage::default() {
                    tx.send(LlmStreamEvent::Usage(state.usage.clone()));
                }
                if !state.full_text.is_empty() {
                    tx.send(LlmStreamEvent::Delta(state.full_text.clone()));
                }
                for part in parts
                    .iter()
                    .filter(|part| matches!(part, LlmOutputPart::Reasoning { .. }))
                {
                    tx.send(LlmStreamEvent::Part(part.clone()));
                }
                for part in parts
                    .iter()
                    .filter(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
                {
                    tx.send(LlmStreamEvent::Part(part.clone()));
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

        let mut state = ChatStreamState::default();
        drive_sse_response(
            resp,
            timeouts.chunk_timeout,
            "OpenAI-compatible chat stream chunk timed out",
            |raw| {
                emit_provider_trace(provider_trace.as_ref(), "openai_compatible", raw);
                let prev_len = state.deltas.len();
                let prev_usage = state.usage.clone();
                Self::process_chat_sse_event(raw, &mut state)?;
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
                } else {
                    state.take_reasoning_deltas();
                }
                Ok(())
            },
        )
        .await?;

        let parts = state.parts();
        if !Self::has_response_content(&parts) {
            return Err(Self::empty_response_error(
                state
                    .final_response
                    .as_ref()
                    .map(Value::to_string)
                    .unwrap_or_default(),
            ));
        }
        if let Some(tx) = &stream_events {
            for part in parts
                .iter()
                .filter(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
            {
                tx.send(LlmStreamEvent::Part(part.clone()));
            }
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
}

impl OpenAiGenericProvider {
    pub fn into_components(self) -> ProviderComponents {
        let model_policy = std::sync::Arc::new(OpenAiModelPolicy::new(self.base_url.clone()));
        ProviderComponents::new(
            Box::new(self.clone()),
            Box::new(NoopProviderAuth),
            Box::new(NoopProviderReadiness::new()),
            Box::new(self),
            model_policy,
        )
    }
}

impl ProviderState for OpenAiGenericProvider {
    fn kind(&self) -> &'static str {
        "openai-compatible"
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
        if self.wire_api != OpenAiWireApi::Auto {
            map.insert(
                "wire_api".to_string(),
                serde_json::to_value(self.wire_api).unwrap_or(serde_json::Value::Null),
            );
        }
        if self.cache_retention != OpenAiCacheRetention::Short {
            map.insert(
                "cache_retention".to_string(),
                serde_json::to_value(self.cache_retention).unwrap_or(serde_json::Value::Null),
            );
        }
        serde_json::Value::Object(map)
    }

    fn clone_boxed(&self) -> Box<dyn ProviderState> {
        Box::new(self.clone())
    }
}

impl ProviderModelPolicy for OpenAiModelPolicy {
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
        if !self.supported_variants(model).contains(&variant) {
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
}

#[async_trait]
impl ProviderTransport for OpenAiGenericProvider {
    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        match self.resolved_wire_api(&req) {
            OpenAiWireApi::Responses | OpenAiWireApi::Auto => self.complete_responses(req).await,
            OpenAiWireApi::ChatCompletions => self.complete_chat_completions(req).await,
        }
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
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
    #[serde(default)]
    wire_api: OpenAiWireApi,
    #[serde(default)]
    cache_retention: OpenAiCacheRetention,
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
        "Any OpenAI-compatible API endpoint"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String> {
        let cfg: OpenAiProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(OpenAiGenericProvider {
            api_key: cfg.api_key,
            base_url: cfg.base_url,
            options: cfg.options,
            wire_api: cfg.wire_api,
            cache_retention: cfg.cache_retention,
            client: build_http_client(),
        }
        .into_components())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::llm::types::{LlmJsonSchema, LlmMessage, LlmToolSpec};
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
        let body = provider.build_responses_request_body(&req, true).unwrap();
        assert_eq!(body["instructions"], "system prompt");
        assert_eq!(body["stream"], true);
        assert!(body.get("messages").is_none());
        assert!(body.get("cache_control").is_none());
        assert_eq!(body["prompt_cache_key"], "session-1");
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    }

    #[test]
    fn wire_api_auto_routes_only_openrouter_claude_to_chat_completions() {
        let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
        let mut claude = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        claude.model = "anthropic/claude-sonnet-4.6".to_string();
        assert_eq!(
            provider.resolved_wire_api(&claude),
            OpenAiWireApi::ChatCompletions
        );

        let mut gpt = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        gpt.model = "openai/gpt-5.4".to_string();
        assert_eq!(provider.resolved_wire_api(&gpt), OpenAiWireApi::Responses);

        let local = OpenAiGenericProvider::new("key", "http://localhost:11434/v1");
        assert_eq!(local.resolved_wire_api(&claude), OpenAiWireApi::Responses);

        let forced = local.with_wire_api(OpenAiWireApi::ChatCompletions);
        assert_eq!(
            forced.resolved_wire_api(&gpt),
            OpenAiWireApi::ChatCompletions
        );
    }

    #[test]
    fn chat_body_uses_messages_and_not_responses_input() {
        let mut req = request(vec![
            LlmMessage::text(LlmRole::System, "system prompt"),
            LlmMessage::text(LlmRole::User, "hello"),
        ]);
        req.model = "anthropic/claude-sonnet-4.6".to_string();
        req.model_variant = Some("high".to_string());
        req.output_spec = Some(LlmOutputSpec::JsonObject);

        let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .build_chat_request_body(&req, true)
            .unwrap();

        assert!(body.get("input").is_none());
        assert!(body.get("instructions").is_none());
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["stream_options"], json!({ "include_usage": true }));
        assert_eq!(body["reasoning"], json!({ "effort": "high" }));
        assert_eq!(body["response_format"], json!({ "type": "json_object" }));
    }

    #[test]
    fn openrouter_claude_chat_body_marks_anthropic_cache_breakpoints() {
        let mut req = request(vec![
            LlmMessage::text(LlmRole::System, "stable system prompt"),
            LlmMessage::text(LlmRole::User, "dynamic tail"),
        ]);
        req.model = "anthropic/claude-sonnet-4.6".to_string();
        req.tools = Arc::new(vec![LlmToolSpec {
            name: "search".to_string(),
            description: "Search".to_string(),
            input_schema: json!({"type": "object"}),
            output_schema: json!({}),
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
        }]);

        let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .build_chat_request_body(&req, true)
            .unwrap();

        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(
            body["tools"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn chat_tools_use_projected_openai_schema_and_preserve_override() {
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        req.model = "anthropic/claude-sonnet-4.6".to_string();
        req.tools = Arc::new(vec![
            LlmToolSpec {
                name: "empty".to_string(),
                description: "Empty".to_string(),
                input_schema: json!({"type": "object"}),
                output_schema: json!({}),
                input_schema_projections: Vec::new(),
                output_schema_projections: Vec::new(),
            },
            LlmToolSpec {
                name: "override".to_string(),
                description: "Override".to_string(),
                input_schema: json!({"type": "object", "properties": {"raw": {"const": "x"}}}),
                output_schema: json!({}),
                input_schema_projections: vec![lash::SchemaProjectionOverride {
                    profile: OpenAiSchemaProfile::ToolParameters
                        .projection_id()
                        .to_string(),
                    schema: json!({
                        "type": "object",
                        "properties": { "raw": { "type": "string", "enum": ["x"] } }
                    }),
                }],
                output_schema_projections: Vec::new(),
            },
        ]);

        let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .build_chat_request_body(&req, true)
            .unwrap();

        assert_eq!(
            body["tools"][0]["function"]["parameters"]["properties"],
            json!({})
        );
        assert_eq!(
            body["tools"][1]["function"]["parameters"]["properties"]["raw"],
            json!({ "type": "string", "enum": ["x"] })
        );
    }

    #[test]
    fn structured_output_schema_is_projected_or_rejected_locally() {
        let mut req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
            name: "result".to_string(),
            schema: json!({
                "type": "object",
                "properties": { "summary": { "type": "string" } }
            }),
            strict: true,
        }));

        let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .build_responses_request_body(&req, false)
            .unwrap();
        assert_eq!(
            body["text"]["format"]["schema"]["required"],
            json!(["summary"])
        );
        assert_eq!(
            body["text"]["format"]["schema"]["additionalProperties"],
            false
        );

        req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
            name: "bad".to_string(),
            schema: json!({"type": "object", "allOf": []}),
            strict: true,
        }));
        let err = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .build_responses_request_body(&req, false)
            .unwrap_err();
        assert_eq!(err.kind, ProviderFailureKind::Validation);
        assert!(err.message.contains("allOf"));
    }

    #[test]
    fn openrouter_claude_chat_body_prefers_explicit_text_cache_breakpoint() {
        let mut req = request(vec![
            LlmMessage::text(LlmRole::System, "stable system prompt"),
            LlmMessage::new(
                LlmRole::User,
                vec![
                    LlmContentBlock::Text {
                        text: "stable history".into(),
                        response_meta: None,
                        cache_breakpoint: true,
                    },
                    LlmContentBlock::Text {
                        text: "dynamic current iteration".into(),
                        response_meta: None,
                        cache_breakpoint: false,
                    },
                ],
            ),
        ]);
        req.model = "anthropic/claude-sonnet-4.6".to_string();

        let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .build_chat_request_body(&req, true)
            .unwrap();

        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert!(
            body["messages"][1]["content"][1]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body["messages"][1]["content"][0]
                .get("__lash_cache_breakpoint")
                .is_none()
        );
    }

    #[test]
    fn cache_retention_none_removes_chat_cache_markers() {
        let mut req = request(vec![
            LlmMessage::text(LlmRole::System, "stable system prompt"),
            LlmMessage::text(LlmRole::User, "dynamic tail"),
        ]);
        req.model = "anthropic/claude-sonnet-4.6".to_string();

        let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .with_cache_retention(OpenAiCacheRetention::None)
            .build_chat_request_body(&req, true)
            .unwrap();

        assert!(body["messages"][0]["content"].is_string());
        assert!(body["messages"][1]["content"].is_string());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn cache_retention_long_uses_anthropic_ttl_on_chat_cache_markers() {
        let mut req = request(vec![
            LlmMessage::text(LlmRole::System, "stable system prompt"),
            LlmMessage::text(LlmRole::User, "dynamic tail"),
        ]);
        req.model = "anthropic/claude-sonnet-4.6".to_string();

        let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .with_cache_retention(OpenAiCacheRetention::Long)
            .build_chat_request_body(&req, true)
            .unwrap();

        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
    }

    #[test]
    fn responses_long_cache_retention_emits_openai_retention() {
        let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .with_cache_retention(OpenAiCacheRetention::Long);
        let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);

        let body = provider.build_responses_request_body(&req, true).unwrap();

        assert_eq!(body["prompt_cache_key"], "session-1");
        assert_eq!(body["prompt_cache_retention"], "24h");
        assert!(body.get("cache_control").is_none());
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
                cache_breakpoint: false,
            }],
        )]);
        let body = provider.build_responses_request_body(&req, false).unwrap();
        assert_eq!(body["input"][0]["type"], "message");
        assert_eq!(body["input"][0]["id"], "msg_1");
        assert_eq!(body["input"][0]["phase"], "final_answer");
    }

    #[test]
    fn legacy_assistant_text_gets_deterministic_id() {
        let provider = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL);
        let req = request(vec![LlmMessage::text(LlmRole::Assistant, "legacy")]);
        let body = provider.build_responses_request_body(&req, false).unwrap();
        assert_eq!(body["input"][0]["id"], "msg_lash_0_0");
        assert_eq!(body["input"][0]["status"], "completed");
        assert!(body["input"][0].get("phase").is_none());
    }

    #[test]
    fn local_base_url_omits_openai_only_fields() {
        let provider = OpenAiGenericProvider::new("key", "http://localhost:11434/v1");
        let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
        let body = provider.build_responses_request_body(&req, true).unwrap();
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
                "prompt_tokens_details": { "cached_tokens": 7, "cache_write_tokens": 5 },
                "completion_tokens_details": { "reasoning_tokens": 4 }
            }
        }));
        assert_eq!(chat_usage.input_tokens, 13);
        assert_eq!(chat_usage.output_tokens, 17);
        assert_eq!(chat_usage.cached_input_tokens, 2);
        assert_eq!(chat_usage.reasoning_tokens, 4);

        let write_only_usage =
            OpenAiGenericProvider::usage_from_response_value(&serde_json::json!({
                "usage": {
                    "prompt_tokens": 5353,
                    "completion_tokens": 433,
                    "prompt_tokens_details": { "cached_tokens": 3, "cache_write_tokens": 5353 }
                }
            }));
        assert_eq!(write_only_usage.cached_input_tokens, 0);
    }

    #[test]
    fn chat_body_replays_openrouter_reasoning_details_on_tool_calls() {
        let req = request(vec![LlmMessage::new(
            LlmRole::Assistant,
            vec![LlmContentBlock::ToolCall {
                call_id: "call_1".to_string(),
                tool_name: "lookup".to_string(),
                input_json: "{\"q\":\"x\"}".to_string(),
                item_id: None,
                signature: Some(
                    json!({
                        "type": "reasoning.encrypted",
                        "id": "call_1",
                        "data": "encrypted"
                    })
                    .to_string(),
                ),
            }],
        )]);

        let body = OpenAiGenericProvider::new("key", OPENROUTER_BASE_URL)
            .build_chat_request_body(&req, false)
            .unwrap();

        assert_eq!(
            body["messages"][0]["reasoning_details"][0],
            json!({
                "type": "reasoning.encrypted",
                "id": "call_1",
                "data": "encrypted"
            })
        );
    }

    #[test]
    fn non_streaming_chat_parser_captures_text_tool_and_usage() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "think",
                    "content": "hello",
                    "reasoning_details": [{
                        "type": "reasoning.encrypted",
                        "id": "call_1",
                        "data": "encrypted"
                    }],
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"q\":\"x\"}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 17,
                "prompt_tokens_details": { "cached_tokens": 2 }
            }
        });

        let parts = OpenAiGenericProvider::chat_response_parts_from_value(&value);
        let usage = OpenAiGenericProvider::usage_from_response_value(&value);

        assert!(matches!(&parts[0], LlmOutputPart::Reasoning { text, .. } if text == "think"));
        assert!(matches!(&parts[1], LlmOutputPart::Text { text, .. } if text == "hello"));
        assert!(matches!(
            &parts[2],
            LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                signature,
                ..
            } if call_id == "call_1"
                && tool_name == "lookup"
                && input_json == "{\"q\":\"x\"}"
                && signature.as_deref().is_some_and(|value| value.contains("encrypted"))
        ));
        assert_eq!(usage.input_tokens, 13);
        assert_eq!(usage.output_tokens, 17);
        assert_eq!(usage.cached_input_tokens, 2);
    }

    #[test]
    fn chat_stream_parser_captures_text_tool_done_and_usage() {
        let mut state = ChatStreamState::default();
        OpenAiGenericProvider::process_chat_sse_event(
            r#"{"choices":[{"delta":{"content":"Hi"}}]}"#,
            &mut state,
        )
        .unwrap();
        OpenAiGenericProvider::process_chat_sse_event(
            r#"{"choices":[{"delta":{"reasoning_content":"Think"}}]}"#,
            &mut state,
        )
        .unwrap();
        OpenAiGenericProvider::process_chat_sse_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}"#,
            &mut state,
        )
        .unwrap();
        OpenAiGenericProvider::process_chat_sse_event(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]}}]}"#,
            &mut state,
        )
        .unwrap();
        OpenAiGenericProvider::process_chat_sse_event(
            r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","id":"call_1","data":"encrypted"}]}}],"usage":{"prompt_tokens":9,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2}}}"#,
            &mut state,
        )
        .unwrap();
        OpenAiGenericProvider::process_chat_sse_event(
            r#"{"choices":[{"usage":{"prompt_tokens":11,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":2}}}]}"#,
            &mut state,
        )
        .unwrap();
        OpenAiGenericProvider::process_chat_sse_event("[DONE]", &mut state).unwrap();

        assert_eq!(state.full_text, "Hi");
        assert_eq!(state.reasoning_text, "Think");
        let parts = state.parts();
        assert!(matches!(&parts[0], LlmOutputPart::Reasoning { text, .. } if text == "Think"));
        assert!(matches!(&parts[1], LlmOutputPart::Text { text, .. } if text == "Hi"));
        assert!(matches!(
            &parts[2],
            LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                signature,
                ..
            } if call_id == "call_1"
                && tool_name == "lookup"
                && input_json == "{\"q\":\"x\"}"
                && signature.as_deref().is_some_and(|value| value.contains("encrypted"))
        ));
        assert_eq!(state.usage.input_tokens, 11);
        assert_eq!(state.usage.output_tokens, 5);
        assert_eq!(state.usage.cached_input_tokens, 2);
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
