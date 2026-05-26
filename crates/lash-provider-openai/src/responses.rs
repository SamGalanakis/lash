use crate::support::*;
use lash_core::llm::types::{ProviderReasoningReplay, ProviderReplayMeta};

impl OpenAiCompatibleProvider {
    #[cfg(test)]
    pub(crate) fn usage_from_response_value(value: &Value) -> LlmUsage {
        usage_from_response_value(value)
    }

    pub(crate) fn role_name(role: &LlmRole) -> &'static str {
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

    pub(crate) fn response_text_meta_from_message_item(item: &Value) -> ResponseTextMeta {
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
                    LlmContentBlock::Reasoning { text, replay, .. } => {
                        Self::flush_pending_content(
                            &mut pending_content,
                            &mut input,
                            role,
                            matches!(msg.role, LlmRole::User),
                            pending_meta.take(),
                            message_index,
                            pending_part_index,
                        );
                        let Some(blob) = replay
                            .as_ref()
                            .and_then(|meta| meta.encrypted_content.as_deref())
                        else {
                            continue;
                        };
                        let summary = replay
                            .as_ref()
                            .map(|meta| meta.summary.as_slice())
                            .unwrap_or_default();
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
                        if let Some(id) = replay.as_ref().and_then(|meta| meta.item_id.as_deref())
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
                        replay,
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
                        if let Some(id) = replay.as_ref().and_then(|meta| meta.item_id.as_deref()) {
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

    pub(crate) fn projected_schema(
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

    pub(crate) fn build_tools(req: &LlmRequest) -> Result<Vec<Value>, LlmTransportError> {
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

    pub(crate) fn tool_choice_value(choice: &LlmToolChoice) -> &'static str {
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

    pub(crate) fn output_token_cap(&self, req: &LlmRequest) -> u64 {
        req.generation
            .output_token_cap_u64()
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
    }

    pub(crate) fn build_responses_request_body(
        &self,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<Value, LlmTransportError> {
        validate_image_attachments(
            req,
            &["image/jpeg", "image/png", "image/gif", "image/webp"],
            "OpenAI Responses",
        )?;
        let tools = Self::build_tools(req)?;
        let (instructions, input) = Self::build_input(req);
        let mut body = json!({
            "model": req.model,
            "instructions": instructions,
            "input": input,
            "tools": tools,
            "stream": stream,
            "max_output_tokens": self.output_token_cap(req),
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
                OpenAiDirectModelPolicy.request_variant_config(&req.model, variant)
            && effort != "none"
        {
            let mut reasoning = json!({
                "effort": clamp_reasoning_effort(&req.model, &effort),
            });
            if self.options.thinking.expose {
                reasoning["summary"] = json!("auto");
            }
            body["reasoning"] = reasoning;
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
        if self.options.cache_retention != CacheRetention::None
            && let Some(session_id) = req.session_id.as_deref()
        {
            body["prompt_cache_key"] = json!(session_id);
        }
        if self.options.cache_retention == CacheRetention::Long
            && Self::supports_openai_request_fields(&self.base_url)
        {
            body["prompt_cache_retention"] = json!("24h");
        }
        Ok(body)
    }

    pub(crate) fn message_text_from_item(item: &Value) -> String {
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

    pub(crate) fn response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
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
                            replay: Some(ProviderReasoningReplay {
                                item_id: item
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .map(str::to_string),
                                encrypted_content: item
                                    .get("encrypted_content")
                                    .and_then(|v| v.as_str())
                                    .map(str::to_string),
                                signature: None,
                                redacted: false,
                                summary,
                            }),
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
                            replay: item.get("id").and_then(|v| v.as_str()).map(|id| {
                                ProviderReplayMeta {
                                    item_id: Some(id.to_string()),
                                    opaque: None,
                                }
                            }),
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

    pub(crate) fn process_sse_event(
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
                .map(responses_error_is_retryable)
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
            merge_usage(&mut state.usage, &usage_from_response_value(resp));
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
                        .retryable(responses_error_is_retryable(&error_value))
                        .with_raw(event.to_string()),
                );
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn parse_sse_payload(
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

#[derive(Clone, Debug, Default)]
pub(crate) struct ResponsesStreamingToolCall {
    pub(crate) call_id: String,
    pub(crate) tool_name: String,
    pub(crate) input_json: String,
    pub(crate) item_id: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ResponsesStreamState {
    pub(crate) full_text: String,
    pub(crate) pending_text_deltas: Vec<String>,
    pub(crate) parts: Vec<LlmOutputPart>,
    pub(crate) usage: LlmUsage,
    pub(crate) provider_usage: Option<Value>,
    pub(crate) current_text_part: Option<usize>,
    pub(crate) current_reasoning_part: Option<usize>,
    pub(crate) reasoning_deltas: Vec<String>,
    pub(crate) tool_calls: HashMap<String, ResponsesStreamingToolCall>,
    pub(crate) final_response: Option<Value>,
}

impl ResponsesStreamState {
    pub(crate) fn begin_message(&mut self, item: Option<&Value>) {
        let meta = item.map(OpenAiCompatibleProvider::response_text_meta_from_message_item);
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Text {
            text: String::new(),
            response_meta: meta,
        });
        self.current_text_part = Some(index);
    }

    pub(crate) fn finish_message(&mut self, item: Option<&Value>) {
        if let Some(item) = item {
            let text = OpenAiCompatibleProvider::message_text_from_item(item);
            let meta = OpenAiCompatibleProvider::response_text_meta_from_message_item(item);
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

    pub(crate) fn push_text_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        let index = self.ensure_text_part_index();
        if let Some(LlmOutputPart::Text { text, .. }) = self.parts.get_mut(index) {
            text.push_str(piece);
        }
        self.pending_text_deltas.push(piece.to_string());
        self.recompute_full_text();
    }

    pub(crate) fn ensure_text_part_index(&mut self) -> usize {
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

    pub(crate) fn recompute_full_text(&mut self) {
        self.full_text.clear();
        for part in &self.parts {
            if let LlmOutputPart::Text { text, .. } = part {
                self.full_text.push_str(text);
            }
        }
    }

    pub(crate) fn begin_reasoning_part(&mut self) {
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Reasoning {
            text: String::new(),
            replay: None,
        });
        self.current_reasoning_part = Some(index);
    }

    pub(crate) fn push_reasoning_delta(&mut self, delta: &str) {
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

    pub(crate) fn finish_reasoning_part(&mut self) {
        if let Some(index) = self.current_reasoning_part.take()
            && let Some(LlmOutputPart::Reasoning { text, .. }) = self.parts.get_mut(index)
        {
            let trimmed = text.trim_end();
            if trimmed.len() != text.len() {
                *text = trimmed.to_string();
            }
        }
    }

    pub(crate) fn finalize_reasoning_item(&mut self, item: &Value) {
        let Some((_, part)) = self
            .parts
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(_, part)| matches!(part, LlmOutputPart::Reasoning { .. }))
        else {
            return;
        };
        let LlmOutputPart::Reasoning { replay, .. } = part else {
            return;
        };
        let meta = replay.get_or_insert_with(ProviderReasoningReplay::default);
        if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
            meta.item_id = Some(id.to_string());
        }
        if let Some(blob) = item.get("encrypted_content").and_then(|v| v.as_str()) {
            meta.encrypted_content = Some(blob.to_string());
        }
        if let Some(arr) = item.get("summary").and_then(|v| v.as_array()) {
            let texts = arr
                .iter()
                .filter_map(|entry| entry.get("text").and_then(|v| v.as_str()).map(String::from))
                .collect::<Vec<_>>();
            if !texts.is_empty() {
                meta.summary = texts;
            }
        }
    }

    pub(crate) fn update_tool_call_from_item(&mut self, item: &Value) -> Option<String> {
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

    pub(crate) fn push_tool_call_delta(&mut self, item_id: &str, delta: &str) {
        if item_id.is_empty() || delta.is_empty() {
            return;
        }
        self.tool_calls
            .entry(item_id.to_string())
            .or_default()
            .input_json
            .push_str(delta);
    }

    pub(crate) fn set_tool_call_arguments(&mut self, item_id: &str, arguments: &str) {
        if item_id.is_empty() {
            return;
        }
        self.tool_calls
            .entry(item_id.to_string())
            .or_default()
            .input_json = arguments.to_string();
    }

    pub(crate) fn finish_tool_call(&mut self, item: &Value) -> Option<LlmOutputPart> {
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
            replay: (!tool_call.item_id.is_empty()).then_some(ProviderReplayMeta {
                item_id: Some(tool_call.item_id),
                opaque: None,
            }),
        };
        self.parts.push(part.clone());
        Some(part)
    }

    pub(crate) fn take_reasoning_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reasoning_deltas)
    }

    pub(crate) fn take_text_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_text_deltas)
    }

    pub(crate) fn response_parts(&self) -> Vec<LlmOutputPart> {
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
            parts = OpenAiCompatibleProvider::response_parts_from_value(final_response);
        }
        parts
    }
}
