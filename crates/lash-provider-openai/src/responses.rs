use crate::responses_shared as shared;
use crate::support::*;

const PROVIDER: &str = "OpenAI-compatible";

impl OpenAiCompatibleProvider {
    #[cfg(test)]
    pub(crate) fn usage_from_response_value(value: &Value) -> LlmUsage {
        shared::usage_from_response_value(value)
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
                let phase = match phase {
                    ResponseTextPhase::Commentary => "commentary",
                    ResponseTextPhase::FinalAnswer => "final_answer",
                };
                item["phase"] = json!(phase);
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

            let role = shared::role_name(&msg.role);
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
                            pending_content.push(shared::input_image_part(att));
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

    pub(crate) fn build_tools(req: &LlmRequest) -> Result<Vec<Value>, LlmTransportError> {
        shared::build_tools(PROVIDER, req)
    }

    pub(crate) fn projected_schema(
        canonical: &Value,
        overrides: &[SchemaProjectionOverride],
        profile: OpenAiSchemaProfile,
    ) -> Result<Value, LlmTransportError> {
        shared::projected_schema(PROVIDER, canonical, overrides, profile)
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

    pub(crate) fn build_responses_request_body(
        &self,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<Value, LlmTransportError> {
        validate_image_attachments(req, OPENAI_IMAGE_MIMES, "OpenAI Responses")?;
        let tools = Self::build_tools(req)?;
        let (instructions, input) = Self::build_input(req);
        let policy = resolve_generation_policy(
            &req.generation,
            &self.options,
            DEFAULT_MAX_OUTPUT_TOKENS,
            (),
        );
        let mut body = json!({
            "model": req.model,
            "instructions": instructions,
            "input": input,
            "tools": tools,
            "stream": stream,
            "max_output_tokens": policy.max_output_tokens,
        });
        if !req.tools.is_empty() {
            body["tool_choice"] = json!(shared::tool_choice_value(&req.tool_choice));
        }
        if Self::supports_openai_request_fields(&self.base_url) {
            body["include"] = json!(["reasoning.encrypted_content"]);
            body["store"] = json!(false);
            body["parallel_tool_calls"] = json!(!req.tools.is_empty());
            body["text"] = json!({"verbosity": "medium"});
        }
        if let Some(variant) = req.model_variant.as_deref()
            && let Some(effort) = OpenAiDirectModelPolicy.reasoning_effort(&req.model, variant)
            && effort != "none"
        {
            let mut reasoning = json!({
                "effort": clamp_reasoning_effort(&req.model, &effort),
            });
            if policy.expose_thinking {
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
        if policy.cache_retention != CacheRetention::None
            && let Some(session_id) = req.session_id.as_deref()
        {
            body["prompt_cache_key"] = json!(session_id);
        }
        if policy.cache_retention == CacheRetention::Long
            && Self::supports_openai_request_fields(&self.base_url)
        {
            body["prompt_cache_retention"] = json!("24h");
        }
        Ok(body)
    }

    pub(crate) fn response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
        shared::response_parts_from_value(value)
    }

    pub(crate) fn process_sse_event(
        raw: &str,
        state: &mut ResponsesStreamState,
        emitted_parts: Option<&mut Vec<LlmOutputPart>>,
    ) -> Result<(), LlmTransportError> {
        shared::process_sse_event(PROVIDER, raw, state, emitted_parts)
    }

    pub(crate) fn parse_sse_payload(
        payload: &str,
        state: &mut ResponsesStreamState,
    ) -> Result<(), LlmTransportError> {
        shared::parse_sse_payload(PROVIDER, payload, state)
    }
}
