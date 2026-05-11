use crate::support::*;

impl OpenAiCompatibleProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            options: ProviderOptions::default(),
            cache_retention: OpenAiCacheRetention::Short,
            client: DEFAULT_HTTP_CLIENT.clone(),
        }
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.options = options;
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

    pub(crate) fn build_responses_request_body(
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
                OpenAiDirectModelPolicy.request_variant_config(&req.model, variant)
            && effort != "none"
        {
            let mut reasoning = json!({
                "effort": Self::clamp_reasoning_effort(&req.model, &effort),
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

    pub(crate) fn build_chat_request_body(
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

    pub(crate) fn usage_from_response_value(value: &Value) -> LlmUsage {
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

    pub(crate) fn chat_response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
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
            || value
                .get("code")
                .or_else(|| value.get("type"))
                .or_else(|| value.get("status"))
                .and_then(|v| v.as_str())
                .is_some_and(|code| {
                    matches!(
                        code,
                        "server_error"
                            | "internal_server_error"
                            | "service_unavailable"
                            | "temporarily_unavailable"
                            | "overloaded"
                            | "rate_limit_exceeded"
                    )
                })
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

    pub(crate) fn process_chat_sse_event(
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
        emit_provider_request_trace(provider_trace.as_ref(), "responses", &body);
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
                if self.options.thinking.expose {
                    for part in &parts {
                        if let LlmOutputPart::Reasoning { text, .. } = part
                            && !text.is_empty()
                        {
                            tx.send(LlmStreamEvent::ReasoningDelta(text.clone()));
                        }
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
        let expose_thinking = self.options.thinking.expose;
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
                        if expose_thinking {
                            tx.send(LlmStreamEvent::ReasoningDelta(delta));
                        }
                    }
                    for part in emitted_parts.drain(..) {
                        if matches!(part, LlmOutputPart::Reasoning { .. }) && !expose_thinking {
                            continue;
                        }
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
        emit_provider_request_trace(provider_trace.as_ref(), "chat/completions", &body);
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
                if self.options.thinking.expose {
                    for part in parts
                        .iter()
                        .filter(|part| matches!(part, LlmOutputPart::Reasoning { .. }))
                    {
                        tx.send(LlmStreamEvent::Part(part.clone()));
                    }
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
        let expose_thinking = self.options.thinking.expose;
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
                        if expose_thinking {
                            tx.send(LlmStreamEvent::ReasoningDelta(delta));
                        }
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

impl OpenAiCompatibleProvider {
    pub fn into_components(self) -> ProviderComponents {
        let model_policy = std::sync::Arc::new(OpenAiModelPolicy::new(self.base_url.clone()));
        ProviderComponents::new(Box::new(self.clone()), Box::new(self), model_policy)
    }
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAiCompatibleProvider::new(api_key, OPENAI_BASE_URL),
        }
    }

    pub fn with_options(mut self, options: ProviderOptions) -> Self {
        self.inner.options = options;
        self
    }

    pub fn with_cache_retention(mut self, retention: OpenAiCacheRetention) -> Self {
        self.inner.cache_retention = retention;
        self
    }

    pub fn with_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.inner.client = (*client).clone();
        self
    }

    pub fn into_components(self) -> ProviderComponents {
        ProviderComponents::new(
            Box::new(self.clone()),
            Box::new(self),
            std::sync::Arc::new(OpenAiDirectModelPolicy),
        )
    }

    #[cfg(test)]
    pub(crate) fn build_responses_request_body(
        &self,
        req: &LlmRequest,
        stream: bool,
    ) -> Result<Value, LlmTransportError> {
        self.inner.build_responses_request_body(req, stream)
    }
}

impl ProviderState for OpenAiCompatibleProvider {
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

impl ProviderState for OpenAiProvider {
    fn kind(&self) -> &'static str {
        "openai"
    }

    fn options(&self) -> ProviderOptions {
        self.inner.options.clone()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.inner.options = options;
    }

    fn serialize_config(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "api_key".to_string(),
            serde_json::Value::String(self.inner.api_key.clone()),
        );
        if !self.inner.options.is_default() {
            map.insert(
                "options".to_string(),
                serde_json::to_value(&self.inner.options).unwrap_or(serde_json::Value::Null),
            );
        }
        if self.inner.cache_retention != OpenAiCacheRetention::Short {
            map.insert(
                "cache_retention".to_string(),
                serde_json::to_value(self.inner.cache_retention).unwrap_or(serde_json::Value::Null),
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

#[derive(Clone, Debug)]
struct OpenAiDirectModelPolicy;

impl ProviderModelPolicy for OpenAiDirectModelPolicy {
    fn default_model(&self) -> &str {
        "gpt-5.4"
    }

    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        let id = model_id(model).to_ascii_lowercase();
        if id.starts_with("gpt-5") || id.starts_with("o") {
            OPENROUTER_REASONING_VARIANTS
        } else {
            &[]
        }
    }

    fn default_model_variant(&self, model: &str) -> Option<&'static str> {
        self.supported_variants(model)
            .contains(&"medium")
            .then_some("medium")
    }

    fn request_variant_config(&self, model: &str, variant: &str) -> Option<VariantRequestConfig> {
        if !self.supported_variants(model).contains(&variant) {
            return None;
        }
        Some(VariantRequestConfig::ReasoningEffort(
            OpenAiCompatibleProvider::clamp_reasoning_effort(model, variant),
        ))
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
                model: "gpt-5.4".to_string(),
                variant: Some("high".to_string()),
            }),
            _ => None,
        }
    }
}

#[async_trait]
impl ProviderTransport for OpenAiCompatibleProvider {
    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        self.complete_chat_completions(req).await
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl ProviderTransport for OpenAiProvider {
    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        self.inner.complete_responses(req).await
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
        Box::new(self.clone())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenAiCompatibleProviderConfig {
    api_key: String,
    base_url: String,
    #[serde(default)]
    options: ProviderOptions,
    #[serde(default)]
    cache_retention: OpenAiCacheRetention,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenAiProviderConfig {
    api_key: String,
    #[serde(default)]
    options: ProviderOptions,
    #[serde(default)]
    cache_retention: OpenAiCacheRetention,
}

pub struct OpenAiCompatibleProviderFactory;
pub struct OpenAiProviderFactory;

impl OpenAiCompatibleProviderFactory {
    pub fn register() {
        lash_core::register_provider_factory(std::sync::Arc::new(Self));
    }
}

impl OpenAiProviderFactory {
    pub fn register() {
        lash_core::register_provider_factory(std::sync::Arc::new(Self));
    }
}

impl ProviderFactory for OpenAiProviderFactory {
    fn kind(&self) -> &'static str {
        "openai"
    }

    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String> {
        let cfg: OpenAiProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(OpenAiProvider {
            inner: OpenAiCompatibleProvider {
                api_key: cfg.api_key,
                base_url: OPENAI_BASE_URL.to_string(),
                options: cfg.options,
                cache_retention: cfg.cache_retention,
                client: build_http_client(),
            },
        }
        .into_components())
    }
}

impl ProviderFactory for OpenAiCompatibleProviderFactory {
    fn kind(&self) -> &'static str {
        "openai-compatible"
    }
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String> {
        let cfg: OpenAiCompatibleProviderConfig =
            serde_json::from_value(config).map_err(|err| err.to_string())?;
        Ok(OpenAiCompatibleProvider {
            api_key: cfg.api_key,
            base_url: cfg.base_url,
            options: cfg.options,
            cache_retention: cfg.cache_retention,
            client: build_http_client(),
        }
        .into_components())
    }
}
