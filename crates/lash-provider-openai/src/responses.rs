use crate::responses_shared as shared;
use crate::support::*;

const PROVIDER: &str = "OpenAI-compatible";

impl OpenAiCompatibleProvider {
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
        let (instructions, input) =
            shared::build_responses_input(req, shared::ResponsesInputOptions::OPENAI);
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
