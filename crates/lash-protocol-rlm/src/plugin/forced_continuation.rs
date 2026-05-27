use std::sync::Arc;

use lash_core::llm::types::{
    LlmContentBlock, LlmJsonSchema, LlmMessage, LlmOutputSpec, LlmRequest, LlmRole, LlmToolChoice,
};
use lash_core::plugin::{PluginError, ProtocolBeforeLlmCallContext};

pub(super) fn process_support_unavailable(err: &PluginError) -> bool {
    matches!(
        err,
        PluginError::Session(message)
            if message.contains("process") && message.contains("unavailable")
    )
}

pub(super) async fn forced_continue_as_args(
    ctx: &ProtocolBeforeLlmCallContext<'_>,
    request: &LlmRequest,
    threshold: usize,
    observed_tokens: usize,
) -> Result<serde_json::Value, PluginError> {
    let mut fallback_request = request.clone();
    fallback_request.tools = Arc::new(Vec::new());
    fallback_request.tool_choice = LlmToolChoice::None;
    fallback_request.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
        name: "continue_as".to_string(),
        schema: crate::control_tools::continue_as_input_schema(),
        strict: true,
    }));
    fallback_request.stream_events = None;
    fallback_request.provider_trace = None;
    fallback_request.messages.push(LlmMessage::new(
        LlmRole::User,
        vec![LlmContentBlock::Text {
            text: forced_continue_as_instruction(threshold, observed_tokens).into(),
            response_meta: None,
            cache_breakpoint: false,
        }],
    ));
    let completion = ctx
        .direct_llm_completion(fallback_request, "continue_as_forced_context_fallback")
        .await
        .map_err(|err| {
            PluginError::Session(format!(
                "forced continue_as fallback LLM call failed: {err}"
            ))
        })?;
    parse_forced_continue_as_args(&completion.response.full_text)
}

fn forced_continue_as_instruction(threshold: usize, observed_tokens: usize) -> String {
    format!(
        "Context budget is above the forced continuation threshold ({observed_tokens} observed, threshold {threshold}). Produce fresh-context continuation arguments as JSON matching the provided schema. Set out the task at hand in `task`. Put only necessary state in `seed`, including only live process handles the successor must await. Live handles omitted from `seed` are not carried forward. Leave bulky logs, transcripts, raw command output, and repeated context behind. Prefer variable names, file paths, projected references, and compact summaries over copying large values. Omit `seed` when no extra state is needed."
    )
}

pub(super) fn parse_forced_continue_as_args(text: &str) -> Result<serde_json::Value, PluginError> {
    let value: serde_json::Value = serde_json::from_str(text.trim()).map_err(|err| {
        PluginError::Session(format!(
            "forced continue_as fallback returned invalid JSON: {err}"
        ))
    })?;
    let validator =
        jsonschema::JSONSchema::compile(&crate::control_tools::continue_as_input_schema())
            .map_err(|err| {
                PluginError::Session(format!(
                    "failed to compile forced continue_as fallback schema: {err}"
                ))
            })?;
    if let Err(errors) = validator.validate(&value) {
        let messages = errors.map(|err| err.to_string()).collect::<Vec<_>>();
        return Err(PluginError::Session(format!(
            "forced continue_as fallback returned schema-invalid JSON: {}",
            messages.join("; ")
        )));
    }
    Ok(value)
}
