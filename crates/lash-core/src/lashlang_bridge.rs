use lashlang::{ExecutionHostError, Value as LashlangValue};

pub fn process_handle_json(id: &str) -> serde_json::Value {
    serde_json::json!({
        "__handle__": "process",
        "id": id,
    })
}

pub fn lashlang_value_to_json(
    value: &LashlangValue,
) -> Result<serde_json::Value, ExecutionHostError> {
    serde_json::to_value(value)
        .map_err(|err| ExecutionHostError::new(format!("failed to serialize value: {err}")))
}

pub fn protocol_tool_reply_to_lashlang_value(
    reply: crate::ToolInvocationReply,
) -> Result<LashlangValue, ExecutionHostError> {
    protocol_tool_output_to_lashlang_value(&reply.output)
}

pub fn protocol_tool_output_to_lashlang_value(
    output: &crate::ToolCallOutput,
) -> Result<LashlangValue, ExecutionHostError> {
    let value = output.value_for_projection();
    if output.is_success() {
        Ok(lashlang::from_json(value))
    } else {
        Err(ExecutionHostError::new(json_error_message(value)))
    }
}

pub fn process_event_payload(
    value: &LashlangValue,
) -> Result<serde_json::Value, ExecutionHostError> {
    Ok(serde_json::json!({
        "value": lashlang_value_to_json(value)?,
        "text": value.to_string(),
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

pub fn json_error_message(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text,
        other => other.to_string(),
    }
}
