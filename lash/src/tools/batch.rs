use crate::{ToolDefinition, ToolExecutionMode, ToolParam};

pub(crate) fn batch_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "batch".to_string(),
        description: "Execute multiple independent tool calls concurrently. Calls start in parallel; ordering is not guaranteed.".to_string(),
        params: vec![ToolParam {
            name: "tool_calls".to_string(),
            r#type: "list".to_string(),
            description: "Array of 1-25 objects like { tool: \"read_file\", parameters: { path: \"src/main.rs\" } }. Use only for independent calls. Do not include another batch call.".to_string(),
            default_value: None,
            required: true,
        }],
        returns: "dict".to_string(),
        examples: vec![
            r#"batch(tool_calls=[{"tool":"read_file","parameters":{"path":"src/main.rs"}},{"tool":"grep","parameters":{"query":"ToolProvider lash/src/"}}])"#.to_string(),
        ],
        enabled: true,
        injected: true,
        input_schema_override: None,
        output_schema_override: None,
        execution_mode: ToolExecutionMode::Parallel,
    }
}
