use crate::{ToolDefinition, ToolExecutionMode, ToolParam};

pub fn batch_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "batch".to_string(),
        description: "Execute up to 25 independent tool calls concurrently. Calls start in parallel; ordering is not guaranteed. Anything past 25 is dropped.".to_string(),
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
        availability: crate::ToolAvailabilityConfig::documented(),
        activation: crate::ToolActivation::Always,
        availability_override: None,
        input_schema_override: None,
        output_schema_override: None,
        discovery: crate::tools::discovery_metadata("runtime", &["parallel_tools"]),
        execution_mode: ToolExecutionMode::Parallel,
    }
}
