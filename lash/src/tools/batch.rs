use crate::{ToolDefinition, ToolExecutionMode};

use super::object_schema;

pub fn batch_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "batch",
        "Execute up to 25 independent tool calls concurrently. Calls start in parallel; ordering is not guaranteed. Anything past 25 is dropped.",
        object_schema(
            serde_json::json!({
                "tool_calls": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": { "type": "string" },
                            "parameters": { "type": "object", "additionalProperties": true }
                        },
                        "required": ["tool", "parameters"],
                        "additionalProperties": false
                    },
                    "description": "Array of 1-25 objects like { tool: \"read_file\", parameters: { path: \"src/main.rs\" } }. Use only for independent calls. Do not include another batch call."
                }
            }),
            &["tool_calls"],
        ),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
            r#"batch(tool_calls=[{"tool":"read_file","parameters":{"path":"src/main.rs"}},{"tool":"grep","parameters":{"query":"ToolProvider lash/src/"}}])"#.to_string(),
        ])
    .with_discovery(crate::tools::discovery_metadata("runtime", &["parallel_tools"]))
    .with_execution_mode(ToolExecutionMode::Parallel)
}
