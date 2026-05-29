use lash_core::{ToolAgentSurface, ToolAvailabilityConfig, ToolDefinition, ToolScheduling};

pub fn batch_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:batch",
        "batch",
        "Execute up to 25 independent tool calls concurrently. Calls start in parallel; ordering is not guaranteed. Calls past index 25 are rejected.",
        object_schema(
            serde_json::json!({
                "tool_calls": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 25,
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": { "type": "string" },
                            "parameters": { "type": "object", "additionalProperties": true }
                        },
                        "required": ["tool", "parameters"],
                        "additionalProperties": false
                    },
                    "description": "Array of 1-25 objects like { tool: \"read_file\", parameters: { path: \"src/main.rs\" } }. Use only for independent calls. Do not include another batch call. More than 25 calls is rejected as a tool error."
                }
            }),
            &["tool_calls"],
        ),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
            r#"await tools.batch({ tool_calls: [{ tool: "read_file", parameters: { path: "src/main.rs" } }, { tool: "grep", parameters: { query: "ToolProvider crates/lash/src/" } }] })?"#.to_string(),
        ])
    .with_agent_surface(
        ToolAgentSurface::new(["tools"], "batch").with_aliases(["parallel_tools"]),
    )
    .with_availability(ToolAvailabilityConfig::callable())
    .with_scheduling(ToolScheduling::Parallel)
}

fn object_schema(properties: serde_json::Value, required: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}
