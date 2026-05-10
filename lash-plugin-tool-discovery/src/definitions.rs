use lash::{ToolActivation, ToolAvailabilityConfig, ToolDefinition, ToolExecutionMode};
use serde_json::json;

pub(crate) fn search_tools_definition() -> ToolDefinition {
    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct SearchToolsArgs {
        #[schemars(
            description = "Concise tool search query. Prefer keywords and short intent phrases with the app/domain, action, object, qualifiers, and important fields; for multi-constraint tasks include every constraint, such as \"spotify liked songs library\"."
        )]
        query: String,
        #[schemars(description = "Optional namespace filter, such as \"appworld\".")]
        namespace: Option<NamespaceFilter>,
        #[schemars(range(min = 1, max = 100))]
        #[schemars(description = "Maximum number of results to return. Defaults to 10.")]
        limit: Option<usize>,
        #[schemars(description = "Exact tool name or names to exclude from results.")]
        exclude: Option<NameFilter>,
    }

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    #[serde(untagged)]
    enum NamespaceFilter {
        One(String),
        Many(Vec<String>),
    }

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    #[serde(untagged)]
    enum NameFilter {
        One(String),
        Many(Vec<String>),
    }

    ToolDefinition::raw(
        "search_tools",
        "Search catalogued tool names, namespaces, aliases, descriptions, signatures, return fields, and examples. Use this when the tool you need is not showcased in the prompt. Query with concise keywords and short intent phrases: include the app/domain, action, object, qualifiers, and important fields or constraints. For initial exploration, print only result names and signatures; inspect descriptions and examples only when you need to choose between close matches or learn call idioms.",
        serde_json::to_value(schemars::schema_for!(SearchToolsArgs))
            .unwrap_or_else(|_| json!({})),
        json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "signature": {
                        "type": "string",
                        "description": "Callable signature with successful return type plus compact parameter and return-field details."
                    },
                    "description": { "type": "string" },
                    "examples": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["name", "signature"],
                "additionalProperties": false
            }
        }),
    )
        .with_examples(vec![
            "search_tools(query=\"spotify liked songs library\", namespace=\"appworld\")".into(),
            "search_tools(query=\"spotify song details play_count genre title song_id\", namespace=\"appworld\")".into(),
            "search_tools(query=\"venmo send money private payment_card receiver_email\", namespace=\"appworld\")".into(),
        ])
        .with_availability(ToolAvailabilityConfig::showcased())
        .with_activation(ToolActivation::Always)
        .with_discovery(lash::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["tool_search".to_string()],
        })
        .with_execution_mode(ToolExecutionMode::Serial)
}

pub(crate) fn load_tools_definition() -> ToolDefinition {
    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct LoadToolsArgs {
        names: Option<Vec<String>>,
    }

    ToolDefinition::typed::<LoadToolsArgs, serde_json::Value>(
        "load_tools",
        "Promote loadable tools into the showcased surface. Callable omitted tools do not need loading and can be called directly.",
    )
        .with_examples(vec!["load_tools(names=[\"tool_name\"])".into()])
        .with_availability(ToolAvailabilityConfig::showcased())
        .with_activation(ToolActivation::Always)
        .with_discovery(lash::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["activate_tools".to_string()],
        })
        .with_execution_mode(ToolExecutionMode::Serial)
}
