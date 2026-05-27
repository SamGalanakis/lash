use lash_core::{ToolActivation, ToolAvailabilityConfig, ToolDefinition, ToolScheduling};
use serde_json::{Value, json};

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

    ToolDefinition::raw_named(
        "search_tools",
        "Search catalogued tool names, namespaces, aliases, descriptions, signatures, return fields, and examples. Use this when the tool you need is not showcased in the prompt. Query with concise keywords and short intent phrases: include the app/domain, action, object, qualifiers, and important fields or constraints. For initial exploration, print only result names and signatures; inspect descriptions and examples only when you need to choose between close matches or learn call idioms.",
        schema_for::<SearchToolsArgs>(),
        search_tools_output_schema(),
    )
        .with_examples(vec![
            "search_tools(query=\"spotify liked songs library\", namespace=\"appworld\")".into(),
            "search_tools(query=\"spotify song details play_count genre title song_id\", namespace=\"appworld\")".into(),
            "search_tools(query=\"venmo send money private payment_card receiver_email\", namespace=\"appworld\")".into(),
        ])
        .with_availability(ToolAvailabilityConfig::showcased())
        .with_activation(ToolActivation::Always)
        .with_discovery(lash_core::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["tool_search".to_string()],
        })
        .with_scheduling(ToolScheduling::Serial)
}

fn schema_for<T>() -> Value
where
    T: schemars::JsonSchema,
{
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or_else(|_| json!({}))
}

fn search_tools_output_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["name", "signature"],
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
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn search_tools_has_typed_result_schema() {
        let definition = search_tools_definition();

        assert_eq!(definition.output_schema["type"], json!("array"));
        let item = &definition.output_schema["items"];
        assert_eq!(item["type"], json!("object"));
        let required = item["required"].as_array().expect("required");
        assert!(required.contains(&json!("name")));
        assert!(required.contains(&json!("signature")));
        let rendered_signature = definition.compact_contract().render_signature();
        assert!(
            rendered_signature.starts_with("search_tools(query: str"),
            "{rendered_signature}"
        );
        assert!(
            rendered_signature.contains("-> list[record{"),
            "{rendered_signature}"
        );
        assert!(
            rendered_signature.contains("name: str"),
            "{rendered_signature}"
        );
        assert!(
            rendered_signature.contains("signature: str"),
            "{rendered_signature}"
        );
    }
}
