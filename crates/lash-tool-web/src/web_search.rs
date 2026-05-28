use serde_json::{Value, json};

use lash_core::{
    ToolCall, ToolContract, ToolDefinition, ToolManifest, ToolProvider, ToolResult, ToolScheduling,
};

use lash_tool_support::object_schema;

/// Web search via Tavily API.
pub struct WebSearch {
    api_key: String,
    client: reqwest::Client,
}

impl WebSearch {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for WebSearch {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![web_search_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<std::sync::Arc<ToolContract>> {
        (name == "search_web").then(|| std::sync::Arc::new(web_search_tool_definition().contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let args = call.args;
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        if self.api_key.trim().is_empty() {
            return ToolResult::err(json!("Tavily API key is required for web.search"));
        }

        let body = json!({
            "api_key": self.api_key,
            "query": query,
            "max_results": limit,
        });

        let resp = self
            .client
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
                Ok(data) => ToolResult::ok(json!({
                    "results": data.get("results").cloned().unwrap_or_else(|| json!([])),
                    "answer": data.get("answer").cloned().unwrap_or(Value::Null),
                })),
                Err(e) => ToolResult::err_fmt(format_args!("Failed to parse response: {e}")),
            },
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                ToolResult::err_fmt(format_args!("Tavily API error ({status}): {body}"))
            }
            Err(e) => ToolResult::err_fmt(format_args!("Request failed: {e}")),
        }
    }
}

fn web_search_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
                "tool:search_web",
                "search_web",
                "Search the web for candidate sources. Returns `{results, answer?}` with snippet text; use `web.fetch` when you need the page itself.",
                object_schema(
                    serde_json::json!({
                        "query": { "type": "string" },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 5,
                            "description": "Maximum results to return (default 5)"
                        }
                    }),
                    &["query"],
                ),
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": { "type": "string" },
                                    "url": { "type": "string" },
                                    "content": {
                                        "type": "string",
                                        "description": "Search-result snippet text."
                                    }
                                },
                                "required": ["title", "url", "content"],
                                "additionalProperties": false
                            }
                        },
                        "answer": {
                            "type": "string",
                            "description": "Optional synthesized answer returned by the search backend."
                        }
                    },
                    "required": ["results"],
                    "additionalProperties": false
                }),
            )
            .with_examples(vec![
                "await web.search({ query: \"latest Rust release notes\", limit: 5 })?".into(),
            ])
            .with_agent_surface(lash_tool_support::agent_surface(
                ["web"],
                "search",
                &["web_search"],
            ))
            .with_scheduling(ToolScheduling::Parallel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_web_uses_limit_argument_in_model_contract() {
        let definition = web_search_tool_definition();

        let properties = definition
            .input_schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .expect("object properties");
        assert!(properties.contains_key("limit"));
        assert!(!properties.contains_key("max_results"));
        assert_eq!(properties["limit"]["default"], serde_json::json!(5));
        assert!(
            definition
                .examples
                .iter()
                .all(|example| example.contains("limit"))
        );
        assert_eq!(
            definition.output_schema["type"],
            serde_json::json!("object")
        );
        assert_eq!(
            definition.output_schema["required"],
            serde_json::json!(["results"])
        );
        assert_eq!(definition.activation, lash_core::ToolActivation::Always);
        assert_eq!(
            definition.availability.base,
            lash_core::ToolAvailability::Showcased
        );
    }
}
