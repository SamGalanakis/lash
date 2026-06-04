use serde_json::{Value, json};

use lash_core::{ToolCall, ToolDefinition, ToolResult, ToolScheduling};

use lash_tool_support::{StaticToolExecute, StaticToolProvider, object_schema};

/// Web search via Tavily API.
pub struct WebSearch {
    api_key: String,
    client: reqwest::Client,
}

impl WebSearch {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }
}

/// Build the cached `search_web` tool provider for the given Tavily API key.
pub fn web_search_provider(api_key: impl Into<String>) -> StaticToolProvider<WebSearch> {
    StaticToolProvider::new(vec![web_search_tool_definition()], WebSearch::new(api_key))
}

#[async_trait::async_trait]
impl StaticToolExecute for WebSearch {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let args = call.args;
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .clamp(1, 20);

        if self.api_key.trim().is_empty() {
            return ToolResult::err(json!("Tavily API key is required for web.search"));
        }

        let body = json!({
            "query": query,
            "max_results": limit,
        });

        let resp = self
            .client
            .post("https://api.tavily.com/search")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
                Ok(data) => ToolResult::ok(json!({
                    "results": sanitize_results(data.get("results")),
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

fn sanitize_results(results: Option<&Value>) -> Vec<Value> {
    results
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|item| {
            json!({
                "title": item.get("title").and_then(Value::as_str).unwrap_or_default(),
                "url": item.get("url").and_then(Value::as_str).unwrap_or_default(),
                "content": item.get("content").and_then(Value::as_str).unwrap_or_default(),
            })
        })
        .collect()
}

fn web_search_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
                "tool:search_web",
                "search_web",
                "Search the web for candidate sources. Returns ranked `results` with snippet text; use `web.fetch` when you need the page itself. This tool does not expose Tavily's optional generated answer; summarize from result snippets and fetched pages.",
                object_schema(
                    serde_json::json!({
                        "query": { "type": "string" },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
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
            .contract
            .input_schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .expect("object properties");
        assert!(properties.contains_key("limit"));
        assert!(!properties.contains_key("max_results"));
        assert_eq!(properties["limit"]["default"], serde_json::json!(5));
        assert_eq!(properties["limit"]["maximum"], serde_json::json!(20));
        assert!(
            definition
                .contract
                .examples
                .iter()
                .all(|example| example.contains("limit"))
        );
        assert_eq!(
            definition.contract.output_schema["type"],
            serde_json::json!("object")
        );
        assert_eq!(
            definition.contract.output_schema["required"],
            serde_json::json!(["results"])
        );
        assert!(
            !definition.contract.output_schema["properties"]
                .as_object()
                .unwrap()
                .contains_key("answer")
        );
        assert_eq!(
            definition.manifest.activation,
            lash_core::ToolActivation::Always
        );
        assert_eq!(
            definition.manifest.availability.base,
            lash_core::ToolAvailability::Showcased
        );
    }

    #[test]
    fn search_web_sanitizes_tavily_results_to_contract() {
        let results = sanitize_results(Some(&serde_json::json!([
            {
                "title": "Title",
                "url": "https://example.com",
                "content": "Snippet",
                "score": 0.9,
                "raw_content": null,
                "favicon": "https://example.com/favicon.ico"
            }
        ])));

        assert_eq!(
            results,
            vec![serde_json::json!({
                "title": "Title",
                "url": "https://example.com",
                "content": "Snippet"
            })]
        );
    }
}
