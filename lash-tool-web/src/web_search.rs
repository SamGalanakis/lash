use serde_json::json;

use lash::{ToolCall, ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult};

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
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::raw(
                "search_web",
                "Search the web for candidate sources. Returns `{results, answer?}` with snippet text; use `fetch_url` when you need the page itself.",
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
                "search_web(query=\"latest Rust release notes\", limit=5)".into(),
            ])
            .with_discovery(lash_tool_support::discovery_metadata("web", &["web_search"]))
            .with_execution_mode(ToolExecutionMode::Parallel),
        ]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let args = call.args;
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5);

        let body = json!({
            "api_key": self.api_key,
            "query": query,
            "max_results": limit,
            "include_answer": true,
        });

        let resp = self
            .client
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
                Ok(data) => {
                    let results: Vec<serde_json::Value> = data
                        .get("results")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .map(|r| {
                                    json!({
                                        "title": r.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                                        "url": r.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                                        "content": r.get("content").and_then(|v| v.as_str()).unwrap_or(""),
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();

                    let mut result = json!({ "results": results });
                    if let Some(answer) = data.get("answer").and_then(|v| v.as_str()) {
                        result["answer"] = json!(answer);
                    }

                    ToolResult::ok(result)
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_web_uses_limit_argument_in_model_contract() {
        let definition = WebSearch::new("test-key")
            .definitions()
            .into_iter()
            .find(|definition| definition.name == "search_web")
            .expect("search_web definition");

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
        assert_eq!(definition.activation, lash::ToolActivation::Always);
        assert_eq!(
            definition.availability.standard,
            lash::ToolAvailability::Showcased
        );
    }
}
