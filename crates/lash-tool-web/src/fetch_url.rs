use serde_json::json;

use lash_core::{
    ToolCall, ToolContract, ToolDefinition, ToolExecutionMode, ToolManifest, ToolProvider,
    ToolResult,
};

use lash_tool_support::{object_schema, require_str};

/// Fetch a URL and return its content as text.
pub struct FetchUrl {
    api_key: String,
    client: reqwest::Client,
}

impl FetchUrl {
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

impl Default for FetchUrl {
    fn default() -> Self {
        Self::new("")
    }
}

#[async_trait::async_trait]
impl ToolProvider for FetchUrl {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![fetch_url_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<std::sync::Arc<ToolContract>> {
        (name == "fetch_url").then(|| std::sync::Arc::new(fetch_url_tool_definition().contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let args = call.args;
        let url = match require_str(args, "url") {
            Ok(s) => s,
            Err(e) => return e,
        };

        if self.api_key.trim().is_empty() {
            return ToolResult::err(json!("Tavily API key is required for fetch_url"));
        }

        let body = json!({
            "api_key": self.api_key,
            "urls": [url],
        });

        let resp = self
            .client
            .post("https://api.tavily.com/extract")
            .json(&body)
            .send()
            .await;
        let resp = match resp {
            Ok(resp) => resp,
            Err(err) => return ToolResult::err(json!(format!("fetch_url request failed: {err}"))),
        };
        let status = resp.status();
        let value: serde_json::Value = match resp.json().await {
            Ok(value) => value,
            Err(err) => return ToolResult::err(json!(format!("fetch_url response failed: {err}"))),
        };
        if !status.is_success() {
            return ToolResult::err(value);
        }
        let content = value
            .get("results")
            .and_then(|value| value.as_array())
            .and_then(|results| results.first())
            .and_then(|item| item.get("raw_content").or_else(|| item.get("content")))
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        ToolResult::ok(json!({
            "url": url,
            "content": content,
        }))
    }
}

fn fetch_url_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
                "tool:fetch_url",
                "fetch_url",
                "Fetch one known URL and extract readable page text.",
                object_schema(
                    serde_json::json!({
                        "url": { "type": "string", "format": "uri" }
                    }),
                    &["url"],
                ),
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Fetched URL."
                        },
                        "content": {
                            "type": "string",
                            "description": "Extracted readable page text. Empty when no extractable content was returned."
                        }
                    },
                    "required": ["url", "content"],
                    "additionalProperties": false
                }),
            )
            .with_examples(vec!["fetch_url(url=\"https://www.rust-lang.org/\")".into()])
            .with_discovery(lash_tool_support::discovery_metadata(
                "web",
                &["fetch", "open_url"],
            ))
            .with_execution_mode(ToolExecutionMode::Parallel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_url_returns_minimal_typed_record_and_is_showcased() {
        let definition = fetch_url_tool_definition();

        assert_eq!(
            definition.output_schema["type"],
            serde_json::json!("object")
        );
        assert_eq!(
            definition.output_schema["required"],
            serde_json::json!(["url", "content"])
        );
        assert_eq!(
            definition.output_schema["additionalProperties"],
            serde_json::json!(false)
        );
        assert_eq!(definition.activation, lash_core::ToolActivation::Always);
        assert_eq!(
            definition.availability.standard,
            lash_core::ToolAvailability::Showcased
        );
    }
}
