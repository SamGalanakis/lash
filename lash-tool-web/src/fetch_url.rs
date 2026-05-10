use serde_json::json;

use lash::{ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult};

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
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
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
            .with_execution_mode(ToolExecutionMode::Parallel),
        ]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
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

        match resp {
            Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
                Ok(data) => {
                    let content = data
                        .get("results")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| {
                            v.get("raw_content")
                                .and_then(|s| s.as_str())
                                .or_else(|| v.get("content").and_then(|s| s.as_str()))
                        })
                        .unwrap_or_default();

                    if content.is_empty() {
                        return ToolResult::ok(json!({
                            "url": url,
                            "content": "",
                        }));
                    }

                    ToolResult::ok(json!({
                        "url": url,
                        "content": content,
                    }))
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
    fn fetch_url_returns_minimal_typed_record_and_is_showcased() {
        let definition = FetchUrl::new("test-key")
            .definitions()
            .into_iter()
            .find(|definition| definition.name == "fetch_url")
            .expect("fetch_url definition");

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
        assert_eq!(definition.activation, lash::ToolActivation::Always);
        assert_eq!(
            definition.availability.standard,
            lash::ToolAvailability::Documented
        );
    }
}
