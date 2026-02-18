use serde_json::json;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::require_str;

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
        vec![ToolDefinition {
            name: "fetch_url".into(),
            description: "Extract a webpage URL via Tavily and return text content.".into(),
            params: vec![ToolParam::typed("url", "str")],
            returns: "str".into(),
            hidden: false,
        }]
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
                        return ToolResult::err(json!("No content returned from Tavily extract"));
                    }

                    ToolResult::ok(json!(content))
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
