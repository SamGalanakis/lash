use serde_json::json;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

/// Fetch a URL and return its content as text.
pub struct FetchUrl {
    client: reqwest::Client,
}

impl FetchUrl {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }
}

impl Default for FetchUrl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolProvider for FetchUrl {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "fetch_url".into(),
            description: "Fetch a URL and return its text content (truncated at 100KB).".into(),
            params: vec![ToolParam::typed("url", "str")],
            returns: "str".into(),
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if url.is_empty() {
            return ToolResult {
                success: false,
                result: json!("Missing required parameter: url"),
            };
        }

        let resp = self.client.get(url).send().await;

        match resp {
            Ok(r) if r.status().is_success() => {
                // Cap response body to avoid blowing up context
                const MAX_BYTES: usize = 100_000;
                match r.bytes().await {
                    Ok(bytes) => {
                        let mut text = String::from_utf8_lossy(
                            &bytes[..bytes.len().min(MAX_BYTES)],
                        ).to_string();
                        if bytes.len() > MAX_BYTES {
                            text.push_str("\n[truncated]");
                        }
                        ToolResult {
                            success: true,
                            result: json!(text),
                        }
                    }
                    Err(e) => ToolResult {
                        success: false,
                        result: json!(format!("Failed to read body: {e}")),
                    },
                }
            }
            Ok(r) => {
                let status = r.status();
                ToolResult {
                    success: false,
                    result: json!(format!("HTTP {status}")),
                }
            }
            Err(e) => ToolResult {
                success: false,
                result: json!(format!("Request failed: {e}")),
            },
        }
    }
}
