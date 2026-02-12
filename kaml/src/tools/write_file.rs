use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

/// Write complete file contents, creating parent directories if needed.
pub struct WriteFile;

impl WriteFile {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WriteFile {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolProvider for WriteFile {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "write_file".into(),
            description: "Write content to a file, creating parent directories if needed.".into(),
            params: vec![
                ToolParam::typed("path", "str"),
                ToolParam::typed("content", "str"),
            ],
            returns: "str".into(),
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if path_str.is_empty() {
            return ToolResult {
                success: false,
                result: json!("Missing required parameter: path"),
            };
        }

        let path = Path::new(path_str);

        // Create parent directories
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return ToolResult {
                        success: false,
                        result: json!(format!("Failed to create directories: {}", e)),
                    };
                }
            }
        }

        match std::fs::write(path, content) {
            Ok(()) => {
                let bytes = content.len();
                ToolResult {
                    success: true,
                    result: json!(format!("Wrote {} bytes to {}", bytes, path_str)),
                }
            }
            Err(e) => ToolResult {
                success: false,
                result: json!(format!("Failed to write file: {}", e)),
            },
        }
    }
}
