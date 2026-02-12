use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::hashline;

/// Read files with hashline-prefixed output.
pub struct ReadFile;

impl ReadFile {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadFile {
    fn default() -> Self {
        Self::new()
    }
}

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE_LEN: usize = 2000;

#[async_trait::async_trait]
impl ToolProvider for ReadFile {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "read_file".into(),
            description: "Read a file and return its content with line:hash prefixes. If path is a directory, lists its entries.".into(),
            params: vec![
                ToolParam::typed("path", "str"),
                ToolParam {
                    name: "offset".into(),
                    r#type: "int".into(),
                    description: "Line offset to start reading from (1-based)".into(),
                    required: false,
                },
                ToolParam {
                    name: "limit".into(),
                    r#type: "int".into(),
                    description: "Maximum number of lines to read (default 2000)".into(),
                    required: false,
                },
            ],
            returns: "str".into(),
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if path_str.is_empty() {
            return ToolResult {
                success: false,
                result: json!("Missing required parameter: path"),
            };
        }

        let path = Path::new(path_str);

        if !path.exists() {
            return ToolResult {
                success: false,
                result: json!(format!("Path does not exist: {}", path_str)),
            };
        }

        // Directory support
        if path.is_dir() {
            return list_directory(path).await;
        }

        // Binary detection
        if is_likely_binary(path) {
            return ToolResult {
                success: false,
                result: json!(format!("Binary file detected: {}", path_str)),
            };
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                return ToolResult {
                    success: false,
                    result: json!(format!("Failed to read file: {}", e)),
                };
            }
        };

        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1)
            .max(1);

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let all_lines: Vec<&str> = content.lines().collect();
        let total_lines = all_lines.len();

        // offset is 1-based
        let start_idx = (offset - 1).min(total_lines);
        let end_idx = (start_idx + limit).min(total_lines);
        let selected: Vec<&str> = all_lines[start_idx..end_idx].to_vec();

        // Truncate long lines and format with hashlines
        let truncated_content: String = selected
            .iter()
            .map(|line| {
                if line.len() > MAX_LINE_LEN {
                    format!("{}...", &line[..MAX_LINE_LEN])
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mut formatted = hashline::format_hashlines(&truncated_content, offset);

        if end_idx < total_lines {
            formatted.push_str(&format!(
                "\n[Showing lines {}-{} of {}. Use offset={} to continue.]",
                offset,
                offset + selected.len() - 1,
                total_lines,
                end_idx + 1,
            ));
        }

        ToolResult {
            success: true,
            result: json!(formatted),
        }
    }
}

async fn list_directory(path: &Path) -> ToolResult {
    match std::fs::read_dir(path) {
        Ok(entries) => {
            let mut items: Vec<String> = Vec::new();
            for entry in entries {
                if let Ok(entry) = entry {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    if is_dir {
                        items.push(format!("{}/", name));
                    } else {
                        items.push(name);
                    }
                }
            }
            items.sort();
            ToolResult {
                success: true,
                result: json!(items.join("\n")),
            }
        }
        Err(e) => ToolResult {
            success: false,
            result: json!(format!("Failed to read directory: {}", e)),
        },
    }
}

/// Simple binary detection: check first 8KB for null bytes.
fn is_likely_binary(path: &Path) -> bool {
    use std::io::Read;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = [0u8; 8192];
    let n = match file.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    buf[..n].contains(&0)
}
