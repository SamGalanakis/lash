use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::hashline::{self, HashlineEdit};

/// Hashline-aware file editing tool.
pub struct EditFile;

impl EditFile {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EditFile {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolProvider for EditFile {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "edit_file".into(),
            description: concat!(
                "Edit a file using hashline anchors. Each edit is one of: ",
                "{\"set_line\": {\"anchor\": \"LINE:HASH\", \"new_text\": \"...\"}}, ",
                "{\"replace_lines\": {\"start_anchor\": \"LINE:HASH\", \"end_anchor\": \"LINE:HASH\", \"new_text\": \"...\"}}, ",
                "{\"insert_after\": {\"anchor\": \"LINE:HASH\", \"text\": \"...\"}}, ",
                "{\"replace\": {\"old_text\": \"...\", \"new_text\": \"...\", \"all\": false}}",
            ).into(),
            params: vec![
                ToolParam::typed("path", "str"),
                ToolParam {
                    name: "edits".into(),
                    r#type: "list".into(),
                    description: "List of edit operations".into(),
                    required: true,
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
                result: json!(format!("File does not exist: {}", path_str)),
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

        let edits_json = match args.get("edits").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => {
                return ToolResult {
                    success: false,
                    result: json!("Missing or invalid 'edits' parameter: expected a list"),
                };
            }
        };

        let edits = match parse_edits(edits_json) {
            Ok(e) => e,
            Err(msg) => {
                return ToolResult {
                    success: false,
                    result: json!(msg),
                };
            }
        };

        match hashline::apply_hashline_edits(&content, edits) {
            Ok(new_content) => {
                if let Err(e) = std::fs::write(path, &new_content) {
                    return ToolResult {
                        success: false,
                        result: json!(format!("Failed to write file: {}", e)),
                    };
                }
                let new_lines = new_content.lines().count();
                ToolResult {
                    success: true,
                    result: json!(format!(
                        "Applied {} edit(s) to {} ({} lines)",
                        edits_json.len(),
                        path_str,
                        new_lines,
                    )),
                }
            }
            Err(msg) => ToolResult {
                success: false,
                result: json!(msg),
            },
        }
    }
}

fn parse_edits(edits: &[serde_json::Value]) -> Result<Vec<HashlineEdit>, String> {
    let mut result = Vec::new();
    for (i, edit) in edits.iter().enumerate() {
        let edit = parse_single_edit(edit)
            .map_err(|e| format!("Edit #{}: {}", i + 1, e))?;
        result.push(edit);
    }
    Ok(result)
}

fn parse_single_edit(edit: &serde_json::Value) -> Result<HashlineEdit, String> {
    if let Some(obj) = edit.get("set_line") {
        let anchor = obj
            .get("anchor")
            .and_then(|v| v.as_str())
            .ok_or("set_line: missing 'anchor'")?
            .to_string();
        let new_text = obj
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or("set_line: missing 'new_text'")?
            .to_string();
        return Ok(HashlineEdit::SetLine { anchor, new_text });
    }

    if let Some(obj) = edit.get("replace_lines") {
        let start_anchor = obj
            .get("start_anchor")
            .and_then(|v| v.as_str())
            .ok_or("replace_lines: missing 'start_anchor'")?
            .to_string();
        let end_anchor = obj
            .get("end_anchor")
            .and_then(|v| v.as_str())
            .ok_or("replace_lines: missing 'end_anchor'")?
            .to_string();
        let new_text = obj
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or("replace_lines: missing 'new_text'")?
            .to_string();
        return Ok(HashlineEdit::ReplaceLines {
            start_anchor,
            end_anchor,
            new_text,
        });
    }

    if let Some(obj) = edit.get("insert_after") {
        let anchor = obj
            .get("anchor")
            .and_then(|v| v.as_str())
            .ok_or("insert_after: missing 'anchor'")?
            .to_string();
        let text = obj
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or("insert_after: missing 'text'")?
            .to_string();
        return Ok(HashlineEdit::InsertAfter { anchor, text });
    }

    if let Some(obj) = edit.get("replace") {
        let old_text = obj
            .get("old_text")
            .and_then(|v| v.as_str())
            .ok_or("replace: missing 'old_text'")?
            .to_string();
        let new_text = obj
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or("replace: missing 'new_text'")?
            .to_string();
        let all = obj
            .get("all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        return Ok(HashlineEdit::Replace {
            old_text,
            new_text,
            all,
        });
    }

    Err(format!(
        "Unknown edit type. Expected one of: set_line, replace_lines, insert_after, replace. Got: {}",
        edit
    ))
}
