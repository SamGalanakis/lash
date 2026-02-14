use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolResult};

/// Simple text find-and-replace tool (no anchors needed).
pub struct FindReplace;

impl FindReplace {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FindReplace {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl crate::ToolProvider for FindReplace {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "find_replace".into(),
            description: concat!(
                "Find and replace text in a file. No anchors needed — works on raw file content.\n\n",
                "Useful for renaming variables/functions, fixing typos, or simple substitutions. ",
                "old_text must match exactly (including whitespace and newlines). ",
                "By default replaces only the first occurrence; set all=true to replace every occurrence.\n\n",
                "For structural edits (inserting/deleting lines, multi-line rewrites), use edit_file instead.",
            ).into(),
            params: vec![
                ToolParam::typed("path", "str"),
                ToolParam {
                    name: "old_text".into(),
                    r#type: "str".into(),
                    description: "Exact text to find".into(),
                    required: true,
                },
                ToolParam {
                    name: "new_text".into(),
                    r#type: "str".into(),
                    description: "Replacement text".into(),
                    required: true,
                },
                ToolParam {
                    name: "all".into(),
                    r#type: "bool".into(),
                    description: "Replace all occurrences (default: false, first only)".into(),
                    required: false,
                },
            ],
            returns: "str".into(),
            hidden: false,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if path_str.is_empty() {
            return ToolResult::err(json!("Missing required parameter: path"));
        }

        let old_text = match args.get("old_text").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return ToolResult::err(json!("Missing required parameter: old_text"));
            }
        };

        let new_text = match args.get("new_text").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return ToolResult::err(json!("Missing required parameter: new_text"));
            }
        };

        let replace_all = args
            .get("all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let path = Path::new(path_str);

        if !path.exists() {
            return ToolResult::err(json!(format!("File does not exist: {}", path_str)));
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err(json!(format!("Failed to read file: {}", e)));
            }
        };

        if !content.contains(old_text) {
            return ToolResult::err(json!("old_text not found in file"));
        }

        let (new_content, count) = if replace_all {
            let count = content.matches(old_text).count();
            (content.replace(old_text, new_text), count)
        } else {
            (content.replacen(old_text, new_text, 1), 1)
        };

        if let Err(e) = std::fs::write(path, &new_content) {
            return ToolResult::err(json!(format!("Failed to write file: {}", e)));
        }

        let label = if count == 1 {
            "1 replacement".to_string()
        } else {
            format!("{} replacements", count)
        };

        ToolResult::ok(json!(format!("{} made in {}", label, path_str)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolProvider;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_replace_first() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "foo bar foo baz").unwrap();
        let tool = FindReplace::default();
        let result = tool
            .execute(
                "find_replace",
                &json!({
                    "path": path.to_str().unwrap(),
                    "old_text": "foo",
                    "new_text": "qux"
                }),
            )
            .await;
        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "qux bar foo baz"
        );
    }

    #[tokio::test]
    async fn test_replace_all() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "foo bar foo baz foo").unwrap();
        let tool = FindReplace::default();
        let result = tool
            .execute(
                "find_replace",
                &json!({
                    "path": path.to_str().unwrap(),
                    "old_text": "foo",
                    "new_text": "qux",
                    "all": true
                }),
            )
            .await;
        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "qux bar qux baz qux"
        );
        let text = result.result.as_str().unwrap();
        assert!(text.contains("3 replacements"));
    }

    #[tokio::test]
    async fn test_no_match() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world").unwrap();
        let tool = FindReplace::default();
        let result = tool
            .execute(
                "find_replace",
                &json!({
                    "path": path.to_str().unwrap(),
                    "old_text": "xyz",
                    "new_text": "abc"
                }),
            )
            .await;
        assert!(!result.success);
    }
}
