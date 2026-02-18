use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

/// List directory tree structure.
#[derive(Default)]
pub struct Ls;

const MAX_ENTRIES: usize = 500;

#[async_trait::async_trait]
impl ToolProvider for Ls {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "ls".into(),
            description: "List directory tree structure (max 3 levels, 500 entries), respecting .gitignore. Use this to explore project structure.".into(),
            params: vec![
                ToolParam {
                    name: "path".into(),
                    r#type: "str".into(),
                    description: "Directory to list (default: current directory)".into(),
                    required: false,
                },
                ToolParam {
                    name: "ignore".into(),
                    r#type: "list".into(),
                    description: "Additional patterns to ignore".into(),
                    required: false,
                },
            ],
            returns: "str".into(),
            hidden: false,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let base_dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let ignore_patterns: Vec<&str> = args
            .get("ignore")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let base = Path::new(base_dir);
        if !base.is_dir() {
            return ToolResult::err_fmt(format_args!("Not a directory: {base_dir}"));
        }

        let mut builder = ignore::WalkBuilder::new(base);
        builder.hidden(true).git_ignore(true).max_depth(Some(3));

        // Add custom ignore patterns
        let mut overrides = ignore::overrides::OverrideBuilder::new(base);
        for pat in &ignore_patterns {
            let _ = overrides.add(&format!("!{}", pat));
        }
        if let Ok(ov) = overrides.build() {
            builder.overrides(ov);
        }

        let walker = builder.build();
        let mut entries: Vec<String> = Vec::new();
        let mut count = 0;

        for entry in walker {
            if count >= MAX_ENTRIES {
                entries.push(format!("... truncated at {} entries", MAX_ENTRIES));
                break;
            }

            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            let rel_path = path.strip_prefix(base).unwrap_or(path);
            let depth = rel_path.components().count();

            if depth == 0 {
                continue; // Skip the root itself
            }

            let indent = "  ".repeat(depth - 1);
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            let is_dir = path.is_dir();
            if is_dir {
                entries.push(format!("{}{}/", indent, name));
            } else {
                entries.push(format!("{}{}", indent, name));
            }
            count += 1;
        }

        let tree = entries.join("\n");

        ToolResult::ok(json!(tree))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolProvider;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_ls_files_and_dirs() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        std::fs::write(dir.path().join("subdir/nested.rs"), "").unwrap();
        let tool = Ls;
        let result = tool
            .execute("ls", &json!({"path": dir.path().to_str().unwrap()}))
            .await;
        assert!(result.success);
        let text = result.result.as_str().unwrap();
        assert!(text.contains("file.txt"));
        assert!(text.contains("subdir/"));
        assert!(text.contains("nested.rs"));
    }

    #[tokio::test]
    async fn test_ls_empty_dir() {
        let dir = TempDir::new().unwrap();
        let tool = Ls;
        let result = tool
            .execute("ls", &json!({"path": dir.path().to_str().unwrap()}))
            .await;
        assert!(result.success);
        assert!(result.result.as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_ls_not_a_dir() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "").unwrap();
        let tool = Ls;
        let result = tool
            .execute("ls", &json!({"path": path.to_str().unwrap()}))
            .await;
        assert!(!result.success);
    }
}
