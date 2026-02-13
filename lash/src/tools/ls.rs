use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

/// List directory tree structure.
pub struct Ls;

impl Ls {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Ls {
    fn default() -> Self {
        Self::new()
    }
}

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
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let base_dir = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let ignore_patterns: Vec<&str> = args
            .get("ignore")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect()
            })
            .unwrap_or_default();

        let base = Path::new(base_dir);
        if !base.is_dir() {
            return ToolResult {
                success: false,
                result: json!(format!("Not a directory: {}", base_dir)),
            };
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

        ToolResult {
            success: true,
            result: json!(tree),
        }
    }
}
