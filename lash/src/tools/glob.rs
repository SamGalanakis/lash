use serde_json::json;
use std::path::{Path, PathBuf};

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::require_str;

/// Find files by glob pattern.
#[derive(Default)]
pub struct Glob;

const MAX_RESULTS: usize = 100;

#[async_trait::async_trait]
impl ToolProvider for Glob {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "glob".into(),
            description: "Find files matching a glob pattern, sorted by modification time (newest first). Returns up to 100 results.".into(),
            params: vec![
                ToolParam::typed("pattern", "str"),
                ToolParam {
                    name: "path".into(),
                    r#type: "str".into(),
                    description: "Base directory to search in (default: current directory)".into(),
                    required: false,
                },
            ],
            returns: "list".into(),
            examples: vec![],
                hidden: false,
                inject_into_prompt: true,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let pattern = match require_str(args, "pattern") {
            Ok(s) => s,
            Err(e) => return e,
        };

        let base_dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let base = Path::new(base_dir);
        if !base.exists() {
            return ToolResult::err_fmt(format_args!("Path does not exist: {base_dir}"));
        }
        if !base.is_dir() {
            return ToolResult::err_fmt(format_args!(
                "{base_dir} is a file, not a directory. Pass the parent directory as path and use the pattern to match files."
            ));
        }

        // Build the glob matcher
        let glob = match globset::GlobBuilder::new(pattern)
            .literal_separator(false)
            .build()
        {
            Ok(g) => g,
            Err(e) => {
                return ToolResult::err_fmt(format_args!("Invalid glob pattern: {e}"));
            }
        };

        let matcher = match globset::GlobSetBuilder::new().add(glob).build() {
            Ok(m) => m,
            Err(e) => {
                return ToolResult::err_fmt(format_args!("Failed to build glob matcher: {e}"));
            }
        };

        // Walk the directory tree, respecting .gitignore
        let walker = ignore::WalkBuilder::new(base)
            .hidden(false)
            .git_ignore(true)
            .build();

        let mut matches: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            // Match against relative path from base
            let rel_path = match path.strip_prefix(base) {
                Ok(r) => r,
                Err(_) => continue,
            };

            if matcher.is_match(rel_path) {
                let mtime = path
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                matches.push((path.to_path_buf(), mtime));
            }
        }

        // Sort by mtime, newest first
        matches.sort_by(|a, b| b.1.cmp(&a.1));
        matches.truncate(MAX_RESULTS);

        let paths: Vec<String> = matches
            .iter()
            .map(|(p, _)| p.to_string_lossy().to_string())
            .collect();

        ToolResult::ok(json!(paths.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_glob_matches() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        let text = result.result.as_str().unwrap();
        assert!(text.contains("a.rs"));
        assert!(text.contains("b.rs"));
        assert!(!text.contains("c.txt"));
    }

    #[tokio::test]
    async fn test_glob_no_matches() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        assert!(result.result.as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_glob_nested() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("sub/deep")).unwrap();
        std::fs::write(dir.path().join("sub/deep/file.rs"), "").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({"pattern": "**/*.rs", "path": dir.path().to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        assert!(result.result.as_str().unwrap().contains("file.rs"));
    }
}
