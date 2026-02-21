use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::require_str;

/// Search file contents using regex patterns.
#[derive(Default)]
pub struct Grep;

const MAX_RESULTS: usize = 100;
const MAX_LINE_LEN: usize = 500;

#[async_trait::async_trait]
impl ToolProvider for Grep {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "grep".into(),
            description:
                "Search file contents for a regex pattern (Rust regex syntax: use | for alternation, not \\|). Returns matching lines with file paths and line numbers (limit 100)."
                    .into(),
            params: vec![
                ToolParam::typed("pattern", "str"),
                ToolParam {
                    name: "path".into(),
                    r#type: "str".into(),
                    description: "Directory or file to search in (default: current directory)"
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "include".into(),
                    r#type: "str".into(),
                    description: "Glob pattern to filter files (e.g. \"*.rs\", \"*.py\")".into(),
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

        let include = args.get("include").and_then(|v| v.as_str());

        let re = match regex::Regex::new(pattern) {
            Ok(r) => r,
            Err(e) => {
                return ToolResult::err_fmt(format_args!("Invalid regex pattern: {e}"));
            }
        };

        let include_glob = if let Some(inc) = include {
            match globset::GlobBuilder::new(inc).build() {
                Ok(g) => globset::GlobSetBuilder::new().add(g).build().ok(),
                Err(_) => None,
            }
        } else {
            None
        };

        let base = Path::new(base_dir);

        // If it's a single file, search just that file
        if base.is_file() {
            let results = search_file(base, &re);
            return ToolResult::ok(json!(format_matches(&results)));
        }

        if !base.is_dir() {
            return ToolResult::err_fmt(format_args!("Not a file or directory: {base_dir}"));
        }

        let walker = ignore::WalkBuilder::new(base)
            .hidden(false)
            .git_ignore(true)
            .build();

        let mut all_matches: Vec<serde_json::Value> = Vec::new();

        for entry in walker {
            if all_matches.len() >= MAX_RESULTS {
                break;
            }

            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            // Apply include filter
            if let Some(ref glob_set) = include_glob {
                let filename = path.file_name().map(Path::new).unwrap_or(path);
                if !glob_set.is_match(filename) {
                    continue;
                }
            }

            let remaining = MAX_RESULTS - all_matches.len();
            let mut results = search_file(path, &re);
            results.truncate(remaining);
            all_matches.extend(results);
        }

        ToolResult::ok(json!(format_matches(&all_matches)))
    }
}

fn format_matches(matches: &[serde_json::Value]) -> Vec<String> {
    matches
        .iter()
        .map(|m| {
            format!(
                "{}:{}:{}",
                m["file"].as_str().unwrap_or(""),
                m["line"],
                m["content"].as_str().unwrap_or(""),
            )
        })
        .collect()
}

fn search_file(path: &Path, re: &regex::Regex) -> Vec<serde_json::Value> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let mut results = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if re.is_match(line) {
            let display_line = if line.len() > MAX_LINE_LEN {
                format!("{}...", &line[..MAX_LINE_LEN])
            } else {
                line.to_string()
            };
            results.push(json!({
                "file": path.to_string_lossy(),
                "line": i + 1,
                "content": display_line,
            }));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_grep_matches() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again",
        )
        .unwrap();
        let tool = Grep;
        let result = tool
            .execute(
                "grep",
                &json!({"pattern": "hello", "path": dir.path().to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        let arr = result.result.as_array().unwrap();
        let text: Vec<&str> = arr.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(text.iter().any(|l| l.contains("hello world")));
        assert!(text.iter().any(|l| l.contains("hello again")));
        assert!(!text.iter().any(|l| l.contains("foo bar")));
    }

    #[tokio::test]
    async fn test_grep_no_matches() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "hello world").unwrap();
        let tool = Grep;
        let result = tool
            .execute(
                "grep",
                &json!({"pattern": "xyz", "path": dir.path().to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        assert!(result.result.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_grep_single_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        std::fs::write(&path, "fn main() {\n    println!(\"hello\");\n}").unwrap();
        let tool = Grep;
        let result = tool
            .execute(
                "grep",
                &json!({"pattern": "println", "path": path.to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        let arr = result.result.as_array().unwrap();
        assert!(arr.iter().any(|v| v.as_str().unwrap().contains("println")));
    }
}
