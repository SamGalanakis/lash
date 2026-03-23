use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path};

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::require_str;

/// Search file contents using regex patterns.
#[derive(Default)]
pub struct Grep;

const MAX_RESULTS: usize = 100;
const MAX_LINE_LEN: usize = 2_000;
const TRUNCATION_HINT: &str = "Consider using a more specific path or pattern.";

#[async_trait::async_trait]
impl ToolProvider for Grep {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "grep".into(),
            description: format!(
                "Search file contents with a ripgrep-style regex. Returns up to {} `file:line:text` matches.",
                MAX_RESULTS
            ),
            params: vec![
                ToolParam::typed("pattern", "str"),
                ToolParam {
                    name: "path".into(),
                    r#type: "str".into(),
                    description: "Directory or file to search in (default: current directory)"
                        .into(),
                    default_value: Some(serde_json::json!(".")),
                    required: false,
                },
                ToolParam {
                    name: "include".into(),
                    r#type: "str".into(),
                    description: "Glob pattern to filter files (e.g. \"*.rs\", \"*.py\")".into(),
                    default_value: None,
                    required: false,
                },
            ],
            returns: "list[str]".into(),
            examples: vec![],
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
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
            let searched = search_file(base, &re, MAX_RESULTS);
            let mut output = format_matches(&searched.entries);
            maybe_append_truncation_line(&mut output, searched.match_count);
            return ToolResult::ok(json!(output));
        }

        if !base.is_dir() {
            return ToolResult::err_fmt(format_args!("Not a file or directory: {base_dir}"));
        }

        let walker = ignore::WalkBuilder::new(base)
            .hidden(false)
            .git_ignore(true)
            .build();

        let mut all_entries: Vec<GrepHit> = Vec::new();
        let mut total_matches: usize = 0;
        let mut shown_matches: usize = 0;

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();

            if !path.is_file() || is_git_internal_path(base, path) {
                continue;
            }

            // Apply include filter
            if let Some(ref glob_set) = include_glob {
                let filename = path.file_name().map(Path::new).unwrap_or(path);
                if !glob_set.is_match(filename) {
                    continue;
                }
            }

            if shown_matches >= MAX_RESULTS {
                break;
            }

            let remaining = MAX_RESULTS.saturating_sub(shown_matches);
            let searched = search_file(path, &re, remaining);
            total_matches += searched.match_count;
            shown_matches += searched.entries.len();
            all_entries.extend(searched.entries);
        }

        let mut output = format_matches(&all_entries);
        maybe_append_truncation_line(&mut output, total_matches);
        ToolResult::ok(json!(output))
    }
}

#[derive(Clone)]
struct GrepHit {
    file: String,
    line: usize,
    content: String,
    is_match: bool,
}

struct SearchResult {
    entries: Vec<GrepHit>,
    match_count: usize,
}

fn maybe_append_truncation_line(output: &mut Vec<String>, total_matches: usize) {
    if total_matches > MAX_RESULTS {
        let omitted = total_matches - MAX_RESULTS;
        output.push(format!(
            "[results truncated: showing {} of {} matches ({} omitted). {}]",
            MAX_RESULTS, total_matches, omitted, TRUNCATION_HINT
        ));
    }
}

fn format_matches(matches: &[GrepHit]) -> Vec<String> {
    matches
        .iter()
        .map(|m| {
            if m.is_match {
                format!("{}:{}:{}", m.file, m.line, m.content)
            } else {
                format!("{}-{}-{}", m.file, m.line, m.content)
            }
        })
        .collect()
}

fn search_file(path: &Path, re: &regex::Regex, max_matches: usize) -> SearchResult {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(_) => {
            return SearchResult {
                entries: vec![],
                match_count: 0,
            };
        }
    };

    let reader = BufReader::new(file);
    let file = path.to_string_lossy().to_string();
    let mut entries = Vec::new();
    let mut match_count = 0usize;

    for (idx, line_result) in reader.lines().enumerate() {
        let Ok(line) = line_result else {
            continue;
        };
        if !re.is_match(&line) {
            continue;
        }
        match_count += 1;
        if entries.len() >= max_matches {
            continue;
        }
        let content = if line.len() > MAX_LINE_LEN {
            format!("{}...", &line[..MAX_LINE_LEN])
        } else {
            line
        };
        entries.push(GrepHit {
            file: file.clone(),
            line: idx + 1,
            content,
            is_match: true,
        });
    }

    SearchResult {
        entries,
        match_count,
    }
}

fn is_git_internal_path(base: &Path, path: &Path) -> bool {
    path.strip_prefix(base)
        .unwrap_or(path)
        .components()
        .any(|component| matches!(component, Component::Normal(name) if name == ".git"))
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

    #[tokio::test]
    async fn test_grep_truncation_marker() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("test.txt"),
            "hello one\nhello two\nhello three\n",
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
        let text: Vec<&str> = arr.iter().map(|v| v.as_str().unwrap_or("")).collect();
        assert_eq!(text.len(), 3);
        assert!(!text[2].contains("results truncated"));
    }

    #[tokio::test]
    async fn test_grep_hard_caps_at_100_results() {
        let dir = TempDir::new().unwrap();
        let lines = (0..150)
            .map(|idx| format!("hello {idx}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("test.txt"), lines).unwrap();
        let tool = Grep;
        let result = tool
            .execute(
                "grep",
                &json!({"pattern": "hello", "path": dir.path().to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        let arr = result.result.as_array().unwrap();
        assert_eq!(arr.len(), 101);
        assert!(
            arr[100]
                .as_str()
                .unwrap_or("")
                .contains("showing 100 of 150")
        );
    }

    #[tokio::test]
    async fn test_grep_ignores_git_internal_paths() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".git/rr-cache")).unwrap();
        std::fs::write(
            dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again",
        )
        .unwrap();
        std::fs::write(
            dir.path().join(".git/rr-cache/ignored.txt"),
            "hello from git internals",
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
        let text: Vec<&str> = arr.iter().map(|v| v.as_str().unwrap_or("")).collect();
        assert!(text.iter().any(|l| l.contains(":1:hello world")));
        assert!(text.iter().any(|l| l.contains(":3:hello again")));
        assert!(!text.iter().any(|l| l.contains("git internals")));
    }
}
