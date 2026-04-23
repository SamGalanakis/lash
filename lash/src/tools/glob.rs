use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::{ToolDefinition, ToolExecutionMode, ToolParam, ToolProvider, ToolResult};

use super::{
    build_path_entry, filesystem_entries_result, parse_optional_bool, parse_optional_usize_arg,
    require_str, rg_file_list, run_blocking,
};

/// Find files by glob pattern.
#[derive(Default)]
pub struct Glob;

const MAX_RESULTS: usize = 100;

#[async_trait::async_trait]
impl ToolProvider for Glob {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "glob".into(),
            description: format!(
                "Find filesystem entries by glob. By default this includes hidden files and respects `.gitignore` only inside Git repos. Returns a record with `items` sorted by `modified_at` (newest first). Each item has `path`, `kind`, `size_bytes`, `lines`, and `modified_at`. Defaults: limit={}, with_lines=false, include_hidden=true, respect_gitignore=true.",
                MAX_RESULTS
            ),
            params: vec![
                ToolParam::typed("pattern", "str"),
                ToolParam {
                    name: "path".into(),
                    r#type: "str".into(),
                    description: "Base directory to search in (default: current directory)".into(),
                    default_value: Some(serde_json::json!(".")),
                    required: false,
                },
                ToolParam {
                    name: "limit".into(),
                    r#type: "int".into(),
                    description: format!(
                        "Maximum results to return (default: {}). Use null or \"none\" for no cap.",
                        MAX_RESULTS
                    ),
                    default_value: Some(serde_json::json!(MAX_RESULTS)),
                    required: false,
                },
                ToolParam {
                    name: "with_lines".into(),
                    r#type: "bool".into(),
                    description: "Count text lines for file entries (`lines`). Default: false."
                        .into(),
                    default_value: Some(serde_json::json!(false)),
                    required: false,
                },
                ToolParam {
                    name: "include_hidden".into(),
                    r#type: "bool".into(),
                    description: "Include dotfiles and dot-directories. Default: true.".into(),
                    default_value: Some(serde_json::json!(true)),
                    required: false,
                },
                ToolParam {
                    name: "respect_gitignore".into(),
                    r#type: "bool".into(),
                    description: "Respect `.gitignore` and related ignore files. When true (default), `.gitignore` is honored only inside Git repos. When false, ignore-file processing is fully disabled.".into(),
                    default_value: Some(serde_json::json!(true)),
                    required: false,
                },
            ],
            returns: "dict".into(),
            examples: vec![],
            availability: crate::ToolAvailabilityConfig::documented(),
            activation: crate::ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
        }]
    }
    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let pattern = match require_str(args, "pattern") {
            Ok(s) => s,
            Err(e) => return e,
        };

        let base_dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let limit = match parse_limit(args) {
            Ok(l) => l,
            Err(e) => return e,
        };
        let with_lines = match parse_optional_bool(args, "with_lines", false) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let include_hidden = match parse_optional_bool(args, "include_hidden", true) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let respect_gitignore = match parse_optional_bool(args, "respect_gitignore", true) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let pattern = pattern.to_string();
        let base = PathBuf::from(base_dir);

        run_blocking(move || {
            if !base.exists() {
                return ToolResult::err_fmt(format_args!("Path does not exist: {}", base.display()));
            }
            if !base.is_dir() {
                return ToolResult::err_fmt(format_args!(
                    "{} is a file, not a directory. Pass the parent directory as path and use the pattern to match files.",
                    base.display()
                ));
            }

            let glob = match globset::GlobBuilder::new(&pattern)
                .literal_separator(false)
                .build()
            {
                Ok(glob) => glob,
                Err(err) => return ToolResult::err_fmt(format_args!("Invalid glob pattern: {err}")),
            };
            let matcher = match globset::GlobSetBuilder::new().add(glob).build() {
                Ok(matcher) => matcher,
                Err(err) => {
                    return ToolResult::err_fmt(format_args!("Failed to build glob matcher: {err}"));
                }
            };

            let files = match rg_file_list(&base, include_hidden, respect_gitignore, None, &[]) {
                Ok(files) => files,
                Err(err) => return err,
            };

            let mut matched_paths = BTreeSet::new();
            for file in files {
                let Ok(rel_path) = file.strip_prefix(&base) else {
                    continue;
                };
                if matcher.is_match(rel_path) {
                    matched_paths.insert(file.clone());
                }
                let components = rel_path.components().collect::<Vec<_>>();
                if components.len() <= 1 {
                    continue;
                }
                let mut current = PathBuf::new();
                for component in components.iter().take(components.len() - 1) {
                    current.push(component.as_os_str());
                    if matcher.is_match(&current) {
                        matched_paths.insert(base.join(&current));
                    }
                }
            }

            let mut matches = matched_paths
                .into_iter()
                .map(|path| build_path_entry(&path, with_lines))
                .collect::<Vec<_>>();
            matches.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.path.cmp(&b.0.path)));
            let total_matches = matches.len();
            if let Some(limit) = limit {
                matches.truncate(limit);
            }

            let items = matches.into_iter().map(|(entry, _)| entry).collect();
            ToolResult::ok(filesystem_entries_result(items, total_matches))
        })
        .await
    }
}

fn parse_limit(args: &serde_json::Value) -> Result<Option<usize>, ToolResult> {
    parse_optional_usize_arg(args, "limit", Some(MAX_RESULTS), true, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn items(result: &ToolResult) -> &Vec<serde_json::Value> {
        let obj = result.result.as_object().unwrap();
        obj.get("items").and_then(|v| v.as_array()).unwrap()
    }

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
        let arr = items(&result);
        let paths: Vec<&str> = arr
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.contains("a.rs")));
        assert!(paths.iter().any(|p| p.contains("b.rs")));
        assert!(!paths.iter().any(|p| p.contains("c.txt")));
        assert!(
            arr.iter()
                .all(|v| v.get("size_bytes").and_then(|x| x.as_u64()).is_some())
        );
        assert!(
            arr.iter()
                .all(|v| v.get("modified_at").and_then(|x| x.as_str()).is_some())
        );
        assert!(
            arr.iter()
                .all(|v| v.get("lines").map(|x| x.is_null()).unwrap_or(false))
        );
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
        assert!(items(&result).is_empty());
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
        let arr = items(&result);
        let paths: Vec<&str> = arr
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.contains("file.rs")));
    }

    #[tokio::test]
    async fn test_glob_truncation_marker() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.rs"), "").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap(), "limit": 2}),
            )
            .await;
        assert!(result.success);
        let arr = items(&result);
        assert_eq!(arr.len(), 2);
        let truncated = result
            .result
            .get("truncated")
            .and_then(|v| v.as_object())
            .unwrap();
        assert_eq!(truncated.get("shown").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(truncated.get("total").and_then(|v| v.as_u64()), Some(3));
        assert_eq!(truncated.get("omitted").and_then(|v| v.as_u64()), Some(1));
    }

    #[tokio::test]
    async fn test_glob_limit_none() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.rs"), "").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap(), "limit": null}),
            )
            .await;
        assert!(result.success);
        let arr = items(&result);
        assert_eq!(arr.len(), 3);
        assert!(
            result
                .result
                .get("truncated")
                .map(|v| v.is_null())
                .unwrap_or(false)
        );
    }

    #[tokio::test]
    async fn test_glob_with_lines() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "line1\nline2\nline3\n").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap(), "with_lines": true}),
            )
            .await;
        assert!(result.success);
        let arr = items(&result);
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].get("lines").and_then(|v| v.as_u64()), Some(3));
    }

    #[tokio::test]
    async fn test_glob_includes_hidden_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hidden.rs"), "").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("/.hidden.rs")));
    }

    #[tokio::test]
    async fn test_glob_respect_gitignore_false_disables_repo_gitignore() {
        let dir = TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .status()
            .unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.rs\n").unwrap();
        std::fs::write(dir.path().join("ignored.rs"), "").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({
                    "pattern": "*.rs",
                    "path": dir.path().to_str().unwrap(),
                    "respect_gitignore": false
                }),
            )
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("/ignored.rs")));
    }

    #[tokio::test]
    async fn test_glob_respect_gitignore_true_hides_repo_ignored_files() {
        let dir = TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .status()
            .unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.rs\n").unwrap();
        std::fs::write(dir.path().join("ignored.rs"), "").unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({
                    "pattern": "*.rs",
                    "path": dir.path().to_str().unwrap()
                }),
            )
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(!paths.iter().any(|p| p.ends_with("/ignored.rs")));
    }

    #[tokio::test]
    async fn test_glob_no_longer_hides_dot_git_entries_by_default() {
        let dir = TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .status()
            .unwrap();
        let tool = Glob;
        let result = tool
            .execute(
                "glob",
                &json!({"pattern": ".git/**", "path": dir.path().to_str().unwrap()}),
            )
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.contains("/.git/")));
    }
}
