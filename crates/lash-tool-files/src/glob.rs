use std::collections::BTreeSet;
use std::path::PathBuf;

use lash_core::{ToolCall, ToolDefinition, ToolResult, ToolRetryPolicy, ToolScheduling};

use lash_tool_support::{
    FS_DEFAULTS_PREAMBLE, StaticToolExecute, StaticToolProvider, build_path_entry,
    filesystem_entries_result, object_schema, parse_optional_bool, parse_optional_usize_arg,
    require_str, rg_file_list, run_blocking,
};

/// Find files by glob pattern.
#[derive(Default)]
pub struct Glob;

/// Build the cached `glob` tool provider.
pub fn glob_provider() -> StaticToolProvider<Glob> {
    StaticToolProvider::new(vec![glob_tool_definition()], Glob)
}

const MAX_RESULTS: usize = 100;

#[async_trait::async_trait]
impl StaticToolExecute for Glob {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let args = call.args;
        let pattern = match require_str(args, "pattern") {
            Ok(s) => s,
            Err(e) => return e,
        };

        let base_dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let limit = match parse_limit(args) {
            Ok(limit) => limit,
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
        let base = PathBuf::from(base_dir);
        let pattern = pattern.to_string();

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

fn glob_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
                "tool:glob",
                "glob",
                [
                    "Find filesystem entries by glob. ",
                    FS_DEFAULTS_PREAMBLE,
                    " Returns a record with `items` sorted by `modified_at` (newest first). Each item has `path`, `kind`, `size_bytes`, `lines`, and `modified_at`. Defaults: limit=100, with_lines=false, include_hidden=true, respect_gitignore=true.",
                ]
                .concat(),
                object_schema(
                    serde_json::json!({
                        "pattern": { "type": "string" },
                        "path": {
                            "type": "string",
                            "default": ".",
                            "description": "Base directory to search in (default: current directory)"
                        },
                        "limit": {
                            "type": ["integer", "null", "string"],
                            "minimum": 1,
                            "default": MAX_RESULTS,
                            "description": "Maximum results to return (default: 100). Use null or \"none\" for no cap."
                        },
                        "with_lines": {
                            "type": "boolean",
                            "default": false,
                            "description": "Count text lines for file entries (`lines`). Default: false."
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "default": true,
                            "description": "Include dotfiles and dot-directories. Default: true."
                        },
                        "respect_gitignore": {
                            "type": "boolean",
                            "default": true,
                            "description": "Respect `.gitignore` and related ignore files. When true (default), `.gitignore` is honored only inside Git repos. When false, ignore-file processing is fully disabled."
                        }
                    }),
                    &["pattern"],
                ),
                serde_json::json!({ "type": "object", "additionalProperties": true }),
            )
            .with_examples(vec![
                r#"await files.glob({ pattern: "**/*.rs", path: "crates/lash/src", limit: 50 })?"#.into(),
                r#"await files.glob({ pattern: "**/Cargo.toml", path: "." })?"#.into(),
            ])
            .with_agent_surface(lash_tool_support::agent_surface(
                ["files"],
                "glob",
                &["find_files"],
            ))
            .with_scheduling(ToolScheduling::Parallel)
            .with_retry_policy(ToolRetryPolicy::safe(2, 25, 100))
}

fn parse_limit(args: &serde_json::Value) -> Result<Option<usize>, ToolResult> {
    parse_optional_usize_arg(args, "limit", Some(MAX_RESULTS), true, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn items(result: &ToolResult) -> Vec<serde_json::Value> {
        let value = result.value_for_projection();
        value
            .get("items")
            .and_then(|v| v.as_array())
            .unwrap()
            .clone()
    }

    #[tokio::test]
    async fn test_glob_matches() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
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
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        assert!(items(&result).is_empty());
    }

    #[tokio::test]
    async fn test_glob_nested() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("sub/deep")).unwrap();
        std::fs::write(dir.path().join("sub/deep/file.rs"), "").unwrap();
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({"pattern": "**/*.rs", "path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
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
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap(), "limit": 2}),
        )
        .await;
        assert!(result.is_success());
        let arr = items(&result);
        assert_eq!(arr.len(), 2);
        let value = result.value_for_projection();
        let truncated = value.get("truncated").and_then(|v| v.as_object()).unwrap();
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
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap(), "limit": null}),
        )
        .await;
        assert!(result.is_success());
        let arr = items(&result);
        assert_eq!(arr.len(), 3);
        assert!(
            result
                .value_for_projection()
                .get("truncated")
                .map(|v| v.is_null())
                .unwrap_or(false)
        );
    }

    #[tokio::test]
    async fn test_glob_with_lines() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "line1\nline2\nline3\n").unwrap();
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap(), "with_lines": true}),
        )
        .await;
        assert!(result.is_success());
        let arr = items(&result);
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].get("lines").and_then(|v| v.as_u64()), Some(3));
    }

    #[tokio::test]
    async fn test_glob_includes_hidden_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hidden.rs"), "").unwrap();
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({"pattern": "*.rs", "path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
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
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({
                "pattern": "*.rs",
                "path": dir.path().to_str().unwrap(),
                "respect_gitignore": false
            }),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
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
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({
                "pattern": "*.rs",
                "path": dir.path().to_str().unwrap()
            }),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
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
        let result = lash_core::testing::run_tool(
            &glob_provider(),
            "glob",
            &json!({"pattern": ".git/**", "path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
            .collect();
        assert!(paths.iter().any(|p| p.contains("/.git/")));
    }
}
