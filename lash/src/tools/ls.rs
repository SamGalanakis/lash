use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::{
    build_path_entry, filesystem_entries_result, parse_optional_bool, parse_optional_usize_arg,
    rg_file_list, run_blocking,
};
use crate::{ToolDefinition, ToolExecutionMode, ToolParam, ToolProvider, ToolResult};

/// List filesystem entries in a directory tree.
#[derive(Default)]
pub struct Ls;

const DEFAULT_DEPTH: usize = 3;
const MAX_ENTRIES: usize = 500;

#[async_trait::async_trait]
impl ToolProvider for Ls {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "ls".into(),
            description: format!(
                "List filesystem entries. By default this includes hidden files and respects `.gitignore` only inside Git repos. Returns a record with `items` sorted by path. Each item has `path`, `kind`, `size_bytes`, `lines`, and `modified_at`. Defaults: depth={}, limit={}, with_lines=false, include_hidden=true, respect_gitignore=true.",
                DEFAULT_DEPTH, MAX_ENTRIES
            ),
            params: vec![
                ToolParam {
                    name: "path".into(),
                    r#type: "str".into(),
                    description: "Directory to list (default: current directory)".into(),
                    default_value: Some(serde_json::json!(".")),
                    required: false,
                },
                ToolParam {
                    name: "ignore".into(),
                    r#type: "list".into(),
                    description: "Additional glob patterns to ignore.".into(),
                    default_value: None,
                    required: false,
                },
                ToolParam {
                    name: "depth".into(),
                    r#type: "int".into(),
                    description: format!(
                        "Maximum directory depth to traverse (default: {}). Use null or \"none\" for no depth cap.",
                        DEFAULT_DEPTH
                    ),
                    default_value: Some(serde_json::json!(DEFAULT_DEPTH)),
                    required: false,
                },
                ToolParam {
                    name: "limit".into(),
                    r#type: "int".into(),
                    description: format!(
                        "Maximum entries to return (default: {}). Use null or \"none\" for no cap.",
                        MAX_ENTRIES
                    ),
                    default_value: Some(serde_json::json!(MAX_ENTRIES)),
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
            discovery: crate::tools::discovery_metadata("filesystem", &["list_files", "list_directory"]),
            execution_mode: ToolExecutionMode::Parallel,
        }]
    }
    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let base_dir = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let ignore_patterns: Vec<&str> = args
            .get("ignore")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let max_depth = match parse_depth(args) {
            Ok(d) => d,
            Err(e) => return e,
        };
        let limit = match parse_limit(args) {
            Ok(d) => d,
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
        let ignore_patterns = ignore_patterns
            .into_iter()
            .map(|pattern| pattern.to_string())
            .collect::<Vec<_>>();

        run_blocking(move || {
            if !base.is_dir() {
                return ToolResult::err_fmt(format_args!("Not a directory: {}", base.display()));
            }

            let globs = ignore_patterns
                .into_iter()
                .map(|pattern| format!("!{pattern}"))
                .collect::<Vec<_>>();

            let files = match rg_file_list(&base, include_hidden, respect_gitignore, None, &globs) {
                Ok(files) => files,
                Err(err) => return err,
            };

            let all_paths = collect_ls_paths(&base, &files, max_depth);
            let total_entries = all_paths.len();
            let shown_paths = match limit {
                Some(limit) => all_paths.into_iter().take(limit).collect::<Vec<_>>(),
                None => all_paths.into_iter().collect::<Vec<_>>(),
            };
            let items = shown_paths
                .into_iter()
                .map(|path| build_path_entry(&path, with_lines).0)
                .collect();
            ToolResult::ok(filesystem_entries_result(items, total_entries))
        })
        .await
    }
}

fn parse_depth(args: &serde_json::Value) -> Result<Option<usize>, ToolResult> {
    parse_optional_usize_arg(args, "depth", Some(DEFAULT_DEPTH), true, 1)
}

fn parse_limit(args: &serde_json::Value) -> Result<Option<usize>, ToolResult> {
    parse_optional_usize_arg(args, "limit", Some(MAX_ENTRIES), true, 1)
}

fn collect_ls_paths(base: &Path, files: &[PathBuf], max_depth: Option<usize>) -> BTreeSet<PathBuf> {
    let mut entries = BTreeSet::new();
    for file in files {
        let Ok(rel_path) = file.strip_prefix(base) else {
            continue;
        };
        let components = rel_path.components().collect::<Vec<_>>();
        if components.is_empty() {
            continue;
        }

        let max_file_depth = max_depth.unwrap_or(usize::MAX);
        if components.len() <= max_file_depth {
            entries.insert(file.clone());
        }

        let dir_depth = components.len().saturating_sub(1);
        let dirs_to_include = max_depth.map_or(dir_depth, |depth| depth.min(dir_depth));
        let mut current = PathBuf::new();
        for component in components.iter().take(dirs_to_include) {
            current.push(component.as_os_str());
            entries.insert(base.join(&current));
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolProvider;
    use serde_json::json;
    use tempfile::TempDir;

    fn items(result: &ToolResult) -> &Vec<serde_json::Value> {
        let obj = result.result.as_object().unwrap();
        obj.get("items").and_then(|v| v.as_array()).unwrap()
    }

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
        let arr = items(&result);
        let paths: Vec<&str> = arr
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.contains("file.txt")));
        assert!(paths.iter().any(|p| p.contains("subdir")));
        assert!(paths.iter().any(|p| p.contains("nested.rs")));
    }

    #[tokio::test]
    async fn test_ls_empty_dir() {
        let dir = TempDir::new().unwrap();
        let tool = Ls;
        let result = tool
            .execute("ls", &json!({"path": dir.path().to_str().unwrap()}))
            .await;
        assert!(result.success);
        assert!(items(&result).is_empty());
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

    #[tokio::test]
    async fn test_ls_depth_limit() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("a/b/c")).unwrap();
        std::fs::write(dir.path().join("a/b/c/file.txt"), "").unwrap();
        let tool = Ls;
        let result = tool
            .execute(
                "ls",
                &json!({"path": dir.path().to_str().unwrap(), "depth": 1}),
            )
            .await;
        assert!(result.success);
        let arr = items(&result);
        let paths: Vec<&str> = arr
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("/a")));
        assert!(!paths.iter().any(|p| p.ends_with("/b")));
        assert!(!paths.iter().any(|p| p.ends_with("/file.txt")));
    }

    #[tokio::test]
    async fn test_ls_limit_truncation_metadata() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();
        let tool = Ls;
        let result = tool
            .execute(
                "ls",
                &json!({"path": dir.path().to_str().unwrap(), "limit": 2}),
            )
            .await;
        assert!(result.success);
        assert_eq!(items(&result).len(), 2);
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
    async fn test_ls_with_lines() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "line1\nline2\n").unwrap();
        let tool = Ls;
        let result = tool
            .execute(
                "ls",
                &json!({"path": dir.path().to_str().unwrap(), "with_lines": true}),
            )
            .await;
        assert!(result.success);
        let arr = items(&result);
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].get("lines").and_then(|v| v.as_u64()), Some(2));
    }

    #[tokio::test]
    async fn test_ls_includes_hidden_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".env"), "KEY=value\n").unwrap();
        let tool = Ls;
        let result = tool
            .execute("ls", &json!({"path": dir.path().to_str().unwrap()}))
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("/.env")));
    }

    #[tokio::test]
    async fn test_ls_respect_gitignore_default_does_not_apply_gitignore_outside_repo() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "").unwrap();
        let tool = Ls;
        let result = tool
            .execute("ls", &json!({"path": dir.path().to_str().unwrap()}))
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("/ignored.txt")));
    }

    #[tokio::test]
    async fn test_ls_respect_gitignore_false_disables_repo_gitignore() {
        let dir = TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .status()
            .unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "").unwrap();
        let tool = Ls;
        let result = tool
            .execute(
                "ls",
                &json!({
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
        assert!(paths.iter().any(|p| p.ends_with("/ignored.txt")));
    }

    #[tokio::test]
    async fn test_ls_respect_gitignore_true_hides_repo_ignored_files() {
        let dir = TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .status()
            .unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "").unwrap();
        let tool = Ls;
        let result = tool
            .execute(
                "ls",
                &json!({
                    "path": dir.path().to_str().unwrap()
                }),
            )
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(!paths.iter().any(|p| p.ends_with("/ignored.txt")));
    }

    #[tokio::test]
    async fn test_ls_no_longer_hides_dot_git_entries_by_default() {
        let dir = TempDir::new().unwrap();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .status()
            .unwrap();
        let tool = Ls;
        let result = tool
            .execute("ls", &json!({"path": dir.path().to_str().unwrap()}))
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("/.git")));
        assert!(paths.iter().any(|p| p.contains("/.git/")));
    }

    #[tokio::test]
    async fn test_ls_does_not_hide_node_modules_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        std::fs::write(dir.path().join("node_modules/pkg/index.js"), "").unwrap();
        let tool = Ls;
        let result = tool
            .execute("ls", &json!({"path": dir.path().to_str().unwrap()}))
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(paths.iter().any(|p| p.contains("node_modules")));
    }

    #[tokio::test]
    async fn test_ls_ignore_parameter_excludes_matching_paths() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        std::fs::write(dir.path().join("node_modules/pkg/index.js"), "").unwrap();
        let tool = Ls;
        let result = tool
            .execute(
                "ls",
                &json!({
                    "path": dir.path().to_str().unwrap(),
                    "ignore": ["**/node_modules/**", "**/node_modules"]
                }),
            )
            .await;
        assert!(result.success);
        let paths: Vec<&str> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()))
            .collect();
        assert!(!paths.iter().any(|p| p.contains("node_modules")));
    }
}
