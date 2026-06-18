use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use lash_core::{ToolCall, ToolDefinition, ToolResult, ToolRetryPolicy, ToolScheduling};
use lash_tool_support::{
    FS_DEFAULTS_PREAMBLE, FilesystemEntriesOutput, OptionalUsizeArg, StaticToolExecute,
    StaticToolProvider, ToolDefinitionLashlangExt, build_path_entry, default_ls_depth,
    default_ls_limit, default_path_dot, default_true, execute_typed_tool,
    filesystem_entries_output, rg_file_list, run_blocking_value,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// List filesystem entries in a directory tree.
#[derive(Default)]
pub struct Ls;

/// Build the cached `ls` tool provider.
pub fn ls_provider() -> StaticToolProvider<Ls> {
    StaticToolProvider::new(vec![ls_tool_definition()], Ls)
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct LsArgs {
    /// Directory to list.
    #[serde(default = "default_path_dot")]
    path: String,
    /// Additional glob patterns to ignore.
    #[serde(default)]
    ignore: Vec<String>,
    /// Maximum directory depth to traverse. Use null or "none" for no cap.
    #[serde(default = "default_ls_depth")]
    depth: OptionalUsizeArg,
    /// Maximum entries to return. Use null or "none" for no cap.
    #[serde(default = "default_ls_limit")]
    limit: OptionalUsizeArg,
    /// Count text lines for file entries.
    #[serde(default)]
    with_lines: bool,
    /// Include dotfiles and dot-directories.
    #[serde(default = "default_true")]
    include_hidden: bool,
    /// Respect `.gitignore` and related ignore files.
    #[serde(default = "default_true")]
    respect_gitignore: bool,
}

#[async_trait::async_trait]
impl StaticToolExecute for Ls {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        execute_typed_tool::<LsArgs, FilesystemEntriesOutput, _, _>(call.args, |args| async move {
            match run_blocking_value(move || execute_ls_sync(args)).await {
                Ok(result) => result,
                Err(err) => Err(ToolResult::err_fmt(format_args!("{err}"))),
            }
        })
        .await
    }
}

fn ls_tool_definition() -> ToolDefinition {
    ToolDefinition::typed::<LsArgs, FilesystemEntriesOutput>(
                "tool:ls",
                "ls",
                [
                    "List filesystem entries. ",
                    FS_DEFAULTS_PREAMBLE,
                    " Returns a record with `items` sorted by path. Each item has `path`, `kind`, `size_bytes`, `lines`, and `modified_at`. Defaults: depth=3, limit=500, with_lines=false, include_hidden=true, respect_gitignore=true.",
                ]
                .concat(),
            )
            .with_examples(vec![
                r#"await files.list({ path: ".", depth: 1, limit: 100 })?"#.into(),
                r#"await files.list({ path: "crates/lash/src/tools", with_lines: true })?"#.into(),
            ])
            .with_lashlang_binding(lash_tool_support::lashlang_binding(
                ["files"],
                "list",
                &["list_files", "list_directory"],
            ))
            .with_scheduling(ToolScheduling::Parallel)
            .with_retry_policy(ToolRetryPolicy::safe(2, 25, 100))
}

fn execute_ls_sync(args: LsArgs) -> Result<FilesystemEntriesOutput, ToolResult> {
    let max_depth = args.depth.into_option("depth", 1)?;
    let limit = args.limit.into_option("limit", 1)?;
    let base = PathBuf::from(args.path);
    if !base.is_dir() {
        return Err(ToolResult::err_fmt(format_args!(
            "Not a directory: {}",
            base.display()
        )));
    }

    let globs = args
        .ignore
        .into_iter()
        .map(|pattern| format!("!{pattern}"))
        .collect::<Vec<_>>();

    let files = rg_file_list(
        &base,
        args.include_hidden,
        args.respect_gitignore,
        None,
        &globs,
    )?;

    let all_paths = collect_ls_paths(&base, &files, max_depth);
    let total_entries = all_paths.len();
    let shown_paths = match limit {
        Some(limit) => all_paths.into_iter().take(limit).collect::<Vec<_>>(),
        None => all_paths.into_iter().collect::<Vec<_>>(),
    };
    let items = shown_paths
        .into_iter()
        .map(|path| build_path_entry(&path, args.with_lines).0)
        .collect();
    Ok(filesystem_entries_output(items, total_entries))
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

    #[test]
    fn ls_contract_documents_result_shape() {
        let definition = ls_tool_definition();
        assert_eq!(definition.contract.output_schema["type"], json!("object"));
        assert!(definition.contract.output_schema["properties"]["items"].is_object());
        assert!(
            definition
                .compact_contract()
                .render_signature()
                .contains("items")
        );
    }

    #[tokio::test]
    async fn test_ls_files_and_dirs() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        std::fs::write(dir.path().join("subdir/nested.rs"), "").unwrap();
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
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
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        assert!(items(&result).is_empty());
    }

    #[tokio::test]
    async fn test_ls_not_a_dir() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "").unwrap();
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": path.to_str().unwrap()}),
        )
        .await;
        assert!(!result.is_success());
    }

    #[tokio::test]
    async fn test_ls_depth_limit() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("a/b/c")).unwrap();
        std::fs::write(dir.path().join("a/b/c/file.txt"), "").unwrap();
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap(), "depth": 1}),
        )
        .await;
        assert!(result.is_success());
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
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap(), "limit": 2}),
        )
        .await;
        assert!(result.is_success());
        assert_eq!(items(&result).len(), 2);
        let value = result.value_for_projection();
        let truncated = value.get("truncated").and_then(|v| v.as_object()).unwrap();
        assert_eq!(truncated.get("shown").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(truncated.get("total").and_then(|v| v.as_u64()), Some(3));
        assert_eq!(truncated.get("omitted").and_then(|v| v.as_u64()), Some(1));
    }

    #[tokio::test]
    async fn test_ls_with_lines() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "line1\nline2\n").unwrap();
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap(), "with_lines": true}),
        )
        .await;
        assert!(result.is_success());
        let arr = items(&result);
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].get("lines").and_then(|v| v.as_u64()), Some(2));
    }

    #[tokio::test]
    async fn test_ls_includes_hidden_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".env"), "KEY=value\n").unwrap();
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("/.env")));
    }

    #[tokio::test]
    async fn test_ls_respect_gitignore_default_does_not_apply_gitignore_outside_repo() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "").unwrap();
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
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
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({
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
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({
                "path": dir.path().to_str().unwrap()
            }),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
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
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("/.git")));
        assert!(paths.iter().any(|p| p.contains("/.git/")));
    }

    #[tokio::test]
    async fn test_ls_does_not_hide_node_modules_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        std::fs::write(dir.path().join("node_modules/pkg/index.js"), "").unwrap();
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({"path": dir.path().to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
            .collect();
        assert!(paths.iter().any(|p| p.contains("node_modules")));
    }

    #[tokio::test]
    async fn test_ls_ignore_parameter_excludes_matching_paths() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        std::fs::write(dir.path().join("node_modules/pkg/index.js"), "").unwrap();
        let result = lash_core::testing::run_tool(
            &ls_provider(),
            "ls",
            &json!({
                "path": dir.path().to_str().unwrap(),
                "ignore": ["**/node_modules/**", "**/node_modules"]
            }),
        )
        .await;
        assert!(result.is_success());
        let paths: Vec<String> = items(&result)
            .iter()
            .filter_map(|v| v.get("path").and_then(|x| x.as_str()).map(str::to_string))
            .collect();
        assert!(!paths.iter().any(|p| p.contains("node_modules")));
    }
}
