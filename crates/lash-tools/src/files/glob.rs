use std::collections::BTreeSet;
use std::path::PathBuf;

use lash_core::{ToolCall, ToolDefinition, ToolResult, ToolRetryPolicy, ToolScheduling};

use lash_tool_support::{
    FS_DEFAULTS_PREAMBLE, FilesystemEntriesOutput, OptionalUsizeArg, StaticToolExecute,
    StaticToolProvider, ToolDefinitionLashlangExt, build_path_entry, default_glob_limit,
    default_path_dot, default_true, execute_typed_tool, filesystem_entries_output,
    invalid_tool_args, non_empty_string, rg_file_list, run_blocking_value,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// Find files by glob pattern.
#[derive(Default)]
pub struct Glob;

/// Build the cached `glob` tool provider.
pub fn glob_provider() -> StaticToolProvider<Glob> {
    StaticToolProvider::new(vec![glob_tool_definition()], Glob)
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GlobArgs {
    /// Glob pattern to match.
    pattern: String,
    /// Base directory to search in.
    #[serde(default = "default_path_dot")]
    path: String,
    /// Maximum results to return. Use null or "none" for no cap.
    #[serde(default = "default_glob_limit")]
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
impl StaticToolExecute for Glob {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        execute_typed_tool::<GlobArgs, FilesystemEntriesOutput, _, _>(
            call.args,
            |args| async move {
                match run_blocking_value(move || execute_glob_sync(args)).await {
                    Ok(result) => result,
                    Err(err) => Err(ToolResult::err_fmt(format_args!("{err}"))),
                }
            },
        )
        .await
    }
}

fn execute_glob_sync(args: GlobArgs) -> Result<FilesystemEntriesOutput, ToolResult> {
    non_empty_string(&args.pattern, "pattern")?;
    let limit = args.limit.into_option("limit", 1)?;
    let base = PathBuf::from(args.path);
    if !base.exists() {
        return Err(ToolResult::err_fmt(format_args!(
            "Path does not exist: {}",
            base.display()
        )));
    }
    if !base.is_dir() {
        return Err(ToolResult::err_fmt(format_args!(
            "{} is a file, not a directory. Pass the parent directory as path and use the pattern to match files.",
            base.display()
        )));
    }

    let glob = globset::GlobBuilder::new(&args.pattern)
        .literal_separator(false)
        .build()
        .map_err(|err| invalid_tool_args(format!("Invalid glob pattern: {err}")))?;
    let matcher = globset::GlobSetBuilder::new()
        .add(glob)
        .build()
        .map_err(|err| ToolResult::err_fmt(format_args!("Failed to build glob matcher: {err}")))?;

    let files = rg_file_list(
        &base,
        args.include_hidden,
        args.respect_gitignore,
        None,
        &[],
    )?;

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
        .map(|path| build_path_entry(&path, args.with_lines))
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.path.cmp(&b.0.path)));
    let total_matches = matches.len();
    if let Some(limit) = limit {
        matches.truncate(limit);
    }

    let items = matches.into_iter().map(|(entry, _)| entry).collect();
    Ok(filesystem_entries_output(items, total_matches))
}

fn glob_tool_definition() -> ToolDefinition {
    ToolDefinition::typed::<GlobArgs, FilesystemEntriesOutput>(
                "tool:glob",
                "glob",
                [
                    "Find filesystem entries by glob. ",
                    FS_DEFAULTS_PREAMBLE,
                    " Returns a record with `items` sorted by `modified_at` (newest first). Each item has `path`, `kind`, `size_bytes`, `lines`, and `modified_at`. Defaults: limit=100, with_lines=false, include_hidden=true, respect_gitignore=true.",
                ]
                .concat(),
            )
            .with_examples(vec![
                r#"await files.glob({ pattern: "**/*.rs", path: "crates/lash/src", limit: 50 })?"#.into(),
                r#"await files.glob({ pattern: "**/Cargo.toml", path: "." })?"#.into(),
            ])
            .with_lashlang_binding(lash_tool_support::lashlang_binding(
                ["files"],
                "glob",
                &["find_files"],
            ))
            .with_scheduling(ToolScheduling::Parallel)
            .with_retry_policy(ToolRetryPolicy::safe(2, 25, 100))
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
    fn glob_contract_documents_result_shape() {
        let definition = glob_tool_definition();
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
