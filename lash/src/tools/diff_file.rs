use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::require_str;

/// Show unified diff for a file against a git ref.
#[derive(Default)]
pub struct DiffFile;

#[async_trait::async_trait]
impl ToolProvider for DiffFile {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "diff_file".into(),
            description: "Show unified diff of a file against a git ref. \
                Returns standard unified diff output for verifying edits."
                .into(),
            params: vec![
                ToolParam::typed("path", "str"),
                ToolParam {
                    name: "ref".into(),
                    r#type: "str".into(),
                    description: "Git ref to diff against (default: HEAD)".into(),
                    required: false,
                },
            ],
            returns: "str".into(),
            hidden: false,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let path_str = match require_str(args, "path") {
            Ok(s) => s,
            Err(e) => return e,
        };

        let git_ref = args.get("ref").and_then(|v| v.as_str()).unwrap_or("HEAD");

        // Resolve to absolute path
        let abs_path = match std::path::absolute(Path::new(path_str)) {
            Ok(p) => p,
            Err(e) => return ToolResult::err_fmt(format_args!("Invalid path: {e}")),
        };

        if !abs_path.is_file() {
            return ToolResult::err_fmt(format_args!("Not a file: {}", abs_path.display()));
        }

        // Read current file content
        let current = match std::fs::read_to_string(&abs_path) {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err_fmt(format_args!(
                    "Failed to read {}: {e}",
                    abs_path.display()
                ));
            }
        };

        // Find git repo root from the file's directory
        let file_dir = abs_path.parent().unwrap_or(Path::new("."));
        let repo_root = match tokio::process::Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .current_dir(file_dir)
            .output()
            .await
        {
            Ok(out) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout).trim().to_string()
            }
            _ => {
                return ToolResult::err_fmt(format_args!(
                    "Not in a git repository: {}",
                    file_dir.display()
                ));
            }
        };

        // Compute path relative to repo root
        let repo_root_path = Path::new(&repo_root);
        let rel_path = match abs_path.strip_prefix(repo_root_path) {
            Ok(p) => p.to_string_lossy().to_string(),
            Err(_) => {
                return ToolResult::err_fmt(format_args!(
                    "File {} is not under repo root {}",
                    abs_path.display(),
                    repo_root
                ));
            }
        };

        // Get old content via git show
        let git_path = format!("{}:{}", git_ref, rel_path);
        let old_content = match tokio::process::Command::new("git")
            .args(["show", &git_path])
            .current_dir(repo_root_path)
            .output()
            .await
        {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                // File doesn't exist in the ref — treat as new file
                if stderr.contains("does not exist") || stderr.contains("not exist") {
                    String::new()
                } else {
                    return ToolResult::err_fmt(format_args!(
                        "git show {} failed: {}",
                        git_path,
                        stderr.trim()
                    ));
                }
            }
            Err(e) => return ToolResult::err_fmt(format_args!("Failed to run git: {e}")),
        };

        // Generate unified diff
        let diff = similar::TextDiff::from_lines(&old_content, &current);
        let unified = diff
            .unified_diff()
            .header(&format!("a/{}", rel_path), &format!("b/{}", rel_path))
            .to_string();

        if unified.is_empty() {
            ToolResult::ok(json!(format!(
                "No changes detected in {} vs {}",
                path_str, git_ref
            )))
        } else {
            ToolResult::ok(json!(unified))
        }
    }
}
