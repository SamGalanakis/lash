use serde_json::json;

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
        let path = match require_str(args, "path") {
            Ok(s) => s,
            Err(e) => return e,
        };

        let git_ref = args.get("ref").and_then(|v| v.as_str()).unwrap_or("HEAD");

        let output = tokio::process::Command::new("git")
            .args(["diff", git_ref, "--", path])
            .output()
            .await;

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);

                if !out.status.success() {
                    return ToolResult::err_fmt(format_args!("git diff failed: {}", stderr.trim()));
                }

                if stdout.is_empty() {
                    ToolResult::ok(json!(format!(
                        "No changes detected in {} vs {}",
                        path, git_ref
                    )))
                } else {
                    ToolResult::ok(json!(stdout.as_ref()))
                }
            }
            Err(e) => ToolResult::err_fmt(format_args!("Failed to run git: {e}")),
        }
    }
}
