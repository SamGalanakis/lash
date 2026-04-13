mod apply_patch;
mod ask;
mod batch;
mod composite;
mod fetch_url;
mod glob;
mod grep;
mod ls;
mod read_file;
mod shell;
mod show_snippet_to_user;
mod state;
mod update_plan;
mod wait;
mod web_search;

pub use apply_patch::ApplyPatchTool;
#[cfg(feature = "sqlite-store")]
pub(crate) use apply_patch::{PatchAction, inspect_patch_ops};
pub use ask::AskTool;
pub(crate) use composite::CompositeToolProvider;
pub use fetch_url::FetchUrl;
pub use glob::Glob;
pub use grep::Grep;
pub use ls::Ls;
pub use read_file::{ReadFile, ReadFilePluginFactory};
pub use shell::StandardShell;
pub use shell::shell_prompt_contributions;
pub use show_snippet_to_user::ShowSnippetToUser;
pub use state::{StateStore, StateToolsPluginFactory};
pub use update_plan::UpdatePlanTool;
pub use wait::WaitTool;
pub use web_search::WebSearch;

use crate::ToolResult;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeTool {
    Batch,
}

impl NativeTool {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Batch => "batch",
        }
    }

    pub(crate) fn definition(self) -> crate::ToolDefinition {
        match self {
            Self::Batch => batch::batch_tool_definition(),
        }
    }
}

pub(crate) fn native_tools(mode: crate::ExecutionMode) -> &'static [NativeTool] {
    match mode {
        crate::ExecutionMode::Standard => &[NativeTool::Batch],
        crate::ExecutionMode::Rlm => &[],
    }
}

pub(crate) fn all_native_tool_names() -> impl Iterator<Item = &'static str> {
    [NativeTool::Batch].into_iter().map(NativeTool::name)
}

pub(crate) fn find_native_tool(mode: crate::ExecutionMode, name: &str) -> Option<NativeTool> {
    if matches!(mode, crate::ExecutionMode::Rlm) && name == NativeTool::Batch.name() {
        return Some(NativeTool::Batch);
    }
    native_tools(mode)
        .iter()
        .copied()
        .find(|tool| tool.name() == name)
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct PathEntry {
    pub path: String,
    pub kind: String,
    pub size_bytes: u64,
    pub lines: Option<u64>,
    pub modified_at: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct TruncationMeta {
    pub shown: usize,
    pub total: usize,
    pub omitted: usize,
}

/// Extract a required non-empty string arg, or return ToolResult::err.
pub(crate) fn require_str<'a>(
    args: &'a serde_json::Value,
    key: &str,
) -> Result<&'a str, ToolResult> {
    args.get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ToolResult::err_fmt(format_args!("Missing required parameter: {key}")))
}

/// Parse optional bool arg with a default.
pub(crate) fn parse_optional_bool(
    args: &serde_json::Value,
    key: &str,
    default: bool,
) -> Result<bool, ToolResult> {
    match args.get(key) {
        None => Ok(default),
        Some(v) if v.is_null() => Ok(default),
        Some(v) => match v.as_bool() {
            Some(b) => Ok(b),
            None => Err(ToolResult::err_fmt(format_args!(
                "Invalid {key}: expected bool"
            ))),
        },
    }
}

/// Parse an optional positive integer arg.
/// Accepts `null` or `"none"` when `allow_none` is true.
pub(crate) fn parse_optional_usize_arg(
    args: &serde_json::Value,
    key: &str,
    default: Option<usize>,
    allow_none: bool,
    min: usize,
) -> Result<Option<usize>, ToolResult> {
    match args.get(key) {
        None => Ok(default),
        Some(v) if v.is_null() => {
            if allow_none {
                Ok(None)
            } else {
                Err(ToolResult::err_fmt(format_args!(
                    "Invalid {key}: expected int >= {min}"
                )))
            }
        }
        Some(v) => {
            if let Some(s) = v.as_str() {
                if allow_none && s.eq_ignore_ascii_case("none") {
                    return Ok(None);
                }
                return Err(ToolResult::err_fmt(format_args!(
                    "Invalid {key}: expected int{}",
                    if allow_none {
                        ", null, or \"none\""
                    } else {
                        ""
                    }
                )));
            }
            let n = v.as_u64().ok_or_else(|| {
                ToolResult::err_fmt(format_args!(
                    "Invalid {key}: expected int{}",
                    if allow_none {
                        ", null, or \"none\""
                    } else {
                        ""
                    }
                ))
            })? as usize;
            if n < min {
                return Err(ToolResult::err_fmt(format_args!(
                    "Invalid {key}: must be >= {min}{}",
                    if allow_none {
                        ", or use null/\"none\" for no cap"
                    } else {
                        ""
                    }
                )));
            }
            Ok(Some(n))
        }
    }
}

pub(crate) fn project_tool_catalog(
    definitions: impl IntoIterator<Item = crate::ToolDefinition>,
) -> Vec<serde_json::Value> {
    definitions
        .into_iter()
        .filter(|d| d.enabled)
        .map(|d| {
            serde_json::json!({
                "name": d.name,
                "description": d.description,
                "params": d.params,
                "returns": d.returns,
                "examples": d.examples,
                "injected": d.injected,
                "enabled": d.enabled,
            })
        })
        .collect()
}

/// Run blocking filesystem work off the async runtime.
pub(crate) async fn run_blocking<F>(f: F) -> ToolResult
where
    F: FnOnce() -> ToolResult + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => result,
        Err(e) => ToolResult::err_fmt(format_args!("blocking task failed: {e}")),
    }
}

/// Build a normalized filesystem entry for tool output.
/// Returns the entry plus raw mtime for optional sorting.
pub(crate) fn build_path_entry(path: &Path, with_lines: bool) -> (PathEntry, SystemTime) {
    let fallback_mtime = UNIX_EPOCH;
    let path_str = path.to_string_lossy().to_string();

    let metadata = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(_) => {
            let entry = PathEntry {
                path: path_str,
                kind: "other".to_string(),
                size_bytes: 0,
                lines: None,
                modified_at: format_time_rfc3339(fallback_mtime),
            };
            return (entry, fallback_mtime);
        }
    };

    let file_type = metadata.file_type();
    let kind = if file_type.is_symlink() {
        "symlink"
    } else if file_type.is_dir() {
        "dir"
    } else if file_type.is_file() {
        "file"
    } else {
        "other"
    };

    let mtime = metadata.modified().unwrap_or(fallback_mtime);
    let lines = if with_lines && kind == "file" {
        count_text_lines(path)
    } else {
        None
    };

    let entry = PathEntry {
        path: path_str,
        kind: kind.to_string(),
        size_bytes: metadata.len(),
        lines,
        modified_at: format_time_rfc3339(mtime),
    };
    (entry, mtime)
}

pub(crate) fn rg_file_list(
    base: &Path,
    include_hidden: bool,
    respect_gitignore: bool,
    max_depth: Option<usize>,
    globs: &[String],
) -> Result<Vec<PathBuf>, ToolResult> {
    let mut cmd = Command::new("rg");
    cmd.arg("--files").arg("--null").arg("--no-config");
    if include_hidden {
        cmd.arg("--hidden");
    }
    if !respect_gitignore {
        cmd.arg("--no-ignore").arg("--no-ignore-parent");
    }
    if let Some(depth) = max_depth {
        cmd.arg(format!("--max-depth={depth}"));
    }
    for glob in globs {
        cmd.arg("--glob").arg(glob);
    }
    cmd.current_dir(base);

    let output = cmd.output().map_err(|e| {
        ToolResult::err_fmt(format_args!("failed to run rg for {}: {e}", base.display()))
    })?;

    if !output.status.success() {
        if output.status.code() == Some(1) && output.stdout.is_empty() {
            return Ok(Vec::new());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        let message = stderr.trim();
        return Err(ToolResult::err_fmt(format_args!(
            "rg --files failed for {}: {}",
            base.display(),
            if message.is_empty() {
                "unknown error"
            } else {
                message
            }
        )));
    }

    let files = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|chunk| !chunk.is_empty())
        .map(|chunk| base.join(String::from_utf8_lossy(chunk).as_ref()))
        .collect();
    Ok(files)
}

/// Build the standard result envelope returned by filesystem listing tools.
pub(crate) fn filesystem_entries_result(
    items: Vec<PathEntry>,
    total_count: usize,
) -> serde_json::Value {
    let shown = items.len();
    let truncated = if total_count > shown {
        Some(TruncationMeta {
            shown,
            total: total_count,
            omitted: total_count - shown,
        })
    } else {
        None
    };
    serde_json::json!({
        "items": items,
        "truncated": truncated,
    })
}

fn count_text_lines(path: &Path) -> Option<u64> {
    let file = std::fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut count = 0_u64;
    for line in reader.lines() {
        if line.is_err() {
            return None;
        }
        count += 1;
    }
    Some(count)
}

fn format_time_rfc3339(ts: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(ts).to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Generate a compact unified diff between old and new content.
/// Truncates to `max_lines` lines if the diff is too long.
pub(crate) fn compact_diff(old: &str, new: &str, path: &str, max_lines: usize) -> String {
    let diff = similar::TextDiff::from_lines(old, new);
    let unified = diff
        .unified_diff()
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();
    if unified.is_empty() {
        return String::new();
    }
    let lines: Vec<&str> = unified.lines().collect();
    if lines.len() <= max_lines {
        unified
    } else {
        let mut truncated: String = lines[..max_lines].join("\n");
        truncated.push_str(&format!("\n... ({} more lines)", lines.len() - max_lines));
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_tool(name: &str) -> crate::ToolDefinition {
        crate::ToolDefinition {
            name: name.to_string(),
            description: format!("desc for {name}"),
            params: Vec::new(),
            returns: String::new(),
            examples: Vec::new(),
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        }
    }

    #[test]
    fn project_tool_catalog_keeps_enabled_tools_with_prompt_metadata() {
        let catalog = project_tool_catalog([dummy_tool("read_file"), dummy_tool("search_tools")]);
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog[0]["name"], serde_json::json!("read_file"));
        assert_eq!(catalog[1]["injected"], serde_json::json!(true));
    }

    #[test]
    fn rlm_does_not_surface_batch_as_native_tool() {
        assert!(native_tools(crate::ExecutionMode::Rlm).is_empty());
    }

    #[test]
    fn rlm_still_dispatches_batch_as_native_tool() {
        assert!(matches!(
            find_native_tool(crate::ExecutionMode::Rlm, "batch"),
            Some(NativeTool::Batch)
        ));
    }
}
