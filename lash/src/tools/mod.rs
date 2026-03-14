mod agent_call;
mod apply_patch;
mod ask;
mod fetch_url;
mod filtered;
mod glob;
mod grep;
pub mod hashline;
mod ls;
mod read_file;
mod shell;
mod skills;
#[cfg(feature = "sqlite-store")]
mod state;
mod switchable;
mod toolset;
mod update_plan;
mod web_search;

pub use agent_call::{AgentCall, AgentCallConfig};
pub use apply_patch::ApplyPatchTool;
pub use ask::AskTool;
pub use fetch_url::FetchUrl;
pub use filtered::FilteredTools;
pub use glob::Glob;
pub use grep::Grep;
pub use ls::Ls;
pub use read_file::ReadFile;
pub use shell::{ReplShell, StandardShell};
pub use skills::SkillStore;
#[cfg(feature = "sqlite-store")]
pub use state::StateStore;
pub use switchable::SwitchableTools;
pub use toolset::{ToolSet, ToolSetDeps};
pub use update_plan::UpdatePlanTool;
pub use web_search::WebSearch;

use crate::ToolResult;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const INTERNAL_TOOL_CATALOG_ARG: &str = "__tool_catalog";

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
    mode: crate::ExecutionMode,
) -> Vec<serde_json::Value> {
    definitions
        .into_iter()
        .filter(|d| !d.hidden && !d.description_for(mode).is_empty())
        .map(|d| {
            let p = d.project(mode);
            serde_json::json!({
                "name": p.name,
                "description": p.description,
                "params": p.params,
                "returns": p.returns,
                "examples": p.examples,
                "inject_into_prompt": p.inject_into_prompt,
                "hidden": d.hidden,
            })
        })
        .collect()
}

pub(crate) fn preflight_tool_args(
    tool_name: &str,
    mut args: serde_json::Value,
    definitions: impl IntoIterator<Item = crate::ToolDefinition>,
    mode: crate::ExecutionMode,
) -> serde_json::Value {
    if tool_name == "search_tools" && args.get(INTERNAL_TOOL_CATALOG_ARG).is_none() {
        if !args.is_object() {
            args = serde_json::Value::Object(serde_json::Map::new());
        }
        let obj = args
            .as_object_mut()
            .expect("search_tools args should be normalized to an object");
        obj.insert(
            INTERNAL_TOOL_CATALOG_ARG.to_string(),
            serde_json::Value::Array(project_tool_catalog(definitions, mode)),
        );
    }
    args
}

/// Read a file to string, or return ToolResult::err.
pub(crate) fn read_to_string(path: &std::path::Path) -> Result<String, ToolResult> {
    std::fs::read_to_string(path)
        .map_err(|e| ToolResult::err_fmt(format_args!("Failed to read {}: {e}", path.display())))
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
            description: vec![crate::ToolText::new(
                format!("desc for {name}"),
                [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
            )],
            params: Vec::new(),
            returns: String::new(),
            examples: Vec::new(),
            hidden: false,
            inject_into_prompt: true,
        }
    }

    #[test]
    fn preflight_tool_args_injects_catalog_for_search_tools() {
        let args = preflight_tool_args(
            "search_tools",
            serde_json::json!({}),
            [dummy_tool("read_file"), dummy_tool("search_tools")],
            crate::ExecutionMode::Repl,
        );

        let catalog = args
            .get(INTERNAL_TOOL_CATALOG_ARG)
            .and_then(|value| value.as_array())
            .cloned()
            .expect("catalog should be injected");
        assert_eq!(catalog.len(), 2);
    }

    #[test]
    fn preflight_tool_args_preserves_existing_catalog() {
        let args = preflight_tool_args(
            "search_tools",
            serde_json::json!({
                "__tool_catalog": [{"name":"already_here"}]
            }),
            [dummy_tool("read_file")],
            crate::ExecutionMode::Repl,
        );

        assert_eq!(
            args.get(INTERNAL_TOOL_CATALOG_ARG),
            Some(&serde_json::json!([{ "name": "already_here" }]))
        );
    }

    #[test]
    fn preflight_tool_args_leaves_other_tools_unchanged() {
        let args = serde_json::json!({"path":"Cargo.toml"});
        assert_eq!(
            preflight_tool_args(
                "read_file",
                args.clone(),
                [dummy_tool("read_file")],
                crate::ExecutionMode::Repl,
            ),
            args
        );
    }
}
