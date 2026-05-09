mod apply_patch;
mod fetch_url;
mod glob;
mod grep;
mod ls;
mod read_file;
mod shell;
mod web_search;

pub use apply_patch::ApplyPatchTool;
pub use apply_patch::{PatchAction, PatchFileOp, inspect_patch_ops};
pub use fetch_url::FetchUrl;
pub use glob::Glob;
pub use grep::Grep;
pub use ls::Ls;
pub use read_file::{ReadFile, ReadFilePluginFactory};
pub use shell::{StandardShell, StandardShellPluginFactory};
pub use shell::{shell_prompt_contributions, shell_prompt_contributions_for_access};
pub use web_search::WebSearch;

use lash::ToolResult;
use lash_rlm_types::unwrap_projected_arg;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Shared preamble describing default filesystem-listing behavior.
/// Used by `ls` and `glob` so both tools document hidden-file and
/// `.gitignore` handling in identical wording.
pub(crate) const FS_DEFAULTS_PREAMBLE: &str =
    "By default this includes hidden files and respects `.gitignore` only inside Git repos.";

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
        .map(unwrap_projected_arg)
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
    match args.get(key).map(unwrap_projected_arg) {
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
    match args.get(key).map(unwrap_projected_arg) {
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

pub(crate) fn object_schema(properties: serde_json::Value, required: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

pub(crate) fn discovery_metadata(namespace: &str, aliases: &[&str]) -> lash::ToolDiscoveryMetadata {
    lash::ToolDiscoveryMetadata {
        namespace: if namespace.is_empty() {
            None
        } else {
            Some(namespace.to_string())
        },
        aliases: aliases.iter().map(|value| value.to_string()).collect(),
    }
}

/// Run blocking filesystem work off the async runtime.
pub async fn run_blocking<F>(f: F) -> ToolResult
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
    let mut builder = ignore::WalkBuilder::new(base);
    builder.hidden(!include_hidden).max_depth(max_depth);

    if respect_gitignore {
        builder.git_ignore(true).git_exclude(true).git_global(true);
        builder.require_git(true);
    } else {
        builder
            .git_ignore(false)
            .git_exclude(false)
            .git_global(false)
            .ignore(false)
            .parents(false)
            .require_git(false);
    }

    if !globs.is_empty() {
        let mut override_builder = ignore::overrides::OverrideBuilder::new(base);
        for glob in globs {
            override_builder.add(glob).map_err(|err| {
                ToolResult::err_fmt(format_args!(
                    "invalid ignore glob for {}: {err}",
                    base.display()
                ))
            })?;
        }

        let overrides = override_builder.build().map_err(|err| {
            ToolResult::err_fmt(format_args!(
                "failed to build ignore globs for {}: {err}",
                base.display()
            ))
        })?;
        builder.overrides(overrides);
    }

    let files = builder
        .build()
        .filter_map(Result::ok)
        .filter(|entry| entry.path() != base)
        .map(ignore::DirEntry::into_path)
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
