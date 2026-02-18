mod agent_call;
mod composite;
mod diff_file;
mod edit_file;
mod fetch_url;
mod find_replace;
mod glob;
mod grep;
pub mod hashline;
mod ls;
mod plan_mode;
mod read_file;
mod shell;
mod skills;
#[cfg(feature = "sqlite-store")]
mod tasks;
#[cfg(feature = "sqlite-store")]
mod view_message;
mod web_search;
mod write_file;

pub use agent_call::AgentCall;
pub use composite::CompositeTools;
pub use diff_file::DiffFile;
pub use edit_file::EditFile;
pub use fetch_url::FetchUrl;
pub use find_replace::FindReplace;
pub use glob::Glob;
pub use grep::Grep;
pub use ls::Ls;
pub use plan_mode::PlanMode;
pub use read_file::ReadFile;
pub use shell::Shell;
pub use skills::SkillStore;
#[cfg(feature = "sqlite-store")]
pub use tasks::TaskStore;
#[cfg(feature = "sqlite-store")]
pub use view_message::ViewMessage;
pub use web_search::WebSearch;
pub use write_file::WriteFile;

use crate::ToolResult;

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
