mod composite;
mod delegate_task;
mod diff_file;
mod edit_file;
mod fetch_url;
mod find_replace;
mod glob;
mod grep;
pub mod hashline;
mod ls;
mod read_file;
mod shell;
mod skills;
mod tasks;
mod view_message;
mod web_search;
mod write_file;

pub use composite::CompositeTools;
pub use delegate_task::{DelegateDeep, DelegateSearch, DelegateTask};
pub use diff_file::DiffFile;
pub use edit_file::EditFile;
pub use fetch_url::FetchUrl;
pub use find_replace::FindReplace;
pub use glob::Glob;
pub use grep::Grep;
pub use ls::Ls;
pub use read_file::ReadFile;
pub use shell::Shell;
pub use skills::SkillStore;
pub use tasks::TaskStore;
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
