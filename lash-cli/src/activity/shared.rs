use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use super::ActivityArtifact;

// ─── Tool name normalization ─────────────────────────────────────────────────

/// Strip the `functions.` / `web.` namespace prefix some providers add
/// so projectors can match on bare tool names like `read_file`.
pub(super) fn activity_tool_name(name: &str) -> &str {
    name.strip_prefix("functions.")
        .or_else(|| name.strip_prefix("web."))
        .unwrap_or(name)
}

// ─── JSON arg accessors ──────────────────────────────────────────────────────

pub(super) fn tool_arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
}

pub(super) fn tool_arg_list(args: &Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| match value {
                    Value::String(text) if !text.is_empty() => Some(inline_text(text)),
                    Value::Bool(_) | Value::Number(_) => Some(value.to_string()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

// ─── Text normalization ──────────────────────────────────────────────────────

pub(super) fn inline_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn inline_snippet(text: &str, max_chars: usize) -> String {
    let compact = inline_text(text);
    let snippet: String = compact.chars().take(max_chars).collect();
    if compact.chars().count() > max_chars {
        format!("{snippet}...")
    } else {
        snippet
    }
}

// ─── Path display ────────────────────────────────────────────────────────────
//
// Absolute paths get progressively shortened: first to repo-relative, then
// home-relative (`~/foo/bar`), then a tail-only absolute (`…/foo/bar/baz`).
// Used by the exploration, edit, shell, and snippet projectors.

pub(super) fn compact_path_display(path: &str) -> String {
    let path = Path::new(path);
    if !path.is_absolute() {
        return path.to_string_lossy().into_owned();
    }

    if let Ok(cwd) = std::env::current_dir()
        && let Ok(relative) = path.strip_prefix(&cwd)
    {
        let label = relative.to_string_lossy();
        return if label.is_empty() {
            ".".to_string()
        } else {
            label.into_owned()
        };
    }

    if let Some(home) = user_home_dir()
        && let Ok(relative) = path.strip_prefix(home)
    {
        return compact_home_relative_path(relative);
    }

    compact_absolute_path(path)
}

fn user_home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn compact_home_relative_path(path: &Path) -> String {
    let components = display_path_components(path);
    if components.is_empty() {
        return "~".to_string();
    }
    if components.len() <= 3 {
        return format!("~/{}", components.join("/"));
    }
    format!("~/…/{}", tail_components(&components, 3))
}

fn compact_absolute_path(path: &Path) -> String {
    let components = display_path_components(path);
    if components.is_empty() {
        return "/".to_string();
    }
    if components.len() <= 3 {
        return format!("/{}", components.join("/"));
    }
    format!("…/{}", tail_components(&components, 3))
}

fn display_path_components(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            Component::CurDir => Some(".".to_string()),
            Component::ParentDir => Some("..".to_string()),
            Component::RootDir | Component::Prefix(_) => None,
        })
        .collect()
}

fn tail_components(components: &[String], count: usize) -> String {
    components
        .iter()
        .skip(components.len().saturating_sub(count))
        .cloned()
        .collect::<Vec<_>>()
        .join("/")
}

// ─── Result preview helpers ──────────────────────────────────────────────────

/// Extract a `TextPreview` artifact from a tool result that's either a
/// bare string or a JSON object with an `answer` field. Used by the
/// fetch, lashlang, and generic-fallback projectors.
pub(super) fn text_preview_artifact(
    title: Option<&str>,
    result: &Value,
) -> Option<ActivityArtifact> {
    let text = if let Some(text) = result.as_str() {
        text.to_string()
    } else if let Some(text) = result.get("answer").and_then(|value| value.as_str()) {
        text.to_string()
    } else {
        return None;
    };

    if text.trim().is_empty() {
        return None;
    }

    Some(ActivityArtifact::TextPreview {
        title: title.map(str::to_string),
        text,
    })
}

/// Render a limited list of `{name, description}` entries as
/// `"name: description"` detail lines. Used by the subagent and
/// search_tools projectors.
pub(super) fn named_description_detail_lines(result: &Value, limit: usize) -> Vec<String> {
    result
        .as_array()
        .map(|items| {
            items
                .iter()
                .take(limit)
                .filter_map(|item| {
                    let name = item.get("name").and_then(|value| value.as_str())?;
                    let description = item
                        .get("description")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default();
                    if description.trim().is_empty() {
                        Some(name.to_string())
                    } else {
                        Some(format!("{name}: {}", inline_snippet(description, 72)))
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

// ─── Generic semantic summary fallback ───────────────────────────────────────
//
// Used by: `apply_patch` when the result has no semantic summary,
// `show_snippet_to_user` when the result is missing path info, and the
// generic projector as the last-ditch label for unknown tools.
// Known tool names get a hand-tuned phrase; others fall back to
// `"tool_name_with_underscores"` → `"tool name with underscores"`.

pub(super) fn semantic_tool_summary(name: &str, args: &Value) -> String {
    match name {
        "read_file" => tool_arg_str(args, "path")
            .map(|path| {
                let mut label = format!("read {}", compact_path_display(path));
                if let Some(offset) = args.get("offset").and_then(|value| value.as_u64())
                    && offset > 1
                {
                    label.push_str(&format!(" @{}", offset));
                }
                label
            })
            .unwrap_or_else(|| "read file".to_string()),
        "apply_patch" => "apply patch".to_string(),
        "grep" => super::projectors::exploration::grep_label(args),
        "glob" => super::projectors::exploration::glob_label(args),
        "ls" => format!(
            "list {}",
            compact_path_display(tool_arg_str(args, "path").unwrap_or("."))
        ),
        "exec_command" => tool_arg_str(args, "cmd")
            .map(inline_text)
            .unwrap_or_else(|| "command".to_string()),
        "write_stdin" => tool_arg_str(args, "chars")
            .map(|chars| {
                if chars.is_empty() {
                    "poll command output".to_string()
                } else {
                    "write to command".to_string()
                }
            })
            .unwrap_or_else(|| "write to command".to_string()),
        "fetch_url" => tool_arg_str(args, "url")
            .map(|url| format!("fetch {}", super::projectors::web::display_url(url)))
            .unwrap_or_else(|| "fetch url".to_string()),
        "search_web" => tool_arg_str(args, "query")
            .map(|query| format!("web \"{}\"", inline_text(query)))
            .unwrap_or_else(|| "search web".to_string()),
        "spawn_agent" => tool_arg_str(args, "task")
            .map(|task| format!("spawn subagent · {}", inline_text(task)))
            .unwrap_or_else(|| "spawn subagent".to_string()),
        "send_message" => tool_arg_str(args, "target")
            .map(|target| format!("message subagent · {}", inline_text(target)))
            .unwrap_or_else(|| "message subagent".to_string()),
        "followup_task" => tool_arg_str(args, "target")
            .map(|target| format!("follow up subagent · {}", inline_text(target)))
            .unwrap_or_else(|| "follow up subagent".to_string()),
        "wait_agent" => "waited on subagents".to_string(),
        "close_agent" => tool_arg_str(args, "target")
            .map(|target| format!("close subagent · {}", inline_text(target)))
            .unwrap_or_else(|| "close subagent".to_string()),
        "list_agents" => "list subagents".to_string(),
        "show_snippet_to_user" => tool_arg_str(args, "path")
            .map(|path| {
                let start = args
                    .get("start_line")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(1);
                let end = args
                    .get("end_line")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(start);
                format!(
                    "show {}:{}-{} to user",
                    compact_path_display(path),
                    start,
                    end
                )
            })
            .unwrap_or_else(|| "show snippet to user".to_string()),
        _ => name.replace('_', " "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_path_display_collapses_external_absolute_paths() {
        assert_eq!(
            compact_path_display("/tmp/std-template/spacetimedb/src/lib.rs"),
            "…/spacetimedb/src/lib.rs"
        );
    }
}
