use serde_json::json;
use std::path::{Path, PathBuf};

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::{compact_diff, require_str, run_blocking};

const BEGIN_PATCH_MARKER: &str = "*** Begin Patch";
const END_PATCH_MARKER: &str = "*** End Patch";
const ADD_FILE_MARKER: &str = "*** Add File: ";
const DELETE_FILE_MARKER: &str = "*** Delete File: ";
const UPDATE_FILE_MARKER: &str = "*** Update File: ";
const MOVE_TO_MARKER: &str = "*** Move to: ";
const EOF_MARKER: &str = "*** End of File";
const CHANGE_CONTEXT_MARKER: &str = "@@ ";
const EMPTY_CHANGE_CONTEXT_MARKER: &str = "@@";

#[derive(Default)]
pub struct ApplyPatchTool;

#[async_trait::async_trait]
impl ToolProvider for ApplyPatchTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "apply_patch".into(),
            description: vec![crate::ToolText::new(
                concat!(
                    "Apply one or more file changes from a structured patch.\n\n",
                    "Use this as the primary file-mutation tool for creating, updating, deleting, or moving files.\n\n",
                    "Patch format:\n",
                    "- Begin with `*** Begin Patch`\n",
                    "- End with `*** End Patch`\n",
                    "- Use `*** Add File: path`, `*** Delete File: path`, or `*** Update File: path`\n",
                    "- Optional move: `*** Move to: new/path`\n",
                    "- Inside updates, each line must start with ` ` (context), `-` (remove), or `+` (add)\n",
                    "- Optional context headers start with `@@`\n\n",
                    "Pass paths relative to `workdir` when provided, otherwise relative to the current working directory."
                ),
                [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
            )],
            params: vec![
                ToolParam {
                    name: "input".into(),
                    r#type: "str".into(),
                    description: "Patch body in apply_patch format".into(),
                    required: true,
                },
                ToolParam {
                    name: "workdir".into(),
                    r#type: "str".into(),
                    description: "Optional working directory used to resolve relative patch paths"
                        .into(),
                    required: false,
                },
            ],
            returns: "dict".into(),
            examples: vec![crate::ToolText::new(
                "apply_patch(input=\"*** Begin Patch\\n*** Update File: src/main.rs\\n@@\\n-old\\n+new\\n*** End Patch\")",
                [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
            )],
            hidden: false,
            inject_into_prompt: true,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let input = match require_str(args, "input") {
            Ok(value) => value.to_string(),
            Err(err) => return err,
        };
        let workdir = args
            .get("workdir")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_string);

        run_blocking(move || execute_apply_patch_sync(&input, workdir.as_deref())).await
    }
}

fn execute_apply_patch_sync(input: &str, workdir: Option<&str>) -> ToolResult {
    let patch = match parse_patch(input) {
        Ok(patch) => patch,
        Err(err) => return ToolResult::err_fmt(err),
    };

    if patch.hunks.is_empty() {
        return ToolResult::err_fmt("No files were modified.");
    }

    let cwd = match workdir {
        Some(path) => PathBuf::from(path),
        None => match std::env::current_dir() {
            Ok(path) => path,
            Err(err) => return ToolResult::err_fmt(format!("Failed to determine cwd: {err}")),
        },
    };

    let mut prepared = Vec::new();
    for hunk in &patch.hunks {
        match prepare_change(hunk, &cwd) {
            Ok(change) => prepared.push(change),
            Err(err) => return ToolResult::err_fmt(err),
        }
    }

    for change in &prepared {
        if let Err(err) = apply_change(change) {
            return ToolResult::err_fmt(err);
        }
    }

    let files = prepared
        .iter()
        .map(PreparedChange::as_json)
        .collect::<Vec<_>>();
    let changed = prepared.len();
    let (total_added, total_removed) = prepared.iter().fold((0usize, 0usize), |acc, change| {
        let (added, removed) = change.line_delta();
        (acc.0 + added, acc.1 + removed)
    });
    let summary = format!(
        "Applied patch to {} file{}",
        changed,
        if changed == 1 { "" } else { "s" }
    );
    let combined_diff = prepared
        .iter()
        .map(PreparedChange::diff)
        .filter(|diff| !diff.trim().is_empty())
        .take(3)
        .map(str::to_string)
        .collect::<Vec<_>>()
        .join("\n");

    ToolResult::ok(json!({
        "__type__": "patch_result",
        "summary": summary,
        "added": total_added,
        "removed": total_removed,
        "files": files,
        "diff": combined_diff,
    }))
}

#[derive(Debug, PartialEq, Clone)]
struct ParsedPatch {
    hunks: Vec<Hunk>,
}

#[derive(Debug, PartialEq, Clone)]
enum Hunk {
    Add {
        path: PathBuf,
        contents: String,
    },
    Delete {
        path: PathBuf,
    },
    Update {
        path: PathBuf,
        move_path: Option<PathBuf>,
        chunks: Vec<UpdateFileChunk>,
    },
}

#[derive(Debug, PartialEq, Clone)]
struct UpdateFileChunk {
    change_context: Option<String>,
    old_lines: Vec<String>,
    new_lines: Vec<String>,
    is_end_of_file: bool,
}

fn parse_patch(input: &str) -> Result<ParsedPatch, String> {
    let lines: Vec<&str> = input.trim().lines().collect();
    let (first, last) = match lines.as_slice() {
        [] => (None, None),
        [first] => (Some(first.trim()), Some(first.trim())),
        [first, .., last] => (Some(first.trim()), Some(last.trim())),
    };

    match (first, last) {
        (Some(BEGIN_PATCH_MARKER), Some(END_PATCH_MARKER)) => {}
        (Some(first), _) if first != BEGIN_PATCH_MARKER => {
            return Err("The first line of the patch must be '*** Begin Patch'".to_string());
        }
        _ => return Err("The last line of the patch must be '*** End Patch'".to_string()),
    }

    if lines.len() <= 2 {
        return Ok(ParsedPatch { hunks: Vec::new() });
    }

    let mut hunks = Vec::new();
    let mut remaining = &lines[1..lines.len() - 1];
    let mut line_number = 2usize;
    while !remaining.is_empty() {
        let (hunk, consumed) = parse_one_hunk(remaining, line_number)?;
        hunks.push(hunk);
        remaining = &remaining[consumed..];
        line_number += consumed;
    }

    Ok(ParsedPatch { hunks })
}

fn parse_one_hunk(lines: &[&str], line_number: usize) -> Result<(Hunk, usize), String> {
    if lines.is_empty() {
        return Err(format!("invalid hunk at line {line_number}: missing hunk"));
    }

    let first_line = lines[0].trim();
    if let Some(path) = first_line.strip_prefix(ADD_FILE_MARKER) {
        let mut contents = String::new();
        let mut consumed = 1usize;
        for line in &lines[1..] {
            if let Some(text) = line.strip_prefix('+') {
                contents.push_str(text);
                contents.push('\n');
                consumed += 1;
            } else {
                break;
            }
        }
        if contents.is_empty() {
            return Err(format!(
                "invalid hunk at line {line_number}: add file hunk for '{path}' is empty"
            ));
        }
        return Ok((
            Hunk::Add {
                path: PathBuf::from(path),
                contents,
            },
            consumed,
        ));
    }

    if let Some(path) = first_line.strip_prefix(DELETE_FILE_MARKER) {
        return Ok((
            Hunk::Delete {
                path: PathBuf::from(path),
            },
            1,
        ));
    }

    if let Some(path) = first_line.strip_prefix(UPDATE_FILE_MARKER) {
        let mut consumed = 1usize;
        let mut remaining = &lines[1..];
        let move_path = remaining
            .first()
            .and_then(|line| line.strip_prefix(MOVE_TO_MARKER))
            .map(PathBuf::from);

        if move_path.is_some() {
            consumed += 1;
            remaining = &remaining[1..];
        }

        let mut chunks = Vec::new();
        while !remaining.is_empty() {
            if remaining[0].trim().is_empty() {
                consumed += 1;
                remaining = &remaining[1..];
                continue;
            }
            if remaining[0].starts_with("***") {
                break;
            }
            let (chunk, chunk_lines) =
                parse_update_file_chunk(remaining, line_number + consumed, chunks.is_empty())?;
            chunks.push(chunk);
            consumed += chunk_lines;
            remaining = &remaining[chunk_lines..];
        }

        if chunks.is_empty() {
            return Err(format!(
                "invalid hunk at line {line_number}: update file hunk for path '{path}' is empty"
            ));
        }

        return Ok((
            Hunk::Update {
                path: PathBuf::from(path),
                move_path,
                chunks,
            },
            consumed,
        ));
    }

    Err(format!(
        "invalid hunk at line {line_number}: '{first_line}' is not a valid hunk header"
    ))
}

fn parse_update_file_chunk(
    lines: &[&str],
    line_number: usize,
    allow_missing_context: bool,
) -> Result<(UpdateFileChunk, usize), String> {
    if lines.is_empty() {
        return Err(format!(
            "invalid hunk at line {line_number}: update hunk does not contain any lines"
        ));
    }

    let (change_context, start_index) = if lines[0] == EMPTY_CHANGE_CONTEXT_MARKER {
        (None, 1)
    } else if let Some(context) = lines[0].strip_prefix(CHANGE_CONTEXT_MARKER) {
        (Some(context.to_string()), 1)
    } else if allow_missing_context {
        (None, 0)
    } else {
        return Err(format!(
            "invalid hunk at line {line_number}: expected update hunk to start with a @@ context marker"
        ));
    };

    if start_index >= lines.len() {
        return Err(format!(
            "invalid hunk at line {}: update hunk does not contain any lines",
            line_number + 1
        ));
    }

    let mut chunk = UpdateFileChunk {
        change_context,
        old_lines: Vec::new(),
        new_lines: Vec::new(),
        is_end_of_file: false,
    };
    let mut parsed_lines = 0usize;
    for line in &lines[start_index..] {
        match *line {
            EOF_MARKER => {
                if parsed_lines == 0 {
                    return Err(format!(
                        "invalid hunk at line {}: update hunk does not contain any lines",
                        line_number + 1
                    ));
                }
                chunk.is_end_of_file = true;
                parsed_lines += 1;
                break;
            }
            line_contents => match line_contents.chars().next() {
                None => {
                    chunk.old_lines.push(String::new());
                    chunk.new_lines.push(String::new());
                    parsed_lines += 1;
                }
                Some(' ') => {
                    chunk.old_lines.push(line_contents[1..].to_string());
                    chunk.new_lines.push(line_contents[1..].to_string());
                    parsed_lines += 1;
                }
                Some('+') => {
                    chunk.new_lines.push(line_contents[1..].to_string());
                    parsed_lines += 1;
                }
                Some('-') => {
                    chunk.old_lines.push(line_contents[1..].to_string());
                    parsed_lines += 1;
                }
                _ => {
                    if parsed_lines == 0 {
                        return Err(format!(
                            "invalid hunk at line {}: unexpected line in update hunk",
                            line_number + 1
                        ));
                    }
                    break;
                }
            },
        }
    }

    Ok((chunk, parsed_lines + start_index))
}

enum PreparedChange {
    Add {
        path: PathBuf,
        display_path: String,
        contents: String,
        diff: String,
    },
    Delete {
        path: PathBuf,
        display_path: String,
        diff: String,
    },
    Update {
        path: PathBuf,
        display_path: String,
        contents: String,
        diff: String,
    },
    Move {
        from: PathBuf,
        from_display_path: String,
        to: PathBuf,
        display_path: String,
        contents: String,
        diff: String,
    },
}

impl PreparedChange {
    fn diff(&self) -> &str {
        match self {
            Self::Add { diff, .. }
            | Self::Delete { diff, .. }
            | Self::Update { diff, .. }
            | Self::Move { diff, .. } => diff,
        }
    }

    fn line_delta(&self) -> (usize, usize) {
        count_diff_delta(self.diff())
    }

    fn as_json(&self) -> serde_json::Value {
        let (added, removed) = self.line_delta();
        match self {
            Self::Add {
                display_path, diff, ..
            } => json!({
                "path": display_path,
                "status": "added",
                "added": added,
                "removed": removed,
                "diff": diff,
            }),
            Self::Delete {
                display_path, diff, ..
            } => json!({
                "path": display_path,
                "status": "deleted",
                "added": added,
                "removed": removed,
                "diff": diff,
            }),
            Self::Update {
                display_path, diff, ..
            } => json!({
                "path": display_path,
                "status": "modified",
                "added": added,
                "removed": removed,
                "diff": diff,
            }),
            Self::Move {
                from_display_path,
                display_path,
                diff,
                ..
            } => json!({
                "path": display_path,
                "from_path": from_display_path,
                "status": "moved",
                "added": added,
                "removed": removed,
                "diff": diff,
            }),
        }
    }
}

fn count_diff_delta(diff: &str) -> (usize, usize) {
    let mut added = 0usize;
    let mut removed = 0usize;
    for line in diff.lines() {
        if line.starts_with("+++ ") || line.starts_with("--- ") || line.starts_with("@@") {
            continue;
        }
        if line.starts_with('+') {
            added += 1;
        } else if line.starts_with('-') {
            removed += 1;
        }
    }
    (added, removed)
}

fn normalize_display_path(cwd: &Path, path: &Path) -> String {
    let display = path.strip_prefix(cwd).unwrap_or(path).display().to_string();
    let display = if display.is_empty() {
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(".")
            .to_string()
    } else {
        display
    };
    display.replace('\\', "/")
}

fn prepare_change(hunk: &Hunk, cwd: &Path) -> Result<PreparedChange, String> {
    match hunk {
        Hunk::Add { path, contents } => {
            let resolved = resolve_path(cwd, path);
            if resolved.exists() {
                return Err(format!("File already exists: {}", resolved.display()));
            }
            let display_path = normalize_display_path(cwd, &resolved);
            let diff = compact_diff("", contents, &display_path, 120);
            Ok(PreparedChange::Add {
                path: resolved,
                display_path,
                contents: contents.clone(),
                diff,
            })
        }
        Hunk::Delete { path } => {
            let resolved = resolve_path(cwd, path);
            let original = std::fs::read_to_string(&resolved)
                .map_err(|err| format!("Failed to read {}: {err}", resolved.display()))?;
            let display_path = normalize_display_path(cwd, &resolved);
            let diff = compact_diff(&original, "", &display_path, 120);
            Ok(PreparedChange::Delete {
                path: resolved,
                display_path,
                diff,
            })
        }
        Hunk::Update {
            path,
            move_path,
            chunks,
        } => {
            let resolved = resolve_path(cwd, path);
            let applied = derive_new_contents_from_chunks(&resolved, chunks)?;
            let target = move_path
                .as_ref()
                .map(|path| resolve_path(cwd, path))
                .unwrap_or_else(|| resolved.clone());
            let display_path = normalize_display_path(cwd, &target);
            let diff = compact_diff(
                &applied.original_contents,
                &applied.new_contents,
                &display_path,
                120,
            );
            if let Some(dest) = move_path.as_ref() {
                Ok(PreparedChange::Move {
                    from_display_path: normalize_display_path(cwd, &resolved),
                    from: resolved,
                    to: resolve_path(cwd, dest),
                    display_path,
                    contents: applied.new_contents,
                    diff,
                })
            } else {
                Ok(PreparedChange::Update {
                    path: resolved,
                    display_path,
                    contents: applied.new_contents,
                    diff,
                })
            }
        }
    }
}

fn apply_change(change: &PreparedChange) -> Result<(), String> {
    match change {
        PreparedChange::Add { path, contents, .. }
        | PreparedChange::Update { path, contents, .. } => {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|err| {
                    format!("Failed to create directories for {}: {err}", path.display())
                })?;
            }
            std::fs::write(path, contents)
                .map_err(|err| format!("Failed to write {}: {err}", path.display()))
        }
        PreparedChange::Delete { path, .. } => std::fs::remove_file(path)
            .map_err(|err| format!("Failed to delete {}: {err}", path.display())),
        PreparedChange::Move {
            from, to, contents, ..
        } => {
            if let Some(parent) = to.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|err| {
                    format!("Failed to create directories for {}: {err}", to.display())
                })?;
            }
            std::fs::write(to, contents)
                .map_err(|err| format!("Failed to write {}: {err}", to.display()))?;
            std::fs::remove_file(from)
                .map_err(|err| format!("Failed to remove original {}: {err}", from.display()))
        }
    }
}

fn resolve_path(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

struct AppliedPatch {
    original_contents: String,
    new_contents: String,
}

fn derive_new_contents_from_chunks(
    path: &Path,
    chunks: &[UpdateFileChunk],
) -> Result<AppliedPatch, String> {
    let original_contents = std::fs::read_to_string(path)
        .map_err(|err| format!("Failed to read file to update {}: {err}", path.display()))?;

    let mut original_lines: Vec<String> = original_contents.split('\n').map(String::from).collect();
    if original_lines.last().is_some_and(String::is_empty) {
        original_lines.pop();
    }

    let replacements = compute_replacements(&original_lines, path, chunks)?;
    let mut new_lines = apply_replacements(original_lines, &replacements);
    if !new_lines.last().is_some_and(String::is_empty) {
        new_lines.push(String::new());
    }
    let new_contents = new_lines.join("\n");

    Ok(AppliedPatch {
        original_contents,
        new_contents,
    })
}

fn compute_replacements(
    original_lines: &[String],
    path: &Path,
    chunks: &[UpdateFileChunk],
) -> Result<Vec<(usize, usize, Vec<String>)>, String> {
    let mut replacements = Vec::new();
    let mut line_index = 0usize;

    for chunk in chunks {
        if let Some(ctx_line) = &chunk.change_context {
            if let Some(index) = seek_sequence(
                original_lines,
                std::slice::from_ref(ctx_line),
                line_index,
                false,
            ) {
                line_index = index + 1;
            } else {
                return Err(format!(
                    "Failed to find context '{}' in {}",
                    ctx_line,
                    path.display()
                ));
            }
        }

        if chunk.old_lines.is_empty() {
            let insertion_idx = if original_lines.last().is_some_and(String::is_empty) {
                original_lines.len().saturating_sub(1)
            } else {
                original_lines.len()
            };
            replacements.push((insertion_idx, 0, chunk.new_lines.clone()));
            continue;
        }

        let mut pattern: &[String] = &chunk.old_lines;
        let mut new_slice: &[String] = &chunk.new_lines;
        let mut found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file);

        if found.is_none() && pattern.last().is_some_and(String::is_empty) {
            pattern = &pattern[..pattern.len() - 1];
            if new_slice.last().is_some_and(String::is_empty) {
                new_slice = &new_slice[..new_slice.len() - 1];
            }
            found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file);
        }

        if let Some(start_idx) = found {
            replacements.push((start_idx, pattern.len(), new_slice.to_vec()));
            line_index = start_idx + pattern.len();
        } else {
            return Err(format!(
                "Failed to find expected lines in {}:\n{}",
                path.display(),
                chunk.old_lines.join("\n")
            ));
        }
    }

    replacements.sort_by_key(|(start_idx, _, _)| *start_idx);
    Ok(replacements)
}

fn apply_replacements(
    mut lines: Vec<String>,
    replacements: &[(usize, usize, Vec<String>)],
) -> Vec<String> {
    for (start_idx, old_len, new_segment) in replacements.iter().rev() {
        for _ in 0..*old_len {
            if *start_idx < lines.len() {
                lines.remove(*start_idx);
            }
        }
        for (offset, line) in new_segment.iter().enumerate() {
            lines.insert(*start_idx + offset, line.clone());
        }
    }
    lines
}

fn seek_sequence(lines: &[String], pattern: &[String], start: usize, eof: bool) -> Option<usize> {
    if pattern.is_empty() {
        return Some(start);
    }
    if pattern.len() > lines.len() {
        return None;
    }

    let search_start = if eof && lines.len() >= pattern.len() {
        lines.len() - pattern.len()
    } else {
        start
    };

    for i in search_start..=lines.len().saturating_sub(pattern.len()) {
        if lines[i..i + pattern.len()] == *pattern {
            return Some(i);
        }
    }
    for i in search_start..=lines.len().saturating_sub(pattern.len()) {
        let mut ok = true;
        for (p_idx, pat) in pattern.iter().enumerate() {
            if lines[i + p_idx].trim_end() != pat.trim_end() {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(i);
        }
    }
    for i in search_start..=lines.len().saturating_sub(pattern.len()) {
        let mut ok = true;
        for (p_idx, pat) in pattern.iter().enumerate() {
            if lines[i + p_idx].trim() != pat.trim() {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn add_file_patch_creates_file() {
        let dir = TempDir::new().unwrap();
        let tool = ApplyPatchTool;
        let result = tool
            .execute(
                "apply_patch",
                &json!({
                    "workdir": dir.path().to_str().unwrap(),
                    "input": "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch"
                }),
            )
            .await;
        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
            "hello\n"
        );
    }

    #[tokio::test]
    async fn update_file_patch_modifies_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {\n    old();\n}\n").unwrap();
        let tool = ApplyPatchTool;
        let result = tool
            .execute(
                "apply_patch",
                &json!({
                    "workdir": dir.path().to_str().unwrap(),
                    "input": "*** Begin Patch\n*** Update File: main.rs\n@@ fn main() {\n-    old();\n+    new();\n*** End Patch"
                }),
            )
            .await;
        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("main.rs")).unwrap(),
            "fn main() {\n    new();\n}\n"
        );
    }

    #[tokio::test]
    async fn delete_file_patch_removes_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("old.txt"), "gone\n").unwrap();
        let tool = ApplyPatchTool;
        let result = tool
            .execute(
                "apply_patch",
                &json!({
                    "workdir": dir.path().to_str().unwrap(),
                    "input": "*** Begin Patch\n*** Delete File: old.txt\n*** End Patch"
                }),
            )
            .await;
        assert!(result.success);
        assert!(!dir.path().join("old.txt").exists());
    }

    #[tokio::test]
    async fn move_patch_renames_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("old.txt"), "line\n").unwrap();
        let tool = ApplyPatchTool;
        let result = tool
            .execute(
                "apply_patch",
                &json!({
                    "workdir": dir.path().to_str().unwrap(),
                    "input": "*** Begin Patch\n*** Update File: old.txt\n*** Move to: new.txt\n@@\n line\n*** End Patch"
                }),
            )
            .await;
        assert!(result.success);
        assert!(!dir.path().join("old.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new.txt")).unwrap(),
            "line\n"
        );
    }

    #[tokio::test]
    async fn patch_result_uses_workdir_relative_display_paths() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("base.txt"), "old\n").unwrap();
        let tool = ApplyPatchTool;
        let result = tool
            .execute(
                "apply_patch",
                &json!({
                    "workdir": dir.path().to_str().unwrap(),
                    "input": "*** Begin Patch\n*** Update File: base.txt\n@@\n-old\n+new\n*** End Patch"
                }),
            )
            .await;

        assert!(result.success);
        let diff = result.result["diff"].as_str().expect("diff");
        assert!(diff.contains("--- a/base.txt"));
        assert!(diff.contains("+++ b/base.txt"));
        assert!(!diff.contains("/tmp/"));
        assert_eq!(result.result["files"][0]["path"], "base.txt");
        assert_eq!(result.result["files"][0]["added"], 1);
        assert_eq!(result.result["files"][0]["removed"], 1);
        assert_eq!(result.result["added"], 1);
        assert_eq!(result.result["removed"], 1);
    }
}
