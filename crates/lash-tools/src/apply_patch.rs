use serde_json::json;
use std::path::{Path, PathBuf};

use lash_core::{ToolCall, ToolDefinition, ToolResult, ToolScheduling};

use lash_tool_support::{
    StaticToolExecute, StaticToolProvider, ToolDefinitionLashlangExt, compact_diff,
    display_relative, normalize_lexical, object_schema, require_str, resolve_under, run_blocking,
};

const BEGIN_PATCH_MARKER: &str = "*** Begin Patch";
const END_PATCH_MARKER: &str = "*** End Patch";
const ADD_FILE_MARKER: &str = "*** Add File: ";
const DELETE_FILE_MARKER: &str = "*** Delete File: ";
const UPDATE_FILE_MARKER: &str = "*** Update File: ";
const MOVE_TO_MARKER: &str = "*** Move to: ";
const EOF_MARKER: &str = "*** End of File";
const CHANGE_CONTEXT_MARKER: &str = "@@ ";
const EMPTY_CHANGE_CONTEXT_MARKER: &str = "@@";
const APPLY_PATCH_INSTRUCTIONS: &str = r#"Use `files.patch(...)` to edit files. The patch body is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create or replace a file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more "hunks", each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

- " " (a space) for unchanged context
- "-" for a line being removed
- "+" for a line being added

For [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:

@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ 	 def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

```
*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch
```

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
- File references can only be relative, NEVER ABSOLUTE.
- Avoid re-reading a file just to confirm a successful patch; if `files.patch` succeeds, trust it and move on to the next targeted check"#;

#[derive(Default)]
pub struct ApplyPatchTool;

/// Build the cached `apply_patch` tool provider.
pub fn apply_patch_provider() -> StaticToolProvider<ApplyPatchTool> {
    StaticToolProvider::new(vec![apply_patch_tool_definition()], ApplyPatchTool)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchAction {
    Add,
    Delete,
    Update,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchFileOp {
    pub action: PatchAction,
    pub path: PathBuf,
    pub move_path: Option<PathBuf>,
}

#[async_trait::async_trait]
impl StaticToolExecute for ApplyPatchTool {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let input = match require_str(call.args, "input") {
            Ok(value) => value.to_string(),
            Err(err) => return err,
        };
        let workdir = call
            .args
            .get("workdir")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_string);

        run_blocking(move || apply_patch(&input, workdir.as_deref())).await
    }
}

fn apply_patch_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
                "tool:apply_patch",
                "apply_patch",
                APPLY_PATCH_INSTRUCTIONS,
                object_schema(
                    serde_json::json!({
                        "input": {
                            "type": "string",
                            "description": "Patch body in the file patch format"
                        },
                        "workdir": {
                            "type": "string",
                            "description": "Optional working directory used to resolve relative patch paths"
                        }
                    }),
                    &["input"],
                ),
                apply_patch_output_schema(),
            )
            .with_examples(vec![
                "await files.patch({ input: \"*** Begin Patch\\n*** Add File: hello.txt\\n+hello\\n*** End Patch\" })?"
                    .into(),
                "await files.patch({ input: \"*** Begin Patch\\n*** Update File: src/main.rs\\n@@ fn main() {\\n-    old();\\n+    new();\\n*** End Patch\" })?"
                    .into(),
            ])
            .with_lashlang_binding(lash_tool_support::lashlang_binding(
                ["files"],
                "patch",
                &["patch", "edit_file"],
            ))
            .with_scheduling(ToolScheduling::Serial)
}

fn apply_patch_output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" },
            "added": { "type": "integer", "minimum": 0 },
            "removed": { "type": "integer", "minimum": 0 },
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "status": { "type": "string", "enum": ["added", "deleted", "modified", "moved"] },
                        "added": { "type": "integer", "minimum": 0 },
                        "removed": { "type": "integer", "minimum": 0 },
                        "diff": { "type": "string" },
                        "from_path": { "type": "string" }
                    },
                    "required": ["path", "status", "added", "removed", "diff"],
                    "additionalProperties": false
                }
            },
            "diff": {
                "type": "string",
                "description": "Combined diff preview capped to the first three changed files."
            }
        },
        "required": ["summary", "added", "removed", "files", "diff"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod description_tests {
    use super::*;

    #[test]
    fn apply_patch_description_mentions_avoiding_rereads() {
        let description = apply_patch_tool_definition().manifest().description;
        assert!(description.contains("Avoid re-reading a file"));
    }
}

pub fn apply_patch(input: &str, workdir: Option<&str>) -> ToolResult {
    let patch = match parse_patch(input) {
        Ok(patch) => patch,
        Err(err) => return ToolResult::err_fmt(err),
    };

    if patch.hunks.is_empty() {
        return ToolResult::err_fmt("No files were modified.");
    }

    let cwd = match resolve_patch_workdir(workdir) {
        Ok(path) => path,
        Err(err) => return ToolResult::err_fmt(err),
    };

    let mut applied = Vec::new();
    for hunk in &patch.hunks {
        match apply_hunk(hunk, &cwd) {
            Ok(change) => applied.push(change),
            Err(err) => return ToolResult::err_fmt(err),
        }
    }

    let files = applied
        .iter()
        .map(PreparedChange::as_json)
        .collect::<Vec<_>>();
    let changed = applied.len();
    let (total_added, total_removed) = applied.iter().fold((0usize, 0usize), |acc, change| {
        let (added, removed) = change.line_delta();
        (acc.0 + added, acc.1 + removed)
    });
    let summary = format!(
        "Applied patch to {} file{}",
        changed,
        if changed == 1 { "" } else { "s" }
    );
    let combined_diff = applied
        .iter()
        .map(PreparedChange::diff)
        .filter(|diff| !diff.trim().is_empty())
        .take(3)
        .map(str::to_string)
        .collect::<Vec<_>>()
        .join("\n");

    ToolResult::ok(json!({
        "summary": summary,
        "added": total_added,
        "removed": total_removed,
        "files": files,
        "diff": combined_diff,
    }))
}

pub fn inspect_patch_ops(input: &str, workdir: Option<&str>) -> Result<Vec<PatchFileOp>, String> {
    let patch = parse_patch(input)?;
    let cwd = resolve_patch_workdir(workdir)?;
    Ok(patch
        .hunks
        .into_iter()
        .map(|hunk| match hunk {
            Hunk::Add { path, .. } => PatchFileOp {
                action: PatchAction::Add,
                path: resolve_under(&cwd, &path),
                move_path: None,
            },
            Hunk::Delete { path } => PatchFileOp {
                action: PatchAction::Delete,
                path: resolve_under(&cwd, &path),
                move_path: None,
            },
            Hunk::Update {
                path, move_path, ..
            } => PatchFileOp {
                action: PatchAction::Update,
                path: resolve_under(&cwd, &path),
                move_path: move_path.as_ref().map(|target| resolve_under(&cwd, target)),
            },
        })
        .collect())
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
    let normalized = normalize_patch_input(input);
    let lines: Vec<&str> = normalized.lines().collect();
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

fn normalize_patch_input(input: &str) -> String {
    let trimmed = input.trim();
    if has_patch_boundaries(trimmed.lines()) {
        return trimmed.to_string();
    }

    strip_heredoc_wrapper(trimmed).unwrap_or_else(|| trimmed.to_string())
}

fn has_patch_boundaries<'a>(mut lines: impl DoubleEndedIterator<Item = &'a str>) -> bool {
    match (lines.next(), lines.next_back()) {
        (Some(first), Some(last)) => {
            first.trim() == BEGIN_PATCH_MARKER && last.trim() == END_PATCH_MARKER
        }
        (Some(only), None) => only.trim() == BEGIN_PATCH_MARKER,
        _ => false,
    }
}

fn strip_heredoc_wrapper(input: &str) -> Option<String> {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() < 4 {
        return None;
    }

    let marker = parse_heredoc_start(lines[0].trim())?;
    let last = lines.last()?.trim();
    if last != marker && !last.ends_with(marker) {
        return None;
    }

    let inner = lines[1..lines.len() - 1].join("\n");
    has_patch_boundaries(inner.lines()).then(|| inner.trim().to_string())
}

fn resolve_patch_workdir(workdir: Option<&str>) -> Result<PathBuf, String> {
    let here = std::env::current_dir().map_err(|err| format!("Failed to determine cwd: {err}"))?;
    // `resolve_under` already passes absolute paths through (normalized) and
    // joins relative ones onto `here`, so it covers both branches; an absent
    // `workdir` is just the cwd itself.
    Ok(match workdir {
        Some(path) => resolve_under(&here, Path::new(path)),
        None => normalize_lexical(&here),
    })
}

fn parse_heredoc_start(line: &str) -> Option<&str> {
    let marker = if let Some(rest) = line.strip_prefix("<<") {
        rest
    } else {
        line.strip_prefix("apply_patch <<")?
    };

    let marker = marker.trim();
    if marker.len() >= 2 {
        let bytes = marker.as_bytes();
        if (bytes[0] == b'\'' && bytes[marker.len() - 1] == b'\'')
            || (bytes[0] == b'"' && bytes[marker.len() - 1] == b'"')
        {
            return Some(&marker[1..marker.len() - 1]);
        }
    }

    Some(marker)
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
        if contents.is_empty()
            && lines
                .get(1)
                .is_some_and(|line| !line.trim_start().starts_with("***"))
        {
            return Err(format!(
                "invalid hunk at line {line_number}: add file hunk for '{path}' is empty; every new file content line must start with '+'"
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

    let (change_context, start_index) = if lines[0].trim_end() == EMPTY_CHANGE_CONTEXT_MARKER {
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
        display_path: String,
        diff: String,
    },
    Delete {
        display_path: String,
        diff: String,
    },
    Update {
        display_path: String,
        diff: String,
    },
    Move {
        from_display_path: String,
        display_path: String,
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

fn apply_hunk(hunk: &Hunk, cwd: &Path) -> Result<PreparedChange, String> {
    match hunk {
        Hunk::Add { path, contents } => {
            let resolved = resolve_under(cwd, path);
            let original_contents = std::fs::read_to_string(&resolved).unwrap_or_default();
            if let Some(parent) = resolved.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|err| {
                    format!(
                        "Failed to create directories for {}: {err}",
                        resolved.display()
                    )
                })?;
            }
            std::fs::write(&resolved, contents)
                .map_err(|err| format!("Failed to write {}: {err}", resolved.display()))?;
            let display_path = display_relative(cwd, &resolved);
            let diff = compact_diff(&original_contents, contents, &display_path, 120);
            Ok(PreparedChange::Add { display_path, diff })
        }
        Hunk::Delete { path } => {
            let resolved = resolve_under(cwd, path);
            let original = std::fs::read_to_string(&resolved).unwrap_or_default();
            std::fs::remove_file(&resolved)
                .map_err(|err| format!("Failed to delete {}: {err}", resolved.display()))?;
            let display_path = display_relative(cwd, &resolved);
            let diff = compact_diff(&original, "", &display_path, 120);
            Ok(PreparedChange::Delete { display_path, diff })
        }
        Hunk::Update {
            path,
            move_path,
            chunks,
        } => {
            let resolved = resolve_under(cwd, path);
            let applied = derive_new_contents_from_chunks(&resolved, chunks)?;
            let target = move_path
                .as_ref()
                .map(|path| -> Result<PathBuf, String> { Ok(resolve_under(cwd, path)) })
                .transpose()?
                .unwrap_or_else(|| resolved.clone());
            if let Some(parent) = target.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|err| {
                    format!(
                        "Failed to create directories for {}: {err}",
                        target.display()
                    )
                })?;
            }
            std::fs::write(&target, &applied.new_contents)
                .map_err(|err| format!("Failed to write {}: {err}", target.display()))?;
            if move_path.is_some() {
                std::fs::remove_file(&resolved).map_err(|err| {
                    format!("Failed to remove original {}: {err}", resolved.display())
                })?;
            }
            let display_path = display_relative(cwd, &target);
            let diff = compact_diff(
                &applied.original_contents,
                &applied.new_contents,
                &display_path,
                120,
            );
            if move_path.is_some() {
                Ok(PreparedChange::Move {
                    from_display_path: display_relative(cwd, &resolved),
                    display_path,
                    diff,
                })
            } else {
                Ok(PreparedChange::Update { display_path, diff })
            }
        }
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
                // The @@ label is an anchor, not a consumed context line.
                // Keep the search window at the matched line so updates whose
                // first old-line equals the label still match correctly.
                line_index = index;
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
    fn normalize_for_match(text: &str) -> String {
        text.trim()
            .chars()
            .map(|ch| match ch {
                '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
                | '\u{2212}' => '-',
                '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
                '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
                '\u{00A0}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}' | '\u{2006}'
                | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200A}' | '\u{202F}' | '\u{205F}'
                | '\u{3000}' => ' ',
                other => other,
            })
            .collect()
    }
    for i in search_start..=lines.len().saturating_sub(pattern.len()) {
        let mut ok = true;
        for (p_idx, pat) in pattern.iter().enumerate() {
            if normalize_for_match(&lines[i + p_idx]) != normalize_for_match(pat) {
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
    use tempfile::TempDir;

    fn run_patch(dir: &TempDir, input: impl AsRef<str>) -> ToolResult {
        apply_patch(input.as_ref(), Some(dir.path().to_str().unwrap()))
    }

    #[test]
    fn apply_patch_contract_documents_result_shape() {
        let definition = apply_patch_tool_definition();

        assert_eq!(
            definition.contract.output_schema["properties"]["files"]["type"],
            serde_json::json!("array")
        );
        let rendered = definition.compact_contract().render_signature();
        assert!(rendered.contains("files"), "{rendered}");
        assert!(rendered.contains("summary"), "{rendered}");
    }

    #[test]
    fn direct_apply_patch_creates_file() {
        let dir = TempDir::new().unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch",
        );
        assert!(result.is_success());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
            "hello\n"
        );
    }

    #[test]
    fn update_file_patch_modifies_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {\n    old();\n}\n").unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: main.rs\n@@ fn main() {\n-    old();\n+    new();\n*** End Patch",
        );
        assert!(result.is_success());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("main.rs")).unwrap(),
            "fn main() {\n    new();\n}\n"
        );
    }

    #[test]
    fn delete_file_patch_removes_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("old.txt"), "gone\n").unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Delete File: old.txt\n*** End Patch",
        );
        assert!(result.is_success());
        assert!(!dir.path().join("old.txt").exists());
    }

    #[test]
    fn move_patch_renames_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("old.txt"), "line\n").unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: old.txt\n*** Move to: new.txt\n@@\n line\n*** End Patch",
        );
        assert!(result.is_success());
        assert!(!dir.path().join("old.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new.txt")).unwrap(),
            "line\n"
        );
    }

    #[test]
    fn patch_result_uses_workdir_relative_display_paths() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("base.txt"), "old\n").unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: base.txt\n@@\n-old\n+new\n*** End Patch",
        );

        assert!(result.is_success());
        let result_value = result.value_for_projection();
        let diff = result_value["diff"].as_str().expect("diff");
        assert!(diff.contains("--- a/base.txt"));
        assert!(diff.contains("+++ b/base.txt"));
        assert!(!diff.contains("/tmp/"));
        assert_eq!(result_value["files"][0]["path"], "base.txt");
        assert_eq!(result_value["files"][0]["added"], 1);
        assert_eq!(result_value["files"][0]["removed"], 1);
        assert_eq!(result_value["added"], 1);
        assert_eq!(result_value["removed"], 1);
    }

    #[test]
    fn add_file_patch_requires_plus_prefixed_lines() {
        let dir = TempDir::new().unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Add File: hello.txt\nhello\n*** End Patch",
        );
        assert!(!result.is_success());
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("must start with '+'")
        );
    }

    #[test]
    fn add_file_patch_can_create_truly_empty_file() {
        let dir = TempDir::new().unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Add File: empty.txt\n*** End Patch",
        );
        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("empty.txt")).unwrap(),
            ""
        );
        assert_eq!(
            result.value_for_projection()["files"][0]["path"],
            "empty.txt"
        );
    }

    #[test]
    fn update_file_patch_allows_first_chunk_without_explicit_marker() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("module.py"), "import alpha\n").unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: module.py\n import alpha\n+import beta\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("module.py")).unwrap(),
            "import alpha\nimport beta\n"
        );
    }

    #[test]
    fn update_file_patch_accepts_whitespace_padded_headers_and_markers() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("pad.txt"), "one\n").unwrap();

        let result = run_patch(
            &dir,
            " *** Begin Patch\n  *** Update File: pad.txt\n@@\n-one\n+two\n *** End Patch ",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("pad.txt")).unwrap(),
            "two\n"
        );
    }

    #[test]
    fn update_file_patch_supports_pure_addition_chunk() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("notes.txt"), "alpha\nbeta\n").unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: notes.txt\n@@\n+gamma\n+delta\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("notes.txt")).unwrap(),
            "alpha\nbeta\ngamma\ndelta\n"
        );
    }

    #[test]
    fn update_file_patch_supports_deletion_only_chunk() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("lines.txt"), "line1\nline2\nline3\nline4\n").unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: lines.txt\n@@\n line1\n-line2\n line3\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("lines.txt")).unwrap(),
            "line1\nline3\nline4\n"
        );
    }

    #[test]
    fn update_file_patch_supports_end_of_file_marker() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("tail.txt"), "first\nsecond\n").unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: tail.txt\n@@\n first\n-second\n+second updated\n*** End of File\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("tail.txt")).unwrap(),
            "first\nsecond updated\n"
        );
    }

    #[test]
    fn update_file_patch_appends_trailing_newline() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("plain.txt"), "just one line").unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: plain.txt\n@@\n-just one line\n+first row\n+second row\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("plain.txt")).unwrap(),
            "first row\nsecond row\n"
        );
    }

    #[test]
    fn empty_patch_returns_no_files_modified() {
        let dir = TempDir::new().unwrap();
        let result = run_patch(&dir, "*** Begin Patch\n*** End Patch");

        assert!(!result.is_success());
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("No files were modified.")
        );
    }

    #[test]
    fn invalid_hunk_header_is_rejected() {
        let dir = TempDir::new().unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Rename File: nope.txt\n*** End Patch",
        );

        assert!(!result.is_success());
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("is not a valid hunk header")
        );
    }

    #[test]
    fn direct_heredoc_wrapper_is_accepted() {
        let dir = TempDir::new().unwrap();
        let result = run_patch(
            &dir,
            "<<EOF\n*** Begin Patch\n*** Add File: tiny.txt\n+ok\n*** End Patch\nEOF",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("tiny.txt")).unwrap(),
            "ok\n"
        );
    }

    #[test]
    fn apply_patch_accepts_absolute_add_paths() {
        let dir = TempDir::new().unwrap();
        let abs = dir.path().join("hello.txt");
        let input = format!(
            "*** Begin Patch\n*** Add File: {}\n+hello\n*** End Patch",
            abs.display()
        );
        let result = run_patch(&dir, input);
        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(std::fs::read_to_string(&abs).unwrap(), "hello\n");
        assert_eq!(
            result.value_for_projection()["files"][0]["path"],
            "hello.txt"
        );
    }

    #[test]
    fn apply_patch_accepts_absolute_update_paths() {
        let dir = TempDir::new().unwrap();
        let abs = dir.path().join("main.rs");
        std::fs::write(&abs, "fn main() {\n    old();\n}\n").unwrap();
        let input = format!(
            "*** Begin Patch\n*** Update File: {}\n@@ fn main() {{\n-    old();\n+    new();\n*** End Patch",
            abs.display()
        );
        let result = run_patch(&dir, input);

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(&abs).unwrap(),
            "fn main() {\n    new();\n}\n"
        );
        assert_eq!(result.value_for_projection()["files"][0]["path"], "main.rs");
    }

    #[test]
    fn apply_patch_accepts_absolute_delete_paths() {
        let dir = TempDir::new().unwrap();
        let abs = dir.path().join("old.txt");
        std::fs::write(&abs, "gone\n").unwrap();
        let input = format!(
            "*** Begin Patch\n*** Delete File: {}\n*** End Patch",
            abs.display()
        );
        let result = run_patch(&dir, input);

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert!(!abs.exists());
        assert_eq!(result.value_for_projection()["files"][0]["path"], "old.txt");
    }

    #[test]
    fn apply_patch_accepts_absolute_move_paths() {
        let dir = TempDir::new().unwrap();
        let source = dir.path().join("old.txt");
        let dest = dir.path().join("nested").join("new.txt");
        std::fs::write(&source, "line\n").unwrap();
        let input = format!(
            "*** Begin Patch\n*** Update File: {}\n*** Move to: {}\n@@\n line\n*** End Patch",
            source.display(),
            dest.display()
        );
        let result = run_patch(&dir, input);

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert!(!source.exists());
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "line\n");
        assert_eq!(
            result.value_for_projection()["files"][0]["path"],
            "nested/new.txt"
        );
    }

    #[test]
    fn apply_patch_accepts_lenient_heredoc_wrapper() {
        let dir = TempDir::new().unwrap();
        let result = run_patch(
            &dir,
            "apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\nPATCH",
        );
        assert!(result.is_success());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
            "hello\n"
        );
    }

    #[test]
    fn apply_patch_treats_unified_diff_header_as_plain_context() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {\n    old();\n}\n").unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: main.rs\n@@ -1,3 +1,3 @@\n-    old();\n+    new();\n*** End Patch",
        );
        assert!(!result.is_success());
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("Failed to find context '-1,3 +1,3 @@'")
        );
    }

    #[test]
    fn update_file_patch_allows_context_label_matching_first_old_line() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("main.rs"),
            "fn main() {\n    println!(\"old\");\n}\n",
        )
        .unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: main.rs\n@@ fn main() {\n fn main() {\n-    println!(\"old\");\n+    println!(\"new\");\n }\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("main.rs")).unwrap(),
            "fn main() {\n    println!(\"new\");\n}\n"
        );
    }

    #[test]
    fn update_file_patch_allows_whitespace_padded_bare_hunk_header() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("hello.txt"),
            "Hello from apply_patch!\nLine two.\n",
        )
        .unwrap();
        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: hello.txt\n@@ \n Hello from apply_patch!\n-Line two.\n+Line two updated by patch.\n+Line three added.\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
            "Hello from apply_patch!\nLine two updated by patch.\nLine three added.\n"
        );
    }

    #[test]
    fn update_file_patch_matches_common_unicode_punctuation() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("unicode.txt"),
            "note - uses an en dash \u{2013} and a nonbreaking hyphen in top\u{2011}level text\n",
        )
        .unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: unicode.txt\n@@\n-note - uses an en dash - and a nonbreaking hyphen in top-level text\n+normalized replacement\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("unicode.txt")).unwrap(),
            "normalized replacement\n"
        );
    }

    #[test]
    fn add_file_patch_overwrites_existing_target() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("dupe.txt"), "original\n").unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Add File: dupe.txt\n+replacement\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("dupe.txt")).unwrap(),
            "replacement\n"
        );
    }

    #[test]
    fn delete_file_patch_rejects_directory_target() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("folder")).unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Delete File: folder\n*** End Patch",
        );

        assert!(!result.is_success());
        assert!(dir.path().join("folder").is_dir());
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("Failed to delete")
        );
    }

    #[test]
    fn delete_missing_file_reports_delete_failure() {
        let dir = TempDir::new().unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Delete File: missing.txt\n*** End Patch",
        );

        assert!(!result.is_success());
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("Failed to delete")
        );
    }

    #[test]
    fn move_patch_overwrites_existing_destination() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("renamed").join("dir")).unwrap();
        std::fs::write(dir.path().join("old.txt"), "from\n").unwrap();
        std::fs::write(
            dir.path().join("renamed").join("dir").join("name.txt"),
            "stale\n",
        )
        .unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: old.txt\n*** Move to: renamed/dir/name.txt\n@@\n-from\n+new\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert!(!dir.path().join("old.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("renamed").join("dir").join("name.txt"))
                .unwrap(),
            "new\n"
        );
    }

    #[test]
    fn later_hunk_sees_earlier_hunk_changes() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("chain.txt"), "old\n").unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Update File: chain.txt\n@@\n-old\n+mid\n*** Update File: chain.txt\n@@\n-mid\n+new\n*** End Patch",
        );

        assert!(result.is_success(), "{}", result.value_for_projection());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("chain.txt")).unwrap(),
            "new\n"
        );
    }

    #[test]
    fn failed_later_hunk_keeps_earlier_successful_changes() {
        let dir = TempDir::new().unwrap();

        let result = run_patch(
            &dir,
            "*** Begin Patch\n*** Add File: created.txt\n+hello\n*** Update File: missing.txt\n@@\n-old\n+new\n*** End Patch",
        );

        assert!(!result.is_success());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("created.txt")).unwrap(),
            "hello\n"
        );
        assert!(
            result
                .value_for_projection()
                .to_string()
                .contains("Failed to read file to update")
        );
    }
}
