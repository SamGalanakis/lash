//! Edit projector: `edit` / `write`.
//!
//! Produces an `Edit` block with a `PatchPreview` artifact listing the
//! changed files. Consecutive edits merge via `merge_edit_activity`
//! through `ActivityState` so a run of edits collapses into a single
//! "Edited N files (+M -K)" block.

use std::collections::HashSet;

use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, PatchFilePreview, ProjectCtx,
    ToolProjector,
    shared::{compact_path_display, semantic_tool_summary},
};

pub(crate) struct EditProjector;

impl ToolProjector for EditProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["edit", "write"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        let artifact = patch_preview_artifact(ctx.name, &ctx.args, &ctx.result);
        let summary = patch_summary(&artifact)
            .or_else(|| {
                ctx.result
                    .get("summary")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| semantic_tool_summary(ctx.name, &ctx.args));
        let args = std::mem::replace(&mut ctx.args, Value::Null);
        let result = std::mem::replace(&mut ctx.result, Value::Null);
        vec![
            ActivityBlock::new(
                ActivityKind::Edit,
                ctx.name,
                args,
                summary,
                status,
                result,
                ctx.duration_ms,
            )
            .with_artifact(artifact),
        ]
    }
}

// ─── Patch preview helpers ───────────────────────────────────────────────────

fn patch_preview_artifact(name: &str, args: &Value, result: &Value) -> Option<ActivityArtifact> {
    let files = if name == "edit" {
        edit_result_file_preview(args, result).into_iter().collect()
    } else {
        Vec::new()
    };

    if files.is_empty() {
        return result
            .get("details")
            .and_then(|details| details.get("diff"))
            .or_else(|| result.get("diff"))
            .and_then(|value| value.as_str())
            .filter(|diff| !diff.trim().is_empty())
            .map(|diff| ActivityArtifact::DiffPreview {
                title: "Diff".to_string(),
                diff: diff.to_string(),
            });
    }

    let total_added = files.iter().map(|file| file.added).sum();
    let total_removed = files.iter().map(|file| file.removed).sum();

    Some(ActivityArtifact::PatchPreview {
        files,
        total_added,
        total_removed,
    })
}

fn edit_result_file_preview(args: &Value, result: &Value) -> Option<PatchFilePreview> {
    let path = result
        .get("path")
        .and_then(|value| value.as_str())
        .or_else(|| args.get("path").and_then(|value| value.as_str()))?;
    let diff = result
        .get("details")
        .and_then(|details| details.get("patch"))
        .or_else(|| {
            result
                .get("details")
                .and_then(|details| details.get("diff"))
        })
        .and_then(|value| value.as_str())
        .filter(|diff| !diff.trim().is_empty())?;
    let (added, removed) = count_diff_delta(diff);
    Some(PatchFilePreview {
        path: path.to_string(),
        from_path: None,
        status: "modified".to_string(),
        added,
        removed,
        diff: diff.to_string(),
    })
}

fn patch_summary(artifact: &Option<ActivityArtifact>) -> Option<String> {
    let ActivityArtifact::PatchPreview {
        files,
        total_added,
        total_removed,
    } = artifact.as_ref()?
    else {
        return None;
    };

    Some(patch_summary_from_preview(
        files,
        *total_added,
        *total_removed,
    ))
}

fn patch_summary_from_preview(
    files: &[PatchFilePreview],
    total_added: usize,
    total_removed: usize,
) -> String {
    if files.len() == 1 {
        let file = &files[0];
        return format!(
            "{} {} {}",
            patch_status_title(&file.status),
            patch_file_subject(file),
            patch_count_suffix(file.added, file.removed)
        );
    }

    let file_count = patch_unique_file_count(files);
    let noun = if file_count == 1 { "file" } else { "files" };
    format!(
        "{} {} {} {}",
        patch_group_title(files),
        file_count,
        noun,
        patch_count_suffix(total_added, total_removed)
    )
}

fn patch_group_title(files: &[PatchFilePreview]) -> &'static str {
    let Some(first) = files.first() else {
        return "Edited";
    };
    if files.iter().all(|file| file.status == first.status) {
        patch_status_title(&first.status)
    } else {
        "Edited"
    }
}

/// Status label for a single file in a patch preview. Consumed by
/// `render/activity.rs` for patch headers — re-exported through
/// `activity::patch_status_title`.
pub(crate) fn patch_status_title(status: &str) -> &'static str {
    match status {
        "added" => "Added",
        "deleted" => "Deleted",
        "moved" => "Moved",
        _ => "Edited",
    }
}

fn patch_unique_file_count(files: &[PatchFilePreview]) -> usize {
    let mut unique = HashSet::new();
    for file in files {
        unique.insert(patch_file_subject(file));
    }
    unique.len()
}

/// Human-facing subject for a patched file (or `from → to` for moves).
/// Consumed by `render/activity.rs` for patch preview rows — re-exported
/// through `activity::patch_file_subject`.
pub(crate) fn patch_file_subject(file: &PatchFilePreview) -> String {
    let path = compact_path_display(&file.path);
    match &file.from_path {
        Some(from_path) => format!("{} → {path}", compact_path_display(from_path)),
        None => path,
    }
}

fn patch_count_suffix(added: usize, removed: usize) -> String {
    format!("(+{} -{})", added, removed)
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

// ─── Merge for adjacent patch activities ─────────────────────────────────────

/// Merge the files from `incoming` into `target`, regenerating the
/// patch summary. Called from `projection.rs` when two adjacent Edit
/// blocks should cluster into one. Returns `true` if the merge
/// succeeded.
pub fn merge_edit_activity(target: &mut ActivityBlock, incoming: ActivityBlock) -> bool {
    let Some(ActivityArtifact::PatchPreview {
        files,
        total_added,
        total_removed,
    }) = target.result.artifact.as_mut()
    else {
        return false;
    };
    let Some(ActivityArtifact::PatchPreview {
        files: incoming_files,
        total_added: incoming_added,
        total_removed: incoming_removed,
    }) = incoming.result.artifact.clone()
    else {
        return false;
    };

    files.extend(incoming_files);
    *total_added += incoming_added;
    *total_removed += incoming_removed;
    target.duration_ms += incoming.duration_ms;
    target.call.summary = patch_summary_from_preview(files, *total_added, *total_removed);
    true
}

#[cfg(test)]
mod tests {
    use crate::activity::{ActivityArtifact, ActivityState};
    use serde_json::json;

    #[test]
    fn edit_summary_uses_result_details_patch() {
        let mut state = ActivityState::new();
        let blocks = state.project_tool_call(
            "edit",
            json!({"path": "crates/lash-cli/src/render/mod.rs"}),
            json!({
                "summary": "Successfully replaced 1 block(s) in crates/lash-cli/src/render/mod.rs.",
                "path": "crates/lash-cli/src/render/mod.rs",
                "replacements": 1,
                "details": {
                    "patch": "--- a/crates/lash-cli/src/render/mod.rs\n+++ b/crates/lash-cli/src/render/mod.rs\n@@\n-old\n+new",
                    "diff": "--- a/crates/lash-cli/src/render/mod.rs\n+++ b/crates/lash-cli/src/render/mod.rs\n@@\n-old\n+new",
                    "firstChangedLine": 12
                }
            }),
            true,
            18,
        );

        assert_eq!(
            blocks[0].call.summary,
            "Edited crates/lash-cli/src/render/mod.rs (+1 -1)"
        );
        assert!(matches!(
            blocks[0].result.artifact,
            Some(ActivityArtifact::PatchPreview {
                total_added: 1,
                total_removed: 1,
                ..
            })
        ));
    }

    #[test]
    fn write_summary_uses_tool_result_summary_without_patch_preview() {
        let mut state = ActivityState::new();
        let blocks = state.project_tool_call(
            "write",
            json!({"path": "new.rs", "content": "fn main() {}\n"}),
            json!({
                "summary": "Successfully wrote 13 bytes to new.rs.",
                "path": "new.rs",
                "bytes": 13
            }),
            true,
            11,
        );

        assert_eq!(
            blocks[0].call.summary,
            "Successfully wrote 13 bytes to new.rs."
        );
        assert!(blocks[0].result.artifact.is_none());
    }
}
