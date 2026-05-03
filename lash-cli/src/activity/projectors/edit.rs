//! Edit projector: `apply_patch`.
//!
//! Produces an `Edit` block with a `PatchPreview` artifact listing the
//! changed files. Consecutive edits merge via `merge_edit_activity`
//! through `ActivityState` so a run of patches collapses into a single
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
        &["apply_patch"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        let artifact = patch_preview_artifact(&ctx.result);
        let summary = patch_summary(&ctx.result)
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

// ─── Patch helpers ───────────────────────────────────────────────────────────

fn patch_preview_artifact(result: &Value) -> Option<ActivityArtifact> {
    let files = result
        .get("files")
        .and_then(|value| value.as_array())
        .map(|files| {
            files
                .iter()
                .filter_map(|file| {
                    let path = file.get("path").and_then(|value| value.as_str())?;
                    let status = file
                        .get("status")
                        .and_then(|value| value.as_str())
                        .unwrap_or("modified")
                        .to_string();
                    let from_path = file
                        .get("from_path")
                        .and_then(|value| value.as_str())
                        .map(str::to_string);
                    let diff = file
                        .get("diff")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let (added, removed) = patch_file_counts(file, &diff);
                    Some(PatchFilePreview {
                        path: path.to_string(),
                        from_path,
                        status,
                        added,
                        removed,
                        diff,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if files.is_empty() {
        return result
            .get("diff")
            .and_then(|value| value.as_str())
            .filter(|diff| !diff.trim().is_empty())
            .map(|diff| ActivityArtifact::DiffPreview {
                title: "Diff".to_string(),
                diff: diff.to_string(),
            });
    }

    let total_added = result
        .get("added")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or_else(|| files.iter().map(|file| file.added).sum());
    let total_removed = result
        .get("removed")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or_else(|| files.iter().map(|file| file.removed).sum());

    Some(ActivityArtifact::PatchPreview {
        files,
        total_added,
        total_removed,
    })
}

fn patch_summary(result: &Value) -> Option<String> {
    let ActivityArtifact::PatchPreview {
        files,
        total_added,
        total_removed,
    } = patch_preview_artifact(result)?
    else {
        return None;
    };

    Some(patch_summary_from_preview(
        &files,
        total_added,
        total_removed,
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

fn patch_file_counts(file: &Value, diff: &str) -> (usize, usize) {
    let added = file
        .get("added")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize);
    let removed = file
        .get("removed")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize);
    match (added, removed) {
        (Some(added), Some(removed)) => (added, removed),
        _ => count_diff_delta(diff),
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
    fn apply_patch_summary_prefers_semantic_single_file_copy() {
        let mut state = ActivityState::new();
        let blocks = state.project_tool_call(
            "apply_patch",
            json!({}),
            json!({
                "summary": "Applied patch to 1 file",
                "added": 3,
                "removed": 1,
                "files": [{
                    "path": "lash-cli/src/render/mod.rs",
                    "status": "modified",
                    "added": 3,
                    "removed": 1,
                    "diff": "--- a/lash-cli/src/render/mod.rs\n+++ b/lash-cli/src/render/mod.rs\n@@\n-old\n+new"
                }]
            }),
            true,
            18,
        );

        assert_eq!(
            blocks[0].call.summary,
            "Edited lash-cli/src/render/mod.rs (+3 -1)"
        );
        assert!(matches!(
            blocks[0].result.artifact,
            Some(ActivityArtifact::PatchPreview {
                total_added: 3,
                total_removed: 1,
                ..
            })
        ));
    }

    #[test]
    fn apply_patch_summary_shows_move_arrow() {
        let mut state = ActivityState::new();
        let blocks = state.project_tool_call(
            "apply_patch",
            json!({}),
            json!({
                "summary": "Applied patch to 1 file",
                "added": 2,
                "removed": 2,
                "files": [{
                    "path": "new.rs",
                    "from_path": "old.rs",
                    "status": "moved",
                    "added": 2,
                    "removed": 2,
                    "diff": "--- a/new.rs\n+++ b/new.rs\n@@\n-old\n+new"
                }]
            }),
            true,
            11,
        );

        assert_eq!(blocks[0].call.summary, "Moved old.rs → new.rs (+2 -2)");
    }
}
