//! Exploration projector: `read_file`, `grep`, `glob`, `ls`.
//!
//! All four produce an `ActivityKind::Exploration` block with one
//! `ExplorationOp`. Consecutive explorations merge into a single
//! `Explored` block via `merge_exploration_activity`, which is called
//! from `projection.rs::UiTimeline::append_activity`.

use serde_json::Value;

use crate::activity::{
    ActivityBlock, ActivityExtra, ActivityKind, ActivityStatus, ExplorationOp, ExplorationOpKind,
    ProjectCtx, ToolProjector,
    shared::{compact_path_display, tool_arg_str},
};

pub(crate) struct ExplorationProjector;

impl ToolProjector for ExplorationProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["read_file", "grep", "glob", "ls"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        let Some(op) = exploration_op_for(ctx.name, &ctx.args, &ctx.result) else {
            return Vec::new();
        };
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        let args = std::mem::replace(&mut ctx.args, Value::Null);
        let result = std::mem::replace(&mut ctx.result, Value::Null);
        vec![exploration_block(
            ctx.name,
            status,
            ctx.duration_ms,
            args,
            result,
            op,
        )]
    }
}

/// Return the `ExplorationOp` for one of the four read-only tools, or
/// `None` for anything else. Call this from the generic semantic summary
/// fallback too.
pub(crate) fn exploration_op_for(
    name: &str,
    args: &Value,
    result: &Value,
) -> Option<ExplorationOp> {
    match name {
        "read_file" => Some(ExplorationOp {
            kind: ExplorationOpKind::Read,
            subject: read_label(result).unwrap_or_else(|| {
                compact_path_display(tool_arg_str(args, "path").unwrap_or("file"))
            }),
        }),
        "grep" => Some(ExplorationOp {
            kind: ExplorationOpKind::Search,
            subject: grep_label(args),
        }),
        "glob" => Some(ExplorationOp {
            kind: ExplorationOpKind::Glob,
            subject: glob_label(args),
        }),
        "ls" => Some(ExplorationOp {
            kind: ExplorationOpKind::List,
            subject: compact_path_display(tool_arg_str(args, "path").unwrap_or(".")),
        }),
        _ => None,
    }
}

pub(crate) fn grep_label(args: &Value) -> String {
    format!("{:?}", tool_arg_str(args, "query").unwrap_or("query"))
}

pub(crate) fn glob_label(args: &Value) -> String {
    let pattern = tool_arg_str(args, "pattern").unwrap_or("*");
    if let Some(path) = tool_arg_str(args, "path") {
        format!("{} in {}", pattern, compact_path_display(path))
    } else {
        pattern.to_string()
    }
}

fn read_label(result: &Value) -> Option<String> {
    let text = result.as_str()?;
    let first_line = text.lines().next()?.trim();
    if first_line.is_empty() {
        return None;
    }
    if let Some(rest) = first_line.strip_prefix("==> ")
        && let Some(path) = rest.strip_suffix(" <==")
    {
        return Some(compact_path_display(path));
    }
    if let Some(path) = first_line.split_whitespace().next()
        && path.contains('/')
    {
        return Some(compact_path_display(path));
    }
    None
}

pub(crate) fn exploration_block(
    name: &str,
    status: ActivityStatus,
    duration_ms: u64,
    args: Value,
    result: Value,
    op: ExplorationOp,
) -> ActivityBlock {
    let mut block = ActivityBlock::new(
        ActivityKind::Exploration,
        name,
        args,
        String::new(),
        status,
        result,
        duration_ms,
    )
    .with_extra(Some(ActivityExtra::Exploration(vec![op])));
    rebuild_exploration_summary(&mut block);
    block
}

fn exploration_step_line(op: &ExplorationOp) -> String {
    match op.kind {
        ExplorationOpKind::Read => format!("Read {}", op.subject),
        ExplorationOpKind::Search => format!("Search {}", op.subject),
        ExplorationOpKind::Glob => format!("Glob {}", op.subject),
        ExplorationOpKind::List => format!("List {}", op.subject),
    }
}

/// Rebuild an Exploration block's display after its op list changes
/// (on construction or after a merge). Two modes:
///
/// - **Single op**: promote the op straight onto the call line, no
///   detail lines. Visually `· Read README.md`. A solo read/search/glob/ls
///   reads cleaner as one line than as a wrapper + indented body —
///   the old `EXPLORE · 1 step` layout doubled the line count for a
///   single operation and duplicated the op text.
/// - **Multiple ops**: show `· Explored` on the call line, list the
///   ops as detail lines below. The `N steps` counter is gone — the
///   detail list is always visible at the default expand level so
///   the count would duplicate what the reader can already see.
fn rebuild_exploration_summary(block: &mut ActivityBlock) {
    let Some(ActivityExtra::Exploration(ops)) = block.call.extra.as_ref() else {
        return;
    };
    block.call.tag = None;
    if ops.len() == 1 {
        block.call.summary = exploration_step_line(&ops[0]);
        block.result.detail_lines = Vec::new();
    } else {
        block.call.summary = "Explored".to_string();
        block.result.detail_lines = ops.iter().map(exploration_step_line).collect();
    }
}

/// Merge the ops from `incoming` into `target`, regenerating the
/// exploration summary. Called from `projection.rs` when two adjacent
/// exploration blocks should cluster into one `Explored` section.
/// Returns `true` if the merge succeeded.
pub fn merge_exploration_activity(target: &mut ActivityBlock, mut incoming: ActivityBlock) -> bool {
    let Some(ActivityExtra::Exploration(target_ops)) = target.call.extra.as_mut() else {
        return false;
    };
    let Some(ActivityExtra::Exploration(incoming_ops)) = incoming.call.extra.take() else {
        return false;
    };
    target_ops.extend(incoming_ops);
    target.duration_ms += incoming.duration_ms;
    rebuild_exploration_summary(target);
    true
}

#[cfg(test)]
mod tests {
    use crate::activity::ActivityState;
    use serde_json::json;

    #[test]
    fn read_file_labels_prefer_repo_relative_paths() {
        let mut state = ActivityState::new();
        let path = std::env::current_dir()
            .expect("cwd")
            .join("lash-cli/src/render/mod.rs");
        let blocks = state.blocks_for_tool_call(
            "read_file",
            json!({ "path": path }),
            json!("fn render() {}"),
            true,
            4,
        );

        assert_eq!(blocks[0].call.summary, "Read lash-cli/src/render/mod.rs");
        assert!(blocks[0].result.detail_lines.is_empty());
    }

    #[test]
    fn grep_labels_prefer_repo_relative_paths() {
        let mut state = ActivityState::new();
        let path = std::env::current_dir()
            .expect("cwd")
            .join("lash-cli/src/render/mod.rs");
        let blocks = state.blocks_for_tool_call(
            "grep",
            json!({ "query": format!("{} render_activity_block", path.display()) }),
            json!("match"),
            true,
            4,
        );

        assert_eq!(
            blocks[0].call.summary,
            format!(
                "Search {:?}",
                format!("{} render_activity_block", path.display())
            )
        );
        assert!(blocks[0].result.detail_lines.is_empty());
    }
}
