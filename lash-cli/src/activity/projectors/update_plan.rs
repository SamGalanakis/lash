//! `update_plan` projector.
//!
//! The `update_plan` tool returns a flat `"Plan updated"` string, which
//! renders as an empty expanded activity when the user hits the expand
//! key. Surface the submitted plan items as detail lines so the
//! expanded view shows what the model actually pushed to the dock.

use serde_json::Value;

use crate::activity::{
    ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_text, tool_arg_str},
};

pub(crate) struct UpdatePlanProjector;

impl ToolProjector for UpdatePlanProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["update_plan"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        if ctx.name != "update_plan" {
            return Vec::new();
        }
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        let summary = update_plan_summary(&ctx.args);
        // On success the sticky plan dock already renders the
        // checklist — repeating every step as activity detail lines
        // just stacks the same information inline above it. Keep
        // detail lines only on failure, where the expanded activity
        // has to explain *why* the call was rejected.
        let detail_lines = if ctx.success {
            Vec::new()
        } else {
            update_plan_failure_lines(&ctx.args, &ctx.result)
        };
        let args = std::mem::replace(&mut ctx.args, Value::Null);
        let result = std::mem::replace(&mut ctx.result, Value::Null);
        vec![
            ActivityBlock::new(
                ActivityKind::GenericTool,
                ctx.name,
                args,
                summary,
                status,
                result,
                ctx.duration_ms,
            )
            .with_detail_lines(detail_lines),
        ]
    }
}

fn update_plan_summary(args: &Value) -> String {
    let count = args
        .get("plan")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    match count {
        0 => "update plan".to_string(),
        1 => "update plan · 1 step".to_string(),
        n => format!("update plan · {n} steps"),
    }
}

/// Render each plan item as a detail line prefixed with the same glyph
/// the sticky dock uses (`✓` done, `■` in-progress, `□` pending), so the
/// expanded activity mirrors what just landed in the dock.
fn update_plan_detail_lines(args: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(explanation) = tool_arg_str(args, "explanation") {
        lines.push(inline_text(explanation));
    }
    let Some(plan) = args.get("plan").and_then(Value::as_array) else {
        return lines;
    };
    for item in plan {
        let step = item
            .get("step")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("(empty)");
        let status = item.get("status").and_then(Value::as_str).unwrap_or("");
        let glyph = match status {
            "completed" => "✓",
            "in_progress" => "■",
            _ => "□",
        };
        lines.push(format!("{glyph} {}", inline_text(step)));
    }
    lines
}

/// When `update_plan` rejects the args, surface the validation error
/// first so the expanded activity explains the red `×` instead of
/// silently listing the plan the model tried to submit.
fn update_plan_failure_lines(args: &Value, result: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(message) = result
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(inline_text(message));
    } else if let Some(message) = result
        .get("error")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(inline_text(message));
    }
    lines.extend(update_plan_detail_lines(args));
    lines
}

#[cfg(test)]
mod tests {
    use crate::activity::ActivityState;
    use serde_json::json;

    #[test]
    fn update_plan_success_has_no_detail_lines_so_dock_is_the_only_render() {
        let mut state = ActivityState::new();
        let blocks = state.blocks_for_tool_call(
            "update_plan",
            json!({
                "explanation": "Cooking-show vibes.",
                "plan": [
                    {"step": "Find a suspiciously ripe tomato", "status": "completed"},
                    {"step": "Chop onions while pretending to be on a cooking show", "status": "in_progress"},
                    {"step": "Simmer everything until it smells impressive", "status": "pending"},
                ]
            }),
            json!("Plan updated"),
            true,
            3,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "update plan · 3 steps");
        assert!(
            blocks[0].result.detail_lines.is_empty(),
            "success path must not duplicate the plan dock's content",
        );
    }

    #[test]
    fn update_plan_failure_surfaces_error_before_submitted_plan() {
        let mut state = ActivityState::new();
        let blocks = state.blocks_for_tool_call(
            "update_plan",
            json!({
                "plan": [
                    {"step": "A", "status": "in_progress"},
                    {"step": "B", "status": "in_progress"}
                ]
            }),
            json!("Plan may contain at most one in_progress step"),
            false,
            2,
        );

        assert_eq!(
            blocks[0].result.detail_lines,
            vec![
                "Plan may contain at most one in_progress step".to_string(),
                "■ A".to_string(),
                "■ B".to_string(),
            ]
        );
    }
}
