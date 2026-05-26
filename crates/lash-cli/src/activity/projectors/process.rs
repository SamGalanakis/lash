//! Process-control projector.
//!
//! Renders generic process handle inspection and cancellation calls as
//! first-class activities instead of generic fallback rows.

use serde_json::Value;

use crate::activity::{
    ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_text, tool_arg_str},
};

pub(crate) struct ProcessProjector;

impl ToolProjector for ProcessProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["list_process_handles", "cancel_process"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        match ctx.name {
            "list_process_handles" => project_list_process_handles(ctx),
            "cancel_process" => project_cancel_process(ctx),
            _ => Vec::new(),
        }
    }
}

fn project_list_process_handles(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let processes = ctx
        .result
        .get("processes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let running = processes
        .iter()
        .filter(|process| process.get("terminal").and_then(Value::as_str) == Some("running"))
        .count();
    let status = if !ctx.success {
        ActivityStatus::Failed
    } else if running > 0 {
        ActivityStatus::Running
    } else {
        ActivityStatus::Completed
    };
    let summary = if ctx.success {
        format!("listed processes · {} visible", processes.len())
    } else {
        "failed to list processes".to_string()
    };
    let detail_lines = processes
        .iter()
        .filter_map(process_list_detail_line)
        .collect::<Vec<_>>();
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

fn project_cancel_process(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let status = if ctx.success {
        ActivityStatus::Completed
    } else {
        ActivityStatus::Failed
    };
    let process_id = tool_arg_str(&ctx.args, "process_id").unwrap_or("process");
    let summary = if ctx.success {
        format!("stopped {}", inline_text(process_id))
    } else {
        format!("failed to stop {}", inline_text(process_id))
    };
    let detail_lines = process_stop_detail_lines(&ctx.result, ctx.success);
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

fn process_list_detail_line(process: &Value) -> Option<String> {
    let terminal = process
        .get("terminal")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let descriptor = process.get("descriptor").unwrap_or(&Value::Null);
    let kind = descriptor
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("process");
    let label = descriptor
        .get("label")
        .and_then(Value::as_str)
        .or_else(|| process.get("process_id").and_then(Value::as_str))
        .unwrap_or("process");
    Some(format!(
        "{} · {} · {}",
        inline_text(terminal),
        inline_text(kind),
        inline_text(label)
    ))
}

fn process_stop_detail_lines(result: &Value, success: bool) -> Vec<String> {
    if !success {
        return error_lines(result);
    }

    let terminal = result
        .get("terminal")
        .and_then(Value::as_str)
        .unwrap_or("cancelled");
    let process_id = result
        .get("process_id")
        .and_then(Value::as_str)
        .unwrap_or("process");

    vec![format!(
        "{} · {}",
        inline_text(terminal),
        inline_text(process_id)
    )]
}

fn error_lines(result: &Value) -> Vec<String> {
    match result {
        Value::String(text) => text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .take(3)
            .map(str::to_string)
            .collect(),
        other => serde_json::to_string(other)
            .ok()
            .map(|text| vec![text])
            .unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use crate::activity::ActivityState;
    use serde_json::json;

    #[test]
    fn process_list_projects_generic_process_status_lines() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
            "list_process_handles",
            json!({}),
            json!({
                "processes": [{
                    "process_id": "call-123",
                    "descriptor": { "kind": "tool", "label": "slow_tool" },
                    "terminal": "running"
                }]
            }),
            true,
            24,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "listed processes · 1 visible");
        assert_eq!(
            blocks[0].result.detail_lines,
            vec!["running · tool · slow_tool".to_string()]
        );
    }

    #[test]
    fn cancel_process_projects_cancelled_process_status() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
            "cancel_process",
            json!({
                "process_id": "tool-call-123",
            }),
            json!({
                "process_id": "tool-call-123",
                "terminal": "cancelled",
            }),
            true,
            5,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "stopped tool-call-123");
        assert_eq!(
            blocks[0].result.detail_lines,
            vec!["cancelled · tool-call-123".to_string()]
        );
    }
}
