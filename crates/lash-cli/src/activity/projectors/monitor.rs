//! Monitor/task-control projector.
//!
//! Renders `monitor` and related task-control calls as first-class
//! activities instead of generic fallback rows.

use serde_json::Value;

use crate::activity::{
    ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_snippet, inline_text, tool_arg_str},
};

pub(crate) struct MonitorProjector;

impl ToolProjector for MonitorProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["monitor", "tasks_list", "tasks_stop"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        match ctx.name {
            "monitor" => project_monitor(ctx),
            "tasks_list" => project_tasks_list(ctx),
            "tasks_stop" => project_tasks_stop(ctx),
            _ => Vec::new(),
        }
    }
}

fn project_monitor(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let status = if ctx.success && result_state(&ctx.result) == Some("running") {
        ActivityStatus::Running
    } else if ctx.success {
        ActivityStatus::Completed
    } else {
        ActivityStatus::Failed
    };
    let summary = monitor_summary(&ctx.args);
    let detail_lines = monitor_detail_lines(&ctx.args, &ctx.result, ctx.success);
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

fn project_tasks_list(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let tasks = ctx
        .result
        .get("tasks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let running = tasks
        .iter()
        .filter(|task| task.get("state").and_then(Value::as_str) == Some("running"))
        .count();
    let idle = tasks
        .iter()
        .filter(|task| task.get("state").and_then(Value::as_str) == Some("idle"))
        .count();
    let status = if !ctx.success {
        ActivityStatus::Failed
    } else if running > 0 {
        ActivityStatus::Running
    } else if idle > 0 {
        ActivityStatus::Partial
    } else {
        ActivityStatus::Completed
    };
    let summary = if ctx.success {
        format!("listed background tasks · {} active", running + idle)
    } else {
        "failed to list background tasks".to_string()
    };
    let detail_lines = tasks
        .iter()
        .filter_map(task_list_detail_line)
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

fn project_tasks_stop(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let status = if ctx.success {
        ActivityStatus::Completed
    } else {
        ActivityStatus::Failed
    };
    let task_id = tool_arg_str(&ctx.args, "task_id").unwrap_or("task");
    let summary = if ctx.success {
        format!("stopped {}", inline_text(task_id))
    } else {
        format!("failed to stop {}", inline_text(task_id))
    };
    let detail_lines = task_stop_detail_lines(&ctx.result, ctx.success);
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

fn monitor_summary(args: &Value) -> String {
    let target = tool_arg_str(args, "description").unwrap_or("monitor");
    format!("watch {}", inline_text(target))
}

fn monitor_detail_lines(args: &Value, result: &Value, success: bool) -> Vec<String> {
    if !success {
        return monitor_error_lines(result);
    }

    let Some(state) = result.get("state").and_then(Value::as_str) else {
        return Vec::new();
    };
    let description = result
        .get("description")
        .and_then(Value::as_str)
        .or_else(|| tool_arg_str(args, "description"))
        .unwrap_or("monitor");
    let persistent = result
        .get("persistent")
        .and_then(Value::as_bool)
        .or_else(|| args.get("persistent").and_then(Value::as_bool))
        .unwrap_or(false);
    let timeout_ms = result
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .or_else(|| args.get("timeout_ms").and_then(Value::as_u64))
        .unwrap_or(300_000);
    let command = result
        .get("command")
        .and_then(Value::as_str)
        .or_else(|| tool_arg_str(args, "command"))
        .unwrap_or_default();

    monitor_status_lines(state, description, command, persistent, timeout_ms)
}

fn monitor_error_lines(result: &Value) -> Vec<String> {
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

fn monitor_status_lines(
    state: &str,
    description: &str,
    command: &str,
    persistent: bool,
    timeout_ms: u64,
) -> Vec<String> {
    let mut lines = vec![format!(
        "{} · {}",
        inline_text(state),
        inline_text(description)
    )];

    if persistent {
        lines.push("Runs until stopped".to_string());
    } else {
        lines.push(format!("Timeout {} ms", timeout_ms));
    }

    if !command.trim().is_empty() {
        lines.push(format!("Command {}", inline_snippet(command, 96)));
    }

    lines
}

fn result_state(result: &Value) -> Option<&str> {
    result.get("state").and_then(Value::as_str)
}

fn task_list_detail_line(task: &Value) -> Option<String> {
    let state = task
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let kind = task.get("kind").and_then(Value::as_str).unwrap_or("task");
    let label = task
        .get("label")
        .and_then(Value::as_str)
        .or_else(|| task.get("task_id").and_then(Value::as_str))
        .unwrap_or("task");
    Some(format!(
        "{} · {} · {}",
        inline_text(state),
        inline_text(kind),
        inline_text(label)
    ))
}

fn task_stop_detail_lines(result: &Value, success: bool) -> Vec<String> {
    if !success {
        return monitor_error_lines(result);
    }

    let state = result
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or("stopped");
    let task_id = result
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or("task");
    let kind = result.get("kind").and_then(Value::as_str).unwrap_or("task");
    let kind_prefix = format!("{kind}:");
    let label = if task_id.starts_with(&kind_prefix) {
        task_id.to_string()
    } else {
        format!("{kind} {task_id}")
    };

    vec![format!("{} · {}", inline_text(state), inline_text(&label))]
}

#[cfg(test)]
mod tests {
    use crate::activity::ActivityState;
    use serde_json::json;

    #[test]
    fn monitor_start_projects_human_friendly_status_lines() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
            "monitor",
            json!({
                "description": "app errors",
                "command": "bash -lc 'echo started'",
            }),
            json!({
                "description": "app errors",
                "command": "bash -lc 'echo started'",
                "persistent": false,
                "timeout_ms": 300000,
                "state": "running"
            }),
            true,
            24,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "watch app errors");
        assert_eq!(blocks[0].result.detail_lines[0], "running · app errors");
        assert!(
            blocks[0]
                .result
                .detail_lines
                .iter()
                .any(|line| line == "Timeout 300000 ms")
        );
    }

    #[test]
    fn persistent_monitor_projects_run_until_stopped() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
            "monitor",
            json!({
                "description": "dev server",
                "command": "bash -lc 'tail -f /tmp/server.log'",
                "persistent": true,
            }),
            json!({
                "description": "dev server",
                "command": "bash -lc 'tail -f /tmp/server.log'",
                "persistent": true,
                "timeout_ms": 300000,
                "state": "running"
            }),
            true,
            17,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "watch dev server");
        assert_eq!(blocks[0].result.detail_lines[0], "running · dev server");
        assert!(
            blocks[0]
                .result
                .detail_lines
                .iter()
                .any(|line| line == "Runs until stopped")
        );
    }

    #[test]
    fn monitor_failure_surfaces_error_lines() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
            "monitor",
            json!({
                "description": "bad",
                "command": "false",
            }),
            json!("monitor requires `description`\nsecond line"),
            false,
            3,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "watch bad");
        assert_eq!(
            blocks[0].result.detail_lines,
            vec![
                "monitor requires `description`".to_string(),
                "second line".to_string()
            ]
        );
    }

    #[test]
    fn tasks_stop_projects_cancelled_task_status() {
        let mut state = ActivityState::default();
        let blocks = state.project_tool_call(
            "tasks_stop",
            json!({
                "task_id": "monitor:abc123",
            }),
            json!({
                "task_id": "monitor:abc123",
                "kind": "monitor",
                "state": "cancelled",
            }),
            true,
            5,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "stopped monitor:abc123");
        assert_eq!(
            blocks[0].result.detail_lines,
            vec!["cancelled · monitor:abc123".to_string()]
        );
    }
}
