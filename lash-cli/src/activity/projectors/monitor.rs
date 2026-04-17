//! Monitor projector.
//!
//! Renders `monitor` tool calls as first-class activities instead of
//! generic fallback rows.

use serde_json::Value;

use crate::activity::{
    ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_snippet, inline_text, tool_arg_str},
};

pub(crate) struct MonitorProjector;

impl ToolProjector for MonitorProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["monitor"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        if ctx.name != "monitor" {
            return Vec::new();
        }

        let status = if ctx.success {
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
}

fn monitor_summary(args: &Value) -> String {
    let target = tool_arg_str(args, "description").unwrap_or("monitor");
    format!("watch {}", inline_text(target))
}

fn monitor_detail_lines(args: &Value, result: &Value, success: bool) -> Vec<String> {
    if !success {
        return monitor_error_lines(result);
    }

    let Some(run_state) = result.get("run_state").and_then(Value::as_str) else {
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

    monitor_status_lines(run_state, description, command, persistent, timeout_ms)
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
    run_state: &str,
    description: &str,
    command: &str,
    persistent: bool,
    timeout_ms: u64,
) -> Vec<String> {
    let mut lines = vec![format!(
        "{} · {}",
        inline_text(run_state),
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

#[cfg(test)]
mod tests {
    use crate::activity::ActivityState;
    use serde_json::json;

    #[test]
    fn monitor_start_projects_human_friendly_status_lines() {
        let mut state = ActivityState::default();
        let blocks = state.blocks_for_tool_call(
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
                "run_state": "running"
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
        let blocks = state.blocks_for_tool_call(
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
                "run_state": "running"
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
        let blocks = state.blocks_for_tool_call(
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
}
