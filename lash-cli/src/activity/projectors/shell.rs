//! Shell projector: `exec_command`, `start_command`, and `write_stdin`.
//!
//! Stateful calls mutate the `shell_handles` map on the ctx — `start_command`
//! registers a running shell session, `write_stdin` clears it on exit
//! or keeps it alive while the session is still running. Stateful
//! tools like these are exactly why `ProjectCtx` exposes the handle
//! maps as split mutable borrows.

use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_text, tool_arg_str},
};

pub(crate) struct ShellProjector;

impl ToolProjector for ShellProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["exec_command", "start_command", "write_stdin"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        match ctx.name {
            "exec_command" | "start_command" => project_exec_command(ctx),
            "write_stdin" => project_write_stdin(ctx),
            _ => Vec::new(),
        }
    }
}

fn project_exec_command(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let command = tool_arg_str(&ctx.args, "cmd")
        .map(str::to_string)
        .unwrap_or_else(|| "command".to_string());
    let running_handle =
        tool_result_shell_id(&ctx.result).filter(|_| shell_result_running(&ctx.result));
    if let Some(handle_id) = running_handle.as_ref() {
        ctx.shell_handles.insert(handle_id.clone(), command.clone());
    }
    let exit_code = shell_result_exit_code(&ctx.result);
    let status = if !ctx.success || exit_code.is_some_and(|value| value != 0) {
        ActivityStatus::Failed
    } else {
        ActivityStatus::Completed
    };
    let mut detail_lines = Vec::new();
    if status == ActivityStatus::Failed {
        if let Some(exit_code) = exit_code {
            detail_lines.push(format!("Exited with {}", exit_code));
        }
        if let Some(workdir) = tool_arg_str(&ctx.args, "workdir") {
            detail_lines.push(format!("In {}", workdir));
        }
    } else if let Some(ref handle_id) = running_handle {
        detail_lines.push(format!("Handle {}", handle_id));
    }
    let artifact = shell_output_artifact(&ctx.result);
    let summary = if running_handle.is_some() {
        shell_start_summary(&command, status)
    } else if status == ActivityStatus::Failed {
        format!("failed {}", inline_text(&command))
    } else {
        inline_text(&command)
    };
    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    vec![
        ActivityBlock::new(
            ActivityKind::ShellCommand,
            ctx.name,
            args,
            summary,
            status,
            result,
            ctx.duration_ms,
        )
        .with_detail_lines(detail_lines)
        .with_artifact(artifact),
    ]
}

fn project_write_stdin(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    // Empty polls that returned nothing new are suppressed so a
    // long-running shell doesn't flood the transcript with noise.
    if should_suppress_shell_poll_activity(&ctx.args, &ctx.result, ctx.success) {
        return Vec::new();
    }

    let handle_id = tool_arg_shell_id(&ctx.args).unwrap_or_default();
    let command = ctx
        .shell_handles
        .get(&handle_id)
        .cloned()
        .unwrap_or_else(|| format!("shell {}", handle_id));
    let exit_code = shell_result_exit_code(&ctx.result);
    let running = shell_result_running(&ctx.result);
    if exit_code.is_some() {
        ctx.shell_handles.remove(&handle_id);
    } else if running && !handle_id.is_empty() {
        ctx.shell_handles
            .entry(handle_id.clone())
            .or_insert_with(|| command.clone());
    }
    let status = if !ctx.success || exit_code.is_some_and(|value| value != 0) {
        ActivityStatus::Failed
    } else {
        ActivityStatus::Completed
    };
    let chars = tool_arg_str(&ctx.args, "chars").unwrap_or("");
    let mut detail_lines = Vec::new();
    if !chars.is_empty() {
        detail_lines.push(format!("Input {}", inline_text(chars)));
    }
    if let Some(exit_code) = exit_code {
        detail_lines.push(format!("Exited with {}", exit_code));
    } else if running && !handle_id.is_empty() {
        detail_lines.push(format!("Handle {}", handle_id));
    }
    let artifact = shell_output_artifact(&ctx.result);
    let summary = shell_write_summary(&command, chars, &ctx.result, status);
    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    vec![
        ActivityBlock::new(
            ActivityKind::ShellInteraction,
            ctx.name,
            args,
            summary,
            status,
            result,
            ctx.duration_ms,
        )
        .with_detail_lines(detail_lines)
        .with_artifact(artifact),
    ]
}

// ─── Helpers (private to this projector) ─────────────────────────────────────

fn tool_result_shell_id(result: &Value) -> Option<String> {
    result
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .or_else(|| {
            result
                .get("session_id")
                .and_then(|value| value.as_i64())
                .map(|value| value.to_string())
        })
        .or_else(|| {
            result
                .get("session_id")
                .and_then(|value| value.as_u64())
                .map(|value| value.to_string())
        })
        .or_else(|| {
            result
                .get("session_id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
}

fn tool_arg_shell_id(args: &Value) -> Option<String> {
    args.get("id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .or_else(|| {
            args.get("session_id")
                .and_then(|value| value.as_i64())
                .map(|value| value.to_string())
        })
        .or_else(|| {
            args.get("session_id")
                .and_then(|value| value.as_u64())
                .map(|value| value.to_string())
        })
        .or_else(|| {
            args.get("session_id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
}

fn shell_result_output(result: &Value) -> Option<String> {
    result
        .get("output")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn shell_output_artifact(result: &Value) -> Option<ActivityArtifact> {
    shell_result_output(result).and_then(|text| {
        if text.trim().is_empty() {
            None
        } else {
            Some(ActivityArtifact::TextPreview {
                title: Some("Shell output".to_string()),
                text,
            })
        }
    })
}

fn shell_start_summary(command: &str, status: ActivityStatus) -> String {
    if status == ActivityStatus::Failed {
        format!("failed {}", inline_text(command))
    } else {
        format!("started {}", inline_text(command))
    }
}

fn shell_write_summary(
    command: &str,
    input: &str,
    result: &Value,
    status: ActivityStatus,
) -> String {
    if status == ActivityStatus::Failed {
        return format!("failed {}", inline_text(command));
    }
    if input.trim().is_empty() {
        if shell_result_output(result).is_some_and(|text| !text.trim().is_empty()) {
            return format!("read {}", inline_text(command));
        }
        if shell_result_running(result) {
            return format!("polled {}", inline_text(command));
        }
        return format!("finished {}", inline_text(command));
    }
    format!(
        "sent {} → {}",
        shell_input_preview(input),
        inline_text(command)
    )
}

fn shell_input_preview(input: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 24;
    let compact = inline_text(input);
    let preview: String = compact.chars().take(MAX_PREVIEW_CHARS).collect();
    if compact.chars().count() > MAX_PREVIEW_CHARS {
        format!("{preview}...")
    } else {
        preview
    }
}

fn shell_result_running(result: &Value) -> bool {
    result
        .get("running")
        .and_then(|value| value.as_bool())
        .unwrap_or_else(|| {
            result
                .get("session_id")
                .is_some_and(|value| !value.is_null())
        })
}

fn shell_result_exit_code(result: &Value) -> Option<i64> {
    result.get("exit_code").and_then(|value| value.as_i64())
}

fn should_suppress_shell_poll_activity(args: &Value, result: &Value, success: bool) -> bool {
    success
        && tool_arg_str(args, "chars").is_none()
        && shell_result_running(result)
        && shell_output_artifact(result).is_none()
        && shell_result_exit_code(result).is_none()
}

#[cfg(test)]
mod tests {
    use crate::activity::ActivityState;
    use serde_json::json;

    #[test]
    fn exec_command_success_uses_plain_command_summary() {
        let mut state = ActivityState::new();
        let blocks = state.blocks_for_tool_call(
            "exec_command",
            json!({
                "cmd": "date '+%Y-%m-%d %H:%M:%S %Z'",
                "workdir": "/home/sam/code/lash"
            }),
            json!({
                "output": "2026-03-12 17:11:12 CET",
                "exit_code": 0
            }),
            true,
            13,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "date '+%Y-%m-%d %H:%M:%S %Z'");
    }

    #[test]
    fn start_command_running_uses_session_id_handle() {
        let mut state = ActivityState::new();
        let blocks = state.blocks_for_tool_call(
            "start_command",
            json!({
                "cmd": "python3 -q",
            }),
            json!({
                "output": ">>> ",
                "status": "running",
                "done": false,
                "running": true,
                "session_id": 7,
                "wall_time_seconds": 0.01
            }),
            true,
            13,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].call.summary, "started python3 -q");
        assert!(
            blocks[0]
                .result
                .detail_lines
                .iter()
                .any(|line| line == "Handle 7")
        );
    }
}
