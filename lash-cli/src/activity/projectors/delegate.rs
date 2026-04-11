//! Delegate projector: `agent_call`, `agent_result`, `agent_kill`.
//!
//! All three share the `delegate_handles` map on the ctx:
//! - `agent_call` stores `handle_id → task` on success so later
//!   `agent_result` / `agent_kill` calls can recover the task label.
//! - `agent_result` removes the handle and rebuilds a completion
//!   summary based on the child's status and token usage.
//! - `agent_kill` removes the handle and produces a "delegate stopped"
//!   block.

use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_text, tool_arg_str},
};

pub(crate) struct DelegateProjector;

impl ToolProjector for DelegateProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["agent_call", "agent_result", "agent_kill"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        match ctx.name {
            "agent_call" => project_agent_call(ctx),
            "agent_result" => project_agent_result(ctx),
            "agent_kill" => project_agent_kill(ctx),
            _ => Vec::new(),
        }
    }
}

fn project_agent_call(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let task = tool_arg_str(&ctx.args, "task")
        .or_else(|| tool_arg_str(&ctx.args, "prompt"))
        .unwrap_or("delegate task")
        .to_string();
    if ctx.success
        && let Some(id) = tool_result_handle_id(&ctx.result)
    {
        ctx.delegate_handles.insert(id.to_string(), task.clone());
    }
    let status = if ctx.success {
        ActivityStatus::Completed
    } else {
        ActivityStatus::Failed
    };
    let mut detail_lines = Vec::new();
    if ctx.success {
        let model = ctx
            .result
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let variant = ctx
            .result
            .get("model_variant")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if !model.is_empty() {
            let label = if variant.is_empty() {
                model.to_string()
            } else {
                format!("{model} ({variant})")
            };
            detail_lines.push(label);
        }
    }
    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    vec![
        ActivityBlock::new(
            ActivityKind::Delegate,
            ctx.name,
            args,
            format!("delegate · {}", inline_text(&task)),
            status,
            result,
            ctx.duration_ms,
        )
        .with_detail_lines(detail_lines),
    ]
}

fn project_agent_result(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let handle_id = tool_arg_str(&ctx.args, "id")
        .unwrap_or_default()
        .to_string();
    let meta = ctx.result.get("session");
    let child_status = ctx
        .result
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let task = meta
        .and_then(|value| value.get("task"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .or_else(|| ctx.delegate_handles.remove(&handle_id))
        .unwrap_or_else(|| handle_id.clone());

    let delegate_status = delegate_activity_status(ctx.success, child_status);
    let delegate_summary = delegate_result_summary(child_status, &task);
    let mut detail_lines = Vec::new();
    if let Some(error) = ctx.result.get("error").and_then(|value| value.as_str())
        && !error.trim().is_empty()
    {
        detail_lines.push(format!("Error {}", inline_text(error)));
    }

    if let Some(meta) = meta {
        // Model info line.
        let model = meta
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let variant = meta
            .get("model_variant")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let tool_calls = meta.get("tool_calls").and_then(|v| v.as_u64());
        let iterations = meta.get("iterations").and_then(|v| v.as_u64());

        // Summary stats line.
        let mut parts = Vec::new();
        if !model.is_empty() {
            let label = if variant.is_empty() {
                model.to_string()
            } else {
                format!("{model} ({variant})")
            };
            parts.push(label);
        }
        if let Some(iters) = iterations {
            parts.push(format!(
                "{} iteration{}",
                iters,
                if iters == 1 { "" } else { "s" }
            ));
        }
        if let Some(tc) = tool_calls {
            parts.push(format!(
                "{} tool call{}",
                tc,
                if tc == 1 { "" } else { "s" }
            ));
        }
        if !parts.is_empty() {
            detail_lines.push(parts.join(" · "));
        }
        if let Some(token_line) = delegate_token_usage_line(meta) {
            detail_lines.push(token_line);
        }
    }

    let artifact = ctx
        .result
        .get("result")
        .and_then(|value| value.as_str())
        .filter(|text| !text.trim().is_empty())
        .map(|text| ActivityArtifact::TextPreview {
            title: Some("Delegate result".to_string()),
            text: text.to_string(),
        });
    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    vec![
        ActivityBlock::new(
            ActivityKind::Delegate,
            ctx.name,
            args,
            delegate_summary,
            delegate_status,
            result,
            ctx.duration_ms,
        )
        .with_detail_lines(detail_lines)
        .with_artifact(artifact),
    ]
}

fn project_agent_kill(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let handle_id = tool_arg_str(&ctx.args, "id")
        .unwrap_or_default()
        .to_string();
    let task = ctx.delegate_handles.remove(&handle_id).unwrap_or(handle_id);
    let status = if ctx.success {
        ActivityStatus::Completed
    } else {
        ActivityStatus::Failed
    };
    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    vec![ActivityBlock::new(
        ActivityKind::Delegate,
        ctx.name,
        args,
        format!("delegate stopped · {}", inline_text(&task)),
        status,
        result,
        ctx.duration_ms,
    )]
}

// ─── Helpers (private to this projector) ─────────────────────────────────────

fn tool_result_handle_id(result: &Value) -> Option<&str> {
    result
        .get("id")
        .and_then(|value| value.as_str())
        .or_else(|| result.get("session_id").and_then(|value| value.as_str()))
}

fn delegate_activity_status(success: bool, child_status: &str) -> ActivityStatus {
    if !success || matches!(child_status, "failed" | "interrupted") {
        ActivityStatus::Failed
    } else {
        ActivityStatus::Completed
    }
}

fn delegate_result_summary(child_status: &str, task: &str) -> String {
    let label = match child_status {
        "interrupted" => "delegate stopped",
        "failed" => "delegate failed",
        _ => "delegate done",
    };
    format!("{label} · {}", inline_text(task))
}

fn delegate_token_usage_line(meta: &Value) -> Option<String> {
    let usage = meta.get("token_usage")?;
    let total = usage.get("total_tokens").and_then(|value| value.as_u64());
    let input = usage.get("input_tokens").and_then(|value| value.as_u64());
    let output = usage.get("output_tokens").and_then(|value| value.as_u64());
    let reasoning = usage
        .get("reasoning_tokens")
        .and_then(|value| value.as_u64())
        .filter(|value| *value > 0);
    let cached = usage
        .get("cached_input_tokens")
        .and_then(|value| value.as_u64())
        .filter(|value| *value > 0);

    let mut parts = Vec::new();
    if let Some(total) = total {
        parts.push(format!("{total} total tokens"));
    }
    if let Some(input) = input {
        parts.push(format!("{input} in"));
    }
    if let Some(output) = output {
        parts.push(format!("{output} out"));
    }
    if let Some(reasoning) = reasoning {
        parts.push(format!("{reasoning} reasoning"));
    }
    if let Some(cached) = cached {
        parts.push(format!("{cached} cached"));
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" · "))
    }
}
