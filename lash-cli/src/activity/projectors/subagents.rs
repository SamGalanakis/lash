use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_snippet, inline_text, tool_arg_str},
};

pub(crate) struct SubagentProjector;

impl ToolProjector for SubagentProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &[
            "spawn_agent",
            "send_message",
            "followup_task",
            "wait_agent",
            "close_agent",
            "list_agents",
        ]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        match ctx.name {
            "spawn_agent" => vec![project_spawn_agent(ctx)],
            "send_message" => vec![project_send_message(ctx)],
            "followup_task" => vec![project_followup_task(ctx)],
            "wait_agent" => project_wait_agent(ctx),
            "close_agent" => vec![project_close_agent(ctx)],
            "list_agents" => vec![project_list_agents(ctx)],
            _ => Vec::new(),
        }
    }
}

fn project_spawn_agent(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let task = tool_arg_str(&ctx.args, "task")
        .unwrap_or("spawn agent")
        .to_string();
    let capability = ctx
        .result
        .get("capability")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let path = ctx
        .result
        .get("path")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let name = subagent_name(
        ctx.result.get("task_name").and_then(|value| value.as_str()),
        Some(path),
        "subagent",
    );
    let mut detail_lines = Vec::new();
    detail_lines.push(format!("Task {}", inline_text(&task)));
    if !path.is_empty() {
        detail_lines.push(format!("Path {}", inline_text(path)));
    }
    if let Some(profile) = profile_line((!capability.is_empty()).then_some(capability), &ctx.result)
    {
        detail_lines.push(format!("Profile {profile}"));
    }
    block(
        ctx,
        format!("spawn subagent · {}", inline_text(&name)),
        detail_lines,
        None,
    )
}

fn project_send_message(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let target = tool_arg_str(&ctx.args, "target")
        .unwrap_or("agent")
        .to_string();
    let mut detail_lines = Vec::new();
    detail_lines.push(format!("Target {}", inline_text(&target)));
    if let Some(message) = tool_arg_str(&ctx.args, "message") {
        detail_lines.push(format!("Message {}", inline_text(message)));
    }
    block(
        ctx,
        format!(
            "message subagent · {}",
            inline_text(&summary_target(&target))
        ),
        detail_lines,
        None,
    )
}

fn project_followup_task(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let target = tool_arg_str(&ctx.args, "target")
        .unwrap_or("agent")
        .to_string();
    let mut detail_lines = Vec::new();
    detail_lines.push(format!("Target {}", inline_text(&target)));
    if let Some(task) = tool_arg_str(&ctx.args, "task") {
        detail_lines.push(format!("Task {}", inline_text(task)));
    }
    if let Some(status) = ctx.result.get("status").and_then(|value| value.as_str())
        && !status.is_empty()
    {
        detail_lines.push(format!("Status {status}"));
    }
    block(
        ctx,
        format!(
            "follow up subagent · {}",
            inline_text(&summary_target(&target))
        ),
        detail_lines,
        None,
    )
}

fn project_wait_agent(ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
    let timed_out = ctx
        .result
        .get("timed_out")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let events = ctx
        .result
        .get("events")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();

    if events.is_empty() {
        let summary = if timed_out {
            "waited on subagents · timeout".to_string()
        } else {
            "waited on subagents".to_string()
        };
        return vec![block(ctx, summary, Vec::new(), None)];
    }

    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    events
        .iter()
        .enumerate()
        .map(|(index, event)| {
            project_wait_event(
                ctx.name,
                args.clone(),
                result.clone(),
                ctx.success,
                ctx.duration_ms,
                index,
                event,
            )
        })
        .collect()
}

fn project_wait_event(
    name: &str,
    args: Value,
    result: Value,
    success: bool,
    duration_ms: u64,
    _index: usize,
    event: &Value,
) -> ActivityBlock {
    let status = if success {
        ActivityStatus::Completed
    } else {
        ActivityStatus::Failed
    };
    let kind = event
        .get("kind")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let tag = wait_event_tag(event);
    let summary = match kind {
        "task_started" => "subagent started".to_string(),
        "message" => {
            let from = event
                .get("from")
                .and_then(|value| value.as_str())
                .unwrap_or("agent");
            let to = event
                .get("to")
                .and_then(|value| value.as_str())
                .unwrap_or("agent");
            format!(
                "subagent message · {} → {}",
                inline_text(&summary_target(from)),
                inline_text(&summary_target(to))
            )
        }
        "task_completed" => completion_label(
            event
                .get("status")
                .and_then(|value| value.as_str())
                .unwrap_or("completed"),
        )
        .to_string(),
        "agent_closed" => "subagent closed".to_string(),
        _ => "subagent event".to_string(),
    };

    let detail_lines = wait_event_detail_lines(event);
    let artifact = wait_event_artifact(event);
    let mut block = ActivityBlock::new(
        ActivityKind::Subagent,
        name,
        args,
        summary,
        event_status(event, status),
        result,
        duration_ms,
    );
    block.call.tag = tag;
    block.result.detail_lines = detail_lines;
    block.result.artifact = artifact;
    block
}

fn project_close_agent(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let target = tool_arg_str(&ctx.args, "target")
        .unwrap_or("agent")
        .to_string();
    let detail_lines = ctx
        .result
        .get("closed")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|path| format!("Closed {}", inline_text(path)))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    block(
        ctx,
        format!("close subagent · {}", inline_text(&summary_target(&target))),
        detail_lines,
        None,
    )
}

fn project_list_agents(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let prefix = tool_arg_str(&ctx.args, "path_prefix")
        .map(inline_text)
        .unwrap_or_else(|| "/root".to_string());
    let detail_lines = ctx
        .result
        .get("agents")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let path = item.get("path").and_then(|value| value.as_str())?;
                    let status = item
                        .get("status")
                        .and_then(|value| value.as_str())
                        .unwrap_or("unknown");
                    let capability = item
                        .get("capability")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default();
                    let queued = item
                        .get("queued_tasks")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0);
                    let inbox = item
                        .get("inbox_messages")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0);
                    let mut parts = vec![inline_text(path), status.to_string()];
                    if !capability.is_empty() {
                        parts.push(capability.to_string());
                    }
                    if queued > 0 {
                        parts.push(format!("{queued} queued"));
                    }
                    if inbox > 0 {
                        parts.push(format!("{inbox} inbox"));
                    }
                    Some(parts.join(" · "))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    block(
        ctx,
        format!("list subagents · {}", inline_text(&summary_target(&prefix))),
        detail_lines,
        None,
    )
}

fn wait_event_detail_lines(event: &Value) -> Vec<String> {
    match event
        .get("kind")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
    {
        "task_started" => {
            let task = event
                .get("task")
                .and_then(|value| value.as_str())
                .unwrap_or("task");
            let path = event
                .get("path")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let capability = event
                .get("capability")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let mut lines = vec![format!("Task {}", inline_text(task))];
            if !path.is_empty() {
                lines.push(format!("Path {}", inline_text(path)));
            }
            if let Some(profile) =
                profile_line((!capability.is_empty()).then_some(capability), event)
            {
                lines.push(format!("Profile {profile}"));
            }
            lines
        }
        "message" => {
            let mut lines = Vec::new();
            if let Some(from) = event.get("from").and_then(|value| value.as_str()) {
                lines.push(format!("From {}", inline_text(from)));
            }
            if let Some(to) = event.get("to").and_then(|value| value.as_str()) {
                lines.push(format!("To {}", inline_text(to)));
            }
            if let Some(text) = event.get("message").and_then(|value| value.as_str()) {
                lines.push(format!("Message {}", inline_text(text)));
            }
            lines
        }
        "task_completed" => {
            let mut lines = Vec::new();
            if let Some(task) = event.get("task").and_then(|value| value.as_str()) {
                lines.push(format!("Task {}", inline_text(task)));
            }
            if let Some(error) = event.get("error").and_then(|value| value.as_str())
                && !error.trim().is_empty()
            {
                lines.push(format!("Error {}", inline_text(error)));
            }
            if let Some(path) = event.get("path").and_then(|value| value.as_str()) {
                lines.push(format!("Path {}", inline_text(path)));
            }
            if let Some(session) = event.get("session") {
                let mut parts = Vec::new();
                if let Some(model) = model_label(session) {
                    parts.push(model);
                }
                if let Some(iterations) = session.get("iterations").and_then(|value| value.as_u64())
                {
                    parts.push(format!(
                        "{} iteration{}",
                        iterations,
                        if iterations == 1 { "" } else { "s" }
                    ));
                }
                if let Some(tool_calls) = session.get("tool_calls").and_then(|value| value.as_u64())
                {
                    parts.push(format!(
                        "{} tool call{}",
                        tool_calls,
                        if tool_calls == 1 { "" } else { "s" }
                    ));
                }
                if !parts.is_empty() {
                    lines.push(format!("Run {}", parts.join(" · ")));
                }
                if let Some(tokens) = token_usage_line(session) {
                    lines.push(format!("Tokens {tokens}"));
                }
            }
            lines
        }
        "agent_closed" => event
            .get("path")
            .and_then(|value| value.as_str())
            .map(|path| vec![format!("Path {}", inline_text(path))])
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn wait_event_artifact(event: &Value) -> Option<ActivityArtifact> {
    let value = event.get("result")?;
    let text = result_preview(value)?;
    Some(ActivityArtifact::TextPreview {
        title: Some("Subagent result".to_string()),
        text,
    })
}

fn completion_label(status: &str) -> &'static str {
    match status {
        "interrupted" => "subagent stopped",
        "failed" => "subagent failed",
        _ => "subagent finished",
    }
}

fn wait_event_tag(event: &Value) -> Option<String> {
    match event
        .get("kind")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
    {
        "task_started" | "task_completed" | "agent_closed" => Some(subagent_name(
            None,
            event.get("path").and_then(|value| value.as_str()),
            "subagent",
        )),
        "message" => event
            .get("from")
            .and_then(|value| value.as_str())
            .map(summary_target),
        _ => None,
    }
}

fn event_status(event: &Value, fallback: ActivityStatus) -> ActivityStatus {
    match event
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
    {
        "failed" | "interrupted" => ActivityStatus::Failed,
        _ => fallback,
    }
}

fn token_usage_line(meta: &Value) -> Option<String> {
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
        parts.push(format!("{total} total"));
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

fn model_label(value: &Value) -> Option<String> {
    let model = value.get("model").and_then(|item| item.as_str())?;
    let variant = value
        .get("model_variant")
        .and_then(|item| item.as_str())
        .unwrap_or_default();
    if variant.is_empty() {
        Some(model.to_string())
    } else {
        Some(format!("{model} ({variant})"))
    }
}

fn profile_line(capability: Option<&str>, value: &Value) -> Option<String> {
    let model = value.get("model").and_then(|item| item.as_str());
    let variant = value
        .get("model_variant")
        .and_then(|item| item.as_str())
        .filter(|item| !item.is_empty());
    let capability = capability.filter(|item| !item.is_empty());

    let mut parts = Vec::new();
    if let Some(capability) = capability {
        parts.push(format!("{capability} capability"));
    }
    if let Some(model) = model {
        let rendered_model = match (variant, capability) {
            (Some(variant), Some(capability)) if variant.eq_ignore_ascii_case(capability) => {
                model.to_string()
            }
            (Some(variant), _) => format!("{model} ({variant})"),
            (None, _) => model.to_string(),
        };
        parts.push(rendered_model);
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" · "))
    }
}

fn subagent_name(task_name: Option<&str>, path: Option<&str>, fallback: &str) -> String {
    task_name
        .filter(|value| !value.trim().is_empty())
        .map(inline_text)
        .or_else(|| path.and_then(path_leaf))
        .unwrap_or_else(|| fallback.to_string())
}

fn summary_target(target: &str) -> String {
    let compact = inline_text(target);
    if compact == "/root" || compact == "root" {
        "/root".to_string()
    } else {
        path_leaf(target).unwrap_or(compact)
    }
}

fn path_leaf(path: &str) -> Option<String> {
    path.trim_matches('/')
        .rsplit('/')
        .find(|segment| !segment.is_empty())
        .map(str::to_string)
}

fn result_preview(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) if text.trim().is_empty() => None,
        Value::String(text) => Some(text.to_string()),
        Value::Object(obj) => {
            let entries = obj
                .iter()
                .map(|(key, value)| format!("{key}: {}", inline_snippet(&value.to_string(), 80)))
                .collect::<Vec<_>>();
            Some(format!("{{ {} }}", entries.join(", ")))
        }
        Value::Array(items) => {
            let rendered = items
                .iter()
                .take(4)
                .map(|item| inline_snippet(&item.to_string(), 40))
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!("[{rendered}]"))
        }
        other => Some(inline_snippet(&other.to_string(), 120)),
    }
}

fn block(
    ctx: &mut ProjectCtx<'_>,
    summary: String,
    detail_lines: Vec<String>,
    artifact: Option<ActivityArtifact>,
) -> ActivityBlock {
    let status = if ctx.success {
        ActivityStatus::Completed
    } else {
        ActivityStatus::Failed
    };
    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    ActivityBlock::new(
        ActivityKind::Subagent,
        ctx.name,
        args,
        summary,
        status,
        result,
        ctx.duration_ms,
    )
    .with_detail_lines(detail_lines)
    .with_artifact(artifact)
}
