use lash_subagents::AgentMetadata;
use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_snippet, inline_text, tool_arg_str},
};

pub(crate) struct SubagentProjector;

impl ToolProjector for SubagentProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["spawn_agent"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        match ctx.name {
            "spawn_agent" => vec![project_spawn_agent(ctx)],
            _ => Vec::new(),
        }
    }
}

fn project_spawn_agent(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let task = tool_arg_str(&ctx.args, "task")
        .unwrap_or("spawn agent")
        .to_string();
    let path = ctx
        .result
        .get("agent_name")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let metadata = lookup_metadata(ctx, path);
    let capability_arg = tool_arg_str(&ctx.args, "capability").unwrap_or_default();
    let capability = metadata
        .as_ref()
        .and_then(|meta| meta.capability.as_deref())
        .unwrap_or(capability_arg);
    let name = subagent_name(
        ctx.result
            .get("agent_name")
            .and_then(|value| value.as_str()),
        Some(path),
        "subagent",
    );
    let mut detail_lines = Vec::new();
    detail_lines.push(format!("Task {}", inline_text(&task)));
    if !path.is_empty() {
        detail_lines.push(format!("Agent {}", inline_text(path)));
    }
    if let Some(profile) = profile_line_from_metadata(
        (!capability.is_empty()).then_some(capability),
        metadata.as_ref(),
    ) {
        detail_lines.push(format!("Profile {profile}"));
    }
    let run_state = metadata.as_ref().map(|meta| meta.run_state.as_str());
    let mut activity = block(
        ctx,
        format!("spawn subagent · {}", inline_text(&name)),
        detail_lines,
        None,
    );
    if activity.result.status == ActivityStatus::Completed && run_state == Some("running") {
        activity.result.status = ActivityStatus::Running;
    }
    activity
}

fn lookup_metadata(ctx: &ProjectCtx<'_>, agent_name: &str) -> Option<AgentMetadata> {
    let _ = (ctx.subagent_host, agent_name);
    None
}

fn project_send_message(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let agent_name = tool_arg_str(&ctx.args, "agent_name")
        .unwrap_or("agent")
        .to_string();
    let mut detail_lines = Vec::new();
    detail_lines.push(format!("Agent {}", inline_text(&agent_name)));
    if let Some(message) = tool_arg_str(&ctx.args, "message") {
        detail_lines.push(format!("Message {}", inline_text(message)));
    }
    if let Some(delivery) = ctx.result.get("delivery").and_then(|value| value.as_str()) {
        detail_lines.push(format!("Delivery {delivery}"));
    }
    block(
        ctx,
        format!(
            "message subagent · {}",
            inline_text(&summary_agent_name(&agent_name))
        ),
        detail_lines,
        None,
    )
}

fn project_followup_task(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let agent_name = tool_arg_str(&ctx.args, "agent_name")
        .unwrap_or("agent")
        .to_string();
    let mut detail_lines = Vec::new();
    detail_lines.push(format!("Agent {}", inline_text(&agent_name)));
    if let Some(task) = tool_arg_str(&ctx.args, "task") {
        detail_lines.push(format!("Task {}", inline_text(task)));
    }
    if let Some(status) = ctx.result.get("status").and_then(|value| value.as_str())
        && !status.is_empty()
    {
        detail_lines.push(format!("Status {status}"));
    }
    if let Some(delivery) = ctx.result.get("delivery").and_then(|value| value.as_str()) {
        detail_lines.push(format!("Delivery {delivery}"));
    }
    block(
        ctx,
        format!(
            "follow up subagent · {}",
            inline_text(&summary_agent_name(&agent_name))
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
    let completed_count = ctx
        .result
        .get("completed")
        .and_then(|value| value.as_object())
        .map(serde_json::Map::len)
        .unwrap_or(0);
    let pending = ctx
        .result
        .get("pending")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    let completed = ctx
        .result
        .get("completed")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();

    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    let pending_count = pending.len();
    let mut parts = Vec::new();
    if timed_out {
        parts.push("timeout".to_string());
    }
    if completed_count > 0 {
        parts.push(format!("{} completed", completed_count));
    }
    if pending_count > 0 {
        parts.push(format!("{} pending", pending_count));
    }
    let summary = if parts.is_empty() {
        "waited on subagents".to_string()
    } else {
        format!("waited on subagents · {}", parts.join(" · "))
    };
    let mut detail_lines = pending
        .keys()
        .map(|agent_name| format!("Pending {}", inline_text(agent_name)))
        .collect::<Vec<_>>();
    if timed_out && pending_count == 0 {
        detail_lines.push("Timed out with no pending task agent_names".to_string());
    }
    let status = if !ctx.success {
        ActivityStatus::Failed
    } else if pending_count > 0 {
        ActivityStatus::Partial
    } else {
        ActivityStatus::Completed
    };
    let mut aggregate = ActivityBlock::new(
        ActivityKind::Subagent,
        ctx.name,
        args.clone(),
        summary,
        status,
        result.clone(),
        ctx.duration_ms,
    )
    .with_detail_lines(detail_lines);
    aggregate.children = completed
        .iter()
        .enumerate()
        .map(|(index, (_agent_name, completion))| {
            project_wait_completion(
                ctx.name,
                args.clone(),
                result.clone(),
                ctx.success,
                ctx.duration_ms,
                index,
                completion,
            )
        })
        .collect();
    vec![aggregate]
}

fn project_wait_completion(
    name: &str,
    args: Value,
    result: Value,
    success: bool,
    duration_ms: u64,
    _index: usize,
    completion: &Value,
) -> ActivityBlock {
    let status = if success {
        ActivityStatus::Completed
    } else {
        ActivityStatus::Failed
    };
    let summary = completion_label(
        completion
            .get("status")
            .and_then(|value| value.as_str())
            .unwrap_or("completed"),
    )
    .to_string();
    let tag = completion
        .get("agent_name")
        .and_then(|value| value.as_str())
        .map(|name| inline_text(&summary_agent_name(name)));
    let detail_lines = wait_completion_detail_lines(completion);
    let artifact = wait_completion_artifact(completion);
    let mut block = ActivityBlock::new(
        ActivityKind::Subagent,
        name,
        args,
        summary,
        completion_status(completion, status),
        result,
        duration_ms,
    );
    block.call.tag = tag;
    block.result.detail_lines = detail_lines;
    block.result.artifact = artifact;
    block
}

fn project_close_agent(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let agent_name = tool_arg_str(&ctx.args, "agent_name")
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
        format!(
            "close subagent · {}",
            inline_text(&summary_agent_name(&agent_name))
        ),
        detail_lines,
        None,
    )
}

fn project_list_agents(ctx: &mut ProjectCtx<'_>) -> ActivityBlock {
    let detail_lines = ctx
        .result
        .get("agents")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let path = item.get("agent_name").and_then(|value| value.as_str())?;
                    let status = item
                        .get("agent_state")
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
                    let messages = item
                        .get("queued_messages")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0);
                    let mut parts = vec![inline_text(path), status.to_string()];
                    if !capability.is_empty() {
                        parts.push(capability.to_string());
                    }
                    if queued > 0 {
                        parts.push(format!("{queued} queued"));
                    }
                    if messages > 0 {
                        parts.push(format!("{messages} messages"));
                    }
                    Some(parts.join(" · "))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    block(ctx, "list subagents".to_string(), detail_lines, None)
}

fn wait_completion_detail_lines(completion: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(agent_name) = completion
        .get("agent_name")
        .and_then(|value| value.as_str())
    {
        lines.push(format!("Agent {}", inline_text(agent_name)));
    }
    if let Some(task) = completion.get("task").and_then(|value| value.as_str()) {
        lines.push(format!("Task {}", inline_text(task)));
    }
    if let Some(error) = completion.get("error").and_then(|value| value.as_str())
        && !error.trim().is_empty()
    {
        lines.push(format!("Error {}", inline_text(error)));
    }
    lines
}

fn wait_completion_artifact(completion: &Value) -> Option<ActivityArtifact> {
    let value = completion.get("result")?;
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

fn completion_status(completion: &Value, fallback: ActivityStatus) -> ActivityStatus {
    match completion
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
    {
        "failed" | "interrupted" => ActivityStatus::Failed,
        _ => fallback,
    }
}

fn profile_line_from_metadata(
    capability: Option<&str>,
    metadata: Option<&AgentMetadata>,
) -> Option<String> {
    let model = metadata
        .map(|meta| meta.model.as_str())
        .filter(|model| !model.is_empty());
    let variant = metadata
        .and_then(|meta| meta.model_variant.as_deref())
        .filter(|v| !v.is_empty());
    let capability = capability.filter(|c| !c.is_empty());

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

fn subagent_name(agent_name: Option<&str>, path: Option<&str>, fallback: &str) -> String {
    agent_name
        .filter(|value| !value.trim().is_empty())
        .map(inline_text)
        .or_else(|| path.map(inline_text))
        .unwrap_or_else(|| fallback.to_string())
}

fn summary_agent_name(agent_name: &str) -> String {
    inline_text(agent_name)
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
