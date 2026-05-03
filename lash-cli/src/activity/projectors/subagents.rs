use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_text, tool_arg_str},
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
    let capability_arg = tool_arg_str(&ctx.args, "capability").unwrap_or_default();
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
    if !capability_arg.is_empty() {
        detail_lines.push(format!(
            "Profile {} capability",
            inline_text(capability_arg)
        ));
    }
    block(
        ctx,
        format!("spawn subagent · {}", inline_text(&name)),
        detail_lines,
        None,
    )
}

fn subagent_name(agent_name: Option<&str>, path: Option<&str>, fallback: &str) -> String {
    agent_name
        .filter(|value| !value.trim().is_empty())
        .map(inline_text)
        .or_else(|| path.map(inline_text))
        .unwrap_or_else(|| fallback.to_string())
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
