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
    let capability_arg = tool_arg_str(&ctx.args, "capability").unwrap_or_default();
    let mut detail_lines = Vec::new();
    detail_lines.push(format!("Task {}", inline_text(&task)));
    if !capability_arg.is_empty() {
        detail_lines.push(format!(
            "Profile {} capability",
            inline_text(capability_arg)
        ));
    }
    if let Some(seed_summary) = summarize_seed_arg(&ctx.args) {
        detail_lines.push(seed_summary);
    }
    block(
        ctx,
        format!("spawn subagent · {}", inline_text(&task)),
        detail_lines,
        None,
    )
}

fn summarize_seed_arg(args: &Value) -> Option<String> {
    let seed = lash_protocol_rlm::RlmSeed::from_tool_args(args).ok()?;
    if seed.is_empty() {
        return None;
    }
    let mut parts = Vec::with_capacity(seed.projected.entries.len() + seed.globals.len());
    for (name, _) in seed.projected.entries {
        parts.push(format!("{name} (projected)"));
    }
    for name in seed.globals.keys() {
        parts.push(name.clone());
    }
    Some(format!("Seed {}", parts.join(", ")))
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
