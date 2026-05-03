//! Generic projector: `search_tools` and the fallback for any unknown
//! tool name.
//!
//! The generic projector is special because it acts as both a
//! registered projector (for `search_tools`, which wants rich detail
//! lines) and as the fallback path in `ActivityState::project_tool_call`
//! for names with no dedicated projector. Anything the fallback
//! produces is an `ActivityKind::GenericTool` with a semantic summary
//! and an optional text preview artifact.

use serde_json::Value;

use crate::activity::{
    ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{
        inline_text, named_description_detail_lines, semantic_tool_summary, text_preview_artifact,
        tool_arg_str,
    },
};

pub(crate) struct GenericProjector;

impl ToolProjector for GenericProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["search_tools"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        match ctx.name {
            "search_tools" => {
                let summary = tool_search_summary(&ctx.args);
                let detail_lines = tool_search_detail_lines(&ctx.result);
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
            _ => vec![fallback_block(ctx, status)],
        }
    }
}

/// Build a `GenericTool` block for an unregistered tool. Called from
/// `ActivityState::project_tool_call` as the default fallback.
pub(crate) fn fallback_block(ctx: &mut ProjectCtx<'_>, status: ActivityStatus) -> ActivityBlock {
    let summary = semantic_tool_summary(ctx.name, &ctx.args);
    let artifact = text_preview_artifact(None, &ctx.result);
    let args = std::mem::replace(&mut ctx.args, Value::Null);
    let result = std::mem::replace(&mut ctx.result, Value::Null);
    ActivityBlock::new(
        ActivityKind::GenericTool,
        ctx.name,
        args,
        summary,
        status,
        result,
        ctx.duration_ms,
    )
    .with_artifact(artifact)
}

fn tool_search_summary(args: &Value) -> String {
    tool_arg_str(args, "query")
        .map(|query| format!("searched tools for {:?}", inline_text(query)))
        .unwrap_or_else(|| "browsed tools".to_string())
}

fn tool_search_detail_lines(result: &Value) -> Vec<String> {
    named_description_detail_lines(result, 4)
}
