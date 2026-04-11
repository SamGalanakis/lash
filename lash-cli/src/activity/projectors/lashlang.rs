//! Lashlang projector: `execute_lashlang`.
//!
//! Produces a `Hidden` kind block with a `TextPreview` artifact
//! containing the raw lashlang source. The `Hidden` kind causes
//! `render/activity.rs` to suppress the call line and only show the
//! artifact when the user expands the block. Stateless.

use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::tool_arg_str,
};

pub(crate) struct LashlangProjector;

impl ToolProjector for LashlangProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["execute_lashlang"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        let code = tool_arg_str(&ctx.args, "code").unwrap_or("").to_string();
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        let args = std::mem::replace(&mut ctx.args, Value::Null);
        let result = std::mem::replace(&mut ctx.result, Value::Null);
        vec![
            ActivityBlock::new(
                ActivityKind::Hidden,
                ctx.name,
                args,
                String::new(),
                status,
                result,
                ctx.duration_ms,
            )
            .with_artifact(Some(ActivityArtifact::TextPreview {
                title: Some("lashlang".to_string()),
                text: code,
            })),
        ]
    }
}
