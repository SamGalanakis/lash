//! Snippet projector: `show_snippet_to_user`.
//!
//! Produces a `GenericTool` block with a `SnippetPreview` artifact
//! that the renderer later unfolds into a highlighted code snippet.

use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx,
    SnippetPreviewArtifact, SnippetRenderMode, ToolProjector,
    shared::{compact_path_display, semantic_tool_summary},
};

pub(crate) struct SnippetProjector;

impl ToolProjector for SnippetProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["show_snippet_to_user"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        let artifact = snippet_preview_artifact(&ctx.result);
        let summary = snippet_summary(&ctx.result)
            .unwrap_or_else(|| semantic_tool_summary(ctx.name, &ctx.args));
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
            .with_artifact(artifact),
        ]
    }
}

fn snippet_preview_artifact(result: &Value) -> Option<ActivityArtifact> {
    let path = result.get("path").and_then(|value| value.as_str())?;
    let start_line = result.get("start_line").and_then(|value| value.as_u64())? as usize;
    let end_line = result.get("end_line").and_then(|value| value.as_u64())? as usize;
    let content = result.get("content").and_then(|value| value.as_str())?;
    let render_mode = match result.get("render_mode").and_then(|value| value.as_str()) {
        Some("markdown") => SnippetRenderMode::Markdown,
        Some("code") => SnippetRenderMode::Code,
        _ => SnippetRenderMode::Text,
    };
    Some(ActivityArtifact::SnippetPreview(SnippetPreviewArtifact {
        title: result
            .get("title")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        path: path.to_string(),
        start_line,
        end_line,
        content: content.to_string(),
        render_mode,
        language: result
            .get("language")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_string),
    }))
}

fn snippet_summary(result: &Value) -> Option<String> {
    let path = result.get("path").and_then(|value| value.as_str())?;
    let start_line = result.get("start_line").and_then(|value| value.as_u64())?;
    let end_line = result.get("end_line").and_then(|value| value.as_u64())?;
    Some(format!(
        "show {}:{}-{} to user",
        compact_path_display(path),
        start_line,
        end_line
    ))
}

#[cfg(test)]
mod tests {
    use crate::activity::{
        ActivityArtifact, ActivityState, SnippetPreviewArtifact, SnippetRenderMode,
    };
    use serde_json::json;

    #[test]
    fn show_snippet_to_user_builds_snippet_preview_artifact() {
        let mut state = ActivityState::new();
        let blocks = state.blocks_for_tool_call(
            "show_snippet_to_user",
            json!({
                "path": "lash-cli/src/render/mod.rs",
                "start_line": 12,
                "end_line": 14
            }),
            json!({
                "path": "lash-cli/src/render/mod.rs",
                "start_line": 12,
                "end_line": 14,
                "title": "Queue preview",
                "content": "fn one() {}\nfn two() {}",
                "render_mode": "code",
                "language": "rs"
            }),
            true,
            5,
        );

        assert_eq!(
            blocks[0].call.summary,
            "show lash-cli/src/render/mod.rs:12-14 to user"
        );
        assert!(matches!(
            blocks[0].result.artifact,
            Some(ActivityArtifact::SnippetPreview(SnippetPreviewArtifact {
                title: Some(ref title),
                start_line: 12,
                end_line: 14,
                render_mode: SnippetRenderMode::Code,
                ..
            })) if title == "Queue preview"
        ));
    }
}
