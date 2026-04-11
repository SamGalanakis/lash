//! Web projector: `search_web` and `fetch_url`.
//!
//! Stateless. `search_web` produces a `WebSearch` block with a list of
//! result sources as a `SourceList` artifact; `fetch_url` produces a
//! `WebFetch` block with a `TextPreview` artifact of the fetched body.

use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_snippet, inline_text, text_preview_artifact, tool_arg_str},
};

pub(crate) struct WebProjector;

impl ToolProjector for WebProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["search_web", "fetch_url"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        match ctx.name {
            "search_web" => {
                let query = tool_arg_str(&ctx.args, "query")
                    .unwrap_or("search web")
                    .to_string();
                let items = web_sources(&ctx.result);
                let artifact = (!items.is_empty()).then(|| ActivityArtifact::SourceList {
                    title: "Sources".to_string(),
                    items,
                });
                let summary = web_search_summary(&query);
                let detail_lines = web_search_detail_lines(&ctx.result);
                let args = std::mem::replace(&mut ctx.args, Value::Null);
                let result = std::mem::replace(&mut ctx.result, Value::Null);
                vec![
                    ActivityBlock::new(
                        ActivityKind::WebSearch,
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
            "fetch_url" => {
                let url = tool_arg_str(&ctx.args, "url").unwrap_or("url").to_string();
                let artifact = text_preview_artifact(Some("Fetched content"), &ctx.result);
                let args = std::mem::replace(&mut ctx.args, Value::Null);
                let result = std::mem::replace(&mut ctx.result, Value::Null);
                vec![
                    ActivityBlock::new(
                        ActivityKind::WebFetch,
                        ctx.name,
                        args,
                        format!("fetch {}", display_url(&url)),
                        status,
                        result,
                        ctx.duration_ms,
                    )
                    .with_artifact(artifact),
                ]
            }
            _ => Vec::new(),
        }
    }
}

fn web_sources(result: &Value) -> Vec<String> {
    result
        .get("results")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("url").and_then(|value| value.as_str()))
                .take(5)
                .map(|url| url.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn web_search_summary(query: &str) -> String {
    format!("searched web for {:?}", inline_text(query))
}

fn web_search_detail_lines(result: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(answer) = result.get("answer").and_then(|value| value.as_str())
        && !answer.trim().is_empty()
    {
        lines.push(format!("Answer {}", inline_snippet(answer, 72)));
    }
    if let Some(results) = result.get("results").and_then(|value| value.as_array()) {
        lines.extend(results.iter().take(3).filter_map(|item| {
            let title = item.get("title").and_then(|value| value.as_str())?;
            let url = item
                .get("url")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            if url.is_empty() {
                Some(inline_snippet(title, 72))
            } else {
                Some(format!(
                    "{} · {}",
                    inline_snippet(title, 48),
                    display_url(url)
                ))
            }
        }));
    }
    lines
}

/// Strip `http(s)://` + trailing slash from a URL for display. Also
/// called from `shared::semantic_tool_summary` for the `fetch_url`
/// fallback summary, so it's `pub(crate)` rather than module-private.
pub(crate) fn display_url(url: &str) -> String {
    url.trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string()
}
