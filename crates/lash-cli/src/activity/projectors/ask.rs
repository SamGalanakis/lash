//! Ask projector.
//!
//! `ask` produces an `Ask` block with a `QuestionPanel` artifact
//! describing the question and its options.

use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx,
    QuestionPanelArtifact, QuestionPanelOption, QuestionPanelSelectionMode, ToolProjector,
    shared::{inline_text, tool_arg_list, tool_arg_str},
};

pub(crate) struct AskProjector;

impl ToolProjector for AskProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["ask"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        if ctx.name != "ask" {
            return Vec::new();
        }
        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };
        let detail_lines = ask_detail_lines(&ctx.args, &ctx.result);
        let artifact = inline_question_panel_artifact(&ctx.args, &ctx.result);
        let args = std::mem::replace(&mut ctx.args, Value::Null);
        let result = std::mem::replace(&mut ctx.result, Value::Null);
        vec![
            ActivityBlock::new(
                ActivityKind::Ask,
                ctx.name,
                args,
                "Question",
                status,
                result,
                ctx.duration_ms,
            )
            .with_detail_lines(detail_lines)
            .with_artifact(artifact),
        ]
    }
}

fn ask_detail_lines(args: &Value, _result: &Value) -> Vec<String> {
    let mut lines = Vec::new();

    if let Some(question) = tool_arg_str(args, "question") {
        let mut question_lines = question
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(inline_text);
        if let Some(first_line) = question_lines.next() {
            lines.push(first_line);
            lines.extend(question_lines);
        }
    }

    for (idx, option) in tool_arg_list(args, "options").into_iter().enumerate() {
        lines.push(format!("{}. {}", idx + 1, option));
    }

    lines
}

fn prompt_lines(question: &str) -> Vec<String> {
    question
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

fn inline_question_panel_artifact(args: &Value, result: &Value) -> Option<ActivityArtifact> {
    let question = tool_arg_str(args, "question")?;
    let mut options = tool_arg_list(args, "options")
        .into_iter()
        .map(|label| QuestionPanelOption {
            label,
            selected: false,
        })
        .collect::<Vec<_>>();
    let mut selection_mode = (!options.is_empty()).then_some(QuestionPanelSelectionMode::Single);
    let note = result
        .get("note")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let mut answer = None;

    match result.get("kind").and_then(|value| value.as_str()) {
        Some("single") => {
            selection_mode = Some(QuestionPanelSelectionMode::Single);
            if let Some(selection) = result.get("selection").and_then(|value| value.as_str()) {
                if let Some(option) = options.iter_mut().find(|option| option.label == selection) {
                    option.selected = true;
                } else if !selection.trim().is_empty() {
                    answer = Some(selection.trim().to_string());
                }
            }
        }
        Some("multi") => {
            selection_mode = Some(QuestionPanelSelectionMode::Multi);
            let mut unmatched = Vec::new();
            for selection in result
                .get("selections")
                .and_then(|value| value.as_array())
                .into_iter()
                .flatten()
                .filter_map(|value| value.as_str())
            {
                if let Some(option) = options.iter_mut().find(|option| option.label == selection) {
                    option.selected = true;
                } else if !selection.trim().is_empty() {
                    unmatched.push(selection.trim().to_string());
                }
            }
            if !unmatched.is_empty() {
                answer = Some(unmatched.join(", "));
            }
        }
        Some("text") => {
            answer = result
                .get("text")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
        }
        _ => {}
    }

    Some(ActivityArtifact::QuestionPanel(QuestionPanelArtifact {
        prompt_lines: prompt_lines(question),
        options,
        selection_mode,
        answer,
        note,
    }))
}
