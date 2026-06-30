//! Plan-mode prompt surface: approval request/response types, the prompt
//! trait, and the user-facing exit/guidance text builders.

use super::*;

pub(crate) fn plan_exit_next_turn_input(display: &str, note: Option<&str>) -> String {
    if let Some(note) = note.filter(|note| !note.trim().is_empty()) {
        format!(
            "The user approved the plan. Execute the plan in `{display}` now — start immediately, do not ask for confirmation.\n\nUser note: {note}"
        )
    } else {
        format!(
            "The user approved the plan. Execute the plan in `{display}` now — start immediately, do not ask for confirmation."
        )
    }
}

pub(crate) fn plan_exit_fresh_context_input(display: &str) -> String {
    format!("Do a full, faithful implementation of the plan found at: {display}")
}

pub(crate) fn plan_exit_confirmation_display(selection: &str, note: Option<&str>) -> String {
    if let Some(note) = note.filter(|note| !note.trim().is_empty()) {
        format!("{selection}\n\nNote: {note}")
    } else {
        selection.to_string()
    }
}

pub(crate) fn plan_mode_guidance_message(plan_path: &Path) -> PluginMessage {
    let display = plan_display_path(plan_path);
    PluginMessage::text(
        lash_core::MessageRole::System,
        format!(
            "Plan mode: use `{display}` as the single source of truth. Use `files.glob`, `files.grep`, `files.read`, `web.search`, `web.fetch`, and `user.ask` as needed, and update only that file with `files.edit` or `files.write`. Do not present the plan with snippets, showcases, or prose checklists; the host can surface the file path while planning. When the plan is ready for review, call `plan.exit`."
        ),
    )
}

pub(crate) fn plan_mode_tool_note(plan_path: Option<&Path>) -> String {
    match plan_path {
        Some(path) => format!(
            "Plan mode tools: `files.glob`, `files.grep`, `files.read`, `web.search`, `web.fetch`, `user.ask`, `files.edit`/`files.write` for `{}`, `plan.exit`. The host can surface the plan file path; full review happens in `plan.exit`.",
            plan_display_path(path)
        ),
        None => "Plan mode tools: `files.glob`, `files.grep`, `files.read`, `web.search`, `web.fetch`, `user.ask`, plan-file `files.edit`/`files.write`, `plan.exit`. The host can surface the plan file path; full review happens in `plan.exit`.".to_string(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanModePromptRequest {
    pub question: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review: Option<PlanModePromptReview>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_note: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanModePromptReview {
    pub title: String,
    pub markdown: String,
}

impl PlanModePromptRequest {
    pub fn single(question: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            question: question.into(),
            options,
            review: None,
            allow_note: false,
        }
    }

    pub fn with_review(mut self, title: impl Into<String>, markdown: impl Into<String>) -> Self {
        self.review = Some(PlanModePromptReview {
            title: title.into(),
            markdown: markdown.into(),
        });
        self
    }

    pub fn with_optional_note(mut self) -> Self {
        self.allow_note = !self.options.is_empty();
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanModePromptResponse {
    Single {
        selection: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
}

#[async_trait::async_trait]
pub trait PlanModePrompt: Send + Sync {
    async fn prompt_user(
        &self,
        request: PlanModePromptRequest,
    ) -> Result<PlanModePromptResponse, PluginError>;
}
