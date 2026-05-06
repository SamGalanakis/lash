//! `update_plan` tool + plugin.
//!
//! A root-only, interactive-only tool that lets the model publish a
//! checklist to the TUI's sticky plan dock. Each call fully replaces
//! the previously-published plan. The plugin:
//!
//! * exposes one tool, `update_plan`, with status values `pending` /
//!   `in_progress` / `completed` (at most one `in_progress` at a time)
//! * stores the latest snapshot on the plugin so it survives resume /
//!   snapshot (the runtime-side TUI reads it off the emitted surface
//!   event)
//! * emits a [`PluginSurfaceEvent::PanelUpsert`] after every successful
//!   call, formatted as the checklist markdown the CLI's plan-dock
//!   parser understands (`- [x]` / `- [~]` / `- [ ]`).
//!
//! Gating: the plugin's [`PluginFactory::build`] returns an inert
//! `SessionPlugin` whenever the session has a parent (i.e. the session
//! is a subagent, compaction child, or any other non-root session).
//! Interactive-vs-batch gating is handled by the registration site in
//! `lash-cli/src/bootstrap.rs`.

use std::sync::{Arc, Mutex};

use serde_json::json;

use lash::plugin::{
    PluginDirective, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    PluginSurfaceEvent, SessionPlugin,
};
use lash::{PromptContribution, ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult};

const PLUGIN_ID: &str = "update_plan";
const PANEL_KEY: &str = "plan";
const PANEL_TITLE: &str = "PLAN";
const PLANNING_GUIDANCE: &str = concat!(
    "Use `update_plan` for substantial multi-step work and skip it for trivial or single-step asks. ",
    "Write short steps and keep exactly one step `in_progress` while work is underway. ",
    "Mark completed work before moving on, use `explanation` when the plan changes, and update the plan as soon as scope or sequencing shifts. ",
    "Do not let the plan go stale while coding or running validation. ",
    "After an `update_plan` call, briefly summarize what changed and what comes next instead of repeating the full checklist. ",
    "Finish by marking every step `completed` when the task is done.",
);

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanItem {
    pub step: String,
    pub status: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanSnapshot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plan: Vec<PlanItem>,
    #[serde(default)]
    pub generation: u64,
}

impl PlanSnapshot {
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Default)]
struct PlanState {
    explanation: Option<String>,
    items: Vec<PlanItem>,
    generation: u64,
}

impl PlanState {
    fn snapshot(&self) -> PlanSnapshot {
        PlanSnapshot {
            explanation: self.explanation.clone(),
            plan: self.items.clone(),
            generation: self.generation,
        }
    }

    fn apply(&mut self, explanation: Option<String>, items: Vec<PlanItem>) {
        self.explanation = explanation;
        self.items = items;
        self.generation = self.generation.wrapping_add(1).max(1);
    }
}

struct UpdatePlanTool {
    state: Arc<Mutex<PlanState>>,
}

#[async_trait::async_trait]
impl ToolProvider for UpdatePlanTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "update_plan",
                "Publish or replace the current plan: a list of short ordered steps with statuses (pending, in_progress, completed), plus an optional explanation. At most one step can be in_progress at a time. Each call fully replaces the previous plan. Use this for substantial multi-step work to keep progress visible to the user. After updating, briefly summarize what changed and what comes next instead of repeating the full checklist.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "explanation": { "type": "string" },
                        "plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": { "type": "string" },
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed"]
                                    }
                                },
                                "required": ["step", "status"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["plan"],
                    "additionalProperties": false
                }),
                serde_json::json!({ "type": "string" }),
            )
            .with_examples(vec![
                "{\"explanation\":\"I found the main renderer.\",\"plan\":[{\"step\":\"Inspect renderer\",\"status\":\"completed\"},{\"step\":\"Patch layout\",\"status\":\"in_progress\"},{\"step\":\"Run tests\",\"status\":\"pending\"}]}"
                    .into(),
            ])
            .with_execution_mode(ToolExecutionMode::Parallel),
        ]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match name {
            "update_plan" => execute_update_plan(&self.state, args),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

fn execute_update_plan(state: &Arc<Mutex<PlanState>>, args: &serde_json::Value) -> ToolResult {
    let explanation = args
        .get("explanation")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let Some(raw_plan) = args.get("plan").and_then(|value| value.as_array()) else {
        return ToolResult::err_fmt("Missing required parameter: plan");
    };
    if raw_plan.is_empty() {
        return ToolResult::err_fmt("Plan must contain at least one step");
    }

    let mut items = Vec::with_capacity(raw_plan.len());
    for (idx, item) in raw_plan.iter().enumerate() {
        let Some(object) = item.as_object() else {
            return ToolResult::err_fmt(format_args!(
                "Invalid plan[{idx}]: expected object with step and status"
            ));
        };
        let Some(step) = object
            .get("step")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return ToolResult::err_fmt(format_args!(
                "Invalid plan[{idx}].step: expected non-empty string"
            ));
        };
        let Some(status) = object
            .get("status")
            .and_then(|value| value.as_str())
            .map(str::trim)
        else {
            return ToolResult::err_fmt(format_args!(
                "Invalid plan[{idx}].status: expected string"
            ));
        };
        if !matches!(status, "pending" | "in_progress" | "completed") {
            return ToolResult::err_fmt(format_args!(
                "Invalid plan[{idx}].status: expected pending, in_progress, or completed"
            ));
        }
        items.push(PlanItem {
            step: step.to_string(),
            status: status.to_string(),
        });
    }

    let in_progress = items
        .iter()
        .filter(|item| item.status == "in_progress")
        .count();
    if in_progress > 1 {
        return ToolResult::err_fmt("Plan may contain at most one in_progress step");
    }

    let mut guard = state.lock().unwrap();
    guard.apply(explanation, items);
    ToolResult::ok(json!("Plan updated"))
}

/// Format a [`PlanSnapshot`] as the checklist markdown the CLI's plan
/// dock parser consumes (`- [x]` / `- [~]` / `- [ ]`). Matches the
/// shape documented in `lash-cli/src/plugin_surface.rs::parse_plan_items`.
fn format_plan_markdown(snapshot: &PlanSnapshot) -> String {
    let mut out = String::new();
    for item in &snapshot.plan {
        let glyph = match item.status.as_str() {
            "completed" => "[x]",
            "in_progress" => "[~]",
            _ => "[ ]",
        };
        out.push_str("- ");
        out.push_str(glyph);
        out.push(' ');
        out.push_str(&item.step);
        out.push('\n');
    }
    out
}

fn plan_panel_event(snapshot: &PlanSnapshot) -> PluginSurfaceEvent {
    PluginSurfaceEvent::PanelUpsert {
        key: PANEL_KEY.to_string(),
        title: PANEL_TITLE.to_string(),
        content: format_plan_markdown(snapshot),
    }
}

fn planning_prompt_contributions() -> Vec<PromptContribution> {
    vec![PromptContribution::guidance("Planning", PLANNING_GUIDANCE)]
}

/// Public plugin factory. Callers that want this plugin installed
/// (`lash-cli` under `profile.interactive_extras`) push an instance
/// onto the plugin factory list. In non-root sessions the factory
/// returns an inert plugin that registers nothing.
pub struct UpdatePlanPluginFactory;

impl UpdatePlanPluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl Default for UpdatePlanPluginFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginFactory for UpdatePlanPluginFactory {
    fn id(&self) -> &'static str {
        PLUGIN_ID
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(UpdatePlanPlugin {
            active: ctx.is_root_session(),
            state: Arc::new(Mutex::new(PlanState::default())),
        }))
    }
}

struct UpdatePlanPlugin {
    active: bool,
    state: Arc<Mutex<PlanState>>,
}

impl SessionPlugin for UpdatePlanPlugin {
    fn id(&self) -> &'static str {
        PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        reg.prompt().contribute(Arc::new(|_ctx| {
            Box::pin(async move { Ok(planning_prompt_contributions()) })
        }));
        reg.tools().provider(Arc::new(UpdatePlanTool {
            state: Arc::clone(&self.state),
        }))?;
        let after_state = Arc::clone(&self.state);
        reg.tool_calls().after(Arc::new(move |ctx| {
            let state = Arc::clone(&after_state);
            Box::pin(async move {
                if ctx.tool_name != "update_plan" {
                    return Ok(Vec::new());
                }
                if !ctx.result.success {
                    tracing::debug!(
                        target: "lash::update_plan",
                        "after_tool_call observed failed update_plan; skipping emit",
                    );
                    return Ok(Vec::new());
                }
                let snapshot = state
                    .lock()
                    .map_err(|_| PluginError::Session("update_plan state poisoned".to_string()))?
                    .snapshot();
                tracing::info!(
                    target: "lash::update_plan",
                    items = snapshot.plan.len(),
                    generation = snapshot.generation,
                    "emitting plan panel upsert",
                );
                Ok(vec![PluginDirective::emit_events(vec![plan_panel_event(
                    &snapshot,
                )])])
            })
        }));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::testing::{MockSessionManager, test_mode_factories};
    use lash::{
        ExecutionMode, PluginHost, PromptHookContext, PromptSlot, SessionReadView,
        SessionStateEnvelope,
    };

    #[tokio::test]
    async fn validates_shape() {
        let tool = UpdatePlanTool {
            state: Arc::new(Mutex::new(PlanState::default())),
        };
        let result = tool
            .execute(
                "update_plan",
                &json!({"plan":[{"step":"","status":"pending"}]}),
            )
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn rejects_multiple_in_progress_steps() {
        let tool = UpdatePlanTool {
            state: Arc::new(Mutex::new(PlanState::default())),
        };
        let result = tool
            .execute(
                "update_plan",
                &json!({
                    "plan":[
                        {"step":"a","status":"in_progress"},
                        {"step":"b","status":"in_progress"}
                    ]
                }),
            )
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn bumps_generation_on_success() {
        let state = Arc::new(Mutex::new(PlanState::default()));
        let tool = UpdatePlanTool {
            state: Arc::clone(&state),
        };
        assert_eq!(state.lock().unwrap().generation, 0);
        let result = tool
            .execute(
                "update_plan",
                &json!({
                    "plan":[{"step":"one","status":"pending"}]
                }),
            )
            .await;
        assert!(result.success);
        assert_eq!(state.lock().unwrap().generation, 1);
    }

    #[test]
    fn format_plan_markdown_uses_parser_compatible_glyphs() {
        let snapshot = PlanSnapshot {
            explanation: None,
            plan: vec![
                PlanItem {
                    step: "done work".into(),
                    status: "completed".into(),
                },
                PlanItem {
                    step: "current".into(),
                    status: "in_progress".into(),
                },
                PlanItem {
                    step: "later".into(),
                    status: "pending".into(),
                },
            ],
            generation: 1,
        };
        let md = format_plan_markdown(&snapshot);
        assert_eq!(
            md, "- [x] done work\n- [~] current\n- [ ] later\n",
            "format must match lash-cli plan dock parser"
        );
    }

    #[test]
    fn factory_marks_child_sessions_inactive() {
        let factory = UpdatePlanPluginFactory::new();
        let root_ctx = PluginSessionContext {
            session_id: "root".into(),
            execution_mode: lash::ExecutionMode::standard(),
            standard_context_approach: Some(lash::StandardContextApproach::default()),
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
            parent_session_id: None,
        };
        let child_ctx = PluginSessionContext {
            session_id: "child".into(),
            execution_mode: lash::ExecutionMode::standard(),
            standard_context_approach: Some(lash::StandardContextApproach::default()),
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
            parent_session_id: Some("root".into()),
        };
        assert!(root_ctx.is_root_session());
        assert!(!child_ctx.is_root_session());
        factory.build(&root_ctx).expect("root build");
        factory.build(&child_ctx).expect("child build");
    }

    #[tokio::test]
    async fn root_session_contributes_planning_guidance() {
        let mut factories = test_mode_factories();
        factories.push(Arc::new(UpdatePlanPluginFactory::new()));
        let plugin_host = PluginHost::new(factories);
        let session = plugin_host
            .build_standard_session("root", None)
            .expect("session");

        let contributions = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "root".to_string(),
                host: Arc::new(MockSessionManager::default()),
                state: SessionReadView::from_exported_state(&SessionStateEnvelope::default()),
                mode_turn_options: lash::ModeTurnOptions::default(),
            })
            .await
            .expect("prompt contributions");

        let contribution = contributions
            .iter()
            .find(|contribution| contribution.title.as_deref() == Some("Planning"))
            .expect("planning guidance");
        assert_eq!(contribution.slot, PromptSlot::Guidance);
        assert_eq!(contribution.content, PLANNING_GUIDANCE);
    }

    #[tokio::test]
    async fn child_session_does_not_contribute_planning_guidance() {
        let mut factories = test_mode_factories();
        factories.push(Arc::new(UpdatePlanPluginFactory::new()));
        let plugin_host = PluginHost::new(factories);
        let session = plugin_host
            .build_session_with_parent(
                "child",
                Some("root".to_string()),
                ExecutionMode::standard(),
                Some(lash::StandardContextApproach::default()),
                None,
                lash::plugin::SessionAuthorityContext::default(),
            )
            .expect("session");

        let contributions = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "child".to_string(),
                host: Arc::new(MockSessionManager::default()),
                state: SessionReadView::from_exported_state(&SessionStateEnvelope::default()),
                mode_turn_options: lash::ModeTurnOptions::default(),
            })
            .await
            .expect("prompt contributions");

        assert!(
            !contributions
                .iter()
                .any(|contribution| contribution.title.as_deref() == Some("Planning"))
        );
    }
}
