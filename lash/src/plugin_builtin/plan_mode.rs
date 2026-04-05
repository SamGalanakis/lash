use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::plugin::{
    ExternalOpDef, ExternalOpKind, PluginDirective, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, PluginSnapshotMeta, SessionParam, SessionPlugin, SnapshotReader,
    SnapshotWriter, ToolSurfaceContribution, ToolSurfaceOverride,
};
use crate::tools::{PatchAction, inspect_patch_ops};
use crate::{
    PluginMessage, PromptRequest, PromptResponse, ToolDefinition, ToolExecutionContext,
    ToolProvider, ToolResult,
};

const PLAN_MODE_BADGE_KEY: &str = "mode";
const PLAN_MODE_BADGE_LABEL: &str = "plan";

fn plan_display_path(path: &Path) -> String {
    let display = std::env::current_dir()
        .ok()
        .and_then(|cwd| path.strip_prefix(&cwd).ok().map(PathBuf::from))
        .unwrap_or_else(|| path.to_path_buf());
    let rendered = display.display().to_string();
    if rendered.is_empty() {
        ".".to_string()
    } else {
        rendered.replace('\\', "/")
    }
}

fn plan_exit_next_turn_input(display: &str, note: Option<&str>) -> String {
    if let Some(note) = note.filter(|note| !note.trim().is_empty()) {
        format!(
            "The plan at `{display}` is approved. Plan mode is off now. Execute that plan.\n\nUser note: {note}"
        )
    } else {
        format!("The plan at `{display}` is approved. Plan mode is off now. Execute that plan.")
    }
}

fn plan_file_exists(path: &Path) -> bool {
    path.is_file()
}

fn resolve_plan_path(run_session_id: &str) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|err| format!("Failed to determine cwd: {err}"))?;
    Ok(cwd
        .join(crate::legacy_repo_local_lash_dir())
        .join("plans")
        .join(format!("{run_session_id}.md")))
}

fn effective_run_session_id(state: &crate::AgentStateEnvelope) -> &str {
    state
        .policy
        .session_id
        .as_deref()
        .unwrap_or(&state.agent_id)
}

fn plan_mode_guidance_message(plan_path: &Path) -> PluginMessage {
    let display = plan_display_path(plan_path);
    let exists = plan_file_exists(plan_path);
    PluginMessage::text(
        crate::MessageRole::System,
        format!(
            r#"
Plan Mode

Plan mode is active. The official plan lives in `{display}`.
{plan_file_instruction}

Mode rules (strict)

- Stay in plan mode until it is explicitly disabled.
- If the user asks for execution while plan mode is active, plan that execution instead of doing it.
- Do not use `update_plan` in plan mode.
- Do not use `<proposed_plan>` tags or any special wrapper blocks.
- Keep the real plan in `{display}` and keep ordinary assistant prose concise.

Execution vs mutation in plan mode

- Read-only exploration is allowed when it improves the plan.
- `apply_patch` is allowed only for `{display}`.
- Do not edit any other file.
- Do not run shell commands or other tools whose purpose is to carry out the implementation.

Planning workflow

1. Ground yourself in the actual repo and current state before asking questions.
2. Use `ask(...)` for decisions that materially change the plan and cannot be discovered locally.
3. Keep `{display}` implementation-ready: clear scope, file paths, interfaces, risks, and verification.
4. When the plan is ready, call `plan_exit()`.

Finalization rule

- End with either an `ask(...)` question or `plan_exit()`.
- If you call `plan_exit()`, keep the post-tool response to a brief handoff. lash will use the approved plan in a fresh execution turn.
"#,
            plan_file_instruction = if exists {
                "A plan file already exists there. Read it first, then update it incrementally with `apply_patch`.".to_string()
            } else {
                "No plan file exists yet. Create it with `apply_patch` and keep updating that same file.".to_string()
            },
        )
        .trim()
        .to_string(),
    )
}

fn plan_mode_tool_note(plan_path: Option<&Path>) -> String {
    match plan_path {
        Some(path) => format!(
            "Plan mode: only `{}` may be edited with `apply_patch`. Use `plan_exit()` when the plan is ready.",
            plan_display_path(path)
        ),
        None => "Plan mode: only the session plan file under `.lash/plans/` may be edited with `apply_patch`. Use `plan_exit()` when the plan is ready.".to_string(),
    }
}

#[derive(Clone, Debug)]
pub struct PlanModePluginConfig {
    pub blocked_tools: BTreeSet<String>,
}

impl Default for PlanModePluginConfig {
    fn default() -> Self {
        Self {
            blocked_tools: [
                "agent_call",
                "agent_kill",
                "agent_result",
                "exec_command",
                "update_plan",
                "write_stdin",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        }
    }
}

impl PlanModePluginConfig {
    pub fn with_blocked_tools<I, S>(mut self, blocked_tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.blocked_tools = blocked_tools.into_iter().map(Into::into).collect();
        self
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct PlanModeSnapshot {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    generation: u64,
    #[serde(default)]
    plan_path: Option<String>,
}

#[derive(Debug, Default)]
struct PlanModeState {
    enabled: bool,
    generation: u64,
    plan_path: Option<PathBuf>,
    active_turn_applied_generation: Option<u64>,
}

impl PlanModeState {
    fn snapshot(&self) -> PlanModeSnapshot {
        PlanModeSnapshot {
            enabled: self.enabled,
            generation: self.generation,
            plan_path: self
                .plan_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
        }
    }

    fn set_enabled(&mut self, enabled: bool) -> PlanModeSnapshot {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.generation = self.generation.wrapping_add(1).max(1);
            self.active_turn_applied_generation = None;
        }
        self.snapshot()
    }

    fn toggle(&mut self) -> PlanModeSnapshot {
        self.set_enabled(!self.enabled)
    }

    fn prepare_turn(&mut self) -> bool {
        self.active_turn_applied_generation = None;
        if !self.enabled {
            return false;
        }
        self.active_turn_applied_generation = Some(self.generation);
        true
    }

    fn checkpoint_injection_needed(&mut self) -> bool {
        if !self.enabled || self.active_turn_applied_generation == Some(self.generation) {
            return false;
        }
        self.active_turn_applied_generation = Some(self.generation);
        true
    }

    fn finish_turn(&mut self) {
        self.active_turn_applied_generation = None;
    }

    fn badge_event(&self) -> crate::plugin::PluginSurfaceEvent {
        crate::plugin::PluginSurfaceEvent::ModeIndicatorUpsert {
            key: PLAN_MODE_BADGE_KEY.to_string(),
            label: PLAN_MODE_BADGE_LABEL.to_string(),
        }
    }

    fn clear_badge_event(&self) -> crate::plugin::PluginSurfaceEvent {
        crate::plugin::PluginSurfaceEvent::ModeIndicatorClear {
            key: PLAN_MODE_BADGE_KEY.to_string(),
        }
    }

    fn plan_path(&self) -> Option<PathBuf> {
        self.plan_path.clone()
    }

    fn ensure_plan_path_from_state(
        &mut self,
        state: &crate::AgentStateEnvelope,
    ) -> Result<PathBuf, PluginError> {
        if let Some(path) = self.plan_path() {
            return Ok(path);
        }
        let path =
            resolve_plan_path(effective_run_session_id(state)).map_err(PluginError::Session)?;
        self.plan_path = Some(path.clone());
        Ok(path)
    }

    fn set_plan_path(&mut self, path: PathBuf) {
        self.plan_path = Some(path);
    }

    fn restore_snapshot(&mut self, snapshot: PlanModeSnapshot) {
        self.enabled = snapshot.enabled;
        self.generation = snapshot.generation;
        self.plan_path = snapshot.plan_path.map(PathBuf::from);
        self.active_turn_applied_generation = None;
    }
}

async fn ensure_plan_path(
    state: &Arc<Mutex<PlanModeState>>,
    session_id: &str,
    host: &Arc<dyn crate::SessionManager>,
) -> Result<PathBuf, PluginError> {
    if let Some(path) = state
        .lock()
        .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
        .plan_path()
    {
        return Ok(path);
    }

    let snapshot = host.snapshot_session(session_id).await?;
    let run_session_id = effective_run_session_id(&snapshot).to_string();
    let path = resolve_plan_path(&run_session_id).map_err(PluginError::Session)?;
    state
        .lock()
        .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
        .set_plan_path(path.clone());
    Ok(path)
}

fn patch_allowed_for_plan_file(args: &serde_json::Value, plan_path: &Path) -> Result<(), String> {
    let input = args
        .get("input")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "plan mode requires `apply_patch.input`".to_string())?;
    let workdir = args
        .get("workdir")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty());
    let ops = inspect_patch_ops(input, workdir)?;
    if ops.is_empty() {
        return Err("plan mode requires a non-empty patch".to_string());
    }
    for op in ops {
        match op.action {
            PatchAction::Add | PatchAction::Update
                if op.path == plan_path && op.move_path.is_none() => {}
            PatchAction::Add | PatchAction::Update => {
                return Err(format!(
                    "plan mode only allows `apply_patch` to edit `{}`",
                    plan_display_path(plan_path)
                ));
            }
            PatchAction::Delete => {
                return Err(format!(
                    "plan mode does not allow deleting `{}`",
                    plan_display_path(plan_path)
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
struct PlanModeTools {
    state: Arc<Mutex<PlanModeState>>,
}

impl PlanModeTools {
    async fn execute_plan_exit(&self, context: &ToolExecutionContext) -> ToolResult {
        let enabled = match self.state.lock() {
            Ok(guard) => guard.enabled,
            Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
        };
        if !enabled {
            return ToolResult::err(json!("plan mode is not active"));
        }

        let plan_path =
            match ensure_plan_path(&self.state, &context.session_id, &context.host).await {
                Ok(path) => path,
                Err(err) => return ToolResult::err(json!(err.to_string())),
            };
        let display = plan_display_path(&plan_path);
        let answer = match context
            .host
            .prompt_user(
                PromptRequest::single(
                    format!(
                        "Plan at {display} is ready. Start implementing it in a fresh execution turn?"
                    ),
                    vec!["Implement it".to_string(), "Keep planning".to_string()],
                )
                .with_optional_note(),
            )
            .await
        {
            Ok(answer) => answer,
            Err(err) => return ToolResult::err(json!(err.to_string())),
        };

        let (approved, note) = match &answer {
            PromptResponse::Single { selection, note } if selection == "Implement it" => {
                (true, note.clone())
            }
            _ => (false, None),
        };
        if !approved {
            return ToolResult::ok(json!({
                "approved": false,
                "plan_path": display,
                "answer": answer,
            }));
        }

        match self.state.lock() {
            Ok(mut guard) => {
                guard.set_enabled(false);
            }
            Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
        }

        ToolResult::ok(json!({
            "approved": true,
            "plan_path": display,
            "next_turn_input": plan_exit_next_turn_input(&display, note.as_deref()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::plan_exit_next_turn_input;

    #[test]
    fn plan_exit_next_turn_input_appends_user_note() {
        assert_eq!(
            plan_exit_next_turn_input(
                ".lash/plans/run-session.md",
                Some("start with the safe slice"),
            ),
            "The plan at `.lash/plans/run-session.md` is approved. Plan mode is off now. Execute that plan.\n\nUser note: start with the safe slice"
        );
        assert_eq!(
            plan_exit_next_turn_input(".lash/plans/run-session.md", Some("   ")),
            "The plan at `.lash/plans/run-session.md` is approved. Plan mode is off now. Execute that plan."
        );
    }
}

#[async_trait::async_trait]
impl ToolProvider for PlanModeTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "plan_exit".into(),
            description: "Finish planning, ask the user whether to start implementation, and if approved hand off to a fresh execution turn that uses the saved `.lash` plan file.".into(),
            params: Vec::new(),
            returns: "dict".into(),
            examples: vec!["plan_exit()".into()],
            enabled: false,
            injected: false,
            input_schema_override: None,
            output_schema_override: None,
        }]
    }

    async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{name}` requires session context and cannot run without it"
        ))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        _args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "plan_exit" => self.execute_plan_exit(context).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        _progress: Option<&crate::ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

pub struct PlanModePluginFactory {
    config: PlanModePluginConfig,
}

impl Default for PlanModePluginFactory {
    fn default() -> Self {
        Self::new(PlanModePluginConfig::default())
    }
}

impl PlanModePluginFactory {
    pub fn new(config: PlanModePluginConfig) -> Self {
        Self { config }
    }
}

impl PluginFactory for PlanModePluginFactory {
    fn id(&self) -> &'static str {
        "plan_mode"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(PlanModePlugin {
            state: Arc::new(Mutex::new(PlanModeState::default())),
            config: self.config.clone(),
        }))
    }
}

struct PlanModePlugin {
    state: Arc<Mutex<PlanModeState>>,
    config: PlanModePluginConfig,
}

impl SessionPlugin for PlanModePlugin {
    fn id(&self) -> &'static str {
        "plan_mode"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools().provider(Arc::new(PlanModeTools {
            state: Arc::clone(&self.state),
        }))?;

        let before_turn_state = Arc::clone(&self.state);
        reg.turn().before(Arc::new(move |ctx| {
            let state = Arc::clone(&before_turn_state);
            Box::pin(async move {
                let mut state = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
                let should_inject = state.prepare_turn();
                if !should_inject {
                    return Ok(Vec::new());
                }
                let plan_path = state.ensure_plan_path_from_state(&ctx.state)?;
                Ok(vec![
                    PluginDirective::emit_events(vec![state.badge_event()]),
                    PluginDirective::EnqueueMessages {
                        messages: vec![plan_mode_guidance_message(&plan_path)],
                    },
                ])
            })
        }));

        let checkpoint_state = Arc::clone(&self.state);
        reg.turn().checkpoint(Arc::new(move |ctx| {
            let state = Arc::clone(&checkpoint_state);
            Box::pin(async move {
                let mut state = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
                let should_inject = state.checkpoint_injection_needed();
                if !should_inject {
                    return Ok(Vec::new());
                }
                let plan_path = state.ensure_plan_path_from_state(&ctx.state)?;
                Ok(vec![
                    PluginDirective::emit_events(vec![state.badge_event()]),
                    PluginDirective::EnqueueMessages {
                        messages: vec![plan_mode_guidance_message(&plan_path)],
                    },
                ])
            })
        }));

        let after_turn_state = Arc::clone(&self.state);
        reg.turn().after(Arc::new(move |_ctx| {
            let state = Arc::clone(&after_turn_state);
            Box::pin(async move {
                state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                    .finish_turn();
                Ok(Vec::new())
            })
        }));

        let before_tool_state = Arc::clone(&self.state);
        let before_tool_config = self.config.clone();
        reg.tool_calls().before(Arc::new(move |ctx| {
            let state = Arc::clone(&before_tool_state);
            let config = before_tool_config.clone();
            Box::pin(async move {
                let enabled = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                    .enabled;
                if !enabled {
                    return Ok(Vec::new());
                }

                if ctx.tool_name == "apply_patch" && !config.blocked_tools.contains("apply_patch") {
                    let plan_path = ensure_plan_path(&state, &ctx.session_id, &ctx.host).await?;
                    if let Err(message) = patch_allowed_for_plan_file(&ctx.args, &plan_path) {
                        return Ok(vec![PluginDirective::AbortTurn {
                            code: "plan_mode_tool_blocked".to_string(),
                            message,
                        }]);
                    }
                    return Ok(Vec::new());
                }

                if !config.blocked_tools.contains(&ctx.tool_name) {
                    return Ok(Vec::new());
                }

                Ok(vec![PluginDirective::AbortTurn {
                    code: "plan_mode_tool_blocked".to_string(),
                    message: format!(
                        "Plan mode blocks `{}`. Disable plan mode to execute implementation tools.",
                        ctx.tool_name
                    ),
                }])
            })
        }));

        let after_tool_state = Arc::clone(&self.state);
        reg.tool_calls().after(Arc::new(move |ctx| {
            let state = Arc::clone(&after_tool_state);
            Box::pin(async move {
                let approved = ctx.tool_name == "plan_exit"
                    && ctx.result.success
                    && ctx
                        .result
                        .result
                        .get("approved")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false);
                if !approved {
                    return Ok(Vec::new());
                }
                let clear = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                    .clear_badge_event();
                Ok(vec![PluginDirective::emit_events(vec![clear])])
            })
        }));

        let tool_surface_state = Arc::clone(&self.state);
        let tool_surface_config = self.config.clone();
        reg.surface().contribute(Arc::new(move |_ctx| {
            let state = tool_surface_state
                .lock()
                .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
            if !state.enabled {
                return Ok(ToolSurfaceContribution::default());
            }

            let mut overrides = tool_surface_config
                .blocked_tools
                .iter()
                .map(|tool_name| ToolSurfaceOverride {
                    tool_name: tool_name.clone(),
                    enabled: Some(false),
                    injected: Some(false),
                })
                .collect::<Vec<_>>();
            overrides.push(ToolSurfaceOverride {
                tool_name: "plan_exit".to_string(),
                enabled: Some(true),
                injected: Some(true),
            });

            Ok(ToolSurfaceContribution {
                overrides,
                tool_list_notes: vec![plan_mode_tool_note(state.plan_path().as_deref())],
            })
        }));

        let status_state = Arc::clone(&self.state);
        reg.external().op(
            ExternalOpDef {
                name: "plan_mode.status".to_string(),
                description: "Read the current plan-mode state for this session.".to_string(),
                kind: ExternalOpKind::Query,
                session_param: SessionParam::Required,
                input_schema: json!({
                    "type": "object",
                    "additionalProperties": false
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string" },
                        "enabled": { "type": "boolean" },
                        "plan_path": { "type": ["string", "null"] }
                    },
                    "required": ["session_id", "enabled", "plan_path"],
                    "additionalProperties": false
                }),
            },
            Arc::new(move |ctx, _args| {
                let state = Arc::clone(&status_state);
                Box::pin(async move {
                    let Some(session_id) = ctx.session_id else {
                        return ToolResult::err(json!("plan_mode.status requires session_id"));
                    };
                    let snapshot = match state.lock() {
                        Ok(guard) => guard.snapshot(),
                        Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
                    };
                    ToolResult::ok(json!({
                        "session_id": session_id,
                        "enabled": snapshot.enabled,
                        "plan_path": snapshot.plan_path,
                    }))
                })
            }),
        )?;

        for (name, description, kind) in [
            (
                "plan_mode.enable",
                "Enable plan mode for this session.",
                ExternalOpKind::Command,
            ),
            (
                "plan_mode.disable",
                "Disable plan mode for this session.",
                ExternalOpKind::Command,
            ),
            (
                "plan_mode.toggle",
                "Toggle plan mode for this session.",
                ExternalOpKind::Command,
            ),
        ] {
            let state = Arc::clone(&self.state);
            reg.external().op(
                ExternalOpDef {
                    name: name.to_string(),
                    description: description.to_string(),
                    kind,
                    session_param: SessionParam::Required,
                    input_schema: json!({
                        "type": "object",
                        "additionalProperties": false
                    }),
                    output_schema: json!({
                        "type": "object",
                        "properties": {
                            "session_id": { "type": "string" },
                            "enabled": { "type": "boolean" },
                            "plan_path": { "type": ["string", "null"] }
                        },
                        "required": ["session_id", "enabled", "plan_path"],
                        "additionalProperties": false
                    }),
                },
                Arc::new(move |ctx, _args| {
                    let state = Arc::clone(&state);
                    let op_name = name.to_string();
                    Box::pin(async move {
                        let Some(session_id) = ctx.session_id else {
                            return ToolResult::err(json!(format!(
                                "{op_name} requires session_id"
                            )));
                        };
                        let snapshot = match state.lock() {
                            Ok(mut guard) => match op_name.as_str() {
                                "plan_mode.enable" => guard.set_enabled(true),
                                "plan_mode.disable" => guard.set_enabled(false),
                                "plan_mode.toggle" => guard.toggle(),
                                _ => unreachable!(),
                            },
                            Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
                        };
                        ToolResult::ok(json!({
                            "session_id": session_id,
                            "enabled": snapshot.enabled,
                            "plan_path": snapshot.plan_path,
                        }))
                    })
                }),
            )?;
        }

        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let snapshot = self
            .state
            .lock()
            .map_err(|_| PluginError::Snapshot("plan mode state poisoned".to_string()))?
            .snapshot();
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: Some(json!({
                "enabled": snapshot.enabled,
                "generation": snapshot.generation,
                "plan_path": snapshot.plan_path,
            })),
        })
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let snapshot = meta
            .state
            .clone()
            .map(serde_json::from_value::<PlanModeSnapshot>)
            .transpose()
            .map_err(|err| PluginError::Snapshot(err.to_string()))?
            .unwrap_or_default();
        self.state
            .lock()
            .map_err(|_| PluginError::Snapshot("plan mode state poisoned".to_string()))?
            .restore_snapshot(snapshot);
        Ok(())
    }
}
