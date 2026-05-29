use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::json;

use lash_core::plugin::{
    PluginAction, PluginActionFailure, PluginActionKind, PluginDirective, PluginError,
    PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta, SessionParam,
    SessionPlugin, SnapshotReader, SnapshotWriter, ToolSurfaceContribution, ToolSurfaceOverride,
};
use lash_core::{
    JsonSchema, PluginMessage, ToolCall, ToolContext, ToolControl, ToolDefinition, ToolResult,
    ToolScheduling,
};
use lash_tool_apply_patch::{PatchAction, inspect_patch_ops};
use lash_tool_support::{StaticToolExecute, StaticToolProvider};

mod prompt;
mod state;

pub use prompt::{
    PlanModePrompt, PlanModePromptRequest, PlanModePromptResponse, PlanModePromptReview,
};
use prompt::{
    plan_exit_confirmation_display, plan_exit_fresh_context_input, plan_exit_next_turn_input,
    plan_mode_guidance_message, plan_mode_tool_note,
};
#[cfg(test)]
use state::PLAN_TEMPLATE;
use state::{
    PlanModeSnapshot, PlanModeState, PlanReport, effective_run_session_id, plan_display_path,
    read_plan_report, resolve_plan_path, seed_plan_template,
};

const PLAN_MODE_STATE_EVENT: &str = "plan_mode.state";

fn default_allowed_tools() -> BTreeSet<String> {
    [
        "ask",
        "fetch_url",
        "glob",
        "grep",
        "ls",
        "read_file",
        "search_tools",
        "search_web",
        "apply_patch",
        "plan_exit",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn fresh_context_frame_id() -> String {
    format!("plan-frame-{}", uuid::Uuid::new_v4().simple())
}

fn plan_protocol_state_event(
    session_id: &str,
    enabled: bool,
    report: Option<&PlanReport>,
) -> Result<lash_core::PluginRuntimeEvent, PluginError> {
    Ok(lash_core::PluginRuntimeEvent::Custom {
        name: PLAN_MODE_STATE_EVENT.to_string(),
        payload: serde_json::to_value(plan_mode_payload(session_id, enabled, report)).map_err(
            |err| PluginError::Session(format!("failed to encode plan mode state: {err}")),
        )?,
    })
}

#[derive(Clone, Debug)]
pub struct PlanModePluginConfig {
    pub allowed_tools: BTreeSet<String>,
}

impl Default for PlanModePluginConfig {
    fn default() -> Self {
        Self {
            allowed_tools: default_allowed_tools(),
        }
    }
}

impl PlanModePluginConfig {
    pub fn with_allowed_tools<I, S>(mut self, allowed_tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools = allowed_tools.into_iter().map(Into::into).collect();
        self.allowed_tools.insert("plan_exit".to_string());
        self
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct PlanModeExternalArgs {}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct PlanModeExternalStatus {
    pub session_id: String,
    pub enabled: bool,
    pub plan_path: Option<String>,
}

pub struct PlanModeEnableOp;
pub struct PlanModeDisableOp;
pub struct PlanModeToggleOp;

impl PluginAction for PlanModeEnableOp {
    const NAME: &'static str = "plan_mode.enable";
    const DESCRIPTION: &'static str = "Enable plan mode for this session.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = PlanModeExternalArgs;
    type Output = PlanModeExternalStatus;
}

impl PluginAction for PlanModeDisableOp {
    const NAME: &'static str = "plan_mode.disable";
    const DESCRIPTION: &'static str = "Disable plan mode for this session.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = PlanModeExternalArgs;
    type Output = PlanModeExternalStatus;
}

impl PluginAction for PlanModeToggleOp {
    const NAME: &'static str = "plan_mode.toggle";
    const DESCRIPTION: &'static str = "Toggle plan mode for this session.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = PlanModeExternalArgs;
    type Output = PlanModeExternalStatus;
}

async fn ensure_plan_path<H>(
    state: &Arc<Mutex<PlanModeState>>,
    session_id: &str,
    host: &Arc<H>,
) -> Result<PathBuf, PluginError>
where
    H: lash_core::plugin::runtime_host::RuntimeSessionHost + ?Sized,
{
    if let Some(path) = state
        .lock()
        .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
        .plan_path()
    {
        return Ok(path);
    }

    let snapshot = host.snapshot_session(session_id).await?;
    let run_session_id =
        effective_run_session_id(&snapshot.session_id, &snapshot.policy).to_string();
    let path = resolve_plan_path(&run_session_id).map_err(PluginError::Session)?;
    state
        .lock()
        .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
        .set_plan_path(path.clone());
    Ok(path)
}

fn ensure_plan_path_from_snapshot(
    state: &Arc<Mutex<PlanModeState>>,
    snapshot: &lash_core::SessionSnapshot,
) -> Result<PathBuf, PluginError> {
    if let Some(path) = state
        .lock()
        .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
        .plan_path()
    {
        return Ok(path);
    }
    let run_session_id =
        effective_run_session_id(&snapshot.session_id, &snapshot.policy).to_string();
    let path = resolve_plan_path(&run_session_id).map_err(PluginError::Session)?;
    state
        .lock()
        .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
        .set_plan_path(path.clone());
    Ok(path)
}

async fn ensure_plan_report_for_tool_context(
    state: &Arc<Mutex<PlanModeState>>,
    context: &ToolContext<'_>,
    seed_if_missing: bool,
) -> Result<PlanReport, PluginError> {
    let snapshot = context.sessions().snapshot_current().await?;
    let path = ensure_plan_path_from_snapshot(state, &snapshot)?;
    if seed_if_missing {
        seed_plan_template(&path).map_err(PluginError::Session)?;
    }
    read_plan_report(&path).map_err(PluginError::Session)
}

async fn ensure_plan_report<H>(
    state: &Arc<Mutex<PlanModeState>>,
    session_id: &str,
    host: &Arc<H>,
    seed_if_missing: bool,
) -> Result<PlanReport, PluginError>
where
    H: lash_core::plugin::runtime_host::RuntimeSessionHost + ?Sized,
{
    let path = ensure_plan_path(state, session_id, host).await?;
    if seed_if_missing {
        seed_plan_template(&path).map_err(PluginError::Session)?;
    }
    read_plan_report(&path).map_err(PluginError::Session)
}

fn tool_state_unavailable(err: &PluginError) -> bool {
    matches!(err, PluginError::Session(message) if message.contains("tool state"))
}

async fn sync_plan_exit_tool_state<H>(
    host: &Arc<H>,
    session_id: &str,
    enabled: bool,
) -> Result<(), PluginError>
where
    H: lash_core::plugin::runtime_host::RuntimeSessionHost + ?Sized,
{
    let availability = if enabled {
        Some(lash_core::ToolAvailability::Showcased)
    } else {
        Some(lash_core::ToolAvailability::Off)
    };
    match host
        .set_tool_availability(session_id, "plan_exit", availability)
        .await
    {
        Ok(_) => Ok(()),
        Err(err) if tool_state_unavailable(&err) => Ok(()),
        Err(err) => Err(err),
    }
}

async fn set_plan_mode_enabled_state<H>(
    state: &Arc<Mutex<PlanModeState>>,
    session_id: &str,
    host: &Arc<H>,
    enabled: bool,
) -> Result<bool, PluginError>
where
    H: lash_core::plugin::runtime_host::RuntimeSessionHost + ?Sized,
{
    let previous = {
        let mut guard = state
            .lock()
            .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
        let previous = guard.enabled;
        if previous != enabled {
            guard.set_enabled(enabled);
        }
        previous
    };
    if let Err(err) = sync_plan_exit_tool_state(host, session_id, enabled).await {
        if previous != enabled {
            let mut guard = state
                .lock()
                .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
            guard.set_enabled(previous);
        }
        return Err(err);
    }
    Ok(enabled)
}

async fn set_plan_mode_enabled_state_for_tool_context(
    state: &Arc<Mutex<PlanModeState>>,
    context: &ToolContext<'_>,
    enabled: bool,
) -> Result<bool, PluginError> {
    let previous = {
        let mut guard = state
            .lock()
            .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
        let previous = guard.enabled;
        if previous != enabled {
            guard.set_enabled(enabled);
        }
        previous
    };
    let availability = if enabled {
        Some(lash_core::ToolAvailability::Showcased)
    } else {
        Some(lash_core::ToolAvailability::Off)
    };
    let names = vec!["plan_exit".to_string()];
    if let Err(err) = context
        .sessions()
        .set_tools_availability(&names, availability)
        .await
        && !tool_state_unavailable(&err)
    {
        if previous != enabled {
            let mut guard = state
                .lock()
                .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
            guard.set_enabled(previous);
        }
        return Err(err);
    }
    Ok(enabled)
}

fn plan_mode_payload(
    session_id: &str,
    enabled: bool,
    report: Option<&PlanReport>,
) -> PlanModeExternalStatus {
    PlanModeExternalStatus {
        session_id: session_id.to_string(),
        enabled,
        plan_path: report.map(|value| value.display_path.clone()),
    }
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
    prompt: Option<Arc<dyn PlanModePrompt>>,
}

impl PlanModeTools {
    async fn execute_plan_exit(&self, context: &ToolContext<'_>) -> ToolResult {
        let enabled = match self.state.lock() {
            Ok(guard) => guard.enabled,
            Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
        };
        if !enabled {
            return ToolResult::err(json!("plan mode is not active"));
        }

        let report = match ensure_plan_report_for_tool_context(&self.state, context, true).await {
            Ok(report) => report,
            Err(err) => return ToolResult::err(json!(err.to_string())),
        };

        let Some(prompt) = &self.prompt else {
            return ToolResult::err(json!(
                "plan approval prompts are unavailable in this session"
            ));
        };
        let answer = match prompt
            .prompt_user(
                PlanModePromptRequest::single(
                    format!("Review the plan in `{}`. What next?", report.display_path),
                    vec![
                        "Start implementing now".to_string(),
                        "Keep planning".to_string(),
                        "Start in fresh context".to_string(),
                    ],
                )
                .with_review("PLAN", report.approval_content())
                .with_optional_note(),
            )
            .await
        {
            Ok(answer) => answer,
            Err(err) => return ToolResult::err(json!(err.to_string())),
        };

        let selection = match &answer {
            PlanModePromptResponse::Single { selection, .. } => selection.as_str(),
        };
        if selection == "Keep planning" {
            return ToolResult::ok(json!({
                "approved": false,
                "plan_path": report.display_path,
                "answer": answer,
            }));
        }

        let note = match &answer {
            PlanModePromptResponse::Single { note, .. } => note.clone(),
        };

        if let Err(err) =
            set_plan_mode_enabled_state_for_tool_context(&self.state, context, false).await
        {
            return ToolResult::err(json!(err.to_string()));
        }

        if selection == "Start in fresh context" {
            return ToolResult::ok(json!({
                "approved": true,
                "plan_path": report.display_path,
                "execution_mode": "fresh_context",
            }));
        }

        ToolResult::ok(json!({
            "approved": true,
            "answer": answer,
            "confirmation_display": plan_exit_confirmation_display(selection, note.as_deref()),
            "plan_path": report.display_path,
            "execution_mode": "current_session",
            "next_turn_input": plan_exit_next_turn_input(&report.display_path, note.as_deref()),
        }))
    }
}

fn plan_mode_provider(
    state: Arc<Mutex<PlanModeState>>,
    prompt: Option<Arc<dyn PlanModePrompt>>,
) -> StaticToolProvider<PlanModeTools> {
    StaticToolProvider::new(
        vec![plan_exit_tool_definition()],
        PlanModeTools { state, prompt },
    )
}

#[async_trait::async_trait]
impl StaticToolExecute for PlanModeTools {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "plan_exit" => self.execute_plan_exit(call.context).await,
            other => ToolResult::err_fmt(format_args!("Unknown tool: {other}")),
        }
    }
}

fn plan_exit_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:plan_exit",
        "plan_exit",
        "Ask whether to exit plan mode.",
        ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec!["plan_exit()".into()])
    .with_availability(lash_core::ToolAvailabilityConfig::off())
    .with_scheduling(ToolScheduling::Parallel)
}

pub struct PlanModePluginFactory {
    config: PlanModePluginConfig,
    prompt: Option<Arc<dyn PlanModePrompt>>,
}

impl Default for PlanModePluginFactory {
    fn default() -> Self {
        Self::new(PlanModePluginConfig::default())
    }
}

impl PlanModePluginFactory {
    pub fn new(config: PlanModePluginConfig) -> Self {
        Self {
            config,
            prompt: None,
        }
    }

    pub fn with_prompt(mut self, prompt: Arc<dyn PlanModePrompt>) -> Self {
        self.prompt = Some(prompt);
        self
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
            prompt: self.prompt.clone(),
        }))
    }
}

struct PlanModePlugin {
    state: Arc<Mutex<PlanModeState>>,
    config: PlanModePluginConfig,
    prompt: Option<Arc<dyn PlanModePrompt>>,
}

impl SessionPlugin for PlanModePlugin {
    fn id(&self) -> &'static str {
        "plan_mode"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools().provider(Arc::new(plan_mode_provider(
            Arc::clone(&self.state),
            self.prompt.clone(),
        )))?;

        let before_turn_state = Arc::clone(&self.state);
        reg.turn().before(Arc::new(move |ctx| {
            let state = Arc::clone(&before_turn_state);
            Box::pin(async move {
                let plan_path = {
                    let mut state = state.lock().map_err(|_| {
                        PluginError::Session("plan mode state poisoned".to_string())
                    })?;
                    let should_inject = state.prepare_turn();
                    if !should_inject {
                        return Ok(Vec::new());
                    }
                    state.ensure_plan_path_from_state(&ctx.state.to_owned_state())?
                };
                seed_plan_template(&plan_path).map_err(PluginError::Session)?;
                let report = read_plan_report(&plan_path).map_err(PluginError::Session)?;
                Ok(vec![
                    PluginDirective::emit_runtime_events(vec![plan_protocol_state_event(
                        &ctx.session_id,
                        true,
                        Some(&report),
                    )?]),
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
                let plan_path = {
                    let mut state = state.lock().map_err(|_| {
                        PluginError::Session("plan mode state poisoned".to_string())
                    })?;
                    let should_inject = state.checkpoint_injection_needed();
                    if !should_inject {
                        return Ok(Vec::new());
                    }
                    state.ensure_plan_path_from_state(&ctx.state.to_owned_state())?
                };
                seed_plan_template(&plan_path).map_err(PluginError::Session)?;
                let report = read_plan_report(&plan_path).map_err(PluginError::Session)?;
                Ok(vec![
                    PluginDirective::emit_runtime_events(vec![plan_protocol_state_event(
                        &ctx.session_id,
                        true,
                        Some(&report),
                    )?]),
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

                if ctx.tool_name != "plan_exit" && !config.allowed_tools.contains(&ctx.tool_name) {
                    return Ok(vec![PluginDirective::AbortTurn {
                        code: "plan_mode_tool_blocked".to_string(),
                        message: format!(
                            "Plan mode blocks `{}`. Use planning tools or `plan_exit()`.",
                            ctx.tool_name
                        ),
                    }]);
                }

                if ctx.tool_name == "apply_patch" {
                    let snapshot = ctx.session_snapshot().await?;
                    let plan_path = ensure_plan_path_from_snapshot(&state, &snapshot)?;
                    if let Err(message) = patch_allowed_for_plan_file(&ctx.args, &plan_path) {
                        return Ok(vec![PluginDirective::AbortTurn {
                            code: "plan_mode_tool_blocked".to_string(),
                            message,
                        }]);
                    }
                }

                Ok(Vec::new())
            })
        }));

        let after_tool_state = Arc::clone(&self.state);
        reg.tool_calls().after(Arc::new(move |ctx| {
            let state = Arc::clone(&after_tool_state);
            Box::pin(async move {
                let result_value = ctx.result.value_for_projection();
                let approved = ctx.tool_name == "plan_exit"
                    && ctx.result.is_success()
                    && result_value
                        .get("approved")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false);
                if approved {
                    let mut directives = vec![PluginDirective::emit_runtime_events(vec![
                        plan_protocol_state_event(&ctx.session_id, false, None)?,
                    ])];
                    if result_value
                        .get("execution_mode")
                        .and_then(|value| value.as_str())
                        == Some("fresh_context")
                    {
                        let plan_path = result_value
                            .get("plan_path")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let frame_id = fresh_context_frame_id();
                        let task = plan_exit_fresh_context_input(&plan_path);
                        directives.push(PluginDirective::short_circuit(
                            ToolResult::ok(json!({
                                "approved": true,
                                "plan_path": plan_path,
                                "execution_mode": "fresh_context",
                                "frame_id": frame_id.clone(),
                            }))
                            .with_control(
                                ToolControl::SwitchAgentFrame {
                                    frame_id,
                                    initial_nodes: Vec::new(),
                                    task: Some(task),
                                },
                            ),
                        ));
                    }
                    return Ok(directives);
                }

                let enabled = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                    .enabled;
                if !enabled || ctx.tool_name != "apply_patch" || !ctx.result.is_success() {
                    return Ok(Vec::new());
                }

                let snapshot = ctx.session_snapshot().await?;
                let path = ensure_plan_path_from_snapshot(&state, &snapshot)?;
                let report = read_plan_report(&path).map_err(PluginError::Session)?;
                Ok(vec![PluginDirective::emit_runtime_events(vec![
                    plan_protocol_state_event(&ctx.session_id, true, Some(&report))?,
                ])])
            })
        }));

        let tool_surface_state = Arc::clone(&self.state);
        let tool_surface_config = self.config.clone();
        reg.surface().contribute(Arc::new(move |ctx| {
            let state = tool_surface_state
                .lock()
                .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?;
            if !state.enabled {
                return Ok(ToolSurfaceContribution::default());
            }

            let mut overrides = ctx
                .tools
                .iter()
                .filter(|tool| !tool_surface_config.allowed_tools.contains(&tool.name))
                .map(|tool| ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(lash_core::ToolAvailability::Off),
                })
                .collect::<Vec<_>>();
            overrides.push(ToolSurfaceOverride {
                tool_name: "plan_exit".to_string(),
                availability: Some(lash_core::ToolAvailability::Showcased),
            });
            if tool_surface_config.allowed_tools.contains("apply_patch") {
                overrides.push(ToolSurfaceOverride {
                    tool_name: "apply_patch".to_string(),
                    availability: Some(lash_core::ToolAvailability::Showcased),
                });
            }

            Ok(ToolSurfaceContribution {
                overrides,
                tool_list_notes: vec![plan_mode_tool_note(state.plan_path().as_deref())],
            })
        }));

        register_plan_mode_op::<PlanModeEnableOp>(reg, Arc::clone(&self.state))?;
        register_plan_mode_op::<PlanModeDisableOp>(reg, Arc::clone(&self.state))?;
        register_plan_mode_op::<PlanModeToggleOp>(reg, Arc::clone(&self.state))?;

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
            revision: snapshot.generation,
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

    fn snapshot_revision(&self) -> u64 {
        self.state
            .lock()
            .map(|state| state.generation)
            .unwrap_or_default()
    }
}

fn register_plan_mode_op<Op>(
    reg: &mut PluginRegistrar,
    state: Arc<Mutex<PlanModeState>>,
) -> Result<(), PluginError>
where
    Op: PluginAction<Args = PlanModeExternalArgs, Output = PlanModeExternalStatus>,
{
    reg.actions().typed::<Op, _, _>(move |ctx, _args| {
        let state = Arc::clone(&state);
        async move {
            let Some(session_id) = ctx.session_id else {
                return Err(PluginActionFailure::new(format!(
                    "{} requires session_id",
                    Op::NAME
                )));
            };
            let target_enabled = match state.lock() {
                Ok(guard) => match Op::NAME {
                    "plan_mode.enable" => true,
                    "plan_mode.disable" => false,
                    "plan_mode.toggle" => !guard.enabled,
                    _ => unreachable!(),
                },
                Err(_) => return Err(PluginActionFailure::new("plan mode state poisoned")),
            };
            let enabled =
                set_plan_mode_enabled_state(&state, &session_id, &ctx.host, target_enabled).await?;
            let report = ensure_plan_report(&state, &session_id, &ctx.host, enabled).await?;
            Ok(plan_mode_payload(&session_id, enabled, Some(&report)))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        PLAN_TEMPLATE, plan_exit_fresh_context_input, plan_exit_next_turn_input, read_plan_report,
    };

    #[test]
    fn plan_exit_next_turn_input_appends_user_note() {
        assert_eq!(
            plan_exit_next_turn_input(
                ".lash/plans/run-session.md",
                Some("start with the safe slice"),
            ),
            "The user approved the plan. Execute the plan in `.lash/plans/run-session.md` now — start immediately, do not ask for confirmation.\n\nUser note: start with the safe slice"
        );
        assert_eq!(
            plan_exit_next_turn_input(".lash/plans/run-session.md", Some("   ")),
            "The user approved the plan. Execute the plan in `.lash/plans/run-session.md` now — start immediately, do not ask for confirmation."
        );
    }

    #[test]
    fn plan_exit_fresh_context_input_is_short_and_direct() {
        assert_eq!(
            plan_exit_fresh_context_input(".lash/plans/run-session.md"),
            "Do a full, faithful implementation of the plan found at: .lash/plans/run-session.md"
        );
    }

    #[test]
    fn seeded_template_is_readable_without_validation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("plan.md");
        std::fs::write(&path, PLAN_TEMPLATE).expect("write template");
        let report = read_plan_report(&path).expect("report");
        assert_eq!(report.content.as_deref(), Some(PLAN_TEMPLATE));
    }
}
