use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::json;

use lash::plugin::{
    ExternalOpDef, ExternalOpKind, PluginDirective, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, PluginSnapshotMeta, SessionParam, SessionPlugin, SnapshotReader,
    SnapshotWriter, ToolSurfaceContribution, ToolSurfaceOverride,
};
use lash::tools::{PatchAction, inspect_patch_ops};
use lash::{
    PluginMessage, PromptRequest, PromptResponse, SessionContextSurface, SessionCreateRequest,
    SessionPluginMode, SessionStartPoint, ToolDefinition, ToolExecutionContext, ToolExecutionMode,
    ToolProvider, ToolResult,
};

const PLAN_MODE_BADGE_KEY: &str = "mode";
const PLAN_MODE_BADGE_LABEL: &str = "plan";
const PLAN_MODE_PANEL_KEY: &str = "panel";
const PLAN_MODE_PANEL_TITLE: &str = "PLAN";
const PLAN_TEMPLATE: &str = r#"# Plan

## Goal
- TBD

## Steps
- TBD

## Files
- TBD

## Risks
- TBD

## Verification
- TBD
"#;

fn default_allowed_tools() -> BTreeSet<String> {
    [
        "ask",
        "fetch_url",
        "glob",
        "grep",
        "ls",
        "read_file",
        "discover_tools",
        "load_tools",
        "search_web",
        "apply_patch",
        "plan_exit",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

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
            "The user approved the plan. Execute the plan in `{display}` now — start immediately, do not ask for confirmation.\n\nUser note: {note}"
        )
    } else {
        format!(
            "The user approved the plan. Execute the plan in `{display}` now — start immediately, do not ask for confirmation."
        )
    }
}

fn plan_exit_fresh_context_input(display: &str) -> String {
    format!("Do a full, faithful implementation of the plan found at: {display}")
}

fn plan_exit_confirmation_display(selection: &str, note: Option<&str>) -> String {
    if let Some(note) = note.filter(|note| !note.trim().is_empty()) {
        format!("{selection}\n\nNote: {note}")
    } else {
        selection.to_string()
    }
}

fn fresh_context_session_id() -> String {
    format!("plan-{}", uuid::Uuid::new_v4().simple())
}

fn resolve_plan_path(run_session_id: &str) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|err| format!("Failed to determine cwd: {err}"))?;
    Ok(cwd
        .join(".lash")
        .join("plans")
        .join(format!("{run_session_id}.md")))
}

fn effective_run_session_id<'a>(session_id: &'a str, policy: &'a lash::SessionPolicy) -> &'a str {
    policy.session_id.as_deref().unwrap_or(session_id)
}

fn seed_plan_template(path: &Path) -> Result<bool, String> {
    if path.is_file() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create plan directory `{}`: {err}",
                plan_display_path(parent)
            )
        })?;
    }
    fs::write(path, PLAN_TEMPLATE).map_err(|err| {
        format!(
            "Failed to seed plan template `{}`: {err}",
            plan_display_path(path)
        )
    })?;
    Ok(true)
}

#[derive(Clone, Debug, Default)]
struct PlanReport {
    display_path: String,
    content: Option<String>,
}

impl PlanReport {
    fn preview_content(&self) -> String {
        format!("Path: `{}`", self.display_path)
    }

    fn approval_content(&self) -> String {
        self.content
            .as_deref()
            .map(str::trim_end)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.preview_content())
    }
}

fn read_plan_report(path: &Path) -> Result<PlanReport, String> {
    let display_path = plan_display_path(path);
    if !path.is_file() {
        return Ok(PlanReport {
            display_path,
            ..Default::default()
        });
    }

    let content = fs::read_to_string(path).map_err(|err| {
        format!(
            "Failed to read plan file `{}`: {err}",
            plan_display_path(path)
        )
    })?;
    Ok(PlanReport {
        display_path,
        content: Some(content),
    })
}

fn plan_panel_event(report: &PlanReport) -> lash::plugin::PluginSurfaceEvent {
    lash::plugin::PluginSurfaceEvent::PanelUpsert {
        key: PLAN_MODE_PANEL_KEY.to_string(),
        title: PLAN_MODE_PANEL_TITLE.to_string(),
        content: report.preview_content(),
    }
}

fn clear_plan_panel_event() -> lash::plugin::PluginSurfaceEvent {
    lash::plugin::PluginSurfaceEvent::PanelClear {
        key: PLAN_MODE_PANEL_KEY.to_string(),
    }
}

fn plan_mode_guidance_message(plan_path: &Path) -> PluginMessage {
    let display = plan_display_path(plan_path);
    PluginMessage::text(
        lash::MessageRole::System,
        format!(
            "Plan mode: use `{display}` as the single source of truth. Read/search/list, web, and `ask(...)` as needed, and update only that file with `apply_patch`. Do not present the plan with snippets, showcases, or prose checklists; the panel only shows the file path while planning. When the plan is ready for review, call `plan_exit()`."
        ),
    )
}

fn plan_mode_tool_note(plan_path: Option<&Path>) -> String {
    match plan_path {
        Some(path) => format!(
            "Plan mode tools: read/search/list, web search/fetch, `ask`, `apply_patch` for `{}`, `plan_exit()`. The panel shows the plan file path; full review happens in `plan_exit()`.",
            plan_display_path(path)
        ),
        None => "Plan mode tools: read/search/list, web search/fetch, `ask`, plan-file `apply_patch`, `plan_exit()`. The panel shows the plan file path; full review happens in `plan_exit()`.".to_string(),
    }
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

    fn prepare_turn(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        if self.active_turn_applied_generation == Some(self.generation) {
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

    fn finish_turn(&mut self) {}

    fn badge_event(&self) -> lash::plugin::PluginSurfaceEvent {
        lash::plugin::PluginSurfaceEvent::ModeIndicatorUpsert {
            key: PLAN_MODE_BADGE_KEY.to_string(),
            label: PLAN_MODE_BADGE_LABEL.to_string(),
        }
    }

    fn clear_badge_event(&self) -> lash::plugin::PluginSurfaceEvent {
        lash::plugin::PluginSurfaceEvent::ModeIndicatorClear {
            key: PLAN_MODE_BADGE_KEY.to_string(),
        }
    }

    fn plan_path(&self) -> Option<PathBuf> {
        self.plan_path.clone()
    }

    fn ensure_plan_path_from_state(
        &mut self,
        state: &lash::SessionStateEnvelope,
    ) -> Result<PathBuf, PluginError> {
        if let Some(path) = self.plan_path() {
            return Ok(path);
        }
        let path = resolve_plan_path(effective_run_session_id(&state.session_id, &state.policy))
            .map_err(PluginError::Session)?;
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
    host: &Arc<dyn lash::SessionManager>,
) -> Result<PathBuf, PluginError> {
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

async fn ensure_plan_report(
    state: &Arc<Mutex<PlanModeState>>,
    session_id: &str,
    host: &Arc<dyn lash::SessionManager>,
    seed_if_missing: bool,
) -> Result<PlanReport, PluginError> {
    let path = ensure_plan_path(state, session_id, host).await?;
    if seed_if_missing {
        seed_plan_template(&path).map_err(PluginError::Session)?;
    }
    read_plan_report(&path).map_err(PluginError::Session)
}

fn dynamic_tool_state_unavailable(err: &PluginError) -> bool {
    matches!(err, PluginError::Session(message) if message.contains("dynamic tool state"))
}

async fn sync_plan_exit_tool_state(
    host: &Arc<dyn lash::SessionManager>,
    session_id: &str,
    enabled: bool,
) -> Result<(), PluginError> {
    let availability = if enabled {
        Some(lash::ToolAvailability::Documented)
    } else {
        Some(lash::ToolAvailability::Hidden)
    };
    match host
        .set_tool_availability(session_id, "plan_exit", availability)
        .await
    {
        Ok(_) => Ok(()),
        Err(err) if dynamic_tool_state_unavailable(&err) => Ok(()),
        Err(err) => Err(err),
    }
}

async fn set_plan_mode_enabled_state(
    state: &Arc<Mutex<PlanModeState>>,
    session_id: &str,
    host: &Arc<dyn lash::SessionManager>,
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

fn plan_status_payload(
    session_id: &str,
    enabled: bool,
    report: Option<&PlanReport>,
) -> serde_json::Value {
    json!({
        "session_id": session_id,
        "enabled": enabled,
        "plan_path": report.map(|value| value.display_path.clone()),
        "panel_title": enabled.then_some(PLAN_MODE_PANEL_TITLE.to_string()),
        "panel_content": if enabled {
            report.map(|value| value.preview_content())
        } else {
            None
        },
    })
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

        let report =
            match ensure_plan_report(&self.state, &context.session_id, &context.host, true).await {
                Ok(report) => report,
                Err(err) => return ToolResult::err(json!(err.to_string())),
            };

        let answer = match context
            .host
            .prompt_user(
                PromptRequest::single(
                    format!("Review the plan in `{}`. What next?", report.display_path),
                    vec![
                        "Start implementing now".to_string(),
                        "Keep planning".to_string(),
                        "Start in fresh context".to_string(),
                    ],
                )
                .with_markdown_panel("PLAN", report.approval_content())
                .with_optional_note(),
            )
            .await
        {
            Ok(answer) => answer,
            Err(err) => return ToolResult::err(json!(err.to_string())),
        };

        let selection = match &answer {
            PromptResponse::Single { selection, .. } => selection.as_str(),
            _ => "Keep planning",
        };
        if selection == "Keep planning" {
            return ToolResult::ok(json!({
                "approved": false,
                "plan_path": report.display_path,
                "answer": answer,
            }));
        }

        let note = match &answer {
            PromptResponse::Single { note, .. } => note.clone(),
            _ => None,
        };

        if let Err(err) =
            set_plan_mode_enabled_state(&self.state, &context.session_id, &context.host, false)
                .await
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

#[async_trait::async_trait]
impl ToolProvider for PlanModeTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "plan_exit".into(),
            description: "Ask whether to exit plan mode.".into(),
            params: Vec::new(),
            returns: "dict".into(),
            examples: vec!["plan_exit()".into()],
            availability: lash::ToolAvailabilityConfig::hidden(),
            activation: lash::ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
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
        _progress: Option<&lash::ProgressSender>,
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
                    PluginDirective::emit_events(vec![
                        state
                            .lock()
                            .map_err(|_| {
                                PluginError::Session("plan mode state poisoned".to_string())
                            })?
                            .badge_event(),
                        plan_panel_event(&report),
                    ]),
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
                    PluginDirective::emit_events(vec![
                        state
                            .lock()
                            .map_err(|_| {
                                PluginError::Session("plan mode state poisoned".to_string())
                            })?
                            .badge_event(),
                        plan_panel_event(&report),
                    ]),
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
                    let plan_path = ensure_plan_path(&state, &ctx.session_id, &ctx.host).await?;
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
                let approved = ctx.tool_name == "plan_exit"
                    && ctx.result.success
                    && ctx
                        .result
                        .result
                        .get("approved")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false);
                if approved {
                    let clear_badge = state
                        .lock()
                        .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                        .clear_badge_event();
                    let mut directives = vec![PluginDirective::emit_events(vec![
                        clear_badge,
                        clear_plan_panel_event(),
                    ])];
                    if ctx
                        .result
                        .result
                        .get("execution_mode")
                        .and_then(|value| value.as_str())
                        == Some("fresh_context")
                    {
                        let plan_path = ctx
                            .result
                            .result
                            .get("plan_path")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let seed = PluginMessage::text(
                            lash::MessageRole::User,
                            plan_exit_fresh_context_input(&plan_path),
                        );
                        let session_id = fresh_context_session_id();
                        directives.push(PluginDirective::CreateSession {
                            request: Box::new(SessionCreateRequest {
                                session_id: Some(session_id.clone()),
                                parent_session_id: Some(ctx.session_id.clone()),
                                start: SessionStartPoint::Empty,
                                policy: None,
                                plugin_mode: SessionPluginMode::Fresh,
                                initial_nodes: Vec::new(),
                                first_turn_input: Some(seed),
                                context_surface: SessionContextSurface::default(),
                                mode_extras: lash::ModeExtras::default(),
                                usage_source: Some("plan_execution".to_string()),
                            }),
                        });
                        directives.push(PluginDirective::short_circuit(ToolResult::ok(json!({
                            "approved": true,
                            "plan_path": plan_path,
                            "execution_mode": "fresh_context",
                            "session_id": session_id,
                        }))));
                    }
                    return Ok(directives);
                }

                let enabled = state
                    .lock()
                    .map_err(|_| PluginError::Session("plan mode state poisoned".to_string()))?
                    .enabled;
                if !enabled || ctx.tool_name != "apply_patch" || !ctx.result.success {
                    return Ok(Vec::new());
                }

                let report = ensure_plan_report(&state, &ctx.session_id, &ctx.host, false).await?;
                Ok(vec![PluginDirective::emit_events(vec![plan_panel_event(
                    &report,
                )])])
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
                    availability: Some(lash::ToolAvailability::Hidden),
                })
                .collect::<Vec<_>>();
            overrides.push(ToolSurfaceOverride {
                tool_name: "plan_exit".to_string(),
                availability: Some(lash::ToolAvailability::Documented),
            });
            if tool_surface_config.allowed_tools.contains("apply_patch") {
                overrides.push(ToolSurfaceOverride {
                    tool_name: "apply_patch".to_string(),
                    availability: Some(lash::ToolAvailability::Documented),
                });
            }

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
                        "plan_path": { "type": ["string", "null"] },
                        "panel_title": { "type": ["string", "null"] },
                        "panel_content": { "type": ["string", "null"] }
                    },
                    "required": ["session_id", "enabled", "plan_path", "panel_title", "panel_content"],
                    "additionalProperties": false
                }),
            },
            Arc::new(move |ctx, _args| {
                let state = Arc::clone(&status_state);
                Box::pin(async move {
                    let Some(session_id) = ctx.session_id else {
                        return ToolResult::err(json!("plan_mode.status requires session_id"));
                    };
                    let enabled = match state.lock() {
                        Ok(guard) => guard.enabled,
                        Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
                    };
                    let report = match ensure_plan_report(&state, &session_id, &ctx.host, enabled)
                        .await
                    {
                        Ok(report) => report,
                        Err(err) => return ToolResult::err(json!(err.to_string())),
                    };
                    ToolResult::ok(plan_status_payload(&session_id, enabled, Some(&report)))
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
                            "plan_path": { "type": ["string", "null"] },
                            "panel_title": { "type": ["string", "null"] },
                            "panel_content": { "type": ["string", "null"] }
                        },
                        "required": ["session_id", "enabled", "plan_path", "panel_title", "panel_content"],
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
                        let target_enabled = match state.lock() {
                            Ok(guard) => match op_name.as_str() {
                                "plan_mode.enable" => true,
                                "plan_mode.disable" => false,
                                "plan_mode.toggle" => !guard.enabled,
                                _ => unreachable!(),
                            },
                            Err(_) => return ToolResult::err(json!("plan mode state poisoned")),
                        };
                        let enabled = match set_plan_mode_enabled_state(
                            &state,
                            &session_id,
                            &ctx.host,
                            target_enabled,
                        )
                        .await
                        {
                            Ok(enabled) => enabled,
                            Err(err) => return ToolResult::err(json!(err.to_string())),
                        };
                        let report = match ensure_plan_report(&state, &session_id, &ctx.host, enabled)
                            .await
                        {
                            Ok(report) => report,
                            Err(err) => return ToolResult::err(json!(err.to_string())),
                        };
                        ToolResult::ok(plan_status_payload(&session_id, enabled, Some(&report)))
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
