use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use lash_core::ToolResult;
use lash_core::plugin::{
    PluginAction, PluginActionContext, PluginActionFailure, PluginActionKind, PluginError,
    PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta, SessionParam,
    SessionPlugin, SnapshotReader, SnapshotWriter,
};

const fn default_monitor_timeout_ms() -> u64 {
    300_000
}

pub const MAX_MONITOR_TIMEOUT_MS: u64 = 3_600_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MonitorArmOn {
    #[default]
    Manual,
    SessionStart,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MonitorWakePolicy {
    Notify,
    #[default]
    QueueTurn,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MonitorSpec {
    pub id: String,
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub persistent: bool,
    #[serde(default = "default_monitor_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub arm_on: MonitorArmOn,
    #[serde(default)]
    pub wake_policy: MonitorWakePolicy,
    #[serde(default)]
    pub restart_on_restore: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MonitorRunState {
    #[default]
    Idle,
    Running,
    Stopped,
    Exited,
    Failed,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MonitorStatus {
    pub spec: MonitorSpec,
    #[serde(default)]
    pub armed: bool,
    #[serde(default)]
    pub state: MonitorRunState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_event: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_exit_status: Option<i32>,
    #[serde(default)]
    pub event_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MonitorSnapshot {
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub active_count: usize,
    #[serde(default)]
    pub monitors: Vec<MonitorStatus>,
}

pub const MONITOR_PLUGIN_ID: &str = "monitor";
pub const MONITOR_LINE_EVENT: &str = "monitor.line";

pub fn monitor_process_id(monitor_id: &str) -> String {
    format!("monitor:{monitor_id}")
}

pub fn monitor_tool_args(spec: &MonitorSpec, description: Option<&str>) -> serde_json::Value {
    let mut args = serde_json::json!({
        "id": spec.id,
        "command": spec.command,
        "description": description.unwrap_or(&spec.command),
        "persistent": spec.persistent,
        "timeout_ms": spec.timeout_ms,
        "wake_policy": spec.wake_policy,
    });
    if let Some(cwd) = spec.cwd.as_ref() {
        args["cwd"] = serde_json::json!(cwd);
    }
    if !spec.env.is_empty() {
        args["env"] = serde_json::json!(spec.env);
    }
    args
}

pub fn monitor_process_descriptor(spec: &MonitorSpec) -> lash_core::ProcessHandleDescriptor {
    lash_core::ProcessHandleDescriptor::new(Some("monitor"), Some(spec.id.clone()))
}

pub fn monitor_process_registration(
    spec: &MonitorSpec,
    description: Option<&str>,
) -> lash_core::ProcessRegistration {
    let process_id = monitor_process_id(&spec.id);
    lash_core::ProcessRegistration::new(
        process_id.clone(),
        lash_core::ProcessInput::ToolCall {
            call: lash_core::PreparedToolCall::from_parts(
                process_id,
                "monitor",
                monitor_tool_args(spec, description),
                None,
                serde_json::Value::Null,
            ),
        },
    )
    .with_extra_event_types([monitor_line_event_type()])
}

pub fn monitor_line_event_type() -> lash_core::ProcessEventType {
    let mut properties = serde_json::Map::new();
    properties.insert("line".to_string(), serde_json::json!({ "type": "string" }));
    properties.insert(
        "stream".to_string(),
        serde_json::json!({ "type": "string" }),
    );
    properties.insert(
        "timestamp".to_string(),
        serde_json::json!({ "type": "string" }),
    );
    properties.insert(
        "wake_input".to_string(),
        serde_json::json!({ "type": "string" }),
    );
    lash_core::ProcessEventType {
        name: MONITOR_LINE_EVENT.to_string(),
        payload_schema: lash_core::LashSchema::object(
            properties,
            vec![
                "line".to_string(),
                "stream".to_string(),
                "timestamp".to_string(),
            ],
        ),
        semantics: lash_core::ProcessEventSemanticsSpec {
            wake: Some(lash_core::ProcessWakeSpec {
                when: Some(lash_core::ProcessValueSelector::Present(
                    "/wake_input".to_string(),
                )),
                input: lash_core::ProcessValueSelector::Pointer("/wake_input".to_string()),
                dedupe_key: lash_core::ProcessWakeDedupeKey::EventIdentity,
            }),
            ..lash_core::ProcessEventSemanticsSpec::default()
        },
    }
}

fn apply_process_terminal(status: &mut MonitorStatus, record: &lash_core::ProcessRecord) {
    status.state = match record.terminal.as_ref().map(|terminal| terminal.state) {
        None => MonitorRunState::Running,
        Some(lash_core::ProcessTerminalState::Completed) => MonitorRunState::Exited,
        Some(lash_core::ProcessTerminalState::Failed) => MonitorRunState::Failed,
        Some(lash_core::ProcessTerminalState::Cancelled) => MonitorRunState::Stopped,
    };
    if let Some(terminal) = record.terminal.as_ref() {
        match &terminal.await_output {
            lash_core::ProcessAwaitOutput::Failure { message, .. } => {
                status.last_error = Some(message.clone());
            }
            lash_core::ProcessAwaitOutput::Cancelled { message, .. } => {
                status.last_error = Some(message.clone());
            }
            lash_core::ProcessAwaitOutput::Success { value, .. } => {
                status.last_error = None;
                status.last_exit_status = value
                    .get("exit_status")
                    .and_then(serde_json::Value::as_i64)
                    .and_then(|code| i32::try_from(code).ok());
            }
        }
    }
}

pub fn monitor_status_from_process_record(
    record: &lash_core::ProcessRecord,
) -> Option<MonitorStatus> {
    let lash_core::ProcessInput::ToolCall { call } = record.input.as_ref() else {
        return None;
    };
    if call.tool_name != "monitor" {
        return None;
    }
    let id = call
        .args
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .or_else(|| record.id.strip_prefix("monitor:").map(str::to_string))?;
    let command = call
        .args
        .get("command")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    let cwd = call
        .args
        .get("cwd")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let env = call
        .args
        .get("env")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();
    let persistent = call
        .args
        .get("persistent")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let timeout_ms = call
        .args
        .get("timeout_ms")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(default_monitor_timeout_ms);
    let wake_policy = match call
        .args
        .get("wake_policy")
        .and_then(serde_json::Value::as_str)
    {
        Some("notify") => MonitorWakePolicy::Notify,
        _ => MonitorWakePolicy::QueueTurn,
    };
    let mut status = MonitorStatus {
        spec: MonitorSpec {
            id,
            command,
            cwd,
            env,
            persistent,
            timeout_ms,
            wake_policy,
            ..MonitorSpec::default()
        },
        armed: !record.is_terminal(),
        state: MonitorRunState::Idle,
        last_event: None,
        last_error: None,
        last_exit_status: None,
        event_count: 0,
    };
    apply_process_terminal(&mut status, record);
    Some(status)
}

#[derive(Default)]
pub struct MonitorPluginFactory;

impl PluginFactory for MonitorPluginFactory {
    fn id(&self) -> &'static str {
        MONITOR_PLUGIN_ID
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(MonitorPlugin::default()))
    }
}

#[derive(Default)]
struct MonitorPlugin {
    state: Arc<Mutex<MonitorPluginState>>,
}

fn tool_result_output<T>(result: ToolResult) -> Result<T, PluginActionFailure>
where
    T: serde::de::DeserializeOwned,
{
    if !result.is_success() {
        return Err(PluginActionFailure::new(
            result.value_for_projection().to_string(),
        ));
    }
    serde_json::from_value(result.into_output().value_for_projection())
        .map_err(|err| PluginActionFailure::new(format!("invalid monitor output: {err}")))
}

fn tool_result_unit(result: ToolResult) -> Result<(), PluginActionFailure> {
    if result.is_success() {
        Ok(())
    } else {
        Err(PluginActionFailure::new(
            result.value_for_projection().to_string(),
        ))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct MonitorSnapshotState {
    #[serde(default)]
    revision: u64,
    #[serde(default)]
    monitors: BTreeMap<String, MonitorEntry>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct MonitorEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    owner_plugin_id: Option<String>,
    status: MonitorStatus,
}

#[derive(Default)]
struct MonitorPluginState {
    snapshot: MonitorSnapshotState,
}

#[derive(Clone, Debug, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct OwnedMonitorSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_id: Option<String>,
    pub spec: MonitorSpec,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct RegisterSpecsArgs {
    #[serde(default)]
    pub specs: Vec<OwnedMonitorSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct StartMonitorArgs {
    pub spec: MonitorSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct StopMonitorArgs {
    pub id: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct MonitorEmptyArgs {}

pub struct MonitorRegisterSpecsOp;
pub struct MonitorStatusOp;
pub struct MonitorStartOp;
pub struct MonitorStopOp;

impl PluginAction for MonitorRegisterSpecsOp {
    const NAME: &'static str = "monitor.register_specs";
    const DESCRIPTION: &'static str = "Register typed monitor specs for the current session.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = RegisterSpecsArgs;
    type Output = ();
}

impl PluginAction for MonitorStatusOp {
    const NAME: &'static str = "monitor.status";
    const DESCRIPTION: &'static str = "Return current monitor status.";
    const KIND: PluginActionKind = PluginActionKind::Query;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = MonitorEmptyArgs;
    type Output = MonitorSnapshot;
}

impl PluginAction for MonitorStartOp {
    const NAME: &'static str = "monitor.start";
    const DESCRIPTION: &'static str = "Start a monitor.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = StartMonitorArgs;
    type Output = MonitorSnapshot;
}

impl PluginAction for MonitorStopOp {
    const NAME: &'static str = "monitor.stop";
    const DESCRIPTION: &'static str = "Stop a monitor.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = StopMonitorArgs;
    type Output = MonitorSnapshot;
}

impl MonitorPlugin {
    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, MonitorPluginState>, PluginError> {
        self.state
            .lock()
            .map_err(|_| PluginError::Session("monitor state poisoned".to_string()))
    }

    fn bump_revision(state: &mut MonitorPluginState) {
        state.snapshot.revision = state.snapshot.revision.saturating_add(1);
    }

    fn visible_id(owner_plugin_id: Option<&str>, spec_id: &str) -> String {
        match owner_plugin_id {
            Some(plugin_id) if !plugin_id.is_empty() && plugin_id != MONITOR_PLUGIN_ID => {
                format!("{plugin_id}:{spec_id}")
            }
            _ => spec_id.to_string(),
        }
    }

    fn snapshot_from_state(state: &MonitorPluginState) -> MonitorSnapshot {
        Self::snapshot_from_state_and_processes(state, &[])
    }

    fn snapshot_from_state_and_processes(
        state: &MonitorPluginState,
        processes: &[lash_core::ProcessHandleGrantEntry],
    ) -> MonitorSnapshot {
        let by_id = processes
            .iter()
            .filter(|(grant, _)| grant.descriptor.kind.as_deref() == Some("monitor"))
            .map(|(_, record)| (record.id.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        let mut monitors = state
            .snapshot
            .monitors
            .values()
            .map(|entry| {
                let mut status = entry.status.clone();
                if let Some(record) = by_id.get(monitor_process_id(&status.spec.id).as_str()) {
                    apply_process_terminal(&mut status, record);
                }
                status
            })
            .collect::<Vec<_>>();
        let known_ids = monitors
            .iter()
            .map(|status| status.spec.id.clone())
            .collect::<std::collections::HashSet<_>>();
        monitors.extend(by_id.values().filter_map(|record| {
            let status = monitor_status_from_process_record(record)?;
            (!known_ids.contains(&status.spec.id)).then_some(status)
        }));
        monitors.sort_by(|left, right| left.spec.id.cmp(&right.spec.id));
        MonitorSnapshot {
            revision: state.snapshot.revision,
            active_count: monitors
                .iter()
                .filter(|status| status.state == MonitorRunState::Running)
                .count(),
            monitors,
        }
    }

    fn record_monitor_notice(state: &mut MonitorPluginState, monitor_id: &str, message: String) {
        if let Some(entry) = state.snapshot.monitors.get_mut(monitor_id) {
            entry.status.last_event = Some(message);
            entry.status.event_count = entry.status.event_count.saturating_add(1);
        }
        Self::bump_revision(state);
    }

    fn upsert_spec(
        state: &mut MonitorPluginState,
        owner_plugin_id: Option<String>,
        mut spec: MonitorSpec,
        armed: Option<bool>,
    ) -> Result<String, PluginError> {
        spec.id = Self::visible_id(owner_plugin_id.as_deref(), spec.id.trim());
        if spec.id.trim().is_empty() {
            return Err(PluginError::Session(
                "monitor id must be a non-empty string".to_string(),
            ));
        }
        if spec.command.trim().is_empty() {
            return Err(PluginError::Session(
                "monitor command must be a non-empty string".to_string(),
            ));
        }
        let default_armed = matches!(spec.arm_on, MonitorArmOn::SessionStart);
        let spec_id = spec.id.clone();
        match state.snapshot.monitors.get_mut(&spec_id) {
            Some(entry) => {
                let was_armed = entry.status.armed;
                entry.owner_plugin_id = owner_plugin_id;
                entry.status.spec = spec.clone();
                entry.status.armed = armed.unwrap_or(was_armed || default_armed);
            }
            None => {
                state.snapshot.monitors.insert(
                    spec_id.clone(),
                    MonitorEntry {
                        owner_plugin_id,
                        status: MonitorStatus {
                            spec,
                            armed: armed.unwrap_or(default_armed),
                            state: MonitorRunState::Idle,
                            last_event: None,
                            last_error: None,
                            last_exit_status: None,
                            event_count: 0,
                        },
                    },
                );
            }
        }
        Self::bump_revision(state);
        Ok(spec_id)
    }

    async fn ensure_running(
        &self,
        session_id: &str,
        processes: Arc<dyn lash_core::ProcessService>,
    ) -> Result<(), PluginError> {
        let to_start = {
            let state = self.lock_state()?;
            state
                .snapshot
                .monitors
                .values()
                .filter(|entry| {
                    entry.status.armed && entry.status.state != MonitorRunState::Running
                })
                .map(|entry| entry.status.spec.clone())
                .collect::<Vec<_>>()
        };
        for spec in to_start {
            self.start_monitor_task(session_id, processes.clone(), spec)
                .await?;
        }
        Ok(())
    }

    async fn start_monitor_task(
        &self,
        session_id: &str,
        processes: Arc<dyn lash_core::ProcessService>,
        spec: MonitorSpec,
    ) -> Result<(), PluginError> {
        let managed_spec = monitor_process_registration(&spec, None);
        match processes
            .start(
                session_id,
                managed_spec,
                lash_core::ProcessStartOptions::new()
                    .with_wake_session_id(session_id)
                    .with_descriptor(monitor_process_descriptor(&spec)),
                lash_core::ProcessOpScope::new(),
            )
            .await
        {
            Ok(_) => {
                let mut state = self.lock_state()?;
                if let Some(entry) = state.snapshot.monitors.get_mut(&spec.id) {
                    entry.status.armed = true;
                    entry.status.state = MonitorRunState::Running;
                    entry.status.last_error = None;
                    entry.status.last_exit_status = None;
                }
                Self::bump_revision(&mut state);
                Ok(())
            }
            Err(err) if err.to_string().contains("already registered") => {
                let mut state = self.lock_state()?;
                if let Some(entry) = state.snapshot.monitors.get_mut(&spec.id) {
                    entry.status.state = MonitorRunState::Running;
                }
                Self::bump_revision(&mut state);
                Ok(())
            }
            Err(err) => {
                let mut state = self.lock_state()?;
                if let Some(entry) = state.snapshot.monitors.get_mut(&spec.id) {
                    entry.status.state = MonitorRunState::Failed;
                    entry.status.last_error = Some(err.to_string());
                    entry.status.armed = false;
                }
                Self::record_monitor_notice(
                    &mut state,
                    &spec.id,
                    format!("Failed to start monitor: {err}"),
                );
                Err(err)
            }
        }
    }

    async fn handle_register_specs(
        &self,
        ctx: PluginActionContext,
        args: serde_json::Value,
    ) -> ToolResult {
        let _ = ctx;
        let parsed = match serde_json::from_value::<RegisterSpecsArgs>(args) {
            Ok(parsed) => parsed,
            Err(err) => return ToolResult::err_fmt(format_args!("invalid monitor specs: {err}")),
        };
        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        for owned in parsed.specs {
            if let Err(err) = Self::upsert_spec(&mut state, owned.plugin_id, owned.spec, None) {
                return ToolResult::err_fmt(err.to_string());
            }
        }
        ToolResult::ok(serde_json::json!(Self::snapshot_from_state(&state)))
    }

    async fn handle_status(&self, ctx: PluginActionContext) -> ToolResult {
        let Some(session_id) = ctx.session_id.as_deref() else {
            return ToolResult::err_fmt("monitor.status requires a session");
        };
        if let Err(err) = self.ensure_running(session_id, ctx.processes.clone()).await {
            return ToolResult::err_fmt(err.to_string());
        }
        let processes = match ctx
            .processes
            .list_visible(session_id, lash_core::ProcessOpScope::new())
            .await
        {
            Ok(processes) => processes,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        let state = match self.lock_state() {
            Ok(state) => state,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        ToolResult::ok(serde_json::json!(Self::snapshot_from_state_and_processes(
            &state, &processes
        )))
    }

    async fn handle_start(&self, ctx: PluginActionContext, args: serde_json::Value) -> ToolResult {
        let Some(session_id) = ctx.session_id.as_deref() else {
            return ToolResult::err_fmt("monitor.start requires a session");
        };
        let parsed = match serde_json::from_value::<StartMonitorArgs>(args) {
            Ok(parsed) => parsed,
            Err(err) => return ToolResult::err_fmt(format_args!("invalid monitor spec: {err}")),
        };
        let entry_spec = {
            let mut state = match self.lock_state() {
                Ok(state) => state,
                Err(err) => return ToolResult::err_fmt(err.to_string()),
            };
            let spec_id = match Self::upsert_spec(&mut state, None, parsed.spec.clone(), Some(true))
            {
                Ok(spec_id) => spec_id,
                Err(err) => return ToolResult::err_fmt(err.to_string()),
            };
            let Some(entry_spec) = state
                .snapshot
                .monitors
                .get(&spec_id)
                .map(|entry| entry.status.spec.clone())
            else {
                return ToolResult::err_fmt("monitor registration failed");
            };
            entry_spec
        };
        if let Err(err) = self
            .start_monitor_task(session_id, ctx.processes.clone(), entry_spec)
            .await
        {
            return ToolResult::err_fmt(err.to_string());
        }
        let state = match self.lock_state() {
            Ok(state) => state,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        ToolResult::ok(serde_json::json!(Self::snapshot_from_state(&state)))
    }

    async fn handle_stop(&self, ctx: PluginActionContext, args: serde_json::Value) -> ToolResult {
        let Some(session_id) = ctx.session_id.as_deref() else {
            return ToolResult::err_fmt("monitor.stop requires a session");
        };
        let parsed = match serde_json::from_value::<StopMonitorArgs>(args) {
            Ok(parsed) => parsed,
            Err(err) => return ToolResult::err_fmt(format_args!("invalid stop payload: {err}")),
        };
        let monitor_id = {
            let state = match self.lock_state() {
                Ok(state) => state,
                Err(err) => return ToolResult::err_fmt(err.to_string()),
            };
            let Some(entry) = state.snapshot.monitors.get(&parsed.id) else {
                return ToolResult::err_fmt(format_args!("unknown monitor `{}`", parsed.id));
            };
            entry.status.spec.id.clone()
        };

        let process_id = monitor_process_id(&parsed.id);
        if let Err(err) = ctx
            .processes
            .cancel(session_id, &process_id, lash_core::ProcessOpScope::new())
            .await
        {
            return ToolResult::err_fmt(err.to_string());
        }

        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        let Some(entry) = state.snapshot.monitors.get_mut(&parsed.id) else {
            return ToolResult::err_fmt(format_args!("unknown monitor `{}`", parsed.id));
        };
        entry.status.armed = false;
        entry.status.state = MonitorRunState::Stopped;
        entry.status.last_error = None;
        entry.status.last_exit_status = None;
        MonitorPlugin::record_monitor_notice(
            &mut state,
            &monitor_id,
            "Monitor stopped".to_string(),
        );
        ToolResult::ok(serde_json::json!(Self::snapshot_from_state(&state)))
    }
}

#[async_trait]
impl SessionPlugin for MonitorPlugin {
    fn id(&self) -> &'static str {
        MONITOR_PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let state = Arc::clone(&self.state);
        reg.actions()
            .typed::<MonitorRegisterSpecsOp, _, _>(move |ctx, args| {
                let plugin = MonitorPlugin {
                    state: Arc::clone(&state),
                };
                async move {
                    tool_result_unit(
                        plugin
                            .handle_register_specs(
                                ctx,
                                serde_json::to_value(args).unwrap_or_default(),
                            )
                            .await,
                    )
                }
            })?;

        let state = Arc::clone(&self.state);
        reg.actions()
            .typed::<MonitorStatusOp, _, _>(move |ctx, _args| {
                let plugin = MonitorPlugin {
                    state: Arc::clone(&state),
                };
                async move { tool_result_output(plugin.handle_status(ctx).await) }
            })?;

        let state = Arc::clone(&self.state);
        reg.actions()
            .typed::<MonitorStartOp, _, _>(move |ctx, args| {
                let plugin = MonitorPlugin {
                    state: Arc::clone(&state),
                };
                async move {
                    tool_result_output(
                        plugin
                            .handle_start(ctx, serde_json::to_value(args).unwrap_or_default())
                            .await,
                    )
                }
            })?;

        let state = Arc::clone(&self.state);
        reg.actions()
            .typed::<MonitorStopOp, _, _>(move |ctx, args| {
                let plugin = MonitorPlugin {
                    state: Arc::clone(&state),
                };
                async move {
                    tool_result_output(
                        plugin
                            .handle_stop(ctx, serde_json::to_value(args).unwrap_or_default())
                            .await,
                    )
                }
            })?;
        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let snapshot = self.lock_state()?.snapshot.clone();
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            revision: snapshot.revision,
            state: Some(serde_json::to_value(snapshot).map_err(|err| {
                PluginError::Snapshot(format!("failed to serialize monitor snapshot: {err}"))
            })?),
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
            .map(serde_json::from_value::<MonitorSnapshotState>)
            .transpose()
            .map_err(|err| PluginError::Snapshot(err.to_string()))?
            .unwrap_or_default();
        let mut state = self.lock_state()?;
        state.snapshot = snapshot;
        for entry in state.snapshot.monitors.values_mut() {
            if entry.status.armed && entry.status.spec.restart_on_restore {
                entry.status.state = MonitorRunState::Idle;
            } else if entry.status.state == MonitorRunState::Running {
                entry.status.state = MonitorRunState::Idle;
                entry.status.armed = false;
            }
        }
        Ok(())
    }

    fn snapshot_revision(&self) -> u64 {
        self.state
            .lock()
            .map(|state| state.snapshot.revision)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_monitor_state(spec: &MonitorSpec) -> Arc<Mutex<MonitorPluginState>> {
        let mut monitors = BTreeMap::new();
        monitors.insert(
            spec.id.clone(),
            MonitorEntry {
                owner_plugin_id: None,
                status: MonitorStatus {
                    spec: spec.clone(),
                    armed: true,
                    state: MonitorRunState::Running,
                    last_event: None,
                    last_error: None,
                    last_exit_status: None,
                    event_count: 0,
                },
            },
        );
        Arc::new(Mutex::new(MonitorPluginState {
            snapshot: MonitorSnapshotState {
                revision: 0,
                monitors,
            },
        }))
    }

    #[test]
    fn monitor_line_event_declares_wake_semantics() {
        let event = monitor_line_event_type();

        assert_eq!(event.name, MONITOR_LINE_EVENT);
        assert!(event.semantics.wake.is_some());
        assert!(
            event
                .payload_schema
                .schema
                .get("required")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|required| required
                    .iter()
                    .any(|field| field.as_str() == Some("line")))
        );
    }

    #[test]
    fn monitor_registration_uses_canonical_process_shape() {
        let spec = MonitorSpec {
            id: "app-errors".to_string(),
            command: "tail -f app.log".to_string(),
            persistent: true,
            ..MonitorSpec::default()
        };
        let registration = monitor_process_registration(&spec, Some("Watch app errors"));
        let descriptor = monitor_process_descriptor(&spec);

        assert_eq!(registration.id, "monitor:app-errors");
        assert_eq!(descriptor.kind.as_deref(), Some("monitor"));
        assert_eq!(descriptor.label.as_deref(), Some("app-errors"));
        let lash_core::ProcessInput::ToolCall { call } = registration.input.as_ref() else {
            panic!("monitor registration should use a tool call");
        };
        assert_eq!(call.tool_name, "monitor");
        assert_eq!(call.args["description"], "Watch app errors");
        assert!(
            registration
                .event_types
                .iter()
                .any(|event_type| event_type.name == MONITOR_LINE_EVENT)
        );
    }

    #[test]
    fn monitor_status_reconstructs_from_process_record() {
        let spec = MonitorSpec {
            id: "restore".to_string(),
            command: "printf ready".to_string(),
            timeout_ms: 1_000,
            ..MonitorSpec::default()
        };
        let record =
            lash_core::ProcessRecord::from_registration(monitor_process_registration(&spec, None));

        let status = monitor_status_from_process_record(&record).expect("monitor status");

        assert_eq!(status.spec.id, "restore");
        assert_eq!(status.spec.command, "printf ready");
        assert_eq!(status.state, MonitorRunState::Running);
    }

    #[test]
    fn status_projection_reads_process_terminal_state() {
        let spec = MonitorSpec {
            id: "slow".to_string(),
            command: "sleep 5".to_string(),
            persistent: false,
            timeout_ms: 50,
            ..Default::default()
        };
        let state = seeded_monitor_state(&spec);
        let mut record =
            lash_core::ProcessRecord::from_registration(lash_core::ProcessRegistration::new(
                "monitor:slow",
                lash_core::ProcessInput::External {
                    metadata: serde_json::Value::Null,
                },
            ));
        record.terminal = Some(lash_core::ProcessTerminalSemantics {
            state: lash_core::ProcessTerminalState::Failed,
            await_output: lash_core::ProcessAwaitOutput::Failure {
                class: lash_core::ToolFailureClass::Execution,
                code: "process_command_timeout".to_string(),
                message: "monitor timed out after 50ms".to_string(),
                raw: None,
                control: None,
            },
        });
        let grant = lash_core::ProcessHandleGrant {
            session_id: "root".to_string(),
            process_id: "monitor:slow".to_string(),
            descriptor: lash_core::ProcessHandleDescriptor::new(Some("monitor"), Some("slow")),
        };

        let guard = state.lock().expect("monitor state");
        let snapshot = MonitorPlugin::snapshot_from_state_and_processes(&guard, &[(grant, record)]);

        assert_eq!(snapshot.monitors[0].state, MonitorRunState::Failed);
        assert_eq!(
            snapshot.monitors[0].last_error.as_deref(),
            Some("monitor timed out after 50ms")
        );
    }
}
