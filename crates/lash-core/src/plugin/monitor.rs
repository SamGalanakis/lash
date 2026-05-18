use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::select;

use crate::ToolResult;
use crate::monitor::{
    MonitorArmOn, MonitorEvent, MonitorRunState, MonitorSnapshot, MonitorSpec, MonitorStatus,
    MonitorUpdateBatch, MonitorWakePolicy,
};
use crate::plugin::{
    PluginAction, PluginActionContext, PluginActionFailure, PluginActionKind, PluginError,
    PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta, SessionParam,
    SessionPlugin, SnapshotReader, SnapshotWriter,
};

pub const MONITOR_PLUGIN_ID: &str = "monitor";

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
    serde_json::from_value(result.into_value_for_projection())
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
    sequence: u64,
    #[serde(default)]
    monitors: BTreeMap<String, MonitorEntry>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct MonitorEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    owner_plugin_id: Option<String>,
    status: MonitorStatus,
    #[serde(default)]
    pending_wake: bool,
    #[serde(skip, default)]
    runtime_pid: Option<u32>,
}

#[derive(Default)]
struct MonitorPluginState {
    snapshot: MonitorSnapshotState,
    updates: Vec<MonitorEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize, crate::JsonSchema)]
pub struct OwnedMonitorSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_id: Option<String>,
    pub spec: MonitorSpec,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, crate::JsonSchema)]
pub struct RegisterSpecsArgs {
    #[serde(default)]
    pub specs: Vec<OwnedMonitorSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, crate::JsonSchema)]
pub struct StartMonitorArgs {
    pub spec: MonitorSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, crate::JsonSchema)]
pub struct StopMonitorArgs {
    pub id: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, crate::JsonSchema)]
pub struct MonitorEmptyArgs {}

#[derive(Clone, Debug, Default, Serialize, Deserialize, crate::JsonSchema)]
pub struct AckWakeArgs {
    #[serde(default)]
    pub ids: Vec<String>,
}

pub struct MonitorRegisterSpecsOp;
pub struct MonitorStatusOp;
pub struct MonitorTakeUpdatesOp;
pub struct MonitorAckWakeOp;
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

impl PluginAction for MonitorTakeUpdatesOp {
    const NAME: &'static str = "monitor.take_updates";
    const DESCRIPTION: &'static str = "Drain pending monitor updates.";
    const KIND: PluginActionKind = PluginActionKind::Task;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = MonitorEmptyArgs;
    type Output = MonitorUpdateBatch;
}

impl PluginAction for MonitorAckWakeOp {
    const NAME: &'static str = "monitor.ack_wake";
    const DESCRIPTION: &'static str = "Acknowledge pending monitor wake-ups.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = AckWakeArgs;
    type Output = ();
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
        let mut monitors = state
            .snapshot
            .monitors
            .values()
            .map(|entry| entry.status.clone())
            .collect::<Vec<_>>();
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

    fn queue_update(
        state: &mut MonitorPluginState,
        monitor_id: &str,
        message: String,
        queue_turn_input: Option<String>,
    ) {
        state.snapshot.sequence = state.snapshot.sequence.saturating_add(1);
        state.updates.push(MonitorEvent {
            sequence: state.snapshot.sequence,
            monitor_id: monitor_id.to_string(),
            message,
            queue_turn_input,
        });
        if state.updates.len() > 128 {
            let drop_count = state.updates.len() - 128;
            state.updates.drain(0..drop_count);
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
                        pending_wake: false,
                        runtime_pid: None,
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
        host: Arc<dyn crate::plugin::runtime_host::RuntimeSessionHost>,
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
            self.start_task(session_id, host.clone(), spec).await?;
        }
        Ok(())
    }

    async fn start_task(
        &self,
        session_id: &str,
        host: Arc<dyn crate::plugin::runtime_host::RuntimeSessionHost>,
        spec: MonitorSpec,
    ) -> Result<(), PluginError> {
        let task_id = format!("monitor:{}", spec.id);
        let state = Arc::clone(&self.state);
        let session_id_owned = session_id.to_string();
        let spec_clone = spec.clone();
        let task_host = host.clone();
        let managed_spec = crate::BackgroundTaskRegistration {
            id: task_id.clone(),
            kind: crate::BackgroundTaskKind::Monitor,
            producer: "monitor",
            child_session_id: None,
            parent_task_id: None,
        };
        match host
            .spawn_managed_task(
                session_id,
                managed_spec,
                Box::pin(async move {
                    run_monitor_task(state, session_id_owned, spec_clone, task_host).await
                }),
            )
            .await
        {
            Ok(()) => {
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
            Err(err) if err.to_string().contains("already running") => {
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
                Self::queue_update(
                    &mut state,
                    &spec.id,
                    format!("Failed to start monitor: {err}"),
                    None,
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
        if let Err(err) = self.ensure_running(session_id, ctx.host.clone()).await {
            return ToolResult::err_fmt(err.to_string());
        }
        let state = match self.lock_state() {
            Ok(state) => state,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        ToolResult::ok(serde_json::json!(Self::snapshot_from_state(&state)))
    }

    async fn handle_take_updates(&self, ctx: PluginActionContext) -> ToolResult {
        let Some(session_id) = ctx.session_id.as_deref() else {
            return ToolResult::err_fmt("monitor.take_updates requires a session");
        };
        if let Err(err) = self.ensure_running(session_id, ctx.host.clone()).await {
            return ToolResult::err_fmt(err.to_string());
        }
        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        let active_count = state
            .snapshot
            .monitors
            .values()
            .filter(|entry| entry.status.state == MonitorRunState::Running)
            .count();
        ToolResult::ok(serde_json::json!(MonitorUpdateBatch {
            revision: state.snapshot.revision,
            active_count,
            events: std::mem::take(&mut state.updates),
        }))
    }

    async fn handle_ack_wake(&self, args: serde_json::Value) -> ToolResult {
        let parsed = match serde_json::from_value::<AckWakeArgs>(args) {
            Ok(parsed) => parsed,
            Err(err) => return ToolResult::err_fmt(format_args!("invalid ack payload: {err}")),
        };
        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        for id in parsed.ids {
            if let Some(entry) = state.snapshot.monitors.get_mut(&id) {
                entry.pending_wake = false;
            }
        }
        Self::bump_revision(&mut state);
        ToolResult::ok(serde_json::json!(Self::snapshot_from_state(&state)))
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
            .start_task(session_id, ctx.host.clone(), entry_spec)
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
        let (monitor_id, runtime_pid) = {
            let state = match self.lock_state() {
                Ok(state) => state,
                Err(err) => return ToolResult::err_fmt(err.to_string()),
            };
            let Some(entry) = state.snapshot.monitors.get(&parsed.id) else {
                return ToolResult::err_fmt(format_args!("unknown monitor `{}`", parsed.id));
            };
            (entry.status.spec.id.clone(), entry.runtime_pid)
        };

        if let Err(err) = terminate_monitor_process_tree(runtime_pid).await {
            return ToolResult::err_fmt(err.to_string());
        }

        if let Err(err) = ctx
            .host
            .cancel_managed_task(session_id, &format!("monitor:{}", parsed.id))
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
        entry.pending_wake = false;
        entry.runtime_pid = None;
        entry.status.last_error = None;
        entry.status.last_exit_status = None;
        MonitorPlugin::queue_update(&mut state, &monitor_id, "Monitor stopped".to_string(), None);
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
            .typed::<MonitorTakeUpdatesOp, _, _>(move |ctx, _args| {
                let plugin = MonitorPlugin {
                    state: Arc::clone(&state),
                };
                async move { tool_result_output(plugin.handle_take_updates(ctx).await) }
            })?;

        let state = Arc::clone(&self.state);
        reg.actions()
            .typed::<MonitorAckWakeOp, _, _>(move |_ctx, args| {
                let plugin = MonitorPlugin {
                    state: Arc::clone(&state),
                };
                async move {
                    tool_result_unit(
                        plugin
                            .handle_ack_wake(serde_json::to_value(args).unwrap_or_default())
                            .await,
                    )
                }
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
        state.updates.clear();
        for entry in state.snapshot.monitors.values_mut() {
            entry.pending_wake = false;
            entry.runtime_pid = None;
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

async fn run_monitor_task(
    state: Arc<Mutex<MonitorPluginState>>,
    _session_id: String,
    spec: MonitorSpec,
    _host: Arc<dyn crate::plugin::runtime_host::RuntimeSessionHost>,
) -> Result<(), PluginError> {
    let timeout_deadline = (!spec.persistent)
        .then(|| tokio::time::Instant::now() + std::time::Duration::from_millis(spec.timeout_ms));
    let mut command = Command::new("bash");
    command.arg("-lc").arg(&spec.command);
    if let Some(cwd) = spec.cwd.as_ref() {
        command.current_dir(cwd);
    }
    if !spec.env.is_empty() {
        command.envs(spec.env.iter());
    }
    command.kill_on_drop(true);
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    configure_monitor_command(&mut command);

    let mut child = command
        .spawn()
        .map_err(|err| PluginError::Session(format!("failed to start monitor process: {err}")))?;
    let runtime_pid = child.id();
    {
        let mut guard = state
            .lock()
            .map_err(|_| PluginError::Session("monitor state poisoned".to_string()))?;
        if let Some(entry) = guard.snapshot.monitors.get_mut(&spec.id) {
            entry.runtime_pid = runtime_pid;
        }
    }
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| PluginError::Session("monitor stdout unavailable".to_string()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| PluginError::Session("monitor stderr unavailable".to_string()))?;
    let mut stdout_lines = BufReader::new(stdout).lines();
    let mut stderr_lines = BufReader::new(stderr).lines();
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut timeout = timeout_deadline.map(|deadline| Box::pin(tokio::time::sleep_until(deadline)));
    let mut timed_out = false;

    let id = spec.id.clone();
    let wake_policy = spec.wake_policy;

    while !stdout_done || !stderr_done {
        select! {
            _ = timeout.as_mut().unwrap(), if timeout.is_some() => {
                timed_out = true;
                break;
            }
            line = stdout_lines.next_line(), if !stdout_done => {
                match line.map_err(|err| PluginError::Session(format!("monitor stdout read failed: {err}")))? {
                    Some(line) => record_monitor_line(&state, &spec, &id, wake_policy, line, true)?,
                    None => stdout_done = true,
                }
            }
            line = stderr_lines.next_line(), if !stderr_done => {
                match line.map_err(|err| PluginError::Session(format!("monitor stderr read failed: {err}")))? {
                    Some(line) => record_monitor_line(
                        &state,
                        &spec,
                        &id,
                        MonitorWakePolicy::Notify,
                        line,
                        false,
                    )?,
                    None => stderr_done = true,
                }
            }
        }
    }

    let exit =
        if timed_out {
            terminate_monitor_process_tree(runtime_pid).await?;
            child
                .wait()
                .await
                .map_err(|err| PluginError::Session(format!("monitor wait failed: {err}")))?
        } else if let Some(deadline) = timeout_deadline {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, child.wait()).await {
                Ok(result) => result
                    .map_err(|err| PluginError::Session(format!("monitor wait failed: {err}")))?,
                Err(_) => {
                    timed_out = true;
                    terminate_monitor_process_tree(runtime_pid).await?;
                    child.wait().await.map_err(|err| {
                        PluginError::Session(format!("monitor wait failed: {err}"))
                    })?
                }
            }
        } else {
            child
                .wait()
                .await
                .map_err(|err| PluginError::Session(format!("monitor wait failed: {err}")))?
        };

    let mut state = state
        .lock()
        .map_err(|_| PluginError::Session("monitor state poisoned".to_string()))?;
    if let Some(entry) = state.snapshot.monitors.get_mut(&id) {
        let was_stopped = entry.status.state == MonitorRunState::Stopped;
        entry.status.state = if was_stopped {
            MonitorRunState::Stopped
        } else if timed_out {
            MonitorRunState::Failed
        } else if exit.success() {
            MonitorRunState::Exited
        } else {
            MonitorRunState::Failed
        };
        entry.status.last_exit_status = exit.code();
        entry.status.armed = false;
        entry.runtime_pid = None;
        if timed_out {
            entry.status.last_error =
                Some(format!("monitor timed out after {}ms", spec.timeout_ms));
        } else if !exit.success() && !was_stopped {
            entry.status.last_error = Some(format!(
                "monitor exited with status {}",
                exit.code().unwrap_or_default()
            ));
        }
        entry.pending_wake = false;
    }
    if state
        .snapshot
        .monitors
        .get(&id)
        .map(|entry| entry.status.state != MonitorRunState::Stopped)
        .unwrap_or(false)
    {
        MonitorPlugin::queue_update(
            &mut state,
            &id,
            if timed_out {
                format!("Monitor timed out after {}ms", spec.timeout_ms)
            } else if exit.success() {
                "Monitor exited".to_string()
            } else {
                format!(
                    "Monitor failed with status {}",
                    exit.code().unwrap_or_default()
                )
            },
            None,
        );
    }
    Ok(())
}

#[cfg(unix)]
fn configure_monitor_command(command: &mut Command) {
    // Put each monitor under its own session/process group so stop can
    // terminate the whole command tree instead of just the shell wrapper.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn configure_monitor_command(_command: &mut Command) {}

#[cfg(unix)]
async fn terminate_monitor_process_tree(runtime_pid: Option<u32>) -> Result<(), PluginError> {
    let Some(pid) = runtime_pid else {
        return Ok(());
    };
    let pgid = -(pid as i32);
    send_process_group_signal(pgid, libc::SIGTERM)?;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    if process_group_exists(pgid) {
        send_process_group_signal(pgid, libc::SIGKILL)?;
    }
    Ok(())
}

#[cfg(not(unix))]
async fn terminate_monitor_process_tree(_runtime_pid: Option<u32>) -> Result<(), PluginError> {
    Ok(())
}

#[cfg(unix)]
fn process_group_exists(pgid: i32) -> bool {
    // `kill(pgid, 0)` probes whether any process in the group still exists.
    let rc = unsafe { libc::kill(pgid, 0) };
    if rc == 0 {
        return true;
    }
    let err = std::io::Error::last_os_error();
    !matches!(err.raw_os_error(), Some(libc::ESRCH))
}

#[cfg(unix)]
fn send_process_group_signal(pgid: i32, signal: libc::c_int) -> Result<(), PluginError> {
    let rc = unsafe { libc::kill(pgid, signal) };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    if matches!(err.raw_os_error(), Some(libc::ESRCH)) {
        return Ok(());
    }
    Err(PluginError::Session(format!(
        "failed to signal monitor process group {pgid}: {err}"
    )))
}

fn record_monitor_line(
    state: &Arc<Mutex<MonitorPluginState>>,
    spec: &MonitorSpec,
    id: &str,
    wake_policy: MonitorWakePolicy,
    line: String,
    from_stdout: bool,
) -> Result<(), PluginError> {
    let message = line.trim().to_string();
    if message.is_empty() {
        return Ok(());
    }
    let mut state = state
        .lock()
        .map_err(|_| PluginError::Session("monitor state poisoned".to_string()))?;
    let Some(entry) = state.snapshot.monitors.get_mut(id) else {
        return Ok(());
    };
    entry.status.last_event = Some(message.clone());
    entry.status.event_count = entry.status.event_count.saturating_add(1);
    let queue_turn_input =
        if from_stdout && wake_policy == MonitorWakePolicy::QueueTurn && !entry.pending_wake {
            entry.pending_wake = true;
            Some(format!("Monitor event \"{}\": {}", spec.id, message))
        } else {
            None
        };
    MonitorPlugin::queue_update(&mut state, id, message, queue_turn_input);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockSessionManager;

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
                pending_wake: false,
                runtime_pid: None,
            },
        );
        Arc::new(Mutex::new(MonitorPluginState {
            snapshot: MonitorSnapshotState {
                revision: 0,
                sequence: 0,
                monitors,
            },
            updates: Vec::new(),
        }))
    }

    #[tokio::test]
    async fn non_persistent_monitor_times_out_and_records_failure() {
        let spec = MonitorSpec {
            id: "slow".to_string(),
            command: "sleep 5".to_string(),
            persistent: false,
            timeout_ms: 50,
            ..Default::default()
        };
        let state = seeded_monitor_state(&spec);
        let host: Arc<dyn crate::plugin::runtime_host::RuntimeSessionHost> =
            Arc::new(MockSessionManager::default());

        run_monitor_task(state.clone(), "root".to_string(), spec, host)
            .await
            .expect("monitor task should complete after timeout");

        let guard = state.lock().expect("monitor state");
        let entry = guard
            .snapshot
            .monitors
            .get("slow")
            .expect("seeded monitor entry");
        assert_eq!(entry.status.state, MonitorRunState::Failed);
        assert!(!entry.status.armed);
        assert!(
            entry
                .status
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("timed out after 50ms"))
        );
        assert!(
            guard
                .updates
                .iter()
                .any(|event| event.message.contains("timed out after 50ms"))
        );
    }
}
