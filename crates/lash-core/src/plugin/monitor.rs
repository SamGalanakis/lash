use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::ToolResult;
use crate::monitor::{MonitorArmOn, MonitorRunState, MonitorSnapshot, MonitorSpec, MonitorStatus};
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
    #[serde(skip, default)]
    runtime_pid: Option<u32>,
}

#[derive(Default)]
struct MonitorPluginState {
    snapshot: MonitorSnapshotState,
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
        processes: &[crate::ProcessRecord],
    ) -> MonitorSnapshot {
        let by_id = processes
            .iter()
            .filter(|record| record.tags.iter().any(|tag| tag == "monitor"))
            .map(|record| (record.id.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        let mut monitors = state
            .snapshot
            .monitors
            .values()
            .map(|entry| {
                let mut status = entry.status.clone();
                if let Some(record) = by_id.get(format!("monitor:{}", status.spec.id).as_str()) {
                    status.state = match record.state {
                        crate::ProcessState::Pending
                        | crate::ProcessState::Scheduled
                        | crate::ProcessState::Running
                        | crate::ProcessState::Waiting
                        | crate::ProcessState::CancelRequested => MonitorRunState::Running,
                        crate::ProcessState::Completed => MonitorRunState::Exited,
                        crate::ProcessState::Failed => MonitorRunState::Failed,
                        crate::ProcessState::Cancelled => MonitorRunState::Stopped,
                    };
                    status.event_count = record
                        .latest_event
                        .as_ref()
                        .map(|event| event.sequence as usize)
                        .unwrap_or(status.event_count);
                    if let Some(event) = record.latest_event.as_ref() {
                        if event.event_type == "monitor.line" {
                            status.last_event = event
                                .payload
                                .get("line")
                                .and_then(serde_json::Value::as_str)
                                .map(ToOwned::to_owned);
                        }
                    }
                    if let Some(terminal) = record.terminal.as_ref() {
                        match &terminal.await_output {
                            crate::ProcessAwaitOutput::Failure { message, .. } => {
                                status.last_error = Some(message.clone());
                            }
                            crate::ProcessAwaitOutput::Cancelled { message, .. } => {
                                status.last_error = Some(message.clone());
                            }
                            crate::ProcessAwaitOutput::Success { value, .. } => {
                                status.last_error = None;
                                status.last_exit_status = value
                                    .get("exit_status")
                                    .and_then(serde_json::Value::as_i64)
                                    .and_then(|code| i32::try_from(code).ok());
                            }
                        }
                    }
                }
                status
            })
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
            self.start_monitor_task(session_id, host.clone(), spec)
                .await?;
        }
        Ok(())
    }

    async fn start_monitor_task(
        &self,
        session_id: &str,
        host: Arc<dyn crate::plugin::runtime_host::RuntimeSessionHost>,
        spec: MonitorSpec,
    ) -> Result<(), PluginError> {
        let process_id = format!("monitor:{}", spec.id);
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
        let line_event = crate::ProcessEventType {
            name: "monitor.line".to_string(),
            payload_schema: crate::LashSchema::object(
                properties,
                vec![
                    "line".to_string(),
                    "stream".to_string(),
                    "timestamp".to_string(),
                ],
            ),
            semantics: crate::ProcessEventSemanticsSpec {
                wake: Some(crate::ProcessWakeSpec {
                    when: Some(crate::ProcessValueSelector::Present(
                        "/wake_input".to_string(),
                    )),
                    input: crate::ProcessValueSelector::Pointer("/wake_input".to_string()),
                    dedupe_key: crate::ProcessWakeDedupeKey::EventIdentity,
                }),
                ..crate::ProcessEventSemanticsSpec::default()
            },
        };
        let managed_spec = crate::ProcessRegistration::new(
            process_id.clone(),
            "monitor",
            crate::ProcessScope {
                session_id: session_id.to_string(),
            },
            crate::ProcessInput::Command {
                command: spec.command.clone(),
                cwd: spec.cwd.clone(),
                env: spec.env.clone(),
                timeout_ms: spec.timeout_ms,
                persistent: spec.persistent,
            },
        )
        .with_extra_event_types([line_event])
        .with_tags(["monitor"])
        .with_metadata(serde_json::json!({
            "monitor_id": spec.id,
            "wake_policy": spec.wake_policy,
        }))
        .with_handle_visible(true)
        .with_cancel_policy(crate::ProcessCancelPolicy::Cooperative)
        .with_close_policy(crate::ProcessClosePolicy::Cancel);
        match host.start_process(session_id, managed_spec).await {
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
        if let Err(err) = self.ensure_running(session_id, ctx.host.clone()).await {
            return ToolResult::err_fmt(err.to_string());
        }
        let processes = match ctx.host.list_processes(session_id).await {
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
            .start_monitor_task(session_id, ctx.host.clone(), entry_spec)
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
        let Some(_session_id) = ctx.session_id.as_deref() else {
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

        let process_id = format!("monitor:{}", parsed.id);
        if let Err(err) = ctx
            .host
            .cancel_process(ctx.session_id.as_deref().unwrap_or_default(), &process_id)
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
        entry.runtime_pid = None;
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
                runtime_pid: None,
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
    fn status_projection_reads_process_terminal_state() {
        let spec = MonitorSpec {
            id: "slow".to_string(),
            command: "sleep 5".to_string(),
            persistent: false,
            timeout_ms: 50,
            ..Default::default()
        };
        let state = seeded_monitor_state(&spec);
        let mut record = crate::ProcessRecord::local_session(
            "root",
            "monitor:slow",
            "monitor",
            crate::ProcessState::Failed,
        );
        record.tags = vec!["monitor".to_string()];
        record.terminal = Some(crate::ProcessTerminalSemantics {
            state: crate::ProcessTerminalState::Failed,
            await_output: crate::ProcessAwaitOutput::Failure {
                class: crate::ToolFailureClass::Execution,
                code: "process_command_timeout".to_string(),
                message: "monitor timed out after 50ms".to_string(),
                raw: None,
                control: None,
            },
        });

        let guard = state.lock().expect("monitor state");
        let snapshot = MonitorPlugin::snapshot_from_state_and_processes(&guard, &[record]);

        assert_eq!(snapshot.monitors[0].state, MonitorRunState::Failed);
        assert_eq!(
            snapshot.monitors[0].last_error.as_deref(),
            Some("monitor timed out after 50ms")
        );
    }
}
