//! Mode-agnostic runtime-control tools (`monitor`, `list_process_handles`,
//! `cancel_process`).
//!
//! Dedicated plugins register these tools into the native-tools surface,
//! so mode crates do not own or duplicate runtime control behavior. RLM
//! hides process-control tools when a mode exposes the same control through
//! native process handles.

use std::sync::Arc;

use serde_json::Value;

use crate::plugin::{
    ModeNativeToolsPlugin, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use crate::tool_dispatch::ToolDispatchContext;
use crate::{
    MAX_MONITOR_TIMEOUT_MS, MonitorRunState, MonitorSpec, ProcessState, ProgressSender,
    ToolContract, ToolDefinition, ToolExecutionMode, ToolManifest, ToolResult,
};

/// Plugin factory for mode-agnostic process-control tools.
#[derive(Default)]
pub struct BuiltinProcessControlsPluginFactory;

impl BuiltinProcessControlsPluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl PluginFactory for BuiltinProcessControlsPluginFactory {
    fn id(&self) -> &'static str {
        "process_controls"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ProcessControlsPlugin {
            enabled: ctx.execution_mode != crate::ExecutionMode::new("rlm"),
        }))
    }
}

struct ProcessControlsPlugin {
    enabled: bool,
}

impl SessionPlugin for ProcessControlsPlugin {
    fn id(&self) -> &'static str {
        "process_controls"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if self.enabled {
            reg.mode()
                .native_tools(Arc::new(ProcessControlsNativeTools))?;
        }
        Ok(())
    }
}

struct ProcessControlsNativeTools;

#[async_trait::async_trait]
impl ModeNativeToolsPlugin for ProcessControlsNativeTools {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        process_control_tool_definitions()
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        process_control_tool_definitions()
            .into_iter()
            .find(|tool| tool.name == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    async fn execute(
        &self,
        context: &ToolDispatchContext<'_>,
        name: &str,
        args: &Value,
        _progress: Option<&ProgressSender>,
    ) -> Option<ToolResult> {
        match name {
            "list_process_handles" => Some(execute_process_list_tool_call(context).await),
            "cancel_process" => Some(execute_process_cancel_tool_call(context, args).await),
            _ => None,
        }
    }
}

/// Plugin factory for the shell-backed `monitor` tool.
#[derive(Default)]
pub struct BuiltinMonitorToolPluginFactory;

impl BuiltinMonitorToolPluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl PluginFactory for BuiltinMonitorToolPluginFactory {
    fn id(&self) -> &'static str {
        "monitor_tool"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(MonitorToolPlugin))
    }
}

struct MonitorToolPlugin;

impl SessionPlugin for MonitorToolPlugin {
    fn id(&self) -> &'static str {
        "monitor_tool"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.mode().native_tools(Arc::new(MonitorNativeTool))
    }
}

struct MonitorNativeTool;

#[async_trait::async_trait]
impl ModeNativeToolsPlugin for MonitorNativeTool {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![monitor_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        (name == "monitor").then(|| Arc::new(monitor_tool_definition().contract()))
    }

    async fn execute(
        &self,
        context: &ToolDispatchContext<'_>,
        name: &str,
        args: &Value,
        _progress: Option<&ProgressSender>,
    ) -> Option<ToolResult> {
        match name {
            "monitor" => {
                let spec = match MonitorToolSpec::from_args(args) {
                    Ok(spec) => spec,
                    Err(result) => return Some(result),
                };
                Some(execute_monitor_tool_call(context, spec).await)
            }
            _ => None,
        }
    }
}

/// Build the `monitor` tool definition.
pub fn monitor_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "monitor",
        "Run a background script that turns each stdout line into a process event and optional turn wake-up. Use for streaming watches (`tell me every time X happens`); for one-shot `wait until X`, run the command synchronously instead. This returns a process handle; use `list_process_handles` to rediscover live monitors and `cancel handle` to stop one.\n\nEvents arrive automatically as user-like input — do not call another tool to collect them. Return your turn after starting the monitor; the runtime wakes a new turn on the first matching line.\n\n**Pipe guards**\n- Always use `grep --line-buffered` in pipes (otherwise pipe buffering delays events by minutes).\n- Merge stderr into stdout (`cmd 2>&1 | grep ...`) — stderr alone is not observed.\n- In poll loops wrap transient failures (`curl ... || true`) and pick intervals ≥30s for remote APIs, 0.5–1s for local checks.\n\n**Coverage — silence is not success.** Your filter must match every terminal state, not just the happy path. A monitor that greps only for the success marker stays silent through a crashloop, a hang, or an unexpected exit — and silence looks identical to `still running`. If you can't enumerate the failure signatures, broaden the alternation rather than narrow it.\n\nSet `persistent: true` for session-length watches. Timeout → killed; exit ends the watch (exit code is reported).",
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command or script. Each stdout line is an event; exit ends the watch. Filter with `grep --line-buffered` (or equivalent) so only the lines you'd act on become events — including failure signatures, not just success."
                },
                "description": {
                    "type": "string",
                    "description": "Short human-readable description of what you are monitoring (shown in every notification). Be specific — \"errors in deploy.log\" beats \"watching logs\"."
                },
                "persistent": {
                    "type": "boolean",
                    "default": false,
                    "description": "Run for the lifetime of the session (no timeout). Use for session-length watches like PR monitoring or log tails."
                },
                "timeout_ms": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": MAX_MONITOR_TIMEOUT_MS,
                    "default": 300000,
                    "description": "Kill the monitor after this deadline. Default 300000ms, max 3600000ms. Ignored when persistent is true."
                }
            },
            "required": ["command", "description"],
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
        r#"monitor(command="tail -f /var/log/app.log | grep -E --line-buffered 'ERROR|Traceback|FAILED'", description="errors in app.log")"#.into(),
        r#"monitor(command="while true; do curl -sf http://localhost:3000/health && echo ready && break; sleep 2; done", description="local server ready", timeout_ms=300000)"#.into(),
    ])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

pub fn process_list_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "list_process_handles",
        "List every model-visible process handle in this session with its process id, producer, tags, and state.",
        ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec!["list_process_handles()".into()])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn process_control_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        process_list_tool_definition(),
        process_cancel_tool_definition(),
    ]
}

pub fn process_cancel_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "cancel_process",
        "Request cancellation for a durable process by `process_id`.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "string",
                    "description": "Process id returned by a process handle or `list_process_handles`."
                }
            },
            "required": ["process_id"],
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
        r#"cancel_process(process_id="monitor:app-errors")"#.into(),
        r#"cancel_process(process_id="subagent:session-01JZK7G4QP9Q4J7W3Q2E1H6M9C")"#.into(),
    ])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

/// Parsed `monitor` arguments ready to hand to the session manager.
pub struct MonitorToolSpec {
    pub command: String,
    pub description: String,
    pub persistent: bool,
    pub timeout_ms: u64,
}

impl MonitorToolSpec {
    #[allow(clippy::result_large_err)]
    pub fn from_args(args: &Value) -> Result<Self, ToolResult> {
        let command = args
            .get("command")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ToolResult::err_fmt("monitor requires `command`"))?;
        let description = args
            .get("description")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ToolResult::err_fmt("monitor requires `description`"))?;
        let persistent = args
            .get("persistent")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(300_000);
        if timeout_ms > MAX_MONITOR_TIMEOUT_MS {
            return Err(ToolResult::err_fmt(format_args!(
                "monitor timeout_ms must be <= {MAX_MONITOR_TIMEOUT_MS}"
            )));
        }
        Ok(Self {
            command: command.to_string(),
            description: description.to_string(),
            persistent,
            timeout_ms,
        })
    }
}

pub async fn execute_monitor_tool_call(
    context: &ToolDispatchContext<'_>,
    spec: MonitorToolSpec,
) -> ToolResult {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let monitor_spec = MonitorSpec {
        id: id.clone(),
        command: spec.command,
        cwd: None,
        env: Default::default(),
        persistent: spec.persistent,
        timeout_ms: spec.timeout_ms,
        arm_on: Default::default(),
        wake_policy: Default::default(),
        restart_on_restore: spec.persistent,
    };
    match context
        .host
        .start_monitor(&context.session_id, monitor_spec)
        .await
    {
        Ok(snapshot) => {
            let started = snapshot
                .monitors
                .iter()
                .find(|status| status.spec.id == id)
                .cloned();
            match started {
                Some(status) => ToolResult::ok(serde_json::json!({
                    "process_id": format!("monitor:{}", status.spec.id),
                    "monitor_id": status.spec.id,
                    "description": status.spec.command,
                    "command": status.spec.command,
                    "persistent": status.spec.persistent,
                    "timeout_ms": status.spec.timeout_ms,
                    "state": match status.state {
                        MonitorRunState::Idle => "idle",
                        MonitorRunState::Running => "running",
                        MonitorRunState::Stopped => "stopped",
                        MonitorRunState::Exited => "exited",
                        MonitorRunState::Failed => "failed",
                    },
                })),
                None => ToolResult::err_fmt("monitor started but status was unavailable"),
            }
        }
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

pub async fn execute_process_list_tool_call(context: &ToolDispatchContext<'_>) -> ToolResult {
    match context
        .host
        .list_processes_scoped(
            &context.session_id,
            context.tool_effect_metadata.clone(),
            Some(context.effect_controller.as_controller()),
        )
        .await
    {
        Ok(processes) => {
            let entries: Vec<Value> = processes
                .into_iter()
                .filter(|process| process.handle_visible)
                .map(|process| {
                    let created_at_iso = chrono::DateTime::<chrono::Utc>::from(process.created_at)
                        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
                    serde_json::json!({
                        "process_id": process.id,
                        "producer": process.producer,
                        "tags": process.tags,
                        "state": state_label(process.state),
                        "created_at": created_at_iso,
                    })
                })
                .collect();
            ToolResult::ok(serde_json::json!({ "processes": entries }))
        }
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

pub async fn execute_process_cancel_tool_call(
    context: &ToolDispatchContext<'_>,
    args: &Value,
) -> ToolResult {
    let Some(id) = args
        .get("process_id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return ToolResult::err_fmt("cancel_process requires `process_id`");
    };
    match context
        .host
        .cancel_process_scoped(
            &context.session_id,
            id,
            context.tool_effect_metadata.clone(),
            Some(context.effect_controller.as_controller()),
        )
        .await
    {
        Ok(status) => ToolResult::ok(serde_json::json!({
            "process_id": status.id,
            "producer": status.producer,
            "state": state_label(status.state),
        })),
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

fn state_label(state: ProcessState) -> &'static str {
    match state {
        ProcessState::Pending => "pending",
        ProcessState::Scheduled => "scheduled",
        ProcessState::Running => "running",
        ProcessState::Waiting => "idle",
        ProcessState::Completed => "completed",
        ProcessState::Failed => "failed",
        ProcessState::CancelRequested => "cancel_requested",
        ProcessState::Cancelled => "cancelled",
    }
}
