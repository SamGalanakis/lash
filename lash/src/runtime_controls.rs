//! Mode-agnostic runtime-control tools (`monitor`, `tasks_list`,
//! `tasks_stop`).
//!
//! These tool definitions + implementations are shared between
//! `lash-mode-standard` and `lash-mode-rlm`: both modes expose them in
//! their native-tools surface so the model can inspect and cancel
//! background subagents and monitors regardless of how it drives the
//! rest of its turn.

use serde_json::Value;

use crate::tool_dispatch::ToolDispatchContext;
use crate::{
    MAX_MONITOR_TIMEOUT_MS, ManagedRunState, ManagedTaskKind, MonitorRunState, MonitorSpec,
    ToolDefinition, ToolExecutionMode, ToolParam, ToolResult,
};

/// Build the `monitor` tool definition. Injected by mode plugins.
pub fn monitor_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "monitor".into(),
        description: "Run a background script that turns each stdout line into a new turn wake-up. Use for streaming watches (`tell me every time X happens`); for one-shot `wait until X`, just run the command synchronously instead. Returns a `task_id` that `tasks_stop` accepts.\n\nEvents arrive automatically as user-like input — do not call `wait_agent` or any other tool to collect them. Return your turn after starting the monitor; the runtime wakes a new turn on the first line.\n\n**Pipe guards**\n- Always use `grep --line-buffered` in pipes (otherwise pipe buffering delays events by minutes).\n- Merge stderr into stdout (`cmd 2>&1 | grep ...`) — stderr alone is not observed.\n- In poll loops wrap transient failures (`curl ... || true`) and pick intervals ≥30s for remote APIs, 0.5–1s for local checks.\n\n**Coverage — silence is not success.** Your filter must match every terminal state, not just the happy path. A monitor that greps only for the success marker stays silent through a crashloop, a hang, or an unexpected exit — and silence looks identical to `still running`. If you can't enumerate the failure signatures, broaden the alternation rather than narrow it.\n\nWrong: `tail -f run.log | grep --line-buffered \"elapsed_steps=\"`\nRight: `tail -f run.log | grep -E --line-buffered \"elapsed_steps=|Traceback|Error|FAILED|Killed|OOM\"`\n\nSet `persistent: true` for session-length watches. Timeout → killed; exit ends the watch (exit code is reported).".into(),
        params: vec![
            ToolParam {
                name: "command".into(),
                r#type: "string".into(),
                description:
                    "Shell command or script. Each stdout line is an event; exit ends the watch. Filter with `grep --line-buffered` (or equivalent) so only the lines you'd act on become events — including failure signatures, not just success."
                        .into(),
                default_value: None,
                required: true,
            },
            ToolParam {
                name: "description".into(),
                r#type: "string".into(),
                description: "Short human-readable description of what you are monitoring (shown in every notification). Be specific — \"errors in deploy.log\" beats \"watching logs\".".into(),
                default_value: None,
                required: true,
            },
            ToolParam {
                name: "persistent".into(),
                r#type: "bool".into(),
                description:
                    "Run for the lifetime of the session (no timeout). Use for session-length watches like PR monitoring or log tails."
                        .into(),
                default_value: Some(serde_json::json!(false)),
                required: false,
            },
            ToolParam {
                name: "timeout_ms".into(),
                r#type: "number".into(),
                description:
                    "Kill the monitor after this deadline. Default 300000ms, max 3600000ms. Ignored when persistent is true."
                        .into(),
                default_value: Some(serde_json::json!(300000)),
                required: false,
            },
        ],
        returns: "json".into(),
        examples: vec![
            "monitor(command=\"tail -f /var/log/app.log | grep -E --line-buffered 'ERROR|Traceback|FAILED'\", description=\"errors in app.log\")".into(),
            "monitor(command=\"while true; do curl -sf http://localhost:3000/health && echo ready && break; sleep 2; done\", description=\"local server ready\", timeout_ms=300000)".into(),
        ],
        availability: crate::ToolAvailabilityConfig::documented(),
        activation: crate::ToolActivation::Always,
        availability_override: None,
        input_schema_override: None,
        output_schema_override: None,
        execution_mode: ToolExecutionMode::Parallel,
    }
}

pub fn tasks_list_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "tasks_list".into(),
        description: "List every background task registered for this session — monitors and subagents — with their `task_id`, kind, label, and run_state. `run_state` is one of `running` (currently working), `idle` (live subagent waiting for a follow-up task), `completed`, `failed`, or `cancelled`. Use this to see what's still running (or idle and available for `followup_task`) before deciding whether to keep waiting, poll again, or stop something.".into(),
        params: vec![],
        returns: "json".into(),
        examples: vec!["tasks_list()".into()],
        availability: crate::ToolAvailabilityConfig::documented(),
        activation: crate::ToolActivation::Always,
        availability_override: None,
        input_schema_override: None,
        output_schema_override: None,
        execution_mode: ToolExecutionMode::Parallel,
    }
}

pub fn tasks_stop_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "tasks_stop".into(),
        description: "Cancel a background task by `task_id`. Monitors and subagents return this id when started; `tasks_list` can also rediscover it. For monitors this terminates the process tree; for subagents it closes the subtree (running turns are cancelled, idle sessions closed).".into(),
        params: vec![ToolParam {
            name: "task_id".into(),
            r#type: "string".into(),
            description: "Task id returned by `monitor`, `spawn_agent`, or `tasks_list`.".into(),
            default_value: None,
            required: true,
        }],
        returns: "json".into(),
        examples: vec![
            "tasks_stop(task_id=\"monitor:app-errors\")".into(),
            "tasks_stop(task_id=\"subagent:/root/inspect_auth\")".into(),
        ],
        availability: crate::ToolAvailabilityConfig::documented(),
        activation: crate::ToolActivation::Always,
        availability_override: None,
        input_schema_override: None,
        output_schema_override: None,
        execution_mode: ToolExecutionMode::Parallel,
    }
}

/// Parsed `monitor` arguments ready to hand to the session manager.
pub struct MonitorToolSpec {
    pub command: String,
    pub description: String,
    pub persistent: bool,
    pub timeout_ms: u64,
}

impl MonitorToolSpec {
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
    context: &ToolDispatchContext,
    spec: MonitorToolSpec,
) -> ToolResult {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let monitor_spec = MonitorSpec {
        id: id.clone(),
        label: spec.description,
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
                    "task_id": format!("monitor:{}", status.spec.id),
                    "monitor_id": status.spec.id,
                    "description": status.spec.label,
                    "command": status.spec.command,
                    "persistent": status.spec.persistent,
                    "timeout_ms": status.spec.timeout_ms,
                    "run_state": match status.run_state {
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

pub async fn execute_tasks_list_tool_call(context: &ToolDispatchContext) -> ToolResult {
    match context
        .host
        .list_background_tasks(&context.session_id)
        .await
    {
        Ok(tasks) => {
            let entries: Vec<Value> = tasks
                .into_iter()
                .map(|task| {
                    let started_at_iso = chrono::DateTime::<chrono::Utc>::from(task.started_at)
                        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
                    serde_json::json!({
                        "task_id": task.id,
                        "label": task.label,
                        "kind": kind_label(task.kind),
                        "producer": task.producer,
                        "run_state": run_state_label(task.run_state),
                        "started_at": started_at_iso,
                    })
                })
                .collect();
            ToolResult::ok(serde_json::json!({ "tasks": entries }))
        }
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

pub async fn execute_tasks_stop_tool_call(
    context: &ToolDispatchContext,
    args: &Value,
) -> ToolResult {
    let Some(id) = args
        .get("task_id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return ToolResult::err_fmt("tasks_stop requires `task_id`");
    };
    match context
        .host
        .cancel_background_task(&context.session_id, id)
        .await
    {
        Ok(status) => ToolResult::ok(serde_json::json!({
            "task_id": status.id,
            "kind": kind_label(status.kind),
            "run_state": run_state_label(status.run_state),
        })),
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

fn run_state_label(state: ManagedRunState) -> &'static str {
    match state {
        ManagedRunState::Running => "running",
        ManagedRunState::Idle => "idle",
        ManagedRunState::Completed => "completed",
        ManagedRunState::Failed => "failed",
        ManagedRunState::Cancelled => "cancelled",
    }
}

fn kind_label(kind: ManagedTaskKind) -> &'static str {
    kind.as_str()
}
