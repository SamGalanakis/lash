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
        description: "Start a background monitor that streams events from a long-running script. Each stdout line is an event that arrives automatically as new turn input — **you do not call any tool to collect events**. Return your turn after starting the monitor and continue with other work; a new turn will be triggered when the monitor emits its first line. Events appear as user-like input but are not replies from the user.\n\n**Do not use `wait_agent` to collect monitor events** — `wait_agent` only receives subagent lifecycle events, never monitor output. There is no monitor-event sink tool; the runtime delivers events for you.\n\nMonitor is for the **streaming** case: \"tell me every time X happens.\" For one-shot \"wait until X is done,\" run the command synchronously instead.\n\nYour script's stdout is the event stream. Each line becomes a turn wake-up. Exit ends the watch.\n\n  # Each matching log line is an event\n  tail -f /var/log/app.log | grep --line-buffered \"ERROR\"\n\n  # Each file change is an event\n  inotifywait -m --format '%e %f' /watched/dir\n\n  # Poll for new PR comments and emit one line per new comment\n  last=$(date -u +%Y-%m-%dT%H:%M:%SZ)\n  while true; do\n    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)\n    gh api \"repos/owner/repo/issues/123/comments?since=$last\" --jq '.[] | \"\\(.user.login): \\(.body)\"'\n    last=$now; sleep 30\n  done\n\n**Script quality:**\n- Always use `grep --line-buffered` in pipes — without it, pipe buffering delays events by minutes.\n- In poll loops, handle transient failures (`curl ... || true`) — one failed request shouldn't kill the monitor.\n- Poll intervals: 30s+ for remote APIs (rate limits), 0.5-1s for local checks.\n- Write a specific `description` — it appears in every notification (\"errors in deploy.log\" not \"watching logs\").\n- Only stdout is the event stream. Stderr does not trigger notifications — for a command you run directly (e.g. `python train.py 2>&1 | grep --line-buffered ...`), merge stderr with `2>&1` so its failures reach your filter.\n\n**Coverage — silence is not success.** When watching a job or process for an outcome, your filter must match every terminal state, not just the happy path. A monitor that greps only for the success marker stays silent through a crashloop, a hung process, or an unexpected exit — and silence looks identical to \"still running.\" Before arming, ask: *if this process crashed right now, would my filter emit anything?* If not, widen it.\n\n  # Wrong — silent on crash, hang, or any non-success exit\n  tail -f run.log | grep --line-buffered \"elapsed_steps=\"\n\n  # Right — one alternation covering progress + the failure signatures you'd act on\n  tail -f run.log | grep -E --line-buffered \"elapsed_steps=|Traceback|Error|FAILED|assert|Killed|OOM\"\n\nFor poll loops checking job state, emit on every terminal status (`succeeded|failed|cancelled|timeout`), not just success. If you cannot confidently enumerate the failure signatures, broaden the grep alternation rather than narrow it — some extra noise is better than missing a crashloop.\n\n**Output volume**: Every stdout line becomes a turn wake-up, so the filter should be selective — but selective means \"the lines you'd act on,\" not \"only good news.\" Never pipe raw logs; use `grep --line-buffered`, `awk`, or a wrapper that emits exactly the success and failure signals you care about.\n\nThe script runs in the same shell environment as a normal shell tool call. Exit ends the watch (exit code is reported). Timeout → killed. Set `persistent: true` for session-length watches (PR monitoring, log tails) — the monitor runs until it exits or the session ends.".into(),
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
        enabled: true,
        injected: true,
        input_schema_override: None,
        output_schema_override: None,
        execution_mode: ToolExecutionMode::Parallel,
    }
}

pub fn tasks_list_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "tasks_list".into(),
        description: "List every background task registered for this session — monitors and subagents — with their id, kind, label, and run_state. `run_state` is one of `running` (currently working), `idle` (live subagent waiting for a follow-up task), `completed`, `failed`, or `cancelled`. Use this to see what's still running (or idle and available for `followup_task`) before deciding whether to keep waiting, poll again, or stop something.".into(),
        params: vec![],
        returns: "json".into(),
        examples: vec!["tasks_list()".into()],
        enabled: true,
        injected: true,
        input_schema_override: None,
        output_schema_override: None,
        execution_mode: ToolExecutionMode::Parallel,
    }
}

pub fn tasks_stop_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "tasks_stop".into(),
        description: "Cancel a background task by id. Ids come from `tasks_list` — monitors look like `monitor:<name>` and subagent subtrees look like `subagent:/root/<name>`. For monitors this terminates the process tree; for subagents it closes the subtree (running turns are cancelled, idle sessions closed).".into(),
        params: vec![ToolParam {
            name: "id".into(),
            r#type: "string".into(),
            description: "Task id as returned by `tasks_list`.".into(),
            default_value: None,
            required: true,
        }],
        returns: "json".into(),
        examples: vec![
            "tasks_stop(id=\"monitor:app-errors\")".into(),
            "tasks_stop(id=\"subagent:/root/inspect_auth\")".into(),
        ],
        enabled: true,
        injected: true,
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
                        "id": task.id,
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
        .get("id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return ToolResult::err_fmt("tasks_stop requires `id`");
    };
    match context
        .host
        .cancel_background_task(&context.session_id, id)
        .await
    {
        Ok(status) => ToolResult::ok(serde_json::json!({
            "id": status.id,
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
