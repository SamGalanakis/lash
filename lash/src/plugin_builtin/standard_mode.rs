use std::sync::Arc;

use crate::plugin::{
    ModeNativeToolsPlugin, ModeSessionPlugin, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, SessionPlugin,
};
use crate::tool_dispatch::{
    ParallelToolCallSpec, ToolDispatchContext, dispatch_parallel_tool_calls,
};
use crate::tools::batch::batch_tool_definition;
use crate::{
    ExecutionMode, MAX_MONITOR_TIMEOUT_MS, ManagedRunState, ManagedTaskKind, MonitorRunState,
    MonitorSpec, ProgressSender, SessionError, ToolDefinition, ToolParam, ToolResult,
};

pub(crate) struct StandardModePluginFactory;

impl PluginFactory for StandardModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_standard"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(StandardModePlugin {
            active: matches!(ctx.execution_mode, ExecutionMode::Standard),
        }))
    }
}

struct StandardModePlugin {
    active: bool,
}

impl SessionPlugin for StandardModePlugin {
    fn id(&self) -> &'static str {
        "mode_standard"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        reg.mode().session(Arc::new(StandardModeSession))?;
        reg.mode().native_tools(Arc::new(StandardModeNativeTools))?;
        Ok(())
    }
}

struct StandardModeSession;

#[async_trait::async_trait]
impl ModeSessionPlugin for StandardModeSession {
    async fn initialize_session(
        &self,
        _session: &mut crate::Session,
        _session_id: &str,
    ) -> Result<(), SessionError> {
        Ok(())
    }
}

struct StandardModeNativeTools;

#[async_trait::async_trait]
impl ModeNativeToolsPlugin for StandardModeNativeTools {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![
            batch_tool_definition(),
            monitor_tool_definition(),
            tasks_list_tool_definition(),
            tasks_stop_tool_definition(),
        ]
    }

    async fn execute(
        &self,
        context: &ToolDispatchContext,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> Option<ToolResult> {
        match name {
            "batch" => Some(execute_batch_tool_call(context, args, progress).await),
            "monitor" => Some(execute_monitor_tool_call(context, args).await),
            "tasks_list" => Some(execute_tasks_list_tool_call(context).await),
            "tasks_stop" => Some(execute_tasks_stop_tool_call(context, args).await),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct BatchCallSpec {
    index: usize,
    tool: String,
    parameters: serde_json::Value,
}

const BATCH_MAX_TOOL_CALLS: usize = 25;

fn monitor_tool_definition() -> ToolDefinition {
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
    }
}

async fn execute_batch_tool_call(
    context: &ToolDispatchContext,
    args: &serde_json::Value,
    progress: Option<&ProgressSender>,
) -> ToolResult {
    let specs = match parse_batch_specs(args) {
        Ok(specs) => specs,
        Err(err) => return err,
    };

    let mut immediate_outcomes = Vec::new();
    let mut parallel_specs = Vec::new();

    for spec in specs.into_iter().take(BATCH_MAX_TOOL_CALLS) {
        if spec.tool == "batch" {
            immediate_outcomes.push(serde_json::json!({
                "index": spec.index,
                "tool": spec.tool,
                "success": false,
                "duration_ms": 0,
                "error": "Tool 'batch' is not allowed inside batch",
            }));
            continue;
        }
        parallel_specs.push(ParallelToolCallSpec {
            index: spec.index,
            tool_name: spec.tool,
            args: spec.parameters,
        });
    }

    let mut images = Vec::new();
    let mut parallel_outcomes =
        dispatch_parallel_tool_calls(Arc::new(context.clone()), parallel_specs, progress).await;
    for outcome in parallel_outcomes.drain(..) {
        images.extend(outcome.images);
        let mut record = serde_json::Map::new();
        record.insert("index".to_string(), serde_json::json!(outcome.index));
        record.insert("tool".to_string(), serde_json::json!(outcome.record.tool));
        record.insert(
            "success".to_string(),
            serde_json::json!(outcome.record.success),
        );
        record.insert(
            "duration_ms".to_string(),
            serde_json::json!(outcome.record.duration_ms),
        );
        record.insert(
            if outcome.record.success {
                "result".to_string()
            } else {
                "error".to_string()
            },
            outcome.record.result,
        );
        immediate_outcomes.push(serde_json::Value::Object(record));
    }

    for overflow_index in BATCH_MAX_TOOL_CALLS
        ..args
            .get("tool_calls")
            .and_then(|value| value.as_array())
            .map(|value| value.len())
            .unwrap_or_default()
    {
        immediate_outcomes.push(serde_json::json!({
            "index": overflow_index,
            "tool": args
                .get("tool_calls")
                .and_then(|value| value.as_array())
                .and_then(|items| items.get(overflow_index))
                .and_then(|item| item.get("tool"))
                .and_then(|value| value.as_str())
                .unwrap_or("unknown"),
            "success": false,
            "duration_ms": 0,
            "error": "Maximum of 25 tool calls allowed in batch",
        }));
    }

    immediate_outcomes.sort_by_key(|outcome| {
        outcome
            .get("index")
            .and_then(|value| value.as_u64())
            .unwrap_or(u64::MAX)
    });
    ToolResult::with_images(
        true,
        serde_json::json!({
            "results": immediate_outcomes,
        }),
        images,
    )
}

fn parse_batch_specs(args: &serde_json::Value) -> Result<Vec<BatchCallSpec>, ToolResult> {
    let Some(raw_calls) = args.get("tool_calls").and_then(|value| value.as_array()) else {
        return Err(ToolResult::err_fmt(
            "Missing required parameter: tool_calls",
        ));
    };
    if raw_calls.is_empty() {
        return Err(ToolResult::err_fmt(
            "Invalid tool_calls: expected at least one call",
        ));
    }

    let mut specs = Vec::with_capacity(raw_calls.len());
    for (index, item) in raw_calls.iter().enumerate() {
        let Some(object) = item.as_object() else {
            return Err(ToolResult::err_fmt(format_args!(
                "Invalid tool_calls[{index}]: expected object with tool and parameters"
            )));
        };
        let Some(tool) = object
            .get("tool")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|tool| !tool.is_empty())
        else {
            return Err(ToolResult::err_fmt(format_args!(
                "Invalid tool_calls[{index}].tool: expected non-empty string"
            )));
        };
        let parameters = object
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        specs.push(BatchCallSpec {
            index,
            tool: tool.to_string(),
            parameters,
        });
    }

    Ok(specs)
}

async fn execute_monitor_tool_call(
    context: &ToolDispatchContext,
    args: &serde_json::Value,
) -> ToolResult {
    let Some(command) = args
        .get("command")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return ToolResult::err_fmt("monitor requires `command`");
    };
    let Some(description) = args
        .get("description")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return ToolResult::err_fmt("monitor requires `description`");
    };

    let persistent = args
        .get("persistent")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let timeout_ms = args
        .get("timeout_ms")
        .and_then(|value| value.as_u64())
        .unwrap_or(300_000);
    if timeout_ms > MAX_MONITOR_TIMEOUT_MS {
        return ToolResult::err_fmt(format_args!(
            "monitor timeout_ms must be <= {MAX_MONITOR_TIMEOUT_MS}"
        ));
    }

    let id = uuid::Uuid::new_v4().simple().to_string();
    let spec = MonitorSpec {
        id: id.clone(),
        label: description.to_string(),
        command: command.to_string(),
        cwd: None,
        env: Default::default(),
        persistent,
        timeout_ms,
        arm_on: Default::default(),
        wake_policy: Default::default(),
        restart_on_restore: persistent,
    };
    match context.host.start_monitor(&context.session_id, spec).await {
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

fn tasks_list_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "tasks_list".into(),
        description: "List every background task registered for this session — monitors and subagents — with their id, kind, label, and run_state. Use this to see what's still running before deciding whether to keep waiting, poll again, or stop something.".into(),
        params: vec![],
        returns: "json".into(),
        examples: vec!["tasks_list()".into()],
        enabled: true,
        injected: true,
        input_schema_override: None,
        output_schema_override: None,
    }
}

fn tasks_stop_tool_definition() -> ToolDefinition {
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
    }
}

fn run_state_label(state: ManagedRunState) -> &'static str {
    match state {
        ManagedRunState::Running => "running",
        ManagedRunState::Completed => "completed",
        ManagedRunState::Failed => "failed",
        ManagedRunState::Cancelled => "cancelled",
    }
}

fn kind_label(kind: ManagedTaskKind) -> &'static str {
    kind.as_str()
}

async fn execute_tasks_list_tool_call(context: &ToolDispatchContext) -> ToolResult {
    match context
        .host
        .list_background_tasks(&context.session_id)
        .await
    {
        Ok(tasks) => {
            let entries: Vec<serde_json::Value> = tasks
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

async fn execute_tasks_stop_tool_call(
    context: &ToolDispatchContext,
    args: &serde_json::Value,
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
