use std::io::{self, BufRead, Write};

use futures_util::StreamExt as _;
use lash::{
    LashSession, TurnActivity, TurnEvent, TurnInput,
    observe::{SessionObservationEventPayload, SessionObservationStreamItem},
    usage::{SessionUsageReport, TokenLedgerEntry, diff_usage_reports},
};
use lash_core::TurnOutcome;

use crate::SkillCatalog;
use crate::app::PreparedTurn;
use crate::turn_runner::{make_turn_input, spawn_session_turn};
use crate::util;

/// Version stamped into the autonomous JSONL surface: the `turn_start` frame in
/// Json mode and the `ready` frame in Rpc mode both carry it under
/// `protocol_version`. Bump when the record shapes below change incompatibly;
/// the docs describe the surface contract keyed off this constant.
pub const AUTONOMOUS_JSON_VERSION: u32 = 1;

/// Human-facing status token for a completed tool call in Print mode.
fn tool_status_label(status: lash_core::ToolCallStatus) -> &'static str {
    match status {
        lash_core::ToolCallStatus::Success => "ok",
        lash_core::ToolCallStatus::Failure => "error",
        lash_core::ToolCallStatus::Cancelled => "cancelled",
    }
}

/// Render the Print-mode `[tool]` summary line for a completed tool call.
/// Sub-second durations are elided to keep the stream quiet (see
/// [`util::format_duration_ms_if_visible`]).
pub(crate) fn format_tool_line(
    name: &str,
    status: lash_core::ToolCallStatus,
    duration_ms: u64,
) -> String {
    let status = tool_status_label(status);
    match util::format_duration_ms_if_visible(duration_ms) {
        Some(duration_text) => format!("[tool] {name} · {status} · {duration_text}"),
        None => format!("[tool] {name} · {status}"),
    }
}

pub(crate) struct AutonomousPersistenceContext {
    pub(crate) await_background_work: bool,
    pub(crate) turn_usage_json: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutonomousMode {
    Print,
    Json,
    Rpc,
}

pub(crate) struct AutonomousRenderer {
    pub(crate) streamed_text: bool,
    pub(crate) wrote_stdout: bool,
    pub(crate) stdout_text: String,
}

impl AutonomousRenderer {
    pub(crate) fn new() -> Self {
        Self {
            streamed_text: false,
            wrote_stdout: false,
            stdout_text: String::new(),
        }
    }

    pub(crate) fn handle(&mut self, activity: TurnActivity) -> Result<(), String> {
        match activity.event {
            TurnEvent::AssistantProseDelta { text } => {
                if !text.is_empty() {
                    self.streamed_text = true;
                    self.wrote_stdout = true;
                    self.stdout_text.push_str(&text);
                    print!("{text}");
                    let _ = io::stdout().flush();
                }
            }
            TurnEvent::ToolCallCompleted {
                name,
                output,
                duration_ms,
                ..
            } => {
                eprintln!("{}", format_tool_line(&name, output.status(), duration_ms));
            }
            TurnEvent::ModelRequestStarted { protocol_iteration } => {
                eprintln!("[thinking] step {}", protocol_iteration + 1);
            }
            TurnEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
            } => {
                eprintln!(
                    "[retry] in {}s · attempt {}/{} · {}",
                    wait_seconds, attempt, max_attempts, reason
                );
            }
            TurnEvent::Error { message } => {
                eprintln!("error: {message}");
            }
            TurnEvent::PluginRuntime { .. } => {}
            // Reasoning summaries are a TUI-only affordance; in
            // autonomous mode we deliberately discard them to keep stdout
            // aligned with the model's final answer.
            TurnEvent::ReasoningDelta { .. } => {}
            TurnEvent::ToolCallStarted { .. }
            | TurnEvent::QueuedWorkStarted { .. }
            | TurnEvent::Usage { .. }
            | TurnEvent::ChildUsage { .. }
            | TurnEvent::QueuedInputAccepted { .. }
            | TurnEvent::QueuedMessagesCommitted { .. }
            | TurnEvent::CodeBlockStarted { .. }
            | TurnEvent::CodeBlockCompleted { .. }
            | TurnEvent::FinalValue { .. }
            | TurnEvent::ToolValue { .. } => {}
        }
        Ok(())
    }

    pub(crate) fn rendered_plugin_output(&self) -> Option<String> {
        None
    }

    pub(crate) fn finish_output(&mut self, final_text: &str) {
        if final_text.is_empty() {
            return;
        }

        let remainder = if self.stdout_text.is_empty() {
            final_text
        } else if final_text.starts_with(&self.stdout_text) {
            &final_text[self.stdout_text.len()..]
        } else if self.stdout_text.ends_with(final_text) {
            ""
        } else {
            final_text
        };

        if remainder.is_empty() {
            if self.wrote_stdout && !self.stdout_text.ends_with('\n') {
                println!();
            }
            return;
        }

        self.wrote_stdout = true;
        self.stdout_text.push_str(remainder);
        print!("{remainder}");
        if !remainder.ends_with('\n') {
            println!();
        }
    }
}

/// Build the Json-mode `turn_start` frame — the first NDJSON record of a
/// `--print --mode json` stream, carrying the [`AUTONOMOUS_JSON_VERSION`] stamp.
fn json_turn_start_record(stream_id: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "turn_start",
        "stream_id": stream_id,
        "protocol_version": AUTONOMOUS_JSON_VERSION,
    })
}

/// Build an `event` frame embedding a [`TurnActivity`] verbatim under
/// `activity`. Rpc streams also echo the originating request `id`.
fn json_event_record(
    stream_id: u64,
    request_id: Option<&serde_json::Value>,
    activity: &TurnActivity,
) -> serde_json::Value {
    let mut record = serde_json::json!({
        "type": "event",
        "stream_id": stream_id,
        "activity": activity,
    });
    if let Some(id) = request_id {
        record["id"] = id.clone();
    }
    record
}

/// Build the terminal `turn_finish` frame pinning the turn's outcome, assistant
/// text, usage, errors, execution summary, and tool-call records.
fn json_finish_record(
    stream_id: u64,
    request_id: Option<&serde_json::Value>,
    turn: &lash::TurnResult,
    cancelled: bool,
) -> serde_json::Value {
    let ok = matches!(
        turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. }
    ) && !cancelled;
    let mut record = serde_json::json!({
        "type": "turn_finish",
        "stream_id": stream_id,
        "ok": ok,
        "cancelled": cancelled,
        "assistant_text": turn.assistant_output.safe_text,
        "outcome": turn.outcome,
        "usage": turn.usage,
        "children_usage": turn.children_usage,
        "errors": turn.errors,
        "execution": turn.execution,
        "tool_calls": turn.tool_calls,
    });
    if let Some(id) = request_id {
        record["id"] = id.clone();
    }
    record
}

struct JsonRenderer {
    stream_id: u64,
    request_id: Option<serde_json::Value>,
    stdout: io::Stdout,
}

impl JsonRenderer {
    fn new(stream_id: u64) -> anyhow::Result<Self> {
        let mut renderer = Self {
            stream_id,
            request_id: None,
            stdout: io::stdout(),
        };
        renderer.write_record(json_turn_start_record(stream_id))?;
        Ok(renderer)
    }

    fn for_rpc(stream_id: u64, request_id: serde_json::Value) -> Self {
        Self {
            stream_id,
            request_id: Some(request_id),
            stdout: io::stdout(),
        }
    }

    fn write_record(&mut self, value: serde_json::Value) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.stdout, &value)?;
        self.stdout.write_all(b"\n")?;
        self.stdout.flush()?;
        Ok(())
    }

    fn handle(&mut self, activity: TurnActivity) -> anyhow::Result<()> {
        let record = json_event_record(self.stream_id, self.request_id.as_ref(), &activity);
        self.write_record(record)
    }

    fn finish(&mut self, turn: &lash::TurnResult, cancelled: bool) -> anyhow::Result<()> {
        let record = json_finish_record(self.stream_id, self.request_id.as_ref(), turn, cancelled);
        self.write_record(record)
    }
}

enum AutonomousOutput {
    Print(AutonomousRenderer),
    Json(JsonRenderer),
}

impl AutonomousOutput {
    fn print() -> Self {
        Self::Print(AutonomousRenderer::new())
    }

    fn json(stream_id: u64) -> anyhow::Result<Self> {
        Ok(Self::Json(JsonRenderer::new(stream_id)?))
    }

    fn handle(&mut self, activity: TurnActivity) -> anyhow::Result<()> {
        match self {
            AutonomousOutput::Print(renderer) => {
                renderer.handle(activity).map_err(anyhow::Error::msg)
            }
            AutonomousOutput::Json(renderer) => renderer.handle(activity),
        }
    }

    fn finish_success(&mut self, turn: &lash::TurnResult, cancelled: bool) -> anyhow::Result<()> {
        match self {
            AutonomousOutput::Print(renderer) => {
                if !turn.assistant_output.safe_text.is_empty() {
                    renderer.finish_output(&turn.assistant_output.safe_text);
                } else if let Some(rendered) = renderer.rendered_plugin_output() {
                    renderer.finish_output(&rendered);
                } else {
                    let raw = turn.assistant_output.raw_text.trim();
                    if raw.is_empty() {
                        eprintln!("error: model returned no usable assistant output");
                    } else {
                        let mut preview: String = raw.chars().take(64).collect();
                        if raw.chars().count() > 64 {
                            preview.push_str("...");
                        }
                        eprintln!("error: model returned malformed assistant output: {preview}");
                    }
                    std::process::exit(2);
                }
                Ok(())
            }
            AutonomousOutput::Json(renderer) => renderer.finish(turn, cancelled),
        }
    }

    fn finish_failure(&mut self, turn: &lash::TurnResult, cancelled: bool) -> anyhow::Result<()> {
        match self {
            AutonomousOutput::Print(_) => {
                for issue in &turn.errors {
                    eprintln!("error: {}", issue.message);
                }
                if turn.errors.is_empty() {
                    eprintln!("error: autonomous turn failed");
                }
                Ok(())
            }
            AutonomousOutput::Json(renderer) => renderer.finish(turn, cancelled),
        }
    }
}

struct AutonomousTurnOutcome {
    done: crate::turn_runner::RuntimeRunResult,
    cancel: lash::CancellationToken,
}

async fn run_autonomous_turn(
    session: LashSession,
    turn_input: TurnInput,
    output: &mut AutonomousOutput,
    stream_id: u64,
) -> anyhow::Result<AutonomousTurnOutcome> {
    let observable = session.observe();
    let cursor = observable.current_observation().cursor;
    let mut observation = Some(observable.subscribe_and_recover(cursor));
    let (cancel, return_rx) = spawn_session_turn(session, turn_input, stream_id);
    #[cfg(unix)]
    {
        let cancel = cancel.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{SignalKind, signal};
            if let Ok(mut sig) = signal(SignalKind::terminate()) {
                sig.recv().await;
                cancel.cancel();
            }
        });
    }
    let mut task = tokio::spawn(async move { (return_rx.await, cancel) });
    let mut returned = None;
    let mut observation_finished = observation.is_none();
    let mut observation_grace_deadline: Option<tokio::time::Instant> = None;
    loop {
        if returned.is_some() && observation_finished {
            break;
        }
        tokio::select! {
            next = async {
                observation.as_mut()?.next().await
            }, if !observation_finished => {
                match next {
                    Some(Ok(SessionObservationStreamItem::Event(event))) => match event.payload {
                        SessionObservationEventPayload::TurnActivity(activity) => {
                            match output.handle(activity) {
                                Ok(()) => {}
                                Err(err) => {
                                    eprintln!("error: {err}");
                                    std::process::exit(2);
                                }
                            }
                        }
                        SessionObservationEventPayload::Committed { .. } => {
                            observation_finished = true;
                        }
                        SessionObservationEventPayload::QueueChanged { .. }
                        | SessionObservationEventPayload::ProcessChanged { .. }
                        | SessionObservationEventPayload::AgentFrameSwitched { .. } => {}
                    },
                    Some(Ok(SessionObservationStreamItem::Gap { gap, .. })) => {
                        eprintln!(
                            "warning: live session observation skipped buffered events ({:?}); continuing from current snapshot",
                            gap.reason
                        );
                    }
                    Some(Err(err)) => {
                        eprintln!("warning: live session observation ended early: {err}");
                        observation_finished = true;
                    }
                    None => observation_finished = true,
                }
            }
            join = &mut task, if returned.is_none() => {
                match join {
                    Ok(result) => {
                        returned = Some(result);
                        observation_grace_deadline = Some(
                            tokio::time::Instant::now() + std::time::Duration::from_millis(250),
                        );
                    }
                    Err(err) => return Err(anyhow::anyhow!("autonomous turn task failed: {err}")),
                }
            }
            _ = async {
                match observation_grace_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending::<()>().await,
                }
            }, if returned.is_some() && !observation_finished => {
                observation_finished = true;
            }
        }
    }
    let (done, cancel) = returned.expect("return task completed");
    let done = done.map_err(|err| anyhow::anyhow!("autonomous turn task channel failed: {err}"))?;

    Ok(AutonomousTurnOutcome { done, cancel })
}

async fn run_prepared_autonomous_turn(
    session: LashSession,
    prompt: String,
    skills: &SkillCatalog,
    rlm_projected_bindings: Option<lash_protocol_rlm::RlmProjectedBindings>,
    stream_id: u64,
    output: &mut AutonomousOutput,
) -> anyhow::Result<AutonomousTurnOutcome> {
    let prepared = PreparedTurn::prepare(prompt, Vec::new(), skills);
    let mut turn_input = make_turn_input(&prepared);
    if let Some(bindings) = rlm_projected_bindings {
        turn_input = lash_protocol_rlm::RlmTurnInputExt::rlm_project(turn_input, bindings)?;
    }
    run_autonomous_turn(session, turn_input, output, stream_id).await
}

async fn finish_autonomous_outcome(
    session: &LashSession,
    output: &mut AutonomousOutput,
    outcome: AutonomousTurnOutcome,
    persistence: &AutonomousPersistenceContext,
) -> anyhow::Result<(crate::turn_runner::RuntimeRunResult, bool)> {
    let (mut done, cancel) = (outcome.done, outcome.cancel);
    if persistence.await_background_work {
        session.refresh_background_graph().await?;
        let state = session.admin().state().persist_current().await?;
        done.result.state = state.to_snapshot();
    }
    match &done.result.outcome {
        TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. } => {
            output.finish_success(&done.result, cancel.is_cancelled())?;
            if cancel.is_cancelled() && matches!(output, AutonomousOutput::Print(_)) {
                std::process::exit(1);
            }
        }
        TurnOutcome::Stopped(_) => {
            output.finish_failure(&done.result, cancel.is_cancelled())?;
            if matches!(output, AutonomousOutput::Print(_)) {
                std::process::exit(1);
            }
        }
    }
    let cancelled = cancel.is_cancelled();
    Ok((done, cancelled))
}

/// Run the session autonomously: send prompt, consume events, print final response to stdout.
pub(crate) async fn run_autonomous(
    session: LashSession,
    prompt: String,
    skills: SkillCatalog,
    persistence: AutonomousPersistenceContext,
    rlm_projected_bindings: Option<lash_protocol_rlm::RlmProjectedBindings>,
    mode: AutonomousMode,
) -> anyhow::Result<()> {
    if mode == AutonomousMode::Rpc {
        return run_rpc(session, skills, persistence).await;
    }
    let before_usage = session.usage_report();
    let mut output = match mode {
        AutonomousMode::Print => AutonomousOutput::print(),
        AutonomousMode::Json => AutonomousOutput::json(1)?,
        AutonomousMode::Rpc => unreachable!("handled above"),
    };
    let outcome = run_prepared_autonomous_turn(
        session.clone(),
        prompt,
        &skills,
        rlm_projected_bindings,
        1,
        &mut output,
    )
    .await?;
    let (done, cancelled) =
        finish_autonomous_outcome(&session, &mut output, outcome, &persistence).await?;
    let cumulative_usage = session.usage_report();
    if let Some(path) = &persistence.turn_usage_json {
        let (delta_entries, delta_error, delta_is_fallback) = match diff_usage_reports(
            &before_usage,
            &cumulative_usage,
        ) {
            Ok(entries) => (entries, None, false),
            Err(err) => {
                tracing::warn!(
                    %err,
                    "failed to diff token ledger for autonomous turn; falling back to assembled turn usage"
                );
                let fallback = if done.result.usage.total() > 0 {
                    vec![TokenLedgerEntry {
                        source: "turn".to_string(),
                        model: done.result.state.policy.model.id.clone(),
                        usage: done.result.usage.clone(),
                    }]
                } else {
                    Vec::new()
                };
                (fallback, Some(err), true)
            }
        };
        let usage_artifact = serde_json::json!({
            "delta_entries": delta_entries,
            "delta": SessionUsageReport::from_entries(&delta_entries),
            "delta_error": delta_error,
            "delta_is_fallback": delta_is_fallback,
            "cumulative_rows": cumulative_usage.by_source_model,
            "cumulative": cumulative_usage,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&usage_artifact)?)?;
    }

    if mode == AutonomousMode::Json
        && (cancelled
            || !matches!(
                done.result.outcome,
                TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. }
            ))
    {
        return Err(anyhow::anyhow!("autonomous turn failed"));
    }

    Ok(())
}

#[derive(serde::Deserialize)]
struct RpcRequest {
    id: serde_json::Value,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

fn rpc_write(value: serde_json::Value) -> anyhow::Result<()> {
    let mut stdout = io::stdout();
    serde_json::to_writer(&mut stdout, &value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn rpc_error(id: serde_json::Value, code: &str, message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "type": "response",
        "id": id,
        "ok": false,
        "error": { "code": code, "message": message.into() },
    })
}

async fn run_rpc(
    session: LashSession,
    skills: SkillCatalog,
    persistence: AutonomousPersistenceContext,
) -> anyhow::Result<()> {
    rpc_write(serde_json::json!({
        "type": "ready",
        "protocol": "lash.rpc.v1",
        "protocol_version": AUTONOMOUS_JSON_VERSION,
        "methods": ["prompt", "ping", "shutdown"],
    }))?;

    let stdin = io::stdin();
    let mut next_stream_id = 1_u64;
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: RpcRequest = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(err) => {
                rpc_write(rpc_error(
                    serde_json::Value::Null,
                    "invalid_json",
                    err.to_string(),
                ))?;
                continue;
            }
        };
        match request.method.as_str() {
            "ping" => rpc_write(serde_json::json!({
                "type": "response",
                "id": request.id,
                "ok": true,
                "result": { "pong": true },
            }))?,
            "shutdown" => {
                rpc_write(serde_json::json!({
                    "type": "response",
                    "id": request.id,
                    "ok": true,
                    "result": { "shutdown": true },
                }))?;
                return Ok(());
            }
            "prompt" => {
                let Some(prompt) = request
                    .params
                    .get("prompt")
                    .and_then(|value| value.as_str())
                else {
                    rpc_write(rpc_error(
                        request.id,
                        "invalid_params",
                        "prompt params require a string `prompt`",
                    ))?;
                    continue;
                };
                let stream_id = next_stream_id;
                next_stream_id += 1;
                rpc_write(serde_json::json!({
                    "type": "turn_start",
                    "id": request.id,
                    "stream_id": stream_id,
                }))?;
                let mut output =
                    AutonomousOutput::Json(JsonRenderer::for_rpc(stream_id, request.id.clone()));
                let outcome = run_prepared_autonomous_turn(
                    session.clone(),
                    prompt.to_string(),
                    &skills,
                    None,
                    stream_id,
                    &mut output,
                )
                .await?;
                let (done, cancelled) =
                    finish_autonomous_outcome(&session, &mut output, outcome, &persistence).await?;
                rpc_write(serde_json::json!({
                    "type": "response",
                    "id": request.id,
                    "ok": !cancelled && matches!(done.result.outcome, TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. }),
                    "result": {
                        "stream_id": stream_id,
                        "assistant_text": done.result.assistant_output.safe_text,
                        "outcome": done.result.outcome,
                        "usage": done.result.usage,
                        "errors": done.result.errors,
                    },
                }))?;
            }
            _ => rpc_write(rpc_error(
                request.id,
                "unknown_method",
                format!("unknown RPC method `{}`", request.method),
            ))?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::TurnResult;
    use lash_core::{
        AssistantOutput, ExecutionSummary, OutputState, SessionSnapshot, TokenUsage,
        ToolCallOutput, ToolCallRecord, ToolCallStatus, TurnActivityId, TurnFinish, TurnOutcome,
        TurnStop,
    };
    use serde_json::json;

    fn completed_tool_activity() -> TurnActivity {
        TurnActivity {
            id: TurnActivityId::new("act-1"),
            correlation_id: TurnActivityId::new("corr-1"),
            event: TurnEvent::ToolCallCompleted {
                call_id: Some("call-1".to_string()),
                name: "read_file".to_string(),
                args: json!({ "path": "README.md" }),
                output: ToolCallOutput::success("ok"),
                duration_ms: 5,
                graph_key: None,
                parent_call_id: None,
            },
        }
    }

    fn sample_turn_result() -> TurnResult {
        TurnResult {
            state: SessionSnapshot::default(),
            outcome: TurnOutcome::Finished(TurnFinish::AssistantMessage {
                text: "hello".to_string(),
            }),
            assistant_output: AssistantOutput {
                safe_text: "hello".to_string(),
                raw_text: "hello".to_string(),
                state: OutputState::Usable,
            },
            usage: TokenUsage {
                input_tokens: 3,
                output_tokens: 4,
                ..TokenUsage::default()
            },
            children_usage: Vec::new(),
            llm_calls: Vec::new(),
            tool_calls: vec![ToolCallRecord {
                call_id: Some("call-1".to_string()),
                tool: "read_file".to_string(),
                args: json!({ "path": "README.md" }),
                output: ToolCallOutput::success("ok"),
                duration_ms: 5,
            }],
            execution: ExecutionSummary {
                had_tool_calls: true,
                had_code_execution: false,
                started_at_ms: 1_000,
                duration_ms: 42,
            },
            errors: Vec::new(),
        }
    }

    #[test]
    fn turn_start_record_stamps_protocol_version() {
        assert_eq!(
            json_turn_start_record(1),
            json!({
                "type": "turn_start",
                "stream_id": 1,
                "protocol_version": AUTONOMOUS_JSON_VERSION,
            })
        );
        assert_eq!(AUTONOMOUS_JSON_VERSION, 1);
    }

    #[test]
    fn event_record_embeds_turn_activity_verbatim() {
        let activity = completed_tool_activity();
        let record = json_event_record(1, None, &activity);

        assert_eq!(record["type"], "event");
        assert_eq!(record["stream_id"], 1);
        assert!(record.get("id").is_none());
        // The activity is embedded verbatim: the flattened event tag, identity,
        // and correlation id all survive the record wrapper unchanged.
        assert_eq!(
            record["activity"],
            serde_json::to_value(&activity).unwrap(),
            "event frame must embed the TurnActivity verbatim"
        );
        assert_eq!(record["activity"]["type"], "tool_call_completed");
        assert_eq!(record["activity"]["id"], "act-1");
        assert_eq!(record["activity"]["correlation_id"], "corr-1");
    }

    #[test]
    fn event_record_echoes_rpc_request_id() {
        let activity = completed_tool_activity();
        let record = json_event_record(7, Some(&json!("req-2")), &activity);

        assert_eq!(record["stream_id"], 7);
        assert_eq!(record["id"], "req-2");
        assert_eq!(record["activity"]["type"], "tool_call_completed");
    }

    #[test]
    fn finish_record_pins_success_shape() {
        let record = json_finish_record(1, None, &sample_turn_result(), false);

        assert_eq!(record["type"], "turn_finish");
        assert_eq!(record["stream_id"], 1);
        assert_eq!(record["ok"], true);
        assert_eq!(record["cancelled"], false);
        assert_eq!(record["assistant_text"], "hello");
        assert!(record.get("id").is_none());
        assert_eq!(
            record["outcome"],
            json!({ "finished": { "assistant_message": { "text": "hello" } } })
        );
        assert_eq!(
            record["usage"],
            json!({
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_read_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "reasoning_output_tokens": 0,
            })
        );
        assert_eq!(record["children_usage"], json!([]));
        assert_eq!(record["errors"], json!([]));
        assert_eq!(
            record["execution"],
            json!({
                "had_tool_calls": true,
                "had_code_execution": false,
                "started_at_ms": 1_000,
                "duration_ms": 42,
            })
        );
        assert_eq!(
            record["tool_calls"],
            json!([{
                "call_id": "call-1",
                "tool": "read_file",
                "args": { "path": "README.md" },
                "output": { "outcome": { "status": "success", "payload": "ok" } },
                "duration_ms": 5,
            }])
        );
    }

    #[test]
    fn finish_record_marks_stopped_turn_not_ok() {
        let mut turn = sample_turn_result();
        turn.outcome = TurnOutcome::Stopped(TurnStop::ToolFailure);
        let record = json_finish_record(1, None, &turn, false);
        assert_eq!(record["ok"], false);
        assert_eq!(record["outcome"], json!({ "stopped": "tool_failure" }));
    }

    #[test]
    fn finish_record_marks_cancelled_turn_not_ok_and_echoes_id() {
        let record = json_finish_record(2, Some(&json!("req-9")), &sample_turn_result(), true);
        assert_eq!(record["id"], "req-9");
        assert_eq!(record["stream_id"], 2);
        assert_eq!(record["cancelled"], true);
        assert_eq!(record["ok"], false, "a cancelled turn is never ok");
    }

    #[test]
    fn tool_line_renders_status_and_visible_duration() {
        assert_eq!(
            format_tool_line("read_file", ToolCallStatus::Success, 1_500),
            "[tool] read_file · ok · 1.5s"
        );
    }

    #[test]
    fn tool_line_elides_subsecond_duration() {
        assert_eq!(
            format_tool_line("read_file", ToolCallStatus::Success, 500),
            "[tool] read_file · ok"
        );
    }

    #[test]
    fn tool_line_renders_failure_status() {
        assert_eq!(
            format_tool_line("write_file", ToolCallStatus::Failure, 2_000),
            "[tool] write_file · error · 2.0s"
        );
    }

    #[test]
    fn tool_line_renders_cancelled_status() {
        assert_eq!(
            format_tool_line("shell", ToolCallStatus::Cancelled, 0),
            "[tool] shell · cancelled"
        );
    }
}
