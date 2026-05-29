use std::io::{self, Write};

use lash::{
    LashSession, TurnActivity, TurnActivitySink, TurnEvent, TurnInput,
    usage::{SessionUsageReport, TokenLedgerEntry, diff_usage_reports},
};
use lash_core::TurnOutcome;
use tokio::sync::mpsc;

use crate::SkillCatalog;
use crate::app::PreparedTurn;
use crate::turn_runner::{make_turn_input, spawn_session_turn};
use crate::util;

pub(crate) struct AutonomousPersistenceContext {
    pub(crate) await_background_work: bool,
    pub(crate) turn_usage_json: Option<std::path::PathBuf>,
}

struct AutonomousChannelSink {
    tx: mpsc::Sender<TurnActivity>,
}

#[async_trait::async_trait]
impl TurnActivitySink for AutonomousChannelSink {
    async fn emit(&self, activity: TurnActivity) {
        let _ = self.tx.send(activity).await;
    }
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
                let status = match output.status() {
                    lash_core::ToolCallStatus::Success => "ok",
                    lash_core::ToolCallStatus::Failure => "error",
                    lash_core::ToolCallStatus::Cancelled => "cancelled",
                };
                if let Some(duration_text) = util::format_duration_ms_if_visible(duration_ms) {
                    eprintln!("[tool] {name} · {status} · {duration_text}");
                } else {
                    eprintln!("[tool] {name} · {status}");
                }
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
            | TurnEvent::SubmittedValue { .. }
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

struct AutonomousTurnOutcome {
    done: crate::turn_runner::RuntimeRunResult,
    cancel: tokio_util::sync::CancellationToken,
}

async fn run_autonomous_turn(
    session: LashSession,
    turn_input: TurnInput,
    renderer: &mut AutonomousRenderer,
    stream_id: u64,
) -> anyhow::Result<AutonomousTurnOutcome> {
    let (event_tx, mut event_rx) = mpsc::channel::<TurnActivity>(100);
    let sink = AutonomousChannelSink { tx: event_tx };
    let (cancel, return_rx) = spawn_session_turn(session, turn_input, sink, stream_id);
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
    let (done, cancel) = loop {
        tokio::select! {
            Some(activity) = event_rx.recv() => {
                match renderer.handle(activity) {
                    Ok(()) => {}
                    Err(err) => {
                        eprintln!("error: {err}");
                        std::process::exit(2);
                    }
                }
            }
            join = &mut task => {
                match join {
                    Ok(result) => break result,
                    Err(err) => return Err(anyhow::anyhow!("autonomous turn task failed: {err}")),
                }
            }
        }
    };
    let done = done.map_err(|err| anyhow::anyhow!("autonomous turn task channel failed: {err}"))?;
    while let Ok(activity) = event_rx.try_recv() {
        match renderer.handle(activity) {
            Ok(()) => {}
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(2);
            }
        }
    }

    Ok(AutonomousTurnOutcome { done, cancel })
}

/// Run the session autonomously: send prompt, consume events, print final response to stdout.
pub(crate) async fn run_autonomous(
    session: LashSession,
    prompt: String,
    skills: SkillCatalog,
    persistence: AutonomousPersistenceContext,
    rlm_projected_bindings: Option<lash_protocol_rlm::RlmProjectedBindings>,
) -> anyhow::Result<()> {
    let before_usage = session.usage_report();
    let prepared = PreparedTurn::prepare(prompt, Vec::new(), &skills);
    let mut turn_input = make_turn_input(&prepared);
    if let Some(bindings) = rlm_projected_bindings {
        turn_input = lash_protocol_rlm::RlmTurnInputExt::rlm_project(turn_input, bindings)?;
    }
    let mut renderer = AutonomousRenderer::new();
    let outcome = run_autonomous_turn(session.clone(), turn_input, &mut renderer, 1).await?;
    let (mut done, cancel) = (outcome.done, outcome.cancel);
    if persistence.await_background_work {
        session.control().state().await_background_work().await?;
        let state = session.control().state().persist_current().await?;
        done.result.state = state.into_envelope();
    }
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
                let fallback =
                    if done.result.usage.total() > 0 || done.result.usage.cached_input_tokens > 0 {
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

    match &done.result.outcome {
        TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. } => {
            let turn = &done.result;
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
                    eprintln!(
                        "error: model returned malformed assistant output: {}",
                        preview
                    );
                }
                std::process::exit(2);
            }
            if cancel.is_cancelled() {
                std::process::exit(1);
            }
        }
        TurnOutcome::Stopped(_) => {
            for issue in &done.result.errors {
                eprintln!("error: {}", issue.message);
            }
            if done.result.errors.is_empty() {
                eprintln!("error: autonomous turn failed");
            }
            std::process::exit(1);
        }
    }
    Ok(())
}
