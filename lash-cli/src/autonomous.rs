use std::collections::BTreeMap;
use std::io::{self, Write};

use lash::*;
use tokio::sync::mpsc;

use crate::app::PreparedTurn;
use crate::turn_runner::{make_turn_input, spawn_runtime_turn};
use crate::{plugin_surface, util};

pub(crate) struct AutonomousPersistenceContext {
    pub(crate) await_background_work: bool,
    pub(crate) turn_usage_json: Option<std::path::PathBuf>,
}

struct AutonomousChannelSink {
    tx: mpsc::Sender<SessionEvent>,
}

#[async_trait::async_trait]
impl EventSink for AutonomousChannelSink {
    async fn emit(&self, event: SessionEvent) {
        let _ = self.tx.send(event).await;
    }
}

pub(crate) struct AutonomousRenderer {
    pub(crate) streamed_text: bool,
    pub(crate) wrote_stdout: bool,
    pub(crate) stdout_text: String,
    pub(crate) plugin_panels: BTreeMap<String, (String, String)>,
}

impl AutonomousRenderer {
    pub(crate) fn new() -> Self {
        Self {
            streamed_text: false,
            wrote_stdout: false,
            stdout_text: String::new(),
            plugin_panels: BTreeMap::new(),
        }
    }

    pub(crate) fn handle(&mut self, event: SessionEvent) -> Result<Option<String>, String> {
        let handoff_session_id = match &event {
            SessionEvent::TurnOutcome {
                outcome: lash::TurnOutcome::Handoff { session_id },
            } => Some(session_id.clone()),
            _ => None,
        };
        match event {
            SessionEvent::TextDelta { content } => {
                if !content.is_empty() {
                    self.streamed_text = true;
                    self.wrote_stdout = true;
                    self.stdout_text.push_str(&content);
                    print!("{content}");
                    let _ = io::stdout().flush();
                }
            }
            SessionEvent::ToolCall {
                name,
                success,
                duration_ms,
                ..
            } => {
                let status = if success { "ok" } else { "error" };
                if let Some(duration_text) = util::format_duration_ms_if_visible(duration_ms) {
                    eprintln!("[tool] {name} · {status} · {duration_text}");
                } else {
                    eprintln!("[tool] {name} · {status}");
                }
            }
            SessionEvent::Message { text, kind } => match kind.as_str() {
                "tool_output" | "final" if !text.trim().is_empty() => {
                    if kind == "final" {
                        self.streamed_text = true;
                        self.wrote_stdout = true;
                        self.stdout_text.push_str(&text);
                        print!("{text}");
                        let _ = io::stdout().flush();
                    } else {
                        eprintln!("{text}");
                    }
                }
                _ => {}
            },
            SessionEvent::LlmRequest { mode_iteration, .. } => {
                eprintln!("[thinking] step {}", mode_iteration + 1);
            }
            SessionEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
                ..
            } => {
                eprintln!(
                    "[retry] in {}s · attempt {}/{} · {}",
                    wait_seconds, attempt, max_attempts, reason
                );
            }
            SessionEvent::Error { message, .. } => {
                eprintln!("error: {message}");
            }
            SessionEvent::Prompt { request, .. } => {
                return Err(format!(
                    "unexpected user prompt in autonomous mode: {}",
                    request.question
                ));
            }
            SessionEvent::PluginEvent { plugin_id, event } => match event {
                PluginSurfaceEvent::PanelUpsert {
                    key,
                    title,
                    content,
                } => {
                    self.plugin_panels.insert(
                        plugin_surface::surface_key(&plugin_id, &key),
                        (title, content),
                    );
                }
                PluginSurfaceEvent::PanelAppend { key, content } => {
                    if !content.is_empty()
                        && let Some((_, existing)) = self
                            .plugin_panels
                            .get_mut(&plugin_surface::surface_key(&plugin_id, &key))
                    {
                        existing.push_str(&content);
                    }
                }
                PluginSurfaceEvent::PanelClear { key } => {
                    self.plugin_panels
                        .remove(&plugin_surface::surface_key(&plugin_id, &key));
                }
                PluginSurfaceEvent::ModeIndicatorUpsert { .. }
                | PluginSurfaceEvent::ModeIndicatorClear { .. }
                | PluginSurfaceEvent::Status { .. }
                | PluginSurfaceEvent::Custom { .. } => {}
            },
            // Reasoning summaries are a TUI-only affordance; in
            // autonomous mode we deliberately discard them to keep stdout
            // aligned with the model's final answer.
            SessionEvent::ReasoningDelta { .. } => {}
            SessionEvent::Done
            | SessionEvent::ToolCallStart { .. }
            | SessionEvent::TokenUsage { .. }
            | SessionEvent::ChildTokenUsage { .. }
            | SessionEvent::InjectedTurnInputAccepted { .. }
            | SessionEvent::InjectedMessagesCommitted { .. }
            | SessionEvent::TurnOutcome { .. }
            | SessionEvent::LlmResponse { .. } => {}
        }
        Ok(handoff_session_id)
    }

    pub(crate) fn rendered_plugin_output(&self) -> Option<String> {
        let sections = self
            .plugin_panels
            .values()
            .map(|(title, content)| {
                let body = content.trim();
                if body.is_empty() {
                    title.clone()
                } else {
                    format!("{title}\n{body}")
                }
            })
            .collect::<Vec<_>>();
        if sections.is_empty() {
            None
        } else {
            Some(sections.join("\n\n"))
        }
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
    handoff_session_id: Option<String>,
}

async fn run_autonomous_turn(
    runtime: LashRuntime,
    turn_input: TurnInput,
    renderer: &mut AutonomousRenderer,
    stream_id: u64,
) -> anyhow::Result<AutonomousTurnOutcome> {
    let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(100);
    let sink = AutonomousChannelSink { tx: event_tx };
    let (cancel, return_rx) = spawn_runtime_turn(runtime, turn_input, sink, stream_id);
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
    let mut handoff_session_id = None;

    let (done, cancel) = loop {
        tokio::select! {
            Some(event) = event_rx.recv() => {
                match renderer.handle(event) {
                    Ok(Some(session_id)) => {
                        handoff_session_id = Some(session_id);
                    }
                    Ok(None) => {}
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
    while let Ok(event) = event_rx.try_recv() {
        match renderer.handle(event) {
            Ok(Some(session_id)) => {
                handoff_session_id = Some(session_id);
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(2);
            }
        }
    }

    Ok(AutonomousTurnOutcome {
        done,
        cancel,
        handoff_session_id,
    })
}

async fn activate_autonomous_handoff(
    runtime: &mut LashRuntime,
    session_id: &str,
) -> anyhow::Result<TurnInput> {
    let session_manager = runtime
        .session_manager()
        .map_err(|err| anyhow::anyhow!("failed to access session manager: {err}"))?;
    runtime
        .activate_managed_session(session_id)
        .await
        .map_err(|err| anyhow::anyhow!("failed to activate session `{session_id}`: {err}"))?;
    let seed = session_manager
        .take_first_turn_input(session_id)
        .await
        .map_err(|err| {
            anyhow::anyhow!("failed to read first turn for session `{session_id}`: {err}")
        })?
        .ok_or_else(|| {
            anyhow::anyhow!("handoff session `{session_id}` did not provide a first turn")
        })?;
    if seed.content.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "handoff session `{session_id}` provided an empty first turn"
        ));
    }
    let prepared =
        PreparedTurn::prepare_with_effective_text(seed.content.clone(), seed.content, Vec::new());
    Ok(make_turn_input(&prepared))
}

/// Run the session autonomously: send prompt, consume events, print final response to stdout.
pub(crate) async fn run_autonomous(
    runtime: LashRuntime,
    prompt: String,
    skills: SkillCatalog,
    persistence: AutonomousPersistenceContext,
) -> anyhow::Result<()> {
    let before_usage = runtime.usage_report();
    let prepared = PreparedTurn::prepare(prompt, Vec::new(), &skills);
    let mut runtime = runtime;
    let mut turn_input = make_turn_input(&prepared);
    let mut renderer = AutonomousRenderer::new();
    let mut stream_id = 1;
    let (mut done, cancel) = loop {
        let outcome = run_autonomous_turn(runtime, turn_input, &mut renderer, stream_id).await?;
        let mut turn_done = outcome.done;
        if let Some(session_id) = outcome.handoff_session_id {
            if !matches!(
                &turn_done.result.outcome,
                lash::TurnOutcome::Finished(_) | lash::TurnOutcome::Handoff { .. }
            ) {
                break (turn_done, outcome.cancel);
            }
            turn_input = activate_autonomous_handoff(&mut turn_done.runtime, &session_id).await?;
            runtime = turn_done.runtime;
            stream_id += 1;
            continue;
        }
        break (turn_done, outcome.cancel);
    };
    if persistence.await_background_work {
        done.runtime.await_background_work().await?;
        done.result.state = done.runtime.export_state();
    }
    let cumulative_usage = done.runtime.usage_report();
    if let Some(path) = &persistence.turn_usage_json {
        let (delta_entries, delta_error, delta_is_fallback) = match lash::diff_usage_reports(
            &before_usage,
            &cumulative_usage,
        ) {
            Ok(entries) => (entries, None, false),
            Err(err) => {
                tracing::warn!(
                    %err,
                    "failed to diff token ledger for autonomous turn; falling back to assembled turn usage"
                );
                let fallback = if done.result.token_usage.total() > 0
                    || done.result.token_usage.cached_input_tokens > 0
                {
                    vec![lash::TokenLedgerEntry {
                        source: "turn".to_string(),
                        model: done.result.state.policy.model.clone(),
                        usage: done.result.token_usage.clone(),
                    }]
                } else {
                    Vec::new()
                };
                (fallback, Some(err), true)
            }
        };
        let usage_artifact = serde_json::json!({
            "delta_entries": delta_entries,
            "delta": lash::SessionUsageReport::from_entries(&delta_entries),
            "delta_error": delta_error,
            "delta_is_fallback": delta_is_fallback,
            "cumulative_rows": cumulative_usage.by_source_model,
            "cumulative": cumulative_usage,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&usage_artifact)?)?;
    }

    match &done.result.outcome {
        lash::TurnOutcome::Finished(_) | lash::TurnOutcome::Handoff { .. } => {
            let turn = &done.result;
            if !turn.assistant_output.safe_text.is_empty() {
                renderer.finish_output(&turn.assistant_output.safe_text);
            } else if turn.has_plugin_visible_output {
                if let Some(rendered) = renderer.rendered_plugin_output() {
                    renderer.finish_output(&rendered);
                }
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
        lash::TurnOutcome::Stopped(_) => {
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
