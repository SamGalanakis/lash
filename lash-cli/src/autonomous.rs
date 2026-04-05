use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;

use lash::provider::Provider;
use lash::*;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::app::{PreparedTurn, UiResumeState};
use crate::input_items::build_items_from_editor_input;
use crate::{hash12, latest_user_prompt_hash, persist_root_agent_state};
use crate::{plugin_surface, util};

pub(crate) struct AutonomousPersistenceContext {
    pub(crate) store: Arc<Store>,
    pub(crate) dynamic_state: DynamicStateSnapshot,
    pub(crate) provider: Provider,
    pub(crate) configured_model: String,
    pub(crate) context_window: u64,
    pub(crate) model_variant: Option<String>,
    pub(crate) toolset_hash: String,
}

async fn persist_autonomous_runtime_state(
    runtime: &mut LashRuntime,
    persistence: &AutonomousPersistenceContext,
    mut state: AgentStateEnvelope,
) {
    let snapshot_hash = if matches!(state.policy.execution_mode, ExecutionMode::Repl) {
        match runtime.snapshot_repl().await {
            Ok(blob) => {
                state = runtime.export_state();
                Some(hash12(&blob))
            }
            Err(err) => {
                tracing::warn!(
                    "failed to snapshot repl state during autonomous persistence: {err}"
                );
                None
            }
        }
    } else {
        state = runtime.export_state();
        None
    };

    let execution_mode = state.policy.execution_mode;
    let context_strategy = state.policy.context_strategy;
    let prompt_hash = latest_user_prompt_hash(&state.messages);
    let ui_state = UiResumeState::default();
    persist_root_agent_state(
        &persistence.store,
        &mut state,
        &ui_state,
        &persistence.dynamic_state,
        &persistence.provider,
        &persistence.configured_model,
        persistence.context_window,
        execution_mode,
        context_strategy,
        persistence.model_variant.as_deref(),
        &persistence.toolset_hash,
        prompt_hash,
        snapshot_hash,
    );
    runtime.set_state(state);
}

struct AutonomousChannelSink {
    tx: mpsc::Sender<AgentEvent>,
}

#[async_trait::async_trait]
impl EventSink for AutonomousChannelSink {
    async fn emit(&self, event: AgentEvent) {
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

    pub(crate) fn handle(&mut self, event: AgentEvent) -> Result<(), String> {
        match event {
            AgentEvent::TextDelta { content } => {
                if !content.is_empty() {
                    self.streamed_text = true;
                    self.wrote_stdout = true;
                    self.stdout_text.push_str(&content);
                    print!("{content}");
                    let _ = io::stdout().flush();
                }
            }
            AgentEvent::CodeBlock { code } => {
                if !code.trim().is_empty() {
                    eprintln!("[code]");
                    eprintln!("{code}");
                    eprintln!("[/code]");
                }
            }
            AgentEvent::ToolCall {
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
            AgentEvent::CodeOutput { output, error } => {
                if !output.trim().is_empty() {
                    eprintln!("{output}");
                }
                if let Some(error) = error.filter(|value| !value.trim().is_empty()) {
                    eprintln!("{error}");
                }
            }
            AgentEvent::Message { text, kind } => match kind.as_str() {
                "tool_output" | "final" => {
                    if !text.trim().is_empty() {
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
                }
                "delegate_start" => {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                        let task = value
                            .get("task")
                            .and_then(|item| item.as_str())
                            .unwrap_or("delegate");
                        let model = value
                            .get("model")
                            .and_then(|item| item.as_str())
                            .unwrap_or_default();
                        let variant = value
                            .get("model_variant")
                            .and_then(|item| item.as_str())
                            .unwrap_or_default();
                        let model_label = if model.is_empty() {
                            String::new()
                        } else if variant.is_empty() {
                            model.to_string()
                        } else {
                            format!("{model} ({variant})")
                        };
                        if model_label.is_empty() {
                            eprintln!("[delegate] {task}");
                        } else {
                            eprintln!("[delegate] {task} · {model_label}");
                        }
                    } else if !text.trim().is_empty() {
                        eprintln!("[delegate] {}", text.trim());
                    }
                }
                _ => {}
            },
            AgentEvent::LlmRequest { iteration, .. } => {
                eprintln!("[thinking] turn {}", iteration + 1);
            }
            AgentEvent::RetryStatus {
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
            AgentEvent::Error { message, .. } => {
                eprintln!("error: {message}");
            }
            AgentEvent::Prompt { request, .. } => {
                return Err(format!(
                    "unexpected user prompt in autonomous mode: {}",
                    request.question
                ));
            }
            AgentEvent::PluginEvent { plugin_id, event } => match event {
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
                | PluginSurfaceEvent::Custom { .. } => {}
            },
            AgentEvent::Done
            | AgentEvent::TokenUsage { .. }
            | AgentEvent::InjectedMessagesCommitted { .. }
            | AgentEvent::LlmResponse { .. }
            | AgentEvent::DurableSnapshot { .. } => {}
        }
        Ok(())
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

/// Run the agent autonomously: send prompt, consume events, print final response to stdout.
pub(crate) async fn run_autonomous(
    mut runtime: LashRuntime,
    prompt: String,
    skills: SkillCatalog,
    persistence: AutonomousPersistenceContext,
) -> anyhow::Result<()> {
    let prepared = PreparedTurn::prepare(prompt, Vec::new(), &skills);
    let (items, image_blobs) = build_items_from_editor_input(&prepared.effective_text, Vec::new());
    let cancel = CancellationToken::new();
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
    let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
    let mut task = tokio::spawn(async move {
        let sink = AutonomousChannelSink { tx: event_tx };
        let result = runtime
            .stream_turn(
                TurnInput {
                    items,
                    image_blobs,
                    mode: Some(RunMode::Normal),
                },
                &sink,
                cancel.clone(),
            )
            .await;
        (runtime, result, cancel)
    });

    let mut renderer = AutonomousRenderer::new();
    let (mut runtime, result, cancel) = loop {
        tokio::select! {
            Some(event) = event_rx.recv() => {
                if let Err(err) = renderer.handle(event) {
                    eprintln!("error: {err}");
                    std::process::exit(2);
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

    match result {
        Ok(turn) => {
            persist_autonomous_runtime_state(&mut runtime, &persistence, turn.state.clone()).await;
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
        Err(e) => {
            let state = runtime.export_state();
            persist_autonomous_runtime_state(&mut runtime, &persistence, state).await;
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
    Ok(())
}
