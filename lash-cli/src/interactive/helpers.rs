use crossterm::event::{KeyCode, KeyModifiers};
use lash::*;
use lash_ui::{KeyChord as UiKeyChord, KeyCode as UiKeyCode, KeyModifiers as UiKeyModifiers};
use tokio::sync::mpsc;

use crate::app::{App, PreparedTurn};
use crate::event::{AppEvent, AppEventTx};
use crate::ui_trace::{UiTraceRecorder, drain_aux_ops_into};
use crate::{collect_ui_snapshot, copy_binding, queued_turn_edit_binding};

#[derive(Clone)]
pub(super) struct TurnReplayPayload {
    pub(super) prepared_turn: PreparedTurn,
    pub(super) turn_input: TurnInput,
    pub(super) execution_mode: ExecutionMode,
}

pub(super) struct AppEventSink {
    tx: mpsc::Sender<SessionEvent>,
}

#[async_trait::async_trait]
impl EventSink for AppEventSink {
    async fn emit(&self, event: SessionEvent) {
        let _ = self.tx.send(event).await;
    }
}

pub(super) struct RuntimeEventBridge;

impl RuntimeEventBridge {
    pub(super) fn spawn(stream_id: u64, app_tx: AppEventTx) -> AppEventSink {
        let (tx, mut rx) = mpsc::channel::<SessionEvent>(256);
        tokio::spawn(async move {
            let mut pending_text = String::new();
            let mut flush = Box::pin(tokio::time::sleep(std::time::Duration::from_secs(86_400)));
            let mut flush_armed = false;
            let mut coalesced = 0usize;

            loop {
                tokio::select! {
                    biased;
                    _ = &mut flush, if flush_armed => {
                        if !pending_text.is_empty() {
                            let content = std::mem::take(&mut pending_text);
                            let _ = app_tx.send(AppEvent::Session {
                                stream_id,
                                event: SessionEvent::TextDelta { content },
                            });
                            if coalesced > 0 {
                                tracing::trace!(stream_id, coalesced, "coalesced runtime text deltas");
                            }
                            coalesced = 0;
                        }
                        flush_armed = false;
                    }
                    maybe_event = rx.recv() => {
                        let Some(event) = maybe_event else {
                            break;
                        };
                        match event {
                            SessionEvent::TextDelta { content } => {
                                if content.is_empty() {
                                    continue;
                                }
                                if pending_text.is_empty() {
                                    flush.as_mut().reset(tokio::time::Instant::now() + TEXT_DELTA_REDRAW_INTERVAL);
                                    flush_armed = true;
                                } else {
                                    coalesced += 1;
                                }
                                pending_text.push_str(&content);
                            }
                            event => {
                                if !pending_text.is_empty() {
                                    let content = std::mem::take(&mut pending_text);
                                    let _ = app_tx.send(AppEvent::Session {
                                        stream_id,
                                        event: SessionEvent::TextDelta { content },
                                    });
                                    if coalesced > 0 {
                                        tracing::trace!(stream_id, coalesced, "coalesced runtime text deltas before structural event");
                                    }
                                    coalesced = 0;
                                    flush_armed = false;
                                }
                                let _ = app_tx.send(AppEvent::Session { stream_id, event });
                            }
                        }
                    }
                }
            }

            if !pending_text.is_empty() {
                let _ = app_tx.send(AppEvent::Session {
                    stream_id,
                    event: SessionEvent::TextDelta {
                        content: pending_text,
                    },
                });
            }
        });
        AppEventSink { tx }
    }
}

pub(super) struct UiSnapshotWorker {
    request_tx: mpsc::UnboundedSender<UiSnapshotRequest>,
    next_generation: u64,
    in_flight: bool,
    pending: bool,
}

struct UiSnapshotRequest {
    generation: u64,
    session_id: String,
    session_manager: std::sync::Arc<dyn RuntimeSessionHost>,
}

impl UiSnapshotWorker {
    pub(super) fn spawn(
        app_tx: AppEventTx,
        ui_extensions: std::sync::Arc<lash_ui::UiExtensions>,
        plugin_host: PluginHost,
    ) -> Self {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<UiSnapshotRequest>();
        tokio::spawn(async move {
            while let Some(request) = request_rx.recv().await {
                let started = std::time::Instant::now();
                let snapshot = tokio::time::timeout(
                    std::time::Duration::from_millis(200),
                    collect_ui_snapshot(
                        request.session_id,
                        ui_extensions.as_ref(),
                        &plugin_host,
                        request.session_manager,
                    ),
                )
                .await;
                let result = match snapshot {
                    Ok(result) => {
                        if result.duration > std::time::Duration::from_millis(100) {
                            tracing::warn!(
                                generation = request.generation,
                                duration_ms = result.duration.as_millis(),
                                "slow UI snapshot"
                            );
                        }
                        result
                    }
                    Err(_) => {
                        tracing::warn!(
                            generation = request.generation,
                            duration_ms = started.elapsed().as_millis(),
                            "UI snapshot exceeded deadline"
                        );
                        crate::event::UiSnapshotResult {
                            effects: Vec::new(),
                            background_tasks: None,
                            duration: started.elapsed(),
                            timed_out: true,
                            diagnostics: vec!["UI snapshot exceeded 200ms deadline".to_string()],
                        }
                    }
                };
                let _ = app_tx.send(AppEvent::UiSnapshot {
                    generation: request.generation,
                    result,
                });
            }
        });
        Self {
            request_tx,
            next_generation: 0,
            in_flight: false,
            pending: false,
        }
    }

    pub(super) fn request(
        &mut self,
        session_id: String,
        session_manager: std::sync::Arc<dyn RuntimeSessionHost>,
    ) {
        if self.in_flight {
            self.pending = true;
            return;
        }
        self.next_generation = self.next_generation.wrapping_add(1);
        let generation = self.next_generation;
        self.in_flight = true;
        self.pending = false;
        let _ = self.request_tx.send(UiSnapshotRequest {
            generation,
            session_id,
            session_manager,
        });
    }

    pub(super) fn complete(
        &mut self,
        generation: u64,
        session_id: String,
        session_manager: std::sync::Arc<dyn RuntimeSessionHost>,
    ) -> bool {
        if generation != self.next_generation {
            return false;
        }
        self.in_flight = false;
        if self.pending {
            self.request(session_id, session_manager);
        }
        true
    }
}

pub(super) fn cleared_session_state(policy: SessionPolicy) -> SessionStateEnvelope {
    SessionStateEnvelope {
        policy,
        ..SessionStateEnvelope::default()
    }
}

pub(super) fn log_runtime_handoff(
    phase: &str,
    app: &App,
    runtime: &Option<LashRuntime>,
    runtime_return_rx_present: bool,
    cancel_token_present: bool,
    active_stream_id: u64,
) {
    tracing::debug!(
        phase,
        app_running = app.running,
        runtime_present = runtime.is_some(),
        runtime_return_rx_present,
        cancel_token_present,
        queued_turns = app.queued_turns.len(),
        pending_steers = app.pending_steers.len(),
        active_stream_id,
        live_turn = app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
        "interactive runtime handoff"
    );
}

pub(super) const TEXT_DELTA_REDRAW_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(33);

pub(super) fn record_queue_turn(ui_trace: &mut Option<UiTraceRecorder>, turn: &PreparedTurn) {
    if let Some(recorder) = ui_trace.as_mut() {
        recorder.record_queue_turn(turn);
    }
}

pub(super) fn record_queue_pending_steer(
    ui_trace: &mut Option<UiTraceRecorder>,
    turn: &PreparedTurn,
) {
    if let Some(recorder) = ui_trace.as_mut() {
        recorder.record_queue_pending_steer(turn);
    }
}

pub(super) fn drain_aux_trace_ops(ui_trace: &mut Option<UiTraceRecorder>) {
    if let Some(recorder) = ui_trace.as_mut() {
        drain_aux_ops_into(recorder);
    }
}

pub(super) fn is_copy_shortcut(key: crossterm::event::KeyEvent) -> bool {
    copy_binding().matches(key)
}

pub(super) fn should_preserve_selection_for_key(key: crossterm::event::KeyEvent) -> bool {
    if is_copy_shortcut(key) {
        return true;
    }

    key.modifiers.intersects(
        KeyModifiers::SHIFT | KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
    )
}

pub(super) fn key_chord_from_event(key: crossterm::event::KeyEvent) -> Option<UiKeyChord> {
    let code = match key.code {
        KeyCode::Tab | KeyCode::BackTab => UiKeyCode::Tab,
        KeyCode::Enter => UiKeyCode::Enter,
        KeyCode::Esc => UiKeyCode::Esc,
        KeyCode::Up => UiKeyCode::Up,
        KeyCode::Down => UiKeyCode::Down,
        KeyCode::PageUp => UiKeyCode::PageUp,
        KeyCode::PageDown => UiKeyCode::PageDown,
        KeyCode::Char(ch) => UiKeyCode::Char(ch),
        _ => return None,
    };
    Some(UiKeyChord {
        code,
        modifiers: UiKeyModifiers {
            shift: key.modifiers.contains(KeyModifiers::SHIFT)
                || matches!(key.code, KeyCode::BackTab),
            control: key.modifiers.contains(KeyModifiers::CONTROL),
            alt: key.modifiers.contains(KeyModifiers::ALT),
        },
    })
}

pub(super) fn monitor_wake_message(input: &str) -> PluginMessage {
    PluginMessage::text(MessageRole::System, input)
}

pub(super) fn queued_turn_edit_matches(key: crossterm::event::KeyEvent) -> bool {
    queued_turn_edit_binding().matches(key)
}
