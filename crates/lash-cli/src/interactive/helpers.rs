use crossterm::event::{KeyCode, KeyModifiers};
use futures_util::StreamExt as _;
use lash::observe::{SessionCursor, SessionObservationEventPayload, SessionObservationStreamItem};
use lash::{LashSession, TurnActivity, TurnActivityId, TurnEvent, TurnInput};
use lash_core::{LiveReplayGap, SessionPolicy, SessionSnapshot};
use lash_tui_extensions::{
    KeyChord as UiKeyChord, KeyCode as UiKeyCode, KeyModifiers as UiKeyModifiers,
};
use tokio::sync::mpsc;

use crate::app::{App, PreparedTurn};
use crate::event::{AppEvent, AppEventTx};
use crate::execution_settings::ExecutionMode;
use crate::keybindings::{copy_binding, queued_turn_edit_binding};
use crate::ui_effects::collect_ui_snapshot;
use crate::ui_trace::{UiTraceRecorder, drain_aux_ops_into};

#[derive(Clone)]
pub(super) struct TurnReplayPayload {
    pub(super) prepared_turn: PreparedTurn,
    pub(super) turn_input: TurnInput,
    pub(super) execution_mode: ExecutionMode,
}

pub(super) struct SessionObservationBridge;

impl SessionObservationBridge {
    pub(super) fn spawn(session: &LashSession, stream_id: u64, app_tx: AppEventTx) {
        let observable = session.observe();
        let cursor = observable.current_observation().cursor;
        Self::spawn_from_cursor(session, cursor, stream_id, app_tx);
    }

    pub(super) fn spawn_from_cursor(
        session: &LashSession,
        cursor: SessionCursor,
        stream_id: u64,
        app_tx: AppEventTx,
    ) {
        let observable = session.observe();
        let mut subscription = observable.subscribe_and_recover(cursor);

        tokio::spawn(async move {
            let mut coalescer = TurnActivityCoalescer::new(stream_id, app_tx.clone());
            loop {
                tokio::select! {
                    biased;
                    _ = &mut coalescer.flush, if coalescer.flush_armed => {
                        coalescer.flush().await;
                    }
                    next = subscription.next() => {
                        match next {
                            Some(Ok(SessionObservationStreamItem::Event(event))) => match event.payload {
                                SessionObservationEventPayload::TurnActivity(activity) => {
                                    coalescer.emit(activity).await;
                                }
                                SessionObservationEventPayload::QueueChanged { .. } => {
                                    coalescer.flush().await;
                                    let _ = app_tx.send(AppEvent::RequestQueuedWorkSnapshot);
                                }
                                SessionObservationEventPayload::ProcessChanged {
                                    kind,
                                    process_ids,
                                } => {
                                    coalescer.flush().await;
                                    let _ = app_tx.send(AppEvent::ProcessChanged {
                                        stream_id,
                                        kind,
                                        process_ids,
                                    });
                                    let _ = app_tx.send(AppEvent::RequestUiSnapshot);
                                }
                                SessionObservationEventPayload::AgentFrameSwitched { .. } => {
                                    coalescer.flush().await;
                                    let _ = app_tx.send(AppEvent::RequestUiSnapshot);
                                }
                                SessionObservationEventPayload::Committed { .. } => {
                                    coalescer.flush().await;
                                    let _ = app_tx.send(AppEvent::SessionObservationFinished { stream_id });
                                    break;
                                }
                            },
                            Some(Ok(SessionObservationStreamItem::Gap { gap, .. })) => {
                                coalescer.flush().await;
                                emit_observation_gap(&app_tx, &gap);
                            }
                            Some(Err(err)) => {
                                coalescer.flush().await;
                                let _ = app_tx.send(AppEvent::SystemMessage {
                                    message: format!("Live session observation ended early: {err}"),
                                });
                                let _ = app_tx.send(AppEvent::RequestUiSnapshot);
                                let _ = app_tx.send(AppEvent::RequestQueuedWorkSnapshot);
                                let _ = app_tx.send(AppEvent::SessionObservationFinished { stream_id });
                                break;
                            }
                            None => {
                                coalescer.flush().await;
                                let _ = app_tx.send(AppEvent::SessionObservationFinished { stream_id });
                                break;
                            }
                        }
                    }
                }
            }
        });
    }
}

pub(super) struct TurnActivityCoalescer {
    stream_id: u64,
    app_tx: AppEventTx,
    pending_text: String,
    pending_correlation_id: Option<TurnActivityId>,
    flush: std::pin::Pin<Box<tokio::time::Sleep>>,
    flush_armed: bool,
    coalesced: usize,
}

impl TurnActivityCoalescer {
    pub(super) fn new(stream_id: u64, app_tx: AppEventTx) -> Self {
        Self {
            stream_id,
            app_tx,
            pending_text: String::new(),
            pending_correlation_id: None,
            flush: Box::pin(tokio::time::sleep(std::time::Duration::from_secs(86_400))),
            flush_armed: false,
            coalesced: 0,
        }
    }

    pub(super) async fn emit(&mut self, activity: TurnActivity) {
        match activity.event {
            TurnEvent::AssistantProseDelta { text } => {
                if text.is_empty() {
                    return;
                }
                if self.pending_text.is_empty() {
                    self.pending_correlation_id = Some(activity.correlation_id);
                    self.flush
                        .as_mut()
                        .reset(tokio::time::Instant::now() + TEXT_DELTA_REDRAW_INTERVAL);
                    self.flush_armed = true;
                } else {
                    self.coalesced += 1;
                }
                self.pending_text.push_str(&text);
            }
            event => {
                self.flush().await;
                let _ = self.app_tx.send(AppEvent::Session {
                    stream_id: self.stream_id,
                    activity: Box::new(TurnActivity { event, ..activity }),
                });
            }
        }
    }

    pub(super) async fn flush(&mut self) {
        if self.pending_text.is_empty() {
            self.flush_armed = false;
            return;
        }
        let text = std::mem::take(&mut self.pending_text);
        let correlation_id = self
            .pending_correlation_id
            .take()
            .unwrap_or_else(TurnActivityId::fresh);
        let _ = self.app_tx.send(AppEvent::Session {
            stream_id: self.stream_id,
            activity: Box::new(TurnActivity::new(
                correlation_id,
                TurnEvent::AssistantProseDelta { text },
            )),
        });
        if self.coalesced > 0 {
            tracing::trace!(
                stream_id = self.stream_id,
                coalesced = self.coalesced,
                "coalesced runtime text deltas"
            );
        }
        self.coalesced = 0;
        self.flush_armed = false;
    }
}

fn emit_observation_gap(app_tx: &AppEventTx, gap: &LiveReplayGap) {
    let _ = app_tx.send(AppEvent::SystemMessage {
        message: format!(
            "Live session observation skipped buffered events ({:?}); refreshed from the current session snapshot.",
            gap.reason
        ),
    });
    let _ = app_tx.send(AppEvent::RequestUiSnapshot);
    let _ = app_tx.send(AppEvent::RequestQueuedWorkSnapshot);
}

pub(super) struct UiSnapshotWorker {
    request_tx: mpsc::UnboundedSender<UiSnapshotRequest>,
    next_generation: u64,
    in_flight: bool,
    pending: bool,
}

struct UiSnapshotRequest {
    generation: u64,
    session: lash::LashSession,
}

impl UiSnapshotWorker {
    pub(super) fn spawn(app_tx: AppEventTx) -> Self {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<UiSnapshotRequest>();
        tokio::spawn(async move {
            while let Some(request) = request_rx.recv().await {
                let started = std::time::Instant::now();
                let snapshot = tokio::time::timeout(
                    std::time::Duration::from_millis(200),
                    collect_ui_snapshot(request.session.clone()),
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
                            processes: None,
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

    pub(super) fn request(&mut self, session: lash::LashSession) {
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
            session,
        });
    }

    pub(super) fn complete(&mut self, generation: u64, session: lash::LashSession) -> bool {
        if generation != self.next_generation {
            return false;
        }
        self.in_flight = false;
        if self.pending {
            self.request(session);
        }
        true
    }
}

pub(super) fn cleared_session_state(policy: SessionPolicy) -> SessionSnapshot {
    SessionSnapshot {
        policy,
        ..SessionSnapshot::default()
    }
}

pub(super) fn log_runtime_transition(
    phase: &str,
    app: &App,
    runtime: &Option<lash::LashSession>,
    runtime_return_rx_present: bool,
    cancel_token_present: bool,
    active_stream_id: u64,
) {
    tracing::debug!(
        phase,
        run_state = ?app.run_state,
        runtime_present = runtime.is_some(),
        runtime_return_rx_present,
        cancel_token_present,
        queued_turn_inputs = app.pending_turn_input_snapshot().len(),
        draft_presentations = app.queues.draft_presentations.len(),
        active_stream_id,
        live_turn = ?app.live.turn.as_ref().map(|turn| turn.run_state),
        "interactive runtime transition"
    );
}

pub(super) const TEXT_DELTA_REDRAW_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(33);

pub(super) fn record_queue_turn(ui_trace: &mut Option<UiTraceRecorder>, turn: &PreparedTurn) {
    if let Some(recorder) = ui_trace.as_mut() {
        recorder.record_queue_turn(turn);
    }
}

pub(super) fn record_queue_current_turn_input(
    ui_trace: &mut Option<UiTraceRecorder>,
    turn: &PreparedTurn,
) {
    if let Some(recorder) = ui_trace.as_mut() {
        recorder.record_queue_current_turn_input(turn);
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

pub(super) fn queued_turn_edit_matches(key: crossterm::event::KeyEvent) -> bool {
    queued_turn_edit_binding().matches(key)
}
