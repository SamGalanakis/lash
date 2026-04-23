use crossterm::event::{KeyCode, KeyModifiers};
use lash::*;
use lash_ui::{KeyChord as UiKeyChord, KeyCode as UiKeyCode, KeyModifiers as UiKeyModifiers};
use tokio::sync::mpsc;

use crate::app::{App, PreparedTurn};
use crate::event::AppEvent;
use crate::ui_trace::{UiTraceRecorder, drain_aux_ops_into};
use crate::{copy_binding, queued_turn_edit_binding};

#[derive(Clone)]
pub(super) struct TurnReplayPayload {
    pub(super) prepared_turn: PreparedTurn,
    pub(super) turn_input: TurnInput,
    pub(super) execution_mode: ExecutionMode,
}

pub(super) struct AppEventSink {
    pub(super) tx: mpsc::UnboundedSender<AppEvent>,
    pub(super) stream_id: u64,
}

#[async_trait::async_trait]
impl EventSink for AppEventSink {
    async fn emit(&self, event: SessionEvent) {
        let _ = self.tx.send(AppEvent::Session {
            stream_id: self.stream_id,
            event,
        });
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

#[derive(Default)]
pub(super) struct PendingTextDeltaBuffer {
    content: String,
    flush_at: Option<tokio::time::Instant>,
}

impl PendingTextDeltaBuffer {
    pub(super) fn push(&mut self, content: String) {
        self.push_at(content, tokio::time::Instant::now());
    }

    pub(super) fn push_at(&mut self, content: String, now: tokio::time::Instant) {
        if content.is_empty() {
            return;
        }
        if self.content.is_empty() {
            self.flush_at = Some(now + TEXT_DELTA_REDRAW_INTERVAL);
        }
        self.content.push_str(&content);
    }

    pub(super) fn flush_deadline(&self) -> Option<tokio::time::Instant> {
        self.flush_at
    }

    pub(super) fn should_flush(&self, now: tokio::time::Instant) -> bool {
        self.flush_at.is_some_and(|deadline| deadline <= now)
    }

    pub(super) fn take_event(&mut self) -> Option<SessionEvent> {
        if self.content.is_empty() {
            self.flush_at = None;
            return None;
        }
        self.flush_at = None;
        Some(SessionEvent::TextDelta {
            content: std::mem::take(&mut self.content),
        })
    }

    pub(super) fn clear(&mut self) {
        self.content.clear();
        self.flush_at = None;
    }
}

pub(super) fn flush_pending_text_deltas(
    app: &mut App,
    pending: &mut PendingTextDeltaBuffer,
    ui_trace: Option<&mut UiTraceRecorder>,
) {
    if let Some(event) = pending.take_event() {
        if let Some(recorder) = ui_trace {
            recorder.record_session_event(&event);
        }
        app.handle_session_event(event);
        app.dirty = true;
    }
}

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
