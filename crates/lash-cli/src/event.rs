use crossterm::event::Event as TermEvent;
use lash_core::{ProcessHandleSummary, SessionProcessEventKind, TurnActivity};
use lash_tui_extensions::TuiHostEffect;
use tokio::sync::mpsc;

use crate::prompt_model::{PromptRequest, PromptResponse};

/// Unified event type for the main loop.
pub enum AppEvent {
    Terminal(TermEvent),
    ClipboardImageReady {
        id: usize,
        png: anyhow::Result<Vec<u8>>,
    },
    UpdateCheckFinished {
        message: String,
    },
    Session {
        stream_id: u64,
        activity: Box<TurnActivity>,
    },
    Prompt {
        request: PromptRequest,
        response_tx: std::sync::mpsc::Sender<PromptResponse>,
    },
    UiSnapshot {
        generation: u64,
        result: UiSnapshotResult,
    },
    ProcessChanged {
        stream_id: u64,
        kind: SessionProcessEventKind,
        process_ids: Vec<String>,
    },
    RequestUiSnapshot,
    RequestPendingTurnInputSnapshot,
    SystemMessage {
        message: String,
    },
    SessionObservationFinished {
        stream_id: u64,
    },
    FrameRequested,
    /// 100ms tick for spinner animation (only triggers redraw when a session is running).
    Tick,
    /// The background `@`-completion file index has finished walking the
    /// project root and is now serving real matches. The interactive loop
    /// re-runs `update_suggestions` so any in-flight "indexing files…"
    /// placeholder is replaced with the real result.
    FileIndexReady,
    /// Graceful shutdown (e.g. SIGTERM).
    Quit,
}

pub struct UiSnapshotResult {
    pub effects: Vec<TuiHostEffect>,
    pub processes: Option<Vec<ProcessHandleSummary>>,
    pub duration: std::time::Duration,
    pub timed_out: bool,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventLane {
    High,
    Normal,
    Low,
}

pub struct QueuedAppEvent {
    pub event: AppEvent,
    pub lane: EventLane,
    pub enqueued_at: std::time::Instant,
}

#[derive(Clone)]
pub struct AppEventTx {
    high: mpsc::UnboundedSender<QueuedAppEvent>,
    normal: mpsc::UnboundedSender<QueuedAppEvent>,
    low: mpsc::UnboundedSender<QueuedAppEvent>,
    frame_pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub struct AppEventPump {
    high_rx: mpsc::UnboundedReceiver<QueuedAppEvent>,
    normal_rx: mpsc::UnboundedReceiver<QueuedAppEvent>,
    low_rx: mpsc::UnboundedReceiver<QueuedAppEvent>,
    tx: AppEventTx,
}

impl AppEventPump {
    pub fn new() -> Self {
        let (high, high_rx) = mpsc::unbounded_channel();
        let (normal, normal_rx) = mpsc::unbounded_channel();
        let (low, low_rx) = mpsc::unbounded_channel();
        let tx = AppEventTx {
            high,
            normal,
            low,
            frame_pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        Self {
            high_rx,
            normal_rx,
            low_rx,
            tx,
        }
    }

    pub fn sender(&self) -> AppEventTx {
        self.tx.clone()
    }

    pub fn mark_frame_consumed(&self) {
        self.tx
            .frame_pending
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn lane_depths(&self) -> (usize, usize, usize) {
        (self.high_rx.len(), self.normal_rx.len(), self.low_rx.len())
    }

    pub async fn recv(&mut self) -> Option<QueuedAppEvent> {
        if let Ok(event) = self.high_rx.try_recv() {
            return Some(event);
        }
        if let Ok(event) = self.normal_rx.try_recv() {
            return Some(event);
        }
        if let Ok(event) = self.low_rx.try_recv() {
            return Some(event);
        }

        tokio::select! {
            biased;
            event = self.high_rx.recv() => event,
            event = self.normal_rx.recv() => event,
            event = self.low_rx.recv() => event,
        }
    }
}

impl AppEventTx {
    pub fn send(&self, event: AppEvent) -> Result<(), ()> {
        if matches!(event, AppEvent::FrameRequested)
            && self
                .frame_pending
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(());
        }
        let lane = lane_for_event(&event);
        let queued = QueuedAppEvent {
            event,
            lane,
            enqueued_at: std::time::Instant::now(),
        };
        let result = match lane {
            EventLane::High => self.high.send(queued),
            EventLane::Normal => self.normal.send(queued),
            EventLane::Low => self.low.send(queued),
        };
        result.map_err(|_| ())
    }

    pub fn request_frame(&self) {
        let _ = self.send(AppEvent::FrameRequested);
    }
}

fn lane_for_event(event: &AppEvent) -> EventLane {
    match event {
        AppEvent::Terminal(_) | AppEvent::ClipboardImageReady { .. } | AppEvent::Quit => {
            EventLane::High
        }
        AppEvent::Session { .. }
        | AppEvent::ProcessChanged { .. }
        | AppEvent::Prompt { .. }
        | AppEvent::SystemMessage { .. }
        | AppEvent::SessionObservationFinished { .. } => EventLane::Normal,
        AppEvent::UpdateCheckFinished { .. }
        | AppEvent::UiSnapshot { .. }
        | AppEvent::RequestUiSnapshot
        | AppEvent::RequestPendingTurnInputSnapshot
        | AppEvent::FrameRequested
        | AppEvent::Tick
        | AppEvent::FileIndexReady => EventLane::Low,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn priority_pump_handles_high_priority_before_low_priority() {
        let mut pump = AppEventPump::new();
        let tx = pump.sender();

        assert!(tx.send(AppEvent::FileIndexReady).is_ok());
        assert!(tx.send(AppEvent::Tick).is_ok());
        assert!(tx.send(AppEvent::Quit).is_ok());

        let event = pump.recv().await.expect("event");
        assert_eq!(event.lane, EventLane::High);
        assert!(matches!(event.event, AppEvent::Quit));
    }

    #[tokio::test]
    async fn frame_requests_are_coalesced_until_consumed() {
        let mut pump = AppEventPump::new();
        let tx = pump.sender();

        tx.request_frame();
        tx.request_frame();

        let first = pump.recv().await.expect("frame");
        assert!(matches!(first.event, AppEvent::FrameRequested));
        assert!(pump.low_rx.try_recv().is_err());

        pump.mark_frame_consumed();
        tx.request_frame();
        let second = pump.recv().await.expect("second frame");
        assert!(matches!(second.event, AppEvent::FrameRequested));
    }
}
