use crossterm::event::Event as TermEvent;
use lash::SessionEvent;

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
        event: SessionEvent,
    },
    /// 100ms tick for spinner animation (only triggers redraw when a session is running).
    Tick,
    /// Graceful shutdown (e.g. SIGTERM).
    Quit,
}
