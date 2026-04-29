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
    /// The background `@`-completion file index has finished walking the
    /// project root and is now serving real matches. The interactive loop
    /// re-runs `update_suggestions` so any in-flight "indexing files…"
    /// placeholder is replaced with the real result.
    FileIndexReady,
    /// Graceful shutdown (e.g. SIGTERM).
    Quit,
}
