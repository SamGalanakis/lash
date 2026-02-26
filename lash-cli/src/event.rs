use crossterm::event::Event as TermEvent;
use lash_core::AgentEvent;

/// Unified event type for the main loop.
pub enum AppEvent {
    Terminal(TermEvent),
    Agent {
        stream_id: u64,
        event: AgentEvent,
    },
    /// 100ms tick for spinner animation (only triggers redraw when agent is running).
    Tick,
    /// Graceful shutdown (e.g. SIGTERM).
    Quit,
}
