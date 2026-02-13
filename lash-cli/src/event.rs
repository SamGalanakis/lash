use crossterm::event::Event as TermEvent;
use lash::AgentEvent;

/// Unified event type for the main loop.
pub enum AppEvent {
    Terminal(TermEvent),
    Agent(AgentEvent),
}
