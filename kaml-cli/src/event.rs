use crossterm::event::Event as TermEvent;
use kaml::AgentEvent;

/// Unified event type for the main loop.
pub enum AppEvent {
    Terminal(TermEvent),
    Agent(AgentEvent),
}
