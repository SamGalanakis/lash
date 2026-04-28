//! Pure RLM execution mode: a trajectory-shaped mode that uses lashlang as
//! the persistent REPL. Model prose is projected as trajectory reasoning,
//! fenced `lashlang` is executed, `print` yields observations, and
//! `submit` yields the final value.

mod driver;
mod plugin;
mod stream_mask;

pub use driver::{RlmpureProjectorConfig, build_rlmpure_preamble, rlmpure_execution_section};
pub use plugin::{BuiltinRlmpureModePluginFactory, RlmpureModePluginConfig};
