//! RLM execution mode: a trajectory-shaped mode that uses lashlang as
//! the persistent REPL. Model prose is projected as trajectory reasoning,
//! fenced `lashlang` is executed, `print` yields observations, and
//! `submit` yields the final value.

mod control_tools;
mod driver;
mod executor;
mod plugin;
mod projected_bindings;
mod projection;
mod protocol;
mod rlm_support;
mod stream_mask;

pub use control_tools::{continue_as_tool_definition, rlm_seed_initial_nodes};
pub use driver::{RlmProjectorConfig, build_rlm_preamble};
pub use plugin::{BuiltinRlmModePluginFactory, RlmModePluginConfig};
pub use projected_bindings::{
    RlmProjectedBindings, RlmTurnInputExt, rlm_session_projection_extension,
};
pub use projection::{
    RlmHistoryProjection, decode_rlm_mode_event, project_rlm_globals_from_events,
    rlm_history_projection, rlm_mode_event,
};
pub use protocol::{
    RlmDriver, RlmPromptFeatures, contains_closed_lashlang_fence, rlm_execution_section,
    rlm_execution_section_with_features,
};
pub use rlm_support::{BoundVariablesCache, format_budget_suffix};
