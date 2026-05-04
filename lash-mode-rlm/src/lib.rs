//! RLM execution mode: a trajectory-shaped mode that uses lashlang as
//! the persistent REPL. Model prose is projected as trajectory reasoning,
//! fenced `lashlang` is executed, `print` yields observations, and
//! `submit` yields the final value.

mod driver;
mod executor;
mod plugin;
mod protocol;
mod rlm_support;
mod stream_mask;

pub use driver::{RlmProjectorConfig, build_rlm_preamble};
pub use plugin::{BuiltinRlmModePluginFactory, RlmModePluginConfig};
pub use protocol::{
    LASHLANG_LANGUAGE_REFERENCE, RlmDriver, contains_closed_lashlang_fence, rlm_execution_section,
};
pub use rlm_support::{
    BoundVariablesCache, apply_globals_patch_nodes, bound_variables_prompt_contributions,
    budget_prompt_contributions, restore_execution_state_and_globals,
};
