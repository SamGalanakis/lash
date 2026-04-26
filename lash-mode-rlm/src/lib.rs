//! RLM execution mode: the model writes a single fenced `lashlang`
//! block per turn and the runtime executes it inside an in-process
//! lashlang runtime with session-scoped globals.
//!
//! Public surface:
//!
//! - [`BuiltinRlmModePluginFactory`] — plugin that wires everything
//!   together: protocol driver, session state management, the shared
//!   discovery/load tools, "Bound Variables" / "Print Output" prompt
//!   contributions, and the stream mask that suppresses the fenced
//!   body + raises `abort_stream` when the fence closes.
//! - [`rlm_execution_section`] — the execution-mode prompt text.
//! - [`LASHLANG_LANGUAGE_REFERENCE`] — the shared lashlang language
//!   reference used by RLM-family modes.
//! - [`contains_closed_lashlang_fence`] — exposed so alternative
//!   fence-close detectors (e.g. integration tests) can reuse the same
//!   rule.

mod driver;
mod plugin;
mod rlm_support;
mod stream_mask;

pub use driver::{
    LASHLANG_LANGUAGE_REFERENCE, RlmDriver, contains_closed_lashlang_fence, rlm_execution_section,
};
pub use plugin::{BuiltinRlmModePluginFactory, RlmModePluginConfig};
