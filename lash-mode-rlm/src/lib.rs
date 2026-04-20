//! RLM execution mode: the model writes a single fenced `lashlang`
//! block per turn and the runtime executes it inside an in-process
//! lashlang runtime with session-scoped globals.
//!
//! Public surface:
//!
//! - [`BuiltinRlmModePluginFactory`] — plugin that wires everything
//!   together: protocol driver, session state management, `search_tools`
//!   provider, "Bound Variables" / "Print Output" prompt contributions,
//!   and the stream mask that suppresses the fenced body + raises
//!   `abort_stream` when the fence closes.
//! - [`RLM_EXECUTION_SECTION`] — the execution-mode prompt text (so
//!   downstream crates can lightly augment the contract if they need).
//! - [`contains_closed_lashlang_fence`] — exposed so alternative
//!   fence-close detectors (e.g. integration tests) can reuse the same
//!   rule.

mod driver;
mod plugin;
mod rlm_support;
mod stream_mask;

pub use driver::{RLM_EXECUTION_SECTION, RlmDriver, contains_closed_lashlang_fence};
pub use plugin::{BuiltinRlmModePluginFactory, RlmModePluginConfig};
