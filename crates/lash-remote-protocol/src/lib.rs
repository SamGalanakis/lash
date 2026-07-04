//! Wire DTOs for driving a lash runtime across a process boundary.
//!
//! Each domain module carries one slice of the protocol vocabulary
//! ([`llm`], [`turn_input`], [`turn_result`], [`processes`], [`triggers`],
//! [`prompt`], [`tools`], [`observations`], [`usage_activity`],
//! [`registry_errors`]); the crate root re-exports all of them, which is the
//! established public API for direct consumers of this crate. The
//! cross-cutting protocol handshake ([`REMOTE_PROTOCOL_VERSION`],
//! [`ensure_protocol_version`]) lives at the root itself.

pub mod llm;
pub mod observations;
pub mod processes;
pub mod prompt;
pub mod registry_errors;
pub mod tools;
pub mod triggers;
pub mod turn_input;
pub mod turn_result;
pub mod usage_activity;

pub use llm::*;
pub use observations::*;
pub use processes::*;
pub use prompt::*;
pub use registry_errors::*;
pub use tools::*;
pub use triggers::*;
pub use turn_input::*;
pub use turn_result::*;
pub use usage_activity::*;

// Bumped to 7: the process wire mirror gained the additive-but-breaking
// `Abandoned` terminal variant (on `RemoteProcessTerminalState`,
// `RemoteProcessAwaitOutput`, `RemoteProcessStatus`, `RemoteProcessLifecycleStatus`,
// `RemoteProcessStatusFilter`) plus the required `disposition` field and the
// `first_started`/`abandon_request`/lease-holder read-side fields (ADR 0019). A
// new tagged-enum variant cannot be deserialized by a version-6 peer, so per
// docs/reporting.html's "bump for a breaking wire change" policy the version
// dial moves; `validate()` rejects a version-6 envelope outright.
pub const REMOTE_PROTOCOL_VERSION: u32 = 7;

pub fn ensure_protocol_version(actual: u32) -> Result<(), RemoteProtocolError> {
    if actual == REMOTE_PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(RemoteProtocolError::UnsupportedProtocolVersion {
            actual,
            expected: REMOTE_PROTOCOL_VERSION,
        })
    }
}

#[cfg(feature = "core-conversions")]
mod core_conversions;

#[cfg(feature = "core-conversions")]
pub use core_conversions::{RemoteTurnActivitySink, replay_collected_activities};

#[cfg(test)]
mod tests;
