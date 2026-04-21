//! Mode-plugin traits and narrow session/runtime context wrappers.
//!
//! Execution modes (standard vs RLM) register their plugin
//! implementations here; the runtime narrows what a mode plugin can
//! poke at so external mode crates don't need direct access to
//! `Session` / `LashRuntime` internals.
//!
//! Split out of `plugin/mod.rs` for file size; `pub use` there keeps
//! the outer module path.

use serde::{Deserialize, Serialize};

use super::{RlmTermination, SessionAppendNode, SessionCreateRequest};
use crate::runtime::PersistedSessionState;
use crate::{ToolDefinition, ToolResult};

/// Session-scoped plugin that initializes, restores, and extends mode
/// state across a session's lifecycle. External mode crates implement
/// this via context wrappers ([`ModeSessionContext`],
/// [`ModeRuntimeContext`]) so they don't need direct access to
/// `Session`/`LashRuntime` internals — the context narrows what a
/// plugin can poke at to the capabilities any execution mode
/// reasonably needs.
#[async_trait::async_trait]
pub trait ModeSessionPlugin: Send + Sync {
    async fn initialize_session(
        &self,
        _ctx: ModeSessionContext<'_>,
    ) -> Result<(), crate::SessionError> {
        Ok(())
    }

    async fn restore_session(
        &self,
        _ctx: ModeSessionContext<'_>,
        _state: &PersistedSessionState,
    ) -> Result<(), crate::SessionError> {
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        _ctx: ModeSessionContext<'_>,
        _nodes: &[SessionAppendNode],
    ) -> Result<(), crate::SessionError> {
        Ok(())
    }

    fn configure_runtime_from_request(
        &self,
        _ctx: ModeRuntimeContext<'_>,
        _request: &SessionCreateRequest,
    ) {
    }
}

/// Narrow wrapper around `Session` that mode plugins use to
/// initialize, restore, and extend their per-session state.
///
/// Exposes only the capabilities every mode reasonably needs
/// (starting the lashlang execution backend, configuring output
/// projection, restoring execution state, applying globals patches).
/// Prevents mode plugins from reaching into unrelated `Session`
/// internals.
pub struct ModeSessionContext<'a> {
    session: &'a mut crate::Session,
    session_id: &'a str,
}

impl<'a> ModeSessionContext<'a> {
    pub(crate) fn new(session: &'a mut crate::Session, session_id: &'a str) -> Self {
        Self {
            session,
            session_id,
        }
    }

    /// ID of the session being initialized/restored. Equivalent to the
    /// `session_id` previously passed as a separate argument.
    pub fn session_id(&self) -> &str {
        self.session_id
    }

    /// Start the embedded lashlang execution backend for this session
    /// (no-op if already running). Typically called in
    /// `initialize_session` by modes that dispatch work via lashlang.
    pub async fn start_lashlang_runtime(&mut self) -> Result<(), crate::SessionError> {
        self.session.start_lashlang_runtime(self.session_id).await
    }

    /// Configure how tool-call / print output is truncated before it
    /// flows back into the model. Mode plugins supply this from their
    /// own config.
    pub fn set_execution_output_projection(
        &mut self,
        config: crate::ToolResultProjectionPluginConfig,
    ) {
        self.session.set_execution_output_projection(config);
    }

    /// Restore the lashlang execution backend's globals from a
    /// persisted snapshot.
    pub async fn restore_execution_state(
        &mut self,
        data: &[u8],
    ) -> Result<(), crate::SessionError> {
        self.session.restore_execution_state(data).await
    }

    /// Apply an RLM globals patch (assignments + unsets) from either
    /// the session graph or an incoming append-nodes request.
    pub async fn apply_rlm_globals_patch(
        &mut self,
        patch: &crate::RlmGlobalsPatchPluginBody,
    ) -> Result<(), crate::SessionError> {
        self.session.apply_rlm_globals_patch(patch).await
    }
}

/// Narrow wrapper around `LashRuntime` that mode plugins use when
/// configuring the runtime from a fresh `SessionCreateRequest`.
///
/// Exposes only the runtime-level capabilities modes need to set
/// (termination contract, etc.) so plugins don't reach into unrelated
/// runtime internals.
pub struct ModeRuntimeContext<'a> {
    runtime: &'a mut crate::runtime::LashRuntime,
}

impl<'a> ModeRuntimeContext<'a> {
    pub(crate) fn new(runtime: &'a mut crate::runtime::LashRuntime) -> Self {
        Self { runtime }
    }

    /// Set how the session's embedded lashlang runtime terminates:
    /// `ProseWithoutFence` for chat-style sessions, or `Finish` with
    /// an optional output schema for typed-RLM sessions.
    pub fn set_termination_mode(&mut self, termination: crate::RlmTermination) {
        self.runtime.set_repl_termination(termination);
    }
}

#[async_trait::async_trait]
pub trait ModeNativeToolsPlugin: Send + Sync {
    fn definitions(&self) -> Vec<ToolDefinition>;

    async fn execute(
        &self,
        context: &crate::tool_dispatch::ToolDispatchContext,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&crate::ProgressSender>,
    ) -> Option<ToolResult>;
}

/// Singleton plugin slot that owns the `ProtocolDriverHandle` and
/// associated preamble (prompt text, tool surface, sync/async flag)
/// for a given execution mode. Mode-specific crates
/// (`lash-mode-standard`, `lash-mode-rlm`) register one implementation
/// each; the runtime picks the one whose `mode_id` matches the session
/// policy's execution mode, falling back to `build_mode_preamble`
/// when no plugin claims the slot.
pub trait ModeProtocolDriverPlugin: Send + Sync {
    /// Execution-mode identifier this driver implements (e.g.
    /// `"standard"`, `"rlm"`). Matched against
    /// `ExecutionMode::plugin_id()` at preamble-build time.
    fn mode_id(&self) -> &'static str;

    /// Build the `ModePreamble` (driver handle + prompt text + tool
    /// surface metadata) for a turn in this mode.
    fn build_preamble(&self, input: crate::ModeBuildInput) -> crate::ModePreamble;
}

/// Mode-specific extras carried on a `SessionCreateRequest`.
///
/// Each variant matches an `ExecutionMode` value and carries the
/// settings only that mode cares about. Adding a new mode means adding
/// a new variant with its own struct — no mode-specific fields ever
/// leak into the base request.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ModeExtras {
    Standard(StandardCreateExtras),
    Rlm(RlmCreateExtras),
}

impl Default for ModeExtras {
    fn default() -> Self {
        Self::Standard(StandardCreateExtras::default())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StandardCreateExtras {}

/// RLM-mode session config. Carries the choice of how the model
/// terminates the session (prose vs `submit`-with-optional-schema).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RlmCreateExtras {
    #[serde(default)]
    pub termination: RlmTermination,
}
