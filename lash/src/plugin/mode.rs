//! Mode-plugin traits and narrow session/runtime context wrappers.
//!
//! Execution modes (standard vs RLM) register their plugin
//! implementations here; the runtime narrows what a mode plugin can
//! poke at so external mode crates don't need direct access to
//! `Session` / `LashRuntime` internals.
//!
//! Split out of `plugin/mod.rs` for file size; `pub use` there keeps
//! the outer module path.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::{SessionAppendNode, SessionCreateRequest};
use crate::runtime::PersistedSessionState;
use crate::{ExecutionMode, ToolDefinition, ToolResult};

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

    /// Apply a mode-owned globals/state patch from either the session
    /// graph or an incoming append-nodes request. Today the only
    /// patch payload is the RLM lashlang globals patch.
    pub async fn apply_mode_globals_patch(
        &mut self,
        patch: &lash_rlm_types::RlmGlobalsPatchPluginBody,
    ) -> Result<(), crate::SessionError> {
        self.session.apply_mode_globals_patch(patch).await
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
    pub fn set_mode_turn_options(&mut self, options: crate::ModeTurnOptions) {
        self.runtime.set_mode_turn_options(options);
    }

    pub fn set_rlm_termination_mode(&mut self, termination: lash_rlm_types::RlmTermination) {
        self.runtime
            .set_mode_turn_options(crate::ModeTurnOptions::rlm(termination));
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
    fn mode_id(&self) -> &str;

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
#[derive(Clone, Debug, Serialize)]
pub struct ModeExtras {
    pub mode_id: ExecutionMode,
    #[serde(default)]
    pub payload: serde_json::Value,
}

impl Default for ModeExtras {
    fn default() -> Self {
        Self::empty(ExecutionMode::standard())
    }
}

impl ModeExtras {
    pub fn empty(mode_id: ExecutionMode) -> Self {
        Self {
            mode_id,
            payload: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    pub fn typed<T>(mode_id: ExecutionMode, extras: T) -> Result<Self, serde_json::Error>
    where
        T: Serialize,
    {
        Ok(Self {
            mode_id,
            payload: serde_json::to_value(extras)?,
        })
    }

    pub fn decode<T>(&self, expected_mode: &ExecutionMode) -> Result<Option<T>, serde_json::Error>
    where
        T: DeserializeOwned,
    {
        if &self.mode_id != expected_mode {
            return Ok(None);
        }
        serde_json::from_value(self.payload.clone()).map(Some)
    }
}

impl<'de> Deserialize<'de> for ModeExtras {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(object) = value.as_object() {
            if let (Some(mode_id), Some(payload)) = (object.get("mode_id"), object.get("payload")) {
                let mode_id = ExecutionMode::deserialize(mode_id.clone())
                    .map_err(serde::de::Error::custom)?;
                return Ok(Self {
                    mode_id,
                    payload: payload.clone(),
                });
            }
            if let Some(mode) = object.get("mode").and_then(serde_json::Value::as_str) {
                let mut payload = object.clone();
                payload.remove("mode");
                return Ok(Self {
                    mode_id: ExecutionMode::new(mode),
                    payload: serde_json::Value::Object(payload),
                });
            }
        }
        Err(serde::de::Error::custom("invalid mode extras payload"))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StandardCreateExtras {}

pub use lash_rlm_types::RlmCreateExtras;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_extras_reads_legacy_tagged_rlm_payload() {
        let extras: ModeExtras = serde_json::from_value(serde_json::json!({
            "mode": "rlm",
            "termination": {
                "kind": "finish",
                "schema": null
            }
        }))
        .expect("legacy extras");
        assert_eq!(extras.mode_id, ExecutionMode::new("rlm"));
        let decoded = extras
            .decode::<RlmCreateExtras>(&ExecutionMode::new("rlm"))
            .expect("decode")
            .expect("matching mode");
        assert!(matches!(
            decoded.termination,
            lash_rlm_types::RlmTermination::Finish { schema: None, .. }
        ));
    }

    #[test]
    fn mode_extras_round_trips_open_payload() {
        let standard = ModeExtras::default();
        assert_eq!(standard.mode_id, ExecutionMode::standard());
        assert_eq!(standard.payload, serde_json::json!({}));

        let extras = ModeExtras::typed(
            ExecutionMode::new("rlm"),
            RlmCreateExtras {
                termination: lash_rlm_types::RlmTermination::ProseWithoutFence,
            },
        )
        .expect("encode");
        let json = serde_json::to_value(&extras).expect("serialize");
        assert_eq!(json["mode_id"], "rlm");
        let decoded: ModeExtras = serde_json::from_value(json).expect("deserialize");
        assert_eq!(decoded.mode_id, ExecutionMode::new("rlm"));
    }
}
