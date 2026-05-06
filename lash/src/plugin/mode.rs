//! Mode-plugin traits and narrow session/runtime context wrappers.
//!
//! Execution modes (standard vs RLM) register their plugin
//! implementations here; the runtime narrows what a mode plugin can
//! poke at so external mode crates don't need direct access to
//! `Session` / `LashRuntime` internals.
//!
//! Split out of `plugin/mod.rs` for file size; `pub use` there keeps
//! the outer module path.

use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::{SessionAppendNode, SessionCreateRequest};
use crate::runtime::PersistedSessionState;
use crate::{
    ExecRequest, ExecResponse, ExecutionMode, ModeExecutionContext, ToolDefinition, ToolResult,
};

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

    async fn apply_session_extension(
        &self,
        _extension: crate::ModeSessionExtensionHandle,
    ) -> Result<(), crate::SessionError> {
        Err(crate::SessionError::Protocol(
            "execution mode does not accept session extensions".to_string(),
        ))
    }

    async fn validate_turn_extension(
        &self,
        _extension: &crate::ModeTurnExtensionHandle,
    ) -> Result<(), crate::SessionError> {
        Ok(())
    }

    async fn execute_code(
        &self,
        _ctx: ModeExecutionContext,
        _request: ExecRequest,
    ) -> Result<ExecResponse, crate::SessionError> {
        Err(crate::SessionError::RlmUnavailable)
    }

    fn execution_state_dirty(&self) -> bool {
        false
    }

    async fn snapshot_execution_state(
        &self,
        _ctx: ModeSessionContext<'_>,
    ) -> Result<Option<Vec<u8>>, crate::SessionError> {
        Ok(None)
    }

    async fn restore_execution_state(
        &self,
        _ctx: ModeSessionContext<'_>,
        _data: &[u8],
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
/// Exposes only generic per-session lifecycle capabilities. Mode-local
/// execution state is owned by the mode plugin itself and is accessed
/// through [`ModeSessionPlugin`] callbacks.
/// Prevents mode plugins from reaching into unrelated `Session`
/// internals.
pub struct ModeSessionContext<'a> {
    session_id: &'a str,
    projected_rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
}

impl<'a> ModeSessionContext<'a> {
    pub(crate) fn new(_session: &'a mut crate::Session, session_id: &'a str) -> Self {
        Self {
            session_id,
            projected_rlm_globals: Arc::new(serde_json::Map::new()),
        }
    }

    pub(crate) fn with_projected_rlm_globals(
        _session: &'a mut crate::Session,
        session_id: &'a str,
        projected_rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
    ) -> Self {
        Self {
            session_id,
            projected_rlm_globals,
        }
    }

    /// ID of the session being initialized/restored. Equivalent to the
    /// `session_id` previously passed as a separate argument.
    pub fn session_id(&self) -> &str {
        self.session_id
    }

    pub fn projected_rlm_globals(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.projected_rlm_globals
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

    pub fn set_mode_turn_options(&mut self, options: crate::ModeTurnOptions) {
        self.runtime.set_mode_turn_options(options);
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
            .decode::<lash_rlm_types::RlmCreateExtras>(&ExecutionMode::new("rlm"))
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
            lash_rlm_types::RlmCreateExtras {
                termination: lash_rlm_types::RlmTermination::default(),
                ..Default::default()
            },
        )
        .expect("encode");
        let json = serde_json::to_value(&extras).expect("serialize");
        assert_eq!(json["mode_id"], "rlm");
        let decoded: ModeExtras = serde_json::from_value(json).expect("deserialize");
        assert_eq!(decoded.mode_id, ExecutionMode::new("rlm"));
    }
}
