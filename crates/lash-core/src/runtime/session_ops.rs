//! `LashRuntime` session-graph and execution-state operations.
//!
//! Extracted from `runtime/mod.rs`. This file re-opens `impl LashRuntime`;
//! no types live here and no public API is changed.

use std::sync::Arc;

use crate::{PluginActionInvokeError, SessionError};

use super::LashRuntime;
use super::state::{
    PersistedSessionState, SessionStateEnvelope, append_session_nodes_to_state,
    normalize_session_graph,
};

impl LashRuntime {
    /// Replace the host-owned state envelope.
    pub fn set_persisted_state(&mut self, state: PersistedSessionState) {
        let mut state = state;
        normalize_session_graph(&mut state);
        if let Some(session) = self.session.as_ref() {
            let snapshot = state.plugin_snapshot.clone().unwrap_or_default();
            if let Err(err) = session.plugins().restore(&snapshot) {
                tracing::warn!("failed to restore plugin snapshot in set_state: {err}");
            }
            state.plugin_snapshot_revision =
                Some(session.plugins().snapshot_revision_fingerprint());
        }
        self.policy = state.policy.clone();
        self.mode_turn_options = state.mode_turn_options.clone();
        self.state = state;
    }

    pub async fn append_session_nodes(
        &mut self,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, SessionError> {
        self.refresh_session_graph_from_store().await;
        if let Some(required) = request.requires_ancestor_node_id.as_deref()
            && !self.state.session_graph.active_path_contains(required)
        {
            return Ok(crate::AppendSessionNodesResult::StaleBranch {
                current_leaf_node_id: self.state.session_graph.leaf_node_id.clone(),
            });
        }
        let node_ids = append_session_nodes_to_state(&mut self.state, &request.nodes);
        if let Some(session) = self.session.as_mut() {
            let mode_session = Arc::clone(session.plugins().mode_session());
            let session_id = self.state.session_id.clone();
            mode_session
                .append_session_nodes(
                    crate::plugin::ModeSessionContext::new(session, &session_id),
                    &request.nodes,
                )
                .await?;
        }
        if let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        {
            let graph = crate::store::GraphCommitDelta::Append {
                nodes: node_ids
                    .iter()
                    .filter_map(|id| self.state.session_graph.find_node(id).cloned())
                    .collect(),
                leaf_node_id: self.state.session_graph.leaf_node_id.clone(),
            };
            let commit = crate::store::RuntimeCommit::persisted_state_with_graph_commit(
                &self.state,
                graph,
                &[],
            );
            match store.commit_runtime_state(commit).await {
                Ok(result) => self.state.apply_persisted_commit_result(result),
                Err(err) => tracing::warn!("failed to persist runtime state: {err}"),
            }
        }
        Ok(crate::AppendSessionNodesResult::Appended {
            node_ids,
            leaf_node_id: self
                .state
                .session_graph
                .leaf_node_id
                .clone()
                .unwrap_or_default(),
        })
    }

    pub async fn apply_mode_session_extension(
        &mut self,
        extension: crate::ModeSessionExtensionHandle,
    ) -> Result<(), SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session is not available".to_string(),
            ));
        };
        let mode_session = Arc::clone(session.plugins().mode_session());
        mode_session.apply_session_extension(extension).await
    }

    pub async fn validate_mode_turn_extension(
        &mut self,
        extension: &crate::ModeTurnExtensionHandle,
    ) -> Result<(), SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session is not available".to_string(),
            ));
        };
        let mode_session = Arc::clone(session.plugins().mode_session());
        mode_session.validate_turn_extension(extension).await
    }

    pub async fn branch_to_node(
        &mut self,
        node_id: Option<String>,
    ) -> Result<SessionStateEnvelope, SessionError> {
        let mut state = self.export_state();
        state.session_graph.branch_to(node_id);
        let mut persisted_state = PersistedSessionState::from_state(state);
        normalize_session_graph(&mut persisted_state);

        let policy = persisted_state.policy.clone();
        let host = self.host.clone();
        let services = self.services.clone();
        let managed_sessions = Arc::clone(&self.managed_sessions);
        let active_handoff_continuations = Arc::clone(&self.active_handoff_continuations);
        let managed_turns = Arc::clone(&self.managed_turns);
        let process_sync_needed = Arc::clone(&self.process_sync_needed);
        let runtime_scope_id = Arc::clone(&self.runtime_scope_id);
        let turn_phase_probe = self.turn_phase_probe.clone();

        let mut rebuilt = Self::from_host_state(policy, host, services, persisted_state).await?;
        rebuilt.managed_sessions = managed_sessions;
        rebuilt.active_handoff_continuations = active_handoff_continuations;
        rebuilt.managed_turns = managed_turns;
        rebuilt.process_sync_needed = process_sync_needed;
        rebuilt.runtime_scope_id = runtime_scope_id;
        rebuilt.turn_phase_probe = turn_phase_probe;

        let exported = rebuilt.export_state();
        *self = rebuilt;
        Ok(exported)
    }

    /// Promote a managed child session into the foreground runtime.
    ///
    /// Child sessions created through `RuntimeSessionHost::create_session` are real
    /// runtimes, not serialized placeholders. Foreground handoff must therefore
    /// claim that runtime instead of reconstructing a new empty state in the UI.
    pub async fn activate_managed_session(&mut self, session_id: &str) -> Result<(), SessionError> {
        let child = {
            let mut registry = self.managed_sessions.lock().await;
            registry.remove(session_id).ok_or_else(|| {
                SessionError::Protocol(format!("unknown managed session `{session_id}`"))
            })?
        };
        let child = child.try_into_runtime().map_err(|_| {
            SessionError::Protocol(format!("managed session `{session_id}` is still in use"))
        })?;
        *self = child;
        Ok(())
    }

    /// Reset the RLM session on the underlying session runtime.
    pub async fn reset_session(&mut self) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        session.reset().await
    }

    /// Explicitly snapshot execution-mode-local state, if any.
    pub async fn snapshot_execution_state(&mut self) -> Result<Option<Vec<u8>>, SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let mode_session = Arc::clone(session.plugins().mode_session());
        let session_id = self.state.session_id.clone();
        let blob = mode_session
            .snapshot_execution_state(crate::plugin::ModeSessionContext::new(session, &session_id))
            .await?;
        self.state.execution_state_snapshot = blob.clone();
        Ok(blob)
    }

    /// Explicitly restore execution-mode-local state from an opaque snapshot blob.
    pub async fn restore_execution_state(&mut self, snapshot: &[u8]) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let mode_session = Arc::clone(session.plugins().mode_session());
        let session_id = self.state.session_id.clone();
        mode_session
            .restore_execution_state(
                crate::plugin::ModeSessionContext::new(session, &session_id),
                snapshot,
            )
            .await?;
        self.state.execution_state_snapshot = Some(snapshot.to_vec());
        Ok(())
    }

    pub async fn invoke_plugin_action(
        &self,
        name: &str,
        args: serde_json::Value,
        session_id: Option<String>,
    ) -> Result<crate::ToolResult, PluginActionInvokeError> {
        let manager = self.runtime_session_manager()?;
        let Some(session) = self.session.as_ref() else {
            return Err(PluginActionInvokeError::Unknown(name.to_string()));
        };
        session
            .plugins()
            .invoke_plugin_action(name, args, session_id, true, manager)
            .await
    }
}
