//! `LashRuntime` session-graph and execution-state operations.
//!
//! Extracted from `runtime/mod.rs`. This file re-opens `impl LashRuntime`;
//! no types live here and no public API is changed.

use std::sync::Arc;

use crate::{PluginActionInvokeError, SessionError};

use super::LashRuntime;
use super::state::{RuntimeSessionState, append_session_nodes_to_state, normalize_session_graph};

impl LashRuntime {
    /// Replace the host-owned state envelope.
    pub fn set_persisted_state(&mut self, state: RuntimeSessionState) -> Result<(), SessionError> {
        let mut state = state;
        normalize_session_graph(&mut state);
        if let Some(session) = self.session.as_ref() {
            session.invalidate_runtime_caches();
            // Restore the persisted tool surface so the live registry matches the
            // state being installed (mirrors `from_host_state`). Without this the
            // registry keeps its prior generation/tools and silently diverges from
            // `state`. `restore_state` adopts the snapshot's generation, so a
            // surface that reached generation >= 2 restores cleanly.
            if let Some(tool_state) = state.tool_state_snapshot.clone() {
                session
                    .plugins()
                    .tool_registry()
                    .restore_state(tool_state)
                    .map_err(|err| SessionError::Protocol(err.to_string()))?;
            }
            let snapshot = state.plugin_snapshot.clone().unwrap_or_default();
            session
                .plugins()
                .restore(&snapshot)
                .map_err(|err| SessionError::Protocol(err.to_string()))?;
            state.plugin_snapshot_revision =
                Some(session.plugins().snapshot_revision_fingerprint());
        }
        self.policy = state.policy.clone();
        self.protocol_turn_options = state.protocol_turn_options.clone();
        self.state = state;
        Ok(())
    }

    pub async fn append_session_nodes(
        &mut self,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, SessionError> {
        self.refresh_session_graph_from_store().await?;
        if let Some(required) = request.requires_ancestor_node_id.as_deref()
            && !self.state.session_graph.active_path_contains(required)
        {
            return Ok(crate::AppendSessionNodesResult::StaleBranch {
                current_leaf_node_id: self.state.session_graph.leaf_node_id.clone(),
            });
        }
        let node_ids = append_session_nodes_to_state(&mut self.state, &request.nodes);
        if let Some(session) = self.session.as_mut() {
            let protocol_session = Arc::clone(session.plugins().protocol_session());
            let session_id = self.state.session_id.clone();
            protocol_session
                .append_session_nodes(
                    crate::plugin::ProtocolSessionContext::new(session, &session_id),
                    &request.nodes,
                )
                .await?;
        }
        self.stamp_live_plugin_state();
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

    pub async fn apply_protocol_session_extension(
        &mut self,
        extension: crate::ProtocolSessionExtensionHandle,
    ) -> Result<(), SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session is not available".to_string(),
            ));
        };
        let protocol_session = Arc::clone(session.plugins().protocol_session());
        protocol_session.apply_session_extension(extension).await
    }

    pub async fn validate_protocol_turn_extension(
        &mut self,
        extension: &crate::ProtocolTurnExtensionHandle,
    ) -> Result<(), SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session is not available".to_string(),
            ));
        };
        let protocol_session = Arc::clone(session.plugins().protocol_session());
        protocol_session.validate_turn_extension(extension).await
    }

    pub async fn branch_to_node(
        &mut self,
        node_id: Option<String>,
    ) -> Result<crate::SessionSnapshot, SessionError> {
        let mut state = self.export_state();
        state.session_graph.branch_to(node_id);
        let mut persisted_state = RuntimeSessionState::from_snapshot(state);
        normalize_session_graph(&mut persisted_state);

        let policy = persisted_state.policy.clone();
        let host = self.host.clone();
        let services = self.services.clone();
        let managed_sessions = Arc::clone(&self.managed_sessions);
        let managed_turns = Arc::clone(&self.managed_turns);
        let process_sync_needed = Arc::clone(&self.process_sync_needed);
        let runtime_scope_id = Arc::clone(&self.runtime_scope_id);
        let turn_phase_probe = self.turn_phase_probe.clone();

        let mut rebuilt = Self::from_host_state(policy, host, services, persisted_state).await?;
        rebuilt.managed_sessions = managed_sessions;
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
    /// runtimes, not serialized placeholders. Foreground activation must therefore
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

    /// Explicitly snapshot protocol-local execution state, if any.
    pub async fn snapshot_execution_state(&mut self) -> Result<Option<Vec<u8>>, SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let code_executor = session
            .plugins()
            .code_executor()
            .ok_or(SessionError::CodeExecutionUnavailable)?;
        let session_id = self.state.session_id.clone();
        let blob = code_executor
            .snapshot_execution_state(crate::plugin::ProtocolSessionContext::new(
                session,
                &session_id,
            ))
            .await?;
        self.state.execution_state_snapshot = blob.clone();
        Ok(blob)
    }

    /// Explicitly restore protocol-local execution state from an opaque snapshot blob.
    pub async fn restore_execution_state(&mut self, snapshot: &[u8]) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let code_executor = session
            .plugins()
            .code_executor()
            .ok_or(SessionError::CodeExecutionUnavailable)?;
        let session_id = self.state.session_id.clone();
        code_executor
            .restore_execution_state(
                crate::plugin::ProtocolSessionContext::new(session, &session_id),
                snapshot,
            )
            .await?;
        self.state.execution_state_snapshot = Some(snapshot.to_vec());
        Ok(())
    }

    pub async fn install_lashlang_trigger_source(
        &mut self,
        source: &str,
    ) -> Result<crate::SessionTriggerInstallReport, SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let tool_surface = session
            .tool_surface(&self.state.session_id)
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        let surface = crate::session::lashlang_surface_from_tool_surface(
            &tool_surface,
            session.plugins().lashlang_abilities(),
            session.plugins().lashlang_resources(),
        );
        let report = session
            .plugins()
            .install_lashlang_trigger_source(
                source,
                surface,
                self.host.core.lashlang_artifact_store.as_ref(),
            )
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        self.stamp_live_plugin_state();
        if let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        {
            let commit = crate::store::RuntimeCommit::persisted_state(&self.state, &[]);
            let result = store.commit_runtime_state(commit).await.map_err(|err| {
                SessionError::Protocol(format!("failed to persist trigger registry state: {err}"))
            })?;
            self.state.apply_persisted_commit_result(result);
        }
        Ok(report)
    }

    pub async fn emit_host_event(
        &mut self,
        resource_type: &str,
        alias: &str,
        event: &str,
        payload: serde_json::Value,
    ) -> Result<crate::HostEventEmitReport, SessionError> {
        {
            let Some(session) = self.session.as_ref() else {
                return Err(SessionError::Protocol(
                    "runtime session not available".to_string(),
                ));
            };
            crate::session::triggers::validate_host_event(
                session.plugins(),
                resource_type,
                alias,
                event,
                &payload,
            )
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        }
        let append = self
            .append_session_nodes(crate::AppendSessionNodesRequest {
                nodes: vec![crate::SessionAppendNode::plugin(
                    "lash.host_event",
                    serde_json::json!({
                        "resource_type": resource_type,
                        "alias": alias,
                        "event": event,
                        "payload": payload.clone(),
                    }),
                )],
                requires_ancestor_node_id: None,
            })
            .await?;
        let host_event_node_id = match append {
            crate::AppendSessionNodesResult::Appended { node_ids, .. } => {
                node_ids.into_iter().next().unwrap_or_default()
            }
            crate::AppendSessionNodesResult::StaleBranch {
                current_leaf_node_id,
            } => {
                return Err(SessionError::Protocol(format!(
                    "host event append targeted a stale session branch at {:?}",
                    current_leaf_node_id
                )));
            }
        };
        let host_event_invocation = crate::runtime::causal::session_node_invocation(
            &self.state.session_id,
            host_event_node_id,
        );
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let manager = self
            .runtime_session_manager()
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        crate::session::triggers::emit_host_event(
            &self.state.session_id,
            Arc::clone(session.plugins()),
            manager as Arc<dyn crate::ProcessService>,
            host_event_invocation,
            resource_type,
            alias,
            event,
            payload,
        )
        .await
        .map_err(|err| SessionError::Protocol(err.to_string()))
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
            .invoke_plugin_action(
                name,
                args,
                session_id,
                true,
                manager.clone() as Arc<dyn crate::plugin::RuntimeSessionHost>,
                manager as Arc<dyn crate::ProcessService>,
            )
            .await
    }
}
