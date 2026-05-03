use super::*;
use std::sync::atomic::Ordering;

impl RuntimeSessionManager {
    pub(in crate::runtime::session_manager) async fn append_session_nodes(
        &self,
        session_id: &str,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, crate::PluginError> {
        if let Some(runtime) = {
            let registry = self.managed.registry.lock().await;
            registry.get(session_id).cloned()
        } {
            let mut runtime = runtime.lock().await;
            return runtime
                .append_session_nodes(request)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()));
        }

        if session_id != self.current.session_id {
            return Err(crate::PluginError::Session(format!(
                "unknown session `{session_id}`"
            )));
        }

        let Some(store) = &self.current.store else {
            return Err(crate::PluginError::Session(
                "session graph mutation requires a runtime store".to_string(),
            ));
        };

        let mut state = if self.usage.persist_to_store {
            self.current_snapshot_for_store_write().await
        } else {
            let mut state = self.current.snapshot.to_snapshot();
            super::normalize_session_graph(&mut state);
            state
        };
        let usage_deltas = if self.usage.persist_to_store {
            self.merge_drained_token_ledger(&mut state)
        } else {
            Vec::new()
        };
        if let Some(required) = request.requires_ancestor_node_id.as_deref()
            && !state.session_graph.active_path_contains(required)
        {
            return Ok(crate::AppendSessionNodesResult::StaleBranch {
                current_leaf_node_id: state.session_graph.leaf_node_id.clone(),
            });
        }
        let node_ids = append_session_nodes_to_state(&mut state, &request.nodes);
        let leaf_node_id = state.session_graph.leaf_node_id.clone().unwrap_or_default();
        let commit = crate::store::PersistedStateCommit::persisted_state(&state, &usage_deltas);
        let result = crate::store::apply_runtime_commit(store.as_ref(), commit)
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        state.apply_persisted_commit_result(result);
        self.background.sync_needed.store(true, Ordering::Release);
        Ok(crate::AppendSessionNodesResult::Appended {
            node_ids,
            leaf_node_id,
        })
    }
}
