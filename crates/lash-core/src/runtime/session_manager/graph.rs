use super::*;
use std::sync::atomic::Ordering;

impl CurrentSessionCapability {
    pub(in crate::runtime::session_manager) async fn append_session_nodes(
        &self,
        managed: &ManagedSessionCapability,
        usage: &UsageCapability,
        background: &ProcessCapability,
        session_id: &str,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, crate::PluginError> {
        if let Some(runtime) = {
            let registry = managed.registry.lock().await;
            registry.get(session_id).cloned()
        } {
            let mut writer = runtime.runtime.lock().await;
            let result = writer
                .append_session_nodes(request)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
            runtime.publish_from(&writer);
            return Ok(result);
        }

        if session_id != self.session_id {
            return Err(crate::PluginError::Session(format!(
                "unknown session `{session_id}`"
            )));
        }

        let Some(store) = &self.store else {
            return Err(crate::PluginError::Session(
                "session graph mutation requires a runtime store".to_string(),
            ));
        };

        let mut state = if usage.persist_to_store {
            self.current_snapshot_for_store_write().await
        } else {
            let mut state = self.snapshot.to_snapshot();
            super::normalize_session_graph(&mut state);
            state
        };
        let usage_deltas = if usage.persist_to_store {
            usage.merge_drained_token_ledger(&mut state)
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
        let graph = crate::store::GraphCommitDelta::Append {
            nodes: node_ids
                .iter()
                .filter_map(|id| state.session_graph.find_node(id).cloned())
                .collect(),
            leaf_node_id: state.session_graph.leaf_node_id.clone(),
        };
        let commit = crate::store::RuntimeCommit::persisted_state_with_graph_commit(
            &state,
            graph,
            &usage_deltas,
        );
        let result = store
            .commit_runtime_state(commit)
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        state.apply_persisted_commit_result(result);
        background.sync_needed.store(true, Ordering::Release);
        Ok(crate::AppendSessionNodesResult::Appended {
            node_ids,
            leaf_node_id,
        })
    }
}
