use std::sync::Arc;

use lash_core::plugin::runtime_host::RuntimeSessionHost;
use lash_core::plugin::{HistoryError, PluginError};
use lash_core::{
    AppendSessionNodesRequest, AppendSessionNodesResult, DirectCompletion, DirectRequest,
    SessionAppendNode, SessionGraph,
};

pub(crate) struct OmRuntimeHost<'a> {
    session_id: &'a str,
    host: &'a Arc<dyn RuntimeSessionHost>,
}

impl<'a> OmRuntimeHost<'a> {
    pub(crate) fn new(session_id: &'a str, host: &'a Arc<dyn RuntimeSessionHost>) -> Self {
        Self { session_id, host }
    }

    pub(crate) fn session_id(&self) -> &str {
        self.session_id
    }

    pub(crate) async fn append_plugin_nodes(
        &self,
        graph: &SessionGraph,
        nodes: Vec<(String, serde_json::Value)>,
    ) -> Result<Option<SessionGraph>, PluginError> {
        append_plugin_nodes(self.session_id, self.host, graph, nodes).await
    }

    pub(crate) async fn direct_completion(
        &self,
        request: DirectRequest,
        usage_source: &str,
    ) -> Result<DirectCompletion, PluginError> {
        self.host.direct_completion(request, usage_source).await
    }
}

pub(crate) async fn await_hidden_tasks_and_snapshot(
    session_id: &str,
    host: &Arc<dyn RuntimeSessionHost>,
) -> Result<SessionGraph, HistoryError> {
    host.await_hidden_tasks(session_id).await?;
    Ok(host.snapshot_current().await?.session_graph)
}

async fn append_plugin_nodes(
    session_id: &str,
    host: &Arc<dyn lash_core::plugin::runtime_host::RuntimeSessionHost>,
    graph: &SessionGraph,
    nodes: Vec<(String, serde_json::Value)>,
) -> Result<Option<SessionGraph>, PluginError> {
    if nodes.is_empty() {
        return Ok(Some(graph.clone()));
    }
    let request_nodes = nodes
        .iter()
        .cloned()
        .map(|(plugin_type, body)| SessionAppendNode::plugin(plugin_type, body))
        .collect::<Vec<_>>();
    match host
        .append_session_nodes(
            session_id,
            AppendSessionNodesRequest {
                nodes: request_nodes,
                requires_ancestor_node_id: graph.leaf_node_id.clone(),
            },
        )
        .await?
    {
        AppendSessionNodesResult::Appended { .. } => {
            let mut next = graph.clone();
            for (plugin_type, body) in nodes {
                next.append_plugin(plugin_type, body);
            }
            Ok(Some(next))
        }
        AppendSessionNodesResult::StaleBranch { .. } => Ok(None),
    }
}
