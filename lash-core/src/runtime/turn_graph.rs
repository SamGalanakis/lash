use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::session_graph::SessionReadModel;
use crate::session_graph::build_active_read_replacement;
use crate::session_graph::tool_call_record_active_read_key;
use crate::session_model::SessionEventRecord;
use crate::store::GraphCommitDelta;
use crate::{
    BaseRenderCache, Message, MessageSequence, SessionGraph, SessionNodeRecord, ToolCallRecord,
};

#[derive(Debug)]
pub(super) struct TurnGraphOverlay {
    base_graph: Arc<SessionGraph>,
    active_events: Arc<Vec<SessionEventRecord>>,
    active_messages: MessageSequence,
    graph_tool_calls: Vec<ToolCallRecord>,
    read_tool_calls: Arc<Vec<ToolCallRecord>>,
    append_builder: crate::session_graph::SessionGraphAppendBuilder,
    appended_nodes: Vec<SessionNodeRecord>,
    appended_node_indices: HashMap<String, usize>,
    committed_node_ids: HashSet<String>,
}

impl TurnGraphOverlay {
    pub(super) fn new(base_graph: Arc<SessionGraph>, base_read_model: SessionReadModel) -> Self {
        let append_builder = base_graph.append_builder();
        let active_messages = MessageSequence::from_base(base_read_model.messages);
        let graph_tool_calls = base_read_model.tool_calls.as_ref().clone();
        Self {
            committed_node_ids: base_graph
                .nodes
                .iter()
                .map(|node| node.node_id.clone())
                .collect(),
            base_graph,
            active_events: base_read_model.active_events,
            active_messages,
            graph_tool_calls,
            read_tool_calls: base_read_model.tool_calls,
            append_builder,
            appended_nodes: Vec::new(),
            appended_node_indices: HashMap::new(),
        }
    }

    pub(super) fn base_graph(&self) -> Arc<SessionGraph> {
        Arc::clone(&self.base_graph)
    }

    pub(super) fn message_sequence(&self) -> MessageSequence {
        self.active_messages.clone()
    }

    pub(super) fn tool_calls_arc(&self) -> Arc<Vec<ToolCallRecord>> {
        Arc::clone(&self.read_tool_calls)
    }

    pub(super) fn graph_tool_calls(&self) -> &[ToolCallRecord] {
        self.graph_tool_calls.as_slice()
    }

    #[allow(dead_code)]
    pub(super) fn read_model(&self) -> SessionReadModel {
        SessionReadModel {
            active_events: Arc::clone(&self.active_events),
            messages: self.active_messages.shared(),
            tool_calls: Arc::clone(&self.read_tool_calls),
            prompt_render_cache: Arc::new(BaseRenderCache::new()),
        }
    }

    pub(super) fn record_tool_calls<I>(&mut self, records: I)
    where
        I: IntoIterator<Item = ToolCallRecord>,
    {
        self.append_events(records.into_iter().map(|record| {
            SessionEventRecord::Tool(crate::session_model::ToolEvent::Invocation {
                stable_key: tool_call_record_active_read_key(&record),
                record,
            })
        }));
    }

    pub(super) fn append_events<I>(&mut self, events: I)
    where
        I: IntoIterator<Item = SessionEventRecord>,
    {
        for event in events {
            let node = self.append_builder.append_event_record(event.clone());
            self.append_appended_nodes(node);
            Arc::make_mut(&mut self.active_events).push(event.clone());
            match event {
                SessionEventRecord::Conversation(record) => {
                    self.active_messages.push(record.to_message());
                }
                SessionEventRecord::Tool(crate::session_model::ToolEvent::Invocation {
                    record,
                    ..
                }) => {
                    self.graph_tool_calls.push(record.clone());
                    Arc::make_mut(&mut self.read_tool_calls).push(record);
                }
                _ => {}
            }
        }
    }

    pub(super) fn message_delta_if_current_preserved<'a>(
        &self,
        next: impl IntoIterator<Item = &'a Message>,
    ) -> Option<Vec<Message>> {
        let mut current = self.active_messages.iter();
        let mut appended = Vec::new();
        for message in next.into_iter().filter(|message| !message.is_transient()) {
            if let Some(current_message) = current.next() {
                if current_message.id != message.id {
                    return None;
                }
            } else {
                appended.push(message.clone());
            }
        }
        current.next().is_none().then_some(appended)
    }

    pub(super) fn append_active_conversation_messages(&mut self, messages: &[Message]) {
        let appendable_messages = messages
            .iter()
            .filter(|message| !message.is_transient())
            .cloned()
            .collect::<Vec<_>>();
        if appendable_messages.is_empty() {
            return;
        }

        let nodes = self
            .append_builder
            .append_messages(appendable_messages.clone());
        Arc::make_mut(&mut self.active_events)
            .extend(nodes.iter().filter_map(|node| node.event().cloned()));
        self.append_appended_nodes(nodes);
        self.active_messages.extend(appendable_messages);
    }

    pub(super) fn append_active_read_delta(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        let appendable_messages = {
            let mut seen_message_ids = self
                .active_messages
                .iter()
                .map(|message| message.id.as_str())
                .collect::<HashSet<_>>();
            messages
                .iter()
                .filter(|message| {
                    !message.is_transient() && seen_message_ids.insert(message.id.as_str())
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        let mut seen_tool_call_keys = self
            .graph_tool_calls
            .iter()
            .map(tool_call_record_active_read_key)
            .collect::<HashSet<_>>();
        let appendable_tool_calls = tool_calls
            .iter()
            .filter(|record| seen_tool_call_keys.insert(tool_call_record_active_read_key(record)))
            .cloned()
            .collect::<Vec<_>>();

        if !appendable_messages.is_empty() {
            let nodes = self
                .append_builder
                .append_messages(appendable_messages.clone());
            Arc::make_mut(&mut self.active_events)
                .extend(nodes.iter().filter_map(|node| node.event().cloned()));
            self.append_appended_nodes(nodes);
            self.active_messages.extend(appendable_messages);
        }
        if !appendable_tool_calls.is_empty() {
            let nodes = self
                .append_builder
                .append_tool_call_records(appendable_tool_calls.clone());
            Arc::make_mut(&mut self.active_events)
                .extend(nodes.iter().filter_map(|node| node.event().cloned()));
            self.append_appended_nodes(nodes);
            self.graph_tool_calls.extend(appendable_tool_calls.clone());
            Arc::make_mut(&mut self.read_tool_calls).extend(appendable_tool_calls);
        }
    }

    pub(super) fn replace_active_read_state(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        let active_path = self.active_path_nodes();
        let replacement = build_active_read_replacement(
            active_path,
            self.append_builder.existing_node_ids(),
            messages,
            tool_calls,
        );
        self.append_builder.register_existing_node_ids(
            replacement
                .new_tail_nodes
                .iter()
                .map(|node| node.node_id.as_str()),
        );
        self.append_appended_nodes(replacement.new_tail_nodes);
        self.append_builder
            .set_leaf_node_id(replacement.leaf_node_id.clone());
        self.active_events = Arc::new(replacement.active_events);
        self.active_messages = MessageSequence::from_owned(replacement.active_messages);
        self.graph_tool_calls = replacement.active_tool_calls.clone();
        self.read_tool_calls = Arc::new(replacement.active_tool_calls);
    }

    pub(super) fn graph_commit(&self, graph_replace_required: bool) -> GraphCommitDelta {
        if graph_replace_required {
            return GraphCommitDelta::ReplaceFull(self.materialized_graph());
        }

        let nodes = self
            .appended_nodes
            .iter()
            .filter(|node| !self.committed_node_ids.contains(&node.node_id))
            .cloned()
            .collect::<Vec<_>>();
        if nodes.is_empty() {
            GraphCommitDelta::Unchanged {
                leaf_node_id: self.leaf_node_id(),
            }
        } else {
            GraphCommitDelta::Append {
                nodes,
                leaf_node_id: self.leaf_node_id(),
            }
        }
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(super) fn mark_graph_commit_persisted(&mut self, graph: &GraphCommitDelta) {
        match graph {
            GraphCommitDelta::Unchanged { .. } => {}
            GraphCommitDelta::Append { nodes, .. } => {
                self.mark_node_ids_persisted(nodes.iter().map(|node| node.node_id.clone()));
            }
            GraphCommitDelta::ReplaceFull(graph) => {
                self.replace_persisted_node_ids(
                    graph.nodes.iter().map(|node| node.node_id.clone()),
                );
            }
        }
    }

    pub(super) fn mark_node_ids_persisted<I>(&mut self, node_ids: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.committed_node_ids.extend(node_ids);
    }

    pub(super) fn replace_persisted_node_ids<I>(&mut self, node_ids: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.committed_node_ids = node_ids.into_iter().collect();
    }

    pub(super) fn into_session_graph(self) -> SessionGraph {
        let leaf_node_id = self.leaf_node_id();
        if self.appended_nodes.is_empty() {
            return Arc::try_unwrap(self.base_graph).unwrap_or_else(|graph| graph.as_ref().clone());
        }
        match Arc::try_unwrap(self.base_graph) {
            Ok(mut graph) => {
                graph.extend_node_records(self.appended_nodes);
                graph.set_leaf_node_id(leaf_node_id);
                graph
            }
            Err(base_graph) => {
                let mut nodes =
                    Vec::with_capacity(base_graph.nodes.len() + self.appended_nodes.len());
                nodes.extend(base_graph.nodes.iter().cloned());
                nodes.extend(self.appended_nodes);
                SessionGraph::from_nodes(nodes, leaf_node_id)
            }
        }
    }

    fn leaf_node_id(&self) -> Option<String> {
        self.append_builder.leaf_node_id().cloned()
    }

    fn materialized_graph(&self) -> SessionGraph {
        if self.appended_nodes.is_empty() {
            return self.base_graph.as_ref().clone();
        }
        let mut nodes = Vec::with_capacity(self.base_graph.nodes.len() + self.appended_nodes.len());
        nodes.extend(self.base_graph.nodes.iter().cloned());
        nodes.extend(self.appended_nodes.iter().cloned());
        SessionGraph::from_nodes(nodes, self.leaf_node_id())
    }

    fn active_path_nodes(&self) -> Vec<&SessionNodeRecord> {
        let mut path = Vec::new();
        let mut current = self.leaf_node_id();
        while let Some(node_id) = current {
            let Some(node) = self
                .appended_node_indices
                .get(node_id.as_str())
                .and_then(|idx| self.appended_nodes.get(*idx))
                .or_else(|| self.base_graph.find_node(node_id.as_str()))
            else {
                break;
            };
            path.push(node);
            current = node.parent_node_id.clone();
        }
        path.reverse();
        path
    }

    fn append_appended_nodes(&mut self, nodes: Vec<SessionNodeRecord>) {
        self.appended_node_indices.reserve(nodes.len());
        self.appended_nodes.reserve(nodes.len());
        for node in nodes {
            self.appended_node_indices
                .insert(node.node_id.clone(), self.appended_nodes.len());
            self.appended_nodes.push(node);
        }
    }
}
