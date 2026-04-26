use std::collections::HashSet;
use std::sync::Arc;

use crate::session_graph::tool_call_record_projection_key;
use crate::session_model::SessionEventRecord;
use crate::store::SessionGraphCommit;
use crate::{Message, MessageSequence, SessionGraph, SessionNodeRecord, ToolCallRecord};

#[derive(Debug)]
pub(super) struct TurnGraphOverlay {
    base_graph: Arc<SessionGraph>,
    active_events: Arc<Vec<SessionEventRecord>>,
    projected_messages: MessageSequence,
    graph_tool_calls: Vec<ToolCallRecord>,
    read_tool_calls: Arc<Vec<ToolCallRecord>>,
    append_builder: crate::session_graph::SessionGraphAppendBuilder,
    appended_nodes: Vec<SessionNodeRecord>,
    materialized: Option<SessionGraph>,
}

impl TurnGraphOverlay {
    pub(super) fn new(
        base_graph: Arc<SessionGraph>,
        base_events: Arc<Vec<SessionEventRecord>>,
        base_messages: Arc<Vec<Message>>,
        base_tool_calls: Arc<Vec<ToolCallRecord>>,
    ) -> Self {
        let append_builder = base_graph.append_builder();
        let projected_messages = MessageSequence::from_base(base_messages);
        Self {
            base_graph,
            active_events: base_events,
            projected_messages,
            graph_tool_calls: base_tool_calls.as_ref().clone(),
            read_tool_calls: base_tool_calls,
            append_builder,
            appended_nodes: Vec::new(),
            materialized: None,
        }
    }

    pub(super) fn base_graph(&self) -> Arc<SessionGraph> {
        Arc::clone(&self.base_graph)
    }

    pub(super) fn message_sequence(&self) -> MessageSequence {
        self.projected_messages.clone()
    }

    pub(super) fn active_events_arc(&self) -> Arc<Vec<SessionEventRecord>> {
        Arc::clone(&self.active_events)
    }

    pub(super) fn tool_calls_arc(&self) -> Arc<Vec<ToolCallRecord>> {
        Arc::clone(&self.read_tool_calls)
    }

    pub(super) fn graph_tool_calls(&self) -> &[ToolCallRecord] {
        self.graph_tool_calls.as_slice()
    }

    pub(super) fn record_tool_calls<I>(&mut self, records: I)
    where
        I: IntoIterator<Item = ToolCallRecord>,
    {
        self.append_events(records.into_iter().map(|record| {
            SessionEventRecord::Tool(crate::session_model::ToolEvent::Invocation {
                stable_key: tool_call_record_projection_key(&record),
                record,
            })
        }));
    }

    pub(super) fn append_events<I>(&mut self, events: I)
    where
        I: IntoIterator<Item = SessionEventRecord>,
    {
        for event in events {
            if let Some(graph) = self.materialized.as_mut() {
                graph.append_event(event);
                self.refresh_from_materialized();
                continue;
            }
            let node = self.append_builder.append_event_record(event.clone());
            self.appended_nodes.extend(node);
            Arc::make_mut(&mut self.active_events).push(event.clone());
            match event {
                SessionEventRecord::Conversation(record) => {
                    self.projected_messages.push(record.to_message());
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
        let mut current = self.projected_messages.iter();
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

    pub(super) fn append_projected_conversation_messages(&mut self, messages: &[Message]) {
        let appendable_messages = messages
            .iter()
            .filter(|message| !message.is_transient())
            .cloned()
            .collect::<Vec<_>>();
        if appendable_messages.is_empty() {
            return;
        }
        if let Some(graph) = self.materialized.as_mut() {
            graph.append_projected_conversation_messages(&appendable_messages);
            self.refresh_from_materialized();
            return;
        }

        let nodes = self
            .append_builder
            .append_messages(appendable_messages.clone());
        self.appended_nodes.extend(nodes);
        self.projected_messages.extend(appendable_messages);
    }

    pub(super) fn append_projection_delta(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        if let Some(graph) = self.materialized.as_mut() {
            graph.append_projection_delta(messages, tool_calls);
            self.refresh_from_materialized();
            return;
        }

        let appendable_messages = {
            let mut seen_message_ids = self
                .projected_messages
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
            .map(tool_call_record_projection_key)
            .collect::<HashSet<_>>();
        let appendable_tool_calls = tool_calls
            .iter()
            .filter(|record| seen_tool_call_keys.insert(tool_call_record_projection_key(record)))
            .cloned()
            .collect::<Vec<_>>();

        if !appendable_messages.is_empty() {
            let nodes = self
                .append_builder
                .append_messages(appendable_messages.clone());
            self.appended_nodes.extend(nodes);
            self.projected_messages.extend(appendable_messages);
        }
        if !appendable_tool_calls.is_empty() {
            let nodes = self
                .append_builder
                .append_tool_call_records(appendable_tool_calls.clone());
            self.appended_nodes.extend(nodes);
            self.graph_tool_calls.extend(appendable_tool_calls);
        }
    }

    pub(super) fn replace_projection(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        let graph = self.materialized_graph_mut();
        graph.merge_active_projection(messages, tool_calls);
        self.read_tool_calls = Arc::new(tool_calls.to_vec());
        self.refresh_from_materialized();
    }

    pub(super) fn graph_commit(
        &self,
        persisted_graph_node_count: usize,
        graph_replace_required: bool,
    ) -> SessionGraphCommit {
        if graph_replace_required {
            return SessionGraphCommit::Replace(self.materialized_graph());
        }
        if let Some(graph) = self.materialized.as_ref() {
            let nodes_len = graph.nodes.len();
            if persisted_graph_node_count > nodes_len {
                return SessionGraphCommit::Replace(graph.clone());
            }
            return SessionGraphCommit::Append {
                nodes: graph.nodes[persisted_graph_node_count..].to_vec(),
                leaf_node_id: graph.leaf_node_id.clone(),
                graph_node_count: nodes_len,
            };
        }

        let base_len = self.base_graph.nodes.len();
        let graph_node_count = base_len + self.appended_nodes.len();
        if persisted_graph_node_count > graph_node_count {
            return SessionGraphCommit::Replace(self.materialized_graph());
        }

        let mut nodes = Vec::new();
        let appended_start = if persisted_graph_node_count < base_len {
            nodes.extend(
                self.base_graph.nodes[persisted_graph_node_count..]
                    .iter()
                    .cloned(),
            );
            0
        } else {
            persisted_graph_node_count - base_len
        };
        nodes.extend(self.appended_nodes[appended_start..].iter().cloned());
        SessionGraphCommit::Append {
            nodes,
            leaf_node_id: self.leaf_node_id(),
            graph_node_count,
        }
    }

    pub(super) fn into_session_graph(self) -> SessionGraph {
        if let Some(graph) = self.materialized {
            return graph;
        }
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
        if let Some(graph) = self.materialized.as_ref() {
            return graph.leaf_node_id.clone();
        }
        self.append_builder.leaf_node_id().cloned()
    }

    fn materialized_graph(&self) -> SessionGraph {
        if let Some(graph) = self.materialized.as_ref() {
            return graph.clone();
        }
        if self.appended_nodes.is_empty() {
            return self.base_graph.as_ref().clone();
        }
        let mut nodes = Vec::with_capacity(self.base_graph.nodes.len() + self.appended_nodes.len());
        nodes.extend(self.base_graph.nodes.iter().cloned());
        nodes.extend(self.appended_nodes.iter().cloned());
        SessionGraph::from_nodes(nodes, self.leaf_node_id())
    }

    fn materialized_graph_mut(&mut self) -> &mut SessionGraph {
        if self.materialized.is_none() {
            let graph = self.materialized_graph();
            self.materialized = Some(graph);
            self.appended_nodes.clear();
        }
        self.materialized
            .as_mut()
            .expect("turn graph materialized before mutation")
    }

    fn refresh_from_materialized(&mut self) {
        let Some(graph) = self.materialized.as_ref() else {
            return;
        };
        self.projected_messages =
            MessageSequence::from_owned(graph.project_conversation_messages());
        self.active_events = Arc::new(graph.active_events());
        self.graph_tool_calls = graph.project_tool_calls();
        self.read_tool_calls = Arc::new(self.graph_tool_calls.clone());
        self.append_builder = graph.append_builder();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_model::ConversationRecord;
    use crate::{
        MessageRole, ModeEvent, Part, PartKind, PruneState, SessionNodePayload, ToolCallRecord,
    };
    use lash_rlm_types::{RlmModeEvent, RlmTrajectoryEntry};

    fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            }],
            user_input: None,
            origin: None,
        }
    }

    fn overlay_from_graph(graph: SessionGraph) -> TurnGraphOverlay {
        let base_graph = Arc::new(graph);
        TurnGraphOverlay::new(
            Arc::clone(&base_graph),
            base_graph.shared_active_events(),
            base_graph.shared_projected_conversation_messages(),
            base_graph.shared_projected_tool_calls(),
        )
    }

    fn node_event_kind(node: &SessionNodeRecord) -> &'static str {
        match &node.payload {
            SessionNodePayload::Event { event } => match event {
                SessionEventRecord::Conversation(record) => match record.role {
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::System => "system",
                },
                SessionEventRecord::Tool(_) => "tool",
                SessionEventRecord::Mode(event) => match event.rlm_event() {
                    Some(RlmModeEvent::RlmTrajectoryEntry(_)) => "rlm_trajectory",
                    _ => "mode",
                },
                SessionEventRecord::StateSnapshot(_) => "state_snapshot",
            },
            SessionNodePayload::Plugin { .. } => "plugin",
        }
    }

    #[test]
    fn append_overlay_commits_only_unpersisted_tail() {
        let first = text_message("u1", MessageRole::User, "hello");
        let second = text_message("a1", MessageRole::Assistant, "hi");
        let mut overlay = overlay_from_graph(SessionGraph::from_projection(
            std::slice::from_ref(&first),
            &[],
        ));

        overlay.append_projected_conversation_messages(std::slice::from_ref(&second));
        let commit = overlay.graph_commit(1, false);
        let SessionGraphCommit::Append {
            nodes,
            leaf_node_id,
            graph_node_count,
        } = commit
        else {
            panic!("overlay should append graph tail");
        };
        assert_eq!(graph_node_count, 2);
        assert_eq!(leaf_node_id.as_deref(), Some("a1"));
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, "a1");

        let graph = overlay.into_session_graph();
        assert_eq!(
            graph
                .project_conversation_messages()
                .iter()
                .map(|message| message.id.as_str())
                .collect::<Vec<_>>(),
            vec!["u1", "a1"]
        );
    }

    #[test]
    fn append_overlay_respects_progress_commit_count() {
        let first = text_message("u1", MessageRole::User, "hello");
        let second = text_message("a1", MessageRole::Assistant, "hi");
        let third = text_message("u2", MessageRole::User, "again");
        let mut overlay = overlay_from_graph(SessionGraph::from_projection(
            std::slice::from_ref(&first),
            &[],
        ));

        overlay.append_projected_conversation_messages(std::slice::from_ref(&second));
        overlay.append_projected_conversation_messages(std::slice::from_ref(&third));
        let commit = overlay.graph_commit(2, false);
        let SessionGraphCommit::Append {
            nodes,
            leaf_node_id,
            graph_node_count,
        } = commit
        else {
            panic!("overlay should append graph tail");
        };
        assert_eq!(graph_node_count, 3);
        assert_eq!(leaf_node_id.as_deref(), Some("u2"));
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, "u2");
    }

    #[test]
    fn exec_tool_call_recorded_before_rlm_trajectory_and_final_answer() {
        let user = text_message("u1", MessageRole::User, "try grep");
        let assistant = text_message("a1", MessageRole::Assistant, "grep worked");
        let tool_call = ToolCallRecord {
            call_id: None,
            tool: "grep".to_string(),
            args: serde_json::json!({ "query": "submit", "path": "." }),
            result: serde_json::json!({ "matches": [] }),
            success: true,
            duration_ms: 7,
        };
        let entry = RlmTrajectoryEntry {
            id: "rlm_step_0".to_string(),
            iteration: 0,
            reasoning: "I'll inspect with grep.".to_string(),
            code: "g = (call grep { query: \"submit\", path: \".\" })?\nsubmit \"grep worked\""
                .to_string(),
            output: String::new(),
            observations: Vec::new(),
            tool_calls: vec![tool_call.clone()],
            error: None,
            final_output: Some(serde_json::json!("grep worked")),
            output_raw_len: 0,
        };
        let mut overlay = overlay_from_graph(SessionGraph::from_projection(
            std::slice::from_ref(&user),
            &[],
        ));

        // ExecCode records lashlang tool effects before the RLM driver
        // appends the trajectory and final assistant event. The later
        // finalization projection sees the same tool call via the live
        // SessionEvent stream, but must not append a duplicate after the
        // assistant answer.
        overlay.record_tool_calls([tool_call.clone()]);
        overlay.append_events([
            SessionEventRecord::Mode(ModeEvent::rlm(RlmModeEvent::RlmTrajectoryEntry(entry))),
            SessionEventRecord::Conversation(ConversationRecord::from_message(assistant.clone())),
        ]);
        overlay.append_projection_delta(std::slice::from_ref(&assistant), &[tool_call]);

        let graph = overlay.into_session_graph();
        let kinds = graph.nodes.iter().map(node_event_kind).collect::<Vec<_>>();
        assert_eq!(kinds, vec!["user", "tool", "rlm_trajectory", "assistant"]);
    }

    #[test]
    fn replace_overlay_materializes_branch_without_losing_base_graph() {
        let first = text_message("u1", MessageRole::User, "hello");
        let old = text_message("a1", MessageRole::Assistant, "old");
        let new = text_message("a2", MessageRole::Assistant, "new");
        let base = SessionGraph::from_projection(&[first.clone(), old], &[]);
        let mut overlay = overlay_from_graph(base);

        overlay.replace_projection(&[first, new], &[]);
        let commit = overlay.graph_commit(2, false);
        let SessionGraphCommit::Append {
            nodes,
            leaf_node_id,
            graph_node_count,
        } = commit
        else {
            panic!("replacement branch should append new tail");
        };
        assert_eq!(graph_node_count, 3);
        assert_eq!(leaf_node_id.as_deref(), Some("a2"));
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, "a2");
    }
}
