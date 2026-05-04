use std::collections::HashSet;
use std::sync::Arc;

use crate::session_graph::tool_call_record_active_read_key;
use crate::session_model::SessionEventRecord;
use crate::{MessageSequence, SessionReadView, ToolCallRecord};

use super::PersistedSessionState;
use super::turn_graph::TurnGraphOverlay;

#[derive(Debug)]
pub(super) struct TurnProgress {
    graph: TurnGraphOverlay,
    state: PersistedSessionState,
    sansio_events_synced: usize,
}

impl TurnProgress {
    pub(super) fn from_state(state: PersistedSessionState) -> Self {
        let base_graph = Arc::new(state.session_graph.clone());
        let base_read_model = base_graph.read_model();
        let sansio_events_synced = base_read_model.active_events.len();
        let graph = TurnGraphOverlay::new(base_graph, base_read_model);
        Self {
            graph,
            state,
            sansio_events_synced,
        }
    }

    pub(super) fn state_mut(&mut self) -> &mut PersistedSessionState {
        &mut self.state
    }

    pub(super) fn state(&self) -> &PersistedSessionState {
        &self.state
    }

    pub(super) fn active_events(&self) -> Arc<Vec<SessionEventRecord>> {
        self.graph.read_model().active_events
    }

    pub(super) fn apply_prepared_messages(&mut self, messages: &MessageSequence) {
        self.apply_message_projection(messages);
    }

    pub(super) fn mirror_sansio_progress(
        &mut self,
        events: &Arc<Vec<SessionEventRecord>>,
    ) -> Vec<SessionEventRecord> {
        if events.len() <= self.sansio_events_synced {
            return Vec::new();
        }
        let mirrored = events[self.sansio_events_synced..].to_vec();
        self.graph.append_events(mirrored.iter().cloned());
        self.sansio_events_synced = events.len();
        mirrored
    }

    pub(super) fn record_tool_calls<I>(&mut self, records: I)
    where
        I: IntoIterator<Item = ToolCallRecord>,
    {
        self.graph.record_tool_calls(records);
    }

    pub(super) fn read_view(
        &self,
        policy: crate::SessionPolicy,
        iteration: usize,
        mode_turn_options: crate::ModeTurnOptions,
        messages: MessageSequence,
    ) -> SessionReadView {
        SessionReadView::derived_from_persisted_state(
            &self.state,
            policy,
            iteration,
            mode_turn_options,
            self.graph.base_graph(),
            messages,
            self.graph.tool_calls_arc(),
        )
    }

    pub(super) fn finalize_turn_read_state(
        &mut self,
        new_messages: MessageSequence,
        tool_calls: &[ToolCallRecord],
        cancelled: bool,
    ) {
        let projected_messages =
            (new_messages.is_empty() && cancelled).then(|| self.graph.message_sequence().shared());
        let appended_messages = if let Some(projected_messages) = projected_messages.as_ref() {
            self.graph
                .message_delta_if_current_preserved(projected_messages.iter())
        } else {
            self.graph
                .message_delta_if_current_preserved(new_messages.iter())
        };

        if let Some(appended_messages) = appended_messages {
            if tool_calls.is_empty() {
                self.graph
                    .append_active_conversation_messages(&appended_messages);
            } else {
                self.graph
                    .append_active_read_delta(&appended_messages, tool_calls);
            }
            return;
        }

        let mut next_tool_calls = self.graph.graph_tool_calls().to_vec();
        append_unique_tool_calls(&mut next_tool_calls, tool_calls);
        let projected_messages = projected_messages.unwrap_or_else(|| new_messages.shared());
        self.graph
            .replace_active_read_state(projected_messages.as_slice(), &next_tool_calls);
    }

    pub(super) fn into_final_state(mut self) -> PersistedSessionState {
        drop(std::mem::take(&mut self.state.session_graph));
        self.state.session_graph = self.graph.into_session_graph();
        self.state
    }

    pub(super) fn graph_commit(
        &self,
        graph_replace_required: bool,
    ) -> crate::store::GraphCommitDelta {
        self.graph.graph_commit(graph_replace_required)
    }

    pub(super) fn mark_graph_commit_persisted(&mut self, graph: &crate::store::GraphCommitDelta) {
        self.graph.mark_graph_commit_persisted(graph);
    }

    #[cfg(test)]
    pub(super) fn into_session_graph(self) -> crate::SessionGraph {
        self.graph.into_session_graph()
    }

    fn apply_message_projection(&mut self, messages: &MessageSequence) {
        if let Some(appended_messages) = self
            .graph
            .message_delta_if_current_preserved(messages.iter())
        {
            self.graph
                .append_active_conversation_messages(&appended_messages);
        } else {
            let read_messages = messages.shared();
            let tool_calls = self.graph.tool_calls_arc();
            self.graph
                .replace_active_read_state(read_messages.as_slice(), tool_calls.as_slice());
        }
    }
}

fn append_unique_tool_calls(out: &mut Vec<ToolCallRecord>, records: &[ToolCallRecord]) {
    let mut seen = out
        .iter()
        .map(tool_call_record_active_read_key)
        .collect::<HashSet<_>>();
    out.extend(
        records
            .iter()
            .filter(|record| seen.insert(tool_call_record_active_read_key(record)))
            .cloned(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_model::ConversationRecord;
    use crate::store::GraphCommitDelta;
    use crate::{
        Message, MessageRole, ModeEvent, Part, PartKind, PruneState, SessionGraph,
        SessionNodePayload,
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
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        }
    }

    fn tool_call(call_id: &str, tool: &str) -> ToolCallRecord {
        ToolCallRecord {
            call_id: Some(call_id.to_string()),
            tool: tool.to_string(),
            args: serde_json::json!({ "query": "submit", "path": "." }),
            result: serde_json::json!({ "matches": [] }),
            success: true,
            duration_ms: 7,
        }
    }

    fn progress_from_graph(graph: SessionGraph) -> TurnProgress {
        TurnProgress::from_state(PersistedSessionState {
            session_graph: graph,
            ..PersistedSessionState::default()
        })
    }

    fn node_event_kind(node: &crate::SessionNodeRecord) -> &'static str {
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
    fn sansio_events_after_runtime_tool_records_are_mirrored_once_in_order() {
        let user = text_message("u1", MessageRole::User, "try grep");
        let assistant = text_message("a1", MessageRole::Assistant, "grep worked");
        let record = tool_call("call-1", "grep");
        let entry = RlmTrajectoryEntry {
            id: "rlm_step_0".to_string(),
            iteration: 0,
            reasoning: "I'll inspect with grep.".to_string(),
            code: "submit \"grep worked\"".to_string(),
            output: Vec::new(),
            tool_calls: vec![record.clone()],
            images: Vec::new(),
            error: None,
            final_output: Some(serde_json::json!("grep worked")),
        };
        let mut progress = progress_from_graph(SessionGraph::from_active_read_state(
            std::slice::from_ref(&user),
            &[],
        ));
        let base_events = progress.active_events();
        let mut sansio_events = base_events.as_ref().clone();

        progress.record_tool_calls([record.clone()]);
        sansio_events.push(SessionEventRecord::Mode(ModeEvent::rlm(
            RlmModeEvent::RlmTrajectoryEntry(entry),
        )));
        sansio_events.push(SessionEventRecord::Conversation(
            ConversationRecord::from_message(assistant.clone()),
        ));

        let events = Arc::new(sansio_events);
        let mirrored = progress.mirror_sansio_progress(&events);
        assert_eq!(mirrored.len(), 2);
        assert!(progress.mirror_sansio_progress(&events).is_empty());

        progress.finalize_turn_read_state(
            MessageSequence::from_base(vec![user, assistant].into()),
            &[record],
            false,
        );
        let graph = progress.into_session_graph();
        let kinds = graph.nodes.iter().map(node_event_kind).collect::<Vec<_>>();
        assert_eq!(kinds, vec!["user", "tool", "rlm_trajectory", "assistant"]);
    }

    #[test]
    fn progress_commits_append_only_unpersisted_graph_tail() {
        let first = text_message("u1", MessageRole::User, "hello");
        let second = text_message("a1", MessageRole::Assistant, "hi");
        let third = text_message("u2", MessageRole::User, "again");
        let mut progress = progress_from_graph(SessionGraph::from_active_read_state(
            std::slice::from_ref(&first),
            &[],
        ));

        progress.apply_prepared_messages(&MessageSequence::from_base(
            vec![first.clone(), second.clone()].into(),
        ));
        let first_commit = progress.graph_commit(false);
        progress.mark_graph_commit_persisted(&first_commit);
        progress.apply_prepared_messages(&MessageSequence::from_base(
            vec![first.clone(), second, third].into(),
        ));
        let commit = progress.graph_commit(false);
        let GraphCommitDelta::Append {
            nodes,
            leaf_node_id,
        } = commit
        else {
            panic!("progress should append graph tail");
        };
        assert_eq!(leaf_node_id.as_deref(), Some("u2"));
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, "u2");
    }

    #[test]
    fn finalization_appends_when_active_read_prefix_is_preserved() {
        let user = text_message("u1", MessageRole::User, "hello");
        let assistant = text_message("a1", MessageRole::Assistant, "hi");
        let mut progress = progress_from_graph(SessionGraph::from_active_read_state(
            std::slice::from_ref(&user),
            &[],
        ));

        progress.finalize_turn_read_state(
            MessageSequence::from_base(vec![user, assistant].into()),
            &[],
            false,
        );
        let graph = progress.into_session_graph();
        assert_eq!(
            graph
                .read_model()
                .messages
                .iter()
                .map(|message| message.id.as_str())
                .collect::<Vec<_>>(),
            vec!["u1", "a1"]
        );
        assert_eq!(graph.nodes.len(), 2);
    }

    #[test]
    fn finalization_replaces_when_active_read_prefix_diverges() {
        let user = text_message("u1", MessageRole::User, "hello");
        let old = text_message("a1", MessageRole::Assistant, "old");
        let replacement = text_message("a2", MessageRole::Assistant, "new");
        let mut progress = progress_from_graph(SessionGraph::from_active_read_state(
            &[user.clone(), old],
            &[],
        ));

        progress.finalize_turn_read_state(
            MessageSequence::from_base(vec![user, replacement].into()),
            &[],
            false,
        );
        let graph = progress.into_session_graph();
        assert_eq!(
            graph
                .read_model()
                .messages
                .iter()
                .map(|message| message.id.as_str())
                .collect::<Vec<_>>(),
            vec!["u1", "a2"]
        );
        assert_eq!(graph.leaf_node_id.as_deref(), Some("a2"));
    }

    #[test]
    fn finalization_dedupes_runtime_and_assembler_tool_calls() {
        let user = text_message("u1", MessageRole::User, "hello");
        let old = text_message("a1", MessageRole::Assistant, "old");
        let assistant = text_message("a2", MessageRole::Assistant, "hi");
        let record = tool_call("call-1", "grep");
        let mut progress = progress_from_graph(SessionGraph::from_active_read_state(
            &[user.clone(), old],
            &[],
        ));

        progress.record_tool_calls([record.clone()]);
        progress.finalize_turn_read_state(
            MessageSequence::from_base(vec![user, assistant].into()),
            &[record],
            false,
        );
        let graph = progress.into_session_graph();
        assert_eq!(graph.read_model().tool_calls.len(), 1);
    }
}
