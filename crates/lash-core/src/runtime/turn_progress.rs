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
    pub(super) fn from_state(mut state: PersistedSessionState) -> Self {
        let base_graph = Arc::new(std::mem::take(&mut state.session_graph));
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
        turn_index: usize,
        mode_turn_options: crate::ModeTurnOptions,
        messages: MessageSequence,
    ) -> SessionReadView {
        SessionReadView::derived_from_persisted_state(
            &self.state,
            policy,
            turn_index,
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
        self.state.session_graph = self.graph.into_session_graph();
        self.state
    }

    pub(super) fn graph_commit(
        &self,
        graph_replace_required: bool,
    ) -> crate::store::GraphCommitDelta {
        self.graph.graph_commit(graph_replace_required)
    }

    pub(super) fn mark_node_ids_persisted<I>(&mut self, node_ids: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.graph.mark_node_ids_persisted(node_ids);
    }

    pub(super) fn replace_persisted_node_ids<I>(&mut self, node_ids: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.graph.replace_persisted_node_ids(node_ids);
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
