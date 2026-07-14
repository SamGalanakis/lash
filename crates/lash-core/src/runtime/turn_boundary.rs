use std::collections::BTreeSet;
use std::sync::Arc;

use crate::session_model::{SessionEventRecord, fresh_message_id};
use crate::store::{GraphCommitDelta, RuntimeCommit, RuntimePersistence, StoreError};
use crate::{
    AssembledTurn, Message, MessageRole, MessageSequence, Part, PartKind, PluginSession,
    PruneState, Session, SessionPolicy, SessionReadView, ToolCallRecord, TurnFinish, TurnOutcome,
    shared_parts,
};

use super::{RuntimeError, RuntimeSessionState, TurnCommitDraft, merge_ledger_entry};

pub(super) struct ProgressBoundaryCommit {
    pub(super) protocol_events: Vec<crate::ProtocolEvent>,
    pub(super) persisted: bool,
}

struct ProgressBoundarySnapshot<'a> {
    policy: SessionPolicy,
    turn_index: usize,
    messages: MessageSequence,
    event_delta: Vec<SessionEventRecord>,
    execution_state_snapshot: Option<Option<Vec<u8>>>,
    plugins: Option<&'a PluginSession>,
    store: Option<&'a (dyn RuntimePersistence + 'a)>,
}

pub(super) struct TurnBoundary {
    stage: TurnCommitStage,
    clock: Arc<dyn crate::Clock>,
    session_execution_lease: Option<crate::SessionExecutionLeaseFence>,
}

/// Explicit two-phase lifecycle for a turn commit.
///
/// A pipeline starts in [`TurnCommitStage::Drafting`] while progress-boundary
/// commits accumulate against a mutable [`TurnCommitDraft`]. The first call
/// that needs the assembled session state transitions it (irreversibly) to
/// [`TurnCommitStage::Finalized`], snapshotting the progress graph commit so
/// later final commits can reconcile it against the materialized graph.
enum TurnCommitStage {
    Drafting(Box<TurnCommitDraft>),
    Finalized(Box<FinalizedTurnCommitStage>),
}

struct FinalizedTurnCommitStage {
    state: RuntimeSessionState,
    progress_graph_commit: GraphCommitDelta,
}

impl TurnCommitStage {
    /// Cheap throwaway value used only to move out of `&mut self` during the
    /// `Drafting` → `Finalized` transition.
    fn placeholder() -> Self {
        Self::Finalized(Box::new(FinalizedTurnCommitStage {
            state: RuntimeSessionState::default(),
            progress_graph_commit: GraphCommitDelta::Unchanged { leaf_node_id: None },
        }))
    }
}

struct FinalCommitInput<'a> {
    returned_state: &'a crate::SessionSnapshot,
    tool_calls: &'a [ToolCallRecord],
    plugins: Option<&'a PluginSession>,
    execution_state_snapshot: Option<Option<Vec<u8>>>,
    store: Option<&'a (dyn RuntimePersistence + 'a)>,
    usage_deltas: &'a [crate::TokenLedgerEntry],
    outcome: &'a TurnOutcome,
    turn_id: Option<&'a str>,
    completed_queue_claims: Vec<crate::QueuedWorkCompletion>,
    completed_turn_input_claims: Vec<crate::TurnInputCompletion>,
    interrupted_turn_input_turn_id: Option<String>,
    pending_attachment_ids: Vec<crate::AttachmentId>,
    session_execution_lease_completion: Option<crate::SessionExecutionLeaseCompletion>,
}

enum PersistedGraphMark {
    Unchanged,
    Append(Vec<String>),
    ReplaceFull(Vec<String>),
}

impl PersistedGraphMark {
    fn from_graph_commit(graph: &GraphCommitDelta) -> Self {
        match graph {
            GraphCommitDelta::Unchanged { .. } => Self::Unchanged,
            GraphCommitDelta::Append { nodes, .. } => {
                Self::Append(nodes.iter().map(|node| node.node_id.clone()).collect())
            }
            GraphCommitDelta::ReplaceFull(graph) => Self::ReplaceFull(
                graph
                    .nodes
                    .iter()
                    .map(|node| node.node_id.clone())
                    .collect(),
            ),
        }
    }
}

impl TurnBoundary {
    #[cfg(test)]
    pub(super) fn from_state(state: RuntimeSessionState) -> Self {
        Self::from_state_with_clock(state, Arc::new(crate::SystemClock))
    }

    pub(super) fn from_state_with_clock(
        state: RuntimeSessionState,
        clock: Arc<dyn crate::Clock>,
    ) -> Self {
        let draft_clock = Arc::clone(&clock);
        Self {
            stage: TurnCommitStage::Drafting(Box::new(TurnCommitDraft::from_state_with_clock(
                state,
                draft_clock,
            ))),
            clock,
            session_execution_lease: None,
        }
    }

    pub(super) fn with_session_execution_lease(
        mut self,
        lease: Option<crate::SessionExecutionLeaseFence>,
    ) -> Self {
        self.session_execution_lease = lease;
        self
    }

    pub(super) fn state_mut(&mut self) -> &mut RuntimeSessionState {
        match &mut self.stage {
            TurnCommitStage::Drafting(draft) => draft.state_mut(),
            TurnCommitStage::Finalized(finalized) => &mut finalized.state,
        }
    }

    pub(super) fn state(&self) -> &RuntimeSessionState {
        match &self.stage {
            TurnCommitStage::Drafting(draft) => draft.state(),
            TurnCommitStage::Finalized(finalized) => &finalized.state,
        }
    }

    pub(super) fn apply_prepared_messages(&mut self, messages: &MessageSequence) {
        self.draft_mut().apply_prepared_messages(messages);
    }

    pub(super) fn read_view(
        &self,
        policy: crate::SessionPolicy,
        turn_index: usize,
        protocol_turn_options: crate::ProtocolTurnOptions,
        messages: MessageSequence,
    ) -> SessionReadView {
        self.draft_ref()
            .read_view(policy, turn_index, protocol_turn_options, messages)
    }

    pub(super) fn active_events(&self) -> Arc<Vec<SessionEventRecord>> {
        self.draft_ref().active_events()
    }

    pub(super) fn finalize_turn_read_state(
        &mut self,
        new_messages: MessageSequence,
        cancelled: bool,
    ) {
        self.draft_mut()
            .finalize_turn_read_state(new_messages, cancelled);
    }

    pub(super) async fn prepared_checkpoint(
        &mut self,
        store: Option<&(dyn RuntimePersistence + '_)>,
        policy: SessionPolicy,
        turn_index: usize,
        messages: &MessageSequence,
        session: Option<&mut Session>,
    ) -> Result<(), StoreError> {
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return Ok(());
        }

        self.apply_prepared_messages(messages);
        let Some(store) = store else {
            return Ok(());
        };

        let plugins = session
            .as_deref()
            .map(|session| Arc::clone(session.plugins()));
        let execution_state_snapshot = match session {
            Some(session) => Self::snapshot_dirty_execution_state(session).await,
            None => None,
        };
        let state = self.draft_mut().state_mut();
        state.policy = policy;
        state.turn_index = turn_index;
        if let Some(execution_state_snapshot) = execution_state_snapshot {
            state.set_execution_state_snapshot(execution_state_snapshot);
        }
        if let Some(plugins) = plugins.as_ref() {
            state.refresh_plugin_snapshots(plugins.as_ref());
        }
        self.commit_progress_graph(store, &[]).await
    }

    pub(super) async fn progress_boundary(
        &mut self,
        session: &mut Session,
        policy: SessionPolicy,
        turn_index: usize,
        messages: MessageSequence,
        event_delta: Vec<SessionEventRecord>,
    ) -> Result<ProgressBoundaryCommit, RuntimeError> {
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return Ok(ProgressBoundaryCommit {
                protocol_events: Vec::new(),
                persisted: false,
            });
        }

        let store = session.history_store();
        let execution_state_snapshot = Self::snapshot_dirty_execution_state(session).await;
        let plugins = Arc::clone(session.plugins());
        self.progress_boundary_with_snapshot(ProgressBoundarySnapshot {
            policy,
            turn_index,
            messages,
            event_delta,
            execution_state_snapshot,
            plugins: Some(plugins.as_ref()),
            store: store.as_ref().map(|store| store.as_ref()),
        })
        .await
    }

    pub(super) async fn progress_boundary_in_memory(
        &mut self,
        session: &mut Session,
        policy: SessionPolicy,
        turn_index: usize,
        messages: MessageSequence,
        event_delta: Vec<SessionEventRecord>,
    ) -> Result<ProgressBoundaryCommit, RuntimeError> {
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return Ok(ProgressBoundaryCommit {
                protocol_events: Vec::new(),
                persisted: false,
            });
        }

        let execution_state_snapshot = Self::snapshot_dirty_execution_state(session).await;
        let plugins = Arc::clone(session.plugins());
        self.progress_boundary_with_snapshot(ProgressBoundarySnapshot {
            policy,
            turn_index,
            messages,
            event_delta,
            execution_state_snapshot,
            plugins: Some(plugins.as_ref()),
            store: None,
        })
        .await
    }

    async fn progress_boundary_with_snapshot(
        &mut self,
        snapshot: ProgressBoundarySnapshot<'_>,
    ) -> Result<ProgressBoundaryCommit, RuntimeError> {
        let ProgressBoundarySnapshot {
            policy,
            turn_index,
            messages,
            event_delta,
            execution_state_snapshot,
            plugins,
            store,
        } = snapshot;
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return Ok(ProgressBoundaryCommit {
                protocol_events: Vec::new(),
                persisted: false,
            });
        }

        let protocol_events = self.apply_event_delta(event_delta);
        {
            let draft = self.draft_mut();
            draft.apply_prepared_messages(&messages);
            let state = draft.state_mut();
            state.policy = policy;
            state.turn_index = turn_index;
            if let Some(execution_state_snapshot) = execution_state_snapshot {
                state.set_execution_state_snapshot(execution_state_snapshot);
            }
            if let Some(plugins) = plugins {
                state.refresh_plugin_snapshots(plugins);
            }
        }

        let Some(store) = store else {
            return Ok(ProgressBoundaryCommit {
                protocol_events,
                persisted: false,
            });
        };
        match self.commit_progress_graph(store, &[]).await {
            Ok(()) => Ok(ProgressBoundaryCommit {
                protocol_events,
                persisted: true,
            }),
            Err(err @ StoreError::SessionExecutionLeaseExpired { .. }) => {
                Err(super::runtime_error_from_store_commit(err))
            }
            Err(err) => {
                tracing::warn!("failed to persist runtime progress boundary: {err}");
                Ok(ProgressBoundaryCommit {
                    protocol_events,
                    persisted: false,
                })
            }
        }
    }

    pub(super) fn export_state_for_assembly(&mut self) -> crate::SessionSnapshot {
        self.final_state_mut().to_snapshot()
    }

    pub(super) fn apply_event_delta(
        &mut self,
        event_delta: Vec<SessionEventRecord>,
    ) -> Vec<crate::ProtocolEvent> {
        let protocol_events = event_delta
            .iter()
            .filter_map(|event| match event {
                SessionEventRecord::Protocol(event) => Some(event.clone()),
                SessionEventRecord::Conversation(_) => None,
            })
            .collect::<Vec<_>>();
        self.draft_mut().append_events(event_delta);
        protocol_events
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn final_commit(
        &mut self,
        returned_turn: &mut AssembledTurn,
        session: Option<&mut Session>,
        usage_deltas: &[crate::TokenLedgerEntry],
        turn_id: Option<&str>,
        completed_queue_claims: Vec<crate::QueuedWorkCompletion>,
        completed_turn_input_claims: Vec<crate::TurnInputCompletion>,
        interrupted_turn_input_turn_id: Option<String>,
        pending_attachment_ids: Vec<crate::AttachmentId>,
        session_execution_lease_completion: Option<crate::SessionExecutionLeaseCompletion>,
    ) -> Result<(), RuntimeError> {
        let (store, plugins, execution_state_snapshot) = match session {
            Some(session) => {
                let store = session.history_store();
                let execution_state_snapshot = Self::snapshot_dirty_execution_state(session).await;
                let plugins = Arc::clone(session.plugins());
                (store, Some(plugins), execution_state_snapshot)
            }
            None => (None, None, None),
        };
        self.final_commit_with_snapshots(FinalCommitInput {
            returned_state: &returned_turn.state,
            tool_calls: &returned_turn.tool_calls,
            plugins: plugins.as_deref(),
            execution_state_snapshot,
            store: store.as_ref().map(|store| store.as_ref()),
            usage_deltas,
            outcome: &returned_turn.outcome,
            turn_id,
            completed_queue_claims,
            completed_turn_input_claims,
            interrupted_turn_input_turn_id,
            pending_attachment_ids,
            session_execution_lease_completion,
        })
        .await
        .map_err(super::runtime_error_from_store_commit)?;
        returned_turn.state = self.final_state_mut().to_snapshot();
        Ok(())
    }

    pub(super) fn into_final_state(self) -> RuntimeSessionState {
        match self.stage {
            TurnCommitStage::Drafting(draft) => (*draft).into_final_state(),
            TurnCommitStage::Finalized(finalized) => finalized.state,
        }
    }

    fn draft_ref(&self) -> &TurnCommitDraft {
        match &self.stage {
            TurnCommitStage::Drafting(draft) => draft.as_ref(),
            TurnCommitStage::Finalized(_) => {
                panic!("turn commit draft is unavailable after final state materialization")
            }
        }
    }

    fn draft_mut(&mut self) -> &mut TurnCommitDraft {
        match &mut self.stage {
            TurnCommitStage::Drafting(draft) => draft.as_mut(),
            TurnCommitStage::Finalized(_) => {
                panic!("turn commit draft is unavailable after final state materialization")
            }
        }
    }

    fn final_state_mut(&mut self) -> &mut RuntimeSessionState {
        self.stage = match std::mem::replace(&mut self.stage, TurnCommitStage::placeholder()) {
            TurnCommitStage::Drafting(draft) => {
                let progress_graph_commit =
                    draft.graph_commit(draft.state().graph_replace_required);
                TurnCommitStage::Finalized(Box::new(FinalizedTurnCommitStage {
                    state: (*draft).into_final_state(),
                    progress_graph_commit,
                }))
            }
            finalized => finalized,
        };
        match &mut self.stage {
            TurnCommitStage::Finalized(finalized) => &mut finalized.state,
            TurnCommitStage::Drafting(_) => unreachable!("stage was just finalized"),
        }
    }

    async fn commit_progress_graph(
        &mut self,
        store: &(dyn RuntimePersistence + '_),
        usage_deltas: &[crate::TokenLedgerEntry],
    ) -> Result<(), StoreError> {
        let draft = self.draft_mut();
        let state = draft.state();
        let graph = draft.graph_commit(state.graph_replace_required);
        self.apply_commit(
            store,
            graph,
            usage_deltas,
            None,
            Vec::new(),
            Vec::new(),
            None,
            Vec::new(),
            None,
        )
        .await
    }

    async fn final_commit_with_snapshots(
        &mut self,
        input: FinalCommitInput<'_>,
    ) -> Result<(), StoreError> {
        let FinalCommitInput {
            returned_state,
            tool_calls,
            plugins,
            execution_state_snapshot,
            store,
            usage_deltas,
            outcome,
            turn_id,
            completed_queue_claims,
            completed_turn_input_claims,
            interrupted_turn_input_turn_id,
            pending_attachment_ids,
            session_execution_lease_completion,
        } = input;
        let clock = Arc::clone(&self.clock);
        let state = self.final_state_mut();
        state.apply_snapshot(returned_state);
        for entry in usage_deltas.iter().cloned() {
            merge_ledger_entry(&mut state.token_ledger, entry);
        }
        if let Some(plugins) = plugins {
            state.refresh_plugin_snapshots(plugins);
        }
        if let Some(execution_state_snapshot) = execution_state_snapshot {
            state.set_execution_state_snapshot(execution_state_snapshot);
        }
        materialize_terminal_output(state, outcome, clock.as_ref());
        materialize_agent_frame_switch(state, outcome, clock.as_ref());
        let progress_graph = match &self.stage {
            TurnCommitStage::Drafting(draft) => {
                Some(draft.graph_commit(draft.state().graph_replace_required))
            }
            TurnCommitStage::Finalized(finalized) => Some(finalized.progress_graph_commit.clone()),
        };
        let state = self.final_state_mut();

        if let Some(store) = store {
            let graph = if state.graph_replace_required {
                GraphCommitDelta::ReplaceFull(state.session_graph.clone())
            } else if state.head_revision.is_none() {
                match progress_graph {
                    Some(GraphCommitDelta::Append {
                        nodes,
                        leaf_node_id,
                    }) if state.session_graph.nodes.is_empty() => GraphCommitDelta::ReplaceFull(
                        crate::SessionGraph::from_nodes(nodes, leaf_node_id),
                    ),
                    _ => GraphCommitDelta::ReplaceFull(state.session_graph.clone()),
                }
            } else {
                match progress_graph {
                    Some(GraphCommitDelta::Unchanged { .. })
                        if !state.session_graph.nodes.is_empty() =>
                    {
                        GraphCommitDelta::ReplaceFull(state.session_graph.clone())
                    }
                    Some(graph) => graph,
                    None => GraphCommitDelta::Unchanged {
                        leaf_node_id: state.session_graph.leaf_node_id.clone(),
                    },
                }
            };
            let committed_attachment_ids =
                committed_attachment_ids(state, tool_calls, pending_attachment_ids);
            self.apply_commit(
                store,
                graph,
                usage_deltas,
                turn_id,
                completed_queue_claims,
                completed_turn_input_claims,
                interrupted_turn_input_turn_id,
                committed_attachment_ids,
                session_execution_lease_completion,
            )
            .await
        } else {
            state.discard_runtime_snapshots();
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn apply_commit(
        &mut self,
        store: &(dyn RuntimePersistence + '_),
        graph: GraphCommitDelta,
        usage_deltas: &[crate::TokenLedgerEntry],
        turn_id: Option<&str>,
        completed_queue_claims: Vec<crate::QueuedWorkCompletion>,
        completed_turn_input_claims: Vec<crate::TurnInputCompletion>,
        interrupted_turn_input_turn_id: Option<String>,
        committed_attachment_ids: Vec<crate::AttachmentId>,
        session_execution_lease_completion: Option<crate::SessionExecutionLeaseCompletion>,
    ) -> Result<(), StoreError> {
        let session_execution_lease = self.session_execution_lease.clone();
        let state = self.state_mut();
        let mark = PersistedGraphMark::from_graph_commit(&graph);
        let mut commit =
            RuntimeCommit::persisted_state_with_graph_commit(state, graph, usage_deltas)
                .with_committed_attachments(committed_attachment_ids);
        if let Some(lease) = session_execution_lease {
            commit = commit.with_session_execution_lease(lease);
        }
        if let Some(completion) = session_execution_lease_completion {
            commit = commit.releasing_session_execution_lease(completion);
        }
        commit.completed_queue_claims = completed_queue_claims;
        commit.completed_turn_input_claims = completed_turn_input_claims;
        commit.interrupted_turn_input_turn_id = interrupted_turn_input_turn_id;
        if let Some(turn_id) = turn_id {
            let turn_commit_hash = commit.turn_commit_hash()?;
            commit.turn_commit = Some(crate::RuntimeTurnCommitStamp::new(
                commit.session_id.clone(),
                turn_id,
                turn_commit_hash,
            ));
        }
        let result = store.commit_runtime_state(commit).await?;
        state.apply_persisted_commit_result(result);
        if let TurnCommitStage::Drafting(draft) = &mut self.stage {
            match mark {
                PersistedGraphMark::Unchanged => {}
                PersistedGraphMark::Append(node_ids) => {
                    draft.mark_node_ids_persisted(node_ids);
                }
                PersistedGraphMark::ReplaceFull(node_ids) => {
                    draft.replace_persisted_node_ids(node_ids);
                }
            }
        }
        Ok(())
    }

    async fn snapshot_dirty_execution_state(session: &mut Session) -> Option<Option<Vec<u8>>> {
        let code_executor = session.plugins().code_executor()?;
        if !code_executor.execution_state_dirty() {
            return None;
        }
        let session_id = session.session_id().to_string();
        match code_executor
            .snapshot_execution_state(crate::plugin::ProtocolSessionContext::new(
                session,
                &session_id,
            ))
            .await
        {
            Ok(snapshot) => Some(snapshot),
            Err(err) => {
                tracing::warn!("failed to snapshot dirty execution state: {err}");
                None
            }
        }
    }
}

fn committed_attachment_ids(
    state: &RuntimeSessionState,
    tool_calls: &[ToolCallRecord],
    pending_attachment_ids: Vec<crate::AttachmentId>,
) -> Vec<crate::AttachmentId> {
    let mut attachment_ids = pending_attachment_ids.into_iter().collect::<BTreeSet<_>>();
    for call in tool_calls {
        for attachment in call.output.attachments() {
            attachment_ids.insert(attachment.id);
        }
    }
    for message in state.read_model().messages.iter() {
        for part in message.parts.iter() {
            if let Some(attachment) = &part.attachment {
                attachment_ids.insert(attachment.reference.id.clone());
            }
        }
    }
    attachment_ids.into_iter().collect()
}

fn materialize_terminal_output(
    state: &mut RuntimeSessionState,
    outcome: &TurnOutcome,
    clock: &dyn crate::Clock,
) {
    let TurnOutcome::Finished(TurnFinish::AssistantMessage { text }) = outcome else {
        return;
    };
    if state
        .read_model()
        .messages
        .iter()
        .rfind(|message| !message.is_transient())
        .is_some_and(|message| {
            message.role == MessageRole::Assistant && message_rendered_text(message) == *text
        })
    {
        return;
    }

    let id = fresh_message_id();
    state.append_active_conversation_messages_with_clock(
        &[Message {
            id: id.clone(),
            role: MessageRole::Assistant,
            parts: shared_parts(vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Prose,
                content: text.clone(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]),
            origin: None,
        }],
        clock,
    );
    state.graph_replace_required = true;
}

fn materialize_agent_frame_switch(
    state: &mut RuntimeSessionState,
    outcome: &TurnOutcome,
    clock: &dyn crate::Clock,
) {
    let TurnOutcome::AgentFrameSwitch {
        frame_id,
        initial_nodes,
        ..
    } = outcome
    else {
        return;
    };
    if frame_id.trim().is_empty() || state.current_agent_frame_id == *frame_id {
        return;
    }
    let nodes = initial_nodes
        .iter()
        .map(|value| {
            serde_json::from_value::<crate::SessionAppendNode>(value.clone())
                .expect("agent frame seed nodes are validated by the protocol producer")
        })
        .collect::<Vec<_>>();
    super::open_agent_frame_in_state_with_clock(
        state,
        crate::OpenAgentFrameRequest::new(frame_id.clone(), crate::AgentFrameReason::continue_as())
            .with_initial_nodes(nodes),
        clock,
    );
}

fn message_rendered_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .filter(|part| {
            matches!(
                part.kind,
                PartKind::Prose | PartKind::Text | PartKind::Image | PartKind::ToolResult
            )
        })
        .map(|part| part.content.as_str())
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tests::helpers::RecordingStore;
    use crate::session_model::{ConversationRecord, MessageRole, Part, PartKind, PruneState};
    use crate::store::SessionExecutionLeaseStore;
    use crate::{Message, SessionGraph, TokenUsage, shared_parts};

    fn lease_owner(owner_id: &str) -> crate::LeaseOwnerIdentity {
        crate::LeaseOwnerIdentity::opaque(owner_id, format!("{owner_id}:incarnation"))
    }

    fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: shared_parts(vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]),
            origin: None,
        }
    }

    fn usage_entry(source: &str, model: &str, input_tokens: i64) -> crate::TokenLedgerEntry {
        crate::TokenLedgerEntry {
            source: source.to_string(),
            model: model.to_string(),
            usage: TokenUsage {
                input_tokens,
                output_tokens: 2,
                cache_read_input_tokens: 1,
                cache_write_input_tokens: 0,
                reasoning_output_tokens: 0,
            },
        }
    }

    fn image_ref(id: &str) -> crate::AttachmentRef {
        crate::AttachmentMeta::new(
            crate::AttachmentId::new(id),
            crate::MediaType::Image(crate::ImageMediaType::Png),
            3,
            Some(1),
            Some(1),
            Some("tiny".to_string()),
        )
        .as_ref()
    }

    fn test_protocol_event(kind: &str) -> crate::ProtocolEvent {
        crate::ProtocolEvent::typed(
            "test_protocol",
            serde_json::json!({
                "kind": kind,
                "payload": { "test": true },
            }),
        )
        .expect("test protocol event serializes")
    }

    fn summarize_protocol_event(event: &crate::ProtocolEvent) -> String {
        let Some(value) = event
            .decode::<serde_json::Value>("test_protocol")
            .expect("test protocol event decodes")
        else {
            return format!("protocol:{}", event.plugin_id);
        };
        let kind = value
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        format!("protocol:{kind}")
    }

    fn persisted_event_order(graph: &SessionGraph) -> Vec<String> {
        graph
            .nodes
            .iter()
            .filter_map(|node| match node.event()? {
                crate::SessionEventRecord::Conversation(record) => {
                    Some(format!("message:{}", record.id))
                }
                crate::SessionEventRecord::Protocol(event) => Some(summarize_protocol_event(event)),
            })
            .collect()
    }

    fn chronological_event_order(graph: &SessionGraph) -> Vec<String> {
        let read_model = graph.read_model();
        crate::chronological::ChronologicalProjection::from_read_model(&read_model)
            .entries()
            .iter()
            .map(|entry| match &entry.payload {
                crate::chronological::ChronologicalPayload::Message(message) => {
                    format!("message:{}", message.id)
                }
                crate::chronological::ChronologicalPayload::ProtocolEvent(event) => {
                    summarize_protocol_event(event)
                }
            })
            .collect()
    }

    fn stored_graph_with_head_leaf(store: &RecordingStore) -> SessionGraph {
        let mut graph = store.session_graph.lock().expect("lock graph").clone();
        graph.set_leaf_node_id(
            store
                .session_head_meta
                .lock()
                .expect("lock head meta")
                .as_ref()
                .and_then(|meta| meta.leaf_node_id.clone()),
        );
        graph
    }

    fn state_with_graph(graph: SessionGraph) -> RuntimeSessionState {
        RuntimeSessionState {
            session_id: "session-1".to_string(),
            session_graph: graph,
            ..RuntimeSessionState::default()
        }
    }

    async fn leased_boundary(
        store: &RecordingStore,
        state: RuntimeSessionState,
    ) -> (TurnBoundary, crate::SessionExecutionLease) {
        let owner = lease_owner("turn-boundary-test");
        let lease = store
            .try_claim_session_execution_lease(&state.session_id, &owner, 60_000)
            .await
            .expect("claim test session execution lease")
            .acquired()
            .expect("test session execution lease");
        (
            TurnBoundary::from_state(state).with_session_execution_lease(Some(lease.fence())),
            lease,
        )
    }

    #[test]
    fn agent_frame_switch_materializes_outcome_seed_without_tool_call_event() {
        let graph = SessionGraph::from_active_read_state(&[text_message(
            "u0",
            MessageRole::User,
            "old frame",
        )]);
        let mut state = state_with_graph(graph);
        state.ensure_agent_frame_initialized();
        let previous_frame_id = state.current_agent_frame_id.clone();
        let frame_id = "frame-2".to_string();
        let seed_node = crate::SessionAppendNode::message(crate::PluginMessage::text(
            MessageRole::User,
            "seed message",
        ));
        materialize_agent_frame_switch(
            &mut state,
            &TurnOutcome::AgentFrameSwitch {
                frame_id: frame_id.clone(),
                task: "next task".to_string(),
                initial_nodes: vec![serde_json::to_value(seed_node).expect("seed node json")],
            },
            &crate::SystemClock,
        );

        assert_eq!(state.session_id, "session-1");
        assert_eq!(state.current_agent_frame_id, frame_id);
        let current = state.current_agent_frame().expect("current frame");
        assert_eq!(
            current.previous_frame_id.as_deref(),
            Some(previous_frame_id.as_str())
        );
        assert_eq!(
            current.reason.as_str(),
            crate::AgentFrameReason::CONTINUE_AS
        );
        let current_read = state
            .session_graph
            .read_model_for_agent_frame(&frame_id, false);
        assert_eq!(current_read.messages.len(), 1);
        assert_eq!(current_read.messages[0].parts[0].content, "seed message");
        let previous_read = state
            .session_graph
            .read_model_for_agent_frame(&previous_frame_id, true);
        assert_eq!(previous_read.messages.len(), 1);
        assert_eq!(previous_read.messages[0].parts[0].content, "old frame");
    }

    #[test]
    fn open_agent_frame_seeds_compaction_frame_and_is_replay_idempotent() {
        let graph = SessionGraph::from_active_read_state(&[text_message(
            "u0",
            MessageRole::User,
            "old durable frame",
        )]);
        let mut state = state_with_graph(graph);
        state.ensure_agent_frame_initialized();
        let previous_frame_id = state.current_agent_frame_id.clone();
        let previous = state
            .current_agent_frame_mut()
            .expect("current frame before compaction");
        previous.assignment.usage_source = Some("root-assignment".to_string());
        previous.protocol_turn_options =
            crate::ProtocolTurnOptions::from_payload(serde_json::json!({ "mode": "test" }));

        let frame_id = "frame-compaction".to_string();
        let seed_node = crate::SessionAppendNode::message(
            crate::PluginMessage::text(MessageRole::Assistant, "Compaction summary:\nold work")
                .with_origin(crate::MessageOrigin::Plugin {
                    plugin_id: "rolling_history".to_string(),
                    transient: false,
                }),
        );

        let opened = super::super::open_agent_frame_in_state_with_clock(
            &mut state,
            crate::OpenAgentFrameRequest::new(
                frame_id.clone(),
                crate::AgentFrameReason::compaction(),
            )
            .with_initial_nodes(vec![seed_node.clone()]),
            &crate::SystemClock,
        );

        assert!(opened.opened);
        assert_eq!(state.current_agent_frame_id, frame_id);
        let current = state.current_agent_frame().expect("current frame");
        assert_eq!(current.reason.as_str(), crate::AgentFrameReason::COMPACTION);
        assert_eq!(
            current.previous_frame_id.as_deref(),
            Some(previous_frame_id.as_str())
        );
        assert_eq!(
            current.assignment.usage_source.as_deref(),
            Some("root-assignment")
        );
        assert_eq!(
            current.protocol_turn_options.payload,
            serde_json::json!({ "mode": "test" })
        );

        let current_read = state
            .session_graph
            .read_model_for_agent_frame(&frame_id, false);
        assert_eq!(current_read.messages.len(), 1);
        assert_eq!(
            current_read.messages[0].parts[0].content,
            "Compaction summary:\nold work"
        );
        assert!(matches!(
            current_read.messages[0].origin.as_ref(),
            Some(crate::MessageOrigin::Plugin { plugin_id, .. }) if plugin_id == "rolling_history"
        ));

        let previous_read = state
            .session_graph
            .read_model_for_agent_frame(&previous_frame_id, true);
        assert_eq!(previous_read.messages.len(), 1);
        assert_eq!(
            previous_read.messages[0].parts[0].content,
            "old durable frame"
        );

        let replay = super::super::open_agent_frame_in_state_with_clock(
            &mut state,
            crate::OpenAgentFrameRequest::new(
                frame_id.clone(),
                crate::AgentFrameReason::compaction(),
            )
            .with_initial_nodes(vec![seed_node]),
            &crate::SystemClock,
        );
        assert!(!replay.opened);
        let replay_read = state
            .session_graph
            .read_model_for_agent_frame(&frame_id, false);
        assert_eq!(replay_read.messages.len(), 1);
    }

    #[tokio::test]
    async fn prepared_checkpoint_writes_only_explicit_progress_graph_tail() {
        let mut graph =
            SessionGraph::from_active_read_state(&[text_message("u0", MessageRole::User, "old")]);
        let base_graph = graph.clone();
        graph.append_message(text_message("a0", MessageRole::Assistant, "new"));
        let state = state_with_graph(base_graph.clone());
        let store = RecordingStore::default();
        store
            .session_graph
            .lock()
            .expect("lock graph")
            .extend_node_records(base_graph.nodes.iter().cloned());
        let (mut pipeline, _lease) = leased_boundary(&store, state).await;

        pipeline
            .prepared_checkpoint(
                Some(&store),
                SessionPolicy::default(),
                0,
                &MessageSequence::from_base(
                    vec![
                        text_message("u0", MessageRole::User, "old"),
                        text_message("a0", MessageRole::Assistant, "new"),
                    ]
                    .into(),
                ),
                None,
            )
            .await
            .expect("commit");

        let mut stored_graph = store.session_graph.lock().expect("lock graph").clone();
        stored_graph.set_leaf_node_id(
            store
                .session_head_meta
                .lock()
                .expect("lock head meta")
                .as_ref()
                .and_then(|meta| meta.leaf_node_id.clone()),
        );
        assert_eq!(stored_graph.nodes.len(), graph.nodes.len());
        assert_eq!(stored_graph.nodes[1].node_id, graph.nodes[1].node_id);
        assert!(pipeline.state_mut().head_revision.is_some());
    }

    #[tokio::test]
    async fn prepared_checkpoint_propagates_store_errors() {
        let graph =
            SessionGraph::from_active_read_state(&[text_message("u0", MessageRole::User, "hello")]);
        let state = state_with_graph(graph);
        let store = RecordingStore::default();
        store
            .save_session_head_meta(crate::SessionHeadMeta {
                session_id: "other-session".to_string(),
                ..crate::SessionHeadMeta::default()
            })
            .await;
        let (mut pipeline, _lease) = leased_boundary(&store, state).await;

        let err = pipeline
            .prepared_checkpoint(
                Some(&store),
                SessionPolicy::default(),
                0,
                &MessageSequence::from_base(
                    vec![text_message("u0", MessageRole::User, "hello")].into(),
                ),
                None,
            )
            .await
            .expect_err("binding mismatch");

        assert!(matches!(
            err,
            StoreError::SessionBindingMismatch {
                bound_session_id,
                attempted_session_id
            } if bound_session_id == "other-session" && attempted_session_id == "session-1"
        ));
    }

    #[tokio::test]
    async fn progress_boundary_uses_typed_messages_without_duplicate_conversation_nodes() {
        let user = text_message("u0", MessageRole::User, "hello");
        let assistant = text_message("a0", MessageRole::Assistant, "hi");
        let graph = SessionGraph::from_active_read_state(std::slice::from_ref(&user));
        let base_graph = graph.clone();
        let event_delta = vec![
            crate::SessionEventRecord::Conversation(ConversationRecord::from_message(user.clone())),
            crate::SessionEventRecord::Conversation(ConversationRecord::from_message(
                assistant.clone(),
            )),
        ];
        let store = RecordingStore::default();
        store
            .session_graph
            .lock()
            .expect("lock graph")
            .extend_node_records(base_graph.nodes.iter().cloned());
        let (mut pipeline, _lease) = leased_boundary(&store, state_with_graph(graph)).await;

        let boundary = pipeline
            .progress_boundary_with_snapshot(ProgressBoundarySnapshot {
                policy: SessionPolicy::default(),
                turn_index: 1,
                messages: MessageSequence::from_base(vec![user.clone(), assistant.clone()].into()),
                event_delta,
                execution_state_snapshot: None,
                plugins: None,
                store: Some(&store),
            })
            .await
            .expect("progress boundary");

        assert!(boundary.persisted);
        assert!(boundary.protocol_events.is_empty());

        let stored_graph = store.session_graph.lock().expect("lock graph").clone();
        let conversation_nodes = stored_graph
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    node.event(),
                    Some(crate::SessionEventRecord::Conversation(_))
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(conversation_nodes.len(), 2);
        assert_eq!(conversation_nodes[0].node_id, "u0");
        assert_eq!(conversation_nodes[1].node_id, "a0");
        assert!(
            stored_graph
                .nodes
                .iter()
                .filter(|node| matches!(
                    node.event(),
                    Some(crate::SessionEventRecord::Conversation(_))
                ))
                .all(|node| !node.node_id.starts_with("plugin:"))
        );
    }

    #[tokio::test]
    async fn prepared_checkpoint_without_store_persists_user_before_protocol_event() {
        let user = text_message("u0", MessageRole::User, "hello");
        let diagnostic = test_protocol_event("diagnostic");
        let store = RecordingStore::default();
        let mut pipeline = TurnBoundary::from_state(state_with_graph(SessionGraph::default()));

        pipeline
            .prepared_checkpoint(
                None,
                SessionPolicy::default(),
                0,
                &MessageSequence::from_base(vec![user.clone()].into()),
                None,
            )
            .await
            .expect("checkpoint without store");
        let owner = lease_owner("turn-boundary-test");
        let lease = store
            .try_claim_session_execution_lease("session-1", &owner, 60_000)
            .await
            .expect("claim test session execution lease")
            .acquired()
            .expect("test session execution lease");
        pipeline = pipeline.with_session_execution_lease(Some(lease.fence()));

        let boundary = pipeline
            .progress_boundary_with_snapshot(ProgressBoundarySnapshot {
                policy: SessionPolicy::default(),
                turn_index: 1,
                messages: MessageSequence::from_base(vec![user].into()),
                event_delta: vec![crate::SessionEventRecord::Protocol(diagnostic)],
                execution_state_snapshot: None,
                plugins: None,
                store: Some(&store),
            })
            .await
            .expect("progress boundary");

        assert!(boundary.persisted);
        let stored_graph = stored_graph_with_head_leaf(&store);
        let expected = vec!["message:u0", "protocol:diagnostic"];
        assert_eq!(persisted_event_order(&stored_graph), expected);
        assert_eq!(chronological_event_order(&stored_graph), expected);
    }

    #[tokio::test]
    async fn progress_boundary_persists_assistant_before_protocol_entry() {
        let user = text_message("u0", MessageRole::User, "hello");
        let assistant = text_message("a0", MessageRole::Assistant, "hi");
        let trajectory = test_protocol_event("trajectory");
        let graph = SessionGraph::from_active_read_state(std::slice::from_ref(&user));
        let store = RecordingStore::default();
        store
            .session_graph
            .lock()
            .expect("lock graph")
            .extend_node_records(graph.nodes.iter().cloned());
        let (mut pipeline, _lease) = leased_boundary(&store, state_with_graph(graph.clone())).await;

        let boundary = pipeline
            .progress_boundary_with_snapshot(ProgressBoundarySnapshot {
                policy: SessionPolicy::default(),
                turn_index: 1,
                messages: MessageSequence::from_base(vec![user, assistant.clone()].into()),
                event_delta: vec![
                    crate::SessionEventRecord::Conversation(ConversationRecord::from_message(
                        assistant,
                    )),
                    crate::SessionEventRecord::Protocol(trajectory),
                ],
                execution_state_snapshot: None,
                plugins: None,
                store: Some(&store),
            })
            .await
            .expect("progress boundary");

        assert!(boundary.persisted);
        let stored_graph = stored_graph_with_head_leaf(&store);
        let expected = vec!["message:u0", "message:a0", "protocol:trajectory"];
        assert_eq!(persisted_event_order(&stored_graph), expected);
        assert_eq!(chronological_event_order(&stored_graph), expected);
    }

    #[tokio::test]
    async fn progress_boundary_can_update_turn_draft_without_store_commit() {
        let user = text_message("u0", MessageRole::User, "hello");
        let assistant = text_message("a0", MessageRole::Assistant, "hi");
        let graph = SessionGraph::from_active_read_state(std::slice::from_ref(&user));
        let mut pipeline = TurnBoundary::from_state(state_with_graph(graph));
        let protocol_event =
            crate::ProtocolEvent::typed("test_protocol", serde_json::json!({"step": "started"}))
                .expect("protocol event serializes");
        let event_delta = vec![crate::SessionEventRecord::Protocol(protocol_event)];

        let boundary = pipeline
            .progress_boundary_with_snapshot(ProgressBoundarySnapshot {
                policy: SessionPolicy::default(),
                turn_index: 1,
                messages: MessageSequence::from_base(vec![user, assistant].into()),
                event_delta,
                execution_state_snapshot: None,
                plugins: None,
                store: None,
            })
            .await
            .expect("progress boundary");

        assert!(!boundary.persisted);
        assert_eq!(boundary.protocol_events.len(), 1);
        assert_eq!(pipeline.state().turn_index, 1);
    }

    #[tokio::test]
    async fn progress_boundary_logs_and_continues_on_store_failure() {
        let user = text_message("u0", MessageRole::User, "hello");
        let assistant = text_message("a0", MessageRole::Assistant, "hi");
        let graph = SessionGraph::from_active_read_state(std::slice::from_ref(&user));
        let protocol_event =
            crate::ProtocolEvent::typed("test_protocol", serde_json::json!({"step": "started"}))
                .expect("protocol event serializes");
        let event_delta = vec![crate::SessionEventRecord::Protocol(protocol_event)];
        let store = RecordingStore::default();
        store
            .save_session_head_meta(crate::SessionHeadMeta {
                session_id: "other-session".to_string(),
                ..crate::SessionHeadMeta::default()
            })
            .await;
        let (mut pipeline, _lease) = leased_boundary(&store, state_with_graph(graph)).await;

        let boundary = pipeline
            .progress_boundary_with_snapshot(ProgressBoundarySnapshot {
                policy: SessionPolicy::default(),
                turn_index: 1,
                messages: MessageSequence::from_base(vec![user, assistant].into()),
                event_delta,
                execution_state_snapshot: None,
                plugins: None,
                store: Some(&store),
            })
            .await
            .expect("progress boundary");

        assert!(!boundary.persisted);
        assert_eq!(boundary.protocol_events.len(), 1);
        assert_eq!(
            *store
                .runtime_commit_count
                .lock()
                .expect("lock runtime commit count"),
            0
        );
    }

    #[test]
    fn committed_attachment_ids_merge_pending_store_writes_with_tool_outputs() {
        let tool_ref = image_ref("tool-output");
        let state = RuntimeSessionState::default();
        let tool_calls = vec![crate::ToolCallRecord {
            call_id: Some("call-1".to_string()),
            tool: "make_attachment".to_string(),
            args: serde_json::json!({}),
            output: crate::ToolCallOutput::success(crate::ToolValue::Attachment(tool_ref)),
            duration_ms: 1,
        }];

        let ids = committed_attachment_ids(
            &state,
            &tool_calls,
            vec![
                crate::AttachmentId::new("tool-output"),
                crate::AttachmentId::new("store-write"),
            ],
        );

        assert_eq!(
            ids,
            vec![
                crate::AttachmentId::new("store-write"),
                crate::AttachmentId::new("tool-output"),
            ]
        );
    }

    #[tokio::test]
    async fn final_commit_merges_usage_and_updates_persisted_graph_count() {
        let graph =
            SessionGraph::from_active_read_state(&[text_message("u0", MessageRole::User, "hello")]);
        let usage = vec![
            usage_entry("child", "gpt", 5),
            usage_entry("turn", "gpt", 17),
        ];
        let store = RecordingStore::default();
        let (mut pipeline, _lease) = leased_boundary(&store, state_with_graph(graph.clone())).await;
        let returned_state = pipeline.export_state_for_assembly();

        pipeline
            .final_commit_with_snapshots(FinalCommitInput {
                returned_state: &returned_state,
                plugins: None,
                execution_state_snapshot: Some(Some(b"runtime".to_vec())),
                store: Some(&store),
                usage_deltas: &usage,
                outcome: &TurnOutcome::Stopped(crate::TurnStop::Cancelled),
                tool_calls: &[],
                turn_id: None,
                completed_queue_claims: Vec::new(),
                completed_turn_input_claims: Vec::new(),
                interrupted_turn_input_turn_id: None,
                pending_attachment_ids: Vec::new(),
                session_execution_lease_completion: None,
            })
            .await
            .expect("commit");

        assert_eq!(
            store.usage_deltas.lock().expect("lock usage deltas").len(),
            2
        );
        assert_eq!(pipeline.state_mut().token_ledger.len(), 2);
        assert!(pipeline.state_mut().execution_state_snapshot().is_none());
        assert!(pipeline.state_mut().head_revision.is_some());
    }

    #[tokio::test]
    async fn no_store_final_commit_discards_snapshots_without_touching_graph_or_usage() {
        let graph =
            SessionGraph::from_active_read_state(&[text_message("u0", MessageRole::User, "hello")]);
        let usage = vec![usage_entry("turn", "model", 5)];
        let mut state = state_with_graph(graph.clone());
        state.token_ledger = usage.clone();
        state.tool_state_snapshot = Some(crate::ToolState::default());
        state.plugin_snapshot = Some(crate::PluginSessionSnapshot::default());
        state.execution_state_snapshot = Some(b"runtime".to_vec());
        let mut pipeline = TurnBoundary::from_state(state);
        let returned_state = pipeline.export_state_for_assembly();

        pipeline
            .final_commit_with_snapshots(FinalCommitInput {
                returned_state: &returned_state,
                plugins: None,
                execution_state_snapshot: None,
                store: None,
                usage_deltas: &[],
                outcome: &TurnOutcome::Stopped(crate::TurnStop::Cancelled),
                tool_calls: &[],
                turn_id: None,
                completed_queue_claims: Vec::new(),
                completed_turn_input_claims: Vec::new(),
                interrupted_turn_input_turn_id: None,
                pending_attachment_ids: Vec::new(),
                session_execution_lease_completion: None,
            })
            .await
            .expect("no-store commit");

        let state = pipeline.state_mut();
        assert_eq!(state.session_graph.nodes.len(), graph.nodes.len());
        assert_eq!(state.token_ledger.len(), usage.len());
        assert!(state.tool_state_snapshot.is_none());
        assert!(state.plugin_snapshot.is_none());
        assert!(state.execution_state_snapshot.is_none());
    }
}
