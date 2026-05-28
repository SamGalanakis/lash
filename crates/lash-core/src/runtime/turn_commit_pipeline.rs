use std::sync::Arc;

use crate::session_model::{SessionEventRecord, fresh_message_id};
use crate::store::{GraphCommitDelta, RuntimeCommit, RuntimePersistence, StoreError};
use crate::{
    AssembledTurn, Message, MessageRole, MessageSequence, Part, PartKind, PluginSession,
    PruneState, Session, SessionPolicy, SessionReadView, ToolCallRecord, TurnFinish, TurnOutcome,
    shared_parts,
};

use super::{
    RuntimeError, RuntimeErrorCode, RuntimeSessionState, TurnCommitDraft, merge_ledger_entry,
};

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

pub(super) struct TurnCommitPipeline {
    draft: Option<TurnCommitDraft>,
    final_state: Option<RuntimeSessionState>,
    final_graph_commit: Option<GraphCommitDelta>,
}

struct FinalCommitInput<'a> {
    returned_state: &'a crate::SessionStateEnvelope,
    tool_calls: &'a [ToolCallRecord],
    plugins: Option<&'a PluginSession>,
    execution_state_snapshot: Option<Option<Vec<u8>>>,
    store: Option<&'a (dyn RuntimePersistence + 'a)>,
    usage_deltas: &'a [crate::TokenLedgerEntry],
    outcome: &'a TurnOutcome,
    completed_turn: Option<crate::RuntimeTurnCompletion>,
    completed_queue_claims: Vec<crate::QueuedWorkCompletion>,
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

impl TurnCommitPipeline {
    pub(super) fn from_state(state: RuntimeSessionState) -> Self {
        Self {
            draft: Some(TurnCommitDraft::from_state(state)),
            final_state: None,
            final_graph_commit: None,
        }
    }

    pub(super) fn state_mut(&mut self) -> &mut RuntimeSessionState {
        match self.draft.as_mut() {
            Some(draft) => draft.state_mut(),
            None => self
                .final_state
                .as_mut()
                .expect("turn commit pipeline final state must be present"),
        }
    }

    pub(super) fn state(&self) -> &RuntimeSessionState {
        match self.draft.as_ref() {
            Some(draft) => draft.state(),
            None => self
                .final_state
                .as_ref()
                .expect("turn commit pipeline final state must be present"),
        }
    }

    pub(super) fn apply_prepared_messages(&mut self, messages: &MessageSequence) {
        self.draft_mut().apply_prepared_messages(messages);
    }

    pub(super) fn record_tool_calls<I>(&mut self, records: I)
    where
        I: IntoIterator<Item = ToolCallRecord>,
    {
        self.draft_mut().record_tool_calls(records);
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
        tool_calls: &[ToolCallRecord],
        cancelled: bool,
    ) {
        self.draft_mut()
            .finalize_turn_read_state(new_messages, tool_calls, cancelled);
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
        let Some(store) = store else {
            return Ok(());
        };

        self.apply_prepared_messages(messages);
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
    ) -> ProgressBoundaryCommit {
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return ProgressBoundaryCommit {
                protocol_events: Vec::new(),
                persisted: false,
            };
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

    async fn progress_boundary_with_snapshot(
        &mut self,
        snapshot: ProgressBoundarySnapshot<'_>,
    ) -> ProgressBoundaryCommit {
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
            return ProgressBoundaryCommit {
                protocol_events: Vec::new(),
                persisted: false,
            };
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
            return ProgressBoundaryCommit {
                protocol_events,
                persisted: false,
            };
        };
        match self.commit_progress_graph(store, &[]).await {
            Ok(()) => ProgressBoundaryCommit {
                protocol_events,
                persisted: true,
            },
            Err(err) => {
                tracing::warn!("failed to persist runtime progress boundary: {err}");
                ProgressBoundaryCommit {
                    protocol_events,
                    persisted: false,
                }
            }
        }
    }

    pub(super) fn export_state_for_assembly(&mut self) -> crate::SessionStateEnvelope {
        self.final_state_mut().export_state()
    }

    pub(super) fn apply_event_delta(
        &mut self,
        event_delta: Vec<SessionEventRecord>,
    ) -> Vec<crate::ProtocolEvent> {
        let protocol_events = event_delta
            .into_iter()
            .filter_map(|event| match event {
                SessionEventRecord::Protocol(event) => Some(event),
                SessionEventRecord::Conversation(_) | SessionEventRecord::Tool(_) => None,
            })
            .collect::<Vec<_>>();
        self.draft_mut()
            .append_protocol_events(protocol_events.iter().cloned());
        protocol_events
    }

    pub(super) async fn final_commit(
        &mut self,
        returned_turn: &mut AssembledTurn,
        session: Option<&mut Session>,
        usage_deltas: &[crate::TokenLedgerEntry],
        completed_turn: Option<crate::RuntimeTurnCompletion>,
        completed_queue_claims: Vec<crate::QueuedWorkCompletion>,
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
            completed_turn,
            completed_queue_claims,
        })
        .await
        .map_err(|err| RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string()))?;
        returned_turn.state = self.final_state_mut().export_state();
        Ok(())
    }

    pub(super) fn into_final_state(mut self) -> RuntimeSessionState {
        if let Some(state) = self.final_state.take() {
            return state;
        }
        self.draft
            .take()
            .expect("turn commit pipeline draft must be present")
            .into_final_state()
    }

    fn draft_ref(&self) -> &TurnCommitDraft {
        self.draft
            .as_ref()
            .expect("turn commit draft is unavailable after final state materialization")
    }

    fn draft_mut(&mut self) -> &mut TurnCommitDraft {
        self.draft
            .as_mut()
            .expect("turn commit draft is unavailable after final state materialization")
    }

    fn final_state_mut(&mut self) -> &mut RuntimeSessionState {
        if self.final_state.is_none() {
            let draft = self
                .draft
                .take()
                .expect("turn commit pipeline draft must be present");
            self.final_graph_commit =
                Some(draft.graph_commit(draft.state().graph_replace_required));
            self.final_state = Some(draft.into_final_state());
        }
        self.final_state
            .as_mut()
            .expect("turn commit pipeline final state must be present")
    }

    async fn commit_progress_graph(
        &mut self,
        store: &(dyn RuntimePersistence + '_),
        usage_deltas: &[crate::TokenLedgerEntry],
    ) -> Result<(), StoreError> {
        let draft = self.draft_mut();
        let state = draft.state();
        let graph = draft.graph_commit(state.graph_replace_required);
        self.apply_commit(store, graph, usage_deltas, None, Vec::new())
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
            completed_turn,
            completed_queue_claims,
        } = input;
        let state = self.final_state_mut();
        state.apply_exported_state(returned_state);
        for entry in usage_deltas.iter().cloned() {
            merge_ledger_entry(&mut state.token_ledger, entry);
        }
        if let Some(plugins) = plugins {
            state.refresh_plugin_snapshots(plugins);
        }
        if let Some(execution_state_snapshot) = execution_state_snapshot {
            state.set_execution_state_snapshot(execution_state_snapshot);
        }
        materialize_terminal_output(state, outcome);
        materialize_agent_frame_switch(state, outcome, tool_calls);
        let progress_graph = self
            .draft
            .as_ref()
            .map(|draft| draft.graph_commit(draft.state().graph_replace_required))
            .or_else(|| self.final_graph_commit.clone());
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
            self.apply_commit(
                store,
                graph,
                usage_deltas,
                completed_turn,
                completed_queue_claims,
            )
            .await
        } else {
            state.discard_runtime_snapshots();
            Ok(())
        }
    }

    async fn apply_commit(
        &mut self,
        store: &(dyn RuntimePersistence + '_),
        graph: GraphCommitDelta,
        usage_deltas: &[crate::TokenLedgerEntry],
        completed_turn: Option<crate::RuntimeTurnCompletion>,
        completed_queue_claims: Vec<crate::QueuedWorkCompletion>,
    ) -> Result<(), StoreError> {
        let state = self.state_mut();
        let mark = PersistedGraphMark::from_graph_commit(&graph);
        let mut commit =
            RuntimeCommit::persisted_state_with_graph_commit(state, graph, usage_deltas);
        commit.completed_turn = completed_turn;
        commit.completed_queue_claims = completed_queue_claims;
        let result = store.commit_runtime_state(commit).await?;
        state.apply_persisted_commit_result(result);
        if let Some(draft) = self.draft.as_mut() {
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

fn materialize_terminal_output(state: &mut RuntimeSessionState, outcome: &TurnOutcome) {
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
    state.append_active_conversation_messages(&[Message {
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
    }]);
    state.graph_replace_required = true;
}

fn materialize_agent_frame_switch(
    state: &mut RuntimeSessionState,
    outcome: &TurnOutcome,
    tool_calls: &[ToolCallRecord],
) {
    let TurnOutcome::AgentFrameSwitch { frame_id } = outcome else {
        return;
    };
    if frame_id.trim().is_empty() || state.current_agent_frame_id == *frame_id {
        return;
    }
    let control = tool_calls
        .iter()
        .find_map(|record| match &record.output.control {
            Some(crate::ToolControl::SwitchAgentFrame {
                frame_id: control_frame_id,
                initial_nodes,
                task,
            }) if control_frame_id == frame_id => Some((initial_nodes, task)),
            _ => None,
        });
    let empty_nodes = Vec::new();
    let empty_task = None;
    let (initial_nodes, _task) = control.unwrap_or((&empty_nodes, &empty_task));
    let previous = state.current_agent_frame().cloned();
    let assignment = previous
        .as_ref()
        .map(|frame| frame.assignment.clone())
        .unwrap_or_else(|| crate::AgentFrameAssignment::from_policy(state.policy.clone()));
    let protocol_turn_options = previous
        .as_ref()
        .map(|frame| frame.protocol_turn_options.clone())
        .unwrap_or_else(|| state.protocol_turn_options.clone());
    let previous_frame_id = previous.map(|frame| frame.frame_id);
    state.append_agent_frame(crate::AgentFrameRecord::new(
        frame_id.clone(),
        state.session_id.clone(),
        previous_frame_id,
        crate::AgentFrameReason::ContinueAs,
        None,
        assignment,
        protocol_turn_options,
    ));

    let nodes = initial_nodes
        .iter()
        .filter_map(|value| {
            match serde_json::from_value::<crate::SessionAppendNode>(value.clone()) {
                Ok(node) => Some(node),
                Err(err) => {
                    tracing::warn!("failed to decode agent frame initial node: {err}");
                    None
                }
            }
        })
        .collect::<Vec<_>>();
    if !nodes.is_empty() {
        super::append_session_nodes_to_state(state, &nodes);
        state.graph_replace_required = true;
    }
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
    use crate::{Message, SessionGraph, TokenUsage, shared_parts};

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
                cached_input_tokens: 1,
                reasoning_tokens: 0,
            },
        }
    }

    fn state_with_graph(graph: SessionGraph) -> RuntimeSessionState {
        RuntimeSessionState {
            session_id: "session-1".to_string(),
            session_graph: graph,
            ..RuntimeSessionState::default()
        }
    }

    #[test]
    fn agent_frame_switch_keeps_session_and_tags_initial_nodes_to_new_frame() {
        let graph = SessionGraph::from_active_read_state(
            &[text_message("u0", MessageRole::User, "old frame")],
            &[],
        );
        let mut state = state_with_graph(graph);
        state.ensure_agent_frame_initialized();
        let previous_frame_id = state.current_agent_frame_id.clone();
        let frame_id = "frame-2".to_string();
        let seed_node = crate::SessionAppendNode::message(crate::PluginMessage::text(
            MessageRole::User,
            "seed message",
        ));
        let tool_calls = vec![crate::ToolCallRecord {
            call_id: Some("continue-call".to_string()),
            tool: "continue_as".to_string(),
            args: serde_json::json!({ "task": "next task" }),
            output: crate::ToolCallOutput::success(serde_json::json!({ "ok": true })).with_control(
                crate::ToolControl::SwitchAgentFrame {
                    frame_id: frame_id.clone(),
                    initial_nodes: vec![serde_json::to_value(seed_node).expect("seed node json")],
                    task: Some("next task".to_string()),
                },
            ),
            duration_ms: 1,
        }];

        materialize_agent_frame_switch(
            &mut state,
            &TurnOutcome::AgentFrameSwitch {
                frame_id: frame_id.clone(),
            },
            &tool_calls,
        );

        assert_eq!(state.session_id, "session-1");
        assert_eq!(state.current_agent_frame_id, frame_id);
        let current = state.current_agent_frame().expect("current frame");
        assert_eq!(
            current.previous_frame_id.as_deref(),
            Some(previous_frame_id.as_str())
        );
        assert!(matches!(
            current.reason,
            crate::AgentFrameReason::ContinueAs
        ));
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

    #[tokio::test]
    async fn prepared_checkpoint_writes_only_explicit_progress_graph_tail() {
        let mut graph = SessionGraph::from_active_read_state(
            &[text_message("u0", MessageRole::User, "old")],
            &[],
        );
        let base_graph = graph.clone();
        graph.append_message(text_message("a0", MessageRole::Assistant, "new"));
        let state = state_with_graph(base_graph.clone());
        let store = RecordingStore::default();
        store
            .session_graph
            .lock()
            .expect("lock graph")
            .extend_node_records(base_graph.nodes.iter().cloned());
        let mut pipeline = TurnCommitPipeline::from_state(state);

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
        let graph = SessionGraph::from_active_read_state(
            &[text_message("u0", MessageRole::User, "hello")],
            &[],
        );
        let state = state_with_graph(graph);
        let store = RecordingStore::default();
        store
            .save_session_head_meta(crate::SessionHeadMeta {
                session_id: "other-session".to_string(),
                ..crate::SessionHeadMeta::default()
            })
            .await;
        let mut pipeline = TurnCommitPipeline::from_state(state);

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
        let graph = SessionGraph::from_active_read_state(std::slice::from_ref(&user), &[]);
        let base_graph = graph.clone();
        let mut pipeline = TurnCommitPipeline::from_state(state_with_graph(graph));
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
            .await;

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
                    Some(
                        crate::SessionEventRecord::Conversation(_)
                            | crate::SessionEventRecord::Tool(_)
                    )
                ))
                .all(|node| !node.node_id.starts_with("plugin:"))
        );
    }

    #[tokio::test]
    async fn progress_boundary_logs_and_continues_on_store_failure() {
        let user = text_message("u0", MessageRole::User, "hello");
        let assistant = text_message("a0", MessageRole::Assistant, "hi");
        let graph = SessionGraph::from_active_read_state(std::slice::from_ref(&user), &[]);
        let mut pipeline = TurnCommitPipeline::from_state(state_with_graph(graph));
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
            .await;

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

    #[tokio::test]
    async fn final_commit_merges_usage_and_updates_persisted_graph_count() {
        let graph = SessionGraph::from_active_read_state(
            &[text_message("u0", MessageRole::User, "hello")],
            &[],
        );
        let usage = vec![
            usage_entry("child", "gpt", 5),
            usage_entry("turn", "gpt", 17),
        ];
        let store = RecordingStore::default();
        let mut pipeline = TurnCommitPipeline::from_state(state_with_graph(graph.clone()));
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
                completed_turn: None,
                completed_queue_claims: Vec::new(),
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
        let graph = SessionGraph::from_active_read_state(
            &[text_message("u0", MessageRole::User, "hello")],
            &[],
        );
        let usage = vec![usage_entry("turn", "model", 5)];
        let mut state = state_with_graph(graph.clone());
        state.token_ledger = usage.clone();
        state.tool_state_snapshot = Some(crate::ToolState::default());
        state.plugin_snapshot = Some(crate::PluginSessionSnapshot::default());
        state.execution_state_snapshot = Some(b"runtime".to_vec());
        let mut pipeline = TurnCommitPipeline::from_state(state);
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
                completed_turn: None,
                completed_queue_claims: Vec::new(),
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
