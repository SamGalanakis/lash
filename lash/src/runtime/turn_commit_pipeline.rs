use std::sync::Arc;

use crate::session_model::SessionEventRecord;
use crate::store::{GraphCommitDelta, RuntimeCommit, RuntimePersistence, StoreError};
use crate::{
    AssembledTurn, MessageSequence, PluginSession, Session, SessionPolicy, SessionReadView,
    ToolCallRecord,
};

use super::{PersistedSessionState, RuntimeError, TurnProgress, merge_ledger_entry};

pub(super) struct ProgressBoundaryCommit {
    pub(super) mirrored_events: Vec<SessionEventRecord>,
    pub(super) persisted: bool,
}

struct ProgressBoundarySnapshot<'a> {
    policy: SessionPolicy,
    iteration: usize,
    messages: MessageSequence,
    events: Arc<Vec<SessionEventRecord>>,
    execution_state_snapshot: Option<Option<Vec<u8>>>,
    plugins: Option<&'a PluginSession>,
    store: Option<&'a (dyn RuntimePersistence + 'a)>,
}

pub(super) struct TurnCommitPipeline {
    progress: Option<TurnProgress>,
    final_state: Option<PersistedSessionState>,
    final_graph_commit: Option<GraphCommitDelta>,
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
    pub(super) fn from_state(state: PersistedSessionState) -> Self {
        Self {
            progress: Some(TurnProgress::from_state(state)),
            final_state: None,
            final_graph_commit: None,
        }
    }

    pub(super) fn state_mut(&mut self) -> &mut PersistedSessionState {
        match self.progress.as_mut() {
            Some(progress) => progress.state_mut(),
            None => self
                .final_state
                .as_mut()
                .expect("turn commit pipeline final state must be present"),
        }
    }

    pub(super) fn apply_prepared_messages(&mut self, messages: &MessageSequence) {
        self.progress_mut().apply_prepared_messages(messages);
    }

    pub(super) fn record_tool_calls<I>(&mut self, records: I)
    where
        I: IntoIterator<Item = ToolCallRecord>,
    {
        self.progress_mut().record_tool_calls(records);
    }

    pub(super) fn read_view(
        &self,
        policy: crate::SessionPolicy,
        iteration: usize,
        mode_turn_options: crate::ModeTurnOptions,
        messages: MessageSequence,
    ) -> SessionReadView {
        self.progress_ref()
            .read_view(policy, iteration, mode_turn_options, messages)
    }

    pub(super) fn active_events(&self) -> Arc<Vec<SessionEventRecord>> {
        self.progress_ref().active_events()
    }

    pub(super) fn finalize_turn_read_state(
        &mut self,
        new_messages: MessageSequence,
        tool_calls: &[ToolCallRecord],
        cancelled: bool,
    ) {
        self.progress_mut()
            .finalize_turn_read_state(new_messages, tool_calls, cancelled);
    }

    pub(super) async fn prepared_checkpoint(
        &mut self,
        store: Option<&(dyn RuntimePersistence + '_)>,
        policy: SessionPolicy,
        iteration: usize,
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
        let state = self.progress_mut().state_mut();
        state.policy = policy;
        state.iteration = iteration;
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
        iteration: usize,
        messages: MessageSequence,
        events: Arc<Vec<SessionEventRecord>>,
    ) -> ProgressBoundaryCommit {
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return ProgressBoundaryCommit {
                mirrored_events: Vec::new(),
                persisted: false,
            };
        }

        let store = session.history_store();
        let execution_state_snapshot = Self::snapshot_dirty_execution_state(session).await;
        let plugins = Arc::clone(session.plugins());
        self.progress_boundary_with_snapshot(ProgressBoundarySnapshot {
            policy,
            iteration,
            messages,
            events,
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
            iteration,
            messages,
            events,
            execution_state_snapshot,
            plugins,
            store,
        } = snapshot;
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return ProgressBoundaryCommit {
                mirrored_events: Vec::new(),
                persisted: false,
            };
        }

        let mirrored_events = self.progress_mut().mirror_sansio_progress(&events);
        {
            let progress = self.progress_mut();
            progress.apply_prepared_messages(&messages);
            let state = progress.state_mut();
            state.policy = policy;
            state.iteration = iteration;
            if let Some(execution_state_snapshot) = execution_state_snapshot {
                state.set_execution_state_snapshot(execution_state_snapshot);
            }
            if let Some(plugins) = plugins {
                state.refresh_plugin_snapshots(plugins);
            }
        }

        let Some(store) = store else {
            return ProgressBoundaryCommit {
                mirrored_events,
                persisted: false,
            };
        };
        match self.commit_progress_graph(store, &[]).await {
            Ok(()) => ProgressBoundaryCommit {
                mirrored_events,
                persisted: true,
            },
            Err(err) => {
                tracing::warn!("failed to persist runtime progress boundary: {err}");
                ProgressBoundaryCommit {
                    mirrored_events,
                    persisted: false,
                }
            }
        }
    }

    pub(super) fn export_state_for_assembly(&mut self) -> crate::SessionStateEnvelope {
        self.final_state_mut().export_state()
    }

    pub(super) async fn final_commit(
        &mut self,
        returned_turn: &mut AssembledTurn,
        session: Option<&mut Session>,
        usage_deltas: &[crate::TokenLedgerEntry],
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
        self.final_commit_with_snapshots(
            &returned_turn.state,
            plugins.as_deref(),
            execution_state_snapshot,
            store.as_ref().map(|store| store.as_ref()),
            usage_deltas,
        )
        .await
        .map_err(|err| RuntimeError {
            code: "store_commit_failed".to_string(),
            message: err.to_string(),
        })?;
        returned_turn.state = self.final_state_mut().export_state();
        Ok(())
    }

    pub(super) fn into_final_state(mut self) -> PersistedSessionState {
        if let Some(state) = self.final_state.take() {
            return state;
        }
        self.progress
            .take()
            .expect("turn commit pipeline progress must be present")
            .into_final_state()
    }

    fn progress_ref(&self) -> &TurnProgress {
        self.progress
            .as_ref()
            .expect("turn progress is unavailable after final state materialization")
    }

    fn progress_mut(&mut self) -> &mut TurnProgress {
        self.progress
            .as_mut()
            .expect("turn progress is unavailable after final state materialization")
    }

    fn final_state_mut(&mut self) -> &mut PersistedSessionState {
        if self.final_state.is_none() {
            let progress = self
                .progress
                .take()
                .expect("turn commit pipeline progress must be present");
            self.final_graph_commit =
                Some(progress.graph_commit(progress.state().graph_replace_required));
            self.final_state = Some(progress.into_final_state());
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
        let progress = self.progress_mut();
        let state = progress.state();
        let graph = progress.graph_commit(state.graph_replace_required);
        self.apply_commit(store, graph, usage_deltas).await
    }

    async fn final_commit_with_snapshots(
        &mut self,
        returned_state: &crate::SessionStateEnvelope,
        plugins: Option<&PluginSession>,
        execution_state_snapshot: Option<Option<Vec<u8>>>,
        store: Option<&(dyn RuntimePersistence + '_)>,
        usage_deltas: &[crate::TokenLedgerEntry],
    ) -> Result<(), StoreError> {
        let progress_graph = self
            .progress
            .as_ref()
            .map(|progress| progress.graph_commit(progress.state().graph_replace_required))
            .or_else(|| self.final_graph_commit.clone());
        let state = self.final_state_mut();
        let materialized_graph = state.session_graph.clone();
        state.apply_exported_state(returned_state);
        if state.session_graph.nodes.is_empty() && !materialized_graph.nodes.is_empty() {
            state.session_graph = materialized_graph;
        }
        for entry in usage_deltas.iter().cloned() {
            merge_ledger_entry(&mut state.token_ledger, entry);
        }
        if let Some(plugins) = plugins {
            state.refresh_plugin_snapshots(plugins);
        }
        if let Some(execution_state_snapshot) = execution_state_snapshot {
            state.set_execution_state_snapshot(execution_state_snapshot);
        }

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
            self.apply_commit(store, graph, usage_deltas).await
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
    ) -> Result<(), StoreError> {
        let state = self.state_mut();
        let mark = PersistedGraphMark::from_graph_commit(&graph);
        let commit = RuntimeCommit::persisted_state_with_graph_commit(state, graph, usage_deltas);
        let result = store.commit_runtime_state(commit).await?;
        state.apply_persisted_commit_result(result);
        if let Some(progress) = self.progress.as_mut() {
            match mark {
                PersistedGraphMark::Unchanged => {}
                PersistedGraphMark::Append(node_ids) => {
                    progress.mark_node_ids_persisted(node_ids);
                }
                PersistedGraphMark::ReplaceFull(node_ids) => {
                    progress.replace_persisted_node_ids(node_ids);
                }
            }
        }
        Ok(())
    }

    async fn snapshot_dirty_execution_state(session: &mut Session) -> Option<Option<Vec<u8>>> {
        let mode_session = std::sync::Arc::clone(session.plugins().mode_session());
        if !mode_session.execution_state_dirty() {
            return None;
        }
        let session_id = session.session_id().to_string();
        match mode_session
            .snapshot_execution_state(crate::plugin::ModeSessionContext::new(session, &session_id))
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use super::*;
    use crate::session_model::{ConversationRecord, MessageRole, Part, PartKind, PruneState};
    use crate::{Message, SessionGraph, TokenUsage, shared_parts};

    #[derive(Default)]
    struct RecordingStore {
        session_head_meta: Mutex<Option<crate::SessionHeadMeta>>,
        session_graph: Mutex<SessionGraph>,
        usage_deltas: Mutex<Vec<crate::TokenLedgerEntry>>,
        runtime_commit_count: Mutex<usize>,
    }

    impl RecordingStore {
        async fn save_session_head_meta(&self, meta: crate::SessionHeadMeta) {
            *self.session_head_meta.lock().expect("lock head meta") = Some(meta);
        }
    }

    #[async_trait::async_trait]
    impl RuntimePersistence for RecordingStore {
        async fn load_session(
            &self,
            scope: crate::store::SessionReadScope,
        ) -> Result<Option<crate::store::PersistedSessionRead>, StoreError> {
            let Some(meta) = self
                .session_head_meta
                .lock()
                .expect("lock head meta")
                .clone()
            else {
                return Ok(None);
            };
            let mut graph = self.session_graph.lock().expect("lock graph").clone();
            if let crate::store::SessionReadScope::ActivePath { leaf_node_id } = scope {
                if let Some(leaf_node_id) = leaf_node_id.or_else(|| meta.leaf_node_id.clone()) {
                    graph.set_leaf_node_id(Some(leaf_node_id));
                }
                graph = graph.fork_current_path();
            }
            Ok(Some(crate::store::PersistedSessionRead {
                session_id: meta.session_id,
                head_revision: meta.head_revision,
                config: meta.config,
                graph,
                checkpoint_ref: meta.checkpoint_ref,
                checkpoint: None,
                token_ledger: self.usage_deltas.lock().expect("lock usage deltas").clone(),
            }))
        }

        async fn load_node(
            &self,
            node_id: &str,
        ) -> Result<Option<crate::SessionNodeRecord>, StoreError> {
            Ok(self
                .session_graph
                .lock()
                .expect("lock graph")
                .find_node(node_id)
                .cloned())
        }

        async fn commit_runtime_state(
            &self,
            commit: RuntimeCommit,
        ) -> Result<crate::store::RuntimeCommitResult, StoreError> {
            let mut meta = self.session_head_meta.lock().expect("lock head meta");
            let actual = meta.as_ref().map_or(0, |meta| meta.head_revision);
            if let Some(bound) = meta.as_ref().map(|meta| meta.session_id.clone())
                && bound != commit.session_id
            {
                return Err(StoreError::SessionBindingMismatch {
                    bound_session_id: bound,
                    attempted_session_id: commit.session_id,
                });
            }
            if commit.expected_head_revision.is_some()
                && commit.expected_head_revision != Some(actual)
            {
                return Err(StoreError::HeadRevisionConflict {
                    expected: commit.expected_head_revision,
                    actual,
                });
            }
            let mut graph = self.session_graph.lock().expect("lock graph");
            let leaf_node_id = match &commit.graph {
                GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
                GraphCommitDelta::Append {
                    nodes,
                    leaf_node_id,
                } => {
                    graph.extend_node_records(nodes.iter().cloned());
                    leaf_node_id.clone()
                }
                GraphCommitDelta::ReplaceFull(next) => {
                    *graph = next.clone();
                    next.leaf_node_id.clone()
                }
            };
            self.usage_deltas
                .lock()
                .expect("lock usage deltas")
                .extend(commit.usage_deltas.iter().cloned());
            let checkpoint_ref = crate::BlobRef(format!("recording-checkpoint-{}", actual + 1));
            let manifest = crate::store::SessionCheckpoint {
                turn_state: commit.checkpoint.turn_state,
                dynamic_state_ref: commit.checkpoint.dynamic_state_ref,
                plugin_snapshot_ref: commit.checkpoint.plugin_snapshot_ref,
                plugin_snapshot_revision: commit.checkpoint.plugin_snapshot_revision,
                execution_state_ref: commit.checkpoint.execution_state_ref,
            };
            let head_revision = actual + 1;
            *meta = Some(crate::SessionHeadMeta {
                session_id: commit.session_id,
                head_revision,
                config: commit.config,
                checkpoint_ref: Some(checkpoint_ref.clone()),
                leaf_node_id,
                graph_node_count: graph.nodes.len(),
                token_ledger: Vec::new(),
            });
            *self
                .runtime_commit_count
                .lock()
                .expect("lock runtime commit count") += 1;
            Ok(crate::store::RuntimeCommitResult {
                head_revision,
                checkpoint_ref,
                manifest,
            })
        }

        async fn save_session_meta(
            &self,
            _meta: crate::store::SessionMeta,
        ) -> Result<(), StoreError> {
            Ok(())
        }

        async fn load_session_meta(&self) -> Result<Option<crate::store::SessionMeta>, StoreError> {
            Ok(None)
        }

        async fn tombstone_nodes(&self, _ids: &[String]) -> Result<(), StoreError> {
            Ok(())
        }

        async fn vacuum(&self) -> Result<crate::store::VacuumReport, StoreError> {
            Ok(crate::store::VacuumReport::default())
        }

        async fn gc_unreachable(&self) -> Result<crate::store::GcReport, StoreError> {
            Ok(crate::store::GcReport::default())
        }
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
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]),
            user_input: None,
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

    fn state_with_graph(graph: SessionGraph) -> PersistedSessionState {
        PersistedSessionState {
            session_id: "session-1".to_string(),
            session_graph: graph,
            ..PersistedSessionState::default()
        }
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

        let stored_graph = store.session_graph.lock().expect("lock graph").clone();
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
    async fn progress_boundary_mirrors_once_and_reports_successful_commit() {
        let user = text_message("u0", MessageRole::User, "hello");
        let assistant = text_message("a0", MessageRole::Assistant, "hi");
        let graph = SessionGraph::from_active_read_state(std::slice::from_ref(&user), &[]);
        let mut pipeline = TurnCommitPipeline::from_state(state_with_graph(graph));
        let events = Arc::new(vec![
            crate::SessionEventRecord::Conversation(ConversationRecord::from_message(user.clone())),
            crate::SessionEventRecord::Conversation(ConversationRecord::from_message(
                assistant.clone(),
            )),
        ]);
        let store = RecordingStore::default();

        let boundary = pipeline
            .progress_boundary_with_snapshot(ProgressBoundarySnapshot {
                policy: SessionPolicy::default(),
                iteration: 1,
                messages: MessageSequence::from_base(vec![user.clone(), assistant.clone()].into()),
                events: Arc::clone(&events),
                execution_state_snapshot: None,
                plugins: None,
                store: Some(&store),
            })
            .await;

        assert!(boundary.persisted);
        assert_eq!(boundary.mirrored_events.len(), 1);
        let second = pipeline
            .progress_boundary_with_snapshot(ProgressBoundarySnapshot {
                policy: SessionPolicy::default(),
                iteration: 1,
                messages: MessageSequence::from_base(vec![user, assistant].into()),
                events,
                execution_state_snapshot: None,
                plugins: None,
                store: Some(&store),
            })
            .await;
        assert!(second.mirrored_events.is_empty());
    }

    #[tokio::test]
    async fn progress_boundary_logs_and_continues_on_store_failure() {
        let user = text_message("u0", MessageRole::User, "hello");
        let assistant = text_message("a0", MessageRole::Assistant, "hi");
        let graph = SessionGraph::from_active_read_state(std::slice::from_ref(&user), &[]);
        let mut pipeline = TurnCommitPipeline::from_state(state_with_graph(graph));
        let events = Arc::new(vec![
            crate::SessionEventRecord::Conversation(ConversationRecord::from_message(user.clone())),
            crate::SessionEventRecord::Conversation(ConversationRecord::from_message(
                assistant.clone(),
            )),
        ]);
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
                iteration: 1,
                messages: MessageSequence::from_base(vec![user, assistant].into()),
                events,
                execution_state_snapshot: None,
                plugins: None,
                store: Some(&store),
            })
            .await;

        assert!(!boundary.persisted);
        assert_eq!(boundary.mirrored_events.len(), 1);
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
            .final_commit_with_snapshots(
                &returned_state,
                None,
                Some(Some(b"runtime".to_vec())),
                Some(&store),
                &usage,
            )
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
        state.dynamic_state_snapshot = Some(crate::DynamicStateSnapshot {
            base_generation: 1,
            tools: BTreeMap::new(),
        });
        state.plugin_snapshot = Some(crate::PluginSessionSnapshot::default());
        state.execution_state_snapshot = Some(b"runtime".to_vec());
        let mut pipeline = TurnCommitPipeline::from_state(state);
        let returned_state = pipeline.export_state_for_assembly();

        pipeline
            .final_commit_with_snapshots(&returned_state, None, None, None, &[])
            .await
            .expect("no-store commit");

        let state = pipeline.state_mut();
        assert_eq!(state.session_graph.nodes.len(), graph.nodes.len());
        assert_eq!(state.token_ledger.len(), usage.len());
        assert!(state.dynamic_state_snapshot.is_none());
        assert!(state.plugin_snapshot.is_none());
        assert!(state.execution_state_snapshot.is_none());
    }
}
