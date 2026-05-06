//! History rewriting and turn-context transform plugin contracts.
//!
//! Split out of `plugin/mod.rs` purely for file size. All types keep
//! their original module path via `pub use` in `plugin/mod.rs`.

use std::sync::{Arc, OnceLock};

use crate::SessionPolicy;
use crate::SessionStateEnvelope;
use crate::runtime::PersistedSessionState;

use super::PluginError;

/// Reason the history pipeline is being invoked.
#[derive(Clone, Debug)]
pub enum RewriteTrigger {
    /// User invoked `/compact` (or an equivalent plugin command).
    Manual { instructions: Option<String> },
    /// The previous turn overflowed the context window; retry with
    /// compacted history.
    OverflowRecovery,
    /// Session config changed to a smaller context window.
    WindowShrink {
        old_max: Option<usize>,
        new_max: Option<usize>,
    },
    /// Reserved for future scheduled compactors — not fired by any call
    /// site today.
    Periodic,
}

/// Metadata accumulated as a history rewrite pipeline runs.
#[derive(Clone, Debug, Default)]
pub struct HistoryRewriteMetadata {
    pub summarized_token_count: Option<u64>,
    pub pruned_message_count: u32,
    pub produced_summary: bool,
}

/// Mutable state passed through the history rewrite pipeline.
#[derive(Clone, Debug)]
pub struct HistoryState {
    pub messages: Vec<crate::Message>,
    pub tool_calls: Vec<crate::ToolCallRecord>,
    pub metadata: HistoryRewriteMetadata,
}

impl HistoryState {
    pub fn from_state(state: &SessionStateEnvelope) -> Self {
        let read_view = state.read_view();
        Self {
            messages: read_view.messages().to_vec(),
            tool_calls: read_view.tool_calls().to_vec(),
            metadata: HistoryRewriteMetadata::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionReadView(Arc<SessionReadState>);

#[derive(Debug)]
struct SessionReadState {
    meta: SessionReadMeta,
    graph: SessionReadGraph,
    read_model: crate::session_graph::SessionReadModel,
    chronological_projection: OnceLock<Arc<crate::ChronologicalProjection>>,
}

#[derive(Clone, Debug)]
struct SessionReadMeta {
    session_id: String,
    policy: SessionPolicy,
    iteration: usize,
    token_usage: crate::TokenUsage,
    last_prompt_usage: Option<crate::runtime::PromptUsage>,
    mode_turn_options: crate::ModeTurnOptions,
}

impl SessionReadMeta {
    fn from_state_ref(state: &SessionStateEnvelope) -> Self {
        Self {
            session_id: state.session_id.clone(),
            policy: state.policy.clone(),
            iteration: state.iteration,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
            mode_turn_options: state.mode_turn_options.clone(),
        }
    }

    fn from_state_owned(state: SessionStateEnvelope) -> Self {
        Self {
            session_id: state.session_id,
            policy: state.policy,
            iteration: state.iteration,
            token_usage: state.token_usage,
            last_prompt_usage: state.last_prompt_usage,
            mode_turn_options: state.mode_turn_options,
        }
    }

    fn from_persisted_ref(state: &PersistedSessionState) -> Self {
        Self {
            session_id: state.session_id.clone(),
            policy: state.policy.clone(),
            iteration: state.iteration,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
            mode_turn_options: state.mode_turn_options.clone(),
        }
    }

    fn with_policy(mut self, policy: SessionPolicy) -> Self {
        self.policy = policy;
        self
    }

    fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = iteration;
        self
    }

    fn with_mode_turn_options(mut self, mode_turn_options: crate::ModeTurnOptions) -> Self {
        self.mode_turn_options = mode_turn_options;
        self
    }

    fn to_owned_state(&self, session_graph: crate::SessionGraph) -> SessionStateEnvelope {
        SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: self.policy.clone(),
            session_graph,
            iteration: self.iteration,
            token_usage: self.token_usage.clone(),
            last_prompt_usage: self.last_prompt_usage.clone(),
            mode_turn_options: self.mode_turn_options.clone(),
        }
    }
}

#[derive(Debug)]
enum SessionReadGraph {
    Owned(Arc<crate::SessionGraph>),
    Derived {
        cache: OnceLock<Arc<crate::SessionGraph>>,
        base_graph: Option<Arc<crate::SessionGraph>>,
    },
}

impl SessionReadView {
    pub fn from_derived_message_view(
        mut state: SessionStateEnvelope,
        active_events: Arc<Vec<crate::SessionEventRecord>>,
        messages: Arc<Vec<crate::Message>>,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_owned({
                state.session_graph = crate::SessionGraph::default();
                state
            }),
            graph: SessionReadGraph::Derived {
                cache: OnceLock::new(),
                base_graph: None,
            },
            read_model: crate::session_graph::SessionReadModel {
                active_events,
                messages,
                tool_calls,
                rlm_globals: Arc::new(serde_json::Map::new()),
                prompt_render_cache: Arc::new(crate::BaseRenderCache::new()),
            },
            chronological_projection: OnceLock::new(),
        }))
    }

    fn from_graph_message_sequence_meta(
        meta: SessionReadMeta,
        base_graph: Arc<crate::SessionGraph>,
        messages: crate::MessageSequence,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        let base_read_model = base_graph.read_model();
        let active_events = base_read_model.active_events;
        let rlm_globals = base_read_model.rlm_globals;
        Self(Arc::new(SessionReadState {
            meta,
            graph: SessionReadGraph::Derived {
                cache: OnceLock::new(),
                base_graph: Some(base_graph),
            },
            read_model: crate::session_graph::SessionReadModel {
                active_events,
                messages: messages.shared(),
                tool_calls,
                rlm_globals,
                prompt_render_cache: Arc::new(crate::BaseRenderCache::new()),
            },
            chronological_projection: OnceLock::new(),
        }))
    }

    #[cfg(test)]
    pub(crate) fn from_graph_message_sequence(
        state: &SessionStateEnvelope,
        base_graph: Arc<crate::SessionGraph>,
        messages: crate::MessageSequence,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        Self::from_graph_message_sequence_meta(
            SessionReadMeta::from_state_ref(state),
            base_graph,
            messages,
            tool_calls,
        )
    }

    pub fn from_exported_state(state: &SessionStateEnvelope) -> Self {
        let read_model = state.session_graph.read_model();
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_ref(state),
            graph: SessionReadGraph::Owned(Arc::new(state.session_graph.clone())),
            read_model,
            chronological_projection: OnceLock::new(),
        }))
    }

    pub fn from_persisted_state(state: &PersistedSessionState) -> Self {
        let graph = Arc::new(state.session_graph.clone());
        let read_model = graph.read_model();
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_persisted_ref(state),
            graph: SessionReadGraph::Owned(graph),
            read_model,
            chronological_projection: OnceLock::new(),
        }))
    }

    pub(crate) fn from_runtime_state(
        state: &PersistedSessionState,
        policy: SessionPolicy,
        mode_turn_options: crate::ModeTurnOptions,
    ) -> Self {
        let graph = Arc::new(state.session_graph.clone());
        let read_model = graph.read_model();
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_persisted_ref(state)
                .with_policy(policy)
                .with_mode_turn_options(mode_turn_options),
            graph: SessionReadGraph::Owned(graph),
            read_model,
            chronological_projection: OnceLock::new(),
        }))
    }

    pub(crate) fn derived_from_persisted_state(
        state: &PersistedSessionState,
        policy: SessionPolicy,
        iteration: usize,
        mode_turn_options: crate::ModeTurnOptions,
        base_graph: Arc<crate::SessionGraph>,
        messages: crate::MessageSequence,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        Self::from_graph_message_sequence_meta(
            SessionReadMeta::from_persisted_ref(state)
                .with_policy(policy)
                .with_iteration(iteration)
                .with_mode_turn_options(mode_turn_options),
            base_graph,
            messages,
            tool_calls,
        )
    }

    fn graph_arc(&self) -> &Arc<crate::SessionGraph> {
        match &self.0.graph {
            SessionReadGraph::Owned(graph) => graph,
            SessionReadGraph::Derived { cache, base_graph } => cache.get_or_init(|| {
                let mut graph = base_graph
                    .as_ref()
                    .map(|graph| graph.as_ref().clone())
                    .unwrap_or_default();
                graph.replace_active_read_state(
                    self.0.read_model.messages.as_slice(),
                    self.0.read_model.tool_calls.as_slice(),
                );
                Arc::new(graph)
            }),
        }
    }

    pub fn session_id(&self) -> &str {
        &self.0.meta.session_id
    }

    pub fn policy(&self) -> &SessionPolicy {
        &self.0.meta.policy
    }

    fn session_graph(&self) -> &crate::SessionGraph {
        self.graph_arc().as_ref()
    }

    pub fn materialized_session_graph(&self) -> crate::SessionGraph {
        self.session_graph().clone()
    }

    pub fn messages(&self) -> &[crate::Message] {
        self.0.read_model.messages.as_slice()
    }

    pub fn tool_calls(&self) -> &[crate::ToolCallRecord] {
        self.0.read_model.tool_calls.as_slice()
    }

    pub fn active_events(&self) -> &[crate::SessionEventRecord] {
        self.0.read_model.active_events.as_slice()
    }

    pub fn rlm_globals(&self) -> serde_json::Map<String, serde_json::Value> {
        self.shared_rlm_globals().as_ref().clone()
    }

    /// Borrow the read model's RLM globals for this view. Hot callers
    /// (per-iteration prompt contributions like `bound_variables`) should use
    /// this instead of [`rlm_globals`] to avoid a deep map clone.
    pub fn shared_rlm_globals(&self) -> Arc<serde_json::Map<String, serde_json::Value>> {
        Arc::clone(&self.0.read_model.rlm_globals)
    }

    #[cfg(test)]
    pub(crate) fn read_model(&self) -> &crate::session_graph::SessionReadModel {
        &self.0.read_model
    }

    pub fn chronological_projection(&self) -> crate::ChronologicalProjection {
        crate::ChronologicalProjection::from_read_model(&self.0.read_model)
    }

    pub(crate) fn shared_chronological_projection(&self) -> Arc<crate::ChronologicalProjection> {
        Arc::clone(self.0.chronological_projection.get_or_init(|| {
            Arc::new(crate::ChronologicalProjection::from_read_model(
                &self.0.read_model,
            ))
        }))
    }

    pub fn rlm_history_len(&self) -> usize {
        crate::ChronologicalProjection::rlm_history_len_from_read_model(&self.0.read_model)
    }

    pub fn rlm_history(&self) -> Vec<lash_rlm_types::RlmHistoryItem> {
        self.shared_chronological_projection().rlm_history()
    }

    pub fn message_tree(&self) -> Vec<crate::SessionMessageTreeNode> {
        self.session_graph().message_tree()
    }

    pub fn iteration(&self) -> usize {
        self.0.meta.iteration
    }

    pub fn token_usage(&self) -> &crate::TokenUsage {
        &self.0.meta.token_usage
    }

    pub fn last_prompt_usage(&self) -> Option<&crate::runtime::PromptUsage> {
        self.0.meta.last_prompt_usage.as_ref()
    }

    pub fn mode_turn_options(&self) -> &crate::ModeTurnOptions {
        &self.0.meta.mode_turn_options
    }

    pub fn to_owned_state(&self) -> SessionStateEnvelope {
        self.0.meta.to_owned_state(self.session_graph().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Message, MessageRole, ModeEvent, Part, PartKind, PruneState, SessionEventRecord,
        ToolCallRecord, ToolEvent,
    };
    use lash_rlm_types::{RlmGlobalsPatchPluginBody, RlmModeEvent, RlmTrajectoryEntry};

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

    fn transient_plugin_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            origin: Some(crate::MessageOrigin::Plugin {
                plugin_id: "test-plugin".to_string(),
                transient: true,
            }),
            ..text_message(id, role, content)
        }
    }

    fn tool_call(call_id: &str, tool: &str) -> ToolCallRecord {
        ToolCallRecord {
            call_id: Some(call_id.to_string()),
            tool: tool.to_string(),
            args: serde_json::json!({"q": tool}),
            result: serde_json::json!({"answer": tool}),
            success: true,
            duration_ms: 5,
        }
    }

    #[test]
    fn read_view_wraps_read_model_with_session_metadata_and_graph() {
        let mut state = SessionStateEnvelope {
            session_id: "session-read-view-test".to_string(),
            iteration: 7,
            mode_turn_options: crate::ModeTurnOptions::default(),
            ..SessionStateEnvelope::default()
        };
        state.policy.max_turns = Some(3);
        let user = text_message("u1", MessageRole::User, "hello");
        let assistant = text_message("a1", MessageRole::Assistant, "hi");
        let tool_call = ToolCallRecord {
            call_id: Some("call_1".to_string()),
            tool: "lookup".to_string(),
            args: serde_json::json!({"q": "hello"}),
            result: serde_json::json!("hi"),
            success: true,
            duration_ms: 5,
        };
        state.session_graph.replace_active_read_state(
            &[user.clone(), assistant.clone()],
            std::slice::from_ref(&tool_call),
        );
        state
            .session_graph
            .append_event(SessionEventRecord::Mode(ModeEvent::rlm(
                RlmModeEvent::RlmGlobalsPatch(RlmGlobalsPatchPluginBody {
                    set: serde_json::Map::from_iter([(
                        "answer".to_string(),
                        serde_json::json!(42),
                    )]),
                    unset: Vec::new(),
                }),
            )));

        let view = SessionReadView::from_exported_state(&state);
        assert_eq!(view.session_id(), "session-read-view-test");
        assert_eq!(view.policy().max_turns, Some(3));
        assert_eq!(view.iteration(), 7);
        assert_eq!(view.messages().len(), 2);
        assert_eq!(view.messages()[0].id, user.id);
        assert_eq!(view.messages()[1].id, assistant.id);
        assert_eq!(view.tool_calls().len(), 1);
        assert_eq!(view.tool_calls()[0].call_id, tool_call.call_id);
        assert_eq!(view.active_events().len(), 4);
        assert_eq!(
            view.shared_rlm_globals().get("answer"),
            Some(&serde_json::json!(42))
        );
        assert_eq!(view.read_model().messages.len(), 2);
        assert_eq!(
            view.materialized_session_graph().nodes.len(),
            state.session_graph.nodes.len()
        );

        let owned = view.to_owned_state();
        assert_eq!(owned.session_id, state.session_id);
        assert_eq!(owned.iteration, state.iteration);
        assert_eq!(
            owned.session_graph.nodes.len(),
            state.session_graph.nodes.len()
        );
        assert_eq!(
            owned.mode_turn_options.mode_id,
            crate::ExecutionMode::standard()
        );
        assert_eq!(
            view.mode_turn_options().mode_id,
            crate::ExecutionMode::standard()
        );
    }

    #[test]
    fn derived_read_view_materializes_graph_lazily_and_preserves_mode_turn_options() {
        let state = SessionStateEnvelope {
            session_id: "derived-read-view".to_string(),
            mode_turn_options: crate::ModeTurnOptions::default(),
            ..SessionStateEnvelope::default()
        };
        let base_graph = Arc::new(crate::SessionGraph::default());
        let user = text_message("u1", MessageRole::User, "hello");

        let view = SessionReadView::from_graph_message_sequence(
            &state,
            base_graph,
            crate::MessageSequence::from_base(vec![user.clone()].into()),
            Arc::new(Vec::new()),
        );

        match &view.0.graph {
            SessionReadGraph::Derived { cache, .. } => assert!(cache.get().is_none()),
            SessionReadGraph::Owned(_) => panic!("expected derived read graph"),
        }
        assert_eq!(view.messages().len(), 1);
        assert_eq!(view.messages()[0].id, user.id);
        match &view.0.graph {
            SessionReadGraph::Derived { cache, .. } => assert!(cache.get().is_none()),
            SessionReadGraph::Owned(_) => panic!("expected derived read graph"),
        }

        let owned = view.to_owned_state();

        assert_eq!(
            owned.mode_turn_options.mode_id,
            crate::ExecutionMode::standard()
        );
        match &view.0.graph {
            SessionReadGraph::Derived { cache, .. } => assert!(cache.get().is_some()),
            SessionReadGraph::Owned(_) => panic!("expected derived read graph"),
        }
    }

    #[test]
    fn shared_chronological_projection_reuses_arc_and_matches_owned_projection() {
        let mut state = SessionStateEnvelope {
            session_id: "shared-projection".to_string(),
            mode_turn_options: crate::ModeTurnOptions::default(),
            ..SessionStateEnvelope::default()
        };
        let user = text_message("u1", MessageRole::User, "hello");
        let transient = transient_plugin_message("t1", MessageRole::System, "hidden");
        let lookup = tool_call("call_lookup", "lookup");
        state.session_graph.append_message(user);
        state.session_graph.append_message(transient);
        state
            .session_graph
            .append_event(SessionEventRecord::Mode(ModeEvent::rlm(
                RlmModeEvent::RlmTrajectoryEntry(RlmTrajectoryEntry {
                    id: "rlm_step_1".to_string(),
                    iteration: 1,
                    reasoning: "inspect".to_string(),
                    code: "lookup()".to_string(),
                    output: vec!["observed".to_string()],
                    tool_calls: vec![lookup.clone(), lookup.clone()],
                    images: Vec::new(),
                    error: None,
                    final_output: None,
                }),
            )));
        state
            .session_graph
            .append_event(SessionEventRecord::Tool(ToolEvent::Invocation {
                stable_key: "call_lookup".to_string(),
                record: lookup,
            }));
        state
            .session_graph
            .append_message(text_message("a1", MessageRole::Assistant, "done"));

        let view = SessionReadView::from_exported_state(&state);
        let first = view.shared_chronological_projection();
        let second = view.shared_chronological_projection();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(
            serde_json::to_value(first.entries()).unwrap(),
            serde_json::to_value(view.chronological_projection().entries()).unwrap()
        );
        assert_eq!(
            view.rlm_history_len(),
            view.chronological_projection().rlm_history_len()
        );
    }

    #[test]
    fn shared_chronological_projection_keeps_derived_graph_cache_cold() {
        let state = SessionStateEnvelope {
            session_id: "derived-shared-projection".to_string(),
            mode_turn_options: crate::ModeTurnOptions::default(),
            ..SessionStateEnvelope::default()
        };
        let base_graph = Arc::new(crate::SessionGraph::default());
        let user = text_message("u1", MessageRole::User, "hello");
        let tool_call = tool_call("call_lookup", "lookup");
        let view = SessionReadView::from_graph_message_sequence(
            &state,
            base_graph,
            crate::MessageSequence::from_base(vec![user].into()),
            Arc::new(vec![tool_call]),
        );

        match &view.0.graph {
            SessionReadGraph::Derived { cache, .. } => assert!(cache.get().is_none()),
            SessionReadGraph::Owned(_) => panic!("expected derived read graph"),
        }

        let shared = view.shared_chronological_projection();
        assert_eq!(
            shared.entries().len(),
            view.chronological_projection().entries().len()
        );

        match &view.0.graph {
            SessionReadGraph::Derived { cache, .. } => assert!(cache.get().is_none()),
            SessionReadGraph::Owned(_) => panic!("expected derived read graph"),
        }
    }

    #[test]
    fn rlm_history_len_matches_full_projection_for_mixed_read_model() {
        let mut state = SessionStateEnvelope {
            session_id: "rlm-history-len".to_string(),
            mode_turn_options: crate::ModeTurnOptions::default(),
            ..SessionStateEnvelope::default()
        };
        let lookup = tool_call("call_lookup", "lookup");
        state
            .session_graph
            .append_message(text_message("u1", MessageRole::User, "hello"));
        state.session_graph.append_message(transient_plugin_message(
            "t1",
            MessageRole::System,
            "hidden",
        ));
        state
            .session_graph
            .append_event(SessionEventRecord::Tool(ToolEvent::Invocation {
                stable_key: "call_lookup".to_string(),
                record: lookup.clone(),
            }));
        state
            .session_graph
            .append_event(SessionEventRecord::Mode(ModeEvent::rlm(
                RlmModeEvent::RlmTrajectoryEntry(RlmTrajectoryEntry {
                    id: "rlm_step_1".to_string(),
                    iteration: 1,
                    reasoning: "inspect".to_string(),
                    code: "lookup()".to_string(),
                    output: vec!["observed".to_string()],
                    tool_calls: vec![lookup],
                    images: Vec::new(),
                    error: None,
                    final_output: None,
                }),
            )));
        state
            .session_graph
            .append_message(text_message("a1", MessageRole::Assistant, "done"));

        let view = SessionReadView::from_exported_state(&state);

        assert_eq!(
            view.rlm_history_len(),
            view.chronological_projection().rlm_history_len()
        );
        assert_eq!(
            view.rlm_history_len(),
            view.shared_chronological_projection().rlm_history_len()
        );
    }
}

/// Context passed to a turn-context transform.
#[derive(Clone)]
pub struct TurnTransformContext {
    pub session_id: String,
    pub state: SessionReadView,
    pub prompt_usage: Option<crate::runtime::PromptUsage>,
    pub max_context_tokens: Option<usize>,
    pub host: Arc<dyn super::HistoryHost>,
}

/// Context passed to a history rewriter.
#[derive(Clone)]
pub struct RewriteContext {
    pub session_id: String,
    pub trigger: RewriteTrigger,
    pub state: SessionReadView,
    pub host: Arc<dyn super::HistoryHost>,
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum HistoryError {
    #[error("history pipeline error: {0}")]
    Pipeline(String),
    #[error("history session error: {0}")]
    Session(String),
}

impl From<PluginError> for HistoryError {
    fn from(value: PluginError) -> Self {
        Self::Session(value.to_string())
    }
}

/// Prepares the ephemeral turn context presented to the model.
#[async_trait::async_trait]
pub trait TurnContextTransform: Send + Sync {
    fn id(&self) -> &'static str;
    async fn transform(
        &self,
        ctx: &TurnTransformContext,
        input: crate::session_model::context::PreparedContext,
    ) -> Result<crate::session_model::context::PreparedContext, HistoryError>;
}

/// Performs a permanent transform on persisted history (compaction,
/// overflow recovery, manual `/compact`, …).
#[async_trait::async_trait]
pub trait HistoryRewriter: Send + Sync {
    fn id(&self) -> &'static str;
    fn accepts(&self, _trigger: &RewriteTrigger) -> bool {
        true
    }
    async fn rewrite(
        &self,
        ctx: &RewriteContext,
        input: HistoryState,
    ) -> Result<HistoryState, HistoryError>;
}
