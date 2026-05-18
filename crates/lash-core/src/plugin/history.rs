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
    turn_index: usize,
    token_usage: crate::TokenUsage,
    last_prompt_usage: Option<crate::runtime::PromptUsage>,
    mode_turn_options: crate::ModeTurnOptions,
}

impl SessionReadMeta {
    fn from_state_ref(state: &SessionStateEnvelope) -> Self {
        Self {
            session_id: state.session_id.clone(),
            policy: state.policy.clone(),
            turn_index: state.turn_index,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
            mode_turn_options: state.mode_turn_options.clone(),
        }
    }

    fn from_persisted_ref(state: &PersistedSessionState) -> Self {
        Self {
            session_id: state.session_id.clone(),
            policy: state.policy.clone(),
            turn_index: state.turn_index,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
            mode_turn_options: state.mode_turn_options.clone(),
        }
    }

    fn with_policy(mut self, policy: SessionPolicy) -> Self {
        self.policy = policy;
        self
    }

    fn with_turn_index(mut self, turn_index: usize) -> Self {
        self.turn_index = turn_index;
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
            turn_index: self.turn_index,
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
        base_graph: Arc<crate::SessionGraph>,
    },
}

impl SessionReadView {
    fn from_graph_message_sequence_meta(
        meta: SessionReadMeta,
        base_graph: Arc<crate::SessionGraph>,
        messages: crate::MessageSequence,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        let base_read_model = base_graph.read_model();
        let active_events = base_read_model.active_events;
        Self(Arc::new(SessionReadState {
            meta,
            graph: SessionReadGraph::Derived {
                cache: OnceLock::new(),
                base_graph,
            },
            read_model: crate::session_graph::SessionReadModel {
                active_events,
                messages: messages.shared(),
                tool_calls,
                prompt_render_cache: Arc::new(crate::BaseRenderCache::new()),
            },
            chronological_projection: OnceLock::new(),
        }))
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
        turn_index: usize,
        mode_turn_options: crate::ModeTurnOptions,
        base_graph: Arc<crate::SessionGraph>,
        messages: crate::MessageSequence,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        Self::from_graph_message_sequence_meta(
            SessionReadMeta::from_persisted_ref(state)
                .with_policy(policy)
                .with_turn_index(turn_index)
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
                let mut graph = base_graph.as_ref().clone();
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

    pub fn message_tree(&self) -> Vec<crate::SessionMessageTreeNode> {
        self.session_graph().message_tree()
    }

    pub fn turn_index(&self) -> usize {
        self.0.meta.turn_index
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

/// Context passed to a turn-context transform.
#[derive(Clone)]
pub struct TurnTransformContext {
    pub session_id: String,
    pub state: SessionReadView,
    pub prompt_usage: Option<crate::runtime::PromptUsage>,
    pub max_context_tokens: Option<usize>,
    pub host: Arc<dyn super::RuntimeSessionHost>,
}

/// Context passed to a history rewriter.
#[derive(Clone)]
pub struct RewriteContext {
    pub session_id: String,
    pub trigger: RewriteTrigger,
    pub state: SessionReadView,
    pub host: Arc<dyn super::RuntimeSessionHost>,
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
/// overflow recovery, manual `/compact`, ...).
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
