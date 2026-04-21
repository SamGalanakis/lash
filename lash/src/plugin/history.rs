//! History rewriting and turn-context transform plugin contracts.
//!
//! Split out of `plugin/mod.rs` purely for file size. All types keep
//! their original module path via `pub use` in `plugin/mod.rs`.

use std::sync::{Arc, OnceLock};

use crate::SessionPolicy;
use crate::SessionStateEnvelope;
use crate::runtime::PersistedSessionState;

use super::{PluginError, SessionManager};

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
        Self {
            messages: state.project_messages(),
            tool_calls: state.project_tool_calls(),
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
    messages: Arc<Vec<crate::Message>>,
    tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    projected_rlm_globals: Arc<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Clone, Debug)]
struct SessionReadMeta {
    session_id: String,
    policy: SessionPolicy,
    iteration: usize,
    token_usage: crate::TokenUsage,
    last_prompt_usage: Option<crate::runtime::PromptUsage>,
}

impl SessionReadMeta {
    fn from_state_ref(state: &SessionStateEnvelope) -> Self {
        Self {
            session_id: state.session_id.clone(),
            policy: state.policy.clone(),
            iteration: state.iteration,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
        }
    }

    fn from_state_owned(state: SessionStateEnvelope) -> Self {
        Self {
            session_id: state.session_id,
            policy: state.policy,
            iteration: state.iteration,
            token_usage: state.token_usage,
            last_prompt_usage: state.last_prompt_usage,
        }
    }

    fn to_owned_state(&self, session_graph: crate::SessionGraph) -> SessionStateEnvelope {
        SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: self.policy.clone(),
            session_graph,
            iteration: self.iteration,
            token_usage: self.token_usage.clone(),
            last_prompt_usage: self.last_prompt_usage.clone(),
        }
    }
}

#[derive(Debug)]
enum SessionReadGraph {
    Owned(Arc<crate::SessionGraph>),
    Derived {
        cache: OnceLock<Arc<crate::SessionGraph>>,
        base_graph: Option<Arc<crate::SessionGraph>>,
        messages: Arc<Vec<crate::Message>>,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    },
}

impl SessionReadView {
    pub fn new(state: SessionStateEnvelope) -> Self {
        let mut state = state;
        let graph = Arc::new(std::mem::take(&mut state.session_graph));
        let messages = graph.shared_projected_messages();
        let tool_calls = graph.shared_projected_tool_calls();
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_owned(state),
            graph: SessionReadGraph::Owned(Arc::clone(&graph)),
            messages,
            tool_calls,
            projected_rlm_globals: graph.shared_projected_rlm_globals(),
        }))
    }

    pub fn from_projection_state(
        mut state: SessionStateEnvelope,
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
                messages: Arc::clone(&messages),
                tool_calls: Arc::clone(&tool_calls),
            },
            messages,
            tool_calls,
            projected_rlm_globals: Arc::new(serde_json::Map::new()),
        }))
    }

    pub fn from_graph_projection(
        state: &SessionStateEnvelope,
        base_graph: crate::SessionGraph,
        messages: Arc<Vec<crate::Message>>,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_ref(state),
            graph: SessionReadGraph::Derived {
                cache: OnceLock::new(),
                base_graph: Some(Arc::new(base_graph.clone())),
                messages: Arc::clone(&messages),
                tool_calls: Arc::clone(&tool_calls),
            },
            messages,
            tool_calls,
            projected_rlm_globals: base_graph.shared_projected_rlm_globals(),
        }))
    }

    pub fn from_state(state: &SessionStateEnvelope) -> Self {
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_ref(state),
            graph: SessionReadGraph::Owned(Arc::new(state.session_graph.clone())),
            messages: state.session_graph.shared_projected_messages(),
            tool_calls: state.session_graph.shared_projected_tool_calls(),
            projected_rlm_globals: state.session_graph.shared_projected_rlm_globals(),
        }))
    }

    pub fn from_persisted_state(state: &PersistedSessionState) -> Self {
        Self::from_state(&state.export_state())
    }

    fn graph_arc(&self) -> &Arc<crate::SessionGraph> {
        match &self.0.graph {
            SessionReadGraph::Owned(graph) => graph,
            SessionReadGraph::Derived {
                cache,
                base_graph,
                messages,
                tool_calls,
            } => cache.get_or_init(|| {
                let mut graph = base_graph
                    .as_ref()
                    .map(|graph| graph.as_ref().clone())
                    .unwrap_or_default();
                graph.merge_active_projection(messages.as_slice(), tool_calls.as_slice());
                Arc::new(graph)
            }),
        }
    }

    fn messages_arc(&self) -> &Arc<Vec<crate::Message>> {
        &self.0.messages
    }

    fn tool_calls_arc(&self) -> &Arc<Vec<crate::ToolCallRecord>> {
        &self.0.tool_calls
    }

    fn projected_rlm_globals_arc(&self) -> &Arc<serde_json::Map<String, serde_json::Value>> {
        &self.0.projected_rlm_globals
    }

    pub fn session_id(&self) -> &str {
        &self.0.meta.session_id
    }

    pub fn policy(&self) -> &SessionPolicy {
        &self.0.meta.policy
    }

    pub fn session_graph(&self) -> &crate::SessionGraph {
        self.graph_arc().as_ref()
    }

    pub fn messages(&self) -> &[crate::Message] {
        self.messages_arc().as_slice()
    }

    pub fn tool_calls(&self) -> &[crate::ToolCallRecord] {
        self.tool_calls_arc().as_slice()
    }

    pub fn projected_rlm_globals(&self) -> &serde_json::Map<String, serde_json::Value> {
        self.projected_rlm_globals_arc().as_ref()
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
    pub host: Arc<dyn SessionManager>,
}

/// Context passed to a history rewriter.
#[derive(Clone)]
pub struct RewriteContext {
    pub session_id: String,
    pub trigger: RewriteTrigger,
    pub state: SessionReadView,
    pub host: Arc<dyn SessionManager>,
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
