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
        let read_model = state.read_model();
        Self {
            messages: read_model.messages.as_ref().clone(),
            tool_calls: read_model.tool_calls.as_ref().clone(),
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
    read_model: crate::SessionReadModel,
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
            mode_turn_options: crate::ModeTurnOptions::default(),
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
    pub fn new(state: SessionStateEnvelope) -> Self {
        let mut state = state;
        let graph = Arc::new(std::mem::take(&mut state.session_graph));
        let read_model = graph.read_model();
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_owned(state),
            graph: SessionReadGraph::Owned(Arc::clone(&graph)),
            read_model,
        }))
    }

    pub fn from_read_model_state(
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
            },
            read_model: crate::SessionReadModel {
                active_events: Arc::new(Vec::new()),
                messages,
                tool_calls,
                rlm_globals: Arc::new(serde_json::Map::new()),
                prompt_render_cache: Arc::new(crate::BaseRenderCache::new()),
            },
        }))
    }

    pub(crate) fn from_graph_message_sequence(
        state: &SessionStateEnvelope,
        base_graph: Arc<crate::SessionGraph>,
        messages: crate::MessageSequence,
        tool_calls: Arc<Vec<crate::ToolCallRecord>>,
    ) -> Self {
        let base_read_model = base_graph.read_model();
        let active_events = base_read_model.active_events;
        let rlm_globals = base_read_model.rlm_globals;
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_ref(state),
            graph: SessionReadGraph::Derived {
                cache: OnceLock::new(),
                base_graph: Some(base_graph),
            },
            read_model: crate::SessionReadModel {
                active_events,
                messages: messages.shared(),
                tool_calls,
                rlm_globals,
                prompt_render_cache: Arc::new(crate::BaseRenderCache::new()),
            },
        }))
    }

    pub fn from_state(state: &SessionStateEnvelope) -> Self {
        let read_model = state.session_graph.read_model();
        Self(Arc::new(SessionReadState {
            meta: SessionReadMeta::from_state_ref(state),
            graph: SessionReadGraph::Owned(Arc::new(state.session_graph.clone())),
            read_model,
        }))
    }

    pub fn from_persisted_state(state: &PersistedSessionState) -> Self {
        Self::from_state(&state.export_state())
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

    pub fn session_graph(&self) -> &crate::SessionGraph {
        self.graph_arc().as_ref()
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

    pub fn read_model(&self) -> &crate::SessionReadModel {
        &self.0.read_model
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Message, MessageRole, ModeEvent, Part, PartKind, PruneState, SessionEventRecord,
        ToolCallRecord,
    };
    use lash_rlm_types::{RlmGlobalsPatchPluginBody, RlmModeEvent};

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

    #[test]
    fn read_view_wraps_read_model_with_session_metadata_and_graph() {
        let mut state = SessionStateEnvelope {
            session_id: "session-read-view-test".to_string(),
            iteration: 7,
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
        state
            .session_graph
            .replace_active_read_state(&[user.clone(), assistant.clone()], &[tool_call.clone()]);
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

        let view = SessionReadView::from_state(&state);
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
            view.session_graph().nodes.len(),
            state.session_graph.nodes.len()
        );

        let owned = view.to_owned_state();
        assert_eq!(owned.session_id, state.session_id);
        assert_eq!(owned.iteration, state.iteration);
        assert_eq!(
            owned.session_graph.nodes.len(),
            state.session_graph.nodes.len()
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
