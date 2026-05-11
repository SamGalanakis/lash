//! Session state envelopes and persistence helpers.
//!
//! Extracted from `runtime/mod.rs`. `SessionStateEnvelope` and
//! `PersistedSessionState` keep their original public paths via `pub use`
//! in `mod.rs`; the helper functions are `pub(super)` so sibling runtime
//! modules (`mod.rs`, `session_manager.rs`) can reach them via
//! `super::*`.

use lash_sansio::PromptUsage;

use crate::session_model::{
    Message, SessionEventRecord, SessionPolicy, TokenUsage, plugin_message_to_message,
};
use crate::{PersistedTurnState, ToolCallRecord};

use super::usage::TokenLedgerEntry;

/// Serializable session read-model exported to hosts and plugins.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionStateEnvelope {
    pub session_id: String,
    #[serde(default)]
    pub policy: SessionPolicy,
    #[serde(default)]
    pub session_graph: crate::SessionGraph,
    #[serde(default)]
    pub turn_index: usize,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
    #[serde(default)]
    pub mode_turn_options: crate::ModeTurnOptions,
}

impl SessionStateEnvelope {
    pub(crate) fn read_model(&self) -> crate::session_graph::SessionReadModel {
        self.session_graph.read_model()
    }

    pub fn replace_active_read_state(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        self.session_graph
            .replace_active_read_state(messages, tool_calls);
    }

    pub fn replace_active_tool_calls(&mut self, tool_calls: &[ToolCallRecord]) {
        self.session_graph.replace_active_tool_calls(tool_calls);
    }

    pub fn append_active_read_delta(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        self.session_graph
            .append_active_read_delta(messages, tool_calls);
    }

    pub fn read_view(&self) -> crate::SessionReadView {
        crate::SessionReadView::from_exported_state(self)
    }
}

impl Default for SessionStateEnvelope {
    fn default() -> Self {
        Self {
            session_id: "root".to_string(),
            policy: SessionPolicy::default(),
            session_graph: crate::SessionGraph::default(),
            turn_index: 0,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
            mode_turn_options: crate::ModeTurnOptions::default(),
        }
    }
}

/// Serializable persistence snapshot used by stores, resume, and child session snapshots.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PersistedSessionState {
    pub session_id: String,
    #[serde(default)]
    pub policy: SessionPolicy,
    #[serde(default)]
    pub session_graph: crate::SessionGraph,
    #[serde(default)]
    pub turn_index: usize,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
    #[serde(default)]
    pub mode_turn_options: crate::ModeTurnOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_state_ref: Option<crate::store::BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_state_generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_state_snapshot: Option<crate::ToolState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot_ref: Option<crate::store::BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot: Option<crate::PluginSessionSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_state_ref: Option<crate::store::BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_state_snapshot: Option<Vec<u8>>,
    /// Cost-accounting ledger. Every LLM call (parent turns, subagent
    /// children, compaction, observers, background helpers) contributes an
    /// entry keyed by `(source, model)`. Separate from `token_usage`
    /// which tracks context-window accounting only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<TokenLedgerEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<crate::store::BlobRef>,
    /// Store head revision observed by the runtime. Commits use it for
    /// optimistic concurrency; `None` means the runtime is creating the
    /// first persisted head.
    #[serde(skip)]
    pub head_revision: Option<u64>,
    /// Signals that the next commit must write the full graph (a
    /// destructive rewrite happened, e.g. `heal_orphaned_leaf`). Cleared
    /// after the next commit.
    #[serde(skip)]
    pub graph_replace_required: bool,
}

impl PersistedSessionState {
    pub fn from_state(state: SessionStateEnvelope) -> Self {
        Self {
            session_id: state.session_id,
            policy: state.policy,
            session_graph: state.session_graph,
            turn_index: state.turn_index,
            token_usage: state.token_usage,
            last_prompt_usage: state.last_prompt_usage,
            mode_turn_options: state.mode_turn_options,
            tool_state_ref: None,
            tool_state_generation: None,
            tool_state_snapshot: None,
            plugin_snapshot_ref: None,
            plugin_snapshot_revision: None,
            plugin_snapshot: None,
            execution_state_ref: None,
            execution_state_snapshot: None,
            token_ledger: Vec::new(),
            checkpoint_ref: None,
            head_revision: None,
            graph_replace_required: false,
        }
    }

    pub fn export_state(&self) -> SessionStateEnvelope {
        SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: self.policy.clone(),
            session_graph: self.session_graph.clone(),
            turn_index: self.turn_index,
            token_usage: self.token_usage.clone(),
            last_prompt_usage: self.last_prompt_usage.clone(),
            mode_turn_options: self.mode_turn_options.clone(),
        }
    }

    pub fn apply_exported_state(&mut self, state: &SessionStateEnvelope) {
        self.session_id = state.session_id.clone();
        self.policy = state.policy.clone();
        self.session_graph = state.session_graph.clone();
        self.turn_index = state.turn_index;
        self.token_usage = state.token_usage.clone();
        self.last_prompt_usage = state.last_prompt_usage.clone();
        self.mode_turn_options = state.mode_turn_options.clone();
    }

    pub fn stamp_runtime_state(
        &mut self,
        tool_state: Option<&crate::ToolState>,
        plugin_snapshot: Option<&crate::PluginSessionSnapshot>,
    ) {
        self.tool_state_snapshot = tool_state.cloned();
        self.tool_state_generation = tool_state.map(|snapshot| snapshot.generation());
        self.plugin_snapshot = plugin_snapshot.cloned();
    }

    pub fn usage_report(&self) -> super::usage::SessionUsageReport {
        super::usage::SessionUsageReport::from_entries(&self.token_ledger)
    }

    pub(crate) fn read_model(&self) -> crate::session_graph::SessionReadModel {
        self.session_graph.read_model()
    }

    pub fn replace_active_read_state(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        self.session_graph
            .replace_active_read_state(messages, tool_calls);
        self.graph_replace_required = false;
    }

    pub fn replace_active_tool_calls(&mut self, tool_calls: &[ToolCallRecord]) {
        self.session_graph.replace_active_tool_calls(tool_calls);
        self.graph_replace_required = false;
    }

    pub fn append_active_read_delta(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        self.session_graph
            .append_active_read_delta(messages, tool_calls);
    }

    pub fn append_active_conversation_messages(&mut self, messages: &[Message]) {
        self.session_graph
            .append_active_conversation_messages(messages);
    }

    pub fn read_view(&self) -> crate::SessionReadView {
        crate::SessionReadView::from_persisted_state(self)
    }

    pub fn session_graph(&self) -> &crate::SessionGraph {
        &self.session_graph
    }

    pub fn policy(&self) -> &SessionPolicy {
        &self.policy
    }

    pub fn turn_state(&self) -> PersistedTurnState {
        PersistedTurnState {
            turn_index: self.turn_index,
            token_usage: self.token_usage.clone(),
            last_prompt_usage: self.last_prompt_usage.clone(),
            mode_turn_options: self.mode_turn_options.clone(),
        }
    }

    pub fn token_ledger(&self) -> &[TokenLedgerEntry] {
        &self.token_ledger
    }

    pub fn apply_persisted_commit_result(&mut self, result: crate::store::RuntimeCommitResult) {
        self.head_revision = Some(result.head_revision);
        self.checkpoint_ref = Some(result.checkpoint_ref);
        self.tool_state_ref = result.manifest.tool_state_ref;
        if let Some(snapshot) = self.tool_state_snapshot.as_ref() {
            self.tool_state_generation = Some(snapshot.generation());
        } else if self.tool_state_ref.is_none() {
            self.tool_state_generation = None;
        }
        self.plugin_snapshot_ref = result.manifest.plugin_snapshot_ref;
        self.plugin_snapshot_revision = result.manifest.plugin_snapshot_revision;
        self.execution_state_ref = result.manifest.execution_state_ref;
        self.graph_replace_required = false;
        self.tool_state_snapshot = None;
        self.plugin_snapshot = None;
        self.execution_state_snapshot = None;
    }

    pub fn discard_runtime_snapshots(&mut self) {
        self.tool_state_snapshot = None;
        self.plugin_snapshot = None;
        self.execution_state_snapshot = None;
    }

    pub fn set_execution_state_snapshot(&mut self, execution_state_snapshot: Option<Vec<u8>>) {
        if execution_state_snapshot.is_none() {
            self.execution_state_ref = None;
        }
        self.execution_state_snapshot = execution_state_snapshot;
    }

    pub fn execution_state_snapshot(&self) -> Option<&[u8]> {
        self.execution_state_snapshot.as_deref()
    }

    pub fn refresh_plugin_snapshots(&mut self, plugins: &crate::PluginSession) {
        let tool_registry = plugins.tool_registry();
        let generation = tool_registry.generation();
        if self.tool_state_ref.is_none() || self.tool_state_generation != Some(generation) {
            let snapshot = tool_registry.export_state();
            self.tool_state_generation = Some(snapshot.generation());
            self.tool_state_snapshot = Some(snapshot);
        }

        let revision = plugins.snapshot_revision_fingerprint();
        if self.plugin_snapshot_ref.is_none() || self.plugin_snapshot_revision != Some(revision) {
            self.plugin_snapshot = plugins.snapshot().ok();
        }
        self.plugin_snapshot_revision = Some(revision);
    }
}

impl Default for PersistedSessionState {
    fn default() -> Self {
        Self {
            session_id: "root".to_string(),
            policy: SessionPolicy::default(),
            session_graph: crate::SessionGraph::default(),
            turn_index: 0,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
            mode_turn_options: crate::ModeTurnOptions::default(),
            tool_state_ref: None,
            tool_state_generation: None,
            tool_state_snapshot: None,
            plugin_snapshot_ref: None,
            plugin_snapshot_revision: None,
            plugin_snapshot: None,
            execution_state_ref: None,
            execution_state_snapshot: None,
            token_ledger: Vec::new(),
            checkpoint_ref: None,
            head_revision: None,
            graph_replace_required: false,
        }
    }
}

pub(super) fn apply_persisted_session_config(
    policy: &mut SessionPolicy,
    config: &crate::PersistedSessionConfig,
) {
    if !config.configured_model.is_empty() {
        policy.model = config.configured_model.clone();
    }
    if config.context_window > 0 {
        policy.max_context_tokens = Some(config.context_window as usize);
    }
    policy.execution_mode = config.execution_mode.clone();
    policy.standard_context_approach = config.standard_context_approach.clone();
    policy.model_variant = config.model_variant.clone();
}

pub(super) fn apply_session_checkpoint(
    state: &mut PersistedSessionState,
    checkpoint: Option<crate::store::HydratedSessionCheckpoint>,
) {
    let Some(checkpoint) = checkpoint else {
        state.tool_state_ref = None;
        state.tool_state_generation = None;
        state.tool_state_snapshot = None;
        state.plugin_snapshot_ref = None;
        state.plugin_snapshot_revision = None;
        state.plugin_snapshot = None;
        state.execution_state_ref = None;
        state.execution_state_snapshot = None;
        return;
    };
    state.turn_index = checkpoint.turn_state.turn_index;
    state.token_usage = checkpoint.turn_state.token_usage;
    state.last_prompt_usage = checkpoint.turn_state.last_prompt_usage;
    state.mode_turn_options = checkpoint.turn_state.mode_turn_options;
    state.tool_state_ref = checkpoint.tool_state_ref.clone();
    state.tool_state_generation = checkpoint
        .tool_state
        .as_ref()
        .map(|snapshot| snapshot.generation());
    state.tool_state_snapshot = checkpoint.tool_state;
    state.plugin_snapshot_ref = checkpoint.plugin_snapshot_ref.clone();
    state.plugin_snapshot_revision = checkpoint.plugin_snapshot_revision;
    state.plugin_snapshot = checkpoint.plugin_snapshot;
    state.execution_state_ref = checkpoint.execution_state_ref.clone();
    state.execution_state_snapshot = None;
}

pub(super) fn apply_session_head(
    state: &mut PersistedSessionState,
    head: &crate::store::SessionHead,
) {
    state.session_graph = head.graph.clone();
    state.checkpoint_ref = head.checkpoint_ref.clone();
    state.token_ledger = head.token_ledger.clone();
    state.tool_state_ref = None;
    state.tool_state_generation = None;
    state.tool_state_snapshot = None;
    state.plugin_snapshot_ref = None;
    state.plugin_snapshot_revision = None;
    state.plugin_snapshot = None;
    state.execution_state_ref = None;
    state.execution_state_snapshot = None;
    state.head_revision = Some(head.head_revision);
    state.graph_replace_required = false;
    apply_persisted_session_config(&mut state.policy, &head.config);
}

pub(super) fn append_session_nodes_to_state(
    state: &mut PersistedSessionState,
    nodes: &[crate::SessionAppendNode],
) -> Vec<String> {
    let mut node_ids = Vec::with_capacity(nodes.len());
    for node in nodes {
        match node {
            crate::SessionAppendNode::Message { message } => {
                let message = plugin_message_to_message(message);
                node_ids.push(
                    state
                        .session_graph
                        .append_event(SessionEventRecord::Conversation(
                            crate::session_model::ConversationRecord::from_message(message),
                        )),
                );
            }
            crate::SessionAppendNode::Event { event } => {
                node_ids.push(state.session_graph.append_event(event.clone()));
            }
            crate::SessionAppendNode::Plugin { plugin_type, body } => {
                node_ids.push(
                    state
                        .session_graph
                        .append_plugin(plugin_type.clone(), body.clone()),
                );
            }
        }
    }
    normalize_session_graph(state);
    node_ids
}

/// Heal any graph corruption (orphaned leaf) on load.
///
/// Must run BEFORE any residency-based trim (phase-9 feature) because
/// healing's fallback search relies on having the full node set in RAM.
/// Under `Residency::ActivePathOnly`, the runtime loads only the active
/// path; if the leaf doesn't resolve against that reduced set, the
/// caller falls back to a full `load_session_graph()` + `normalize` +
/// trim.
pub(super) fn normalize_session_graph(state: &mut PersistedSessionState) {
    if state.session_graph.heal_orphaned_leaf() {
        state.graph_replace_required = true;
    }
}

/// Trim the resident node set according to `Residency`. Called AFTER
/// `normalize_session_graph` during `from_environment` load. Under
/// `KeepAll` this is a no-op; under `ActivePathOnly` it replaces the
/// resident graph with just the active path. Orphans remain on disk —
/// the host decides whether/when to tombstone + vacuum them via
/// `LashRuntime::orphaned_node_ids` + the store primitives.
///
pub(super) fn apply_residency_on_load(
    state: &mut PersistedSessionState,
    residency: crate::Residency,
) {
    match residency {
        crate::Residency::KeepAll => {}
        crate::Residency::ActivePathOnly => {
            state.session_graph = state.session_graph.fork_current_path();
        }
    }
}
