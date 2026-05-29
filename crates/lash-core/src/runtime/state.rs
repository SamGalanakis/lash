//! Session state envelopes and persistence helpers.
//!
//! Extracted from `runtime/mod.rs`. `SessionStateEnvelope` and
//! `RuntimeSessionState` keep their original public paths via `pub use`
//! in `mod.rs`; the helper functions are `pub(super)` so sibling runtime
//! modules (`mod.rs`, `session_manager.rs`) can reach them via
//! `super::*`.

use lash_sansio::PromptUsage;

use crate::session_model::{Message, SessionPolicy, TokenUsage, plugin_message_to_message};
use crate::{PersistedTurnState, ToolCallRecord};

use super::usage::TokenLedgerEntry;

/// Serializable session read-model exported to hosts and plugins.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionStateEnvelope {
    pub session_id: String,
    #[serde(default)]
    pub policy: SessionPolicy,
    #[serde(default)]
    pub agent_frames: Vec<crate::AgentFrameRecord>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub current_agent_frame_id: crate::AgentFrameId,
    #[serde(default)]
    pub session_graph: crate::SessionGraph,
    #[serde(default)]
    pub turn_index: usize,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
    #[serde(default)]
    pub protocol_turn_options: crate::ProtocolTurnOptions,
}

impl SessionStateEnvelope {
    pub(crate) fn read_model(&self) -> crate::session_graph::SessionReadModel {
        self.session_graph.read_model_for_agent_frame(
            &self.current_agent_frame_id,
            self.agent_frames
                .iter()
                .find(|frame| frame.frame_id == self.current_agent_frame_id)
                .map(|frame| frame.previous_frame_id.is_none())
                .unwrap_or(true),
        )
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
            agent_frames: default_agent_frames("root", &SessionPolicy::default()),
            current_agent_frame_id: default_agent_frame_id("root"),
            session_graph: crate::SessionGraph::default(),
            turn_index: 0,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
            protocol_turn_options: crate::ProtocolTurnOptions::default(),
        }
    }
}

/// Plain-data view of the **persistable** subset of a
/// [`RuntimeSessionState`]. Hosts and store backends that want to
/// inspect "what would be persisted right now" use this — it never
/// carries runtime-only scratch fields (`head_revision`,
/// `graph_replace_required`, the dirty `*_snapshot` write buffers).
///
/// Build one via [`RuntimeSessionState::persisted_snapshot`]. The
/// shape mirrors the persisted fields of `RuntimeSessionState` 1:1;
/// adding a field here is also a contract change for every backend's
/// on-disk layout, so do it deliberately.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PersistedSessionSnapshot {
    pub session_id: String,
    pub policy: SessionPolicy,
    #[serde(default)]
    pub agent_frames: Vec<crate::AgentFrameRecord>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub current_agent_frame_id: crate::AgentFrameId,
    pub session_graph: crate::SessionGraph,
    pub turn_index: usize,
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
    pub protocol_turn_options: crate::ProtocolTurnOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_state_ref: Option<crate::store::BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_state_generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot_ref: Option<crate::store::BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_state_ref: Option<crate::store::BlobRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<TokenLedgerEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<crate::store::BlobRef>,
}

/// The runtime's view of a session: the persistable snapshot fields
/// **plus** scratch fields the runtime tracks but never persists
/// (head-revision CAS guard, pending dirty-write buffers, replace-graph
/// flag). The non-persisted fields are marked `#[serde(skip)]` so the
/// type still round-trips correctly when used directly as a wire
/// format, but `persisted_snapshot()` is the preferred way to extract
/// "what gets saved" — it returns a separate value type and rules out
/// runtime-only fields by construction.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeSessionState {
    pub session_id: String,
    #[serde(default)]
    pub policy: SessionPolicy,
    #[serde(default)]
    pub agent_frames: Vec<crate::AgentFrameRecord>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub current_agent_frame_id: crate::AgentFrameId,
    #[serde(default)]
    pub session_graph: crate::SessionGraph,
    #[serde(default)]
    pub turn_index: usize,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
    #[serde(default)]
    pub protocol_turn_options: crate::ProtocolTurnOptions,
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

impl From<RuntimeSessionState> for SessionStateEnvelope {
    fn from(state: RuntimeSessionState) -> Self {
        state.into_envelope()
    }
}

impl RuntimeSessionState {
    pub fn from_state(state: SessionStateEnvelope) -> Self {
        let mut state = Self {
            session_id: state.session_id,
            policy: state.policy,
            agent_frames: state.agent_frames,
            current_agent_frame_id: state.current_agent_frame_id,
            session_graph: state.session_graph,
            turn_index: state.turn_index,
            token_usage: state.token_usage,
            last_prompt_usage: state.last_prompt_usage,
            protocol_turn_options: state.protocol_turn_options,
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
        };
        state.ensure_agent_frame_initialized();
        state
    }

    /// Return the persistable subset of this runtime state as a
    /// plain-data [`PersistedSessionSnapshot`]. Runtime-only scratch
    /// fields (`head_revision`, `graph_replace_required`, the dirty
    /// `*_snapshot` write buffers) are dropped on purpose — they
    /// don't belong on the wire.
    pub fn persisted_snapshot(&self) -> PersistedSessionSnapshot {
        PersistedSessionSnapshot {
            session_id: self.session_id.clone(),
            policy: self.policy.clone(),
            agent_frames: self.agent_frames.clone(),
            current_agent_frame_id: self.current_agent_frame_id.clone(),
            session_graph: self.session_graph.clone(),
            turn_index: self.turn_index,
            token_usage: self.token_usage.clone(),
            last_prompt_usage: self.last_prompt_usage.clone(),
            protocol_turn_options: self.protocol_turn_options.clone(),
            tool_state_ref: self.tool_state_ref.clone(),
            tool_state_generation: self.tool_state_generation,
            plugin_snapshot_ref: self.plugin_snapshot_ref.clone(),
            plugin_snapshot_revision: self.plugin_snapshot_revision,
            execution_state_ref: self.execution_state_ref.clone(),
            token_ledger: self.token_ledger.clone(),
            checkpoint_ref: self.checkpoint_ref.clone(),
        }
    }

    pub fn export_state(&self) -> SessionStateEnvelope {
        SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: self.policy.clone(),
            agent_frames: self.agent_frames.clone(),
            current_agent_frame_id: self.current_agent_frame_id.clone(),
            session_graph: self.session_graph.clone(),
            turn_index: self.turn_index,
            token_usage: self.token_usage.clone(),
            last_prompt_usage: self.last_prompt_usage.clone(),
            protocol_turn_options: self.protocol_turn_options.clone(),
        }
    }

    /// Owned conversion into the persistable read-model envelope.
    ///
    /// Like [`Self::export_state`] but consumes `self` and moves the
    /// persistable fields out instead of cloning. Use this when the
    /// `RuntimeSessionState` is no longer needed afterwards.
    pub fn into_envelope(self) -> SessionStateEnvelope {
        SessionStateEnvelope {
            session_id: self.session_id,
            policy: self.policy,
            agent_frames: self.agent_frames,
            current_agent_frame_id: self.current_agent_frame_id,
            session_graph: self.session_graph,
            turn_index: self.turn_index,
            token_usage: self.token_usage,
            last_prompt_usage: self.last_prompt_usage,
            protocol_turn_options: self.protocol_turn_options,
        }
    }

    pub fn apply_exported_state(&mut self, state: &SessionStateEnvelope) {
        self.session_id = state.session_id.clone();
        self.policy = state.policy.clone();
        self.agent_frames = state.agent_frames.clone();
        self.current_agent_frame_id = state.current_agent_frame_id.clone();
        self.ensure_agent_frame_initialized();
        self.session_graph = state.session_graph.clone();
        self.turn_index = state.turn_index;
        self.token_usage = state.token_usage.clone();
        self.last_prompt_usage = state.last_prompt_usage.clone();
        self.protocol_turn_options = state.protocol_turn_options.clone();
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
        self.session_graph.read_model_for_agent_frame(
            &self.current_agent_frame_id,
            self.current_agent_frame_is_initial(),
        )
    }

    pub fn replace_active_read_state(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        self.session_graph
            .replace_active_read_state_for_agent_frame(
                &self.current_agent_frame_id,
                messages,
                tool_calls,
            );
        self.graph_replace_required = false;
    }

    pub fn replace_active_tool_calls(&mut self, tool_calls: &[ToolCallRecord]) {
        let messages = self.read_model().messages;
        self.session_graph
            .replace_active_read_state_for_agent_frame(
                &self.current_agent_frame_id,
                messages.as_slice(),
                tool_calls,
            );
        self.graph_replace_required = false;
    }

    pub fn append_active_read_delta(
        &mut self,
        messages: &[Message],
        tool_calls: &[ToolCallRecord],
    ) {
        self.session_graph.append_active_read_delta_for_agent_frame(
            &self.current_agent_frame_id,
            messages,
            tool_calls,
        );
    }

    pub fn append_active_conversation_messages(&mut self, messages: &[Message]) {
        self.session_graph
            .append_active_conversation_messages_for_agent_frame(
                &self.current_agent_frame_id,
                messages,
            );
    }

    pub fn read_view(&self) -> crate::SessionReadView {
        crate::SessionReadView::from_persisted_state(self)
    }

    pub fn session_graph(&self) -> &crate::SessionGraph {
        &self.session_graph
    }

    pub fn policy(&self) -> &SessionPolicy {
        self.effective_policy()
    }

    pub fn turn_state(&self) -> PersistedTurnState {
        PersistedTurnState {
            turn_index: self.turn_index,
            token_usage: self.token_usage.clone(),
            last_prompt_usage: self.last_prompt_usage.clone(),
            protocol_turn_options: self.protocol_turn_options.clone(),
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
        let execution_state_ref = self.execution_state_ref.clone();
        if let Some(frame) = self.current_agent_frame_mut() {
            frame.execution_state_ref = execution_state_ref;
            frame.execution_state_snapshot = None;
        }
        self.graph_replace_required = false;
        self.tool_state_snapshot = None;
        self.plugin_snapshot = None;
        self.execution_state_snapshot = None;
        if let Some(frame) = self.current_agent_frame_mut() {
            frame.execution_state_snapshot = None;
        }
    }

    pub fn discard_runtime_snapshots(&mut self) {
        self.tool_state_snapshot = None;
        self.plugin_snapshot = None;
        self.execution_state_snapshot = None;
        if let Some(frame) = self.current_agent_frame_mut() {
            frame.execution_state_snapshot = None;
        }
    }

    pub fn set_execution_state_snapshot(&mut self, execution_state_snapshot: Option<Vec<u8>>) {
        if execution_state_snapshot.is_none() {
            self.execution_state_ref = None;
        }
        self.execution_state_snapshot = execution_state_snapshot.clone();
        if let Some(frame) = self.current_agent_frame_mut() {
            if execution_state_snapshot.is_none() {
                frame.execution_state_ref = None;
            }
            frame.execution_state_snapshot = execution_state_snapshot;
        }
    }

    pub fn execution_state_snapshot(&self) -> Option<&[u8]> {
        self.current_agent_frame()
            .and_then(|frame| frame.execution_state_snapshot.as_deref())
            .or(self.execution_state_snapshot.as_deref())
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

impl RuntimeSessionState {
    pub fn current_agent_frame(&self) -> Option<&crate::AgentFrameRecord> {
        self.agent_frames
            .iter()
            .find(|frame| frame.frame_id == self.current_agent_frame_id)
    }

    pub fn current_agent_frame_mut(&mut self) -> Option<&mut crate::AgentFrameRecord> {
        let current_agent_frame_id = self.current_agent_frame_id.clone();
        self.agent_frames
            .iter_mut()
            .find(|frame| frame.frame_id == current_agent_frame_id)
    }

    pub fn effective_policy(&self) -> &SessionPolicy {
        self.current_agent_frame()
            .map(|frame| &frame.assignment.policy)
            .unwrap_or(&self.policy)
    }

    pub fn effective_protocol_turn_options(&self) -> &crate::ProtocolTurnOptions {
        self.current_agent_frame()
            .map(|frame| &frame.protocol_turn_options)
            .unwrap_or(&self.protocol_turn_options)
    }

    pub fn ensure_agent_frame_initialized(&mut self) {
        if self.current_agent_frame_id.is_empty() {
            self.current_agent_frame_id = default_agent_frame_id(&self.session_id);
        }
        if self
            .agent_frames
            .iter()
            .any(|frame| frame.frame_id == self.current_agent_frame_id)
        {
            return;
        }
        let mut frame = default_agent_frame(&self.session_id, &self.policy);
        frame.frame_id = self.current_agent_frame_id.clone();
        frame.protocol_turn_options = self.protocol_turn_options.clone();
        frame.execution_state_ref = self.execution_state_ref.clone();
        frame.execution_state_snapshot = self.execution_state_snapshot.clone();
        self.agent_frames.push(frame);
    }

    pub fn reset_initial_agent_frame(
        &mut self,
        assignment: crate::AgentFrameAssignment,
        protocol_turn_options: crate::ProtocolTurnOptions,
    ) {
        let frame_id = default_agent_frame_id(&self.session_id);
        self.policy = assignment.policy.clone();
        self.protocol_turn_options = protocol_turn_options.clone();
        self.current_agent_frame_id = frame_id.clone();
        self.agent_frames = vec![crate::AgentFrameRecord::new(
            frame_id,
            self.session_id.clone(),
            None,
            crate::AgentFrameReason::Initial,
            None,
            assignment,
            protocol_turn_options,
        )];
    }

    pub fn append_agent_frame(&mut self, mut frame: crate::AgentFrameRecord) {
        let previous_frame_id = self.current_agent_frame_id.clone();
        for existing in &mut self.agent_frames {
            if existing.frame_id == previous_frame_id {
                existing.status = crate::AgentFrameStatus::Superseded;
            }
        }
        if frame.previous_frame_id.is_none() && !previous_frame_id.is_empty() {
            frame.previous_frame_id = Some(previous_frame_id);
        }
        frame.status = crate::AgentFrameStatus::Active;
        self.policy = frame.assignment.policy.clone();
        self.protocol_turn_options = frame.protocol_turn_options.clone();
        self.current_agent_frame_id = frame.frame_id.clone();
        self.execution_state_ref = frame.execution_state_ref.clone();
        self.execution_state_snapshot = frame.execution_state_snapshot.clone();
        self.agent_frames.push(frame);
    }

    fn current_agent_frame_is_initial(&self) -> bool {
        self.current_agent_frame()
            .map(|frame| frame.previous_frame_id.is_none())
            .unwrap_or(true)
    }
}

impl Default for RuntimeSessionState {
    fn default() -> Self {
        Self {
            session_id: "root".to_string(),
            policy: SessionPolicy::default(),
            agent_frames: default_agent_frames("root", &SessionPolicy::default()),
            current_agent_frame_id: default_agent_frame_id("root"),
            session_graph: crate::SessionGraph::default(),
            turn_index: 0,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
            protocol_turn_options: crate::ProtocolTurnOptions::default(),
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
    policy.model = config.model.clone();
}

pub(super) fn apply_session_checkpoint(
    state: &mut RuntimeSessionState,
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
        state.ensure_agent_frame_initialized();
        return;
    };
    state.turn_index = checkpoint.turn_state.turn_index;
    state.token_usage = checkpoint.turn_state.token_usage;
    state.last_prompt_usage = checkpoint.turn_state.last_prompt_usage;
    state.protocol_turn_options = checkpoint.turn_state.protocol_turn_options;
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
    state.ensure_agent_frame_initialized();
    if let Some(frame) = state.current_agent_frame_mut() {
        frame.execution_state_ref = checkpoint.execution_state_ref.clone();
        frame.execution_state_snapshot = checkpoint.execution_state;
    }
}

pub(super) fn apply_session_head(
    state: &mut RuntimeSessionState,
    head: &crate::store::SessionHead,
) {
    state.session_graph = head.graph.clone();
    state.agent_frames = head.agent_frames.clone();
    state.current_agent_frame_id = head.current_agent_frame_id.clone();
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
    state.ensure_agent_frame_initialized();
    state.head_revision = Some(head.head_revision);
    state.graph_replace_required = false;
    apply_persisted_session_config(&mut state.policy, &head.config);
}

pub(super) fn append_session_nodes_to_state(
    state: &mut RuntimeSessionState,
    nodes: &[crate::SessionAppendNode],
) -> Vec<String> {
    let drafts = nodes
        .iter()
        .map(session_append_node_draft)
        .collect::<Vec<_>>();
    state.ensure_agent_frame_initialized();
    let node_ids = state
        .session_graph
        .append_node_drafts_for_agent_frame(&state.current_agent_frame_id, drafts);
    normalize_session_graph(state);
    node_ids
}

fn session_append_node_draft(
    node: &crate::SessionAppendNode,
) -> crate::session_graph::SessionNodeDraft {
    match node {
        crate::SessionAppendNode::Message { message, caused_by } => {
            crate::session_graph::SessionNodeDraft::message(plugin_message_to_message(message))
                .with_caused_by(caused_by.clone())
        }
        crate::SessionAppendNode::ProtocolEvent { event, caused_by } => {
            crate::session_graph::SessionNodeDraft::protocol_event(event.clone())
                .with_caused_by(caused_by.clone())
        }
        crate::SessionAppendNode::Plugin {
            plugin_type,
            body,
            caused_by,
        } => crate::session_graph::SessionNodeDraft::plugin(plugin_type.clone(), body.clone())
            .with_caused_by(caused_by.clone()),
    }
}

fn default_agent_frame_id(session_id: &str) -> crate::AgentFrameId {
    format!("{session_id}:frame:initial")
}

fn default_agent_frames(session_id: &str, policy: &SessionPolicy) -> Vec<crate::AgentFrameRecord> {
    vec![default_agent_frame(session_id, policy)]
}

fn default_agent_frame(session_id: &str, policy: &SessionPolicy) -> crate::AgentFrameRecord {
    crate::AgentFrameRecord::new(
        default_agent_frame_id(session_id),
        session_id.to_string(),
        None,
        crate::AgentFrameReason::Initial,
        None,
        crate::AgentFrameAssignment::from_policy(policy.clone()),
        crate::ProtocolTurnOptions::default(),
    )
}

/// Heal any graph corruption (orphaned leaf) on load.
///
/// Must run BEFORE any residency-based trim (phase-9 feature) because
/// healing's fallback search relies on having the full node set in RAM.
/// Under `Residency::ActivePathOnly`, the runtime loads only the active
/// path; if the leaf doesn't resolve against that reduced set, the
/// caller falls back to a full `load_session_graph()` + `normalize` +
/// trim.
pub(super) fn normalize_session_graph(state: &mut RuntimeSessionState) {
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
    state: &mut RuntimeSessionState,
    residency: crate::Residency,
) {
    match residency {
        crate::Residency::KeepAll => {}
        crate::Residency::ActivePathOnly => {
            state.session_graph = state.session_graph.fork_current_path();
        }
    }
}
