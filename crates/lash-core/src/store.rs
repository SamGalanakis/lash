fn default_root_session_id() -> String {
    "root".to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(
        "store is already bound to session `{bound_session_id}` and cannot be reused for `{attempted_session_id}`"
    )]
    SessionBindingMismatch {
        bound_session_id: String,
        attempted_session_id: String,
    },
    #[error("store does not support read scope {0:?}")]
    UnsupportedReadScope(SessionReadScope),
    #[error("store head revision conflict: expected {expected:?}, actual {actual}")]
    HeadRevisionConflict { expected: Option<u64>, actual: u64 },
    #[error(
        "turn `{turn_id}` for session `{session_id}` is already leased by `{owner_id}` until {expires_at_epoch_ms}"
    )]
    RuntimeTurnLeaseConflict {
        session_id: String,
        turn_id: String,
        owner_id: String,
        expires_at_epoch_ms: u64,
    },
    #[error("runtime turn lease for `{session_id}`/`{turn_id}` is missing or expired")]
    RuntimeTurnLeaseExpired { session_id: String, turn_id: String },
    #[error("runtime effect journal hash mismatch for idempotency key `{idempotency_key}`")]
    RuntimeEffectJournalHashMismatch { idempotency_key: String },
    #[error("runtime turn checkpoint hash mismatch for `{session_id}`/`{turn_id}`")]
    RuntimeTurnCheckpointHashMismatch { session_id: String, turn_id: String },
    #[error(
        "{record_kind} schema_version {actual} is not supported by this binary (expected {expected})"
    )]
    UnsupportedRecordSchemaVersion {
        record_kind: &'static str,
        actual: u32,
        expected: u32,
    },
    #[error("store backend error: {0}")]
    Backend(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub session_name: String,
    pub created_at: String,
    pub model: String,
    pub cwd: Option<String>,
    pub relation: crate::SessionRelation,
}

impl SessionMeta {
    /// Returns the parent session id, if any, derived from the canonical
    /// [`SessionRelation`] field.
    pub fn parent_session_id(&self) -> Option<&str> {
        self.relation.parent_session_id()
    }
}

/// Lightweight session info for the resume picker.
#[derive(Clone, Debug)]
pub struct SessionPickerInfo {
    pub session_id: String,
    pub cwd: Option<String>,
    pub relation: crate::SessionRelation,
    pub first_user_message: String,
    pub user_message_count: usize,
}

impl SessionPickerInfo {
    pub fn parent_session_id(&self) -> Option<&str> {
        self.relation.parent_session_id()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct BlobRef(pub String);

impl BlobRef {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BlobRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for BlobRef {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GcReport {
    pub root_count: usize,
    pub retained_blob_count: usize,
    pub deleted_blob_count: usize,
}

/// Result of a `RuntimePersistence::vacuum()` call.
/// `removed_node_count` counts the tombstoned graph-node rows that were
/// physically deleted from the store. Returned so hosts can emit metrics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VacuumReport {
    pub removed_node_count: usize,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionCheckpoint {
    #[serde(default)]
    pub turn_state: crate::PersistedTurnState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_state_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_state_ref: Option<BlobRef>,
}

#[derive(Clone, Debug, Default)]
pub struct HydratedSessionCheckpoint {
    pub turn_state: crate::PersistedTurnState,
    pub tool_state_ref: Option<BlobRef>,
    pub tool_state: Option<crate::ToolState>,
    pub plugin_snapshot_ref: Option<BlobRef>,
    pub plugin_snapshot: Option<crate::PluginSessionSnapshot>,
    pub plugin_snapshot_revision: Option<u64>,
    pub execution_state_ref: Option<BlobRef>,
    pub execution_state: Option<Vec<u8>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionHead {
    #[serde(default = "default_root_session_id")]
    pub session_id: String,
    #[serde(default)]
    pub head_revision: u64,
    pub graph: crate::SessionGraph,
    pub config: crate::PersistedSessionConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<crate::TokenLedgerEntry>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionHeadMeta {
    #[serde(default = "default_root_session_id")]
    pub session_id: String,
    #[serde(default)]
    pub head_revision: u64,
    pub config: crate::PersistedSessionConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_ref: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_node_id: Option<String>,
    #[serde(default)]
    pub graph_node_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<crate::TokenLedgerEntry>,
}

fn persisted_session_config_from_state(
    state: &crate::RuntimeSessionState,
) -> crate::PersistedSessionConfig {
    crate::PersistedSessionConfig {
        provider_id: state.policy.provider.kind().to_string(),
        model: state.policy.model.clone(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionReadScope {
    FullGraph,
    ActivePath { leaf_node_id: Option<String> },
}

#[derive(Clone, Debug)]
pub struct PersistedSessionRead {
    pub session_id: String,
    pub head_revision: u64,
    pub config: crate::PersistedSessionConfig,
    pub graph: crate::SessionGraph,
    pub checkpoint_ref: Option<BlobRef>,
    pub checkpoint: Option<HydratedSessionCheckpoint>,
    pub token_ledger: Vec<crate::TokenLedgerEntry>,
}

#[derive(Clone, Debug)]
pub enum GraphCommitDelta {
    Unchanged {
        leaf_node_id: Option<String>,
    },
    Append {
        nodes: Vec<crate::SessionNodeRecord>,
        leaf_node_id: Option<String>,
    },
    ReplaceFull(crate::SessionGraph),
}

impl GraphCommitDelta {
    pub fn leaf_node_id(&self) -> Option<&String> {
        match self {
            Self::Unchanged { leaf_node_id } | Self::Append { leaf_node_id, .. } => {
                leaf_node_id.as_ref()
            }
            Self::ReplaceFull(graph) => graph.leaf_node_id.as_ref(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeCommit {
    pub session_id: String,
    pub expected_head_revision: Option<u64>,
    pub config: crate::PersistedSessionConfig,
    pub graph: GraphCommitDelta,
    pub checkpoint: HydratedSessionCheckpoint,
    pub usage_deltas: Vec<crate::TokenLedgerEntry>,
    pub completed_turn: Option<RuntimeTurnCompletion>,
    /// Attachment ids whose bytes are referenced by this commit and
    /// should be stamped `committed` in the write-ahead manifest as
    /// part of the same SQL transaction. The backend marks each id
    /// committed via [`AttachmentManifest::commit_refs`] before the
    /// commit returns success. Hosts populate this from the
    /// attachments emitted by tool calls and inline LLM-request
    /// attachments produced during the turn.
    pub committed_attachment_ids: Vec<crate::AttachmentId>,
}

#[derive(Clone, Debug)]
pub struct RuntimeCommitResult {
    pub head_revision: u64,
    pub checkpoint_ref: BlobRef,
    pub manifest: SessionCheckpoint,
}

// =============================================================================
// Attachment write-ahead manifest
// =============================================================================

/// A pending attachment write recorded *before* the bytes hit the
/// [`AttachmentStore`](crate::AttachmentStore) backend.
///
/// The runtime calls [`AttachmentManifest::record_intent`] from the
/// [`SessionScopedAttachmentStore`](crate::SessionScopedAttachmentStore)
/// wrapper before each `put`, so the manifest is a durable record that
/// "some bytes are about to land at this URI." When the turn that
/// references the attachment commits successfully via
/// [`RuntimePersistence::commit_runtime_state`], the same transaction
/// stamps `committed_at_epoch_ms`. Periodic GC sweeps manifest rows
/// whose intent has aged past a host-chosen threshold without ever
/// being committed and deletes the corresponding bytes — that's how we
/// reconcile orphaned files left behind by crashes between `put` and
/// the next turn commit.
#[derive(Clone, Debug)]
pub struct AttachmentIntent {
    pub attachment_id: crate::AttachmentId,
    pub session_id: String,
    /// Canonical URI for the attachment payload in the backing store.
    /// For file-backed stores this is the absolute on-disk path; for
    /// blob-backed stores it can be any stable identifier the host
    /// uses to clean the payload up.
    pub canonical_uri: String,
    pub intent_at_epoch_ms: u64,
}

#[derive(Clone, Debug)]
pub struct AttachmentManifestEntry {
    pub attachment_id: crate::AttachmentId,
    pub session_id: String,
    pub canonical_uri: String,
    pub intent_at_epoch_ms: u64,
    pub committed_at_epoch_ms: Option<u64>,
}

/// Trait alias for the synchronous attachment-manifest surface on
/// [`RuntimePersistence`]. Used by
/// [`SessionScopedAttachmentStore`](crate::SessionScopedAttachmentStore)
/// to record intent rows before `put` and by GC sweeps to reconcile
/// orphans. See the [`AttachmentIntent`] doc comment for the full
/// crash-safety story.
///
/// Backends with no attachment story (in-memory tests, mock stores)
/// inherit the default no-op impls on [`RuntimePersistence`] and
/// participate transparently — `record_intent` is a no-op, the
/// scoped wrapper still works, and GC sweeps return empty.
pub trait AttachmentManifest: Send + Sync {
    fn record_intent(&self, intent: AttachmentIntent) -> Result<(), StoreError>;

    /// Mark a set of attachment ids as committed (i.e. now referenced
    /// by a durable session-graph commit). Backends that store
    /// commits and manifest in the same database stamp this inside
    /// the commit transaction; the trait-level method is the
    /// out-of-band entry point for hosts that want to commit an id
    /// outside the normal turn-commit flow.
    fn commit_refs(
        &self,
        session_id: &str,
        attachment_ids: &[crate::AttachmentId],
    ) -> Result<(), StoreError>;

    /// Return manifest entries whose intent has aged past
    /// `older_than_epoch_ms` without ever being committed. Hosts run
    /// this periodically to find orphans left by crashes between
    /// `record_intent` and the next turn commit.
    fn list_uncommitted(
        &self,
        older_than_epoch_ms: u64,
    ) -> Result<Vec<AttachmentManifestEntry>, StoreError>;

    /// Remove a manifest row entirely. Called by the GC coordinator
    /// after the corresponding bytes have been removed from the
    /// backing [`AttachmentStore`](crate::AttachmentStore).
    fn forget(&self, attachment_id: &crate::AttachmentId) -> Result<(), StoreError>;
}

/// Mixin macro for [`RuntimePersistence`] implementors that have no
/// attachment-write story (mock backends, in-memory test stores,
/// runtime-perf harnesses). Pastes no-op impls of every
/// [`AttachmentManifest`] method.
#[macro_export]
macro_rules! impl_noop_attachment_manifest {
    ($ty:ty) => {
        impl $crate::AttachmentManifest for $ty {
            fn record_intent(
                &self,
                _intent: $crate::AttachmentIntent,
            ) -> ::std::result::Result<(), $crate::StoreError> {
                Ok(())
            }

            fn commit_refs(
                &self,
                _session_id: &str,
                _attachment_ids: &[$crate::AttachmentId],
            ) -> ::std::result::Result<(), $crate::StoreError> {
                Ok(())
            }

            fn list_uncommitted(
                &self,
                _older_than_epoch_ms: u64,
            ) -> ::std::result::Result<Vec<$crate::AttachmentManifestEntry>, $crate::StoreError>
            {
                Ok(Vec::new())
            }

            fn forget(
                &self,
                _attachment_id: &$crate::AttachmentId,
            ) -> ::std::result::Result<(), $crate::StoreError> {
                Ok(())
            }
        }
    };
}

/// Wire-format version stamped on every persisted [`RuntimeTurnCheckpoint`].
///
/// Bump when the on-wire shape of `RuntimeTurnCheckpoint` changes in a way that
/// older code cannot safely deserialize (enum-variant renames, field-meaning
/// changes, removed required fields). Bytes carrying older or newer versions
/// are rejected at load via [`ensure_supported_schema_version`] so a binary
/// upgrade against in-flight durable state (Restate / SQLite) fails closed
/// rather than silently misparsing.
pub const RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION: u32 = 1;
/// Wire-format version for [`RuntimeTurnLease`]. See
/// [`RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION`] for upgrade semantics.
pub const RUNTIME_TURN_LEASE_SCHEMA_VERSION: u32 = 1;
/// Wire-format version for [`RuntimeEffectJournalRecord`]. See
/// [`RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION`] for upgrade semantics.
pub const RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION: u32 = 1;

/// Reject a persisted record whose `schema_version` does not match the
/// version this binary supports. Backends call this immediately after
/// deserializing a record from durable storage.
pub fn ensure_supported_schema_version(
    record_kind: &'static str,
    actual: u32,
    expected: u32,
) -> Result<(), StoreError> {
    if actual == expected {
        Ok(())
    } else {
        Err(StoreError::UnsupportedRecordSchemaVersion {
            record_kind,
            actual,
            expected,
        })
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeTurnMachineConfigSnapshot {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_session_id: Option<String>,
    pub autonomous: bool,
    pub model: crate::ModelSpec,
    #[serde(default)]
    pub generation: crate::GenerationOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    pub sync_execution_surface: bool,
    pub tool_specs: Vec<crate::llm::types::LlmToolSpec>,
    pub system_prompt: String,
    pub termination: crate::ProtocolTurnOptions,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeTurnCheckpoint {
    pub schema_version: u32,
    pub session_id: String,
    pub turn_id: String,
    pub turn_index: usize,
    pub protocol_iteration: usize,
    pub checkpoint_hash: String,
    pub machine_config: RuntimeTurnMachineConfigSnapshot,
    pub checkpoint: lash_sansio::TurnCheckpoint<crate::HostTurnProtocol>,
    pub protocol_turn_options: crate::ProtocolTurnOptions,
    pub turn_prompt_layer: crate::PromptLayer,
    pub provider_id: String,
    pub model: crate::ModelSpec,
    pub updated_at_epoch_ms: u64,
}

/// Durable lease over a runtime turn.
///
/// The lease pair `(owner_id, lease_token)` plus `fencing_token` are how lash
/// guarantees that one logical turn is driven by exactly one worker at a
/// time — even after a crash, even across two runtimes that both think
/// they own the session. The local SQLite backend (`lash-sqlite-store`)
/// uses these to serialize concurrent claims on the same `(session_id,
/// turn_id)`; future distributed durable backends (Restate, hosted KV) use
/// the *same* fields to coordinate workers that don't share a file system.
///
/// **This is not single-process theatre.** The owner / fencing-token /
/// lease-token triple is the public contract that lets any future backend
/// detect and reject stale writers. Treat it as load-bearing, not
/// defensive.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeTurnLease {
    pub schema_version: u32,
    pub session_id: String,
    pub turn_id: String,
    pub owner_id: String,
    pub lease_token: String,
    pub fencing_token: u64,
    pub claimed_at_epoch_ms: u64,
    pub expires_at_epoch_ms: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeTurnCompletion {
    pub session_id: String,
    pub turn_id: String,
    pub lease_token: String,
}

impl RuntimeTurnCompletion {
    pub fn from_lease(lease: &RuntimeTurnLease) -> Self {
        Self {
            session_id: lease.session_id.clone(),
            turn_id: lease.turn_id.clone(),
            lease_token: lease.lease_token.clone(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeEffectJournalRecord {
    pub schema_version: u32,
    pub session_id: String,
    pub turn_id: String,
    pub idempotency_key: String,
    pub envelope_hash: String,
    pub effect_kind: crate::RuntimeEffectKind,
    pub outcome: crate::RuntimeEffectOutcome,
    pub created_at_epoch_ms: u64,
}

pub fn runtime_turn_checkpoint_hash(
    checkpoint: &lash_sansio::TurnCheckpoint<crate::HostTurnProtocol>,
) -> Result<String, StoreError> {
    crate::stable_hash::stable_json_sha256_hex(checkpoint)
        .map_err(|err| StoreError::Backend(format!("failed to serialize turn checkpoint: {err}")))
}

fn build_persisted_turn_state(state: &crate::RuntimeSessionState) -> crate::PersistedTurnState {
    crate::PersistedTurnState {
        turn_index: state.turn_index,
        token_usage: state.token_usage.clone(),
        last_prompt_usage: state.last_prompt_usage.clone(),
        protocol_turn_options: state.protocol_turn_options.clone(),
    }
}

fn build_checkpoint_from_persisted_state(
    state: &crate::RuntimeSessionState,
) -> HydratedSessionCheckpoint {
    HydratedSessionCheckpoint {
        turn_state: build_persisted_turn_state(state),
        tool_state_ref: state.tool_state_ref.clone(),
        tool_state: state.tool_state_snapshot.clone(),
        plugin_snapshot_ref: state.plugin_snapshot_ref.clone(),
        plugin_snapshot_revision: state.plugin_snapshot_revision,
        plugin_snapshot: state.plugin_snapshot.clone(),
        execution_state_ref: state.execution_state_ref.clone(),
        execution_state: state.execution_state_snapshot.clone(),
    }
}

impl RuntimeCommit {
    pub fn persisted_state(
        state: &crate::RuntimeSessionState,
        usage_deltas: &[crate::TokenLedgerEntry],
    ) -> Self {
        Self {
            session_id: state.session_id.clone(),
            expected_head_revision: state.head_revision,
            config: persisted_session_config_from_state(state),
            graph: if state.graph_replace_required || state.head_revision.is_none() {
                GraphCommitDelta::ReplaceFull(state.session_graph.clone())
            } else {
                GraphCommitDelta::Unchanged {
                    leaf_node_id: state.session_graph.leaf_node_id.clone(),
                }
            },
            checkpoint: build_checkpoint_from_persisted_state(state),
            usage_deltas: usage_deltas.to_vec(),
            completed_turn: None,
            committed_attachment_ids: Vec::new(),
        }
    }

    pub(crate) fn persisted_state_with_graph_commit(
        state: &crate::RuntimeSessionState,
        graph: GraphCommitDelta,
        usage_deltas: &[crate::TokenLedgerEntry],
    ) -> Self {
        Self {
            session_id: state.session_id.clone(),
            expected_head_revision: state.head_revision,
            config: persisted_session_config_from_state(state),
            graph,
            checkpoint: build_checkpoint_from_persisted_state(state),
            usage_deltas: usage_deltas.to_vec(),
            completed_turn: None,
            committed_attachment_ids: Vec::new(),
        }
    }

    pub fn clearing_completed_turn(mut self, completed_turn: RuntimeTurnCompletion) -> Self {
        self.completed_turn = Some(completed_turn);
        self
    }

    pub fn with_committed_attachments(
        mut self,
        attachment_ids: impl IntoIterator<Item = crate::AttachmentId>,
    ) -> Self {
        self.committed_attachment_ids = attachment_ids.into_iter().collect();
        self
    }
}

fn persisted_session_state_from_head(
    head: SessionHead,
    checkpoint: Option<HydratedSessionCheckpoint>,
) -> crate::RuntimeSessionState {
    let mut state = crate::RuntimeSessionState {
        session_id: head.session_id,
        policy: crate::SessionPolicy::default(),
        session_graph: head.graph,
        turn_index: 0,
        token_usage: crate::TokenUsage::default(),
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
        token_ledger: head.token_ledger,
        checkpoint_ref: head.checkpoint_ref.clone(),
        head_revision: Some(head.head_revision),
        graph_replace_required: false,
    };
    state.policy.model = head.config.model.clone();
    if let Some(checkpoint) = checkpoint {
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
        state.execution_state_snapshot = checkpoint.execution_state;
    }
    state
}

impl Default for SessionHead {
    fn default() -> Self {
        Self {
            session_id: default_root_session_id(),
            head_revision: 0,
            graph: crate::SessionGraph::default(),
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            token_ledger: Vec::new(),
        }
    }
}

impl Default for SessionHeadMeta {
    fn default() -> Self {
        Self {
            session_id: default_root_session_id(),
            head_revision: 0,
            config: crate::PersistedSessionConfig::default(),
            checkpoint_ref: None,
            leaf_node_id: None,
            graph_node_count: 0,
            token_ledger: Vec::new(),
        }
    }
}

/// Exact persistence protocol required by the runtime.
///
/// This is intentionally the runtime's atomic transaction facade: one backend
/// owns session graph/head commits, durable turn leases, turn checkpoints,
/// effect-journal rows, and the attachment write-ahead manifest together.
/// Keep this monolithic until a second real backend proves that splitting
/// store facets removes more complexity than it adds.
///
/// The [`AttachmentManifest`] supertrait is required so the runtime can wrap
/// any persistence backend with a
/// [`SessionScopedAttachmentStore`](crate::SessionScopedAttachmentStore)
/// without dual-trait casting. Backends with no attachment-write story can
/// implement the manifest methods as no-ops via
/// [`NoopAttachmentManifest`]'s blanket helpers.
#[async_trait::async_trait]
pub trait RuntimePersistence: AttachmentManifest + Send + Sync {
    async fn load_session(
        &self,
        scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, StoreError>;

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<crate::SessionNodeRecord>, StoreError>;

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError>;

    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError>;

    async fn renew_runtime_turn_lease(
        &self,
        lease: &RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError>;

    async fn abandon_runtime_turn_lease(&self, lease: &RuntimeTurnLease) -> Result<(), StoreError>;

    async fn save_runtime_turn_checkpoint(
        &self,
        lease: &RuntimeTurnLease,
        checkpoint: RuntimeTurnCheckpoint,
    ) -> Result<(), StoreError>;

    async fn load_runtime_turn_checkpoint(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<RuntimeTurnCheckpoint>, StoreError>;

    async fn save_runtime_effect_outcome(
        &self,
        lease: &RuntimeTurnLease,
        record: RuntimeEffectJournalRecord,
    ) -> Result<(), StoreError>;

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<RuntimeEffectJournalRecord>, StoreError>;

    async fn save_session_meta(&self, meta: SessionMeta) -> Result<(), StoreError>;
    async fn load_session_meta(&self) -> Result<Option<SessionMeta>, StoreError>;

    async fn tombstone_nodes(&self, ids: &[String]) -> Result<(), StoreError>;
    async fn vacuum(&self) -> Result<VacuumReport, StoreError>;
    async fn gc_unreachable(&self) -> Result<GcReport, StoreError>;
}

fn persisted_session_state_from_read(read: PersistedSessionRead) -> crate::RuntimeSessionState {
    persisted_session_state_from_head(
        SessionHead {
            session_id: read.session_id,
            head_revision: read.head_revision,
            graph: read.graph,
            config: read.config,
            checkpoint_ref: read.checkpoint_ref,
            token_ledger: read.token_ledger,
        },
        read.checkpoint,
    )
}

pub async fn load_persisted_session_state(
    store: &(dyn RuntimePersistence + '_),
) -> Result<Option<crate::RuntimeSessionState>, StoreError> {
    Ok(store
        .load_session(SessionReadScope::FullGraph)
        .await?
        .map(persisted_session_state_from_read))
}

pub async fn load_persisted_session_state_active_path(
    store: &(dyn RuntimePersistence + '_),
    leaf_node_id: Option<String>,
) -> Result<Option<crate::RuntimeSessionState>, StoreError> {
    Ok(store
        .load_session(SessionReadScope::ActivePath { leaf_node_id })
        .await?
        .map(persisted_session_state_from_read))
}

pub async fn refresh_persisted_session_state(
    store: &(dyn RuntimePersistence + '_),
    state: &mut crate::RuntimeSessionState,
) -> Result<(), StoreError> {
    if let Some(mut fresh) = load_persisted_session_state(store).await? {
        // The store owns persisted graph/checkpoint/config state, but not
        // live provider credentials or other runtime-only policy fields.
        fresh.policy.provider = state.policy.provider.clone();
        fresh.policy.session_id = state.policy.session_id.clone();
        fresh.policy.max_turns = state.policy.max_turns;
        *state = fresh;
    }
    Ok(())
}
