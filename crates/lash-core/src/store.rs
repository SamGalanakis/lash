use sha2::Digest;

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
    #[error("store backend error: {0}")]
    Backend(String),
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub session_name: String,
    pub created_at: String,
    pub model: String,
    pub cwd: Option<String>,
    pub parent_session_id: Option<String>,
    pub relation: crate::SessionRelation,
}

impl<'de> serde::Deserialize<'de> for SessionMeta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawSessionMeta {
            session_id: String,
            session_name: String,
            created_at: String,
            model: String,
            cwd: Option<String>,
            #[serde(default)]
            parent_session_id: Option<String>,
            #[serde(default)]
            relation: Option<crate::SessionRelation>,
        }

        let raw = RawSessionMeta::deserialize(deserializer)?;
        let relation = raw.relation.unwrap_or_else(|| {
            raw.parent_session_id
                .as_ref()
                .map(|parent_session_id| crate::SessionRelation::Child {
                    parent_session_id: parent_session_id.clone(),
                    originating_tool_call_id: None,
                })
                .unwrap_or_default()
        });
        let parent_session_id = raw
            .parent_session_id
            .or_else(|| relation.parent_session_id().map(ToOwned::to_owned));
        Ok(Self {
            session_id: raw.session_id,
            session_name: raw.session_name,
            created_at: raw.created_at,
            model: raw.model,
            cwd: raw.cwd,
            parent_session_id,
            relation,
        })
    }
}

impl SessionMeta {
    pub fn relation_parent_session_id(&self) -> Option<&str> {
        self.relation
            .parent_session_id()
            .or(self.parent_session_id.as_deref())
    }
}

/// Lightweight session info for the resume picker.
#[derive(Clone, Debug)]
pub struct SessionPickerInfo {
    pub session_id: String,
    pub cwd: Option<String>,
    pub parent_session_id: Option<String>,
    pub first_user_message: String,
    pub user_message_count: usize,
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
    state: &crate::PersistedSessionState,
) -> crate::PersistedSessionConfig {
    crate::PersistedSessionConfig {
        provider_id: state.policy.provider.kind().to_string(),
        configured_model: state.policy.model.clone(),
        context_window: state.policy.max_context_tokens.unwrap_or_default() as u64,
        execution_mode: state.policy.execution_mode.clone(),
        standard_context_approach: state.policy.standard_context_approach.clone(),
        model_variant: state.policy.model_variant.clone(),
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
}

#[derive(Clone, Debug)]
pub struct RuntimeCommitResult {
    pub head_revision: u64,
    pub checkpoint_ref: BlobRef,
    pub manifest: SessionCheckpoint,
}

pub const RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION: u32 = 1;
pub const RUNTIME_TURN_LEASE_SCHEMA_VERSION: u32 = 1;
pub const RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeTurnMachineConfigSnapshot {
    pub execution_mode: crate::ExecutionMode,
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_session_id: Option<String>,
    pub autonomous: bool,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    pub sync_execution_surface: bool,
    pub tool_specs: Vec<crate::llm::types::LlmToolSpec>,
    pub system_prompt: String,
    pub termination: crate::ModeTurnOptions,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeTurnCheckpoint {
    pub schema_version: u32,
    pub session_id: String,
    pub turn_id: String,
    pub turn_index: usize,
    pub mode_iteration: usize,
    pub checkpoint_hash: String,
    pub machine_config: RuntimeTurnMachineConfigSnapshot,
    pub checkpoint: lash_sansio::TurnCheckpoint<crate::HostModeProtocol>,
    pub mode_turn_options: crate::ModeTurnOptions,
    pub turn_prompt_layer: crate::PromptLayer,
    pub provider_id: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    pub updated_at_epoch_ms: u64,
}


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
    checkpoint: &lash_sansio::TurnCheckpoint<crate::HostModeProtocol>,
) -> Result<String, StoreError> {
    let bytes = serde_json::to_vec(checkpoint).map_err(|err| {
        StoreError::Backend(format!("failed to serialize turn checkpoint: {err}"))
    })?;
    Ok(format!("{:x}", sha2::Sha256::digest(&bytes)))
}

fn build_persisted_turn_state(state: &crate::PersistedSessionState) -> crate::PersistedTurnState {
    crate::PersistedTurnState {
        turn_index: state.turn_index,
        token_usage: state.token_usage.clone(),
        last_prompt_usage: state.last_prompt_usage.clone(),
        mode_turn_options: state.mode_turn_options.clone(),
    }
}

fn build_checkpoint_from_persisted_state(
    state: &crate::PersistedSessionState,
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
        state: &crate::PersistedSessionState,
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
        }
    }

    pub(crate) fn persisted_state_with_graph_commit(
        state: &crate::PersistedSessionState,
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
        }
    }

    pub fn clearing_completed_turn(mut self, completed_turn: RuntimeTurnCompletion) -> Self {
        self.completed_turn = Some(completed_turn);
        self
    }
}

fn persisted_session_state_from_head(
    head: SessionHead,
    checkpoint: Option<HydratedSessionCheckpoint>,
) -> crate::PersistedSessionState {
    let mut state = crate::PersistedSessionState {
        session_id: head.session_id,
        policy: crate::SessionPolicy::default(),
        session_graph: head.graph,
        turn_index: 0,
        token_usage: crate::TokenUsage::default(),
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
        token_ledger: head.token_ledger,
        checkpoint_ref: head.checkpoint_ref.clone(),
        head_revision: Some(head.head_revision),
        graph_replace_required: false,
    };
    state.policy.model = head.config.configured_model.clone();
    if head.config.context_window > 0 {
        state.policy.max_context_tokens = Some(head.config.context_window as usize);
    }
    state.policy.execution_mode = head.config.execution_mode;
    state.policy.standard_context_approach = head.config.standard_context_approach.clone();
    state.policy.model_variant = head.config.model_variant.clone();
    if let Some(checkpoint) = checkpoint {
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
/// owns session graph/head commits, durable turn leases, turn checkpoints, and
/// effect-journal rows together. Keep this monolithic until a second real
/// backend proves that splitting store facets removes more complexity than it
/// adds.
#[async_trait::async_trait]
pub trait RuntimePersistence: Send + Sync {
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

fn persisted_session_state_from_read(read: PersistedSessionRead) -> crate::PersistedSessionState {
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
) -> Result<Option<crate::PersistedSessionState>, StoreError> {
    Ok(store
        .load_session(SessionReadScope::FullGraph)
        .await?
        .map(persisted_session_state_from_read))
}

pub async fn load_persisted_session_state_active_path(
    store: &(dyn RuntimePersistence + '_),
    leaf_node_id: Option<String>,
) -> Result<Option<crate::PersistedSessionState>, StoreError> {
    Ok(store
        .load_session(SessionReadScope::ActivePath { leaf_node_id })
        .await?
        .map(persisted_session_state_from_read))
}

pub async fn refresh_persisted_session_state(
    store: &(dyn RuntimePersistence + '_),
    state: &mut crate::PersistedSessionState,
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
