mod assembly;
mod builder;
mod config_ops;
mod environment;
mod host;
mod io;
mod session_manager;
mod session_ops;
mod state;
#[cfg(test)]
mod tests;
mod turn_driver;
mod usage;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::llm::types::{LlmOutputPart, LlmRequest, LlmResponse, LlmStreamEvent, LlmUsage};
use crate::plugin::{
    CheckpointHookContext, PluginMessage, PrepareTurnRequest, SessionConfigChangedContext,
};
use crate::sansio::{Effect, LlmCallError, Response, TurnMachine};
use crate::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, SessionEvent, SessionPolicy, TokenUsage,
    fresh_message_id, make_error_event, reassign_part_ids, transport_stream_events,
};
use crate::tool_dispatch::{ToolDispatchContext, dispatch_tool_call_with_execution_context};
use crate::{
    CheckpointKind, ExecutionMode, ExternalInvokeError, PersistentRuntimeServices,
    PromptHookContext, RuntimeServices, SandboxMessage, Session, SessionCreateRequest,
    SessionError, SessionHandle, SessionManager, SessionSnapshot, SessionStartPoint,
    ToolCallRecord,
};

use host::*;
use session_manager::*;
use turn_driver::*;

// `PromptUsage` is re-exported below alongside the runtime's own types.
pub use lash_sansio::PromptUsage;

use assembly::{
    LlmDebugText, LlmDebugToolCall, LlmStreamDebugState, LlmStreamEventLog, LlmStreamSummary,
    StandardStreamFallback, StandardStreamState, TurnAssembler,
};
#[cfg(test)]
#[allow(unused_imports)]
use assembly::{classify_output_state, sanitize_assistant_output};
pub use builder::EmbeddedRuntimeBuilder;
pub use environment::{ParkedSession, Residency, RuntimeEnvironment, RuntimeEnvironmentBuilder};
pub use host::{
    BackgroundRuntimeHost, DefaultPathResolver, EmbeddedRuntimeHost, FileLlmCallLogger,
    LlmCallLogger, ManagedRunState, ManagedTaskCancel, ManagedTaskKind, ManagedTaskSpec,
    ManagedTaskStatus, RuntimeCoreConfig, SessionTaskExecutor, TokioSessionTaskExecutor,
};
use io::{normalize_input_items, projection_message_delta_if_base_preserved};
pub use state::{PersistedSessionState, SessionStateEnvelope};
use state::{
    append_session_nodes_to_state, apply_residency_on_load, apply_session_checkpoint,
    apply_session_head, clear_persisted_runtime_caches, load_session_checkpoint,
    normalize_session_graph, persist_session_graph_and_head,
};
pub use usage::{
    SessionUsageReport, TokenLedgerEntry, UsageReportRow, UsageTotals, diff_token_ledger,
    diff_usage_reports,
};
use usage::{merge_ledger_entry, merge_usage_delta_entries, normalize_prompt_usage};

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeTurnPhase {
    ContextTransform,
    BeforeTurnHooks,
    PromptBuild,
    EffectLoop,
    FinalizeTurn,
    PersistTurn,
}

#[doc(hidden)]
pub trait RuntimeTurnPhaseProbe: Send + Sync {
    fn begin(&self, phase: RuntimeTurnPhase);
    fn end(&self, phase: RuntimeTurnPhase);
}

/// Runtime execution mode for a turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RunMode {
    Normal,
}

/// Host-provided per-turn input.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Text { text: String },
    FileRef { path: String },
    DirRef { path: String },
    ImageRef { id: String },
}

/// Host-provided per-turn input.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnInput {
    pub items: Vec<InputItem>,
    #[serde(default)]
    pub image_blobs: HashMap<String, Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_input: Option<crate::UserInputProvenance>,
    #[serde(default)]
    pub mode: Option<RunMode>,
    /// Per-turn override for the session's RLM termination contract.
    /// When `Some`, this turn validates `submit` against the supplied
    /// schema (or, for `ProseWithoutFence`, drops validation entirely)
    /// without mutating the session-scoped default. Used by
    /// `followup_task` to retype a subagent for a single turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rlm_termination_override: Option<crate::RlmTermination>,
}

#[derive(Clone, Debug)]
pub(super) enum NormalizedItem {
    Text(String),
    Image(Vec<u8>),
}

/// Canonical assistant output payload.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssistantOutput {
    pub safe_text: String,
    pub raw_text: String,
    pub state: OutputState,
}

/// Quality and usability of assembled terminal output.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputState {
    Usable,
    EmptyOutput,
    TracebackOnly,
    RecoveredFromError,
}

/// Structured terminal status for a turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Completed,
    Interrupted,
    Failed,
}

/// Canonical reason a turn ended.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DoneReason {
    ModelStop,
    MaxTurns,
    UserAbort,
    ToolFailure,
    RuntimeError,
}

/// RLM code execution output observed during a turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CodeOutputRecord {
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// High-level execution summary shared across execution modes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecutionSummary {
    pub mode: ExecutionMode,
    #[serde(default)]
    pub had_tool_calls: bool,
    #[serde(default)]
    pub had_code_execution: bool,
}

/// Structured issue surfaced during turn execution.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnIssue {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

/// Canonical high-level turn result returned to hosts.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssembledTurn {
    pub state: SessionStateEnvelope,
    pub status: TurnStatus,
    pub assistant_output: AssistantOutput,
    #[serde(default)]
    pub has_plugin_visible_output: bool,
    pub done_reason: DoneReason,
    pub execution: ExecutionSummary,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default)]
    pub errors: Vec<TurnIssue>,
    /// When the session was started in typed RLM termination mode AND
    /// the lashlang program ended with `submit <expr>`, this is the
    /// captured (and schema-validated, if a schema was supplied) value.
    /// `None` for chat-style sessions and for typed sessions that
    /// timed out without finishing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_finish: Option<serde_json::Value>,
}

/// Runtime error for unexpected failures.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeError {
    pub code: String,
    pub message: String,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RuntimeError {}

/// Pluggable path resolver for file and directory references.
pub trait PathResolver: Send + Sync {
    fn resolve(&self, path: &str, expect_file: bool, base_dir: &Path) -> Result<PathBuf, String>;
}

/// Sanitization policy knobs.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SanitizerPolicy {}

/// Termination policy knobs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TerminationPolicy {
    #[serde(default)]
    pub treat_missing_done_as_failure: bool,
}

impl Default for TerminationPolicy {
    fn default() -> Self {
        Self {
            treat_missing_done_as_failure: true,
        }
    }
}

/// Host event sink for low-level streaming runtime events.
/// `SessionEvent` is intentionally mode-specific and should be treated as preview/progress data.
#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: SessionEvent);
}

/// No-op sink useful for callers that only care about final state.
pub struct NoopEventSink;

#[async_trait::async_trait]
impl EventSink for NoopEventSink {
    async fn emit(&self, _event: SessionEvent) {}
}

enum RuntimeStreamEvent {
    Session(SessionEvent),
}

#[derive(Clone)]
pub struct SessionStoreCreateRequest {
    pub session_id: String,
    pub parent_session_id: Option<String>,
    pub policy: SessionPolicy,
}

pub trait SessionStoreFactory: Send + Sync {
    fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn crate::store::RuntimeStore>, String>;
}

fn debug_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?.trim();
        let kb = value.split_whitespace().next()?.parse::<u64>().ok()?;
        Some(kb)
    })
}

/// Generic runtime for CLI or programmatic embedding.
pub struct LashRuntime {
    pub(in crate::runtime) session: Option<Session>,
    pub(in crate::runtime) policy: SessionPolicy,
    pub(in crate::runtime) host: RuntimeHost,
    pub(in crate::runtime) services: RuntimeServices,
    pub(in crate::runtime) state: PersistedSessionState,
    pub(in crate::runtime) runtime_scope_id: Arc<str>,
    pub(in crate::runtime) managed_sessions: Arc<Mutex<HashMap<String, Arc<Mutex<LashRuntime>>>>>,
    pub(in crate::runtime) managed_turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    pub(in crate::runtime) overflow_recovery_attempted: bool,
    /// RLM termination contract for this session.
    pub(in crate::runtime) rlm_termination: crate::RlmTermination,
    /// Session-scoped token cost ledger. Shared by ALL
    /// `RuntimeSessionManager` instances created from this runtime
    /// (both per-turn and async maintenance). Entries accumulate here
    /// and are drained into `state.token_ledger` at turn-commit time.
    pub(in crate::runtime) shared_token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    pub(in crate::runtime) background_sync_needed: Arc<AtomicBool>,
    pub(in crate::runtime) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}

impl LashRuntime {
    pub(super) fn stamp_live_plugin_state(&mut self) {
        if let Some(session) = self.session.as_ref() {
            if let Some(dynamic_tools) = session.plugins().dynamic_tools() {
                let snapshot = dynamic_tools.export_state();
                self.state.dynamic_state_generation = Some(snapshot.base_generation);
                self.state.dynamic_state_snapshot = Some(snapshot);
            } else {
                self.state.dynamic_state_generation = None;
                self.state.dynamic_state_snapshot = None;
            }
            self.state.plugin_snapshot = session.plugins().snapshot().ok();
            self.state.plugin_snapshot_revision =
                Some(session.plugins().snapshot_revision_fingerprint());
        } else {
            self.state.dynamic_state_generation = None;
            self.state.dynamic_state_snapshot = None;
            self.state.plugin_snapshot = None;
            self.state.plugin_snapshot_revision = None;
        }
    }

    fn has_overflow_error(assembled: &AssembledTurn) -> bool {
        assembled.errors.iter().any(|issue| {
            let lower = issue.message.to_lowercase();
            lower.contains("prompt is too long")
                || lower.contains("context_length_exceeded")
                || lower.contains("maximum context length")
                || lower.contains("too many tokens")
                || lower.contains("exceeds the maximum number of tokens")
                || lower.contains("request too large")
        })
    }

    fn max_context_tokens(&self) -> usize {
        self.policy
            .max_context_tokens
            .expect("lash runtime requires explicit max_context_tokens")
    }

    fn active_tool_catalog(&self) -> Vec<serde_json::Value> {
        self.active_tool_catalog_shared().as_ref().clone()
    }

    fn active_tool_catalog_shared(&self) -> Arc<Vec<serde_json::Value>> {
        self.session
            .as_ref()
            .map(|session| {
                session.shared_tool_catalog(&self.state.session_id, self.policy.execution_mode)
            })
            .unwrap_or_else(|| Arc::new(Vec::new()))
    }

    pub fn dynamic_tool_state(&self) -> Result<crate::DynamicStateSnapshot, SessionError> {
        let Some(session) = self.session.as_ref() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let Some(dynamic_tools) = session.plugins().dynamic_tools() else {
            return Err(SessionError::Protocol(
                "dynamic tools are unavailable in this runtime session".to_string(),
            ));
        };
        Ok(dynamic_tools.export_state())
    }

    pub(super) async fn from_host_state(
        policy: SessionPolicy,
        host: RuntimeHost,
        services: RuntimeServices,
        mut state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        if state.session_id.is_empty() {
            state.session_id = "root".to_string();
        }
        // Defaulted state (e.g. `PersistedSessionState::default()` used
        // by fresh-session constructors) carries an unconfigured policy.
        // Fill it in from the caller's policy so tests and hosts that
        // pass a real policy alongside default state don't trip the
        // max_context_tokens guard below.
        if state.policy.provider.kind() == "unconfigured" {
            state.policy = policy.clone();
        }
        normalize_session_graph(&mut state);
        if policy.max_context_tokens.is_none() {
            return Err(SessionError::Protocol(
                "session policy missing max_context_tokens; hosts must supply explicit model metadata"
                    .to_string(),
            ));
        }
        let mut session = Session::new(
            services.clone(),
            &state.session_id,
            state.policy.execution_mode,
        )
        .await?;
        if let Some(dynamic_state) = state.dynamic_state_snapshot.clone()
            && let Some(dynamic_tools) = session.plugins().dynamic_tools()
            && let Err(err) = dynamic_tools.apply_state(dynamic_state)
        {
            tracing::warn!("failed to restore dynamic tool state from checkpoint: {err}");
        }
        if let Some(snapshot) = state.plugin_snapshot.clone() {
            session
                .plugins()
                .restore(&snapshot)
                .map_err(|err| SessionError::Protocol(err.to_string()))?;
        }
        let mode_session = Arc::clone(session.plugins().mode_session());
        let session_id = state.session_id.clone();
        mode_session
            .restore_session(
                crate::plugin::ModeSessionContext::new(&mut session, &session_id),
                &state,
            )
            .await?;
        clear_persisted_runtime_caches(&mut state);
        session
            .plugins()
            .emit_runtime_event(crate::PluginRuntimeEvent::SessionRestored(
                state.read_view(),
            ))
            .await;
        Ok(Self {
            session: Some(session),
            policy,
            host,
            services,
            state,
            runtime_scope_id: Arc::<str>::from(uuid::Uuid::new_v4().to_string()),
            managed_sessions: Arc::new(Mutex::new(HashMap::new())),
            managed_turns: Arc::new(Mutex::new(HashMap::new())),
            overflow_recovery_attempted: false,
            rlm_termination: crate::RlmTermination::default(),
            shared_token_ledger: Arc::new(std::sync::Mutex::new(Vec::new())),
            background_sync_needed: Arc::new(AtomicBool::new(false)),
            turn_phase_probe: None,
        })
    }

    /// Build a runtime for an embedded host with no background worker support.
    pub async fn from_embedded_state(
        policy: SessionPolicy,
        host: EmbeddedRuntimeHost,
        services: RuntimeServices,
        state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        Self::from_host_state(policy, host.into(), services, state).await
    }

    /// Build a runtime for a host that supports background plugin work.
    pub async fn from_background_state(
        policy: SessionPolicy,
        host: BackgroundRuntimeHost,
        services: RuntimeServices,
        state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        Self::from_host_state(policy, host.into(), services, state).await
    }

    /// Build a runtime for an embedded host with persistent store support.
    pub async fn from_persistent_embedded_state(
        policy: SessionPolicy,
        host: EmbeddedRuntimeHost,
        services: PersistentRuntimeServices,
        state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        Self::from_host_state(policy, host.into(), services.into_runtime_services(), state).await
    }

    /// Build a runtime for a background-capable host with persistent store support.
    pub async fn from_persistent_background_state(
        policy: SessionPolicy,
        host: BackgroundRuntimeHost,
        services: PersistentRuntimeServices,
        state: PersistedSessionState,
    ) -> Result<Self, SessionError> {
        Self::from_host_state(policy, host.into(), services.into_runtime_services(), state).await
    }

    /// Embedder-preferred constructor: build a `LashRuntime` from a
    /// shared `RuntimeEnvironment`.
    ///
    /// Everything expensive (plugin factories, HTTP client pool, prompt
    /// template, path resolver) lives on the environment and is
    /// reused across every runtime the embedder builds. This call is
    /// O(plugin-session-registration + state-hydration), not
    /// O(full-infrastructure-init).
    ///
    /// * `env` — the shared environment. `env.plugin_host` must be set.
    /// * `policy` — per-session policy (model, provider, execution mode).
    /// * `state` — persisted session state (empty for a fresh session).
    /// * `store` — per-session store. `None` builds an embedded runtime
    ///   with no persistence; `Some` builds a persistent
    ///   background-capable runtime.
    pub async fn from_environment(
        env: &RuntimeEnvironment,
        policy: SessionPolicy,
        mut state: PersistedSessionState,
        store: Option<Arc<dyn crate::store::RuntimeStore>>,
    ) -> Result<Self, SessionError> {
        // ActivePathOnly without a store is a data-loss footgun: trim
        // drops orphans from RAM with nowhere to reload them from.
        if matches!(env.residency, Residency::ActivePathOnly) && store.is_none() {
            return Err(SessionError::Protocol(
                "Residency::ActivePathOnly requires a persistent store — \
                 without one, trimmed orphans are irrecoverable"
                    .to_string(),
            ));
        }
        // Heal FIRST (against the full resident set), then trim.
        // `heal_orphaned_leaf` is driven by `normalize_session_graph`
        // which runs again inside `from_host_state`. Running it here
        // too lets us trim safely before delegating.
        normalize_session_graph(&mut state);
        apply_residency_on_load(&mut state, env.residency);
        let plugin_host = env.plugin_host.as_ref().ok_or_else(|| {
            SessionError::Protocol(
                "RuntimeEnvironment.plugin_host is required for from_environment".to_string(),
            )
        })?;
        let plugin_session = plugin_host
            .build_session(
                state.session_id.as_str(),
                policy.execution_mode,
                policy.context_approach.clone(),
                state.plugin_snapshot.as_ref(),
            )
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        let core = env.to_runtime_core_config();
        let embedded = EmbeddedRuntimeHost::new(core);
        let runtime = if let Some(store) = store {
            let services = PersistentRuntimeServices::new_with_bridges(
                plugin_session,
                crate::session::TurnInjectionBridge::new(),
                crate::session::TurnInputInjectionBridge::new(),
                store,
            );
            match env.session_task_executor.as_ref() {
                Some(executor) => {
                    let host = BackgroundRuntimeHost::new(embedded, Arc::clone(executor));
                    Self::from_persistent_background_state(policy, host, services, state).await?
                }
                None => {
                    Self::from_persistent_embedded_state(policy, embedded, services, state).await?
                }
            }
        } else {
            let services = RuntimeServices::new(plugin_session);
            match env.session_task_executor.as_ref() {
                Some(executor) => {
                    let host = BackgroundRuntimeHost::new(embedded, Arc::clone(executor));
                    Self::from_background_state(policy, host, services, state).await?
                }
                None => Self::from_embedded_state(policy, embedded, services, state).await?,
            }
        };
        Ok(runtime)
    }

    /// Persist any dirty state and drop the runtime, returning a lightweight
    /// handle the embedder can cache and resume later via
    /// [`LashRuntime::resume`]. This is the webserver-embedder handoff
    /// primitive: the handle holds only the session id, policy, and store
    /// reference — no graph nodes, no plugin session, no HTTP client.
    pub async fn park(mut self) -> Result<ParkedSession, SessionError> {
        let store = self.services.store.clone().ok_or_else(|| {
            SessionError::Protocol(
                "park() requires a persistent runtime (store is not set)".to_string(),
            )
        })?;
        let session_id = self.state.session_id.clone();
        let policy = self.policy.clone();
        // Flush any dirty resident state to the store before dropping.
        persist_session_graph_and_head(store.as_ref(), &mut self.state).await;
        // Drain pending tombstones if any. Under KeepHistory this is a
        // no-op (tombstones never get added). Under DropOrphans,
        // Phase-9's not-yet-wired rewrite path would have populated the
        // set — wired fully in Phase 10's vacuum() design.
        Ok(ParkedSession {
            session_id,
            store,
            policy,
        })
    }

    /// Resume a previously parked session against a shared environment.
    /// Loads only the active-path graph when
    /// `env.residency == ActivePathOnly`; under `KeepAll`
    /// loads the full graph (current behavior).
    pub async fn resume(
        parked: ParkedSession,
        env: &RuntimeEnvironment,
    ) -> Result<Self, SessionError> {
        // Under ActivePathOnly, skip the full-graph load: fetch head
        // metadata + the active-path chain only. SQLite impls can
        // override `load_active_path_graph` with a recursive CTE for a
        // real O(active-path) query; the default still loads the full
        // graph then forks, which is correct but slower on large
        // histories.
        let loaded = match env.residency {
            Residency::KeepAll => parked.store.load_persisted_session_state().await,
            Residency::ActivePathOnly => {
                parked
                    .store
                    .load_persisted_session_state_active_path()
                    .await
            }
        };
        let state = loaded.unwrap_or_else(|| PersistedSessionState {
            session_id: parked.session_id.clone(),
            policy: parked.policy.clone(),
            ..PersistedSessionState::default()
        });
        Self::from_environment(env, parked.policy, state, Some(parked.store)).await
    }

    /// Opt-in async read for historic (non-active-path) nodes under
    /// `Residency::ActivePathOnly`. Plugins that walk the full graph
    /// call this instead of `session_graph().find_node()` so missing
    /// nodes surface as `Ok(None)` rather than silently missing.
    pub async fn get_historic_node(
        &self,
        node_id: &str,
    ) -> Result<Option<crate::SessionNodeRecord>, SessionError> {
        if let Some(node) = self.state.session_graph.find_node(node_id) {
            return Ok(Some(node.clone()));
        }
        let store = self.services.store.clone().ok_or_else(|| {
            SessionError::Protocol("get_historic_node() requires a persistent runtime".to_string())
        })?;
        Ok(store.get_node(node_id).await)
    }

    /// Store-resident node IDs that are NOT reachable from the current
    /// leaf — i.e. orphans eligible for tombstoning. lash owns RAM; the
    /// host owns disk lifecycle, so this is a primitive the host calls
    /// on its own schedule (e.g. every N turns, or off-peak).
    ///
    /// Typical autonomous-agent loop:
    ///
    /// ```ignore
    /// let orphans = runtime.orphaned_node_ids().await?;
    /// if !orphans.is_empty() {
    ///     store.tombstone_nodes(&orphans).await;
    /// }
    /// // And less often:
    /// store.vacuum().await;
    /// ```
    pub async fn orphaned_node_ids(&self) -> Result<Vec<String>, SessionError> {
        let store = self.services.store.clone().ok_or_else(|| {
            SessionError::Protocol("orphaned_node_ids() requires a persistent runtime".to_string())
        })?;
        let active: std::collections::HashSet<&str> = self
            .state
            .session_graph
            .active_path_nodes()
            .iter()
            .map(|node| node.node_id.as_str())
            .collect();
        let full = store.load_session_graph().await;
        Ok(full
            .nodes
            .iter()
            .filter(|node| !active.contains(node.node_id.as_str()))
            .map(|node| node.node_id.clone())
            .collect())
    }

    #[doc(hidden)]
    pub fn set_turn_phase_probe(&mut self, probe: Arc<dyn RuntimeTurnPhaseProbe>) {
        self.turn_phase_probe = Some(probe);
    }

    fn mark_phase_begin(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.begin(phase);
        }
    }

    fn mark_phase_end(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.end(phase);
        }
    }

    /// Override the RLM termination contract for this session. Defaults
    /// to `ProseWithoutFence` (today's chat-style behavior). Typed
    /// subagent sessions call this with `Finish { schema }` to require
    /// typed termination.
    pub(crate) fn set_repl_termination(&mut self, termination: crate::RlmTermination) {
        self.rlm_termination = termination;
    }

    /// Export current session state for inspection/UI purposes.
    /// This keeps persistence-heavy snapshots untouched; callers that need a
    /// fully persisted view should use `export_persisted_state`.
    pub fn export_state(&self) -> SessionStateEnvelope {
        self.state.export_state()
    }

    /// Export the narrow persistence snapshot used by stores and resume logic.
    pub fn export_persistence_state(&self) -> PersistedSessionState {
        self.state.clone()
    }

    pub fn apply_persistence_state(&mut self, state: PersistedSessionState) {
        self.set_persisted_state(state);
    }

    pub(crate) fn export_graph_first_state(&self) -> PersistedSessionState {
        self.state.clone()
    }

    /// Export a persistence-ready state envelope with dynamic/plugin snapshots
    /// refreshed from the live session.
    pub fn export_persisted_state(&self) -> PersistedSessionState {
        let mut state = self.state.clone();
        if let Some(session) = self.session.as_ref() {
            if let Some(dynamic_tools) = session.plugins().dynamic_tools() {
                let snapshot = dynamic_tools.export_state();
                state.dynamic_state_generation = Some(snapshot.base_generation);
                state.dynamic_state_snapshot = Some(snapshot);
            } else {
                state.dynamic_state_generation = None;
                state.dynamic_state_snapshot = None;
            }
            state.plugin_snapshot = session.plugins().snapshot().ok();
            state.plugin_snapshot_revision =
                Some(session.plugins().snapshot_revision_fingerprint());
        }
        normalize_session_graph(&mut state);
        state
    }

    pub fn usage_report(&self) -> SessionUsageReport {
        let mut entries = self.state.token_ledger.clone();
        let drained = self.shared_token_ledger.lock().expect("token ledger lock");
        for entry in drained.iter().cloned() {
            merge_ledger_entry(&mut entries, entry);
        }
        SessionUsageReport::from_entries(&entries)
    }

    pub async fn await_background_work(&mut self) -> Result<(), SessionError> {
        let manager = self
            .runtime_session_manager()
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        manager
            .await_hidden_tasks(&self.state.session_id)
            .await
            .map_err(|err| SessionError::Protocol(format!("session task failed: {err}")))?;
        if self.background_sync_needed.swap(false, Ordering::AcqRel) {
            self.refresh_session_graph_from_store().await;
        }
        self.refresh_session_tool_surface().await?;
        Ok(())
    }

    pub(super) async fn refresh_session_graph_from_store(&mut self) {
        let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        else {
            return;
        };
        let Some(head_meta) = store.load_session_head_meta().await else {
            return;
        };
        let has_newer_graph = head_meta.graph_node_count > self.state.persisted_graph_node_count
            || head_meta.leaf_node_id != self.state.session_graph.leaf_node_id
            || head_meta.checkpoint_ref != self.state.checkpoint_ref;
        if !has_newer_graph {
            return;
        }
        let mut graph = store.load_session_graph().await;
        graph.set_leaf_node_id(head_meta.leaf_node_id.clone());
        let head = crate::store::SessionHead {
            session_id: head_meta.session_id.clone(),
            graph,
            config: head_meta.config.clone(),
            checkpoint_ref: head_meta.checkpoint_ref.clone(),
            token_ledger: merge_usage_delta_entries(store.load_usage_deltas().await),
        };
        apply_session_head(&mut self.state, &head);
        let checkpoint =
            load_session_checkpoint(store.as_ref(), head.checkpoint_ref.as_ref()).await;
        apply_session_checkpoint(&mut self.state, checkpoint);
    }

    pub(super) fn runtime_session_manager(
        &self,
    ) -> Result<Arc<dyn SessionManager>, ExternalInvokeError> {
        self.runtime_session_manager_with_prompt_bridge(None)
    }

    fn runtime_session_manager_for_turn(
        &self,
        prompt_bridge: Option<HostPromptBridge>,
        child_usage_event_relay: Option<ChildUsageEventRelay>,
    ) -> Result<Arc<dyn SessionManager>, ExternalInvokeError> {
        Ok(Arc::new(RuntimeSessionManager::new(
            self,
            prompt_bridge,
            false,
            child_usage_event_relay,
        )?))
    }

    fn runtime_session_manager_with_prompt_bridge(
        &self,
        prompt_bridge: Option<HostPromptBridge>,
    ) -> Result<Arc<dyn SessionManager>, ExternalInvokeError> {
        Ok(Arc::new(RuntimeSessionManager::new(
            self,
            prompt_bridge,
            true,
            None,
        )?))
    }

    pub fn session_manager(&self) -> Result<Arc<dyn SessionManager>, ExternalInvokeError> {
        self.runtime_session_manager()
    }

    /// The plugin session bound to the currently active runtime session, if any.
    pub fn plugin_session(&self) -> Option<Arc<crate::PluginSession>> {
        self.session.as_ref().map(|s| Arc::clone(s.plugins()))
    }

    /// Run the registered history rewrite pipeline against the current
    /// state, applying the resulting messages back onto the runtime.
    /// Returns true when at least one rewriter produced a summary or
    /// otherwise mutated the message list.
    pub async fn rewrite_history(
        &mut self,
        trigger: crate::RewriteTrigger,
    ) -> Result<bool, ExternalInvokeError> {
        let manager = self.runtime_session_manager()?;
        let Some(plugin_session) = self.session.as_ref().map(|s| Arc::clone(s.plugins())) else {
            return Err(ExternalInvokeError::Unknown(
                "runtime session not available".to_string(),
            ));
        };
        let ctx = crate::RewriteContext {
            session_id: self.state.session_id.clone(),
            trigger,
            state: self.state.read_view(),
            host: manager,
        };
        let input = crate::HistoryState::from_state(&self.state.export_state());
        let baseline_messages = input.messages.len();
        let outcome = plugin_session
            .rewrite_history(&ctx, input)
            .await
            .map_err(|err| {
                ExternalInvokeError::Unknown(format!("rewrite_history failed: {err}"))
            })?;
        let mutated =
            outcome.metadata.produced_summary || outcome.messages.len() != baseline_messages;
        if mutated {
            self.state
                .replace_projection(&outcome.messages, &outcome.tool_calls);
            if let Some(session) = self.session.as_ref() {
                self.state.dynamic_state_snapshot = session
                    .plugins()
                    .dynamic_tools()
                    .map(|tools| tools.export_state());
                self.state.plugin_snapshot = session.plugins().snapshot().ok();
                self.state.plugin_snapshot_revision =
                    Some(session.plugins().snapshot_revision_fingerprint());
            }
        }
        Ok(mutated)
    }

    pub(super) fn session_policy(&self) -> SessionPolicy {
        self.policy.clone()
    }

    pub(super) async fn notify_session_config_changed(&self, previous: SessionPolicy) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let current = self.session_policy();
        if current == previous {
            return;
        }
        let Ok(host) = self.runtime_session_manager() else {
            return;
        };
        session
            .plugins()
            .emit_runtime_event(crate::PluginRuntimeEvent::SessionConfigChanged(Box::new(
                SessionConfigChangedContext {
                    session_id: self.state.session_id.clone(),
                    previous,
                    current,
                    host,
                },
            )))
            .await;
    }

    pub(super) async fn apply_session_config_mutations(&mut self, previous: SessionPolicy) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let current = self.session_policy();
        if current == previous {
            return;
        }
        let Ok(host) = self.runtime_session_manager() else {
            return;
        };
        self.policy = session
            .plugins()
            .mutate_session_config(
                SessionConfigChangedContext {
                    session_id: self.state.session_id.clone(),
                    previous,
                    current,
                    host,
                },
                self.policy.clone(),
            )
            .await;
        self.state.policy = self.policy.clone();
    }

    /// Run a single turn and stream events to the host sink.
    /// Includes overflow recovery: if the LLM rejects the prompt as too long,
    /// the context is force-compacted and the turn is retried once.
    pub async fn stream_turn(
        &mut self,
        input: TurnInput,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let saved_messages = self.state.session_graph.shared_projected_messages();
        let saved_tool_calls = self.state.session_graph.shared_projected_tool_calls();
        let saved_prompt_usage = self.state.last_prompt_usage.clone();

        let assembled = self
            .stream_turn_inner(input.clone(), events, cancel.clone())
            .await?;

        if !self.overflow_recovery_attempted && Self::has_overflow_error(&assembled) {
            self.overflow_recovery_attempted = true;
            // Restore pre-turn state so the retry appends the user message cleanly.
            self.state
                .replace_projection(saved_messages.as_slice(), saved_tool_calls.as_slice());
            self.state.last_prompt_usage = saved_prompt_usage;
            // Force-compact: strip images, prune, summarize.
            let _ = self
                .rewrite_history(crate::RewriteTrigger::OverflowRecovery)
                .await;
            let retry = self.stream_turn_inner(input, events, cancel).await?;
            self.overflow_recovery_attempted = false;
            return Ok(retry);
        }
        self.overflow_recovery_attempted = false;
        Ok(assembled)
    }

    async fn stream_turn_inner(
        &mut self,
        input: TurnInput,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.refresh_session_graph_from_store().await;
        let previous_prompt_usage = self.state.last_prompt_usage.clone();
        let normalized = match self.normalize_input_items(&input.items, &input.image_blobs) {
            Ok(items) => items,
            Err(e) => {
                self.state.last_prompt_usage = None;
                let mut assembler = TurnAssembler::default();
                let error_event = SessionEvent::Error {
                    message: e.clone(),
                    envelope: Some(crate::session_model::ErrorEnvelope {
                        kind: "input_validation".to_string(),
                        code: Some("invalid_turn_input".to_string()),
                        user_message: e,
                        raw: None,
                    }),
                };
                assembler.push(&error_event);
                events.emit(error_event).await;
                assembler.push(&SessionEvent::Done);
                events.emit(SessionEvent::Done).await;
                return Ok(assembler.finish(
                    self.state.export_state(),
                    false,
                    None,
                    &self.host.core.sanitizer,
                    &self.host.core.termination,
                ));
            }
        };

        let base_messages = self.state.session_graph.shared_projected_messages();
        let base_rendered_prompt = self.state.session_graph.shared_projected_rendered_prompt();
        let mut turn_delta = Vec::new();
        let mode = input.mode.unwrap_or(RunMode::Normal);
        let mode_msg = match mode {
            RunMode::Normal => None,
        };
        if let Some(content) = mode_msg {
            let sys_id = fresh_message_id();
            turn_delta.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Text,
                    content,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                }],
                user_input: None,
                origin: None,
            });
        }

        let user_id = fresh_message_id();
        let mut user_parts: Vec<Part> = Vec::new();
        for item in normalized {
            match item {
                NormalizedItem::Text(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    user_parts.push(Part {
                        id: format!("{}.p{}", user_id, user_parts.len()),
                        kind: PartKind::Text,
                        content: text,
                        attachment: None,
                        tool_call_id: None,
                        tool_name: None,
                        tool_item_id: None,
                        tool_signature: None,
                        prune_state: PruneState::Intact,
                        reasoning_meta: None,
                    });
                }
                NormalizedItem::Image(bytes) => {
                    user_parts.push(Part {
                        id: format!("{}.p{}", user_id, user_parts.len()),
                        kind: PartKind::Image,
                        content: String::new(),
                        attachment: Some(crate::session_model::message::PartAttachment {
                            mime: "image/png".to_string(),
                            url: crate::session_model::message::data_url_for_bytes(
                                "image/png",
                                &bytes,
                            ),
                            filename: None,
                        }),
                        tool_call_id: None,
                        tool_name: None,
                        tool_item_id: None,
                        tool_signature: None,
                        prune_state: PruneState::Intact,
                        reasoning_meta: None,
                    });
                }
            }
        }
        if user_parts.is_empty() {
            user_parts.push(Part {
                id: format!("{}.p0", user_id),
                kind: PartKind::Text,
                content: String::new(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            });
        }
        reassign_part_ids(&user_id, &mut user_parts);
        turn_delta.push(Message {
            id: user_id.clone(),
            role: MessageRole::User,
            parts: user_parts,
            user_input: input.user_input.clone(),
            origin: None,
        });

        let manager = self
            .runtime_session_manager_for_turn(None, None)
            .map_err(|err| RuntimeError {
                code: "plugin_session_manager".to_string(),
                message: err.to_string(),
            })?;
        let plugin_session = self
            .session
            .as_ref()
            .map(|s| Arc::clone(s.plugins()))
            .ok_or_else(|| RuntimeError {
                code: "context_prepare_turn".to_string(),
                message: "runtime session not available".to_string(),
            })?;
        let turn_ctx = crate::TurnTransformContext {
            session_id: self.state.session_id.clone(),
            state: self.state.read_view(),
            prompt_usage: previous_prompt_usage.clone(),
            max_context_tokens: Some(LashRuntime::max_context_tokens(self)),
            host: Arc::clone(&manager),
        };
        self.mark_phase_begin(RuntimeTurnPhase::ContextTransform);
        let prepared_context = plugin_session
            .prepare_turn_context(
                &turn_ctx,
                crate::session_model::context::PreparedContext {
                    messages: crate::MessageSequence::from_base_and_delta(
                        base_messages,
                        turn_delta,
                    )
                    .with_base_rendered_prompt(Some(base_rendered_prompt)),
                    ..Default::default()
                },
            )
            .await
            .map_err(|err| RuntimeError {
                code: "context_prepare_turn".to_string(),
                message: err.to_string(),
            })?;
        self.mark_phase_end(RuntimeTurnPhase::ContextTransform);
        // Release the read-view's graph clone before the rest of the turn
        // runs. Keeping it alive into `stream_prepared_turn` forces the
        // post-turn `append_projection_delta` to deep-clone the session
        // graph (Arc::make_mut with refcount > 1).
        drop(turn_ctx);
        let messages = prepared_context.messages;
        if let Some(session) = self.session.as_mut() {
            session.set_context_surface(
                prepared_context.tool_providers,
                prepared_context.prompt_contributions,
                prepared_context.include_base_tools,
            );
        }

        self.state.last_prompt_usage = None;

        self.stream_prepared_turn(
            messages,
            previous_prompt_usage,
            input.rlm_termination_override.clone(),
            events,
            cancel,
        )
        .await
    }

    /// Run a single turn and return only the assembled terminal result.
    pub async fn run_turn_assembled(
        &mut self,
        input: TurnInput,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.stream_turn(input, &NoopEventSink, cancel).await
    }

    /// Run a turn using host-prepared message history.
    pub async fn stream_prepared_turn(
        &mut self,
        messages: crate::MessageSequence,
        _previous_prompt_usage: Option<PromptUsage>,
        rlm_termination_override: Option<crate::RlmTermination>,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let prompt_bridge = HostPromptBridge::new();
        let (event_tx, mut event_rx) = mpsc::channel::<RuntimeStreamEvent>(100);
        let child_usage_event_relay = ChildUsageEventRelay::new(event_tx.clone());
        let manager = self
            .runtime_session_manager_for_turn(
                Some(prompt_bridge.clone()),
                Some(child_usage_event_relay.clone()),
            )
            .map_err(|err| RuntimeError {
                code: "plugin_session_manager".to_string(),
                message: err.to_string(),
            })?;
        let (prompt_tx, mut prompt_rx) = tokio::sync::mpsc::unbounded_channel::<PendingPrompt>();
        prompt_bridge.set_sender(prompt_tx);
        let prompt_event_tx = event_tx.clone();
        let prompt_hook_manager = Arc::clone(&manager);
        let prompt_plugins = self
            .session
            .as_ref()
            .map(|session| Arc::clone(session.plugins()));
        let prompt_forward = tokio::spawn(async move {
            while let Some(prompt) = prompt_rx.recv().await {
                if let Some(plugins) = prompt_plugins.as_ref() {
                    match plugins
                        .on_prompt_request(crate::PromptRequestHookContext {
                            session_id: plugins.session_id().to_string(),
                            request: prompt.request.clone(),
                            host: Arc::clone(&prompt_hook_manager),
                        })
                        .await
                    {
                        Ok(emitted) => {
                            for surface in emitted {
                                let events = crate::plugin::plugin_surface_session_events(
                                    &surface.plugin_id,
                                    vec![surface.value],
                                );
                                for event in events {
                                    let _ = prompt_event_tx
                                        .send(RuntimeStreamEvent::Session(event))
                                        .await;
                                }
                            }
                        }
                        Err(err) => {
                            let _ = prompt_event_tx
                                .send(RuntimeStreamEvent::Session(make_error_event(
                                    "plugin_prompt_request",
                                    None,
                                    err.to_string(),
                                    Some(err.to_string()),
                                )))
                                .await;
                        }
                    }
                }
                if !prompt_event_tx.is_closed() {
                    let _ = prompt_event_tx
                        .send(RuntimeStreamEvent::Session(SessionEvent::Prompt {
                            request: prompt.request,
                            response_tx: prompt.response_tx,
                        }))
                        .await;
                }
            }
        });
        let mut assembler = TurnAssembler::default();
        let plugins = {
            let session = self
                .session
                .as_ref()
                .expect("lash runtime session must be available");
            Arc::clone(session.plugins())
        };
        self.mark_phase_begin(RuntimeTurnPhase::BeforeTurnHooks);
        // Block-scope the pinned future so it (and its captured
        // `SessionReadView` clone of the session graph) drops before the
        // post-turn `append_projection_delta` mutation. Keeping it alive
        // across the turn forces `Arc::make_mut` to deep-clone
        // `SessionGraphData`.
        let prepared = {
            let prepare_turn = plugins.prepare_turn(PrepareTurnRequest {
                session_id: self.state.session_id.clone(),
                state: self.state.read_view(),
                messages,
                host: Arc::clone(&manager),
            });
            tokio::pin!(prepare_turn);

            loop {
                tokio::select! {
                    prepared = &mut prepare_turn => {
                        let prepared = prepared.map_err(|err| RuntimeError {
                            code: "plugin_prepare_turn".to_string(),
                            message: err.to_string(),
                        })?;
                        self.mark_phase_end(RuntimeTurnPhase::BeforeTurnHooks);
                        break prepared;
                    }
                    maybe_event = event_rx.recv() => {
                        if let Some(event) = maybe_event {
                            match event {
                                RuntimeStreamEvent::Session(event) => {
                                    assembler.push(&event);
                                    events.emit(event).await;
                                }
                            }
                        }
                    }
                }
            }
        };
        for event in &prepared.events {
            assembler.push(event);
        }
        emit_session_events_to_sink(events, prepared.events).await;
        if let Some(abort) = prepared.abort {
            prompt_bridge.clear_sender();
            let _ = prompt_forward.await;
            drop(event_tx);

            let mut state = self.state.clone();
            if let Some(appended_messages) = projection_message_delta_if_base_preserved(
                state.projected_messages(),
                prepared.messages.as_slice(),
            ) {
                state.append_projection_delta(&appended_messages, &[]);
            } else {
                let tool_calls = state.project_tool_calls();
                state.replace_projection(prepared.messages.as_slice(), &tool_calls);
            }
            let issue = TurnIssue {
                kind: "plugin".to_string(),
                code: Some(abort.code),
                message: abort.message.clone(),
                raw: None,
            };
            let error_event = SessionEvent::Error {
                message: abort.message,
                envelope: Some(crate::session_model::ErrorEnvelope {
                    kind: "plugin".to_string(),
                    code: issue.code.clone(),
                    user_message: issue.message.clone(),
                    raw: None,
                }),
            };
            assembler.push(&error_event);
            events.emit(error_event).await;
            assembler.push(&SessionEvent::Done);
            events.emit(SessionEvent::Done).await;
            return Ok(assembler.finish(
                state.export_state(),
                cancel.is_cancelled(),
                Some(issue),
                &self.host.core.sanitizer,
                &self.host.core.termination,
            ));
        }
        let cancel_state = cancel.clone();
        let session = self
            .session
            .take()
            .expect("lash runtime session must be available");
        let mut driver = RuntimeTurnDriver {
            session,
            policy: self.policy.clone(),
            host: self.host.clone(),
            session_id: self.state.session_id.clone(),
            base_graph: self.state.session_graph.clone(),
            tool_calls: self.state.session_graph.shared_projected_tool_calls(),
            llm_stream_summaries: HashMap::new(),
            session_manager: manager,
            prompt_bridge,
            rlm_termination: rlm_termination_override
                .clone()
                .unwrap_or_else(|| self.rlm_termination.clone()),
            turn_phase_probe: self.turn_phase_probe.clone(),
        };
        let run_offset = self.state.iteration;
        let run_task = tokio::spawn(async move {
            let (new_messages, new_iteration) = driver
                .run(prepared.messages, event_tx, cancel, run_offset)
                .await;
            (driver, new_messages, new_iteration)
        });
        tokio::pin!(run_task);

        self.mark_phase_begin(RuntimeTurnPhase::EffectLoop);
        let (driver, new_messages, new_iteration) = loop {
            tokio::select! {
                maybe_event = event_rx.recv() => {
                    if let Some(event) = maybe_event {
                        match event {
                            RuntimeStreamEvent::Session(event) => {
                                assembler.push(&event);
                                events.emit(event).await;
                            }
                        }
                    }
                }
                joined = &mut run_task => {
                    child_usage_event_relay.clear();
                    let joined = match joined {
                        Ok(v) => v,
                        Err(e) => {
                            let issue = TurnIssue {
                                kind: "runtime".to_string(),
                                code: Some("run_task_join_failed".to_string()),
                                message: format!("Runtime turn task failed: {e}"),
                                raw: None,
                            };
                            return Ok(assembler.finish(
                                self.state.export_state(),
                                cancel_state.is_cancelled(),
                                Some(issue),
                                &self.host.core.sanitizer,
                                &self.host.core.termination,
                            ));
                        }
                    };
                    break joined;
                }
            }
        };
        let _ = prompt_forward.await;
        while let Some(event) = event_rx.recv().await {
            match event {
                RuntimeStreamEvent::Session(event) => {
                    assembler.push(&event);
                    events.emit(event).await;
                }
            }
        }
        self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            new_message_count = new_messages.len(),
            tool_call_count = assembler.tool_calls.len(),
            "runtime post-run_task"
        );

        // Drain the shared token ledger (child sessions + direct
        // completions + async OM observers/reflectors) and merge into
        // the session state. Also record the parent's own turn usage.
        let child_ledger = {
            let mut ledger = self.shared_token_ledger.lock().expect("token ledger lock");
            std::mem::take(&mut *ledger)
        };
        let mut turn_usage_delta = child_ledger.clone();
        for entry in child_ledger {
            merge_ledger_entry(&mut self.state.token_ledger, entry);
        }
        if assembler.token_usage.total() > 0 || assembler.token_usage.cached_input_tokens > 0 {
            let entry = TokenLedgerEntry {
                source: "turn".to_string(),
                model: driver.policy.model.clone(),
                usage: assembler.token_usage.clone(),
            };
            merge_ledger_entry(&mut self.state.token_ledger, entry.clone());
            turn_usage_delta.push(entry);
        }
        let turn_usage_delta = merge_usage_delta_entries(turn_usage_delta);

        let RuntimeTurnDriver {
            session,
            policy,
            base_graph,
            ..
        } = driver;
        // Explicit drop: `..` elision keeps the base_graph clone alive
        // until end of scope in practice, which forces
        // `append_projection_delta`'s `Arc::make_mut` to deep-clone.
        drop(base_graph);
        self.session = Some(session);
        self.policy = policy;
        self.state.policy = self.policy.clone();
        self.state.iteration = new_iteration;
        if let Some(appended_messages) = projection_message_delta_if_base_preserved(
            self.state.projected_messages(),
            new_messages.as_slice(),
        ) {
            self.state
                .append_projection_delta(&appended_messages, &assembler.tool_calls);
        } else {
            let mut next_tool_calls = self.state.project_tool_calls();
            if !assembler.tool_calls.is_empty() {
                next_tool_calls.extend(assembler.tool_calls.clone());
            }
            self.state
                .replace_projection(new_messages.as_slice(), &next_tool_calls);
        }
        if assembler.token_usage.total() > 0 || assembler.token_usage.cached_input_tokens > 0 {
            self.state.token_usage = assembler.token_usage.clone();
        }

        let last_prompt_usage = assembler
            .last_llm_usage()
            .and_then(|usage| normalize_prompt_usage(self.policy.provider.as_dyn(), usage));
        let finalize_manager = if self.session.is_some() {
            Some(
                self.runtime_session_manager_for_turn(None, None)
                    .map_err(|err| RuntimeError {
                        code: "plugin_session_manager".to_string(),
                        message: err.to_string(),
                    })?,
            )
        } else {
            None
        };
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            state_message_count = self.state.projected_messages().len(),
            graph_node_count = self.state.session_graph.nodes.len(),
            token_ledger_entries = self.state.token_ledger.len(),
            "runtime before assembler.finish"
        );
        let mut assembled_state = std::mem::take(&mut self.state);
        assembled_state.last_prompt_usage = last_prompt_usage.clone();
        let assembled = assembler.finish(
            assembled_state.export_state(),
            cancel_state.is_cancelled(),
            None,
            &self.host.core.sanitizer,
            &self.host.core.termination,
        );
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            assembled_message_count = assembled.state.projected_messages().len(),
            assembled_graph_node_count = assembled.state.session_graph.nodes.len(),
            "runtime after assembler.finish"
        );
        if let Some(session) = self.session.as_ref() {
            let plugins = Arc::clone(session.plugins());
            let manager = finalize_manager.expect("finalize manager should exist with session");
            tracing::debug!(rss_kb = debug_rss_kb(), "runtime before finalize_turn");
            self.mark_phase_begin(RuntimeTurnPhase::FinalizeTurn);
            let finalized = plugins
                .finalize_turn(assembled, manager)
                .await
                .map_err(|err| RuntimeError {
                    code: "plugin_finalize_turn".to_string(),
                    message: err.to_string(),
                })?;
            self.mark_phase_end(RuntimeTurnPhase::FinalizeTurn);
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                finalized_message_count = finalized.turn.state.projected_messages().len(),
                "runtime after finalize_turn"
            );
            let mut returned_turn = finalized.turn;
            let dynamic_state = plugins.dynamic_tools().map(|tools| tools.export_state());
            let plugin_snapshot = plugins.snapshot().ok();
            let plugin_snapshot_revision = Some(plugins.snapshot_revision_fingerprint());
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                dynamic_state_present = dynamic_state.is_some(),
                plugin_snapshot_present = plugin_snapshot.is_some(),
                "runtime before stamp_runtime_state"
            );
            self.mark_phase_begin(RuntimeTurnPhase::PersistTurn);
            assembled_state.apply_exported_state(&returned_turn.state);
            assembled_state.dynamic_state_snapshot = dynamic_state;
            assembled_state.dynamic_state_generation = assembled_state
                .dynamic_state_snapshot
                .as_ref()
                .map(|snapshot| snapshot.base_generation);
            assembled_state.plugin_snapshot = plugin_snapshot;
            assembled_state.plugin_snapshot_revision = plugin_snapshot_revision;
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                persisted_graph_node_count = assembled_state.session_graph.nodes.len(),
                persisted_message_count = assembled_state.projected_messages().len(),
                "runtime after stamp_runtime_state"
            );
            if let Some(store) = self
                .session
                .as_ref()
                .and_then(|session| session.history_store())
            {
                let commit = crate::store::RuntimeCommit::persisted_state(
                    &assembled_state,
                    &turn_usage_delta,
                );
                let crate::store::RuntimeCommitResult::PersistedState(result) = store
                    .apply_runtime_commit(commit)
                    .await
                    .map_err(|err| RuntimeError {
                        code: "store_commit_failed".to_string(),
                        message: err.to_string(),
                    })?
                else {
                    unreachable!("persisted state commit should return persisted result");
                };
                assembled_state.apply_persisted_commit_result(result);
            } else {
                clear_persisted_runtime_caches(&mut assembled_state);
            }
            returned_turn.state = assembled_state.export_state();
            emit_session_events_to_sink(events, finalized.events).await;
            self.state = assembled_state;
            if let Some(session) = self.session.as_ref()
                && let Ok(host) = self.runtime_session_manager()
            {
                session
                    .plugins()
                    .emit_runtime_event(crate::PluginRuntimeEvent::TurnPersisted(
                        crate::SessionStateChangedContext {
                            session_id: self.state.session_id.clone(),
                            state: returned_turn.state.read_view(),
                            host,
                        },
                    ))
                    .await;
            }
            self.mark_phase_end(RuntimeTurnPhase::PersistTurn);
            Ok(returned_turn)
        } else {
            self.state.apply_exported_state(&assembled.state);
            Ok(assembled)
        }
    }
}

impl LashRuntime {
    fn normalize_input_items(
        &self,
        items: &[InputItem],
        image_blobs: &HashMap<String, Vec<u8>>,
    ) -> Result<Vec<NormalizedItem>, String> {
        normalize_input_items(
            items,
            image_blobs,
            self.host.core.base_dir.as_path(),
            self.host.core.path_resolver.as_ref(),
        )
    }
}
