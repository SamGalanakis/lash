mod assembly;
mod builder;
mod config_ops;
mod environment;
mod host;
mod io;
mod lifecycle;
mod session_api;
mod session_manager;
mod session_ops;
mod state;
#[cfg(test)]
mod tests;
mod turn_commit_pipeline;
mod turn_driver;
mod turn_graph;
mod turn_loop;
mod turn_progress;
mod usage;

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::llm::types::{
    LlmOutputPart, LlmProviderTraceEvent, LlmProviderTraceSender, LlmRequest, LlmResponse,
    LlmStreamEvent, LlmUsage,
};
use crate::plugin::{
    CheckpointHookContext, PluginMessage, PrepareTurnRequest, SessionConfigChangedContext,
};
use crate::sansio::{LlmCallError, Response};
use crate::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, SessionEvent, SessionPolicy, TokenUsage,
    fresh_message_id, make_error_event, reassign_part_ids, shared_parts, transport_stream_events,
};
use crate::tool_dispatch::{ToolDispatchContext, dispatch_tool_call_with_execution_context};
use crate::{
    CheckpointKind, ExecutionMode, ExternalInvokeError, PersistentRuntimeServices,
    PromptHookContext, RuntimeServices, RuntimeSessionHost, SandboxMessage, Session,
    SessionCreateRequest, SessionError, SessionHandle, SessionSnapshot, SessionStartPoint,
    ToolCallRecord, TurnFinish, TurnOutcome, TurnStop,
};
use crate::{Effect, TurnMachine};

use host::*;
use session_manager::*;
use turn_commit_pipeline::*;
use turn_driver::*;
use turn_progress::*;

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
    BackgroundRuntimeHost, DefaultPathResolver, EmbeddedRuntimeHost, ManagedRunState,
    ManagedTaskCancel, ManagedTaskKind, ManagedTaskSpec, ManagedTaskStatus, RuntimeCoreConfig,
    SessionTaskExecutor, TokioSessionTaskExecutor,
};
use io::normalize_input_items;
pub use state::{PersistedSessionState, SessionStateEnvelope};
use state::{
    append_session_nodes_to_state, apply_residency_on_load, apply_session_checkpoint,
    apply_session_head, normalize_session_graph,
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
    /// Per-turn override for mode-owned turn options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode_turn_options: Option<crate::ModeTurnOptions>,
    /// Optional externally-stable trace turn id. Normal runtime callers leave
    /// this empty and the runtime generates one per outer turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_turn_id: Option<String>,
}

#[derive(Clone)]
pub struct ModeTurnSidecarHandle(Arc<dyn ModeTurnSidecar>);

impl ModeTurnSidecarHandle {
    pub fn new(sidecar: impl ModeTurnSidecar + 'static) -> Self {
        Self(Arc::new(sidecar))
    }

    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    pub fn prompt_contributions(&self) -> Vec<crate::PromptContribution> {
        self.0.prompt_contributions()
    }
}

impl fmt::Debug for ModeTurnSidecarHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ModeTurnSidecarHandle(..)")
    }
}

pub trait ModeTurnSidecar: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn prompt_contributions(&self) -> Vec<crate::PromptContribution> {
        Vec::new()
    }
}

static MODE_TURN_SIDECARS: OnceLock<StdMutex<HashMap<String, ModeTurnSidecarHandle>>> =
    OnceLock::new();

fn mode_turn_sidecars() -> &'static StdMutex<HashMap<String, ModeTurnSidecarHandle>> {
    MODE_TURN_SIDECARS.get_or_init(|| StdMutex::new(HashMap::new()))
}

impl TurnInput {
    pub fn set_mode_sidecar(&mut self, sidecar: ModeTurnSidecarHandle) {
        let turn_id = self
            .trace_turn_id
            .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
            .clone();
        if let Ok(mut sidecars) = mode_turn_sidecars().lock() {
            sidecars.insert(turn_id, sidecar);
        }
    }

    pub fn mode_sidecar_handle(&self) -> Option<ModeTurnSidecarHandle> {
        self.trace_turn_id.as_ref().and_then(|turn_id| {
            mode_turn_sidecars()
                .lock()
                .ok()
                .and_then(|sidecars| sidecars.get(turn_id).cloned())
        })
    }

    pub(crate) fn take_mode_sidecar_handle(&mut self) -> Option<ModeTurnSidecarHandle> {
        self.trace_turn_id.as_ref().and_then(|turn_id| {
            mode_turn_sidecars()
                .lock()
                .ok()
                .and_then(|mut sidecars| sidecars.remove(turn_id))
        })
    }
}

#[derive(Clone, Debug)]
pub(super) enum NormalizedItem {
    Text(String),
    Image(crate::AttachmentRef),
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
    pub outcome: crate::TurnOutcome,
    pub assistant_output: AssistantOutput,
    #[serde(default)]
    pub has_plugin_visible_output: bool,
    pub execution: ExecutionSummary,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default)]
    pub errors: Vec<TurnIssue>,
}

/// Result of driving one logical host turn through any foreground handoffs.
///
/// A handoff is an internal runtime continuation, similar to compaction from a
/// host's perspective. Callers that need a final answer can use
/// [`LashRuntime::stream_turn_following_handoffs`] and inspect `final_turn()`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FollowedTurn {
    pub turns: Vec<AssembledTurn>,
}

impl FollowedTurn {
    pub fn final_turn(&self) -> Option<&AssembledTurn> {
        self.turns.last()
    }

    pub fn into_final_turn(mut self) -> Option<AssembledTurn> {
        self.turns.pop()
    }

    pub fn handoff_count(&self) -> usize {
        self.turns
            .iter()
            .filter(|turn| matches!(turn.outcome, crate::TurnOutcome::Handoff { .. }))
            .count()
    }
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
    fn is_noop(&self) -> bool {
        false
    }

    async fn emit(&self, event: SessionEvent);
}

/// No-op sink useful for callers that only care about final state.
pub struct NoopEventSink;

#[async_trait::async_trait]
impl EventSink for NoopEventSink {
    fn is_noop(&self) -> bool {
        true
    }

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
    ) -> Result<Arc<dyn crate::store::RuntimePersistence>, String>;
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
    pub(in crate::runtime) active_handoff_continuations: Arc<Mutex<HashMap<String, String>>>,
    pub(in crate::runtime) managed_turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    pub(in crate::runtime) overflow_recovery_attempted: bool,
    /// Mode-owned turn options for this session.
    pub(in crate::runtime) mode_turn_options: crate::ModeTurnOptions,
    /// Session-scoped token cost ledger. Shared by ALL
    /// `RuntimeSessionManager` instances created from this runtime
    /// (both per-turn and async maintenance). Entries accumulate here
    /// and are drained into `state.token_ledger` at turn-commit time.
    pub(in crate::runtime) shared_token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    pub(in crate::runtime) background_sync_needed: Arc<AtomicBool>,
    /// Seed `PluginMessage`s queued via
    /// `SessionCreateRequest::first_turn_input` for child sessions.
    /// Shared across `RuntimeSessionManager` instances built from this
    /// runtime so the seed remains visible after the parent turn that
    /// created the session has ended.
    pub(in crate::runtime) pending_first_turn_inputs:
        Arc<std::sync::Mutex<HashMap<String, crate::PluginMessage>>>,
    pub(in crate::runtime) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}
