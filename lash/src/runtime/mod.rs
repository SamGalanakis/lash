mod session_manager;
#[cfg(test)]
mod tests;
mod turn_driver;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::llm::factory::adapter_for;
use crate::llm::transport::LlmTransport;
use crate::llm::types::{LlmOutputPart, LlmRequest, LlmResponse, LlmStreamEvent, LlmUsage};
use crate::plugin::{
    CheckpointHookContext, PluginMessage, PrepareTurnRequest, SessionConfigChangedContext,
    ToolResultProjectionContext, ToolResultProjectionHook, emit_plugin_surface_events,
    plugin_surface_event_renders_visible_output,
};
use crate::provider::Provider;
use crate::sansio::{Effect, LlmCallError, Response, TurnMachine, TurnMachineConfig};
use crate::session_model::{
    DurableTurnSnapshot, Message, MessageRole, Part, PartKind, PruneState, SessionEvent,
    SessionPolicy, TokenUsage, build_execution_preamble, finalize_prompt_context, fresh_message_id,
    make_error_event, plugin_message_to_message, reassign_part_ids, transport_stream_events,
};
use crate::tool_dispatch::{ToolDispatchContext, dispatch_tool_call};
use crate::{
    CheckpointKind, ExecutionMode, ExternalInvokeError, PromptHookContext, PromptRenderer,
    PromptSectionOverride, RuntimeServices, SandboxMessage, Session, SessionCreateRequest,
    SessionError, SessionHandle, SessionManager, SessionSnapshot, SessionStartPoint,
    ToolCallRecord,
};

use session_manager::*;
use turn_driver::*;

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
}

#[derive(Clone, Debug)]
enum NormalizedItem {
    Text(String),
    Image(Vec<u8>),
}

/// Exact prompt-usage snapshot from the most recent completed LLM call.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PromptUsage {
    pub prompt_context_tokens: usize,
    pub input_tokens: usize,
    pub cached_input_tokens: usize,
    #[serde(default)]
    pub context_budget_tokens: usize,
}

/// A single row in the token cost ledger. One per unique
/// `(source, model)` pair — accumulated, not per-call.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenLedgerEntry {
    /// Caller-supplied label: `"turn"`, `"predict"`, `"compaction"`,
    /// `"observer"`, `"reflector"`, `"agent_call"`, or any plugin-defined
    /// string. Core doesn't interpret the value; the UI uses it for
    /// grouping and display.
    pub source: String,
    /// Model identifier used for the LLM call (e.g.
    /// `"anthropic/claude-haiku-4-5"`).
    pub model: String,
    /// Accumulated token counts for this `(source, model)` pair.
    pub usage: TokenUsage,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct UsageTotals {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    #[serde(default)]
    pub reasoning_tokens: i64,
    pub total_tokens: i64,
    pub context_total_tokens: i64,
}

impl UsageTotals {
    fn from_usage(usage: &TokenUsage) -> Self {
        let total_tokens = usage.total();
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_input_tokens: usage.cached_input_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            total_tokens,
            context_total_tokens: total_tokens + usage.cached_input_tokens,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct UsageReportRow {
    pub source: String,
    pub model: String,
    pub usage: UsageTotals,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SessionUsageReport {
    pub entry_count: usize,
    pub usage: UsageTotals,
    pub by_source: BTreeMap<String, UsageTotals>,
    pub by_model: BTreeMap<String, UsageTotals>,
    pub by_source_model: Vec<UsageReportRow>,
}

impl SessionUsageReport {
    pub fn from_entries(entries: &[TokenLedgerEntry]) -> Self {
        let mut total = TokenUsage::default();
        let mut by_source_usage = BTreeMap::<String, TokenUsage>::new();
        let mut by_model_usage = BTreeMap::<String, TokenUsage>::new();
        let mut by_source_model = Vec::with_capacity(entries.len());

        for entry in entries {
            total.add(&entry.usage);
            by_source_usage
                .entry(entry.source.clone())
                .or_default()
                .add(&entry.usage);
            by_model_usage
                .entry(entry.model.clone())
                .or_default()
                .add(&entry.usage);
            by_source_model.push(UsageReportRow {
                source: entry.source.clone(),
                model: entry.model.clone(),
                usage: UsageTotals::from_usage(&entry.usage),
            });
        }

        Self {
            entry_count: entries.len(),
            usage: UsageTotals::from_usage(&total),
            by_source: by_source_usage
                .into_iter()
                .map(|(key, usage)| (key, UsageTotals::from_usage(&usage)))
                .collect(),
            by_model: by_model_usage
                .into_iter()
                .map(|(key, usage)| (key, UsageTotals::from_usage(&usage)))
                .collect(),
            by_source_model,
        }
    }
}

pub fn diff_token_ledger(
    before: &[TokenLedgerEntry],
    after: &[TokenLedgerEntry],
) -> Result<Vec<TokenLedgerEntry>, String> {
    let before_index = before
        .iter()
        .map(|entry| ((entry.source.as_str(), entry.model.as_str()), &entry.usage))
        .collect::<HashMap<_, _>>();
    let after_index = after
        .iter()
        .map(|entry| ((entry.source.as_str(), entry.model.as_str()), &entry.usage))
        .collect::<HashMap<_, _>>();

    let mut keys = before_index
        .keys()
        .copied()
        .chain(after_index.keys().copied())
        .collect::<Vec<_>>();
    keys.sort_unstable();
    keys.dedup();

    let mut out = Vec::new();
    for (source, model) in keys {
        let before_usage = before_index
            .get(&(source, model))
            .copied()
            .cloned()
            .unwrap_or_default();
        let after_usage = after_index
            .get(&(source, model))
            .copied()
            .cloned()
            .unwrap_or_default();
        let delta = TokenUsage {
            input_tokens: after_usage.input_tokens - before_usage.input_tokens,
            output_tokens: after_usage.output_tokens - before_usage.output_tokens,
            cached_input_tokens: after_usage.cached_input_tokens - before_usage.cached_input_tokens,
            reasoning_tokens: after_usage.reasoning_tokens - before_usage.reasoning_tokens,
        };
        if delta.input_tokens < 0
            || delta.output_tokens < 0
            || delta.cached_input_tokens < 0
            || delta.reasoning_tokens < 0
        {
            return Err(format!(
                "token ledger decreased for source/model ({source}, {model})"
            ));
        }
        if delta.total() == 0 && delta.cached_input_tokens == 0 {
            continue;
        }
        out.push(TokenLedgerEntry {
            source: source.to_string(),
            model: model.to_string(),
            usage: delta,
        });
    }
    Ok(out)
}

/// Serializable host-owned session envelope.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionStateEnvelope {
    pub session_id: String,
    #[serde(default)]
    pub policy: SessionPolicy,
    #[serde(default)]
    pub session_graph: crate::SessionGraph,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default)]
    pub iteration: usize,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_prompt_usage: Option<PromptUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_state_snapshot: Option<Vec<u8>>,
    /// Cost-accounting ledger. Every LLM call (parent turns, predict
    /// children, compaction, observers, agent_call) contributes an
    /// entry keyed by `(source, model)`. Separate from `token_usage`
    /// which tracks context-window accounting only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ledger: Vec<TokenLedgerEntry>,
}

impl SessionStateEnvelope {
    pub fn refresh_from_session_graph(&mut self) {
        normalize_session_graph(self);
    }

    pub fn stamp_runtime_state(
        &mut self,
        dynamic_state: Option<&crate::DynamicStateSnapshot>,
        plugin_snapshot: Option<&crate::PluginSessionSnapshot>,
    ) {
        stamp_session_graph_runtime_state(self, plugin_snapshot.cloned(), dynamic_state.cloned());
    }

    pub fn usage_report(&self) -> SessionUsageReport {
        SessionUsageReport::from_entries(&self.token_ledger)
    }
}

#[derive(Clone, Debug, Default)]
struct StandardStreamFallback {
    parts: Vec<LlmOutputPart>,
}

#[derive(Clone, Copy, Debug)]
struct LlmStreamDebugState {
    started_at: std::time::Instant,
    sequence: u64,
    summary: LlmStreamSummary,
}

#[derive(Clone, Copy)]
struct LlmDebugText<'a> {
    raw: Option<&'a str>,
    visible: Option<&'a str>,
}

#[derive(Clone, Copy)]
struct LlmDebugToolCall<'a> {
    call_id: &'a str,
    tool_name: &'a str,
    input_json: &'a str,
}

#[derive(Clone, Copy)]
struct LlmStreamEventLog<'a> {
    session_id: &'a str,
    iteration: usize,
    event_type: &'a str,
    text: LlmDebugText<'a>,
    usage: Option<&'a LlmUsage>,
    tool_call: Option<LlmDebugToolCall<'a>>,
}

struct StandardStreamState<'a> {
    text_streamed: &'a mut bool,
    streamed_usage: &'a mut LlmUsage,
    streamed_output: &'a mut StandardStreamFallback,
    debug: &'a mut LlmStreamDebugState,
    iteration: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct LlmStreamSummary {
    first_visible_token_latency_ms: Option<u64>,
    last_visible_chunk_latency_ms: Option<u64>,
    text_delta_count: u64,
    visible_chunk_count: u64,
    total_visible_chars: u64,
    max_visible_chunk_chars: u64,
}

impl LlmStreamDebugState {
    fn new() -> Self {
        Self {
            started_at: std::time::Instant::now(),
            sequence: 0,
            summary: LlmStreamSummary::default(),
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let sequence = self.sequence;
        self.sequence += 1;
        sequence
    }

    fn elapsed_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }
}

impl LlmStreamSummary {
    fn record_text_chunk(&mut self, visible_text: Option<&str>, elapsed_ms: u64) {
        self.text_delta_count += 1;

        let visible_chars = visible_text
            .map(|text| text.chars().count() as u64)
            .unwrap_or(0);
        if visible_chars == 0 {
            return;
        }

        if self.first_visible_token_latency_ms.is_none() {
            self.first_visible_token_latency_ms = Some(elapsed_ms);
        }
        self.last_visible_chunk_latency_ms = Some(elapsed_ms);
        self.visible_chunk_count += 1;
        self.total_visible_chars += visible_chars;
        self.max_visible_chunk_chars = self.max_visible_chunk_chars.max(visible_chars);
    }

    fn to_json(self) -> serde_json::Value {
        let avg_visible_chunk_chars = if self.visible_chunk_count == 0 {
            None
        } else {
            Some(self.total_visible_chars as f64 / self.visible_chunk_count as f64)
        };
        let stream_duration_ms = match (
            self.first_visible_token_latency_ms,
            self.last_visible_chunk_latency_ms,
        ) {
            (Some(first), Some(last)) => Some(last.saturating_sub(first)),
            _ => None,
        };
        serde_json::json!({
            "first_visible_token_latency_ms": self.first_visible_token_latency_ms,
            "stream_duration_ms": stream_duration_ms,
            "text_delta_count": self.text_delta_count,
            "visible_chunk_count": self.visible_chunk_count,
            "avg_visible_chunk_chars": avg_visible_chunk_chars,
            "max_visible_chunk_chars": if self.visible_chunk_count == 0 {
                serde_json::Value::Null
            } else {
                serde_json::Value::from(self.max_visible_chunk_chars)
            },
        })
    }
}

impl StandardStreamFallback {
    fn push_text(&mut self, piece: String) {
        if piece.is_empty() {
            return;
        }
        match self.parts.last_mut() {
            Some(LlmOutputPart::Text { text }) => append_stream_piece(text, &piece),
            _ => self.parts.push(LlmOutputPart::Text { text: piece }),
        }
    }

    fn push_tool_call(&mut self, call_id: String, tool_name: String, input_json: String) {
        self.parts.push(LlmOutputPart::ToolCall {
            call_id,
            tool_name,
            input_json,
        });
    }

    fn is_empty(&self) -> bool {
        !self.parts.iter().any(|part| match part {
            LlmOutputPart::Text { text } => !text.is_empty(),
            LlmOutputPart::ToolCall { .. } => true,
        })
    }

    fn full_text(&self) -> String {
        let mut full_text = String::new();
        for part in &self.parts {
            if let LlmOutputPart::Text { text } = part {
                full_text.push_str(text);
            }
        }
        full_text
    }

    fn apply_to_response(&self, response: &mut LlmResponse) {
        if llm_response_has_content(response) || self.is_empty() {
            return;
        }
        response.parts = self.parts.clone();
        if response.full_text.is_empty() {
            response.full_text = self.full_text();
        }
    }
}

fn append_stream_piece(full: &mut String, piece: &str) {
    if piece.is_empty() {
        return;
    }
    if piece.starts_with(full.as_str()) {
        full.push_str(&piece[full.len()..]);
    } else {
        full.push_str(piece);
    }
}

impl Default for SessionStateEnvelope {
    fn default() -> Self {
        Self {
            session_id: "root".to_string(),
            policy: SessionPolicy::default(),
            session_graph: crate::SessionGraph::default(),
            messages: Vec::new(),
            tool_calls: Vec::new(),
            iteration: 0,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
            execution_state_snapshot: None,
            token_ledger: Vec::new(),
        }
    }
}

fn normalize_session_graph(state: &mut SessionStateEnvelope) {
    if state.session_graph.nodes.is_empty() {
        state.session_graph =
            crate::SessionGraph::from_projection(&state.messages, &state.tool_calls);
    } else if !state.messages.is_empty() || !state.tool_calls.is_empty() {
        state
            .session_graph
            .merge_active_projection(&state.messages, &state.tool_calls);
    }
    if let Some(config) = state.session_graph.latest_session_config() {
        apply_persisted_session_config(&mut state.policy, &config);
    }
    if let Some(turn_state) = state.session_graph.latest_turn_state() {
        state.iteration = turn_state.iteration;
        state.token_usage = turn_state.token_usage;
        state.last_prompt_usage = turn_state.last_prompt_usage;
    }
    if let Some(token_ledger) = state.session_graph.latest_token_ledger() {
        state.token_ledger = token_ledger;
    }
    if let Some(execution_state_snapshot) = state.session_graph.latest_execution_state() {
        state.execution_state_snapshot = execution_state_snapshot;
    }
    state.messages = state.session_graph.project_messages();
    state.tool_calls = state.session_graph.project_tool_calls();
}

fn persisted_session_config(policy: &SessionPolicy) -> crate::PersistedSessionConfig {
    crate::PersistedSessionConfig {
        provider_id: policy.provider.id().to_string(),
        configured_model: policy.model.clone(),
        context_window: policy.max_context_tokens.unwrap_or_default() as u64,
        execution_mode: policy.execution_mode,
        context_approach: policy.context_approach.clone(),
        model_variant: policy.model_variant.clone(),
    }
}

fn apply_persisted_session_config(
    policy: &mut SessionPolicy,
    config: &crate::PersistedSessionConfig,
) {
    if !config.configured_model.is_empty() {
        policy.model = config.configured_model.clone();
    }
    if config.context_window > 0 {
        policy.max_context_tokens = Some(config.context_window as usize);
    }
    policy.execution_mode = config.execution_mode;
    policy.context_approach = config.context_approach.clone();
    policy.model_variant = config.model_variant.clone();
}

fn stamp_session_graph_runtime_state(
    state: &mut SessionStateEnvelope,
    plugin_snapshot: Option<crate::PluginSessionSnapshot>,
    dynamic_state: Option<crate::DynamicStateSnapshot>,
) {
    state
        .session_graph
        .merge_active_projection(&state.messages, &state.tool_calls);
    state.session_graph.record_runtime_state(
        &persisted_session_config(&state.policy),
        &crate::PersistedTurnState {
            iteration: state.iteration,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
        },
        dynamic_state.as_ref(),
        plugin_snapshot.as_ref(),
        state.execution_state_snapshot.as_deref(),
        &state.token_ledger,
    );
}

fn append_session_nodes_to_state(
    state: &mut SessionStateEnvelope,
    nodes: &[crate::SessionAppendNode],
) -> Vec<String> {
    state
        .session_graph
        .merge_active_projection(&state.messages, &state.tool_calls);
    state.messages.clear();
    state.tool_calls.clear();

    let mut node_ids = Vec::with_capacity(nodes.len());
    for node in nodes {
        match node {
            crate::SessionAppendNode::Message { message } => {
                let message = plugin_message_to_message(message, None);
                node_ids.push(state.session_graph.append_message(message));
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

async fn apply_graph_execution_mode_state(
    session: &mut Session,
    state: &SessionStateEnvelope,
) -> Result<(), SessionError> {
    if !matches!(state.policy.execution_mode, ExecutionMode::Rlm) {
        return Ok(());
    }
    for body in state
        .session_graph
        .active_path_plugins(crate::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE)
    {
        let patch = serde_json::from_value::<crate::RlmGlobalsPatchPluginBody>(body.clone())
            .map_err(|err| {
                SessionError::Protocol(format!("invalid RLM globals patch node: {err}"))
            })?;
        session.apply_rlm_globals_patch(&patch).await?;
    }
    Ok(())
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
    /// the lashlang program ended with `finish <expr>`, this is the
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

/// Host profile presets for runtime policies.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum HostProfile {
    #[default]
    Interactive,
    Embedded,
}

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

type LlmFactory = Arc<dyn Fn(&Provider) -> Box<dyn LlmTransport> + Send + Sync>;

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

fn merge_ledger_entry(ledger: &mut Vec<TokenLedgerEntry>, entry: TokenLedgerEntry) {
    if entry.usage.total() == 0 {
        return;
    }
    if let Some(existing) = ledger
        .iter_mut()
        .find(|e| e.source == entry.source && e.model == entry.model)
    {
        existing.usage.input_tokens += entry.usage.input_tokens;
        existing.usage.output_tokens += entry.usage.output_tokens;
        existing.usage.cached_input_tokens += entry.usage.cached_input_tokens;
        existing.usage.reasoning_tokens += entry.usage.reasoning_tokens;
    } else {
        ledger.push(entry);
    }
}

fn debug_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?.trim();
        let kb = value.split_whitespace().next()?.parse::<u64>().ok()?;
        Some(kb)
    })
}

fn rlm_termination_to_sansio(termination: &crate::RlmTermination) -> crate::sansio::RlmTermination {
    match termination {
        crate::RlmTermination::ProseWithoutFence => {
            crate::sansio::RlmTermination::ProseWithoutFence
        }
        crate::RlmTermination::Finish { schema } => crate::sansio::RlmTermination::Finish {
            schema: schema.clone(),
        },
    }
}

fn default_llm_factory() -> LlmFactory {
    Arc::new(|provider| adapter_for(provider))
}

fn normalize_prompt_usage(provider: &Provider, usage: &TokenUsage) -> Option<PromptUsage> {
    let input_tokens = usage.input_tokens.max(0) as usize;
    let output_tokens = usage.output_tokens.max(0) as usize;
    let cached_input_tokens = usage.cached_input_tokens.max(0) as usize;
    if input_tokens == 0 && cached_input_tokens == 0 && output_tokens == 0 {
        return None;
    }

    let prompt_context_tokens = if provider.input_usage_excludes_cached_tokens() {
        input_tokens.saturating_add(cached_input_tokens)
    } else {
        input_tokens
    };
    let adjusted_input_tokens = if provider.input_usage_excludes_cached_tokens() {
        input_tokens
    } else {
        input_tokens.saturating_sub(cached_input_tokens)
    };
    let context_budget_tokens = adjusted_input_tokens
        .saturating_add(output_tokens)
        .saturating_add(cached_input_tokens);

    Some(PromptUsage {
        prompt_context_tokens,
        input_tokens,
        cached_input_tokens,
        context_budget_tokens,
    })
}

#[derive(Default)]
struct TurnAssembler {
    final_message: Option<String>,
    text_deltas: String,
    tool_calls: Vec<ToolCallRecord>,
    token_usage: TokenUsage,
    last_llm_usage: Option<TokenUsage>,
    issues: Vec<TurnIssue>,
    saw_done: bool,
    saw_tool_failure: bool,
    has_plugin_visible_output: bool,
    typed_finish: Option<serde_json::Value>,
}

impl TurnAssembler {
    fn push(&mut self, event: &SessionEvent) {
        match event {
            SessionEvent::TextDelta { content } => {
                self.text_deltas.push_str(content);
            }
            SessionEvent::ToolCall {
                call_id,
                name,
                args,
                result,
                success,
                duration_ms,
            } => {
                self.tool_calls.push(ToolCallRecord {
                    call_id: call_id.clone(),
                    tool: name.clone(),
                    args: args.clone(),
                    result: result.clone(),
                    success: *success,
                    duration_ms: *duration_ms,
                });
                if !success {
                    self.saw_tool_failure = true;
                }
            }
            SessionEvent::Message { text, kind } if kind == "final" => {
                self.final_message = Some(text.clone());
            }
            SessionEvent::TokenUsage {
                usage, cumulative, ..
            } => {
                self.token_usage = cumulative.clone();
                self.last_llm_usage = Some(usage.clone());
            }
            SessionEvent::Error { message, envelope } => {
                let (kind, code) = if let Some(envelope) = envelope {
                    (envelope.kind.clone(), envelope.code.clone())
                } else {
                    ("runtime".to_string(), None)
                };
                self.issues.push(TurnIssue {
                    kind,
                    code,
                    message: message.clone(),
                });
            }
            SessionEvent::Done => {
                self.saw_done = true;
            }
            SessionEvent::TypedFinish { value } => {
                self.typed_finish = Some(value.clone());
            }
            SessionEvent::PluginEvent { event, .. } => {
                if plugin_surface_event_renders_visible_output(event) {
                    self.has_plugin_visible_output = true;
                }
            }
            _ => {}
        }
    }

    fn finish(
        mut self,
        state: SessionStateEnvelope,
        interrupted: bool,
        force_runtime_error: Option<TurnIssue>,
        sanitizer: &SanitizerPolicy,
        termination: &TerminationPolicy,
    ) -> AssembledTurn {
        let mut issues = self.issues;
        if let Some(issue) = force_runtime_error {
            issues.push(issue);
        }
        let max_turn_reached = state.messages.iter().rev().take(8).any(|msg| {
            msg.role == MessageRole::System
                && msg
                    .parts
                    .iter()
                    .any(|part| part.content.contains("Turn limit reached ("))
        });

        let raw_output = if let Some(final_message) = self.final_message {
            final_message
        } else {
            let streamed = self.text_deltas.trim().to_string();
            let state_output = fallback_assistant_output_from_state(&state);
            if streamed.is_empty()
                || (!state_output.is_empty()
                    && state_output.len() >= streamed.len()
                    && state_output.starts_with(&streamed))
            {
                state_output
            } else {
                streamed
            }
        };
        let safe_output = sanitize_assistant_output(raw_output.clone(), sanitizer);
        let output_state = classify_output_state(&raw_output, &safe_output, &issues);

        let (status, done_reason) = if interrupted {
            (TurnStatus::Interrupted, DoneReason::UserAbort)
        } else if !self.saw_done && termination.treat_missing_done_as_failure {
            (TurnStatus::Failed, DoneReason::RuntimeError)
        } else if !issues.is_empty() {
            if self.saw_tool_failure {
                (TurnStatus::Failed, DoneReason::ToolFailure)
            } else {
                (TurnStatus::Failed, DoneReason::RuntimeError)
            }
        } else if max_turn_reached {
            (TurnStatus::Completed, DoneReason::MaxTurns)
        } else {
            (TurnStatus::Completed, DoneReason::ModelStop)
        };

        AssembledTurn {
            execution: ExecutionSummary {
                mode: state.policy.execution_mode,
                had_tool_calls: !self.tool_calls.is_empty(),
                had_code_execution: false,
            },
            state,
            status,
            assistant_output: AssistantOutput {
                safe_text: safe_output,
                raw_text: raw_output,
                state: output_state,
            },
            has_plugin_visible_output: self.has_plugin_visible_output,
            done_reason,
            token_usage: self.token_usage,
            tool_calls: self.tool_calls,
            errors: issues,
            typed_finish: self.typed_finish.take(),
        }
    }

    fn last_llm_usage(&self) -> Option<&TokenUsage> {
        self.last_llm_usage.as_ref()
    }
}

fn fallback_assistant_output_from_state(state: &SessionStateEnvelope) -> String {
    state
        .messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::Assistant)
        .map(|message| {
            message
                .parts
                .iter()
                .map(|part| part.content.as_str())
                .collect::<String>()
        })
        .unwrap_or_default()
}

/// Host-owned runtime knobs that are not part of the session contract.
#[derive(Clone)]
pub struct RuntimeHostConfig {
    pub host_profile: HostProfile,
    pub user_prompts_enabled: bool,
    pub base_dir: Option<PathBuf>,
    pub path_resolver: Option<Arc<dyn PathResolver>>,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub prompt_renderer: Arc<dyn PromptRenderer>,
    pub prompt_overrides: Vec<PromptSectionOverride>,
    pub llm_log_path: Option<PathBuf>,
    pub sanitizer: SanitizerPolicy,
    pub termination: TerminationPolicy,
}

impl Default for RuntimeHostConfig {
    fn default() -> Self {
        Self {
            host_profile: HostProfile::Interactive,
            user_prompts_enabled: true,
            base_dir: None,
            path_resolver: None,
            session_store_factory: None,
            prompt_renderer: crate::default_prompt_renderer(),
            prompt_overrides: Vec::new(),
            llm_log_path: None,
            sanitizer: SanitizerPolicy::default(),
            termination: TerminationPolicy::default(),
        }
    }
}

/// Generic runtime for CLI or programmatic embedding.
pub struct LashRuntime {
    session: Option<Session>,
    policy: SessionPolicy,
    host: RuntimeHostConfig,
    services: RuntimeServices,
    state: SessionStateEnvelope,
    llm_factory: LlmFactory,
    managed_sessions: Arc<Mutex<HashMap<String, Arc<Mutex<LashRuntime>>>>>,
    managed_turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    overflow_recovery_attempted: bool,
    /// RLM termination contract for this session.
    rlm_termination: crate::RlmTermination,
    /// Session-scoped token cost ledger. Shared by ALL
    /// `RuntimeSessionManager` instances created from this runtime
    /// (both per-turn and async maintenance). Entries accumulate here
    /// and are drained into `state.token_ledger` at turn-commit time.
    shared_token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    background_jobs:
        Arc<Mutex<HashMap<String, Vec<tokio::task::JoinHandle<Result<(), crate::PluginError>>>>>>,
}

impl LashRuntime {
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
        self.session
            .as_ref()
            .map(|session| session.tool_catalog(&self.state.session_id, self.policy.execution_mode))
            .unwrap_or_default()
    }

    /// Build a runtime from host-provided state + tools.
    pub async fn from_state(
        policy: SessionPolicy,
        mut host: RuntimeHostConfig,
        services: RuntimeServices,
        mut state: SessionStateEnvelope,
    ) -> Result<Self, SessionError> {
        host.base_dir = Some(
            host.base_dir
                .clone()
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        );
        if state.session_id.is_empty() {
            state.session_id = "root".to_string();
        }
        if state.policy == SessionPolicy::default() {
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
        if let Some(dynamic_state) = state.session_graph.latest_dynamic_state()
            && let Some(dynamic_tools) = session.plugins().dynamic_tools()
            && let Err(err) = dynamic_tools.apply_state(dynamic_state)
        {
            tracing::warn!("failed to restore dynamic tool state from graph: {err}");
        }
        if let Some(snapshot) = state.session_graph.latest_plugin_snapshot() {
            session
                .plugins()
                .restore(&snapshot)
                .map_err(|err| SessionError::Protocol(err.to_string()))?;
        }
        if matches!(state.policy.execution_mode, ExecutionMode::Rlm)
            && let Some(snapshot) = state.execution_state_snapshot.clone()
        {
            session.restore_execution_state(&snapshot).await?;
        }
        apply_graph_execution_mode_state(&mut session, &state).await?;
        session
            .plugins()
            .emit_runtime_event(crate::PluginRuntimeEvent::SessionRestored(state.clone()))
            .await;
        Ok(Self {
            session: Some(session),
            policy,
            host,
            services,
            state,
            llm_factory: default_llm_factory(),
            managed_sessions: Arc::new(Mutex::new(HashMap::new())),
            managed_turns: Arc::new(Mutex::new(HashMap::new())),
            overflow_recovery_attempted: false,
            rlm_termination: crate::RlmTermination::default(),
            shared_token_ledger: Arc::new(std::sync::Mutex::new(Vec::new())),
            background_jobs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Override the RLM termination contract for this session. Defaults
    /// to `ProseWithoutFence` (today's chat-style behavior). Sub-sessions
    /// spawned via the typed `predict` path call this with
    /// `Finish { schema }` to require typed termination.
    pub(crate) fn set_repl_termination(&mut self, termination: crate::RlmTermination) {
        self.rlm_termination = termination;
    }

    /// Export current host-owned state envelope.
    pub fn export_state(&self) -> SessionStateEnvelope {
        let mut state = self.state.clone();
        if let Some(session) = self.session.as_ref() {
            let dynamic_state = session
                .plugins()
                .dynamic_tools()
                .map(|tools| tools.export_state());
            let plugin_snapshot = session.plugins().snapshot().ok();
            state.stamp_runtime_state(dynamic_state.as_ref(), plugin_snapshot.as_ref());
        }
        state.refresh_from_session_graph();
        state
    }

    pub async fn await_background_work(&mut self) -> Result<(), SessionError> {
        loop {
            let jobs = {
                let mut jobs = self.background_jobs.lock().await;
                jobs.remove(&self.state.session_id).unwrap_or_default()
            };
            if jobs.is_empty() {
                self.refresh_session_graph_from_store().await;
                return Ok(());
            }
            for job in jobs {
                match job.await {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => {
                        self.refresh_session_graph_from_store().await;
                        return Err(SessionError::Protocol(format!(
                            "background job failed: {err}"
                        )));
                    }
                    Err(err) => {
                        self.refresh_session_graph_from_store().await;
                        return Err(SessionError::Protocol(format!(
                            "background job task failed: {err}"
                        )));
                    }
                }
            }
        }
    }

    async fn refresh_session_graph_from_store(&mut self) {
        let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        else {
            return;
        };
        let Some(graph) = store.load_session_graph().await else {
            return;
        };
        let current = &self.state.session_graph;
        let has_newer_graph =
            graph.nodes.len() > current.nodes.len() || graph.leaf_node_id != current.leaf_node_id;
        if !has_newer_graph {
            return;
        }
        self.state.session_graph = graph;
        normalize_session_graph(&mut self.state);
    }

    fn runtime_session_manager(&self) -> Result<Arc<dyn SessionManager>, ExternalInvokeError> {
        self.runtime_session_manager_with_prompt_bridge(None)
    }

    fn runtime_session_manager_for_turn(
        &self,
        prompt_bridge: Option<HostPromptBridge>,
    ) -> Result<Arc<dyn SessionManager>, ExternalInvokeError> {
        Ok(Arc::new(RuntimeSessionManager::new(
            self,
            prompt_bridge,
            false,
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
            state: self.state.clone(),
            host: manager,
        };
        let input = crate::HistoryState::from_state(&self.state);
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
            self.state.messages = outcome.messages;
            self.state.tool_calls = outcome.tool_calls;
            self.state.session_graph =
                crate::SessionGraph::from_projection(&self.state.messages, &self.state.tool_calls);
            if let Some(session) = self.session.as_ref() {
                let dynamic_state = session
                    .plugins()
                    .dynamic_tools()
                    .map(|tools| tools.export_state());
                let plugin_snapshot = session.plugins().snapshot().ok();
                stamp_session_graph_runtime_state(&mut self.state, plugin_snapshot, dynamic_state);
            } else {
                self.state.refresh_from_session_graph();
            }
        }
        Ok(mutated)
    }

    fn session_policy(&self) -> SessionPolicy {
        self.policy.clone()
    }

    async fn notify_session_config_changed(&self, previous: SessionPolicy) {
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
            .emit_runtime_event(crate::PluginRuntimeEvent::SessionConfigChanged(
                SessionConfigChangedContext {
                    session_id: self.state.session_id.clone(),
                    previous,
                    current,
                    host,
                },
            ))
            .await;
    }

    async fn apply_session_config_mutations(&mut self, previous: SessionPolicy) {
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
        self.state = session
            .plugins()
            .mutate_session_config(
                SessionConfigChangedContext {
                    session_id: self.state.session_id.clone(),
                    previous,
                    current,
                    host,
                },
                self.state.clone(),
            )
            .await;
        normalize_session_graph(&mut self.state);
    }

    /// Replace the host-owned state envelope.
    pub fn set_state(&mut self, state: SessionStateEnvelope) {
        let mut state = state;
        normalize_session_graph(&mut state);
        if let Some(session) = self.session.as_ref() {
            let snapshot = state
                .session_graph
                .latest_plugin_snapshot()
                .unwrap_or_default();
            if let Err(err) = session.plugins().restore(&snapshot) {
                tracing::warn!("failed to restore plugin snapshot in set_state: {err}");
            }
        }
        self.policy = state.policy.clone();
        self.state = state;
    }

    pub async fn append_session_nodes(
        &mut self,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, SessionError> {
        self.refresh_session_graph_from_store().await;
        if let Some(required) = request.requires_ancestor_node_id.as_deref()
            && !self.state.session_graph.active_path_contains(required)
        {
            return Ok(crate::AppendSessionNodesResult::StaleBranch {
                current_leaf_node_id: self.state.session_graph.leaf_node_id.clone(),
            });
        }
        let node_ids = append_session_nodes_to_state(&mut self.state, &request.nodes);
        if matches!(self.state.policy.execution_mode, ExecutionMode::Rlm)
            && let Some(session) = self.session.as_mut()
        {
            for node in &request.nodes {
                let crate::SessionAppendNode::Plugin { plugin_type, body } = node else {
                    continue;
                };
                if plugin_type != crate::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE {
                    continue;
                }
                let patch =
                    serde_json::from_value::<crate::RlmGlobalsPatchPluginBody>(body.clone())
                        .map_err(|err| {
                            SessionError::Protocol(format!(
                                "invalid RLM globals patch node body: {err}"
                            ))
                        })?;
                session.apply_rlm_globals_patch(&patch).await?;
            }
        }
        if let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        {
            store
                .save_session_graph(self.state.session_graph.clone())
                .await;
        }
        Ok(crate::AppendSessionNodesResult::Appended {
            node_ids,
            leaf_node_id: self
                .state
                .session_graph
                .leaf_node_id
                .clone()
                .unwrap_or_default(),
        })
    }

    pub async fn branch_to_node(
        &mut self,
        node_id: Option<String>,
    ) -> Result<SessionStateEnvelope, SessionError> {
        let mut state = self.export_state();
        state.session_graph.branch_to(node_id);
        normalize_session_graph(&mut state);

        let policy = state.policy.clone();
        let host = self.host.clone();
        let services = self.services.clone();
        let llm_factory = Arc::clone(&self.llm_factory);
        let managed_sessions = Arc::clone(&self.managed_sessions);
        let managed_turns = Arc::clone(&self.managed_turns);

        let mut rebuilt = Self::from_state(policy, host, services, state).await?;
        rebuilt.llm_factory = llm_factory;
        rebuilt.managed_sessions = managed_sessions;
        rebuilt.managed_turns = managed_turns;

        let exported = rebuilt.export_state();
        *self = rebuilt;
        Ok(exported)
    }

    /// Update model on the runtime config.
    pub fn set_model(&mut self, model: String) {
        self.policy.model = model;
        self.state.policy.model = self.policy.model.clone();
    }

    /// Update model variant on the runtime config.
    pub fn set_model_variant(&mut self, model_variant: Option<String>) {
        self.policy.model_variant = model_variant;
        self.state.policy.model_variant = self.policy.model_variant.clone();
    }

    /// Update explicit model context metadata on the runtime config.
    pub fn set_max_context_tokens(&mut self, max_context_tokens: usize) {
        self.policy.max_context_tokens = Some(max_context_tokens);
        self.state.policy.max_context_tokens = self.policy.max_context_tokens;
    }

    /// Update provider on the runtime config.
    pub fn set_provider(&mut self, provider: Provider) {
        self.policy.provider = provider;
        self.state.policy.provider = self.policy.provider.clone();
    }

    /// Update session ID metadata on the runtime config.
    pub fn set_session_id(&mut self, session_id: Option<String>) {
        self.policy.session_id = session_id;
        self.state.policy.session_id = self.policy.session_id.clone();
    }

    pub async fn update_session_config(
        &mut self,
        provider: Option<Provider>,
        model: Option<String>,
        model_variant: Option<Option<String>>,
        max_context_tokens: Option<usize>,
    ) {
        let previous = self.session_policy();
        if let Some(provider) = provider {
            self.policy.provider = provider;
        }
        if let Some(model) = model {
            self.policy.model = model;
        }
        if let Some(model_variant) = model_variant {
            self.policy.model_variant = model_variant;
        }
        if let Some(max_context_tokens) = max_context_tokens {
            self.policy.max_context_tokens = Some(max_context_tokens);
        }
        self.state.policy = self.policy.clone();
        // Eagerly compact messages if the context window shrunk.
        let new_max = self.policy.max_context_tokens;
        let old_max = previous.max_context_tokens;
        if new_max < old_max || (new_max.is_some() && old_max.is_none()) {
            let rewrite_result = self
                .rewrite_history(crate::RewriteTrigger::WindowShrink { old_max, new_max })
                .await;
            eprintln!("[update_session_config] rewrite_history result: {rewrite_result:?}");
        }
        self.apply_session_config_mutations(previous.clone()).await;
        self.notify_session_config_changed(previous).await;
    }

    /// Re-register the current execution surface in the live RLM session.
    pub async fn refresh_session_execution_surface(&mut self) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        session.refresh_execution_surface().await
    }

    /// Reset the RLM session on the underlying session runtime.
    pub async fn reset_session(&mut self) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        session.reset().await
    }

    /// Explicitly snapshot execution-mode-local state, if any.
    pub async fn snapshot_execution_state(&mut self) -> Result<Option<Vec<u8>>, SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let blob = session.snapshot_execution_state().await?;
        self.state.execution_state_snapshot = blob.clone();
        Ok(blob)
    }

    /// Explicitly restore execution-mode-local state from an opaque snapshot blob.
    pub async fn restore_execution_state(&mut self, snapshot: &[u8]) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        session.restore_execution_state(snapshot).await?;
        self.state.execution_state_snapshot = Some(snapshot.to_vec());
        Ok(())
    }

    pub async fn invoke_external(
        &self,
        name: &str,
        args: serde_json::Value,
        session_id: Option<String>,
    ) -> Result<crate::ToolResult, ExternalInvokeError> {
        let manager = self.runtime_session_manager()?;
        let Some(session) = self.session.as_ref() else {
            return Err(ExternalInvokeError::Unknown(name.to_string()));
        };
        session
            .plugins()
            .invoke_external(name, args, session_id, true, manager)
            .await
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
        let saved_messages = self.state.messages.clone();
        let saved_prompt_usage = self.state.last_prompt_usage.clone();

        let assembled = self
            .stream_turn_inner(input.clone(), events, cancel.clone())
            .await?;

        if !self.overflow_recovery_attempted && Self::has_overflow_error(&assembled) {
            self.overflow_recovery_attempted = true;
            // Restore pre-turn state so the retry appends the user message cleanly.
            self.state.messages = saved_messages;
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
                    self.state.clone(),
                    false,
                    None,
                    &self.host.sanitizer,
                    &self.host.termination,
                ));
            }
        };

        let mut messages = self.state.messages.clone();
        let mode = input.mode.unwrap_or(match self.host.host_profile {
            HostProfile::Interactive | HostProfile::Embedded => RunMode::Normal,
        });
        let mode_msg = match mode {
            RunMode::Normal => None,
        };
        if let Some(content) = mode_msg {
            let sys_id = fresh_message_id();
            messages.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Text,
                    content,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
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
                        prune_state: PruneState::Intact,
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
                        prune_state: PruneState::Intact,
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
                prune_state: PruneState::Intact,
            });
        }
        reassign_part_ids(&user_id, &mut user_parts);
        messages.push(Message {
            id: user_id.clone(),
            role: MessageRole::User,
            parts: user_parts,
            user_input: input.user_input.clone(),
            origin: None,
        });

        let manager = self
            .runtime_session_manager_for_turn(None)
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
            state: self.export_state(),
            prompt_usage: previous_prompt_usage.clone(),
            max_context_tokens: Some(LashRuntime::max_context_tokens(self)),
            host: Arc::clone(&manager),
        };
        let prepared_context = plugin_session
            .prepare_turn_context(
                &turn_ctx,
                crate::session_model::context::PreparedContext {
                    messages,
                    ..Default::default()
                },
            )
            .await
            .map_err(|err| RuntimeError {
                code: "context_prepare_turn".to_string(),
                message: err.to_string(),
            })?;
        let messages = prepared_context.messages;
        if let Some(session) = self.session.as_mut() {
            session.set_context_surface(
                prepared_context.tool_providers,
                prepared_context.prompt_contributions,
                Vec::new(),
                prepared_context.include_base_tools,
            );
        }

        self.state.last_prompt_usage = None;

        self.stream_prepared_turn(messages, previous_prompt_usage, events, cancel)
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
        messages: Vec<Message>,
        _previous_prompt_usage: Option<PromptUsage>,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let prompt_bridge = HostPromptBridge::new();
        let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(100);
        let manager = self
            .runtime_session_manager_for_turn(Some(prompt_bridge.clone()))
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
                                emit_plugin_surface_events(
                                    &prompt_event_tx,
                                    &surface.plugin_id,
                                    vec![surface.value],
                                )
                                .await;
                            }
                        }
                        Err(err) => {
                            let _ = prompt_event_tx
                                .send(make_error_event(
                                    "plugin_prompt_request",
                                    None,
                                    err.to_string(),
                                    Some(err.to_string()),
                                ))
                                .await;
                        }
                    }
                }
                if !prompt_event_tx.is_closed() {
                    let _ = prompt_event_tx
                        .send(SessionEvent::Prompt {
                            request: prompt.request,
                            response_tx: prompt.response_tx,
                        })
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
        let prepare_turn = plugins.prepare_turn(PrepareTurnRequest {
            session_id: self.state.session_id.clone(),
            state: self.export_state(),
            messages,
            host: Arc::clone(&manager),
        });
        tokio::pin!(prepare_turn);

        let prepared = loop {
            tokio::select! {
                prepared = &mut prepare_turn => {
                    break prepared.map_err(|err| RuntimeError {
                        code: "plugin_prepare_turn".to_string(),
                        message: err.to_string(),
                    })?;
                }
                maybe_event = event_rx.recv() => {
                    if let Some(event) = maybe_event {
                        assembler.push(&event);
                        events.emit(event).await;
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
            state.messages = prepared.messages;
            let issue = TurnIssue {
                kind: "plugin".to_string(),
                code: Some(abort.code),
                message: abort.message.clone(),
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
                state,
                cancel.is_cancelled(),
                Some(issue),
                &self.host.sanitizer,
                &self.host.termination,
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
            tool_calls: self.state.tool_calls.clone(),
            llm_stream_summaries: HashMap::new(),
            llm_factory: Arc::clone(&self.llm_factory),
            session_manager: manager,
            prompt_bridge,
            rlm_termination: self.rlm_termination.clone(),
        };
        let run_offset = self.state.iteration;
        let run_task = tokio::spawn(async move {
            let (new_messages, new_iteration) = driver
                .run(prepared.messages, event_tx, cancel, run_offset)
                .await;
            (driver, new_messages, new_iteration)
        });

        while let Some(event) = event_rx.recv().await {
            assembler.push(&event);
            events.emit(event).await;
        }
        let _ = prompt_forward.await;

        let (driver, new_messages, new_iteration) = match run_task.await {
            Ok(v) => v,
            Err(e) => {
                let issue = TurnIssue {
                    kind: "runtime".to_string(),
                    code: Some("run_task_join_failed".to_string()),
                    message: format!("Runtime turn task failed: {e}"),
                };
                return Ok(assembler.finish(
                    self.state.clone(),
                    cancel_state.is_cancelled(),
                    Some(issue),
                    &self.host.sanitizer,
                    &self.host.termination,
                ));
            }
        };
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            new_message_count = new_messages.len(),
            tool_call_count = self.state.tool_calls.len(),
            "runtime post-run_task"
        );

        // Drain the shared token ledger (child sessions + direct
        // completions + async OM observers/reflectors) and merge into
        // the session state. Also record the parent's own turn usage.
        let child_ledger = {
            let mut ledger = self.shared_token_ledger.lock().expect("token ledger lock");
            std::mem::take(&mut *ledger)
        };
        for entry in child_ledger {
            merge_ledger_entry(&mut self.state.token_ledger, entry);
        }
        if assembler.token_usage.total() > 0 {
            merge_ledger_entry(
                &mut self.state.token_ledger,
                TokenLedgerEntry {
                    source: "turn".to_string(),
                    model: driver.policy.model.clone(),
                    usage: assembler.token_usage.clone(),
                },
            );
        }

        self.session = Some(driver.session);
        self.policy = driver.policy;
        self.state.policy = self.policy.clone();
        self.state.messages = new_messages;
        self.state.iteration = new_iteration;
        if !assembler.tool_calls.is_empty() {
            self.state.tool_calls.extend(assembler.tool_calls.clone());
        }
        if assembler.token_usage.total() > 0 {
            self.state.token_usage = assembler.token_usage.clone();
        }

        let last_prompt_usage = assembler
            .last_llm_usage()
            .and_then(|usage| normalize_prompt_usage(&self.policy.provider, usage));
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            state_message_count = self.state.messages.len(),
            graph_node_count = self.state.session_graph.nodes.len(),
            token_ledger_entries = self.state.token_ledger.len(),
            "runtime before assembler.finish"
        );
        let mut assembled = assembler.finish(
            self.state.clone(),
            cancel_state.is_cancelled(),
            None,
            &self.host.sanitizer,
            &self.host.termination,
        );
        assembled.state.last_prompt_usage = last_prompt_usage.clone();
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            assembled_message_count = assembled.state.messages.len(),
            assembled_graph_node_count = assembled.state.session_graph.nodes.len(),
            "runtime after assembler.finish"
        );
        if let Some(session) = self.session.as_ref() {
            let plugins = Arc::clone(session.plugins());
            let manager =
                self.runtime_session_manager_for_turn(None)
                    .map_err(|err| RuntimeError {
                        code: "plugin_session_manager".to_string(),
                        message: err.to_string(),
                    })?;
            tracing::debug!(rss_kb = debug_rss_kb(), "runtime before finalize_turn");
            let finalized = plugins
                .finalize_turn(assembled, manager)
                .await
                .map_err(|err| RuntimeError {
                    code: "plugin_finalize_turn".to_string(),
                    message: err.to_string(),
                })?;
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                finalized_message_count = finalized.turn.state.messages.len(),
                "runtime after finalize_turn"
            );
            let mut persisted_state = finalized.turn.state.clone();
            let dynamic_state = plugins.dynamic_tools().map(|tools| tools.export_state());
            let plugin_snapshot = plugins.snapshot().ok();
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                dynamic_state_present = dynamic_state.is_some(),
                plugin_snapshot_present = plugin_snapshot.is_some(),
                "runtime before stamp_runtime_state"
            );
            persisted_state.stamp_runtime_state(dynamic_state.as_ref(), plugin_snapshot.as_ref());
            persisted_state.refresh_from_session_graph();
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                persisted_graph_node_count = persisted_state.session_graph.nodes.len(),
                persisted_message_count = persisted_state.messages.len(),
                "runtime after stamp_runtime_state"
            );
            if let Some(store) = self
                .session
                .as_ref()
                .and_then(|session| session.history_store())
            {
                store
                    .save_turn_checkpoint(persisted_state.session_graph.clone())
                    .await;
            }
            emit_session_events_to_sink(events, finalized.events).await;
            let mut returned_turn = finalized.turn;
            returned_turn.state = persisted_state.clone();
            self.state = persisted_state;
            if let Some(session) = self.session.as_ref()
                && let Ok(host) = self.runtime_session_manager()
            {
                session
                    .plugins()
                    .emit_runtime_event(crate::PluginRuntimeEvent::TurnPersisted(
                        crate::SessionStateChangedContext {
                            session_id: self.state.session_id.clone(),
                            state: self.export_state(),
                            host,
                        },
                    ))
                    .await;
            }
            Ok(returned_turn)
        } else {
            let mut next_state = assembled.state.clone();
            next_state
                .session_graph
                .merge_active_projection(&next_state.messages, &next_state.tool_calls);
            normalize_session_graph(&mut next_state);
            self.state = next_state;
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
            self.host
                .base_dir
                .as_deref()
                .unwrap_or_else(|| Path::new(".")),
            self.host.path_resolver.as_deref(),
        )
    }
}

fn normalize_input_items(
    items: &[InputItem],
    image_blobs: &HashMap<String, Vec<u8>>,
    base_dir: &Path,
    path_resolver: Option<&dyn PathResolver>,
) -> Result<Vec<NormalizedItem>, String> {
    let mut out: Vec<NormalizedItem> = Vec::new();
    for item in items {
        match item {
            InputItem::Text { text } => push_text(&mut out, text.clone()),
            InputItem::FileRef { path } => {
                let abs = resolve_existing_path(path, true, base_dir, path_resolver)?;
                push_text(&mut out, format!("[file: {}]", abs.display()));
            }
            InputItem::DirRef { path } => {
                let abs = resolve_existing_path(path, false, base_dir, path_resolver)?;
                push_text(
                    &mut out,
                    format!(
                        "[directory: {}]",
                        abs.to_string_lossy().trim_end_matches('/')
                    ),
                );
            }
            InputItem::ImageRef { id } => {
                if id.is_empty() {
                    return Err("Invalid image_ref: id cannot be empty".to_string());
                }
                let Some(blob) = image_blobs.get(id) else {
                    return Err(format!("Invalid image_ref: missing blob for id '{id}'"));
                };
                out.push(NormalizedItem::Image(blob.clone()));
            }
        }
    }
    Ok(out)
}

fn push_text(out: &mut Vec<NormalizedItem>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(NormalizedItem::Text(last)) = out.last_mut() {
        last.push_str(&text);
    } else {
        out.push(NormalizedItem::Text(text));
    }
}

fn resolve_existing_path(
    path: &str,
    expect_file: bool,
    base_dir: &Path,
    path_resolver: Option<&dyn PathResolver>,
) -> Result<PathBuf, String> {
    if let Some(resolver) = path_resolver {
        return resolver.resolve(path, expect_file, base_dir);
    }
    if path.is_empty() {
        return Err("Path reference cannot be empty".to_string());
    }
    let p = Path::new(path);
    let candidate = if p.is_absolute() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    };
    if !candidate.exists() {
        return Err(format!(
            "Referenced path does not exist: {}",
            candidate.display()
        ));
    }
    if expect_file && !candidate.is_file() {
        return Err(format!(
            "Referenced path is not a file: {}",
            candidate.display()
        ));
    }
    if !expect_file && !candidate.is_dir() {
        return Err(format!(
            "Referenced path is not a directory: {}",
            candidate.display()
        ));
    }
    candidate
        .canonicalize()
        .map_err(|e| format!("Failed to canonicalize {}: {e}", candidate.display()))
}

fn sanitize_assistant_output(text: String, _policy: &SanitizerPolicy) -> String {
    text.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join(
            "
",
        )
        .trim()
        .to_string()
}
fn classify_output_state(raw_text: &str, safe_text: &str, issues: &[TurnIssue]) -> OutputState {
    if safe_text.is_empty() && raw_text.is_empty() {
        return OutputState::EmptyOutput;
    }
    if safe_text.is_empty() && contains_traceback_only(raw_text) {
        return OutputState::TracebackOnly;
    }
    if !issues.is_empty() && !safe_text.is_empty() {
        return OutputState::RecoveredFromError;
    }
    OutputState::Usable
}

fn contains_traceback_only(raw_text: &str) -> bool {
    if raw_text.is_empty() {
        return false;
    }
    let has_traceback = raw_text.contains("Traceback (most recent call last)")
        || raw_text.lines().any(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("Runtime error:")
                || trimmed.starts_with("NameError:")
                || trimmed.starts_with("TypeError:")
                || trimmed.starts_with("ValueError:")
                || trimmed.starts_with("KeyError:")
                || trimmed.starts_with("AttributeError:")
                || trimmed.starts_with("SyntaxError:")
                || trimmed.starts_with("ImportError:")
                || trimmed.starts_with("ModuleNotFoundError:")
        });
    if !has_traceback {
        return false;
    }
    // If no alphabetic prose besides traceback/exception formatting, treat as traceback-only.
    !raw_text.lines().any(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("Traceback")
            || trimmed.starts_with("File ")
            || trimmed.starts_with("Runtime error:")
        {
            return false;
        }
        !trimmed.contains(':')
    })
}
