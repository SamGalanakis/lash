use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::agent::message::IMAGE_REF_PREFIX;
use crate::agent::message::MessageOrigin;
use crate::agent::{
    AgentConfig, AgentEvent, Message, MessageRole, Part, PartKind, PromptSectionOverride,
    PruneState, TokenUsage, build_execution_preamble, make_error_event, transport_stream_events,
};
use crate::capabilities::AgentCapabilities;
use crate::instructions::{FsInstructionSource, InstructionSource};
use crate::llm::factory::adapter_for;
use crate::llm::transport::LlmTransport;
use crate::llm::types::{LlmOutputPart, LlmRequest, LlmResponse, LlmStreamEvent, LlmUsage};
use crate::provider::Provider;
use crate::sansio::{Effect, LlmCallError, Response, TurnMachine, TurnMachineConfig};
use crate::strip_repl_fragments;
use crate::{
    CapabilityId, ContextFoldingConfig, ExecutionMode, ExternalInvokeError, PluginMessage,
    PluginSessionSnapshot, PromptHookContext, RuntimeServices, SandboxMessage, Session,
    SessionConfigOverrides, SessionCreateRequest, SessionError, SessionHandle, SessionManager,
    SessionSnapshot, SessionStartPoint, ToolCallRecord, TurnHookContext, TurnResultHookContext,
};

/// Runtime execution mode for a turn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RunMode {
    Normal,
    Plan,
}

/// Host-provided per-turn input.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Text { text: String },
    FileRef { path: String },
    DirRef { path: String },
    ImageRef { id: String },
    SkillRef { name: String, args: Option<String> },
}

/// Host-provided per-turn input.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnInput {
    pub items: Vec<InputItem>,
    #[serde(default)]
    pub image_blobs: HashMap<String, Vec<u8>>,
    #[serde(default)]
    pub mode: Option<RunMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_file: Option<String>,
}

#[derive(Clone, Debug)]
enum NormalizedItem {
    Text(String),
    Image(Vec<u8>),
}

/// Serializable host-owned session envelope.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgentStateEnvelope {
    pub agent_id: String,
    #[serde(default)]
    pub execution_mode: ExecutionMode,
    #[serde(default)]
    pub context_folding: ContextFoldingConfig,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub iteration: usize,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_state: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_state: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_manifest: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_snapshot: Option<PluginSessionSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repl_snapshot: Option<Vec<u8>>,
}

impl Default for AgentStateEnvelope {
    fn default() -> Self {
        Self {
            agent_id: "root".to_string(),
            execution_mode: crate::default_execution_mode(),
            context_folding: ContextFoldingConfig::default(),
            messages: Vec::new(),
            iteration: 0,
            token_usage: TokenUsage::default(),
            task_state: None,
            subagent_state: None,
            replay_manifest: None,
            plugin_snapshot: None,
            repl_snapshot: None,
        }
    }
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
    Sanitized,
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

/// REPL code execution output observed during a turn.
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
/// This contract is stable across execution modes; mode-specific detail is summarized in
/// `execution`, while REPL-only detail is exposed through `code_outputs`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssembledTurn {
    pub state: AgentStateEnvelope,
    pub status: TurnStatus,
    pub assistant_output: AssistantOutput,
    pub done_reason: DoneReason,
    pub execution: ExecutionSummary,
    #[serde(default)]
    pub token_usage: TokenUsage,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default)]
    pub code_outputs: Vec<CodeOutputRecord>,
    #[serde(default)]
    pub errors: Vec<TurnIssue>,
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
    Headless,
    Embedded,
}

/// Pluggable path resolver for file and directory references.
pub trait PathResolver: Send + Sync {
    fn resolve(&self, path: &str, expect_file: bool, base_dir: &Path) -> Result<PathBuf, String>;
}

/// Sanitization policy knobs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SanitizerPolicy {
    #[serde(default = "SanitizerPolicy::default_strip_repl_fragments")]
    pub strip_repl_fragments: bool,
}

impl SanitizerPolicy {
    fn default_strip_repl_fragments() -> bool {
        true
    }
}

impl Default for SanitizerPolicy {
    fn default() -> Self {
        Self {
            strip_repl_fragments: true,
        }
    }
}

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
/// `AgentEvent` is intentionally mode-specific and should be treated as preview/progress data.
#[async_trait::async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: AgentEvent);
}

/// No-op sink useful for callers that only care about final state.
pub struct NoopEventSink;

#[async_trait::async_trait]
impl EventSink for NoopEventSink {
    async fn emit(&self, _event: AgentEvent) {}
}

type LlmFactory = Arc<dyn Fn(&Provider) -> Box<dyn LlmTransport> + Send + Sync>;

fn default_llm_factory() -> LlmFactory {
    Arc::new(|provider| adapter_for(provider))
}

#[derive(Default)]
struct TurnAssembler {
    final_message: Option<String>,
    text_deltas: String,
    tool_calls: Vec<ToolCallRecord>,
    code_outputs: Vec<CodeOutputRecord>,
    token_usage: TokenUsage,
    issues: Vec<TurnIssue>,
    saw_done: bool,
    saw_tool_failure: bool,
}

impl TurnAssembler {
    fn push(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::TextDelta { content } => {
                self.text_deltas.push_str(content);
            }
            AgentEvent::ToolCall {
                name,
                args,
                result,
                success,
                duration_ms,
            } => {
                self.tool_calls.push(ToolCallRecord {
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
            AgentEvent::CodeOutput { output, error } => {
                if error.is_some() {
                    self.saw_tool_failure = true;
                }
                self.code_outputs.push(CodeOutputRecord {
                    output: output.clone(),
                    error: error.clone(),
                });
            }
            AgentEvent::Message { text, kind } if kind == "final" => {
                self.final_message = Some(text.clone());
            }
            AgentEvent::TokenUsage { cumulative, .. } => {
                self.token_usage = cumulative.clone();
            }
            AgentEvent::Error { message, envelope } => {
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
            AgentEvent::Done => {
                self.saw_done = true;
            }
            _ => {}
        }
    }

    fn finish(
        self,
        state: AgentStateEnvelope,
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
            self.text_deltas.trim().to_string()
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
                mode: state.execution_mode,
                had_tool_calls: !self.tool_calls.is_empty(),
                had_code_execution: !self.code_outputs.is_empty(),
            },
            state,
            status,
            assistant_output: AssistantOutput {
                safe_text: safe_output,
                raw_text: raw_output,
                state: output_state,
            },
            done_reason,
            token_usage: self.token_usage,
            tool_calls: self.tool_calls,
            code_outputs: self.code_outputs,
            errors: issues,
        }
    }
}

#[cfg(feature = "sqlite-store")]
pub(crate) fn latest_turn_history_payload(turn: &AssembledTurn) -> serde_json::Value {
    let messages = &turn.state.messages;
    let turn_index = messages
        .iter()
        .filter(|msg| matches!(msg.role, MessageRole::User))
        .count() as i64;
    let last_user_idx = messages
        .iter()
        .rposition(|msg| matches!(msg.role, MessageRole::User));

    let user_message = last_user_idx
        .and_then(|idx| messages.get(idx))
        .map(message_text_for_history)
        .unwrap_or_default();

    let mut prose_parts = Vec::new();
    let mut code_parts = Vec::new();
    if let Some(idx) = last_user_idx {
        for msg in messages.iter().skip(idx + 1) {
            if !matches!(msg.role, MessageRole::Assistant) {
                continue;
            }
            for part in &msg.parts {
                match part.kind {
                    PartKind::Text | PartKind::Prose => {
                        if !part.content.trim().is_empty() {
                            prose_parts.push(part.content.clone());
                        }
                    }
                    PartKind::Code => {
                        if !part.content.trim().is_empty() {
                            code_parts.push(part.content.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let prose = if prose_parts.is_empty() {
        turn.assistant_output.raw_text.clone()
    } else {
        prose_parts.join("\n\n")
    };
    let code = code_parts.join("\n\n");
    let output = turn
        .code_outputs
        .iter()
        .map(|record| match (&record.output, &record.error) {
            (output, Some(error)) if !output.is_empty() && !error.is_empty() => {
                format!("{output}\n{error}")
            }
            (output, _) if !output.is_empty() => output.clone(),
            (_, Some(error)) => error.clone(),
            _ => String::new(),
        })
        .filter(|chunk| !chunk.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let error = turn.errors.first().map(|issue| issue.message.clone());

    serde_json::json!({
        "index": turn_index,
        "user_message": user_message,
        "prose": prose,
        "code": code,
        "output": output,
        "error": error,
        "tool_calls": turn.tool_calls,
    })
}

#[cfg(feature = "sqlite-store")]
fn message_text_for_history(msg: &Message) -> String {
    msg.parts
        .iter()
        .filter_map(|part| match part.kind {
            PartKind::Text | PartKind::Prose | PartKind::Code => Some(part.content.as_str()),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Runtime config used by embedders to construct an engine.
#[derive(Clone)]
pub struct RuntimeConfig {
    pub capabilities: AgentCapabilities,
    pub model: String,
    pub provider: Provider,
    pub execution_mode: ExecutionMode,
    pub context_folding: ContextFoldingConfig,
    pub sub_agent: bool,
    pub reasoning_effort: Option<String>,
    pub session_id: Option<String>,
    pub max_context_tokens: Option<usize>,
    pub max_turns: Option<usize>,
    pub include_soul: bool,
    pub llm_log_path: Option<PathBuf>,
    pub headless: bool,
    pub host_profile: HostProfile,
    pub prompt_overrides: Vec<PromptSectionOverride>,
    pub base_dir: Option<PathBuf>,
    pub path_resolver: Option<Arc<dyn PathResolver>>,
    pub sanitizer: SanitizerPolicy,
    pub termination: TerminationPolicy,
    pub instruction_source: Arc<dyn InstructionSource>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let cfg = AgentConfig::default();
        Self {
            capabilities: cfg.capabilities,
            model: cfg.model,
            provider: cfg.provider,
            execution_mode: cfg.execution_mode,
            context_folding: cfg.context_folding,
            sub_agent: cfg.sub_agent,
            reasoning_effort: cfg.reasoning_effort,
            session_id: cfg.session_id,
            max_context_tokens: cfg.max_context_tokens,
            max_turns: cfg.max_turns,
            include_soul: cfg.include_soul,
            llm_log_path: cfg.llm_log_path,
            headless: cfg.headless,
            host_profile: if cfg.headless {
                HostProfile::Headless
            } else {
                HostProfile::Interactive
            },
            prompt_overrides: cfg.prompt_overrides,
            base_dir: None,
            path_resolver: None,
            sanitizer: SanitizerPolicy::default(),
            termination: TerminationPolicy::default(),
            instruction_source: Arc::new(FsInstructionSource::new()),
        }
    }
}

impl From<RuntimeConfig> for AgentConfig {
    fn from(value: RuntimeConfig) -> Self {
        let headless = value.headless || matches!(value.host_profile, HostProfile::Headless);
        Self {
            capabilities: value.capabilities,
            model: value.model,
            provider: value.provider,
            execution_mode: value.execution_mode,
            context_folding: value.context_folding,
            session_id: value.session_id,
            max_context_tokens: value.max_context_tokens,
            sub_agent: value.sub_agent,
            reasoning_effort: value.reasoning_effort,
            max_turns: value.max_turns,
            include_soul: value.include_soul,
            llm_log_path: value.llm_log_path,
            headless,
            prompt_overrides: value.prompt_overrides,
            instruction_source: value.instruction_source,
        }
    }
}

impl From<AgentConfig> for RuntimeConfig {
    fn from(value: AgentConfig) -> Self {
        let host_profile = if value.headless {
            HostProfile::Headless
        } else {
            HostProfile::Interactive
        };
        Self {
            capabilities: value.capabilities,
            model: value.model,
            provider: value.provider,
            execution_mode: value.execution_mode,
            context_folding: value.context_folding,
            sub_agent: value.sub_agent,
            reasoning_effort: value.reasoning_effort,
            session_id: value.session_id,
            max_context_tokens: value.max_context_tokens,
            max_turns: value.max_turns,
            include_soul: value.include_soul,
            llm_log_path: value.llm_log_path,
            headless: value.headless,
            host_profile,
            prompt_overrides: value.prompt_overrides,
            base_dir: None,
            path_resolver: None,
            sanitizer: SanitizerPolicy::default(),
            termination: TerminationPolicy::default(),
            instruction_source: value.instruction_source,
        }
    }
}

/// Generic runtime for CLI or programmatic embedding.
pub struct LashRuntime {
    session: Option<Session>,
    config: RuntimeConfig,
    state: AgentStateEnvelope,
    capabilities: AgentCapabilities,
    host_profile: HostProfile,
    base_dir: PathBuf,
    path_resolver: Option<Arc<dyn PathResolver>>,
    sanitizer: SanitizerPolicy,
    termination: TerminationPolicy,
    cached_base_context: Option<String>,
    llm_factory: LlmFactory,
    managed_sessions: Arc<Mutex<HashMap<String, Arc<Mutex<LashRuntime>>>>>,
}

#[derive(Clone)]
struct RuntimeSessionManager {
    current_agent_id: String,
    current_snapshot: SessionSnapshot,
    current_config: RuntimeConfig,
    current_tools: Arc<dyn crate::ToolProvider>,
    current_plugins: Arc<crate::PluginSession>,
    registry: Arc<Mutex<HashMap<String, Arc<Mutex<LashRuntime>>>>>,
}

impl RuntimeSessionManager {
    fn build_runtime_config(&self, overrides: &SessionConfigOverrides) -> RuntimeConfig {
        let mut config = self.current_config.clone();
        if let Some(model) = &overrides.model {
            config.model = model.clone();
        }
        if let Some(reasoning_effort) = &overrides.reasoning_effort {
            config.reasoning_effort = Some(reasoning_effort.clone());
        }
        if let Some(execution_mode) = overrides.execution_mode {
            config.execution_mode = execution_mode;
        }
        if let Some(capabilities) = &overrides.capabilities {
            config.capabilities = capabilities.clone();
        }
        if let Some(context_folding) = overrides.context_folding {
            config.context_folding = context_folding;
        }
        if let Some(session_id) = &overrides.session_id {
            config.session_id = Some(session_id.clone());
        }
        config
    }

    fn build_runtime_state(
        &self,
        agent_id: String,
        request: &SessionCreateRequest,
        mut base: SessionSnapshot,
    ) -> SessionSnapshot {
        base.agent_id = agent_id;
        if let Some(execution_mode) = request.config_overrides.execution_mode {
            base.execution_mode = execution_mode;
        }
        if let Some(context_folding) = request.config_overrides.context_folding {
            base.context_folding = context_folding;
        }
        let existing_messages = base.messages.clone();
        let appended = request
            .initial_messages
            .iter()
            .enumerate()
            .map(|(idx, message)| plugin_message_to_message(&existing_messages, idx, message))
            .collect::<Vec<_>>();
        base.messages.extend(appended);
        base
    }

    async fn snapshot_by_id(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        if session_id == self.current_agent_id {
            return Ok(self.current_snapshot.clone());
        }
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        Ok(runtime.export_state())
    }
}

#[async_trait::async_trait]
impl SessionManager for RuntimeSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, crate::PluginError> {
        Ok(self.current_snapshot.clone())
    }

    async fn snapshot_session(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        self.snapshot_by_id(session_id).await
    }

    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, crate::PluginError> {
        let agent_id = request
            .agent_id
            .clone()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let snapshot = match &request.start {
            SessionStartPoint::Empty => SessionSnapshot {
                agent_id: agent_id.clone(),
                execution_mode: request
                    .config_overrides
                    .execution_mode
                    .unwrap_or(self.current_config.execution_mode),
                context_folding: request
                    .config_overrides
                    .context_folding
                    .unwrap_or(self.current_config.context_folding),
                ..Default::default()
            },
            SessionStartPoint::CurrentSession => self.current_snapshot.clone(),
            SessionStartPoint::ExistingSession { session_id } => {
                self.snapshot_by_id(session_id).await?
            }
            SessionStartPoint::Snapshot { snapshot } => (**snapshot).clone(),
        };
        let state = self.build_runtime_state(agent_id.clone(), &request, snapshot);
        let config = self.build_runtime_config(&request.config_overrides);
        let tools = if let Some(snapshot) = self.current_tools.dynamic_snapshot() {
            self.current_tools
                .fork_dynamic_with_snapshot(snapshot)
                .unwrap_or_else(|| Arc::clone(&self.current_tools))
        } else {
            Arc::clone(&self.current_tools)
        };
        let plugins = self
            .current_plugins
            .fork_for_agent(&agent_id)
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let runtime = LashRuntime::from_state(config, RuntimeServices::new(tools, plugins), state)
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        self.registry
            .lock()
            .await
            .insert(agent_id.clone(), Arc::new(Mutex::new(runtime)));
        Ok(SessionHandle {
            session_id: agent_id,
        })
    }

    async fn close_session(&self, session_id: &str) -> Result<(), crate::PluginError> {
        if session_id == self.current_agent_id {
            return Err(crate::PluginError::Session(
                "cannot close the current session".to_string(),
            ));
        }
        self.registry.lock().await.remove(session_id);
        Ok(())
    }

    async fn start_turn(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn, crate::PluginError> {
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let mut runtime = runtime.lock().await;
        runtime
            .run_turn_assembled(input, CancellationToken::new())
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }
}

fn plugin_message_to_message(
    existing_messages: &[Message],
    offset: usize,
    plugin_message: &PluginMessage,
) -> Message {
    let next_index = existing_messages.len() + offset;
    let message_id = format!("m{next_index}");
    Message {
        id: message_id.clone(),
        role: plugin_message.role,
        parts: vec![Part {
            id: format!("{message_id}.p0"),
            kind: PartKind::Text,
            content: plugin_message.content.clone(),
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }],
        origin: Some(MessageOrigin::Plugin {
            plugin_id: "plugin".to_string(),
        }),
    }
}

fn append_plugin_messages(messages: &mut Vec<Message>, plugin_messages: &[PluginMessage]) {
    let new_messages = plugin_messages
        .iter()
        .filter(|message| matches!(message.role, MessageRole::User | MessageRole::System))
        .enumerate()
        .map(|(idx, message)| plugin_message_to_message(messages, idx, message))
        .collect::<Vec<_>>();
    messages.extend(new_messages);
}

impl LashRuntime {
    /// Build a runtime from host-provided state + tools.
    pub async fn from_state(
        config: RuntimeConfig,
        services: RuntimeServices,
        mut state: AgentStateEnvelope,
    ) -> Result<Self, SessionError> {
        let capabilities = config.capabilities.clone();
        let host_profile = config.host_profile.clone();
        let base_dir = config
            .base_dir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let path_resolver = config.path_resolver.clone();
        let sanitizer = config.sanitizer.clone();
        let termination = config.termination.clone();
        if state.agent_id.is_empty() {
            state.agent_id = "root".to_string();
        }
        if matches!(state.execution_mode, ExecutionMode::Repl)
            && !matches!(config.execution_mode, ExecutionMode::Repl)
        {
            state.execution_mode = config.execution_mode;
        }
        if state.context_folding.is_default() && !config.context_folding.is_default() {
            state.context_folding = config.context_folding;
        }
        services.tools.set_execution_mode(state.execution_mode);
        let mut session = Session::new(
            services,
            &state.agent_id,
            config.headless,
            capabilities.clone(),
            state.execution_mode,
        )
        .await?;
        if let Some(snapshot) = state.plugin_snapshot.clone() {
            session
                .plugins()
                .restore(&snapshot)
                .map_err(|err| SessionError::Protocol(err.to_string()))?;
        }
        if matches!(state.execution_mode, ExecutionMode::Repl)
            && let Some(snapshot) = state.repl_snapshot.clone()
        {
            session.restore(&snapshot).await?;
        }
        session.plugins().on_session_restored(&state).await;
        Ok(Self {
            session: Some(session),
            config,
            state,
            capabilities,
            host_profile,
            base_dir,
            path_resolver,
            sanitizer,
            termination,
            cached_base_context: None,
            llm_factory: default_llm_factory(),
            managed_sessions: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Export current host-owned state envelope.
    pub fn export_state(&self) -> AgentStateEnvelope {
        let mut state = self.state.clone();
        if let Some(session) = self.session.as_ref() {
            state.plugin_snapshot = session.plugins().snapshot().ok();
        }
        state
    }

    fn runtime_session_manager(&self) -> Result<Arc<dyn SessionManager>, ExternalInvokeError> {
        let Some(session) = self.session.as_ref() else {
            return Err(ExternalInvokeError::Unknown("session_manager".to_string()));
        };
        Ok(Arc::new(RuntimeSessionManager {
            current_agent_id: self.state.agent_id.clone(),
            current_snapshot: self.export_state(),
            current_config: self.config.clone(),
            current_tools: Arc::clone(session.tools()),
            current_plugins: Arc::clone(session.plugins()),
            registry: Arc::clone(&self.managed_sessions),
        }))
    }

    /// Replace the host-owned state envelope.
    pub fn set_state(&mut self, state: AgentStateEnvelope) {
        if let Some(session) = self.session.as_ref() {
            session.tools().set_execution_mode(state.execution_mode);
            if let Some(snapshot) = state.plugin_snapshot.as_ref()
                && let Err(err) = session.plugins().restore(snapshot)
            {
                tracing::warn!("failed to restore plugin snapshot in set_state: {err}");
            }
        }
        self.state = state;
    }

    /// Update model on the runtime config.
    pub fn set_model(&mut self, model: String) {
        self.config.model = model;
    }

    /// Update reasoning effort on the runtime config.
    pub fn set_reasoning_effort(&mut self, reasoning_effort: Option<String>) {
        self.config.reasoning_effort = reasoning_effort;
    }

    /// Update provider on the runtime config.
    pub fn set_provider(&mut self, provider: Provider) {
        self.config.provider = provider;
    }

    /// Update session ID metadata on the runtime config.
    pub fn set_session_id(&mut self, session_id: Option<String>) {
        self.config.session_id = session_id;
    }

    /// Update execution mode on the runtime and persisted envelope.
    pub fn set_execution_mode(&mut self, execution_mode: ExecutionMode) {
        self.state.execution_mode = execution_mode;
        self.config.execution_mode = execution_mode;
        if let Some(session) = self.session.as_ref() {
            session.tools().set_execution_mode(execution_mode);
        }
    }

    /// Update context folding policy on the runtime and persisted envelope.
    pub fn set_context_folding(&mut self, context_folding: ContextFoldingConfig) {
        self.state.context_folding = context_folding;
        self.config.context_folding = context_folding;
    }

    /// Update the active prompt-facing capability set.
    pub fn set_capabilities(&mut self, capabilities: AgentCapabilities) {
        self.capabilities = capabilities.clone();
        self.config.capabilities = capabilities;
    }

    /// Re-register the current tool/capability projection in the live Python session.
    pub async fn reconfigure_session(
        &mut self,
        capabilities_json: String,
        generation: u64,
    ) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        session.reconfigure(capabilities_json, generation).await
    }

    /// Reset the REPL session on the underlying agent.
    pub async fn reset_session(&mut self) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        session.reset().await
    }

    /// Explicitly snapshot REPL state; does not run an LLM turn.
    pub async fn snapshot_repl(&mut self) -> Result<Vec<u8>, SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        let blob = session.snapshot().await?;
        self.state.repl_snapshot = Some(blob.clone());
        Ok(blob)
    }

    /// Explicitly restore REPL state from an opaque snapshot blob.
    pub async fn restore_repl(&mut self, snapshot: &[u8]) -> Result<(), SessionError> {
        let Some(session) = self.session.as_mut() else {
            return Err(SessionError::Protocol(
                "runtime session not available".to_string(),
            ));
        };
        session.restore(snapshot).await?;
        self.state.repl_snapshot = Some(snapshot.to_vec());
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
    pub async fn stream_turn(
        &mut self,
        input: TurnInput,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let normalized = match self.normalize_input_items(&input.items, &input.image_blobs) {
            Ok(items) => items,
            Err(e) => {
                let mut assembler = TurnAssembler::default();
                let error_event = AgentEvent::Error {
                    message: e.clone(),
                    envelope: Some(crate::agent::ErrorEnvelope {
                        kind: "input_validation".to_string(),
                        code: Some("invalid_turn_input".to_string()),
                        user_message: e,
                        raw: None,
                    }),
                };
                assembler.push(&error_event);
                events.emit(error_event).await;
                assembler.push(&AgentEvent::Done);
                events.emit(AgentEvent::Done).await;
                return Ok(assembler.finish(
                    self.state.clone(),
                    false,
                    None,
                    &self.sanitizer,
                    &self.termination,
                ));
            }
        };

        let mut messages = self.state.messages.clone();
        let mode = input.mode.unwrap_or(match self.host_profile {
            HostProfile::Interactive | HostProfile::Headless | HostProfile::Embedded => {
                RunMode::Normal
            }
        });
        let plan_file = input
            .plan_file
            .clone()
            .unwrap_or_else(|| ".lash_plan".into());
        let mode_msg = match mode {
            RunMode::Plan => Some(format!(
                "## Plan Mode\n\n\
                You are in PLAN mode. Think, explore, and design — do NOT execute changes.\n\n\
                **Rules:**\n\
                - READ-ONLY: Do not modify project files or run destructive commands\n\
                - You MAY use: read_file, glob, grep, ls, web_search, fetch_url, agent_call\n\
                - You MUST NOT use: edit_file, find_replace, write_file (except the plan file), or shell with write commands\n\
                - Exception: Write your plan to `{}` using write_file\n\n\
                **Workflow:**\n\
                1. Understand the request and note key assumptions\n\
                2. Explore the codebase — read files, search for patterns, understand architecture\n\
                3. Design your approach — consider tradeoffs, identify risks\n\
                4. Write a clear, step-by-step plan to the plan file\n\n\
                When the user switches back to normal mode, you will execute the plan.",
                plan_file
            )),
            RunMode::Normal => input.plan_file.as_ref().and_then(|path| {
                let plan_content = std::fs::read_to_string(path).unwrap_or_default();
                if plan_content.is_empty() {
                    None
                } else {
                    let mut available = Vec::new();
                    if self.capabilities.enabled(CapabilityId::History) {
                        available.push(
                            "- `search_history(\"query\", mode=\"hybrid\")` — search planning exploration"
                                .to_string(),
                        );
                        available.push(
                            "- Prior user requests are included in turn history results".to_string(),
                        );
                    }
                    if self.capabilities.enabled(CapabilityId::Memory) {
                        available
                            .push("- `mem_get(...)`, `mem_all()`, and `search_mem(...)` — persistent memory".to_string());
                    }
                    let available_context = if available.is_empty() {
                        "No persisted planning context capabilities are enabled for this agent."
                            .to_string()
                    } else {
                        available.join("\n")
                    };
                    Some(format!(
                        "## Executing Plan\n\n\
                        You are executing a plan from a previous planning session. \
                        Use available context from this agent configuration.\n\n\
                        **Plan file:** `{}`\n\n\
                        ---\n{}\n---\n\n\
                        **Available context:**\n\
                        {}\n\n\
                        Execute the plan step by step.",
                        path, plan_content, available_context
                    ))
                }
            }),
        };
        if let Some(content) = mode_msg {
            let sys_id = format!("m{}", messages.len());
            messages.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Text,
                    content,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                }],
                origin: None,
            });
        }

        let user_id = format!("m{}", messages.len());
        let mut user_parts: Vec<Part> = Vec::new();
        let mut user_images_png: Vec<Vec<u8>> = Vec::new();
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
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: PruneState::Intact,
                    });
                }
                NormalizedItem::Image(bytes) => {
                    let image_idx = user_images_png.len();
                    user_images_png.push(bytes);
                    user_parts.push(Part {
                        id: format!("{}.p{}", user_id, user_parts.len()),
                        kind: PartKind::Text,
                        content: format!("{IMAGE_REF_PREFIX}{image_idx}"),
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
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
        }
        messages.push(Message {
            id: user_id.clone(),
            role: MessageRole::User,
            parts: user_parts,
            origin: None,
        });

        self.stream_prepared_turn(messages, user_images_png, events, cancel)
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
        mut messages: Vec<Message>,
        images_png: Vec<Vec<u8>>,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let manager = self.runtime_session_manager().map_err(|err| RuntimeError {
            code: "plugin_session_manager".to_string(),
            message: err.to_string(),
        })?;
        let plugins = {
            let session = self
                .session
                .as_ref()
                .expect("lash runtime session must be available");
            Arc::clone(session.plugins())
        };
        let mut hook_state = self.export_state();
        hook_state.messages = messages.clone();
        let before_turn = plugins
            .before_turn(TurnHookContext {
                session_id: self.state.agent_id.clone(),
                state: hook_state,
                host: Arc::clone(&manager),
            })
            .await
            .map_err(|err| RuntimeError {
                code: "plugin_before_turn".to_string(),
                message: err.to_string(),
            })?;
        let mut maybe_abort: Option<(String, String)> = None;
        for directive in before_turn {
            match directive {
                crate::PluginDirective::AbortTurn { code, message } => {
                    maybe_abort = Some((code, message));
                }
                crate::PluginDirective::EnqueueMessages {
                    messages: plugin_messages,
                } => {
                    append_plugin_messages(&mut messages, &plugin_messages);
                }
                crate::PluginDirective::CreateSession { request } => {
                    manager
                        .create_session(*request)
                        .await
                        .map_err(|err| RuntimeError {
                            code: "plugin_create_session".to_string(),
                            message: err.to_string(),
                        })?;
                }
                crate::PluginDirective::ReplaceToolArgs { .. }
                | crate::PluginDirective::ShortCircuitTool { .. } => {
                    return Err(RuntimeError {
                        code: "plugin_invalid_before_turn".to_string(),
                        message: "tool directives are not valid in before_turn".to_string(),
                    });
                }
            }
        }
        if let Some((code, message)) = maybe_abort {
            let mut state = self.state.clone();
            state.messages = messages;
            let issue = TurnIssue {
                kind: "plugin".to_string(),
                code: Some(code),
                message: message.clone(),
            };
            let error_event = AgentEvent::Error {
                message,
                envelope: Some(crate::agent::ErrorEnvelope {
                    kind: "plugin".to_string(),
                    code: issue.code.clone(),
                    user_message: issue.message.clone(),
                    raw: None,
                }),
            };
            let mut assembler = TurnAssembler::default();
            assembler.push(&error_event);
            events.emit(error_event).await;
            assembler.push(&AgentEvent::Done);
            events.emit(AgentEvent::Done).await;
            return Ok(assembler.finish(
                state,
                cancel.is_cancelled(),
                Some(issue),
                &self.sanitizer,
                &self.termination,
            ));
        }

        let plugin_prompt_sections = plugins
            .collect_prompt_contributions(PromptHookContext {
                session_id: self.state.agent_id.clone(),
                host: Arc::clone(&manager),
            })
            .map_err(|err| RuntimeError {
                code: "plugin_prompt".to_string(),
                message: err.to_string(),
            })?
            .into_iter()
            .map(|contribution| contribution.content)
            .collect::<Vec<_>>();

        let cancel_state = cancel.clone();
        let session = self
            .session
            .take()
            .expect("lash runtime session must be available");
        let mut driver = RuntimeTurnDriver {
            session,
            config: self.config.clone(),
            cached_base_context: self.cached_base_context.take(),
            agent_id: self.state.agent_id.clone(),
            llm_factory: Arc::clone(&self.llm_factory),
            plugin_prompt_sections,
            session_manager: manager,
        };
        let run_offset = self.state.iteration;
        let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(100);
        let run_task = tokio::spawn(async move {
            let (new_messages, new_iteration) = driver
                .run(messages, images_png, event_tx, cancel, run_offset)
                .await;
            (driver, new_messages, new_iteration)
        });

        let mut assembler = TurnAssembler::default();
        while let Some(event) = event_rx.recv().await {
            assembler.push(&event);
            events.emit(event).await;
        }

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
                    &self.sanitizer,
                    &self.termination,
                ));
            }
        };

        self.session = Some(driver.session);
        self.config = driver.config;
        self.cached_base_context = driver.cached_base_context;
        self.state.messages = new_messages;
        self.state.iteration = new_iteration;
        if assembler.token_usage.total() > 0 {
            self.state.token_usage = assembler.token_usage.clone();
        }

        let mut assembled = assembler.finish(
            self.state.clone(),
            cancel_state.is_cancelled(),
            None,
            &self.sanitizer,
            &self.termination,
        );
        if let Some(session) = self.session.as_ref() {
            let plugins = Arc::clone(session.plugins());
            let manager = self.runtime_session_manager().map_err(|err| RuntimeError {
                code: "plugin_session_manager".to_string(),
                message: err.to_string(),
            })?;
            let after_turn = plugins
                .after_turn(TurnResultHookContext {
                    session_id: self.state.agent_id.clone(),
                    turn: assembled.clone(),
                    host: Arc::clone(&manager),
                })
                .await
                .map_err(|err| RuntimeError {
                    code: "plugin_after_turn".to_string(),
                    message: err.to_string(),
                })?;
            for directive in after_turn {
                match directive {
                    crate::PluginDirective::EnqueueMessages {
                        messages: plugin_messages,
                    } => append_plugin_messages(&mut self.state.messages, &plugin_messages),
                    crate::PluginDirective::CreateSession { request } => {
                        manager
                            .create_session(*request)
                            .await
                            .map_err(|err| RuntimeError {
                                code: "plugin_create_session".to_string(),
                                message: err.to_string(),
                            })?;
                    }
                    crate::PluginDirective::AbortTurn { .. }
                    | crate::PluginDirective::ReplaceToolArgs { .. }
                    | crate::PluginDirective::ShortCircuitTool { .. } => {
                        return Err(RuntimeError {
                            code: "plugin_invalid_after_turn".to_string(),
                            message:
                                "only message enqueue and session creation are valid in after_turn"
                                    .to_string(),
                        });
                    }
                }
            }
            assembled.state.messages = self.state.messages.clone();
            plugins.on_turn_committed(&assembled).await;
            self.state.plugin_snapshot = plugins.snapshot().ok();
        }
        Ok(assembled)
    }
}

struct RuntimeTurnDriver {
    session: Session,
    config: RuntimeConfig,
    cached_base_context: Option<String>,
    agent_id: String,
    llm_factory: LlmFactory,
    plugin_prompt_sections: Vec<String>,
    session_manager: Arc<dyn SessionManager>,
}

impl RuntimeTurnDriver {
    fn llm(&self, provider: &Provider) -> Box<dyn LlmTransport> {
        (self.llm_factory)(provider)
    }

    async fn run(
        &mut self,
        messages: Vec<Message>,
        images: Vec<Vec<u8>>,
        event_tx: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
        run_offset: usize,
    ) -> (Vec<Message>, usize) {
        macro_rules! emit {
            ($event:expr) => {
                crate::agent::send_event(&event_tx, $event).await
            };
        }

        if matches!(self.config.execution_mode, ExecutionMode::Standard) {
            return self
                .run_standard(messages, images, event_tx, cancel, run_offset)
                .await;
        }

        let capabilities_json = self
            .session
            .tools()
            .dynamic_capabilities_payload_json()
            .unwrap_or_else(|| {
                let available_defs = self.session.tools().definitions();
                let capability_defs = crate::default_dynamic_capability_defs();
                let resolved = crate::resolve_capability_projection(
                    &capability_defs,
                    &self.config.capabilities,
                    &available_defs,
                )
                .unwrap_or_else(|_| crate::ResolvedProjection {
                    enabled_capabilities: self
                        .config
                        .capabilities
                        .enabled_capabilities
                        .iter()
                        .map(|id| id.as_str().to_string())
                        .collect(),
                    effective_tools: self
                        .config
                        .capabilities
                        .enabled_tools
                        .iter()
                        .filter(|tool| available_defs.iter().any(|def| def.name == tool.as_str()))
                        .cloned()
                        .collect(),
                    helper_bindings: BTreeSet::new(),
                    prompt_sections: Vec::new(),
                });
                serde_json::json!({
                    "enabled_capabilities": resolved.enabled_capabilities.into_iter().collect::<Vec<_>>(),
                    "enabled_tools": resolved.effective_tools.into_iter().collect::<Vec<_>>(),
                    "helper_bindings": resolved.helper_bindings.into_iter().collect::<Vec<_>>(),
                })
                .to_string()
            });
        let generation = self.session.tools().dynamic_generation().unwrap_or(0);
        if let Err(e) = self
            .session
            .reconfigure(capabilities_json, generation)
            .await
        {
            emit!(make_error_event(
                "tool_projection",
                Some("reconfigure_failed"),
                format!("Failed to refresh REPL tool projection: {e}"),
                Some(e.to_string()),
            ));
            emit!(AgentEvent::Done);
            return (messages, run_offset);
        }

        let mut agent_config = self.runtime_agent_config();
        let model = match self.prepare_provider(&mut agent_config).await {
            Ok(model) => model,
            Err(event) => {
                emit!(event);
                emit!(AgentEvent::Done);
                return (messages, run_offset);
            }
        };
        let preamble = build_execution_preamble(
            &self.session,
            &agent_config,
            &mut self.cached_base_context,
            ExecutionMode::Repl,
            model,
            self.plugin_prompt_sections.clone(),
        );
        self.config = agent_config.into();

        let max_context = self.max_context_tokens(&preamble.model);
        let machine_config = self.machine_config(preamble, max_context, ExecutionMode::Repl);
        let mut machine = TurnMachine::new(machine_config, messages, images, run_offset);

        loop {
            let Some(effect) = machine.poll_effect() else {
                break;
            };
            match effect {
                Effect::Emit(event) => emit!(event),
                Effect::Done {
                    messages,
                    iteration,
                } => return (messages, iteration),
                Effect::LlmCall { id, request } => {
                    if cancel.is_cancelled() {
                        emit!(AgentEvent::Done);
                        return (Vec::new(), run_offset);
                    }
                    let llm_response = self
                        .run_repl_llm_call(&mut machine, id, request, &event_tx, &cancel)
                        .await;
                    machine.handle_response(Response::LlmComplete {
                        id,
                        result: llm_response,
                        text_streamed: false,
                    });
                }
                Effect::ExecCode { id, code } => {
                    let result = self.run_exec_code(&code, &event_tx).await;
                    let response = match result {
                        Ok(output) => Response::ExecResult {
                            id,
                            result: Ok(output),
                        },
                        Err(error) => Response::ExecResult {
                            id,
                            result: Err(error),
                        },
                    };
                    machine.handle_response(response);
                }
                Effect::Sleep { id, duration } => {
                    tokio::time::sleep(duration).await;
                    machine.handle_response(Response::Timeout { id });
                }
                Effect::CancelLlm { .. } => {}
                Effect::ToolCall { .. } => {}
            }
        }

        (Vec::new(), run_offset)
    }

    async fn run_standard(
        &mut self,
        messages: Vec<Message>,
        images: Vec<Vec<u8>>,
        event_tx: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
        run_offset: usize,
    ) -> (Vec<Message>, usize) {
        macro_rules! emit {
            ($event:expr) => {
                crate::agent::send_event(&event_tx, $event).await
            };
        }

        let mut agent_config = self.runtime_agent_config();
        let model = match self.prepare_provider(&mut agent_config).await {
            Ok(model) => model,
            Err(event) => {
                emit!(event);
                emit!(AgentEvent::Done);
                return (messages, run_offset);
            }
        };
        let preamble = build_execution_preamble(
            &self.session,
            &agent_config,
            &mut self.cached_base_context,
            ExecutionMode::Standard,
            model,
            self.plugin_prompt_sections.clone(),
        );
        self.config = agent_config.into();

        let max_context = self.max_context_tokens(&preamble.model);
        let machine_config = self.machine_config(preamble, max_context, ExecutionMode::Standard);
        let mut machine = TurnMachine::new(machine_config, messages, images, run_offset);

        loop {
            let Some(effect) = machine.poll_effect() else {
                break;
            };
            match effect {
                Effect::Emit(event) => emit!(event),
                Effect::Done {
                    messages,
                    iteration,
                } => return (messages, iteration),
                Effect::LlmCall { id, request } => {
                    if cancel.is_cancelled() {
                        emit!(AgentEvent::Done);
                        return (Vec::new(), run_offset);
                    }
                    let (result, text_streamed) = self
                        .run_standard_llm_call(request, &event_tx, &cancel)
                        .await;
                    machine.handle_response(Response::LlmComplete {
                        id,
                        result,
                        text_streamed,
                    });
                }
                Effect::ToolCall {
                    id,
                    call_id,
                    tool_name,
                    args,
                } => {
                    let mut pending_tools = vec![(id, call_id, tool_name, args)];
                    while let Some(next) = machine.poll_effect() {
                        match next {
                            Effect::ToolCall {
                                id,
                                call_id,
                                tool_name,
                                args,
                            } => pending_tools.push((id, call_id, tool_name, args)),
                            Effect::Emit(event) => emit!(event),
                            Effect::Done {
                                messages,
                                iteration,
                            } => return (messages, iteration),
                            _ => break,
                        }
                    }
                    for (id, call_id, tool_name, _args, result, duration_ms) in
                        self.run_tool_calls(pending_tools, &event_tx).await
                    {
                        machine.handle_response(Response::ToolResult {
                            id,
                            call_id,
                            tool_name,
                            result,
                            duration_ms,
                        });
                    }
                }
                Effect::Sleep { id, duration } => {
                    tokio::time::sleep(duration).await;
                    machine.handle_response(Response::Timeout { id });
                }
                Effect::CancelLlm { .. } => {}
                Effect::ExecCode { .. } => {}
            }
        }

        (Vec::new(), run_offset)
    }

    fn runtime_agent_config(&self) -> AgentConfig {
        self.config.clone().into()
    }

    async fn prepare_provider(&mut self, config: &mut AgentConfig) -> Result<String, AgentEvent> {
        match config.provider.ensure_fresh().await {
            Ok(true) => {
                let _ = crate::provider::save_provider(&config.provider);
            }
            Err(e) => {
                return Err(make_error_event(
                    "token_refresh",
                    Some("refresh_failed"),
                    format!(
                        "Token refresh failed: {}. Re-authenticate with /provider and retry.",
                        e
                    ),
                    Some(e.to_string()),
                ));
            }
            _ => {}
        }

        let llm = self.llm(&config.provider);
        let model = llm.normalize_model(&config.model);
        match llm.ensure_ready(&mut config.provider).await {
            Ok(changed) => {
                if changed {
                    let _ = crate::provider::save_provider(&config.provider);
                }
            }
            Err(e) => {
                return Err(make_error_event(
                    "llm_provider",
                    e.code.as_deref(),
                    format!(
                        "LLM provider initialization failed: {}. Run /provider to reconfigure credentials, then retry.",
                        e.message
                    ),
                    e.raw,
                ));
            }
        }

        Ok(model)
    }

    fn machine_config(
        &self,
        preamble: crate::agent::ExecutionPreamble,
        max_context_tokens: usize,
        execution_mode: ExecutionMode,
    ) -> TurnMachineConfig {
        TurnMachineConfig {
            execution_mode,
            model: preamble.model,
            context_folding: self.config.context_folding,
            max_context_tokens,
            max_turns: self.config.max_turns,
            headless: self.config.headless,
            sub_agent: self.config.sub_agent,
            include_soul: self.config.include_soul,
            reasoning_effort: self.config.reasoning_effort.clone(),
            session_id: self.config.session_id.clone(),
            tool_list: preamble.tool_list,
            tool_specs: preamble.tool_specs,
            tool_names: preamble.tool_names,
            helper_bindings: preamble.helper_bindings,
            capability_prompt_sections: preamble.capability_prompt_sections,
            plugin_prompt_sections: preamble.plugin_prompt_sections,
            can_write: preamble.can_write,
            history_enabled: preamble.history_enabled,
            project_instructions: preamble.project_instructions,
            prompt_overrides: self.config.prompt_overrides.clone(),
            base_context: preamble.base_context,
            instruction_source: preamble.instruction_source,
            llm_log_path: self.config.llm_log_path.clone(),
            agent_id: self.agent_id.clone(),
        }
    }

    fn max_context_tokens(&self, model: &str) -> usize {
        self.config
            .max_context_tokens
            .or_else(|| {
                self.config
                    .provider
                    .context_window(model)
                    .map(|v| v as usize)
            })
            .unwrap_or(200_000)
    }

    async fn run_exec_code(
        &mut self,
        code: &str,
        event_tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<crate::ExecResponse, String> {
        let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel::<SandboxMessage>();
        self.session.set_message_sender(msg_tx);
        let event_tx_clone = event_tx.clone();
        let drain_handle = tokio::spawn(async move {
            while let Some(sandbox_msg) = msg_rx.recv().await {
                if sandbox_msg.kind != "final" && !event_tx_clone.is_closed() {
                    let _ = event_tx_clone
                        .send(AgentEvent::Message {
                            text: sandbox_msg.text,
                            kind: sandbox_msg.kind,
                        })
                        .await;
                }
            }
        });
        let result = self.session.run_code(code).await.map_err(|e| e.to_string());
        self.session.clear_message_sender();
        let _ = drain_handle.await;
        result
    }

    async fn run_repl_llm_call(
        &mut self,
        machine: &mut TurnMachine,
        effect_id: crate::sansio::EffectId,
        request: LlmRequest,
        event_tx: &mpsc::Sender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<LlmResponse, LlmCallError> {
        let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel::<SandboxMessage>();
        self.session.set_message_sender(msg_tx);
        let (prompt_tx, mut prompt_rx) =
            tokio::sync::mpsc::unbounded_channel::<crate::session::UserPrompt>();
        self.session.set_prompt_sender(prompt_tx);
        let (llm_stream_tx, mut llm_stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
        let llm_request = LlmRequest {
            stream_events: transport_stream_events(&self.config.provider, Some(llm_stream_tx)),
            ..request
        };

        let mut call_provider = self.config.provider.clone();
        let llm_factory = Arc::clone(&self.llm_factory);
        let mut llm_task = tokio::spawn(async move {
            let llm = llm_factory(&call_provider);
            let result = llm.complete(&mut call_provider, llm_request).await;
            (result, call_provider)
        });

        let result = loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    llm_task.abort();
                    break Err(LlmCallError {
                        message: "cancelled".to_string(),
                        retryable: false,
                        raw: None,
                        code: Some("cancelled".to_string()),
                    });
                }
                Some(sandbox_msg) = msg_rx.recv() => {
                    if sandbox_msg.kind != "final" && !event_tx.is_closed() {
                        let _ = event_tx.send(AgentEvent::Message {
                            text: sandbox_msg.text,
                            kind: sandbox_msg.kind,
                        }).await;
                    }
                }
                Some(user_prompt) = prompt_rx.recv() => {
                    if !event_tx.is_closed() {
                        let _ = event_tx.send(AgentEvent::Prompt {
                            question: user_prompt.question,
                            options: user_prompt.options,
                            response_tx: user_prompt.response_tx,
                        }).await;
                    }
                }
                Some(stream_event) = llm_stream_rx.recv() => {
                    match stream_event {
                        LlmStreamEvent::Delta(delta) => {
                            if !machine.handle_llm_delta(effect_id, &delta) {
                                llm_task.abort();
                                break Ok(LlmResponse::default());
                            }
                        }
                        LlmStreamEvent::Part(LlmOutputPart::Text { text }) => {
                            if !machine.handle_llm_delta(effect_id, &text) {
                                llm_task.abort();
                                break Ok(LlmResponse::default());
                            }
                        }
                        LlmStreamEvent::Part(LlmOutputPart::ToolCall { .. }) => {}
                        LlmStreamEvent::Usage(usage) => machine.handle_llm_usage(effect_id, &usage),
                    }
                }
                join = &mut llm_task => {
                    let (result, provider_after) = match join {
                        Ok(v) => v,
                        Err(e) => break Err(LlmCallError {
                            message: format!("internal task failed: {e}"),
                            retryable: false,
                            raw: None,
                            code: Some("task_join_failed".to_string()),
                        }),
                    };
                    self.config.provider = provider_after;
                    let mut completed_from_stream: Option<LlmResponse> = None;
                    while let Ok(stream_event) = llm_stream_rx.try_recv() {
                        match stream_event {
                            LlmStreamEvent::Delta(delta) => {
                                if !machine.handle_llm_delta(effect_id, &delta) {
                                    completed_from_stream = Some(LlmResponse::default());
                                    break;
                                }
                            }
                            LlmStreamEvent::Part(LlmOutputPart::Text { text }) => {
                                if !machine.handle_llm_delta(effect_id, &text) {
                                    completed_from_stream = Some(LlmResponse::default());
                                    break;
                                }
                            }
                            LlmStreamEvent::Part(LlmOutputPart::ToolCall { .. }) => {}
                            LlmStreamEvent::Usage(usage) => {
                                machine.handle_llm_usage(effect_id, &usage);
                            }
                        }
                    }
                    if let Some(response) = completed_from_stream {
                        break Ok(response);
                    }
                    match result {
                        Ok(resp) => break Ok(resp),
                        Err(e) => break Err(LlmCallError {
                            message: e.message,
                            retryable: e.retryable,
                            raw: e.raw,
                            code: e.code,
                        }),
                    }
                }
            }
        };

        self.session.clear_message_sender();
        self.session.clear_prompt_sender();
        while let Ok(sandbox_msg) = msg_rx.try_recv() {
            if sandbox_msg.kind != "final" && !event_tx.is_closed() {
                let _ = event_tx
                    .send(AgentEvent::Message {
                        text: sandbox_msg.text,
                        kind: sandbox_msg.kind,
                    })
                    .await;
            }
        }
        while let Ok(user_prompt) = prompt_rx.try_recv() {
            if !event_tx.is_closed() {
                let _ = event_tx
                    .send(AgentEvent::Prompt {
                        question: user_prompt.question,
                        options: user_prompt.options,
                        response_tx: user_prompt.response_tx,
                    })
                    .await;
            }
        }
        result
    }

    async fn run_standard_llm_call(
        &mut self,
        request: LlmRequest,
        event_tx: &mpsc::Sender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        let (llm_stream_tx, mut llm_stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
        let llm_request = LlmRequest {
            stream_events: transport_stream_events(&self.config.provider, Some(llm_stream_tx)),
            ..request
        };

        let mut call_provider = self.config.provider.clone();
        let llm_factory = Arc::clone(&self.llm_factory);
        let mut llm_task = tokio::spawn(async move {
            let llm = llm_factory(&call_provider);
            let result = llm.complete(&mut call_provider, llm_request).await;
            (result, call_provider)
        });

        let mut text_streamed = false;
        let mut streamed_usage = LlmUsage::default();
        let result = loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    llm_task.abort();
                    break Err(LlmCallError {
                        message: "cancelled".to_string(),
                        retryable: false,
                        raw: None,
                        code: Some("cancelled".to_string()),
                    });
                }
                Some(stream_event) = llm_stream_rx.recv() => {
                    forward_standard_stream_event(
                        event_tx,
                        stream_event,
                        &mut text_streamed,
                        &mut streamed_usage,
                    ).await;
                }
                join = &mut llm_task => {
                    let (result, provider_after) = match join {
                        Ok(v) => v,
                        Err(e) => break Err(LlmCallError {
                            message: format!("internal task failed: {e}"),
                            retryable: false,
                            raw: None,
                            code: Some("task_join_failed".to_string()),
                        }),
                    };
                    self.config.provider = provider_after;
                    drain_standard_stream_queue(
                        event_tx,
                        &mut llm_stream_rx,
                        &mut text_streamed,
                        &mut streamed_usage,
                    ).await;
                    match result {
                        Ok(mut resp) => {
                            if response_usage_is_empty(&resp.usage) {
                                resp.usage = streamed_usage.clone();
                            }
                            break Ok(resp)
                        }
                        Err(e) => break Err(LlmCallError {
                            message: e.message,
                            retryable: e.retryable,
                            raw: e.raw,
                            code: e.code,
                        }),
                    }
                }
            }
        };

        (result, text_streamed)
    }

    async fn run_tool_calls(
        &mut self,
        pending_tools: Vec<(crate::sansio::EffectId, String, String, serde_json::Value)>,
        event_tx: &mpsc::Sender<AgentEvent>,
    ) -> Vec<(
        crate::sansio::EffectId,
        String,
        String,
        serde_json::Value,
        crate::ToolResult,
        u64,
    )> {
        let tool_provider = Arc::clone(self.session.tools());
        let plugins = Arc::clone(self.session.plugins());
        let manager = Arc::clone(&self.session_manager);
        let mut join_set = tokio::task::JoinSet::new();
        let mut immediate = Vec::new();
        for (eid, call_id, tool_name, mut args) in pending_tools {
            let directives = match plugins
                .before_tool_call(crate::plugin::ToolCallHookContext {
                    session_id: self.agent_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                    host: Arc::clone(&manager),
                })
                .await
            {
                Ok(directives) => directives,
                Err(err) => {
                    immediate.push((
                        eid,
                        call_id,
                        tool_name,
                        args,
                        crate::ToolResult::err_fmt(err.to_string()),
                        0,
                    ));
                    continue;
                }
            };
            let mut short_circuit: Option<crate::ToolResult> = None;
            for directive in directives {
                match directive {
                    crate::PluginDirective::CreateSession { request } => {
                        if let Err(err) = manager.create_session(*request).await {
                            short_circuit = Some(crate::ToolResult::err_fmt(err.to_string()));
                            break;
                        }
                    }
                    crate::PluginDirective::ReplaceToolArgs { args: replacement } => {
                        args = replacement;
                    }
                    crate::PluginDirective::ShortCircuitTool { .. } => {
                        short_circuit = directive.into_tool_result();
                    }
                    crate::PluginDirective::AbortTurn { message, .. } => {
                        short_circuit = Some(crate::ToolResult::err_fmt(message));
                    }
                    crate::PluginDirective::EnqueueMessages { .. } => {
                        short_circuit = Some(crate::ToolResult::err_fmt(
                            "before_tool_call does not support message injection",
                        ));
                    }
                }
            }
            if let Some(result) = short_circuit {
                immediate.push((eid, call_id, tool_name, args, result, 0));
                continue;
            }

            if (tool_name == "list_tools" || tool_name == "search_tools")
                && let Some(obj) = args.as_object_mut()
                && !obj.contains_key("catalog")
            {
                let catalog: Vec<serde_json::Value> = tool_provider
                    .definitions()
                    .into_iter()
                    .filter(|d| !d.hidden)
                    .filter(|d| !d.description_for(ExecutionMode::Standard).is_empty())
                    .map(|d| {
                        let p = d.project(ExecutionMode::Standard);
                        serde_json::json!({
                            "name": p.name,
                            "description": p.description,
                            "params": p.params,
                            "returns": p.returns,
                            "examples": p.examples,
                            "hidden": p.hidden,
                            "inject_into_prompt": p.inject_into_prompt,
                        })
                    })
                    .collect();
                obj.insert("catalog".to_string(), serde_json::Value::Array(catalog));
            }

            let provider = Arc::clone(&tool_provider);
            let event_tx_clone = event_tx.clone();
            let plugins = Arc::clone(&plugins);
            let manager = Arc::clone(&manager);
            let session_id = self.agent_id.clone();
            join_set.spawn(async move {
                let (progress_tx, mut progress_rx) =
                    tokio::sync::mpsc::unbounded_channel::<SandboxMessage>();
                let progress_handle = tokio::spawn(async move {
                    while let Some(sandbox_msg) = progress_rx.recv().await {
                        if sandbox_msg.kind != "final" {
                            let _ = event_tx_clone
                                .send(AgentEvent::Message {
                                    text: sandbox_msg.text,
                                    kind: sandbox_msg.kind,
                                })
                                .await;
                        }
                    }
                });
                let tool_start = std::time::Instant::now();
                let result = provider
                    .execute_streaming(&tool_name, &args, Some(&progress_tx))
                    .await;
                drop(progress_tx);
                let _ = progress_handle.await;
                let result = match plugins
                    .after_tool_call(crate::plugin::ToolResultHookContext {
                        session_id,
                        tool_name: tool_name.clone(),
                        args: args.clone(),
                        result: result.clone(),
                        duration_ms: tool_start.elapsed().as_millis() as u64,
                        host: Arc::clone(&manager),
                    })
                    .await
                {
                    Ok(directives) => {
                        let mut final_result = result;
                        for directive in directives {
                            match directive {
                                crate::PluginDirective::CreateSession { request } => {
                                    if let Err(err) = manager.create_session(*request).await {
                                        final_result = crate::ToolResult::err_fmt(err.to_string());
                                        break;
                                    }
                                }
                                crate::PluginDirective::ShortCircuitTool { .. } => {
                                    if let Some(replacement) = directive.into_tool_result() {
                                        final_result = replacement;
                                    }
                                }
                                crate::PluginDirective::AbortTurn { message, .. } => {
                                    final_result = crate::ToolResult::err_fmt(message);
                                }
                                crate::PluginDirective::ReplaceToolArgs { .. }
                                | crate::PluginDirective::EnqueueMessages { .. } => {
                                    final_result = crate::ToolResult::err_fmt(
                                        "after_tool_call only supports abort, short-circuit, and session creation",
                                    );
                                }
                            }
                        }
                        final_result
                    }
                    Err(err) => crate::ToolResult::err_fmt(err.to_string()),
                };
                (
                    eid,
                    call_id,
                    tool_name,
                    args,
                    result,
                    tool_start.elapsed().as_millis() as u64,
                )
            });
        }

        let mut outcomes = Vec::new();
        outcomes.extend(immediate);
        while let Some(joined) = join_set.join_next().await {
            match joined {
                Ok(outcome) => outcomes.push(outcome),
                Err(e) => outcomes.push((
                    crate::sansio::EffectId(0),
                    uuid::Uuid::new_v4().to_string(),
                    "unknown".to_string(),
                    serde_json::json!({}),
                    crate::ToolResult::err_fmt(format!("tool task panicked: {e}")),
                    0,
                )),
            }
        }
        outcomes
    }
}

async fn forward_standard_stream_event(
    event_tx: &mpsc::Sender<AgentEvent>,
    stream_event: LlmStreamEvent,
    text_streamed: &mut bool,
    streamed_usage: &mut LlmUsage,
) {
    match stream_event {
        LlmStreamEvent::Delta(delta) => {
            if !delta.is_empty() {
                *text_streamed = true;
                crate::agent::send_event(event_tx, AgentEvent::TextDelta { content: delta }).await;
            }
        }
        LlmStreamEvent::Part(LlmOutputPart::Text { text }) => {
            if !text.is_empty() {
                *text_streamed = true;
                crate::agent::send_event(event_tx, AgentEvent::TextDelta { content: text }).await;
            }
        }
        LlmStreamEvent::Part(LlmOutputPart::ToolCall { .. }) => {}
        LlmStreamEvent::Usage(usage) => *streamed_usage = usage,
    }
}

async fn drain_standard_stream_queue(
    event_tx: &mpsc::Sender<AgentEvent>,
    llm_stream_rx: &mut tokio::sync::mpsc::UnboundedReceiver<LlmStreamEvent>,
    text_streamed: &mut bool,
    streamed_usage: &mut LlmUsage,
) {
    while let Ok(stream_event) = llm_stream_rx.try_recv() {
        forward_standard_stream_event(event_tx, stream_event, text_streamed, streamed_usage).await;
    }
}

fn response_usage_is_empty(usage: &LlmUsage) -> bool {
    usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_input_tokens == 0
        && usage.reasoning_tokens == 0
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
            &self.base_dir,
            self.path_resolver.as_deref(),
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
            InputItem::SkillRef { name, args } => {
                if name.is_empty() {
                    return Err("Invalid skill_ref: name cannot be empty".to_string());
                }
                let mut marker = format!("[SKILL:{name}]");
                if let Some(args) = args.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    marker.push(' ');
                    marker.push_str(args);
                }
                push_text(&mut out, marker);
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

fn sanitize_assistant_output(text: String, policy: &SanitizerPolicy) -> String {
    if !policy.strip_repl_fragments || text.is_empty() {
        return text;
    }
    let stripped = strip_repl_fragments(&text);
    let normalized = stripped
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    normalized.trim().to_string()
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
    if safe_text != raw_text {
        return OutputState::Sanitized;
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use crate::llm::transport::LlmTransportError;
    use crate::llm::types::{LlmRequest, LlmUsage};
    use crate::provider::Provider;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    fn default_state() -> AgentStateEnvelope {
        AgentStateEnvelope::default()
    }

    #[derive(Clone, Default)]
    struct RecordingSink {
        events: Arc<Mutex<Vec<AgentEvent>>>,
    }

    #[async_trait::async_trait]
    impl EventSink for RecordingSink {
        async fn emit(&self, event: AgentEvent) {
            self.events.lock().expect("lock sink").push(event);
        }
    }

    impl RecordingSink {
        fn snapshot(&self) -> Vec<AgentEvent> {
            self.events.lock().expect("lock sink").clone()
        }
    }

    struct MockCall {
        stream_events: Vec<LlmStreamEvent>,
        response: Result<LlmResponse, LlmTransportError>,
    }

    #[derive(Clone)]
    struct MockTransport {
        calls: Arc<Mutex<Vec<MockCall>>>,
    }

    impl MockTransport {
        fn new(calls: Vec<MockCall>) -> Self {
            Self {
                calls: Arc::new(Mutex::new(calls)),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmTransport for MockTransport {
        fn default_root_model(&self) -> &'static str {
            "mock-model"
        }

        fn default_agent_model(&self, _tier: &str) -> Option<crate::llm::types::ModelSelection> {
            None
        }

        fn requires_streaming(&self) -> bool {
            true
        }

        fn normalize_model(&self, model: &str) -> String {
            model.to_string()
        }

        fn context_lookup_model(&self, model: &str) -> String {
            model.to_string()
        }

        async fn ensure_ready(&self, _provider: &mut Provider) -> Result<bool, LlmTransportError> {
            Ok(false)
        }

        async fn complete(
            &self,
            _provider: &mut Provider,
            req: LlmRequest,
        ) -> Result<LlmResponse, LlmTransportError> {
            let call = self.calls.lock().expect("lock calls").remove(0);
            if let Some(tx) = req.stream_events.as_ref() {
                for event in &call.stream_events {
                    let _ = tx.send(event.clone());
                }
            }
            call.response
        }
    }

    fn standard_test_config() -> RuntimeConfig {
        RuntimeConfig {
            execution_mode: ExecutionMode::Standard,
            provider: Provider::OpenAiGeneric {
                api_key: "test-key".to_string(),
                base_url: "https://example.invalid/v1".to_string(),
            },
            model: "mock-model".to_string(),
            headless: true,
            host_profile: HostProfile::Headless,
            ..RuntimeConfig::default()
        }
    }

    async fn standard_runtime_with_transport(transport: MockTransport) -> LashRuntime {
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(crate::ToolSet::new());
        let mut runtime = LashRuntime::from_state(
            standard_test_config(),
            crate::RuntimeServices::tools_only(tools, "root").expect("services"),
            AgentStateEnvelope::default(),
        )
        .await
        .expect("runtime");
        runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
        runtime
    }

    struct RuntimeTestPluginFactory {
        build: Arc<
            dyn Fn(
                    &crate::PluginSessionContext,
                ) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError>
                + Send
                + Sync,
        >,
    }

    impl crate::PluginFactory for RuntimeTestPluginFactory {
        fn id(&self) -> &'static str {
            "runtime-test"
        }

        fn build(
            &self,
            ctx: &crate::PluginSessionContext,
        ) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError> {
            (self.build)(ctx)
        }
    }

    struct RuntimeTestPlugin {
        before_turn: Option<crate::plugin::BeforeTurnHook>,
        external_registrar: Option<
            Arc<
                dyn Fn(&mut crate::PluginRegistrar) -> Result<(), crate::PluginError> + Send + Sync,
            >,
        >,
    }

    impl crate::SessionPlugin for RuntimeTestPlugin {
        fn id(&self) -> &'static str {
            "runtime-test"
        }

        fn register(&self, reg: &mut crate::PluginRegistrar) -> Result<(), crate::PluginError> {
            if let Some(hook) = &self.before_turn {
                reg.before_turn(Arc::clone(hook));
            }
            if let Some(register) = &self.external_registrar {
                register(reg)?;
            }
            Ok(())
        }
    }

    async fn runtime_with_plugins(
        plugins: Vec<Arc<dyn crate::PluginFactory>>,
        transport: MockTransport,
    ) -> LashRuntime {
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(crate::ToolSet::new());
        let plugin_host = crate::PluginHost::new(plugins);
        let plugin_session = plugin_host.build_session("root", None).expect("plugins");
        let mut runtime = LashRuntime::from_state(
            standard_test_config(),
            crate::RuntimeServices::new(tools, plugin_session),
            AgentStateEnvelope::default(),
        )
        .await
        .expect("runtime");
        runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
        runtime
    }

    #[tokio::test]
    async fn plugin_before_turn_can_abort_and_inject_messages() {
        let plugin = Arc::new(RuntimeTestPluginFactory {
            build: Arc::new(|_| {
                Ok(Arc::new(RuntimeTestPlugin {
                    before_turn: Some(Arc::new(|_| {
                        Box::pin(async {
                            Ok(vec![
                                crate::PluginDirective::EnqueueMessages {
                                    messages: vec![crate::PluginMessage {
                                        role: crate::MessageRole::System,
                                        content: "plugin preface".to_string(),
                                    }],
                                },
                                crate::PluginDirective::AbortTurn {
                                    code: "blocked".to_string(),
                                    message: "plugin stopped the turn".to_string(),
                                },
                            ])
                        })
                    })),
                    external_registrar: None,
                }))
            }),
        });
        let transport = MockTransport::new(Vec::new());
        let mut runtime = runtime_with_plugins(vec![plugin], transport).await;

        let turn = runtime
            .run_turn_assembled(
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "hello".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    mode: None,
                    plan_file: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("turn");

        assert_eq!(turn.status, TurnStatus::Failed);
        assert_eq!(turn.done_reason, DoneReason::RuntimeError);
        assert!(turn.errors.iter().any(|issue| issue.kind == "plugin"));
        assert!(turn.state.messages.iter().any(|message| {
            message
                .parts
                .iter()
                .any(|part| part.content.contains("plugin preface"))
        }));
    }

    #[tokio::test]
    async fn external_invoke_can_create_session_from_current_snapshot() {
        let plugin = Arc::new(RuntimeTestPluginFactory {
            build: Arc::new(|_| {
                Ok(Arc::new(RuntimeTestPlugin {
                    before_turn: None,
                    external_registrar: Some(Arc::new(|reg| {
                        reg.register_external_op(
                            crate::ExternalOpDef {
                                name: "test.spawn".to_string(),
                                description: "spawn".to_string(),
                                kind: crate::ExternalOpKind::Command,
                                session_param: crate::SessionParam::Optional,
                                input_schema: json!({}),
                                output_schema: json!({}),
                            },
                            Arc::new(|ctx, _args| {
                                Box::pin(async move {
                                    let handle = ctx
                                        .host
                                        .create_session(crate::SessionCreateRequest {
                                            agent_id: Some("branched".to_string()),
                                            start: crate::SessionStartPoint::CurrentSession,
                                            config_overrides:
                                                crate::SessionConfigOverrides::default(),
                                            initial_messages: vec![crate::PluginMessage {
                                                role: crate::MessageRole::User,
                                                content: "branch seed".to_string(),
                                            }],
                                        })
                                        .await
                                        .map_err(|err| crate::ToolResult::err_fmt(err.to_string()));
                                    match handle {
                                        Ok(handle) => {
                                            let snapshot = ctx
                                                .host
                                                .snapshot_session(&handle.session_id)
                                                .await
                                                .map_err(|err| {
                                                    crate::ToolResult::err_fmt(err.to_string())
                                                });
                                            match snapshot {
                                                Ok(snapshot) => crate::ToolResult::ok(json!({
                                                    "session_id": handle.session_id,
                                                    "message_count": snapshot.messages.len(),
                                                })),
                                                Err(err) => err,
                                            }
                                        }
                                        Err(err) => err,
                                    }
                                })
                            }),
                        )
                    })),
                }))
            }),
        });
        let transport = MockTransport::new(Vec::new());
        let mut runtime = runtime_with_plugins(vec![plugin], transport).await;

        runtime.state.messages.push(Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Text,
                content: "root msg".to_string(),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            origin: None,
        });

        let result = runtime
            .invoke_external("test.spawn", json!({}), None)
            .await
            .expect("invoke");
        assert!(result.success);
        assert_eq!(
            result
                .result
                .get("session_id")
                .and_then(|value| value.as_str()),
            Some("branched")
        );
        assert_eq!(
            result
                .result
                .get("message_count")
                .and_then(|value| value.as_u64()),
            Some(2)
        );
    }

    #[test]
    fn assembler_prefers_final_message() {
        let mut assembler = TurnAssembler::default();
        assembler.push(&AgentEvent::TextDelta {
            content: "stream".to_string(),
        });
        assembler.push(&AgentEvent::Message {
            text: "final".to_string(),
            kind: "final".to_string(),
        });
        assembler.push(&AgentEvent::Done);
        let out = assembler.finish(
            default_state(),
            false,
            None,
            &SanitizerPolicy::default(),
            &TerminationPolicy::default(),
        );
        assert_eq!(out.status, TurnStatus::Completed);
        assert_eq!(out.done_reason, DoneReason::ModelStop);
        assert_eq!(out.assistant_output.safe_text, "final");
        assert_eq!(out.assistant_output.raw_text, "final");
        assert_eq!(out.assistant_output.state, OutputState::Usable);
    }

    #[test]
    fn assembler_marks_tool_failure() {
        let mut assembler = TurnAssembler::default();
        assembler.push(&AgentEvent::ToolCall {
            name: "x".to_string(),
            args: serde_json::json!({}),
            result: serde_json::json!({"error": true}),
            success: false,
            duration_ms: 1,
        });
        assembler.push(&AgentEvent::Error {
            message: "tool failed".to_string(),
            envelope: None,
        });
        assembler.push(&AgentEvent::Done);
        let out = assembler.finish(
            default_state(),
            false,
            None,
            &SanitizerPolicy::default(),
            &TerminationPolicy::default(),
        );
        assert_eq!(out.status, TurnStatus::Failed);
        assert_eq!(out.done_reason, DoneReason::ToolFailure);
        assert_eq!(out.tool_calls.len(), 1);
    }

    #[test]
    fn assembler_marks_missing_done_as_failure() {
        let mut assembler = TurnAssembler::default();
        assembler.push(&AgentEvent::TextDelta {
            content: "partial".to_string(),
        });
        let out = assembler.finish(
            default_state(),
            false,
            None,
            &SanitizerPolicy::default(),
            &TerminationPolicy::default(),
        );
        assert_eq!(out.status, TurnStatus::Failed);
        assert_eq!(out.done_reason, DoneReason::RuntimeError);
    }

    #[test]
    fn assembler_detects_max_turn_message() {
        let mut state = default_state();
        state.messages.push(Message {
            id: "m0".to_string(),
            role: MessageRole::System,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Text,
                content: "Turn limit reached (5).".to_string(),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            origin: None,
        });
        let mut assembler = TurnAssembler::default();
        assembler.push(&AgentEvent::Done);
        let out = assembler.finish(
            state,
            false,
            None,
            &SanitizerPolicy::default(),
            &TerminationPolicy::default(),
        );
        assert_eq!(out.status, TurnStatus::Completed);
        assert_eq!(out.done_reason, DoneReason::MaxTurns);
    }

    #[test]
    fn sanitizer_strips_repl_tags() {
        let cleaned = sanitize_assistant_output(
            "<repl>print('x')</repl> done".to_string(),
            &SanitizerPolicy {
                strip_repl_fragments: true,
            },
        );
        assert_eq!(cleaned, "print('x') done");
    }

    #[test]
    fn sanitizer_strips_dangling_repl_fragments() {
        let cleaned = sanitize_assistant_output(
            "status <repl\nstill here </repl".to_string(),
            &SanitizerPolicy {
                strip_repl_fragments: true,
            },
        );
        assert_eq!(cleaned, "status\nstill here");
    }

    #[test]
    fn output_state_empty_output() {
        assert_eq!(classify_output_state("", "", &[]), OutputState::EmptyOutput);
    }

    #[test]
    fn output_state_traceback_only() {
        let raw = "Runtime error: Traceback (most recent call last):\nFile \"repl_1.py\", line 2, in <module>\nNameError: name 'now' is not defined";
        assert_eq!(
            classify_output_state(raw, "", &[]),
            OutputState::TracebackOnly
        );
    }

    #[test]
    fn output_state_sanitized() {
        let raw = "<repl>print('x')</repl> done";
        let safe = "print('x') done";
        assert_eq!(
            classify_output_state(raw, safe, &[]),
            OutputState::Sanitized
        );
    }

    #[test]
    fn output_state_recovered_from_error() {
        let issues = vec![TurnIssue {
            kind: "runtime".to_string(),
            code: Some("example".to_string()),
            message: "something failed".to_string(),
        }];
        assert_eq!(
            classify_output_state("raw", "usable", &issues),
            OutputState::RecoveredFromError
        );
    }

    #[test]
    fn normalize_items_resolves_relative_paths_with_base_dir() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let file_path = tmp.path().join("a.txt");
        std::fs::write(&file_path, "x").expect("write");
        let dir_path = tmp.path().join("sub");
        std::fs::create_dir_all(&dir_path).expect("mkdir");

        let items = vec![
            InputItem::FileRef {
                path: "a.txt".to_string(),
            },
            InputItem::DirRef {
                path: "sub".to_string(),
            },
        ];
        let out =
            normalize_input_items(&items, &HashMap::new(), tmp.path(), None).expect("normalized");
        assert_eq!(out.len(), 1);
        match &out[0] {
            NormalizedItem::Text(text) => {
                assert!(text.contains("[file:"));
                assert!(text.contains("[directory:"));
            }
            _ => panic!("expected merged text item"),
        }
    }

    #[tokio::test]
    async fn standard_runtime_assembles_stream_only_text_response() {
        let transport = MockTransport::new(vec![MockCall {
            stream_events: vec![
                LlmStreamEvent::Delta("What time ".to_string()),
                LlmStreamEvent::Part(LlmOutputPart::Text {
                    text: "is it?".to_string(),
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 11,
                    output_tokens: 4,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse {
                full_text: "What time is it?".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "What time is it?".to_string(),
                }],
                ..LlmResponse::default()
            }),
        }]);
        let mut runtime = standard_runtime_with_transport(transport).await;
        let sink = RecordingSink::default();

        let turn = runtime
            .stream_turn(
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "hi".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    mode: None,
                    plan_file: None,
                },
                &sink,
                CancellationToken::new(),
            )
            .await
            .expect("turn");

        assert_eq!(turn.status, TurnStatus::Completed);
        assert_eq!(turn.done_reason, DoneReason::ModelStop);
        assert_eq!(turn.assistant_output.safe_text, "What time is it?");

        let streamed_text: String = sink
            .snapshot()
            .into_iter()
            .filter_map(|event| match event {
                AgentEvent::TextDelta { content } => Some(content),
                _ => None,
            })
            .collect();
        assert_eq!(streamed_text, "What time is it?");
    }

    #[tokio::test]
    async fn standard_runtime_uses_streamed_usage_when_final_usage_missing() {
        let transport = MockTransport::new(vec![MockCall {
            stream_events: vec![
                LlmStreamEvent::Delta("Hi".to_string()),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 9,
                    output_tokens: 3,
                    cached_input_tokens: 2,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse {
                full_text: "Hi".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Hi".to_string(),
                }],
                usage: LlmUsage::default(),
                ..LlmResponse::default()
            }),
        }]);
        let mut runtime = standard_runtime_with_transport(transport).await;

        let turn = runtime
            .run_turn_assembled(
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "hello".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    mode: None,
                    plan_file: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("turn");

        assert_eq!(turn.token_usage.input_tokens, 9);
        assert_eq!(turn.token_usage.output_tokens, 3);
        assert_eq!(turn.token_usage.cached_input_tokens, 2);
    }

    #[tokio::test]
    async fn standard_runtime_prefers_final_usage_over_streamed_usage() {
        let transport = MockTransport::new(vec![MockCall {
            stream_events: vec![
                LlmStreamEvent::Delta("Hi".to_string()),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 9,
                    output_tokens: 3,
                    cached_input_tokens: 2,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse {
                full_text: "Hi".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Hi".to_string(),
                }],
                usage: LlmUsage {
                    input_tokens: 12,
                    output_tokens: 4,
                    cached_input_tokens: 1,
                    reasoning_tokens: 0,
                },
                ..LlmResponse::default()
            }),
        }]);
        let mut runtime = standard_runtime_with_transport(transport).await;

        let turn = runtime
            .run_turn_assembled(
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "hello".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    mode: None,
                    plan_file: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("turn");

        assert_eq!(turn.token_usage.input_tokens, 12);
        assert_eq!(turn.token_usage.output_tokens, 4);
        assert_eq!(turn.token_usage.cached_input_tokens, 1);
    }

    #[cfg(feature = "sqlite-store")]
    #[tokio::test]
    async fn completed_turns_are_persisted_for_search_history() {
        let transport = MockTransport::new(vec![MockCall {
            stream_events: vec![LlmStreamEvent::Delta("Stored answer".to_string())],
            response: Ok(LlmResponse {
                full_text: "Stored answer".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Stored answer".to_string(),
                }],
                ..LlmResponse::default()
            }),
        }]);

        let store = Arc::new(crate::store::Store::memory().expect("store"));
        let base_provider: Arc<dyn crate::ToolProvider> =
            Arc::new(crate::ToolSet::new() + crate::tools::StateStore::new(Vec::new()));
        let plugin_host = crate::PluginHost::new(vec![Arc::new(
            crate::BuiltinHistoryPluginFactory::new(Arc::clone(&store)),
        )]);
        let plugins = plugin_host.build_session("root", None).expect("plugins");
        let mut toolset = crate::ToolSet::new() + Arc::clone(&base_provider);
        for provider in plugins.tool_providers() {
            toolset = toolset + Arc::clone(provider);
        }
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(toolset);
        let mut runtime = LashRuntime::from_state(
            standard_test_config(),
            crate::RuntimeServices::new(tools, Arc::clone(&plugins)),
            AgentStateEnvelope::default(),
        )
        .await
        .expect("runtime");
        runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));

        let _turn = runtime
            .run_turn_assembled(
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "where did this go?".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    mode: None,
                    plan_file: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("turn");

        let history_provider = plugins
            .tool_providers()
            .iter()
            .find(|provider| {
                provider
                    .definitions()
                    .iter()
                    .any(|def| def.name == "search_history")
            })
            .cloned()
            .expect("history provider");
        let result = history_provider
            .execute(
                "search_history",
                &serde_json::json!({
                    "__agent_id__":"root",
                    "query":"where did this go",
                    "mode":"hybrid",
                    "limit":10
                }),
            )
            .await;
        assert!(result.success);
        let items = result.result.as_array().cloned().unwrap_or_default();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].get("turn").and_then(|v| v.as_i64()), Some(1));
        assert!(
            items[0]
                .get("preview")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .contains("where did this go")
        );
    }

    #[tokio::test]
    async fn drain_standard_stream_queue_forwards_prequeued_text() {
        let (event_tx, mut event_rx) = mpsc::channel(8);
        let (llm_stream_tx, mut llm_stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
        llm_stream_tx
            .send(LlmStreamEvent::Delta("Hello".to_string()))
            .expect("delta");
        llm_stream_tx
            .send(LlmStreamEvent::Part(LlmOutputPart::Text {
                text: " there".to_string(),
            }))
            .expect("part");
        drop(llm_stream_tx);

        let mut text_streamed = false;
        let mut streamed_usage = LlmUsage::default();
        drain_standard_stream_queue(
            &event_tx,
            &mut llm_stream_rx,
            &mut text_streamed,
            &mut streamed_usage,
        )
        .await;
        drop(event_tx);

        let mut streamed_text = String::new();
        while let Some(event) = event_rx.recv().await {
            if let AgentEvent::TextDelta { content } = event {
                streamed_text.push_str(&content);
            }
        }

        assert!(text_streamed);
        assert_eq!(streamed_text, "Hello there");
    }
}
