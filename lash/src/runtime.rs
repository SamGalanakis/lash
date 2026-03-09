use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent::message::IMAGE_REF_PREFIX;
use crate::agent::{
    AgentConfig, AgentEvent, Message, MessageRole, Part, PartKind, PromptSectionOverride,
    PruneState, TokenUsage, build_execution_preamble, make_error_event, transport_stream_events,
};
use crate::capabilities::AgentCapabilities;
use crate::instructions::{FsInstructionSource, InstructionSource};
use crate::llm::factory::adapter_for;
use crate::llm::types::{LlmOutputPart, LlmRequest, LlmResponse, LlmStreamEvent};
use crate::provider::Provider;
use crate::sansio::{Effect, LlmCallError, Response, TurnMachine, TurnMachineConfig};
use crate::strip_repl_fragments;
use crate::{
    CapabilityId, ContextFoldingConfig, ExecutionMode, SandboxMessage, Session, SessionError,
    ToolCallRecord, ToolProvider,
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

/// Generic runtime engine for CLI or programmatic embedding.
pub struct RuntimeEngine {
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
}

impl RuntimeEngine {
    /// Build a runtime from host-provided state + tools.
    pub async fn from_state(
        config: RuntimeConfig,
        tools: Arc<dyn ToolProvider>,
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
        tools.set_execution_mode(state.execution_mode);
        let mut session = Session::new(
            tools,
            &state.agent_id,
            config.headless,
            capabilities.clone(),
            state.execution_mode,
        )
        .await?;
        if matches!(state.execution_mode, ExecutionMode::Repl)
            && let Some(snapshot) = state.repl_snapshot.clone()
        {
            session.restore(&snapshot).await?;
        }
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
        })
    }

    /// Export current host-owned state envelope.
    pub fn export_state(&self) -> AgentStateEnvelope {
        self.state.clone()
    }

    /// Replace the host-owned state envelope.
    pub fn set_state(&mut self, state: AgentStateEnvelope) {
        if let Some(session) = self.session.as_ref() {
            session.tools().set_execution_mode(state.execution_mode);
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
        messages: Vec<Message>,
        images_png: Vec<Vec<u8>>,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let cancel_state = cancel.clone();
        let session = self
            .session
            .take()
            .expect("runtime engine session must be available");
        let mut driver = RuntimeTurnDriver {
            session,
            config: self.config.clone(),
            cached_base_context: self.cached_base_context.take(),
            agent_id: self.state.agent_id.clone(),
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

        Ok(assembler.finish(
            self.state.clone(),
            cancel_state.is_cancelled(),
            None,
            &self.sanitizer,
            &self.termination,
        ))
    }
}

struct RuntimeTurnDriver {
    session: Session,
    config: RuntimeConfig,
    cached_base_context: Option<String>,
    agent_id: String,
}

impl RuntimeTurnDriver {
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

        if matches!(self.config.execution_mode, ExecutionMode::NativeTools) {
            return self
                .run_native_tools(messages, images, event_tx, cancel, run_offset)
                .await;
        }

        let capabilities_json = self
            .session
            .tools()
            .dynamic_capabilities_payload_json()
            .unwrap_or_else(|| {
                serde_json::json!({
                    "enabled_capabilities": self
                        .config
                        .capabilities
                        .enabled_capabilities
                        .iter()
                        .map(|id| id.as_str())
                        .collect::<Vec<_>>(),
                    "enabled_tools": self
                        .config
                        .capabilities
                        .enabled_tools
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>(),
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

    async fn run_native_tools(
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
            ExecutionMode::NativeTools,
            model,
        );
        self.config = agent_config.into();

        let max_context = self.max_context_tokens(&preamble.model);
        let machine_config = self.machine_config(preamble, max_context, ExecutionMode::NativeTools);
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
                    let (result, text_streamed) =
                        self.run_native_llm_call(request, &event_tx, &cancel).await;
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

        let llm = adapter_for(&config.provider);
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
        let history_enabled = if matches!(execution_mode, ExecutionMode::NativeTools) {
            preamble.helper_bindings.contains("search_history")
                || preamble.enabled_capability_ids.contains("history")
        } else {
            preamble.history_enabled
        };
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
            enabled_capability_ids: preamble.enabled_capability_ids,
            helper_bindings: preamble.helper_bindings,
            capability_prompt_sections: preamble.capability_prompt_sections,
            can_write: preamble.can_write,
            history_enabled,
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
        let mut llm_task = tokio::spawn(async move {
            let llm = adapter_for(&call_provider);
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

    async fn run_native_llm_call(
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
        let mut llm_task = tokio::spawn(async move {
            let llm = adapter_for(&call_provider);
            let result = llm.complete(&mut call_provider, llm_request).await;
            (result, call_provider)
        });

        let mut text_streamed = false;
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
                    match stream_event {
                        LlmStreamEvent::Delta(delta) => {
                            if !delta.is_empty() {
                                text_streamed = true;
                                crate::agent::send_event(event_tx, AgentEvent::TextDelta { content: delta }).await;
                            }
                        }
                        LlmStreamEvent::Part(LlmOutputPart::Text { text }) => {
                            if !text.is_empty() {
                                text_streamed = true;
                                crate::agent::send_event(event_tx, AgentEvent::TextDelta { content: text }).await;
                            }
                        }
                        LlmStreamEvent::Part(LlmOutputPart::ToolCall { .. }) => {}
                        LlmStreamEvent::Usage(_) => {}
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
        let mut join_set = tokio::task::JoinSet::new();
        for (eid, call_id, tool_name, mut args) in pending_tools {
            if (tool_name == "list_tools" || tool_name == "search_tools")
                && let Some(obj) = args.as_object_mut()
                && !obj.contains_key("catalog")
            {
                let catalog: Vec<serde_json::Value> = tool_provider
                    .definitions()
                    .into_iter()
                    .filter(|d| {
                        !d.hidden && !d.description_for(ExecutionMode::NativeTools).is_empty()
                    })
                    .map(|d| {
                        let p = d.project(ExecutionMode::NativeTools);
                        serde_json::json!({
                            "name": p.name,
                            "description": p.description,
                            "examples": p.examples,
                            "inject_into_prompt": p.inject_into_prompt,
                        })
                    })
                    .collect();
                obj.insert("catalog".to_string(), serde_json::Value::Array(catalog));
            }

            let provider = Arc::clone(&tool_provider);
            let event_tx_clone = event_tx.clone();
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

impl RuntimeEngine {
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

    fn default_state() -> AgentStateEnvelope {
        AgentStateEnvelope::default()
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
}
