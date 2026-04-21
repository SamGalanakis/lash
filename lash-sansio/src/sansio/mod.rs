//! Sans-IO state machine for session turns.
//!
//! `TurnMachine` owns the generic effect engine. Protocol-specific behavior
//! lives behind `ProtocolDriverHandle`, which returns declarative
//! `DriverAction`s that the machine applies.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::llm::types::{
    LlmAttachment, LlmOutputPart, LlmRequest, LlmResponse, LlmToolChoice, LlmToolSpec,
};
use crate::session_model::message::{MessageOrigin, PartAttachment, data_url_for_bytes};
use crate::session_model::{
    Message, MessageRole, MessageSequence, Part, PartKind, PruneState, RetryPolicy, SessionEvent,
    TokenUsage, TurnTerminationPolicyState, fresh_message_id, make_error_envelope,
    make_error_event, reassign_part_ids,
};
use crate::{CheckpointKind, PluginMessage, ToolResult};

// ─── Public types ───

/// Opaque identifier linking an effect to its response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectId(pub u64);

#[derive(Clone, Debug)]
pub struct PendingToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub args: Value,
    /// Provider-specific item-id (e.g. Codex `fc_...`). Carried through so
    /// it can be re-emitted on the next request body. `None` for providers
    /// that don't surface a distinct item-id.
    pub item_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CompletedToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub args: Value,
    pub state_result: ToolResult,
    pub model_result: ToolResult,
    pub duration_ms: u64,
    /// See [`PendingToolCall::item_id`].
    pub item_id: Option<String>,
}

#[derive(Clone, Debug)]
pub enum LogEvent {
    LlmDebug {
        session_id: String,
        iteration: usize,
        usage: TokenUsage,
        provider_usage: Option<Value>,
        request_body: Option<String>,
        response_text: String,
        response_parts: Option<Value>,
    },
    LlmError {
        session_id: String,
        iteration: usize,
        request_body: Option<String>,
        message: String,
        retryable: bool,
        raw: Option<String>,
        code: Option<String>,
    },
}

/// An effect the host must fulfil.
#[derive(Debug)]
pub enum Effect {
    /// Sync the live RLM execution surface before the turn proceeds.
    SyncExecutionSurface { id: EffectId },
    /// Start an LLM call.
    LlmCall { id: EffectId, request: LlmRequest },
    /// Cancel an in-progress LLM stream.
    CancelLlm { id: EffectId },
    /// Execute one or more standard-mode tool calls.
    ToolCalls {
        id: EffectId,
        calls: Vec<PendingToolCall>,
    },
    /// Execute a RLM code block.
    ExecCode { id: EffectId, code: String },
    /// Run a host/plugin checkpoint before the machine continues or completes.
    Checkpoint {
        id: EffectId,
        checkpoint: CheckpointKind,
    },
    /// Retry backoff.
    Sleep { id: EffectId, duration: Duration },
    /// Host-implemented fire-and-forget logging.
    Log { event: LogEvent },
    /// Fire-and-forget event (no response needed).
    Emit(SessionEvent),
    /// Turn is done.
    Done {
        messages: MessageSequence,
        iteration: usize,
    },
}

/// Error details from a failed LLM call.
#[derive(Clone, Debug)]
pub struct LlmCallError {
    pub message: String,
    pub retryable: bool,
    pub raw: Option<String>,
    pub code: Option<String>,
    pub request_body: Option<String>,
}

/// A response to a previously emitted effect.
pub enum Response {
    /// Live RLM execution surface sync completed.
    ExecutionSurfaceSynced {
        id: EffectId,
        result: Result<(), String>,
    },
    /// Full LLM response.
    LlmComplete {
        id: EffectId,
        result: Result<LlmResponse, LlmCallError>,
        /// When true, text deltas were already emitted during streaming,
        /// so the driver should skip emitting `TextDelta` events.
        text_streamed: bool,
    },
    /// Native tool results.
    ToolResults {
        id: EffectId,
        results: Vec<CompletedToolCall>,
    },
    /// RLM code execution result.
    ExecResult {
        id: EffectId,
        result: Result<crate::ExecResponse, String>,
    },
    /// Checkpoint result with optional injected messages.
    Checkpoint {
        id: EffectId,
        messages: Vec<PluginMessage>,
        transient_messages: Vec<PluginMessage>,
    },
    /// Sleep completed.
    Timeout { id: EffectId },
}

pub type DriverState = Box<dyn Any + Send + Sync>;

pub fn driver_state<T>(state: T) -> DriverState
where
    T: Any + Send + Sync,
{
    Box::new(state)
}

pub struct WaitingLlmState {
    pub retry_attempt: usize,
    pub request: LlmRequest,
    driver_state: Option<DriverState>,
}

impl WaitingLlmState {
    pub fn take_driver_state<T>(&mut self) -> Option<T>
    where
        T: Any + Send + Sync,
    {
        self.driver_state
            .take()
            .and_then(|state| state.downcast::<T>().ok())
            .map(|state| *state)
    }
}

pub struct WaitingExecState {
    driver_state: DriverState,
}

impl WaitingExecState {
    pub fn into_driver_state<T>(self) -> Option<T>
    where
        T: Any + Send + Sync,
    {
        self.driver_state.downcast::<T>().ok().map(|state| *state)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointResumeAction {
    PrepareIteration,
    Finish,
}

pub enum DriverAction {
    Emit(SessionEvent),
    AppendMessages(Vec<Message>),
    StartLlm {
        request: LlmRequest,
        driver_state: Option<DriverState>,
    },
    StartTools {
        calls: Vec<PendingToolCall>,
    },
    StartExec {
        code: String,
        driver_state: DriverState,
    },
    StartCheckpoint {
        checkpoint: CheckpointKind,
        on_empty: CheckpointResumeAction,
    },
    AdvanceIteration,
    ScheduleTurnLimitFinal,
    Finish,
}

pub struct DriverContextView<'a> {
    config: &'a TurnMachineConfig,
    messages: &'a MessageSequence,
    iteration: usize,
    run_offset: usize,
    termination: &'a TurnTerminationPolicyState,
}

impl<'a> DriverContextView<'a> {
    pub fn build_llm_request(&self, use_tools: bool) -> LlmRequest {
        let rendered_prompt = self.messages.render_prompt();
        let attachments: Vec<LlmAttachment> = rendered_prompt.attachments;
        let mut messages = rendered_prompt.messages;
        if !self.config.system_prompt.trim().is_empty() {
            messages.insert(
                0,
                crate::llm::types::LlmMessage::text(
                    crate::llm::types::LlmRole::System,
                    self.config.system_prompt.clone(),
                ),
            );
        }

        LlmRequest {
            model: self.config.model.clone(),
            messages,
            attachments,
            tools: if use_tools {
                Arc::clone(&self.config.tool_specs)
            } else {
                Arc::new(Vec::new())
            },
            tool_choice: if use_tools {
                LlmToolChoice::Auto
            } else {
                LlmToolChoice::None
            },
            model_variant: self.config.model_variant.clone(),
            session_id: self.config.run_session_id.clone(),
            output_spec: None,
            stream_events: None,
        }
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    pub fn run_offset(&self) -> usize {
        self.run_offset
    }

    pub fn max_turns(&self) -> Option<usize> {
        self.config.max_turns
    }

    pub fn rlm_termination(&self) -> &RlmTermination {
        &self.config.rlm_termination
    }

    pub fn should_force_exit_after_grace_turn(&self) -> bool {
        self.termination.should_force_exit_after_grace_turn()
    }

    pub fn messages(&self) -> &MessageSequence {
        self.messages
    }
}

pub trait ProtocolDriverHandle: Send + Sync {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction>;
    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_>,
        waiting: WaitingLlmState,
        llm_response: LlmResponse,
        text_streamed: bool,
    ) -> Vec<DriverAction>;
    fn handle_tool_results(
        &self,
        ctx: DriverContextView<'_>,
        completed: Vec<CompletedToolCall>,
    ) -> Vec<DriverAction>;
    fn handle_exec_result(
        &self,
        ctx: DriverContextView<'_>,
        waiting: WaitingExecState,
        result: Result<crate::ExecResponse, String>,
    ) -> Vec<DriverAction>;
}

/// Configuration for a `TurnMachine` instance.
pub struct TurnMachineConfig {
    pub protocol_driver: Arc<dyn ProtocolDriverHandle>,
    pub sync_execution_surface: bool,
    pub model: String,
    pub max_turns: Option<usize>,
    pub model_variant: Option<String>,
    pub run_session_id: Option<String>,
    pub tool_specs: Arc<Vec<LlmToolSpec>>,
    pub system_prompt: String,
    pub session_id: String,
    pub emit_llm_debug_log: bool,
    pub rlm_termination: RlmTermination,
    pub retry_policy: RetryPolicy,
}

/// How a RLM session terminates. Mirrors `lash::RlmTermination`;
/// duplicated here so the sans-io layer doesn't depend on the lash
/// host crate.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RlmTermination {
    /// Prose with no fenced lashlang block terminates the turn.
    #[default]
    ProseWithoutFence,
    /// The session terminates only when the code calls `submit <expr>`.
    Finish { schema: Option<serde_json::Value> },
}

// ─── Internal state ───

enum MachineState {
    PreparingMode,
    WaitingExecutionSurface {
        effect_id: EffectId,
    },
    PrepareIteration,
    WaitingLlm {
        effect_id: EffectId,
        request: LlmRequest,
        driver_state: Option<DriverState>,
        retry_attempt: usize,
    },
    WaitingRetry {
        effect_id: EffectId,
        retry_attempt: usize,
        request: LlmRequest,
        driver_state: Option<DriverState>,
    },
    WaitingTools {
        effect_id: EffectId,
    },
    WaitingExec {
        effect_id: EffectId,
        driver_state: DriverState,
    },
    WaitingCheckpoint {
        effect_id: EffectId,
        checkpoint: CheckpointKind,
        on_empty: CheckpointResumeAction,
    },
    Finished,
}

/// Sans-IO state machine for a single session run (multi-turn).
pub struct TurnMachine {
    config: TurnMachineConfig,
    state: MachineState,
    pending_effects: VecDeque<Effect>,
    next_effect_id: u64,
    messages: MessageSequence,
    iteration: usize,
    run_offset: usize,
    cumulative_usage: TokenUsage,
    termination: TurnTerminationPolicyState,
}

impl TurnMachine {
    /// Create a new machine in `PrepareIteration` state.
    pub fn new(config: TurnMachineConfig, messages: Vec<Message>, run_offset: usize) -> Self {
        Self::new_shared(config, MessageSequence::from_owned(messages), run_offset)
    }

    pub fn new_shared(
        config: TurnMachineConfig,
        messages: MessageSequence,
        run_offset: usize,
    ) -> Self {
        Self {
            config,
            state: MachineState::PreparingMode,
            pending_effects: VecDeque::new(),
            next_effect_id: 1,
            messages,
            iteration: run_offset,
            run_offset,
            cumulative_usage: TokenUsage::default(),
            termination: TurnTerminationPolicyState::new(),
        }
    }

    /// Whether the machine has finished.
    pub fn is_done(&self) -> bool {
        matches!(self.state, MachineState::Finished)
    }

    pub fn messages(&self) -> Arc<Vec<Message>> {
        self.messages.shared()
    }

    pub fn materialized_messages(&self) -> Arc<Vec<Message>> {
        self.messages.shared()
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    fn driver_context(&self) -> DriverContextView<'_> {
        DriverContextView {
            config: &self.config,
            messages: &self.messages,
            iteration: self.iteration,
            run_offset: self.run_offset,
            termination: &self.termination,
        }
    }

    fn next_id(&mut self) -> EffectId {
        let id = EffectId(self.next_effect_id);
        self.next_effect_id += 1;
        id
    }

    fn emit(&mut self, event: SessionEvent) {
        self.pending_effects.push_back(Effect::Emit(event));
    }

    pub fn fail_turn(&mut self, event: SessionEvent) {
        self.emit(event);
        self.finish();
    }

    fn finish(&mut self) {
        self.emit(SessionEvent::Done);
        let msgs = std::mem::take(&mut self.messages);
        let iteration = self.iteration;
        self.state = MachineState::Finished;
        self.pending_effects.push_back(Effect::Done {
            messages: msgs,
            iteration,
        });
    }

    /// Drain the next pending effect. Returns `None` when the host must call
    /// `handle_response()` before more effects become available.
    pub fn poll_effect(&mut self) -> Option<Effect> {
        if let Some(effect) = self.pending_effects.pop_front() {
            return Some(effect);
        }

        match &self.state {
            MachineState::PreparingMode => {
                self.prepare_mode();
                self.pending_effects.pop_front()
            }
            MachineState::PrepareIteration => {
                self.prepare_iteration();
                self.pending_effects.pop_front()
            }
            _ => None,
        }
    }

    // ─── State transitions ───

    fn prepare_mode(&mut self) {
        if self.config.sync_execution_surface {
            let id = self.next_id();
            self.state = MachineState::WaitingExecutionSurface { effect_id: id };
            self.pending_effects
                .push_back(Effect::SyncExecutionSurface { id });
            return;
        }

        self.prepare_iteration();
    }

    fn prepare_iteration(&mut self) {
        let actions = {
            let driver = Arc::clone(&self.config.protocol_driver);
            let ctx = self.driver_context();
            driver.prepare_iteration(ctx)
        };
        self.apply_actions(actions);
    }

    fn start_llm_request(
        &mut self,
        request: LlmRequest,
        retry_attempt: usize,
        driver_state: Option<DriverState>,
    ) {
        let tool_list = self
            .config
            .tool_specs
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        self.emit(SessionEvent::LlmRequest {
            iteration: self.iteration,
            message_count: self.messages.len(),
            tool_list,
        });

        let id = self.next_id();
        self.state = MachineState::WaitingLlm {
            effect_id: id,
            request: request.clone(),
            driver_state,
            retry_attempt,
        };
        self.pending_effects
            .push_back(Effect::LlmCall { id, request });
    }

    fn start_tool_calls(&mut self, calls: Vec<PendingToolCall>) {
        let effect_id = self.next_id();
        self.state = MachineState::WaitingTools { effect_id };
        self.pending_effects.push_back(Effect::ToolCalls {
            id: effect_id,
            calls,
        });
    }

    fn start_exec(&mut self, code: String, driver_state: DriverState) {
        let effect_id = self.next_id();
        self.state = MachineState::WaitingExec {
            effect_id,
            driver_state,
        };
        self.pending_effects.push_back(Effect::ExecCode {
            id: effect_id,
            code,
        });
    }

    pub fn apply_actions(&mut self, actions: Vec<DriverAction>) {
        for action in actions {
            match action {
                DriverAction::Emit(event) => self.emit(event),
                DriverAction::AppendMessages(messages) => {
                    if !messages.is_empty() {
                        self.messages.extend(messages);
                    }
                }
                DriverAction::StartLlm {
                    request,
                    driver_state,
                } => self.start_llm_request(request, 0, driver_state),
                DriverAction::StartTools { calls } => self.start_tool_calls(calls),
                DriverAction::StartExec { code, driver_state } => {
                    self.start_exec(code, driver_state)
                }
                DriverAction::StartCheckpoint {
                    checkpoint,
                    on_empty,
                } => self.request_checkpoint(checkpoint, on_empty),
                DriverAction::AdvanceIteration => self.iteration += 1,
                DriverAction::ScheduleTurnLimitFinal => {
                    self.termination.maybe_schedule_turn_limit_final(
                        self.iteration,
                        self.run_offset,
                        self.config.max_turns,
                        self.messages.make_mut(),
                    );
                }
                DriverAction::Finish => {
                    self.finish();
                    break;
                }
            }
        }
    }

    /// Feed a response to a previously emitted effect.
    pub fn handle_response(&mut self, response: Response) {
        match response {
            Response::ExecutionSurfaceSynced { id, result } => {
                self.handle_execution_surface_synced(id, result)
            }
            Response::LlmComplete {
                id,
                result,
                text_streamed,
            } => self.handle_llm_complete(id, result, text_streamed),
            Response::ToolResults { id, results } => self.handle_tool_results(id, results),
            Response::ExecResult { id, result } => self.handle_exec_result(id, result),
            Response::Checkpoint {
                id,
                messages,
                transient_messages,
            } => self.handle_checkpoint(id, messages, transient_messages),
            Response::Timeout { id } => self.handle_timeout(id),
        }
    }

    fn request_checkpoint(&mut self, checkpoint: CheckpointKind, on_empty: CheckpointResumeAction) {
        let id = self.next_id();
        self.state = MachineState::WaitingCheckpoint {
            effect_id: id,
            checkpoint,
            on_empty,
        };
        self.pending_effects
            .push_back(Effect::Checkpoint { id, checkpoint });
    }

    fn handle_execution_surface_synced(&mut self, id: EffectId, result: Result<(), String>) {
        let waiting_id = match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingExecutionSurface { effect_id } => effect_id,
            other => {
                self.state = other;
                return;
            }
        };
        if waiting_id != id {
            self.state = MachineState::WaitingExecutionSurface {
                effect_id: waiting_id,
            };
            return;
        }

        match result {
            Ok(()) => {
                self.state = MachineState::PrepareIteration;
            }
            Err(error) => {
                self.fail_turn(make_error_event(
                    "execution_surface",
                    Some("reconfigure_failed"),
                    format!("Failed to refresh execution surface: {error}"),
                    Some(error),
                ));
            }
        }
    }

    fn append_checkpoint_messages(&mut self, plugin_messages: &[PluginMessage], transient: bool) {
        let appended = plugin_messages
            .iter()
            .filter(|message| matches!(message.role, MessageRole::User | MessageRole::System))
            .map(|message| {
                let message_id = fresh_message_id();
                let mut parts = if message.parts.is_empty() {
                    vec![Part {
                        id: format!("{message_id}.p0"),
                        kind: PartKind::Text,
                        content: message.content.clone(),
                        attachment: None,
                        tool_call_id: None,
                        tool_name: None,
                        tool_item_id: None,
                        tool_signature: None,
                        prune_state: PruneState::Intact,
                        reasoning_meta: None,
                    }]
                } else {
                    message.parts.clone()
                };
                let has_image_parts = parts
                    .iter()
                    .any(|part| matches!(part.kind, PartKind::Image));
                if matches!(message.role, MessageRole::User) && !has_image_parts {
                    parts.extend(message.images.iter().map(|bytes| Part {
                        id: String::new(),
                        kind: PartKind::Image,
                        content: String::new(),
                        attachment: Some(PartAttachment {
                            mime: "image/png".to_string(),
                            url: data_url_for_bytes("image/png", bytes),
                            filename: None,
                        }),
                        tool_call_id: None,
                        tool_name: None,
                        tool_item_id: None,
                        tool_signature: None,
                        prune_state: PruneState::Intact,
                        reasoning_meta: None,
                    }));
                }
                reassign_part_ids(&message_id, &mut parts);
                Message {
                    id: message_id.clone(),
                    role: message.role,
                    parts,
                    user_input: message.user_input.clone(),
                    origin: Some(MessageOrigin::Plugin {
                        plugin_id: "plugin".to_string(),
                        transient,
                    }),
                }
            })
            .collect::<Vec<_>>();
        if !appended.is_empty() {
            self.messages.extend(appended);
        }
    }

    fn handle_checkpoint(
        &mut self,
        id: EffectId,
        messages: Vec<PluginMessage>,
        transient_messages: Vec<PluginMessage>,
    ) {
        let (effect_id, checkpoint, on_empty) =
            match std::mem::replace(&mut self.state, MachineState::Finished) {
                MachineState::WaitingCheckpoint {
                    effect_id,
                    checkpoint,
                    on_empty,
                } => (effect_id, checkpoint, on_empty),
                other => {
                    self.state = other;
                    return;
                }
            };
        if effect_id != id {
            self.state = MachineState::WaitingCheckpoint {
                effect_id,
                checkpoint,
                on_empty,
            };
            return;
        }

        if !messages.is_empty() || !transient_messages.is_empty() {
            self.append_checkpoint_messages(&messages, false);
            self.append_checkpoint_messages(&transient_messages, true);
            if matches!(checkpoint, CheckpointKind::BeforeCompletion) {
                self.iteration += 1;
                if self.termination.should_force_exit_after_grace_turn() {
                    self.finish();
                    return;
                }
                self.termination.maybe_schedule_turn_limit_final(
                    self.iteration,
                    self.run_offset,
                    self.config.max_turns,
                    self.messages.make_mut(),
                );
            }
            self.state = MachineState::PrepareIteration;
            return;
        }

        match on_empty {
            CheckpointResumeAction::PrepareIteration => {
                self.state = MachineState::PrepareIteration;
            }
            CheckpointResumeAction::Finish => self.finish(),
        }
    }

    fn take_waiting_llm_state(&mut self, id: EffectId) -> Option<WaitingLlmState> {
        match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingLlm {
                effect_id,
                request,
                driver_state,
                retry_attempt,
            } if effect_id == id => Some(WaitingLlmState {
                retry_attempt,
                request,
                driver_state,
            }),
            other => {
                self.state = other;
                None
            }
        }
    }

    fn handle_llm_complete(
        &mut self,
        id: EffectId,
        result: Result<LlmResponse, LlmCallError>,
        text_streamed: bool,
    ) {
        let Some(waiting) = self.take_waiting_llm_state(id) else {
            return;
        };
        match result {
            Err(error) => {
                if error.retryable && waiting.retry_attempt < self.config.retry_policy.max_retries {
                    self.schedule_llm_retry(
                        waiting.retry_attempt,
                        error,
                        waiting.request,
                        waiting.driver_state,
                    );
                    return;
                }
                self.emit_llm_error(error);
            }
            Ok(llm_response) => {
                self.record_llm_usage(&llm_response, self.llm_response_text(&llm_response));
                let actions = {
                    let driver = Arc::clone(&self.config.protocol_driver);
                    let ctx = self.driver_context();
                    driver.handle_llm_success(ctx, waiting, llm_response, text_streamed)
                };
                self.apply_actions(actions);
            }
        }
    }

    fn schedule_llm_retry(
        &mut self,
        retry_attempt: usize,
        error: LlmCallError,
        request: LlmRequest,
        driver_state: Option<DriverState>,
    ) {
        self.record_llm_error(&error);
        let delay = self.config.retry_policy.delay_for_attempt(retry_attempt);
        let reason = error.message.clone();
        self.emit(SessionEvent::RetryStatus {
            wait_seconds: delay.as_secs(),
            attempt: retry_attempt + 2,
            max_attempts: self.config.retry_policy.max_retries + 1,
            reason: reason.clone(),
            envelope: Some(make_error_envelope(
                "llm_provider",
                error.code.as_deref(),
                format!("LLM error: {reason}"),
                error.raw,
            )),
        });
        let sleep_id = self.next_id();
        self.state = MachineState::WaitingRetry {
            effect_id: sleep_id,
            retry_attempt: retry_attempt + 1,
            request,
            driver_state,
        };
        self.pending_effects.push_back(Effect::Sleep {
            id: sleep_id,
            duration: delay,
        });
    }

    fn llm_response_text<'a>(&self, llm_response: &'a LlmResponse) -> &'a str {
        &llm_response.full_text
    }

    fn llm_response_debug_parts(&self, llm_response: &LlmResponse) -> Option<Value> {
        let parts = llm_response
            .parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text } if !text.is_empty() => Some(serde_json::json!({
                    "type": "text",
                    "text": text,
                })),
                LlmOutputPart::Text { .. } => None,
                LlmOutputPart::Reasoning {
                    text,
                    item_id,
                    summary,
                    encrypted_content,
                    signature,
                    redacted,
                } => Some(serde_json::json!({
                    "type": "reasoning",
                    "id": item_id,
                    "summary": summary,
                    "text": text,
                    "has_encrypted": encrypted_content.is_some() || signature.is_some(),
                    "redacted": redacted,
                })),
                LlmOutputPart::ToolCall {
                    call_id,
                    tool_name,
                    input_json,
                    item_id,
                    signature,
                } => Some(serde_json::json!({
                    "type": "tool_call",
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "input_json": input_json,
                    "id": item_id,
                    "has_signature": signature.is_some(),
                })),
            })
            .collect::<Vec<_>>();
        (!parts.is_empty()).then_some(Value::Array(parts))
    }

    fn record_llm_usage(&mut self, llm_response: &LlmResponse, response_text: &str) {
        let usage = token_usage_from_llm_usage(&llm_response.usage);
        self.cumulative_usage.add(&usage);
        self.emit(SessionEvent::TokenUsage {
            iteration: self.iteration,
            usage: usage.clone(),
            cumulative: self.cumulative_usage.clone(),
        });
        if self.config.emit_llm_debug_log {
            let response_parts = self.llm_response_debug_parts(llm_response);
            self.pending_effects.push_back(Effect::Log {
                event: LogEvent::LlmDebug {
                    session_id: self.config.session_id.clone(),
                    iteration: self.iteration,
                    usage,
                    provider_usage: llm_response.provider_usage.clone(),
                    request_body: llm_response.request_body.clone(),
                    response_text: response_text.to_string(),
                    response_parts,
                },
            });
        }
    }

    fn record_llm_error(&mut self, error: &LlmCallError) {
        if self.config.emit_llm_debug_log {
            self.pending_effects.push_back(Effect::Log {
                event: LogEvent::LlmError {
                    session_id: self.config.session_id.clone(),
                    iteration: self.iteration,
                    request_body: error.request_body.clone(),
                    message: error.message.clone(),
                    retryable: error.retryable,
                    raw: error.raw.clone(),
                    code: error.code.clone(),
                },
            });
        }
    }

    fn emit_llm_error(&mut self, error: LlmCallError) {
        self.record_llm_error(&error);
        self.emit(make_error_event(
            "llm_provider",
            error.code.as_deref(),
            format!("LLM error: {}", error.message),
            error.raw,
        ));
        self.finish();
    }

    fn handle_tool_results(&mut self, id: EffectId, completed: Vec<CompletedToolCall>) {
        let waiting_effect_id = match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingTools { effect_id } => effect_id,
            other => {
                self.state = other;
                return;
            }
        };

        if waiting_effect_id != id {
            self.state = MachineState::WaitingTools {
                effect_id: waiting_effect_id,
            };
            return;
        }

        for outcome in &completed {
            self.emit(SessionEvent::ToolCall {
                call_id: Some(outcome.call_id.clone()),
                name: outcome.tool_name.clone(),
                args: outcome.args.clone(),
                result: outcome.state_result.result.clone(),
                success: outcome.state_result.success,
                duration_ms: outcome.duration_ms,
            });
        }

        let actions = {
            let driver = Arc::clone(&self.config.protocol_driver);
            let ctx = self.driver_context();
            driver.handle_tool_results(ctx, completed)
        };
        self.apply_actions(actions);
    }

    fn take_waiting_exec_state(&mut self, id: EffectId) -> Option<WaitingExecState> {
        match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingExec {
                effect_id,
                driver_state,
            } if effect_id == id => Some(WaitingExecState { driver_state }),
            other => {
                self.state = other;
                None
            }
        }
    }

    fn handle_exec_result(&mut self, id: EffectId, result: Result<crate::ExecResponse, String>) {
        let Some(waiting) = self.take_waiting_exec_state(id) else {
            return;
        };
        let actions = {
            let driver = Arc::clone(&self.config.protocol_driver);
            let ctx = self.driver_context();
            driver.handle_exec_result(ctx, waiting, result)
        };
        self.apply_actions(actions);
    }

    fn handle_timeout(&mut self, id: EffectId) {
        let (effect_id, retry_attempt, request, driver_state) =
            match std::mem::replace(&mut self.state, MachineState::Finished) {
                MachineState::WaitingRetry {
                    effect_id,
                    retry_attempt,
                    request,
                    driver_state,
                } => (effect_id, retry_attempt, request, driver_state),
                other => {
                    self.state = other;
                    return;
                }
            };

        if effect_id != id {
            self.state = MachineState::WaitingRetry {
                effect_id,
                retry_attempt,
                request,
                driver_state,
            };
            return;
        }

        self.start_llm_request(request, retry_attempt, driver_state);
    }
}

fn token_usage_from_llm_usage(usage: &crate::llm::types::LlmUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    }
}

#[cfg(test)]
mod tests;
