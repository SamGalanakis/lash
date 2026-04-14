//! Sans-IO state machine for session turns.
//!
//! `TurnMachine` encapsulates all protocol logic for both Standard and RLM
//! execution modes. The host event loop drives the machine by calling
//! `poll_effect()` and feeding responses back via `handle_response()`.

use std::collections::VecDeque;
use std::time::Duration;

use serde_json::Value;

use crate::llm::types::{
    LlmAttachment, LlmOutputPart, LlmRequest, LlmResponse, LlmToolChoice, LlmToolSpec, LlmUsage,
};
use crate::session_model::exec::ExecAccumulator;
use crate::session_model::message::{MessageOrigin, PartAttachment, data_url_for_bytes};
use crate::session_model::{
    LLM_MAX_RETRIES, LLM_RETRY_DELAYS, Message, MessageRole, MessageSequence, Part, PartKind,
    PruneState, SessionEvent, TokenUsage, TurnTerminationPolicyState, format_tool_result_content,
    fresh_message_id, make_error_envelope, make_error_event, reassign_part_ids,
};
use crate::{CheckpointKind, ExecutionMode, PluginMessage, ToolCallRecord, ToolResult};

// ─── Public types ───

/// Opaque identifier linking an effect to its response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectId(pub u64);

#[derive(Clone, Debug)]
pub struct PendingToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub args: Value,
}

#[derive(Clone, Debug)]
pub struct CompletedToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub args: Value,
    pub state_result: ToolResult,
    pub model_result: ToolResult,
    pub duration_ms: u64,
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
}

/// An effect the host must fulfil.
#[derive(Debug)]
pub enum Effect {
    /// Sync the live RLM execution surface before the turn proceeds.
    SyncExecutionSurface { id: EffectId },
    /// Start an LLM call.
    /// For Standard the host returns `Response::LlmComplete`.
    /// For RLM the host returns `Response::LlmComplete`.
    LlmCall { id: EffectId, request: LlmRequest },
    /// Cancel an in-progress LLM stream (RLM: code fence detected mid-stream).
    CancelLlm { id: EffectId },
    /// Execute one or more standard-mode tool calls. Host returns `Response::ToolResults`.
    ToolCalls {
        id: EffectId,
        calls: Vec<PendingToolCall>,
    },
    /// Execute a RLM code block. Host returns `Response::ExecResult`.
    ExecCode { id: EffectId, code: String },
    /// Run a host/plugin checkpoint before the machine continues or completes.
    Checkpoint {
        id: EffectId,
        checkpoint: CheckpointKind,
    },
    /// Retry backoff. Host returns `Response::Timeout`.
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
}

/// A response to a previously emitted effect.
pub enum Response {
    /// Live RLM execution surface sync completed.
    ExecutionSurfaceSynced {
        id: EffectId,
        result: Result<(), String>,
    },
    /// Full LLM response (Standard), or final response after streaming (RLM).
    LlmComplete {
        id: EffectId,
        result: Result<LlmResponse, LlmCallError>,
        /// When true, text deltas were already emitted during streaming,
        /// so the machine should skip emitting `TextDelta` events.
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
    },
    /// Sleep completed.
    Timeout { id: EffectId },
}

/// Configuration for a `TurnMachine` instance.
pub struct TurnMachineConfig {
    pub execution_mode: ExecutionMode,
    pub model: String,
    pub max_turns: Option<usize>,
    pub model_variant: Option<String>,
    pub run_session_id: Option<String>,
    pub tool_specs: Arc<Vec<LlmToolSpec>>,
    pub system_prompt: String,
    pub session_id: String,
    pub emit_llm_debug_log: bool,
    /// RLM termination contract for this session. Only meaningful when
    /// `execution_mode == Rlm`. Defaults to `ProseWithoutFence`.
    pub rlm_termination: RlmTermination,
}

/// How a RLM session terminates. Mirrors `lash::RlmTermination`;
/// duplicated here so the sans-io layer doesn't depend on the lash
/// host crate.
#[derive(Clone, Debug, Default)]
pub enum RlmTermination {
    /// Today's behavior — terminate when the model writes prose with
    /// no fenced lashlang block. The prose IS the assistant's final
    /// reply.
    #[default]
    ProseWithoutFence,
    /// Terminate when the model calls `finish <expr>` from inside a
    /// fenced lashlang block. The captured value is the terminal
    /// result. Prose-without-fence becomes a soft error that loops the
    /// model. When `schema` is `Some`, the captured value is validated
    /// against the JSON Schema before being accepted; mismatches loop
    /// with an explanation.
    Finish { schema: Option<serde_json::Value> },
}

// ─── Internal state ───

/// RLM iteration state carried across lashlang tool executions.
struct RlmState {
    acc: ExecAccumulator,
    latest_usage: LlmUsage,
}

impl RlmState {
    fn new() -> Self {
        Self {
            acc: ExecAccumulator::new(),
            latest_usage: LlmUsage::default(),
        }
    }
}

/// Accumulated RLM turn state carried across exec cycles.
struct RlmTurnState {
    state: RlmState,
}

struct WaitingLlmState {
    retry_attempt: usize,
    rlm: Option<RlmState>,
    request: LlmRequest,
}

enum CheckpointResume {
    PrepareIteration,
    Finish,
}

enum MachineState {
    PreparingMode,
    WaitingExecutionSurface {
        effect_id: EffectId,
    },
    PrepareIteration,
    WaitingLlm {
        _effect_id: EffectId,
        request: LlmRequest,
        // RLM state (None for Standard)
        rlm: Option<RlmState>,
        retry_attempt: usize,
    },
    WaitingRetry {
        effect_id: EffectId,
        retry_attempt: usize,
        last_error: String,
        request: LlmRequest,
        /// Saved RLM state for retry continuation
        rlm: Option<RlmState>,
    },
    WaitingTools {
        effect_id: EffectId,
    },
    WaitingExec {
        rlm: RlmTurnState,
    },
    ProcessReplResult {
        rlm: RlmTurnState,
    },
    WaitingCheckpoint {
        effect_id: EffectId,
        checkpoint: CheckpointKind,
        on_empty: CheckpointResume,
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
        // If we have queued effects, drain them first.
        if let Some(effect) = self.pending_effects.pop_front() {
            return Some(effect);
        }

        // Otherwise, advance the state machine.
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
        if matches!(self.config.execution_mode, ExecutionMode::Rlm) {
            let id = self.next_id();
            self.state = MachineState::WaitingExecutionSurface { effect_id: id };
            self.pending_effects
                .push_back(Effect::SyncExecutionSurface { id });
            return;
        }

        self.prepare_iteration();
    }

    fn prepare_iteration(&mut self) {
        let rendered_prompt = self.messages.render_prompt(self.config.execution_mode);

        let use_tools = !self.config.tool_specs.is_empty();
        let attachments: Vec<LlmAttachment> = rendered_prompt.attachments;
        let mut messages = rendered_prompt.messages;
        if !self.config.system_prompt.trim().is_empty() {
            messages.insert(
                0,
                crate::llm::types::LlmMessage {
                    role: crate::llm::types::LlmRole::System,
                    content: self.config.system_prompt.clone(),
                    kind: "text".to_string(),
                    image_idx: -1,
                    tool_call_id: None,
                    tool_name: None,
                },
            );
        }

        let llm_request = LlmRequest {
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
        };
        self.queue_llm_request(llm_request, 0, None);
    }

    fn queue_llm_request(
        &mut self,
        request: LlmRequest,
        retry_attempt: usize,
        rlm: Option<RlmState>,
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
        let rlm = if matches!(self.config.execution_mode, ExecutionMode::Rlm) {
            Some(rlm.unwrap_or_else(RlmState::new))
        } else {
            None
        };
        self.state = MachineState::WaitingLlm {
            _effect_id: id,
            request: request.clone(),
            rlm,
            retry_attempt,
        };
        self.pending_effects
            .push_back(Effect::LlmCall { id, request });
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
            Response::Checkpoint { id, messages } => self.handle_checkpoint(id, messages),
            Response::Timeout { id } => self.handle_timeout(id),
        }
    }

    fn request_checkpoint(&mut self, checkpoint: CheckpointKind, on_empty: CheckpointResume) {
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

    fn append_checkpoint_messages(&mut self, plugin_messages: &[PluginMessage]) {
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
                        prune_state: PruneState::Intact,
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
                        prune_state: PruneState::Intact,
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
                        transient: false,
                    }),
                }
            })
            .collect::<Vec<_>>();
        if !appended.is_empty() {
            self.messages.extend(appended);
        }
    }

    fn handle_checkpoint(&mut self, id: EffectId, messages: Vec<PluginMessage>) {
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

        if !messages.is_empty() {
            self.append_checkpoint_messages(&messages);
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
            CheckpointResume::PrepareIteration => {
                self.state = MachineState::PrepareIteration;
            }
            CheckpointResume::Finish => self.finish(),
        }
    }

    fn handle_llm_complete(
        &mut self,
        _id: EffectId,
        result: Result<LlmResponse, LlmCallError>,
        text_streamed: bool,
    ) {
        let waiting = self.take_waiting_llm_state();
        let Some(waiting) = waiting else {
            return;
        };
        match result {
            Err(error) => {
                if error.retryable && waiting.retry_attempt < LLM_MAX_RETRIES {
                    self.schedule_llm_retry(
                        waiting.retry_attempt,
                        error,
                        waiting.request,
                        waiting.rlm,
                    );
                    return;
                }
                self.emit_llm_error(error);
            }
            Ok(llm_response) => {
                let response_text = self.llm_response_text(&llm_response);
                self.record_llm_usage(
                    &llm_response,
                    response_text,
                    waiting.rlm.as_ref().map(|rlm| &rlm.latest_usage),
                );
                match self.config.execution_mode {
                    ExecutionMode::Standard => {
                        self.handle_standard_llm_success(llm_response, text_streamed)
                    }
                    ExecutionMode::Rlm => self.handle_repl_llm_success(
                        llm_response,
                        waiting.request,
                        waiting.rlm.unwrap_or_else(RlmState::new),
                        waiting.retry_attempt,
                    ),
                }
            }
        }
    }

    fn take_waiting_llm_state(&mut self) -> Option<WaitingLlmState> {
        match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingLlm {
                request,
                rlm,
                retry_attempt,
                ..
            } => Some(WaitingLlmState {
                retry_attempt,
                rlm,
                request,
            }),
            other => {
                self.state = other;
                None
            }
        }
    }

    fn schedule_llm_retry(
        &mut self,
        retry_attempt: usize,
        error: LlmCallError,
        request: LlmRequest,
        rlm: Option<RlmState>,
    ) {
        let delay = LLM_RETRY_DELAYS[retry_attempt];
        let reason = error.message.clone();
        self.emit(SessionEvent::RetryStatus {
            wait_seconds: delay.as_secs(),
            attempt: retry_attempt + 2,
            max_attempts: LLM_MAX_RETRIES + 1,
            reason: reason.clone(),
            envelope: Some(make_error_envelope(
                "llm_provider",
                error.code.as_deref(),
                format!("LLM error: {}", reason),
                error.raw,
            )),
        });
        let sleep_id = self.next_id();
        self.state = MachineState::WaitingRetry {
            effect_id: sleep_id,
            retry_attempt: retry_attempt + 1,
            last_error: reason,
            request,
            rlm,
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
                LlmOutputPart::ToolCall {
                    call_id,
                    tool_name,
                    input_json,
                } => Some(serde_json::json!({
                    "type": "tool_call",
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "input_json": input_json,
                })),
            })
            .collect::<Vec<_>>();
        (!parts.is_empty()).then_some(Value::Array(parts))
    }

    fn record_llm_usage(
        &mut self,
        llm_response: &LlmResponse,
        response_text: &str,
        fallback_usage: Option<&LlmUsage>,
    ) {
        let usage = if llm_usage_is_empty(&llm_response.usage) {
            fallback_usage
                .map(token_usage_from_llm_usage)
                .unwrap_or_default()
        } else {
            token_usage_from_llm_usage(&llm_response.usage)
        };
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

    fn emit_llm_error(&mut self, error: LlmCallError) {
        self.emit(make_error_event(
            "llm_provider",
            error.code.as_deref(),
            format!("LLM error: {}", error.message),
            error.raw,
        ));
        self.finish();
    }

    // ─── Standard path ───

    fn handle_standard_llm_success(&mut self, llm_response: LlmResponse, text_streamed: bool) {
        let response_parts = if llm_response.parts.is_empty() && !llm_response.full_text.is_empty()
        {
            vec![LlmOutputPart::Text {
                text: llm_response.full_text.clone(),
            }]
        } else {
            llm_response.parts.clone()
        };

        let mut assistant_text = String::new();
        let mut tool_calls: Vec<(String, String, String)> = Vec::new();
        for part in response_parts {
            match part {
                LlmOutputPart::Text { text } => {
                    if !text.is_empty() {
                        let previous_len = assistant_text.len();
                        append_assistant_text_part(&mut assistant_text, &text);
                        if !text_streamed {
                            self.emit(SessionEvent::TextDelta {
                                content: assistant_text[previous_len..].to_string(),
                            });
                        }
                    }
                }
                LlmOutputPart::ToolCall {
                    call_id,
                    tool_name,
                    input_json,
                } => {
                    tool_calls.push((call_id, tool_name, input_json));
                }
            }
        }
        self.emit(SessionEvent::LlmResponse {
            iteration: self.iteration,
            content: assistant_text.clone(),
            duration_ms: 0, // Host can provide timing via response
        });

        // No tool calls → prose-only → done
        if tool_calls.is_empty() {
            if assistant_text.trim().is_empty() {
                self.emit(make_error_event(
                    "llm_provider",
                    Some("empty_response"),
                    "Model returned no assistant text or tool calls.",
                    None,
                ));
                self.finish();
                return;
            }
            let mid = fresh_message_id();
            self.messages.push(Message {
                id: mid.clone(),
                role: MessageRole::Assistant,
                parts: vec![Part {
                    id: format!("{}.p0", mid),
                    kind: PartKind::Prose,
                    content: assistant_text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            });
            self.request_checkpoint(CheckpointKind::BeforeCompletion, CheckpointResume::Finish);
            return;
        }

        // Build assistant message with tool call parts
        let asst_id = fresh_message_id();
        let mut assistant_parts = Vec::new();
        if !assistant_text.trim().is_empty() {
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::Prose,
                content: assistant_text.clone(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
        }

        let mut calls = Vec::new();
        for (call_id, tool_name, input_json) in &tool_calls {
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::ToolCall,
                content: input_json.clone(),
                attachment: None,
                tool_call_id: Some(call_id.clone()),
                tool_name: Some(tool_name.clone()),
                prune_state: PruneState::Intact,
            });

            let args =
                serde_json::from_str::<Value>(input_json).unwrap_or_else(|_| serde_json::json!({}));
            calls.push(PendingToolCall {
                call_id: call_id.clone(),
                tool_name: tool_name.clone(),
                args,
            });
        }

        if !assistant_parts.is_empty() {
            self.messages.push(Message {
                id: asst_id,
                role: MessageRole::Assistant,
                parts: assistant_parts,
                user_input: None,
                origin: None,
            });
        }

        let effect_id = self.next_id();
        self.pending_effects.push_back(Effect::ToolCalls {
            id: effect_id,
            calls,
        });
        self.state = MachineState::WaitingTools { effect_id };
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

        self.process_standard_tool_results(completed);
    }

    fn process_standard_tool_results(&mut self, completed: Vec<CompletedToolCall>) {
        let mut result_parts = Vec::new();
        let mut tool_records = Vec::new();

        for outcome in completed {
            tool_records.push(ToolCallRecord {
                call_id: Some(outcome.call_id.clone()),
                tool: outcome.tool_name.clone(),
                args: outcome.args.clone(),
                result: outcome.model_result.result.clone(),
                success: outcome.model_result.success,
                duration_ms: outcome.duration_ms,
            });

            result_parts.push(Part {
                id: String::new(),
                kind: PartKind::ToolResult,
                content: format_tool_result_content(
                    outcome.model_result.success,
                    &outcome.model_result.result,
                ),
                attachment: None,
                tool_call_id: Some(outcome.call_id.clone()),
                tool_name: Some(outcome.tool_name.clone()),
                prune_state: PruneState::Intact,
            });

            for (image_offset, image) in outcome.model_result.images.into_iter().enumerate() {
                result_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[Tool image: {}]", image.label),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
                result_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(PartAttachment {
                        mime: image.mime.clone(),
                        url: data_url_for_bytes(&image.mime, &image.data),
                        filename: Some(format!("tool-image-{image_offset}")),
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
        }

        if !result_parts.is_empty() {
            let user_id = fresh_message_id();
            reassign_part_ids(&user_id, &mut result_parts);
            self.messages.push(Message {
                id: user_id,
                role: MessageRole::User,
                parts: result_parts,
                user_input: None,
                origin: None,
            });
        }

        self.iteration += 1;
        if let Some(max_turns) = self.config.max_turns
            && self.iteration >= self.run_offset + max_turns
        {
            let sys_id = fresh_message_id();
            self.messages.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Error,
                    content: format!(
                        "Turn limit reached ({max_turns}) before a final assistant response."
                    ),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            });
            self.finish();
            return;
        }

        self.request_checkpoint(
            CheckpointKind::AfterWork,
            CheckpointResume::PrepareIteration,
        );
    }

    // ─── RLM path ───

    fn handle_repl_llm_success(
        &mut self,
        llm_response: LlmResponse,
        _request: LlmRequest,
        rlm: RlmState,
        _retry_attempt: usize,
    ) {
        self.emit(SessionEvent::LlmResponse {
            iteration: self.iteration,
            content: llm_response.full_text.clone(),
            duration_ms: 0,
        });

        let mut assistant_text = String::new();
        let response_parts = if llm_response.parts.is_empty() && !llm_response.full_text.is_empty()
        {
            vec![LlmOutputPart::Text {
                text: llm_response.full_text.clone(),
            }]
        } else {
            llm_response.parts.clone()
        };

        // Native tool calls have no place in RLM mode anymore — the model
        // writes lashlang inside a fenced block in its prose. If a provider
        // somehow returns a tool call here we just drop it; the absence of
        // a fence will surface as the existing "no work step" error path.
        for part in response_parts {
            if let LlmOutputPart::Text { text } = part {
                append_assistant_text_part(&mut assistant_text, &text);
            }
        }

        if assistant_text.trim().is_empty() {
            self.emit(make_error_event(
                "llm_provider",
                Some("empty_response"),
                "Model returned no assistant text.",
                None,
            ));
            self.finish();
            return;
        }

        let extraction = extract_first_lashlang_fence(&assistant_text);
        let Some(fence) = extraction else {
            // No fenced lashlang block. What happens next depends on
            // the session's termination contract:
            // - `ProseWithoutFence` (default): the prose IS the
            //   terminal response. Persist and finish the turn.
            // - `Finish { .. }`: prose-only is invalid. The model must
            //   call `finish <expr>` from inside a fenced block. Push
            //   the prose into history and inject a system reminder
            //   before looping.
            match &self.config.rlm_termination {
                RlmTermination::ProseWithoutFence => {
                    let mid = fresh_message_id();
                    self.messages.push(Message {
                        id: mid.clone(),
                        role: MessageRole::Assistant,
                        parts: vec![Part {
                            id: format!("{}.p0", mid),
                            kind: PartKind::Prose,
                            content: assistant_text,
                            attachment: None,
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: PruneState::Intact,
                        }],
                        user_input: None,
                        origin: None,
                    });
                    self.request_checkpoint(
                        CheckpointKind::BeforeCompletion,
                        CheckpointResume::Finish,
                    );
                }
                RlmTermination::Finish { .. } => {
                    let asst_id = fresh_message_id();
                    self.messages.push(Message {
                        id: asst_id.clone(),
                        role: MessageRole::Assistant,
                        parts: vec![Part {
                            id: format!("{}.p0", asst_id),
                            kind: PartKind::Prose,
                            content: assistant_text,
                            attachment: None,
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: PruneState::Intact,
                        }],
                        user_input: None,
                        origin: None,
                    });
                    let reminder_id = fresh_message_id();
                    self.messages.push(Message {
                        id: reminder_id.clone(),
                        role: MessageRole::User,
                        parts: vec![Part {
                            id: format!("{}.p0", reminder_id),
                            kind: PartKind::Text,
                            content: "[runtime] You're in a typed RLM session. End by emitting a fenced ```lashlang block that calls `finish <expr>` with a value matching the required output schema. Prose-only replies are not accepted as the final answer here.".to_string(),
                            attachment: None,
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: PruneState::Intact,
                        }],
                        user_input: None,
                        origin: None,
                    });
                    self.iteration += 1;
                    self.request_checkpoint(
                        CheckpointKind::AfterWork,
                        CheckpointResume::PrepareIteration,
                    );
                }
            }
            return;
        };

        // Multiple fenced blocks: only the first runs. The prompt
        // already instructs the model to consolidate; if it forgets, the
        // next iteration's results implicitly tell it. Intentionally no
        // warning event — keeping the protocol minimal.
        let _ = fence.had_extra_fences;

        let asst_id = fresh_message_id();
        self.messages.push(Message {
            id: asst_id.clone(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: format!("{}.p0", asst_id),
                kind: PartKind::Prose,
                content: assistant_text,
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        });

        let exec_id = self.next_id();
        let mut rlm = rlm;
        rlm.acc.executed_code = Some(fence.code.clone());
        self.state = MachineState::WaitingExec {
            rlm: RlmTurnState { state: rlm },
        };
        self.pending_effects.push_back(Effect::ExecCode {
            id: exec_id,
            code: fence.code,
        });
    }

    fn handle_exec_result(&mut self, _id: EffectId, result: Result<crate::ExecResponse, String>) {
        let mut rlm = match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingExec { rlm, .. } => rlm,
            other => {
                self.state = other;
                return;
            }
        };

        match result {
            Ok(r) => {
                for tc in &r.tool_calls {
                    self.emit(SessionEvent::ToolCall {
                        call_id: None,
                        name: tc.tool.clone(),
                        args: tc.args.clone(),
                        result: tc.result.clone(),
                        success: tc.success,
                        duration_ms: tc.duration_ms,
                    });
                }
                rlm.state.acc.tool_calls.extend(r.tool_calls);
                rlm.state.acc.images.extend(r.images);
                if !r.output.is_empty() {
                    rlm.state.acc.combined_output.push_str(&r.output);
                }
                for observation in r.observations {
                    if !observation.is_empty() {
                        if !rlm.state.acc.combined_output.is_empty()
                            && !rlm.state.acc.combined_output.ends_with('\n')
                        {
                            rlm.state.acc.combined_output.push('\n');
                        }
                        rlm.state.acc.combined_output.push_str(&observation);
                        if !rlm.state.acc.combined_output.ends_with('\n') {
                            rlm.state.acc.combined_output.push('\n');
                        }
                    }
                }
                if let Some(raw_error) = r.error {
                    rlm.state.acc.exec_error = Some(raw_error);
                }
                if let Some(finish_value) = r.terminal_finish {
                    rlm.state.acc.terminal_finish = Some(finish_value);
                }
            }
            Err(e) => {
                rlm.state.acc.exec_error = Some(e);
            }
        }

        self.state = MachineState::ProcessReplResult { rlm };
        self.process_repl_result();
    }

    fn process_repl_result(&mut self) {
        let rlm = match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::ProcessReplResult { rlm } => rlm,
            other => {
                self.state = other;
                return;
            }
        };

        let rlm_state = rlm.state;
        let next_tool_images = rlm_state.acc.images.clone();
        let result_call_id = format!("rlm_exec_{}", self.iteration);

        // Typed-mode terminal finish: when the lashlang program ended
        // with `finish <expr>` AND the session uses
        // `RlmTermination::Finish`, validate the captured value
        // against the schema (if any) and terminate cleanly. On
        // validation failure, feed the error back so the model can
        // retry with a corrected `finish` call.
        if let Some(finish_value) = &rlm_state.acc.terminal_finish
            && let RlmTermination::Finish { schema } = &self.config.rlm_termination
        {
            if let Some(schema) = schema
                && let Err(error_text) = validate_finish_value(finish_value, schema)
            {
                let asst_id = fresh_message_id();
                self.messages.push(Message {
                    id: asst_id.clone(),
                    role: MessageRole::User,
                    parts: vec![Part {
                        id: format!("{}.p0", asst_id),
                        kind: PartKind::Text,
                        content: format!(
                            "[runtime] Your `finish` value didn't match the required output schema:\n{error_text}\n\nFix the value and call `finish <corrected>` from another fenced ```lashlang block."
                        ),
                        attachment: None,
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: PruneState::Intact,
                    }],
                    user_input: None,
                    origin: None,
                });
                self.iteration += 1;
                self.request_checkpoint(
                    CheckpointKind::AfterWork,
                    CheckpointResume::PrepareIteration,
                );
                return;
            }

            // Validation passed (or no schema). Terminate the turn
            // with the captured value as the assistant's final
            // structured response.
            let mid = fresh_message_id();
            let rendered = match finish_value {
                serde_json::Value::String(text) => text.clone(),
                other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
            };
            self.messages.push(Message {
                id: mid.clone(),
                role: MessageRole::Assistant,
                parts: vec![Part {
                    id: format!("{}.p0", mid),
                    kind: PartKind::Prose,
                    content: rendered,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            });
            self.emit(SessionEvent::TypedFinish {
                value: finish_value.clone(),
            });
            self.request_checkpoint(CheckpointKind::BeforeCompletion, CheckpointResume::Finish);
            return;
        }

        let mut result_payload = serde_json::json!({
            "observations": rlm_state.acc.combined_output,
            "tool_calls": rlm_state.acc.tool_calls,
            "error": rlm_state.acc.exec_error,
        });
        if !next_tool_images.is_empty() {
            let images = next_tool_images
                .iter()
                .map(|img| {
                    serde_json::json!({
                        "label": img.label,
                        "mime": img.mime,
                    })
                })
                .collect::<Vec<_>>();
            result_payload["images"] = serde_json::Value::Array(images);
        }
        let success = result_payload
            .get("error")
            .is_none_or(|value| value.is_null());
        // Telemetry only — surfaced as a synthetic tool_call event so the
        // host UI keeps rendering "execute_lashlang" runs in the activity
        // panel without changing the wire-format-visible message history.
        let execute_args = rlm_state
            .acc
            .executed_code
            .as_ref()
            .map(|code| serde_json::json!({"code": code}))
            .unwrap_or_else(|| serde_json::json!({}));
        self.emit(SessionEvent::ToolCall {
            call_id: Some(result_call_id),
            name: "execute_lashlang".to_string(),
            args: execute_args,
            result: result_payload.clone(),
            success,
            duration_ms: 0,
        });
        // Render the exec result as a plain user text message — there is
        // no longer a preceding native tool call for it to reference.
        let user_id = fresh_message_id();
        let result_text = format_repl_result_text(success, &result_payload);
        let mut result_parts = vec![Part {
            id: format!("{}.p0", user_id),
            kind: PartKind::Text,
            content: result_text,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }];
        for (image_offset, img) in next_tool_images.iter().enumerate() {
            result_parts.push(Part {
                id: format!("{}.p{}", user_id, result_parts.len()),
                kind: PartKind::Text,
                content: format!("[Tool image: {}]", img.label),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
            result_parts.push(Part {
                id: format!("{}.p{}", user_id, result_parts.len()),
                kind: PartKind::Image,
                content: String::new(),
                attachment: Some(PartAttachment {
                    mime: img.mime.clone(),
                    url: data_url_for_bytes(&img.mime, &img.data),
                    filename: Some(format!("tool-image-{image_offset}")),
                }),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
        }
        self.messages.push(Message {
            id: user_id,
            role: MessageRole::User,
            parts: result_parts,
            user_input: None,
            origin: None,
        });

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

        self.request_checkpoint(
            CheckpointKind::AfterWork,
            CheckpointResume::PrepareIteration,
        );
    }

    fn handle_timeout(&mut self, id: EffectId) {
        let (effect_id, retry_attempt, last_error, request, rlm) =
            match std::mem::replace(&mut self.state, MachineState::Finished) {
                MachineState::WaitingRetry {
                    effect_id,
                    retry_attempt,
                    last_error,
                    request,
                    rlm,
                    ..
                } => (effect_id, retry_attempt, last_error, request, rlm),
                other => {
                    self.state = other;
                    return;
                }
            };

        if effect_id != id {
            self.state = MachineState::WaitingRetry {
                effect_id,
                retry_attempt,
                last_error,
                request,
                rlm,
            };
            return;
        }

        self.queue_llm_request(request, retry_attempt, rlm);
    }
}

mod helpers;
use helpers::*;

#[cfg(test)]
mod tests;
use std::sync::Arc;
