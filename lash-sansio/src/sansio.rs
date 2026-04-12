//! Sans-IO state machine for session turns.
//!
//! `TurnMachine` encapsulates all protocol logic for both Standard and REPL
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
    LLM_MAX_RETRIES, LLM_RETRY_DELAYS, Message, MessageRole, Part, PartKind, PromptSectionOverride,
    PruneState, SessionEvent, TokenUsage, TurnTerminationPolicyState, format_tool_result_content,
    fresh_message_id, make_error_envelope, make_error_event, reassign_part_ids, render_prompt,
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
        request_body: Option<String>,
        response_text: String,
        response_parts: Option<Value>,
    },
}

/// An effect the host must fulfil.
#[derive(Debug)]
pub enum Effect {
    /// Sync the live REPL execution surface before the turn proceeds.
    SyncExecutionSurface { id: EffectId },
    /// Start an LLM call.
    /// For Standard the host returns `Response::LlmComplete`.
    /// For REPL the host streams via `handle_llm_delta()` then `Response::LlmComplete`.
    LlmCall { id: EffectId, request: LlmRequest },
    /// Cancel an in-progress LLM stream (REPL: code fence detected mid-stream).
    CancelLlm { id: EffectId },
    /// Execute one or more standard-mode tool calls. Host returns `Response::ToolResults`.
    ToolCalls {
        id: EffectId,
        calls: Vec<PendingToolCall>,
    },
    /// Execute a REPL code block. Host returns `Response::ExecResult`.
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
        messages: Vec<Message>,
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
    /// Live REPL execution surface sync completed.
    ExecutionSurfaceSynced {
        id: EffectId,
        result: Result<(), String>,
    },
    /// Full LLM response (Standard), or final response after streaming (REPL).
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
    /// REPL code execution result.
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
    pub tool_specs: Vec<LlmToolSpec>,
    pub prompt: crate::PromptContext,
    pub prompt_renderer: std::sync::Arc<dyn crate::PromptRenderer>,
    pub prompt_overrides: Vec<PromptSectionOverride>,
    pub session_id: String,
    pub emit_llm_debug_log: bool,
    /// REPL termination contract for this session. Only meaningful when
    /// `execution_mode == Repl`. Defaults to `ProseWithoutFence`.
    pub repl_termination: ReplTermination,
}

/// How a REPL session terminates. Mirrors `lash::ReplTermination`;
/// duplicated here so the sans-io layer doesn't depend on the lash
/// host crate.
#[derive(Clone, Debug, Default)]
pub enum ReplTermination {
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
    Finish {
        schema: Option<serde_json::Value>,
    },
}

// ─── Internal state ───

/// REPL iteration state carried across lashlang tool executions.
struct ReplState {
    acc: ExecAccumulator,
    latest_usage: LlmUsage,
}

impl ReplState {
    fn new() -> Self {
        Self {
            acc: ExecAccumulator::new(),
            latest_usage: LlmUsage::default(),
        }
    }
}

/// Accumulated REPL turn state carried across exec cycles.
struct ReplTurnState {
    state: ReplState,
}

struct WaitingLlmState {
    retry_attempt: usize,
    repl: Option<ReplState>,
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
        // REPL state (None for Standard)
        repl: Option<ReplState>,
        retry_attempt: usize,
    },
    WaitingRetry {
        effect_id: EffectId,
        retry_attempt: usize,
        last_error: String,
        request: LlmRequest,
        /// Saved REPL state for retry continuation
        repl: Option<ReplState>,
    },
    WaitingTools {
        effect_id: EffectId,
    },
    WaitingExec {
        repl: ReplTurnState,
    },
    ProcessReplResult {
        repl: ReplTurnState,
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
    messages: Vec<Message>,
    iteration: usize,
    run_offset: usize,
    cumulative_usage: TokenUsage,
    termination: TurnTerminationPolicyState,
}

impl TurnMachine {
    /// Create a new machine in `PrepareIteration` state.
    pub fn new(config: TurnMachineConfig, messages: Vec<Message>, run_offset: usize) -> Self {
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

    pub fn messages(&self) -> &[Message] {
        &self.messages
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
        if matches!(self.config.execution_mode, ExecutionMode::Repl) {
            let id = self.next_id();
            self.state = MachineState::WaitingExecutionSurface { effect_id: id };
            self.pending_effects
                .push_back(Effect::SyncExecutionSurface { id });
            return;
        }

        self.prepare_iteration();
    }

    fn prepare_iteration(&mut self) {
        let system_prompt = self
            .config
            .prompt_renderer
            .render(&self.config.prompt, &self.config.prompt_overrides);

        let rendered_prompt = render_prompt(&self.messages, self.config.execution_mode);

        let use_tools = !self.config.tool_specs.is_empty();
        let attachments: Vec<LlmAttachment> = rendered_prompt.attachments;
        let mut messages = rendered_prompt.messages;
        if !system_prompt.trim().is_empty() {
            messages.insert(
                0,
                crate::llm::types::LlmMessage {
                    role: crate::llm::types::LlmRole::System,
                    content: system_prompt,
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
                self.config.tool_specs.clone()
            } else {
                Vec::new()
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
        repl: Option<ReplState>,
    ) {
        self.emit(SessionEvent::LlmRequest {
            iteration: self.iteration,
            message_count: self.messages.len(),
            tool_list: self.config.prompt.tool_list.clone(),
        });

        let id = self.next_id();
        let repl = if matches!(self.config.execution_mode, ExecutionMode::Repl) {
            Some(repl.unwrap_or_else(ReplState::new))
        } else {
            None
        };
        self.state = MachineState::WaitingLlm {
            _effect_id: id,
            request: request.clone(),
            repl,
            retry_attempt,
        };
        self.pending_effects
            .push_back(Effect::LlmCall { id, request });
    }

    /// Feed an incremental LLM text delta. Tool-call based modes do not use text deltas for control flow.
    pub fn handle_llm_delta(&mut self, _id: EffectId, _text: &str) -> bool {
        true
    }

    /// Feed a streaming usage update.
    pub fn handle_llm_usage(&mut self, _id: EffectId, usage: &LlmUsage) {
        let usage = TokenUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_input_tokens: usage.cached_input_tokens,
            reasoning_tokens: usage.reasoning_tokens,
        };
        let mut cumulative = self.cumulative_usage.clone();
        cumulative.add(&usage);
        self.emit(SessionEvent::TokenUsage {
            iteration: self.iteration,
            usage: usage.clone(),
            cumulative,
        });
        if let MachineState::WaitingLlm { repl, .. } = &mut self.state
            && let Some(repl) = repl
        {
            repl.latest_usage = LlmUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cached_input_tokens: usage.cached_input_tokens,
                reasoning_tokens: usage.reasoning_tokens,
            };
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
            .enumerate()
            .map(|(_, message)| {
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
                    }),
                }
            })
            .collect::<Vec<_>>();
        self.messages.extend(appended);
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
                    &mut self.messages,
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
                        waiting.repl,
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
                    waiting.repl.as_ref().map(|repl| &repl.latest_usage),
                );
                match self.config.execution_mode {
                    ExecutionMode::Standard => {
                        self.handle_standard_llm_success(llm_response, text_streamed)
                    }
                    ExecutionMode::Repl => self.handle_repl_llm_success(
                        llm_response,
                        waiting.request,
                        waiting.repl.unwrap_or_else(ReplState::new),
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
                repl,
                retry_attempt,
                ..
            } => Some(WaitingLlmState {
                retry_attempt,
                repl,
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
        repl: Option<ReplState>,
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
            repl,
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

    // ─── REPL path ───

    fn handle_repl_llm_success(
        &mut self,
        llm_response: LlmResponse,
        _request: LlmRequest,
        repl: ReplState,
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

        // Native tool calls have no place in REPL mode anymore — the model
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
            match &self.config.repl_termination {
                ReplTermination::ProseWithoutFence => {
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
                ReplTermination::Finish { .. } => {
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
                            content: "[runtime] You're in a typed REPL session. End by emitting a fenced ```lashlang block that calls `finish <expr>` with a value matching the required output schema. Prose-only replies are not accepted as the final answer here.".to_string(),
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
        let repl = repl;
        self.state = MachineState::WaitingExec {
            repl: ReplTurnState { state: repl },
        };
        self.pending_effects.push_back(Effect::ExecCode {
            id: exec_id,
            code: fence.code,
        });
    }

    fn handle_exec_result(&mut self, _id: EffectId, result: Result<crate::ExecResponse, String>) {
        let mut repl = match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingExec { repl, .. } => repl,
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
                repl.state.acc.tool_calls.extend(r.tool_calls);
                repl.state.acc.images.extend(r.images);
                if !r.output.is_empty() {
                    repl.state.acc.combined_output.push_str(&r.output);
                }
                for observation in r.observations {
                    if !observation.is_empty() {
                        if !repl.state.acc.combined_output.is_empty()
                            && !repl.state.acc.combined_output.ends_with('\n')
                        {
                            repl.state.acc.combined_output.push('\n');
                        }
                        repl.state.acc.combined_output.push_str(&observation);
                        if !repl.state.acc.combined_output.ends_with('\n') {
                            repl.state.acc.combined_output.push('\n');
                        }
                    }
                }
                if let Some(raw_error) = r.error {
                    repl.state.acc.exec_error = Some(raw_error);
                }
                if let Some(finish_value) = r.terminal_finish {
                    repl.state.acc.terminal_finish = Some(finish_value);
                }
            }
            Err(e) => {
                repl.state.acc.exec_error = Some(e);
            }
        }

        self.state = MachineState::ProcessReplResult { repl };
        self.process_repl_result();
    }

    fn process_repl_result(&mut self) {
        let repl = match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::ProcessReplResult { repl } => repl,
            other => {
                self.state = other;
                return;
            }
        };

        let repl_state = repl.state;
        let next_tool_images = repl_state.acc.images.clone();
        let result_call_id = format!("repl_exec_{}", self.iteration);

        // Typed-mode terminal finish: when the lashlang program ended
        // with `finish <expr>` AND the session uses
        // `ReplTermination::Finish`, validate the captured value
        // against the schema (if any) and terminate cleanly. On
        // validation failure, feed the error back so the model can
        // retry with a corrected `finish` call.
        if let Some(finish_value) = &repl_state.acc.terminal_finish {
            if let ReplTermination::Finish { schema } = &self.config.repl_termination {
                if let Some(schema) = schema {
                    if let Err(error_text) = validate_finish_value(finish_value, schema) {
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
                }

                // Validation passed (or no schema). Terminate the turn
                // with the captured value as the assistant's final
                // structured response.
                let mid = fresh_message_id();
                let rendered = match finish_value {
                    serde_json::Value::String(text) => text.clone(),
                    other => serde_json::to_string_pretty(other)
                        .unwrap_or_else(|_| other.to_string()),
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
                self.request_checkpoint(
                    CheckpointKind::BeforeCompletion,
                    CheckpointResume::Finish,
                );
                return;
            }
        }

        let mut result_payload = serde_json::json!({
            "observations": repl_state.acc.combined_output,
            "tool_calls": repl_state.acc.tool_calls,
            "error": repl_state.acc.exec_error,
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
        self.emit(SessionEvent::ToolCall {
            call_id: Some(result_call_id),
            name: "execute_lashlang".to_string(),
            args: serde_json::json!({}),
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
            &mut self.messages,
        );

        self.request_checkpoint(
            CheckpointKind::AfterWork,
            CheckpointResume::PrepareIteration,
        );
    }

    fn handle_timeout(&mut self, id: EffectId) {
        let (effect_id, retry_attempt, last_error, request, repl) =
            match std::mem::replace(&mut self.state, MachineState::Finished) {
                MachineState::WaitingRetry {
                    effect_id,
                    retry_attempt,
                    last_error,
                    request,
                    repl,
                    ..
                } => (effect_id, retry_attempt, last_error, request, repl),
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
                repl,
            };
            return;
        }

        self.queue_llm_request(request, retry_attempt, repl);
    }
}

/// Outcome of pulling the first ` ```lashlang ` (or ` ```repl `) fenced
/// block out of an assistant prose response.
struct FenceExtraction {
    code: String,
    /// True when the prose contained more than one matching fenced block.
    /// The dispatch loop only runs the first; this signals "tell the
    /// model to consolidate next time."
    had_extra_fences: bool,
}

/// Find the first ` ```lashlang ` or ` ```repl ` fenced block in `text`
/// and return its code body. Both language tags are accepted as aliases.
/// `None` ⇒ no recognized fence ⇒ caller treats the prose as a "finish
/// in prose" terminal response.
fn extract_first_lashlang_fence(text: &str) -> Option<FenceExtraction> {
    fn find_fence(text: &str) -> Option<(usize, usize, usize)> {
        // Returns (open_byte, body_start, body_end_open_fence_byte). The
        // open fence is exactly three backticks followed by a recognized
        // language tag and a newline. The closing fence is three
        // backticks at the start of a line. Trailing characters on the
        // closing line are tolerated.
        let mut search_from = 0usize;
        while let Some(rel) = text[search_from..].find("```") {
            let open = search_from + rel;
            // Require either start-of-string or a newline immediately
            // before the opening fence so we don't match inline triple
            // backticks inside prose.
            let preceded_by_newline =
                open == 0 || text.as_bytes().get(open - 1).copied() == Some(b'\n');
            if !preceded_by_newline {
                search_from = open + 3;
                continue;
            }
            let after_open = open + 3;
            let rest = &text[after_open..];
            let lang_end = rest.find('\n').unwrap_or(rest.len());
            let lang = rest[..lang_end].trim();
            if !matches!(lang, "lashlang" | "repl") {
                search_from = after_open;
                continue;
            }
            let body_start = after_open + lang_end + 1;
            if body_start > text.len() {
                return None;
            }
            // Find a closing fence at the start of a line.
            let mut cursor = body_start;
            loop {
                let Some(rel) = text[cursor..].find("```") else {
                    // No closing fence ⇒ treat the rest of the buffer as
                    // the code body. Forgiving for partial responses.
                    return Some((open, body_start, text.len()));
                };
                let close = cursor + rel;
                let preceded_by_newline =
                    close == 0 || text.as_bytes().get(close - 1).copied() == Some(b'\n');
                if preceded_by_newline {
                    return Some((open, body_start, close));
                }
                cursor = close + 3;
            }
        }
        None
    }

    let (_open, body_start, body_end) = find_fence(text)?;
    let code = text[body_start..body_end]
        .trim_end_matches('\n')
        .to_string();

    // Look for any *additional* fenced block past the closing fence so
    // we can tell the model only the first ran.
    let after_close = (body_end + 3).min(text.len());
    let had_extra_fences = find_fence(&text[after_close..]).is_some();

    Some(FenceExtraction {
        code,
        had_extra_fences,
    })
}

/// Validate a `finish` value against the JSON Schema embedded in the
/// session's `ReplTermination::Finish { schema }`. Returns `Ok(())` on
/// success or a human-readable error string on mismatch. Supports the
/// subset of JSON Schema that `predict` generates: `type` (string,
/// number, integer, boolean, array, object, null), nested object
/// `properties` with `required`, and `items` for arrays. Unsupported
/// keywords are ignored (permissive).
fn validate_finish_value(value: &Value, schema: &Value) -> Result<(), String> {
    fn matches_type(value: &Value, expected: &str) -> bool {
        match expected {
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value
                .as_f64()
                .is_some_and(|n| n.is_finite() && n.fract() == 0.0),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            "null" => value.is_null(),
            _ => true, // unknown ⇒ permissive
        }
    }

    fn type_name(value: &Value) -> &'static str {
        match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    fn check(value: &Value, schema: &Value, path: &str) -> Result<(), String> {
        let Some(schema_obj) = schema.as_object() else {
            return Ok(());
        };

        if let Some(ty) = schema_obj.get("type").and_then(Value::as_str)
            && !matches_type(value, ty)
        {
            return Err(format!(
                "{path}: expected {ty}, got {}",
                type_name(value)
            ));
        }

        if let Some(properties) = schema_obj.get("properties").and_then(Value::as_object)
            && let Some(obj) = value.as_object()
        {
            // Required fields
            if let Some(required) = schema_obj.get("required").and_then(Value::as_array) {
                for required_field in required {
                    if let Some(name) = required_field.as_str()
                        && !obj.contains_key(name)
                    {
                        return Err(format!(
                            "{path}: missing required field `{name}`"
                        ));
                    }
                }
            }
            for (name, sub_schema) in properties {
                if let Some(sub_value) = obj.get(name) {
                    let sub_path = if path.is_empty() {
                        name.clone()
                    } else {
                        format!("{path}.{name}")
                    };
                    check(sub_value, sub_schema, &sub_path)?;
                }
            }
        }

        if let Some(items_schema) = schema_obj.get("items")
            && let Some(arr) = value.as_array()
        {
            for (idx, item) in arr.iter().enumerate() {
                let sub_path = format!("{path}[{idx}]");
                check(item, items_schema, &sub_path)?;
            }
        }

        Ok(())
    }

    check(value, schema, "")
}

/// Render the lashlang exec result as a plain user-message text body.
/// Mirrors the JSON shape `format_tool_result_content` produced for the
/// legacy `execute_lashlang` tool result, but without the
/// tool-result-message envelope.
fn format_repl_result_text(success: bool, result_payload: &Value) -> String {
    let mut sections: Vec<String> = Vec::new();
    sections.push("[Lashlang execution result]".to_string());
    if !success {
        sections.push("status: error".to_string());
    }
    if let Some(observations) = result_payload
        .get("observations")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        sections.push(format!("observations:\n{observations}"));
    }
    if let Some(tool_calls) = result_payload
        .get("tool_calls")
        .filter(|value| !value.is_null())
    {
        if let Some(arr) = tool_calls.as_array() {
            if !arr.is_empty() {
                sections.push(format!(
                    "tool_calls: {}",
                    serde_json::to_string(arr).unwrap_or_else(|_| "[]".into())
                ));
            }
        }
    }
    if let Some(error) = result_payload
        .get("error")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        sections.push(format!("error:\n{error}"));
    }
    if let Some(images) = result_payload
        .get("images")
        .and_then(Value::as_array)
        .filter(|arr| !arr.is_empty())
    {
        sections.push(format!(
            "images: {}",
            serde_json::to_string(images).unwrap_or_else(|_| "[]".into())
        ));
    }
    sections.join("\n\n")
}

fn token_usage_from_llm_usage(usage: &LlmUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    }
}

fn llm_usage_is_empty(usage: &LlmUsage) -> bool {
    usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_input_tokens == 0
        && usage.reasoning_tokens == 0
}

fn append_assistant_text_part(out: &mut String, next: &str) {
    if out.is_empty() {
        out.push_str(next);
        return;
    }

    let prev_trailing_newlines = out.chars().rev().take_while(|ch| *ch == '\n').count();
    let next_leading_newlines = next.chars().take_while(|ch| *ch == '\n').count();
    let total_boundary_newlines = prev_trailing_newlines + next_leading_newlines;
    if total_boundary_newlines < 2 {
        out.push_str(&"\n".repeat(2 - total_boundary_newlines));
    }

    out.push_str(next);
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_model::{Message, MessageRole, Part, PartKind, PruneState};

    fn test_config(mode: ExecutionMode) -> TurnMachineConfig {
        TurnMachineConfig {
            execution_mode: mode,
            model: "test-model".to_string(),
            max_turns: None,
            model_variant: None,
            run_session_id: None,
            tool_specs: Vec::new(),
            prompt: crate::PromptContext {
                mode,
                ..crate::PromptContext::default()
            },
            prompt_renderer: crate::default_prompt_renderer(),
            prompt_overrides: Vec::new(),
            session_id: "test".to_string(),
            emit_llm_debug_log: false,
            repl_termination: ReplTermination::default(),
        }
    }

    fn user_message(content: &str) -> Message {
        Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }
    }

    /// Collect effects until a specific variant is found or exhausted.
    fn drain_effects(machine: &mut TurnMachine) -> Vec<Effect> {
        let mut effects = Vec::new();
        while let Some(effect) = machine.poll_effect() {
            if let Effect::SyncExecutionSurface { id } = effect {
                effects.push(effect);
                machine.handle_response(Response::ExecutionSurfaceSynced { id, result: Ok(()) });
                continue;
            }
            effects.push(effect);
        }
        effects
    }

    fn find_llm_call(effects: &[Effect]) -> Option<&EffectId> {
        effects.iter().find_map(|e| match e {
            Effect::LlmCall { id, .. } => Some(id),
            _ => None,
        })
    }

    fn find_llm_request(effects: &[Effect]) -> Option<&LlmRequest> {
        effects.iter().find_map(|e| match e {
            Effect::LlmCall { request, .. } => Some(request),
            _ => None,
        })
    }

    fn find_done(effects: &[Effect]) -> Option<(&Vec<Message>, usize)> {
        effects.iter().find_map(|e| match e {
            Effect::Done {
                messages,
                iteration,
            } => Some((messages, *iteration)),
            _ => None,
        })
    }

    fn find_retry_status(effects: &[Effect]) -> Option<(u64, usize, usize, String)> {
        effects.iter().find_map(|e| match e {
            Effect::Emit(SessionEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
                envelope: _,
            }) => Some((*wait_seconds, *attempt, *max_attempts, reason.clone())),
            _ => None,
        })
    }

    fn find_retry_envelope(effects: &[Effect]) -> Option<crate::session_model::ErrorEnvelope> {
        effects.iter().find_map(|e| match e {
            Effect::Emit(SessionEvent::RetryStatus {
                envelope: Some(envelope),
                ..
            }) => Some(envelope.clone()),
            _ => None,
        })
    }

    fn has_error_event(effects: &[Effect], needle: &str) -> bool {
        effects.iter().any(|e| match e {
            Effect::Emit(SessionEvent::Error { message, .. }) => message.contains(needle),
            _ => false,
        })
    }

    fn find_checkpoint(effects: &[Effect]) -> Option<(EffectId, CheckpointKind)> {
        effects.iter().find_map(|e| match e {
            Effect::Checkpoint { id, checkpoint } => Some((*id, *checkpoint)),
            _ => None,
        })
    }

    fn find_llm_debug(effects: &[Effect]) -> Option<(TokenUsage, String, Option<Value>)> {
        effects.iter().find_map(|e| match e {
            Effect::Log {
                event:
                    LogEvent::LlmDebug {
                        usage,
                        response_text,
                        response_parts,
                        ..
                    },
            } => Some((usage.clone(), response_text.clone(), response_parts.clone())),
            _ => None,
        })
    }

    // ─── Standard tests ───

    #[test]
    fn standard_prose_only_response_emits_done() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

        // Respond with prose-only
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "Hello there!".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Hello there!".to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        assert!(find_done(&effects).is_some());
        assert!(machine.is_done());
    }

    #[test]
    fn standard_multiple_text_parts_preserve_block_boundaries() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                parts: vec![
                    LlmOutputPart::Text {
                        text: "What’s working:".to_string(),
                    },
                    LlmOutputPart::Text {
                        text: "- one\n- two".to_string(),
                    },
                ],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        let (messages, _) = find_done(&effects).expect("done");
        let assistant = messages
            .iter()
            .find(|message| message.role == MessageRole::Assistant)
            .expect("assistant message");
        let prose = assistant
            .parts
            .iter()
            .find(|part| matches!(part.kind, PartKind::Prose))
            .map(|part| part.content.as_str())
            .expect("prose part");

        assert_eq!(prose, "What’s working:\n\n- one\n- two");
    }

    #[test]
    fn standard_multiple_text_parts_do_not_accumulate_extra_blank_lines() {
        let mut combined = "Heading\n".to_string();
        append_assistant_text_part(&mut combined, "\nBody");
        assert_eq!(combined, "Heading\n\nBody");
    }

    #[test]
    fn llm_request_includes_image_prompt_parts_for_attached_images() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![
                Part {
                    id: "m0.p0".to_string(),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(PartAttachment {
                        mime: "image/png".to_string(),
                        url: data_url_for_bytes("image/png", &[1, 2, 3]),
                        filename: None,
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                },
                Part {
                    id: "m0.p1".to_string(),
                    kind: PartKind::Text,
                    content: "explain this".to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                },
            ],
            user_input: None,
            origin: None,
        }];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let request = effects
            .into_iter()
            .find_map(|effect| match effect {
                Effect::LlmCall { request, .. } => Some(request),
                _ => None,
            })
            .expect("llm call");

        assert_eq!(request.attachments.len(), 1);
        // Images and text now always flow through ordered request messages.
        assert!(
            request
                .messages
                .iter()
                .any(|msg| msg.kind == "image" && msg.image_idx == 0)
        );
        assert!(
            request
                .messages
                .iter()
                .any(|msg| msg.kind == "text" && msg.content.contains("explain this"))
        );
    }

    #[test]
    fn standard_tool_calls_produce_effects_and_loop() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("read file")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

        // Respond with tool call
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                parts: vec![
                    LlmOutputPart::Text {
                        text: "Let me read that.".to_string(),
                    },
                    LlmOutputPart::ToolCall {
                        call_id: "tc1".to_string(),
                        tool_name: "read_file".to_string(),
                        input_json: r#"{"path":"foo.txt"}"#.to_string(),
                    },
                ],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        // Should have ToolCalls effect
        let tool_effect = effects.iter().find_map(|e| match e {
            Effect::ToolCalls { id, calls } => calls.first().map(|call| {
                (
                    *id,
                    call.call_id.clone(),
                    call.tool_name.clone(),
                    call.args.clone(),
                )
            }),
            _ => None,
        });
        assert!(tool_effect.is_some());
        let (tool_id, call_id, tool_name, args) = tool_effect.unwrap();
        assert_eq!(args, serde_json::json!({"path":"foo.txt"}));

        // Feed tool result
        machine.handle_response(Response::ToolResults {
            id: tool_id,
            results: vec![CompletedToolCall {
                call_id,
                tool_name,
                args,
                state_result: crate::ToolResult::ok(serde_json::json!("file contents")),
                model_result: crate::ToolResult::ok(serde_json::json!("file contents")),
                duration_ms: 10,
            }],
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::AfterWork);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        let llm_id2 = find_llm_call(&effects);
        assert!(
            llm_id2.is_some(),
            "should emit another LlmCall after tool results"
        );

        // Respond with prose to end
        machine.handle_response(Response::LlmComplete {
            id: *llm_id2.unwrap(),
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "Done.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Done.".to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        assert!(find_done(&effects).is_some());
    }

    #[test]
    fn standard_tool_results_preserve_original_args() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("what time is it")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                parts: vec![LlmOutputPart::ToolCall {
                    call_id: "tc1".to_string(),
                    tool_name: "exec_command".to_string(),
                    input_json: r#"{"cmd":"date","workdir":"/tmp"}"#.to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (tool_id, call_id, tool_name, args) = effects
            .iter()
            .find_map(|e| match e {
                Effect::ToolCalls { id, calls } => calls.first().map(|call| {
                    (
                        *id,
                        call.call_id.clone(),
                        call.tool_name.clone(),
                        call.args.clone(),
                    )
                }),
                _ => None,
            })
            .expect("should emit ToolCalls effect");
        assert_eq!(args, serde_json::json!({"cmd":"date","workdir":"/tmp"}));

        machine.handle_response(Response::ToolResults {
            id: tool_id,
            results: vec![CompletedToolCall {
                call_id,
                tool_name,
                args: args.clone(),
                state_result: crate::ToolResult::ok(serde_json::json!({
                    "output": "ok",
                    "exit_code": 0,
                    "timed_out": false,
                    "duration_ms": 1
                })),
                model_result: crate::ToolResult::ok(serde_json::json!({
                    "output": "ok",
                    "exit_code": 0,
                    "timed_out": false,
                    "duration_ms": 1
                })),
                duration_ms: 1,
            }],
        });

        let effects = drain_effects(&mut machine);
        let tool_event = effects
            .iter()
            .find_map(|e| match e {
                Effect::Emit(SessionEvent::ToolCall { args, .. }) => Some(args.clone()),
                _ => None,
            })
            .expect("should emit ToolCall event");
        assert_eq!(
            tool_event,
            serde_json::json!({"cmd":"date","workdir":"/tmp"})
        );
    }

    #[test]
    fn standard_retryable_error_sleeps_then_retries() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        // Retryable error
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Err(LlmCallError {
                message: "rate limited".to_string(),
                retryable: true,
                raw: None,
                code: None,
            }),
        });

        let effects = drain_effects(&mut machine);
        let retry = find_retry_status(&effects).expect("retry status");
        assert_eq!(retry.1, 2);
        assert_eq!(retry.2, LLM_MAX_RETRIES + 1);
        let envelope = find_retry_envelope(&effects).expect("retry envelope");
        assert_eq!(envelope.kind, "llm_provider");
        assert_eq!(envelope.user_message, "LLM error: rate limited");
        let sleep_effect = effects.iter().find_map(|e| match e {
            Effect::Sleep { id, .. } => Some(*id),
            _ => None,
        });
        assert!(sleep_effect.is_some(), "should emit Sleep for retry");

        // Feed timeout
        machine.handle_response(Response::Timeout {
            id: sleep_effect.unwrap(),
        });

        // Should get new LlmCall
        let effects = drain_effects(&mut machine);
        assert!(find_retry_status(&effects).is_none());
        assert!(find_llm_call(&effects).is_some());
    }

    #[test]
    fn standard_retryable_error_exhaustion_emits_error_and_done() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let mut effects = drain_effects(&mut machine);
        let mut llm_id = *find_llm_call(&effects).expect("initial llm call");

        for expected_attempt in 2..=(LLM_MAX_RETRIES + 1) {
            machine.handle_response(Response::LlmComplete {
                id: llm_id,
                text_streamed: false,
                result: Err(LlmCallError {
                    message: "provider unavailable".to_string(),
                    retryable: true,
                    raw: None,
                    code: Some("http_500".to_string()),
                }),
            });

            effects = drain_effects(&mut machine);

            let retry = find_retry_status(&effects).expect("retry status");
            assert_eq!(retry.1, expected_attempt);
            let envelope = find_retry_envelope(&effects).expect("retry envelope");
            assert_eq!(envelope.code.as_deref(), Some("http_500"));
            let sleep_id = effects
                .iter()
                .find_map(|e| match e {
                    Effect::Sleep { id, .. } => Some(*id),
                    _ => None,
                })
                .expect("sleep effect");
            machine.handle_response(Response::Timeout { id: sleep_id });
            effects = drain_effects(&mut machine);
            llm_id = *find_llm_call(&effects).expect("retried llm call");
        }

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Err(LlmCallError {
                message: "provider unavailable".to_string(),
                retryable: true,
                raw: None,
                code: Some("http_500".to_string()),
            }),
        });

        effects = drain_effects(&mut machine);
        assert!(has_error_event(&effects, "LLM error: provider unavailable"));
        assert!(find_done(&effects).is_some());
        assert!(machine.is_done());
    }

    #[test]
    fn standard_fatal_error_emits_done() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Err(LlmCallError {
                message: "auth failed".to_string(),
                retryable: false,
                raw: None,
                code: None,
            }),
        });

        let effects = drain_effects(&mut machine);
        assert!(find_done(&effects).is_some());
        assert!(machine.is_done());
    }

    #[test]
    fn standard_empty_response_emits_error() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse::default()),
        });

        let effects = drain_effects(&mut machine);
        let has_error = effects
            .iter()
            .any(|e| matches!(e, Effect::Emit(SessionEvent::Error { .. })));
        assert!(has_error);
        assert!(find_done(&effects).is_some());
    }

    #[test]
    fn standard_max_turns_stops_iteration() {
        let mut config = test_config(ExecutionMode::Standard);
        config.max_turns = Some(1);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        // Respond with tool call to trigger iteration
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                parts: vec![LlmOutputPart::ToolCall {
                    call_id: "tc1".to_string(),
                    tool_name: "test".to_string(),
                    input_json: "{}".to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let tool_id = effects
            .iter()
            .find_map(|e| match e {
                Effect::ToolCalls { id, .. } => Some(*id),
                _ => None,
            })
            .unwrap();

        machine.handle_response(Response::ToolResults {
            id: tool_id,
            results: vec![CompletedToolCall {
                call_id: "tc1".to_string(),
                tool_name: "test".to_string(),
                args: serde_json::json!({}),
                state_result: crate::ToolResult::ok(serde_json::json!("ok")),
                model_result: crate::ToolResult::ok(serde_json::json!("ok")),
                duration_ms: 1,
            }],
        });

        let effects = drain_effects(&mut machine);
        assert!(find_done(&effects).is_some());
        assert!(machine.is_done());
    }

    // ─── REPL tests ───

    #[test]
    fn repl_prose_only_response_emits_done() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "Hello there!".to_string(),
                deltas: vec!["Hello there!".to_string()],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        assert!(find_done(&effects).is_some());
        assert!(machine.is_done());
    }

    #[test]
    fn standard_checkpoint_messages_continue_turn_before_completion() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "Hello there!".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Hello there!".to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);

        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: vec![PluginMessage::text(MessageRole::User, "one more thing")],
        });

        let effects = drain_effects(&mut machine);
        assert!(find_llm_call(&effects).is_some());
        assert!(machine.messages().iter().any(|message| {
            message.role == MessageRole::User
                && message
                    .parts
                    .iter()
                    .any(|part| part.content == "one more thing")
        }));
    }

    #[test]
    fn repl_fenced_lashlang_block_runs_exec_and_continues() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("run some code")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("llm call");
        let request = find_llm_request(&effects).expect("request");
        assert!(
            request.tools.is_empty(),
            "REPL mode no longer advertises native tool specs",
        );

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "Quick check.\n\n```lashlang\nobserve \"hi\"\n```\n".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Quick check.\n\n```lashlang\nobserve \"hi\"\n```\n".to_string(),
                }],
                usage: LlmUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                },
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let exec_effect = effects.iter().find_map(|e| match e {
            Effect::ExecCode { id, code } => Some((*id, code.clone())),
            _ => None,
        });
        assert_eq!(
            exec_effect.as_ref().map(|(_, code)| code.as_str()),
            Some("observe \"hi\"")
        );

        machine.handle_response(Response::ExecResult {
            id: exec_effect.expect("exec").0,
            result: Ok(crate::ExecResponse {
                output: "hi\n".to_string(),
                observations: Vec::new(),
                tool_calls: Vec::new(),
                images: Vec::new(),
                error: None,
                duration_ms: 1,
                terminal_finish: None,
            }),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::AfterWork);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        assert!(find_llm_call(&effects).is_some());
    }

    #[test]
    fn repl_prose_after_exec_finishes_turn() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("run code then summarize")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "Let me check.\n\n```lashlang\nobserve \"hi\"\n```\n".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Let me check.\n\n```lashlang\nobserve \"hi\"\n```\n".to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let exec_id = effects
            .iter()
            .find_map(|effect| match effect {
                Effect::ExecCode { id, .. } => Some(*id),
                _ => None,
            })
            .expect("exec effect");

        machine.handle_response(Response::ExecResult {
            id: exec_id,
            result: Ok(crate::ExecResponse {
                output: "hi\n".to_string(),
                observations: Vec::new(),
                tool_calls: Vec::new(),
                images: Vec::new(),
                error: None,
                duration_ms: 5,
                terminal_finish: None,
            }),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::AfterWork);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        let next_llm_id = *find_llm_call(&effects).expect("next llm call");
        machine.handle_response(Response::LlmComplete {
            id: next_llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "All done.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "All done.".to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("completion checkpoint");
        assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        assert!(find_done(&effects).is_some());
    }

    #[test]
    fn repl_takes_first_fenced_block_when_multiple_present() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("run some code")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        let response_text = "Step 1.\n\n```lashlang\nobserve \"first\"\n```\n\nStep 2.\n\n```lashlang\nobserve \"second\"\n```\n";
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: response_text.to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: response_text.to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let exec = effects
            .iter()
            .find_map(|e| match e {
                Effect::ExecCode { code, .. } => Some(code.clone()),
                _ => None,
            })
            .expect("exec effect");
        assert_eq!(exec, "observe \"first\"");
    }

    #[test]
    fn repl_debug_log_preserves_text_only_responses() {
        let mut config = test_config(ExecutionMode::Repl);
        config.emit_llm_debug_log = true;
        let msgs = vec![user_message("run code")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("llm call");
        let response_text = "Working on it.\n\n```lashlang\nobserve \"hi\"\n```\n";
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: response_text.to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: response_text.to_string(),
                }],
                usage: LlmUsage {
                    input_tokens: 321,
                    output_tokens: 123,
                    cached_input_tokens: 45,
                    reasoning_tokens: 67,
                },
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (debug_usage, debug_text, _debug_parts) = find_llm_debug(&effects).expect("llm debug");
        assert_eq!(debug_usage.input_tokens, 321);
        assert_eq!(debug_usage.output_tokens, 123);
        assert_eq!(debug_usage.cached_input_tokens, 45);
        assert_eq!(debug_usage.reasoning_tokens, 67);
        assert_eq!(debug_text, response_text);
    }

    #[test]
    fn llm_debug_log_preserves_tool_call_only_responses() {
        let mut config = test_config(ExecutionMode::Standard);
        config.emit_llm_debug_log = true;
        let msgs = vec![user_message("call a tool")];
        let mut machine = TurnMachine::new(config, msgs, 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("llm call");

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                parts: vec![LlmOutputPart::ToolCall {
                    call_id: "tc1".to_string(),
                    tool_name: "read_file".to_string(),
                    input_json: r#"{"path":"foo.txt"}"#.to_string(),
                }],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let (usage, response_text, response_parts) = find_llm_debug(&effects).expect("llm debug");
        assert_eq!(usage.total(), 0);
        assert!(response_text.is_empty());
        assert_eq!(
            response_parts,
            Some(Value::Array(vec![serde_json::json!({
                "type": "tool_call",
                "call_id": "tc1",
                "tool_name": "read_file",
                "input_json": r#"{"path":"foo.txt"}"#,
            })]))
        );
    }

    #[test]
    fn fence_extractor_handles_lashlang_and_repl_aliases() {
        let lashlang = "Plan.\n\n```lashlang\nfoo = 1\n```\n";
        let extracted = super::extract_first_lashlang_fence(lashlang).expect("fence");
        assert_eq!(extracted.code, "foo = 1");
        assert!(!extracted.had_extra_fences);

        let repl_alias = "```repl\nbar = 2\n```";
        let extracted = super::extract_first_lashlang_fence(repl_alias).expect("fence");
        assert_eq!(extracted.code, "bar = 2");
    }

    #[test]
    fn fence_extractor_returns_none_for_pure_prose() {
        assert!(super::extract_first_lashlang_fence("All done!").is_none());
        // Other languages don't count.
        assert!(super::extract_first_lashlang_fence("```python\nprint('x')\n```").is_none(),);
        // Inline triple-backticks in prose don't fool it either.
        assert!(super::extract_first_lashlang_fence("see ```lashlang inline``` here").is_none(),);
    }

    #[test]
    fn fence_extractor_flags_extra_blocks() {
        let two_blocks = "first\n\n```lashlang\na = 1\n```\n\nbridge\n\n```lashlang\nb = 2\n```\n";
        let extracted = super::extract_first_lashlang_fence(two_blocks).expect("fence");
        assert_eq!(extracted.code, "a = 1");
        assert!(extracted.had_extra_fences);
    }

    #[test]
    fn finish_value_validator_accepts_matching_record() {
        let value = serde_json::json!({"answer": "yes", "confidence": 0.92});
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer", "confidence"]
        });
        assert!(super::validate_finish_value(&value, &schema).is_ok());
    }

    #[test]
    fn finish_value_validator_flags_missing_required_field() {
        let value = serde_json::json!({"answer": "yes"});
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer", "confidence"]
        });
        let err = super::validate_finish_value(&value, &schema).expect_err("missing field");
        assert!(err.contains("missing required field `confidence`"), "{err}");
    }

    #[test]
    fn finish_value_validator_flags_type_mismatch() {
        let value = serde_json::json!({"answer": "yes", "confidence": "high"});
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer", "confidence"]
        });
        let err = super::validate_finish_value(&value, &schema).expect_err("type mismatch");
        assert!(err.contains("confidence"), "{err}");
        assert!(err.contains("number"), "{err}");
    }

    #[test]
    fn finish_value_validator_recurses_into_array_items() {
        let value = serde_json::json!({"items": ["a", 2, "c"]});
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["items"]
        });
        let err = super::validate_finish_value(&value, &schema).expect_err("inner type mismatch");
        assert!(err.contains("items[1]"), "{err}");
    }

    #[test]
    fn fence_extractor_takes_unterminated_remainder_when_no_close() {
        let unterminated = "loading...\n\n```lashlang\nx = 1\nobserve x\n";
        let extracted = super::extract_first_lashlang_fence(unterminated).expect("fence");
        assert_eq!(extracted.code, "x = 1\nobserve x");
        assert!(!extracted.had_extra_fences);
    }
}
