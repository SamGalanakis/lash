//! Sans-IO state machine for agent turns.
//!
//! `TurnMachine` encapsulates all protocol logic for both Standard and REPL
//! execution modes. The host event loop drives the machine by calling
//! `poll_effect()` and feeding responses back via `handle_response()`.

use std::collections::VecDeque;
use std::time::Duration;

use serde_json::Value;

use crate::agent::exec::ExecAccumulator;
use crate::agent::message::IMAGE_REF_PREFIX;
use crate::agent::message::MessageOrigin;
use crate::agent::{
    AgentEvent, LLM_MAX_RETRIES, LLM_RETRY_DELAYS, Message, MessageRole, Part, PartKind,
    PromptSectionOverride, PruneState, TokenUsage, TurnTerminationPolicyState,
    build_assistant_parts, format_tool_result_content, is_malformed_assistant_output,
    make_error_envelope, make_error_event, parse_fence_line, render_prompt, truncate_raw_error,
};
use crate::llm::types::{
    LlmAttachment, LlmOutputPart, LlmPromptPart, LlmRequest, LlmResponse, LlmToolChoice,
    LlmToolSpec, LlmUsage,
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
    pub raw_result: ToolResult,
    pub model_result: ToolResult,
    pub duration_ms: u64,
}

#[derive(Clone, Debug)]
pub enum LogEvent {
    LlmDebug {
        agent_id: String,
        iteration: usize,
        usage: TokenUsage,
        request_body: Option<String>,
        response_text: String,
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
    Emit(AgentEvent),
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
    pub context_strategy: crate::ContextStrategy,
    pub model: String,
    pub max_turns: Option<usize>,
    pub model_variant: Option<String>,
    pub session_id: Option<String>,
    pub tool_specs: Vec<LlmToolSpec>,
    pub prompt: crate::PromptContext,
    pub prompt_renderer: std::sync::Arc<dyn crate::PromptRenderer>,
    pub prompt_overrides: Vec<PromptSectionOverride>,
    pub agent_id: String,
    pub emit_llm_debug_log: bool,
}

// ─── Internal state ───

/// REPL fence parser state.
struct FenceState {
    response: String,
    in_code_fence: bool,
    current_prose: String,
    current_code: String,
    prose_parts: Vec<String>,
    code_parts: Vec<String>,
    last_line_start: usize,
    code_executed: bool,
    acc: ExecAccumulator,
    latest_usage: LlmUsage,
}

impl FenceState {
    fn new() -> Self {
        Self {
            response: String::new(),
            in_code_fence: false,
            current_prose: String::new(),
            current_code: String::new(),
            prose_parts: Vec::new(),
            code_parts: Vec::new(),
            last_line_start: 0,
            code_executed: false,
            acc: ExecAccumulator::new(),
            latest_usage: LlmUsage::default(),
        }
    }

    fn reset_for_retry(&mut self) {
        self.response.clear();
        self.in_code_fence = false;
        self.current_prose.clear();
        self.current_code.clear();
        self.prose_parts.clear();
        self.code_parts.clear();
        self.last_line_start = 0;
        self.code_executed = false;
        self.acc = ExecAccumulator::new();
    }
}

/// Accumulated REPL turn state carried across exec cycles.
struct ReplTurnState {
    fence: FenceState,
}

struct PendingReplLlmCompletion {
    response_text: String,
    latest_usage: LlmUsage,
}

struct WaitingLlmState {
    retry_attempt: usize,
    fence: Option<FenceState>,
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
        effect_id: EffectId,
        request: LlmRequest,
        // REPL streaming state (None for Standard)
        fence: Option<FenceState>,
        retry_attempt: usize,
        stop_stream_processing: bool,
    },
    WaitingRetry {
        effect_id: EffectId,
        retry_attempt: usize,
        last_error: String,
        request: LlmRequest,
        /// Saved REPL fence state for retry continuation
        fence: Option<FenceState>,
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

/// Sans-IO state machine for a single agent run (multi-turn).
pub struct TurnMachine {
    config: TurnMachineConfig,
    state: MachineState,
    pending_effects: VecDeque<Effect>,
    next_effect_id: u64,
    messages: Vec<Message>,
    /// User-provided images (mime, data).
    user_images: Vec<(String, Vec<u8>)>,
    /// Tool-produced images for the next turn.
    tool_images: Vec<(String, Vec<u8>)>,
    iteration: usize,
    run_offset: usize,
    cumulative_usage: TokenUsage,
    pending_repl_completion: Option<PendingReplLlmCompletion>,
    termination: TurnTerminationPolicyState,
}

impl TurnMachine {
    /// Create a new machine in `PrepareIteration` state.
    pub fn new(
        config: TurnMachineConfig,
        messages: Vec<Message>,
        images: Vec<Vec<u8>>,
        run_offset: usize,
    ) -> Self {
        let user_images: Vec<(String, Vec<u8>)> = images
            .into_iter()
            .map(|png_bytes| ("image/png".to_string(), png_bytes))
            .collect();
        Self {
            config,
            state: MachineState::PreparingMode,
            pending_effects: VecDeque::new(),
            next_effect_id: 1,
            messages,
            user_images,
            tool_images: Vec::new(),
            iteration: run_offset,
            run_offset,
            cumulative_usage: TokenUsage::default(),
            pending_repl_completion: None,
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

    fn emit(&mut self, event: AgentEvent) {
        self.pending_effects.push_back(Effect::Emit(event));
    }

    pub fn fail_turn(&mut self, event: AgentEvent) {
        self.emit(event);
        self.finish();
    }

    fn finish(&mut self) {
        self.emit(AgentEvent::Done);
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

        let all_images: Vec<(String, Vec<u8>)> = self
            .user_images
            .iter()
            .chain(self.tool_images.iter())
            .map(|(mime, data)| (mime.clone(), data.clone()))
            .collect();
        let rendered_prompt = render_prompt(&self.messages, self.config.execution_mode);

        let is_standard = matches!(self.config.execution_mode, ExecutionMode::Standard);
        let has_structured_messages = !rendered_prompt.messages.is_empty();

        let attachments: Vec<LlmAttachment> = if has_structured_messages {
            all_images
                .iter()
                .map(|(mime, data)| LlmAttachment {
                    mime: mime.clone(),
                    data: data.clone(),
                })
                .collect()
        } else {
            rendered_prompt
                .image_indices
                .into_iter()
                .filter_map(|idx| all_images.get(idx))
                .map(|(mime, data)| LlmAttachment {
                    mime: mime.clone(),
                    data: data.clone(),
                })
                .collect()
        };
        let mut user_prompt = rendered_prompt.user_prompt;
        if !has_structured_messages {
            user_prompt.extend((0..attachments.len()).map(LlmPromptPart::Image));
        }

        let llm_request = LlmRequest {
            model: self.config.model.clone(),
            system_prompt,
            user_prompt,
            messages: rendered_prompt.messages,
            attachments,
            tools: if is_standard {
                self.config.tool_specs.clone()
            } else {
                Vec::new()
            },
            tool_choice: if is_standard && !self.config.tool_specs.is_empty() {
                LlmToolChoice::Auto
            } else {
                LlmToolChoice::None
            },
            model_variant: self.config.model_variant.clone(),
            session_id: self.config.session_id.clone(),
            stream_events: None,
        };

        self.tool_images.clear();
        self.queue_llm_request(llm_request, 0, None);
    }

    fn queue_llm_request(
        &mut self,
        request: LlmRequest,
        retry_attempt: usize,
        fence: Option<FenceState>,
    ) {
        self.emit(AgentEvent::LlmRequest {
            iteration: self.iteration,
            message_count: self.messages.len(),
            tool_list: self.config.prompt.tool_list.clone(),
        });

        let id = self.next_id();
        let is_standard = matches!(self.config.execution_mode, ExecutionMode::Standard);
        let fence = if is_standard {
            None
        } else {
            Some(fence.unwrap_or_else(FenceState::new))
        };
        self.state = MachineState::WaitingLlm {
            effect_id: id,
            request: request.clone(),
            fence,
            retry_attempt,
            stop_stream_processing: false,
        };
        self.pending_effects
            .push_back(Effect::LlmCall { id, request });
    }

    /// Feed an incremental LLM text delta (REPL mode only).
    /// Returns `true` if the host should continue streaming, `false` to cancel.
    pub fn handle_llm_delta(&mut self, _id: EffectId, text: &str) -> bool {
        // Extract state we need — avoid holding &mut self.state across pushes to pending_effects
        let (fence, effect_id) = match &mut self.state {
            MachineState::WaitingLlm {
                fence: Some(fence),
                stop_stream_processing,
                effect_id,
                ..
            } => {
                if *stop_stream_processing || text.is_empty() {
                    return !*stop_stream_processing;
                }
                (fence as *mut FenceState, *effect_id)
            }
            _ => return true,
        };

        // SAFETY: We hold &mut self exclusively. The pointer avoids the borrow checker
        // issue of borrowing self.state and self.pending_effects simultaneously.
        // We never invalidate the pointer (no state transitions) until the explicit
        // mem::replace below.
        let fence = unsafe { &mut *fence };

        fence.response.push_str(text);

        let mut queued_effects: Vec<Effect> = Vec::new();
        let mut transition_to_exec: Option<String> = None;

        while let Some(nl) = fence.response[fence.last_line_start..].find('\n') {
            let line_end = fence.last_line_start + nl;
            let line = fence.response[fence.last_line_start..line_end].to_string();
            fence.last_line_start = line_end + 1;

            let parsed = parse_fence_line(
                &line,
                &mut fence.in_code_fence,
                &mut fence.current_prose,
                &mut fence.current_code,
                &mut fence.prose_parts,
            );

            if !parsed.prose_delta.is_empty() {
                queued_effects.push(Effect::Emit(AgentEvent::TextDelta {
                    content: format!("{}\n", parsed.prose_delta),
                }));
            }

            for code in parsed.codes_to_execute {
                fence.code_parts.push(code.clone());
                queued_effects.push(Effect::Emit(AgentEvent::CodeBlock { code: code.clone() }));

                if !fence.acc.had_failure {
                    fence.code_executed = true;
                    transition_to_exec = Some(code);
                    break;
                }
                fence.code_executed = true;
            }

            if transition_to_exec.is_some() || fence.code_executed {
                break;
            }
        }

        // Flush queued effects
        for e in queued_effects {
            self.pending_effects.push_back(e);
        }

        if let Some(code) = transition_to_exec {
            let exec_id = self.next_id();
            let fence_taken = match std::mem::replace(&mut self.state, MachineState::Finished) {
                MachineState::WaitingLlm { fence: Some(f), .. } => f,
                _ => unreachable!(),
            };
            self.pending_repl_completion = Some(PendingReplLlmCompletion {
                response_text: fence_taken.response.clone(),
                latest_usage: fence_taken.latest_usage.clone(),
            });

            self.pending_effects
                .push_back(Effect::CancelLlm { id: effect_id });

            self.state = MachineState::WaitingExec {
                repl: ReplTurnState { fence: fence_taken },
            };
            self.pending_effects
                .push_back(Effect::ExecCode { id: exec_id, code });
            return false;
        }

        // Mark stop if code was executed
        if let MachineState::WaitingLlm {
            fence: Some(f),
            stop_stream_processing,
            ..
        } = &mut self.state
            && f.code_executed
        {
            *stop_stream_processing = true;
            return false;
        }

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
        self.emit(AgentEvent::TokenUsage {
            iteration: self.iteration,
            usage: usage.clone(),
            cumulative,
        });
        if let MachineState::WaitingLlm { fence, .. } = &mut self.state
            && let Some(fence) = fence
        {
            fence.latest_usage = LlmUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cached_input_tokens: usage.cached_input_tokens,
                reasoning_tokens: usage.reasoning_tokens,
            };
        }
        if let Some(pending) = &mut self.pending_repl_completion {
            pending.latest_usage = LlmUsage {
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
        let base_len = self.messages.len();
        let appended = plugin_messages
            .iter()
            .filter(|message| matches!(message.role, MessageRole::User | MessageRole::System))
            .enumerate()
            .map(|(offset, message)| {
                let message_id = format!("m{}", base_len + offset);
                Message {
                    id: message_id.clone(),
                    role: message.role,
                    parts: vec![Part {
                        id: format!("{message_id}.p0"),
                        kind: PartKind::Text,
                        content: message.content.clone(),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: PruneState::Intact,
                    }],
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
            let pending = self.pending_repl_completion.take();
            if let (Some(pending), Ok(llm_response)) = (pending, &result) {
                self.record_llm_usage(
                    llm_response,
                    &pending.response_text,
                    Some(&pending.latest_usage),
                );
            }
            return;
        };
        self.pending_repl_completion = None;

        match result {
            Err(error) => {
                if error.retryable && waiting.retry_attempt < LLM_MAX_RETRIES {
                    self.schedule_llm_retry(
                        waiting.retry_attempt,
                        error,
                        waiting.request,
                        waiting.fence,
                    );
                    return;
                }
                self.emit_llm_error(error);
            }
            Ok(llm_response) => {
                let response_text = self.llm_response_text(&llm_response, waiting.fence.as_ref());
                self.record_llm_usage(
                    &llm_response,
                    response_text,
                    waiting.fence.as_ref().map(|fence| &fence.latest_usage),
                );
                match self.config.execution_mode {
                    ExecutionMode::Standard => {
                        self.handle_standard_llm_success(llm_response, text_streamed)
                    }
                    ExecutionMode::Repl => self.handle_repl_llm_success(
                        llm_response,
                        waiting.request,
                        waiting.fence.unwrap_or_else(FenceState::new),
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
                fence,
                retry_attempt,
                ..
            } => Some(WaitingLlmState {
                retry_attempt,
                fence,
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
        fence: Option<FenceState>,
    ) {
        let delay = LLM_RETRY_DELAYS[retry_attempt];
        let reason = error.message.clone();
        self.emit(AgentEvent::RetryStatus {
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
            fence,
        };
        self.pending_effects.push_back(Effect::Sleep {
            id: sleep_id,
            duration: delay,
        });
    }

    fn llm_response_text<'a>(
        &self,
        llm_response: &'a LlmResponse,
        fence: Option<&'a FenceState>,
    ) -> &'a str {
        match (self.config.execution_mode, fence) {
            (ExecutionMode::Standard, _) => &llm_response.full_text,
            (ExecutionMode::Repl, Some(fence)) if !fence.response.is_empty() => &fence.response,
            (ExecutionMode::Repl, _) => &llm_response.full_text,
        }
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
        self.emit(AgentEvent::TokenUsage {
            iteration: self.iteration,
            usage: usage.clone(),
            cumulative: self.cumulative_usage.clone(),
        });
        if self.config.emit_llm_debug_log {
            self.pending_effects.push_back(Effect::Log {
                event: LogEvent::LlmDebug {
                    agent_id: self.config.agent_id.clone(),
                    iteration: self.iteration,
                    usage,
                    request_body: llm_response.request_body.clone(),
                    response_text: response_text.to_string(),
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
                        if !text_streamed {
                            self.emit(AgentEvent::TextDelta {
                                content: text.clone(),
                            });
                        }
                        assistant_text.push_str(&text);
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
        self.emit(AgentEvent::LlmResponse {
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
            let mid = format!("m{}", self.messages.len());
            self.messages.push(Message {
                id: mid.clone(),
                role: MessageRole::Assistant,
                parts: vec![Part {
                    id: format!("{}.p0", mid),
                    kind: PartKind::Prose,
                    content: assistant_text,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                }],
                origin: None,
            });
            self.request_checkpoint(CheckpointKind::BeforeCompletion, CheckpointResume::Finish);
            return;
        }

        // Build assistant message with tool call parts
        let asst_id = format!("m{}", self.messages.len());
        let mut assistant_parts = Vec::new();
        if !assistant_text.trim().is_empty() {
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::Prose,
                content: assistant_text.clone(),
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
            self.emit(AgentEvent::ToolCall {
                call_id: Some(outcome.call_id.clone()),
                name: outcome.tool_name.clone(),
                args: outcome.args.clone(),
                result: outcome.raw_result.result.clone(),
                success: outcome.raw_result.success,
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
                tool_call_id: Some(outcome.call_id.clone()),
                tool_name: Some(outcome.tool_name.clone()),
                prune_state: PruneState::Intact,
            });

            let base_image_idx = self.user_images.len() + self.tool_images.len();
            for (image_offset, image) in outcome.model_result.images.into_iter().enumerate() {
                self.tool_images
                    .push((image.mime.clone(), image.data.clone()));
                let image_idx = base_image_idx + image_offset;
                result_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[Tool image: {}]", image.label),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
                result_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("{IMAGE_REF_PREFIX}{image_idx}"),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
        }

        if !result_parts.is_empty() {
            let user_id = format!("m{}", self.messages.len());
            for (idx, part) in result_parts.iter_mut().enumerate() {
                part.id = format!("{}.p{}", user_id, idx);
            }
            self.messages.push(Message {
                id: user_id,
                role: MessageRole::User,
                parts: result_parts,
                origin: None,
            });
        }

        self.iteration += 1;
        if let Some(max_turns) = self.config.max_turns
            && self.iteration >= self.run_offset + max_turns
        {
            let sys_id = format!("m{}", self.messages.len());
            self.messages.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Error,
                    content: format!(
                        "Turn limit reached ({max_turns}) before a final assistant response."
                    ),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                }],
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
        request: LlmRequest,
        mut fence: FenceState,
        retry_attempt: usize,
    ) {
        // If we already executed code mid-stream, go to processing
        if fence.code_executed {
            self.emit(AgentEvent::LlmResponse {
                iteration: self.iteration,
                content: fence.response.clone(),
                duration_ms: 0,
            });
            self.state = MachineState::ProcessReplResult {
                repl: ReplTurnState { fence },
            };
            self.process_repl_result();
            return;
        }

        // Process non-streamed deltas (buffered response)
        if fence.response.is_empty() && !llm_response.full_text.is_empty() {
            // Apply full text through fence parser
            for delta in &llm_response.deltas {
                if delta.is_empty() {
                    continue;
                }
                fence.response.push_str(delta);

                while let Some(nl) = fence.response[fence.last_line_start..].find('\n') {
                    let line_end = fence.last_line_start + nl;
                    let line = fence.response[fence.last_line_start..line_end].to_string();
                    fence.last_line_start = line_end + 1;

                    let parsed = parse_fence_line(
                        &line,
                        &mut fence.in_code_fence,
                        &mut fence.current_prose,
                        &mut fence.current_code,
                        &mut fence.prose_parts,
                    );

                    if !parsed.prose_delta.is_empty() {
                        self.emit(AgentEvent::TextDelta {
                            content: format!("{}\n", parsed.prose_delta),
                        });
                    }

                    if let Some(code) = parsed.codes_to_execute.into_iter().next() {
                        fence.code_parts.push(code.clone());
                        self.emit(AgentEvent::CodeBlock { code: code.clone() });
                        fence.code_executed = true;

                        let exec_id = self.next_id();
                        self.emit(AgentEvent::LlmResponse {
                            iteration: self.iteration,
                            content: fence.response.clone(),
                            duration_ms: 0,
                        });
                        self.state = MachineState::WaitingExec {
                            repl: ReplTurnState { fence },
                        };
                        self.pending_effects
                            .push_back(Effect::ExecCode { id: exec_id, code });
                        return;
                    }
                }
                if fence.code_executed {
                    break;
                }
            }

            if fence.code_executed {
                self.emit(AgentEvent::LlmResponse {
                    iteration: self.iteration,
                    content: fence.response.clone(),
                    duration_ms: 0,
                });
                self.state = MachineState::ProcessReplResult {
                    repl: ReplTurnState { fence },
                };
                self.process_repl_result();
                return;
            }

            if fence.response.is_empty() {
                fence.response = llm_response.full_text.clone();
            }
        }

        self.emit(AgentEvent::LlmResponse {
            iteration: self.iteration,
            content: if fence.response.is_empty() {
                llm_response.full_text.clone()
            } else {
                fence.response.clone()
            },
            duration_ms: 0,
        });

        // Process trailing text
        if fence.last_line_start < fence.response.len() {
            let trailing = fence.response[fence.last_line_start..].to_string();
            let parsed = parse_fence_line(
                &trailing,
                &mut fence.in_code_fence,
                &mut fence.current_prose,
                &mut fence.current_code,
                &mut fence.prose_parts,
            );
            if !parsed.prose_delta.is_empty() {
                self.emit(AgentEvent::TextDelta {
                    content: parsed.prose_delta,
                });
            }
            if let Some(code) = parsed.codes_to_execute.into_iter().next() {
                fence.code_parts.push(code.clone());
                self.emit(AgentEvent::CodeBlock { code: code.clone() });
                fence.code_executed = true;

                let exec_id = self.next_id();
                self.state = MachineState::WaitingExec {
                    repl: ReplTurnState { fence },
                };
                self.pending_effects
                    .push_back(Effect::ExecCode { id: exec_id, code });
                return;
            }
        }

        // Unclosed code fence at end of response
        if fence.in_code_fence && !fence.current_code.trim().is_empty() {
            let code = fence.current_code.clone();
            fence.code_parts.push(code.clone());
            self.emit(AgentEvent::CodeBlock { code: code.clone() });
            fence.current_code.clear();
            fence.code_executed = true;

            let exec_id = self.next_id();
            self.state = MachineState::WaitingExec {
                repl: ReplTurnState { fence },
            };
            self.pending_effects
                .push_back(Effect::ExecCode { id: exec_id, code });
            return;
        }

        // Flush remaining prose
        let remaining_prose = fence.current_prose.trim().to_string();
        if !remaining_prose.is_empty() {
            fence.prose_parts.push(remaining_prose);
            fence.current_prose.clear();
        }

        if fence.response.is_empty() && !llm_response.full_text.is_empty() {
            fence.response = llm_response.full_text;
        }

        // Check for malformed output
        if is_malformed_assistant_output(&fence.response) {
            let preview = truncate_raw_error(fence.response.trim());
            if retry_attempt < LLM_MAX_RETRIES {
                fence.reset_for_retry();
                self.schedule_llm_retry(
                    retry_attempt,
                    LlmCallError {
                        message: "malformed assistant output from model (partial repl fragment)"
                            .to_string(),
                        retryable: true,
                        raw: Some(preview.clone()),
                        code: Some("malformed_output".to_string()),
                    },
                    request,
                    Some(fence),
                );
                return;
            }
            self.emit(make_error_event(
                "llm_provider",
                Some("malformed_output"),
                "Model returned malformed output. Use /retry to replay this turn.",
                Some(preview),
            ));
            self.finish();
            return;
        }

        self.state = MachineState::ProcessReplResult {
            repl: ReplTurnState { fence },
        };
        self.process_repl_result();
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
                    self.emit(AgentEvent::ToolCall {
                        call_id: None,
                        name: tc.tool.clone(),
                        args: tc.args.clone(),
                        result: tc.result.clone(),
                        success: tc.success,
                        duration_ms: tc.duration_ms,
                    });
                }
                if !r.output.is_empty() || r.error.is_some() {
                    self.emit(AgentEvent::CodeOutput {
                        output: r.output.clone(),
                        error: r.error.clone(),
                    });
                }

                repl.fence.acc.tool_calls.extend(r.tool_calls);
                repl.fence.acc.images.extend(r.images);
                if !r.output.is_empty() {
                    repl.fence.acc.combined_output.push_str(&r.output);
                }
                for observation in r.observations {
                    if !observation.is_empty() {
                        if !repl.fence.acc.combined_output.is_empty()
                            && !repl.fence.acc.combined_output.ends_with('\n')
                        {
                            repl.fence.acc.combined_output.push('\n');
                        }
                        repl.fence.acc.combined_output.push_str(&observation);
                        if !repl.fence.acc.combined_output.ends_with('\n') {
                            repl.fence.acc.combined_output.push('\n');
                        }
                    }
                }
                if let Some(raw_error) = r.error {
                    repl.fence.acc.exec_error = Some(raw_error);
                    repl.fence.acc.had_failure = true;
                }
            }
            Err(e) => {
                self.emit(AgentEvent::CodeOutput {
                    output: String::new(),
                    error: Some(e.clone()),
                });
                repl.fence.acc.exec_error = Some(e);
                repl.fence.acc.had_failure = true;
            }
        }

        // Move to processing the repl result
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

        let fence = repl.fence;
        let executed_text = &fence.response;
        let has_code = !fence.code_parts.is_empty();

        // Check for empty response with no execution
        if executed_text.trim().is_empty()
            && fence.acc.tool_calls.is_empty()
            && fence.acc.combined_output.is_empty()
            && !fence.acc.had_failure
        {
            self.emit(make_error_event(
                "llm_provider",
                Some("empty_response"),
                "I didn't get a response. Use /retry to replay this turn, or /provider if credentials changed.",
                None,
            ));
            self.finish();
            return;
        }

        // Keep tool images
        let mut next_tool_image_refs: Vec<(usize, String)> = Vec::new();
        let base_image_idx = self.user_images.len();
        for (i, img) in fence.acc.images.iter().enumerate() {
            self.tool_images.push((img.mime.clone(), img.data.clone()));
            next_tool_image_refs.push((base_image_idx + i, img.label.clone()));
        }

        let has_output = !fence.acc.combined_output.is_empty();
        let has_tool_calls = !fence.acc.tool_calls.is_empty();

        // Pure prose response finalizes the turn, even after prior REPL cycles.
        if !has_code && !has_output && !has_tool_calls && !fence.acc.had_failure {
            let mid = format!("m{}", self.messages.len());
            let asst_parts = build_assistant_parts(&mid, &fence.prose_parts, &fence.code_parts);
            self.messages.push(Message {
                id: mid,
                role: MessageRole::Assistant,
                parts: asst_parts,
                origin: None,
            });
            self.request_checkpoint(CheckpointKind::BeforeCompletion, CheckpointResume::Finish);
            return;
        }

        // Build feedback parts for execution results. The executed code itself already lives
        // in the assistant message and should not be replayed a second time via system feedback.
        let mut feedback_parts: Vec<Part> = Vec::new();

        if has_output {
            let mut output_text = fence.acc.combined_output.clone();
            if has_tool_calls {
                output_text.push_str(&format!(
                    "\n[{} tool call(s) executed]",
                    fence.acc.tool_calls.len()
                ));
            }
            feedback_parts.push(Part {
                id: String::new(),
                kind: PartKind::Output,
                content: output_text,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
        } else if has_tool_calls {
            feedback_parts.push(Part {
                id: String::new(),
                kind: PartKind::Output,
                content: format!("[{} tool call(s) executed]", fence.acc.tool_calls.len()),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
        }
        if let Some(err) = &fence.acc.exec_error {
            feedback_parts.push(Part {
                id: String::new(),
                kind: PartKind::Error,
                content: format!("{}\nFix and retry.", err),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
        }
        if !next_tool_image_refs.is_empty() {
            for (idx, label) in &next_tool_image_refs {
                feedback_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[Tool image: {}]", label),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
                feedback_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("{IMAGE_REF_PREFIX}{idx}"),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
        }

        // Push assistant message
        let asst_id = format!("m{}", self.messages.len());
        let asst_parts = build_assistant_parts(&asst_id, &fence.prose_parts, &fence.code_parts);
        self.messages.push(Message {
            id: asst_id,
            role: MessageRole::Assistant,
            parts: asst_parts,
            origin: None,
        });

        // Push system feedback only when execution produced something new to feed back.
        if !feedback_parts.is_empty() {
            let sys_id = format!("m{}", self.messages.len());
            for (idx, part) in feedback_parts.iter_mut().enumerate() {
                part.id = format!("{}.p{}", sys_id, idx);
            }
            self.messages.push(Message {
                id: sys_id,
                role: MessageRole::System,
                parts: feedback_parts,
                origin: None,
            });
        }

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
        let (effect_id, retry_attempt, last_error, request, fence) =
            match std::mem::replace(&mut self.state, MachineState::Finished) {
                MachineState::WaitingRetry {
                    effect_id,
                    retry_attempt,
                    last_error,
                    request,
                    fence,
                    ..
                } => (effect_id, retry_attempt, last_error, request, fence),
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
                fence,
            };
            return;
        }

        self.queue_llm_request(request, retry_attempt, fence);
    }
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

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Message, MessageRole, Part, PartKind, PruneState};

    fn test_config(mode: ExecutionMode) -> TurnMachineConfig {
        TurnMachineConfig {
            execution_mode: mode,
            context_strategy: crate::default_context_strategy(),
            model: "test-model".to_string(),
            max_turns: None,
            model_variant: None,
            session_id: None,
            tool_specs: Vec::new(),
            prompt: crate::PromptContext {
                mode,
                include_soul: false,
                ..crate::PromptContext::default()
            },
            prompt_renderer: crate::default_prompt_renderer(),
            prompt_overrides: Vec::new(),
            agent_id: "test".to_string(),
            emit_llm_debug_log: false,
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
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
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
            Effect::Emit(AgentEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
                envelope: _,
            }) => Some((*wait_seconds, *attempt, *max_attempts, reason.clone())),
            _ => None,
        })
    }

    fn find_retry_envelope(effects: &[Effect]) -> Option<crate::agent::ErrorEnvelope> {
        effects.iter().find_map(|e| match e {
            Effect::Emit(AgentEvent::RetryStatus {
                envelope: Some(envelope),
                ..
            }) => Some(envelope.clone()),
            _ => None,
        })
    }

    fn has_error_event(effects: &[Effect], needle: &str) -> bool {
        effects.iter().any(|e| match e {
            Effect::Emit(AgentEvent::Error { message, .. }) => message.contains(needle),
            _ => false,
        })
    }

    fn find_checkpoint(effects: &[Effect]) -> Option<(EffectId, CheckpointKind)> {
        effects.iter().find_map(|e| match e {
            Effect::Checkpoint { id, checkpoint } => Some((*id, *checkpoint)),
            _ => None,
        })
    }

    fn find_token_usage(effects: &[Effect]) -> Option<TokenUsage> {
        effects.iter().find_map(|e| match e {
            Effect::Emit(AgentEvent::TokenUsage { usage, .. }) => Some(usage.clone()),
            _ => None,
        })
    }

    fn find_llm_debug(effects: &[Effect]) -> Option<(TokenUsage, String)> {
        effects.iter().find_map(|e| match e {
            Effect::Log {
                event:
                    LogEvent::LlmDebug {
                        usage,
                        response_text,
                        ..
                    },
            } => Some((usage.clone(), response_text.clone())),
            _ => None,
        })
    }

    // ─── Standard tests ───

    #[test]
    fn standard_prose_only_response_emits_done() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
    fn llm_request_includes_image_prompt_parts_for_attached_images() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![
                Part {
                    id: "m0.p0".to_string(),
                    kind: PartKind::Text,
                    content: "__LASH_IMAGE_IDX:0".to_string(),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                },
                Part {
                    id: "m0.p1".to_string(),
                    kind: PartKind::Text,
                    content: "explain this".to_string(),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                },
            ],
            origin: None,
        }];
        let mut machine = TurnMachine::new(config, msgs, vec![vec![1, 2, 3]], 0);

        let effects = drain_effects(&mut machine);
        let request = effects
            .into_iter()
            .find_map(|effect| match effect {
                Effect::LlmCall { request, .. } => Some(request),
                _ => None,
            })
            .expect("llm call");

        assert_eq!(request.attachments.len(), 1);
        assert!(request.user_prompt.iter().any(|part| matches!(
            part,
            LlmPromptPart::Text(text) if text.contains("[Image attached]") && text.contains("explain this")
        )));
        assert!(
            request
                .user_prompt
                .iter()
                .any(|part| matches!(part, LlmPromptPart::Image(0)))
        );
    }

    #[test]
    fn standard_tool_calls_produce_effects_and_loop() {
        let config = test_config(ExecutionMode::Standard);
        let msgs = vec![user_message("read file")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
                raw_result: crate::ToolResult::ok(serde_json::json!("file contents")),
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
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
                raw_result: crate::ToolResult::ok(serde_json::json!({
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
                Effect::Emit(AgentEvent::ToolCall { args, .. }) => Some(args.clone()),
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
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
            .any(|e| matches!(e, Effect::Emit(AgentEvent::Error { .. })));
        assert!(has_error);
        assert!(find_done(&effects).is_some());
    }

    #[test]
    fn standard_max_turns_stops_iteration() {
        let mut config = test_config(ExecutionMode::Standard);
        config.max_turns = Some(1);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
                raw_result: crate::ToolResult::ok(serde_json::json!("ok")),
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
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

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
            messages: vec![PluginMessage {
                role: MessageRole::User,
                content: "one more thing".to_string(),
            }],
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
    fn repl_checkpoint_after_work_continues_turn() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("run some code")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("llm call");

        let cont =
            machine.handle_llm_delta(llm_id, "Here's the code:\n<repl>\nprint('hi')\n</repl>\n");
        assert!(!cont);

        let effects = drain_effects(&mut machine);
        let (exec_id, _) = effects
            .iter()
            .find_map(|e| match e {
                Effect::ExecCode { id, code } => Some((*id, code.clone())),
                _ => None,
            })
            .expect("exec");

        machine.handle_response(Response::ExecResult {
            id: exec_id,
            result: Ok(crate::ExecResponse {
                output: "hi\n".to_string(),
                observations: Vec::new(),
                tool_calls: Vec::new(),
                images: Vec::new(),
                error: None,
                duration_ms: 1,
            }),
        });
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse::default()),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::AfterWork);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: vec![PluginMessage {
                role: MessageRole::User,
                content: "keep going".to_string(),
            }],
        });

        let effects = drain_effects(&mut machine);
        assert!(find_llm_call(&effects).is_some());
    }

    #[test]
    fn repl_code_block_triggers_exec() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("run some code")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        // Stream deltas with a code block
        let cont =
            machine.handle_llm_delta(llm_id, "Here's the code:\n<repl>\nprint('hi')\n</repl>\n");
        assert!(!cont, "should signal to cancel stream after code block");

        let effects = drain_effects(&mut machine);
        let exec_effect = effects.iter().find_map(|e| match e {
            Effect::ExecCode { id, code } => Some((*id, code.clone())),
            _ => None,
        });
        assert!(exec_effect.is_some(), "should emit ExecCode");
        let (exec_id, code) = exec_effect.unwrap();
        assert_eq!(code, "print('hi')");

        // Feed exec result
        machine.handle_response(Response::ExecResult {
            id: exec_id,
            result: Ok(crate::ExecResponse {
                output: "hi\n".to_string(),
                observations: Vec::new(),
                tool_calls: Vec::new(),
                images: Vec::new(),
                error: None,
                duration_ms: 5,
            }),
        });

        // Now feed the LlmComplete since stream was cancelled
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
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
                full_text: "done".to_string(),
                deltas: vec!["done".to_string()],
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
    fn repl_stream_cancel_still_records_usage_and_debug_log() {
        let mut config = test_config(ExecutionMode::Repl);
        config.emit_llm_debug_log = true;
        let msgs = vec![user_message("run some code")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("llm call");

        machine.handle_llm_usage(
            llm_id,
            &LlmUsage {
                input_tokens: 321,
                output_tokens: 123,
                cached_input_tokens: 45,
                reasoning_tokens: 67,
            },
        );
        let cont =
            machine.handle_llm_delta(llm_id, "Here's the code:\n<repl>\nprint('hi')\n</repl>\n");
        assert!(!cont);

        let effects = drain_effects(&mut machine);
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::ExecCode { .. })),
            "should emit ExecCode"
        );

        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: true,
            result: Ok(LlmResponse::default()),
        });

        let effects = drain_effects(&mut machine);
        let usage = find_token_usage(&effects).expect("token usage");
        assert_eq!(usage.input_tokens, 321);
        assert_eq!(usage.output_tokens, 123);
        assert_eq!(usage.cached_input_tokens, 45);
        assert_eq!(usage.reasoning_tokens, 67);

        let (debug_usage, response_text) = find_llm_debug(&effects).expect("llm debug");
        assert_eq!(debug_usage.input_tokens, 321);
        assert_eq!(debug_usage.output_tokens, 123);
        assert_eq!(debug_usage.cached_input_tokens, 45);
        assert_eq!(debug_usage.reasoning_tokens, 67);
        assert_eq!(
            response_text,
            "Here's the code:\n<repl>\nprint('hi')\n</repl>\n"
        );
    }

    #[test]
    fn repl_prose_after_exec_finishes_turn() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("run code then summarize")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        let cont = machine.handle_llm_delta(llm_id, "<repl>\nprint('hi')\n</repl>\n");
        assert!(!cont);

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
            }),
        });
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
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
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::AfterWork);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        let next_llm_id = *find_llm_call(&effects).expect("next llm call");
        assert!(find_done(&effects).is_none());

        machine.handle_response(Response::LlmComplete {
            id: next_llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "All done.".to_string(),
                deltas: vec!["All done.".to_string()],
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
    fn repl_followup_prompt_replays_executed_code_once() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("run code then continue")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).expect("llm call");

        let cont = machine.handle_llm_delta(llm_id, "<repl>\nprint('hi')\n</repl>\n");
        assert!(!cont);

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
            }),
        });
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse::default()),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::AfterWork);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        let request = find_llm_request(&effects).expect("next llm request");
        let prompt_text = request
            .user_prompt
            .iter()
            .filter_map(|part| match part {
                LlmPromptPart::Text(text) => Some(text.as_str()),
                LlmPromptPart::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("");
        let replay_text = request
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let code_block = "<repl>\nprint('hi')\n</repl>";
        assert_eq!(
            format!("{prompt_text}\n{replay_text}")
                .matches(code_block)
                .count(),
            1
        );
        assert_eq!(
            format!("{prompt_text}\n{replay_text}")
                .matches("<output>\nhi\n\n</output>")
                .count(),
            1
        );
    }

    #[test]
    fn repl_exec_error_produces_feedback() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("run code")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        // Feed deltas with code block via buffered response
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "<repl>\nbad_code()\n</repl>\n".to_string(),
                deltas: vec!["<repl>\nbad_code()\n</repl>\n".to_string()],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let exec_effect = effects.iter().find_map(|e| match e {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        });
        assert!(exec_effect.is_some());

        // Return error
        machine.handle_response(Response::ExecResult {
            id: exec_effect.unwrap(),
            result: Err("NameError: name 'bad_code' is not defined".to_string()),
        });

        let effects = drain_effects(&mut machine);
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::AfterWork);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let effects = drain_effects(&mut machine);
        // Should have error event and loop to next iteration (not done, since there was output)
        let has_llm_call = effects.iter().any(|e| matches!(e, Effect::LlmCall { .. }));
        assert!(
            has_llm_call,
            "should loop to next iteration after exec error"
        );
    }

    #[test]
    fn repl_observations_feed_back_without_visible_code_output() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("inspect a value")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        let cont = machine.handle_llm_delta(llm_id, "Check it:\n<repl>\nobserve value\n</repl>\n");
        assert!(!cont);

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
                output: String::new(),
                observations: vec!["value={\"ok\":true}".to_string()],
                tool_calls: Vec::new(),
                images: Vec::new(),
                error: None,
                duration_ms: 1,
            }),
        });
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse::default()),
        });

        let effects = drain_effects(&mut machine);
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, Effect::Emit(AgentEvent::CodeOutput { .. })))
        );
        let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
        assert_eq!(checkpoint, CheckpointKind::AfterWork);
        machine.handle_response(Response::Checkpoint {
            id: checkpoint_id,
            messages: Vec::new(),
        });

        let messages = machine.messages();
        let system_output = messages
            .iter()
            .rev()
            .find(|message| matches!(message.role, MessageRole::System))
            .and_then(|message| {
                message
                    .parts
                    .iter()
                    .find(|part| matches!(part.kind, PartKind::Output))
            })
            .map(|part| part.content.clone())
            .expect("system output feedback");
        assert!(system_output.contains("value={\"ok\":true}"));
    }

    #[test]
    fn repl_malformed_output_retries() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("hello")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        // Malformed output
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Ok(LlmResponse {
                full_text: "<repl>".to_string(),
                deltas: vec!["<repl>".to_string()],
                ..LlmResponse::default()
            }),
        });

        let effects = drain_effects(&mut machine);
        let has_sleep = effects.iter().any(|e| matches!(e, Effect::Sleep { .. }));
        assert!(has_sleep, "should retry with sleep on malformed output");
    }

    #[test]
    fn repl_mid_stream_cancel_when_fence_closes() {
        let config = test_config(ExecutionMode::Repl);
        let msgs = vec![user_message("code")];
        let mut machine = TurnMachine::new(config, msgs, Vec::new(), 0);

        let effects = drain_effects(&mut machine);
        let llm_id = *find_llm_call(&effects).unwrap();

        // Stream: first delta opens fence
        let cont = machine.handle_llm_delta(llm_id, "<repl>\n");
        assert!(cont, "should continue while fence is open");

        // Stream: code line
        let cont = machine.handle_llm_delta(llm_id, "x = 42\n");
        assert!(cont, "should continue with code");

        // Stream: close fence
        let cont = machine.handle_llm_delta(llm_id, "</repl>\n");
        assert!(!cont, "should cancel after fence closes");

        let effects = drain_effects(&mut machine);
        let has_cancel = effects
            .iter()
            .any(|e| matches!(e, Effect::CancelLlm { .. }));
        let has_exec = effects.iter().any(|e| matches!(e, Effect::ExecCode { .. }));
        assert!(has_cancel, "should emit CancelLlm");
        assert!(has_exec, "should emit ExecCode");
    }
}
