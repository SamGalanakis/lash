//! Sans-IO state machine for session turns.
//!
//! `TurnMachine` owns the generic effect engine. Protocol-specific behavior
//! lives behind `ProtocolDriverHandle`, which returns declarative
//! `DriverAction`s that the machine applies.

use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::llm::types::{
    LlmAttachment, LlmOutputPart, LlmRequest, LlmResponse, LlmTerminalReason, LlmToolChoice,
    LlmToolSpec, ProviderReplayMeta,
};
use crate::session_model::message::MessageOrigin;
use crate::session_model::{
    Message, MessageRole, MessageSequence, Part, PartKind, PruneState, SessionEvent,
    SessionEventRecord, TokenUsage, ToolEvent, TurnTerminationPolicyState, make_error_event,
    reassign_part_ids, render_prompt,
};
use crate::{
    CheckpointKind, ModelToolReturn, PluginMessage, ToolCallOutput, TurnOutcome, TurnStop,
};

// ─── Public types ───

pub trait TurnProtocol: Send + Sync + 'static {
    type Event: Clone + Serialize + DeserializeOwned + Debug + Send + Sync + 'static;
    type Termination: Clone + Default + Debug + Send + Sync + 'static;
    type DriverState: Clone + Default + Serialize + DeserializeOwned + Debug + Send + Sync + 'static;
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct UnitTurnProtocol;

impl TurnProtocol for UnitTurnProtocol {
    type Event = ();
    type Termination = ();
    type DriverState = serde_json::Value;
}

/// Opaque identifier linking an effect to its response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct EffectId(pub u64);

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct PendingToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub args: Value,
    /// Opaque provider replay state carried through for the next request.
    pub replay: Option<ProviderReplayMeta>,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct CompletedToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub args: Value,
    pub output: ToolCallOutput,
    pub model_return: ModelToolReturn,
    pub duration_ms: u64,
    /// See [`PendingToolCall::replay`].
    pub replay: Option<ProviderReplayMeta>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct TurnCause {
    pub id: String,
    pub event_type: String,
    pub origin: MessageOrigin,
    pub text: String,
}

impl TurnCause {
    pub fn to_event_message(&self) -> Message {
        Message {
            id: self.id.clone(),
            role: MessageRole::Event,
            parts: Arc::new(vec![Part {
                id: format!("{}.p0", self.id),
                kind: PartKind::Text,
                content: self.text.clone(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]),
            origin: Some(self.origin.clone()),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, serde::Deserialize)]
pub struct CheckpointDelivery {
    pub messages: Vec<PluginMessage>,
    pub transient_messages: Vec<PluginMessage>,
    pub turn_causes: Vec<TurnCause>,
}

pub fn render_turn_causes_prompt(causes: &[TurnCause]) -> Option<String> {
    if causes.is_empty() {
        return None;
    }

    let mut rendered = String::from("=== TURN EVENTS ===");
    for (index, cause) in causes.iter().enumerate() {
        rendered.push_str("\n\n");
        rendered.push_str(&format!(
            "--- event[{index}] · {} · {} ---\n",
            cause.event_type, cause.id
        ));
        rendered.push_str("Origin: ");
        rendered.push_str(&render_message_origin(&cause.origin));
        rendered.push_str("\n\n");
        rendered.push_str(cause.text.trim());
    }
    Some(rendered)
}

fn render_message_origin(origin: &MessageOrigin) -> String {
    match origin {
        MessageOrigin::Plugin {
            plugin_id,
            transient,
        } => {
            if *transient {
                format!("plugin {plugin_id} (transient)")
            } else {
                format!("plugin {plugin_id}")
            }
        }
        MessageOrigin::Process {
            process_id,
            event_type,
            sequence,
            wake_id,
        } => match wake_id {
            Some(wake_id) => {
                format!("process {process_id} {event_type} #{sequence} ({wake_id})")
            }
            None => format!("process {process_id} {event_type} #{sequence}"),
        },
    }
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub enum LogEvent {
    LlmDebug {
        session_id: String,
        protocol_iteration: usize,
        usage: TokenUsage,
        provider_usage: Option<Value>,
        request_body: Option<String>,
        response_text: String,
        response_parts: Option<Value>,
    },
    LlmError {
        session_id: String,
        protocol_iteration: usize,
        request_body: Option<String>,
        message: String,
        retryable: bool,
        raw: Option<String>,
        code: Option<String>,
        terminal_reason: LlmTerminalReason,
    },
}

/// An effect the host must fulfil.
//
// `Clone` is implemented by hand below rather than derived: the derive would
// demand `M: Clone`, but only `M::Event` is ever cloned (and `TurnProtocol`
// already guarantees `Event: Clone`), so a manual impl keeps `Effect<M>`
// cloneable for every protocol — which the turn checkpoint relies on.
#[derive(Debug, Serialize, serde::Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Effect<M: TurnProtocol = UnitTurnProtocol> {
    /// Sync the live execution surface before the turn proceeds.
    ///
    /// `update_machine_config` is only needed after the turn has
    /// already advanced at least once and the host may need to swap in
    /// a refreshed system prompt or tool schema for the next
    /// protocol iteration. Initial syncs are host-only because the machine was
    /// already constructed from a fresh execution surface.
    SyncExecutionSurface {
        id: EffectId,
        update_machine_config: bool,
    },
    /// Start an LLM call.
    LlmCall {
        id: EffectId,
        request: Arc<LlmRequest>,
    },
    /// Cancel an in-progress LLM stream.
    CancelLlm { id: EffectId },
    /// Execute one or more driver-scheduled tool calls.
    ToolCalls {
        id: EffectId,
        calls: Vec<PendingToolCall>,
    },
    /// Execute a protocol-owned code block.
    ExecCode { id: EffectId, code: String },
    /// Run a host/plugin checkpoint before the machine continues or completes.
    Checkpoint {
        id: EffectId,
        checkpoint: CheckpointKind,
    },
    /// Host-implemented fire-and-forget logging.
    Log { event: LogEvent },
    /// Fire-and-forget event (no response needed).
    Emit(SessionEvent),
    /// Prompt-history progress that may be durably persisted by the host.
    ///
    /// This is separate from [`SessionEvent`]: UI stream events can be partial,
    /// duplicated, or display-only, while `Progress` is emitted only after the
    /// state machine has applied semantic message or protocol-step changes.
    Progress {
        messages: MessageSequence,
        event_delta: Vec<SessionEventRecord<M::Event>>,
        protocol_iteration: usize,
    },
    /// Turn is done.
    Done {
        messages: MessageSequence,
        event_delta: Vec<SessionEventRecord<M::Event>>,
        protocol_iteration: usize,
    },
}

impl<M: TurnProtocol> Clone for Effect<M> {
    fn clone(&self) -> Self {
        match self {
            Self::SyncExecutionSurface {
                id,
                update_machine_config,
            } => Self::SyncExecutionSurface {
                id: *id,
                update_machine_config: *update_machine_config,
            },
            Self::LlmCall { id, request } => Self::LlmCall {
                id: *id,
                request: Arc::clone(request),
            },
            Self::CancelLlm { id } => Self::CancelLlm { id: *id },
            Self::ToolCalls { id, calls } => Self::ToolCalls {
                id: *id,
                calls: calls.clone(),
            },
            Self::ExecCode { id, code } => Self::ExecCode {
                id: *id,
                code: code.clone(),
            },
            Self::Checkpoint { id, checkpoint } => Self::Checkpoint {
                id: *id,
                checkpoint: *checkpoint,
            },
            Self::Log { event } => Self::Log {
                event: event.clone(),
            },
            Self::Emit(event) => Self::Emit(event.clone()),
            Self::Progress {
                messages,
                event_delta,
                protocol_iteration,
            } => Self::Progress {
                messages: messages.clone(),
                event_delta: event_delta.clone(),
                protocol_iteration: *protocol_iteration,
            },
            Self::Done {
                messages,
                event_delta,
                protocol_iteration,
            } => Self::Done {
                messages: messages.clone(),
                event_delta: event_delta.clone(),
                protocol_iteration: *protocol_iteration,
            },
        }
    }
}

impl<M: TurnProtocol> Effect<M> {
    fn id(&self) -> Option<EffectId> {
        match self {
            Self::SyncExecutionSurface { id, .. }
            | Self::LlmCall { id, .. }
            | Self::CancelLlm { id }
            | Self::ToolCalls { id, .. }
            | Self::ExecCode { id, .. }
            | Self::Checkpoint { id, .. } => Some(*id),
            Self::Log { .. } | Self::Emit(_) | Self::Progress { .. } | Self::Done { .. } => None,
        }
    }
}

/// Error details from a failed LLM call.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct LlmCallError {
    pub message: String,
    pub retryable: bool,
    pub raw: Option<String>,
    pub code: Option<String>,
    pub terminal_reason: LlmTerminalReason,
    pub request_body: Option<String>,
}

/// A response to a previously emitted effect.
pub enum Response {
    /// Live execution surface sync completed.
    ExecutionSurfaceSynced {
        id: EffectId,
        result: Result<Option<ExecutionSurfaceSync>, String>,
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
    /// Mode code execution result.
    ExecResult {
        id: EffectId,
        result: Result<crate::ExecResponse, String>,
    },
    /// Checkpoint result with optional injected messages.
    Checkpoint {
        id: EffectId,
        delivery: CheckpointDelivery,
    },
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct ExecutionSurfaceSync {
    pub system_prompt: Arc<str>,
    pub tool_specs: Arc<Vec<LlmToolSpec>>,
}

pub struct WaitingLlmState<M: TurnProtocol = UnitTurnProtocol> {
    pub request: Arc<LlmRequest>,
    driver_state: Option<M::DriverState>,
}

impl<M: TurnProtocol> WaitingLlmState<M> {
    pub fn take_driver_state(&mut self) -> Option<M::DriverState> {
        self.driver_state.take()
    }
}

pub struct WaitingExecState<M: TurnProtocol = UnitTurnProtocol> {
    driver_state: M::DriverState,
}

impl<M: TurnProtocol> WaitingExecState<M> {
    pub fn into_driver_state(self) -> M::DriverState {
        self.driver_state
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub enum CheckpointResumeAction {
    PrepareIteration,
    Finish(TurnOutcome),
}

#[allow(clippy::large_enum_variant)]
pub enum DriverAction<M: TurnProtocol = UnitTurnProtocol> {
    Emit(SessionEvent),
    AppendEvents(Vec<SessionEventRecord<M::Event>>),
    StartLlm {
        request: Arc<LlmRequest>,
        driver_state: Option<M::DriverState>,
    },
    StartTools {
        calls: Vec<PendingToolCall>,
    },
    StartExec {
        code: String,
        driver_state: M::DriverState,
    },
    StartCheckpoint {
        checkpoint: CheckpointKind,
        on_empty: CheckpointResumeAction,
    },
    AdvanceProtocolIteration,
    ScheduleTurnLimitFinal {
        message: Message,
    },
    Finish(TurnOutcome),
}

pub struct DriverContextView<'a, M: TurnProtocol = UnitTurnProtocol> {
    config: &'a TurnMachineConfig<M>,
    messages: &'a MessageSequence,
    events: &'a [SessionEventRecord<M::Event>],
    turn_causes: &'a [TurnCause],
    protocol_iteration: usize,
    protocol_run_offset: usize,
    termination: &'a TurnTerminationPolicyState,
}

impl<'a, M: TurnProtocol> DriverContextView<'a, M> {
    pub fn project_llm_request(&self, use_tools: bool) -> Arc<LlmRequest> {
        self.config.projector.project(ProjectorContext {
            config: self.config,
            messages: self.messages,
            events: self.events,
            turn_causes: self.turn_causes,
            protocol_iteration: self.protocol_iteration,
            use_tools,
        })
    }

    pub fn protocol_iteration(&self) -> usize {
        self.protocol_iteration
    }

    pub fn protocol_run_offset(&self) -> usize {
        self.protocol_run_offset
    }

    pub fn max_turns(&self) -> Option<usize> {
        self.config.max_turns
    }

    pub fn termination(&self) -> &M::Termination {
        &self.config.termination
    }

    pub fn autonomous(&self) -> bool {
        self.config.autonomous
    }

    pub fn should_force_exit_after_grace_turn(&self) -> bool {
        self.termination.should_force_exit_after_grace_turn()
    }

    pub fn turn_limit_final_to_schedule(&self) -> Option<usize> {
        self.termination.turn_limit_final_to_schedule(
            self.protocol_iteration,
            self.protocol_run_offset,
            self.config.max_turns,
        )
    }

    pub fn messages(&self) -> &MessageSequence {
        self.messages
    }

    pub fn events(&self) -> &[SessionEventRecord<M::Event>] {
        self.events
    }

    pub fn turn_causes(&self) -> &[TurnCause] {
        self.turn_causes
    }
}

pub struct ProjectorContext<'a, M: TurnProtocol = UnitTurnProtocol> {
    pub config: &'a TurnMachineConfig<M>,
    pub messages: &'a MessageSequence,
    pub events: &'a [SessionEventRecord<M::Event>],
    pub turn_causes: &'a [TurnCause],
    pub protocol_iteration: usize,
    pub use_tools: bool,
}

pub trait ContextProjector<M: TurnProtocol = UnitTurnProtocol>: Send + Sync {
    fn project(&self, ctx: ProjectorContext<'_, M>) -> Arc<LlmRequest>;
}

#[derive(Clone, Debug, Default)]
pub struct ChatContextProjector;

impl<M: TurnProtocol> ContextProjector<M> for ChatContextProjector {
    fn project(&self, ctx: ProjectorContext<'_, M>) -> Arc<LlmRequest> {
        let rendered_prompt = render_messages_for_projector(ctx.messages, ctx.turn_causes);
        let attachments: Vec<LlmAttachment> = rendered_prompt.attachments;
        let mut messages = rendered_prompt.messages;
        if let Some(turn_events) = render_turn_causes_prompt(ctx.turn_causes) {
            messages.push(crate::llm::types::LlmMessage::text(
                crate::llm::types::LlmRole::User,
                Arc::from(turn_events),
            ));
        }
        if !ctx.config.system_prompt.trim().is_empty() {
            messages.insert(
                0,
                crate::llm::types::LlmMessage::text(
                    crate::llm::types::LlmRole::System,
                    Arc::clone(&ctx.config.system_prompt),
                ),
            );
        }

        Arc::new(LlmRequest {
            model: ctx.config.model.clone(),
            messages,
            attachments,
            tools: if ctx.use_tools {
                Arc::clone(&ctx.config.tool_specs)
            } else {
                Arc::new(Vec::new())
            },
            tool_choice: if ctx.use_tools {
                LlmToolChoice::Auto
            } else {
                LlmToolChoice::None
            },
            model_variant: ctx.config.model_variant.clone(),
            generation: ctx.config.generation.clone(),
            session_id: ctx.config.run_session_id.clone(),
            output_spec: None,
            stream_events: None,
            provider_trace: None,
        })
    }
}

fn render_messages_for_projector(
    messages: &MessageSequence,
    turn_causes: &[TurnCause],
) -> crate::RenderedPrompt {
    if turn_causes.is_empty() {
        return messages.render_prompt();
    }

    let active_cause_ids = turn_causes
        .iter()
        .map(|cause| cause.id.as_str())
        .collect::<HashSet<_>>();
    let filtered = messages
        .iter()
        .filter(|message| {
            !(matches!(message.role, MessageRole::Event)
                && active_cause_ids.contains(message.id.as_str()))
        })
        .cloned()
        .collect::<Vec<_>>();
    render_prompt(filtered.as_slice())
}

pub trait ProtocolDriverHandle<M: TurnProtocol = UnitTurnProtocol>: Send + Sync {
    fn prepare_protocol_iteration(&self, ctx: DriverContextView<'_, M>) -> Vec<DriverAction<M>>;
    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_, M>,
        waiting: WaitingLlmState<M>,
        llm_response: LlmResponse,
        text_streamed: bool,
    ) -> Vec<DriverAction<M>>;
    fn handle_tool_results(
        &self,
        ctx: DriverContextView<'_, M>,
        completed: Vec<CompletedToolCall>,
    ) -> Vec<DriverAction<M>>;
    fn handle_exec_result(
        &self,
        ctx: DriverContextView<'_, M>,
        waiting: WaitingExecState<M>,
        result: Result<crate::ExecResponse, String>,
    ) -> Vec<DriverAction<M>>;
}

/// Configuration for a `TurnMachine` instance.
pub struct TurnMachineConfig<M: TurnProtocol = UnitTurnProtocol> {
    pub protocol_driver: Arc<dyn ProtocolDriverHandle<M>>,
    pub projector: Arc<dyn ContextProjector<M>>,
    pub sync_execution_surface: bool,
    pub model: String,
    pub max_turns: Option<usize>,
    pub model_variant: Option<String>,
    pub generation: crate::llm::types::GenerationOptions,
    pub run_session_id: Option<String>,
    pub autonomous: bool,
    pub tool_specs: Arc<Vec<LlmToolSpec>>,
    pub system_prompt: Arc<str>,
    pub session_id: String,
    pub emit_llm_trace: bool,
    pub termination: M::Termination,
    pub turn_limit_final_message: crate::TurnLimitFinalMessage,
}

// ─── Internal state ───

#[derive(Debug, Serialize, serde::Deserialize)]
enum MachineState<M: TurnProtocol = UnitTurnProtocol> {
    PreparingProtocol,
    WaitingExecutionSurface {
        effect_id: EffectId,
        update_machine_config: bool,
    },
    PrepareIteration,
    WaitingLlm {
        effect_id: EffectId,
        request: Arc<LlmRequest>,
        driver_state: Option<M::DriverState>,
    },
    WaitingTools {
        effect_id: EffectId,
        calls: Vec<PendingToolCall>,
    },
    WaitingExec {
        effect_id: EffectId,
        code: String,
        driver_state: M::DriverState,
    },
    WaitingCheckpoint {
        effect_id: EffectId,
        checkpoint: CheckpointKind,
        on_empty: CheckpointResumeAction,
    },
    Finished,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct TurnCheckpoint<M: TurnProtocol = UnitTurnProtocol> {
    state: MachineState<M>,
    pending_effects: Vec<Effect<M>>,
    next_effect_id: u64,
    #[serde(default)]
    next_synthetic_message_id: u64,
    messages: Vec<Message>,
    events: Vec<SessionEventRecord<M::Event>>,
    #[serde(default)]
    turn_causes: Vec<TurnCause>,
    #[serde(default)]
    progress_event_cursor: usize,
    protocol_iteration: usize,
    protocol_run_offset: usize,
    cumulative_usage: TokenUsage,
    termination: TurnTerminationPolicyState,
    synced_protocol_iteration: Option<usize>,
}

impl<M: TurnProtocol> Clone for MachineState<M> {
    fn clone(&self) -> Self {
        match self {
            Self::PreparingProtocol => Self::PreparingProtocol,
            Self::WaitingExecutionSurface {
                effect_id,
                update_machine_config,
            } => Self::WaitingExecutionSurface {
                effect_id: *effect_id,
                update_machine_config: *update_machine_config,
            },
            Self::PrepareIteration => Self::PrepareIteration,
            Self::WaitingLlm {
                effect_id,
                request,
                driver_state,
            } => Self::WaitingLlm {
                effect_id: *effect_id,
                request: Arc::clone(request),
                driver_state: driver_state.clone(),
            },
            Self::WaitingTools { effect_id, calls } => Self::WaitingTools {
                effect_id: *effect_id,
                calls: calls.clone(),
            },
            Self::WaitingExec {
                effect_id,
                code,
                driver_state,
            } => Self::WaitingExec {
                effect_id: *effect_id,
                code: code.clone(),
                driver_state: driver_state.clone(),
            },
            Self::WaitingCheckpoint {
                effect_id,
                checkpoint,
                on_empty,
            } => Self::WaitingCheckpoint {
                effect_id: *effect_id,
                checkpoint: *checkpoint,
                on_empty: on_empty.clone(),
            },
            Self::Finished => Self::Finished,
        }
    }
}

impl<M: TurnProtocol> MachineState<M> {
    fn outstanding_effect_id(&self) -> Option<EffectId> {
        match self {
            Self::WaitingExecutionSurface { effect_id, .. }
            | Self::WaitingLlm { effect_id, .. }
            | Self::WaitingTools { effect_id, .. }
            | Self::WaitingExec { effect_id, .. }
            | Self::WaitingCheckpoint { effect_id, .. } => Some(*effect_id),
            Self::PreparingProtocol | Self::PrepareIteration | Self::Finished => None,
        }
    }

    fn outstanding_effect(&self) -> Option<Effect<M>> {
        match self {
            Self::WaitingExecutionSurface {
                effect_id,
                update_machine_config,
            } => Some(Effect::SyncExecutionSurface {
                id: *effect_id,
                update_machine_config: *update_machine_config,
            }),
            Self::WaitingLlm {
                effect_id, request, ..
            } => Some(Effect::LlmCall {
                id: *effect_id,
                request: Arc::clone(request),
            }),
            Self::WaitingTools { effect_id, calls } => Some(Effect::ToolCalls {
                id: *effect_id,
                calls: calls.clone(),
            }),
            Self::WaitingExec {
                effect_id, code, ..
            } => Some(Effect::ExecCode {
                id: *effect_id,
                code: code.clone(),
            }),
            Self::WaitingCheckpoint {
                effect_id,
                checkpoint,
                ..
            } => Some(Effect::Checkpoint {
                id: *effect_id,
                checkpoint: *checkpoint,
            }),
            Self::PreparingProtocol | Self::PrepareIteration | Self::Finished => None,
        }
    }
}

/// Sans-IO state machine for a single session run (multi-turn).
pub struct TurnMachine<M: TurnProtocol = UnitTurnProtocol> {
    config: TurnMachineConfig<M>,
    state: MachineState<M>,
    pending_effects: VecDeque<Effect<M>>,
    active_effect_redelivery: bool,
    next_effect_id: u64,
    next_synthetic_message_id: u64,
    messages: MessageSequence,
    events: Arc<Vec<SessionEventRecord<M::Event>>>,
    turn_causes: Vec<TurnCause>,
    progress_event_cursor: usize,
    protocol_iteration: usize,
    protocol_run_offset: usize,
    cumulative_usage: TokenUsage,
    termination: TurnTerminationPolicyState,
    synced_protocol_iteration: Option<usize>,
}

impl<M: TurnProtocol> TurnMachine<M> {
    /// Create a new machine in `PrepareIteration` state.
    pub fn new(
        config: TurnMachineConfig<M>,
        messages: Vec<Message>,
        events: Arc<Vec<SessionEventRecord<M::Event>>>,
        protocol_run_offset: usize,
    ) -> Self {
        Self::new_shared(
            config,
            MessageSequence::from_owned(messages),
            events,
            protocol_run_offset,
        )
    }

    pub fn new_shared(
        config: TurnMachineConfig<M>,
        messages: MessageSequence,
        events: Arc<Vec<SessionEventRecord<M::Event>>>,
        protocol_run_offset: usize,
    ) -> Self {
        Self::new_shared_with_turn_causes(config, messages, events, protocol_run_offset, Vec::new())
    }

    pub fn new_shared_with_turn_causes(
        config: TurnMachineConfig<M>,
        messages: MessageSequence,
        events: Arc<Vec<SessionEventRecord<M::Event>>>,
        protocol_run_offset: usize,
        turn_causes: Vec<TurnCause>,
    ) -> Self {
        let next_synthetic_message_id = messages.len() as u64;
        Self {
            config,
            state: MachineState::PreparingProtocol,
            pending_effects: VecDeque::new(),
            active_effect_redelivery: false,
            next_effect_id: 1,
            next_synthetic_message_id,
            messages,
            progress_event_cursor: events.len(),
            events,
            turn_causes,
            protocol_iteration: protocol_run_offset,
            protocol_run_offset,
            cumulative_usage: TokenUsage::default(),
            termination: TurnTerminationPolicyState::new(),
            synced_protocol_iteration: None,
        }
    }

    /// Whether the machine has finished.
    pub fn is_done(&self) -> bool {
        matches!(self.state, MachineState::Finished)
    }

    pub fn messages(&self) -> Arc<Vec<Message>> {
        self.messages.shared()
    }

    pub fn events(&self) -> Arc<Vec<SessionEventRecord<M::Event>>> {
        Arc::clone(&self.events)
    }

    pub fn message_sequence(&self) -> MessageSequence {
        self.messages.clone()
    }

    pub fn protocol_iteration(&self) -> usize {
        self.protocol_iteration
    }

    pub fn checkpoint(&self) -> TurnCheckpoint<M> {
        let active_effect_id = self.state.outstanding_effect_id();
        let pending_effects = self
            .pending_effects
            .iter()
            .filter(|effect| active_effect_id.is_none_or(|id| effect.id() != Some(id)))
            .cloned()
            .collect::<Vec<_>>();
        TurnCheckpoint {
            state: self.state.clone(),
            pending_effects,
            next_effect_id: self.next_effect_id,
            next_synthetic_message_id: self.next_synthetic_message_id,
            messages: self.messages.iter().cloned().collect(),
            events: self.events.as_ref().clone(),
            turn_causes: self.turn_causes.clone(),
            progress_event_cursor: self.progress_event_cursor,
            protocol_iteration: self.protocol_iteration,
            protocol_run_offset: self.protocol_run_offset,
            cumulative_usage: self.cumulative_usage.clone(),
            termination: self.termination.clone(),
            synced_protocol_iteration: self.synced_protocol_iteration,
        }
    }

    pub fn restore_from_checkpoint(
        config: TurnMachineConfig<M>,
        checkpoint: TurnCheckpoint<M>,
    ) -> Self {
        let active_effect_id = checkpoint.state.outstanding_effect_id();
        let pending_effects = checkpoint
            .pending_effects
            .into_iter()
            .collect::<VecDeque<_>>();
        let active_effect_redelivery = active_effect_id.is_some()
            && !pending_effects
                .iter()
                .any(|effect| effect.id() == active_effect_id);
        Self {
            config,
            state: checkpoint.state,
            pending_effects,
            active_effect_redelivery,
            next_effect_id: checkpoint.next_effect_id,
            next_synthetic_message_id: checkpoint.next_synthetic_message_id,
            messages: MessageSequence::from_owned(checkpoint.messages),
            events: Arc::new(checkpoint.events),
            turn_causes: checkpoint.turn_causes,
            progress_event_cursor: checkpoint.progress_event_cursor,
            protocol_iteration: checkpoint.protocol_iteration,
            protocol_run_offset: checkpoint.protocol_run_offset,
            cumulative_usage: checkpoint.cumulative_usage,
            termination: checkpoint.termination,
            synced_protocol_iteration: checkpoint.synced_protocol_iteration,
        }
    }

    fn driver_context(&self) -> DriverContextView<'_, M> {
        DriverContextView {
            config: &self.config,
            messages: &self.messages,
            events: self.events.as_slice(),
            turn_causes: &self.turn_causes,
            protocol_iteration: self.protocol_iteration,
            protocol_run_offset: self.protocol_run_offset,
            termination: &self.termination,
        }
    }

    fn next_id(&mut self) -> EffectId {
        let id = EffectId(self.next_effect_id);
        self.next_effect_id += 1;
        id
    }

    fn next_synthetic_message_id(&mut self, scope: &str) -> String {
        let id = format!(
            "m_sansio_{}_{}_{}",
            self.protocol_run_offset, scope, self.next_synthetic_message_id
        );
        self.next_synthetic_message_id += 1;
        id
    }

    fn emit(&mut self, event: SessionEvent) {
        self.pending_effects.push_back(Effect::Emit(event));
    }

    fn emit_progress(&mut self) {
        let event_delta = self.next_event_delta();
        self.pending_effects.push_back(Effect::Progress {
            messages: self.messages.clone(),
            event_delta,
            protocol_iteration: self.protocol_iteration,
        });
    }

    pub fn fail_turn(&mut self, event: SessionEvent) {
        self.emit(event);
        self.finish(TurnOutcome::Stopped(TurnStop::RuntimeError));
    }

    pub fn finish_with_outcome(&mut self, outcome: TurnOutcome) {
        self.finish(outcome);
    }

    fn finish(&mut self, outcome: TurnOutcome) {
        self.emit(SessionEvent::TurnOutcome { outcome });
        self.emit(SessionEvent::Done);
        let msgs = std::mem::take(&mut self.messages);
        let event_delta = self.next_event_delta();
        let protocol_iteration = self.protocol_iteration;
        self.state = MachineState::Finished;
        self.pending_effects.push_back(Effect::Done {
            messages: msgs,
            event_delta,
            protocol_iteration,
        });
    }

    fn next_event_delta(&mut self) -> Vec<SessionEventRecord<M::Event>> {
        if self.progress_event_cursor >= self.events.len() {
            self.progress_event_cursor = self.events.len();
            return Vec::new();
        }
        let delta = self.events[self.progress_event_cursor..].to_vec();
        self.progress_event_cursor = self.events.len();
        delta
    }

    /// Drain the next pending effect. Returns `None` when the host must call
    /// `handle_response()` before more effects become available.
    pub fn poll_effect(&mut self) -> Option<Effect<M>> {
        if let Some(effect) = self.pending_effects.pop_front() {
            return Some(effect);
        }
        if self.active_effect_redelivery {
            self.active_effect_redelivery = false;
            if let Some(effect) = self.state.outstanding_effect() {
                return Some(effect);
            }
        }

        match &self.state {
            MachineState::PreparingProtocol => {
                self.prepare_protocol();
                self.pending_effects.pop_front()
            }
            MachineState::PrepareIteration => {
                self.prepare_protocol_iteration();
                self.pending_effects.pop_front()
            }
            _ => None,
        }
    }

    // ─── State transitions ───

    fn prepare_protocol(&mut self) {
        if self.config.sync_execution_surface {
            let id = self.next_id();
            self.state = MachineState::WaitingExecutionSurface {
                effect_id: id,
                update_machine_config: false,
            };
            self.pending_effects
                .push_back(Effect::SyncExecutionSurface {
                    id,
                    update_machine_config: false,
                });
            return;
        }

        self.prepare_protocol_iteration();
    }

    fn prepare_protocol_iteration(&mut self) {
        if self.config.sync_execution_surface
            && self.synced_protocol_iteration != Some(self.protocol_iteration)
        {
            let id = self.next_id();
            self.state = MachineState::WaitingExecutionSurface {
                effect_id: id,
                update_machine_config: true,
            };
            self.pending_effects
                .push_back(Effect::SyncExecutionSurface {
                    id,
                    update_machine_config: true,
                });
            return;
        }
        let actions = {
            let driver = Arc::clone(&self.config.protocol_driver);
            let ctx = self.driver_context();
            driver.prepare_protocol_iteration(ctx)
        };
        self.apply_actions(actions);
    }

    fn start_llm_request(
        &mut self,
        request: Arc<LlmRequest>,
        driver_state: Option<M::DriverState>,
    ) {
        let tool_list = self
            .config
            .tool_specs
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        self.emit(SessionEvent::LlmRequest {
            protocol_iteration: self.protocol_iteration,
            message_count: self.messages.len(),
            tool_list,
        });

        let id = self.next_id();
        self.state = MachineState::WaitingLlm {
            effect_id: id,
            request: Arc::clone(&request),
            driver_state,
        };
        self.pending_effects
            .push_back(Effect::LlmCall { id, request });
    }

    fn start_tool_calls(&mut self, calls: Vec<PendingToolCall>) {
        let effect_id = self.next_id();
        self.state = MachineState::WaitingTools {
            effect_id,
            calls: calls.clone(),
        };
        self.pending_effects.push_back(Effect::ToolCalls {
            id: effect_id,
            calls,
        });
    }

    fn start_exec(&mut self, code: String, driver_state: M::DriverState) {
        let effect_id = self.next_id();
        self.state = MachineState::WaitingExec {
            effect_id,
            code: code.clone(),
            driver_state,
        };
        self.pending_effects.push_back(Effect::ExecCode {
            id: effect_id,
            code,
        });
    }

    fn schedule_turn_limit_final(&mut self, message: Message) -> bool {
        let Some(_max_turns) = self.termination.turn_limit_final_to_schedule(
            self.protocol_iteration,
            self.protocol_run_offset,
            self.config.max_turns,
        ) else {
            return false;
        };
        self.termination.mark_turn_limit_final_scheduled();
        self.messages.push(message);
        true
    }

    fn schedule_configured_turn_limit_final(&mut self) -> bool {
        let Some(max_turns) = self.termination.turn_limit_final_to_schedule(
            self.protocol_iteration,
            self.protocol_run_offset,
            self.config.max_turns,
        ) else {
            return false;
        };
        let message_id = self.next_synthetic_message_id("turn_limit");
        let message = (self.config.turn_limit_final_message)(message_id, max_turns);
        self.termination.mark_turn_limit_final_scheduled();
        self.messages.push(message);
        true
    }

    fn append_event(&mut self, event: SessionEventRecord<M::Event>) {
        match event {
            SessionEventRecord::Conversation(record) => {
                Arc::make_mut(&mut self.events)
                    .push(SessionEventRecord::Conversation(record.clone()));
                self.messages.push(record.to_message());
            }
            SessionEventRecord::Tool(ToolEvent::Invocation { stable_key, record }) => {
                Arc::make_mut(&mut self.events).push(SessionEventRecord::Tool(
                    ToolEvent::Invocation { stable_key, record },
                ));
            }
            SessionEventRecord::Protocol(protocol_event) => {
                Arc::make_mut(&mut self.events).push(SessionEventRecord::Protocol(protocol_event));
            }
        }
    }

    pub fn apply_actions(&mut self, actions: Vec<DriverAction<M>>) {
        let mut progress_dirty = false;
        for action in actions {
            match action {
                DriverAction::Emit(event) => self.emit(event),
                DriverAction::AppendEvents(events) => {
                    if !events.is_empty() {
                        for event in events {
                            self.append_event(event);
                        }
                        progress_dirty = true;
                    }
                }
                DriverAction::StartLlm {
                    request,
                    driver_state,
                } => self.start_llm_request(request, driver_state),
                DriverAction::StartTools { calls } => self.start_tool_calls(calls),
                DriverAction::StartExec { code, driver_state } => {
                    self.start_exec(code, driver_state)
                }
                DriverAction::StartCheckpoint {
                    checkpoint,
                    on_empty,
                } => self.request_checkpoint(checkpoint, on_empty),
                DriverAction::AdvanceProtocolIteration => {
                    self.protocol_iteration += 1;
                    self.synced_protocol_iteration = None;
                    progress_dirty = true;
                }
                DriverAction::ScheduleTurnLimitFinal { message } => {
                    if self.schedule_turn_limit_final(message) {
                        progress_dirty = true;
                    }
                }
                DriverAction::Finish(outcome) => {
                    if progress_dirty {
                        self.emit_progress();
                        progress_dirty = false;
                    }
                    self.finish(outcome);
                    break;
                }
            }
        }
        if progress_dirty {
            self.emit_progress();
        }
    }

    /// Feed a response to a previously emitted effect.
    pub fn handle_response(&mut self, response: Response) {
        self.active_effect_redelivery = false;
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
            Response::Checkpoint { id, delivery } => self.handle_checkpoint(id, delivery),
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

    fn handle_execution_surface_synced(
        &mut self,
        id: EffectId,
        result: Result<Option<ExecutionSurfaceSync>, String>,
    ) {
        let (waiting_id, waiting_update_machine_config) =
            match std::mem::replace(&mut self.state, MachineState::Finished) {
                MachineState::WaitingExecutionSurface {
                    effect_id,
                    update_machine_config,
                } => (effect_id, update_machine_config),
                other => {
                    self.state = other;
                    return;
                }
            };
        if waiting_id != id {
            self.state = MachineState::WaitingExecutionSurface {
                effect_id: waiting_id,
                update_machine_config: waiting_update_machine_config,
            };
            return;
        }

        match result {
            Ok(update) => {
                if let Some(update) = update {
                    self.config.system_prompt = update.system_prompt;
                    self.config.tool_specs = update.tool_specs;
                }
                self.synced_protocol_iteration = Some(self.protocol_iteration);
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
        let mut appended = Vec::new();
        for message in plugin_messages
            .iter()
            .filter(|message| matches!(message.role, MessageRole::User | MessageRole::System))
        {
            let message_id = self.next_synthetic_message_id("checkpoint");
            let mut parts = if message.parts.is_empty() {
                vec![Part {
                    id: format!("{message_id}.p0"),
                    kind: PartKind::Text,
                    content: message.content.clone(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                }]
            } else {
                message.parts.clone()
            };
            reassign_part_ids(&message_id, &mut parts);
            appended.push(Message {
                id: message_id.clone(),
                role: message.role,
                parts: Arc::new(parts),
                origin: message.origin.clone().or_else(|| {
                    Some(MessageOrigin::Plugin {
                        plugin_id: "plugin".to_string(),
                        transient,
                    })
                }),
            });
        }
        if !appended.is_empty() {
            self.messages.extend(appended);
        }
    }

    fn append_turn_causes(&mut self, causes: Vec<TurnCause>) {
        if causes.is_empty() {
            return;
        }
        let mut existing_ids = self
            .turn_causes
            .iter()
            .map(|cause| cause.id.clone())
            .collect::<HashSet<_>>();
        for cause in causes {
            if !existing_ids.insert(cause.id.clone()) {
                continue;
            }
            self.messages.push(cause.to_event_message());
            self.turn_causes.push(cause);
        }
    }

    fn handle_checkpoint(&mut self, id: EffectId, delivery: CheckpointDelivery) {
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

        if !delivery.messages.is_empty()
            || !delivery.transient_messages.is_empty()
            || !delivery.turn_causes.is_empty()
        {
            self.append_checkpoint_messages(&delivery.messages, false);
            self.append_checkpoint_messages(&delivery.transient_messages, true);
            self.append_turn_causes(delivery.turn_causes);
            if matches!(checkpoint, CheckpointKind::BeforeCompletion) {
                self.protocol_iteration += 1;
                if self.termination.should_force_exit_after_grace_turn() {
                    self.emit_progress();
                    self.finish(TurnOutcome::Stopped(TurnStop::MaxTurns));
                    return;
                }
                self.schedule_configured_turn_limit_final();
            }
            self.state = MachineState::PrepareIteration;
            self.emit_progress();
            return;
        }

        match on_empty {
            CheckpointResumeAction::PrepareIteration => {
                self.state = MachineState::PrepareIteration;
            }
            CheckpointResumeAction::Finish(outcome) => self.finish(outcome),
        }
    }

    fn take_waiting_llm_state(&mut self, id: EffectId) -> Option<WaitingLlmState<M>> {
        match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingLlm {
                effect_id,
                request,
                driver_state,
            } if effect_id == id => Some(WaitingLlmState {
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
                self.emit_llm_error(error);
            }
            Ok(llm_response) => {
                self.record_llm_usage(&llm_response, self.llm_response_text(&llm_response));
                if self.handle_terminal_llm_response(&llm_response, text_streamed) {
                    return;
                }
                let actions = {
                    let driver = Arc::clone(&self.config.protocol_driver);
                    let ctx = self.driver_context();
                    driver.handle_llm_success(ctx, waiting, llm_response, text_streamed)
                };
                self.apply_actions(actions);
            }
        }
    }

    fn handle_terminal_llm_response(
        &mut self,
        llm_response: &LlmResponse,
        text_streamed: bool,
    ) -> bool {
        let outcome = match llm_response.terminal_reason {
            LlmTerminalReason::OutputLimit => TurnOutcome::Stopped(TurnStop::Incomplete),
            LlmTerminalReason::ContextOverflow => TurnOutcome::Stopped(TurnStop::ProviderError),
            LlmTerminalReason::ContentFilter => TurnOutcome::Stopped(TurnStop::ProviderError),
            LlmTerminalReason::ProviderError => TurnOutcome::Stopped(TurnStop::ProviderError),
            LlmTerminalReason::Cancelled => TurnOutcome::Stopped(TurnStop::Cancelled),
            LlmTerminalReason::Stop | LlmTerminalReason::ToolUse | LlmTerminalReason::Unknown => {
                return false;
            }
        };

        if !text_streamed && !llm_response.full_text.is_empty() {
            self.emit(SessionEvent::TextDelta {
                content: llm_response.full_text.clone(),
            });
        }
        self.emit(SessionEvent::LlmResponse {
            protocol_iteration: self.protocol_iteration,
            content: llm_response.full_text.clone(),
            duration_ms: 0,
        });
        let reason = llm_response.terminal_reason;
        let diagnostic = llm_response
            .terminal_diagnostic
            .clone()
            .unwrap_or_else(|| format!("Model call ended with terminal reason {reason:?}."));
        self.emit(SessionEvent::Error {
            message: diagnostic.clone(),
            envelope: Some(crate::session_model::make_error_envelope(
                "llm_provider",
                Some(reason.code()),
                Some(reason),
                diagnostic,
                None,
            )),
        });
        self.finish(outcome);
        true
    }

    fn llm_response_text<'a>(&self, llm_response: &'a LlmResponse) -> &'a str {
        &llm_response.full_text
    }

    fn llm_response_debug_parts(&self, llm_response: &LlmResponse) -> Option<Value> {
        let parts = llm_response
            .parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text, .. } if !text.is_empty() => Some(serde_json::json!({
                    "type": "text",
                    "text": text,
                })),
                LlmOutputPart::Text { .. } => None,
                LlmOutputPart::Reasoning {
                    text,
                    replay,
                } => Some(serde_json::json!({
                    "type": "reasoning",
                    "id": replay.as_ref().and_then(|meta| meta.item_id.as_ref()),
                    "summary": replay.as_ref().map(|meta| &meta.summary),
                    "text": text,
                    "has_encrypted": replay.as_ref().is_some_and(|meta| meta.encrypted_content.is_some() || meta.signature.is_some()),
                    "redacted": replay.as_ref().is_some_and(|meta| meta.redacted),
                })),
                LlmOutputPart::ToolCall {
                    call_id,
                    tool_name,
                    input_json,
                    replay,
                } => Some(serde_json::json!({
                    "type": "tool_call",
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "input_json": input_json,
                    "id": replay.as_ref().and_then(|meta| meta.item_id.as_ref()),
                    "has_opaque": replay.as_ref().is_some_and(|meta| meta.opaque.is_some()),
                })),
            })
            .collect::<Vec<_>>();
        (!parts.is_empty()).then_some(Value::Array(parts))
    }

    fn record_llm_usage(&mut self, llm_response: &LlmResponse, response_text: &str) {
        let usage = token_usage_from_llm_usage(&llm_response.usage);
        self.cumulative_usage.add(&usage);
        self.emit(SessionEvent::TokenUsage {
            protocol_iteration: self.protocol_iteration,
            usage: usage.clone(),
            cumulative: self.cumulative_usage.clone(),
        });
        if self.config.emit_llm_trace {
            let response_parts = self.llm_response_debug_parts(llm_response);
            self.pending_effects.push_back(Effect::Log {
                event: LogEvent::LlmDebug {
                    session_id: self.config.session_id.clone(),
                    protocol_iteration: self.protocol_iteration,
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
        if self.config.emit_llm_trace {
            self.pending_effects.push_back(Effect::Log {
                event: LogEvent::LlmError {
                    session_id: self.config.session_id.clone(),
                    protocol_iteration: self.protocol_iteration,
                    request_body: error.request_body.clone(),
                    message: error.message.clone(),
                    retryable: error.retryable,
                    raw: error.raw.clone(),
                    code: error.code.clone(),
                    terminal_reason: error.terminal_reason,
                },
            });
        }
    }

    fn emit_llm_error(&mut self, error: LlmCallError) {
        self.record_llm_error(&error);
        self.emit(SessionEvent::Error {
            message: format!("LLM error: {}", error.message),
            envelope: Some(crate::session_model::make_error_envelope(
                "llm_provider",
                error.code.as_deref(),
                Some(error.terminal_reason),
                format!("LLM error: {}", error.message),
                error.raw,
            )),
        });
        self.finish(TurnOutcome::Stopped(TurnStop::ProviderError));
    }

    fn handle_tool_results(&mut self, id: EffectId, completed: Vec<CompletedToolCall>) {
        let (waiting_effect_id, waiting_calls) =
            match std::mem::replace(&mut self.state, MachineState::Finished) {
                MachineState::WaitingTools { effect_id, calls } => (effect_id, calls),
                other => {
                    self.state = other;
                    return;
                }
            };

        if waiting_effect_id != id {
            self.state = MachineState::WaitingTools {
                effect_id: waiting_effect_id,
                calls: waiting_calls,
            };
            return;
        }

        for outcome in &completed {
            self.emit(SessionEvent::ToolCall {
                call_id: Some(outcome.call_id.clone()),
                name: outcome.tool_name.clone(),
                args: outcome.args.clone(),
                output: outcome.output.clone(),
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

    fn take_waiting_exec_state(&mut self, id: EffectId) -> Option<WaitingExecState<M>> {
        match std::mem::replace(&mut self.state, MachineState::Finished) {
            MachineState::WaitingExec {
                effect_id,
                code: _,
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
