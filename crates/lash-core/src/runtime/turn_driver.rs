use super::*;

mod context;
mod effects;
mod events;
mod failures;
mod handlers;
mod lease;
mod machine;
mod streaming;
mod surface;
mod tools;
mod trace;

pub(super) use events::{emit_semantic_response_parts, send_session_event, send_turn_activity};
pub(super) use trace::protocol_step_trace_event;

pub(super) struct RuntimeTurnDriver<'a> {
    pub(super) session: Session,
    pub(super) policy: SessionPolicy,
    pub(super) host: RuntimeHost,
    pub(super) effect_scope: RuntimeEffectControllerScope<'a>,
    pub(super) session_id: String,
    pub(super) turn_id: String,
    pub(super) turn_index: usize,
    pub(super) turn_pipeline: TurnCommitPipeline,
    pub(super) llm_stream_summaries: HashMap<usize, LlmStreamSummary>,
    pub(super) next_llm_ordinal: usize,
    pub(super) session_manager: Arc<RuntimeSessionManager>,
    pub(super) protocol_turn_options: crate::ProtocolTurnOptions,
    pub(super) protocol_extension: Option<crate::ProtocolTurnExtensionHandle>,
    pub(super) turn_context: crate::TurnContext,
    pub(super) turn_causes: Vec<crate::TurnCause>,
    pub(super) pending_process_wake_acks: Vec<String>,
    pub(super) turn_lease: Option<crate::RuntimeTurnLease>,
    pub(super) machine_config_snapshot: Option<crate::RuntimeTurnMachineConfigSnapshot>,
    pub(super) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}
