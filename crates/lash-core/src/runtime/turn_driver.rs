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

pub(in crate::runtime) use crate::runtime::turn_loop::{
    queued_work_trace_payload, send_queued_work_started_event,
};
pub(super) use events::{emit_semantic_response_parts, send_session_event, send_turn_activity};
pub(super) use trace::protocol_step_trace_event;

pub(super) struct RuntimeTurnDriver<'a> {
    pub(super) session: Session,
    pub(super) policy: RuntimeSessionPolicy,
    pub(super) host: RuntimeHost,
    pub(super) scoped_effect_controller: ScopedEffectController<'a>,
    pub(super) session_id: String,
    pub(super) turn_id: String,
    pub(super) turn_index: usize,
    pub(super) turn_pipeline: TurnBoundary,
    pub(super) llm_stream_summaries: HashMap<usize, LlmStreamSummary>,
    pub(super) next_llm_ordinal: usize,
    pub(super) session_services: Arc<RuntimeSessionServices>,
    pub(super) protocol_turn_options: crate::ProtocolTurnOptions,
    pub(super) protocol_extension: Option<crate::ProtocolTurnExtensionHandle>,
    pub(super) turn_context: crate::TurnContext,
    pub(super) turn_causes: Vec<crate::TurnCause>,
    pub(super) pending_queue_claims: Vec<crate::QueuedWorkClaim>,
    pub(super) checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer,
    pub(super) pending_tool_host_events: Vec<crate::tool_dispatch::ToolHostEventEffectOutcome>,
    pub(super) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}
