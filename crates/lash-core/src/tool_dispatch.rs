mod context;
mod directives;
mod execution;
mod preparation;
mod retry;
mod scheduling;
#[cfg(test)]
mod tests;

pub use context::{ToolDispatchContext, ToolTriggerEffectOutcome};
#[cfg(test)]
pub(crate) use scheduling::{ParallelToolCallSpec, dispatch_parallel_tool_calls};

pub(crate) use context::{
    CheckpointMessageBuffer, PendingToolDispatchOutcome, ToolCallLaunch, ToolDispatchOutcome,
    ToolPreparationOutcome, ToolTriggerOutcomeBuffer,
};
pub(crate) use execution::{
    dispatch_granted_prepared_tool_attempt_launch_with_execution_context,
    dispatch_granted_prepared_tool_call_launch_with_execution_context,
    dispatch_prepared_tool_attempt_launch_with_execution_context,
    dispatch_prepared_tool_call_launch_with_execution_context,
    finalize_tool_result_with_execution_context,
};
#[cfg(test)]
pub(crate) use preparation::dispatch_tool_call;
#[cfg(test)]
pub(crate) use preparation::dispatch_tool_call_with_execution_context;
pub(crate) use preparation::{
    prepare_granted_tool_call_with_context, prepare_tool_call_with_context,
    resolve_callable_manifest, resolve_callable_manifest_by_id,
    resolve_tool_argument_projection_policy,
};
pub(crate) use retry::{mark_retry_exhausted, retry_after_ms};
pub(crate) use scheduling::{resolve_tool_scheduling, schedule_tool_batch};
