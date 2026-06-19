use std::sync::Arc;

use crate::plugin::ToolResultHookContext;
use crate::{PreparedToolCall, ProgressSender, ToolContext, ToolFailureClass, ToolResult};

#[cfg(test)]
use super::context::ToolDispatchOutcome;
use super::context::{
    PendingToolDispatchOutcome, ToolCallLaunch, ToolDispatchContext, launch_done, outcome,
    runtime_failure,
};
use super::directives::apply_after_tool_directives;
use super::retry::execute_tool_call;

#[cfg(test)]
pub(crate) async fn dispatch_prepared_tool_call_with_execution_context<'run>(
    context: &ToolDispatchContext<'run>,
    prepared: PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolDispatchOutcome {
    dispatch_prepared_tool_call_launch_with_execution_context(
        context,
        prepared,
        progress,
        tool_context,
    )
    .await
    .into_done_or_runtime_failure()
}

pub(crate) async fn dispatch_prepared_tool_call_launch_with_execution_context<'run>(
    context: &ToolDispatchContext<'run>,
    prepared: PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolCallLaunch {
    let prepared_tool_name = prepared.tool_name.clone();
    let args = prepared.args.clone();
    let Some(manifest) =
        super::preparation::resolve_callable_manifest_by_id(context, &prepared.tool_id)
    else {
        return launch_done(outcome(
            prepared_tool_name,
            args,
            runtime_failure(
                ToolFailureClass::Unavailable,
                "tool_unavailable",
                "Tool is unavailable in this session",
            ),
            0,
        ));
    };
    let tool_name = manifest.name.clone();

    let tool_start = context.clock.now();
    let tool_context = tool_context.with_prepared_payload(prepared.prepared_payload.clone());
    let completion_context = tool_context.clone();
    let result = Box::pin(execute_tool_call(
        context,
        &manifest,
        &prepared,
        progress,
        tool_context,
    ))
    .await;
    let duration_ms = context.clock.now().duration_since(tool_start).as_millis() as u64;
    let result = match result {
        ToolResult::Done(_) => result,
        ToolResult::Pending(pending) => {
            let key = match completion_context.take_completion_key() {
                Ok(Some(key)) => key,
                Ok(None) => {
                    return launch_done(outcome(
                        tool_name,
                        args,
                        runtime_failure(
                            ToolFailureClass::Internal,
                            "pending_tool_missing_completion_key",
                            "tool returned Pending without first obtaining a completion key",
                        ),
                        duration_ms,
                    ));
                }
                Err(err) => {
                    return launch_done(outcome(
                        tool_name,
                        args,
                        runtime_failure(
                            ToolFailureClass::Internal,
                            "pending_tool_completion_key_failed",
                            err.to_string(),
                        ),
                        duration_ms,
                    ));
                }
            };
            return ToolCallLaunch::Pending(PendingToolDispatchOutcome {
                tool_name,
                args,
                key,
                pending,
                duration_ms,
            });
        }
    };

    let result = finalize_tool_result_with_execution_context(
        context,
        &tool_name,
        &args,
        result,
        duration_ms,
    )
    .await;

    launch_done(outcome(tool_name, args, result, duration_ms))
}

pub(crate) async fn finalize_tool_result_with_execution_context(
    context: &ToolDispatchContext<'_>,
    tool_name: &str,
    args: &serde_json::Value,
    result: ToolResult,
    duration_ms: u64,
) -> ToolResult {
    match context
        .plugins
        .after_tool_call(ToolResultHookContext::new(
            context.session_id.clone(),
            tool_name.to_string(),
            args.clone(),
            result.clone(),
            duration_ms,
            context.turn_context.clone(),
            Arc::clone(&context.sessions),
        ))
        .await
    {
        Ok(directives) => apply_after_tool_directives(context, result, directives).await,
        Err(err) => runtime_failure(
            ToolFailureClass::Internal,
            "after_tool_call_failed",
            err.to_string(),
        ),
    }
}

#[cfg(test)]
trait ToolCallLaunchExt {
    fn into_done_or_runtime_failure(self) -> ToolDispatchOutcome;
}

#[cfg(test)]
impl ToolCallLaunchExt for ToolCallLaunch {
    fn into_done_or_runtime_failure(self) -> ToolDispatchOutcome {
        match self {
            ToolCallLaunch::Done(outcome) => outcome,
            ToolCallLaunch::Pending(pending) => outcome(
                pending.tool_name,
                pending.args,
                runtime_failure(
                    ToolFailureClass::Internal,
                    "pending_tool_not_supported_here",
                    "pending tool completion is not supported on this dispatch path",
                ),
                pending.duration_ms,
            ),
        }
    }
}
