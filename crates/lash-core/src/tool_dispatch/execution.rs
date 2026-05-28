use std::sync::Arc;
use std::time::Instant;

use crate::plugin::ToolResultHookContext;
use crate::{PreparedToolCall, ProgressSender, ToolContext, ToolFailureClass};

use super::context::{ToolDispatchContext, ToolDispatchOutcome, outcome, runtime_failure};
use super::directives::apply_after_tool_directives;
use super::preparation::resolve_callable_manifest;
use super::retry::execute_tool_call;

pub(crate) async fn dispatch_prepared_tool_call_with_execution_context<'run>(
    context: &ToolDispatchContext<'run>,
    prepared: PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolDispatchOutcome {
    let tool_name = prepared.tool_name.clone();
    let args = prepared.args.clone();
    let Some(manifest) = resolve_callable_manifest(context, &tool_name) else {
        return outcome(
            tool_name,
            args,
            runtime_failure(
                ToolFailureClass::Unavailable,
                "tool_unavailable",
                "Tool is unavailable in this session",
            ),
            0,
        );
    };

    let tool_start = Instant::now();
    let result = execute_tool_call(
        context,
        &manifest,
        &prepared,
        progress,
        tool_context.with_prepared_payload(prepared.prepared_payload.clone()),
    )
    .await;
    let duration_ms = tool_start.elapsed().as_millis() as u64;

    let result = match context
        .plugins
        .after_tool_call(ToolResultHookContext::new(
            context.session_id.clone(),
            tool_name.clone(),
            args.clone(),
            result.clone(),
            duration_ms,
            context.turn_context.clone(),
            Arc::clone(&context.host),
        ))
        .await
    {
        Ok(directives) => apply_after_tool_directives(context, result, directives).await,
        Err(err) => runtime_failure(
            ToolFailureClass::Internal,
            "after_tool_call_failed",
            err.to_string(),
        ),
    };

    outcome(tool_name, args, result, duration_ms)
}
