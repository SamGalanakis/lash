use crate::{
    PreparedToolCall, ProgressSender, ToolCallOutcome, ToolContext, ToolManifest, ToolResult,
    ToolRetryDisposition, ToolRetryPolicy,
};

use super::context::ToolDispatchContext;

pub(super) async fn execute_tool_attempt<'run>(
    context: &ToolDispatchContext<'run>,
    manifest: &ToolManifest,
    prepared: &PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
    attempt: u32,
    max_attempts: u32,
) -> ToolResult {
    let tool_name = manifest.name.as_str();
    execute_once(
        context,
        prepared,
        progress,
        tool_context.with_retry_context(tool_name, attempt, max_attempts),
    )
    .await
}

pub(super) async fn execute_granted_tool_attempt<'run>(
    context: &ToolDispatchContext<'run>,
    grant: &crate::ToolExecutionGrant,
    prepared: &PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
    attempt: u32,
    max_attempts: u32,
) -> ToolResult {
    let tool_name = grant.manifest.name.as_str();
    execute_granted_once(
        context,
        grant,
        prepared,
        progress,
        tool_context.with_retry_context(tool_name, attempt, max_attempts),
    )
    .await
}

async fn execute_once<'run>(
    context: &ToolDispatchContext<'run>,
    prepared: &PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolResult {
    let args = &prepared.args;
    context
        .tools
        .execute_by_id(&prepared.tool_id, args, &tool_context, progress)
        .await
}

async fn execute_granted_once<'run>(
    context: &ToolDispatchContext<'run>,
    grant: &crate::ToolExecutionGrant,
    prepared: &PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolResult {
    context
        .tools
        .execute_granted(grant, &prepared.args, &tool_context, progress)
        .await
}

pub(crate) fn retry_after_ms(
    result: &ToolResult,
    retry_policy: ToolRetryPolicy,
    retry_index: u32,
) -> Option<u64> {
    if matches!(retry_policy, ToolRetryPolicy::Never) {
        return None;
    }
    let output = result.as_done_output()?;
    let ToolCallOutcome::Failure(failure) = &output.outcome else {
        return None;
    };
    let ToolRetryDisposition::Safe { after_ms } = &failure.retry else {
        return None;
    };
    Some(retry_policy.delay_ms_for_retry(retry_index, *after_ms))
}

pub(crate) fn mark_retry_exhausted(result: ToolResult, attempts: u32) -> ToolResult {
    let mut output = match result.into_done_output() {
        Ok(output) => output,
        Err(pending) => return ToolResult::pending(pending),
    };
    if let ToolCallOutcome::Failure(failure) = &mut output.outcome {
        failure.retry = ToolRetryDisposition::Exhausted { attempts };
    }
    ToolResult::from_output(output)
}
