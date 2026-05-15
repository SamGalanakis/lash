use crate::tool_dispatch::ToolDispatchContext;
use crate::{
    ProgressSender, ToolCall, ToolCallOutcome, ToolContext, ToolManifest, ToolResult,
    ToolRetryDisposition, ToolRetryPolicy,
};

/// Runtime-owned execution policy for a validated tool call.
///
/// Dispatch is responsible for visibility, plugin hooks, contract resolution,
/// and input validation. This executor owns the actual native/provider
/// invocation loop: attempt context, retry policy, delay, idempotency key
/// requirements, and final exhaustion marking.
pub(crate) async fn execute_tool_call(
    context: &ToolDispatchContext,
    manifest: &ToolManifest,
    tool_name: &str,
    args: &serde_json::Value,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext,
) -> ToolResult {
    let retry_policy = manifest.retry_policy;
    let max_attempts = retry_policy.max_attempts();
    let idempotency_key = derive_idempotency_key(&tool_context, tool_name);
    if retry_policy.requires_idempotency_key() && idempotency_key.is_none() {
        return execute_once(
            context,
            tool_name,
            args,
            progress,
            tool_context.with_retry_context(tool_name, 1, 1),
        )
        .await;
    }

    let max_attempts = max_attempts.max(1);
    for attempt in 1..=max_attempts {
        let attempt_context =
            tool_context
                .clone()
                .with_retry_context(tool_name, attempt, max_attempts);
        let result = execute_once(context, tool_name, args, progress, attempt_context).await;
        let retry_after_ms = retry_after_ms(&result, retry_policy, attempt - 1);
        let Some(retry_after_ms) = retry_after_ms else {
            return result;
        };
        if attempt >= max_attempts {
            return mark_retry_exhausted(result, attempt);
        }
        if retry_after_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(retry_after_ms)).await;
        }
    }

    execute_once(
        context,
        tool_name,
        args,
        progress,
        tool_context.with_retry_context(tool_name, 1, 1),
    )
    .await
}

async fn execute_once(
    context: &ToolDispatchContext,
    tool_name: &str,
    args: &serde_json::Value,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext,
) -> ToolResult {
    let native_tools = context.plugins.mode_native_tools().to_vec();
    for provider in native_tools {
        if let Some(result) = provider.execute(context, tool_name, args, progress).await {
            return result;
        }
    }

    context
        .tools
        .execute(ToolCall {
            name: tool_name,
            args,
            context: &tool_context,
            progress,
        })
        .await
}

fn derive_idempotency_key(tool_context: &ToolContext, tool_name: &str) -> Option<String> {
    tool_context.tool_call_id().map(|call_id| {
        format!(
            "lash-tool:{}:{call_id}:{tool_name}",
            tool_context.session_id()
        )
    })
}

fn retry_after_ms(
    result: &ToolResult,
    retry_policy: ToolRetryPolicy,
    retry_index: u32,
) -> Option<u64> {
    if matches!(retry_policy, ToolRetryPolicy::Never) {
        return None;
    }
    let ToolCallOutcome::Failure(failure) = &result.output.outcome else {
        return None;
    };
    let ToolRetryDisposition::Safe { after_ms } = &failure.retry else {
        return None;
    };
    Some(retry_policy.delay_ms_for_retry(retry_index, *after_ms))
}

fn mark_retry_exhausted(mut result: ToolResult, attempts: u32) -> ToolResult {
    if let ToolCallOutcome::Failure(failure) = &mut result.output.outcome {
        failure.retry = ToolRetryDisposition::Exhausted { attempts };
    }
    result
}
