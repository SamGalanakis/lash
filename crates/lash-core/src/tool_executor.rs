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
pub(crate) async fn execute_tool_call<'run>(
    context: &ToolDispatchContext<'run>,
    manifest: &ToolManifest,
    tool_name: &str,
    args: &serde_json::Value,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
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
        if retry_after_ms > 0
            && let Err(err) =
                sleep_before_retry(context, &tool_context, tool_name, attempt, retry_after_ms).await
        {
            return ToolResult::failure(crate::ToolFailure::runtime(
                crate::ToolFailureClass::Internal,
                "tool_retry_sleep_failed",
                format!("retry sleep for tool `{tool_name}` failed after attempt {attempt}: {err}"),
            ));
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

async fn execute_once<'run>(
    context: &ToolDispatchContext<'run>,
    tool_name: &str,
    args: &serde_json::Value,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
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

async fn sleep_before_retry(
    context: &ToolDispatchContext<'_>,
    tool_context: &ToolContext<'_>,
    tool_name: &str,
    attempt: u32,
    retry_after_ms: u64,
) -> Result<(), crate::RuntimeEffectControllerError> {
    let duration = std::time::Duration::from_millis(retry_after_ms);
    let cancellation = tool_context
        .cancellation_token()
        .cloned()
        .unwrap_or_else(tokio_util::sync::CancellationToken::new);
    let metadata = if let Some(parent) = tool_context.tool_effect_metadata.as_ref() {
        crate::runtime::tool_retry_sleep_metadata(parent, tool_name, attempt)
    } else {
        let idempotency_base =
            derive_idempotency_key(tool_context, tool_name).unwrap_or_else(|| {
                format!(
                    "lash-tool:{}:unknown:{tool_name}",
                    tool_context.session_id()
                )
            });
        let effect_id = format!("{idempotency_base}:attempt:{attempt}:sleep");
        crate::EffectInvocationMetadata {
            session_id: tool_context.session_id().to_string(),
            origin: crate::EffectOrigin::Turn,
            turn_id: None,
            turn_index: None,
            mode_iteration: None,
            effect_id: effect_id.clone(),
            effect_kind: crate::RuntimeEffectKind::Sleep,
            idempotency_key: effect_id,
            turn_checkpoint_hash: None,
        }
    };
    let outcome = context
        .effect_controller
        .as_controller()
        .execute_effect(
            crate::RuntimeEffectEnvelope::new(
                metadata,
                crate::RuntimeEffectCommand::Sleep {
                    duration_ms: duration.as_millis().try_into().unwrap_or(u64::MAX),
                },
            ),
            crate::RuntimeEffectLocalExecutor::sleep(cancellation),
        )
        .await?;
    match outcome {
        crate::RuntimeEffectOutcome::Sleep => Ok(()),
        other => Err(crate::RuntimeEffectControllerError::new(
            "runtime_effect_wrong_outcome",
            format!("expected sleep outcome, got {}", other.kind().as_str()),
        )),
    }
}

fn derive_idempotency_key(tool_context: &ToolContext<'_>, tool_name: &str) -> Option<String> {
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
    let ToolCallOutcome::Failure(failure) = &result.as_output().outcome else {
        return None;
    };
    let ToolRetryDisposition::Safe { after_ms } = &failure.retry else {
        return None;
    };
    Some(retry_policy.delay_ms_for_retry(retry_index, *after_ms))
}

fn mark_retry_exhausted(result: ToolResult, attempts: u32) -> ToolResult {
    let mut output = result.into_output();
    if let ToolCallOutcome::Failure(failure) = &mut output.outcome {
        failure.retry = ToolRetryDisposition::Exhausted { attempts };
    }
    ToolResult::from_output(output)
}
