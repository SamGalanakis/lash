use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use futures_util::stream::{FuturesUnordered, StreamExt};
use tokio::sync::mpsc;

use crate::plugin::{
    PluginDirective, PluginSession, RuntimeSessionHost, ToolCallHookContext, ToolResultHookContext,
    emit_plugin_runtime_events,
};
use crate::tool_schema::validate_tool_input;
use crate::{
    PreparedToolCall, ProgressSender, SessionEvent, ToolCall, ToolCallOutcome, ToolCallRecord,
    ToolContext, ToolExecutionMode, ToolFailure, ToolFailureClass, ToolManifest, ToolPrepareCall,
    ToolPrepareContext, ToolProvider, ToolResult, ToolRetryDisposition, ToolRetryPolicy,
    ToolSurface, TurnInjectionBridge,
};

#[derive(Clone)]
pub struct ToolDispatchContext<'run> {
    pub plugins: Arc<PluginSession>,
    pub tools: Arc<dyn ToolProvider>,
    pub surface: Arc<ToolSurface>,
    pub host: Arc<dyn RuntimeSessionHost>,
    pub processes: Arc<dyn crate::ProcessService>,
    pub(crate) effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    pub(crate) direct_completions: crate::DirectCompletionClient<'run>,
    pub(crate) tool_effect_metadata: Option<crate::EffectInvocationMetadata>,
    pub session_id: String,
    pub event_tx: mpsc::Sender<SessionEvent>,
    pub turn_injection_bridge: TurnInjectionBridge,
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub turn_context: crate::TurnContext,
}

impl<'run> ToolDispatchContext<'run> {
    pub fn process_scope(&self) -> crate::ProcessOpScope<'_> {
        crate::ProcessOpScope::new()
            .with_effect_metadata(self.tool_effect_metadata.clone())
            .with_effect_controller(self.effect_controller.as_controller())
    }
}

#[derive(Clone)]
pub(crate) struct ToolDispatchOutcome {
    pub record: ToolCallRecord,
}

pub(crate) enum ToolPreparationOutcome {
    Prepared(PreparedToolCall),
    Completed(Box<ToolDispatchOutcome>),
}

fn completed_preparation(outcome: ToolDispatchOutcome) -> ToolPreparationOutcome {
    ToolPreparationOutcome::Completed(Box::new(outcome))
}

#[derive(Clone)]
pub struct ParallelToolCallSpec {
    pub index: usize,
    pub tool_name: String,
    pub args: serde_json::Value,
}

#[derive(Clone)]
pub struct ParallelToolCallOutcome {
    pub index: usize,
    pub record: ToolCallRecord,
}

pub(crate) async fn dispatch_tool_call(
    context: &ToolDispatchContext<'_>,
    tool_name: String,
    args: serde_json::Value,
    progress: Option<&ProgressSender>,
) -> ToolDispatchOutcome {
    let tool_context = ToolContext::new(
        context.session_id.clone(),
        Arc::clone(&context.host),
        Arc::clone(&context.attachment_store),
        context.direct_completions.clone(),
        None,
    );
    dispatch_tool_call_with_execution_context(context, tool_name, args, progress, tool_context)
        .await
}

pub(crate) async fn dispatch_tool_call_with_execution_context<'run>(
    context: &ToolDispatchContext<'run>,
    tool_name: String,
    args: serde_json::Value,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolDispatchOutcome {
    let pending = crate::sansio::PendingToolCall {
        call_id: tool_context
            .tool_call_id()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("tool:{}", uuid::Uuid::new_v4())),
        tool_name,
        args,
        replay: None,
    };
    match prepare_tool_call_with_context(
        context,
        pending,
        tool_context.tool_call_id().map(str::to_string),
    )
    .await
    {
        ToolPreparationOutcome::Prepared(prepared) => {
            dispatch_prepared_tool_call_with_execution_context(
                context,
                prepared,
                progress,
                tool_context,
            )
            .await
        }
        ToolPreparationOutcome::Completed(outcome) => *outcome,
    }
}

pub(crate) async fn prepare_tool_call_with_context(
    context: &ToolDispatchContext<'_>,
    pending: crate::sansio::PendingToolCall,
    tool_call_id: Option<String>,
) -> ToolPreparationOutcome {
    let tool_name = pending.tool_name.clone();
    let Some(manifest) = resolve_callable_manifest(context, &tool_name) else {
        return completed_preparation(outcome(
            tool_name,
            pending.args,
            runtime_failure(
                ToolFailureClass::Unavailable,
                "tool_unavailable",
                "Tool is unavailable in this session",
            ),
            0,
        ));
    };
    let mut pending = pending;
    let mut args = pending.args;

    let directives = match context
        .plugins
        .before_tool_call(ToolCallHookContext::new(
            context.session_id.clone(),
            tool_name.clone(),
            args.clone(),
            manifest.argument_projection.clone(),
            context.turn_context.clone(),
            Arc::clone(&context.host),
        ))
        .await
    {
        Ok(directives) => directives,
        Err(err) => {
            return completed_preparation(outcome(
                tool_name,
                args,
                runtime_failure(
                    ToolFailureClass::Internal,
                    "before_tool_call_failed",
                    err.to_string(),
                ),
                0,
            ));
        }
    };

    let mut short_circuit: Option<ToolResult> = None;
    for emitted in directives {
        let plugin_id = emitted.plugin_id;
        let directive = emitted.value;
        match directive {
            PluginDirective::CreateSession { request } => {
                if let Err(err) = context.host.create_session(*request).await {
                    short_circuit = Some(ToolResult::err_fmt(err.to_string()));
                    break;
                }
            }
            PluginDirective::HandoffSession { .. } => {
                short_circuit = Some(ToolResult::err_fmt(
                    "before_tool_call does not support session handoff",
                ));
                break;
            }
            PluginDirective::ReplaceToolArgs { args: replacement } => {
                args = replacement;
            }
            PluginDirective::ShortCircuitTool { output } => {
                short_circuit = Some(ToolResult::from_output(output));
            }
            PluginDirective::AbortTurn { message, .. } => {
                short_circuit = Some(ToolResult::err_fmt(message));
            }
            PluginDirective::EmitRuntimeEvents { events } => {
                emit_plugin_runtime_events(&context.event_tx, &plugin_id, events).await;
            }
            PluginDirective::EmitTrace {
                name,
                payload,
                context: trace_context,
            } => {
                if let Err(err) = context
                    .host
                    .emit_trace_event(
                        *trace_context,
                        lash_trace::TraceEvent::Custom {
                            name: format!("plugin.{plugin_id}.{name}"),
                            payload,
                        },
                    )
                    .await
                {
                    short_circuit = Some(ToolResult::err_fmt(err.to_string()));
                    break;
                }
            }
            PluginDirective::EnqueueMessages { .. } => {
                short_circuit = Some(ToolResult::err_fmt(
                    "before_tool_call does not support message injection",
                ));
            }
        }
    }
    if let Some(result) = short_circuit {
        return completed_preparation(outcome(tool_name, args, result, 0));
    }

    let contract = context
        .plugins
        .mode_native_tools()
        .iter()
        .find_map(|provider| provider.resolve_contract(&tool_name))
        .or_else(|| context.tools.resolve_contract(&tool_name));
    let Some(contract) = contract else {
        return completed_preparation(outcome(
            tool_name,
            args,
            runtime_failure(
                ToolFailureClass::Unavailable,
                "tool_contract_unavailable",
                "Tool contract is unavailable in this session",
            ),
            0,
        ));
    };
    if let Err(err) = validate_tool_input(&contract, &args) {
        return completed_preparation(outcome(
            tool_name,
            args,
            runtime_failure(ToolFailureClass::InvalidRequest, "invalid_tool_args", err),
            0,
        ));
    }

    pending.args = args.clone();
    let is_mode_native = context
        .plugins
        .mode_native_tools()
        .iter()
        .any(|provider| provider.resolve_manifest(&tool_name).is_some());
    if is_mode_native {
        return ToolPreparationOutcome::Prepared(PreparedToolCall::identity(pending));
    }
    let prepare_context = ToolPrepareContext::new(
        context.session_id.clone(),
        Arc::clone(&context.host),
        context.turn_context.clone(),
        tool_call_id,
    );
    match context
        .tools
        .prepare_tool_call(ToolPrepareCall {
            pending,
            context: &prepare_context,
        })
        .await
    {
        Ok(prepared) => ToolPreparationOutcome::Prepared(prepared),
        Err(result) => completed_preparation(outcome(tool_name, args, result, 0)),
    }
}

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
        Ok(directives) => {
            let mut final_result = result;
            for emitted in directives {
                let plugin_id = emitted.plugin_id;
                let directive = emitted.value;
                match directive {
                    PluginDirective::CreateSession { request } => {
                        if let Err(err) = context.host.create_session(*request).await {
                            final_result = ToolResult::failure(ToolFailure::runtime(
                                ToolFailureClass::Internal,
                                "plugin_session_create_failed",
                                err.to_string(),
                            ));
                            break;
                        }
                    }
                    PluginDirective::HandoffSession { .. } => {
                        final_result =
                            ToolResult::err_fmt("after_tool_call does not support session handoff");
                        break;
                    }
                    PluginDirective::ShortCircuitTool { output } => {
                        final_result = ToolResult::from_output(output);
                    }
                    PluginDirective::AbortTurn { message, .. } => {
                        final_result = ToolResult::err_fmt(message);
                    }
                    PluginDirective::EmitRuntimeEvents { events } => {
                        emit_plugin_runtime_events(&context.event_tx, &plugin_id, events).await;
                    }
                    PluginDirective::EmitTrace {
                        name,
                        payload,
                        context: trace_context,
                    } => {
                        if let Err(err) = context
                            .host
                            .emit_trace_event(
                                *trace_context,
                                lash_trace::TraceEvent::Custom {
                                    name: format!("plugin.{plugin_id}.{name}"),
                                    payload,
                                },
                            )
                            .await
                        {
                            final_result = ToolResult::err_fmt(err.to_string());
                            break;
                        }
                    }
                    PluginDirective::EnqueueMessages { messages } => {
                        if let Err(err) = context.turn_injection_bridge.enqueue(messages) {
                            final_result = ToolResult::err_fmt(err);
                            break;
                        }
                    }
                    PluginDirective::ReplaceToolArgs { .. } => {
                        final_result = ToolResult::err_fmt(
                            "after_tool_call only supports abort, short-circuit, session creation, events, and message injection",
                        );
                    }
                }
            }
            final_result
        }
        Err(err) => runtime_failure(
            ToolFailureClass::Internal,
            "after_tool_call_failed",
            err.to_string(),
        ),
    };

    outcome(tool_name, args, result, duration_ms)
}

pub(crate) fn resolve_callable_manifest(
    context: &ToolDispatchContext<'_>,
    tool_name: &str,
) -> Option<ToolManifest> {
    if let Some(entry) = context
        .surface
        .tools
        .iter()
        .find(|tool| tool.manifest.name == tool_name)
    {
        return entry
            .availability
            .is_callable()
            .then(|| entry.manifest.clone());
    }

    let mode = context.plugins.execution_mode();
    let visible_and_callable = |manifest: ToolManifest| {
        if context.plugins.tool_access().hides(&manifest.name) {
            return None;
        }
        manifest
            .effective_availability(&mode)
            .is_callable()
            .then_some(manifest)
    };

    for provider in context.plugins.mode_native_tools() {
        if let Some(manifest) = provider
            .resolve_manifest(tool_name)
            .and_then(&visible_and_callable)
        {
            return Some(manifest);
        }
    }

    context
        .tools
        .resolve_manifest(tool_name)
        .and_then(visible_and_callable)
}

pub(crate) fn resolve_callable_manifest_by_id(
    context: &ToolDispatchContext<'_>,
    tool_id: &crate::ToolId,
) -> Option<ToolManifest> {
    if let Some(entry) = context
        .surface
        .tools
        .iter()
        .find(|tool| tool.manifest.id == *tool_id)
    {
        return entry
            .availability
            .is_callable()
            .then(|| entry.manifest.clone());
    }

    let mode = context.plugins.execution_mode();
    let visible_and_callable = |manifest: ToolManifest| {
        if context.plugins.tool_access().hides(&manifest.name) {
            return None;
        }
        manifest
            .effective_availability(&mode)
            .is_callable()
            .then_some(manifest)
    };

    for provider in context.plugins.mode_native_tools() {
        if let Some(manifest) = provider
            .tool_manifests()
            .into_iter()
            .find(|manifest| manifest.id == *tool_id)
            .and_then(visible_and_callable)
        {
            return Some(manifest);
        }
    }

    context
        .tools
        .tool_manifests()
        .into_iter()
        .find(|manifest| manifest.id == *tool_id)
        .and_then(visible_and_callable)
}

pub(crate) async fn dispatch_parallel_tool_call(
    context: Arc<ToolDispatchContext<'_>>,
    spec: ParallelToolCallSpec,
    progress: Option<ProgressSender>,
) -> ParallelToolCallOutcome {
    let outcome = dispatch_tool_call(&context, spec.tool_name, spec.args, progress.as_ref()).await;
    ParallelToolCallOutcome {
        index: spec.index,
        record: outcome.record,
    }
}

/// Resolve the [`ToolExecutionMode`] declared on a tool's definition. Unknown
/// tool names default to [`ToolExecutionMode::Parallel`] — the dispatcher
/// will still surface an "unknown tool" error via the normal path.
pub(crate) fn resolve_tool_execution_mode(
    context: &ToolDispatchContext<'_>,
    tool_name: &str,
) -> ToolExecutionMode {
    context
        .surface
        .tools
        .iter()
        .find(|def| def.manifest.name == tool_name)
        .map(|def| def.manifest.execution_mode)
        .unwrap_or_default()
}

pub(crate) fn resolve_tool_argument_projection_policy(
    context: &ToolDispatchContext<'_>,
    tool_name: &str,
) -> crate::ToolArgumentProjectionPolicy {
    context
        .surface
        .tools
        .iter()
        .find(|def| def.manifest.name == tool_name)
        .map(|def| def.manifest.argument_projection.clone())
        .unwrap_or_default()
}

/// Schedule a batch using Lash's tool execution policy.
///
/// Parallel-safe tools run concurrently first, then serial tools run
/// one-at-a-time in original index order. Returned outputs are sorted by the
/// same original index so callers keep their source/model ordering.
pub(crate) async fn schedule_tool_batch<T, O, IndexOf, ModeOf, Run, Fut>(
    items: Vec<T>,
    index_of: IndexOf,
    mode_of: ModeOf,
    run: Run,
) -> Vec<O>
where
    T: Send + 'static,
    O: Send + 'static,
    IndexOf: Fn(&T) -> usize,
    ModeOf: Fn(&T) -> ToolExecutionMode,
    Run: Fn(T) -> Fut,
    Fut: Future<Output = O> + Send,
{
    let mut parallel_items = Vec::new();
    let mut serial_items = Vec::new();
    for item in items {
        let index = index_of(&item);
        match mode_of(&item) {
            ToolExecutionMode::Parallel => parallel_items.push((index, item)),
            ToolExecutionMode::Serial => serial_items.push((index, item)),
        }
    }

    let mut outcomes = Vec::new();

    let mut pending = FuturesUnordered::new();
    for (index, item) in parallel_items {
        let future = run(item);
        pending.push(async move { (index, future.await) });
    }
    while let Some(outcome) = pending.next().await {
        outcomes.push(outcome);
    }

    serial_items.sort_by_key(|(index, _)| *index);
    for (index, item) in serial_items {
        outcomes.push((index, run(item).await));
    }

    outcomes.sort_by_key(|(index, _)| *index);
    outcomes.into_iter().map(|(_, outcome)| outcome).collect()
}

/// Dispatch a batch of tool calls produced by one model response.
pub async fn dispatch_parallel_tool_calls(
    context: Arc<ToolDispatchContext<'_>>,
    specs: Vec<ParallelToolCallSpec>,
    progress: Option<&ProgressSender>,
) -> Vec<ParallelToolCallOutcome> {
    let progress = progress.cloned();
    schedule_tool_batch(
        specs,
        |spec| spec.index,
        {
            let context = Arc::clone(&context);
            move |spec| resolve_tool_execution_mode(&context, &spec.tool_name)
        },
        move |spec| dispatch_parallel_tool_call(Arc::clone(&context), spec, progress.clone()),
    )
    .await
}

fn outcome(
    tool_name: String,
    args: serde_json::Value,
    result: ToolResult,
    duration_ms: u64,
) -> ToolDispatchOutcome {
    let record = ToolCallRecord {
        call_id: None,
        tool: tool_name,
        args,
        output: result.into_output(),
        duration_ms,
    };
    ToolDispatchOutcome { record }
}

fn runtime_failure(
    class: ToolFailureClass,
    code: impl Into<String>,
    message: impl Into<String>,
) -> ToolResult {
    ToolResult::failure(ToolFailure::runtime(class, code, message))
}

async fn execute_tool_call<'run>(
    context: &ToolDispatchContext<'run>,
    manifest: &ToolManifest,
    prepared: &PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolResult {
    let tool_name = prepared.tool_name.as_str();
    let retry_policy = manifest.retry_policy;
    let max_attempts = retry_policy.max_attempts();
    let idempotency_key = derive_idempotency_key(&tool_context, tool_name);
    if retry_policy.requires_idempotency_key() && idempotency_key.is_none() {
        return execute_once(
            context,
            prepared,
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
        let result = execute_once(context, prepared, progress, attempt_context).await;
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
            return ToolResult::failure(ToolFailure::runtime(
                ToolFailureClass::Internal,
                "tool_retry_sleep_failed",
                format!("retry sleep for tool `{tool_name}` failed after attempt {attempt}: {err}"),
            ));
        }
    }

    execute_once(
        context,
        prepared,
        progress,
        tool_context.with_retry_context(tool_name, 1, 1),
    )
    .await
}

async fn execute_once<'run>(
    context: &ToolDispatchContext<'run>,
    prepared: &PreparedToolCall,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolResult {
    let tool_name = prepared.tool_name.as_str();
    let args = &prepared.args;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{PluginHost, StaticPluginFactory};
    use crate::runtime::RuntimeEffectControllerHandle;
    use crate::{
        ExecutionMode, ToolCall, ToolCallOutcome, ToolProvider, ToolRetryDisposition,
        ToolRetryPolicy,
    };
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Barrier;
    use tokio::time::{Duration, timeout};

    type ExecutionWindow = (&'static str, Instant, Instant);
    type SharedExecutionWindows = Arc<std::sync::Mutex<Vec<ExecutionWindow>>>;
    type AttemptObservation = (u32, u32, Option<String>);
    type SharedAttemptObservations = Arc<std::sync::Mutex<Vec<AttemptObservation>>>;

    fn test_tool(name: &str, execution_mode: ToolExecutionMode) -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            format!("tool:{name}"),
            name,
            "",
            crate::ToolDefinition::default_input_schema(),
            json!({ "type": "string" }),
        )
        .with_execution_mode(execution_mode)
    }

    fn beta_tool() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:beta",
            "beta",
            "",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"],
                "additionalProperties": false
            }),
            json!({ "type": "string" }),
        )
        .with_execution_mode(ToolExecutionMode::Parallel)
    }

    fn named_beta_tool(name: &str) -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            format!("tool:{name}"),
            name,
            "",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"],
                "additionalProperties": false
            }),
            json!({ "type": "string" }),
        )
        .with_execution_mode(ToolExecutionMode::Parallel)
    }

    fn manifests(definitions: Vec<crate::ToolDefinition>) -> Vec<crate::ToolManifest> {
        definitions
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn contract_from(
        definitions: Vec<crate::ToolDefinition>,
        name: &str,
    ) -> Option<Arc<crate::ToolContract>> {
        definitions
            .into_iter()
            .find(|tool| tool.name == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    #[derive(Clone)]
    struct ScheduledProbe {
        index: usize,
        name: &'static str,
        mode: ToolExecutionMode,
        delay: Duration,
    }

    #[tokio::test]
    async fn scheduler_runs_parallel_bucket_then_serial_and_preserves_order() {
        let windows: SharedExecutionWindows = Arc::new(std::sync::Mutex::new(Vec::new()));
        let probes = vec![
            ScheduledProbe {
                index: 0,
                name: "parallel_slow",
                mode: ToolExecutionMode::Parallel,
                delay: Duration::from_millis(40),
            },
            ScheduledProbe {
                index: 1,
                name: "serial",
                mode: ToolExecutionMode::Serial,
                delay: Duration::from_millis(10),
            },
            ScheduledProbe {
                index: 2,
                name: "parallel_fast",
                mode: ToolExecutionMode::Parallel,
                delay: Duration::from_millis(5),
            },
        ];

        let outputs = schedule_tool_batch(probes, |probe| probe.index, |probe| probe.mode, {
            let windows = Arc::clone(&windows);
            move |probe| {
                let windows = Arc::clone(&windows);
                async move {
                    let start = Instant::now();
                    tokio::time::sleep(probe.delay).await;
                    let end = Instant::now();
                    windows
                        .lock()
                        .expect("windows")
                        .push((probe.name, start, end));
                    probe.name
                }
            }
        })
        .await;

        assert_eq!(outputs, ["parallel_slow", "serial", "parallel_fast"]);

        let recorded = windows.lock().expect("windows").clone();
        let parallel_slow = recorded
            .iter()
            .find(|(name, _, _)| *name == "parallel_slow")
            .expect("parallel_slow");
        let parallel_fast = recorded
            .iter()
            .find(|(name, _, _)| *name == "parallel_fast")
            .expect("parallel_fast");
        let serial = recorded
            .iter()
            .find(|(name, _, _)| *name == "serial")
            .expect("serial");

        assert!(
            parallel_fast.1 < parallel_slow.2,
            "parallel tools should overlap even when completion order differs"
        );
        assert!(
            serial.1 >= parallel_slow.2 && serial.1 >= parallel_fast.2,
            "serial tool should start after the parallel bucket completes"
        );
    }

    struct MockTools;

    #[async_trait::async_trait]
    impl ToolProvider for MockTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![
                test_tool("alpha", ToolExecutionMode::Parallel),
                beta_tool(),
            ])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            contract_from(
                vec![test_tool("alpha", ToolExecutionMode::Parallel), beta_tool()],
                name,
            )
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            match call.name {
                "alpha" => ToolResult::ok(json!("alpha")),
                "beta" => {
                    if call.args.get("value").and_then(|value| value.as_str()) == Some("fail") {
                        ToolResult::err_fmt("beta failed")
                    } else {
                        ToolResult::ok(json!(
                            call.args.get("value").cloned().unwrap_or(json!(null))
                        ))
                    }
                }
                other => ToolResult::err_fmt(format!("Unknown tool: {other}")),
            }
        }
    }

    struct ParallelProbeTools {
        barrier: Arc<Barrier>,
        started: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for ParallelProbeTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![
                test_tool("probe_a", ToolExecutionMode::Parallel),
                test_tool("probe_b", ToolExecutionMode::Parallel),
            ])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            contract_from(
                vec![
                    test_tool("probe_a", ToolExecutionMode::Parallel),
                    test_tool("probe_b", ToolExecutionMode::Parallel),
                ],
                name,
            )
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            self.started.fetch_add(1, Ordering::SeqCst);
            let waited = timeout(Duration::from_millis(100), self.barrier.wait()).await;
            match waited {
                Ok(_) => ToolResult::ok(json!(call.name)),
                Err(_) => ToolResult::err_fmt(format!("{} did not overlap with peer", call.name)),
            }
        }
    }

    struct StrictMcpTools {
        executed: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for StrictMcpTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![strict_mcp_tool_definition()])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "mcp__appworld__venmo_show_transactions")
                .then(|| Arc::new(strict_mcp_tool_definition().contract()))
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            self.executed.fetch_add(1, Ordering::SeqCst);
            ToolResult::ok(json!({ "executed": true }))
        }
    }

    fn strict_mcp_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:mcp__appworld__venmo_show_transactions",
            "mcp__appworld__venmo_show_transactions",
            "Show Venmo transactions",
            json!({
                "type": "object",
                "properties": {
                    "min_created_at": { "type": "string" },
                    "max_created_at": { "type": "string" },
                    "limit": { "type": "integer", "maximum": 100 }
                },
                "required": ["limit"]
            }),
            json!({ "type": "object", "additionalProperties": true }),
        )
    }

    struct ProjectionPolicyTools;

    #[async_trait::async_trait]
    impl ToolProvider for ProjectionPolicyTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![projection_policy_tool_definition()])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "seedy").then(|| Arc::new(projection_policy_tool_definition().contract()))
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(json!("ok"))
        }
    }

    fn projection_policy_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:seedy",
            "seedy",
            "Seed-aware",
            crate::ToolDefinition::default_input_schema(),
            json!({ "type": "string" }),
        )
        .with_argument_projection(
            crate::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"),
        )
    }

    fn strict_mcp_dispatch_context(executed: Arc<AtomicUsize>) -> ToolDispatchContext<'static> {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::new(StrictMcpTools { executed }));
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        }
    }

    fn test_plugins(provider: Arc<dyn ToolProvider>) -> Arc<PluginSession> {
        PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "test_tools",
            crate::PluginSpec::new().with_tool_provider(Arc::clone(&provider)),
        ))])
        .build_standard_session("root", None)
        .expect("plugin session")
    }

    use crate::testing::MockSessionManager;

    fn dispatch_context() -> ToolDispatchContext<'static> {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::new(MockTools));
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        }
    }

    fn projection_policy_dispatch_context(
        captured: Arc<std::sync::Mutex<Option<crate::ToolArgumentProjectionPolicy>>>,
    ) -> ToolDispatchContext<'static> {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let provider: Arc<dyn ToolProvider> = Arc::new(ProjectionPolicyTools);
        let hook_captured = Arc::clone(&captured);
        let hook: crate::plugin::BeforeToolCallHook = Arc::new(move |ctx| {
            let hook_captured = Arc::clone(&hook_captured);
            Box::pin(async move {
                *hook_captured.lock().expect("captured policy") =
                    Some(ctx.argument_projection.clone());
                Ok(Vec::new())
            })
        });
        let plugins = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "projection_policy_tools",
            crate::PluginSpec::new()
                .with_tool_provider(Arc::clone(&provider))
                .with_before_tool_call(hook),
        ))])
        .build_standard_session("root", None)
        .expect("plugin session");
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        }
    }

    struct CountingContractTools {
        contracts_resolved: Arc<AtomicUsize>,
        executed: Arc<AtomicUsize>,
    }

    struct ExactDispatchTools {
        contracts_resolved: Arc<AtomicUsize>,
        executed: Arc<AtomicUsize>,
        contract_available: bool,
    }

    struct HiddenDispatchTools {
        contracts_resolved: Arc<AtomicUsize>,
        executed: Arc<AtomicUsize>,
    }

    struct RetryProbeTools {
        definition: crate::ToolDefinition,
        attempts: Arc<AtomicUsize>,
        successes_after: usize,
        cancel_on_first: bool,
        observed_attempts: SharedAttemptObservations,
        retry_after_ms: Option<u64>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for CountingContractTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![beta_tool()])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            self.contracts_resolved.fetch_add(1, Ordering::SeqCst);
            (name == "beta").then(|| Arc::new(beta_tool().contract()))
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            self.executed.fetch_add(1, Ordering::SeqCst);
            ToolResult::ok(json!("ok"))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for ExactDispatchTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            Vec::new()
        }

        fn resolve_manifest(&self, name: &str) -> Option<crate::ToolManifest> {
            (name == "host_only").then(|| named_beta_tool("host_only").manifest())
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            self.contracts_resolved.fetch_add(1, Ordering::SeqCst);
            (self.contract_available && name == "host_only")
                .then(|| Arc::new(named_beta_tool("host_only").contract()))
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            self.executed.fetch_add(1, Ordering::SeqCst);
            ToolResult::ok(json!("host"))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for HiddenDispatchTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![
                named_beta_tool("hidden").with_availability(crate::ToolAvailabilityConfig::off()),
            ])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            self.contracts_resolved.fetch_add(1, Ordering::SeqCst);
            (name == "hidden").then(|| Arc::new(named_beta_tool("hidden").contract()))
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            self.executed.fetch_add(1, Ordering::SeqCst);
            ToolResult::ok(json!("hidden"))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for RetryProbeTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![self.definition.clone()])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == self.definition.name).then(|| Arc::new(self.definition.contract()))
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            self.observed_attempts.lock().expect("attempts").push((
                call.context.attempt_number(),
                call.context.max_attempts(),
                call.context.idempotency_key().map(str::to_string),
            ));
            let attempt_index = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
            if self.cancel_on_first {
                return ToolResult::cancelled("cancelled");
            }
            if attempt_index >= self.successes_after {
                return ToolResult::ok(json!({ "attempt": attempt_index }));
            }
            ToolResult::retryable_failure(
                crate::ToolFailureClass::External,
                "transient",
                "transient failure",
                self.retry_after_ms,
            )
        }
    }

    fn lazy_contract_dispatch_context(
        contracts_resolved: Arc<AtomicUsize>,
        executed: Arc<AtomicUsize>,
    ) -> ToolDispatchContext<'static> {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let provider: Arc<dyn ToolProvider> = Arc::new(CountingContractTools {
            contracts_resolved,
            executed,
        });
        let tools = Arc::clone(&provider);
        let surface = Arc::new(crate::ToolSurface::from_tools(
            provider.tool_manifests(),
            ExecutionMode::standard(),
            BTreeMap::new(),
        ));
        ToolDispatchContext {
            plugins: test_plugins(provider),
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        }
    }

    fn exact_dispatch_context(provider: Arc<dyn ToolProvider>) -> ToolDispatchContext<'static> {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::clone(&provider));
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        }
    }

    fn retry_tool(name: &str, retry_policy: ToolRetryPolicy) -> crate::ToolDefinition {
        named_beta_tool(name)
            .with_execution_mode(ToolExecutionMode::Parallel)
            .with_retry_policy(retry_policy)
    }

    fn retry_dispatch_context(
        retry_policy: ToolRetryPolicy,
        attempts: Arc<AtomicUsize>,
        successes_after: usize,
        cancel_on_first: bool,
        observed_attempts: SharedAttemptObservations,
    ) -> ToolDispatchContext<'static> {
        exact_dispatch_context(Arc::new(RetryProbeTools {
            definition: retry_tool("retry_probe", retry_policy),
            attempts,
            successes_after,
            cancel_on_first,
            observed_attempts,
            retry_after_ms: Some(0),
        }))
    }

    fn parallel_dispatch_context(
        barrier: Arc<Barrier>,
        started: Arc<AtomicUsize>,
    ) -> ToolDispatchContext<'static> {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::new(ParallelProbeTools { barrier, started }));
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        }
    }

    #[tokio::test]
    async fn dispatch_rejects_invalid_args_before_provider_execution() {
        let outcome =
            dispatch_tool_call(&dispatch_context(), "beta".to_string(), json!({}), None).await;

        assert!(!outcome.record.output.is_success());
        assert_eq!(
            outcome.record.output.value_for_projection()["message"],
            json!("value: required property missing")
        );
    }

    #[tokio::test]
    async fn dispatch_resolves_contract_only_for_called_tool_before_execution() {
        let contracts_resolved = Arc::new(AtomicUsize::new(0));
        let executed = Arc::new(AtomicUsize::new(0));
        let outcome = dispatch_tool_call(
            &lazy_contract_dispatch_context(Arc::clone(&contracts_resolved), Arc::clone(&executed)),
            "beta".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(outcome.record.output.is_success());
        assert_eq!(contracts_resolved.load(Ordering::SeqCst), 1);
        assert_eq!(executed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn before_tool_hook_receives_resolved_argument_projection_policy() {
        let captured = Arc::new(std::sync::Mutex::new(None));
        let outcome = dispatch_tool_call(
            &projection_policy_dispatch_context(Arc::clone(&captured)),
            "seedy".to_string(),
            json!({}),
            None,
        )
        .await;

        assert!(outcome.record.output.is_success());
        assert_eq!(
            captured.lock().expect("captured policy").clone(),
            Some(crate::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"))
        );
    }

    #[tokio::test]
    async fn dispatch_exact_resolves_missing_surface_tool_and_executes_owner() {
        let contracts_resolved = Arc::new(AtomicUsize::new(0));
        let executed = Arc::new(AtomicUsize::new(0));
        let provider: Arc<dyn ToolProvider> = Arc::new(ExactDispatchTools {
            contracts_resolved: Arc::clone(&contracts_resolved),
            executed: Arc::clone(&executed),
            contract_available: true,
        });
        let outcome = dispatch_tool_call(
            &exact_dispatch_context(provider),
            "host_only".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(outcome.record.output.is_success());
        assert_eq!(outcome.record.output.value_for_projection(), json!("host"));
        assert_eq!(contracts_resolved.load(Ordering::SeqCst), 1);
        assert_eq!(executed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_contract_unavailable_skips_execution() {
        let contracts_resolved = Arc::new(AtomicUsize::new(0));
        let executed = Arc::new(AtomicUsize::new(0));
        let provider: Arc<dyn ToolProvider> = Arc::new(ExactDispatchTools {
            contracts_resolved: Arc::clone(&contracts_resolved),
            executed: Arc::clone(&executed),
            contract_available: false,
        });
        let outcome = dispatch_tool_call(
            &exact_dispatch_context(provider),
            "host_only".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(!outcome.record.output.is_success());
        assert_eq!(
            outcome.record.output.value_for_projection()["message"],
            json!("Tool contract is unavailable in this session")
        );
        assert_eq!(contracts_resolved.load(Ordering::SeqCst), 1);
        assert_eq!(executed.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn dispatch_rejects_hidden_tool_before_contract_resolution() {
        let contracts_resolved = Arc::new(AtomicUsize::new(0));
        let executed = Arc::new(AtomicUsize::new(0));
        let provider: Arc<dyn ToolProvider> = Arc::new(HiddenDispatchTools {
            contracts_resolved: Arc::clone(&contracts_resolved),
            executed: Arc::clone(&executed),
        });
        let outcome = dispatch_tool_call(
            &exact_dispatch_context(provider),
            "hidden".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(!outcome.record.output.is_success());
        assert_eq!(
            outcome.record.output.value_for_projection()["message"],
            json!("Tool is unavailable in this session")
        );
        assert_eq!(contracts_resolved.load(Ordering::SeqCst), 0);
        assert_eq!(executed.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn dispatch_rejects_unknown_mcp_args_before_provider_execution() {
        let executed = Arc::new(AtomicUsize::new(0));
        let outcome = dispatch_tool_call(
            &strict_mcp_dispatch_context(Arc::clone(&executed)),
            "mcp__appworld__venmo_show_transactions".to_string(),
            json!({
                "min_datetime": "2024-01-01T00:00:00Z",
                "limit": 20
            }),
            None,
        )
        .await;

        assert!(!outcome.record.output.is_success());
        assert_eq!(
            outcome.record.output.value_for_projection()["message"],
            json!("min_datetime: unexpected property")
        );
        assert_eq!(executed.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn default_retry_policy_never_retries_safe_failures() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let outcome = dispatch_tool_call(
            &retry_dispatch_context(
                ToolRetryPolicy::Never,
                Arc::clone(&attempts),
                usize::MAX,
                false,
                Arc::clone(&observed),
            ),
            "retry_probe".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(!outcome.record.output.is_success());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(observed.lock().expect("observed")[0].0, 1);
    }

    #[tokio::test]
    async fn safe_retry_policy_retries_safe_failure_and_stops_on_success() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let outcome = dispatch_tool_call(
            &retry_dispatch_context(
                ToolRetryPolicy::safe(3, 0, 0),
                Arc::clone(&attempts),
                2,
                false,
                Arc::clone(&observed),
            ),
            "retry_probe".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(outcome.record.output.is_success());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(
            observed
                .lock()
                .expect("observed")
                .iter()
                .map(|(attempt, max, _)| (*attempt, *max))
                .collect::<Vec<_>>(),
            vec![(1, 3), (2, 3)]
        );
    }

    #[derive(Default)]
    struct SleepRecordingEffectController {
        sleeps: Arc<std::sync::Mutex<Vec<crate::EffectInvocationMetadata>>>,
    }

    #[async_trait::async_trait]
    impl crate::RuntimeEffectController for SleepRecordingEffectController {
        async fn execute_effect(
            &self,
            envelope: crate::RuntimeEffectEnvelope,
            _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
            self.sleeps
                .lock()
                .expect("sleep records")
                .push(envelope.metadata);
            Ok(crate::RuntimeEffectOutcome::Sleep)
        }
    }

    struct FailingSleepEffectController;

    #[async_trait::async_trait]
    impl crate::RuntimeEffectController for FailingSleepEffectController {
        async fn execute_effect(
            &self,
            envelope: crate::RuntimeEffectEnvelope,
            _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
            Err(crate::RuntimeEffectControllerError::new(
                "test_sleep_rejected",
                format!("rejected {}", envelope.command.kind().as_str()),
            ))
        }
    }

    #[tokio::test]
    async fn retry_delay_crosses_effect_controller_as_sleep_effect() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = Arc::new(SleepRecordingEffectController::default());
        let mut context = exact_dispatch_context(Arc::new(RetryProbeTools {
            definition: retry_tool("retry_probe", ToolRetryPolicy::safe(3, 25, 25)),
            attempts: Arc::clone(&attempts),
            successes_after: 2,
            cancel_on_first: false,
            observed_attempts: Arc::clone(&observed),
            retry_after_ms: Some(25),
        }));
        context.effect_controller = RuntimeEffectControllerHandle::shared(recorder.clone());
        let tool_context = ToolContext::new(
            context.session_id.clone(),
            Arc::clone(&context.host),
            Arc::clone(&context.attachment_store),
            context.direct_completions.clone(),
            Some("call-1".to_string()),
        );

        let outcome = dispatch_tool_call_with_execution_context(
            &context,
            "retry_probe".to_string(),
            json!({ "value": "ok" }),
            None,
            tool_context,
        )
        .await;

        assert!(outcome.record.output.is_success());
        let sleeps = recorder.sleeps.lock().expect("sleep records");
        assert_eq!(sleeps.len(), 1);
        assert_eq!(sleeps[0].effect_kind, crate::RuntimeEffectKind::Sleep);
        assert_eq!(
            sleeps[0].idempotency_key,
            "lash-tool:session:call-1:retry_probe:attempt:1:sleep"
        );
    }

    #[tokio::test]
    async fn retry_sleep_controller_rejection_returns_explicit_tool_failure() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut context = exact_dispatch_context(Arc::new(RetryProbeTools {
            definition: retry_tool("retry_probe", ToolRetryPolicy::safe(3, 25, 25)),
            attempts: Arc::clone(&attempts),
            successes_after: 2,
            cancel_on_first: false,
            observed_attempts: Arc::clone(&observed),
            retry_after_ms: Some(25),
        }));
        context.effect_controller =
            RuntimeEffectControllerHandle::shared(Arc::new(FailingSleepEffectController));
        let tool_context = ToolContext::new(
            context.session_id.clone(),
            Arc::clone(&context.host),
            Arc::clone(&context.attachment_store),
            context.direct_completions.clone(),
            Some("call-1".to_string()),
        );

        let outcome = dispatch_tool_call_with_execution_context(
            &context,
            "retry_probe".to_string(),
            json!({ "value": "ok" }),
            None,
            tool_context,
        )
        .await;

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        let ToolCallOutcome::Failure(failure) = outcome.record.output.outcome else {
            panic!("expected failure");
        };
        assert_eq!(failure.code, "tool_retry_sleep_failed");
    }

    #[tokio::test]
    async fn safe_retry_policy_marks_exhausted_after_final_attempt() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let outcome = dispatch_tool_call(
            &retry_dispatch_context(
                ToolRetryPolicy::safe(2, 0, 0),
                Arc::clone(&attempts),
                usize::MAX,
                false,
                Arc::clone(&observed),
            ),
            "retry_probe".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(!outcome.record.output.is_success());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        let ToolCallOutcome::Failure(failure) = outcome.record.output.outcome else {
            panic!("expected failure");
        };
        assert_eq!(
            failure.retry,
            ToolRetryDisposition::Exhausted { attempts: 2 }
        );
    }

    #[tokio::test]
    async fn cancellation_stops_retry_immediately() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let outcome = dispatch_tool_call(
            &retry_dispatch_context(
                ToolRetryPolicy::safe(3, 0, 0),
                Arc::clone(&attempts),
                usize::MAX,
                true,
                Arc::clone(&observed),
            ),
            "retry_probe".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(!outcome.record.output.is_success());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(matches!(
            outcome.record.output.outcome,
            ToolCallOutcome::Cancelled(_)
        ));
    }

    #[tokio::test]
    async fn retry_context_has_stable_idempotency_key_across_attempts() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let context = retry_dispatch_context(
            ToolRetryPolicy::safe(3, 0, 0),
            Arc::clone(&attempts),
            3,
            false,
            Arc::clone(&observed),
        );
        let tool_context = ToolContext::new(
            context.session_id.clone(),
            Arc::clone(&context.host),
            Arc::clone(&context.attachment_store),
            context.direct_completions.clone(),
            Some("call-1".to_string()),
        );
        let outcome = dispatch_tool_call_with_execution_context(
            &context,
            "retry_probe".to_string(),
            json!({ "value": "ok" }),
            None,
            tool_context,
        )
        .await;

        assert!(outcome.record.output.is_success());
        let observed = observed.lock().expect("observed");
        assert_eq!(observed.len(), 3);
        assert_eq!(
            observed
                .iter()
                .map(|(attempt, max, _)| (*attempt, *max))
                .collect::<Vec<_>>(),
            vec![(1, 3), (2, 3), (3, 3)]
        );
        let keys = observed
            .iter()
            .map(|(_, _, key)| key.clone())
            .collect::<Vec<_>>();
        assert!(keys.iter().all(|key| key == &keys[0]));
        assert_eq!(
            keys[0].as_deref(),
            Some("lash-tool:session:call-1:retry_probe")
        );
    }

    #[tokio::test]
    async fn idempotent_retry_policy_requires_stable_key() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let outcome = dispatch_tool_call(
            &retry_dispatch_context(
                ToolRetryPolicy::idempotent(3, 0, 0),
                Arc::clone(&attempts),
                usize::MAX,
                false,
                Arc::clone(&observed),
            ),
            "retry_probe".to_string(),
            json!({ "value": "ok" }),
            None,
        )
        .await;

        assert!(!outcome.record.output.is_success());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(observed.lock().expect("observed")[0].1, 1);
    }

    #[tokio::test]
    async fn batch_executes_nested_calls_and_preserves_partial_failures() {
        let outcome = dispatch_tool_call(
            &dispatch_context(),
            "batch".to_string(),
            json!({
                "tool_calls": [
                    {"tool": "alpha", "parameters": {}},
                    {"tool": "beta", "parameters": {"value": "ok"}},
                    {"tool": "beta", "parameters": {"value": "fail"}}
                ]
            }),
            None,
        )
        .await;

        assert!(outcome.record.output.is_success());
        assert_eq!(outcome.record.tool, "batch");
        let value = outcome.record.output.value_for_projection();
        let results = value
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(results.len(), 3);
        assert_eq!(
            results
                .iter()
                .filter(|item| item.get("success").and_then(|value| value.as_bool()) == Some(true))
                .count(),
            2
        );
        assert_eq!(results[0].get("tool"), Some(&json!("alpha")));
        assert_eq!(
            results[2]
                .get("error")
                .and_then(|value| value.get("message")),
            Some(&json!("beta failed"))
        );
    }

    #[tokio::test]
    async fn batch_rejects_nested_batch_as_partial_failure() {
        let outcome = dispatch_tool_call(
            &dispatch_context(),
            "batch".to_string(),
            json!({
                "tool_calls": [
                    {"tool": "batch", "parameters": {"tool_calls": []}}
                ]
            }),
            None,
        )
        .await;

        assert!(outcome.record.output.is_success());
        let value = outcome.record.output.value_for_projection();
        let first = value
            .get("results")
            .and_then(|value| value.as_array())
            .and_then(|items| items.first())
            .expect("first result");
        assert_eq!(
            first.get("error"),
            Some(&json!("Tool 'batch' is not allowed inside batch"))
        );
    }

    #[tokio::test]
    async fn batch_marks_overflow_calls_as_failures() {
        let tool_calls = (0..26)
            .map(|_| json!({"tool": "alpha", "parameters": {}}))
            .collect::<Vec<_>>();

        let outcome = dispatch_tool_call(
            &dispatch_context(),
            "batch".to_string(),
            json!({ "tool_calls": tool_calls }),
            None,
        )
        .await;

        assert!(!outcome.record.output.is_success());
        let value = outcome.record.output.value_for_projection();
        let error = value
            .get("message")
            .and_then(|value| value.as_str())
            .expect("string error result");
        assert!(
            error.contains("tool_calls") && error.contains("items <= 25"),
            "{error}",
        );
    }

    #[tokio::test]
    async fn batch_calls_make_progress_concurrently() {
        let barrier = Arc::new(Barrier::new(2));
        let started = Arc::new(AtomicUsize::new(0));
        let outcome = dispatch_tool_call(
            &parallel_dispatch_context(Arc::clone(&barrier), Arc::clone(&started)),
            "batch".to_string(),
            json!({
                "tool_calls": [
                    {"tool": "probe_a", "parameters": {}},
                    {"tool": "probe_b", "parameters": {}}
                ]
            }),
            None,
        )
        .await;

        assert!(outcome.record.output.is_success());
        assert_eq!(started.load(Ordering::SeqCst), 2);
        let value = outcome.record.output.value_for_projection();
        let results = value
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|item| item.get("success").and_then(|value| value.as_bool()) == Some(true))
        );
    }

    /// A tool provider whose tools are marked [`ToolExecutionMode::Serial`]
    /// and log (start, end) instants around a sleep into a shared `Mutex`.
    struct SerialProbeTools {
        /// (tool_name, start_instant, end_instant)
        log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for SerialProbeTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![
                test_tool("serial_a", ToolExecutionMode::Serial),
                test_tool("serial_b", ToolExecutionMode::Serial),
            ])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            contract_from(
                vec![
                    test_tool("serial_a", ToolExecutionMode::Serial),
                    test_tool("serial_b", ToolExecutionMode::Serial),
                ],
                name,
            )
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            let start = Instant::now();
            // Sleep long enough that if the two tools *were* dispatched
            // concurrently, their windows would overlap by a detectable
            // margin.
            tokio::time::sleep(Duration::from_millis(40)).await;
            let end = Instant::now();
            self.log
                .lock()
                .expect("serial probe log")
                .push((call.name.to_string(), start, end));
            ToolResult::ok(json!(call.name))
        }
    }

    fn serial_dispatch_context(
        log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
    ) -> ToolDispatchContext<'static> {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::new(SerialProbeTools { log }));
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        }
    }

    /// Two Serial tools in the same batch must not interleave: the second
    /// call's start instant must be at or after the first call's end
    /// instant.
    #[tokio::test]
    async fn serial_tools_do_not_interleave() {
        let log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let context = Arc::new(serial_dispatch_context(Arc::clone(&log)));

        let specs = vec![
            ParallelToolCallSpec {
                index: 0,
                tool_name: "serial_a".to_string(),
                args: json!({}),
            },
            ParallelToolCallSpec {
                index: 1,
                tool_name: "serial_b".to_string(),
                args: json!({}),
            },
        ];

        let outcomes = dispatch_parallel_tool_calls(context, specs, None).await;

        assert_eq!(outcomes.len(), 2);
        assert!(
            outcomes
                .iter()
                .all(|outcome| outcome.record.output.is_success())
        );
        // Outcomes are sorted by original index.
        assert_eq!(outcomes[0].index, 0);
        assert_eq!(outcomes[1].index, 1);
        assert_eq!(outcomes[0].record.tool, "serial_a");
        assert_eq!(outcomes[1].record.tool, "serial_b");

        let entries = log.lock().expect("log").clone();
        assert_eq!(entries.len(), 2, "both serial tools must have executed");
        // Sort entries by start time so we compare the first-to-run vs
        // second-to-run regardless of which tool happened to go first.
        let mut sorted = entries;
        sorted.sort_by_key(|(_, start, _)| *start);
        let (first_name, _first_start, first_end) = &sorted[0];
        let (second_name, second_start, _second_end) = &sorted[1];
        assert_ne!(first_name, second_name, "both tools should have run");
        assert!(
            second_start >= first_end,
            "serial tool ranges must not overlap: first ended at {:?}, second started at {:?}",
            first_end,
            second_start,
        );
    }

    struct SerialRetryProbeTools {
        log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
        attempts_a: Arc<AtomicUsize>,
        attempts_b: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for SerialRetryProbeTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![
                test_tool("serial_retry_a", ToolExecutionMode::Serial)
                    .with_retry_policy(ToolRetryPolicy::safe(2, 0, 0)),
                test_tool("serial_retry_b", ToolExecutionMode::Serial)
                    .with_retry_policy(ToolRetryPolicy::safe(2, 0, 0)),
            ])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            contract_from(
                vec![
                    test_tool("serial_retry_a", ToolExecutionMode::Serial)
                        .with_retry_policy(ToolRetryPolicy::safe(2, 0, 0)),
                    test_tool("serial_retry_b", ToolExecutionMode::Serial)
                        .with_retry_policy(ToolRetryPolicy::safe(2, 0, 0)),
                ],
                name,
            )
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            let start = Instant::now();
            tokio::time::sleep(Duration::from_millis(20)).await;
            let end = Instant::now();
            self.log
                .lock()
                .expect("serial retry log")
                .push((call.name.to_string(), start, end));

            let attempt = match call.name {
                "serial_retry_a" => self.attempts_a.fetch_add(1, Ordering::SeqCst) + 1,
                "serial_retry_b" => self.attempts_b.fetch_add(1, Ordering::SeqCst) + 1,
                _ => 1,
            };
            if attempt == 1 {
                ToolResult::retryable_failure(
                    crate::ToolFailureClass::External,
                    "transient",
                    "transient failure",
                    Some(0),
                )
            } else {
                ToolResult::ok(json!(call.name))
            }
        }
    }

    #[tokio::test]
    async fn serial_tool_retries_do_not_overlap_other_serial_calls() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let attempts_a = Arc::new(AtomicUsize::new(0));
        let attempts_b = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(SerialRetryProbeTools {
            log: Arc::clone(&log),
            attempts_a: Arc::clone(&attempts_a),
            attempts_b: Arc::clone(&attempts_b),
        });
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(provider);
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        let context = Arc::new(ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });

        let outcomes = dispatch_parallel_tool_calls(
            context,
            vec![
                ParallelToolCallSpec {
                    index: 0,
                    tool_name: "serial_retry_a".to_string(),
                    args: json!({}),
                },
                ParallelToolCallSpec {
                    index: 1,
                    tool_name: "serial_retry_b".to_string(),
                    args: json!({}),
                },
            ],
            None,
        )
        .await;

        assert!(
            outcomes
                .iter()
                .all(|outcome| outcome.record.output.is_success())
        );
        assert_eq!(attempts_a.load(Ordering::SeqCst), 2);
        assert_eq!(attempts_b.load(Ordering::SeqCst), 2);

        let mut entries = log.lock().expect("serial retry log").clone();
        entries.sort_by_key(|(_, start, _)| *start);
        assert_eq!(entries.len(), 4);
        for window in entries.windows(2) {
            assert!(
                window[1].1 >= window[0].2,
                "serial retry windows must not overlap: {:?} then {:?}",
                window[0],
                window[1],
            );
        }
    }

    /// When a batch contains a mix of parallel and serial tools, the
    /// parallel-safe ones should still run concurrently with each other
    /// (verified via a Barrier), and the serial one should run separately
    /// without interleaving with any parallel peer's window.
    #[tokio::test]
    async fn mixed_batch_runs_parallel_tools_concurrently_and_serial_alone() {
        struct MixedTools {
            barrier: Arc<Barrier>,
            serial_window: Arc<std::sync::Mutex<Option<(Instant, Instant)>>>,
            parallel_windows: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
        }

        #[async_trait::async_trait]
        impl ToolProvider for MixedTools {
            fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
                manifests(vec![
                    test_tool("par_a", ToolExecutionMode::Parallel),
                    test_tool("par_b", ToolExecutionMode::Parallel),
                    test_tool("ser", ToolExecutionMode::Serial),
                ])
            }

            fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
                contract_from(
                    vec![
                        test_tool("par_a", ToolExecutionMode::Parallel),
                        test_tool("par_b", ToolExecutionMode::Parallel),
                        test_tool("ser", ToolExecutionMode::Serial),
                    ],
                    name,
                )
            }

            async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
                let name = call.name;
                if name == "ser" {
                    let start = Instant::now();
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    let end = Instant::now();
                    *self.serial_window.lock().expect("serial window") = Some((start, end));
                    ToolResult::ok(json!(name))
                } else {
                    let start = Instant::now();
                    // Block until both parallel tools have reached this
                    // barrier — proves they're running concurrently.
                    let waited = timeout(Duration::from_millis(200), self.barrier.wait()).await;
                    let end = Instant::now();
                    self.parallel_windows
                        .lock()
                        .expect("parallel windows")
                        .push((name.to_string(), start, end));
                    match waited {
                        Ok(_) => ToolResult::ok(json!(name)),
                        Err(_) => ToolResult::err_fmt(format!("{name} did not overlap with peer")),
                    }
                }
            }
        }

        let barrier = Arc::new(Barrier::new(2));
        let serial_window = Arc::new(std::sync::Mutex::new(None));
        let parallel_windows = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (event_tx, _event_rx) = mpsc::channel(8);
        let provider = Arc::new(MixedTools {
            barrier: Arc::clone(&barrier),
            serial_window: Arc::clone(&serial_window),
            parallel_windows: Arc::clone(&parallel_windows),
        });
        let plugins = test_plugins(provider);
        let tools = plugins.tools();
        let surface = plugins
            .tool_surface("session", ExecutionMode::standard())
            .expect("tool surface");
        let context = Arc::new(ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            processes: Arc::new(crate::UnavailableProcessService),
            effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController::default(),
            )),
            direct_completions: crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
            tool_effect_metadata: None,
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            turn_context: crate::TurnContext::default(),
        });

        let specs = vec![
            ParallelToolCallSpec {
                index: 0,
                tool_name: "par_a".to_string(),
                args: json!({}),
            },
            ParallelToolCallSpec {
                index: 1,
                tool_name: "ser".to_string(),
                args: json!({}),
            },
            ParallelToolCallSpec {
                index: 2,
                tool_name: "par_b".to_string(),
                args: json!({}),
            },
        ];

        let outcomes = dispatch_parallel_tool_calls(context, specs, None).await;

        assert_eq!(outcomes.len(), 3);
        assert!(
            outcomes
                .iter()
                .all(|outcome| outcome.record.output.is_success()),
            "all tools should succeed: {:?}",
            outcomes
                .iter()
                .map(|outcome| (&outcome.record.tool, outcome.record.output.is_success()))
                .collect::<Vec<_>>()
        );

        let pw = parallel_windows.lock().expect("parallel windows");
        assert_eq!(pw.len(), 2);
        let sw = serial_window
            .lock()
            .expect("serial window")
            .expect("serial window recorded");

        // The serial tool's window must not overlap either parallel
        // tool's window (Option A: serial runs after parallel).
        for (name, p_start, p_end) in pw.iter() {
            assert!(
                sw.0 >= *p_end || sw.1 <= *p_start,
                "serial window {:?} overlaps parallel window {} {:?}..{:?}",
                sw,
                name,
                p_start,
                p_end,
            );
        }
    }
}
