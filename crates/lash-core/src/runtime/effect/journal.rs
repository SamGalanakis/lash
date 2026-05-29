use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::LlmResponse;
use crate::llm::transport::LlmTransportError;
use crate::runtime::session_manager::{CurrentSessionCapability, UsageCapability};
use crate::sansio::LlmCallError;
use crate::{LlmRequest as CoreLlmRequest, PluginError, session_model::TokenUsage};

use super::envelope::{CausalRef, RuntimeEffectEnvelope, RuntimeEffectOutcome, RuntimeInvocation};
use super::executor::{
    RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectLocalExecutor,
};

// =============================================================================
// LLM trace helpers
// =============================================================================

pub(crate) fn token_usage_from_llm(usage: &crate::llm::types::LlmUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    }
}

pub(crate) fn emit_llm_trace_started(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    request: &CoreLlmRequest,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallStarted {
            request: crate::trace::trace_llm_request(request),
        },
    );
}

pub(crate) fn emit_llm_trace_completed(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    response: &LlmResponse,
    duration_ms: u64,
    stream_summary: Option<serde_json::Value>,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallCompleted {
            response: crate::trace::trace_llm_response(
                response.full_text.clone(),
                duration_ms,
                Some(response.terminal_reason),
                crate::trace::trace_output_parts(&response.parts),
            ),
            usage: Some(crate::trace::trace_usage_from_llm(&response.usage)),
            provider_usage: response.provider_usage.clone(),
            stream_summary,
        },
    );
}

pub(crate) struct LlmTraceFailure {
    message: String,
    retryable: bool,
    terminal_reason: crate::LlmTerminalReason,
    code: Option<String>,
    raw: Option<String>,
}

impl From<&LlmTransportError> for LlmTraceFailure {
    fn from(err: &LlmTransportError) -> Self {
        Self {
            message: err.message.clone(),
            retryable: err.retryable,
            terminal_reason: err.terminal_reason,
            code: err.code.clone(),
            raw: err.raw.clone(),
        }
    }
}

impl From<&LlmCallError> for LlmTraceFailure {
    fn from(err: &LlmCallError) -> Self {
        Self {
            message: err.message.clone(),
            retryable: err.retryable,
            terminal_reason: err.terminal_reason,
            code: err.code.clone(),
            raw: err.raw.clone(),
        }
    }
}

pub(crate) fn emit_llm_trace_failed(
    trace_sink: &Option<Arc<dyn lash_trace::TraceSink>>,
    base_context: &lash_trace::TraceContext,
    context: lash_trace::TraceContext,
    failure: LlmTraceFailure,
    stream_summary: Option<serde_json::Value>,
) {
    crate::trace::emit_trace(
        trace_sink,
        base_context,
        context,
        lash_trace::TraceEvent::LlmCallFailed {
            error: lash_trace::TraceError {
                message: failure.message,
                retryable: failure.retryable,
                terminal_reason: Some(failure.terminal_reason.code().to_string()),
                code: failure.code,
                raw: failure.raw,
            },
            stream_summary,
        },
    );
}

pub(crate) fn llm_call_error_from_transport(err: LlmTransportError) -> LlmCallError {
    LlmCallError {
        message: err.message,
        retryable: err.retryable,
        raw: err.raw,
        code: err.code,
        terminal_reason: err.terminal_reason,
        request_body: err.request_body,
    }
}

// =============================================================================
// Direct-completion outcome plumbing
// =============================================================================

/// Applies a journaled direct-effect outcome: records usage/trace against the
/// session and yields the raw provider response. Both the text-only
/// (`DirectCompletion`) and full-output (`DirectLlmCompletion`) client methods
/// project from this single result.
pub(crate) async fn apply_direct_outcome(
    current: &CurrentSessionCapability,
    usage_capability: &UsageCapability,
    request: &CoreLlmRequest,
    usage_source: &str,
    caused_by: Option<&CausalRef>,
    outcome: RuntimeEffectOutcome,
) -> Result<(LlmResponse, TokenUsage), PluginError> {
    let result = outcome
        .into_direct_response()
        .map_err(|err| PluginError::Session(err.to_string()))?;
    apply_direct_llm_result(
        current,
        usage_capability,
        request,
        usage_source,
        &request.model.clone(),
        caused_by,
        result,
    )
    .await
}

async fn apply_direct_llm_result(
    current: &CurrentSessionCapability,
    usage_capability: &UsageCapability,
    request: &CoreLlmRequest,
    usage_source: &str,
    usage_model: &str,
    caused_by: Option<&CausalRef>,
    result: Result<LlmResponse, LlmCallError>,
) -> Result<(LlmResponse, TokenUsage), PluginError> {
    let llm_call_id = emit_direct_llm_trace_started(current, request, caused_by);
    match result {
        Ok(response) => {
            emit_direct_llm_trace_completed(current, llm_call_id.as_deref(), caused_by, &response);
            let usage = token_usage_from_llm(&response.usage);
            usage_capability.record_token_usage(usage_source, usage_model, &usage);
            usage_capability
                .persist_current_usage_ledger(current)
                .await?;
            Ok((response, usage))
        }
        Err(err) => {
            emit_direct_llm_trace_failed(current, llm_call_id.as_deref(), caused_by, &err);
            Err(PluginError::Session(err.message))
        }
    }
}

fn emit_direct_llm_trace_started(
    current: &CurrentSessionCapability,
    request: &CoreLlmRequest,
    caused_by: Option<&CausalRef>,
) -> Option<String> {
    current.host.core.trace_sink.as_ref()?;
    let llm_call_id = uuid::Uuid::new_v4().to_string();
    emit_llm_trace_started(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(&current.session_id, Some(&llm_call_id), caused_by),
        request,
    );
    Some(llm_call_id)
}

fn emit_direct_llm_trace_completed(
    current: &CurrentSessionCapability,
    llm_call_id: Option<&str>,
    caused_by: Option<&CausalRef>,
    response: &LlmResponse,
) {
    let Some(llm_call_id) = llm_call_id else {
        return;
    };
    emit_llm_trace_completed(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(&current.session_id, Some(llm_call_id), caused_by),
        response,
        0,
        None,
    );
}

fn emit_direct_llm_trace_failed(
    current: &CurrentSessionCapability,
    llm_call_id: Option<&str>,
    caused_by: Option<&CausalRef>,
    err: &LlmCallError,
) {
    let Some(llm_call_id) = llm_call_id else {
        return;
    };
    emit_llm_trace_failed(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(&current.session_id, Some(llm_call_id), caused_by),
        LlmTraceFailure::from(err),
        None,
    );
}

fn direct_trace_context(
    session_id: &str,
    llm_call_id: Option<&str>,
    caused_by: Option<&CausalRef>,
) -> lash_trace::TraceContext {
    let mut context = lash_trace::TraceContext::default().for_session(session_id.to_string());
    if let Some(llm_call_id) = llm_call_id {
        context = context.for_llm_call(llm_call_id.to_string());
    }
    if let Some(caused_by) = caused_by {
        context = crate::trace::trace_context_with_causal_ref(context, caused_by);
    }
    context
}

// =============================================================================
// Per-turn effect journaling (durable replay)
// =============================================================================

pub(crate) async fn renew_runtime_turn_lease_for_effect(
    store: &(dyn crate::RuntimePersistence + '_),
    lease: &crate::RuntimeTurnLease,
    invocation: &RuntimeInvocation,
) -> Result<crate::RuntimeTurnLease, RuntimeEffectControllerError> {
    require_matching_turn_lease(Some(lease), invocation)?;
    store
        .renew_runtime_turn_lease(lease, crate::runtime::RUNTIME_TURN_LEASE_TTL_MS)
        .await
        .map_err(RuntimeEffectControllerError::from)
}

pub(crate) async fn execute_effect_with_journal(
    store: Option<&(dyn crate::RuntimePersistence + '_)>,
    lease: Option<&crate::RuntimeTurnLease>,
    controller: &dyn RuntimeEffectController,
    envelope: RuntimeEffectEnvelope,
    local_executor: RuntimeEffectLocalExecutor<'_>,
) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
    let Some(turn_id) = envelope.invocation.scope.turn_id.clone() else {
        return controller.execute_effect(envelope, local_executor).await;
    };
    let Some(store) = store else {
        return controller.execute_effect(envelope, local_executor).await;
    };
    let mut active_lease = require_matching_turn_lease(lease, &envelope.invocation)?;
    let envelope_hash = envelope.stable_hash()?;
    let replay_key = envelope
        .invocation
        .replay_key()
        .ok_or_else(|| {
            RuntimeEffectControllerError::new(
                "runtime_effect_replay_required",
                "runtime effect envelope requires replay.key",
            )
        })?
        .to_string();
    if let Some(record) = store
        .load_runtime_effect_outcome(&envelope.invocation.scope.session_id, &turn_id, &replay_key)
        .await
        .map_err(RuntimeEffectControllerError::from)?
    {
        if record.envelope_hash != envelope_hash {
            return Err(RuntimeEffectControllerError::new(
                "runtime_effect_journal_hash_mismatch",
                format!(
                    "recorded runtime effect `{}` has envelope hash `{}` but replay requested `{}`",
                    replay_key, record.envelope_hash, envelope_hash
                ),
            ));
        }
        return Ok(record.outcome);
    }

    let invocation = envelope.invocation.clone();
    let effect_kind = invocation.effect_kind().ok_or_else(|| {
        RuntimeEffectControllerError::new(
            "runtime_effect_invocation_subject",
            "runtime effect envelope subject must be an effect",
        )
    })?;
    let outcome = execute_pending_effect_with_lease_renewal(
        store,
        &mut active_lease,
        &invocation,
        controller,
        envelope,
        local_executor,
    )
    .await?;
    active_lease = renew_runtime_turn_lease_for_effect(store, &active_lease, &invocation).await?;
    store
        .save_runtime_effect_outcome(
            &active_lease,
            crate::RuntimeEffectJournalRecord {
                schema_version: crate::RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
                session_id: invocation.scope.session_id,
                turn_id,
                replay_key,
                envelope_hash,
                effect_kind,
                outcome: outcome.clone(),
                created_at_epoch_ms: current_epoch_ms(),
            },
        )
        .await
        .map_err(RuntimeEffectControllerError::from)?;
    Ok(outcome)
}

pub(crate) struct JournaledEffectInvocation<'a> {
    store: Option<&'a (dyn crate::RuntimePersistence + 'a)>,
    lease: Option<&'a crate::RuntimeTurnLease>,
    controller: &'a dyn RuntimeEffectController,
    envelope: RuntimeEffectEnvelope,
    local_executor: RuntimeEffectLocalExecutor<'a>,
}

impl<'a> JournaledEffectInvocation<'a> {
    pub(crate) fn new(
        store: Option<&'a (dyn crate::RuntimePersistence + 'a)>,
        lease: Option<&'a crate::RuntimeTurnLease>,
        controller: &'a dyn RuntimeEffectController,
        envelope: RuntimeEffectEnvelope,
        local_executor: RuntimeEffectLocalExecutor<'a>,
    ) -> Self {
        Self {
            store,
            lease,
            controller,
            envelope,
            local_executor,
        }
    }
}

pub(crate) async fn invoke_journaled_effect<T, E, F, Fut>(
    invocation: JournaledEffectInvocation<'_>,
    apply_outcome: F,
) -> Result<T, E>
where
    E: From<RuntimeEffectControllerError>,
    F: FnOnce(RuntimeEffectOutcome) -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let outcome = execute_effect_with_journal(
        invocation.store,
        invocation.lease,
        invocation.controller,
        invocation.envelope,
        invocation.local_executor,
    )
    .await
    .map_err(E::from)?;
    apply_outcome(outcome).await
}

async fn execute_pending_effect_with_lease_renewal(
    store: &(dyn crate::RuntimePersistence + '_),
    active_lease: &mut crate::RuntimeTurnLease,
    invocation: &RuntimeInvocation,
    controller: &dyn RuntimeEffectController,
    envelope: RuntimeEffectEnvelope,
    local_executor: RuntimeEffectLocalExecutor<'_>,
) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
    let pending = controller.execute_effect(envelope, local_executor);
    tokio::pin!(pending);
    loop {
        tokio::select! {
            outcome = &mut pending => return outcome,
            _ = tokio::time::sleep(pending_effect_lease_renew_interval()) => {
                *active_lease =
                    renew_runtime_turn_lease_for_effect(store, active_lease, invocation).await?;
            }
        }
    }
}

fn pending_effect_lease_renew_interval() -> Duration {
    Duration::from_millis(pending_effect_lease_renew_interval_ms())
}

#[cfg(test)]
fn pending_effect_lease_renew_interval_ms() -> u64 {
    25
}

#[cfg(not(test))]
fn pending_effect_lease_renew_interval_ms() -> u64 {
    30_000
}

fn require_matching_turn_lease(
    lease: Option<&crate::RuntimeTurnLease>,
    invocation: &RuntimeInvocation,
) -> Result<crate::RuntimeTurnLease, RuntimeEffectControllerError> {
    let replay_key = invocation.replay_key().unwrap_or("<missing replay>");
    let Some(turn_id) = invocation.scope.turn_id.as_deref() else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` does not carry a turn id for lease validation",
                replay_key
            ),
        ));
    };
    let Some(lease) = lease else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` for turn `{}` requires a runtime turn lease",
                replay_key, turn_id
            ),
        ));
    };
    if lease.session_id != invocation.scope.session_id || lease.turn_id != turn_id {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` lease targets `{}`/`{}` but metadata targets `{}`/`{}`",
                replay_key, lease.session_id, lease.turn_id, invocation.scope.session_id, turn_id
            ),
        ));
    }
    Ok(lease.clone())
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use crate::RuntimeEffectKind;

    #[test]
    fn direct_effect_invocation_preserves_runtime_scope() {
        let invocation = crate::runtime::causal::direct_effect_invocation(
            "s",
            "tool",
            "request:k".to_string(),
            None,
            None,
        );

        assert_eq!(invocation.scope.session_id, "s");
        assert_eq!(invocation.effect_kind(), Some(RuntimeEffectKind::Direct));
        assert!(
            invocation
                .replay_key()
                .expect("replay key")
                .starts_with("s:direct:tool:request:k")
        );
        assert!(invocation.checkpoint_hash.is_none());
    }

    #[test]
    fn tool_retry_sleep_invocation_preserves_parent_checkpoint_digest() {
        let mut parent = crate::runtime::causal::direct_effect_invocation(
            "s",
            "tool",
            "request:k".to_string(),
            Some("turn"),
            None,
        );
        parent.checkpoint_hash = Some("a".repeat(64));

        let sleep = crate::runtime::causal::tool_retry_sleep_invocation(&parent, "probe", 2);

        assert_eq!(sleep.effect_kind(), Some(RuntimeEffectKind::Sleep));
        assert_eq!(sleep.checkpoint_hash, parent.checkpoint_hash);
        assert!(
            sleep
                .replay_key()
                .expect("replay key")
                .ends_with(":probe:attempt:2:sleep")
        );
    }
}
