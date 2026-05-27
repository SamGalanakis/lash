use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::LlmResponse;
use crate::llm::transport::LlmTransportError;
use crate::plugin::{DirectCompletion, DirectLlmCompletion};
use crate::runtime::session_manager::{CurrentSessionCapability, UsageCapability};
use crate::sansio::{EffectId, LlmCallError};
use crate::{DirectRequest, LlmRequest as CoreLlmRequest, PluginError, session_model::TokenUsage};

use super::envelope::{
    EffectInvocationMetadata, EffectOrigin, RuntimeEffectEnvelope, RuntimeEffectKind,
    RuntimeEffectOutcome,
};
use super::executor::{
    RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectLocalExecutor,
};

// =============================================================================
// Effect identity / idempotency helpers
// =============================================================================

pub(crate) fn turn_idempotency_key(
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    protocol_iteration: usize,
    kind: RuntimeEffectKind,
    effect_id: EffectId,
) -> String {
    format!(
        "{session_id}:{turn_id}:{turn_index}:{protocol_iteration}:{}:{}",
        kind.as_str(),
        effect_id.0
    )
}

pub(crate) fn direct_effect_metadata(
    session_id: &str,
    usage_source: &str,
    effect_kind: RuntimeEffectKind,
    idempotency_discriminator: String,
    turn_id: Option<&str>,
) -> EffectInvocationMetadata {
    let origin = match effect_kind {
        RuntimeEffectKind::DirectCompletion => EffectOrigin::DirectCompletion {
            usage_source: usage_source.to_string(),
        },
        RuntimeEffectKind::DirectLlmCompletion => EffectOrigin::DirectLlmCompletion {
            usage_source: usage_source.to_string(),
        },
        _ => unreachable!("direct invocation requires a direct effect kind"),
    };
    let idempotency_key = match turn_id.filter(|value| !value.is_empty()) {
        Some(turn_id) => format!(
            "{session_id}:{turn_id}:direct:{}:{usage_source}:{idempotency_discriminator}",
            effect_kind.as_str()
        ),
        None => format!(
            "{session_id}:direct:{}:{usage_source}:{idempotency_discriminator}",
            effect_kind.as_str()
        ),
    };
    EffectInvocationMetadata {
        session_id: session_id.to_string(),
        origin,
        turn_id: turn_id.map(str::to_string),
        turn_index: None,
        protocol_iteration: None,
        effect_id: idempotency_discriminator,
        effect_kind,
        idempotency_key,
        turn_checkpoint_hash: None,
    }
}

pub(crate) fn tool_retry_sleep_metadata(
    parent: &EffectInvocationMetadata,
    tool_name: &str,
    attempt: u32,
) -> EffectInvocationMetadata {
    let effect_id = format!("{}:{tool_name}:attempt:{attempt}:sleep", parent.effect_id);
    let idempotency_key = format!(
        "{}:{tool_name}:attempt:{attempt}:sleep",
        parent.idempotency_key
    );
    EffectInvocationMetadata {
        session_id: parent.session_id.clone(),
        origin: parent.origin.clone(),
        turn_id: parent.turn_id.clone(),
        turn_index: parent.turn_index,
        protocol_iteration: parent.protocol_iteration,
        effect_id,
        effect_kind: RuntimeEffectKind::Sleep,
        idempotency_key,
        turn_checkpoint_hash: parent.turn_checkpoint_hash.clone(),
    }
}

pub(crate) fn direct_request_discriminator<T>(
    request: &T,
    explicit_key: Option<&str>,
    parent_tool_call_id: Option<&str>,
) -> Result<String, RuntimeEffectControllerError>
where
    T: Serialize,
{
    if let Some(explicit_key) = explicit_key.filter(|key| !key.is_empty()) {
        return Ok(match parent_tool_call_id.filter(|id| !id.is_empty()) {
            Some(parent) => format!("tool:{parent}:request:{explicit_key}"),
            None => format!("request:{explicit_key}"),
        });
    }
    let digest = crate::stable_hash::stable_json_sha256_hex(request).map_err(|err| {
        RuntimeEffectControllerError::new(
            "runtime_effect_discriminator",
            format!("failed to serialize runtime effect discriminator: {err}"),
        )
    })?;
    Ok(match parent_tool_call_id.filter(|id| !id.is_empty()) {
        Some(parent) => format!("tool:{parent}:sha256:{digest}"),
        None => format!("sha256:{digest}"),
    })
}

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

pub(crate) async fn apply_direct_completion_outcome(
    current: &CurrentSessionCapability,
    usage_capability: &UsageCapability,
    request: &DirectRequest,
    normalized_request: &CoreLlmRequest,
    model: &str,
    usage_source: &str,
    outcome: RuntimeEffectOutcome,
) -> Result<DirectCompletion, PluginError> {
    let result = outcome
        .into_direct_completion_response()
        .map_err(|err| PluginError::Session(err.to_string()))?;
    let (response, usage) = apply_direct_llm_result(
        current,
        usage_capability,
        normalized_request,
        usage_source,
        model,
        request.originating_tool_call_id.as_deref(),
        result,
    )
    .await?;
    Ok(DirectCompletion {
        text: response.full_text,
        usage,
    })
}

pub(crate) async fn apply_direct_llm_completion_outcome(
    current: &CurrentSessionCapability,
    usage_capability: &UsageCapability,
    request: &CoreLlmRequest,
    usage_source: &str,
    outcome: RuntimeEffectOutcome,
) -> Result<DirectLlmCompletion, PluginError> {
    let result = outcome
        .into_direct_llm_completion_response()
        .map_err(|err| PluginError::Session(err.to_string()))?;
    let model = request.model.clone();
    let (response, usage) = apply_direct_llm_result(
        current,
        usage_capability,
        request,
        usage_source,
        &model,
        None,
        result,
    )
    .await?;
    Ok(DirectLlmCompletion { response, usage })
}

async fn apply_direct_llm_result(
    current: &CurrentSessionCapability,
    usage_capability: &UsageCapability,
    request: &CoreLlmRequest,
    usage_source: &str,
    usage_model: &str,
    originating_tool_call_id: Option<&str>,
    result: Result<LlmResponse, LlmCallError>,
) -> Result<(LlmResponse, TokenUsage), PluginError> {
    let llm_call_id = emit_direct_llm_trace_started(current, request, originating_tool_call_id);
    match result {
        Ok(response) => {
            emit_direct_llm_trace_completed(
                current,
                llm_call_id.as_deref(),
                originating_tool_call_id,
                &response,
            );
            let usage = token_usage_from_llm(&response.usage);
            usage_capability.record_token_usage(usage_source, usage_model, &usage);
            usage_capability
                .persist_current_usage_ledger(current)
                .await?;
            Ok((response, usage))
        }
        Err(err) => {
            emit_direct_llm_trace_failed(
                current,
                llm_call_id.as_deref(),
                originating_tool_call_id,
                &err,
            );
            Err(PluginError::Session(err.message))
        }
    }
}

fn emit_direct_llm_trace_started(
    current: &CurrentSessionCapability,
    request: &CoreLlmRequest,
    originating_tool_call_id: Option<&str>,
) -> Option<String> {
    current.host.core.trace_sink.as_ref()?;
    let llm_call_id = uuid::Uuid::new_v4().to_string();
    emit_llm_trace_started(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(
            &current.session_id,
            Some(&llm_call_id),
            originating_tool_call_id,
        ),
        request,
    );
    Some(llm_call_id)
}

fn emit_direct_llm_trace_completed(
    current: &CurrentSessionCapability,
    llm_call_id: Option<&str>,
    originating_tool_call_id: Option<&str>,
    response: &LlmResponse,
) {
    let Some(llm_call_id) = llm_call_id else {
        return;
    };
    emit_llm_trace_completed(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(
            &current.session_id,
            Some(llm_call_id),
            originating_tool_call_id,
        ),
        response,
        0,
        None,
    );
}

fn emit_direct_llm_trace_failed(
    current: &CurrentSessionCapability,
    llm_call_id: Option<&str>,
    originating_tool_call_id: Option<&str>,
    err: &LlmCallError,
) {
    let Some(llm_call_id) = llm_call_id else {
        return;
    };
    emit_llm_trace_failed(
        &current.host.core.trace_sink,
        &current.host.core.trace_context,
        direct_trace_context(
            &current.session_id,
            Some(llm_call_id),
            originating_tool_call_id,
        ),
        LlmTraceFailure::from(err),
        None,
    );
}

fn direct_trace_context(
    session_id: &str,
    llm_call_id: Option<&str>,
    originating_tool_call_id: Option<&str>,
) -> lash_trace::TraceContext {
    let mut context = lash_trace::TraceContext::default().for_session(session_id.to_string());
    if let Some(llm_call_id) = llm_call_id {
        context = context.for_llm_call(llm_call_id.to_string());
    }
    if let Some(originating_tool_call_id) = originating_tool_call_id {
        context = context.for_originating_tool_call(originating_tool_call_id.to_string());
    }
    context
}

// =============================================================================
// Per-turn effect journaling (idempotent durable replay)
// =============================================================================

pub(crate) async fn renew_runtime_turn_lease_for_effect(
    store: &(dyn crate::RuntimePersistence + '_),
    lease: &crate::RuntimeTurnLease,
    metadata: &EffectInvocationMetadata,
) -> Result<crate::RuntimeTurnLease, RuntimeEffectControllerError> {
    require_matching_turn_lease(Some(lease), metadata)?;
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
    let Some(turn_id) = envelope.metadata.turn_id.clone() else {
        return controller.execute_effect(envelope, local_executor).await;
    };
    let Some(store) = store else {
        return controller.execute_effect(envelope, local_executor).await;
    };
    let mut active_lease = require_matching_turn_lease(lease, &envelope.metadata)?;
    if !matches!(envelope.metadata.origin, EffectOrigin::Turn) {
        active_lease =
            renew_runtime_turn_lease_for_effect(store, &active_lease, &envelope.metadata).await?;
    }
    let envelope_hash = envelope.stable_hash()?;
    if let Some(record) = store
        .load_runtime_effect_outcome(
            &envelope.metadata.session_id,
            &turn_id,
            &envelope.metadata.idempotency_key,
        )
        .await
        .map_err(RuntimeEffectControllerError::from)?
    {
        if record.envelope_hash != envelope_hash {
            return Err(RuntimeEffectControllerError::new(
                "runtime_effect_journal_hash_mismatch",
                format!(
                    "recorded runtime effect `{}` has envelope hash `{}` but replay requested `{}`",
                    envelope.metadata.idempotency_key, record.envelope_hash, envelope_hash
                ),
            ));
        }
        return Ok(record.outcome);
    }

    let metadata = envelope.metadata.clone();
    let effect_kind = metadata.effect_kind;
    let outcome = execute_pending_effect_with_lease_renewal(
        store,
        &mut active_lease,
        &metadata,
        controller,
        envelope,
        local_executor,
    )
    .await?;
    active_lease = renew_runtime_turn_lease_for_effect(store, &active_lease, &metadata).await?;
    store
        .save_runtime_effect_outcome(
            &active_lease,
            crate::RuntimeEffectJournalRecord {
                schema_version: crate::RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
                session_id: metadata.session_id,
                turn_id,
                idempotency_key: metadata.idempotency_key,
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

async fn execute_pending_effect_with_lease_renewal(
    store: &(dyn crate::RuntimePersistence + '_),
    active_lease: &mut crate::RuntimeTurnLease,
    metadata: &EffectInvocationMetadata,
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
                    renew_runtime_turn_lease_for_effect(store, active_lease, metadata).await?;
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
    metadata: &EffectInvocationMetadata,
) -> Result<crate::RuntimeTurnLease, RuntimeEffectControllerError> {
    let Some(turn_id) = metadata.turn_id.as_deref() else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` does not carry a turn id for lease validation",
                metadata.idempotency_key
            ),
        ));
    };
    let Some(lease) = lease else {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` for turn `{}` requires a runtime turn lease",
                metadata.idempotency_key, turn_id
            ),
        ));
    };
    if lease.session_id != metadata.session_id || lease.turn_id != turn_id {
        return Err(RuntimeEffectControllerError::new(
            "runtime_turn_lease_required",
            format!(
                "runtime effect `{}` lease targets `{}`/`{}` but metadata targets `{}`/`{}`",
                metadata.idempotency_key,
                lease.session_id,
                lease.turn_id,
                metadata.session_id,
                turn_id
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
    use super::*;

    #[test]
    fn direct_effect_metadata_preserves_metadata_shape() {
        let metadata = direct_effect_metadata(
            "s",
            "tool",
            RuntimeEffectKind::DirectCompletion,
            "request:k".to_string(),
            None,
        );

        assert_eq!(metadata.session_id, "s");
        assert_eq!(
            metadata.origin,
            EffectOrigin::DirectCompletion {
                usage_source: "tool".to_string()
            }
        );
        assert!(
            metadata
                .idempotency_key
                .starts_with("s:direct:direct_completion:tool:request:k")
        );
        assert!(metadata.turn_checkpoint_hash.is_none());
    }

    #[test]
    fn tool_retry_sleep_metadata_preserves_parent_checkpoint_digest() {
        let mut parent = direct_effect_metadata(
            "s",
            "tool",
            RuntimeEffectKind::DirectCompletion,
            "request:k".to_string(),
            Some("turn"),
        );
        parent.turn_checkpoint_hash = Some("a".repeat(64));

        let sleep = tool_retry_sleep_metadata(&parent, "probe", 2);

        assert_eq!(sleep.effect_kind, RuntimeEffectKind::Sleep);
        assert_eq!(sleep.turn_checkpoint_hash, parent.turn_checkpoint_hash);
        assert!(sleep.idempotency_key.ends_with(":probe:attempt:2:sleep"));
    }
}
