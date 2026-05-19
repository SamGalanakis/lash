use crate::plugin::{DirectCompletion, DirectLlmCompletion};
use crate::sansio::LlmCallError;
use crate::{
    DirectRequest, LlmRequest as CoreLlmRequest, LlmResponse, PluginError,
    session_model::TokenUsage,
};

use crate::runtime::session_manager::{CurrentSessionCapability, UsageCapability};

use super::envelope::RuntimeEffectOutcome;
use super::trace::{
    LlmTraceFailure, emit_llm_trace_completed, emit_llm_trace_failed, emit_llm_trace_started,
    token_usage_from_llm,
};

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
