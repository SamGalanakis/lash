use super::*;

impl DirectCompletionCapability {
    pub(in crate::runtime::session_manager) async fn direct_completion(
        &self,
        current: &CurrentSessionCapability,
        usage_capability: &UsageCapability,
        mut request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        let mut provider = current.policy.provider.clone();
        let model = if let Some(selection) = provider.default_agent_model(&request.model) {
            if request.model_variant.is_none() {
                request.model_variant = selection.variant;
            }
            provider.resolve_model(&selection.model)
        } else {
            provider.resolve_model(&request.model)
        };
        if let Some(variant) = request.model_variant.as_deref() {
            provider
                .validate_variant(&model, variant)
                .map_err(crate::PluginError::Session)?;
        }
        let originating_tool_call_id = request.originating_tool_call_id.clone();
        let trace_context = |id: Option<&str>| {
            let mut ctx =
                lash_trace::TraceContext::default().for_session(current.session_id.clone());
            if let Some(id) = id {
                ctx = ctx.for_llm_call(id.to_string());
            }
            if let Some(tcid) = &originating_tool_call_id {
                ctx = ctx.for_originating_tool_call(tcid.clone());
            }
            ctx
        };
        let llm_request = crate::direct::build_llm_request(&provider, request, model.clone());
        let llm_call_id = if current.host.core.trace_sink.is_some() {
            let id = uuid::Uuid::new_v4().to_string();
            crate::trace::emit_trace(
                &current.host.core.trace_sink,
                &current.host.core.trace_context,
                trace_context(Some(&id)),
                lash_trace::TraceEvent::LlmCallStarted {
                    request: crate::trace::trace_llm_request(&llm_request),
                },
            );
            Some(id)
        } else {
            None
        };
        let response = match provider.complete(llm_request).await {
            Ok(response) => response,
            Err(err) => {
                if let Some(llm_call_id) = llm_call_id {
                    crate::trace::emit_trace(
                        &current.host.core.trace_sink,
                        &current.host.core.trace_context,
                        trace_context(Some(&llm_call_id)),
                        lash_trace::TraceEvent::LlmCallFailed {
                            error: lash_trace::TraceError {
                                message: err.message.clone(),
                                retryable: err.retryable,
                                code: err.code.clone(),
                                raw: err.raw.clone(),
                            },
                            stream_summary: None,
                        },
                    );
                }
                return Err(crate::PluginError::Session(err.message.clone()));
            }
        };
        if let Some(llm_call_id) = llm_call_id {
            crate::trace::emit_trace(
                &current.host.core.trace_sink,
                &current.host.core.trace_context,
                trace_context(Some(&llm_call_id)),
                lash_trace::TraceEvent::LlmCallCompleted {
                    response: crate::trace::trace_llm_response(
                        response.full_text.clone(),
                        0,
                        crate::trace::trace_output_parts(&response.parts),
                    ),
                    usage: Some(crate::trace::trace_usage_from_llm(&response.usage)),
                    provider_usage: response.provider_usage.clone(),
                    stream_summary: None,
                },
            );
        }
        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            cached_input_tokens: response.usage.cached_input_tokens,
            reasoning_tokens: response.usage.reasoning_tokens,
        };
        usage_capability.record_token_usage(usage_source, &model, &usage);
        usage_capability
            .persist_current_usage_ledger(current)
            .await?;
        Ok(crate::DirectCompletion {
            text: response.full_text,
            usage,
        })
    }

    pub(in crate::runtime::session_manager) async fn direct_llm_completion(
        &self,
        current: &CurrentSessionCapability,
        usage_capability: &UsageCapability,
        request: crate::LlmRequest,
        usage_source: &str,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        let mut provider = current.policy.provider.clone();
        let request = crate::attachments::resolve_llm_request_attachments(
            request,
            current.host.core.attachment_store.as_ref(),
        )
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let model = request.model.clone();
        let llm_call_id = if current.host.core.trace_sink.is_some() {
            let id = uuid::Uuid::new_v4().to_string();
            crate::trace::emit_trace(
                &current.host.core.trace_sink,
                &current.host.core.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(current.session_id.clone())
                    .for_llm_call(id.clone()),
                lash_trace::TraceEvent::LlmCallStarted {
                    request: crate::trace::trace_llm_request(&request),
                },
            );
            Some(id)
        } else {
            None
        };
        let response = match provider.complete(request).await {
            Ok(response) => response,
            Err(err) => {
                if let Some(llm_call_id) = llm_call_id {
                    crate::trace::emit_trace(
                        &current.host.core.trace_sink,
                        &current.host.core.trace_context,
                        lash_trace::TraceContext::default()
                            .for_session(current.session_id.clone())
                            .for_llm_call(llm_call_id),
                        lash_trace::TraceEvent::LlmCallFailed {
                            error: lash_trace::TraceError {
                                message: err.message.clone(),
                                retryable: err.retryable,
                                code: err.code.clone(),
                                raw: err.raw.clone(),
                            },
                            stream_summary: None,
                        },
                    );
                }
                return Err(crate::PluginError::Session(err.message.clone()));
            }
        };
        if let Some(llm_call_id) = llm_call_id {
            crate::trace::emit_trace(
                &current.host.core.trace_sink,
                &current.host.core.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(current.session_id.clone())
                    .for_llm_call(llm_call_id),
                lash_trace::TraceEvent::LlmCallCompleted {
                    response: crate::trace::trace_llm_response(
                        response.full_text.clone(),
                        0,
                        crate::trace::trace_output_parts(&response.parts),
                    ),
                    usage: Some(crate::trace::trace_usage_from_llm(&response.usage)),
                    provider_usage: response.provider_usage.clone(),
                    stream_summary: None,
                },
            );
        }
        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            cached_input_tokens: response.usage.cached_input_tokens,
            reasoning_tokens: response.usage.reasoning_tokens,
        };
        usage_capability.record_token_usage(usage_source, &model, &usage);
        usage_capability
            .persist_current_usage_ledger(current)
            .await?;
        Ok(crate::DirectLlmCompletion { response, usage })
    }
}
