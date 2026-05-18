use super::*;

impl DirectCompletionCapability {
    pub(in crate::runtime::session_manager) async fn direct_completion(
        &self,
        current: &CurrentSessionCapability,
        usage_capability: &UsageCapability,
        request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        self.invoke_direct_completion(current, usage_capability, request, usage_source)
            .await
    }

    async fn invoke_direct_completion(
        &self,
        current: &CurrentSessionCapability,
        usage_capability: &UsageCapability,
        mut request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        let active_effect_scope = crate::runtime::effect_host::active_effect_scope();
        let default_effect_host = Arc::clone(&current.host.core.effect_host);
        let effect_host = active_effect_scope
            .as_ref()
            .map(|scope| scope.host())
            .unwrap_or(default_effect_host.as_ref());
        let provider = current.policy.provider.clone();
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
        let llm_request =
            crate::direct::build_llm_request(&provider, request.clone(), model.clone());
        let discriminator = crate::runtime::effect_host::direct_request_discriminator(
            &request,
            request.idempotency_key.as_deref(),
            request.originating_tool_call_id.as_deref(),
        );
        let invocation = crate::runtime::effect_host::direct_effect_invocation(
            &current.session_id,
            usage_source,
            crate::RuntimeEffectKind::DirectCompletion,
            discriminator,
            active_effect_scope.as_ref().map(|scope| scope.turn_id()),
        );
        let usage_source = usage_source.to_string();
        effect_host
            .direct_completion(
                invocation,
                request,
                llm_request,
                model.clone(),
                usage_source.clone(),
                crate::DirectEffectLocalExecutor::new(
                    current.clone(),
                    usage_capability.clone(),
                    provider,
                ),
            )
            .await
    }

    pub(in crate::runtime::session_manager) async fn direct_llm_completion(
        &self,
        current: &CurrentSessionCapability,
        usage_capability: &UsageCapability,
        request: crate::LlmRequest,
        usage_source: &str,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        self.invoke_direct_llm_completion(current, usage_capability, request, usage_source)
            .await
    }

    async fn invoke_direct_llm_completion(
        &self,
        current: &CurrentSessionCapability,
        usage_capability: &UsageCapability,
        request: crate::LlmRequest,
        usage_source: &str,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        let active_effect_scope = crate::runtime::effect_host::active_effect_scope();
        let default_effect_host = Arc::clone(&current.host.core.effect_host);
        let effect_host = active_effect_scope
            .as_ref()
            .map(|scope| scope.host())
            .unwrap_or(default_effect_host.as_ref());
        let provider = current.policy.provider.clone();
        let request = crate::attachments::resolve_llm_request_attachments(
            request,
            current.host.core.attachment_store.as_ref(),
        )
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let discriminator =
            crate::runtime::effect_host::direct_request_discriminator(&request, None, None);
        let invocation = crate::runtime::effect_host::direct_effect_invocation(
            &current.session_id,
            usage_source,
            crate::RuntimeEffectKind::DirectLlmCompletion,
            discriminator,
            active_effect_scope.as_ref().map(|scope| scope.turn_id()),
        );
        let usage_source = usage_source.to_string();
        effect_host
            .direct_llm_completion(
                invocation,
                request,
                usage_source.clone(),
                crate::DirectEffectLocalExecutor::new(
                    current.clone(),
                    usage_capability.clone(),
                    provider,
                ),
            )
            .await
    }
}
