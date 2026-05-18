use super::*;

type DirectCompletionFuture<'a, T> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, crate::PluginError>> + Send + 'a>>;

trait DirectCompletionInvoker: Send + Sync {
    fn direct_completion<'a>(
        &'a self,
        request: crate::DirectRequest,
        usage_source: &'a str,
    ) -> DirectCompletionFuture<'a, crate::DirectCompletion>;

    fn direct_llm_completion<'a>(
        &'a self,
        request: crate::LlmRequest,
        usage_source: &'a str,
    ) -> DirectCompletionFuture<'a, crate::DirectLlmCompletion>;
}

#[derive(Clone)]
pub struct DirectCompletionClient<'run> {
    inner: Arc<dyn DirectCompletionInvoker + 'run>,
}

impl<'run> DirectCompletionClient<'run> {
    pub(super) fn runtime(
        manager: Arc<RuntimeSessionManager>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        turn_id: Option<String>,
    ) -> Self {
        Self {
            inner: Arc::new(RuntimeDirectCompletionInvoker {
                manager,
                effect_controller,
                turn_id,
            }),
        }
    }

    pub async fn direct_completion(
        &self,
        request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        self.inner.direct_completion(request, usage_source).await
    }

    pub async fn direct_llm_completion(
        &self,
        request: crate::LlmRequest,
        usage_source: &str,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        self.inner
            .direct_llm_completion(request, usage_source)
            .await
    }

    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(UnavailableDirectCompletionInvoker {
                message: message.into(),
            }),
        }
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn from_fn<F>(invoke: F) -> Self
    where
        F: Fn(crate::DirectRequest, String) -> Result<crate::DirectCompletion, crate::PluginError>
            + Send
            + Sync
            + 'static,
    {
        Self {
            inner: Arc::new(TestDirectCompletionInvoker {
                invoke: Arc::new(invoke),
            }),
        }
    }
}

struct RuntimeDirectCompletionInvoker<'run> {
    manager: Arc<RuntimeSessionManager>,
    effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    turn_id: Option<String>,
}

impl DirectCompletionInvoker for RuntimeDirectCompletionInvoker<'_> {
    fn direct_completion<'a>(
        &'a self,
        request: crate::DirectRequest,
        usage_source: &'a str,
    ) -> DirectCompletionFuture<'a, crate::DirectCompletion> {
        Box::pin(async move {
            self.manager
                .direct
                .invoke_direct_completion(
                    &self.manager.current,
                    &self.manager.usage,
                    request,
                    usage_source,
                    self.effect_controller.as_controller(),
                    self.turn_id.as_deref(),
                )
                .await
        })
    }

    fn direct_llm_completion<'a>(
        &'a self,
        request: crate::LlmRequest,
        usage_source: &'a str,
    ) -> DirectCompletionFuture<'a, crate::DirectLlmCompletion> {
        Box::pin(async move {
            self.manager
                .direct
                .invoke_direct_llm_completion(
                    &self.manager.current,
                    &self.manager.usage,
                    request,
                    usage_source,
                    self.effect_controller.as_controller(),
                    self.turn_id.as_deref(),
                )
                .await
        })
    }
}

struct UnavailableDirectCompletionInvoker {
    message: String,
}

impl DirectCompletionInvoker for UnavailableDirectCompletionInvoker {
    fn direct_completion<'a>(
        &'a self,
        _request: crate::DirectRequest,
        _usage_source: &'a str,
    ) -> DirectCompletionFuture<'a, crate::DirectCompletion> {
        Box::pin(async move { Err(crate::PluginError::Session(self.message.clone())) })
    }

    fn direct_llm_completion<'a>(
        &'a self,
        _request: crate::LlmRequest,
        _usage_source: &'a str,
    ) -> DirectCompletionFuture<'a, crate::DirectLlmCompletion> {
        Box::pin(async move { Err(crate::PluginError::Session(self.message.clone())) })
    }
}

#[cfg(any(test, feature = "testing"))]
struct TestDirectCompletionInvoker {
    invoke: Arc<
        dyn Fn(crate::DirectRequest, String) -> Result<crate::DirectCompletion, crate::PluginError>
            + Send
            + Sync,
    >,
}

#[cfg(any(test, feature = "testing"))]
impl DirectCompletionInvoker for TestDirectCompletionInvoker {
    fn direct_completion<'a>(
        &'a self,
        request: crate::DirectRequest,
        usage_source: &'a str,
    ) -> DirectCompletionFuture<'a, crate::DirectCompletion> {
        let invoke = Arc::clone(&self.invoke);
        let usage_source = usage_source.to_string();
        Box::pin(async move { invoke(request, usage_source) })
    }

    fn direct_llm_completion<'a>(
        &'a self,
        _request: crate::LlmRequest,
        _usage_source: &'a str,
    ) -> DirectCompletionFuture<'a, crate::DirectLlmCompletion> {
        Box::pin(async {
            Err(crate::PluginError::Session(
                "direct LLM completions are unavailable in this test context".to_string(),
            ))
        })
    }
}

impl DirectCompletionCapability {
    pub(in crate::runtime::session_manager) async fn invoke_direct_completion(
        &self,
        current: &CurrentSessionCapability,
        usage_capability: &UsageCapability,
        mut request: crate::DirectRequest,
        usage_source: &str,
        effect_controller: &dyn crate::RuntimeEffectController,
        turn_id: Option<&str>,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
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
        let request_spec = crate::DirectRequestSpec::from_request(
            &request,
            current.host.core.attachment_store.as_ref(),
        )
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let normalized_spec = crate::LlmRequestSpec::from_request(
            &llm_request,
            current.host.core.attachment_store.as_ref(),
        )
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let discriminator = crate::runtime::effect_controller::direct_request_discriminator(
            &request_spec,
            request.idempotency_key.as_deref(),
            request.originating_tool_call_id.as_deref(),
        )
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let metadata = crate::runtime::effect_controller::direct_effect_metadata(
            &current.session_id,
            usage_source,
            crate::RuntimeEffectKind::DirectCompletion,
            discriminator,
            turn_id,
        );
        let usage_source = usage_source.to_string();
        let envelope = crate::RuntimeEffectEnvelope::new(
            metadata,
            crate::RuntimeEffectCommand::DirectCompletion {
                request: request_spec,
                normalized_request: normalized_spec,
                model: model.clone(),
                usage_source: usage_source.clone(),
            },
        );
        let outcome = effect_controller
            .execute_effect(
                envelope,
                crate::RuntimeEffectLocalExecutor::direct(
                    provider,
                    Arc::clone(&current.host.core.attachment_store),
                ),
            )
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        crate::runtime::effect_controller::apply_direct_completion_outcome(
            current,
            usage_capability,
            &request,
            &llm_request,
            &model,
            &usage_source,
            outcome,
        )
        .await
    }

    pub(in crate::runtime::session_manager) async fn invoke_direct_llm_completion(
        &self,
        current: &CurrentSessionCapability,
        usage_capability: &UsageCapability,
        request: crate::LlmRequest,
        usage_source: &str,
        effect_controller: &dyn crate::RuntimeEffectController,
        turn_id: Option<&str>,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        let provider = current.policy.provider.clone();
        let request_spec = crate::LlmRequestSpec::from_request(
            &request,
            current.host.core.attachment_store.as_ref(),
        )
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let discriminator = crate::runtime::effect_controller::direct_request_discriminator(
            &request_spec,
            None,
            None,
        )
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let metadata = crate::runtime::effect_controller::direct_effect_metadata(
            &current.session_id,
            usage_source,
            crate::RuntimeEffectKind::DirectLlmCompletion,
            discriminator,
            turn_id,
        );
        let usage_source = usage_source.to_string();
        let envelope = crate::RuntimeEffectEnvelope::new(
            metadata,
            crate::RuntimeEffectCommand::DirectLlmCompletion {
                request: request_spec,
                usage_source: usage_source.clone(),
            },
        );
        let outcome = effect_controller
            .execute_effect(
                envelope,
                crate::RuntimeEffectLocalExecutor::direct(
                    provider,
                    Arc::clone(&current.host.core.attachment_store),
                ),
            )
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        crate::runtime::effect_controller::apply_direct_llm_completion_outcome(
            current,
            usage_capability,
            &request,
            &usage_source,
            outcome,
        )
        .await
    }
}
