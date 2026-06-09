use super::*;

/// Runtime-backed direct completion source.
///
/// Carries everything needed to plan and journal a direct LLM effect against
/// the owning session manager.
#[derive(Clone)]
struct RuntimeDirectSource<'run> {
    manager: Arc<RuntimeSessionServices>,
    effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    turn_id: Option<String>,
}

#[cfg(any(test, feature = "testing"))]
type TestDirectFn = Arc<
    dyn Fn(crate::DirectRequest, String) -> Result<crate::DirectCompletion, crate::PluginError>
        + Send
        + Sync,
>;

/// Source of direct (single-shot) LLM completions for plugins and tools.
///
/// In production this is always backed by the runtime session manager; the
/// test/testing variants exist only so that out-of-runtime test harnesses can
/// inject a canned completion without standing up a full runtime.
#[derive(Clone)]
enum DirectCompletionSource<'run> {
    Runtime(RuntimeDirectSource<'run>),
    #[cfg(any(test, feature = "testing"))]
    Unavailable(String),
    #[cfg(any(test, feature = "testing"))]
    TestFn(TestDirectFn),
}

#[derive(Clone)]
pub struct DirectCompletionClient<'run> {
    source: DirectCompletionSource<'run>,
}

impl<'run> DirectCompletionClient<'run> {
    pub(super) fn runtime(
        manager: Arc<RuntimeSessionServices>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        turn_id: Option<String>,
    ) -> Self {
        Self {
            source: DirectCompletionSource::Runtime(RuntimeDirectSource {
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
        match &self.source {
            DirectCompletionSource::Runtime(source) => {
                source
                    .manager
                    .direct
                    .invoke_direct_completion(source.invocation_context(), request, usage_source)
                    .await
            }
            #[cfg(any(test, feature = "testing"))]
            DirectCompletionSource::Unavailable(message) => {
                Err(crate::PluginError::Session(message.clone()))
            }
            #[cfg(any(test, feature = "testing"))]
            DirectCompletionSource::TestFn(invoke) => invoke(request, usage_source.to_string()),
        }
    }

    pub async fn direct_llm_completion(
        &self,
        request: crate::LlmRequest,
        usage_source: &str,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        match &self.source {
            DirectCompletionSource::Runtime(source) => {
                source
                    .manager
                    .direct
                    .invoke_direct_llm_completion(
                        source.invocation_context(),
                        request,
                        usage_source,
                    )
                    .await
            }
            #[cfg(any(test, feature = "testing"))]
            DirectCompletionSource::Unavailable(message) => {
                Err(crate::PluginError::Session(message.clone()))
            }
            #[cfg(any(test, feature = "testing"))]
            DirectCompletionSource::TestFn(_) => Err(crate::PluginError::Session(
                "direct LLM completions are unavailable in this test context".to_string(),
            )),
        }
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self {
            source: DirectCompletionSource::Unavailable(message.into()),
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
            source: DirectCompletionSource::TestFn(Arc::new(invoke)),
        }
    }
}

impl<'run> RuntimeDirectSource<'run> {
    fn invocation_context(&self) -> DirectInvocationContext<'_> {
        DirectInvocationContext {
            current: &self.manager.current,
            usage_capability: &self.manager.usage,
            effect_controller: self.effect_controller.controller(),
            turn_id: self.turn_id.as_deref(),
        }
    }
}

pub(in crate::runtime::session_manager) struct DirectInvocationContext<'a> {
    current: &'a CurrentSessionCapability,
    usage_capability: &'a UsageCapability,
    effect_controller: &'a dyn crate::RuntimeEffectController,
    turn_id: Option<&'a str>,
}

struct DirectEffectPlan {
    provider: crate::ProviderHandle,
    envelope: crate::RuntimeEffectEnvelope,
    request: Box<crate::LlmRequest>,
    usage_source: String,
}

impl DirectCompletionCapability {
    /// Plans a single direct LLM effect from a normalized [`crate::LlmRequest`].
    ///
    /// Both the text-only (`DirectRequest`) and full-output entry points feed
    /// the same effect lane; they differ only in how the caller projects the
    /// resulting [`crate::LlmResponse`].
    async fn plan_direct_effect(
        &self,
        context: &DirectInvocationContext<'_>,
        provider: crate::ProviderHandle,
        request: crate::LlmRequest,
        usage_source: &str,
        replay: Option<&crate::RuntimeReplay>,
        caused_by: Option<&crate::CausalRef>,
    ) -> Result<DirectEffectPlan, crate::PluginError> {
        let current = context.current;
        let usage_source = usage_source.to_string();
        let request_spec = crate::LlmRequestSpec::from_request(
            &request,
            current.host.core.durability.attachment_store.as_ref(),
        )
        .await?;
        let discriminator =
            crate::runtime::causal::direct_request_discriminator(&request_spec, replay, caused_by)?;
        let invocation = crate::runtime::causal::direct_effect_invocation(
            &current.session_id,
            &usage_source,
            discriminator,
            context.turn_id,
            caused_by.cloned(),
        );
        let envelope = crate::RuntimeEffectEnvelope::new(
            invocation,
            crate::RuntimeEffectCommand::Direct {
                request: Box::new(request_spec),
                usage_source: usage_source.clone(),
            },
        );
        Ok(DirectEffectPlan {
            provider,
            envelope,
            request: Box::new(request),
            usage_source,
        })
    }

    /// Runs a planned direct effect across the journal/controller boundary and
    /// applies usage/trace bookkeeping, yielding the raw provider response.
    async fn run_direct_effect(
        &self,
        context: DirectInvocationContext<'_>,
        plan: DirectEffectPlan,
        caused_by: Option<crate::CausalRef>,
    ) -> Result<(crate::LlmResponse, crate::TokenUsage), crate::PluginError> {
        let current = context.current;
        let DirectEffectPlan {
            provider,
            envelope,
            request,
            usage_source,
        } = plan;
        let outcome = context
            .effect_controller
            .execute_effect(
                envelope,
                crate::RuntimeEffectLocalExecutor::direct(
                    provider,
                    Arc::clone(&current.host.core.durability.attachment_store),
                ),
            )
            .await?;
        crate::runtime::effect::apply_direct_outcome(
            current,
            context.usage_capability,
            &request,
            &usage_source,
            caused_by.as_ref(),
            outcome,
        )
        .await
    }

    pub(in crate::runtime::session_manager) async fn invoke_direct_completion(
        &self,
        context: DirectInvocationContext<'_>,
        request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        let resolved = context.current.resolve_policy()?;
        let provider = resolved.provider().clone();
        let model = request.model.clone();
        if let Some(variant) = request.model_variant.as_deref() {
            provider
                .validate_variant(&model, variant)
                .map_err(crate::PluginError::Session)?;
        }
        let replay = request.replay.clone();
        let caused_by = request.caused_by.clone();
        let normalized = crate::direct::build_llm_request(&provider, request, model);
        let plan = self
            .plan_direct_effect(
                &context,
                provider,
                normalized,
                usage_source,
                replay.as_ref(),
                caused_by.as_ref(),
            )
            .await?;
        let (response, usage) = self.run_direct_effect(context, plan, caused_by).await?;
        Ok(crate::DirectCompletion {
            text: response.full_text,
            usage,
        })
    }

    pub(in crate::runtime::session_manager) async fn invoke_direct_llm_completion(
        &self,
        context: DirectInvocationContext<'_>,
        request: crate::LlmRequest,
        usage_source: &str,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        let resolved = context.current.resolve_policy()?;
        let plan = self
            .plan_direct_effect(
                &context,
                resolved.binding.provider,
                request,
                usage_source,
                None,
                None,
            )
            .await?;
        let (response, usage) = self.run_direct_effect(context, plan, None).await?;
        Ok(crate::DirectLlmCompletion { response, usage })
    }
}
