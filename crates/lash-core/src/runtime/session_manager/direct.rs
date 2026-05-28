use super::*;

/// Runtime-backed direct completion source.
///
/// Carries everything needed to plan and journal a direct LLM effect against
/// the owning session manager.
#[derive(Clone)]
struct RuntimeDirectSource<'run> {
    manager: Arc<RuntimeSessionManager>,
    effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
    turn_id: Option<String>,
    turn_lease: Option<crate::RuntimeTurnLease>,
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
        manager: Arc<RuntimeSessionManager>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        turn_id: Option<String>,
        turn_lease: Option<crate::RuntimeTurnLease>,
    ) -> Self {
        Self {
            source: DirectCompletionSource::Runtime(RuntimeDirectSource {
                manager,
                effect_controller,
                turn_id,
                turn_lease,
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
            effect_controller: self.effect_controller.as_controller(),
            turn_id: self.turn_id.as_deref(),
            turn_lease: self.turn_lease.as_ref(),
        }
    }
}

pub(in crate::runtime::session_manager) struct DirectInvocationContext<'a> {
    current: &'a CurrentSessionCapability,
    usage_capability: &'a UsageCapability,
    effect_controller: &'a dyn crate::RuntimeEffectController,
    turn_id: Option<&'a str>,
    turn_lease: Option<&'a crate::RuntimeTurnLease>,
}

enum DirectLlmPlanInput {
    Text(crate::DirectRequest),
    Full(crate::LlmRequest),
}

struct DirectLlmEffectPlan {
    provider: crate::ProviderHandle,
    envelope: crate::RuntimeEffectEnvelope,
    projection: DirectLlmProjection,
}

enum DirectLlmProjection {
    Text {
        request: Box<crate::DirectRequest>,
        normalized_request: Box<crate::LlmRequest>,
        model: String,
        usage_source: String,
    },
    Full {
        request: Box<crate::LlmRequest>,
        usage_source: String,
    },
}

impl DirectCompletionCapability {
    fn plan_direct_llm_effect(
        &self,
        context: &DirectInvocationContext<'_>,
        input: DirectLlmPlanInput,
        usage_source: &str,
    ) -> Result<DirectLlmEffectPlan, crate::PluginError> {
        let current = context.current;
        let provider = current.policy.provider.clone();
        let usage_source = usage_source.to_string();
        match input {
            DirectLlmPlanInput::Text(request) => {
                let model = request.model.clone();
                if let Some(variant) = request.model_variant.as_deref() {
                    provider
                        .validate_variant(&model, variant)
                        .map_err(crate::PluginError::Session)?;
                }
                let normalized_request =
                    crate::direct::build_llm_request(&provider, request.clone(), model.clone());
                let request_spec = crate::DirectRequestSpec::from_request(
                    &request,
                    current.host.core.attachment_store.as_ref(),
                )?;
                let normalized_spec = crate::LlmRequestSpec::from_request(
                    &normalized_request,
                    current.host.core.attachment_store.as_ref(),
                )?;
                let discriminator = crate::runtime::causal::direct_request_discriminator(
                    &request_spec,
                    request.replay.as_ref(),
                    request.caused_by.as_ref(),
                )?;
                let invocation = crate::runtime::causal::direct_effect_invocation(
                    &current.session_id,
                    &usage_source,
                    crate::RuntimeEffectKind::DirectCompletion,
                    discriminator,
                    context.turn_id,
                    request.caused_by.clone(),
                );
                let envelope = crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::DirectCompletion {
                        request: Box::new(request_spec),
                        normalized_request: Box::new(normalized_spec),
                        model: model.clone(),
                        usage_source: usage_source.clone(),
                    },
                );
                Ok(DirectLlmEffectPlan {
                    provider,
                    envelope,
                    projection: DirectLlmProjection::Text {
                        request: Box::new(request),
                        normalized_request: Box::new(normalized_request),
                        model,
                        usage_source,
                    },
                })
            }
            DirectLlmPlanInput::Full(request) => {
                let request_spec = crate::LlmRequestSpec::from_request(
                    &request,
                    current.host.core.attachment_store.as_ref(),
                )?;
                let discriminator = crate::runtime::causal::direct_request_discriminator(
                    &request_spec,
                    None,
                    None,
                )?;
                let invocation = crate::runtime::causal::direct_effect_invocation(
                    &current.session_id,
                    &usage_source,
                    crate::RuntimeEffectKind::DirectLlmCompletion,
                    discriminator,
                    context.turn_id,
                    None,
                );
                let envelope = crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::DirectLlmCompletion {
                        request: Box::new(request_spec),
                        usage_source: usage_source.clone(),
                    },
                );
                Ok(DirectLlmEffectPlan {
                    provider,
                    envelope,
                    projection: DirectLlmProjection::Full {
                        request: Box::new(request),
                        usage_source,
                    },
                })
            }
        }
    }

    pub(in crate::runtime::session_manager) async fn invoke_direct_completion(
        &self,
        context: DirectInvocationContext<'_>,
        request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        let current = context.current;
        let plan =
            self.plan_direct_llm_effect(&context, DirectLlmPlanInput::Text(request), usage_source)?;
        let DirectLlmEffectPlan {
            provider,
            envelope,
            projection:
                DirectLlmProjection::Text {
                    request,
                    normalized_request,
                    model,
                    usage_source,
                },
        } = plan
        else {
            unreachable!("direct completion planner returned non-text projection")
        };
        crate::runtime::effect::invoke_journaled_effect(
            crate::runtime::effect::JournaledEffectInvocation::new(
                current.store.as_ref().map(|store| store.as_ref()),
                context.turn_lease,
                context.effect_controller,
                envelope,
                crate::RuntimeEffectLocalExecutor::direct(
                    provider,
                    Arc::clone(&current.host.core.attachment_store),
                ),
            ),
            |outcome| async move {
                crate::runtime::effect::apply_direct_completion_outcome(
                    current,
                    context.usage_capability,
                    &request,
                    &normalized_request,
                    &model,
                    &usage_source,
                    outcome,
                )
                .await
            },
        )
        .await
    }

    pub(in crate::runtime::session_manager) async fn invoke_direct_llm_completion(
        &self,
        context: DirectInvocationContext<'_>,
        request: crate::LlmRequest,
        usage_source: &str,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        let current = context.current;
        let plan =
            self.plan_direct_llm_effect(&context, DirectLlmPlanInput::Full(request), usage_source)?;
        let DirectLlmEffectPlan {
            provider,
            envelope,
            projection:
                DirectLlmProjection::Full {
                    request,
                    usage_source,
                },
        } = plan
        else {
            unreachable!("direct LLM planner returned non-full projection")
        };
        crate::runtime::effect::invoke_journaled_effect(
            crate::runtime::effect::JournaledEffectInvocation::new(
                current.store.as_ref().map(|store| store.as_ref()),
                context.turn_lease,
                context.effect_controller,
                envelope,
                crate::RuntimeEffectLocalExecutor::direct(
                    provider,
                    Arc::clone(&current.host.core.attachment_store),
                ),
            ),
            |outcome| async move {
                crate::runtime::effect::apply_direct_llm_completion_outcome(
                    current,
                    context.usage_capability,
                    &request,
                    &usage_source,
                    outcome,
                )
                .await
            },
        )
        .await
    }
}
