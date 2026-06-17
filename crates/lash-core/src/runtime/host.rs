use lash_trace::{TraceContext, TraceLevel, TraceSink};
use std::sync::Arc;

use super::process::{
    InMemoryProcessExecutionEnvStore, ProcessEngineRegistry, ProcessExecutionEnvStore,
    ProcessRegistry,
};
use super::{
    EffectHost, InlineEffectHost, ProcessWorkPoke, QueuedWorkPoke, SessionStoreFactory,
    TerminationPolicy,
};

/// Required host configuration for all runtimes.
#[derive(Clone)]
pub struct RuntimeHostConfig {
    pub durability: RuntimeDurabilityConfig,
    pub process_engines: ProcessEngineRegistry,
    pub providers: RuntimeProviderConfig,
    pub prompt: RuntimePromptConfig,
    pub control: RuntimeControlConfig,
    pub tracing: RuntimeTracingConfig,
}

#[derive(Clone)]
pub struct RuntimeDurabilityConfig {
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub process_env_store: Arc<dyn ProcessExecutionEnvStore>,
}

#[derive(Clone)]
pub struct RuntimeProviderConfig {
    pub provider_resolver: Arc<dyn crate::RuntimeProviderResolver>,
}

#[derive(Clone)]
pub struct RuntimePromptConfig {
    pub prompt: crate::PromptLayer,
}

#[derive(Clone)]
pub struct RuntimeControlConfig {
    pub effect_host: Arc<dyn EffectHost>,
    pub process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
    pub termination: TerminationPolicy,
}

#[derive(Clone)]
pub struct RuntimeTracingConfig {
    pub trace_sink: Option<Arc<dyn TraceSink>>,
    pub trace_level: TraceLevel,
    pub trace_context: TraceContext,
}

impl RuntimeHostConfig {
    /// Construct a config with the three host-owned dependencies named
    /// explicitly.
    ///
    /// There is intentionally no `Default`. The effect host and stores decide
    /// a runtime's durability, so hosts must choose them rather than silently
    /// inheriting in-memory implementations. Use [`RuntimeHostConfig::in_memory`]
    /// to opt into the in-process / in-memory versions by name.
    pub fn new(
        effect_host: Arc<dyn EffectHost>,
        attachment_store: Arc<dyn crate::AttachmentStore>,
        process_env_store: Arc<dyn ProcessExecutionEnvStore>,
    ) -> Self {
        Self {
            durability: RuntimeDurabilityConfig {
                attachment_store,
                process_env_store,
            },
            process_engines: ProcessEngineRegistry::new(),
            providers: RuntimeProviderConfig {
                provider_resolver: Arc::new(crate::EmptyProviderResolver),
            },
            prompt: RuntimePromptConfig {
                prompt: crate::PromptLayer::new(),
            },
            control: RuntimeControlConfig {
                termination: TerminationPolicy::default(),
                effect_host,
                process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
            },
            tracing: RuntimeTracingConfig {
                trace_sink: None,
                trace_level: TraceLevel::Standard,
                trace_context: TraceContext::default(),
            },
        }
    }

    /// Explicit in-process / in-memory configuration: an
    /// [`InlineEffectHost`] and in-memory stores.
    ///
    /// Convenient for tests and local experiments; not durable. Named so the
    /// choice is never silent.
    pub fn in_memory() -> Self {
        Self::new(
            Arc::new(InlineEffectHost::default()),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            Arc::new(InMemoryProcessExecutionEnvStore::new()),
        )
    }

    pub fn with_process_env_store(
        mut self,
        process_env_store: Arc<dyn ProcessExecutionEnvStore>,
    ) -> Self {
        self.durability.process_env_store = process_env_store;
        self
    }

    pub fn with_process_cancel_ability(
        mut self,
        process_cancel_ability: Arc<dyn crate::ProcessCancelAbility>,
    ) -> Self {
        self.control.process_cancel_ability = process_cancel_ability;
        self
    }

    pub fn with_process_engine(mut self, engine: Arc<dyn crate::ProcessEngine>) -> Self {
        self.process_engines = self.process_engines.with_engine(engine);
        self
    }
}

/// Base host shape for embedded runtimes.
#[derive(Clone)]
pub struct EmbeddedRuntimeHost {
    pub core: RuntimeHostConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub trigger_store: Option<Arc<dyn crate::TriggerStore>>,
}

impl EmbeddedRuntimeHost {
    pub fn new(core: RuntimeHostConfig) -> Self {
        Self {
            core,
            session_store_factory: None,
            trigger_store: Some(Arc::new(crate::InMemoryTriggerStore::default())),
        }
    }

    pub fn with_session_store_factory(
        mut self,
        session_store_factory: Arc<dyn SessionStoreFactory>,
    ) -> Self {
        self.session_store_factory = Some(session_store_factory);
        self
    }

    pub fn with_trigger_store(mut self, store: Arc<dyn crate::TriggerStore>) -> Self {
        self.trigger_store = Some(store);
        self
    }
}

/// Host shape for runtimes that support background plugin work.
#[derive(Clone)]
pub struct ProcessRuntimeHost {
    pub embedded: EmbeddedRuntimeHost,
    pub process_registry: Arc<dyn ProcessRegistry>,
}

impl ProcessRuntimeHost {
    pub fn new(embedded: EmbeddedRuntimeHost, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        Self {
            embedded,
            process_registry,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeHost {
    pub core: RuntimeHostConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub trigger_store: Option<Arc<dyn crate::TriggerStore>>,
    pub process_registry: Option<Arc<dyn ProcessRegistry>>,
    /// Wakes the host's [`ProcessWorkRunner`](super::ProcessWorkRunner) so a
    /// successful process start is consumed promptly. Absent when no work runner
    /// is wired (e.g. a registry-less host); poking is then a no-op.
    pub process_work_poke: Option<ProcessWorkPoke>,
    /// Wakes the host's [`QueuedWorkRunner`](super::QueuedWorkRunner) so queued
    /// turn work drains promptly after queue ingress.
    pub queued_work_poke: Option<QueuedWorkPoke>,
}

impl RuntimeHost {
    pub(crate) fn resolve_session_policy(
        &self,
        session_id: &str,
        policy: crate::SessionPolicy,
    ) -> Result<crate::RuntimeSessionPolicy, crate::SessionError> {
        let provider_id = policy.recorded_provider_id();
        let binding = self
            .core
            .providers
            .provider_resolver
            .resolve_provider_binding(provider_id)
            .map_err(|err| match err {
                crate::ProviderResolutionError::MissingProviderId => {
                    crate::SessionError::ProviderUnconfigured {
                        session_id: session_id.to_string(),
                    }
                }
                crate::ProviderResolutionError::UnknownProvider { provider_id } => {
                    crate::SessionError::ProviderUnavailable {
                        provider_id,
                        session_id: session_id.to_string(),
                    }
                }
                crate::ProviderResolutionError::ProviderIdMismatch { expected, actual } => {
                    crate::SessionError::ProviderMismatch {
                        expected,
                        actual,
                        session_id: session_id.to_string(),
                    }
                }
            })?;
        Ok(crate::RuntimeSessionPolicy::new(policy, binding))
    }
}

impl From<EmbeddedRuntimeHost> for RuntimeHost {
    fn from(value: EmbeddedRuntimeHost) -> Self {
        Self {
            core: value.core,
            session_store_factory: value.session_store_factory,
            trigger_store: value.trigger_store,
            process_registry: None,
            process_work_poke: None,
            queued_work_poke: None,
        }
    }
}

impl From<ProcessRuntimeHost> for RuntimeHost {
    fn from(value: ProcessRuntimeHost) -> Self {
        Self {
            core: value.embedded.core,
            session_store_factory: value.embedded.session_store_factory,
            trigger_store: value.embedded.trigger_store,
            process_registry: Some(value.process_registry),
            process_work_poke: None,
            queued_work_poke: None,
        }
    }
}
