use std::sync::Arc;
use std::sync::Mutex;

use lash_trace::{TraceContext, TraceLevel, TraceSink};

use super::process::ProcessRegistry;
use super::{
    InlineRuntimeEffectController, ProcessWorkPoke, RuntimeEffectController, SessionStoreFactory,
    TerminationPolicy,
};

/// Required host configuration for all runtimes.
#[derive(Clone)]
pub struct RuntimeCoreConfig {
    pub host_profile_id: String,
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub lashlang_artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    pub lashlang_process_cache: Arc<Mutex<lashlang::CompiledProcessCache>>,
    pub provider_resolver: Arc<dyn crate::RuntimeProviderResolver>,
    pub prompt: crate::PromptLayer,
    pub trace_sink: Option<Arc<dyn TraceSink>>,
    pub trace_level: TraceLevel,
    pub trace_context: TraceContext,
    pub termination: TerminationPolicy,
    pub effect_controller: Arc<dyn RuntimeEffectController>,
}

impl RuntimeCoreConfig {
    /// Construct a config with the three host-owned dependencies named
    /// explicitly.
    ///
    /// There is intentionally no `Default`. The effect controller, Lashlang
    /// artifact store, and attachment store decide a runtime's durability, so
    /// hosts must choose them rather than silently inheriting in-memory
    /// implementations. Use [`RuntimeCoreConfig::in_memory`] to opt into the
    /// in-process / in-memory versions by name.
    pub fn new(
        effect_controller: Arc<dyn RuntimeEffectController>,
        lashlang_artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
        attachment_store: Arc<dyn crate::AttachmentStore>,
    ) -> Self {
        Self {
            host_profile_id: "default".to_string(),
            attachment_store,
            lashlang_artifact_store,
            lashlang_process_cache: Arc::new(Mutex::new(lashlang::CompiledProcessCache::new())),
            provider_resolver: Arc::new(crate::EmptyProviderResolver),
            prompt: crate::PromptLayer::new(),
            trace_sink: None,
            trace_level: TraceLevel::Standard,
            trace_context: TraceContext::default(),
            termination: TerminationPolicy::default(),
            effect_controller,
        }
    }

    /// Explicit in-process / in-memory configuration: an
    /// [`InlineRuntimeEffectController`], the process-global in-memory Lashlang
    /// artifact store, and an in-memory attachment store.
    ///
    /// Convenient for tests and local experiments; not durable. Named so the
    /// choice is never silent.
    pub fn in_memory() -> Self {
        Self::new(
            Arc::new(InlineRuntimeEffectController::default()),
            lashlang::global_in_memory_lashlang_artifact_store(),
            Arc::new(crate::InMemoryAttachmentStore::new()),
        )
    }

    pub fn with_attachment_store(
        mut self,
        attachment_store: Arc<dyn crate::AttachmentStore>,
    ) -> Self {
        self.attachment_store = attachment_store;
        self
    }

    pub fn with_lashlang_artifact_store(
        mut self,
        artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    ) -> Self {
        self.lashlang_artifact_store = artifact_store;
        self
    }

    pub fn with_lashlang_process_cache(
        mut self,
        process_cache: Arc<Mutex<lashlang::CompiledProcessCache>>,
    ) -> Self {
        self.lashlang_process_cache = process_cache;
        self
    }

    pub fn with_provider_resolver(
        mut self,
        provider_resolver: Arc<dyn crate::RuntimeProviderResolver>,
    ) -> Self {
        self.provider_resolver = provider_resolver;
        self
    }

    pub fn with_host_profile_id(mut self, host_profile_id: impl Into<String>) -> Self {
        self.host_profile_id = host_profile_id.into();
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: crate::PromptTemplate) -> Self {
        self.prompt.template = Some(prompt_template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.prompt.add_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.prompt.replace_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.prompt.clear_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.prompt = prompt;
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.trace_sink = sink;
        self
    }

    pub fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.trace_level = level;
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.trace_context = context;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.termination = termination;
        self
    }

    pub fn with_effect_controller(
        mut self,
        effect_controller: Arc<dyn RuntimeEffectController>,
    ) -> Self {
        self.effect_controller = effect_controller;
        self
    }
}

/// Base host shape for embedded runtimes.
#[derive(Clone)]
pub struct EmbeddedRuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
}

impl EmbeddedRuntimeHost {
    pub fn new(core: RuntimeCoreConfig) -> Self {
        Self {
            core,
            session_store_factory: None,
        }
    }

    pub fn with_session_store_factory(
        mut self,
        session_store_factory: Arc<dyn SessionStoreFactory>,
    ) -> Self {
        self.session_store_factory = Some(session_store_factory);
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
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub process_registry: Option<Arc<dyn ProcessRegistry>>,
    /// Wakes the host's [`ProcessWorkRunner`](super::ProcessWorkRunner) so a
    /// successful process start is consumed promptly. Absent when no work runner
    /// is wired (e.g. a registry-less host); poking is then a no-op.
    pub process_work_poke: Option<ProcessWorkPoke>,
}

impl RuntimeHost {
    pub(crate) fn resolve_session_policy(
        &self,
        session_id: &str,
        policy: crate::SessionPolicy,
    ) -> Result<crate::ResolvedSessionPolicy, crate::SessionError> {
        let provider_id = policy.recorded_provider_id();
        let provider = self
            .core
            .provider_resolver
            .resolve_provider(provider_id)
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
        Ok(crate::ResolvedSessionPolicy::new(policy, provider))
    }
}

impl From<EmbeddedRuntimeHost> for RuntimeHost {
    fn from(value: EmbeddedRuntimeHost) -> Self {
        Self {
            core: value.core,
            session_store_factory: value.session_store_factory,
            process_registry: None,
            process_work_poke: None,
        }
    }
}

impl From<ProcessRuntimeHost> for RuntimeHost {
    fn from(value: ProcessRuntimeHost) -> Self {
        Self {
            core: value.embedded.core,
            session_store_factory: value.embedded.session_store_factory,
            process_registry: Some(value.process_registry),
            process_work_poke: None,
        }
    }
}
