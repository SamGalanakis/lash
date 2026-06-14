use std::sync::Arc;

use crate::plugin::{PluginFactory, PluginHost, PluginSession};
use crate::{
    EffectHost, EmbeddedRuntimeHost, LashRuntime, PluginStack, ProcessRegistry, Residency,
    RuntimeHostConfig, RuntimePersistence, RuntimeSessionState, SessionError, SessionPolicy,
    SessionStoreFactory, TerminationPolicy,
};

enum PluginSource {
    Host(PluginHost),
    Session(Arc<PluginSession>),
}

pub(super) fn lashlang_abilities_for_process_registry(
    mut abilities: lashlang::LashlangAbilities,
    process_registry_available: bool,
) -> lashlang::LashlangAbilities {
    abilities = abilities.with_sleep();
    if process_registry_available {
        abilities.with_processes().with_process_signals()
    } else {
        abilities.processes = false;
        abilities.process_signals = false;
        abilities
    }
}

pub struct EmbeddedRuntimeBuilder {
    session_id: Option<String>,
    policy: Option<SessionPolicy>,
    plugin_options: crate::PluginOptions,
    initial_state: Option<RuntimeSessionState>,
    plugin_source: PluginSource,
    core: RuntimeHostConfig,
    session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    trigger_store: Option<Arc<dyn crate::TriggerStore>>,
    store: Option<Arc<dyn RuntimePersistence>>,
    process_registry: Option<Arc<dyn ProcessRegistry>>,
    residency: Residency,
}

impl Default for EmbeddedRuntimeBuilder {
    fn default() -> Self {
        Self {
            session_id: None,
            policy: None,
            plugin_options: crate::PluginOptions::default(),
            initial_state: None,
            plugin_source: PluginSource::Host(PluginHost::empty()),
            // `RuntimeHostConfig` has no `Default`; start from an explicitly
            // named in-memory core. Callers that need durable stores override
            // it with `with_runtime_host`.
            core: RuntimeHostConfig::in_memory(),
            session_store_factory: None,
            trigger_store: Some(Arc::new(crate::InMemoryTriggerStore::default())),
            store: None,
            process_registry: None,
            residency: Residency::default(),
        }
    }
}

impl EmbeddedRuntimeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub fn policy(&self) -> Option<&SessionPolicy> {
        self.policy.as_ref()
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_policy(mut self, policy: SessionPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    pub fn with_plugin_options(mut self, plugin_options: crate::PluginOptions) -> Self {
        self.plugin_options = plugin_options;
        self
    }

    pub fn with_initial_state(mut self, state: RuntimeSessionState) -> Self {
        self.initial_state = Some(state);
        self
    }

    pub fn with_plugin_host(mut self, plugin_host: PluginHost) -> Self {
        self.plugin_source = PluginSource::Host(plugin_host);
        self
    }

    pub fn with_plugin_session(mut self, plugin_session: Arc<PluginSession>) -> Self {
        self.plugin_source = PluginSource::Session(plugin_session);
        self
    }

    pub fn with_plugin_factories(mut self, factories: Vec<Arc<dyn PluginFactory>>) -> Self {
        let host = PluginHost::new(factories);
        self.plugin_source = PluginSource::Host(host);
        self
    }

    pub fn with_plugin_stack(self, stack: PluginStack) -> Self {
        self.with_plugin_factories(stack.into_factories())
    }

    pub fn with_runtime_host(mut self, core: RuntimeHostConfig) -> Self {
        self.core = core;
        self
    }

    pub fn with_attachment_store(
        mut self,
        attachment_store: Arc<dyn crate::AttachmentStore>,
    ) -> Self {
        self.core.durability.attachment_store = attachment_store;
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: crate::PromptTemplate) -> Self {
        self.core.prompt.prompt.template = Some(prompt_template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.core.prompt.prompt.add_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.core.prompt.prompt.replace_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.core.prompt.prompt.clear_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.core.prompt.prompt = prompt;
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn lash_trace::TraceSink>>) -> Self {
        self.core.tracing.trace_sink = sink;
        self
    }

    pub fn with_lashlang_execution_sink(
        mut self,
        sink: Option<Arc<dyn lash_trace::TraceSink>>,
    ) -> Self {
        self.core.tracing.lashlang_execution_sink = sink;
        self
    }

    pub fn with_lashlang_execution_jsonl_path(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.core.tracing.lashlang_execution_sink = path.map(|path| {
            Arc::new(lash_trace::JsonlTraceSink::new(path)) as Arc<dyn lash_trace::TraceSink>
        });
        self
    }

    pub fn with_trace_level(mut self, level: lash_trace::TraceLevel) -> Self {
        self.core.tracing.trace_level = level;
        self
    }

    pub fn with_trace_context(mut self, context: lash_trace::TraceContext) -> Self {
        self.core.tracing.trace_context = context;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.core.control.termination = termination;
        self
    }

    pub fn with_effect_host(mut self, effect_host: Arc<dyn EffectHost>) -> Self {
        self.core.control.effect_host = effect_host;
        self
    }

    pub fn with_provider_resolver(
        mut self,
        provider_resolver: Arc<dyn crate::RuntimeProviderResolver>,
    ) -> Self {
        self.core.providers.provider_resolver = provider_resolver;
        self
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

    pub fn with_store(mut self, store: Arc<dyn RuntimePersistence>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_process_registry(mut self, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        self.process_registry = Some(process_registry);
        if let PluginSource::Host(host) = &mut self.plugin_source {
            let abilities =
                lashlang_abilities_for_process_registry(host.lashlang_abilities(), true);
            *host = host.clone().with_lashlang_abilities(abilities);
        }
        self
    }

    /// Trim a rebuilt session's resident graph to match the host's residency.
    ///
    /// Defaults to [`Residency::KeepAll`]. Setting [`Residency::ActivePathOnly`]
    /// makes a rebuilt runtime (e.g. a durable worker reconstructing a session to
    /// run a background process) keep only the active path resident, matching the
    /// live runtime's behavior instead of silently retaining the full graph.
    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.residency = residency;
        self
    }

    fn resolve_state_from_defaults(&self) -> RuntimeSessionState {
        let mut state = self.initial_state.clone().unwrap_or_default();
        if let Some(session_id) = &self.session_id {
            state.session_id = session_id.clone();
        }
        if let Some(policy) = &self.policy {
            state.policy = policy.clone();
        }
        state
    }

    async fn resolve_state(&self) -> Result<RuntimeSessionState, SessionError> {
        if let Some(state) = &self.initial_state {
            return Ok({
                let mut state = state.clone();
                if let Some(session_id) = &self.session_id {
                    state.session_id = session_id.clone();
                }
                if let Some(policy) = &self.policy {
                    let recorded_provider_id = state.policy.recorded_provider_id().to_string();
                    state.policy.provider_id = recorded_provider_id;
                    state.policy.session_id = policy.session_id.clone();
                    if state.policy.model.id.trim().is_empty() {
                        state.policy.model = policy.model.clone();
                    }
                }
                state
            });
        }
        if let Some(store) = &self.store {
            if let Some(mut state) = crate::store::load_persisted_session_state(store.as_ref())
                .await
                .map_err(|err| SessionError::Protocol(format!("failed to load store: {err}")))?
            {
                if let Some(session_id) = &self.session_id
                    && &state.session_id != session_id
                {
                    return Err(SessionError::Protocol(format!(
                        "store is bound to session `{}` but builder requested `{session_id}`",
                        state.session_id
                    )));
                }
                if let Some(policy) = &self.policy {
                    let recorded_provider_id = state.policy.recorded_provider_id().to_string();
                    state.policy.provider_id = recorded_provider_id;
                    state.policy.session_id = policy.session_id.clone();
                    if state.policy.model.id.trim().is_empty() {
                        state.policy.model = policy.model.clone();
                    }
                }
                return Ok(state);
            }
            let mut state = self.resolve_state_from_defaults();
            if let Some(policy) = &self.policy {
                state.policy = policy.clone();
            }
            return Ok(state);
        }
        Ok(self.resolve_state_from_defaults())
    }

    fn resolve_plugins(
        &self,
        state: &RuntimeSessionState,
    ) -> Result<Arc<PluginSession>, SessionError> {
        match &self.plugin_source {
            PluginSource::Session(session) => Ok(Arc::clone(session)),
            PluginSource::Host(host) => host
                .clone()
                .with_lashlang_abilities(lashlang_abilities_for_process_registry(
                    host.lashlang_abilities(),
                    self.process_registry.is_some(),
                ))
                .isolated_registry()
                .build_session_with_parent(
                    state.session_id.clone(),
                    None,
                    None,
                    crate::plugin::SessionAuthorityContext {
                        plugin_options: self.plugin_options.clone(),
                        ..crate::plugin::SessionAuthorityContext::default()
                    },
                )
                .map_err(|err| SessionError::Protocol(err.to_string())),
        }
    }

    pub async fn build(self) -> Result<LashRuntime, SessionError> {
        let state = self.resolve_state().await?;
        let plugins = self.resolve_plugins(&state)?;
        let embedded_host = EmbeddedRuntimeHost::new(self.core)
            .with_session_store_factory_option(self.session_store_factory.clone())
            .with_trigger_store_option(self.trigger_store.clone());
        // `assemble_runtime` owns the (store, registry) wiring + residency so the
        // worker rebuild cannot drift from the live open path.
        LashRuntime::assemble_runtime(
            state.policy.clone(),
            embedded_host,
            plugins,
            self.store,
            self.process_registry,
            state,
            self.residency,
        )
        .await
    }

    pub async fn build_ephemeral(mut self) -> Result<LashRuntime, SessionError> {
        self.store = None;
        self.build().await
    }

    pub async fn build_persistent(
        mut self,
        store: Arc<dyn RuntimePersistence>,
    ) -> Result<LashRuntime, SessionError> {
        self.store = Some(store);
        self.build().await
    }

    pub async fn build_background_persistent(
        mut self,
        store: Arc<dyn RuntimePersistence>,
        process_registry: Arc<dyn ProcessRegistry>,
    ) -> Result<LashRuntime, SessionError> {
        self.store = Some(store);
        self = self.with_process_registry(process_registry);
        self.build().await
    }
}

impl LashRuntime {
    pub fn builder() -> EmbeddedRuntimeBuilder {
        EmbeddedRuntimeBuilder::new()
    }
}

trait EmbeddedRuntimeHostExt {
    fn with_session_store_factory_option(
        self,
        session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    ) -> Self;

    fn with_trigger_store_option(self, trigger_store: Option<Arc<dyn crate::TriggerStore>>)
    -> Self;
}

impl EmbeddedRuntimeHostExt for EmbeddedRuntimeHost {
    fn with_session_store_factory_option(
        mut self,
        session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    ) -> Self {
        self.session_store_factory = session_store_factory;
        self
    }

    fn with_trigger_store_option(
        mut self,
        trigger_store: Option<Arc<dyn crate::TriggerStore>>,
    ) -> Self {
        self.trigger_store = trigger_store;
        self
    }
}
