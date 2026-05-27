use std::sync::Arc;

use crate::plugin::{PluginFactory, PluginHost, PluginSession};
use crate::{
    EmbeddedRuntimeHost, LashRuntime, PersistentRuntimeServices, PluginStack, ProcessRegistry,
    ProcessRuntimeHost, RuntimeCoreConfig, RuntimeEffectController, RuntimePersistence,
    RuntimeServices, RuntimeSessionState, SessionError, SessionPolicy, SessionStoreFactory,
    TerminationPolicy, TurnInjectionBridge, TurnInputInjectionBridge,
};

enum PluginSource {
    Host(PluginHost),
    Session(Arc<PluginSession>),
}

pub(super) fn lashlang_abilities_for_process_registry(
    mut abilities: lashlang::LashlangAbilities,
    process_registry_available: bool,
) -> lashlang::LashlangAbilities {
    if process_registry_available {
        abilities.with_processes().with_process_lifecycle()
    } else {
        abilities.processes = false;
        abilities.process_sleep = false;
        abilities.process_signals = false;
        abilities
    }
}

pub struct EmbeddedRuntimeBuilder {
    session_id: Option<String>,
    policy: Option<SessionPolicy>,
    initial_state: Option<RuntimeSessionState>,
    plugin_source: PluginSource,
    turn_injection_bridge: TurnInjectionBridge,
    turn_input_injection_bridge: TurnInputInjectionBridge,
    core: RuntimeCoreConfig,
    session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    store: Option<Arc<dyn RuntimePersistence>>,
    process_registry: Option<Arc<dyn ProcessRegistry>>,
}

impl Default for EmbeddedRuntimeBuilder {
    fn default() -> Self {
        Self {
            session_id: None,
            policy: None,
            initial_state: None,
            plugin_source: PluginSource::Host(PluginHost::empty()),
            turn_injection_bridge: TurnInjectionBridge::new(),
            turn_input_injection_bridge: TurnInputInjectionBridge::new(),
            core: RuntimeCoreConfig::default(),
            session_store_factory: None,
            store: None,
            process_registry: None,
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

    pub fn with_turn_injection_bridge(mut self, bridge: TurnInjectionBridge) -> Self {
        self.turn_injection_bridge = bridge;
        self
    }

    pub fn with_turn_input_injection_bridge(mut self, bridge: TurnInputInjectionBridge) -> Self {
        self.turn_input_injection_bridge = bridge;
        self
    }

    pub fn with_runtime_core(mut self, core: RuntimeCoreConfig) -> Self {
        self.core = core;
        self
    }

    pub fn with_attachment_store(
        mut self,
        attachment_store: Arc<dyn crate::AttachmentStore>,
    ) -> Self {
        self.core = self.core.with_attachment_store(attachment_store);
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: crate::PromptTemplate) -> Self {
        self.core = self.core.with_prompt_template(prompt_template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.core = self.core.with_prompt_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.core = self.core.with_replaced_prompt_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.core = self.core.with_cleared_prompt_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.core = self.core.with_prompt_layer(prompt);
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn lash_trace::TraceSink>>) -> Self {
        self.core = self.core.with_trace_sink(sink);
        self
    }

    pub fn with_trace_level(mut self, level: lash_trace::TraceLevel) -> Self {
        self.core = self.core.with_trace_level(level);
        self
    }

    pub fn with_trace_context(mut self, context: lash_trace::TraceContext) -> Self {
        self.core = self.core.with_trace_context(context);
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.core = self.core.with_termination(termination);
        self
    }

    pub fn with_effect_controller(
        mut self,
        effect_controller: Arc<dyn RuntimeEffectController>,
    ) -> Self {
        self.core = self.core.with_effect_controller(effect_controller);
        self
    }

    pub fn with_session_store_factory(
        mut self,
        session_store_factory: Arc<dyn SessionStoreFactory>,
    ) -> Self {
        self.session_store_factory = Some(session_store_factory);
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
                    state.policy = policy.clone();
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
                    state.policy = policy.clone();
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
                .build_session(state.session_id.clone(), None)
                .map_err(|err| SessionError::Protocol(err.to_string())),
        }
    }

    pub async fn build(self) -> Result<LashRuntime, SessionError> {
        let state = self.resolve_state().await?;
        let plugins = self.resolve_plugins(&state)?;
        let embedded_host = EmbeddedRuntimeHost::new(self.core)
            .with_session_store_factory_option(self.session_store_factory.clone());
        match (self.store, self.process_registry) {
            (Some(store), Some(process_registry)) => {
                LashRuntime::from_persistent_background_state(
                    state.policy.clone(),
                    ProcessRuntimeHost::new(embedded_host, process_registry),
                    PersistentRuntimeServices::new_with_bridges(
                        plugins,
                        self.turn_injection_bridge,
                        self.turn_input_injection_bridge,
                        store,
                    ),
                    state,
                )
                .await
            }
            (Some(store), None) => {
                LashRuntime::from_persistent_embedded_state(
                    state.policy.clone(),
                    embedded_host,
                    PersistentRuntimeServices::new_with_bridges(
                        plugins,
                        self.turn_injection_bridge,
                        self.turn_input_injection_bridge,
                        store,
                    ),
                    state,
                )
                .await
            }
            (None, Some(process_registry)) => {
                LashRuntime::from_background_state(
                    state.policy.clone(),
                    ProcessRuntimeHost::new(embedded_host, process_registry),
                    RuntimeServices::new_with_bridges(
                        plugins,
                        self.turn_injection_bridge,
                        self.turn_input_injection_bridge,
                    ),
                    state,
                )
                .await
            }
            (None, None) => {
                LashRuntime::from_embedded_state(
                    state.policy.clone(),
                    embedded_host,
                    RuntimeServices::new_with_bridges(
                        plugins,
                        self.turn_injection_bridge,
                        self.turn_input_injection_bridge,
                    ),
                    state,
                )
                .await
            }
        }
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
}

impl EmbeddedRuntimeHostExt for EmbeddedRuntimeHost {
    fn with_session_store_factory_option(
        mut self,
        session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    ) -> Self {
        self.session_store_factory = session_store_factory;
        self
    }
}
