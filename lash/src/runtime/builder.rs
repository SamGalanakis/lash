use std::path::PathBuf;
use std::sync::Arc;

use crate::plugin::{PluginFactory, PluginHost, PluginSession};
use crate::{
    BackgroundRuntimeHost, EmbeddedRuntimeHost, LashRuntime, PathResolver, PersistedSessionState,
    PersistentRuntimeServices, RuntimeCoreConfig, RuntimeServices, RuntimeStore, SanitizerPolicy,
    SessionError, SessionPolicy, SessionStoreFactory, SessionTaskExecutor, TerminationPolicy,
    TurnInjectionBridge, TurnInputInjectionBridge,
};

enum PluginSource {
    Host(PluginHost),
    Session(Arc<PluginSession>),
}

pub struct EmbeddedRuntimeBuilder {
    session_id: Option<String>,
    policy: Option<SessionPolicy>,
    initial_state: Option<PersistedSessionState>,
    plugin_source: PluginSource,
    turn_injection_bridge: TurnInjectionBridge,
    turn_input_injection_bridge: TurnInputInjectionBridge,
    core: RuntimeCoreConfig,
    session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    store: Option<Arc<dyn RuntimeStore>>,
    session_task_executor: Option<Arc<dyn SessionTaskExecutor>>,
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
            session_task_executor: None,
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

    pub fn with_initial_state(mut self, state: PersistedSessionState) -> Self {
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
        self.plugin_source = PluginSource::Host(PluginHost::new(factories));
        self
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

    pub fn with_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.core = self.core.with_base_dir(base_dir);
        self
    }

    pub fn with_path_resolver(mut self, path_resolver: Arc<dyn PathResolver>) -> Self {
        self.core = self.core.with_path_resolver(path_resolver);
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: crate::PromptTemplate) -> Self {
        self.core = self.core.with_prompt_template(prompt_template);
        self
    }

    pub fn with_llm_log_path(mut self, llm_log_path: Option<PathBuf>) -> Self {
        self.core = self.core.with_llm_log_path(llm_log_path);
        self
    }

    pub fn with_sanitizer(mut self, sanitizer: SanitizerPolicy) -> Self {
        self.core = self.core.with_sanitizer(sanitizer);
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.core = self.core.with_termination(termination);
        self
    }

    pub fn with_session_store_factory(
        mut self,
        session_store_factory: Arc<dyn SessionStoreFactory>,
    ) -> Self {
        self.session_store_factory = Some(session_store_factory);
        self
    }

    pub fn with_store(mut self, store: Arc<dyn RuntimeStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_session_task_executor(
        mut self,
        session_task_executor: Arc<dyn SessionTaskExecutor>,
    ) -> Self {
        self.session_task_executor = Some(session_task_executor);
        self
    }

    fn resolve_state_from_defaults(&self) -> PersistedSessionState {
        let mut state = self.initial_state.clone().unwrap_or_default();
        if let Some(session_id) = &self.session_id {
            state.session_id = session_id.clone();
        }
        if let Some(policy) = &self.policy {
            state.policy = policy.clone();
        }
        state
    }

    async fn resolve_state(&self) -> Result<PersistedSessionState, SessionError> {
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
            if let Some(mut state) = store.load_persisted_session_state().await {
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
        state: &PersistedSessionState,
    ) -> Result<Arc<PluginSession>, SessionError> {
        match &self.plugin_source {
            PluginSource::Session(session) => Ok(Arc::clone(session)),
            PluginSource::Host(host) => host
                .isolated_registry()
                .build_session(
                    state.session_id.clone(),
                    state.policy.execution_mode,
                    state.policy.context_approach.clone(),
                    None,
                )
                .map_err(|err| SessionError::Protocol(err.to_string())),
        }
    }

    pub async fn build(self) -> Result<LashRuntime, SessionError> {
        let state = self.resolve_state().await?;
        let plugins = self.resolve_plugins(&state)?;
        let embedded_host = EmbeddedRuntimeHost::new(self.core)
            .with_session_store_factory_option(self.session_store_factory.clone());
        match (self.store, self.session_task_executor) {
            (Some(store), Some(session_task_executor)) => {
                LashRuntime::from_persistent_background_state(
                    state.policy.clone(),
                    BackgroundRuntimeHost::new(embedded_host, session_task_executor),
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
            (None, Some(session_task_executor)) => {
                LashRuntime::from_background_state(
                    state.policy.clone(),
                    BackgroundRuntimeHost::new(embedded_host, session_task_executor),
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
        self.session_task_executor = None;
        self.build().await
    }

    pub async fn build_persistent(
        mut self,
        store: Arc<dyn RuntimeStore>,
    ) -> Result<LashRuntime, SessionError> {
        self.store = Some(store);
        self.session_task_executor = None;
        self.build().await
    }

    pub async fn build_background_persistent(
        mut self,
        store: Arc<dyn RuntimeStore>,
        session_task_executor: Arc<dyn SessionTaskExecutor>,
    ) -> Result<LashRuntime, SessionError> {
        self.store = Some(store);
        self.session_task_executor = Some(session_task_executor);
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
