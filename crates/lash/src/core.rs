use crate::support::*;
use lash_core::runtime::{
    ProcessCommand, ProcessEffectOutcome, RuntimeEffectCommand, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeInvocation,
    RuntimeScope,
};

#[derive(Clone)]
pub struct LashCore {
    pub(crate) env: RuntimeEnvironment,
    pub(crate) policy: SessionPolicy,
    pub(crate) modes: Arc<BTreeMap<ModeId, ModePreset>>,
    pub(crate) default_mode: ModeId,
    pub(crate) store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub(crate) plugin_factories: Arc<Vec<Arc<dyn PluginFactory>>>,
    pub(crate) provider: Option<ProviderHandle>,
    pub(crate) live_replay_store: Arc<dyn LiveReplayStore>,
    pub(crate) process_observer: Option<ProcessWorkObserver>,
    /// Shared resolution of the process work runner. The poke it yields is
    /// threaded onto every session's host so the process control seam can wake
    /// the runner after a successful start. Shared across `LashCore` clones so
    /// the default inline runner is spawned at most once (Decision 3).
    pub(crate) process_work_runner: Arc<ProcessWorkRunnerSlot>,
}

/// How a [`LashCore`] resolves its process work runner, decided at `build()`
/// and shared across clones.
pub(crate) enum ProcessWorkRunnerSetup {
    /// No process registry is wired; there is nothing to run and no poke.
    None,
    /// Lazily spawn the default inline [`ProcessWorkRunner`] on first
    /// `session().open()` (Decision 3: the runtime is guaranteed by then, and
    /// `build()` is sync — some tests call it outside a tokio runtime). A store
    /// factory is required to build the config (the worker rebuilds a session
    /// runtime per process); a registry with no store factory is rejected at
    /// build with [`EmbedError::ProcessRegistryRequiresStoreFactory`].
    LazyDefault {
        config: Box<DurableProcessWorkerConfig>,
    },
    /// The host wired an external runner (e.g. the Restate ingress-client
    /// runner) and handed its driver to the core.
    External { driver: ProcessWorkDriver },
}

#[derive(Clone, Default)]
pub(crate) enum ProcessWorkSource {
    #[default]
    None,
    Inline {
        registry: Arc<dyn ProcessRegistry>,
    },
    External(ProcessWorkDriver),
}

impl ProcessWorkSource {
    fn process_registry(&self) -> Option<Arc<dyn ProcessRegistry>> {
        match self {
            Self::None => None,
            Self::Inline { registry } => Some(Arc::clone(registry)),
            Self::External(driver) => Some(driver.process_registry()),
        }
    }

    fn has_registry(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Shared, lazily-initialized process-work-runner state for a [`LashCore`].
///
/// The once-guard ([`tokio::sync::OnceCell`]) makes the default inline runner
/// spawn exactly once across `LashCore` clones, on the first `session().open()`
/// that needs it. The resolved [`ProcessWorkPoke`] (if any) is then reused for
/// every session host.
pub(crate) struct ProcessWorkRunnerSlot {
    setup: ProcessWorkRunnerSetup,
    poke: tokio::sync::OnceCell<Option<ProcessWorkPoke>>,
}

impl ProcessWorkRunnerSlot {
    fn new(setup: ProcessWorkRunnerSetup) -> Self {
        Self {
            setup,
            poke: tokio::sync::OnceCell::new(),
        }
    }

    /// Resolve the poke for a session host, spawning the default inline runner
    /// on first use. Idempotent: the once-guard ensures a single spawn.
    pub(crate) async fn poke(&self) -> Option<ProcessWorkPoke> {
        self.poke
            .get_or_init(|| async {
                match &self.setup {
                    ProcessWorkRunnerSetup::None => None,
                    ProcessWorkRunnerSetup::External { driver } => Some(driver.poke_handle()),
                    ProcessWorkRunnerSetup::LazyDefault { config } => {
                        let worker = DurableProcessWorker::new((**config).clone());
                        Some(ProcessWorkRunner::inline(worker).spawn())
                    }
                }
            })
            .await
            .clone()
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionDeleteReport {
    pub session_id: String,
    pub process: Option<lash_core::ProcessSessionDeleteReport>,
}

impl LashCore {
    pub fn builder() -> LashCoreBuilder {
        LashCoreBuilder::default()
    }

    /// Preset for `standard` mode.
    ///
    /// Storage and effect durability are still host-owned choices. Provide the
    /// effect host, Lashlang artifact store, and attachment store facets with
    /// the builder setters before calling [`LashCoreBuilder::build`].
    pub fn standard() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::standard())
            .default_mode(ModeId::standard())
            .plugins(default_runtime_stack())
    }

    /// Preset for `rlm` mode.
    ///
    /// Storage and effect durability are still host-owned choices. Provide the
    /// effect host, Lashlang artifact store, and attachment store facets with
    /// the builder setters before calling [`LashCoreBuilder::build`].
    pub fn rlm() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::rlm())
            .default_mode(ModeId::rlm())
            .plugins(default_runtime_stack())
    }

    pub fn session(&self, session_id: impl Into<String>) -> SessionBuilder {
        SessionBuilder {
            core: self.clone(),
            session_id: session_id.into(),
            spec: SessionSpec::inherit(),
            mode: None,
            parent_session_id: None,
            store: None,
            provider: None,
            active_plugins: Vec::new(),
            plugin_factories: Vec::new(),
            rlm_final_answer_format: None,
        }
    }

    pub fn host_events(&self) -> crate::control::HostEventsControl {
        crate::control::HostEventsControl { core: self.clone() }
    }

    pub fn effect_host(&self) -> Arc<dyn EffectHost> {
        Arc::clone(&self.env.core.control.effect_host)
    }

    pub async fn delete_session(
        &self,
        session_id: impl AsRef<str>,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<SessionDeleteReport> {
        let session_id = session_id.as_ref().to_string();
        let Some(store_factory) = self.store_factory.as_ref() else {
            return Err(EmbedError::MissingSessionStoreFactory);
        };
        let process = if let Some(process_registry) = self.env.process_registry.as_ref() {
            let invocation = RuntimeInvocation::effect(
                RuntimeScope::new(session_id.clone()),
                format!("process:delete-session:{session_id}"),
                RuntimeEffectKind::Process,
                format!("{session_id}:delete-session"),
            );
            let outcome = scoped_effect_controller
                .controller()
                .execute_effect(
                    RuntimeEffectEnvelope::new(
                        invocation,
                        RuntimeEffectCommand::Process {
                            command: ProcessCommand::DeleteSession {
                                session_id: session_id.clone(),
                            },
                        },
                    ),
                    RuntimeEffectLocalExecutor::process_control(Arc::clone(process_registry)),
                )
                .await
                .map_err(|err| EmbedError::SessionDeleteProcess {
                    session_id: session_id.clone(),
                    message: err.to_string(),
                })?;
            match outcome {
                RuntimeEffectOutcome::Process {
                    result: ProcessEffectOutcome::DeleteSession { report },
                } => Some(report),
                other => {
                    return Err(EmbedError::SessionDeleteProcess {
                        session_id,
                        message: format!(
                            "process delete returned the wrong outcome: {}",
                            other.kind().as_str()
                        ),
                    });
                }
            }
        } else {
            None
        };
        store_factory
            .delete_session(&session_id)
            .await
            .map_err(|message| EmbedError::StoreFactory {
                session_id: session_id.clone(),
                message,
            })?;
        Ok(SessionDeleteReport {
            session_id,
            process,
        })
    }

    pub fn installed_modes(&self) -> impl Iterator<Item = &ModeId> {
        self.modes.keys()
    }

    pub fn process_observer(&self) -> Option<&ProcessWorkObserver> {
        self.process_observer.as_ref()
    }

    pub fn process_registry(&self) -> Option<Arc<dyn ProcessRegistry>> {
        self.env.process_registry.as_ref().cloned()
    }

    pub fn durable_process_worker_config(&self) -> Result<DurableProcessWorkerConfig> {
        self.durable_process_worker_config_with_plugins(std::iter::empty::<Arc<dyn PluginFactory>>())
    }

    pub fn durable_process_worker_config_with_plugins(
        &self,
        extra_plugin_factories: impl IntoIterator<Item = Arc<dyn PluginFactory>>,
    ) -> Result<DurableProcessWorkerConfig> {
        let Some(process_registry) = self.process_registry() else {
            return Err(EmbedError::MissingProcessRegistry);
        };
        let Some(store_factory) = self.store_factory.as_ref() else {
            return Err(EmbedError::MissingProcessWorkerStoreFactory);
        };
        let plugin_host = build_plugin_host_for_mode(
            &self.modes,
            &self.default_mode,
            self.plugin_factories.as_ref(),
            extra_plugin_factories.into_iter().collect(),
            true,
        )?;
        let mut config = DurableProcessWorkerConfig::new(
            Arc::new(plugin_host),
            self.env.core.clone(),
            Arc::clone(store_factory),
            process_registry,
        )
        .with_session_policy(self.policy.clone())
        .with_residency(self.env.residency);
        if let Some(host_event_store) = self.env.host_event_store.as_ref() {
            config = config.with_host_event_store(Arc::clone(host_event_store));
        }
        Ok(config)
    }
}

fn default_runtime_stack() -> PluginStack {
    lash_plugin_tool_output_budget::tool_output_budget_stack()
}

#[derive(Default)]
pub struct LashCoreBuilder {
    pub(crate) modes: BTreeMap<ModeId, ModePreset>,
    pub(crate) default_mode: Option<ModeId>,
    session_spec: SessionSpec,
    provider: Option<ProviderHandle>,
    pub(crate) store_factory: Option<Arc<dyn SessionStoreFactory>>,
    child_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    // `RuntimeHostConfig` has no `Default`: the three host-owned durability
    // dependencies must be named. They are collected here and resolved in
    // `build()`, which errors if any is unset.
    effect_host: Option<Arc<dyn EffectHost>>,
    attachment_store: Option<Arc<dyn AttachmentStore>>,
    lashlang_artifact_store: Option<Arc<dyn lash_core::LashlangArtifactStore>>,
    host_event_store: Option<Arc<dyn lash_core::HostEventStore>>,
    // Benign core overrides applied on top of the resolved core.
    prompt: Option<PromptLayer>,
    trace_sink: Option<Arc<dyn lash_trace::TraceSink>>,
    lashlang_execution_sink: Option<Arc<dyn lash_trace::TraceSink>>,
    trace_level: Option<lash_trace::TraceLevel>,
    trace_context: Option<lash_trace::TraceContext>,
    termination: Option<TerminationPolicy>,
    // Advanced full-config override; used as the base core when present.
    runtime_host_config: Option<RuntimeHostConfig>,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    plugin_stack: PluginStack,
    plugin_host: Option<PluginHost>,
    residency: Option<Residency>,
    // Single source of truth for process lifecycle support and process-work
    // consumption.
    process_work_source: ProcessWorkSource,
    queued_work_poke: Option<QueuedWorkPoke>,
    live_replay_store: Option<Arc<dyn LiveReplayStore>>,
}

impl LashCoreBuilder {
    pub fn install_mode(mut self, preset: ModePreset) -> Self {
        let mode_id = preset.mode_id.clone();
        if self.default_mode.is_none() {
            self.default_mode = Some(mode_id.clone());
        }
        self.modes.insert(mode_id, preset);
        self
    }

    pub fn default_mode(mut self, mode: ModeId) -> Self {
        self.default_mode = Some(mode);
        self
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.session_spec = self.session_spec.provider_id(provider.kind());
        self.provider = Some(provider);
        self
    }

    pub fn model(mut self, model: lash_core::ModelSpec) -> Self {
        self.session_spec = self.session_spec.model(model);
        self
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.session_spec = self.session_spec.max_turns(max_turns);
        self
    }

    pub fn session_spec(mut self, spec: SessionSpec) -> Self {
        self.session_spec = spec;
        self
    }

    /// Configure a factory that can create a persistence store for any root
    /// session opened from this core.
    ///
    /// The factory must honor `SessionStoreCreateRequest::session_id` and
    /// return a store for that specific session. Do not use this to wrap one
    /// pre-opened root store; pass root-only stores with
    /// `LashCore::session(...).store(store)` instead.
    pub fn store_factory(mut self, store_factory: Arc<dyn SessionStoreFactory>) -> Self {
        self.store_factory = Some(store_factory);
        self
    }

    /// Configure the persistence factory used by managed child sessions, such
    /// as local subagents.
    ///
    /// Child factories must return a distinct store bound to the requested
    /// child session id. Hosts that pass an explicit root store with
    /// `SessionBuilder::store` should set this when child sessions need
    /// persistence.
    pub fn child_store_factory(mut self, store_factory: Arc<dyn SessionStoreFactory>) -> Self {
        self.child_store_factory = Some(store_factory);
        self
    }

    pub fn attachment_store(mut self, attachment_store: Arc<dyn AttachmentStore>) -> Self {
        self.attachment_store = Some(attachment_store);
        self
    }

    /// Set the deployment-level Lashlang artifact store (compiled module
    /// cache, shared across the session tree). A durable store such as
    /// `lash_sqlite_store::Store` implements it.
    pub fn lashlang_artifact_store(
        mut self,
        artifact_store: Arc<dyn lash_core::LashlangArtifactStore>,
    ) -> Self {
        self.lashlang_artifact_store = Some(artifact_store);
        self
    }

    /// Set the deployment effect host — the durability boundary every operation
    /// crosses. Pass [`InlineEffectHost`](crate::durability::InlineEffectHost)
    /// for in-process execution, or a workflow-backed host for durable
    /// execution.
    pub fn effect_host(mut self, effect_host: Arc<dyn EffectHost>) -> Self {
        self.effect_host = Some(effect_host);
        self
    }

    pub fn tools(mut self, tools: Arc<dyn ToolProvider>) -> Self {
        self.tool_providers.push(tools);
        self
    }

    pub fn plugin(mut self, plugin: Arc<dyn PluginFactory>) -> Self {
        self.plugin_stack.push(plugin);
        self
    }

    pub fn plugins(mut self, stack: PluginStack) -> Self {
        self.plugin_stack = stack;
        self
    }

    pub fn configure_plugins(mut self, configure: impl FnOnce(&mut PluginStack)) -> Self {
        configure(&mut self.plugin_stack);
        self
    }

    pub fn trace_sink(mut self, trace_sink: Arc<dyn lash_trace::TraceSink>) -> Self {
        self.trace_sink = Some(trace_sink);
        self
    }

    pub fn trace_jsonl_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.trace_sink = Some(Arc::new(lash_trace::JsonlTraceSink::new(path.into())));
        self
    }

    pub fn lashlang_execution_sink(
        mut self,
        lashlang_execution_sink: Arc<dyn lash_trace::TraceSink>,
    ) -> Self {
        self.lashlang_execution_sink = Some(lashlang_execution_sink);
        self
    }

    pub fn lashlang_execution_jsonl_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.lashlang_execution_sink = Some(Arc::new(lash_trace::JsonlTraceSink::new(path.into())));
        self
    }

    pub fn trace_level(mut self, trace_level: lash_trace::TraceLevel) -> Self {
        self.trace_level = Some(trace_level);
        self
    }

    pub fn trace_context(mut self, trace_context: lash_trace::TraceContext) -> Self {
        self.trace_context = Some(trace_context);
        self
    }

    pub fn termination(mut self, termination: TerminationPolicy) -> Self {
        self.termination = Some(termination);
        self
    }

    pub fn residency(mut self, residency: Residency) -> Self {
        self.residency = Some(residency);
        self
    }

    /// Configure the bounded live replay buffer used by session observation
    /// cursors. This is best-effort reconnect recovery only; durable state
    /// still comes from the session store and [`SessionReadView`].
    pub fn live_replay_store(mut self, live_replay_store: Arc<dyn LiveReplayStore>) -> Self {
        self.live_replay_store = Some(live_replay_store);
        self
    }

    /// Resolve the runtime host config, requiring the three host-owned
    /// durability dependencies to have been named.
    fn resolve_runtime_host_config(&mut self) -> Result<RuntimeHostConfig> {
        if let Some(base) = self.runtime_host_config.take() {
            return Ok(self.apply_core_overrides(base));
        }
        let effect_host = self
            .effect_host
            .take()
            .ok_or(EmbedError::MissingEffectHost)?;
        let lashlang_artifact_store = self
            .lashlang_artifact_store
            .take()
            .ok_or(EmbedError::MissingLashlangArtifactStore)?;
        let attachment_store = self
            .attachment_store
            .take()
            .ok_or(EmbedError::MissingAttachmentStore)?;
        let core = RuntimeHostConfig::new(effect_host, lashlang_artifact_store, attachment_store);
        Ok(self.apply_core_overrides(core))
    }

    /// Apply benign + still-set dependency overrides on top of a base core.
    fn apply_core_overrides(&mut self, mut core: RuntimeHostConfig) -> RuntimeHostConfig {
        if let Some(effect_host) = self.effect_host.take() {
            core.control.effect_host = effect_host;
        }
        if let Some(attachment_store) = self.attachment_store.take() {
            core.durability.attachment_store = attachment_store;
        }
        if let Some(artifact_store) = self.lashlang_artifact_store.take() {
            core.durability.lashlang_artifact_store = artifact_store;
        }
        if let Some(prompt) = self.prompt.take() {
            core.prompt.prompt = prompt;
        }
        if let Some(trace_sink) = self.trace_sink.take() {
            core.tracing.trace_sink = Some(trace_sink);
        }
        if let Some(lashlang_execution_sink) = self.lashlang_execution_sink.take() {
            core.tracing.lashlang_execution_sink = Some(lashlang_execution_sink);
        }
        if let Some(trace_level) = self.trace_level.take() {
            core.tracing.trace_level = trace_level;
        }
        if let Some(trace_context) = self.trace_context.take() {
            core.tracing.trace_context = trace_context;
        }
        if let Some(termination) = self.termination.take() {
            core.control.termination = termination;
        }
        core
    }

    /// Validate store peer-coherence of the wired durability dependencies.
    ///
    /// Durability is established by what the host wired; the per-invocation
    /// durable controller is not visible here (the build-time controller is
    /// inline by construction), so this checks the stores against each other
    /// only — never the controller (see A5 in the durable-first wiring spec):
    ///
    /// - a durable session store factory requires a durable attachment and
    ///   artifact store (they back the same session state);
    /// - a durable process registry requires a session store factory that is
    ///   itself durable (the registry's process records are meaningless without
    ///   a durable session behind them).
    fn ensure_store_peer_coherence(&self) -> Result<()> {
        // Match `build()`'s wiring exactly: the session store factory it installs
        // is `child_store_factory.or(store_factory)` (child takes precedence, root
        // is the fallback). The coherence check must read the tier of that same
        // effective factory, or a host that wires only a durable child factory
        // (no root) is wrongly rejected though `build()` would wire it durably.
        let session_store_tier = self
            .child_store_factory
            .as_ref()
            .or(self.store_factory.as_ref())
            .map(|factory| factory.durability_tier());
        let attachment_tier = self
            .attachment_store
            .as_ref()
            .map(|store| store.persistence().durability_tier());
        let artifact_tier = self
            .lashlang_artifact_store
            .as_ref()
            .map(|store| store.durability_tier());
        let host_event_store_tier = self
            .host_event_store
            .as_ref()
            .map(|store| store.durability_tier());

        if session_store_tier == Some(DurabilityTier::Durable) {
            if attachment_tier == Some(DurabilityTier::Inline) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "attachment store",
                });
            }
            if artifact_tier == Some(DurabilityTier::Inline) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "artifact store",
                });
            }
        }

        if let Some(process_registry) = self.process_work_source.process_registry().as_ref() {
            if process_registry.durability_tier() == DurabilityTier::Durable {
                if session_store_tier != Some(DurabilityTier::Durable) {
                    return Err(EmbedError::DurableProcessRegistryRequiresStoreFactory);
                }
                if host_event_store_tier != Some(DurabilityTier::Durable) {
                    return Err(EmbedError::DurableStorePeerRequired {
                        facet: "host event store",
                    });
                }
            }
        }

        if host_event_store_tier == Some(DurabilityTier::Durable) {
            if session_store_tier != Some(DurabilityTier::Durable) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "session store factory",
                });
            }
            if let Some(process_registry) = self.process_work_source.process_registry().as_ref()
                && process_registry.durability_tier() == DurabilityTier::Inline
            {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "process registry",
                });
            }
        }

        Ok(())
    }

    pub fn build(mut self) -> Result<LashCore> {
        if self.modes.is_empty() {
            return Err(EmbedError::NoModesInstalled);
        }
        self.ensure_store_peer_coherence()?;
        let default_mode = self
            .default_mode
            .clone()
            .ok_or(EmbedError::NoModesInstalled)?;
        if !self.modes.contains_key(&default_mode) {
            return Err(EmbedError::DefaultModeNotInstalled { mode: default_mode });
        }
        let provider_id = self
            .session_spec
            .provider_id
            .clone()
            .or_else(|| {
                self.provider
                    .as_ref()
                    .map(|provider| provider.kind().to_string())
            })
            .unwrap_or_default();
        let model = self
            .session_spec
            .model
            .clone()
            .ok_or(EmbedError::MissingModelSpec)?;

        let base_policy = SessionPolicy {
            provider_id,
            model,
            max_turns: self.session_spec.max_turns.flatten(),
            ..SessionPolicy::default()
        };
        let policy = self.session_spec.resolve_against(&base_policy);

        let mut core = self.resolve_runtime_host_config()?;
        if let Some(provider) = self.provider.clone() {
            core.providers.provider_resolver =
                Arc::new(lash_core::SingleProviderResolver::new(provider));
        }

        let plugin_factories = if let Some(plugin_host) = self.plugin_host {
            plugin_host.factories().to_vec()
        } else {
            let mut factories = Vec::new();
            if !self.tool_providers.is_empty() {
                let spec = self
                    .tool_providers
                    .into_iter()
                    .fold(PluginSpec::new(), PluginSpec::with_tool_provider);
                factories.push(Arc::new(StaticPluginFactory::new("embed_tools", spec))
                    as Arc<dyn PluginFactory>);
            }
            factories.extend(self.plugin_stack.into_factories());
            factories
        };
        let default_plugin_host = build_plugin_host_for_mode(
            &self.modes,
            &default_mode,
            &plugin_factories,
            Vec::new(),
            self.process_work_source.has_registry(),
        )?;

        let process_registry = self.process_work_source.process_registry();

        // Resolve the process work runner before the process source is moved
        // into the environment. The default inline runner's config is built
        // eagerly so a missing store factory fails loudly at build, not at
        // first open. It is built from the *default-mode* plugin host (preset
        // protocol plugin + process-lifecycle abilities), the same host the
        // live runtime uses, so the worker can rebuild a runtime for a
        // default-mode process.
        let process_work_runner = Self::resolve_process_work_runner(
            &self.process_work_source,
            &default_plugin_host,
            &core,
            // The worker rebuilds sessions with the same factory `build()` wires
            // below: `child_store_factory.or(store_factory)`.
            self.child_store_factory
                .as_ref()
                .or(self.store_factory.as_ref()),
            &policy,
            self.residency.unwrap_or_default(),
            self.host_event_store.as_ref(),
        )?;

        let mut env_builder = RuntimeEnvironment::builder()
            .with_plugin_host(Arc::new(default_plugin_host))
            .with_runtime_host_config(core);
        if let Some(process_registry) = process_registry.as_ref() {
            env_builder = env_builder.with_process_registry(Arc::clone(process_registry));
        }
        if let Some(residency) = self.residency {
            env_builder = env_builder.with_residency(residency);
        }
        if let Some(child_store_factory) = self
            .child_store_factory
            .as_ref()
            .or(self.store_factory.as_ref())
        {
            env_builder = env_builder.with_session_store_factory(Arc::clone(child_store_factory));
        }
        if let Some(host_event_store) = self.host_event_store.as_ref() {
            env_builder = env_builder.with_host_event_store(Arc::clone(host_event_store));
        }
        if let Some(queued_work_poke) = self.queued_work_poke.clone() {
            env_builder = env_builder.with_queued_work_poke(queued_work_poke);
        }

        let live_replay_store = self
            .live_replay_store
            .take()
            .unwrap_or_else(|| Arc::new(InMemoryLiveReplayStore::default()));

        Ok(LashCore {
            env: env_builder.build(),
            policy,
            modes: Arc::new(self.modes),
            default_mode,
            store_factory: self.store_factory,
            plugin_factories: Arc::new(plugin_factories),
            provider: self.provider,
            live_replay_store,
            process_observer: process_registry.map(ProcessWorkObserver::new),
            process_work_runner: Arc::new(ProcessWorkRunnerSlot::new(process_work_runner)),
        })
    }

    /// Decide how a built [`LashCore`] sources its process work runner.
    ///
    /// - no registry => nothing to run ([`ProcessWorkRunnerSetup::None`]);
    /// - external driver wired => use it ([`ProcessWorkRunnerSetup::External`]);
    /// - inline registry wired => lazily spawn the default inline runner on first open. Its
    ///   [`DurableProcessWorkerConfig`] is built eagerly when a store factory is
    ///   present; without one the inline worker cannot rebuild session runtimes.
    fn resolve_process_work_runner(
        process_work_source: &ProcessWorkSource,
        worker_plugin_host: &PluginHost,
        core: &RuntimeHostConfig,
        store_factory: Option<&Arc<dyn SessionStoreFactory>>,
        policy: &SessionPolicy,
        residency: lash_core::Residency,
        host_event_store: Option<&Arc<dyn lash_core::HostEventStore>>,
    ) -> Result<ProcessWorkRunnerSetup> {
        let process_registry = match process_work_source {
            ProcessWorkSource::None => return Ok(ProcessWorkRunnerSetup::None),
            ProcessWorkSource::External(driver) => {
                return Ok(ProcessWorkRunnerSetup::External {
                    driver: driver.clone(),
                });
            }
            ProcessWorkSource::Inline { registry } => Arc::clone(registry),
        };
        // The worker rebuilds a session runtime per process, so it needs a store
        // factory; without one the default runner could not execute anything, so
        // fail loudly rather than silently leave processes unexecuted.
        let Some(store_factory) = store_factory else {
            return Err(EmbedError::ProcessRegistryRequiresStoreFactory);
        };
        // The worker rebuilds with the *same* plugin host the live runtime uses
        // for the default mode — including the mode preset's protocol plugin
        // (which supplies the protocol session capability) and the
        // process-lifecycle abilities. The bare builder `plugin_factories` omit
        // the preset factory (added per-mode by `build_plugin_host_for_mode`), so
        // a worker built from them would fail to rebuild a runtime ("missing
        // protocol session capability").
        let config = Box::new(
            DurableProcessWorkerConfig::new(
                Arc::new(worker_plugin_host.clone()),
                core.clone(),
                Arc::clone(store_factory),
                process_registry,
            )
            .with_session_policy(policy.clone())
            .with_host_event_store(
                host_event_store
                    .cloned()
                    .unwrap_or_else(|| Arc::new(lash_core::InMemoryHostEventStore::default())),
            )
            .with_residency(residency),
        );
        Ok(ProcessWorkRunnerSetup::LazyDefault { config })
    }

    pub fn advanced(self) -> AdvancedLashCoreBuilder {
        AdvancedLashCoreBuilder { builder: self }
    }

    pub fn process_registry(mut self, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        self.process_work_source = ProcessWorkSource::Inline {
            registry: process_registry,
        };
        self
    }

    pub fn host_event_store(mut self, store: Arc<dyn lash_core::HostEventStore>) -> Self {
        self.host_event_store = Some(store);
        self
    }

    /// Configure an externally owned process work runner.
    ///
    /// Durable hosts construct a [`ProcessWorkDriver`] from the same process
    /// registry and wake handle used by their deployment runner, then pass it
    /// here. The driver registry becomes the core's process registry and no
    /// inline runner is spawned.
    pub fn process_work_driver(mut self, driver: ProcessWorkDriver) -> Self {
        self.process_work_source = ProcessWorkSource::External(driver);
        self
    }

    /// Wire the wake handle of an externally owned queued-work runner. The
    /// runtime pokes it whenever new queued work lands for a session.
    pub fn queued_work_poke(mut self, poke: QueuedWorkPoke) -> Self {
        self.queued_work_poke = Some(poke);
        self
    }
}

pub(crate) fn build_plugin_host_for_mode(
    modes: &BTreeMap<ModeId, ModePreset>,
    mode: &ModeId,
    common_factories: &[Arc<dyn PluginFactory>],
    extra_factories: Vec<Arc<dyn PluginFactory>>,
    process_lifecycle: bool,
) -> Result<PluginHost> {
    let preset = modes
        .get(mode)
        .ok_or_else(|| EmbedError::ModeNotInstalled { mode: mode.clone() })?;
    let mut factories = Vec::with_capacity(1 + common_factories.len() + extra_factories.len());
    factories.push(Arc::clone(&preset.factory));
    factories.extend(common_factories.iter().cloned());
    factories.extend(extra_factories);
    let mut plugin_host = PluginHost::new(factories);
    if process_lifecycle {
        let abilities = plugin_host
            .lashlang_abilities()
            .with_processes()
            .with_sleep()
            .with_process_signals();
        plugin_host = plugin_host.with_lashlang_abilities(abilities);
    }
    Ok(plugin_host)
}

impl PromptLayerSink for LashCoreBuilder {
    fn prompt_layer_mut(&mut self) -> &mut PromptLayer {
        self.prompt.get_or_insert_with(PromptLayer::new)
    }
}

pub struct AdvancedLashCoreBuilder {
    builder: LashCoreBuilder,
}

impl AdvancedLashCoreBuilder {
    pub fn runtime_host_config(mut self, core: lash_core::RuntimeHostConfig) -> Self {
        self.builder.runtime_host_config = Some(core);
        self
    }

    pub fn plugin_host(mut self, plugin_host: PluginHost) -> Self {
        self.builder.plugin_host = Some(plugin_host);
        self
    }

    pub fn build(self) -> Result<LashCore> {
        self.builder.build()
    }
}
