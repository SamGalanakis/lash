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
    /// The host wired an explicit runner (e.g. the Restate ingress-client
    /// runner) and handed its poke to the core.
    Explicit { poke: ProcessWorkPoke },
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
                    ProcessWorkRunnerSetup::Explicit { poke } => Some(poke.clone()),
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

    /// Quickstart preset for `standard` mode. Defaults to the named in-memory
    /// effect controller and stores ([`LashCoreBuilder::in_memory_stores`]);
    /// for durable deployments build from [`LashCore::builder`] and name the
    /// stores explicitly, or override them with the builder setters.
    pub fn standard() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::standard())
            .default_mode(ModeId::standard())
            .plugins(default_runtime_stack())
            .in_memory_stores()
    }

    /// Quickstart preset for `rlm` mode. Defaults to the named in-memory
    /// effect controller and stores ([`LashCoreBuilder::in_memory_stores`]);
    /// for durable deployments build from [`LashCore::builder`] and name the
    /// stores explicitly, or override them with the builder setters.
    pub fn rlm() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::rlm())
            .default_mode(ModeId::rlm())
            .plugins(default_runtime_stack())
            .in_memory_stores()
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
        }
    }

    pub async fn delete_session(&self, session_id: impl AsRef<str>) -> Result<SessionDeleteReport> {
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
                None,
            );
            let outcome = self
                .env
                .core
                .effect_controller
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

    pub fn durable_process_worker_config(
        &self,
        process_registry: Arc<dyn ProcessRegistry>,
    ) -> Result<DurableProcessWorkerConfig> {
        self.durable_process_worker_config_with_plugins(process_registry, Vec::new())
    }

    pub fn durable_process_worker_config_with_plugins(
        &self,
        process_registry: Arc<dyn ProcessRegistry>,
        extra_plugin_factories: impl IntoIterator<Item = Arc<dyn PluginFactory>>,
    ) -> Result<DurableProcessWorkerConfig> {
        let Some(store_factory) = self.store_factory.as_ref() else {
            return Err(EmbedError::MissingProcessWorkerStoreFactory);
        };
        let mut factories = self.plugin_factories.as_ref().clone();
        factories.extend(extra_plugin_factories);
        Ok(DurableProcessWorkerConfig::from_plugin_factories(
            factories,
            self.env.core.clone(),
            Arc::clone(store_factory),
            process_registry,
        )
        .with_session_policy(self.policy.clone()))
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
    // `RuntimeCoreConfig` has no `Default`: the three host-owned durability
    // dependencies must be named. They are collected here and resolved in
    // `build()`, which errors if any is unset — unless `in_memory_stores()`
    // opted into the named in-memory versions.
    effect_controller: Option<Arc<dyn RuntimeEffectController>>,
    attachment_store: Option<Arc<dyn AttachmentStore>>,
    lashlang_artifact_store: Option<Arc<dyn lash_core::LashlangArtifactStore>>,
    in_memory_stores: bool,
    // Benign core overrides applied on top of the resolved core.
    prompt: Option<PromptLayer>,
    trace_sink: Option<Arc<dyn lash_trace::TraceSink>>,
    trace_level: Option<lash_trace::TraceLevel>,
    trace_context: Option<lash_trace::TraceContext>,
    termination: Option<TerminationPolicy>,
    // Advanced full-config override; used as the base core when present.
    runtime_core_config: Option<RuntimeCoreConfig>,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    plugin_stack: PluginStack,
    plugin_host: Option<PluginHost>,
    residency: Option<Residency>,
    process_registry: Option<Arc<dyn ProcessRegistry>>,
    // Process-work-runner wiring. A durable host registers its own runner and
    // hands the core the poke via `with_process_work_runner`; otherwise the
    // default inline runner is spawned lazily on first `session().open()`
    // unless `disable_default_process_work_runner` opted out (the loud guard
    // then rejects a registry with no runner).
    process_work_poke: Option<ProcessWorkPoke>,
    disable_default_process_work_runner: bool,
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

    pub fn mode(mut self, mode: ModeId) -> Self {
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
    /// cache, shared across the session tree). Required for RLM durability
    /// unless [`in_memory_stores`](Self::in_memory_stores) is used; a SQLite
    /// store such as `lash_sqlite_store::Store` implements it.
    pub fn lashlang_artifact_store(
        mut self,
        artifact_store: Arc<dyn lash_core::LashlangArtifactStore>,
    ) -> Self {
        self.lashlang_artifact_store = Some(artifact_store);
        self
    }

    /// Set the runtime effect controller — the durability boundary every turn
    /// crosses. Required unless [`in_memory_stores`](Self::in_memory_stores) is
    /// used. Pass [`InlineRuntimeEffectController`](crate::advanced::InlineRuntimeEffectController)
    /// for in-process execution, or a workflow-backed controller for durable
    /// turn execution.
    pub fn effect_controller(
        mut self,
        effect_controller: Arc<dyn RuntimeEffectController>,
    ) -> Self {
        self.effect_controller = Some(effect_controller);
        self
    }

    /// Opt into the named in-memory effect controller, Lashlang artifact
    /// store, and attachment store.
    ///
    /// Convenient for quickstarts, tests, and local experiments; not durable.
    /// Explicit calls to [`effect_controller`](Self::effect_controller),
    /// [`lashlang_artifact_store`](Self::lashlang_artifact_store), or
    /// [`attachment_store`](Self::attachment_store) still override the
    /// corresponding in-memory default.
    pub fn in_memory_stores(mut self) -> Self {
        self.in_memory_stores = true;
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

    pub fn trace_sink(mut self, trace_sink: Option<Arc<dyn lash_trace::TraceSink>>) -> Self {
        self.trace_sink = trace_sink;
        self
    }

    pub fn trace_jsonl_path(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.trace_sink = path.map(|path| {
            Arc::new(lash_trace::JsonlTraceSink::new(path)) as Arc<dyn lash_trace::TraceSink>
        });
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

    /// Resolve the runtime core config, requiring the three host-owned
    /// durability dependencies to have been named (or `in_memory_stores()` to
    /// have opted into the in-memory versions).
    fn resolve_runtime_core_config(&mut self) -> Result<RuntimeCoreConfig> {
        if let Some(base) = self.runtime_core_config.take() {
            return Ok(self.apply_core_overrides(base));
        }
        if self.in_memory_stores {
            return Ok(self.apply_core_overrides(RuntimeCoreConfig::in_memory()));
        }
        let effect_controller = self
            .effect_controller
            .take()
            .ok_or(EmbedError::MissingEffectController)?;
        let lashlang_artifact_store = self
            .lashlang_artifact_store
            .take()
            .ok_or(EmbedError::MissingLashlangArtifactStore)?;
        let attachment_store = self
            .attachment_store
            .take()
            .ok_or(EmbedError::MissingAttachmentStore)?;
        let core =
            RuntimeCoreConfig::new(effect_controller, lashlang_artifact_store, attachment_store);
        Ok(self.apply_core_overrides(core))
    }

    /// Apply benign + still-set dependency overrides on top of a base core.
    fn apply_core_overrides(&mut self, mut core: RuntimeCoreConfig) -> RuntimeCoreConfig {
        if let Some(effect_controller) = self.effect_controller.take() {
            core = core.with_effect_controller(effect_controller);
        }
        if let Some(attachment_store) = self.attachment_store.take() {
            core = core.with_attachment_store(attachment_store);
        }
        if let Some(artifact_store) = self.lashlang_artifact_store.take() {
            core = core.with_lashlang_artifact_store(artifact_store);
        }
        if let Some(prompt) = self.prompt.take() {
            core = core.with_prompt_layer(prompt);
        }
        if let Some(trace_sink) = self.trace_sink.take() {
            core = core.with_trace_sink(Some(trace_sink));
        }
        if let Some(trace_level) = self.trace_level.take() {
            core = core.with_trace_level(trace_level);
        }
        if let Some(trace_context) = self.trace_context.take() {
            core = core.with_trace_context(trace_context);
        }
        if let Some(termination) = self.termination.take() {
            core = core.with_termination(termination);
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

        if let Some(process_registry) = self.process_registry.as_ref() {
            if process_registry.durability_tier() == DurabilityTier::Durable
                && session_store_tier != Some(DurabilityTier::Durable)
            {
                return Err(EmbedError::DurableProcessRegistryRequiresStoreFactory);
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

        let mut core = self.resolve_runtime_core_config()?;
        if let Some(provider) = self.provider.clone() {
            core = core
                .with_provider_resolver(Arc::new(lash_core::SingleProviderResolver::new(provider)));
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
            self.process_registry.is_some(),
        )?;

        // Resolve the process work runner before the registry is moved into the
        // environment. The default inline runner's config is built eagerly so a
        // missing store factory fails loudly at build, not at first open. It is
        // built from the *default-mode* plugin host (preset protocol plugin +
        // process-lifecycle abilities), the same host the live runtime uses, so
        // the worker can rebuild a runtime for a default-mode process.
        let process_work_runner = Self::resolve_process_work_runner(
            self.process_registry.as_ref(),
            &default_plugin_host,
            &core,
            // The worker rebuilds sessions with the same factory `build()` wires
            // below: `child_store_factory.or(store_factory)`.
            self.child_store_factory
                .as_ref()
                .or(self.store_factory.as_ref()),
            &policy,
            self.process_work_poke.take(),
            self.disable_default_process_work_runner,
            self.residency.unwrap_or_default(),
        )?;

        let mut env_builder = RuntimeEnvironment::builder()
            .with_plugin_host(Arc::new(default_plugin_host))
            .with_runtime_core_config(core);
        if let Some(process_registry) = self.process_registry {
            env_builder = env_builder.with_process_registry(process_registry);
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

        Ok(LashCore {
            env: env_builder.build(),
            policy,
            modes: Arc::new(self.modes),
            default_mode,
            store_factory: self.store_factory,
            plugin_factories: Arc::new(plugin_factories),
            provider: self.provider,
            process_work_runner: Arc::new(ProcessWorkRunnerSlot::new(process_work_runner)),
        })
    }

    /// Decide how a built [`LashCore`] sources its process work runner.
    ///
    /// - no registry => nothing to run ([`ProcessWorkRunnerSetup::None`]);
    /// - explicit poke wired => use it ([`ProcessWorkRunnerSetup::Explicit`]);
    /// - default disabled, registry present, no explicit poke => loud guard
    ///   ([`EmbedError::ProcessRegistryWithoutWorkRunner`], Decision 4);
    /// - otherwise lazily spawn the default inline runner on first open. Its
    ///   [`DurableProcessWorkerConfig`] is built eagerly when a store factory is
    ///   present; without one the inline worker cannot rebuild session runtimes,
    ///   so no runner is spawned (the registry-only inline host keeps its prior
    ///   behavior — `build()` still succeeds).
    fn resolve_process_work_runner(
        process_registry: Option<&Arc<dyn ProcessRegistry>>,
        worker_plugin_host: &PluginHost,
        core: &RuntimeCoreConfig,
        store_factory: Option<&Arc<dyn SessionStoreFactory>>,
        policy: &SessionPolicy,
        explicit_poke: Option<ProcessWorkPoke>,
        default_disabled: bool,
        residency: lash_core::Residency,
    ) -> Result<ProcessWorkRunnerSetup> {
        let Some(process_registry) = process_registry else {
            return Ok(ProcessWorkRunnerSetup::None);
        };
        if let Some(poke) = explicit_poke {
            return Ok(ProcessWorkRunnerSetup::Explicit { poke });
        }
        if default_disabled {
            return Err(EmbedError::ProcessRegistryWithoutWorkRunner);
        }
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
                Arc::clone(process_registry),
            )
            .with_session_policy(policy.clone())
            .with_residency(residency),
        );
        Ok(ProcessWorkRunnerSetup::LazyDefault { config })
    }

    pub fn advanced(self) -> AdvancedLashCoreBuilder {
        AdvancedLashCoreBuilder { builder: self }
    }

    pub fn process_registry(mut self, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        self.process_registry = Some(process_registry);
        self
    }

    /// Register an explicit process work runner by handing the core the poke
    /// that wakes it. Durable hosts spawn their own [`ProcessWorkRunner`] (e.g.
    /// the Restate ingress-client runner), take its
    /// [`poke_handle`](lash_core::ProcessWorkRunner::poke_handle), and pass it
    /// here so the process control seam can make consumption prompt. Suppresses
    /// the default inline runner.
    pub fn with_process_work_runner(mut self, poke: ProcessWorkPoke) -> Self {
        self.process_work_poke = Some(poke);
        self
    }

    /// Disable the default inline process work runner. With a process registry
    /// configured and no explicit runner supplied via
    /// [`with_process_work_runner`](Self::with_process_work_runner), `build()`
    /// then fails with [`EmbedError::ProcessRegistryWithoutWorkRunner`] rather
    /// than silently leaving non-terminal processes unexecuted.
    pub fn disable_default_process_work_runner(mut self) -> Self {
        self.disable_default_process_work_runner = true;
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
    pub fn runtime_core_config(mut self, core: lash_core::RuntimeCoreConfig) -> Self {
        self.builder.runtime_core_config = Some(core);
        self
    }

    pub fn plugin_host(mut self, plugin_host: PluginHost) -> Self {
        self.builder.plugin_host = Some(plugin_host);
        self
    }

    pub fn residency(mut self, residency: Residency) -> Self {
        self.builder.residency = Some(residency);
        self
    }

    pub fn process_registry(mut self, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        self.builder.process_registry = Some(process_registry);
        self
    }

    pub fn effect_controller(
        mut self,
        effect_controller: Arc<dyn RuntimeEffectController>,
    ) -> Self {
        self.builder.effect_controller = Some(effect_controller);
        self
    }

    pub fn termination(mut self, termination: TerminationPolicy) -> Self {
        self.builder.termination = Some(termination);
        self
    }

    /// Register an explicit process work runner by handing the core its poke.
    /// See [`LashCoreBuilder::with_process_work_runner`].
    pub fn with_process_work_runner(mut self, poke: ProcessWorkPoke) -> Self {
        self.builder.process_work_poke = Some(poke);
        self
    }

    /// Disable the default inline process work runner. See
    /// [`LashCoreBuilder::disable_default_process_work_runner`].
    pub fn disable_default_process_work_runner(mut self) -> Self {
        self.builder.disable_default_process_work_runner = true;
        self
    }

    pub fn build(self) -> Result<LashCore> {
        self.builder.build()
    }
}
