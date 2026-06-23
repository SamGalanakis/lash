use crate::support::*;
use lash_core::runtime::{
    ProcessCommand, ProcessEffectOutcome, RuntimeEffectCommand, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeInvocation,
    RuntimeScope,
};

type RuntimeHostInstaller =
    Arc<dyn Fn(RuntimeHostConfig, &PluginHost) -> Result<RuntimeHostConfig> + Send + Sync>;

#[derive(Clone)]
pub struct LashCore {
    pub(crate) env: RuntimeEnvironment,
    pub(crate) policy: SessionPolicy,
    pub(crate) protocol_factory: Option<Arc<dyn PluginFactory>>,
    pub(crate) store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub(crate) plugin_factories: Arc<Vec<Arc<dyn PluginFactory>>>,
    pub(crate) provider: Option<ProviderHandle>,
    pub(crate) live_replay_store: Arc<dyn LiveReplayStore>,
    pub(crate) runtime_host_installer: Option<RuntimeHostInstaller>,
    /// Shared resolution of host-owned work drivers. Shared across `LashCore`
    /// clones so inline process and queued drivers are constructed at most once.
    pub(crate) work_driver: Arc<InlineWorkDriverSlot>,
}

/// How a [`LashCore`] resolves its process work driver, decided at `build()`
/// and shared across clones.
pub(crate) enum ProcessWorkDriverSetup {
    /// No process registry is wired; there is nothing to run.
    None,
    /// Lazily construct the default inline process driver on first
    /// `session().open()`. A store factory is required to build the config (the
    /// worker rebuilds a session runtime per process); a registry with no store
    /// factory is rejected at build with
    /// [`EmbedError::ProcessRegistryRequiresStoreFactory`].
    LazyDefault {
        config: Box<DurableProcessWorkerConfig>,
    },
    /// The host wired an external driver.
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

    #[cfg(feature = "rlm")]
    fn has_registry(&self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Clone, Default)]
pub(crate) enum QueuedWorkSource {
    None,
    #[default]
    LazyDefault,
    External(QueuedWorkDriver),
}

pub(crate) enum QueuedWorkDriverSetup {
    None,
    LazyDefault {
        config: Arc<InlineQueuedWorkRunConfig>,
    },
    External {
        driver: QueuedWorkDriver,
    },
}

pub(crate) struct InlineWorkDriverSetup {
    process: ProcessWorkDriverSetup,
    queued: QueuedWorkDriverSetup,
}

#[derive(Clone, Default)]
pub(crate) struct ResolvedWorkDrivers {
    pub(crate) process: Option<ProcessWorkDriver>,
    pub(crate) queued: Option<QueuedWorkDriver>,
    pub(crate) drive_process_on_open: bool,
}

/// Shared, lazily-initialized host-work state for a [`LashCore`].
///
/// The once-guard ([`tokio::sync::OnceCell`]) constructs inline drivers exactly
/// once across `LashCore` clones, on the first `session().open()` or admin path
/// that needs them.
pub(crate) struct InlineWorkDriverSlot {
    setup: InlineWorkDriverSetup,
    drivers: tokio::sync::OnceCell<ResolvedWorkDrivers>,
    phase_probe_slot: Option<lash_core::runtime::RuntimeTurnPhaseProbeSlot>,
}

impl InlineWorkDriverSlot {
    fn new(setup: InlineWorkDriverSetup) -> Self {
        let phase_probe_slot = match &setup.process {
            ProcessWorkDriverSetup::LazyDefault { config } => {
                Some(config.turn_phase_probe_slot.clone())
            }
            ProcessWorkDriverSetup::None | ProcessWorkDriverSetup::External { .. } => None,
        };
        Self {
            setup,
            drivers: tokio::sync::OnceCell::new(),
            phase_probe_slot,
        }
    }

    /// Resolve host work drivers for a session host. Idempotent: the once-guard
    /// ensures inline drivers are constructed once.
    pub(crate) async fn drivers(&self) -> ResolvedWorkDrivers {
        self.drivers
            .get_or_init(|| async {
                let queued = match &self.setup.queued {
                    QueuedWorkDriverSetup::None => None,
                    QueuedWorkDriverSetup::External { driver } => Some(driver.clone()),
                    QueuedWorkDriverSetup::LazyDefault { config } => Some(QueuedWorkDriver::new(
                        Arc::new(InlineQueuedWorkRunHandle::new(Arc::clone(config))),
                    )),
                };
                let (process, drive_process_on_open) = match &self.setup.process {
                    ProcessWorkDriverSetup::None => (None, false),
                    ProcessWorkDriverSetup::External { driver } => (Some(driver.clone()), false),
                    ProcessWorkDriverSetup::LazyDefault { config } => {
                        let mut config = (**config).clone();
                        if let Some(driver) = queued.clone() {
                            config = config.with_queued_work_driver(driver);
                        }
                        let registry = Arc::clone(&config.process_registry);
                        let worker = DurableProcessWorker::new(config);
                        (Some(ProcessWorkDriver::inline(registry, worker)), true)
                    }
                };
                ResolvedWorkDrivers {
                    process,
                    queued,
                    drive_process_on_open,
                }
            })
            .await
            .clone()
    }

    pub(crate) fn phase_probe_slot(&self) -> Option<lash_core::runtime::RuntimeTurnPhaseProbeSlot> {
        self.phase_probe_slot.clone()
    }

    fn configured_process_work_driver(&self) -> Option<ProcessWorkDriver> {
        match &self.setup.process {
            ProcessWorkDriverSetup::External { driver } => Some(driver.clone()),
            ProcessWorkDriverSetup::None | ProcessWorkDriverSetup::LazyDefault { .. } => None,
        }
    }

    fn configured_queued_work_driver(&self) -> Option<QueuedWorkDriver> {
        match &self.setup.queued {
            QueuedWorkDriverSetup::External { driver } => Some(driver.clone()),
            QueuedWorkDriverSetup::None | QueuedWorkDriverSetup::LazyDefault { .. } => None,
        }
    }
}

pub(crate) struct InlineQueuedWorkRunConfig {
    env: RuntimeEnvironment,
    policy: SessionPolicy,
    protocol_factory: Option<Arc<dyn PluginFactory>>,
    plugin_factories: Arc<Vec<Arc<dyn PluginFactory>>>,
    store_factory: Arc<dyn SessionStoreFactory>,
    live_replay_store: Arc<dyn LiveReplayStore>,
    runtime_host_installer: Option<RuntimeHostInstaller>,
}

impl InlineQueuedWorkRunConfig {
    fn new(
        env: RuntimeEnvironment,
        policy: SessionPolicy,
        protocol_factory: Option<Arc<dyn PluginFactory>>,
        plugin_factories: Arc<Vec<Arc<dyn PluginFactory>>>,
        store_factory: Arc<dyn SessionStoreFactory>,
        live_replay_store: Arc<dyn LiveReplayStore>,
        runtime_host_installer: Option<RuntimeHostInstaller>,
    ) -> Self {
        Self {
            env,
            policy,
            protocol_factory,
            plugin_factories,
            store_factory,
            live_replay_store,
            runtime_host_installer,
        }
    }
}

struct InlineQueuedWorkRunHandle {
    config: Arc<InlineQueuedWorkRunConfig>,
}

impl InlineQueuedWorkRunHandle {
    fn new(config: Arc<InlineQueuedWorkRunConfig>) -> Self {
        Self { config }
    }
}

#[async_trait]
impl QueuedWorkRunHandle for InlineQueuedWorkRunHandle {
    async fn run_queued_work(
        &self,
        request: QueuedWorkRunRequest,
    ) -> std::result::Result<(), lash_core::PluginError> {
        let Some(session_id) = request.session_id else {
            return Ok(());
        };
        let reason = request.reason;
        let mut policy = self.config.policy.clone();
        policy.session_id = Some(session_id.clone());
        let store = self
            .config
            .store_factory
            .create_store(&SessionStoreCreateRequest {
                session_id: session_id.clone(),
                relation: SessionRelation::default(),
                policy: policy.clone(),
            })
            .await
            .map_err(lash_core::PluginError::Session)?;
        let state = crate::session::load_state_for_residency(
            self.config.env.residency,
            &session_id,
            &policy,
            store.as_ref(),
        )
        .await
        .map_err(|err| lash_core::PluginError::Session(err.to_string()))?;
        let plugin_host = build_plugin_host(
            self.config.protocol_factory.as_ref(),
            self.config.plugin_factories.as_ref(),
            Vec::new(),
        )
        .map_err(|err| lash_core::PluginError::Session(err.to_string()))?;
        let mut env = self.config.env.clone();
        env.core = match &self.config.runtime_host_installer {
            Some(install) => install(env.core.clone(), &plugin_host)
                .map_err(|err| lash_core::PluginError::Session(err.to_string()))?,
            None => env.core.clone(),
        };
        env.plugin_host = Some(Arc::new(plugin_host));
        let effect_host = Arc::clone(&env.core.control.effect_host);
        let runtime = LashRuntime::from_environment(&env, policy, state, Some(store))
            .await
            .map_err(|err| lash_core::PluginError::Session(err.to_string()))?;
        let handle = RuntimeHandle::with_live_replay_store(
            runtime,
            Arc::clone(&self.config.live_replay_store),
        );
        let scope = lash_core::ExecutionScope::queue_drain(session_id, reason);
        let scoped = effect_host
            .scoped(scope)
            .map_err(|err| lash_core::PluginError::Session(err.to_string()))?;
        crate::turn::stream_next_queued_prepared_turn(
            &handle,
            crate::turn::TurnSinks::default(),
            scoped,
            CancellationToken::new(),
            &[],
        )
        .await
        .map_err(|err| lash_core::PluginError::Session(err.to_string()))?;
        Ok(())
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

    pub fn session(&self, session_id: impl Into<String>) -> SessionBuilder {
        SessionBuilder {
            core: self.clone(),
            session_id: session_id.into(),
            spec: SessionSpec::inherit(),
            parent_session_id: None,
            session_execution_owner_id: None,
            store: None,
            provider: None,
            active_plugins: Vec::new(),
            plugin_factories: Vec::new(),
        }
    }

    pub fn triggers(&self) -> crate::admin::CoreTriggerAdmin {
        crate::admin::CoreTriggerAdmin { core: self.clone() }
    }

    pub fn processes(&self) -> crate::admin::Processes {
        crate::admin::Processes { core: self.clone() }
    }

    pub fn completions(&self) -> crate::admin::Completions {
        crate::admin::Completions { core: self.clone() }
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
                        RuntimeEffectCommand::process(ProcessCommand::DeleteSession {
                            session_id: session_id.clone(),
                        }),
                    ),
                    RuntimeEffectLocalExecutor::processes(Arc::clone(process_registry)),
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
        if let Some(trigger_store) = self.env.trigger_store.as_ref() {
            trigger_store
                .delete_session_subscriptions(&session_id)
                .await
                .map_err(|err| EmbedError::SessionDeleteProcess {
                    session_id: session_id.clone(),
                    message: err.to_string(),
                })?;
        }
        self.env
            .core
            .control
            .effect_host
            .revoke_await_events_for_session(&session_id)
            .await
            .map_err(|err| EmbedError::SessionDeleteProcess {
                session_id: session_id.clone(),
                message: err.to_string(),
            })?;
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
        let plugin_host = build_plugin_host(
            self.protocol_factory.as_ref(),
            self.plugin_factories.as_ref(),
            extra_plugin_factories.into_iter().collect(),
        )?;
        let runtime_host =
            self.runtime_host_for_plugin_host(self.env.core.clone(), &plugin_host)?;
        let mut config = DurableProcessWorkerConfig::new(
            Arc::new(plugin_host),
            runtime_host,
            Arc::clone(store_factory),
            process_registry,
        )
        .with_session_policy(self.policy.clone())
        .with_residency(self.env.residency);
        if let Some(trigger_store) = self.env.trigger_store.as_ref() {
            config = config.with_trigger_store(Arc::clone(trigger_store));
        }
        if let Some(driver) = self.work_driver.configured_process_work_driver() {
            config = config.with_process_work_driver(driver);
        }
        if let Some(driver) = self.work_driver.configured_queued_work_driver() {
            config = config.with_queued_work_driver(driver);
        }
        Ok(config)
    }

    pub(crate) fn runtime_host_for_plugin_host(
        &self,
        runtime_host: RuntimeHostConfig,
        plugin_host: &PluginHost,
    ) -> Result<RuntimeHostConfig> {
        match &self.runtime_host_installer {
            Some(install) => install(runtime_host, plugin_host),
            None => Ok(runtime_host),
        }
    }
}

fn default_runtime_stack() -> PluginStack {
    lash_plugin_tool_output_budget::tool_output_budget_stack()
}

#[derive(Clone)]
pub struct StandardCore {
    core: LashCore,
}

impl StandardCore {
    pub fn builder() -> StandardCoreBuilder {
        StandardCoreBuilder {
            inner: LashCore::builder()
                .protocol_plugin(Arc::new(
                    lash_protocol_standard::StandardProtocolPluginFactory::new(),
                ))
                .plugins(default_runtime_stack()),
        }
    }

    pub fn session(&self, session_id: impl Into<String>) -> SessionBuilder {
        self.core.session(session_id)
    }

    pub fn into_inner(self) -> LashCore {
        self.core
    }
}

impl std::ops::Deref for StandardCore {
    type Target = LashCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

pub struct StandardCoreBuilder {
    inner: LashCoreBuilder,
}

impl StandardCoreBuilder {
    pub fn build(self) -> Result<StandardCore> {
        self.inner.build().map(|core| StandardCore { core })
    }
}

impl PromptLayerSink for StandardCoreBuilder {
    fn prompt_layer_mut(&mut self) -> &mut PromptLayer {
        self.inner.prompt_layer_mut()
    }
}

#[cfg(feature = "rlm")]
#[derive(Clone)]
pub struct RlmCore {
    core: LashCore,
    surface_config: lash_protocol_rlm::RlmProtocolPluginConfig,
    process_lifecycle_available: bool,
    lashlang_artifact_store: Arc<dyn lash_lashlang_runtime::LashlangArtifactStore>,
}

#[cfg(feature = "rlm")]
impl RlmCore {
    pub fn builder() -> RlmCoreBuilder {
        RlmCoreBuilder {
            inner: LashCore::builder().plugins(default_runtime_stack()),
            config: lash_protocol_rlm::RlmProtocolPluginConfig::default(),
            projection_resolver: Arc::new(lash_protocol_rlm::ProjectionRegistry::default()),
            deferred_tool_resolver: None,
            lashlang_artifact_store: None,
            lashlang_execution_sink: None,
        }
    }

    pub fn session(&self, session_id: impl Into<String>) -> RlmSessionBuilder {
        RlmSessionBuilder {
            builder: self.core.session(session_id),
            rlm_final_answer_format: None,
        }
    }

    pub fn into_inner(self) -> LashCore {
        self.core
    }

    pub fn lashlang_compile_surface(
        &self,
        request: crate::rlm::LashlangCompileSurfaceRequest,
    ) -> Result<crate::rlm::LashlangCompileSurface> {
        let plugin_host = build_plugin_host(
            self.core.protocol_factory.as_ref(),
            self.core.plugin_factories.as_ref(),
            request.extra_plugin_factories,
        )?;
        let plugins = plugin_host.build_session_with_parent(
            &request.session_id,
            None,
            None,
            lash_core::plugin::SessionAuthorityContext {
                plugin_options: request.execution_env_spec.plugin_options,
                ..Default::default()
            },
        )?;
        let tool_catalog = plugins.resolved_tool_catalog(&request.session_id)?;
        let surface = crate::rlm::rlm_lashlang_surface(
            &self.surface_config,
            self.process_lifecycle_available,
        )
        .with_plugin_extensions(plugin_host.extensions())
        .map_err(lash_core::PluginError::Registration)?;
        let host_environment = surface
            .host_environment(&tool_catalog)
            .map_err(lash_core::PluginError::Registration)?;
        Ok(crate::rlm::LashlangCompileSurface {
            host_environment,
            tool_catalog,
            surface,
        })
    }

    pub async fn compile_lashlang_module(
        &self,
        request: crate::rlm::LashlangModuleCompileRequest,
    ) -> std::result::Result<crate::rlm::ModuleCompileOutput, crate::rlm::LashlangModuleCompileError>
    {
        let surface = self
            .lashlang_compile_surface(crate::rlm::LashlangCompileSurfaceRequest {
                session_id: request.session_id,
                execution_env_spec: request.execution_env_spec,
                extra_plugin_factories: request.extra_plugin_factories,
            })
            .map_err(|err| {
                lashlang::ModuleCompileError::Link(lashlang::ModuleCompileDiagnostic {
                    stage: lashlang::ModuleCompileStage::Link,
                    message: err.to_string(),
                    offset: None,
                    span: None,
                    line: None,
                    column: None,
                    diagnostic: Some(err.to_string()),
                })
            })?;
        lashlang::compile_module(lashlang::ModuleCompileRequest {
            source: &request.source,
            environment: &surface.host_environment,
            artifact_store: Some(self.lashlang_artifact_store.as_ref()),
        })
        .await
    }
}

#[cfg(feature = "rlm")]
impl std::ops::Deref for RlmCore {
    type Target = LashCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

#[cfg(feature = "rlm")]
pub struct RlmCoreBuilder {
    inner: LashCoreBuilder,
    config: lash_protocol_rlm::RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn lash_protocol_rlm::ProjectionResolver>,
    deferred_tool_resolver: Option<lash_lashlang_runtime::SharedDeferredToolResolver>,
    lashlang_artifact_store: Option<Arc<dyn lash_lashlang_runtime::LashlangArtifactStore>>,
    lashlang_execution_sink: Option<Arc<dyn lash_trace::TraceSink>>,
}

#[cfg(feature = "rlm")]
impl RlmCoreBuilder {
    pub fn rlm_protocol_config(
        mut self,
        config: lash_protocol_rlm::RlmProtocolPluginConfig,
    ) -> Self {
        self.config = config;
        self
    }

    pub fn projection_resolver(
        mut self,
        projection_resolver: Arc<dyn lash_protocol_rlm::ProjectionResolver>,
    ) -> Self {
        self.projection_resolver = projection_resolver;
        self
    }

    /// Wire a host-provided RLM Deferred Tool Resolver. Most hosts ship none;
    /// the CLI's MCP-discovery example wires one.
    pub fn deferred_tool_resolver(
        mut self,
        resolver: lash_lashlang_runtime::SharedDeferredToolResolver,
    ) -> Self {
        self.deferred_tool_resolver = Some(resolver);
        self
    }

    pub fn lashlang_artifact_store(
        mut self,
        artifact_store: Arc<dyn lash_lashlang_runtime::LashlangArtifactStore>,
    ) -> Self {
        self.lashlang_artifact_store = Some(artifact_store);
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

    pub fn build(mut self) -> Result<RlmCore> {
        let artifact_store = self
            .lashlang_artifact_store
            .clone()
            .ok_or(EmbedError::MissingLashlangArtifactStore)?;
        if self.inner.effective_session_store_tier() == Some(DurabilityTier::Durable)
            && artifact_store.durability_tier()
                == lash_lashlang_runtime::LashlangDurabilityTier::Inline
        {
            return Err(EmbedError::DurableStorePeerRequired {
                facet: "artifact store",
            });
        }
        let process_lifecycle_available = self.inner.process_work_source.has_registry();
        let config = crate::rlm::rlm_protocol_config(self.config, process_lifecycle_available);
        let trace_context = self.inner.resolved_trace_context();
        let mut protocol_factory =
            lash_protocol_rlm::RlmProtocolPluginFactory::new(config.clone())
                .with_projection_resolver(Arc::clone(&self.projection_resolver))
                .with_lashlang_artifact_store(Arc::clone(&artifact_store))
                .with_lashlang_execution_trace(
                    self.lashlang_execution_sink.clone(),
                    trace_context.clone(),
                );
        if let Some(resolver) = self.deferred_tool_resolver.clone() {
            protocol_factory = protocol_factory.with_deferred_tool_resolver(resolver);
        }
        let protocol_factory = Arc::new(protocol_factory);
        let engine_artifact_store = Arc::clone(&artifact_store);
        let engine_config = config.clone();
        let engine_sink = self.lashlang_execution_sink.clone();
        self.inner.protocol_factory = Some(protocol_factory);
        self.inner.runtime_host_installer = Some(Arc::new(move |runtime_host, plugin_host| {
            let surface =
                crate::rlm::rlm_lashlang_surface(&engine_config, process_lifecycle_available)
                    .with_plugin_extensions(plugin_host.extensions())
                    .map_err(lash_core::PluginError::Registration)?;
            let engine = lash_lashlang_runtime::LashlangProcessEngine::new(
                Arc::clone(&engine_artifact_store),
                surface,
            )
            .with_execution_trace(
                engine_sink.clone(),
                runtime_host.tracing.trace_context.clone(),
            );
            Ok(runtime_host.with_process_engine(Arc::new(engine)))
        }));
        self.inner.build().map(|core| RlmCore {
            core,
            surface_config: config,
            process_lifecycle_available,
            lashlang_artifact_store: artifact_store,
        })
    }
}

#[cfg(feature = "rlm")]
impl PromptLayerSink for RlmCoreBuilder {
    fn prompt_layer_mut(&mut self) -> &mut PromptLayer {
        self.inner.prompt_layer_mut()
    }
}

macro_rules! forward_core_builder_methods {
    ($builder:ident) => {
        impl $builder {
            pub fn provider(mut self, provider: ProviderHandle) -> Self {
                self.inner = self.inner.provider(provider);
                self
            }

            pub fn model(mut self, model: lash_core::ModelSpec) -> Self {
                self.inner = self.inner.model(model);
                self
            }

            pub fn max_turns(mut self, max_turns: usize) -> Self {
                self.inner = self.inner.max_turns(max_turns);
                self
            }

            pub fn session_spec(mut self, spec: SessionSpec) -> Self {
                self.inner = self.inner.session_spec(spec);
                self
            }

            pub fn store_factory(mut self, store_factory: Arc<dyn SessionStoreFactory>) -> Self {
                self.inner = self.inner.store_factory(store_factory);
                self
            }

            pub fn child_store_factory(
                mut self,
                store_factory: Arc<dyn SessionStoreFactory>,
            ) -> Self {
                self.inner = self.inner.child_store_factory(store_factory);
                self
            }

            pub fn attachment_store(mut self, attachment_store: Arc<dyn AttachmentStore>) -> Self {
                self.inner = self.inner.attachment_store(attachment_store);
                self
            }

            pub fn process_env_store(
                mut self,
                process_env_store: Arc<dyn ProcessExecutionEnvStore>,
            ) -> Self {
                self.inner = self.inner.process_env_store(process_env_store);
                self
            }

            pub fn effect_host(mut self, effect_host: Arc<dyn EffectHost>) -> Self {
                self.inner = self.inner.effect_host(effect_host);
                self
            }

            pub fn tools(mut self, tools: Arc<dyn ToolProvider>) -> Self {
                self.inner = self.inner.tools(tools);
                self
            }

            pub fn plugin(mut self, plugin: Arc<dyn PluginFactory>) -> Self {
                self.inner = self.inner.plugin(plugin);
                self
            }

            pub fn plugins(mut self, stack: PluginStack) -> Self {
                self.inner = self.inner.plugins(stack);
                self
            }

            pub fn configure_plugins(mut self, configure: impl FnOnce(&mut PluginStack)) -> Self {
                self.inner = self.inner.configure_plugins(configure);
                self
            }

            pub fn trace_sink(mut self, trace_sink: Arc<dyn lash_trace::TraceSink>) -> Self {
                self.inner = self.inner.trace_sink(trace_sink);
                self
            }

            pub fn trace_jsonl_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
                self.inner = self.inner.trace_jsonl_path(path);
                self
            }

            pub fn trace_level(mut self, trace_level: lash_trace::TraceLevel) -> Self {
                self.inner = self.inner.trace_level(trace_level);
                self
            }

            pub fn trace_context(mut self, trace_context: lash_trace::TraceContext) -> Self {
                self.inner = self.inner.trace_context(trace_context);
                self
            }

            pub fn termination(mut self, termination: TerminationPolicy) -> Self {
                self.inner = self.inner.termination(termination);
                self
            }

            pub fn residency(mut self, residency: Residency) -> Self {
                self.inner = self.inner.residency(residency);
                self
            }

            pub fn live_replay_store(
                mut self,
                live_replay_store: Arc<dyn LiveReplayStore>,
            ) -> Self {
                self.inner = self.inner.live_replay_store(live_replay_store);
                self
            }

            pub fn process_registry(mut self, process_registry: Arc<dyn ProcessRegistry>) -> Self {
                self.inner = self.inner.process_registry(process_registry);
                self
            }

            pub fn trigger_store(mut self, store: Arc<dyn lash_core::TriggerStore>) -> Self {
                self.inner = self.inner.trigger_store(store);
                self
            }

            pub fn process_work_driver(mut self, driver: ProcessWorkDriver) -> Self {
                self.inner = self.inner.process_work_driver(driver);
                self
            }

            pub fn queued_work_driver(mut self, driver: QueuedWorkDriver) -> Self {
                self.inner = self.inner.queued_work_driver(driver);
                self
            }

            pub fn disable_queued_work_driver(mut self) -> Self {
                self.inner = self.inner.disable_queued_work_driver();
                self
            }

            pub fn runtime_host_config(mut self, core: RuntimeHostConfig) -> Self {
                self.inner.runtime_host_config = Some(core);
                self
            }
        }
    };
}

forward_core_builder_methods!(StandardCoreBuilder);
#[cfg(feature = "rlm")]
forward_core_builder_methods!(RlmCoreBuilder);

#[derive(Default)]
pub struct LashCoreBuilder {
    pub(crate) protocol_factory: Option<Arc<dyn PluginFactory>>,
    session_spec: SessionSpec,
    provider: Option<ProviderHandle>,
    pub(crate) store_factory: Option<Arc<dyn SessionStoreFactory>>,
    child_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    // `RuntimeHostConfig` has no `Default`: the generic host-owned durability
    // dependencies must be named. They are collected here and resolved in
    // `build()`, which errors if any is unset.
    effect_host: Option<Arc<dyn EffectHost>>,
    attachment_store: Option<Arc<dyn AttachmentStore>>,
    process_env_store: Option<Arc<dyn ProcessExecutionEnvStore>>,
    trigger_store: Option<Arc<dyn lash_core::TriggerStore>>,
    // Benign core overrides applied on top of the resolved core.
    prompt: Option<PromptLayer>,
    trace_sink: Option<Arc<dyn lash_trace::TraceSink>>,
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
    queued_work_source: QueuedWorkSource,
    live_replay_store: Option<Arc<dyn LiveReplayStore>>,
    runtime_host_installer: Option<RuntimeHostInstaller>,
}

impl LashCoreBuilder {
    pub fn protocol_plugin(mut self, plugin: Arc<dyn PluginFactory>) -> Self {
        self.protocol_factory = Some(plugin);
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

    pub fn process_env_store(
        mut self,
        process_env_store: Arc<dyn ProcessExecutionEnvStore>,
    ) -> Self {
        self.process_env_store = Some(process_env_store);
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

    /// Resolve the runtime host config, requiring the generic host-owned
    /// durability dependencies to have been named.
    fn resolve_runtime_host_config(&mut self) -> Result<RuntimeHostConfig> {
        if let Some(base) = self.runtime_host_config.take() {
            return Ok(self.apply_core_overrides(base));
        }
        let effect_host = self
            .effect_host
            .take()
            .ok_or(EmbedError::MissingEffectHost)?;
        let attachment_store = self
            .attachment_store
            .take()
            .ok_or(EmbedError::MissingAttachmentStore)?;
        let process_env_store = self
            .process_env_store
            .take()
            .ok_or(EmbedError::MissingProcessEnvStore)?;
        let core = RuntimeHostConfig::new(effect_host, attachment_store, process_env_store);
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
        if let Some(process_env_store) = self.process_env_store.take() {
            core.durability.process_env_store = process_env_store;
        }
        if let Some(prompt) = self.prompt.take() {
            core.prompt.prompt = prompt;
        }
        if let Some(trace_sink) = self.trace_sink.take() {
            core.tracing.trace_sink = Some(trace_sink);
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
    fn effective_session_store_tier(&self) -> Option<DurabilityTier> {
        self.child_store_factory
            .as_ref()
            .or(self.store_factory.as_ref())
            .map(|factory| factory.durability_tier())
    }

    #[cfg(feature = "rlm")]
    fn resolved_trace_context(&self) -> lash_trace::TraceContext {
        self.trace_context
            .clone()
            .or_else(|| {
                self.runtime_host_config
                    .as_ref()
                    .map(|core| core.tracing.trace_context.clone())
            })
            .unwrap_or_default()
    }

    fn ensure_store_peer_coherence(&self) -> Result<()> {
        // Match `build()`'s wiring exactly: the session store factory it installs
        // is `child_store_factory.or(store_factory)` (child takes precedence, root
        // is the fallback). The coherence check must read the tier of that same
        // effective factory, or a host that wires only a durable child factory
        // (no root) is wrongly rejected though `build()` would wire it durably.
        let session_store_tier = self.effective_session_store_tier();
        let attachment_tier = self
            .attachment_store
            .as_ref()
            .map(|store| store.persistence().durability_tier())
            .or_else(|| {
                self.runtime_host_config.as_ref().map(|core| {
                    core.durability
                        .attachment_store
                        .persistence()
                        .durability_tier()
                })
            });
        let process_env_tier = self
            .process_env_store
            .as_ref()
            .map(|store| store.durability_tier())
            .or_else(|| {
                self.runtime_host_config
                    .as_ref()
                    .map(|core| core.durability.process_env_store.durability_tier())
            });
        let effect_host_tier = self
            .effect_host
            .as_ref()
            .map(|host| host.durability_tier())
            .or_else(|| {
                self.runtime_host_config
                    .as_ref()
                    .map(|core| core.control.effect_host.durability_tier())
            });
        let trigger_store_tier = self
            .trigger_store
            .as_ref()
            .map(|store| store.durability_tier());

        if session_store_tier == Some(DurabilityTier::Durable) {
            if attachment_tier == Some(DurabilityTier::Inline) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "attachment store",
                });
            }
            if process_env_tier == Some(DurabilityTier::Inline) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "process execution environment store",
                });
            }
        }

        if let Some(process_registry) = self.process_work_source.process_registry().as_ref()
            && process_registry.durability_tier() == DurabilityTier::Durable
        {
            if session_store_tier != Some(DurabilityTier::Durable) {
                return Err(EmbedError::DurableProcessRegistryRequiresStoreFactory);
            }
            if trigger_store_tier != Some(DurabilityTier::Durable) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "trigger store",
                });
            }
            if process_env_tier != Some(DurabilityTier::Durable) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "process execution environment store",
                });
            }
        }

        if trigger_store_tier == Some(DurabilityTier::Durable) {
            if session_store_tier != Some(DurabilityTier::Durable) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "session store factory",
                });
            }
            if process_env_tier != Some(DurabilityTier::Durable) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "process execution environment store",
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

        if effect_host_tier == Some(DurabilityTier::Durable) {
            if attachment_tier != Some(DurabilityTier::Durable) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "attachment store",
                });
            }
            if process_env_tier != Some(DurabilityTier::Durable) {
                return Err(EmbedError::DurableStorePeerRequired {
                    facet: "process execution environment store",
                });
            }
        }

        Ok(())
    }

    pub fn build(mut self) -> Result<LashCore> {
        self.ensure_store_peer_coherence()?;
        let protocol_factory = self.protocol_factory.clone();
        if protocol_factory.is_none() && self.plugin_host.is_none() {
            return Err(EmbedError::MissingProtocolPlugin);
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
        let default_plugin_host =
            build_plugin_host(protocol_factory.as_ref(), &plugin_factories, Vec::new())?;
        if let Some(install) = &self.runtime_host_installer {
            core = install(core, &default_plugin_host)?;
        }

        let process_registry = self.process_work_source.process_registry();

        // Resolve process work before the process source is moved into the
        // environment. The default inline driver's config is built
        // eagerly so a missing store factory fails loudly at build, not at
        // first open. It is built from the same single-protocol plugin host the
        // live runtime uses, so the worker can rebuild a runtime for a process.
        let process_work_driver = Self::resolve_process_work_driver(
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
            self.trigger_store.as_ref(),
        )?;

        let live_replay_clock = Arc::clone(&core.clock);
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
        if let Some(trigger_store) = self.trigger_store.as_ref() {
            env_builder = env_builder.with_trigger_store(Arc::clone(trigger_store));
        }
        let live_replay_store = self.live_replay_store.take().unwrap_or_else(|| {
            Arc::new(InMemoryLiveReplayStore::with_clock(
                lash_core::InMemoryLiveReplayStoreConfig::default(),
                live_replay_clock,
            ))
        });
        let env = env_builder.build();
        let queued_work_driver = Self::resolve_queued_work_driver(
            &self.queued_work_source,
            env.clone(),
            policy.clone(),
            protocol_factory.clone(),
            Arc::new(plugin_factories.clone()),
            self.child_store_factory
                .as_ref()
                .or(self.store_factory.as_ref()),
            Arc::clone(&live_replay_store),
            self.runtime_host_installer.clone(),
        );
        let work_driver = InlineWorkDriverSetup {
            process: process_work_driver,
            queued: queued_work_driver,
        };

        Ok(LashCore {
            env,
            policy,
            store_factory: self.store_factory,
            plugin_factories: Arc::new(plugin_factories),
            provider: self.provider,
            live_replay_store,
            protocol_factory,
            runtime_host_installer: self.runtime_host_installer,
            work_driver: Arc::new(InlineWorkDriverSlot::new(work_driver)),
        })
    }

    /// Decide how a built [`LashCore`] sources its process work driver.
    ///
    /// - no registry => nothing to run ([`ProcessWorkDriverSetup::None`]);
    /// - external driver wired => use it ([`ProcessWorkDriverSetup::External`]);
    /// - inline registry wired => lazily construct the default inline driver on first open. Its
    ///   [`DurableProcessWorkerConfig`] is built eagerly when a store factory is
    ///   present; without one the inline worker cannot rebuild session runtimes.
    fn resolve_process_work_driver(
        process_work_source: &ProcessWorkSource,
        worker_plugin_host: &PluginHost,
        core: &RuntimeHostConfig,
        store_factory: Option<&Arc<dyn SessionStoreFactory>>,
        policy: &SessionPolicy,
        residency: lash_core::Residency,
        trigger_store: Option<&Arc<dyn lash_core::TriggerStore>>,
    ) -> Result<ProcessWorkDriverSetup> {
        let process_registry = match process_work_source {
            ProcessWorkSource::None => return Ok(ProcessWorkDriverSetup::None),
            ProcessWorkSource::External(driver) => {
                return Ok(ProcessWorkDriverSetup::External {
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
        // The worker rebuilds with the same plugin host the live runtime uses,
        // including the protocol plugin that supplies the protocol session
        // capability.
        let phase_probe_slot = lash_core::runtime::RuntimeTurnPhaseProbeSlot::default();
        let config = Box::new(
            DurableProcessWorkerConfig::new(
                Arc::new(worker_plugin_host.clone()),
                core.clone(),
                Arc::clone(store_factory),
                process_registry,
            )
            .with_session_policy(policy.clone())
            .with_trigger_store(trigger_store.cloned().unwrap_or_else(|| {
                Arc::new(lash_core::InMemoryTriggerStore::with_clock(Arc::clone(
                    &core.clock,
                )))
            }))
            .with_residency(residency)
            .with_turn_phase_probe_slot(phase_probe_slot),
        );
        Ok(ProcessWorkDriverSetup::LazyDefault { config })
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_queued_work_driver(
        queued_work_source: &QueuedWorkSource,
        env: RuntimeEnvironment,
        policy: SessionPolicy,
        protocol_factory: Option<Arc<dyn PluginFactory>>,
        plugin_factories: Arc<Vec<Arc<dyn PluginFactory>>>,
        store_factory: Option<&Arc<dyn SessionStoreFactory>>,
        live_replay_store: Arc<dyn LiveReplayStore>,
        runtime_host_installer: Option<RuntimeHostInstaller>,
    ) -> QueuedWorkDriverSetup {
        match queued_work_source {
            QueuedWorkSource::None => QueuedWorkDriverSetup::None,
            QueuedWorkSource::External(driver) => QueuedWorkDriverSetup::External {
                driver: driver.clone(),
            },
            QueuedWorkSource::LazyDefault => match store_factory {
                Some(store_factory) => QueuedWorkDriverSetup::LazyDefault {
                    config: Arc::new(InlineQueuedWorkRunConfig::new(
                        env,
                        policy,
                        protocol_factory,
                        plugin_factories,
                        Arc::clone(store_factory),
                        live_replay_store,
                        runtime_host_installer,
                    )),
                },
                None => QueuedWorkDriverSetup::None,
            },
        }
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

    pub fn trigger_store(mut self, store: Arc<dyn lash_core::TriggerStore>) -> Self {
        self.trigger_store = Some(store);
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

    /// Configure an externally owned queued-work driver.
    pub fn queued_work_driver(mut self, driver: QueuedWorkDriver) -> Self {
        self.queued_work_source = QueuedWorkSource::External(driver);
        self
    }

    pub fn disable_queued_work_driver(mut self) -> Self {
        self.queued_work_source = QueuedWorkSource::None;
        self
    }
}

pub(crate) fn build_plugin_host(
    protocol_factory: Option<&Arc<dyn PluginFactory>>,
    common_factories: &[Arc<dyn PluginFactory>],
    extra_factories: Vec<Arc<dyn PluginFactory>>,
) -> Result<PluginHost> {
    let mut factories = Vec::with_capacity(
        usize::from(protocol_factory.is_some()) + common_factories.len() + extra_factories.len(),
    );
    if let Some(protocol_factory) = protocol_factory {
        factories.push(Arc::clone(protocol_factory));
    }
    factories.extend(common_factories.iter().cloned());
    factories.extend(extra_factories);
    Ok(PluginHost::new(factories))
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
