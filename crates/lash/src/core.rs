use crate::support::*;

#[derive(Clone)]
pub struct LashCore {
    pub(crate) env: RuntimeEnvironment,
    pub(crate) policy: SessionPolicy,
    pub(crate) modes: Arc<BTreeMap<ModeId, ModePreset>>,
    pub(crate) default_mode: ModeId,
    pub(crate) store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub(crate) plugin_factories: Arc<Vec<Arc<dyn PluginFactory>>>,
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

    pub fn standard() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::standard())
            .default_mode(ModeId::standard())
            .plugins(default_runtime_stack())
    }

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
            let metadata = lash_core::EffectInvocationMetadata {
                session_id: session_id.clone(),
                origin: lash_core::EffectOrigin::Turn,
                turn_id: None,
                turn_index: None,
                protocol_iteration: None,
                effect_id: format!("process:delete-session:{session_id}"),
                effect_kind: lash_core::RuntimeEffectKind::Process,
                idempotency_key: format!("{session_id}:delete-session"),
                turn_checkpoint_hash: None,
            };
            let outcome = self
                .env
                .core
                .effect_controller
                .execute_effect(
                    lash_core::RuntimeEffectEnvelope::new(
                        metadata,
                        lash_core::RuntimeEffectCommand::Process {
                            command: lash_core::ProcessCommand::DeleteSession {
                                session_id: session_id.clone(),
                            },
                        },
                    ),
                    lash_core::RuntimeEffectLocalExecutor::process_control(Arc::clone(
                        process_registry,
                    )),
                )
                .await
                .map_err(|err| EmbedError::SessionDeleteProcess {
                    session_id: session_id.clone(),
                    message: err.to_string(),
                })?;
            match outcome {
                lash_core::RuntimeEffectOutcome::Process {
                    result: lash_core::ProcessEffectOutcome::DeleteSession { report },
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
    pub(crate) store_factory: Option<Arc<dyn SessionStoreFactory>>,
    child_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    core: RuntimeCoreConfig,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    plugin_stack: PluginStack,
    plugin_host: Option<PluginHost>,
    residency: Option<Residency>,
    process_registry: Option<Arc<dyn ProcessRegistry>>,
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
        self.session_spec = self.session_spec.provider(provider);
        self
    }

    pub fn prompt_template(mut self, template: PromptTemplate) -> Self {
        self.core = self.core.with_prompt_template(template);
        self
    }

    pub fn prompt_contribution(mut self, contribution: PromptContribution) -> Self {
        self.core = self.core.with_prompt_contribution(contribution);
        self
    }

    pub fn replace_prompt_slot(
        mut self,
        slot: PromptSlot,
        contributions: impl IntoIterator<Item = PromptContribution>,
    ) -> Self {
        self.core = self.core.with_replaced_prompt_slot(slot, contributions);
        self
    }

    pub fn clear_prompt_slot(mut self, slot: PromptSlot) -> Self {
        self.core = self.core.with_cleared_prompt_slot(slot);
        self
    }

    pub fn prompt_layer(mut self, layer: PromptLayer) -> Self {
        self.core = self.core.with_prompt_layer(layer);
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
        self.core = self.core.with_attachment_store(attachment_store);
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
        self.core = self.core.with_trace_sink(trace_sink);
        self
    }

    pub fn trace_jsonl_path(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.core = self.core.with_trace_sink(path.map(|path| {
            Arc::new(lash_trace::JsonlTraceSink::new(path)) as Arc<dyn lash_trace::TraceSink>
        }));
        self
    }

    pub fn trace_level(mut self, trace_level: lash_trace::TraceLevel) -> Self {
        self.core = self.core.with_trace_level(trace_level);
        self
    }

    pub fn trace_context(mut self, trace_context: lash_trace::TraceContext) -> Self {
        self.core = self.core.with_trace_context(trace_context);
        self
    }

    pub fn build(self) -> Result<LashCore> {
        if self.modes.is_empty() {
            return Err(EmbedError::NoModesInstalled);
        }
        let default_mode = self
            .default_mode
            .clone()
            .ok_or(EmbedError::NoModesInstalled)?;
        if !self.modes.contains_key(&default_mode) {
            return Err(EmbedError::DefaultModeNotInstalled { mode: default_mode });
        }
        let provider = self.session_spec.provider.clone().unwrap_or_default();
        let model = self
            .session_spec
            .model
            .clone()
            .ok_or(EmbedError::MissingModelSpec)?;

        let base_policy = SessionPolicy {
            provider,
            model,
            max_turns: self.session_spec.max_turns.flatten(),
            ..SessionPolicy::default()
        };
        let policy = self.session_spec.resolve_against(&base_policy);

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

        let mut env_builder = RuntimeEnvironment::builder()
            .with_plugin_host(Arc::new(default_plugin_host))
            .with_runtime_core_config(self.core);
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
        })
    }

    pub fn advanced(self) -> AdvancedLashCoreBuilder {
        AdvancedLashCoreBuilder { builder: self }
    }

    pub fn process_registry(mut self, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        self.process_registry = Some(process_registry);
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
            .with_process_lifecycle();
        plugin_host = plugin_host.with_lashlang_abilities(abilities);
    }
    Ok(plugin_host)
}

pub struct AdvancedLashCoreBuilder {
    builder: LashCoreBuilder,
}

impl AdvancedLashCoreBuilder {
    pub fn runtime_core_config(mut self, core: lash_core::RuntimeCoreConfig) -> Self {
        self.builder.core = core;
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
        self.builder.core = self.builder.core.with_effect_controller(effect_controller);
        self
    }

    pub fn termination(mut self, termination: TerminationPolicy) -> Self {
        self.builder.core = self.builder.core.with_termination(termination);
        self
    }

    pub fn build(self) -> Result<LashCore> {
        self.builder.build()
    }
}
