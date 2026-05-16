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

impl LashCore {
    pub fn builder() -> LashCoreBuilder {
        LashCoreBuilder::default()
    }

    pub fn standard() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::standard())
            .default_mode(ModeId::standard())
            .plugins(PluginStack::runtime())
    }

    pub fn rlm() -> LashCoreBuilder {
        Self::builder()
            .install_mode(ModePreset::rlm())
            .default_mode(ModeId::rlm())
            .plugins(PluginStack::runtime())
    }

    pub fn session(&self, session_id: impl Into<String>) -> SessionBuilder {
        SessionBuilder {
            core: self.clone(),
            session_id: session_id.into(),
            spec: SessionSpec::inherit(),
            parent_session_id: None,
            store: None,
            active_plugins: Vec::new(),
            plugin_factories: Vec::new(),
        }
    }

    pub fn installed_modes(&self) -> impl Iterator<Item = &ModeId> {
        self.modes.keys()
    }
}

#[derive(Default)]
pub struct LashCoreBuilder {
    pub(crate) modes: BTreeMap<ModeId, ModePreset>,
    pub(crate) default_mode: Option<ModeId>,
    session_spec: SessionSpec,
    pub(crate) store_factory: Option<Arc<dyn SessionStoreFactory>>,
    child_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    attachment_store: Option<Arc<dyn AttachmentStore>>,
    tool_providers: Vec<Arc<dyn ToolProvider>>,
    plugin_stack: PluginStack,
    plugin_host: Option<PluginHost>,
    trace_sink: Option<Arc<dyn lash_trace::TraceSink>>,
    trace_level: Option<lash_trace::TraceLevel>,
    trace_context: Option<lash_trace::TraceContext>,
    residency: Option<Residency>,
    background_task_host: Option<Arc<dyn BackgroundTaskHost>>,
    termination: Option<TerminationPolicy>,
    prompt: PromptLayer,
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
        self.session_spec = self.session_spec.mode(mode.execution_mode());
        self
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.session_spec = self.session_spec.provider(provider);
        self
    }

    pub fn prompt_template(mut self, template: PromptTemplate) -> Self {
        self.prompt.template = Some(template);
        self
    }

    pub fn prompt_contribution(mut self, contribution: PromptContribution) -> Self {
        self.prompt.add_contribution(contribution);
        self
    }

    pub fn replace_prompt_slot(
        mut self,
        slot: PromptSlot,
        contributions: impl IntoIterator<Item = PromptContribution>,
    ) -> Self {
        self.prompt.replace_slot(slot, contributions);
        self
    }

    pub fn clear_prompt_slot(mut self, slot: PromptSlot) -> Self {
        self.prompt.clear_slot(slot);
        self
    }

    pub fn prompt_layer(mut self, layer: PromptLayer) -> Self {
        self.prompt = layer;
        self
    }

    pub fn model(mut self, model: impl Into<String>, variant: Option<String>) -> Self {
        self.session_spec = self.session_spec.model(model, variant);
        self
    }

    pub fn max_context_tokens(mut self, max_context_tokens: usize) -> Self {
        self.session_spec = self.session_spec.max_context_tokens(max_context_tokens);
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
            Arc::new(lash_core::JsonlTraceSink::new(path)) as Arc<dyn lash_trace::TraceSink>
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

    pub fn build(self) -> Result<LashCore> {
        if self.modes.is_empty() {
            return Err(EmbedError::NoModesInstalled);
        }
        let default_mode = self
            .session_spec
            .execution_mode
            .as_ref()
            .map(|mode| ModeId::new(mode.plugin_id()))
            .or_else(|| self.default_mode.clone())
            .ok_or(EmbedError::NoModesInstalled)?;
        if !self.modes.contains_key(&default_mode) {
            return Err(EmbedError::DefaultModeNotInstalled { mode: default_mode });
        }
        let max_context_tokens = self
            .session_spec
            .max_context_tokens
            .ok_or(EmbedError::MissingMaxContextTokens)?;
        let provider = self.session_spec.provider.clone().unwrap_or_default();
        let model = self
            .session_spec
            .model
            .clone()
            .unwrap_or_else(|| provider.default_model().to_string());
        let model_variant = self
            .session_spec
            .model_variant
            .clone()
            .unwrap_or_else(|| provider.default_model_variant(&model).map(str::to_string));
        let model = ModelSelection {
            model,
            variant: model_variant,
        };

        let default_preset = self
            .modes
            .get(&default_mode)
            .expect("default mode was validated");
        let base_policy = SessionPolicy {
            provider,
            model: model.model,
            model_variant: model.variant,
            max_context_tokens: Some(max_context_tokens),
            max_turns: self.session_spec.max_turns.flatten(),
            execution_mode: default_mode.execution_mode(),
            standard_context_approach: default_preset.standard_context_approach.clone(),
            ..SessionPolicy::default()
        };
        let policy = self.session_spec.resolve_against(&base_policy);

        let plugin_factories = if let Some(plugin_host) = self.plugin_host {
            plugin_host.factories().to_vec()
        } else {
            let mut factories = Vec::new();
            factories.extend(
                self.modes
                    .values()
                    .map(|preset| Arc::clone(&preset.factory)),
            );
            if !self.tool_providers.is_empty() {
                let spec = self
                    .tool_providers
                    .into_iter()
                    .fold(PluginSpec::new(), PluginSpec::with_tool_provider);
                factories.push(Arc::new(StaticPluginFactory::new("embed_tools", spec))
                    as Arc<dyn PluginFactory>);
            }
            factories.extend(self.plugin_stack.into_factories());
            PluginHost::new(factories).factories().to_vec()
        };
        let plugin_host = PluginHost::new(plugin_factories.clone());

        let background_task_host = self
            .background_task_host
            .unwrap_or_else(|| Arc::new(LocalBackgroundTaskHost::default()));
        let mut env_builder = RuntimeEnvironment::builder()
            .with_plugin_host(Arc::new(plugin_host.with_background_tasks()))
            .with_background_task_host(background_task_host)
            .with_prompt_layer(self.prompt);
        if let Some(attachment_store) = self.attachment_store {
            env_builder = env_builder.with_attachment_store(attachment_store);
        }
        if let Some(trace_sink) = self.trace_sink {
            env_builder = env_builder.with_trace_sink(Some(trace_sink));
        }
        if let Some(trace_level) = self.trace_level {
            env_builder = env_builder.with_trace_level(trace_level);
        }
        if let Some(trace_context) = self.trace_context {
            env_builder = env_builder.with_trace_context(trace_context);
        }
        if let Some(residency) = self.residency {
            env_builder = env_builder.with_residency(residency);
        }
        if let Some(termination) = self.termination {
            env_builder = env_builder.with_termination(termination);
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

    pub fn background_task_host(
        mut self,
        background_task_host: Arc<dyn BackgroundTaskHost>,
    ) -> Self {
        self.background_task_host = Some(background_task_host);
        self
    }
}

pub struct AdvancedLashCoreBuilder {
    builder: LashCoreBuilder,
}

impl AdvancedLashCoreBuilder {
    pub fn runtime_core_config(mut self, core: lash_core::RuntimeCoreConfig) -> Self {
        self.builder.attachment_store = Some(core.attachment_store);
        self.builder.trace_sink = core.trace_sink;
        self.builder.trace_level = Some(core.trace_level);
        self.builder.trace_context = Some(core.trace_context);
        self.builder.termination = Some(core.termination);
        self.builder.prompt = core.prompt;
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

    pub fn background_task_host(
        mut self,
        background_task_host: Arc<dyn BackgroundTaskHost>,
    ) -> Self {
        self.builder.background_task_host = Some(background_task_host);
        self
    }

    pub fn termination(mut self, termination: TerminationPolicy) -> Self {
        self.builder.termination = Some(termination);
        self
    }

    pub fn build(self) -> Result<LashCore> {
        self.builder.build()
    }
}
