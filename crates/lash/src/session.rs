use crate::support::*;

pub struct SessionBuilder {
    pub(crate) core: LashCore,
    pub(crate) session_id: String,
    pub(crate) spec: SessionSpec,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) store: Option<Arc<dyn RuntimePersistence>>,
    pub(crate) active_plugins: Vec<ActivePluginBinding>,
    pub(crate) plugin_factories: Vec<Arc<dyn PluginFactory>>,
}

impl SessionBuilder {
    pub fn standard(mut self) -> Self {
        self.spec = self.spec.mode(ModeId::standard().execution_mode());
        self
    }

    pub fn rlm(mut self) -> Self {
        self.spec = self.spec.mode(ModeId::rlm().execution_mode());
        self
    }

    pub fn mode(mut self, mode: ModeId) -> Self {
        self.spec = self.spec.mode(mode.execution_mode());
        self
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.spec = self.spec.provider(provider);
        self
    }

    pub fn prompt_template(mut self, template: PromptTemplate) -> Self {
        let mut prompt = self.spec.prompt.take().unwrap_or_default();
        prompt.template = Some(template);
        self.spec = self.spec.prompt_layer(prompt);
        self
    }

    pub fn prompt_contribution(mut self, contribution: PromptContribution) -> Self {
        let mut prompt = self.spec.prompt.take().unwrap_or_default();
        prompt.add_contribution(contribution);
        self.spec = self.spec.prompt_layer(prompt);
        self
    }

    pub fn replace_prompt_slot(
        mut self,
        slot: PromptSlot,
        contributions: impl IntoIterator<Item = PromptContribution>,
    ) -> Self {
        let mut prompt = self.spec.prompt.take().unwrap_or_default();
        prompt.replace_slot(slot, contributions);
        self.spec = self.spec.prompt_layer(prompt);
        self
    }

    pub fn clear_prompt_slot(mut self, slot: PromptSlot) -> Self {
        let mut prompt = self.spec.prompt.take().unwrap_or_default();
        prompt.clear_slot(slot);
        self.spec = self.spec.prompt_layer(prompt);
        self
    }

    pub fn prompt_layer(mut self, layer: PromptLayer) -> Self {
        self.spec = self.spec.prompt_layer(layer);
        self
    }

    pub fn session_spec(mut self, spec: SessionSpec) -> Self {
        self.spec = spec;
        self
    }

    pub fn parent(mut self, parent_session_id: impl Into<String>) -> Self {
        self.parent_session_id = Some(parent_session_id.into());
        self
    }

    /// Use a specific persistence store for this root session.
    ///
    /// This is the right API for a host-owned, pre-opened session database.
    /// Managed child sessions never reuse this store; configure
    /// `LashCoreBuilder::child_store_factory` when child sessions should also
    /// persist.
    pub fn store(mut self, store: Arc<dyn RuntimePersistence>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn plugin<P: PluginBinding>(mut self, config: P::SessionConfig) -> Self {
        self.active_plugins.push(ActivePluginBinding {
            id: P::ID,
            requires_turn_input: P::requires_turn_input(&config),
        });
        self.plugin_factories.push(P::factory(&config));
        self
    }

    pub async fn open(self) -> Result<LashSession> {
        let (policy, mode) = self.session_policy()?;
        let store = self.create_store(&policy)?;
        let state = self
            .load_or_default_state(&policy, store.as_deref())
            .await?;
        self.open_resolved(policy, mode, state, store).await
    }

    pub async fn open_with_state(self, mut state: RuntimeSessionState) -> Result<LashSession> {
        let (policy, mode) = self.session_policy()?;
        let store = self.create_store(&policy)?;
        if state.session_id != self.session_id {
            return Err(EmbedError::StoreSessionMismatch {
                loaded: state.session_id,
                requested: self.session_id,
            });
        }
        state.policy = policy.clone();
        Self::normalize_tool_state(&mut state);
        self.open_resolved(policy, mode, state, store).await
    }

    fn session_policy(&self) -> Result<(SessionPolicy, ModeId)> {
        let execution_mode = self
            .spec
            .execution_mode
            .clone()
            .unwrap_or_else(|| self.core.default_mode.execution_mode());
        let mode = ModeId::new(execution_mode.plugin_id());
        let preset = self
            .core
            .modes
            .get(&mode)
            .ok_or_else(|| EmbedError::ModeNotInstalled { mode: mode.clone() })?;
        let mut base = self.core.policy.clone();
        base.execution_mode = execution_mode;
        base.standard_context_approach = preset.standard_context_approach.clone();
        let mut policy = self.spec.resolve_against(&base);
        policy.session_id = Some(self.session_id.clone());
        Ok((policy, mode))
    }

    async fn load_or_default_state(
        &self,
        policy: &SessionPolicy,
        store: Option<&dyn RuntimePersistence>,
    ) -> Result<RuntimeSessionState> {
        let state = match store {
            Some(store) => {
                let loaded = self.load_persisted_state_for_residency(store).await?;
                let mut state = loaded.unwrap_or_else(|| RuntimeSessionState {
                    session_id: self.session_id.clone(),
                    policy: policy.clone(),
                    ..RuntimeSessionState::default()
                });
                if state.session_id != self.session_id {
                    return Err(EmbedError::StoreSessionMismatch {
                        loaded: state.session_id,
                        requested: self.session_id.clone(),
                    });
                }
                state.policy = policy.clone();
                Self::normalize_tool_state(&mut state);
                state
            }
            None => RuntimeSessionState {
                session_id: self.session_id.clone(),
                policy: policy.clone(),
                ..RuntimeSessionState::default()
            },
        };
        Ok(state)
    }

    async fn load_persisted_state_for_residency(
        &self,
        store: &dyn RuntimePersistence,
    ) -> Result<Option<RuntimeSessionState>> {
        match self.core.env.residency {
            Residency::KeepAll => {
                let loaded = lash_core::load_persisted_session_state(store)
                    .await
                    .map_err(|err| {
                        SessionError::Protocol(format!("failed to load store: {err}"))
                    })?;
                Ok(loaded)
            }
            Residency::ActivePathOnly => {
                let active = lash_core::load_persisted_session_state_active_path(store, None)
                    .await
                    .map_err(|err| {
                        SessionError::Protocol(format!("failed to load active-path store: {err}"))
                    })?;
                if active
                    .as_ref()
                    .is_some_and(|state| state.session_graph.nodes.is_empty())
                {
                    let mut full = lash_core::load_persisted_session_state(store)
                        .await
                        .map_err(|err| {
                            SessionError::Protocol(format!(
                                "failed to heal active-path store from full graph: {err}"
                            ))
                        })?;
                    if let Some(state) = full.as_mut() {
                        state.graph_replace_required = true;
                    }
                    return Ok(full);
                }
                Ok(active)
            }
        }
    }

    fn normalize_tool_state(state: &mut RuntimeSessionState) {
        if let Some(snapshot) = state.tool_state_snapshot.as_mut() {
            let normalized = snapshot.clone().with_generation(1);
            state.tool_state_generation = Some(normalized.generation());
            *snapshot = normalized;
        }
    }

    async fn open_resolved(
        self,
        policy: SessionPolicy,
        mode: ModeId,
        state: RuntimeSessionState,
        store: Option<Arc<dyn RuntimePersistence>>,
    ) -> Result<LashSession> {
        let mut env = self.core.env.clone();
        if !self.plugin_factories.is_empty() {
            let mut factories = self.core.plugin_factories.as_ref().clone();
            factories.extend(self.plugin_factories);
            let mut plugin_host = PluginHost::new(factories);
            if env.process_registry.is_some() {
                let abilities = plugin_host
                    .lashlang_abilities()
                    .with_processes()
                    .with_process_lifecycle();
                plugin_host = plugin_host.with_lashlang_abilities(abilities);
            }
            env.plugin_host = Some(Arc::new(plugin_host));
        }
        let runtime = LashRuntime::from_environment(&env, policy, state, store).await?;
        let turn_input_injection_bridge = runtime.turn_input_injection_bridge()?;
        let handle = RuntimeHandle::new(runtime);
        Ok(LashSession {
            runtime: handle,
            turn_input_injection_bridge,
            mode,
            parent_session_id: self.parent_session_id,
            active_plugins: self.active_plugins,
        })
    }

    fn create_store(&self, policy: &SessionPolicy) -> Result<Option<Arc<dyn RuntimePersistence>>> {
        if let Some(store) = self.store.as_ref() {
            return Ok(Some(Arc::clone(store)));
        }
        let Some(factory) = self.core.store_factory.as_ref() else {
            return Ok(None);
        };
        let request = SessionStoreCreateRequest {
            session_id: self.session_id.clone(),
            relation: self
                .parent_session_id
                .as_ref()
                .map(|parent_session_id| lash_core::SessionRelation::Child {
                    parent_session_id: parent_session_id.clone(),
                    originating_tool_call_id: None,
                })
                .unwrap_or_default(),
            policy: policy.clone(),
        };
        factory
            .create_store(&request)
            .map(Some)
            .map_err(|message| EmbedError::StoreFactory {
                session_id: self.session_id.clone(),
                message,
            })
    }
}

#[derive(Clone)]
pub struct LashSession {
    pub(crate) runtime: RuntimeHandle,
    pub(crate) turn_input_injection_bridge: lash_core::TurnInputInjectionBridge,
    pub(crate) mode: ModeId,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) active_plugins: Vec<ActivePluginBinding>,
}

#[derive(Clone, Debug, Default)]
pub struct SessionConfigPatch {
    pub provider: Option<ProviderHandle>,
    pub model: Option<lash_core::ModelSpec>,
    pub prompt: Option<PromptLayer>,
}

impl LashSession {
    pub async fn close(self) -> Result<()> {
        let runtime = self.runtime.writer();
        let runtime = runtime.lock().await;
        runtime.unregister_plugin_session()?;
        Ok(())
    }

    pub fn session_id(&self) -> String {
        self.runtime.observe().session_id().to_string()
    }

    pub fn policy_snapshot(&self) -> SessionPolicy {
        self.runtime.observe().policy.clone()
    }

    pub fn observe(&self) -> ObservableSession {
        ObservableSession {
            runtime: self.runtime.clone(),
        }
    }

    pub fn mode(&self) -> &ModeId {
        &self.mode
    }

    pub fn parent_session_id(&self) -> Option<&str> {
        self.parent_session_id.as_deref()
    }

    pub async fn run(&self, input: TurnInput) -> Result<TurnOutput> {
        self.turn(input).run().await
    }

    pub fn resume_turn(&self, turn_id: impl Into<String>) -> ResumeTurnBuilder {
        ResumeTurnBuilder {
            runtime: self.runtime.clone(),
            turn_id: turn_id.into(),
            cancel: CancellationToken::new(),
        }
    }

    pub fn turn(&self, input: TurnInput) -> TurnBuilder {
        TurnBuilder {
            runtime: self.runtime.clone(),
            active_plugins: self.active_plugins.clone(),
            input,
            cancel: CancellationToken::new(),
            mode_turn_options: None,
            provider: None,
            model: None,
        }
    }

    pub fn control(&self) -> SessionControl {
        SessionControl {
            runtime: self.runtime.clone(),
            turn_input_injection_bridge: self.turn_input_injection_bridge.clone(),
        }
    }

    pub async fn configure(&self, patch: SessionConfigPatch) -> Result<()> {
        self.control().config().update(patch).await
    }

    pub fn tools(&self) -> ToolsControl {
        ToolsControl::new(self.control())
    }

    pub fn process_control(&self) -> ProcessControl {
        ProcessControl::new(self.control())
    }

    pub fn handoffs(&self) -> Handoffs {
        Handoffs::new(self.control())
    }

    pub fn plugin_actions(&self) -> PluginActions {
        PluginActions {
            control: self.control(),
        }
    }

    pub fn queue(&self, input: TurnInput) -> QueueInputBuilder<'_> {
        QueueInputBuilder {
            session: self,
            input,
            id: None,
        }
    }

    pub fn read_view(&self) -> SessionReadView {
        self.runtime.observe().read_view.clone()
    }

    pub fn usage_report(&self) -> SessionUsageReport {
        self.runtime.observe().usage_report.clone()
    }

    pub async fn set_turn_phase_probe(
        &self,
        probe: Arc<dyn lash_core::runtime::RuntimeTurnPhaseProbe>,
    ) {
        let writer = self.runtime.writer();
        let mut runtime = writer.lock().await;
        runtime.set_turn_phase_probe(probe);
        self.runtime.publish_from(&runtime);
    }
}

#[derive(Clone)]
pub struct ObservableSession {
    pub(crate) runtime: RuntimeHandle,
}

impl ObservableSession {
    fn snapshot(&self) -> Arc<RuntimeObservation> {
        self.runtime.observe()
    }

    pub fn session_id(&self) -> String {
        self.snapshot().session_id().to_string()
    }

    pub fn policy_snapshot(&self) -> SessionPolicy {
        self.snapshot().policy.clone()
    }

    pub fn read_view(&self) -> SessionReadView {
        self.snapshot().read_view.clone()
    }

    pub fn usage_report(&self) -> SessionUsageReport {
        self.snapshot().usage_report.clone()
    }

    pub fn tool_state(&self) -> Option<ToolState> {
        self.snapshot().tool_state.clone()
    }

    pub fn active_tool_definitions(&self) -> Vec<ToolManifest> {
        self.snapshot()
            .tool_state
            .as_ref()
            .map(ToolState::tool_manifests)
            .unwrap_or_default()
    }

    pub async fn list_process_handles(&self) -> Vec<ProcessHandleGrantEntry> {
        self.snapshot().list_process_handles().await
    }
}
pub struct QueueInputBuilder<'a> {
    session: &'a LashSession,
    input: TurnInput,
    id: Option<String>,
}

impl<'a> QueueInputBuilder<'a> {
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub async fn send(self) -> Result<()> {
        let message = queued_input_message(self.input)?;
        self.session
            .turn_input_injection_bridge
            .enqueue(vec![lash_core::InjectedTurnInput {
                id: self.id,
                message,
            }])
            .map_err(|message| EmbedError::Session(SessionError::Protocol(message)))
    }
}

impl<'a> std::future::IntoFuture for QueueInputBuilder<'a> {
    type Output = Result<()>;
    type IntoFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

fn queued_input_message(input: TurnInput) -> Result<PluginMessage> {
    let mut text = Vec::new();
    let mut images = Vec::new();
    for item in input.items {
        match item {
            InputItem::Text { text: item_text } => text.push(item_text),
            InputItem::ImageRef { id } => {
                let Some(bytes) = input.image_blobs.get(&id).cloned() else {
                    return Err(EmbedError::MissingQueuedImageBlob { id });
                };
                images.push(bytes);
            }
        }
    }
    let content = text.join("\n");
    Ok(PluginMessage {
        role: MessageRole::User,
        content,
        parts: Vec::new(),
        images,
    })
}
