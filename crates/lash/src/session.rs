use crate::support::*;
use lash_core::runtime::{DeliveryPolicy, QueuedWorkBatch, SlotPolicy};

pub struct SessionBuilder {
    pub(crate) core: LashCore,
    pub(crate) session_id: String,
    pub(crate) spec: SessionSpec,
    pub(crate) mode: Option<ModeId>,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) store: Option<Arc<dyn RuntimePersistence>>,
    pub(crate) provider: Option<ProviderHandle>,
    pub(crate) active_plugins: Vec<ActivePluginBinding>,
    pub(crate) plugin_factories: Vec<Arc<dyn PluginFactory>>,
    #[cfg(feature = "rlm")]
    pub(crate) rlm_final_answer_format: Option<lash_rlm_types::RlmFinalAnswerFormat>,
}

impl SessionBuilder {
    pub fn standard(mut self) -> Self {
        self.mode = Some(ModeId::standard());
        self
    }

    #[cfg(feature = "rlm")]
    pub fn rlm(mut self) -> Self {
        self.mode = Some(ModeId::rlm());
        self
    }

    pub fn mode(mut self, mode: ModeId) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.spec = self.spec.provider_id(provider.kind());
        self.provider = Some(provider);
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
        let store = self.create_store(&policy).await?;
        #[cfg_attr(not(feature = "rlm"), allow(unused_mut))]
        let mut state = self
            .load_or_default_state(&policy, store.as_deref())
            .await?;
        #[cfg(feature = "rlm")]
        self.apply_rlm_session_options(&mode, &mut state)?;
        self.open_resolved(policy, mode, state, store).await
    }

    /// Open this session with a fresh resident graph, ignoring any persisted
    /// session graph/checkpoint state that may already exist for the same
    /// session id.
    ///
    /// The next successful commit writes a full replacement graph, so normal
    /// embedders can use this to start over without manually calling
    /// `load_persisted_session_state` or constructing a `RuntimeSessionState`.
    /// Use [`Self::open`] for resume and [`Self::open_with_state`] only when
    /// restoring explicit host-owned state.
    pub async fn open_fresh(self) -> Result<LashSession> {
        let (policy, mode) = self.session_policy()?;
        let store = self.create_store(&policy).await?;
        #[cfg_attr(not(feature = "rlm"), allow(unused_mut))]
        let mut state = RuntimeSessionState {
            session_id: self.session_id.clone(),
            policy: policy.clone(),
            graph_replace_required: true,
            ..RuntimeSessionState::default()
        };
        #[cfg(feature = "rlm")]
        self.apply_rlm_session_options(&mode, &mut state)?;
        self.open_resolved(policy, mode, state, store).await
    }

    /// Open with an explicitly supplied runtime state.
    ///
    /// This is for advanced hosts that already own a complete state snapshot.
    /// Normal embedders should use [`Self::open`] to resume according to Lash's
    /// residency policy or [`Self::open_fresh`] to start over and replace prior
    /// persisted state on the next commit.
    pub async fn open_with_state(self, mut state: RuntimeSessionState) -> Result<LashSession> {
        let (policy, mode) = self.session_policy()?;
        let store = self.create_store(&policy).await?;
        if state.session_id != self.session_id {
            return Err(EmbedError::StoreSessionMismatch {
                loaded: state.session_id,
                requested: self.session_id,
            });
        }
        let recorded_provider_id = state.policy.recorded_provider_id().to_string();
        state.policy = policy.clone();
        state.policy.provider_id = recorded_provider_id;
        #[cfg(feature = "rlm")]
        self.apply_rlm_session_options(&mode, &mut state)?;
        self.open_resolved(policy, mode, state, store).await
    }

    fn session_policy(&self) -> Result<(SessionPolicy, ModeId)> {
        let mode = self
            .mode
            .clone()
            .unwrap_or_else(|| self.core.default_mode.clone());
        if !self.core.modes.contains_key(&mode) {
            return Err(EmbedError::ModeNotInstalled { mode });
        }
        let mut policy = self.spec.resolve_against(&self.core.policy);
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
                let recorded_provider_id = state.policy.recorded_provider_id().to_string();
                state.policy = policy.clone();
                state.policy.provider_id = recorded_provider_id;
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
                let loaded = lash_core::store::load_persisted_session_state(store)
                    .await
                    .map_err(|err| {
                        SessionError::Protocol(format!("failed to load store: {err}"))
                    })?;
                Ok(loaded)
            }
            Residency::ActivePathOnly => {
                let active =
                    lash_core::store::load_persisted_session_state_active_path(store, None)
                        .await
                        .map_err(|err| {
                            SessionError::Protocol(format!(
                                "failed to load active-path store: {err}"
                            ))
                        })?;
                if active
                    .as_ref()
                    .is_some_and(|state| state.session_graph.nodes.is_empty())
                {
                    let mut full = lash_core::store::load_persisted_session_state(store)
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

    #[cfg(feature = "rlm")]
    fn apply_rlm_session_options(
        &self,
        mode: &ModeId,
        state: &mut RuntimeSessionState,
    ) -> Result<()> {
        let Some(final_answer_format) = self.rlm_session_final_answer_format(mode) else {
            return Ok(());
        };
        let mut extras = if state.protocol_turn_options.is_empty() {
            lash_rlm_types::RlmCreateExtras::default()
        } else {
            state.protocol_turn_options.decode()?
        };
        extras.final_answer_format = Some(final_answer_format);
        let options = ProtocolTurnOptions::typed(extras)?;
        state.protocol_turn_options = options.clone();
        for frame in &mut state.agent_frames {
            frame.protocol_turn_options = options.clone();
        }
        Ok(())
    }

    #[cfg(feature = "rlm")]
    fn rlm_session_final_answer_format(
        &self,
        mode: &ModeId,
    ) -> Option<lash_rlm_types::RlmFinalAnswerFormat> {
        if mode != &ModeId::rlm() {
            return None;
        }
        self.rlm_final_answer_format.clone().or_else(|| {
            if self.parent_session_id.is_none() {
                Some(lash_rlm_types::RlmFinalAnswerFormat::Markdown)
            } else {
                Some(lash_rlm_types::RlmFinalAnswerFormat::RawSubmitValue)
            }
        })
    }

    async fn open_resolved(
        self,
        policy: SessionPolicy,
        mode: ModeId,
        state: RuntimeSessionState,
        store: Option<Arc<dyn RuntimePersistence>>,
    ) -> Result<LashSession> {
        let mut env = self.core.env.clone();
        if let Some(provider) = self.provider.clone().or_else(|| self.core.provider.clone()) {
            env.core.providers.provider_resolver =
                Arc::new(lash_core::SingleProviderResolver::new(provider));
        }
        let plugin_host = build_plugin_host_for_mode(
            &self.core.modes,
            &mode,
            self.core.plugin_factories.as_ref(),
            self.plugin_factories,
            env.process_registry.is_some(),
            #[cfg(feature = "rlm")]
            self.core.lashlang_artifact_store.as_ref(),
            #[cfg(feature = "rlm")]
            self.core.lashlang_execution_sink.clone(),
            #[cfg(feature = "rlm")]
            env.core.tracing.trace_context.clone(),
        )?;
        env.plugin_host = Some(Arc::new(plugin_host));
        let effect_host = Arc::clone(&env.core.control.effect_host);
        // Lazily spawn the default process work runner (Decision 3: deferred to
        // the first open so a tokio runtime is guaranteed; idempotent via the
        // shared once-guard) and thread its poke onto this session's host so the
        // process admin seam can wake the runner after a successful start.
        env.process_work_poke = self.core.process_work_runner.poke().await;
        let runtime = LashRuntime::from_environment(&env, policy, state, store).await?;
        let handle = RuntimeHandle::with_live_replay_store(
            runtime,
            Arc::clone(&self.core.live_replay_store),
        );
        Ok(LashSession {
            runtime: handle,
            effect_host,
            mode,
            parent_session_id: self.parent_session_id,
            active_plugins: self.active_plugins,
            process_phase_probe_slot: self.core.process_work_runner.phase_probe_slot(),
            turn_cancels: crate::turn::TurnCancelRegistry::default(),
        })
    }

    async fn create_store(
        &self,
        policy: &SessionPolicy,
    ) -> Result<Option<Arc<dyn RuntimePersistence>>> {
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
                    caused_by: None,
                })
                .unwrap_or_default(),
            policy: policy.clone(),
        };
        factory
            .create_store(&request)
            .await
            .map(Some)
            .map_err(|message| EmbedError::StoreFactory {
                session_id: self.session_id.clone(),
                message,
            })
    }
}

impl PromptLayerSink for SessionBuilder {
    fn prompt_layer_mut(&mut self) -> &mut PromptLayer {
        self.spec.prompt.get_or_insert_with(PromptLayer::new)
    }
}

#[derive(Clone)]
pub struct LashSession {
    pub(crate) runtime: RuntimeHandle,
    pub(crate) effect_host: Arc<dyn EffectHost>,
    pub(crate) mode: ModeId,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) active_plugins: Vec<ActivePluginBinding>,
    pub(crate) process_phase_probe_slot: Option<lash_core::runtime::RuntimeTurnPhaseProbeSlot>,
    pub(crate) turn_cancels: crate::turn::TurnCancelRegistry,
}

#[derive(Clone, Debug, Default)]
pub struct SessionConfigPatch {
    pub provider: Option<ProviderHandle>,
    pub model: Option<ModelSpec>,
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

    pub fn effect_host(&self) -> Arc<dyn EffectHost> {
        Arc::clone(&self.effect_host)
    }

    pub fn turn(&self, input: TurnInput) -> TurnBuilder {
        TurnBuilder {
            runtime: self.runtime.clone(),
            effect_host: Arc::clone(&self.effect_host),
            active_plugins: self.active_plugins.clone(),
            input,
            cancel: CancellationToken::new(),
            cancels: self.turn_cancels.clone(),
            protocol_turn_options: None,
            provider: None,
            model: None,
            turn_id: None,
        }
    }

    pub fn queued_turn(&self) -> QueuedTurnBuilder {
        QueuedTurnBuilder {
            runtime: self.runtime.clone(),
            effect_host: Arc::clone(&self.effect_host),
            cancel: CancellationToken::new(),
            cancels: self.turn_cancels.clone(),
            batch_ids: Vec::new(),
            drain_id: None,
        }
    }

    /// Cancel every turn currently executing through this opened session
    /// (including its clones) and report how many were signalled.
    ///
    /// This is the affordance behind a UI "stop" control: hold a clone of the
    /// session wherever the stop arrives and call this, instead of threading a
    /// [`CancellationToken`](crate::CancellationToken) into every turn call
    /// ([`TurnBuilder::cancel`](crate::TurnBuilder::cancel) remains the
    /// per-turn hook when you need one). A cancelled turn finishes with
    /// `TurnOutcome::Stopped(TurnStop::Cancelled)` and commits like any other
    /// turn; the session stays usable.
    ///
    /// Scope: turns started from this `LashSession` instance and its clones.
    /// A handle opened separately for the same session id has its own
    /// registry and is not reached.
    pub fn cancel_running_turns(&self) -> usize {
        self.turn_cancels.cancel_all()
    }

    pub fn admin(&self) -> SessionAdmin {
        SessionAdmin {
            runtime: self.runtime.clone(),
        }
    }

    pub async fn configure(&self, patch: SessionConfigPatch) -> Result<()> {
        self.admin().config().update(patch).await
    }

    pub fn tools(&self) -> ToolAdmin {
        ToolAdmin::new(self.admin())
    }

    pub fn commands(&self) -> SessionCommandAdmin {
        self.admin().commands()
    }

    pub fn triggers(&self) -> SessionTriggerAdmin {
        self.admin().triggers()
    }

    pub fn processes(&self) -> SessionProcessAdmin {
        SessionProcessAdmin::new(self.admin())
    }

    pub fn plugin_actions(&self) -> PluginActions {
        PluginActions {
            control: self.admin(),
        }
    }

    pub fn enqueue(&self, input: TurnInput) -> EnqueueTurnBuilder<'_> {
        EnqueueTurnBuilder {
            session: self,
            input,
            id: None,
            delivery_policy: DeliveryPolicy::AfterCurrentTurnCommit,
            slot_policy: SlotPolicy::Exclusive,
        }
    }

    pub async fn queued_work(&self) -> Result<Vec<QueuedWorkBatch>> {
        let observation = self.runtime.observe();
        let store = observation.queue_store.as_ref().ok_or_else(|| {
            EmbedError::Runtime(lash_core::RuntimeError::new(
                lash_core::RuntimeErrorCode::StoreCommitFailed,
                "queued work inspection requires a persistent runtime store",
            ))
        })?;
        store
            .list_pending_queued_work(observation.session_id())
            .await
            .map_err(|err| {
                EmbedError::Runtime(lash_core::RuntimeError::new(
                    lash_core::RuntimeErrorCode::StoreCommitFailed,
                    err.to_string(),
                ))
            })
    }

    pub async fn cancel_queued_work_batch(
        &self,
        batch_id: &str,
    ) -> Result<Option<QueuedWorkBatch>> {
        let session_id = self.session_id();
        self.runtime
            .cancel_queued_work_batch(&session_id, batch_id)
            .await
            .map_err(EmbedError::Runtime)
    }

    /// Resolve once `batch_id` is no longer pending in the queue store —
    /// drained by whoever runs queued work (a queued-work runner, a durable
    /// worker, or another handle's [`queued_turn`](Self::queued_turn)) or
    /// cancelled. This is the enqueue-and-observe side of the queue: the
    /// caller never claims the work itself.
    ///
    /// Completion is read from the persistent queue store, so it observes
    /// drains performed by other session handles and other processes alike.
    /// There is no built-in deadline — nothing resolves if nothing drains the
    /// queue, so bound it with `tokio::time::timeout` when the worker may be
    /// unavailable. A batch id the store has never seen resolves immediately.
    pub async fn await_queued_work_batch(&self, batch_id: &str) -> Result<()> {
        let observation = self.runtime.observe();
        let store = observation.queue_store.clone().ok_or_else(|| {
            EmbedError::Runtime(lash_core::RuntimeError::new(
                lash_core::RuntimeErrorCode::StoreCommitFailed,
                "queued work inspection requires a persistent runtime store",
            ))
        })?;
        let session_id = observation.session_id().to_string();
        drop(observation);
        let mut delay = std::time::Duration::from_millis(25);
        loop {
            let pending = store
                .list_pending_queued_work(&session_id)
                .await
                .map_err(|err| {
                    EmbedError::Runtime(lash_core::RuntimeError::new(
                        lash_core::RuntimeErrorCode::StoreCommitFailed,
                        err.to_string(),
                    ))
                })?;
            if !pending.iter().any(|batch| batch.batch_id == batch_id) {
                return Ok(());
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(std::time::Duration::from_millis(400));
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
        runtime.set_turn_phase_probe(Arc::clone(&probe));
        self.runtime.publish_from(&runtime);
        if let Some(slot) = &self.process_phase_probe_slot {
            let observation = self.runtime.observe();
            slot.set_for_session(observation.session_id(), Arc::clone(&probe));
            let current_frame = observation.persisted_state.current_agent_frame_id.as_str();
            if !current_frame.is_empty() {
                let scope = lash_core::SessionScope::for_agent_frame(
                    observation.session_id(),
                    current_frame,
                );
                slot.set_for_scope(&scope, probe);
            }
        }
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

    pub fn current_observation(&self) -> SessionObservation {
        self.runtime.current_session_observation()
    }

    pub fn resume_from_cursor(&self, cursor: &SessionCursor) -> Result<SessionResume> {
        self.runtime
            .resume_session_observation(cursor)
            .map_err(live_replay_error)
    }

    pub fn subscribe_from_cursor(
        &self,
        cursor: &SessionCursor,
    ) -> Result<SessionObservationSubscription> {
        self.runtime
            .subscribe_session_observation(cursor)
            .map_err(live_replay_error)
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

    pub fn active_tool_manifests(&self) -> Vec<ToolManifest> {
        self.snapshot()
            .tool_state
            .as_ref()
            .map(ToolState::tool_manifests)
            .unwrap_or_default()
    }

    pub async fn list_process_handles(&self) -> Vec<ProcessHandleSummary> {
        self.snapshot().list_process_handles().await
    }

    pub async fn list_all_process_handles(&self) -> Vec<ProcessHandleSummary> {
        self.snapshot().list_all_process_handles().await
    }

    pub fn process_scope(&self) -> SessionScope {
        self.snapshot().process_scope()
    }
}

fn live_replay_error(err: lash_core::LiveReplayStoreError) -> EmbedError {
    EmbedError::Runtime(lash_core::RuntimeError::new(
        RuntimeErrorCode::Other("live_replay".to_string()),
        err.to_string(),
    ))
}

pub struct EnqueueTurnBuilder<'a> {
    session: &'a LashSession,
    input: TurnInput,
    id: Option<String>,
    delivery_policy: DeliveryPolicy,
    slot_policy: SlotPolicy,
}

impl<'a> EnqueueTurnBuilder<'a> {
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn delivery_policy(mut self, policy: DeliveryPolicy) -> Self {
        self.delivery_policy = policy;
        self
    }

    pub fn slot_policy(mut self, policy: SlotPolicy) -> Self {
        self.slot_policy = policy;
        self
    }

    pub async fn send(self) -> Result<QueuedWorkBatch> {
        let source_key = self.id.map(|id| format!("host:{id}"));
        self.session
            .runtime
            .enqueue_turn_input(
                self.input,
                self.delivery_policy,
                self.slot_policy,
                source_key,
            )
            .await
            .map_err(EmbedError::Runtime)
    }
}

impl<'a> std::future::IntoFuture for EnqueueTurnBuilder<'a> {
    type Output = Result<QueuedWorkBatch>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Result<QueuedWorkBatch>> + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}
