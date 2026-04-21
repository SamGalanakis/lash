use super::host::{BackgroundRuntimeHost, EmbeddedRuntimeHost};
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
enum CurrentSnapshot {
    Owned(SessionSnapshot),
    Projection {
        meta: SessionSnapshot,
        messages: Arc<Vec<Message>>,
        tool_calls: Arc<Vec<ToolCallRecord>>,
    },
}

impl CurrentSnapshot {
    fn to_snapshot(&self) -> SessionSnapshot {
        match self {
            Self::Owned(snapshot) => snapshot.clone(),
            Self::Projection {
                meta,
                messages,
                tool_calls,
            } => {
                let mut snapshot = meta.clone();
                snapshot.replace_projection(messages.as_slice(), tool_calls.as_slice());
                snapshot
            }
        }
    }
}

pub(super) struct ManagedSessionTurn {
    pub(super) session_id: String,
    pub(super) cancel: CancellationToken,
    pub(super) task: tokio::task::JoinHandle<Result<AssembledTurn, crate::PluginError>>,
}

#[derive(Clone)]
pub(super) struct RuntimeSessionManager {
    runtime_scope_id: Arc<str>,
    current_session_id: String,
    current_snapshot: CurrentSnapshot,
    current_policy: SessionPolicy,
    current_host: RuntimeHost,
    current_plugins: Arc<crate::PluginSession>,
    current_tool_catalog: Arc<Vec<serde_json::Value>>,
    current_prompt_bridge: Option<HostPromptBridge>,
    current_store: Option<Arc<dyn crate::store::RuntimeStore>>,
    registry: Arc<Mutex<HashMap<String, Arc<Mutex<LashRuntime>>>>>,
    turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    /// Session-scoped token cost ledger shared with the parent
    /// `LashRuntime`. All managers created from the same runtime
    /// write to the same Arc. Drained at turn-commit time.
    token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    background_sync_needed: Arc<AtomicBool>,
    /// Maps child session_id → usage_source label.
    child_usage_sources: Arc<std::sync::Mutex<HashMap<String, String>>>,
    /// Tracks live child-turn usage already bubbled into the shared
    /// token ledger so `await_turn` can reconcile the final turn usage
    /// without double counting.
    child_turn_live_usage: Arc<std::sync::Mutex<HashMap<String, TokenUsage>>>,
    /// Optional relay for bubbling child-session token usage into the
    /// parent turn's live event stream.
    child_usage_event_relay: Option<ChildUsageEventRelay>,
    /// Out-of-turn managers persist drained usage back into the
    /// current session graph. Turn-time managers leave the shared
    /// ledger alone so the parent turn can commit it once.
    persist_usage_to_store: bool,
}

pub(super) struct ChannelEventSink {
    pub(super) tx: mpsc::Sender<SessionEvent>,
    live_usage: Option<LiveChildUsageForwarder>,
}

#[derive(Clone)]
struct LiveChildUsageForwarder {
    turn_id: String,
    session_id: String,
    source: String,
    model: String,
    token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    child_turn_live_usage: Arc<std::sync::Mutex<HashMap<String, TokenUsage>>>,
    relay: Option<ChildUsageEventRelay>,
}

#[derive(Clone, Default)]
pub(super) struct ChildUsageEventRelay {
    tx: Arc<StdMutex<Option<mpsc::Sender<RuntimeStreamEvent>>>>,
}

impl ChildUsageEventRelay {
    pub(super) fn new(tx: mpsc::Sender<RuntimeStreamEvent>) -> Self {
        Self {
            tx: Arc::new(StdMutex::new(Some(tx))),
        }
    }

    pub(super) fn clear(&self) {
        self.tx.lock().expect("child usage relay lock").take();
    }

    async fn emit(&self, event: SessionEvent) {
        let tx = self.tx.lock().expect("child usage relay lock").clone();
        if let Some(tx) = tx
            && !tx.is_closed()
        {
            let _ = tx.send(RuntimeStreamEvent::Session(event)).await;
        }
    }
}

fn usage_has_any_tokens(usage: &TokenUsage) -> bool {
    usage.input_tokens != 0
        || usage.output_tokens != 0
        || usage.cached_input_tokens != 0
        || usage.reasoning_tokens != 0
}

fn record_token_usage_shared(
    token_ledger: &Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    source: &str,
    model: &str,
    usage: &TokenUsage,
) {
    if !usage_has_any_tokens(usage) {
        return;
    }
    let mut ledger = token_ledger.lock().expect("token ledger lock");
    if let Some(entry) = ledger
        .iter_mut()
        .find(|entry| entry.source == source && entry.model == model)
    {
        entry.usage.input_tokens += usage.input_tokens;
        entry.usage.output_tokens += usage.output_tokens;
        entry.usage.cached_input_tokens += usage.cached_input_tokens;
        entry.usage.reasoning_tokens += usage.reasoning_tokens;
    } else {
        ledger.push(TokenLedgerEntry {
            source: source.to_string(),
            model: model.to_string(),
            usage: usage.clone(),
        });
    }
}

fn subtract_usage(reported_total: &TokenUsage, final_total: &TokenUsage) -> Option<TokenUsage> {
    let delta = TokenUsage {
        input_tokens: final_total
            .input_tokens
            .saturating_sub(reported_total.input_tokens),
        output_tokens: final_total
            .output_tokens
            .saturating_sub(reported_total.output_tokens),
        cached_input_tokens: final_total
            .cached_input_tokens
            .saturating_sub(reported_total.cached_input_tokens),
        reasoning_tokens: final_total
            .reasoning_tokens
            .saturating_sub(reported_total.reasoning_tokens),
    };
    usage_has_any_tokens(&delta).then_some(delta)
}

impl LiveChildUsageForwarder {
    async fn relay_token_usage(
        &self,
        iteration: usize,
        _usage: &TokenUsage,
        cumulative_usage: &TokenUsage,
    ) {
        let (delta, cumulative) = {
            let mut live_usage = self
                .child_turn_live_usage
                .lock()
                .expect("child turn live usage lock");
            let reported = live_usage.entry(self.turn_id.clone()).or_default();
            let Some(delta) = subtract_usage(reported, cumulative_usage) else {
                return;
            };
            *reported = cumulative_usage.clone();
            (delta, reported.clone())
        };
        record_token_usage_shared(&self.token_ledger, &self.source, &self.model, &delta);
        if let Some(relay) = &self.relay {
            relay
                .emit(SessionEvent::ChildTokenUsage {
                    session_id: self.session_id.clone(),
                    source: self.source.clone(),
                    model: self.model.clone(),
                    iteration,
                    usage: delta,
                    cumulative,
                })
                .await;
        }
    }
}

#[async_trait::async_trait]
impl EventSink for ChannelEventSink {
    async fn emit(&self, event: SessionEvent) {
        if let SessionEvent::TokenUsage {
            iteration,
            usage,
            cumulative,
        } = &event
            && let Some(live_usage) = &self.live_usage
        {
            live_usage
                .relay_token_usage(*iteration, usage, cumulative)
                .await;
        }
        if !self.tx.is_closed() {
            let _ = self.tx.send(event).await;
        }
    }
}

pub(super) struct PendingPrompt {
    pub(super) request: crate::PromptRequest,
    pub(super) response_tx: std::sync::mpsc::Sender<crate::PromptResponse>,
}

#[derive(Clone, Default)]
pub(super) struct HostPromptBridge {
    sender: Arc<StdMutex<Option<tokio::sync::mpsc::UnboundedSender<PendingPrompt>>>>,
}

impl HostPromptBridge {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn set_sender(&self, tx: tokio::sync::mpsc::UnboundedSender<PendingPrompt>) {
        *self.sender.lock().expect("prompt bridge poisoned") = Some(tx);
    }

    pub(super) fn clear_sender(&self) {
        *self.sender.lock().expect("prompt bridge poisoned") = None;
    }

    async fn prompt(
        &self,
        request: crate::PromptRequest,
    ) -> Result<crate::PromptResponse, crate::PluginError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| crate::PluginError::Session("prompt bridge poisoned".to_string()))?
            .clone()
            .ok_or_else(|| {
                crate::PluginError::Session("user prompts are unavailable in this session".into())
            })?;
        let (response_tx, response_rx) = std::sync::mpsc::channel::<crate::PromptResponse>();
        sender
            .send(PendingPrompt {
                request,
                response_tx,
            })
            .map_err(|_| crate::PluginError::Session("prompt channel closed".to_string()))?;
        tokio::task::spawn_blocking(move || response_rx.recv())
            .await
            .map_err(|err| crate::PluginError::Session(format!("prompt task failed: {err}")))?
            .map_err(|_| crate::PluginError::Session("prompt response channel closed".to_string()))
    }
}

impl RuntimeSessionManager {
    fn current_snapshot_meta_without_graph(runtime: &LashRuntime) -> SessionSnapshot {
        SessionSnapshot {
            session_id: runtime.state.session_id.clone(),
            policy: runtime.state.policy.clone(),
            session_graph: crate::SessionGraph::default(),
            iteration: runtime.state.iteration,
            token_usage: runtime.state.token_usage.clone(),
            last_prompt_usage: runtime.state.last_prompt_usage.clone(),
            dynamic_state_ref: runtime.state.dynamic_state_ref.clone(),
            dynamic_state_generation: runtime.state.dynamic_state_generation,
            dynamic_state_snapshot: None,
            plugin_snapshot_ref: runtime.state.plugin_snapshot_ref.clone(),
            plugin_snapshot_revision: runtime.state.plugin_snapshot_revision,
            plugin_snapshot: None,
            execution_state_snapshot: None,
            token_ledger: runtime.state.token_ledger.clone(),
            checkpoint_ref: runtime.state.checkpoint_ref.clone(),
            persisted_graph_node_count: runtime.state.persisted_graph_node_count,
            graph_replace_required: runtime.state.graph_replace_required,
        }
    }

    pub(super) fn new(
        runtime: &LashRuntime,
        prompt_bridge: Option<HostPromptBridge>,
        persist_usage_to_store: bool,
        child_usage_event_relay: Option<ChildUsageEventRelay>,
    ) -> Result<Self, ExternalInvokeError> {
        let Some(session) = runtime.session.as_ref() else {
            return Err(ExternalInvokeError::Unknown("session_manager".to_string()));
        };
        Ok(Self {
            runtime_scope_id: Arc::clone(&runtime.runtime_scope_id),
            current_session_id: runtime.state.session_id.clone(),
            current_snapshot: if persist_usage_to_store {
                CurrentSnapshot::Owned(runtime.export_graph_first_state())
            } else {
                CurrentSnapshot::Projection {
                    meta: Self::current_snapshot_meta_without_graph(runtime),
                    messages: runtime.state.session_graph.shared_projected_messages(),
                    tool_calls: runtime.state.session_graph.shared_projected_tool_calls(),
                }
            },
            current_policy: runtime.policy.clone(),
            current_host: runtime.host.clone(),
            current_plugins: Arc::clone(session.plugins()),
            current_tool_catalog: runtime.active_tool_catalog_shared(),
            current_prompt_bridge: prompt_bridge,
            current_store: runtime.services.store.clone(),
            registry: Arc::clone(&runtime.managed_sessions),
            turns: Arc::clone(&runtime.managed_turns),
            token_ledger: Arc::clone(&runtime.shared_token_ledger),
            background_sync_needed: Arc::clone(&runtime.background_sync_needed),
            child_usage_sources: Arc::new(std::sync::Mutex::new(HashMap::new())),
            child_turn_live_usage: Arc::new(std::sync::Mutex::new(HashMap::new())),
            child_usage_event_relay,
            persist_usage_to_store,
        })
    }

    fn record_token_usage(&self, source: &str, model: &str, usage: &TokenUsage) {
        record_token_usage_shared(&self.token_ledger, source, model, usage);
    }

    fn drain_token_ledger(&self) -> Vec<TokenLedgerEntry> {
        let mut ledger = self.token_ledger.lock().expect("token ledger lock");
        std::mem::take(&mut *ledger)
    }

    fn merge_drained_token_ledger(&self, state: &mut SessionSnapshot) -> bool {
        let drained = self.drain_token_ledger();
        if drained.is_empty() {
            return false;
        }
        for entry in drained {
            merge_ledger_entry(&mut state.token_ledger, entry);
        }
        true
    }

    fn background_scope_key(&self, session_id: &str) -> String {
        format!("{}:{session_id}", self.runtime_scope_id)
    }

    async fn current_snapshot_for_store_write(&self) -> SessionSnapshot {
        let mut state = self.current_snapshot.to_snapshot();
        if let Some(store) = &self.current_store
            && let Some(head) = store.load_session_head().await
        {
            super::apply_session_head(&mut state, &head);
            let checkpoint =
                super::load_session_checkpoint(store.as_ref(), head.checkpoint_ref.as_ref()).await;
            super::apply_session_checkpoint(&mut state, checkpoint);
        }
        super::normalize_session_graph(&mut state);
        state
    }

    async fn persist_current_usage_ledger(&self) -> Result<(), crate::PluginError> {
        if !self.persist_usage_to_store {
            return Ok(());
        }
        let Some(store) = &self.current_store else {
            return Ok(());
        };
        let mut state = self.current_snapshot_for_store_write().await;
        let drained = self.drain_token_ledger();
        if drained.is_empty() {
            return Ok(());
        }
        for entry in drained.iter().cloned() {
            merge_ledger_entry(&mut state.token_ledger, entry);
        }
        crate::store::append_usage_deltas(store.as_ref(), &drained).await;
        super::persist_session_graph_and_head(store.as_ref(), &mut state).await;
        Ok(())
    }

    fn build_runtime_state(
        &self,
        session_id: String,
        request: &SessionCreateRequest,
        mut base: SessionSnapshot,
        policy: &SessionPolicy,
    ) -> SessionSnapshot {
        normalize_session_graph(&mut base);
        base.session_id = session_id;
        base.policy = policy.clone();
        append_session_nodes_to_state(&mut base, &request.initial_nodes);
        normalize_session_graph(&mut base);
        base
    }

    async fn snapshot_by_id(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        if session_id == self.current_session_id {
            let mut snapshot = self.current_snapshot.to_snapshot();
            super::normalize_session_graph(&mut snapshot);
            self.enrich_current_snapshot(&mut snapshot);
            return Ok(snapshot);
        }
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        Ok(runtime.export_persisted_state())
    }

    async fn tool_catalog_by_id(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        if session_id == self.current_session_id {
            if let Some(runtime) = self.registry.lock().await.get(session_id).cloned() {
                let runtime = runtime.lock().await;
                return Ok(runtime.active_tool_catalog());
            }
            if self.current_plugins.dynamic_tools().is_some() {
                return Ok(self
                    .current_plugins
                    .tool_catalog(session_id, self.current_policy.execution_mode));
            }
            return Ok(self.current_tool_catalog.as_ref().clone());
        }
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        Ok(runtime.active_tool_catalog())
    }

    fn enrich_current_snapshot(&self, snapshot: &mut SessionSnapshot) {
        if let Some(dynamic_tools) = self.current_plugins.dynamic_tools() {
            let dynamic_state = dynamic_tools.export_state();
            snapshot.dynamic_state_generation = Some(dynamic_state.base_generation);
            snapshot.dynamic_state_snapshot = Some(dynamic_state);
        } else {
            snapshot.dynamic_state_generation = None;
            snapshot.dynamic_state_snapshot = None;
        }
        snapshot.plugin_snapshot = self.current_plugins.snapshot().ok();
        snapshot.plugin_snapshot_revision =
            Some(self.current_plugins.snapshot_revision_fingerprint());
    }

    fn current_dynamic_tools(&self) -> Result<Arc<crate::DynamicToolProvider>, crate::PluginError> {
        self.current_plugins.dynamic_tools().ok_or_else(|| {
            crate::PluginError::Session("dynamic tools are unavailable in this session".to_string())
        })
    }

    async fn invoke_monitor_external(
        &self,
        session_id: &str,
        name: &str,
        args: serde_json::Value,
    ) -> Result<crate::ToolResult, crate::PluginError> {
        self.current_plugins
            .host()
            .invoke_external_for_session(session_id, name, args, Arc::new(self.clone()))
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }

    async fn ensure_registered_monitor_specs(
        &self,
        session_id: &str,
    ) -> Result<(), crate::PluginError> {
        let specs = self
            .current_plugins
            .host()
            .monitor_specs_for_session(session_id)
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        if specs.is_empty() {
            return Ok(());
        }
        let specs = specs
            .into_iter()
            .map(|owned| {
                serde_json::json!({
                    "plugin_id": owned.plugin_id,
                    "spec": owned.value,
                })
            })
            .collect::<Vec<_>>();
        let result = self
            .invoke_monitor_external(
                session_id,
                "monitor.register_specs",
                serde_json::json!({
                    "specs": specs,
                }),
            )
            .await?;
        if result.success {
            Ok(())
        } else {
            Err(crate::PluginError::Session(result.result.to_string()))
        }
    }
}

#[async_trait::async_trait]
impl SessionManager for RuntimeSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, crate::PluginError> {
        let mut snapshot = self.current_snapshot.to_snapshot();
        super::normalize_session_graph(&mut snapshot);
        self.enrich_current_snapshot(&mut snapshot);
        Ok(snapshot)
    }

    async fn snapshot_session(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        self.snapshot_by_id(session_id).await
    }

    async fn tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        self.tool_catalog_by_id(session_id).await
    }

    async fn dynamic_tool_state(
        &self,
        session_id: &str,
    ) -> Result<crate::DynamicStateSnapshot, crate::PluginError> {
        if session_id == self.current_session_id {
            if let Some(runtime) = self.registry.lock().await.get(session_id).cloned() {
                let runtime = runtime.lock().await;
                return runtime
                    .dynamic_tool_state()
                    .map_err(|err| crate::PluginError::Session(err.to_string()));
            }
            return Ok(self.current_dynamic_tools()?.export_state());
        }

        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        runtime
            .dynamic_tool_state()
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }

    async fn apply_dynamic_tool_state(
        &self,
        session_id: &str,
        snapshot: crate::DynamicStateSnapshot,
    ) -> Result<u64, crate::PluginError> {
        if session_id == self.current_session_id {
            if let Some(runtime) = self.registry.lock().await.get(session_id).cloned() {
                let mut runtime = runtime.lock().await;
                return runtime
                    .apply_dynamic_tool_state(snapshot)
                    .await
                    .map_err(|err| crate::PluginError::Session(err.to_string()));
            }
            let dynamic_tools = self.current_dynamic_tools()?;
            return dynamic_tools
                .apply_state(snapshot)
                .map_err(|err| crate::PluginError::Session(err.to_string()));
        }

        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let mut runtime = runtime.lock().await;
        runtime
            .apply_dynamic_tool_state(snapshot)
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }

    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, crate::PluginError> {
        let session_id = request
            .session_id
            .clone()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if session_id == self.current_session_id
            || self.registry.lock().await.contains_key(&session_id)
        {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` already exists"
            )));
        }
        let snapshot = match &request.start {
            SessionStartPoint::Empty => SessionSnapshot {
                session_id: session_id.clone(),
                ..Default::default()
            },
            SessionStartPoint::CurrentSession => self.current_snapshot.to_snapshot(),
            SessionStartPoint::ExistingSession { session_id } => {
                self.snapshot_by_id(session_id).await?
            }
            SessionStartPoint::Snapshot { snapshot } => (**snapshot).clone(),
        };
        let mut policy = request
            .policy
            .clone()
            .unwrap_or_else(|| match &request.start {
                SessionStartPoint::Empty => self.current_policy.clone(),
                _ => snapshot.policy.clone(),
            });
        if request.parent_session_id.is_some() {
            policy.session_id = Some(session_id.clone());
        }
        let state = self.build_runtime_state(session_id.clone(), &request, snapshot, &policy);
        let plugins = match request.plugin_mode {
            crate::SessionPluginMode::Fresh => self
                .current_plugins
                .host()
                .build_session_with_parent(
                    &session_id,
                    request.parent_session_id.clone(),
                    policy.execution_mode,
                    policy.context_approach.clone(),
                    None,
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?,
            crate::SessionPluginMode::InheritCurrent => self
                .current_plugins
                .fork_for_session(
                    &session_id,
                    policy.execution_mode,
                    policy.context_approach.clone(),
                )
                .map_err(|err| crate::PluginError::Session(err.to_string()))?,
        };
        let session_store = match &self.current_host.session_store_factory {
            Some(factory) => Some(
                factory
                    .create_store(&SessionStoreCreateRequest {
                        session_id: session_id.clone(),
                        parent_session_id: request.parent_session_id.clone(),
                        policy: policy.clone(),
                    })
                    .map_err(crate::PluginError::Session)?,
            ),
            None => None,
        };
        let mut runtime = match (&self.current_host.session_task_executor, &session_store) {
            (Some(executor), Some(store)) => {
                let host = BackgroundRuntimeHost::new(
                    EmbeddedRuntimeHost {
                        core: self.current_host.core.clone(),
                        session_store_factory: self.current_host.session_store_factory.clone(),
                    },
                    Arc::clone(executor),
                );
                LashRuntime::from_persistent_background_state(
                    policy.clone(),
                    host,
                    crate::PersistentRuntimeServices::new(plugins, Arc::clone(store)),
                    state,
                )
                .await
            }
            (Some(executor), None) => {
                let host = BackgroundRuntimeHost::new(
                    EmbeddedRuntimeHost {
                        core: self.current_host.core.clone(),
                        session_store_factory: self.current_host.session_store_factory.clone(),
                    },
                    Arc::clone(executor),
                );
                LashRuntime::from_background_state(
                    policy.clone(),
                    host,
                    RuntimeServices::new(plugins),
                    state,
                )
                .await
            }
            (None, Some(store)) => {
                let host = EmbeddedRuntimeHost {
                    core: self.current_host.core.clone(),
                    session_store_factory: self.current_host.session_store_factory.clone(),
                };
                LashRuntime::from_persistent_embedded_state(
                    policy.clone(),
                    host,
                    crate::PersistentRuntimeServices::new(plugins, Arc::clone(store)),
                    state,
                )
                .await
            }
            (None, None) => {
                let host = EmbeddedRuntimeHost {
                    core: self.current_host.core.clone(),
                    session_store_factory: self.current_host.session_store_factory.clone(),
                };
                LashRuntime::from_embedded_state(
                    policy.clone(),
                    host,
                    RuntimeServices::new(plugins),
                    state,
                )
                .await
            }
        }
        .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        let mode_session = runtime
            .session
            .as_ref()
            .map(|session| Arc::clone(session.plugins().mode_session()));
        if let Some(mode_session) = mode_session {
            mode_session.configure_runtime_from_request(
                crate::plugin::ModeRuntimeContext::new(&mut runtime),
                &request,
            );
        }
        if let Some(session) = runtime.session.as_mut() {
            session.set_context_surface(
                request.context_surface.tool_providers.clone(),
                request.context_surface.prompt_contributions.clone(),
                request.context_surface.include_base_tools,
            );
        }
        if let Some(store) = &session_store {
            let mut persisted_state = runtime.export_persisted_state();
            super::normalize_session_graph(&mut persisted_state);
            let crate::store::RuntimeCommitResult::PersistedState(result) = store
                .apply_runtime_commit(crate::store::RuntimeCommit::persisted_state(
                    &persisted_state,
                    &[],
                ))
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?
            else {
                unreachable!("persisted state commit should return persisted result");
            };
            persisted_state.apply_persisted_commit_result(result);
        }
        self.registry
            .lock()
            .await
            .insert(session_id.clone(), Arc::new(Mutex::new(runtime)));
        // Stash the usage_source label so await_turn can tag the child's
        // token cost in the parent's ledger.
        if let Some(source) = &request.usage_source {
            self.child_usage_sources
                .lock()
                .expect("child usage sources lock")
                .insert(session_id.clone(), source.clone());
        }
        Ok(SessionHandle {
            session_id,
            parent_session_id: request.parent_session_id,
            policy,
        })
    }

    async fn close_session(&self, session_id: &str) -> Result<(), crate::PluginError> {
        if session_id == self.current_session_id {
            return Err(crate::PluginError::Session(
                "cannot close the current session".to_string(),
            ));
        }
        if self
            .turns
            .lock()
            .await
            .values()
            .any(|turn| turn.session_id == session_id)
        {
            return Err(crate::PluginError::Session(format!(
                "cannot close session `{session_id}` while a turn is running"
            )));
        }
        self.registry.lock().await.remove(session_id);
        self.child_usage_sources
            .lock()
            .expect("child usage sources lock")
            .remove(session_id);
        self.current_plugins.host().unregister_session(session_id)?;
        Ok(())
    }

    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, crate::PluginError> {
        if self
            .turns
            .lock()
            .await
            .values()
            .any(|turn| turn.session_id == session_id)
        {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` already has a running turn"
            )));
        }
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let policy = {
            let runtime = runtime.lock().await;
            runtime.session_policy()
        };
        let turn_id = uuid::Uuid::new_v4().to_string();
        let cancel = CancellationToken::new();
        let (event_tx, event_rx) = mpsc::channel::<SessionEvent>(100);
        let usage_source = self
            .child_usage_sources
            .lock()
            .expect("child usage sources lock")
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| "child".to_string());
        let runtime_clone = Arc::clone(&runtime);
        let cancel_clone = cancel.clone();
        let sink = ChannelEventSink {
            tx: event_tx,
            live_usage: Some(LiveChildUsageForwarder {
                turn_id: turn_id.clone(),
                session_id: session_id.to_string(),
                source: usage_source,
                model: policy.model.clone(),
                token_ledger: Arc::clone(&self.token_ledger),
                child_turn_live_usage: Arc::clone(&self.child_turn_live_usage),
                relay: self.child_usage_event_relay.clone(),
            }),
        };
        let task = tokio::spawn(async move {
            let mut runtime = runtime_clone.lock().await;
            runtime
                .refresh_session_tool_surface()
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
            runtime
                .stream_turn(input, &sink, cancel_clone)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))
        });
        self.turns.lock().await.insert(
            turn_id.clone(),
            ManagedSessionTurn {
                session_id: session_id.to_string(),
                cancel,
                task,
            },
        );
        Ok(crate::plugin::SessionTurnHandle {
            turn_id,
            session_id: session_id.to_string(),
            policy,
            events: event_rx,
        })
    }

    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, crate::PluginError> {
        let managed = self
            .turns
            .lock()
            .await
            .remove(turn_id)
            .ok_or_else(|| crate::PluginError::Session(format!("unknown turn `{turn_id}`")))?;
        let session_id = managed.session_id.clone();
        let turn = managed
            .task
            .await
            .map_err(|err| crate::PluginError::Session(format!("turn task failed: {err}")))?;
        let live_reported = self
            .child_turn_live_usage
            .lock()
            .expect("child turn live usage lock")
            .remove(turn_id)
            .unwrap_or_default();
        if let Ok(turn) = &turn {
            let source = self
                .child_usage_sources
                .lock()
                .expect("child usage sources lock")
                .get(&session_id)
                .cloned()
                .unwrap_or_else(|| "child".to_string());
            if let Some(remainder) = subtract_usage(&live_reported, &turn.token_usage) {
                self.record_token_usage(&source, &turn.state.policy.model, &remainder);
            }
        }
        self.persist_current_usage_ledger().await?;
        turn
    }

    async fn cancel_turn(&self, turn_id: &str) -> Result<(), crate::PluginError> {
        let turns = self.turns.lock().await;
        let managed = turns
            .get(turn_id)
            .ok_or_else(|| crate::PluginError::Session(format!("unknown turn `{turn_id}`")))?;
        managed.cancel.cancel();
        Ok(())
    }

    async fn spawn_hidden_task(
        &self,
        session_id: &str,
        label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        if session_id != self.current_session_id {
            let known = self.registry.lock().await.contains_key(session_id);
            if !known {
                return Err(crate::PluginError::Session(format!(
                    "unknown session `{session_id}`"
                )));
            }
        }
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "session tasks are unavailable in this runtime".to_string(),
            ));
        };
        if session_id == self.current_session_id {
            self.background_sync_needed.store(true, Ordering::Release);
        }
        executor
            .spawn_hidden(&self.background_scope_key(session_id), label, task)
            .await
    }

    async fn await_hidden_tasks(&self, session_id: &str) -> Result<(), crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Ok(());
        };
        executor
            .await_hidden(&self.background_scope_key(session_id))
            .await
    }

    async fn spawn_managed_task(
        &self,
        session_id: &str,
        spec: crate::ManagedTaskSpec,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        if session_id != self.current_session_id {
            let known = self.registry.lock().await.contains_key(session_id);
            if !known {
                return Err(crate::PluginError::Session(format!(
                    "unknown session `{session_id}`"
                )));
            }
        }
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "managed session tasks are unavailable in this runtime".to_string(),
            ));
        };
        if session_id == self.current_session_id {
            self.background_sync_needed.store(true, Ordering::Release);
        }
        executor
            .spawn_managed(&self.background_scope_key(session_id), spec, task)
            .await
    }

    async fn cancel_managed_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<(), crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Ok(());
        };
        executor
            .cancel_managed(&self.background_scope_key(session_id), task_id)
            .await
    }

    async fn register_background_task(
        &self,
        session_id: &str,
        spec: crate::ManagedTaskSpec,
        cancel: Option<crate::ManagedTaskCancel>,
    ) -> Result<(), crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        executor
            .register_external(&self.background_scope_key(session_id), spec, cancel)
            .await
    }

    async fn complete_background_task(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        let Some(executor) = &self.current_host.session_task_executor else {
            return;
        };
        executor
            .mark_terminal(&self.background_scope_key(session_id), task_id, run_state)
            .await;
    }

    async fn transition_background_task_live_state(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        let Some(executor) = &self.current_host.session_task_executor else {
            return;
        };
        executor
            .mark_live_state(&self.background_scope_key(session_id), task_id, run_state)
            .await;
    }

    async fn inject_turn_input(
        &self,
        session_id: &str,
        message: crate::PluginMessage,
    ) -> Result<(), crate::PluginError> {
        let runtime_arc = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        };
        let Some(runtime_arc) = runtime_arc else {
            return Err(crate::PluginError::Session(format!(
                "unknown or inactive session `{session_id}` for turn input injection"
            )));
        };
        let runtime = runtime_arc.lock().await;
        let Some(session) = runtime.session.as_ref() else {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` has no live turn-input bridge"
            )));
        };
        session
            .turn_input_injection_bridge()
            .enqueue(vec![crate::InjectedTurnInput { message }])
            .map_err(crate::PluginError::Session)
    }

    async fn list_background_tasks(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::ManagedTaskStatus>, crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        Ok(executor
            .list_managed(&self.background_scope_key(session_id))
            .await)
    }

    async fn cancel_background_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<crate::ManagedTaskStatus, crate::PluginError> {
        let Some(executor) = &self.current_host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        let scope_key = self.background_scope_key(session_id);
        let Some(status) = executor.get_managed(&scope_key, task_id).await else {
            return Err(crate::PluginError::Session(format!(
                "unknown background task `{task_id}`"
            )));
        };
        match status.kind {
            crate::ManagedTaskKind::Monitor => {
                let monitor_id = task_id
                    .strip_prefix("monitor:")
                    .unwrap_or(task_id)
                    .to_string();
                self.stop_monitor(session_id, &monitor_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
            crate::ManagedTaskKind::Subagent => {
                // Producer-owned cancel: ask the task owner to terminate via
                // its registered cancellation channel. For now we just mark
                // the record; subagent host wires up a close path in its
                // register_background_task adapter.
                executor.cancel_managed(&scope_key, task_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
            _ => {
                executor.cancel_managed(&scope_key, task_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
        }
        let updated = executor
            .get_managed(&scope_key, task_id)
            .await
            .unwrap_or(status);
        Ok(updated)
    }

    async fn monitor_snapshot(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(session_id, "monitor.status", serde_json::json!({}))
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }

    async fn take_monitor_updates(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorUpdateBatch, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(session_id, "monitor.take_updates", serde_json::json!({}))
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor updates: {err}")))
    }

    async fn start_monitor(
        &self,
        session_id: &str,
        spec: crate::MonitorSpec,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(
                session_id,
                "monitor.start",
                serde_json::json!({ "spec": spec }),
            )
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }

    async fn stop_monitor(
        &self,
        session_id: &str,
        monitor_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(
                session_id,
                "monitor.stop",
                serde_json::json!({ "id": monitor_id }),
            )
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }

    async fn prompt_user(
        &self,
        request: crate::PromptRequest,
    ) -> Result<crate::PromptResponse, crate::PluginError> {
        let Some(prompt_bridge) = &self.current_prompt_bridge else {
            return Err(crate::PluginError::Session(
                "user prompts are unavailable in this session".to_string(),
            ));
        };
        prompt_bridge.prompt(request).await
    }

    async fn append_session_nodes(
        &self,
        session_id: &str,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, crate::PluginError> {
        if let Some(runtime) = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        } {
            let mut runtime = runtime.lock().await;
            return runtime
                .append_session_nodes(request)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()));
        }

        if session_id != self.current_session_id {
            return Err(crate::PluginError::Session(format!(
                "unknown session `{session_id}`"
            )));
        }

        let Some(store) = &self.current_store else {
            return Err(crate::PluginError::Session(
                "session graph mutation requires a runtime store".to_string(),
            ));
        };

        let mut state = if self.persist_usage_to_store {
            self.current_snapshot_for_store_write().await
        } else {
            let mut state = self.current_snapshot.to_snapshot();
            super::normalize_session_graph(&mut state);
            state
        };
        if self.persist_usage_to_store {
            let _ = self.merge_drained_token_ledger(&mut state);
        }
        if let Some(required) = request.requires_ancestor_node_id.as_deref()
            && !state.session_graph.active_path_contains(required)
        {
            return Ok(crate::AppendSessionNodesResult::StaleBranch {
                current_leaf_node_id: state.session_graph.leaf_node_id.clone(),
            });
        }
        let node_ids = append_session_nodes_to_state(&mut state, &request.nodes);
        let leaf_node_id = state.session_graph.leaf_node_id.clone().unwrap_or_default();
        let crate::store::RuntimeCommitResult::PersistedState(result) = store
            .apply_runtime_commit(crate::store::RuntimeCommit::persisted_state(&state, &[]))
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))?
        else {
            unreachable!("persisted state commit should return persisted result");
        };
        state.apply_persisted_commit_result(result);
        self.background_sync_needed.store(true, Ordering::Release);
        Ok(crate::AppendSessionNodesResult::Appended {
            node_ids,
            leaf_node_id,
        })
    }

    async fn direct_completion(
        &self,
        request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        let mut provider = self.current_policy.provider.clone();
        let model = provider.resolve_model(&request.model);
        if let Some(variant) = request.model_variant.as_deref() {
            provider
                .validate_variant(&model, variant)
                .map_err(crate::PluginError::Session)?;
        }
        provider
            .ensure_ready()
            .await
            .map_err(|err| crate::PluginError::Session(err.message.clone()))?;
        let llm_request = crate::direct::build_llm_request(&provider, request, model.clone());
        let response = provider
            .complete(llm_request)
            .await
            .map_err(|err| crate::PluginError::Session(err.message.clone()))?;
        let usage = TokenUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            cached_input_tokens: response.usage.cached_input_tokens,
            reasoning_tokens: response.usage.reasoning_tokens,
        };
        self.record_token_usage(usage_source, &model, &usage);
        self.persist_current_usage_ledger().await?;
        Ok(crate::DirectCompletion {
            text: response.full_text,
            usage,
        })
    }
}

pub(super) async fn emit_session_events_to_sink(
    events: &dyn EventSink,
    plugin_events: Vec<SessionEvent>,
) {
    for event in plugin_events {
        events.emit(event).await;
    }
}

pub(super) async fn emit_session_events(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    plugin_events: Vec<SessionEvent>,
) {
    for event in plugin_events {
        if !event_tx.is_closed() {
            let _ = event_tx.send(RuntimeStreamEvent::Session(event)).await;
        }
    }
}
