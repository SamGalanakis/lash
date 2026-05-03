use super::*;
use std::sync::atomic::AtomicBool;

mod api;
mod prompt;
mod usage;

pub(in crate::runtime) use prompt::{HostPromptBridge, PendingPrompt};
pub(in crate::runtime) use usage::ChildUsageEventRelay;
use usage::record_token_usage_shared;
pub(in crate::runtime::session_manager) use usage::{
    ChannelEventSink, LiveChildUsageForwarder, subtract_usage,
};

#[derive(Clone)]
enum CurrentSnapshot {
    Owned(SessionSnapshot),
    ReadModel {
        meta: SessionSnapshot,
        messages: Arc<Vec<Message>>,
        tool_calls: Arc<Vec<ToolCallRecord>>,
    },
}

impl CurrentSnapshot {
    fn to_snapshot(&self) -> SessionSnapshot {
        match self {
            Self::Owned(snapshot) => snapshot.clone(),
            Self::ReadModel {
                meta,
                messages,
                tool_calls,
            } => {
                let mut snapshot = meta.clone();
                snapshot.replace_active_read_state(messages.as_slice(), tool_calls.as_slice());
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
struct CurrentSessionBacking {
    session_id: String,
    snapshot: CurrentSnapshot,
    policy: SessionPolicy,
    host: RuntimeHost,
    plugins: Arc<crate::PluginSession>,
    tool_catalog: Arc<Vec<serde_json::Value>>,
    prompt_bridge: Option<HostPromptBridge>,
    store: Option<Arc<dyn crate::store::RuntimePersistence>>,
}

#[derive(Clone)]
struct ManagedSessionsBacking {
    registry: Arc<Mutex<HashMap<String, Arc<Mutex<LashRuntime>>>>>,
    turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    /// Maps child session_id → seed PluginMessage queued via
    /// `SessionCreateRequest::first_turn_input`. Drained by
    /// `take_first_turn_input` when a host claims it.
    pending_first_turn_inputs: Arc<std::sync::Mutex<HashMap<String, crate::PluginMessage>>>,
}

#[derive(Clone)]
struct UsageBridgeBacking {
    /// Session-scoped token cost ledger shared with the parent
    /// `LashRuntime`. All managers created from the same runtime
    /// write to the same Arc. Drained at turn-commit time.
    token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    /// Maps child session_id → usage_source label.
    child_sources: Arc<std::sync::Mutex<HashMap<String, String>>>,
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
    persist_to_store: bool,
}

#[derive(Clone)]
struct BackgroundTaskBacking {
    runtime_scope_id: Arc<str>,
    sync_needed: Arc<AtomicBool>,
}

#[derive(Clone)]
pub(super) struct RuntimeSessionManager {
    current: CurrentSessionBacking,
    managed: ManagedSessionsBacking,
    usage: UsageBridgeBacking,
    background: BackgroundTaskBacking,
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
            mode_turn_options: runtime.mode_turn_options.clone(),
            dynamic_state_ref: runtime.state.dynamic_state_ref.clone(),
            dynamic_state_generation: runtime.state.dynamic_state_generation,
            dynamic_state_snapshot: None,
            plugin_snapshot_ref: runtime.state.plugin_snapshot_ref.clone(),
            plugin_snapshot_revision: runtime.state.plugin_snapshot_revision,
            plugin_snapshot: None,
            execution_state_ref: runtime.state.execution_state_ref.clone(),
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
            current: CurrentSessionBacking {
                session_id: runtime.state.session_id.clone(),
                snapshot: if persist_usage_to_store {
                    CurrentSnapshot::Owned(runtime.export_graph_first_state())
                } else {
                    let read_model = runtime.state.read_model();
                    CurrentSnapshot::ReadModel {
                        meta: Self::current_snapshot_meta_without_graph(runtime),
                        messages: read_model.messages,
                        tool_calls: read_model.tool_calls,
                    }
                },
                policy: runtime.policy.clone(),
                host: runtime.host.clone(),
                plugins: Arc::clone(session.plugins()),
                tool_catalog: runtime.active_tool_catalog_shared(),
                prompt_bridge,
                store: runtime.services.store.clone(),
            },
            managed: ManagedSessionsBacking {
                registry: Arc::clone(&runtime.managed_sessions),
                turns: Arc::clone(&runtime.managed_turns),
                pending_first_turn_inputs: Arc::clone(&runtime.pending_first_turn_inputs),
            },
            usage: UsageBridgeBacking {
                token_ledger: Arc::clone(&runtime.shared_token_ledger),
                child_sources: Arc::new(std::sync::Mutex::new(HashMap::new())),
                child_turn_live_usage: Arc::new(std::sync::Mutex::new(HashMap::new())),
                child_usage_event_relay,
                persist_to_store: persist_usage_to_store,
            },
            background: BackgroundTaskBacking {
                runtime_scope_id: Arc::clone(&runtime.runtime_scope_id),
                sync_needed: Arc::clone(&runtime.background_sync_needed),
            },
        })
    }

    fn record_token_usage(&self, source: &str, model: &str, usage: &TokenUsage) {
        record_token_usage_shared(&self.usage.token_ledger, source, model, usage);
    }

    fn drain_token_ledger(&self) -> Vec<TokenLedgerEntry> {
        let mut ledger = self.usage.token_ledger.lock().expect("token ledger lock");
        std::mem::take(&mut *ledger)
    }

    fn merge_drained_token_ledger(&self, state: &mut SessionSnapshot) -> Vec<TokenLedgerEntry> {
        let drained = self.drain_token_ledger();
        for entry in drained.iter().cloned() {
            merge_ledger_entry(&mut state.token_ledger, entry);
        }
        drained
    }

    fn background_scope_key(&self, session_id: &str) -> String {
        format!("{}:{session_id}", self.background.runtime_scope_id)
    }

    async fn current_snapshot_for_store_write(&self) -> SessionSnapshot {
        let mut state = self.current.snapshot.to_snapshot();
        if let Some(store) = &self.current.store {
            crate::store::refresh_persisted_session_state(store.as_ref(), &mut state).await;
        }
        super::normalize_session_graph(&mut state);
        state
    }

    async fn persist_current_usage_ledger(&self) -> Result<(), crate::PluginError> {
        if !self.usage.persist_to_store {
            return Ok(());
        }
        let Some(store) = &self.current.store else {
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
        let commit = crate::store::PersistedStateCommit::persisted_state(&state, &drained);
        match crate::store::apply_runtime_commit(store.as_ref(), commit).await {
            Ok(result) => state.apply_persisted_commit_result(result),
            Err(err) => tracing::warn!("failed to persist current usage ledger: {err}"),
        }
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
        if session_id == self.current.session_id {
            let mut snapshot = self.current.snapshot.to_snapshot();
            super::normalize_session_graph(&mut snapshot);
            self.enrich_current_snapshot(&mut snapshot);
            return Ok(snapshot);
        }
        let runtime = {
            let registry = self.managed.registry.lock().await;
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
        if session_id == self.current.session_id {
            if let Some(runtime) = self.managed.registry.lock().await.get(session_id).cloned() {
                let runtime = runtime.lock().await;
                return Ok(runtime.active_tool_catalog());
            }
            if self.current.plugins.dynamic_tools().is_some() {
                return Ok(self
                    .current
                    .plugins
                    .tool_catalog(session_id, self.current.policy.execution_mode.clone()));
            }
            return Ok(self.current.tool_catalog.as_ref().clone());
        }
        let runtime = {
            let registry = self.managed.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        Ok(runtime.active_tool_catalog())
    }

    fn enrich_current_snapshot(&self, snapshot: &mut SessionSnapshot) {
        if let Some(dynamic_tools) = self.current.plugins.dynamic_tools() {
            let dynamic_state = dynamic_tools.export_state();
            snapshot.dynamic_state_generation = Some(dynamic_state.base_generation);
            snapshot.dynamic_state_snapshot = Some(dynamic_state);
        } else {
            snapshot.dynamic_state_generation = None;
            snapshot.dynamic_state_snapshot = None;
        }
        snapshot.plugin_snapshot = self.current.plugins.snapshot().ok();
        snapshot.plugin_snapshot_revision =
            Some(self.current.plugins.snapshot_revision_fingerprint());
    }

    fn current_dynamic_tools(&self) -> Result<Arc<crate::DynamicToolProvider>, crate::PluginError> {
        self.current.plugins.dynamic_tools().ok_or_else(|| {
            crate::PluginError::Session("dynamic tools are unavailable in this session".to_string())
        })
    }

    async fn invoke_monitor_external(
        &self,
        session_id: &str,
        name: &str,
        args: serde_json::Value,
    ) -> Result<crate::ToolResult, crate::PluginError> {
        self.current
            .plugins
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
            .current
            .plugins
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

pub(super) async fn emit_session_events_to_sink(
    events: &dyn EventSink,
    plugin_events: Vec<SessionEvent>,
) {
    if events.is_noop() {
        return;
    }
    for event in plugin_events {
        events.emit(event).await;
    }
}

pub(super) async fn emit_session_event_to_sink(events: &dyn EventSink, event: SessionEvent) {
    if !events.is_noop() {
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
