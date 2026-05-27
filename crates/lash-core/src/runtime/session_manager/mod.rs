use super::*;
use std::sync::atomic::AtomicBool;

mod api;
mod create_plan;
mod current;
mod direct;
mod graph;
mod managed;
mod materialize;
mod process_runners;
mod turns;
mod usage;

pub use direct::DirectCompletionClient;
pub(in crate::runtime) use usage::ChildUsageEventRelay;
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
    pub(super) task: tokio::task::JoinHandle<Result<AssembledTurn, crate::PluginError>>,
}

#[derive(Clone)]
pub(in crate::runtime) struct CurrentSessionCapability {
    pub(in crate::runtime) session_id: String,
    snapshot: CurrentSnapshot,
    policy: SessionPolicy,
    pub(in crate::runtime) host: RuntimeHost,
    plugins: Arc<crate::PluginSession>,
    store: Option<Arc<dyn crate::store::RuntimePersistence>>,
    turn_lease: Option<crate::RuntimeTurnLease>,
}

#[derive(Clone)]
struct ManagedSessionCapability {
    registry: Arc<Mutex<HashMap<String, RuntimeHandle>>>,
    active_handoff_continuations: Arc<Mutex<HashMap<String, String>>>,
    turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    /// Maps child session_id → seed PluginMessage queued via
    /// `SessionCreateRequest::first_turn_input`. Drained by
    /// `take_first_turn_input` when a host claims it.
    pending_first_turn_inputs: Arc<std::sync::Mutex<HashMap<String, crate::PluginMessage>>>,
}

#[derive(Clone)]
pub(in crate::runtime) struct UsageCapability {
    /// Session-scoped token cost ledger shared with the parent
    /// `LashRuntime`. All managers created from the same runtime
    /// write to the same Arc. Drained at turn-commit time.
    token_ledger: Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    /// Maps child session_id → usage_source label.
    child_sources: Arc<std::sync::Mutex<HashMap<String, String>>>,
    /// Tracks live child-turn usage already bubbled into the shared
    /// token ledger so child turn completion can reconcile final usage
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
struct ProcessCapability {
    runtime_scope_id: Arc<str>,
    sync_needed: Arc<AtomicBool>,
}

#[derive(Clone, Default)]
struct DirectCompletionCapability;

#[derive(Clone)]
pub(super) struct RuntimeSessionManager {
    current: CurrentSessionCapability,
    managed: ManagedSessionCapability,
    processes: ProcessCapability,
    usage: UsageCapability,
    direct: DirectCompletionCapability,
}

impl CurrentSessionCapability {
    fn snapshot_meta_without_graph(runtime: &LashRuntime) -> SessionSnapshot {
        SessionSnapshot {
            session_id: runtime.state.session_id.clone(),
            policy: runtime.state.policy.clone(),
            session_graph: crate::SessionGraph::default(),
            turn_index: runtime.state.turn_index,
            token_usage: runtime.state.token_usage.clone(),
            last_prompt_usage: runtime.state.last_prompt_usage.clone(),
            protocol_turn_options: runtime.protocol_turn_options.clone(),
            tool_state_ref: runtime.state.tool_state_ref.clone(),
            tool_state_generation: runtime.state.tool_state_generation,
            tool_state_snapshot: None,
            plugin_snapshot_ref: runtime.state.plugin_snapshot_ref.clone(),
            plugin_snapshot_revision: runtime.state.plugin_snapshot_revision,
            plugin_snapshot: None,
            execution_state_ref: runtime.state.execution_state_ref.clone(),
            execution_state_snapshot: None,
            token_ledger: runtime.state.token_ledger.clone(),
            checkpoint_ref: runtime.state.checkpoint_ref.clone(),
            head_revision: runtime.state.head_revision,
            graph_replace_required: runtime.state.graph_replace_required,
        }
    }

    fn new(
        runtime: &LashRuntime,
        plugins: Arc<crate::PluginSession>,
        persist_usage_to_store: bool,
        turn_lease: Option<crate::RuntimeTurnLease>,
    ) -> Self {
        Self {
            session_id: runtime.state.session_id.clone(),
            snapshot: if persist_usage_to_store {
                CurrentSnapshot::Owned(runtime.export_graph_first_state())
            } else {
                let read_model = runtime.state.read_model();
                CurrentSnapshot::ReadModel {
                    meta: Self::snapshot_meta_without_graph(runtime),
                    messages: read_model.messages,
                    tool_calls: read_model.tool_calls,
                }
            },
            policy: runtime.policy.clone(),
            host: runtime.host.clone(),
            plugins,
            store: runtime.services.store.clone(),
            turn_lease,
        }
    }
}

impl ManagedSessionCapability {
    fn new(runtime: &LashRuntime) -> Self {
        Self {
            registry: Arc::clone(&runtime.managed_sessions),
            active_handoff_continuations: Arc::clone(&runtime.active_handoff_continuations),
            turns: Arc::clone(&runtime.managed_turns),
            pending_first_turn_inputs: Arc::clone(&runtime.pending_first_turn_inputs),
        }
    }
}

impl ProcessCapability {
    fn new(runtime: &LashRuntime) -> Self {
        Self {
            runtime_scope_id: Arc::clone(&runtime.runtime_scope_id),
            sync_needed: Arc::clone(&runtime.process_sync_needed),
        }
    }
}

impl UsageCapability {
    fn new(
        runtime: &LashRuntime,
        persist_to_store: bool,
        child_usage_event_relay: Option<ChildUsageEventRelay>,
    ) -> Self {
        Self {
            token_ledger: Arc::clone(&runtime.shared_token_ledger),
            child_sources: Arc::new(std::sync::Mutex::new(HashMap::new())),
            child_turn_live_usage: Arc::new(std::sync::Mutex::new(HashMap::new())),
            child_usage_event_relay,
            persist_to_store,
        }
    }
}

impl RuntimeSessionManager {
    pub(in crate::runtime) fn process_scope_id(&self, session_id: &str) -> crate::ProcessScopeId {
        self.processes.process_scope_id(session_id)
    }

    pub(in crate::runtime) fn process_registry(&self) -> Option<Arc<dyn crate::ProcessRegistry>> {
        self.current.host.process_registry.clone()
    }

    pub(super) fn direct_completion_client<'run>(
        self: &Arc<Self>,
        effect_controller: crate::runtime::RuntimeEffectControllerHandle<'run>,
        turn_id: Option<String>,
        turn_lease: Option<crate::RuntimeTurnLease>,
    ) -> DirectCompletionClient<'run> {
        DirectCompletionClient::runtime(Arc::clone(self), effect_controller, turn_id, turn_lease)
    }

    pub(super) fn new(
        runtime: &LashRuntime,
        persist_usage_to_store: bool,
        child_usage_event_relay: Option<ChildUsageEventRelay>,
        turn_lease: Option<crate::RuntimeTurnLease>,
    ) -> Result<Self, PluginActionInvokeError> {
        let Some(session) = runtime.session.as_ref() else {
            return Err(PluginActionInvokeError::Unknown(
                "session_manager".to_string(),
            ));
        };
        Ok(Self {
            current: CurrentSessionCapability::new(
                runtime,
                Arc::clone(session.plugins()),
                persist_usage_to_store,
                turn_lease,
            ),
            managed: ManagedSessionCapability::new(runtime),
            processes: ProcessCapability::new(runtime),
            usage: UsageCapability::new(runtime, persist_usage_to_store, child_usage_event_relay),
            direct: DirectCompletionCapability,
        })
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
