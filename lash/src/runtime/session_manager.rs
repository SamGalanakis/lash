use super::*;
use std::sync::atomic::AtomicBool;

mod api;
mod background;
mod current;
mod direct;
mod graph;
mod managed;
mod monitor;
mod prompt;
mod turns;
mod usage;

pub(in crate::runtime) use prompt::{HostPromptBridge, PendingPrompt};
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
    pub(super) cancel: CancellationToken,
    pub(super) task: tokio::task::JoinHandle<Result<AssembledTurn, crate::PluginError>>,
}

#[derive(Clone)]
struct CurrentSessionCapability {
    session_id: String,
    snapshot: CurrentSnapshot,
    policy: SessionPolicy,
    host: RuntimeHost,
    plugins: Arc<crate::PluginSession>,
    tool_catalog: Arc<Vec<serde_json::Value>>,
    store: Option<Arc<dyn crate::store::RuntimePersistence>>,
}

#[derive(Clone)]
struct ManagedSessionCapability {
    registry: Arc<Mutex<HashMap<String, Arc<Mutex<LashRuntime>>>>>,
    active_handoff_continuations: Arc<Mutex<HashMap<String, String>>>,
    turns: Arc<Mutex<HashMap<String, ManagedSessionTurn>>>,
    /// Maps child session_id → seed PluginMessage queued via
    /// `SessionCreateRequest::first_turn_input`. Drained by
    /// `take_first_turn_input` when a host claims it.
    pending_first_turn_inputs: Arc<std::sync::Mutex<HashMap<String, crate::PluginMessage>>>,
}

#[derive(Clone)]
struct UsageCapability {
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
struct BackgroundTaskCapability {
    runtime_scope_id: Arc<str>,
    sync_needed: Arc<AtomicBool>,
}

#[derive(Clone)]
struct PromptCapability {
    prompt_bridge: Option<HostPromptBridge>,
}

#[derive(Clone, Default)]
struct DirectCompletionCapability;

#[derive(Clone)]
pub(super) struct RuntimeSessionManager {
    current: CurrentSessionCapability,
    managed: ManagedSessionCapability,
    background: BackgroundTaskCapability,
    usage: UsageCapability,
    prompt: PromptCapability,
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
            head_revision: runtime.state.head_revision,
            graph_replace_required: runtime.state.graph_replace_required,
        }
    }

    fn new(
        runtime: &LashRuntime,
        plugins: Arc<crate::PluginSession>,
        persist_usage_to_store: bool,
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
            tool_catalog: runtime.active_tool_catalog_shared(),
            store: runtime.services.store.clone(),
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

impl BackgroundTaskCapability {
    fn new(runtime: &LashRuntime) -> Self {
        Self {
            runtime_scope_id: Arc::clone(&runtime.runtime_scope_id),
            sync_needed: Arc::clone(&runtime.background_sync_needed),
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

impl PromptCapability {
    fn new(prompt_bridge: Option<HostPromptBridge>) -> Self {
        Self { prompt_bridge }
    }
}

impl RuntimeSessionManager {
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
            current: CurrentSessionCapability::new(
                runtime,
                Arc::clone(session.plugins()),
                persist_usage_to_store,
            ),
            managed: ManagedSessionCapability::new(runtime),
            background: BackgroundTaskCapability::new(runtime),
            usage: UsageCapability::new(runtime, persist_usage_to_store, child_usage_event_relay),
            prompt: PromptCapability::new(prompt_bridge),
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
