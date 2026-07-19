#[derive(Clone)]
struct AppState {
    core: LashCore,
    process_observer: lash::process::ProcessWorkObserver,
    process_work_driver: lash::process::ProcessWorkDriver,
    session_ids: WorkbenchSessionIds,
    messages: Arc<Mutex<Vec<ChatMessage>>>,
    selected_model: Arc<Mutex<ModelSelection>>,
    web_configured: bool,
    trace_sink: Option<Arc<dyn TraceSink>>,
    lashlang_execution: Arc<TraceLashlangGraphStore>,
    event_tx: broadcast::Sender<StreamItem>,
    queued_work_driver: lash::runtime::QueuedWorkDriver,
    restate_ingress_url: String,
    #[cfg_attr(not(test), allow(dead_code))]
    restate_admin_url: String,
    restate_http: reqwest::Client,
    restate_cron_job_keys: Arc<Mutex<BTreeSet<String>>>,
    mail_world: mail::MailWorld,
    active_turns: ActiveTurns,
}

#[derive(Clone, Debug, Serialize)]
struct Settings {
    model: String,
    model_variant: Option<String>,
    web_configured: bool,
    model_variants: Vec<&'static str>,
    session_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelSelection {
    model: String,
    model_variant: Option<String>,
}

impl ModelSelection {
    fn from_spec(model: &lash::ModelSpec) -> Self {
        Self {
            model: model.id.clone(),
            model_variant: model.variant.effort().map(str::to_string),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct StateSnapshot {
    settings: Settings,
    messages: Vec<ChatMessage>,
    active_turns: Vec<lash::TurnAddress>,
}

#[derive(Clone, Debug, Serialize)]
struct ChatMessage {
    id: String,
    role: String,
    text: String,
    at: String,
}

#[derive(Debug, Deserialize)]
struct TurnRequest {
    text: String,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EventsQuery {
    cursor: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(crate) enum ButtonChoice {
    Red,
    Blue,
}

impl ButtonChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Red => "Red",
            Self::Blue => "Blue",
        }
    }

    fn lower(self) -> &'static str {
        match self {
            Self::Red => "red",
            Self::Blue => "blue",
        }
    }
}

#[derive(Debug, Deserialize)]
struct ButtonEventRequest {
    button: ButtonChoice,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddAccountRequest {
    name: String,
}

#[derive(Debug, Deserialize)]
struct InjectMessageRequest {
    title: Option<String>,
    text: Option<String>,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamItem {
    Observation {
        event: Box<RemoteSessionObservationEvent>,
    },
    ReplayCursor {
        cursor: String,
    },
    ReplayGap {
        observation: Box<RemoteSessionObservation>,
        gap: Box<RemoteLiveReplayGap>,
    },
    Message { message: ChatMessage },
    Error { message: String },
    Done,
}

#[derive(Clone, Default)]
struct ActiveTurns {
    inner: Arc<Mutex<BTreeSet<(String, String)>>>,
    path: Option<Arc<PathBuf>>,
}

impl ActiveTurns {
    fn persistent(path: PathBuf) -> AnyhowResult<Self> {
        let turns = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("decode active turns `{}`", path.display()))?,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => BTreeSet::new(),
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("read active turns `{}`", path.display()));
            }
        };
        let active = Self {
            inner: Arc::new(Mutex::new(turns)),
            path: Some(Arc::new(path)),
        };
        active.persist();
        Ok(active)
    }

    fn insert(&self, session_id: impl Into<String>, turn_id: impl Into<String>) {
        let mut active = self.inner.lock().expect("active turn lock");
        active.insert((session_id.into(), turn_id.into()));
        self.persist_snapshot(&active);
    }

    fn remove(&self, session_id: &str, turn_id: &str) {
        let mut active = self.inner.lock().expect("active turn lock");
        active.remove(&(session_id.to_string(), turn_id.to_string()));
        self.persist_snapshot(&active);
    }

    fn guard(&self, session_id: &str, turn_id: &str) -> ActiveTurnGuard {
        ActiveTurnGuard {
            active: self.clone(),
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
        }
    }

    fn for_session(&self, session_id: &str) -> Vec<lash::TurnAddress> {
        self.inner
            .lock()
            .expect("active turn lock")
            .iter()
            .filter(|(active_session_id, _)| active_session_id == session_id)
            .map(|(session_id, turn_id)| lash::TurnAddress::new(session_id, turn_id))
            .collect()
    }

    fn persist(&self) {
        let active = self.inner.lock().expect("active turn lock");
        self.persist_snapshot(&active);
    }

    fn persist_snapshot(&self, active: &BTreeSet<(String, String)>) {
        let Some(path) = self.path.as_deref() else {
            return;
        };
        let bytes = serde_json::to_vec(active).expect("serialize active turns");
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, bytes)
            .unwrap_or_else(|err| panic!("write active turns `{}`: {err}", temporary.display()));
        std::fs::rename(&temporary, path).unwrap_or_else(|err| {
            panic!(
                "replace active turns `{}` from `{}`: {err}",
                path.display(),
                temporary.display()
            )
        });
    }
}

struct ActiveTurnGuard {
    active: ActiveTurns,
    session_id: String,
    turn_id: String,
}

impl Drop for ActiveTurnGuard {
    fn drop(&mut self) {
        self.active.remove(&self.session_id, &self.turn_id);
    }
}

#[derive(Debug, Serialize)]
struct CommandAccepted {
    accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
struct TurnCancelReceipt {
    address: lash::TurnAddress,
    outcome: lash::TurnCancelOutcome,
    terminal: Option<lash::TurnTerminal>,
}

#[derive(Clone, Debug, Serialize)]
struct TurnCancelResponse {
    accepted: bool,
    cancellations: Vec<TurnCancelReceipt>,
}

/// Best-effort [`ProcessEventSink`](lash::process::ProcessEventSink) that hands
/// each appended process event to a channel (ADR 0017). `emit` runs inline on
/// the registry append path, so it must return fast: it does no I/O, only a
/// non-blocking `try_send`. Dropping on a full channel is intentional — the
/// durable event log (`events_after`) is the reconcile source, not this feed.
#[derive(Clone)]
struct ChannelProcessEventSink {
    tx: mpsc::Sender<lash::process::ProcessEvent>,
}

impl ChannelProcessEventSink {
    fn new(tx: mpsc::Sender<lash::process::ProcessEvent>) -> Self {
        Self { tx }
    }
}

#[async_trait]
impl lash::process::ProcessEventSink for ChannelProcessEventSink {
    async fn emit(&self, event: &lash::process::ProcessEvent) {
        // Non-blocking: drop on a full channel rather than slow every append.
        let _ = self.tx.try_send(event.clone());
    }
}

#[derive(Clone)]
struct WorkbenchQueuedWorkSubmitter {
    session_ids: WorkbenchSessionIds,
    store_factory: Arc<dyn lash::persistence::SessionStoreFactory>,
    restate_ingress_url: String,
    restate_http: reqwest::Client,
    active_turns: ActiveTurns,
}

#[async_trait]
impl lash::runtime::QueuedWorkRunHandle for WorkbenchQueuedWorkSubmitter {
    async fn run_queued_work(
        &self,
        request: lash::runtime::QueuedWorkRunRequest,
    ) -> std::result::Result<(), PluginError> {
        let session_id = request
            .session_id
            .unwrap_or_else(|| self.session_ids.current());
        if !self.has_queued_work(&session_id).await? {
            return Ok(());
        }
        let workflow_request = restate::WorkbenchQueuedTurnWorkflowRequest {
            turn_id: format!("workbench-queued-{}", uuid::Uuid::new_v4()),
            session_id: session_id.clone(),
            reason: request.reason,
        };
        self.active_turns
            .insert(&session_id, &workflow_request.turn_id);
        if let Err(err) = restate::submit_queued_turn_request(
            &self.restate_http,
            &self.restate_ingress_url,
            &workflow_request,
        )
        .await
        {
            self.active_turns
                .remove(&session_id, &workflow_request.turn_id);
            return Err(PluginError::Session(err.to_string()));
        }
        Ok(())
    }
}

impl WorkbenchQueuedWorkSubmitter {
    async fn has_queued_work(&self, session_id: &str) -> std::result::Result<bool, PluginError> {
        let store = self
            .store_factory
            .create_store(&lash::persistence::SessionStoreCreateRequest {
                session_id: session_id.to_string(),
                relation: lash::persistence::SessionRelation::default(),
                policy: lash::runtime::SessionPolicy::default(),
            })
            .await
            .map_err(PluginError::Session)?;
        let queued = store
            .list_queued_work(session_id)
            .await
            .map_err(|err| PluginError::Session(err.to_string()))?;
        Ok(!queued.is_empty())
    }
}

#[cfg(test)]
struct NoopQueuedWorkRunHandle;

#[cfg(test)]
#[async_trait]
impl lash::runtime::QueuedWorkRunHandle for NoopQueuedWorkRunHandle {
    async fn run_queued_work(
        &self,
        _request: lash::runtime::QueuedWorkRunRequest,
    ) -> std::result::Result<(), PluginError> {
        Ok(())
    }
}

#[cfg(test)]
fn inert_queued_work_driver() -> lash::runtime::QueuedWorkDriver {
    lash::runtime::QueuedWorkDriver::new(Arc::new(NoopQueuedWorkRunHandle))
}

#[cfg(test)]
struct NoopProcessRunHandle;

#[cfg(test)]
#[async_trait]
impl lash::process::ProcessRunHandle for NoopProcessRunHandle {
    async fn claim_and_run_pending(&self) -> std::result::Result<(), PluginError> {
        Ok(())
    }
}

/// A driver that reads the registry directly (no external run handle) — enough
/// for tests that build state but do not drive process execution.
#[cfg(test)]
fn inert_process_work_driver(
    registry: Arc<dyn lash::process::ProcessRegistry>,
) -> lash::process::ProcessWorkDriver {
    lash::process::ProcessWorkDriver::new(registry, Arc::new(NoopProcessRunHandle))
}

#[derive(Debug, Serialize)]
struct WorkItem {
    process: WorkProcess,
    descriptor: lash::process::ProcessHandleDescriptor,
    events: Vec<WorkEvent>,
    kind: String,
    label: String,
}

#[derive(Debug, Serialize)]
struct WorkProcess {
    process_id: String,
    graph_key: String,
    lifecycle: lash::process::ProcessLifecycleStatus,
    status_label: String,
    terminal: bool,
    error: Option<String>,
    created_at_ms: u64,
    updated_at_ms: u64,
    input: Value,
    external_ref: Option<Value>,
    child_session_id: Option<String>,
    label: String,
}

#[derive(Debug, Serialize)]
struct WorkEvent {
    sequence: u64,
    event_type: String,
    occurred_at_ms: u64,
    payload: Value,
}

#[derive(Debug, Serialize)]
struct WorkAwaitResult {
    process_id: String,
    outcome: lash::process::ProcessAwaitOutput,
    /// Reconciled from the durable log at terminal (ADR 0017): the authoritative,
    /// complete record, unlike the best-effort event sink.
    events: Vec<WorkAwaitEvent>,
}

#[derive(Debug, Serialize)]
struct WorkAwaitEvent {
    sequence: u64,
    event_type: String,
}
