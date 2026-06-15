#[derive(Clone)]
struct AppState {
    core: LashCore,
    process_observer: lash::process::ProcessWorkObserver,
    session_ids: WorkbenchSessionIds,
    messages: Arc<Mutex<Vec<ChatMessage>>>,
    timeline: Arc<Mutex<Vec<StreamItem>>>,
    selected_model: Arc<Mutex<ModelSelection>>,
    web_configured: bool,
    trace_sink: Option<Arc<dyn TraceSink>>,
    lashlang_execution: Arc<TraceLashlangGraphStore>,
    event_tx: broadcast::Sender<StreamItem>,
    queued_work_poke: lash::runtime::QueuedWorkPoke,
    restate_ingress_url: String,
    restate_http: reqwest::Client,
    restate_cron_job_keys: Arc<Mutex<BTreeSet<String>>>,
    mail_world: mail::MailWorld,
    /// Session handles of in-flight user turns, keyed by turn id. The Restate
    /// turn workflow runs in this same process, so /api/turn/cancel calls
    /// [`lash::LashSession::cancel_running_turns`] on the live handle.
    turn_cancels: Arc<Mutex<BTreeMap<String, lash::LashSession>>>,
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
            model_variant: model.variant.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct StateSnapshot {
    settings: Settings,
    messages: Vec<ChatMessage>,
    timeline: Vec<StreamItem>,
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
    Event { event: Box<TurnActivity> },
    Message { message: ChatMessage },
    Error { message: String },
    Done,
}

impl StreamItem {
    fn event(event: TurnActivity) -> Self {
        Self::Event {
            event: Box::new(event),
        }
    }
}

struct TurnCancelGuard {
    registry: Arc<Mutex<BTreeMap<String, lash::LashSession>>>,
    turn_id: String,
}

impl Drop for TurnCancelGuard {
    fn drop(&mut self) {
        self.registry
            .lock()
            .expect("turn cancel lock")
            .remove(&self.turn_id);
    }
}

#[derive(Debug, Serialize)]
struct CommandAccepted {
    accepted: bool,
}

#[derive(Clone)]
struct WorkbenchQueuedWorkSubmitter {
    session_ids: WorkbenchSessionIds,
    store_factory: Arc<dyn lash::persistence::SessionStoreFactory>,
    restate_ingress_url: String,
    restate_http: reqwest::Client,
}

#[async_trait]
impl lash::runtime::QueuedWorkRunHandle for WorkbenchQueuedWorkSubmitter {
    async fn run_queued_work(
        &self,
        request: lash::runtime::QueuedWorkRunRequest,
    ) -> std::result::Result<lash::runtime::QueuedWorkRunOutcome, PluginError> {
        let session_id = request
            .session_id
            .unwrap_or_else(|| self.session_ids.current());
        if !self.has_queued_work(&session_id).await? {
            return Ok(lash::runtime::QueuedWorkRunOutcome::Idle);
        }
        let workflow_request = restate::WorkbenchQueuedTurnWorkflowRequest {
            turn_id: format!("workbench-queued-{}", uuid::Uuid::new_v4()),
            session_id: session_id.clone(),
            reason: request.reason,
        };
        restate::submit_queued_turn_request(
            &self.restate_http,
            &self.restate_ingress_url,
            &workflow_request,
        )
        .await
        .map_err(|err| PluginError::Session(err.to_string()))?;
        Ok(lash::runtime::QueuedWorkRunOutcome::Submitted { session_id })
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
    ) -> std::result::Result<lash::runtime::QueuedWorkRunOutcome, PluginError> {
        Ok(lash::runtime::QueuedWorkRunOutcome::Idle)
    }
}

#[cfg(test)]
fn inert_queued_work_poke() -> lash::runtime::QueuedWorkPoke {
    lash::runtime::QueuedWorkRunner::new(Arc::new(NoopQueuedWorkRunHandle)).poke_handle()
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
