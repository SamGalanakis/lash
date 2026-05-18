use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use futures_util::stream::{self, BoxStream, StreamExt};
use lash_trace::{JsonlTraceSink, TraceContext, TraceLevel, TraceSink};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, broadcast};

use crate::plugin::PluginError;

use super::{LocalRuntimeEffectHost, RuntimeEffectHost, SessionStoreFactory, TerminationPolicy};

/// Category of a registered background task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundTaskKind {
    Tool,
    SessionTurn,
    Monitor,
    Observer,
    External,
    Other,
}

impl BackgroundTaskKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackgroundTaskKind::Tool => "tool",
            BackgroundTaskKind::SessionTurn => "session_turn",
            BackgroundTaskKind::Monitor => "monitor",
            BackgroundTaskKind::Observer => "observer",
            BackgroundTaskKind::External => "external",
            BackgroundTaskKind::Other => "other",
        }
    }
}

pub type BackgroundTaskId = String;

/// Lifecycle state of a registered background task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundTaskState {
    Pending,
    Running,
    Waiting,
    Completed,
    Failed,
    CancelRequested,
    Cancelled,
}

impl BackgroundTaskState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Serializable task input understood by background-task hosts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackgroundTaskInput {
    ToolCall {
        call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    SessionTurn {
        create_request: Box<crate::SessionCreateRequest>,
        turn_input: Box<crate::TurnInput>,
        output_contract: crate::ToolOutputContract,
    },
    Monitor {
        spec: crate::MonitorSpec,
    },
    External {
        #[serde(default)]
        metadata: serde_json::Value,
    },
}

/// Serializable task spec used to start or recover a background task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackgroundTaskRegistration {
    pub id: BackgroundTaskId,
    pub kind: BackgroundTaskKind,
    pub producer: String,
    pub scope: BackgroundTaskScope,
    pub child_session_id: Option<String>,
    pub parent_task_id: Option<BackgroundTaskId>,
    pub input: BackgroundTaskInput,
    pub attempt: BackgroundTaskAttempt,
    pub cancel_policy: BackgroundCancelPolicy,
    pub close_policy: BackgroundClosePolicy,
}

impl BackgroundTaskRegistration {
    pub fn new(
        id: impl Into<BackgroundTaskId>,
        kind: BackgroundTaskKind,
        producer: impl Into<String>,
        scope: BackgroundTaskScope,
        input: BackgroundTaskInput,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            producer: producer.into(),
            scope,
            child_session_id: None,
            parent_task_id: None,
            input,
            attempt: BackgroundTaskAttempt::default(),
            cancel_policy: BackgroundCancelPolicy::Cooperative,
            close_policy: BackgroundClosePolicy::Keep,
        }
    }

    pub fn with_child_session_id(mut self, child_session_id: impl Into<String>) -> Self {
        self.child_session_id = Some(child_session_id.into());
        self
    }

    pub fn with_parent_task_id(mut self, parent_task_id: impl Into<BackgroundTaskId>) -> Self {
        self.parent_task_id = Some(parent_task_id.into());
        self
    }

    pub fn with_optional_parent_task_id(
        mut self,
        parent_task_id: Option<impl Into<BackgroundTaskId>>,
    ) -> Self {
        self.parent_task_id = parent_task_id.map(Into::into);
        self
    }

    pub fn with_attempt(mut self, attempt: BackgroundTaskAttempt) -> Self {
        self.attempt = attempt;
        self
    }

    pub fn with_cancel_policy(mut self, cancel_policy: BackgroundCancelPolicy) -> Self {
        self.cancel_policy = cancel_policy;
        self
    }

    pub fn with_close_policy(mut self, close_policy: BackgroundClosePolicy) -> Self {
        self.close_policy = close_policy;
        self
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackgroundTaskScope {
    pub session_id: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundCancelPolicy {
    #[default]
    Cooperative,
    AbortLocal,
    External,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundClosePolicy {
    #[default]
    Keep,
    Cancel,
    Transfer,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackgroundTaskAttempt {
    pub attempt: u32,
    pub max_attempts: Option<u32>,
    pub idempotency_key: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackgroundTaskOutcome {
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<crate::ToolCallOutput>,
}

/// Serializable host-owned background task record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackgroundTaskRecord {
    pub id: BackgroundTaskId,
    pub kind: BackgroundTaskKind,
    pub producer: String,
    pub scope: BackgroundTaskScope,
    pub parent_task_id: Option<BackgroundTaskId>,
    pub child_session_id: Option<String>,
    pub input: BackgroundTaskInput,
    pub state: BackgroundTaskState,
    pub cancel_policy: BackgroundCancelPolicy,
    pub close_policy: BackgroundClosePolicy,
    pub attempt: BackgroundTaskAttempt,
    pub result: Option<BackgroundTaskOutcome>,
    pub failure: Option<BackgroundTaskOutcome>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub completed_at: Option<SystemTime>,
}

impl BackgroundTaskRecord {
    pub fn local_session(
        session_id: impl Into<String>,
        id: impl Into<BackgroundTaskId>,
        kind: BackgroundTaskKind,
        producer: impl Into<String>,
        state: BackgroundTaskState,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            id: id.into(),
            kind,
            producer: producer.into(),
            scope: BackgroundTaskScope {
                session_id: session_id.into(),
            },
            parent_task_id: None,
            child_session_id: None,
            input: BackgroundTaskInput::External {
                metadata: serde_json::Value::Null,
            },
            state,
            cancel_policy: BackgroundCancelPolicy::Cooperative,
            close_policy: BackgroundClosePolicy::Keep,
            attempt: BackgroundTaskAttempt::default(),
            result: None,
            failure: None,
            created_at: now,
            updated_at: now,
            completed_at: state.is_terminal().then_some(now),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackgroundTaskEvent {
    Registered {
        record: BackgroundTaskRecord,
    },
    StateChanged {
        task_id: BackgroundTaskId,
        state: BackgroundTaskState,
    },
    Progress {
        task_id: BackgroundTaskId,
        message: String,
    },
    Completed {
        record: BackgroundTaskRecord,
    },
    Failed {
        record: BackgroundTaskRecord,
    },
    CancelRequested {
        task_id: BackgroundTaskId,
        reason: Option<String>,
    },
    Cancelled {
        record: BackgroundTaskRecord,
    },
    Transferred {
        task_id: BackgroundTaskId,
        scope: BackgroundTaskScope,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackgroundTaskFilter {
    pub session_id: Option<String>,
    pub kind: Option<BackgroundTaskKind>,
    pub include_terminal: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BackgroundTaskUpdate {
    pub state: Option<BackgroundTaskState>,
    pub progress: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackgroundTaskCompletion {
    pub state: BackgroundTaskState,
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<crate::ToolCallOutput>,
}

/// Durability-neutral background task registry.
#[async_trait::async_trait]
pub trait BackgroundTaskRegistry: Send + Sync {
    async fn register(
        &self,
        registration: BackgroundTaskRegistration,
    ) -> Result<BackgroundTaskRecord, PluginError>;

    async fn await_completion(
        &self,
        task_id: &str,
    ) -> Result<BackgroundTaskCompletion, PluginError>;

    async fn mark_running(&self, task_id: &str) -> Result<BackgroundTaskRecord, PluginError> {
        self.update(
            task_id,
            BackgroundTaskUpdate {
                state: Some(BackgroundTaskState::Running),
                progress: None,
            },
        )
        .await
    }

    async fn update(
        &self,
        task_id: &str,
        update: BackgroundTaskUpdate,
    ) -> Result<BackgroundTaskRecord, PluginError>;

    async fn complete(
        &self,
        task_id: &str,
        outcome: BackgroundTaskCompletion,
    ) -> Result<BackgroundTaskRecord, PluginError>;

    async fn request_cancel(
        &self,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError>;

    async fn get(&self, task_id: &str) -> Option<BackgroundTaskRecord>;

    async fn list(&self, filter: BackgroundTaskFilter) -> Vec<BackgroundTaskRecord>;

    async fn transfer(
        &self,
        task_id: &str,
        new_scope: BackgroundTaskScope,
    ) -> Result<BackgroundTaskRecord, PluginError>;

    fn subscribe(&self, filter: BackgroundTaskFilter) -> BoxStream<'static, BackgroundTaskEvent>;
}

/// In-memory background task registry shared across runtime sessions.
pub struct LocalBackgroundTaskRegistry {
    managed: Arc<Mutex<ManagedTaskMap>>,
    events: broadcast::Sender<BackgroundTaskEvent>,
}

impl Default for LocalBackgroundTaskRegistry {
    fn default() -> Self {
        let (events, _) = broadcast::channel(256);
        Self {
            managed: Arc::new(Mutex::new(HashMap::new())),
            events,
        }
    }
}

type ManagedTaskMap = HashMap<String, HashMap<String, ManagedTaskRecord>>;

struct ManagedTaskRecord {
    status: BackgroundTaskRecord,
    completion: Option<BackgroundTaskCompletion>,
    notify: Arc<tokio::sync::Notify>,
}

fn new_background_task_record(
    spec: &BackgroundTaskRegistration,
    state: BackgroundTaskState,
) -> BackgroundTaskRecord {
    let mut record = BackgroundTaskRecord::local_session(
        spec.scope.session_id.clone(),
        spec.id.clone(),
        spec.kind,
        spec.producer.clone(),
        state,
    );
    record.parent_task_id = spec.parent_task_id.clone();
    record.child_session_id = spec.child_session_id.clone();
    record.input = spec.input.clone();
    record.attempt = spec.attempt.clone();
    record.cancel_policy = spec.cancel_policy.clone();
    record.close_policy = spec.close_policy.clone();
    record
}

fn event_matches_filter(event: &BackgroundTaskEvent, filter: &BackgroundTaskFilter) -> bool {
    let record = match event {
        BackgroundTaskEvent::Registered { record }
        | BackgroundTaskEvent::Completed { record }
        | BackgroundTaskEvent::Failed { record }
        | BackgroundTaskEvent::Cancelled { record } => Some(record),
        _ => None,
    };
    if let Some(record) = record {
        if filter
            .session_id
            .as_ref()
            .is_some_and(|session_id| &record.scope.session_id != session_id)
        {
            return false;
        }
        if filter.kind.is_some_and(|kind| record.kind != kind) {
            return false;
        }
        return filter.include_terminal || !record.state.is_terminal();
    }
    true
}

fn record_matches_filter(record: &BackgroundTaskRecord, filter: &BackgroundTaskFilter) -> bool {
    if filter
        .session_id
        .as_ref()
        .is_some_and(|session_id| &record.scope.session_id != session_id)
    {
        return false;
    }
    if filter.kind.is_some_and(|kind| record.kind != kind) {
        return false;
    }
    filter.include_terminal || !record.state.is_terminal()
}

impl LocalBackgroundTaskRegistry {
    fn publish(&self, event: BackgroundTaskEvent) {
        let _ = self.events.send(event);
    }

    async fn insert_task(
        &self,
        registration: BackgroundTaskRegistration,
        state: BackgroundTaskState,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let mut managed = self.managed.lock().await;
        let tasks = managed
            .entry(registration.scope.session_id.clone())
            .or_default();
        if tasks
            .get(&registration.id)
            .is_some_and(|record| !record.status.state.is_terminal())
        {
            return Err(PluginError::Session(format!(
                "background task `{}` is already registered",
                registration.id
            )));
        }
        let record = new_background_task_record(&registration, state);
        tasks.insert(
            registration.id,
            ManagedTaskRecord {
                status: record.clone(),
                completion: None,
                notify: Arc::new(tokio::sync::Notify::new()),
            },
        );
        drop(managed);
        self.publish(BackgroundTaskEvent::Registered {
            record: record.clone(),
        });
        if state != BackgroundTaskState::Pending {
            self.publish(BackgroundTaskEvent::StateChanged {
                task_id: record.id.clone(),
                state,
            });
        }
        Ok(record)
    }
}

#[async_trait::async_trait]
impl BackgroundTaskRegistry for LocalBackgroundTaskRegistry {
    async fn register(
        &self,
        registration: BackgroundTaskRegistration,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        self.insert_task(registration, BackgroundTaskState::Pending)
            .await
    }

    async fn await_completion(
        &self,
        task_id: &str,
    ) -> Result<BackgroundTaskCompletion, PluginError> {
        loop {
            let notify = {
                let managed = self.managed.lock().await;
                let Some(record) = managed.values().find_map(|tasks| tasks.get(task_id)) else {
                    return Err(PluginError::Session(format!(
                        "unknown background task `{task_id}`"
                    )));
                };
                if let Some(completion) = record.completion.clone() {
                    return Ok(completion);
                }
                if record.status.state.is_terminal() {
                    return Ok(BackgroundTaskCompletion {
                        state: record.status.state,
                        summary: record
                            .status
                            .result
                            .as_ref()
                            .or(record.status.failure.as_ref())
                            .and_then(|outcome| outcome.summary.clone()),
                        output: record
                            .status
                            .result
                            .as_ref()
                            .or(record.status.failure.as_ref())
                            .and_then(|outcome| outcome.output.clone()),
                    });
                }
                Arc::clone(&record.notify)
            };
            notify.notified().await;
        }
    }

    async fn update(
        &self,
        task_id: &str,
        update: BackgroundTaskUpdate,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let mut managed = self.managed.lock().await;
        for tasks in managed.values_mut() {
            if let Some(record) = tasks.get_mut(task_id) {
                if let Some(state) = update.state {
                    record.status.state = state;
                    record.status.updated_at = SystemTime::now();
                }
                let status = record.status.clone();
                drop(managed);
                if let Some(message) = update.progress {
                    self.publish(BackgroundTaskEvent::Progress {
                        task_id: task_id.to_string(),
                        message,
                    });
                }
                self.publish(BackgroundTaskEvent::StateChanged {
                    task_id: task_id.to_string(),
                    state: status.state,
                });
                return Ok(status);
            }
        }
        Err(PluginError::Session(format!(
            "unknown background task `{task_id}`"
        )))
    }

    async fn complete(
        &self,
        task_id: &str,
        outcome: BackgroundTaskCompletion,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        if !outcome.state.is_terminal() {
            return Err(PluginError::Session(
                "background task completion must use a terminal state".to_string(),
            ));
        }
        let mut managed = self.managed.lock().await;
        for tasks in managed.values_mut() {
            if let Some(record) = tasks.get_mut(task_id) {
                if record.status.state.is_terminal() {
                    return Ok(record.status.clone());
                }
                record.status.state = outcome.state;
                record.status.updated_at = SystemTime::now();
                record.status.completed_at = Some(record.status.updated_at);
                let summary = BackgroundTaskOutcome {
                    summary: outcome.summary.clone(),
                    output: outcome.output.clone(),
                };
                if outcome.state == BackgroundTaskState::Failed {
                    record.status.failure = Some(summary);
                } else {
                    record.status.result = Some(summary);
                }
                record.completion = Some(outcome);
                record.notify.notify_waiters();
                let status = record.status.clone();
                drop(managed);
                self.publish(match status.state {
                    BackgroundTaskState::Failed => BackgroundTaskEvent::Failed {
                        record: status.clone(),
                    },
                    BackgroundTaskState::Cancelled => BackgroundTaskEvent::Cancelled {
                        record: status.clone(),
                    },
                    _ => BackgroundTaskEvent::Completed {
                        record: status.clone(),
                    },
                });
                return Ok(status);
            }
        }
        Err(PluginError::Session(format!(
            "unknown background task `{task_id}`"
        )))
    }

    async fn request_cancel(
        &self,
        task_id: &str,
        reason: Option<String>,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let status = self
            .update(
                task_id,
                BackgroundTaskUpdate {
                    state: Some(BackgroundTaskState::CancelRequested),
                    progress: None,
                },
            )
            .await?;
        self.publish(BackgroundTaskEvent::CancelRequested {
            task_id: task_id.to_string(),
            reason,
        });
        Ok(status)
    }

    async fn get(&self, task_id: &str) -> Option<BackgroundTaskRecord> {
        let managed = self.managed.lock().await;
        managed
            .values()
            .find_map(|tasks| tasks.get(task_id).map(|record| record.status.clone()))
    }

    async fn list(&self, filter: BackgroundTaskFilter) -> Vec<BackgroundTaskRecord> {
        let managed = self.managed.lock().await;
        let mut out = managed
            .values()
            .flat_map(|tasks| tasks.values())
            .map(|record| record.status.clone())
            .filter(|record| record_matches_filter(record, &filter))
            .collect::<Vec<_>>();
        out.sort_by_key(|record| record.created_at);
        out
    }

    async fn transfer(
        &self,
        task_id: &str,
        new_scope: BackgroundTaskScope,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let mut managed = self.managed.lock().await;
        let mut moved = None;
        for tasks in managed.values_mut() {
            if let Some(record) = tasks.remove(task_id) {
                moved = Some(record);
                break;
            }
        }
        let Some(mut record) = moved else {
            return Err(PluginError::Session(format!(
                "unknown background task `{task_id}`"
            )));
        };
        record.status.scope = new_scope.clone();
        record.status.updated_at = SystemTime::now();
        let status = record.status.clone();
        managed
            .entry(new_scope.session_id.clone())
            .or_default()
            .insert(task_id.to_string(), record);
        drop(managed);
        self.publish(BackgroundTaskEvent::Transferred {
            task_id: task_id.to_string(),
            scope: new_scope,
        });
        Ok(status)
    }

    fn subscribe(&self, filter: BackgroundTaskFilter) -> BoxStream<'static, BackgroundTaskEvent> {
        let rx = self.events.subscribe();
        stream::unfold((rx, filter), |(mut rx, filter)| async move {
            loop {
                match rx.recv().await {
                    Ok(event) if event_matches_filter(&event, &filter) => {
                        return Some((event, (rx, filter)));
                    }
                    Ok(_) => continue,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        })
        .boxed()
    }
}

/// Required host configuration for all runtimes.
#[derive(Clone)]
pub struct RuntimeCoreConfig {
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub prompt: crate::PromptLayer,
    pub trace_sink: Option<Arc<dyn TraceSink>>,
    pub trace_level: TraceLevel,
    pub trace_context: TraceContext,
    pub termination: TerminationPolicy,
    pub effect_host: Arc<dyn RuntimeEffectHost>,
}

impl Default for RuntimeCoreConfig {
    fn default() -> Self {
        Self {
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            prompt: crate::PromptLayer::new(),
            trace_sink: None,
            trace_level: TraceLevel::Standard,
            trace_context: TraceContext::default(),
            termination: TerminationPolicy::default(),
            effect_host: Arc::new(LocalRuntimeEffectHost::default()),
        }
    }
}

impl RuntimeCoreConfig {
    pub fn with_attachment_store(
        mut self,
        attachment_store: Arc<dyn crate::AttachmentStore>,
    ) -> Self {
        self.attachment_store = attachment_store;
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: crate::PromptTemplate) -> Self {
        self.prompt.template = Some(prompt_template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.prompt.add_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.prompt.replace_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.prompt.clear_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.prompt = prompt;
        self
    }

    pub fn with_trace_jsonl_path(mut self, trace_path: Option<PathBuf>) -> Self {
        self.trace_sink =
            trace_path.map(|path| Arc::new(JsonlTraceSink::new(path)) as Arc<dyn TraceSink>);
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.trace_sink = sink;
        self
    }

    pub fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.trace_level = level;
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.trace_context = context;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.termination = termination;
        self
    }

    pub fn with_effect_host(mut self, effect_host: Arc<dyn RuntimeEffectHost>) -> Self {
        self.effect_host = effect_host;
        self
    }
}

/// Base host shape for embedded runtimes.
#[derive(Clone)]
pub struct EmbeddedRuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
}

impl EmbeddedRuntimeHost {
    pub fn new(core: RuntimeCoreConfig) -> Self {
        Self {
            core,
            session_store_factory: None,
        }
    }

    pub fn with_session_store_factory(
        mut self,
        session_store_factory: Arc<dyn SessionStoreFactory>,
    ) -> Self {
        self.session_store_factory = Some(session_store_factory);
        self
    }
}

/// Host shape for runtimes that support background plugin work.
#[derive(Clone)]
pub struct BackgroundRuntimeHost {
    pub embedded: EmbeddedRuntimeHost,
    pub background_task_registry: Arc<dyn BackgroundTaskRegistry>,
}

impl BackgroundRuntimeHost {
    pub fn new(
        embedded: EmbeddedRuntimeHost,
        background_task_registry: Arc<dyn BackgroundTaskRegistry>,
    ) -> Self {
        Self {
            embedded,
            background_task_registry,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub background_task_registry: Option<Arc<dyn BackgroundTaskRegistry>>,
}

impl From<EmbeddedRuntimeHost> for RuntimeHost {
    fn from(value: EmbeddedRuntimeHost) -> Self {
        Self {
            core: value.core,
            session_store_factory: value.session_store_factory,
            background_task_registry: None,
        }
    }
}

impl From<BackgroundRuntimeHost> for RuntimeHost {
    fn from(value: BackgroundRuntimeHost) -> Self {
        Self {
            core: value.embedded.core,
            session_store_factory: value.embedded.session_store_factory,
            background_task_registry: Some(value.background_task_registry),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackgroundTaskExecutor;

    fn spec(id: &str, kind: BackgroundTaskKind) -> BackgroundTaskRegistration {
        BackgroundTaskRegistration {
            id: id.to_string(),
            kind,
            producer: "test".to_string(),
            scope: BackgroundTaskScope {
                session_id: "s1".to_string(),
            },
            child_session_id: None,
            parent_task_id: None,
            input: BackgroundTaskInput::External {
                metadata: serde_json::Value::Null,
            },
            attempt: BackgroundTaskAttempt::default(),
            cancel_policy: BackgroundCancelPolicy::Cooperative,
            close_policy: BackgroundClosePolicy::Keep,
        }
    }

    #[tokio::test]
    async fn background_task_registry_register_update_await_complete_and_filter() {
        let registry = LocalBackgroundTaskRegistry::default();
        let registered = registry
            .register(BackgroundTaskRegistration {
                child_session_id: Some("child".to_string()),
                cancel_policy: BackgroundCancelPolicy::External,
                close_policy: BackgroundClosePolicy::Transfer,
                attempt: BackgroundTaskAttempt {
                    attempt: 1,
                    max_attempts: Some(3),
                    idempotency_key: Some("idem".to_string()),
                },
                ..spec("task:one", BackgroundTaskKind::Other)
            })
            .await
            .expect("register");

        assert_eq!(registered.state, BackgroundTaskState::Pending);
        assert_eq!(registered.child_session_id.as_deref(), Some("child"));

        let updated = registry
            .mark_running("task:one")
            .await
            .expect("mark running");
        assert_eq!(updated.state, BackgroundTaskState::Running);

        assert_eq!(
            registry
                .list(BackgroundTaskFilter {
                    session_id: Some("s1".to_string()),
                    kind: Some(BackgroundTaskKind::Other),
                    include_terminal: false,
                })
                .await
                .len(),
            1
        );

        let output = crate::ToolCallOutput::success(serde_json::json!({"ok": true}));
        let completed = registry
            .complete(
                "task:one",
                BackgroundTaskCompletion {
                    state: BackgroundTaskState::Completed,
                    summary: Some("done".to_string()),
                    output: Some(output.clone()),
                },
            )
            .await
            .expect("complete");
        assert_eq!(completed.state, BackgroundTaskState::Completed);
        assert_eq!(
            completed
                .result
                .as_ref()
                .and_then(|outcome| outcome.summary.as_deref()),
            Some("done")
        );
        assert!(completed.completed_at.is_some());
        let awaited = registry.await_completion("task:one").await.expect("await");
        assert_eq!(awaited.state, BackgroundTaskState::Completed);
        assert_eq!(awaited.output, Some(output));
        assert!(
            registry
                .list(BackgroundTaskFilter {
                    session_id: Some("s1".to_string()),
                    kind: None,
                    include_terminal: false,
                })
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn background_task_registry_request_cancel_is_not_terminal() {
        let registry = LocalBackgroundTaskRegistry::default();
        registry
            .register(spec("task:cancel", BackgroundTaskKind::Tool))
            .await
            .expect("register");
        registry
            .mark_running("task:cancel")
            .await
            .expect("mark running");

        let requested = registry
            .request_cancel("task:cancel", Some("stop".to_string()))
            .await
            .expect("request cancel");

        assert_eq!(requested.state, BackgroundTaskState::CancelRequested);
        assert_eq!(
            registry.get("task:cancel").await.expect("task").state,
            BackgroundTaskState::CancelRequested
        );
    }

    #[tokio::test]
    async fn background_task_transfer_moves_live_task_visibility() {
        let registry = LocalBackgroundTaskRegistry::default();
        registry
            .register(spec("monitor:one", BackgroundTaskKind::Monitor))
            .await
            .expect("register");
        registry
            .mark_running("monitor:one")
            .await
            .expect("mark running");

        registry
            .transfer(
                "monitor:one",
                BackgroundTaskScope {
                    session_id: "s2".to_string(),
                },
            )
            .await
            .expect("transfer");

        assert!(
            registry
                .list(BackgroundTaskFilter {
                    session_id: Some("s1".to_string()),
                    kind: None,
                    include_terminal: true,
                })
                .await
                .is_empty()
        );
        let tasks = registry
            .list(BackgroundTaskFilter {
                session_id: Some("s2".to_string()),
                kind: None,
                include_terminal: true,
            })
            .await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "monitor:one");
    }

    #[tokio::test]
    async fn local_effect_host_runs_background_task_and_completes_registry() {
        let registry = Arc::new(LocalBackgroundTaskRegistry::default());
        let effect_host = LocalRuntimeEffectHost::default();
        let registration = spec("task:one", BackgroundTaskKind::Tool);
        registry
            .register(registration.clone())
            .await
            .expect("register");

        effect_host
            .start_background_task(
                registry.clone(),
                registration,
                BackgroundTaskExecutor::new(|_| async {
                    BackgroundTaskCompletion {
                        state: BackgroundTaskState::Completed,
                        summary: Some("done".to_string()),
                        output: Some(crate::ToolCallOutput::success(
                            serde_json::json!({"ok": true}),
                        )),
                    }
                }),
            )
            .await
            .expect("start");

        let completion = registry
            .await_completion("task:one")
            .await
            .expect("completion");
        assert_eq!(completion.state, BackgroundTaskState::Completed);
    }

    #[tokio::test]
    async fn local_effect_host_cancel_requests_reach_background_executor() {
        let registry = Arc::new(LocalBackgroundTaskRegistry::default());
        let effect_host = LocalRuntimeEffectHost::default();
        let registration = spec("task:cancel", BackgroundTaskKind::Tool);
        registry
            .register(registration.clone())
            .await
            .expect("register");

        effect_host
            .start_background_task(
                registry.clone(),
                registration,
                BackgroundTaskExecutor::new(|cancellation| async move {
                    cancellation.cancelled().await;
                    BackgroundTaskCompletion {
                        state: BackgroundTaskState::Cancelled,
                        summary: Some("cancelled".to_string()),
                        output: Some(crate::ToolCallOutput::cancelled(
                            crate::ToolCancellation::runtime("cancelled"),
                        )),
                    }
                }),
            )
            .await
            .expect("start");
        effect_host
            .request_background_task_cancel(registry.clone(), "task:cancel", Some("stop".into()))
            .await
            .expect("request cancel");

        let completion = registry
            .await_completion("task:cancel")
            .await
            .expect("completion");
        assert_eq!(completion.state, BackgroundTaskState::Cancelled);
    }

    struct FakeDurableEffectHost;

    #[async_trait::async_trait]
    impl RuntimeEffectHost for FakeDurableEffectHost {
        async fn start_background_task(
            &self,
            registry: Arc<dyn BackgroundTaskRegistry>,
            registration: BackgroundTaskRegistration,
            _executor: BackgroundTaskExecutor,
        ) -> Result<BackgroundTaskRecord, PluginError> {
            registry.mark_running(&registration.id).await
        }
    }

    #[tokio::test]
    async fn durable_effect_host_can_schedule_without_tokio_executor() {
        let registry = Arc::new(LocalBackgroundTaskRegistry::default());
        let effect_host = FakeDurableEffectHost;
        let registration = spec("durable", BackgroundTaskKind::External);
        registry
            .register(registration.clone())
            .await
            .expect("register");

        let scheduled = effect_host
            .start_background_task(
                registry.clone(),
                registration,
                BackgroundTaskExecutor::new(|_| async {
                    panic!("durable host should not run the local executor")
                }),
            )
            .await
            .expect("schedule");
        assert_eq!(scheduled.state, BackgroundTaskState::Running);

        let output = crate::ToolCallOutput::success(serde_json::json!({"cached": true}));
        registry
            .complete(
                "durable",
                BackgroundTaskCompletion {
                    state: BackgroundTaskState::Completed,
                    summary: Some("from durable store".to_string()),
                    output: Some(output.clone()),
                },
            )
            .await
            .expect("complete");

        let completion = registry.await_completion("durable").await.expect("await");
        assert_eq!(completion.state, BackgroundTaskState::Completed);
        assert_eq!(completion.output, Some(output));
    }
}
