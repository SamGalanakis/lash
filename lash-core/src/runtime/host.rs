use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use futures_util::stream::{self, BoxStream, StreamExt};
use lash_trace::{JsonlTraceSink, TraceContext, TraceLevel, TraceSink};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, broadcast};

use crate::plugin::PluginError;

use super::{SessionStoreFactory, TerminationPolicy};

/// Category of a registered background task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundTaskKind {
    Monitor,
    Subagent,
    Observer,
    Other,
}

impl BackgroundTaskKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackgroundTaskKind::Monitor => "monitor",
            BackgroundTaskKind::Subagent => "subagent",
            BackgroundTaskKind::Observer => "observer",
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

/// Metadata required to register a local background task.
#[derive(Clone, Debug)]
pub struct BackgroundTaskRegistration {
    pub id: BackgroundTaskId,
    pub kind: BackgroundTaskKind,
    pub producer: &'static str,
    pub child_session_id: Option<String>,
    pub parent_task_id: Option<BackgroundTaskId>,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackgroundTaskRegisterRequest {
    pub scope: BackgroundTaskScope,
    pub id: BackgroundTaskId,
    pub kind: BackgroundTaskKind,
    pub producer: String,
    pub parent_task_id: Option<BackgroundTaskId>,
    pub child_session_id: Option<String>,
    pub cancel_policy: BackgroundCancelPolicy,
    pub close_policy: BackgroundClosePolicy,
    pub attempt: BackgroundTaskAttempt,
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
}

/// Local-only cancellation hook for in-process tasks such as subagent trees.
pub type LocalBackgroundTaskCancel =
    Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;

/// Host-owned background task lifecycle policy.
#[async_trait::async_trait]
pub trait BackgroundTaskHost: Send + Sync {
    async fn register(
        &self,
        request: BackgroundTaskRegisterRequest,
    ) -> Result<BackgroundTaskRecord, PluginError>;

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

    async fn spawn_hidden(
        &self,
        session_id: &str,
        label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError>;

    async fn await_hidden(&self, session_id: &str) -> Result<(), PluginError>;

    /// Spawn a tokio-backed background task and register it under `spec.id`.
    /// Local host owns the Tokio future and can abort it via `cancel_managed`.
    async fn spawn_managed(
        &self,
        session_id: &str,
        spec: BackgroundTaskRegistration,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError>;

    /// Register an externally-owned task (e.g. a subagent session) so it
    /// shows up in `list_managed`. The local host does not hold a `JoinHandle`;
    /// if `cancel` is supplied it will be invoked when the registry receives
    /// a cancel request, otherwise cancellation only updates bookkeeping.
    async fn register_external(
        &self,
        session_id: &str,
        spec: BackgroundTaskRegistration,
        cancel: Option<LocalBackgroundTaskCancel>,
    ) -> Result<(), PluginError>;

    async fn unregister_external(&self, session_id: &str, task_id: &str);

    /// Update the state of a registered task to a terminal value.
    /// No-op if the task is unknown or already terminal.
    async fn mark_terminal(&self, session_id: &str, task_id: &str, state: BackgroundTaskState);

    /// Transition a still-live task between the non-terminal `Running`
    /// and `Waiting` states. Used by subagents to reflect whether the
    /// session is actively working on a task or waiting for a
    /// follow-up. No-op if the task is unknown or already terminal.
    async fn mark_live_state(&self, session_id: &str, task_id: &str, state: BackgroundTaskState);

    /// Cancel a tokio-backed task by id. No-op for externally-owned entries;
    /// callers should also mark those terminal via `mark_terminal`.
    async fn cancel_managed(&self, session_id: &str, task_id: &str) -> Result<(), PluginError>;

    /// Read-only snapshot of every registered task for the session.
    async fn list_managed(&self, session_id: &str) -> Vec<BackgroundTaskRecord>;

    /// Look up a single task's metadata.
    async fn get_managed(&self, session_id: &str, task_id: &str) -> Option<BackgroundTaskRecord>;

    /// Move still-registered tasks from one session scope to another.
    async fn transfer_managed(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        task_ids: &[String],
    ) -> Result<(), PluginError>;

    /// Cancel every live task in a session scope and return updated snapshots.
    async fn cancel_all_managed(
        &self,
        session_id: &str,
    ) -> Result<Vec<BackgroundTaskRecord>, PluginError>;
}

/// Tokio-backed background task host shared across runtime sessions.
pub struct LocalBackgroundTaskHost {
    hidden: Mutex<HiddenTaskMap>,
    managed: Arc<Mutex<ManagedTaskMap>>,
    events: broadcast::Sender<BackgroundTaskEvent>,
}

impl Default for LocalBackgroundTaskHost {
    fn default() -> Self {
        let (events, _) = broadcast::channel(256);
        Self {
            hidden: Mutex::new(HashMap::new()),
            managed: Arc::new(Mutex::new(HashMap::new())),
            events,
        }
    }
}

type SessionTaskHandle = tokio::task::JoinHandle<Result<(), PluginError>>;
type HiddenTaskMap = HashMap<String, Vec<SessionTaskHandle>>;
type ManagedTaskMap = HashMap<String, HashMap<String, ManagedTaskRecord>>;

struct ManagedTaskRecord {
    status: BackgroundTaskRecord,
    handle: Option<SessionTaskHandle>,
    cancel: Option<LocalBackgroundTaskCancel>,
}

fn new_background_task_record(
    scope_session_id: &str,
    spec: &BackgroundTaskRegistration,
    state: BackgroundTaskState,
) -> BackgroundTaskRecord {
    let mut record = BackgroundTaskRecord::local_session(
        scope_session_id,
        spec.id.clone(),
        spec.kind,
        spec.producer,
        state,
    );
    record.parent_task_id = spec.parent_task_id.clone();
    record.child_session_id = spec.child_session_id.clone();
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

impl LocalBackgroundTaskHost {
    fn publish(&self, event: BackgroundTaskEvent) {
        let _ = self.events.send(event);
    }
}

#[async_trait::async_trait]
impl BackgroundTaskHost for LocalBackgroundTaskHost {
    async fn register(
        &self,
        request: BackgroundTaskRegisterRequest,
    ) -> Result<BackgroundTaskRecord, PluginError> {
        let mut managed = self.managed.lock().await;
        let tasks = managed.entry(request.scope.session_id.clone()).or_default();
        if tasks
            .get(&request.id)
            .is_some_and(|record| !record.status.state.is_terminal())
        {
            return Err(PluginError::Session(format!(
                "background task `{}` is already registered",
                request.id
            )));
        }
        let now = SystemTime::now();
        let record = BackgroundTaskRecord {
            id: request.id.clone(),
            kind: request.kind,
            producer: request.producer,
            scope: request.scope,
            parent_task_id: request.parent_task_id,
            child_session_id: request.child_session_id,
            state: BackgroundTaskState::Pending,
            cancel_policy: request.cancel_policy,
            close_policy: request.close_policy,
            attempt: request.attempt,
            result: None,
            failure: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
        };
        tasks.insert(
            request.id,
            ManagedTaskRecord {
                status: record.clone(),
                handle: None,
                cancel: None,
            },
        );
        drop(managed);
        self.publish(BackgroundTaskEvent::Registered {
            record: record.clone(),
        });
        Ok(record)
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
                record.status.state = outcome.state;
                record.status.updated_at = SystemTime::now();
                record.status.completed_at = Some(record.status.updated_at);
                let summary = BackgroundTaskOutcome {
                    summary: outcome.summary,
                };
                if outcome.state == BackgroundTaskState::Failed {
                    record.status.failure = Some(summary);
                } else {
                    record.status.result = Some(summary);
                }
                record.handle = None;
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

    async fn spawn_hidden(
        &self,
        session_id: &str,
        _label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError> {
        let handle = tokio::spawn(task);
        self.hidden
            .lock()
            .await
            .entry(session_id.to_string())
            .or_default()
            .push(handle);
        Ok(())
    }

    async fn await_hidden(&self, session_id: &str) -> Result<(), PluginError> {
        loop {
            let tasks = self
                .hidden
                .lock()
                .await
                .remove(session_id)
                .unwrap_or_default();
            if tasks.is_empty() {
                return Ok(());
            }
            for task in tasks {
                match task.await {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => return Err(err),
                    Err(err) => {
                        return Err(PluginError::Session(format!(
                            "hidden background task failed: {err}"
                        )));
                    }
                }
            }
        }
    }

    async fn spawn_managed(
        &self,
        session_id: &str,
        spec: BackgroundTaskRegistration,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError> {
        let mut managed = self.managed.lock().await;
        let tasks = managed.entry(session_id.to_string()).or_default();
        if tasks
            .get(&spec.id)
            .is_some_and(|record| !record.status.state.is_terminal())
        {
            return Err(PluginError::Session(format!(
                "managed session task `{}` is already running",
                spec.id
            )));
        }
        let records = Arc::clone(&self.managed);
        let session_key = session_id.to_string();
        let task_id = spec.id.clone();
        let handle = tokio::spawn(async move {
            let result = task.await;
            let terminal = match &result {
                Ok(()) => BackgroundTaskState::Completed,
                Err(_) => BackgroundTaskState::Failed,
            };
            if let Some(record) = records
                .lock()
                .await
                .get_mut(&session_key)
                .and_then(|tasks| tasks.get_mut(&task_id))
                && !record.status.state.is_terminal()
            {
                record.status.state = terminal;
                record.status.updated_at = SystemTime::now();
                record.status.completed_at = Some(record.status.updated_at);
                record.handle = None;
            }
            result
        });
        let record = ManagedTaskRecord {
            status: new_background_task_record(session_id, &spec, BackgroundTaskState::Running),
            handle: Some(handle),
            cancel: None,
        };
        let status = record.status.clone();
        tasks.insert(spec.id, record);
        drop(managed);
        self.publish(BackgroundTaskEvent::Registered { record: status });
        Ok(())
    }

    async fn register_external(
        &self,
        session_id: &str,
        spec: BackgroundTaskRegistration,
        cancel: Option<LocalBackgroundTaskCancel>,
    ) -> Result<(), PluginError> {
        let mut managed = self.managed.lock().await;
        let tasks = managed.entry(session_id.to_string()).or_default();
        if tasks
            .get(&spec.id)
            .is_some_and(|record| !record.status.state.is_terminal())
        {
            return Err(PluginError::Session(format!(
                "background task `{}` is already registered",
                spec.id
            )));
        }
        let record = ManagedTaskRecord {
            status: new_background_task_record(session_id, &spec, BackgroundTaskState::Running),
            handle: None,
            cancel,
        };
        let status = record.status.clone();
        tasks.insert(spec.id, record);
        drop(managed);
        self.publish(BackgroundTaskEvent::Registered { record: status });
        Ok(())
    }

    async fn unregister_external(&self, session_id: &str, task_id: &str) {
        let mut managed = self.managed.lock().await;
        if let Some(tasks) = managed.get_mut(session_id) {
            tasks.remove(task_id);
            if tasks.is_empty() {
                managed.remove(session_id);
            }
        }
    }

    async fn mark_terminal(&self, session_id: &str, task_id: &str, state: BackgroundTaskState) {
        if !state.is_terminal() {
            return;
        }
        let mut event = None;
        let mut managed = self.managed.lock().await;
        if let Some(record) = managed
            .get_mut(session_id)
            .and_then(|tasks| tasks.get_mut(task_id))
            && !record.status.state.is_terminal()
        {
            record.status.state = state;
            record.status.updated_at = SystemTime::now();
            record.status.completed_at = Some(record.status.updated_at);
            record.handle = None;
            event = Some(match state {
                BackgroundTaskState::Failed => BackgroundTaskEvent::Failed {
                    record: record.status.clone(),
                },
                BackgroundTaskState::Cancelled => BackgroundTaskEvent::Cancelled {
                    record: record.status.clone(),
                },
                _ => BackgroundTaskEvent::Completed {
                    record: record.status.clone(),
                },
            });
        }
        drop(managed);
        if let Some(event) = event {
            self.publish(event);
        }
    }

    async fn mark_live_state(&self, session_id: &str, task_id: &str, state: BackgroundTaskState) {
        if !matches!(
            state,
            BackgroundTaskState::Running | BackgroundTaskState::Waiting
        ) {
            return;
        }
        let mut event = None;
        let mut managed = self.managed.lock().await;
        if let Some(record) = managed
            .get_mut(session_id)
            .and_then(|tasks| tasks.get_mut(task_id))
            && !record.status.state.is_terminal()
        {
            record.status.state = state;
            record.status.updated_at = SystemTime::now();
            event = Some(BackgroundTaskEvent::StateChanged {
                task_id: task_id.to_string(),
                state,
            });
        }
        drop(managed);
        if let Some(event) = event {
            self.publish(event);
        }
    }

    async fn cancel_managed(&self, session_id: &str, task_id: &str) -> Result<(), PluginError> {
        let (handle, cancel, event) = {
            let mut managed = self.managed.lock().await;
            let Some(record) = managed
                .get_mut(session_id)
                .and_then(|tasks| tasks.get_mut(task_id))
            else {
                return Ok(());
            };
            let taken_handle = record.handle.take();
            let taken_cancel = record.cancel.take();
            if !record.status.state.is_terminal() {
                record.status.state = BackgroundTaskState::Cancelled;
                record.status.updated_at = SystemTime::now();
                record.status.completed_at = Some(record.status.updated_at);
            }
            (
                taken_handle,
                taken_cancel,
                BackgroundTaskEvent::Cancelled {
                    record: record.status.clone(),
                },
            )
        };
        if let Some(handle) = handle {
            handle.abort();
        }
        if let Some(cancel) = cancel {
            cancel().await;
        }
        self.publish(event);
        Ok(())
    }

    async fn list_managed(&self, session_id: &str) -> Vec<BackgroundTaskRecord> {
        let managed = self.managed.lock().await;
        let Some(tasks) = managed.get(session_id) else {
            return Vec::new();
        };
        let mut out: Vec<BackgroundTaskRecord> =
            tasks.values().map(|record| record.status.clone()).collect();
        out.sort_by_key(|left| left.created_at);
        out
    }

    async fn get_managed(&self, session_id: &str, task_id: &str) -> Option<BackgroundTaskRecord> {
        let managed = self.managed.lock().await;
        managed
            .get(session_id)
            .and_then(|tasks| tasks.get(task_id))
            .map(|record| record.status.clone())
    }

    async fn transfer_managed(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        task_ids: &[String],
    ) -> Result<(), PluginError> {
        if from_session_id == to_session_id || task_ids.is_empty() {
            return Ok(());
        }
        let mut managed = self.managed.lock().await;
        for task_id in task_ids {
            if managed
                .get(to_session_id)
                .and_then(|tasks| tasks.get(task_id))
                .is_some_and(|record| !record.status.state.is_terminal())
            {
                return Err(PluginError::Session(format!(
                    "background task `{task_id}` already exists in successor session"
                )));
            }
        }

        let mut moved = Vec::new();
        if let Some(from_tasks) = managed.get_mut(from_session_id) {
            for task_id in task_ids {
                if let Some(record) = from_tasks.remove(task_id) {
                    moved.push((task_id.clone(), record));
                }
            }
            if from_tasks.is_empty() {
                managed.remove(from_session_id);
            }
        }
        if moved.is_empty() {
            return Ok(());
        }
        let to_tasks = managed.entry(to_session_id.to_string()).or_default();
        for (task_id, record) in moved {
            to_tasks.insert(task_id.clone(), record);
            self.publish(BackgroundTaskEvent::Transferred {
                task_id: task_id.clone(),
                scope: BackgroundTaskScope {
                    session_id: to_session_id.to_string(),
                },
            });
        }
        Ok(())
    }

    async fn cancel_all_managed(
        &self,
        session_id: &str,
    ) -> Result<Vec<BackgroundTaskRecord>, PluginError> {
        let live_task_ids = {
            let managed = self.managed.lock().await;
            managed
                .get(session_id)
                .map(|tasks| {
                    tasks
                        .values()
                        .filter(|record| !record.status.state.is_terminal())
                        .map(|record| record.status.id.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let mut out = Vec::new();
        for task_id in live_task_ids {
            self.cancel_managed(session_id, &task_id).await?;
            if let Some(status) = self.get_managed(session_id, &task_id).await {
                out.push(status);
            }
        }
        Ok(out)
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
    pub background_task_host: Arc<dyn BackgroundTaskHost>,
}

impl BackgroundRuntimeHost {
    pub fn new(
        embedded: EmbeddedRuntimeHost,
        background_task_host: Arc<dyn BackgroundTaskHost>,
    ) -> Self {
        Self {
            embedded,
            background_task_host,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub background_task_host: Option<Arc<dyn BackgroundTaskHost>>,
}

impl From<EmbeddedRuntimeHost> for RuntimeHost {
    fn from(value: EmbeddedRuntimeHost) -> Self {
        Self {
            core: value.core,
            session_store_factory: value.session_store_factory,
            background_task_host: None,
        }
    }
}

impl From<BackgroundRuntimeHost> for RuntimeHost {
    fn from(value: BackgroundRuntimeHost) -> Self {
        Self {
            core: value.embedded.core,
            session_store_factory: value.embedded.session_store_factory,
            background_task_host: Some(value.background_task_host),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn spec(id: &str, kind: BackgroundTaskKind) -> BackgroundTaskRegistration {
        BackgroundTaskRegistration {
            id: id.to_string(),
            kind,
            producer: "test",
            child_session_id: None,
            parent_task_id: None,
        }
    }

    #[tokio::test]
    async fn background_task_spawn_managed_records_metadata_and_terminates_on_exit() {
        let executor = LocalBackgroundTaskHost::default();
        executor
            .spawn_managed(
                "s1",
                spec("t1", BackgroundTaskKind::Monitor),
                Box::pin(async { Ok(()) }),
            )
            .await
            .expect("spawn");
        for _ in 0..50 {
            let tasks = executor.list_managed("s1").await;
            if tasks
                .iter()
                .all(|task| !matches!(task.state, BackgroundTaskState::Running))
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let tasks = executor.list_managed("s1").await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].kind, BackgroundTaskKind::Monitor);
        assert_eq!(tasks[0].state, BackgroundTaskState::Completed);
    }

    #[tokio::test]
    async fn background_task_mark_live_state_flips_running_and_waiting_but_preserves_terminal() {
        let executor = LocalBackgroundTaskHost::default();
        executor
            .register_external("s1", spec("sub", BackgroundTaskKind::Subagent), None)
            .await
            .expect("register");
        assert_eq!(
            executor.get_managed("s1", "sub").await.unwrap().state,
            BackgroundTaskState::Running
        );

        executor
            .mark_live_state("s1", "sub", BackgroundTaskState::Waiting)
            .await;
        assert_eq!(
            executor.get_managed("s1", "sub").await.unwrap().state,
            BackgroundTaskState::Waiting
        );

        executor
            .mark_live_state("s1", "sub", BackgroundTaskState::Running)
            .await;
        assert_eq!(
            executor.get_managed("s1", "sub").await.unwrap().state,
            BackgroundTaskState::Running
        );

        // Terminal transitions win over live-state flips.
        executor
            .mark_terminal("s1", "sub", BackgroundTaskState::Completed)
            .await;
        executor
            .mark_live_state("s1", "sub", BackgroundTaskState::Running)
            .await;
        assert_eq!(
            executor.get_managed("s1", "sub").await.unwrap().state,
            BackgroundTaskState::Completed
        );
    }

    #[tokio::test]
    async fn background_task_cancel_managed_fires_external_callback_and_marks_cancelled() {
        let executor = LocalBackgroundTaskHost::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_inner = Arc::clone(&calls);
        let cancel: LocalBackgroundTaskCancel = Arc::new(move || {
            let calls = Arc::clone(&calls_inner);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
            })
        });
        executor
            .register_external(
                "s1",
                spec("sub", BackgroundTaskKind::Subagent),
                Some(cancel),
            )
            .await
            .expect("register");
        executor.cancel_managed("s1", "sub").await.expect("cancel");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let status = executor.get_managed("s1", "sub").await.expect("status");
        assert_eq!(status.state, BackgroundTaskState::Cancelled);
    }

    #[tokio::test]
    async fn background_task_transfer_managed_moves_live_task_visibility() {
        let executor = LocalBackgroundTaskHost::default();
        executor
            .register_external("s1", spec("monitor:one", BackgroundTaskKind::Monitor), None)
            .await
            .expect("register");

        executor
            .transfer_managed("s1", "s2", &["monitor:one".to_string()])
            .await
            .expect("transfer");

        assert!(executor.list_managed("s1").await.is_empty());
        let tasks = executor.list_managed("s2").await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "monitor:one");
        assert_eq!(tasks[0].state, BackgroundTaskState::Running);
    }

    #[tokio::test]
    async fn background_task_cancel_all_managed_cancels_each_live_task() {
        let executor = LocalBackgroundTaskHost::default();
        let calls = Arc::new(AtomicUsize::new(0));
        for id in ["a", "b"] {
            let calls_inner = Arc::clone(&calls);
            let cancel: LocalBackgroundTaskCancel = Arc::new(move || {
                let calls = Arc::clone(&calls_inner);
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                })
            });
            executor
                .register_external("s1", spec(id, BackgroundTaskKind::Other), Some(cancel))
                .await
                .expect("register");
        }

        let statuses = executor.cancel_all_managed("s1").await.expect("cancel all");

        assert_eq!(statuses.len(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(
            statuses
                .iter()
                .all(|status| status.state == BackgroundTaskState::Cancelled)
        );
    }

    #[tokio::test]
    async fn background_task_host_contract_register_update_complete_and_filter() {
        let host = LocalBackgroundTaskHost::default();
        let registered = host
            .register(BackgroundTaskRegisterRequest {
                scope: BackgroundTaskScope {
                    session_id: "s1".to_string(),
                },
                id: "task:one".to_string(),
                kind: BackgroundTaskKind::Other,
                producer: "test".to_string(),
                parent_task_id: None,
                child_session_id: Some("child".to_string()),
                cancel_policy: BackgroundCancelPolicy::External,
                close_policy: BackgroundClosePolicy::Transfer,
                attempt: BackgroundTaskAttempt {
                    attempt: 1,
                    max_attempts: Some(3),
                    idempotency_key: Some("idem".to_string()),
                },
            })
            .await
            .expect("register");

        assert_eq!(registered.state, BackgroundTaskState::Pending);
        assert_eq!(registered.child_session_id.as_deref(), Some("child"));

        let updated = host
            .update(
                "task:one",
                BackgroundTaskUpdate {
                    state: Some(BackgroundTaskState::Running),
                    progress: Some("started".to_string()),
                },
            )
            .await
            .expect("update");
        assert_eq!(updated.state, BackgroundTaskState::Running);

        assert_eq!(
            host.list(BackgroundTaskFilter {
                session_id: Some("s1".to_string()),
                kind: Some(BackgroundTaskKind::Other),
                include_terminal: false,
            })
            .await
            .len(),
            1
        );

        let completed = host
            .complete(
                "task:one",
                BackgroundTaskCompletion {
                    state: BackgroundTaskState::Completed,
                    summary: Some("done".to_string()),
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
        assert!(
            host.list(BackgroundTaskFilter {
                session_id: Some("s1".to_string()),
                kind: None,
                include_terminal: false,
            })
            .await
            .is_empty()
        );
    }
}
