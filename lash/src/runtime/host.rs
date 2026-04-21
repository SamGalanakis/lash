use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::plugin::PluginError;

use super::{PathResolver, SanitizerPolicy, SessionStoreFactory, TerminationPolicy};

/// Category of a registered background task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManagedTaskKind {
    Monitor,
    Subagent,
    Observer,
    Other,
}

impl ManagedTaskKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ManagedTaskKind::Monitor => "monitor",
            ManagedTaskKind::Subagent => "subagent",
            ManagedTaskKind::Observer => "observer",
            ManagedTaskKind::Other => "other",
        }
    }
}

/// Lifecycle state of a registered background task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManagedRunState {
    /// Actively working on a task right now.
    Running,
    /// Long-lived task is alive but has nothing to do right now. Used for
    /// subagent sessions that finished their last task and are waiting
    /// for a follow-up via `followup_task` or `send_message`. A new
    /// follow-up transitions the task back to `Running`.
    Idle,
    Completed,
    Failed,
    Cancelled,
}

impl ManagedRunState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Metadata required to register a background task with the executor.
#[derive(Clone, Debug)]
pub struct ManagedTaskSpec {
    pub id: String,
    pub label: String,
    pub kind: ManagedTaskKind,
    pub producer: &'static str,
}

/// Public snapshot of a registered background task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManagedTaskStatus {
    pub id: String,
    pub label: String,
    pub kind: ManagedTaskKind,
    pub producer: String,
    pub run_state: ManagedRunState,
    pub started_at: SystemTime,
}

/// Cancellation hook for externally-owned tasks (e.g. subagent subtrees).
/// The registry invokes it when `cancel_managed` is called on a task that
/// has no tokio handle.
pub type ManagedTaskCancel =
    Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;

/// Destination for LLM request/response/stream debug entries. Lash
/// emits structured JSON entries at several turn-loop checkpoints; the
/// host decides where they go.
///
/// Default: none (lash emits nothing). lash-cli wires a
/// [`FileLlmCallLogger`] pointing at a rotating file. Webserver
/// embedders typically provide their own impl that routes entries to
/// an observability pipeline (OTel, Datadog, in-memory ring buffer).
pub trait LlmCallLogger: Send + Sync {
    /// Append a structured entry. Implementations should not block the
    /// caller for long; buffer + flush asynchronously if needed.
    fn append(&self, entry: &serde_json::Value);
}

/// File-backed [`LlmCallLogger`] that writes one JSON line per entry.
/// Uses an internal mutex so concurrent turn drivers don't interleave
/// lines. Failures log at warn-level and are otherwise silent.
pub struct FileLlmCallLogger {
    path: PathBuf,
    lock: StdMutex<()>,
}

impl FileLlmCallLogger {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: StdMutex::new(()),
        }
    }
}

impl LlmCallLogger for FileLlmCallLogger {
    fn append(&self, entry: &serde_json::Value) {
        use std::io::Write;
        let Ok(line) = serde_json::to_string(entry) else {
            return;
        };
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path);
        match file {
            Ok(mut file) => {
                let _guard = self.lock.lock().ok();
                if let Err(err) = writeln!(file, "{}", line) {
                    tracing::warn!(
                        error = %err,
                        path = %self.path.display(),
                        "failed to append llm debug log"
                    );
                }
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %self.path.display(),
                    "failed to open llm debug log"
                );
            }
        }
    }
}

/// Default resolver for file and directory references.
#[derive(Default)]
pub struct DefaultPathResolver;

impl PathResolver for DefaultPathResolver {
    fn resolve(&self, path: &str, expect_file: bool, base_dir: &Path) -> Result<PathBuf, String> {
        if path.is_empty() {
            return Err("Path reference cannot be empty".to_string());
        }
        let p = Path::new(path);
        let candidate = if p.is_absolute() {
            p.to_path_buf()
        } else {
            base_dir.join(p)
        };
        if !candidate.exists() {
            return Err(format!(
                "Referenced path does not exist: {}",
                candidate.display()
            ));
        }
        if expect_file && !candidate.is_file() {
            return Err(format!(
                "Referenced path is not a file: {}",
                candidate.display()
            ));
        }
        if !expect_file && !candidate.is_dir() {
            return Err(format!(
                "Referenced path is not a directory: {}",
                candidate.display()
            ));
        }
        candidate
            .canonicalize()
            .map_err(|e| format!("Failed to canonicalize {}: {e}", candidate.display()))
    }
}

/// Host-owned session task execution policy.
#[async_trait::async_trait]
pub trait SessionTaskExecutor: Send + Sync {
    async fn spawn_hidden(
        &self,
        session_id: &str,
        label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError>;

    async fn await_hidden(&self, session_id: &str) -> Result<(), PluginError>;

    /// Spawn a tokio-backed background task and register it under `spec.id`.
    /// The executor owns the `JoinHandle` and can abort it via `cancel_managed`.
    async fn spawn_managed(
        &self,
        session_id: &str,
        spec: ManagedTaskSpec,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError>;

    /// Register an externally-owned task (e.g. a subagent session) so it
    /// shows up in `list_managed`. The executor does not hold a `JoinHandle`;
    /// if `cancel` is supplied it will be invoked when the registry receives
    /// a cancel request, otherwise cancellation only updates bookkeeping.
    async fn register_external(
        &self,
        session_id: &str,
        spec: ManagedTaskSpec,
        cancel: Option<ManagedTaskCancel>,
    ) -> Result<(), PluginError>;

    /// Update the run_state of a registered task to a terminal value.
    /// No-op if the task is unknown or already terminal.
    async fn mark_terminal(&self, session_id: &str, task_id: &str, run_state: ManagedRunState);

    /// Transition a still-live task between the non-terminal `Running`
    /// and `Idle` states. Used by subagents to reflect whether the
    /// session is actively working on a task or waiting for a
    /// follow-up. No-op if the task is unknown or already terminal.
    async fn mark_live_state(&self, session_id: &str, task_id: &str, run_state: ManagedRunState);

    /// Cancel a tokio-backed task by id. No-op for externally-owned entries;
    /// callers should also mark those terminal via `mark_terminal`.
    async fn cancel_managed(&self, session_id: &str, task_id: &str) -> Result<(), PluginError>;

    /// Read-only snapshot of every registered task for the session.
    async fn list_managed(&self, session_id: &str) -> Vec<ManagedTaskStatus>;

    /// Look up a single task's metadata.
    async fn get_managed(&self, session_id: &str, task_id: &str) -> Option<ManagedTaskStatus>;
}

/// Tokio-backed session task executor shared across runtime sessions.
#[derive(Default)]
pub struct TokioSessionTaskExecutor {
    hidden: Mutex<HiddenTaskMap>,
    managed: Arc<Mutex<ManagedTaskMap>>,
}

type SessionTaskHandle = tokio::task::JoinHandle<Result<(), PluginError>>;
type HiddenTaskMap = HashMap<String, Vec<SessionTaskHandle>>;
type ManagedTaskMap = HashMap<String, HashMap<String, ManagedTaskRecord>>;

struct ManagedTaskRecord {
    status: ManagedTaskStatus,
    handle: Option<SessionTaskHandle>,
    cancel: Option<ManagedTaskCancel>,
}

#[async_trait::async_trait]
impl SessionTaskExecutor for TokioSessionTaskExecutor {
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
                            "hidden session task failed: {err}"
                        )));
                    }
                }
            }
        }
    }

    async fn spawn_managed(
        &self,
        session_id: &str,
        spec: ManagedTaskSpec,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError> {
        let mut managed = self.managed.lock().await;
        let tasks = managed.entry(session_id.to_string()).or_default();
        if tasks
            .get(&spec.id)
            .is_some_and(|record| !record.status.run_state.is_terminal())
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
                Ok(()) => ManagedRunState::Completed,
                Err(_) => ManagedRunState::Failed,
            };
            if let Some(record) = records
                .lock()
                .await
                .get_mut(&session_key)
                .and_then(|tasks| tasks.get_mut(&task_id))
                && !record.status.run_state.is_terminal()
            {
                record.status.run_state = terminal;
                record.handle = None;
            }
            result
        });
        let record = ManagedTaskRecord {
            status: ManagedTaskStatus {
                id: spec.id.clone(),
                label: spec.label,
                kind: spec.kind,
                producer: spec.producer.to_string(),
                run_state: ManagedRunState::Running,
                started_at: SystemTime::now(),
            },
            handle: Some(handle),
            cancel: None,
        };
        tasks.insert(spec.id, record);
        Ok(())
    }

    async fn register_external(
        &self,
        session_id: &str,
        spec: ManagedTaskSpec,
        cancel: Option<ManagedTaskCancel>,
    ) -> Result<(), PluginError> {
        let mut managed = self.managed.lock().await;
        let tasks = managed.entry(session_id.to_string()).or_default();
        if tasks
            .get(&spec.id)
            .is_some_and(|record| !record.status.run_state.is_terminal())
        {
            return Err(PluginError::Session(format!(
                "background task `{}` is already registered",
                spec.id
            )));
        }
        let record = ManagedTaskRecord {
            status: ManagedTaskStatus {
                id: spec.id.clone(),
                label: spec.label,
                kind: spec.kind,
                producer: spec.producer.to_string(),
                run_state: ManagedRunState::Running,
                started_at: SystemTime::now(),
            },
            handle: None,
            cancel,
        };
        tasks.insert(spec.id, record);
        Ok(())
    }

    async fn mark_terminal(&self, session_id: &str, task_id: &str, run_state: ManagedRunState) {
        if !run_state.is_terminal() {
            return;
        }
        let mut managed = self.managed.lock().await;
        if let Some(record) = managed
            .get_mut(session_id)
            .and_then(|tasks| tasks.get_mut(task_id))
            && !record.status.run_state.is_terminal()
        {
            record.status.run_state = run_state;
            record.handle = None;
        }
    }

    async fn mark_live_state(&self, session_id: &str, task_id: &str, run_state: ManagedRunState) {
        if !matches!(run_state, ManagedRunState::Running | ManagedRunState::Idle) {
            return;
        }
        let mut managed = self.managed.lock().await;
        if let Some(record) = managed
            .get_mut(session_id)
            .and_then(|tasks| tasks.get_mut(task_id))
            && !record.status.run_state.is_terminal()
        {
            record.status.run_state = run_state;
        }
    }

    async fn cancel_managed(&self, session_id: &str, task_id: &str) -> Result<(), PluginError> {
        let (handle, cancel) = {
            let mut managed = self.managed.lock().await;
            let Some(record) = managed
                .get_mut(session_id)
                .and_then(|tasks| tasks.get_mut(task_id))
            else {
                return Ok(());
            };
            let taken_handle = record.handle.take();
            let taken_cancel = record.cancel.take();
            if !record.status.run_state.is_terminal() {
                record.status.run_state = ManagedRunState::Cancelled;
            }
            (taken_handle, taken_cancel)
        };
        if let Some(handle) = handle {
            handle.abort();
        }
        if let Some(cancel) = cancel {
            cancel().await;
        }
        Ok(())
    }

    async fn list_managed(&self, session_id: &str) -> Vec<ManagedTaskStatus> {
        let managed = self.managed.lock().await;
        let Some(tasks) = managed.get(session_id) else {
            return Vec::new();
        };
        let mut out: Vec<ManagedTaskStatus> =
            tasks.values().map(|record| record.status.clone()).collect();
        out.sort_by_key(|left| left.started_at);
        out
    }

    async fn get_managed(&self, session_id: &str, task_id: &str) -> Option<ManagedTaskStatus> {
        let managed = self.managed.lock().await;
        managed
            .get(session_id)
            .and_then(|tasks| tasks.get(task_id))
            .map(|record| record.status.clone())
    }
}

/// Required host configuration for all runtimes.
#[derive(Clone)]
pub struct RuntimeCoreConfig {
    pub base_dir: PathBuf,
    pub path_resolver: Arc<dyn PathResolver>,
    pub prompt_template: crate::PromptTemplate,
    pub llm_logger: Option<Arc<dyn LlmCallLogger>>,
    pub sanitizer: SanitizerPolicy,
    pub termination: TerminationPolicy,
    pub retry_policy: lash_sansio::RetryPolicy,
    /// Host-owned destination for refreshed OAuth credentials. When
    /// `Some`, lash writes refreshed provider tokens here so they
    /// persist across runs. When `None`, token refresh succeeds but
    /// nothing is written — the host either doesn't need persistence
    /// (tests, one-shot calls) or handles it via a different channel.
    pub credential_store_path: Option<PathBuf>,
}

impl Default for RuntimeCoreConfig {
    fn default() -> Self {
        Self {
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            path_resolver: Arc::new(DefaultPathResolver),
            prompt_template: crate::default_prompt_template(),
            llm_logger: None,
            sanitizer: SanitizerPolicy::default(),
            termination: TerminationPolicy::default(),
            retry_policy: lash_sansio::RetryPolicy::default(),
            credential_store_path: None,
        }
    }
}

impl RuntimeCoreConfig {
    pub fn with_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = base_dir.into();
        self
    }

    pub fn with_path_resolver(mut self, path_resolver: Arc<dyn PathResolver>) -> Self {
        self.path_resolver = path_resolver;
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: crate::PromptTemplate) -> Self {
        self.prompt_template = prompt_template;
        self
    }

    pub fn with_llm_log_path(mut self, llm_log_path: Option<PathBuf>) -> Self {
        self.llm_logger = llm_log_path
            .map(|path| Arc::new(FileLlmCallLogger::new(path)) as Arc<dyn LlmCallLogger>);
        self
    }

    pub fn with_llm_logger(mut self, logger: Option<Arc<dyn LlmCallLogger>>) -> Self {
        self.llm_logger = logger;
        self
    }

    pub fn with_sanitizer(mut self, sanitizer: SanitizerPolicy) -> Self {
        self.sanitizer = sanitizer;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.termination = termination;
        self
    }

    pub fn with_retry_policy(mut self, policy: lash_sansio::RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn with_credential_store_path(mut self, path: Option<PathBuf>) -> Self {
        self.credential_store_path = path;
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
    pub session_task_executor: Arc<dyn SessionTaskExecutor>,
}

impl BackgroundRuntimeHost {
    pub fn new(
        embedded: EmbeddedRuntimeHost,
        session_task_executor: Arc<dyn SessionTaskExecutor>,
    ) -> Self {
        Self {
            embedded,
            session_task_executor,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub session_task_executor: Option<Arc<dyn SessionTaskExecutor>>,
}

impl From<EmbeddedRuntimeHost> for RuntimeHost {
    fn from(value: EmbeddedRuntimeHost) -> Self {
        Self {
            core: value.core,
            session_store_factory: value.session_store_factory,
            session_task_executor: None,
        }
    }
}

impl From<BackgroundRuntimeHost> for RuntimeHost {
    fn from(value: BackgroundRuntimeHost) -> Self {
        Self {
            core: value.embedded.core,
            session_store_factory: value.embedded.session_store_factory,
            session_task_executor: Some(value.session_task_executor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn spec(id: &str, kind: ManagedTaskKind) -> ManagedTaskSpec {
        ManagedTaskSpec {
            id: id.to_string(),
            label: id.to_string(),
            kind,
            producer: "test",
        }
    }

    #[tokio::test]
    async fn spawn_managed_records_metadata_and_terminates_on_exit() {
        let executor = TokioSessionTaskExecutor::default();
        executor
            .spawn_managed(
                "s1",
                spec("t1", ManagedTaskKind::Monitor),
                Box::pin(async { Ok(()) }),
            )
            .await
            .expect("spawn");
        for _ in 0..50 {
            let tasks = executor.list_managed("s1").await;
            if tasks
                .iter()
                .all(|task| !matches!(task.run_state, ManagedRunState::Running))
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let tasks = executor.list_managed("s1").await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].kind, ManagedTaskKind::Monitor);
        assert_eq!(tasks[0].run_state, ManagedRunState::Completed);
    }

    #[tokio::test]
    async fn mark_live_state_flips_running_and_idle_but_preserves_terminal() {
        let executor = TokioSessionTaskExecutor::default();
        executor
            .register_external("s1", spec("sub", ManagedTaskKind::Subagent), None)
            .await
            .expect("register");
        assert_eq!(
            executor.get_managed("s1", "sub").await.unwrap().run_state,
            ManagedRunState::Running
        );

        executor
            .mark_live_state("s1", "sub", ManagedRunState::Idle)
            .await;
        assert_eq!(
            executor.get_managed("s1", "sub").await.unwrap().run_state,
            ManagedRunState::Idle
        );

        executor
            .mark_live_state("s1", "sub", ManagedRunState::Running)
            .await;
        assert_eq!(
            executor.get_managed("s1", "sub").await.unwrap().run_state,
            ManagedRunState::Running
        );

        // Terminal transitions win over live-state flips.
        executor
            .mark_terminal("s1", "sub", ManagedRunState::Completed)
            .await;
        executor
            .mark_live_state("s1", "sub", ManagedRunState::Running)
            .await;
        assert_eq!(
            executor.get_managed("s1", "sub").await.unwrap().run_state,
            ManagedRunState::Completed
        );
    }

    #[tokio::test]
    async fn cancel_managed_fires_external_callback_and_marks_cancelled() {
        let executor = TokioSessionTaskExecutor::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_inner = Arc::clone(&calls);
        let cancel: ManagedTaskCancel = Arc::new(move || {
            let calls = Arc::clone(&calls_inner);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
            })
        });
        executor
            .register_external("s1", spec("sub", ManagedTaskKind::Subagent), Some(cancel))
            .await
            .expect("register");
        executor.cancel_managed("s1", "sub").await.expect("cancel");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let status = executor.get_managed("s1", "sub").await.expect("status");
        assert_eq!(status.run_state, ManagedRunState::Cancelled);
    }
}
