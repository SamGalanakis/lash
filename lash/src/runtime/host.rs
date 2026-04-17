use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use tokio::sync::Mutex;

use crate::llm::factory::adapter_for;
use crate::llm::transport::LlmTransport;
use crate::plugin::PluginError;
use crate::provider::Provider;

use super::{PathResolver, SanitizerPolicy, SessionStoreFactory, TerminationPolicy};

pub(crate) type LlmFactory = Arc<dyn Fn(&Provider) -> Box<dyn LlmTransport> + Send + Sync>;

fn default_llm_factory() -> LlmFactory {
    Arc::new(|provider| adapter_for(provider))
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

    async fn spawn_managed(
        &self,
        session_id: &str,
        task_id: &str,
        label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError>;

    async fn cancel_managed(&self, session_id: &str, task_id: &str) -> Result<(), PluginError>;
}

/// Tokio-backed session task executor shared across runtime sessions.
#[derive(Default)]
pub struct TokioSessionTaskExecutor {
    hidden: Mutex<HiddenTaskMap>,
    managed: Mutex<ManagedTaskMap>,
}

type SessionTaskHandle = tokio::task::JoinHandle<Result<(), PluginError>>;
type HiddenTaskMap = HashMap<String, Vec<SessionTaskHandle>>;
type ManagedTaskMap = HashMap<String, HashMap<String, SessionTaskHandle>>;

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
        task_id: &str,
        _label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), PluginError> {
        self.prune_finished_managed(session_id).await;
        let mut managed = self.managed.lock().await;
        let tasks = managed.entry(session_id.to_string()).or_default();
        if tasks.contains_key(task_id) {
            return Err(PluginError::Session(format!(
                "managed session task `{task_id}` is already running"
            )));
        }
        tasks.insert(task_id.to_string(), tokio::spawn(task));
        Ok(())
    }

    async fn cancel_managed(&self, session_id: &str, task_id: &str) -> Result<(), PluginError> {
        self.prune_finished_managed(session_id).await;
        let mut managed = self.managed.lock().await;
        let Some(handle) = managed
            .get_mut(session_id)
            .and_then(|tasks| tasks.remove(task_id))
        else {
            return Ok(());
        };
        handle.abort();
        Ok(())
    }
}

impl TokioSessionTaskExecutor {
    async fn prune_finished_managed(&self, session_id: &str) {
        let finished = {
            let managed = self.managed.lock().await;
            managed
                .get(session_id)
                .map(|tasks| {
                    tasks
                        .iter()
                        .filter_map(|(task_id, handle)| {
                            handle.is_finished().then_some(task_id.clone())
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        if finished.is_empty() {
            return;
        }
        let mut removed = Vec::new();
        {
            let mut managed = self.managed.lock().await;
            if let Some(tasks) = managed.get_mut(session_id) {
                for task_id in &finished {
                    if let Some(handle) = tasks.remove(task_id) {
                        removed.push(handle);
                    }
                }
                if tasks.is_empty() {
                    managed.remove(session_id);
                }
            }
        }
        for handle in removed {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    tracing::warn!("managed session task finished with error: {err}");
                }
                Err(err) => {
                    tracing::warn!("managed session task join failed: {err}");
                }
            }
        }
    }
}

/// Required host configuration for all runtimes.
#[derive(Clone)]
pub struct RuntimeCoreConfig {
    pub base_dir: PathBuf,
    pub path_resolver: Arc<dyn PathResolver>,
    pub prompt_template: crate::PromptTemplate,
    pub llm_log_path: Option<PathBuf>,
    pub(crate) llm_log_lock: Arc<StdMutex<()>>,
    pub sanitizer: SanitizerPolicy,
    pub termination: TerminationPolicy,
    pub(crate) llm_factory: LlmFactory,
}

impl Default for RuntimeCoreConfig {
    fn default() -> Self {
        Self {
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            path_resolver: Arc::new(DefaultPathResolver),
            prompt_template: crate::default_prompt_template(),
            llm_log_path: None,
            llm_log_lock: Arc::new(StdMutex::new(())),
            sanitizer: SanitizerPolicy::default(),
            termination: TerminationPolicy::default(),
            llm_factory: default_llm_factory(),
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
        self.llm_log_path = llm_log_path;
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

    pub fn with_llm_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn(&Provider) -> Box<dyn LlmTransport> + Send + Sync + 'static,
    {
        self.llm_factory = Arc::new(factory);
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
