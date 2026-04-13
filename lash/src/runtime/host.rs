use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

/// Host-owned background execution policy.
#[async_trait::async_trait]
pub trait BackgroundExecutor: Send + Sync {
    async fn spawn(
        &self,
        session_id: &str,
        label: &str,
        job: crate::plugin::PluginBackgroundJob,
    ) -> Result<(), PluginError>;

    async fn await_all(&self, session_id: &str) -> Result<(), PluginError>;
}

/// Tokio-backed background executor shared across runtime sessions.
#[derive(Default)]
pub struct TokioBackgroundExecutor {
    jobs: Mutex<HashMap<String, Vec<tokio::task::JoinHandle<Result<(), PluginError>>>>>,
}

#[async_trait::async_trait]
impl BackgroundExecutor for TokioBackgroundExecutor {
    async fn spawn(
        &self,
        session_id: &str,
        _label: &str,
        job: crate::plugin::PluginBackgroundJob,
    ) -> Result<(), PluginError> {
        let handle = tokio::spawn(job);
        self.jobs
            .lock()
            .await
            .entry(session_id.to_string())
            .or_default()
            .push(handle);
        Ok(())
    }

    async fn await_all(&self, session_id: &str) -> Result<(), PluginError> {
        loop {
            let jobs = self
                .jobs
                .lock()
                .await
                .remove(session_id)
                .unwrap_or_default();
            if jobs.is_empty() {
                return Ok(());
            }
            for job in jobs {
                match job.await {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => return Err(err),
                    Err(err) => {
                        return Err(PluginError::Session(format!(
                            "background job task failed: {err}"
                        )));
                    }
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
    pub prompt_renderer: Arc<dyn crate::PromptRenderer>,
    pub prompt_overrides: Vec<crate::PromptSectionOverride>,
    pub llm_log_path: Option<PathBuf>,
    pub sanitizer: SanitizerPolicy,
    pub termination: TerminationPolicy,
    pub(crate) llm_factory: LlmFactory,
}

impl Default for RuntimeCoreConfig {
    fn default() -> Self {
        Self {
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            path_resolver: Arc::new(DefaultPathResolver),
            prompt_renderer: crate::default_prompt_renderer(),
            prompt_overrides: Vec::new(),
            llm_log_path: None,
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

    pub fn with_prompt_renderer(mut self, prompt_renderer: Arc<dyn crate::PromptRenderer>) -> Self {
        self.prompt_renderer = prompt_renderer;
        self
    }

    pub fn with_prompt_overrides(
        mut self,
        prompt_overrides: Vec<crate::PromptSectionOverride>,
    ) -> Self {
        self.prompt_overrides = prompt_overrides;
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
    pub background_executor: Arc<dyn BackgroundExecutor>,
}

impl BackgroundRuntimeHost {
    pub fn new(
        embedded: EmbeddedRuntimeHost,
        background_executor: Arc<dyn BackgroundExecutor>,
    ) -> Self {
        Self {
            embedded,
            background_executor,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub background_executor: Option<Arc<dyn BackgroundExecutor>>,
}

impl From<EmbeddedRuntimeHost> for RuntimeHost {
    fn from(value: EmbeddedRuntimeHost) -> Self {
        Self {
            core: value.core,
            session_store_factory: value.session_store_factory,
            background_executor: None,
        }
    }
}

impl From<BackgroundRuntimeHost> for RuntimeHost {
    fn from(value: BackgroundRuntimeHost) -> Self {
        Self {
            core: value.embedded.core,
            session_store_factory: value.embedded.session_store_factory,
            background_executor: Some(value.background_executor),
        }
    }
}
