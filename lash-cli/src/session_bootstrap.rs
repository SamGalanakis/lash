use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use lash::{LashCore, LashSession, ModeId, ModePreset};
use lash_core::{
    PersistedSessionConfig, PersistedSessionState, PluginHost, RuntimeCoreConfig,
    RuntimePersistence, SessionGraph, SessionHead, SessionPolicy, SessionTaskExecutor,
};
use lash_sqlite_store::Store;

use crate::session_log::{self, SessionLogger, SessionStart};

pub(crate) enum SessionBootstrapSource {
    Fresh,
    Resume(String),
    ForkChild { parent_session_id: String },
}

pub(crate) struct SessionBootstrap {
    source: SessionBootstrapSource,
    sessions_dir: PathBuf,
    filename: String,
    store: Arc<Store>,
    resume_start: Option<SessionStart>,
    resume_head: Option<SessionHead>,
    session_name: String,
}

pub(crate) struct OpenedCliLashSession {
    pub(crate) bootstrap: SessionBootstrap,
    pub(crate) logger: SessionLogger,
    pub(crate) session: LashSession,
}

#[derive(Clone)]
pub(crate) struct CliSessionOpener {
    plugin_host: PluginHost,
    core: RuntimeCoreConfig,
    session_task_executor: Arc<dyn SessionTaskExecutor>,
}

fn policy_with_persisted_config(
    mut policy: SessionPolicy,
    session_id: String,
    config: Option<&PersistedSessionConfig>,
) -> SessionPolicy {
    if let Some(config) = config {
        if !config.configured_model.is_empty() {
            policy.model = config.configured_model.clone();
        }
        if config.context_window > 0 {
            policy.max_context_tokens = Some(config.context_window as usize);
        }
        policy.execution_mode = config.execution_mode.clone();
        policy.standard_context_approach = config.standard_context_approach.clone();
        policy.model_variant = config.model_variant.clone();
    }
    policy.session_id = Some(session_id);
    policy
}

impl SessionBootstrapSource {
    pub(crate) fn from_resume_arg(resume: Option<String>) -> Self {
        match resume {
            Some(identifier) => Self::Resume(
                session_log::filename_for_session_identifier(&identifier).unwrap_or(identifier),
            ),
            None => Self::Fresh,
        }
    }
}

impl SessionBootstrap {
    pub(crate) fn open(source: SessionBootstrapSource) -> Result<Self> {
        let sessions_dir = session_log::sessions_dir();
        std::fs::create_dir_all(&sessions_dir)?;
        let filename = match &source {
            SessionBootstrapSource::Fresh | SessionBootstrapSource::ForkChild { .. } => {
                session_log::new_session_filename()
            }
            SessionBootstrapSource::Resume(filename) => filename.clone(),
        };
        let db_path = sessions_dir.join(&filename);
        if matches!(source, SessionBootstrapSource::Resume(_)) && !db_path.is_file() {
            return Err(anyhow::anyhow!("Could not resolve session `{filename}`"));
        }
        let store = Arc::new(Store::open(&db_path)?);
        let resume_start = if matches!(source, SessionBootstrapSource::Resume(_)) {
            store.load_session_meta().map(|meta| SessionStart {
                session_id: meta.session_id,
                session_name: meta.session_name,
            })
        } else {
            None
        };
        let session_name = resume_start
            .as_ref()
            .map(|start| start.session_name.clone())
            .unwrap_or_else(|| crate::generate_session_name(&sessions_dir));
        let resume_head = if matches!(source, SessionBootstrapSource::Resume(_)) {
            store.load_session_head()
        } else {
            None
        };
        Ok(Self {
            source,
            sessions_dir,
            filename,
            store,
            resume_start,
            resume_head,
            session_name,
        })
    }

    pub(crate) fn sessions_dir(&self) -> &Path {
        &self.sessions_dir
    }

    pub(crate) fn filename(&self) -> &str {
        &self.filename
    }

    pub(crate) fn store(&self) -> Arc<Store> {
        Arc::clone(&self.store)
    }

    pub(crate) fn run_session_id(&self) -> Option<String> {
        self.resume_start
            .as_ref()
            .map(|start| start.session_id.clone())
            .or_else(|| Some(uuid::Uuid::new_v4().to_string()))
    }

    pub(crate) fn persisted_config(&self) -> Option<PersistedSessionConfig> {
        self.resume_head.as_ref().map(|head| head.config.clone())
    }

    pub(crate) fn initial_graph(&self) -> SessionGraph {
        self.resume_head
            .as_ref()
            .map(|head| head.graph.clone())
            .unwrap_or_default()
    }

    pub(crate) fn session_name(&self) -> String {
        self.session_name.clone()
    }

    pub(crate) fn logger(&self, model: &str, session_id: Option<String>) -> Result<SessionLogger> {
        match &self.source {
            SessionBootstrapSource::Resume(_) => {
                SessionLogger::resume(Arc::clone(&self.store), &self.filename)
            }
            SessionBootstrapSource::Fresh => SessionLogger::new(
                Arc::clone(&self.store),
                self.filename.clone(),
                model,
                session_id,
                self.session_name(),
            ),
            SessionBootstrapSource::ForkChild { parent_session_id } => {
                let logger = SessionLogger::new(
                    Arc::clone(&self.store),
                    self.filename.clone(),
                    model,
                    session_id,
                    self.session_name(),
                )?;
                logger.mark_as_child_of(parent_session_id)?;
                Ok(logger)
            }
        }
    }

    pub(crate) fn fork_child(parent_session_id: &str, model: &str) -> Result<Self> {
        let bootstrap = Self::open(SessionBootstrapSource::ForkChild {
            parent_session_id: parent_session_id.to_string(),
        })?;
        let _logger = bootstrap.logger(model, Some(uuid::Uuid::new_v4().to_string()))?;
        Ok(bootstrap)
    }
}

impl CliSessionOpener {
    pub(crate) fn new(
        plugin_host: PluginHost,
        core: RuntimeCoreConfig,
        session_task_executor: Arc<dyn SessionTaskExecutor>,
    ) -> Self {
        Self {
            plugin_host,
            core,
            session_task_executor,
        }
    }

    pub(crate) async fn open(
        &self,
        source: SessionBootstrapSource,
        fallback_policy: SessionPolicy,
    ) -> Result<OpenedCliLashSession> {
        let bootstrap = SessionBootstrap::open(source)?;
        let session_id = bootstrap
            .run_session_id()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let policy = policy_with_persisted_config(
            fallback_policy,
            session_id.clone(),
            bootstrap.persisted_config().as_ref(),
        );
        let logger = bootstrap.logger(&policy.model, Some(session_id.clone()))?;
        let state = PersistedSessionState {
            session_id: session_id.clone(),
            policy: policy.clone(),
            session_graph: bootstrap.initial_graph(),
            ..PersistedSessionState::default()
        };
        let store: Arc<dyn RuntimePersistence> = bootstrap.store();
        let mut core_builder = LashCore::builder()
            .install_mode(ModePreset::standard())
            .install_mode(ModePreset::rlm())
            .default_mode(ModeId::new(policy.execution_mode.plugin_id().to_string()))
            .provider(policy.provider.clone())
            .model(policy.model.clone(), policy.model_variant.clone())
            .child_store_factory(Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
                bootstrap.sessions_dir().to_path_buf(),
            )));
        if let Some(max_context_tokens) = policy.max_context_tokens {
            core_builder = core_builder.max_context_tokens(max_context_tokens);
        }
        let core = core_builder
            .advanced()
            .plugin_host(self.plugin_host.clone())
            .runtime_core_config(self.core.clone())
            .session_task_executor(Arc::clone(&self.session_task_executor))
            .build()?;
        let session = core
            .session(session_id)
            .mode(ModeId::new(policy.execution_mode.plugin_id().to_string()))
            .store(store)
            .open_with_state(state)
            .await?;
        Ok(OpenedCliLashSession {
            bootstrap,
            logger,
            session,
        })
    }

    pub(crate) async fn fresh(
        &self,
        fallback_policy: SessionPolicy,
    ) -> Result<OpenedCliLashSession> {
        self.open(SessionBootstrapSource::Fresh, fallback_policy)
            .await
    }

    pub(crate) async fn resume(
        &self,
        identifier: &str,
        fallback_policy: SessionPolicy,
    ) -> Result<OpenedCliLashSession> {
        self.open(
            SessionBootstrapSource::from_resume_arg(Some(identifier.to_string())),
            fallback_policy,
        )
        .await
    }
}
