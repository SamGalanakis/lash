use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use lash::{LashCore, LashSession, ModeId, ModePreset, PluginStack, PromptLayerSink};
use lash_core::provider::ProviderHandle;
use lash_core::runtime::RuntimeSessionState;
use lash_core::store::SessionHead;
use lash_core::{
    AttachmentStore, PersistedSessionConfig, RuntimePersistence, SessionGraph, SessionPolicy,
};
use lash_sqlite_store::Store;
use lash_standard_plugins::StandardContextApproach;

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
    host_config: Option<CliSessionHostConfig>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct CliSessionHostConfig {
    pub(crate) execution_mode: ModeId,
    pub(crate) standard_context_approach: Option<StandardContextApproach>,
}

impl CliSessionHostConfig {
    pub(crate) fn new(
        execution_mode: ModeId,
        standard_context_approach: Option<StandardContextApproach>,
    ) -> Self {
        Self {
            execution_mode,
            standard_context_approach,
        }
    }
}

pub(crate) struct OpenedCliLashSession {
    pub(crate) bootstrap: SessionBootstrap,
    pub(crate) logger: SessionLogger,
    pub(crate) session: LashSession,
}

#[derive(Clone)]
pub(crate) struct CliSessionOpener {
    plugin_stack: PluginStack,
    prompt_layer: lash_core::PromptLayer,
    attachment_store: Arc<dyn AttachmentStore>,
    provider: ProviderHandle,
    trace_jsonl_path: Option<PathBuf>,
    trace_level: lash::tracing::TraceLevel,
}

fn policy_with_persisted_config(
    mut policy: SessionPolicy,
    session_id: String,
    config: Option<&PersistedSessionConfig>,
) -> SessionPolicy {
    if let Some(config) = config {
        policy.model = config.model.clone();
        policy.provider_id = config.provider_id.clone();
    }
    policy.session_id = Some(session_id);
    policy
}

fn host_config_path(sessions_dir: &Path, filename: &str) -> PathBuf {
    sessions_dir.join(format!("{filename}.host.json"))
}

fn load_host_config(sessions_dir: &Path, filename: &str) -> Option<CliSessionHostConfig> {
    let path = host_config_path(sessions_dir, filename);
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

fn save_host_config(
    sessions_dir: &Path,
    filename: &str,
    config: &CliSessionHostConfig,
) -> Result<()> {
    std::fs::write(
        host_config_path(sessions_dir, filename),
        serde_json::to_vec_pretty(config)?,
    )?;
    Ok(())
}

impl SessionBootstrapSource {
    pub(crate) async fn from_resume_arg(resume: Option<String>) -> Self {
        match resume {
            Some(identifier) => Self::Resume(
                session_log::filename_for_session_identifier(&identifier)
                    .await
                    .unwrap_or(identifier),
            ),
            None => Self::Fresh,
        }
    }
}

impl SessionBootstrap {
    pub(crate) async fn open(source: SessionBootstrapSource) -> Result<Self> {
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
        let store = Arc::new(Store::open(&db_path).await?);
        let resume_start = if matches!(source, SessionBootstrapSource::Resume(_)) {
            store.load_session_meta().await.map(|meta| SessionStart {
                session_id: meta.session_id,
                session_name: meta.session_name,
            })
        } else {
            None
        };
        let session_name = match resume_start.as_ref() {
            Some(start) => start.session_name.clone(),
            None => crate::generate_session_name(&sessions_dir).await,
        };
        let resume_head = if matches!(source, SessionBootstrapSource::Resume(_)) {
            store.load_session_head().await
        } else {
            None
        };
        let host_config = load_host_config(&sessions_dir, &filename);
        Ok(Self {
            source,
            sessions_dir,
            filename,
            store,
            resume_start,
            resume_head,
            session_name,
            host_config,
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

    fn sidecar_db_path(&self, suffix: &str) -> PathBuf {
        self.sessions_dir
            .join(format!("{}.{}", self.filename, suffix))
    }

    pub(crate) fn artifacts_db_file(&self) -> PathBuf {
        self.sidecar_db_path("artifacts.db")
    }

    pub(crate) fn effects_db_file(&self) -> PathBuf {
        self.sidecar_db_path("effects.db")
    }

    pub(crate) fn processes_db_file(&self) -> PathBuf {
        self.sidecar_db_path("processes.db")
    }

    pub(crate) fn host_events_db_file(&self) -> PathBuf {
        self.sidecar_db_path("host-events.db")
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

    pub(crate) fn persisted_host_config(&self) -> Option<CliSessionHostConfig> {
        self.host_config.clone()
    }

    pub(crate) fn save_host_config(&self, config: &CliSessionHostConfig) -> Result<()> {
        save_host_config(&self.sessions_dir, &self.filename, config)
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

    pub(crate) async fn logger(
        &self,
        model: &str,
        session_id: Option<String>,
    ) -> Result<SessionLogger> {
        match &self.source {
            SessionBootstrapSource::Resume(_) => {
                SessionLogger::resume(Arc::clone(&self.store), &self.filename).await
            }
            SessionBootstrapSource::Fresh => {
                SessionLogger::new(
                    Arc::clone(&self.store),
                    self.filename.clone(),
                    model,
                    session_id,
                    self.session_name(),
                )
                .await
            }
            SessionBootstrapSource::ForkChild { parent_session_id } => {
                let logger = SessionLogger::new(
                    Arc::clone(&self.store),
                    self.filename.clone(),
                    model,
                    session_id,
                    self.session_name(),
                )
                .await?;
                logger.mark_as_child_of(parent_session_id).await?;
                Ok(logger)
            }
        }
    }

    pub(crate) async fn fork_child(parent_session_id: &str, model: &str) -> Result<Self> {
        let bootstrap = Self::open(SessionBootstrapSource::ForkChild {
            parent_session_id: parent_session_id.to_string(),
        })
        .await?;
        let _logger = bootstrap
            .logger(model, Some(uuid::Uuid::new_v4().to_string()))
            .await?;
        Ok(bootstrap)
    }
}

impl CliSessionOpener {
    pub(crate) fn new(
        plugin_stack: PluginStack,
        prompt_layer: lash_core::PromptLayer,
        attachment_store: Arc<dyn AttachmentStore>,
        provider: ProviderHandle,
        trace_jsonl_path: Option<PathBuf>,
        trace_level: lash::tracing::TraceLevel,
    ) -> Self {
        Self {
            plugin_stack,
            prompt_layer,
            attachment_store,
            provider,
            trace_jsonl_path,
            trace_level,
        }
    }

    pub(crate) async fn open(
        &self,
        source: SessionBootstrapSource,
        fallback_policy: SessionPolicy,
        fallback_execution_mode: ModeId,
        fallback_standard_context_approach: Option<StandardContextApproach>,
    ) -> Result<OpenedCliLashSession> {
        let bootstrap = SessionBootstrap::open(source).await?;
        let session_id = bootstrap
            .run_session_id()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let policy = policy_with_persisted_config(
            fallback_policy,
            session_id.clone(),
            bootstrap.persisted_config().as_ref(),
        );
        let host_config = bootstrap.persisted_host_config().unwrap_or_else(|| {
            CliSessionHostConfig::new(fallback_execution_mode, fallback_standard_context_approach)
        });
        bootstrap.save_host_config(&host_config)?;
        let logger = bootstrap
            .logger(&policy.model.id, Some(session_id.clone()))
            .await?;
        let state = RuntimeSessionState {
            session_id: session_id.clone(),
            policy: policy.clone(),
            session_graph: bootstrap.initial_graph(),
            ..RuntimeSessionState::default()
        };
        let store: Arc<dyn RuntimePersistence> = bootstrap.store();
        let artifact_store = Arc::new(Store::open(&bootstrap.artifacts_db_file()).await?)
            as Arc<dyn lash::persistence::LashlangArtifactStore>;
        let effect_host = Arc::new(
            lash_sqlite_store::SqliteEffectHost::open(&bootstrap.effects_db_file()).await?,
        );
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&bootstrap.processes_db_file()).await?,
        );
        let host_event_store = Arc::new(
            lash_sqlite_store::SqliteHostEventStore::open(&bootstrap.host_events_db_file()).await?,
        );
        let mut core_builder = LashCore::builder()
            .install_mode(ModePreset::standard())
            .install_mode(ModePreset::rlm())
            .default_mode(host_config.execution_mode.clone())
            .provider(self.provider.clone())
            .model(policy.model.clone())
            .child_store_factory(Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
                bootstrap.sessions_dir().to_path_buf(),
            )))
            .plugins(self.plugin_stack.clone())
            .prompt_layer(self.prompt_layer.clone())
            .effect_host(effect_host)
            .lashlang_artifact_store(artifact_store)
            .attachment_store(Arc::clone(&self.attachment_store))
            .trace_level(self.trace_level)
            .process_registry(process_registry)
            .host_event_store(host_event_store);
        if let Some(trace_jsonl_path) = self.trace_jsonl_path.clone() {
            core_builder = core_builder.trace_jsonl_path(trace_jsonl_path);
        }
        let core = core_builder.build()?;
        let session = core
            .session(session_id)
            .mode(host_config.execution_mode.clone())
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
        execution_mode: ModeId,
        standard_context_approach: Option<StandardContextApproach>,
    ) -> Result<OpenedCliLashSession> {
        self.open(
            SessionBootstrapSource::Fresh,
            fallback_policy,
            execution_mode,
            standard_context_approach,
        )
        .await
    }

    pub(crate) async fn resume(
        &self,
        identifier: &str,
        fallback_policy: SessionPolicy,
        execution_mode: ModeId,
        standard_context_approach: Option<StandardContextApproach>,
    ) -> Result<OpenedCliLashSession> {
        self.open(
            SessionBootstrapSource::from_resume_arg(Some(identifier.to_string())).await,
            fallback_policy,
            execution_mode,
            standard_context_approach,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, TempDirGuard, env_lock};

    #[tokio::test]
    async fn fresh_cli_session_opens_with_durable_artifact_store() -> Result<()> {
        let _env = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-cli-session-bootstrap");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let provider = lash_core::testing::TestProvider::builder()
            .build()
            .into_handle();
        let opener = CliSessionOpener::new(
            PluginStack::new(),
            lash_core::PromptLayer::new(),
            Arc::new(lash::persistence::FileAttachmentStore::new(
                crate::paths::attachments_dir(),
            )),
            provider,
            None,
            lash::tracing::TraceLevel::Standard,
        );
        let policy = SessionPolicy {
            model: lash_core::ModelSpec::from_token_limits("mock-model", None, 200_000, None)
                .expect("model spec"),
            provider_id: "test".to_string(),
            ..Default::default()
        };

        let opened = opener
            .fresh(
                policy,
                ModeId::standard(),
                Some(StandardContextApproach::default()),
            )
            .await?;

        assert!(!opened.session.session_id().is_empty());
        assert!(opened.bootstrap.artifacts_db_file().is_file());
        assert!(opened.bootstrap.effects_db_file().is_file());
        assert!(opened.bootstrap.processes_db_file().is_file());
        assert!(opened.bootstrap.host_events_db_file().is_file());
        Ok(())
    }
}
