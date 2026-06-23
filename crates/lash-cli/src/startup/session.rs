//! Opening and resuming CLI sessions: the on-disk session bootstrap (store +
//! sidecar databases + host config) and the [`CliSessionOpener`] that builds
//! a `LashCore` and opens the session on top of it.
//!
//! The bootstrap source (fresh / resume / fork-child) is matched exactly once
//! in [`SessionBootstrap::open`]; everything downstream works from the
//! resolved plan instead of re-matching the variants.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use lash::{LashSession, PluginStack, PromptLayerSink, RlmCore, StandardCore};
use lash_core::provider::ProviderHandle;
use lash_core::runtime::RuntimeSessionState;
use lash_core::store::SessionHead;
use lash_core::{
    AttachmentStore, PersistedSessionConfig, RuntimePersistence, SessionGraph, SessionPolicy,
};
use lash_sqlite_store::Store;
use lash_standard_plugins::StandardContextApproach;

use crate::execution_settings::{
    ExecutionMode, RlmTerminationMode, default_rlm_termination_for_mode,
};
use crate::session_log::{self, SessionLogger, SessionStart};

pub(crate) enum SessionBootstrapSource {
    Fresh,
    Resume(String),
    ForkChild { parent_session_id: String },
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

pub(crate) struct SessionBootstrap {
    sessions_dir: PathBuf,
    filename: String,
    store: Arc<Store>,
    resume_start: Option<SessionStart>,
    resume_head: Option<SessionHead>,
    session_name: String,
    host_config: Option<CliSessionHostConfig>,
    /// `Some` when this session was forked from a parent session.
    fork_parent_session_id: Option<String>,
    /// `true` when an existing session database was resumed.
    resumed: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct CliSessionHostConfig {
    pub(crate) execution_mode: ExecutionMode,
    pub(crate) standard_context_approach: Option<StandardContextApproach>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) rlm_termination: Option<RlmTerminationMode>,
}

impl CliSessionHostConfig {
    pub(crate) fn new(
        execution_mode: ExecutionMode,
        standard_context_approach: Option<StandardContextApproach>,
        rlm_termination: Option<RlmTerminationMode>,
    ) -> Self {
        let rlm_termination = if execution_mode.is_rlm() {
            rlm_termination.or_else(|| default_rlm_termination_for_mode(execution_mode))
        } else {
            None
        };
        Self {
            execution_mode,
            standard_context_approach,
            rlm_termination,
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
    deferred_tool_resolver: Option<lash::tools::SharedDeferredToolResolver>,
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

impl SessionBootstrap {
    pub(crate) async fn open(source: SessionBootstrapSource) -> Result<Self> {
        let sessions_dir = session_log::sessions_dir();
        std::fs::create_dir_all(&sessions_dir)?;
        // The single place the bootstrap-source variants are matched: resolve
        // them into a plan (filename + resume/fork markers) up front.
        let (filename, fork_parent_session_id, resumed) = match source {
            SessionBootstrapSource::Fresh => (session_log::new_session_filename(), None, false),
            SessionBootstrapSource::ForkChild { parent_session_id } => (
                session_log::new_session_filename(),
                Some(parent_session_id),
                false,
            ),
            SessionBootstrapSource::Resume(filename) => (filename, None, true),
        };
        let db_path = sessions_dir.join(&filename);
        if resumed && !db_path.is_file() {
            return Err(anyhow::anyhow!("Could not resolve session `{filename}`"));
        }
        let store = Arc::new(Store::open(&db_path).await?);
        let (resume_start, resume_head) = if resumed {
            (
                store.load_session_meta().await.map(|meta| SessionStart {
                    session_id: meta.session_id,
                    session_name: meta.session_name,
                }),
                store.load_session_head().await,
            )
        } else {
            (None, None)
        };
        let session_name = match resume_start.as_ref() {
            Some(start) => start.session_name.clone(),
            None => crate::generate_session_name(&sessions_dir).await,
        };
        let host_config = load_host_config(&sessions_dir, &filename);
        Ok(Self {
            sessions_dir,
            filename,
            store,
            resume_start,
            resume_head,
            session_name,
            host_config,
            fork_parent_session_id,
            resumed,
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

    pub(crate) fn process_env_db_file(&self) -> PathBuf {
        self.sidecar_db_path("process-env.db")
    }

    pub(crate) fn triggers_db_file(&self) -> PathBuf {
        self.sidecar_db_path("triggers.db")
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
        if self.resumed {
            return SessionLogger::resume(Arc::clone(&self.store), &self.filename).await;
        }
        let logger = SessionLogger::new(
            Arc::clone(&self.store),
            self.filename.clone(),
            model,
            session_id,
            self.session_name(),
        )
        .await?;
        if let Some(parent_session_id) = self.fork_parent_session_id.as_deref() {
            logger.mark_as_child_of(parent_session_id).await?;
        }
        Ok(logger)
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        plugin_stack: PluginStack,
        prompt_layer: lash_core::PromptLayer,
        attachment_store: Arc<dyn AttachmentStore>,
        provider: ProviderHandle,
        deferred_tool_resolver: Option<lash::tools::SharedDeferredToolResolver>,
        trace_jsonl_path: Option<PathBuf>,
        trace_level: lash::tracing::TraceLevel,
    ) -> Self {
        Self {
            plugin_stack,
            prompt_layer,
            attachment_store,
            provider,
            deferred_tool_resolver,
            trace_jsonl_path,
            trace_level,
        }
    }

    /// Build the `LashCore` and open the session for an already-opened
    /// bootstrap. The startup pipeline opens the bootstrap once (it needs the
    /// persisted host config before this point) and hands it in here instead
    /// of re-opening the store.
    pub(crate) async fn open_prepared(
        &self,
        bootstrap: SessionBootstrap,
        fallback_policy: SessionPolicy,
        host_config: CliSessionHostConfig,
    ) -> Result<OpenedCliLashSession> {
        let session_id = bootstrap
            .run_session_id()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let policy = policy_with_persisted_config(
            fallback_policy,
            session_id.clone(),
            bootstrap.persisted_config().as_ref(),
        );
        bootstrap.save_host_config(&host_config)?;
        let logger = bootstrap
            .logger(&policy.model.id, Some(session_id.clone()))
            .await?;
        let mut state = RuntimeSessionState {
            session_id: session_id.clone(),
            policy: policy.clone(),
            session_graph: bootstrap.initial_graph(),
            ..RuntimeSessionState::default()
        };
        if host_config.execution_mode.is_rlm()
            && let Some(termination) = host_config.rlm_termination
        {
            state.protocol_turn_options =
                lash_core::ProtocolTurnOptions::typed(lash_rlm_types::RlmCreateExtras {
                    termination: termination.as_rlm_termination(),
                    final_answer_format: None,
                })?;
        }
        let store: Arc<dyn RuntimePersistence> = bootstrap.store();
        let effect_host = Arc::new(
            lash_sqlite_store::SqliteEffectHost::open(&bootstrap.effects_db_file()).await?,
        );
        let process_registry = Arc::new(
            lash_sqlite_store::SqliteProcessRegistry::open(&bootstrap.processes_db_file()).await?,
        );
        let trigger_store = Arc::new(
            lash_sqlite_store::SqliteTriggerStore::open(&bootstrap.triggers_db_file()).await?,
        );
        let process_env_store: Arc<dyn lash::persistence::ProcessExecutionEnvStore> =
            Arc::new(Store::open(&bootstrap.process_env_db_file()).await?);
        let child_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            bootstrap.sessions_dir().to_path_buf(),
        ));
        let session = match host_config.execution_mode {
            ExecutionMode::Standard => {
                let mut core_builder = StandardCore::builder()
                    .provider(self.provider.clone())
                    .model(policy.model.clone())
                    .child_store_factory(child_store_factory)
                    .plugins(self.plugin_stack.clone())
                    .prompt_layer(self.prompt_layer.clone())
                    .effect_host(effect_host)
                    .attachment_store(Arc::clone(&self.attachment_store))
                    .process_env_store(Arc::clone(&process_env_store))
                    .trace_level(self.trace_level)
                    .process_registry(process_registry)
                    .trigger_store(trigger_store);
                if let Some(trace_jsonl_path) = self.trace_jsonl_path.clone() {
                    core_builder = core_builder.trace_jsonl_path(trace_jsonl_path);
                }
                core_builder
                    .build()?
                    .session(session_id)
                    .store(store)
                    .open_with_state(state)
                    .await?
            }
            ExecutionMode::Rlm => {
                let artifact_store = Arc::new(Store::open(&bootstrap.artifacts_db_file()).await?)
                    as Arc<dyn lash::persistence::LashlangArtifactStore>;
                let mut core_builder = RlmCore::builder()
                    .provider(self.provider.clone())
                    .model(policy.model.clone())
                    .child_store_factory(child_store_factory)
                    .plugins(self.plugin_stack.clone())
                    .prompt_layer(self.prompt_layer.clone())
                    .effect_host(effect_host)
                    .lashlang_artifact_store(artifact_store)
                    .attachment_store(Arc::clone(&self.attachment_store))
                    .process_env_store(Arc::clone(&process_env_store))
                    .trace_level(self.trace_level)
                    .process_registry(process_registry)
                    .trigger_store(trigger_store);
                if let Some(trace_jsonl_path) = self.trace_jsonl_path.clone() {
                    core_builder = core_builder.trace_jsonl_path(trace_jsonl_path);
                }
                if let Some(resolver) = self.deferred_tool_resolver.clone() {
                    core_builder = core_builder.deferred_tool_resolver(resolver);
                }
                core_builder
                    .build()?
                    .session(session_id)
                    .store(store)
                    .open_with_state(state)
                    .await?
            }
        };
        Ok(OpenedCliLashSession {
            bootstrap,
            logger,
            session,
        })
    }

    pub(crate) async fn fresh(
        &self,
        fallback_policy: SessionPolicy,
        execution_mode: ExecutionMode,
        standard_context_approach: Option<StandardContextApproach>,
    ) -> Result<OpenedCliLashSession> {
        let bootstrap = SessionBootstrap::open(SessionBootstrapSource::Fresh).await?;
        self.open_prepared(
            bootstrap,
            fallback_policy,
            CliSessionHostConfig::new(
                execution_mode,
                standard_context_approach,
                default_rlm_termination_for_mode(execution_mode),
            ),
        )
        .await
    }

    pub(crate) async fn resume(
        &self,
        identifier: &str,
        fallback_policy: SessionPolicy,
        execution_mode: ExecutionMode,
        standard_context_approach: Option<StandardContextApproach>,
    ) -> Result<OpenedCliLashSession> {
        let bootstrap = SessionBootstrap::open(
            SessionBootstrapSource::from_resume_arg(Some(identifier.to_string())).await,
        )
        .await?;
        let persisted_host_config = bootstrap.persisted_host_config();
        let host_config = CliSessionHostConfig::new(
            persisted_host_config
                .as_ref()
                .map(|config| config.execution_mode)
                .unwrap_or(execution_mode),
            persisted_host_config
                .as_ref()
                .and_then(|config| config.standard_context_approach.clone())
                .or(standard_context_approach),
            persisted_host_config
                .as_ref()
                .and_then(|config| config.rlm_termination)
                .or_else(|| default_rlm_termination_for_mode(execution_mode)),
        );
        self.open_prepared(bootstrap, fallback_policy, host_config)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, TempDirGuard, env_lock};
    use lash_core::SessionMeta;

    #[test]
    fn host_config_round_trips_rlm_termination_and_accepts_old_files() -> Result<()> {
        let temp = TempDirGuard::new("lash-cli-host-config");
        save_host_config(
            temp.path(),
            "rlm.db",
            &CliSessionHostConfig::new(
                ExecutionMode::Rlm,
                None,
                Some(RlmTerminationMode::SubmitRequired),
            ),
        )?;
        let loaded = load_host_config(temp.path(), "rlm.db").expect("saved host config");
        assert_eq!(loaded.execution_mode, ExecutionMode::Rlm);
        assert_eq!(
            loaded.rlm_termination,
            Some(RlmTerminationMode::SubmitRequired)
        );

        std::fs::write(
            host_config_path(temp.path(), "old.db"),
            r#"{"execution_mode":"rlm","standard_context_approach":null}"#,
        )?;
        let old = load_host_config(temp.path(), "old.db").expect("old host config");
        assert_eq!(old.execution_mode, ExecutionMode::Rlm);
        assert_eq!(old.rlm_termination, None);
        Ok(())
    }

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
                ExecutionMode::Rlm,
                Some(StandardContextApproach::default()),
            )
            .await?;

        assert!(!opened.session.session_id().is_empty());
        assert!(opened.bootstrap.artifacts_db_file().is_file());
        assert!(opened.bootstrap.effects_db_file().is_file());
        assert!(opened.bootstrap.processes_db_file().is_file());
        assert!(opened.bootstrap.process_env_db_file().is_file());
        assert!(opened.bootstrap.triggers_db_file().is_file());
        let host_config =
            load_host_config(opened.bootstrap.sessions_dir(), opened.bootstrap.filename())
                .expect("saved host config");
        assert_eq!(
            host_config.rlm_termination,
            Some(RlmTerminationMode::ProseOrSubmit)
        );
        Ok(())
    }

    #[tokio::test]
    async fn file_backed_store_creates_wal_file() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-cli-store-wal");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let db_path = temp.path().join("session.db");
        let store = Store::open(&db_path).await.expect("store");

        store
            .save_session_meta(SessionMeta {
                session_id: "s1".to_string(),
                session_name: "demo".to_string(),
                created_at: "2026-03-26T10:00:00Z".to_string(),
                model: "gpt-5".to_string(),
                cwd: Some("/tmp/demo".to_string()),
                relation: lash_core::SessionRelation::Root,
            })
            .await;

        assert!(db_path.with_extension("db-wal").exists());
    }
}
