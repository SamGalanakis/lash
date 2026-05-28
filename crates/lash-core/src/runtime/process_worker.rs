use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use super::effect::ProcessRunner;
use super::session_manager::RuntimeSessionManager;
use super::{EmbeddedRuntimeBuilder, RuntimeCoreConfig};
use crate::{
    LashRuntime, PluginError, PluginFactory, PluginHost, PluginStack, ProcessAwaitOutput,
    ProcessExecutionContext, ProcessInput, ProcessRegistration, ProcessRegistry,
    SessionStoreCreateRequest, SessionStoreFactory,
};

/// Deployment-local configuration for rebuilding durable process executions.
///
/// Process rows intentionally carry only portable process input and provenance.
/// Workers provide the host profile, plugins, providers, stores, secrets, and
/// host capabilities for the deployment that owns those rows.
#[derive(Clone)]
pub struct DurableProcessWorkerConfig {
    pub plugin_host: Arc<PluginHost>,
    pub runtime_core: RuntimeCoreConfig,
    pub session_policy: crate::SessionPolicy,
    pub session_store_factory: Arc<dyn SessionStoreFactory>,
    pub process_registry: Arc<dyn ProcessRegistry>,
}

impl DurableProcessWorkerConfig {
    pub fn new(
        plugin_host: Arc<PluginHost>,
        runtime_core: RuntimeCoreConfig,
        session_store_factory: Arc<dyn SessionStoreFactory>,
        process_registry: Arc<dyn ProcessRegistry>,
    ) -> Self {
        Self {
            plugin_host,
            runtime_core,
            session_policy: crate::SessionPolicy::default(),
            session_store_factory,
            process_registry,
        }
    }

    pub fn with_session_policy(mut self, policy: crate::SessionPolicy) -> Self {
        self.session_policy = policy;
        self
    }

    pub fn from_plugin_factories(
        plugin_factories: impl IntoIterator<Item = Arc<dyn PluginFactory>>,
        runtime_core: RuntimeCoreConfig,
        session_store_factory: Arc<dyn SessionStoreFactory>,
        process_registry: Arc<dyn ProcessRegistry>,
    ) -> Self {
        Self::new(
            Arc::new(PluginHost::new(plugin_factories.into_iter().collect())),
            runtime_core,
            session_store_factory,
            process_registry,
        )
    }

    pub fn from_plugin_stack(
        plugin_stack: PluginStack,
        runtime_core: RuntimeCoreConfig,
        session_store_factory: Arc<dyn SessionStoreFactory>,
        process_registry: Arc<dyn ProcessRegistry>,
    ) -> Self {
        Self::from_plugin_factories(
            plugin_stack.into_factories(),
            runtime_core,
            session_store_factory,
            process_registry,
        )
    }
}

/// Reconstructable background-process worker.
#[derive(Clone)]
pub struct DurableProcessWorker {
    config: Arc<DurableProcessWorkerConfig>,
}

impl DurableProcessWorker {
    pub fn new(config: DurableProcessWorkerConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    pub fn from_shared_config(config: Arc<DurableProcessWorkerConfig>) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DurableProcessWorkerConfig {
        &self.config
    }

    pub async fn run_process(
        &self,
        registration: ProcessRegistration,
        execution_context: ProcessExecutionContext,
        cancellation: CancellationToken,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        self.ensure_host_profile_matches(&registration)?;
        if let ProcessInput::External { metadata } = registration.input.as_ref() {
            return Ok(ProcessAwaitOutput::Success {
                value: serde_json::json!({ "metadata": metadata.clone() }),
                control: None,
            });
        }
        let session_id = registration.provenance.owner_scope.session_id.as_str();
        if session_id.is_empty() {
            return Err(PluginError::Session(format!(
                "process `{}` is missing a structured owner scope",
                registration.id
            )));
        }
        let runtime = self.rebuild_runtime(session_id).await?;
        let manager = RuntimeSessionManager::new(&runtime, true, None, None).map_err(|err| {
            PluginError::Session(format!(
                "failed to rebuild runtime session `{session_id}` for process `{}`: {err}",
                registration.id
            ))
        })?;
        Ok(manager
            .run_process(
                registration,
                execution_context,
                Arc::clone(&self.config.process_registry),
                cancellation,
            )
            .await)
    }

    pub async fn request_process_cancel(
        &self,
        process_id: &str,
        reason: Option<String>,
    ) -> Result<(), PluginError> {
        self.config
            .process_registry
            .append_event(
                process_id,
                crate::ProcessEventAppendRequest::cancel_requested(process_id, reason),
            )
            .await
            .map(|_| ())
    }

    async fn rebuild_runtime(&self, session_id: &str) -> Result<LashRuntime, PluginError> {
        let store = self
            .config
            .session_store_factory
            .create_store(&SessionStoreCreateRequest {
                session_id: session_id.to_string(),
                relation: crate::SessionRelation::Root,
                policy: self.config.session_policy.clone(),
            })
            .map_err(|err| {
                PluginError::Session(format!(
                    "failed to open session store for process worker session `{session_id}`: {err}"
                ))
            })?;
        EmbeddedRuntimeBuilder::new()
            .with_session_id(session_id.to_string())
            .with_plugin_host(self.config.plugin_host.as_ref().clone())
            .with_runtime_core(self.config.runtime_core.clone())
            .with_policy(self.config.session_policy.clone())
            .with_session_store_factory(Arc::clone(&self.config.session_store_factory))
            .with_process_registry(Arc::clone(&self.config.process_registry))
            .with_store(store)
            .build()
            .await
            .map_err(|err| {
                PluginError::Session(format!(
                    "failed to rebuild process worker runtime for session `{session_id}`: {err}"
                ))
            })
    }

    fn ensure_host_profile_matches(
        &self,
        registration: &ProcessRegistration,
    ) -> Result<(), PluginError> {
        let actual = registration.provenance.host_profile_id.as_str();
        let expected = self.config.runtime_core.host_profile_id.as_str();
        if actual.is_empty() || actual == expected {
            return Ok(());
        }
        Err(PluginError::Session(format!(
            "process `{}` was created for host profile `{actual}` but this worker is `{expected}`",
            registration.id
        )))
    }
}
