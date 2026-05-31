use std::sync::Arc;

use super::*;

#[derive(Debug, thiserror::Error)]
pub enum PluginActionInvokeError {
    #[error("unknown plugin action `{0}`")]
    Unknown(String),
    #[error("unknown plugin session `{0}`")]
    UnknownSession(String),
    #[error("plugin action `{0}` requires a session")]
    MissingSession(String),
    #[error("plugin action `{0}` does not accept a session")]
    UnexpectedSession(String),
    #[error("plugin session registry is unavailable")]
    SessionRegistryPoisoned,
}

#[derive(Clone)]
pub struct RuntimeServices {
    pub plugins: Arc<PluginSession>,
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub lashlang_artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    pub(crate) store: Option<Arc<dyn crate::store::RuntimePersistence>>,
}

#[derive(Clone)]
pub struct PersistentRuntimeServices(RuntimeServices);

impl std::ops::Deref for PersistentRuntimeServices {
    type Target = RuntimeServices;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(crate) struct NoopSessionManager;

impl SessionStateService for NoopSessionManager {}
impl SessionLifecycleService for NoopSessionManager {}
impl SessionGraphService for NoopSessionManager {}
impl RuntimeServices {
    pub fn new(plugins: Arc<PluginSession>) -> Self {
        Self {
            plugins,
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            lashlang_artifact_store: lashlang::global_in_memory_lashlang_artifact_store(),
            store: None,
        }
    }

    pub(crate) fn with_attachment_store(
        mut self,
        attachment_store: Arc<dyn crate::AttachmentStore>,
    ) -> Self {
        self.attachment_store = attachment_store;
        self
    }

    pub(crate) fn with_lashlang_artifact_store(
        mut self,
        artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    ) -> Self {
        self.lashlang_artifact_store = artifact_store;
        self
    }
}

impl PersistentRuntimeServices {
    pub fn new(
        plugins: Arc<PluginSession>,
        store: Arc<dyn crate::store::RuntimePersistence>,
    ) -> Self {
        Self(RuntimeServices {
            plugins,
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            lashlang_artifact_store: lashlang::global_in_memory_lashlang_artifact_store(),
            store: Some(store),
        })
    }

    pub(crate) fn into_runtime_services(self) -> RuntimeServices {
        self.0
    }

    pub fn store(&self) -> Arc<dyn crate::store::RuntimePersistence> {
        self.0
            .store
            .as_ref()
            .expect("persistent runtime services must carry a store")
            .clone()
    }
}
