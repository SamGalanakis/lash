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
    pub turn_injection_bridge: crate::session::TurnInjectionBridge,
    pub turn_input_injection_bridge: crate::session::TurnInputInjectionBridge,
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
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

impl RuntimeSessionHost for NoopSessionManager {}
impl RuntimeServices {
    pub fn new(plugins: Arc<PluginSession>) -> Self {
        Self::new_with_bridges(
            plugins,
            crate::session::TurnInjectionBridge::new(),
            crate::session::TurnInputInjectionBridge::new(),
        )
    }

    pub fn new_with_bridges(
        plugins: Arc<PluginSession>,
        turn_injection_bridge: crate::session::TurnInjectionBridge,
        turn_input_injection_bridge: crate::session::TurnInputInjectionBridge,
    ) -> Self {
        Self {
            plugins,
            turn_injection_bridge,
            turn_input_injection_bridge,
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
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
}

impl PersistentRuntimeServices {
    pub fn new(
        plugins: Arc<PluginSession>,
        store: Arc<dyn crate::store::RuntimePersistence>,
    ) -> Self {
        Self::new_with_bridges(
            plugins,
            crate::session::TurnInjectionBridge::new(),
            crate::session::TurnInputInjectionBridge::new(),
            store,
        )
    }

    pub fn new_with_bridges(
        plugins: Arc<PluginSession>,
        turn_injection_bridge: crate::session::TurnInjectionBridge,
        turn_input_injection_bridge: crate::session::TurnInputInjectionBridge,
        store: Arc<dyn crate::store::RuntimePersistence>,
    ) -> Self {
        Self(RuntimeServices {
            plugins,
            turn_injection_bridge,
            turn_input_injection_bridge,
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
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
