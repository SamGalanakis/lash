use std::sync::Arc;

use super::*;

#[derive(Debug, thiserror::Error)]
pub enum ExternalInvokeError {
    #[error("unknown external invoke `{0}`")]
    Unknown(String),
    #[error("unknown plugin session `{0}`")]
    UnknownSession(String),
    #[error("external invoke `{0}` requires a session")]
    MissingSession(String),
    #[error("external invoke `{0}` does not accept a session")]
    UnexpectedSession(String),
    #[error("plugin session registry is unavailable")]
    SessionRegistryPoisoned,
}

#[derive(Clone)]
pub struct RuntimeServices {
    pub plugins: Arc<PluginSession>,
    pub turn_injection_bridge: crate::session::TurnInjectionBridge,
    pub turn_input_injection_bridge: crate::session::TurnInputInjectionBridge,
    pub(crate) store: Option<Arc<dyn crate::store::RuntimeStore>>,
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

#[async_trait::async_trait]
impl SessionManager for NoopSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        Err(PluginError::Session(
            "session snapshots are unavailable in this runtime".to_string(),
        ))
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Err(PluginError::Session(
            "session lookup is unavailable in this runtime".to_string(),
        ))
    }

    async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        Err(PluginError::Session(
            "tool catalogs are unavailable in this runtime".to_string(),
        ))
    }

    async fn create_session(
        &self,
        _request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        Err(PluginError::Session(
            "session creation is unavailable in this runtime".to_string(),
        ))
    }

    async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "session closing is unavailable in this runtime".to_string(),
        ))
    }

    async fn start_turn_stream(
        &self,
        _session_id: &str,
        _input: TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
        Err(PluginError::Session(
            "session execution is unavailable in this runtime".to_string(),
        ))
    }

    async fn await_turn(&self, _turn_id: &str) -> Result<AssembledTurn, PluginError> {
        Err(PluginError::Session(
            "session execution is unavailable in this runtime".to_string(),
        ))
    }

    async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "session execution is unavailable in this runtime".to_string(),
        ))
    }
}

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
            store: None,
        }
    }
}

impl PersistentRuntimeServices {
    pub fn new(plugins: Arc<PluginSession>, store: Arc<dyn crate::store::RuntimeStore>) -> Self {
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
        store: Arc<dyn crate::store::RuntimeStore>,
    ) -> Self {
        Self(RuntimeServices {
            plugins,
            turn_injection_bridge,
            turn_input_injection_bridge,
            store: Some(store),
        })
    }

    pub(crate) fn into_runtime_services(self) -> RuntimeServices {
        self.0
    }

    pub fn store(&self) -> Arc<dyn crate::store::RuntimeStore> {
        self.0
            .store
            .as_ref()
            .expect("persistent runtime services must carry a store")
            .clone()
    }
}
