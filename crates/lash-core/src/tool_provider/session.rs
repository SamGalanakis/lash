use std::sync::Arc;

use crate::plugin::{PluginError, RuntimeSessionHost, SessionHandle, SessionSnapshot};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolSessionModel {
    pub model: String,
    pub model_variant: Option<String>,
}

#[derive(Clone)]
pub struct ToolSessionControl {
    pub(super) session_id: String,
    pub(super) host: Arc<dyn RuntimeSessionHost>,
}

impl ToolSessionControl {
    pub async fn model(&self) -> Result<ToolSessionModel, PluginError> {
        let snapshot = self.snapshot_current().await?;
        Ok(ToolSessionModel {
            model: snapshot.policy.model.id,
            model_variant: snapshot.policy.model.variant,
        })
    }

    pub async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        self.snapshot(&self.session_id).await
    }

    pub async fn snapshot(
        &self,
        session_id: impl AsRef<str>,
    ) -> Result<SessionSnapshot, PluginError> {
        self.host.snapshot_session(session_id.as_ref()).await
    }

    pub async fn create_session(
        &self,
        request: crate::SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        self.host.create_session(request).await
    }

    pub async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        self.host.close_session(session_id).await
    }

    pub async fn start_turn(
        &self,
        session_id: &str,
        input: crate::TurnInput,
    ) -> Result<crate::AssembledTurn, PluginError> {
        self.host.start_turn(session_id, input).await
    }

    pub async fn tool_catalog(&self) -> Result<Vec<serde_json::Value>, PluginError> {
        self.host.tool_catalog(&self.session_id).await
    }

    pub async fn shared_tool_catalog(&self) -> Result<Arc<Vec<serde_json::Value>>, PluginError> {
        self.host.shared_tool_catalog(&self.session_id).await
    }

    pub async fn set_tools_availability(
        &self,
        names: &[String],
        availability: Option<crate::ToolAvailability>,
    ) -> Result<u64, PluginError> {
        self.host
            .set_tools_availability(&self.session_id, names, availability)
            .await
    }
}

#[async_trait::async_trait]
impl RuntimeSessionHost for ToolSessionControl {
    async fn tool_catalog(&self, session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        self.host.tool_catalog(session_id).await
    }

    async fn shared_tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Arc<Vec<serde_json::Value>>, PluginError> {
        self.host.shared_tool_catalog(session_id).await
    }

    async fn create_session(
        &self,
        request: crate::SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        ToolSessionControl::create_session(self, request).await
    }

    async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        ToolSessionControl::close_session(self, session_id).await
    }

    async fn start_turn(
        &self,
        session_id: &str,
        input: crate::TurnInput,
    ) -> Result<crate::AssembledTurn, PluginError> {
        ToolSessionControl::start_turn(self, session_id, input).await
    }
}
