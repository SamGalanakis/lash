use std::sync::Arc;

use async_trait::async_trait;
use lash::{PluginHost, RuntimeSessionHost, ToolResult};
use lash_tui_extensions::TuiExtensionSession;

pub(crate) struct LegacyTuiExtensionSession<'a> {
    pub(crate) plugin_host: &'a PluginHost,
    pub(crate) session_id: &'a str,
    pub(crate) session_manager: Arc<dyn RuntimeSessionHost>,
}

#[async_trait]
impl TuiExtensionSession for LegacyTuiExtensionSession<'_> {
    async fn invoke_external(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult, String> {
        self.plugin_host
            .invoke_external_for_session(
                self.session_id,
                name,
                args,
                Arc::clone(&self.session_manager),
            )
            .await
            .map_err(|err| err.to_string())
    }
}
