use super::*;

impl RuntimeSessionManager {
    pub(in crate::runtime::session_manager) async fn invoke_monitor_external(
        &self,
        session_id: &str,
        name: &str,
        args: serde_json::Value,
    ) -> Result<crate::ToolResult, crate::PluginError> {
        self.current
            .plugins
            .host()
            .invoke_external_for_session(session_id, name, args, Arc::new(self.clone()))
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }

    pub(in crate::runtime::session_manager) async fn ensure_registered_monitor_specs(
        &self,
        session_id: &str,
    ) -> Result<(), crate::PluginError> {
        let specs = self
            .current
            .plugins
            .host()
            .monitor_specs_for_session(session_id)
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        if specs.is_empty() {
            return Ok(());
        }
        let specs = specs
            .into_iter()
            .map(|owned| {
                serde_json::json!({
                    "plugin_id": owned.plugin_id,
                    "spec": owned.value,
                })
            })
            .collect::<Vec<_>>();
        let result = self
            .invoke_monitor_external(
                session_id,
                "monitor.register_specs",
                serde_json::json!({
                    "specs": specs,
                }),
            )
            .await?;
        if result.success {
            Ok(())
        } else {
            Err(crate::PluginError::Session(result.result.to_string()))
        }
    }

    pub(in crate::runtime::session_manager) async fn monitor_snapshot(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(session_id, "monitor.status", serde_json::json!({}))
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }

    pub(in crate::runtime::session_manager) async fn take_monitor_updates(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorUpdateBatch, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(session_id, "monitor.take_updates", serde_json::json!({}))
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor updates: {err}")))
    }

    pub(in crate::runtime::session_manager) async fn start_monitor(
        &self,
        session_id: &str,
        spec: crate::MonitorSpec,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(
                session_id,
                "monitor.start",
                serde_json::json!({ "spec": spec }),
            )
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }

    pub(in crate::runtime::session_manager) async fn stop_monitor(
        &self,
        session_id: &str,
        monitor_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(session_id).await?;
        let result = self
            .invoke_monitor_external(
                session_id,
                "monitor.stop",
                serde_json::json!({ "id": monitor_id }),
            )
            .await?;
        if !result.success {
            return Err(crate::PluginError::Session(result.result.to_string()));
        }
        serde_json::from_value(result.result)
            .map_err(|err| crate::PluginError::Session(format!("invalid monitor status: {err}")))
    }
}
