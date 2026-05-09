use super::*;

impl BackgroundTaskCapability {
    pub(in crate::runtime::session_manager) async fn ensure_registered_monitor_specs(
        &self,
        current: &CurrentSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
    ) -> Result<(), crate::PluginError> {
        let specs = current
            .plugins
            .host()
            .monitor_specs_for_session(session_id)
            .map_err(|err| crate::PluginError::Session(err.to_string()))?;
        if specs.is_empty() {
            return Ok(());
        }
        let specs = specs
            .into_iter()
            .map(|owned| crate::OwnedMonitorSpec {
                plugin_id: Some(owned.plugin_id),
                spec: owned.value,
            })
            .collect::<Vec<_>>();
        current
            .plugins
            .invoke_external_typed::<crate::MonitorRegisterSpecsOp>(
                crate::RegisterSpecsArgs { specs },
                Some(session_id.to_string()),
                false,
                host,
            )
            .await?;
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn monitor_snapshot(
        &self,
        current: &CurrentSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(current, Arc::clone(&host), session_id)
            .await?;
        current
            .plugins
            .invoke_external_typed::<crate::MonitorStatusOp>(
                crate::MonitorEmptyArgs {},
                Some(session_id.to_string()),
                false,
                host,
            )
            .await
    }

    pub(in crate::runtime::session_manager) async fn take_monitor_updates(
        &self,
        current: &CurrentSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
    ) -> Result<crate::MonitorUpdateBatch, crate::PluginError> {
        self.ensure_registered_monitor_specs(current, Arc::clone(&host), session_id)
            .await?;
        current
            .plugins
            .invoke_external_typed::<crate::MonitorTakeUpdatesOp>(
                crate::MonitorEmptyArgs {},
                Some(session_id.to_string()),
                false,
                host,
            )
            .await
    }

    pub(in crate::runtime::session_manager) async fn start_monitor(
        &self,
        current: &CurrentSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        spec: crate::MonitorSpec,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(current, Arc::clone(&host), session_id)
            .await?;
        current
            .plugins
            .invoke_external_typed::<crate::MonitorStartOp>(
                crate::StartMonitorArgs { spec },
                Some(session_id.to_string()),
                false,
                host,
            )
            .await
    }

    pub(in crate::runtime::session_manager) async fn stop_monitor(
        &self,
        current: &CurrentSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        monitor_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.ensure_registered_monitor_specs(current, Arc::clone(&host), session_id)
            .await?;
        current
            .plugins
            .invoke_external_typed::<crate::MonitorStopOp>(
                crate::StopMonitorArgs {
                    id: monitor_id.to_string(),
                },
                Some(session_id.to_string()),
                false,
                host,
            )
            .await
    }
}
