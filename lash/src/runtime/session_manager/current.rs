use super::*;

impl RuntimeSessionManager {
    pub(in crate::runtime::session_manager) async fn current_snapshot_for_store_write(
        &self,
    ) -> SessionSnapshot {
        let mut state = self.current.snapshot.to_snapshot();
        if let Some(store) = &self.current.store
            && let Err(err) =
                crate::store::refresh_persisted_session_state(store.as_ref(), &mut state).await
        {
            tracing::warn!("failed to refresh persisted session state: {err}");
        }
        super::normalize_session_graph(&mut state);
        state
    }

    pub(in crate::runtime::session_manager) async fn snapshot_by_id(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        if session_id == self.current.session_id {
            let mut snapshot = self.current.snapshot.to_snapshot();
            super::normalize_session_graph(&mut snapshot);
            self.enrich_current_snapshot(&mut snapshot);
            return Ok(snapshot);
        }
        let runtime = {
            let registry = self.managed.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        Ok(runtime.export_persisted_state())
    }

    pub(in crate::runtime::session_manager) async fn tool_catalog_by_id(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        if session_id == self.current.session_id {
            if let Some(runtime) = self.managed.registry.lock().await.get(session_id).cloned() {
                let runtime = runtime.lock().await;
                return Ok(runtime.active_tool_catalog());
            }
            if self.current.plugins.dynamic_tools().is_some() {
                return Ok(self
                    .current
                    .plugins
                    .tool_catalog(session_id, self.current.policy.execution_mode.clone()));
            }
            return Ok(self.current.tool_catalog.as_ref().clone());
        }
        let runtime = {
            let registry = self.managed.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        Ok(runtime.active_tool_catalog())
    }

    pub(in crate::runtime::session_manager) fn enrich_current_snapshot(
        &self,
        snapshot: &mut SessionSnapshot,
    ) {
        if let Some(dynamic_tools) = self.current.plugins.dynamic_tools() {
            let dynamic_state = dynamic_tools.export_state();
            snapshot.dynamic_state_generation = Some(dynamic_state.base_generation);
            snapshot.dynamic_state_snapshot = Some(dynamic_state);
        } else {
            snapshot.dynamic_state_generation = None;
            snapshot.dynamic_state_snapshot = None;
        }
        snapshot.plugin_snapshot = self.current.plugins.snapshot().ok();
        snapshot.plugin_snapshot_revision =
            Some(self.current.plugins.snapshot_revision_fingerprint());
    }

    pub(in crate::runtime::session_manager) fn current_dynamic_tools(
        &self,
    ) -> Result<Arc<crate::DynamicToolProvider>, crate::PluginError> {
        self.current.plugins.dynamic_tools().ok_or_else(|| {
            crate::PluginError::Session("dynamic tools are unavailable in this session".to_string())
        })
    }

    pub(in crate::runtime::session_manager) async fn snapshot_current(
        &self,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        let mut snapshot = self.current.snapshot.to_snapshot();
        super::normalize_session_graph(&mut snapshot);
        self.enrich_current_snapshot(&mut snapshot);
        Ok(snapshot)
    }

    pub(in crate::runtime::session_manager) async fn snapshot_session(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        self.snapshot_by_id(session_id).await
    }

    pub(in crate::runtime::session_manager) async fn tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        self.tool_catalog_by_id(session_id).await
    }

    pub(in crate::runtime::session_manager) async fn dynamic_tool_state(
        &self,
        session_id: &str,
    ) -> Result<crate::DynamicStateSnapshot, crate::PluginError> {
        if session_id == self.current.session_id {
            if let Some(runtime) = self.managed.registry.lock().await.get(session_id).cloned() {
                let runtime = runtime.lock().await;
                return runtime
                    .dynamic_tool_state()
                    .map_err(|err| crate::PluginError::Session(err.to_string()));
            }
            return Ok(self.current_dynamic_tools()?.export_state());
        }

        let runtime = {
            let registry = self.managed.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let runtime = runtime.lock().await;
        runtime
            .dynamic_tool_state()
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }

    pub(in crate::runtime::session_manager) async fn apply_dynamic_tool_state(
        &self,
        session_id: &str,
        snapshot: crate::DynamicStateSnapshot,
    ) -> Result<u64, crate::PluginError> {
        if session_id == self.current.session_id {
            if let Some(runtime) = self.managed.registry.lock().await.get(session_id).cloned() {
                let mut runtime = runtime.lock().await;
                return runtime
                    .apply_dynamic_tool_state(snapshot)
                    .await
                    .map_err(|err| crate::PluginError::Session(err.to_string()));
            }
            let dynamic_tools = self.current_dynamic_tools()?;
            return dynamic_tools
                .apply_state(snapshot)
                .map_err(|err| crate::PluginError::Session(err.to_string()));
        }

        let runtime = {
            let registry = self.managed.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let mut runtime = runtime.lock().await;
        runtime
            .apply_dynamic_tool_state(snapshot)
            .await
            .map_err(|err| crate::PluginError::Session(err.to_string()))
    }

    pub(in crate::runtime::session_manager) async fn emit_trace_event(
        &self,
        context: lash_trace::TraceContext,
        event: lash_trace::TraceEvent,
    ) -> Result<(), crate::PluginError> {
        crate::trace::emit_trace(
            &self.current.host.core.trace_sink,
            &self.current.host.core.trace_context,
            context.for_session(self.current.session_id.clone()),
            event,
        );
        Ok(())
    }
}
