use super::*;

#[async_trait::async_trait]
impl crate::plugin::RuntimeSessionHost for RuntimeSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, crate::PluginError> {
        self.current.snapshot_current().await
    }

    async fn snapshot_session(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        self.current
            .snapshot_session(&self.managed, session_id)
            .await
    }
    async fn tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        self.current.tool_catalog(&self.managed, session_id).await
    }
    async fn tool_state(&self, session_id: &str) -> Result<crate::ToolState, crate::PluginError> {
        self.current.tool_state(&self.managed, session_id).await
    }

    async fn apply_tool_state(
        &self,
        session_id: &str,
        snapshot: crate::ToolState,
    ) -> Result<u64, crate::PluginError> {
        self.current
            .apply_tool_state(&self.managed, session_id, snapshot)
            .await
    }
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, crate::PluginError> {
        self.managed
            .create_session(&self.current, &self.usage, request)
            .await
    }

    async fn take_first_turn_input(
        &self,
        session_id: &str,
    ) -> Result<Option<crate::PluginMessage>, crate::PluginError> {
        self.managed.take_first_turn_input(session_id).await
    }

    async fn close_session(&self, session_id: &str) -> Result<(), crate::PluginError> {
        self.managed
            .close_session(&self.current, &self.usage, session_id)
            .await
    }
    async fn start_turn(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn, crate::PluginError> {
        self.managed
            .start_turn(&self.current, &self.usage, session_id, input)
            .await
    }
    async fn inject_turn_input(
        &self,
        session_id: &str,
        input: crate::InjectedTurnInput,
    ) -> Result<(), crate::PluginError> {
        self.managed.inject_turn_input(session_id, input).await
    }

    async fn start_process(
        &self,
        session_id: &str,
        registration: crate::ProcessRegistration,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.processes
            .start_process(
                &self.current,
                &self.managed,
                session_id,
                registration,
                Arc::new(self.clone()),
            )
            .await
    }

    async fn start_process_scoped(
        &self,
        session_id: &str,
        registration: crate::ProcessRegistration,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.processes
            .start_process_scoped(
                &self.current,
                &self.managed,
                session_id,
                registration,
                Arc::new(self.clone()),
                effect_metadata,
                effect_controller,
            )
            .await
    }

    async fn await_process(
        &self,
        process_id: &str,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        self.processes
            .await_process(&self.current, process_id)
            .await
    }

    async fn await_process_scoped(
        &self,
        process_id: &str,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        self.processes
            .await_process_scoped(
                &self.current,
                process_id,
                effect_metadata,
                effect_controller,
            )
            .await
    }

    async fn list_processes(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        self.processes
            .list_processes(&self.current, session_id)
            .await
    }

    async fn list_processes_scoped(
        &self,
        session_id: &str,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        self.processes
            .list_processes_scoped(
                &self.current,
                session_id,
                effect_metadata,
                effect_controller,
            )
            .await
    }

    async fn cancel_process(
        &self,
        session_id: &str,
        process_id: &str,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.processes
            .cancel_process(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                session_id,
                process_id,
            )
            .await
    }

    async fn cancel_process_scoped(
        &self,
        session_id: &str,
        process_id: &str,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.processes
            .cancel_process_scoped(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                session_id,
                process_id,
                effect_metadata,
                effect_controller,
            )
            .await
    }

    async fn cancel_all_processes(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        self.processes
            .cancel_all_processes(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                session_id,
            )
            .await
    }

    async fn validate_process_handles_visible(
        &self,
        session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        self.processes
            .validate_process_handles_visible(&self.current, &self.managed, session_id, handle_ids)
            .await
    }

    async fn transfer_process_handles(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        self.processes
            .transfer_process_handles(
                &self.current,
                &self.managed,
                from_session_id,
                to_session_id,
                handle_ids,
            )
            .await
    }

    async fn cancel_unreferenced_process_handles(
        &self,
        session_id: &str,
        keep_handle_ids: &[String],
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        self.processes
            .cancel_unreferenced_process_handles(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                session_id,
                keep_handle_ids,
            )
            .await
    }
    async fn monitor_snapshot(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.processes
            .monitor_snapshot(&self.current, Arc::new(self.clone()), session_id)
            .await
    }

    async fn start_monitor(
        &self,
        session_id: &str,
        spec: crate::MonitorSpec,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.processes
            .start_monitor(&self.current, Arc::new(self.clone()), session_id, spec)
            .await
    }

    async fn stop_monitor(
        &self,
        session_id: &str,
        monitor_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.processes
            .stop_monitor(
                &self.current,
                Arc::new(self.clone()),
                session_id,
                monitor_id,
            )
            .await
    }
    async fn append_session_nodes(
        &self,
        session_id: &str,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, crate::PluginError> {
        self.current
            .append_session_nodes(
                &self.managed,
                &self.usage,
                &self.processes,
                session_id,
                request,
            )
            .await
    }
    async fn emit_trace_event(
        &self,
        context: lash_trace::TraceContext,
        event: lash_trace::TraceEvent,
    ) -> Result<(), crate::PluginError> {
        self.current.emit_trace_event(context, event).await
    }
}
