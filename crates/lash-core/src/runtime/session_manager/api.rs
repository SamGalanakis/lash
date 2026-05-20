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
        request: crate::ProcessStartRequest<'_>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.processes
            .start_process(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                request,
            )
            .await
    }

    async fn await_process(
        &self,
        request: crate::ProcessAwaitRequest<'_>,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        self.processes.await_process(&self.current, request).await
    }

    async fn list_process_handles(
        &self,
        request: crate::ProcessListRequest<'_>,
    ) -> Result<Vec<crate::ProcessHandleGrantEntry>, crate::PluginError> {
        self.processes
            .list_process_handles(&self.current, request)
            .await
    }

    async fn cancel_process(
        &self,
        request: crate::ProcessCancelRequest<'_>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.processes
            .cancel_process(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                request,
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
        request: crate::ProcessTransferRequest<'_>,
    ) -> Result<(), crate::PluginError> {
        self.processes
            .transfer_process_handles(&self.current, &self.managed, request)
            .await
    }

    async fn cancel_unreferenced_process_handles(
        &self,
        request: crate::ProcessCleanupRequest<'_>,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        self.processes
            .cancel_unreferenced_process_handles(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                request,
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
