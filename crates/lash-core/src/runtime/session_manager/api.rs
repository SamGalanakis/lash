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

    async fn shared_tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Arc<Vec<serde_json::Value>>, crate::PluginError> {
        self.current
            .shared_tool_catalog(&self.managed, session_id)
            .await
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

#[async_trait::async_trait]
impl crate::ProcessService for RuntimeSessionManager {
    async fn start(
        &self,
        session_id: &str,
        registration: crate::ProcessRegistration,
        options: crate::ProcessStartOptions,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.processes
            .start_process(
                &self.current,
                &self.managed,
                session_id,
                registration,
                options,
                scope,
            )
            .await
    }

    async fn await_process(
        &self,
        process_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        self.processes
            .await_process(&self.current, process_id, scope)
            .await
    }

    async fn list_visible(
        &self,
        session_id: &str,
        mode: crate::ProcessListMode,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<Vec<crate::ProcessHandleGrantEntry>, crate::PluginError> {
        self.processes
            .list_process_handles(&self.current, session_id, mode, scope)
            .await
    }

    async fn validate_visible(
        &self,
        session_id: &str,
        handle_ids: &[String],
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<(), crate::PluginError> {
        self.processes
            .validate_process_handles_visible(
                &self.current,
                &self.managed,
                session_id,
                handle_ids,
                scope,
            )
            .await
    }

    async fn cancel(
        &self,
        session_id: &str,
        process_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.processes
            .cancel_process(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                session_id,
                process_id,
                scope,
            )
            .await
    }

    async fn signal(
        &self,
        session_id: &str,
        process_id: &str,
        signal_id: String,
        payload: serde_json::Value,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessEvent, crate::PluginError> {
        self.processes
            .signal_process(
                &self.current,
                session_id,
                process_id,
                signal_id,
                payload,
                scope,
            )
            .await
    }

    async fn cancel_all(
        &self,
        session_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        self.processes
            .cancel_all_processes(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                session_id,
                scope,
            )
            .await
    }

    async fn transfer(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        process_ids: Vec<String>,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<(), crate::PluginError> {
        self.processes
            .transfer_process_handles(
                &self.current,
                &self.managed,
                from_session_id,
                to_session_id,
                process_ids,
                scope,
            )
            .await
    }

    async fn cancel_unreferenced(
        &self,
        session_id: &str,
        keep_process_ids: Vec<String>,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        self.processes
            .cancel_unreferenced_process_handles(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                session_id,
                keep_process_ids,
                scope,
            )
            .await
    }
}
