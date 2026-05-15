use super::*;

#[async_trait::async_trait]
impl crate::plugin::SessionSnapshotHost for RuntimeSessionManager {
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
}

#[async_trait::async_trait]
impl crate::plugin::ToolCatalogHost for RuntimeSessionManager {
    async fn tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        self.current.tool_catalog(&self.managed, session_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::ToolStateHost for RuntimeSessionManager {
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
}

#[async_trait::async_trait]
impl crate::plugin::SessionLifecycleHost for RuntimeSessionManager {
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
}

#[async_trait::async_trait]
impl crate::plugin::TurnHost for RuntimeSessionManager {
    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, crate::PluginError> {
        self.managed
            .start_turn_stream(&self.usage, session_id, input)
            .await
    }

    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, crate::PluginError> {
        self.managed
            .await_turn(&self.current, &self.usage, turn_id)
            .await
    }

    async fn cancel_turn(&self, turn_id: &str) -> Result<(), crate::PluginError> {
        self.managed.cancel_turn(turn_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::TaskHost for RuntimeSessionManager {
    async fn inject_turn_input(
        &self,
        session_id: &str,
        input: crate::InjectedTurnInput,
    ) -> Result<(), crate::PluginError> {
        self.managed.inject_turn_input(session_id, input).await
    }

    async fn spawn_hidden_task(
        &self,
        session_id: &str,
        label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        self.background
            .spawn_hidden_task(&self.current, &self.managed, session_id, label, task)
            .await
    }

    async fn await_hidden_tasks(&self, session_id: &str) -> Result<(), crate::PluginError> {
        self.background
            .await_hidden_tasks(&self.current, session_id)
            .await
    }

    async fn spawn_managed_task(
        &self,
        session_id: &str,
        spec: crate::BackgroundTaskRegistration,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        self.background
            .spawn_managed_task(&self.current, &self.managed, session_id, spec, task)
            .await
    }

    async fn cancel_managed_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<(), crate::PluginError> {
        self.background
            .cancel_managed_task(&self.current, session_id, task_id)
            .await
    }

    async fn register_background_task(
        &self,
        session_id: &str,
        spec: crate::BackgroundTaskRegistration,
        cancel: Option<crate::LocalBackgroundTaskCancel>,
    ) -> Result<(), crate::PluginError> {
        self.background
            .register_background_task(&self.current, session_id, spec, cancel)
            .await
    }

    async fn unregister_background_task(&self, session_id: &str, task_id: &str) {
        self.background
            .unregister_background_task(&self.current, session_id, task_id)
            .await;
    }

    async fn complete_background_task(
        &self,
        session_id: &str,
        task_id: &str,
        state: crate::BackgroundTaskState,
    ) {
        self.background
            .complete_background_task(&self.current, session_id, task_id, state)
            .await;
    }

    async fn transition_background_task_live_state(
        &self,
        session_id: &str,
        task_id: &str,
        state: crate::BackgroundTaskState,
    ) {
        self.background
            .transition_background_task_live_state(&self.current, session_id, task_id, state)
            .await;
    }

    async fn list_background_tasks(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::BackgroundTaskRecord>, crate::PluginError> {
        self.background
            .list_background_tasks(&self.current, session_id)
            .await
    }

    async fn cancel_background_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<crate::BackgroundTaskRecord, crate::PluginError> {
        self.background
            .cancel_background_task(&self.current, Arc::new(self.clone()), session_id, task_id)
            .await
    }

    async fn cancel_all_background_tasks(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::BackgroundTaskRecord>, crate::PluginError> {
        self.background
            .cancel_all_background_tasks(&self.current, Arc::new(self.clone()), session_id)
            .await
    }

    async fn validate_async_handles_visible(
        &self,
        session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        self.background
            .validate_async_handles_visible(&self.current, &self.managed, session_id, handle_ids)
            .await
    }

    async fn transfer_async_handles(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        self.background
            .transfer_async_handles(
                &self.current,
                &self.managed,
                from_session_id,
                to_session_id,
                handle_ids,
            )
            .await
    }

    async fn cancel_unreferenced_async_handles(
        &self,
        session_id: &str,
        keep_handle_ids: &[String],
    ) -> Result<Vec<crate::BackgroundTaskRecord>, crate::PluginError> {
        self.background
            .cancel_unreferenced_async_handles(
                &self.current,
                &self.managed,
                Arc::new(self.clone()),
                session_id,
                keep_handle_ids,
            )
            .await
    }
}

#[async_trait::async_trait]
impl crate::plugin::MonitorHost for RuntimeSessionManager {
    async fn monitor_snapshot(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.background
            .monitor_snapshot(&self.current, Arc::new(self.clone()), session_id)
            .await
    }

    async fn take_monitor_updates(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorUpdateBatch, crate::PluginError> {
        self.background
            .take_monitor_updates(&self.current, Arc::new(self.clone()), session_id)
            .await
    }

    async fn start_monitor(
        &self,
        session_id: &str,
        spec: crate::MonitorSpec,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.background
            .start_monitor(&self.current, Arc::new(self.clone()), session_id, spec)
            .await
    }

    async fn stop_monitor(
        &self,
        session_id: &str,
        monitor_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        self.background
            .stop_monitor(
                &self.current,
                Arc::new(self.clone()),
                session_id,
                monitor_id,
            )
            .await
    }
}

#[async_trait::async_trait]
impl crate::plugin::SessionGraphHost for RuntimeSessionManager {
    async fn append_session_nodes(
        &self,
        session_id: &str,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, crate::PluginError> {
        self.current
            .append_session_nodes(
                &self.managed,
                &self.usage,
                &self.background,
                session_id,
                request,
            )
            .await
    }
}

#[async_trait::async_trait]
impl crate::plugin::DirectCompletionHost for RuntimeSessionManager {
    async fn direct_completion(
        &self,
        request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        self.direct
            .direct_completion(&self.current, &self.usage, request, usage_source)
            .await
    }

    async fn direct_llm_completion(
        &self,
        request: crate::LlmRequest,
        usage_source: &str,
    ) -> Result<crate::DirectLlmCompletion, crate::PluginError> {
        self.direct
            .direct_llm_completion(&self.current, &self.usage, request, usage_source)
            .await
    }
}

#[async_trait::async_trait]
impl crate::plugin::TraceHost for RuntimeSessionManager {
    async fn emit_trace_event(
        &self,
        context: lash_trace::TraceContext,
        event: lash_trace::TraceEvent,
    ) -> Result<(), crate::PluginError> {
        self.current.emit_trace_event(context, event).await
    }
}
