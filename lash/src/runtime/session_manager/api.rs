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
impl crate::plugin::DynamicToolHost for RuntimeSessionManager {
    async fn dynamic_tool_state(
        &self,
        session_id: &str,
    ) -> Result<crate::DynamicStateSnapshot, crate::PluginError> {
        self.current
            .dynamic_tool_state(&self.managed, session_id)
            .await
    }

    async fn apply_dynamic_tool_state(
        &self,
        session_id: &str,
        snapshot: crate::DynamicStateSnapshot,
    ) -> Result<u64, crate::PluginError> {
        self.current
            .apply_dynamic_tool_state(&self.managed, session_id, snapshot)
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
        message: crate::PluginMessage,
    ) -> Result<(), crate::PluginError> {
        self.managed.inject_turn_input(session_id, message).await
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
        spec: crate::ManagedTaskSpec,
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
        spec: crate::ManagedTaskSpec,
        cancel: Option<crate::ManagedTaskCancel>,
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
        run_state: crate::ManagedRunState,
    ) {
        self.background
            .complete_background_task(&self.current, session_id, task_id, run_state)
            .await;
    }

    async fn transition_background_task_live_state(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        self.background
            .transition_background_task_live_state(&self.current, session_id, task_id, run_state)
            .await;
    }

    async fn list_background_tasks(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::ManagedTaskStatus>, crate::PluginError> {
        self.background
            .list_background_tasks(&self.current, session_id)
            .await
    }

    async fn cancel_background_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<crate::ManagedTaskStatus, crate::PluginError> {
        self.background
            .cancel_background_task(&self.current, Arc::new(self.clone()), session_id, task_id)
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
impl crate::plugin::PromptHost for RuntimeSessionManager {
    async fn prompt_user(
        &self,
        request: crate::PromptRequest,
    ) -> Result<crate::PromptResponse, crate::PluginError> {
        self.prompt.prompt_user(request).await
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
