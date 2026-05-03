use super::*;

#[async_trait::async_trait]
impl crate::plugin::SessionSnapshotHost for RuntimeSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, crate::PluginError> {
        RuntimeSessionManager::snapshot_current(self).await
    }

    async fn snapshot_session(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, crate::PluginError> {
        RuntimeSessionManager::snapshot_session(self, session_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::ToolCatalogHost for RuntimeSessionManager {
    async fn tool_catalog(
        &self,
        session_id: &str,
    ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
        RuntimeSessionManager::tool_catalog(self, session_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::DynamicToolHost for RuntimeSessionManager {
    async fn dynamic_tool_state(
        &self,
        session_id: &str,
    ) -> Result<crate::DynamicStateSnapshot, crate::PluginError> {
        RuntimeSessionManager::dynamic_tool_state(self, session_id).await
    }

    async fn apply_dynamic_tool_state(
        &self,
        session_id: &str,
        snapshot: crate::DynamicStateSnapshot,
    ) -> Result<u64, crate::PluginError> {
        RuntimeSessionManager::apply_dynamic_tool_state(self, session_id, snapshot).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::SessionLifecycleHost for RuntimeSessionManager {
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, crate::PluginError> {
        RuntimeSessionManager::create_session(self, request).await
    }

    async fn take_first_turn_input(
        &self,
        session_id: &str,
    ) -> Result<Option<crate::PluginMessage>, crate::PluginError> {
        RuntimeSessionManager::take_first_turn_input(self, session_id).await
    }

    async fn close_session(&self, session_id: &str) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::close_session(self, session_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::TurnHost for RuntimeSessionManager {
    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, crate::PluginError> {
        RuntimeSessionManager::start_turn_stream(self, session_id, input).await
    }

    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, crate::PluginError> {
        RuntimeSessionManager::await_turn(self, turn_id).await
    }

    async fn cancel_turn(&self, turn_id: &str) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::cancel_turn(self, turn_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::TaskHost for RuntimeSessionManager {
    async fn inject_turn_input(
        &self,
        session_id: &str,
        message: crate::PluginMessage,
    ) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::inject_turn_input(self, session_id, message).await
    }

    async fn spawn_hidden_task(
        &self,
        session_id: &str,
        label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::spawn_hidden_task(self, session_id, label, task).await
    }

    async fn await_hidden_tasks(&self, session_id: &str) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::await_hidden_tasks(self, session_id).await
    }

    async fn spawn_managed_task(
        &self,
        session_id: &str,
        spec: crate::ManagedTaskSpec,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::spawn_managed_task(self, session_id, spec, task).await
    }

    async fn cancel_managed_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::cancel_managed_task(self, session_id, task_id).await
    }

    async fn register_background_task(
        &self,
        session_id: &str,
        spec: crate::ManagedTaskSpec,
        cancel: Option<crate::ManagedTaskCancel>,
    ) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::register_background_task(self, session_id, spec, cancel).await
    }

    async fn unregister_background_task(&self, session_id: &str, task_id: &str) {
        RuntimeSessionManager::unregister_background_task(self, session_id, task_id).await;
    }

    async fn complete_background_task(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        RuntimeSessionManager::complete_background_task(self, session_id, task_id, run_state).await;
    }

    async fn transition_background_task_live_state(
        &self,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        RuntimeSessionManager::transition_background_task_live_state(
            self, session_id, task_id, run_state,
        )
        .await;
    }

    async fn list_background_tasks(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::ManagedTaskStatus>, crate::PluginError> {
        RuntimeSessionManager::list_background_tasks(self, session_id).await
    }

    async fn cancel_background_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> Result<crate::ManagedTaskStatus, crate::PluginError> {
        RuntimeSessionManager::cancel_background_task(self, session_id, task_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::MonitorHost for RuntimeSessionManager {
    async fn monitor_snapshot(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        RuntimeSessionManager::monitor_snapshot(self, session_id).await
    }

    async fn take_monitor_updates(
        &self,
        session_id: &str,
    ) -> Result<crate::MonitorUpdateBatch, crate::PluginError> {
        RuntimeSessionManager::take_monitor_updates(self, session_id).await
    }

    async fn start_monitor(
        &self,
        session_id: &str,
        spec: crate::MonitorSpec,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        RuntimeSessionManager::start_monitor(self, session_id, spec).await
    }

    async fn stop_monitor(
        &self,
        session_id: &str,
        monitor_id: &str,
    ) -> Result<crate::MonitorSnapshot, crate::PluginError> {
        RuntimeSessionManager::stop_monitor(self, session_id, monitor_id).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::SessionGraphHost for RuntimeSessionManager {
    async fn append_session_nodes(
        &self,
        session_id: &str,
        request: crate::AppendSessionNodesRequest,
    ) -> Result<crate::AppendSessionNodesResult, crate::PluginError> {
        RuntimeSessionManager::append_session_nodes(self, session_id, request).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::PromptHost for RuntimeSessionManager {
    async fn prompt_user(
        &self,
        request: crate::PromptRequest,
    ) -> Result<crate::PromptResponse, crate::PluginError> {
        RuntimeSessionManager::prompt_user(self, request).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::DirectCompletionHost for RuntimeSessionManager {
    async fn direct_completion(
        &self,
        request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<crate::DirectCompletion, crate::PluginError> {
        RuntimeSessionManager::direct_completion(self, request, usage_source).await
    }
}

#[async_trait::async_trait]
impl crate::plugin::TraceHost for RuntimeSessionManager {
    async fn emit_trace_event(
        &self,
        context: lash_trace::TraceContext,
        event: lash_trace::TraceEvent,
    ) -> Result<(), crate::PluginError> {
        RuntimeSessionManager::emit_trace_event(self, context, event).await
    }
}
