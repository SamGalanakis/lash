use super::*;
use std::sync::Arc;

impl RuntimeSessionServices {
    pub(in crate::runtime::session_manager::process_runners) async fn run_process_tool_call(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        call: crate::PreparedToolCall,
        parent_invocation: Option<crate::RuntimeInvocation>,
        wake_target_scope: Option<crate::ProcessScope>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let result = self
            .execute_process_tool_call(
                registration,
                registry,
                call,
                parent_invocation,
                wake_target_scope,
                cancellation,
            )
            .await;
        match result {
            Ok(output) => crate::ProcessAwaitOutput::from_tool_output(output),
            Err(err) => crate::ProcessAwaitOutput::from_tool_output(
                crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
                    crate::ToolFailureClass::Internal,
                    "process_tool_failed",
                    err.to_string(),
                )),
            ),
        }
    }

    async fn execute_process_tool_call(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        call: crate::PreparedToolCall,
        parent_invocation: Option<crate::RuntimeInvocation>,
        wake_target_scope: Option<crate::ProcessScope>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<crate::ToolCallOutput, crate::PluginError> {
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<crate::SessionEvent>(64);
        let event_drain = tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
        let services = Arc::new(self.clone());
        let direct_completions = services.direct_completion_client(
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.current.host.core.effect_controller,
            )),
            parent_invocation
                .as_ref()
                .and_then(|invocation: &crate::RuntimeInvocation| invocation.scope.turn_id.clone()),
            self.current.turn_lease.clone(),
        );
        let dispatch = Arc::new(crate::tool_dispatch::ToolDispatchContext {
            plugins: Arc::clone(&self.current.plugins),
            tools: self.current.plugins.tools(),
            surface: self
                .current
                .plugins
                .tool_surface(&self.current.session_id)?,
            sessions: services.state_service(),
            session_lifecycle: services.lifecycle_service(),
            session_graph: services.graph_service(),
            processes: services.process_service(),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.current.host.core.effect_controller,
            )),
            direct_completions: direct_completions.clone(),
            parent_invocation: parent_invocation.clone(),
            session_id: self.current.session_id.clone(),
            agent_frame_id: String::new(),
            event_tx,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            attachment_store: Arc::clone(&self.current.host.core.attachment_store),
            turn_context: crate::TurnContext::default(),
        });
        let tool_context = crate::ToolContext::from_dispatch(Arc::clone(&dispatch))
            .prepared_call(&call)
            .async_process(registration.id.clone(), cancellation)
            .process_events(
                registration.id.clone(),
                registry,
                wake_target_scope,
                self.current.store.clone(),
            )
            .build();
        let outcome = crate::tool_dispatch::dispatch_prepared_tool_call_with_execution_context(
            dispatch.as_ref(),
            call,
            None,
            tool_context,
        )
        .await;
        drop(dispatch);
        let _ = event_drain.await;
        Ok(outcome.record.output)
    }
}
