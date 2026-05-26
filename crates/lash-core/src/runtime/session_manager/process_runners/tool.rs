use super::*;
use std::sync::Arc;

impl RuntimeSessionManager {
    pub(in crate::runtime::session_manager::process_runners) async fn run_process_tool_call(
        &self,
        registration: crate::ProcessRegistration,
        registry: Arc<dyn crate::ProcessRegistry>,
        call: crate::PreparedToolCall,
        tool_effect_metadata: Option<crate::EffectInvocationMetadata>,
        wake_target_scope_key: Option<String>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let result = self
            .execute_process_tool_call(
                registration,
                registry,
                call,
                tool_effect_metadata,
                wake_target_scope_key,
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
        tool_effect_metadata: Option<crate::EffectInvocationMetadata>,
        wake_target_scope_key: Option<String>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<crate::ToolCallOutput, crate::PluginError> {
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<crate::SessionEvent>(64);
        let event_drain = tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
        let host = Arc::new(self.clone()) as Arc<dyn crate::plugin::RuntimeSessionHost>;
        let direct_completions = crate::DirectCompletionClient::runtime(
            Arc::new(self.clone()),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.current.host.core.effect_controller,
            )),
            tool_effect_metadata
                .as_ref()
                .and_then(|metadata: &crate::EffectInvocationMetadata| metadata.turn_id.clone()),
            self.current.turn_lease.clone(),
        );
        let dispatch = crate::tool_dispatch::ToolDispatchContext {
            plugins: Arc::clone(&self.current.plugins),
            tools: self.current.plugins.tools(),
            surface: self.current.plugins.tool_surface(
                &self.current.session_id,
                self.current.policy.execution_mode.clone(),
            )?,
            host: Arc::clone(&host),
            processes: Arc::new(self.clone()),
            effect_controller: crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.current.host.core.effect_controller,
            )),
            direct_completions: direct_completions.clone(),
            tool_effect_metadata: None,
            session_id: self.current.session_id.clone(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
            attachment_store: Arc::clone(&self.current.host.core.attachment_store),
            turn_context: crate::TurnContext::default(),
        };
        let tool_context = crate::ToolContext::new(
            self.current.session_id.clone(),
            host,
            Arc::new(self.clone()),
            crate::TurnContext::default(),
            Arc::clone(&self.current.host.core.attachment_store),
            direct_completions,
            dispatch.effect_controller.clone_scoped(),
            Some(call.call_id.clone()),
        )
        .with_async_process(registration.id.clone(), cancellation)
        .with_process_events(registration.id.clone(), registry, wake_target_scope_key)
        .with_tool_effect_metadata(tool_effect_metadata);
        let outcome = crate::tool_dispatch::dispatch_prepared_tool_call_with_execution_context(
            &dispatch,
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
