use super::*;
use std::sync::Arc;

impl RuntimeSessionServices {
    pub(in crate::runtime::session_manager::process_runners) async fn run_process_tool_call(
        &self,
        run: ProcessToolCallRun<'_>,
    ) -> crate::ProcessAwaitOutput {
        let result = self.execute_process_tool_call(run).await;
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
        run: ProcessToolCallRun<'_>,
    ) -> Result<crate::ToolCallOutput, crate::PluginError> {
        let ProcessToolCallRun {
            registration,
            registry,
            call,
            parent_invocation,
            wake_target_scope,
            scoped_effect_controller,
            cancellation,
        } = run;
        let run_context = ProcessRunContext::builder(self)
            .surface(
                self.current
                    .plugins
                    .tool_surface(&self.current.session_id)?,
            )
            .scoped_effect_controller(scoped_effect_controller)
            .causal_invocation(parent_invocation.clone())
            .dispatch_parent_invocation(parent_invocation)
            .build()?;
        let dispatch = run_context.dispatch();
        let tool_context = crate::ToolContext::from_dispatch(Arc::clone(&dispatch))
            .prepared_call(&call)
            .async_process(registration.id.clone(), cancellation)
            .process_events(
                registration.id.clone(),
                registry,
                wake_target_scope,
                self.current.store.clone(),
                self.current.host.queued_work_poke.clone(),
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
        run_context.shutdown().await;
        Ok(outcome.record.output)
    }
}
