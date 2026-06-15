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
            scoped_effect_controller,
            cancellation,
        } = run;
        let await_parent_invocation = parent_invocation.clone();
        let await_cancellation = cancellation.clone();
        let run_context = ProcessRunContext::builder(self)
            .tool_catalog(
                self.current
                    .plugins
                    .resolved_tool_catalog(&self.current.session_id)?,
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
                self.current.store.clone(),
                self.current.host.session_store_factory.clone(),
                self.current.host.queued_work_poke.clone(),
            )
            .build();
        let launch = Box::pin(
            crate::tool_dispatch::dispatch_prepared_tool_call_launch_with_execution_context(
                dispatch.as_ref(),
                call,
                None,
                tool_context,
            ),
        )
        .await;
        let output = match launch {
            crate::tool_dispatch::ToolCallLaunch::Done(outcome) => outcome.record.output,
            crate::tool_dispatch::ToolCallLaunch::Pending(pending) => {
                let fallback;
                let parent = if let Some(parent) = await_parent_invocation.as_ref() {
                    parent
                } else {
                    fallback = crate::RuntimeInvocation::effect(
                        crate::RuntimeScope::new(&self.current.session_id),
                        format!(
                            "process:{}:tool:{}:await",
                            registration.id, pending.tool_name
                        ),
                        crate::RuntimeEffectKind::AwaitEvent,
                        format!(
                            "process:{}:tool:{}:await",
                            registration.id, pending.tool_name
                        ),
                    );
                    &fallback
                };
                let invocation = crate::runtime::causal::child_effect_invocation(
                    parent,
                    format!(
                        "process:{}:tool:{}:await",
                        registration.id, pending.tool_name
                    ),
                    crate::RuntimeEffectKind::AwaitEvent,
                    format!(
                        "process:{}:tool:{}:await",
                        registration.id, pending.tool_name
                    ),
                );
                let outcome = Box::pin(
                    dispatch.effect_controller.controller().execute_effect(
                        crate::RuntimeEffectEnvelope::new(
                            invocation,
                            crate::RuntimeEffectCommand::AwaitEvent { key: pending.key },
                        ),
                        crate::RuntimeEffectLocalExecutor::await_event(
                            await_cancellation.clone(),
                            pending
                                .pending
                                .deadline
                                .map(|duration| std::time::Instant::now() + duration),
                        ),
                    ),
                )
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
                let resolution = outcome
                    .into_await_event()
                    .map_err(|err| crate::PluginError::Session(err.to_string()))?;
                crate::tool_result::tool_output_from_completion_resolution(resolution)
            }
        };
        drop(dispatch);
        run_context.shutdown().await;
        Ok(output)
    }
}
