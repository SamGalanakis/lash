use super::*;
use std::sync::Arc;

#[async_trait::async_trait]
impl crate::runtime::effect::ProcessRunner for RuntimeSessionServices {
    async fn run_process(
        &self,
        registration: crate::ProcessRegistration,
        execution_context: crate::ProcessExecutionContext,
        registry: Arc<dyn crate::ProcessRegistry>,
        scoped_effect_controller: crate::ScopedEffectController<'_>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let input = Arc::clone(&registration.input);
        match input.as_ref() {
            crate::ProcessInput::ToolCall { call } => {
                self.run_process_tool_call(ProcessToolCallRun {
                    registration,
                    registry: Arc::clone(&registry),
                    call: call.clone(),
                    parent_invocation: execution_context.causal_invocation,
                    wake_target_scope: execution_context.wake_target_scope,
                    scoped_effect_controller,
                    cancellation,
                })
                .await
            }
            crate::ProcessInput::SessionTurn {
                create_request,
                turn_input,
                ..
            } => {
                self.run_process_session_turn(
                    registration,
                    *create_request.clone(),
                    *turn_input.clone(),
                    cancellation,
                )
                .await
            }
            crate::ProcessInput::LashlangProcess {
                module_ref,
                process_ref,
                required_surface_ref,
                process_name,
                args,
                ..
            } => {
                self.run_lashlang_process(
                    registration,
                    registry,
                    module_ref.clone(),
                    process_ref.clone(),
                    required_surface_ref.clone(),
                    process_name.clone(),
                    args.clone(),
                    execution_context,
                    scoped_effect_controller,
                    cancellation,
                )
                .await
            }
            crate::ProcessInput::External { metadata } => crate::ProcessAwaitOutput::Success {
                value: serde_json::json!({ "metadata": metadata.clone() }),
                control: None,
            },
        }
    }
}
