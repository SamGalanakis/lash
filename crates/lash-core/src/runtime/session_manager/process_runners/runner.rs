use super::*;
use std::sync::Arc;

#[async_trait::async_trait]
impl crate::runtime::effect::ProcessRunner for RuntimeSessionManager {
    async fn run_process(
        &self,
        registration: crate::ProcessRegistration,
        execution_context: crate::ProcessExecutionContext,
        registry: Arc<dyn crate::ProcessRegistry>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        let input = Arc::clone(&registration.input);
        match input.as_ref() {
            crate::ProcessInput::ToolCall { call } => {
                self.run_process_tool_call(
                    registration,
                    call.clone(),
                    execution_context.tool_effect_metadata,
                    cancellation,
                )
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
            crate::ProcessInput::Command {
                command,
                cwd,
                env,
                timeout_ms,
                persistent,
                line_event,
            } => {
                self.run_command_process(
                    registration,
                    registry,
                    command.clone(),
                    cwd.clone(),
                    env.clone(),
                    *timeout_ms,
                    *persistent,
                    line_event.clone(),
                    execution_context.wake_session_id,
                    cancellation,
                )
                .await
            }
            crate::ProcessInput::LashlangBlock {
                program,
                input,
                tool_bindings,
                timeout_ms,
                display_name: _,
            } => {
                self.run_lashlang_process(
                    registration,
                    registry,
                    program.clone(),
                    input.clone(),
                    tool_bindings.clone(),
                    *timeout_ms,
                    execution_context,
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
