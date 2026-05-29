use super::*;

impl RuntimeSessionManager {
    pub(in crate::runtime::session_manager::process_runners) async fn run_process_session_turn(
        &self,
        registration: crate::ProcessRegistration,
        mut create_request: crate::SessionCreateRequest,
        turn_input: crate::TurnInput,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> crate::ProcessAwaitOutput {
        // `ProcessInput::SessionTurn` is a durable input: its `create_request` is
        // serialized into the process registry, and `SessionPolicy` persists only
        // the provider *id* — the live `ProviderHandle` is dropped on the wire by
        // design. Re-supply it from this worker/runtime's live provider before the
        // child session is built, the deployment-level binding that lets a child
        // session turn (subagent spawn, deferred work, durable recovery) run its
        // LLM calls. Mirrors `RuntimeSessionState::rebind_provider`: only fill a
        // child whose recorded provider matches ours (or is unset), never override
        // a child that explicitly named a different provider.
        self.resupply_session_turn_provider(&mut create_request);
        let child = match self
            .managed
            .create_session(&self.current, &self.usage, create_request)
            .await
        {
            Ok(child) => child,
            Err(err) => {
                return crate::ProcessAwaitOutput::from_tool_output(
                    crate::ToolCallOutput::failure(crate::ToolFailure::tool(
                        crate::ToolFailureClass::Execution,
                        "process_session_create_failed",
                        err.to_string(),
                    )),
                );
            }
        };
        let child_session_id = child.session_id.clone();
        let turn =
            self.managed
                .start_turn(&self.current, &self.usage, &child_session_id, turn_input);
        tokio::pin!(turn);
        let outcome = tokio::select! {
            _ = cancellation.cancelled() => {
                let _ = self
                    .managed
                    .close_session(&self.current, &self.usage, &child_session_id)
                    .await;
                return crate::ProcessAwaitOutput::from_tool_output(
                    crate::ToolCallOutput::cancelled(
                        crate::ToolCancellation::runtime("background session turn was cancelled"),
                    ),
                );
            }
            outcome = &mut turn => outcome,
        };
        let _ = self
            .managed
            .close_session(&self.current, &self.usage, &child_session_id)
            .await;
        match outcome {
            Ok(turn) => {
                let state = process_terminal_state_for_turn(&turn);
                crate::ProcessAwaitOutput::from_tool_output(output_from_process_turn(
                    &registration,
                    &child_session_id,
                    turn,
                    state,
                ))
            }
            Err(err) => crate::ProcessAwaitOutput::from_tool_output(
                crate::ToolCallOutput::failure(crate::ToolFailure::tool(
                    crate::ToolFailureClass::Execution,
                    "process_session_turn_failed",
                    err.to_string(),
                )),
            ),
        }
    }

    /// Re-bind the live provider handle onto a serialized session-turn
    /// `create_request` from this runtime's live provider. The wire shape of
    /// `SessionPolicy` carries only the provider *id*, so a request that round-
    /// tripped through the process registry arrives with an unconfigured handle;
    /// install ours when the recorded id agrees (or is unset). A request that
    /// already names a configured provider is left untouched.
    fn resupply_session_turn_provider(&self, create_request: &mut crate::SessionCreateRequest) {
        let Some(policy) = create_request.policy.as_mut() else {
            return;
        };
        if policy.provider.kind() != "unconfigured" {
            return;
        }
        let live = &self.current.policy.provider;
        let recorded = policy.recorded_provider_id().trim();
        if recorded.is_empty() || recorded == "unconfigured" || recorded == live.kind() {
            policy.install_provider(live.clone());
        }
    }
}

fn process_terminal_state_for_turn(turn: &crate::AssembledTurn) -> crate::ProcessTerminalState {
    match &turn.outcome {
        crate::TurnOutcome::Finished(_) | crate::TurnOutcome::AgentFrameSwitch { .. } => {
            crate::ProcessTerminalState::Completed
        }
        crate::TurnOutcome::Stopped(crate::TurnStop::Cancelled) => {
            crate::ProcessTerminalState::Cancelled
        }
        crate::TurnOutcome::Stopped(_) => crate::ProcessTerminalState::Failed,
    }
}

fn process_turn_summary(
    turn: &crate::AssembledTurn,
    state: crate::ProcessTerminalState,
) -> Option<String> {
    if state != crate::ProcessTerminalState::Failed {
        return None;
    }
    match &turn.outcome {
        crate::TurnOutcome::Stopped(
            crate::TurnStop::SubmittedError { value } | crate::TurnStop::ToolError { value, .. },
        ) => value
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned),
        _ => Some("background session turn failed".to_string()),
    }
}

fn output_from_process_turn(
    registration: &crate::ProcessRegistration,
    child_session_id: &str,
    turn: crate::AssembledTurn,
    state: crate::ProcessTerminalState,
) -> crate::ToolCallOutput {
    if state == crate::ProcessTerminalState::Cancelled {
        return crate::ToolCallOutput::cancelled(crate::ToolCancellation::runtime(
            "background session turn was cancelled",
        ));
    }
    if state == crate::ProcessTerminalState::Failed {
        return crate::ToolCallOutput::failure(crate::ToolFailure::tool(
            crate::ToolFailureClass::Execution,
            "process_session_turn_failed",
            process_turn_summary(&turn, state)
                .unwrap_or_else(|| "background session turn failed".to_string()),
        ));
    }
    crate::ToolCallOutput::success(serde_json::json!({
        "process_id": registration.id,
        "child_session_id": child_session_id,
        "turn": turn,
    }))
}
