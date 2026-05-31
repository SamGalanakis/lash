use super::*;

impl RuntimeTurnDriver<'_> {
    pub(super) async fn handle_llm_call_effect(
        &mut self,
        machine: &mut TurnMachine,
        id: crate::sansio::EffectId,
        request: Arc<LlmRequest>,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), RuntimeError> {
        if cancel.is_cancelled() {
            send_session_event(event_tx, SessionEvent::Done).await;
            machine.finish_with_outcome(crate::TurnOutcome::Stopped(TurnStop::Cancelled));
            return Ok(());
        }
        match self.before_llm_call(machine, &request).await {
            Ok(Some(crate::ProtocolLlmCallAction::SwitchAgentFrame { frame_id })) => {
                machine.finish_with_outcome(crate::TurnOutcome::AgentFrameSwitch { frame_id });
                return Ok(());
            }
            Ok(None) => {}
            Err(err) => {
                let err_string = err.to_string();
                if self.should_abort_for_runtime_effect_error() {
                    return Err(RuntimeError::new(
                        RuntimeErrorCode::ProtocolBeforeLlmCall,
                        err_string,
                    ));
                }
                machine.fail_turn(make_error_event(
                    "protocol_before_llm_call",
                    Some("before_llm_call_failed"),
                    err_string.clone(),
                    Some(err_string),
                ));
                return Ok(());
            }
        }
        let (result, text_streamed) = match self
            .invoke_turn_llm_effect(machine, id, request, event_tx, cancel)
            .await
        {
            Ok(result) => result,
            Err(err) => {
                self.fail_or_abort_runtime_effect_controller(machine, err)?;
                return Ok(());
            }
        };
        if let Ok(response) = &result {
            let usage = crate::runtime::effect::token_usage_from_llm(&response.usage);
            self.turn_pipeline.state_mut().last_prompt_usage =
                normalize_prompt_usage(&self.policy.provider, &usage);
            if !text_streamed {
                let prose_projector = self.session.plugins().assistant_prose_projector();
                emit_semantic_response_parts(event_tx, response, prose_projector.as_deref()).await;
            }
        }
        machine.handle_response(Response::LlmComplete {
            id,
            result,
            text_streamed,
        });
        Ok(())
    }

    pub(super) async fn handle_checkpoint_effect(
        &mut self,
        machine: &mut TurnMachine,
        id: crate::sansio::EffectId,
        checkpoint: CheckpointKind,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), RuntimeError> {
        let result = self
            .invoke_turn_checkpoint_effect(machine, id, checkpoint, event_tx, cancel)
            .await;
        match result {
            Ok(delivery) => {
                machine.handle_response(Response::Checkpoint { id, delivery });
            }
            Err(err) => {
                self.fail_or_abort_runtime_effect_controller(machine, err.into())?;
            }
        }
        Ok(())
    }

    pub(super) async fn handle_execution_surface_sync_effect(
        &mut self,
        machine: &mut TurnMachine,
        id: crate::sansio::EffectId,
        update_machine_config: bool,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), RuntimeError> {
        let result = match self
            .invoke_turn_execution_surface_sync_effect(
                machine,
                id,
                update_machine_config,
                event_tx,
                cancel,
            )
            .await
        {
            Ok(result) => result,
            Err(err) => {
                self.fail_or_abort_runtime_effect_controller(machine, err)?;
                return Ok(());
            }
        };
        machine.handle_response(Response::ExecutionSurfaceSynced { id, result });
        Ok(())
    }

    pub(super) async fn handle_tool_calls_effect(
        &mut self,
        machine: &mut TurnMachine,
        id: crate::sansio::EffectId,
        calls: Vec<crate::sansio::PendingToolCall>,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), RuntimeError> {
        if self.host.core.trace_sink.is_some() {
            for pending in &calls {
                self.emit_tool_call_started_trace(
                    machine.protocol_iteration(),
                    Some(pending.call_id.clone()),
                    pending.tool_name.clone(),
                    pending.args.clone(),
                );
            }
        }
        let results = match self
            .invoke_turn_tool_calls_effect(machine, id, calls, event_tx, cancel)
            .await
        {
            Ok(results) => results,
            Err(err) => {
                self.fail_or_abort_runtime_effect_controller(machine, err)?;
                return Ok(());
            }
        };
        if self.host.core.trace_sink.is_some() {
            for outcome in &results {
                let record = ToolCallRecord {
                    call_id: Some(outcome.call_id.clone()),
                    tool: outcome.tool_name.clone(),
                    args: outcome.args.clone(),
                    output: outcome.output.clone(),
                    duration_ms: outcome.duration_ms,
                };
                self.emit_tool_call_trace(machine.protocol_iteration(), &record);
            }
        }
        self.turn_pipeline
            .record_tool_calls(results.iter().map(|outcome| ToolCallRecord {
                call_id: Some(outcome.call_id.clone()),
                tool: outcome.tool_name.clone(),
                args: outcome.args.clone(),
                output: outcome.output.clone(),
                duration_ms: outcome.duration_ms,
            }));
        machine.handle_response(Response::ToolResults { id, results });
        Ok(())
    }

    pub(super) async fn handle_exec_code_effect(
        &mut self,
        machine: &mut TurnMachine,
        id: crate::sansio::EffectId,
        code: String,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), RuntimeError> {
        let code_correlation_id = TurnActivityId::new(format!("code:{id:?}"));
        let iteration = machine.protocol_iteration();
        if self.host.core.trace_sink.is_some() {
            self.emit_protocol_diagnostic_trace(
                iteration,
                "exec_code_started",
                serde_json::json!({
                    "code": code,
                    "code_chars": code.chars().count(),
                }),
            );
        }
        send_turn_activity(
            event_tx,
            code_correlation_id.clone(),
            TurnEvent::CodeBlockStarted {
                language: "lashlang".to_string(),
                code: code.clone(),
            },
        )
        .await;
        let exec_created_at = std::time::Instant::now();
        let result = match self
            .invoke_turn_exec_effect(machine, id, code.clone(), event_tx, cancel)
            .await
        {
            Ok(result) => result,
            Err(err) => {
                let message = err.to_string();
                send_turn_activity(
                    event_tx,
                    code_correlation_id.clone(),
                    TurnEvent::CodeBlockCompleted {
                        language: "lashlang".to_string(),
                        output: String::new(),
                        error: Some(message),
                        success: false,
                        duration_ms: exec_created_at.elapsed().as_millis() as u64,
                        tool_call_ids: Vec::new(),
                    },
                )
                .await;
                self.fail_or_abort_runtime_effect_controller(machine, err)?;
                return Ok(());
            }
        };
        match &result {
            Ok(output) => {
                send_turn_activity(
                    event_tx,
                    code_correlation_id.clone(),
                    TurnEvent::CodeBlockCompleted {
                        language: "lashlang".to_string(),
                        output: output.observations.join("\n"),
                        error: output.error.clone(),
                        success: output.error.is_none(),
                        duration_ms: output.duration_ms,
                        tool_call_ids: output
                            .tool_calls
                            .iter()
                            .filter_map(|record| record.call_id.clone())
                            .collect(),
                    },
                )
                .await;
            }
            Err(error) => {
                send_turn_activity(
                    event_tx,
                    code_correlation_id.clone(),
                    TurnEvent::CodeBlockCompleted {
                        language: "lashlang".to_string(),
                        output: String::new(),
                        error: Some(error.clone()),
                        success: false,
                        duration_ms: exec_created_at.elapsed().as_millis() as u64,
                        tool_call_ids: Vec::new(),
                    },
                )
                .await;
            }
        }
        if let Ok(output) = &result {
            if self.host.core.trace_sink.is_some() {
                let observations_text = output.observations.join("\n");
                self.emit_protocol_diagnostic_trace(
                    iteration,
                    "exec_code_completed",
                    serde_json::json!({
                        "duration_ms": output.duration_ms,
                        "output": observations_text,
                        "output_chars": observations_text.chars().count(),
                        "observation_count": output.observations.len(),
                        "observation_truncation": output.observation_truncation,
                        "error": output.error,
                        "terminal_finish": output.terminal_finish,
                        "terminal_finish_present": output.terminal_finish.is_some(),
                        "tool_call_count": output.tool_calls.len(),
                    }),
                );
                if !output.observation_truncation.is_empty() {
                    self.emit_protocol_diagnostic_trace(
                        iteration,
                        "observation_projection",
                        serde_json::json!({
                            "projections": output.observation_truncation,
                        }),
                    );
                }
            }
            self.turn_pipeline
                .record_tool_calls(output.tool_calls.iter().cloned());
        } else if let Err(error) = &result
            && self.host.core.trace_sink.is_some()
        {
            self.emit_protocol_diagnostic_trace(
                iteration,
                "exec_code_failed",
                serde_json::json!({ "error": error }),
            );
        }
        machine.handle_response(match result {
            Ok(output) => Response::ExecResult {
                id,
                result: Ok(output),
            },
            Err(error) => Response::ExecResult {
                id,
                result: Err(error),
            },
        });
        Ok(())
    }
}
