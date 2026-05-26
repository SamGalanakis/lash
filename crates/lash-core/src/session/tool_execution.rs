use super::execution_context::ModeExecutionContext;
use crate::tool_dispatch::{
    ToolDispatchOutcome, ToolPreparationOutcome, prepare_tool_call_with_context,
    schedule_tool_batch,
};
use crate::{
    ModelToolReturn, SessionEvent, ToolCallOutput, ToolCallRecord, ToolCancellation, ToolFailure,
    ToolFailureClass, TurnActivityId, TurnEvent,
};

#[derive(Clone, Debug)]
pub struct ModeToolBatchItem {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ModeToolReply {
    pub output: ToolCallOutput,
    pub record: Option<ToolCallRecord>,
}

impl ModeToolReply {
    pub fn success(value: serde_json::Value) -> Self {
        Self {
            output: ToolCallOutput::success(value),
            record: None,
        }
    }

    pub fn error(value: serde_json::Value) -> Self {
        let message = value
            .as_str()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| value.to_string());
        let mut failure = ToolFailure::tool(ToolFailureClass::Execution, "tool_error", message);
        failure.raw =
            Some(serde_json::from_value(value).unwrap_or_else(|_| {
                crate::ToolValue::String("unserializable tool error".to_string())
            }));
        Self {
            output: ToolCallOutput::failure(failure),
            record: None,
        }
    }

    pub fn from_output(output: ToolCallOutput) -> Self {
        Self {
            output,
            record: None,
        }
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::from_output(ToolCallOutput::cancelled(ToolCancellation::runtime(
            message,
        )))
    }

    pub(crate) fn with_record(mut self, record: ToolCallRecord) -> Self {
        self.record = Some(record);
        self
    }
}

fn process_tool_failure(code: &str, message: String) -> ToolCallOutput {
    ToolCallOutput::failure(ToolFailure::runtime(
        ToolFailureClass::Internal,
        code.to_string(),
        message,
    ))
}

#[derive(Clone, Debug)]
pub(crate) struct CompletedModeToolCall {
    pub index: usize,
    pub completed: crate::sansio::CompletedToolCall,
    pub record: ToolCallRecord,
}

impl ModeExecutionContext<'_> {
    pub(crate) async fn execute_tool_call(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
        index: usize,
        replay: Option<crate::llm::types::ProviderReplayMeta>,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
    ) -> CompletedModeToolCall {
        let _ = self
            .dispatch
            .event_tx
            .send(SessionEvent::ToolCallStart {
                call_id: Some(call_id.clone()),
                name: name.clone(),
                args: args.clone(),
            })
            .await;
        let tool_correlation_id = TurnActivityId::new(format!("tool:{call_id}"));
        self.emit_turn_activity(
            tool_correlation_id.clone(),
            TurnEvent::ToolCallStarted {
                call_id: Some(call_id.clone()),
                name: name.clone(),
                args: args.clone(),
            },
        )
        .await;

        let effect_metadata = effect_metadata.or_else(|| self.effect_metadata.clone());
        let mut dispatch = (*self.dispatch).clone();
        dispatch.tool_effect_metadata = effect_metadata.clone();
        let pending = crate::sansio::PendingToolCall {
            call_id: call_id.clone(),
            tool_name: name,
            args,
            replay: replay.clone(),
        };
        let mut outcome =
            match prepare_tool_call_with_context(&dispatch, pending, Some(call_id.clone())).await {
                ToolPreparationOutcome::Prepared(prepared) => {
                    self.execute_prepared_tool_process(prepared, effect_metadata)
                        .await
                }
                ToolPreparationOutcome::Completed(outcome) => *outcome,
            };
        outcome.record.call_id = Some(call_id.clone());

        let output = outcome.record.output.clone();
        let model_return = match self
            .dispatch
            .plugins
            .project_tool_result(crate::plugin::ToolResultProjectionContext {
                session_id: self.dispatch.session_id.clone(),
                tool_name: outcome.record.tool.clone(),
                args: outcome.record.args.clone(),
                output: output.clone(),
                duration_ms: outcome.record.duration_ms,
                call_id: call_id.clone(),
            })
            .await
        {
            Ok(projected) => projected,
            Err(err) => ModelToolReturn::text(
                call_id.clone(),
                outcome.record.tool.clone(),
                err.to_string(),
            ),
        };
        tokio::task::yield_now().await;
        self.emit_turn_activity(
            tool_correlation_id,
            TurnEvent::ToolCallCompleted {
                call_id: Some(call_id.clone()),
                name: outcome.record.tool.clone(),
                args: outcome.record.args.clone(),
                output: output.clone(),
                duration_ms: outcome.record.duration_ms,
            },
        )
        .await;

        let record = ToolCallRecord {
            call_id: Some(call_id.clone()),
            tool: outcome.record.tool.clone(),
            args: outcome.record.args.clone(),
            output: output.clone(),
            duration_ms: outcome.record.duration_ms,
        };
        CompletedModeToolCall {
            index,
            completed: crate::sansio::CompletedToolCall {
                call_id,
                tool_name: outcome.record.tool,
                args: outcome.record.args,
                output,
                model_return,
                duration_ms: outcome.record.duration_ms,
                replay,
            },
            record,
        }
    }

    pub(crate) async fn prepare_tool_call(
        &self,
        pending: crate::sansio::PendingToolCall,
    ) -> ToolPreparationOutcome {
        let call_id = Some(pending.call_id.clone());
        prepare_tool_call_with_context(self.dispatch.as_ref(), pending, call_id).await
    }

    pub(crate) async fn execute_prepared_tool_call(
        &self,
        prepared: crate::PreparedToolCall,
        index: usize,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
    ) -> CompletedModeToolCall {
        self.execute_prepared_tool_call_inner(prepared, index, effect_metadata)
            .await
    }

    async fn execute_prepared_tool_call_inner(
        &self,
        prepared: crate::PreparedToolCall,
        index: usize,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
    ) -> CompletedModeToolCall {
        let call_id = prepared.call_id.clone();
        let name = prepared.tool_name.clone();
        let args = prepared.args.clone();
        let replay = prepared.replay.clone();
        let _ = self
            .dispatch
            .event_tx
            .send(SessionEvent::ToolCallStart {
                call_id: Some(call_id.clone()),
                name: name.clone(),
                args: args.clone(),
            })
            .await;
        let tool_correlation_id = TurnActivityId::new(format!("tool:{call_id}"));
        self.emit_turn_activity(
            tool_correlation_id.clone(),
            TurnEvent::ToolCallStarted {
                call_id: Some(call_id.clone()),
                name: name.clone(),
                args: args.clone(),
            },
        )
        .await;

        let effect_metadata = effect_metadata.or_else(|| self.effect_metadata.clone());
        let mut outcome = self
            .execute_prepared_tool_process(prepared, effect_metadata)
            .await;
        outcome.record.call_id = Some(call_id.clone());
        tokio::task::yield_now().await;

        self.complete_tool_call(index, call_id, replay, outcome, tool_correlation_id)
            .await
    }

    async fn execute_prepared_tool_process(
        &self,
        prepared: crate::PreparedToolCall,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
    ) -> ToolDispatchOutcome {
        let started = std::time::Instant::now();
        let process_id = prepared.call_id.clone();
        let tool_name = prepared.tool_name.clone();
        let args = prepared.args.clone();
        let registration = crate::ProcessRegistration::new(
            process_id.clone(),
            crate::ProcessInput::ToolCall { call: prepared },
        );
        let execution_context = crate::ProcessExecutionContext::default()
            .with_tool_effect_metadata(effect_metadata.clone())
            .with_wake_session_id(self.dispatch.session_id.clone());
        let output = match self
            .dispatch
            .host
            .start_process(
                crate::ProcessStartRequest::new(
                    &self.dispatch.session_id,
                    registration,
                    execution_context,
                )
                .with_scope(self.process_request_scope(effect_metadata.clone())),
            )
            .await
        {
            Ok(record) => match self
                .await_foreground_tool_process(&record.id, effect_metadata)
                .await
            {
                Ok(output) => output.into_tool_output(),
                Err(err) => process_tool_failure("process_tool_await_failed", err.to_string()),
            },
            Err(err) => process_tool_failure("process_tool_start_failed", err.to_string()),
        };
        ToolDispatchOutcome {
            record: ToolCallRecord {
                call_id: Some(process_id),
                tool: tool_name,
                args,
                output,
                duration_ms: started.elapsed().as_millis() as u64,
            },
        }
    }

    async fn await_foreground_tool_process(
        &self,
        process_id: &str,
        effect_metadata: Option<crate::EffectInvocationMetadata>,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        if let Some(cancellation) = self.cancellation_token.clone() {
            tokio::select! {
                result = self.dispatch.host.await_process(
                    crate::ProcessAwaitRequest::new(process_id)
                        .with_scope(self.process_request_scope(effect_metadata.clone())),
                ) => result,
                _ = cancellation.cancelled() => {
                    let _ = self.dispatch.host.cancel_process(
                        crate::ProcessCancelRequest::new(&self.dispatch.session_id, process_id)
                            .with_scope(self.process_request_scope(effect_metadata.clone())),
                    ).await;
                    self.dispatch.host.await_process(
                        crate::ProcessAwaitRequest::new(process_id)
                            .with_scope(self.process_request_scope(effect_metadata)),
                    ).await
                }
            }
        } else {
            self.dispatch
                .host
                .await_process(
                    crate::ProcessAwaitRequest::new(process_id)
                        .with_scope(self.process_request_scope(effect_metadata)),
                )
                .await
        }
    }

    pub(crate) async fn complete_tool_call(
        &self,
        index: usize,
        call_id: String,
        replay: Option<crate::llm::types::ProviderReplayMeta>,
        outcome: ToolDispatchOutcome,
        tool_correlation_id: TurnActivityId,
    ) -> CompletedModeToolCall {
        let output = outcome.record.output.clone();
        let projection_output = output.clone();
        let projection_tool_name = outcome.record.tool.clone();
        let projection_args = outcome.record.args.clone();
        let projection_duration_ms = outcome.record.duration_ms;
        let projection_call_id = call_id.clone();
        tokio::task::yield_now().await;
        let plugins = std::sync::Arc::clone(&self.dispatch.plugins);
        let projection_context = crate::plugin::ToolResultProjectionContext {
            session_id: self.dispatch.session_id.clone(),
            tool_name: projection_tool_name,
            args: projection_args,
            output: projection_output,
            duration_ms: projection_duration_ms,
            call_id: projection_call_id,
        };
        let model_return = match plugins.project_tool_result(projection_context).await {
            Ok(projected) => projected,
            Err(err) => ModelToolReturn::text(
                call_id.clone(),
                outcome.record.tool.clone(),
                err.to_string(),
            ),
        };

        self.emit_turn_activity(
            tool_correlation_id,
            TurnEvent::ToolCallCompleted {
                call_id: Some(call_id.clone()),
                name: outcome.record.tool.clone(),
                args: outcome.record.args.clone(),
                output: output.clone(),
                duration_ms: outcome.record.duration_ms,
            },
        )
        .await;

        let record = ToolCallRecord {
            call_id: Some(call_id.clone()),
            tool: outcome.record.tool.clone(),
            args: outcome.record.args.clone(),
            output: output.clone(),
            duration_ms: outcome.record.duration_ms,
        };
        CompletedModeToolCall {
            index,
            completed: crate::sansio::CompletedToolCall {
                call_id,
                tool_name: outcome.record.tool,
                args: outcome.record.args,
                output,
                model_return,
                duration_ms: outcome.record.duration_ms,
                replay,
            },
            record,
        }
    }

    pub async fn call_tool(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
        index: usize,
    ) -> ModeToolReply {
        if name == "list_process_handles" {
            let live_processes = self.live_processes().await;
            return self.list_process_handles(live_processes);
        }
        if name == "monitor" {
            return self.start_monitor_handle_call(call_id, args, index).await;
        }
        let executed = self
            .execute_tool_call(call_id, name, args, index, None, None)
            .await;
        let reply = ModeToolReply::from_output(executed.completed.output);
        reply.with_record(executed.record)
    }

    pub async fn call_tool_batch(&self, calls: Vec<ModeToolBatchItem>) -> Vec<ModeToolReply> {
        let indexed_calls = calls.into_iter().enumerate().collect::<Vec<_>>();
        schedule_tool_batch(
            indexed_calls,
            |(index, _)| *index,
            |(_, call)| self.tool_execution_mode(&call.name),
            |(index, call)| {
                let ctx = self.clone();
                async move { ctx.call_tool(call.id, call.name, call.args, index).await }
            },
        )
        .await
    }

    pub async fn start_tool_call(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
    ) -> ModeToolReply {
        if name == "monitor" {
            return self.start_monitor_handle_call(call_id, args, 0).await;
        }
        self.start_tool_process(call_id, name, args).await
    }

    pub async fn await_tool_handle(
        &self,
        _call_id: String,
        handle: serde_json::Value,
    ) -> ModeToolReply {
        self.await_process_handle(handle).await
    }

    pub async fn cancel_tool_handle(
        &self,
        _call_id: String,
        handle: serde_json::Value,
    ) -> ModeToolReply {
        self.cancel_process_handle(handle).await
    }
}
