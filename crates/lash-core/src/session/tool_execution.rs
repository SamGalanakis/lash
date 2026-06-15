use super::execution_context::RuntimeExecutionContext;
use crate::tool_dispatch::{
    ToolCallLaunch, ToolDispatchOutcome, ToolPreparationOutcome,
    dispatch_prepared_tool_call_launch_with_execution_context,
    finalize_tool_result_with_execution_context, prepare_tool_call_with_context,
    schedule_tool_batch,
};
use crate::{
    ModelToolReturn, SessionEvent, ToolCallOutput, ToolCallRecord, ToolCancellation, ToolFailure,
    ToolFailureClass, ToolResult, TurnActivityId, TurnEvent,
};

#[derive(Clone, Debug)]
pub struct ToolInvocation {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ToolInvocationReply {
    pub output: ToolCallOutput,
    pub record: Option<ToolCallRecord>,
}

impl ToolInvocationReply {
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

#[derive(Clone, Debug)]
pub(crate) struct CompletedProtocolToolCall {
    pub completed: crate::sansio::CompletedToolCall,
    pub record: ToolCallRecord,
}

pub(crate) enum ProtocolToolCallLaunch {
    Done(CompletedProtocolToolCall),
    Pending(crate::tool_dispatch::PendingToolDispatchOutcome),
}

#[derive(Clone)]
pub(crate) struct PreparedToolRun {
    pub prepared: crate::PreparedToolCall,
    pub index: usize,
    pub parent_invocation: Option<crate::RuntimeInvocation>,
    pub activity_id: TurnActivityId,
}

impl RuntimeExecutionContext<'_> {
    fn prepared_tool_run(
        &self,
        prepared: crate::PreparedToolCall,
        index: usize,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> PreparedToolRun {
        let activity_id = TurnActivityId::new(format!("tool:{}", prepared.call_id));
        PreparedToolRun {
            prepared,
            index,
            parent_invocation,
            activity_id,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "tool execution carries explicit runtime call metadata"
    )]
    pub(crate) async fn execute_tool_call(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
        index: usize,
        replay: Option<crate::llm::types::ProviderReplayMeta>,
        parent_invocation: Option<crate::RuntimeInvocation>,
        lashlang_execution_call_site: Option<crate::ToolLashlangExecutionCallSite>,
    ) -> CompletedProtocolToolCall {
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

        let parent_invocation = parent_invocation.or_else(|| self.parent_invocation.clone());
        let mut dispatch = (*self.dispatch).clone();
        dispatch.parent_invocation = parent_invocation.clone();
        let pending = crate::sansio::PendingToolCall {
            call_id: call_id.clone(),
            tool_name: name,
            args,
            replay: replay.clone(),
        };
        let launch =
            match prepare_tool_call_with_context(&dispatch, pending, Some(call_id.clone())).await {
                ToolPreparationOutcome::Prepared(prepared) => {
                    let dispatch_context = std::sync::Arc::new(dispatch.clone());
                    let tool_context =
                        crate::ToolContext::from_dispatch(std::sync::Arc::clone(&dispatch_context))
                            .prepared_call(&prepared)
                            .cancellation_token(self.cancellation_token.clone())
                            .runtime_process_id(self.runtime_process_id.clone())
                            .parent_invocation(parent_invocation.clone())
                            .lashlang_execution_call_site(lashlang_execution_call_site.clone())
                            .build();
                    dispatch_prepared_tool_call_launch_with_execution_context(
                        dispatch_context.as_ref(),
                        prepared,
                        None,
                        tool_context,
                    )
                    .await
                }
                ToolPreparationOutcome::Completed(outcome) => ToolCallLaunch::Done(*outcome),
            };
        let mut outcome = match launch {
            ToolCallLaunch::Done(outcome) => outcome,
            ToolCallLaunch::Pending(pending) => {
                self.await_pending_tool_dispatch_outcome(
                    &call_id,
                    parent_invocation.clone(),
                    pending,
                    self.cancellation_token.clone(),
                )
                .await
            }
        };
        outcome.record.call_id = Some(call_id.clone());

        self.complete_tool_call(index, call_id, replay, outcome, tool_correlation_id)
            .await
    }

    pub(crate) async fn prepare_tool_call(
        &self,
        pending: crate::sansio::PendingToolCall,
    ) -> ToolPreparationOutcome {
        let call_id = Some(pending.call_id.clone());
        prepare_tool_call_with_context(self.dispatch.as_ref(), pending, call_id).await
    }

    pub(crate) async fn execute_prepared_tool_call_launch(
        &self,
        prepared: crate::PreparedToolCall,
        index: usize,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> crate::runtime::ToolCallLaunch {
        match Box::pin(self.execute_prepared_tool_call_launch_inner(
            prepared,
            index,
            parent_invocation,
        ))
        .await
        {
            ProtocolToolCallLaunch::Done(completed) => crate::runtime::ToolCallLaunch::Done {
                result: completed.completed,
            },
            ProtocolToolCallLaunch::Pending(pending) => crate::runtime::ToolCallLaunch::Pending {
                key: pending.key,
                pending: pending.pending,
                duration_ms: pending.duration_ms,
            },
        }
    }

    async fn execute_prepared_tool_call_launch_inner(
        &self,
        prepared: crate::PreparedToolCall,
        index: usize,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> ProtocolToolCallLaunch {
        let call_id = prepared.call_id.clone();
        let name = prepared.tool_name.clone();
        let args = prepared.args.clone();
        let replay = prepared.replay.clone();
        let parent_invocation = parent_invocation.or_else(|| self.parent_invocation.clone());
        let run = self.prepared_tool_run(prepared, index, parent_invocation);
        let prepared = run.prepared.clone();
        let _ = self
            .dispatch
            .event_tx
            .send(SessionEvent::ToolCallStart {
                call_id: Some(call_id.clone()),
                name: name.clone(),
                args: args.clone(),
            })
            .await;
        let tool_correlation_id = run.activity_id.clone();
        self.emit_turn_activity(
            tool_correlation_id.clone(),
            TurnEvent::ToolCallStarted {
                call_id: Some(call_id.clone()),
                name: name.clone(),
                args: args.clone(),
            },
        )
        .await;

        let tool_context = crate::ToolContext::from_dispatch(std::sync::Arc::clone(&self.dispatch))
            .prepared_call(&prepared)
            .cancellation_token(self.cancellation_token.clone())
            .runtime_process_id(self.runtime_process_id.clone())
            .parent_invocation(run.parent_invocation.clone())
            .build();
        let outcome = Box::pin(dispatch_prepared_tool_call_launch_with_execution_context(
            self.dispatch.as_ref(),
            prepared,
            None,
            tool_context,
        ))
        .await;
        match outcome {
            ToolCallLaunch::Done(mut outcome) => {
                outcome.record.call_id = Some(call_id.clone());
                tokio::task::yield_now().await;
                let completed = self
                    .complete_tool_call(run.index, call_id, replay, outcome, tool_correlation_id)
                    .await;
                ProtocolToolCallLaunch::Done(completed)
            }
            ToolCallLaunch::Pending(pending) => ProtocolToolCallLaunch::Pending(pending),
        }
    }

    pub(super) async fn await_process_with_cancellation(
        &self,
        process_id: &str,
        parent_invocation: Option<crate::RuntimeInvocation>,
        cancellation: Option<tokio_util::sync::CancellationToken>,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        if let Some(cancellation) = cancellation {
            tokio::select! {
                result = self.dispatch.processes.await_process(
                    process_id,
                    self.process_scope(parent_invocation.clone()),
                ) => result,
                _ = cancellation.cancelled() => {
                    let _ = self.dispatch.processes.cancel(
                        &self.dispatch.session_id,
                        process_id,
                        self.process_scope(parent_invocation.clone()),
                    ).await;
                    self.dispatch.processes.await_process(
                        process_id,
                        self.process_scope(parent_invocation),
                    ).await
                }
            }
        } else {
            self.dispatch
                .processes
                .await_process(process_id, self.process_scope(parent_invocation))
                .await
        }
    }

    pub(crate) async fn complete_tool_call(
        &self,
        _index: usize,
        call_id: String,
        replay: Option<crate::llm::types::ProviderReplayMeta>,
        outcome: ToolDispatchOutcome,
        tool_correlation_id: TurnActivityId,
    ) -> CompletedProtocolToolCall {
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
        CompletedProtocolToolCall {
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

    pub(crate) async fn pending_completion_dispatch_outcome(
        &self,
        tool_name: String,
        args: serde_json::Value,
        resolution: crate::Resolution,
        duration_ms: u64,
    ) -> ToolDispatchOutcome {
        let output = crate::tool_result::tool_output_from_completion_resolution(resolution);
        let result = finalize_tool_result_with_execution_context(
            self.dispatch.as_ref(),
            &tool_name,
            &args,
            ToolResult::from_output(output),
            duration_ms,
        )
        .await;
        let output = result.into_done_output().unwrap_or_else(|_| {
            ToolCallOutput::failure(ToolFailure::runtime(
                ToolFailureClass::Internal,
                "pending_tool_not_finalized",
                "pending tool result reached a completed-output projection path",
            ))
        });
        ToolDispatchOutcome {
            record: ToolCallRecord {
                call_id: None,
                tool: tool_name,
                args,
                output,
                duration_ms,
            },
        }
    }

    async fn await_pending_tool_dispatch_outcome(
        &self,
        call_id: &str,
        parent_invocation: Option<crate::RuntimeInvocation>,
        pending: crate::tool_dispatch::PendingToolDispatchOutcome,
        cancellation: Option<tokio_util::sync::CancellationToken>,
    ) -> ToolDispatchOutcome {
        let fallback;
        let parent = if let Some(parent) = parent_invocation.as_ref() {
            parent
        } else {
            fallback = crate::RuntimeInvocation::effect(
                crate::RuntimeScope::new(&self.dispatch.session_id),
                format!("tool:{call_id}:await"),
                crate::RuntimeEffectKind::AwaitEvent,
                format!("tool:{call_id}:await"),
            );
            &fallback
        };
        let parent_effect_id = parent.effect_id().unwrap_or("tool");
        let invocation = crate::runtime::causal::child_effect_invocation(
            parent,
            format!("{parent_effect_id}:{call_id}:await"),
            crate::RuntimeEffectKind::AwaitEvent,
            format!("{call_id}:await"),
        );
        let cancellation = cancellation.unwrap_or_else(tokio_util::sync::CancellationToken::new);
        let deadline = pending
            .pending
            .deadline
            .map(|duration| std::time::Instant::now() + duration);
        let outcome = self
            .dispatch
            .effect_controller
            .controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::AwaitEvent { key: pending.key },
                ),
                crate::RuntimeEffectLocalExecutor::await_event(cancellation, deadline),
            )
            .await;
        let resolution = match outcome.and_then(crate::RuntimeEffectOutcome::into_await_event) {
            Ok(resolution) => resolution,
            Err(err) => {
                return ToolDispatchOutcome {
                    record: ToolCallRecord {
                        call_id: None,
                        tool: pending.tool_name,
                        args: pending.args,
                        output: ToolCallOutput::failure(ToolFailure::runtime(
                            ToolFailureClass::Internal,
                            "pending_tool_completion_failed",
                            err.to_string(),
                        )),
                        duration_ms: pending.duration_ms,
                    },
                };
            }
        };
        self.pending_completion_dispatch_outcome(
            pending.tool_name,
            pending.args,
            resolution,
            pending.duration_ms,
        )
        .await
    }

    pub async fn call_tool(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
        index: usize,
    ) -> ToolInvocationReply {
        let executed = self
            .execute_tool_call(call_id, name, args, index, None, None, None)
            .await;
        let reply = ToolInvocationReply::from_output(executed.completed.output);
        reply.with_record(executed.record)
    }

    pub async fn call_tool_with_lashlang_execution_call_site(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
        index: usize,
        call_site: crate::ToolLashlangExecutionCallSite,
    ) -> ToolInvocationReply {
        let executed = self
            .execute_tool_call(call_id, name, args, index, None, None, Some(call_site))
            .await;
        let reply = ToolInvocationReply::from_output(executed.completed.output);
        reply.with_record(executed.record)
    }

    pub async fn call_tool_batch(&self, calls: Vec<ToolInvocation>) -> Vec<ToolInvocationReply> {
        let indexed_calls = calls.into_iter().enumerate().collect::<Vec<_>>();
        schedule_tool_batch(
            indexed_calls,
            |(index, _)| *index,
            |(_, call)| self.tool_scheduling(&call.name),
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
    ) -> ToolInvocationReply {
        self.start_tool_process(call_id, name, args).await
    }

    pub async fn await_tool_handle(
        &self,
        call_id: String,
        handle: serde_json::Value,
    ) -> ToolInvocationReply {
        self.await_process_handle(call_id, handle).await
    }

    pub async fn cancel_tool_handle(
        &self,
        call_id: String,
        handle: serde_json::Value,
    ) -> ToolInvocationReply {
        self.cancel_process_handle(call_id, handle).await
    }

    pub async fn signal_tool_handle(
        &self,
        call_id: String,
        handle: serde_json::Value,
        signal_name: String,
        payload: serde_json::Value,
    ) -> ToolInvocationReply {
        self.signal_process_handle(call_id, handle, signal_name, payload)
            .await
    }
}
