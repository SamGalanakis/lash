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
use std::collections::HashMap;

#[derive(Clone)]
pub struct ToolInvocation {
    pub id: String,
    pub tool_id: crate::ToolId,
    pub args: serde_json::Value,
    pub child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
}

impl ToolInvocation {
    pub fn new(id: impl Into<String>, tool_id: crate::ToolId, args: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            tool_id,
            args,
            child_execution_trace_hook: None,
        }
    }

    pub fn label(&self) -> String {
        self.tool_id.to_string()
    }

    pub fn with_child_execution_trace_hook(
        mut self,
        hook: crate::ToolChildExecutionTraceHook,
    ) -> Self {
        self.child_execution_trace_hook = Some(hook);
        self
    }
}

impl std::fmt::Debug for ToolInvocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolInvocation")
            .field("id", &self.id)
            .field("tool_id", &self.tool_id)
            .field("args", &self.args)
            .field(
                "child_execution_trace_hook",
                &self.child_execution_trace_hook.as_ref().map(|_| "<hook>"),
            )
            .finish()
    }
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

fn cancelled_runtime_tool_call_launch(
    call_id: String,
    tool_name: String,
    args: serde_json::Value,
    replay: Option<crate::llm::types::ProviderReplayMeta>,
) -> crate::runtime::ToolCallLaunch {
    crate::runtime::ToolCallLaunch::Done {
        result: cancelled_completed_tool_call(call_id, tool_name, args, replay),
    }
}

fn cancelled_completed_tool_call(
    call_id: String,
    tool_name: String,
    args: serde_json::Value,
    replay: Option<crate::llm::types::ProviderReplayMeta>,
) -> crate::sansio::CompletedToolCall {
    let output = ToolCallOutput::cancelled(ToolCancellation::runtime("tool call cancelled"));
    crate::sansio::CompletedToolCall {
        call_id: call_id.clone(),
        tool_name: tool_name.clone(),
        args,
        model_return: ModelToolReturn {
            call_id,
            tool_name,
            parts: vec![crate::ModelToolReturnPart::text(
                "[Tool execution cancelled]\ntool call cancelled".to_string(),
            )],
        },
        output,
        duration_ms: 0,
        replay,
    }
}

fn deterministic_tool_invocation_batch_id(calls: &[ToolInvocation]) -> String {
    let identity = calls
        .iter()
        .map(|call| {
            serde_json::json!({
                "id": call.id.clone(),
                "tool_id": call.tool_id.to_string(),
                "args": call.args.clone(),
            })
        })
        .collect::<Vec<_>>();
    let digest = crate::stable_hash::stable_json_sha256_hex(&identity)
        .unwrap_or_else(|_| format!("len-{}", calls.len()));
    format!("tool-batch:{digest}")
}

#[derive(Clone)]
pub(crate) struct PreparedToolRun {
    pub prepared: crate::PreparedToolCall,
    pub index: usize,
    pub parent_invocation: Option<crate::RuntimeInvocation>,
    pub activity_id: TurnActivityId,
}

impl RuntimeExecutionContext<'_> {
    fn tool_batch_invocation(&self, batch_id: &str) -> crate::RuntimeInvocation {
        let suffix = format!("tool-batch:{batch_id}");
        if let Some(parent) = self.parent_invocation.as_ref() {
            let parent_effect_id = parent.effect_id().unwrap_or("effect");
            return crate::runtime::causal::child_effect_invocation(
                parent,
                format!("{parent_effect_id}:{suffix}"),
                crate::RuntimeEffectKind::ToolBatch,
                suffix,
            );
        }
        let replay_key = format!("{}:{suffix}", self.execution_scope_id());
        crate::RuntimeInvocation::effect(
            crate::RuntimeScope::new(self.session_id.clone()),
            suffix,
            crate::RuntimeEffectKind::ToolBatch,
            replay_key,
        )
    }

    fn should_execute_child_tool_batch_locally(&self) -> bool {
        self.parent_invocation
            .as_ref()
            .and_then(crate::RuntimeInvocation::effect_kind)
            == Some(crate::RuntimeEffectKind::ToolBatch)
            && self
                .dispatch
                .effect_controller
                .controller()
                .durability_tier()
                == crate::DurabilityTier::Durable
    }

    pub(crate) async fn execute_prepared_tool_batch_launches(
        &self,
        batch: crate::PreparedToolBatch,
        parent_invocation: crate::RuntimeInvocation,
        child_trace_hooks: HashMap<String, crate::ToolChildExecutionTraceHook>,
    ) -> Result<crate::ToolBatchEffectOutcome, crate::RuntimeEffectControllerError> {
        let indexed_tools = batch.calls.into_iter().enumerate().collect::<Vec<_>>();
        let cancellation = self.cancellation_token.clone().unwrap_or_default();
        let tool_cancel = cancellation.child_token();
        let child_trace_hooks = std::sync::Arc::new(child_trace_hooks);
        let outcomes = schedule_tool_batch(
            indexed_tools,
            |(index, _)| *index,
            |(_, child)| self.tool_scheduling(&child.call.tool_name),
            {
                let context = self.clone();
                let cancellation = cancellation.clone();
                let tool_cancel = tool_cancel.clone();
                let child_trace_hooks = std::sync::Arc::clone(&child_trace_hooks);
                move |(index, child)| {
                    let context = context.clone().with_cancellation_token(tool_cancel.clone());
                    let cancellation = cancellation.clone();
                    let tool_cancel = tool_cancel.clone();
                    let parent_invocation = parent_invocation.clone();
                    let cancelled_tool = child.call.clone();
                    let child_execution_trace_hook =
                        child_trace_hooks.get(&child.call.call_id).cloned();
                    async move {
                        let tool_call = context.execute_prepared_tool_call_launch(
                            child.call,
                            index,
                            Some(parent_invocation),
                            child_execution_trace_hook,
                        );
                        tokio::pin!(tool_call);
                        tokio::select! {
                            biased;
                            _ = cancellation.cancelled() => {
                                tool_cancel.cancel();
                                let grace = context
                                    .dispatch
                                    .clock
                                    .sleep(std::time::Duration::from_millis(50));
                                tokio::pin!(grace);
                                tokio::select! {
                                    biased;
                                    outcome = &mut tool_call => outcome,
                                    _ = &mut grace => cancelled_runtime_tool_call_launch(
                                        cancelled_tool.call_id,
                                        cancelled_tool.tool_name,
                                        cancelled_tool.args,
                                        cancelled_tool.replay,
                                    ),
                                }
                            }
                            outcome = &mut tool_call => outcome,
                        }
                    }
                }
            },
        )
        .await;

        let triggers = self.drain_tool_trigger_outcomes().map_err(|err| {
            crate::RuntimeEffectControllerError::new("tool_trigger_outcome_drain", err.to_string())
        })?;
        Ok(crate::ToolBatchEffectOutcome {
            launches: outcomes,
            triggers,
        })
    }

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
        child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
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
        let launch = match prepare_tool_call_with_context(&dispatch, pending, Some(call_id.clone()))
            .await
        {
            ToolPreparationOutcome::Prepared(prepared) => {
                let dispatch_context = std::sync::Arc::new(dispatch.clone());
                let runtime_context = if let Some(parent_invocation) = parent_invocation.clone() {
                    self.clone().with_parent_invocation(parent_invocation)
                } else {
                    self.clone()
                };
                let mut tool_context =
                    crate::ToolContext::from_dispatch(std::sync::Arc::clone(&dispatch_context))
                        .runtime_execution_context(runtime_context)
                        .prepared_call(&prepared)
                        .cancellation_token(self.cancellation_token.clone())
                        .runtime_process_id(self.runtime_process_id.clone())
                        .parent_invocation(parent_invocation.clone())
                        .child_execution_trace_hook(child_execution_trace_hook.clone());
                if let Some(process_events) = self.process_event_context.as_ref() {
                    tool_context = tool_context.process_events(
                        process_events.process_id.clone(),
                        std::sync::Arc::clone(&process_events.registry),
                        process_events.store.clone(),
                        process_events.session_store_factory.clone(),
                        process_events.queued_work_driver.clone(),
                    );
                }
                let tool_context = tool_context.build();
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

    #[expect(
        clippy::too_many_arguments,
        reason = "tool execution carries explicit runtime call metadata"
    )]
    pub(crate) async fn execute_tool_call_by_id(
        &self,
        call_id: String,
        tool_id: crate::ToolId,
        args: serde_json::Value,
        index: usize,
        replay: Option<crate::llm::types::ProviderReplayMeta>,
        parent_invocation: Option<crate::RuntimeInvocation>,
        child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
    ) -> CompletedProtocolToolCall {
        let Some(manifest) =
            crate::tool_dispatch::resolve_callable_manifest_by_id(self.dispatch.as_ref(), &tool_id)
        else {
            let outcome = ToolDispatchOutcome {
                record: ToolCallRecord {
                    call_id: Some(call_id.clone()),
                    tool: tool_id.to_string(),
                    args,
                    output: ToolCallOutput::failure(ToolFailure::runtime(
                        ToolFailureClass::Unavailable,
                        "tool_unavailable",
                        format!("Tool id `{tool_id}` is unavailable in this session"),
                    )),
                    duration_ms: 0,
                },
            };
            let activity_id = TurnActivityId::new(format!("tool:{call_id}"));
            return self
                .complete_tool_call(index, call_id, replay, outcome, activity_id)
                .await;
        };
        self.execute_tool_call(
            call_id,
            manifest.name,
            args,
            index,
            replay,
            parent_invocation,
            child_execution_trace_hook,
        )
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
        child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
    ) -> crate::runtime::ToolCallLaunch {
        match Box::pin(self.execute_prepared_tool_call_launch_inner(
            prepared,
            index,
            parent_invocation,
            child_execution_trace_hook,
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
        child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
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

        let runtime_context = if let Some(parent_invocation) = run.parent_invocation.clone() {
            self.clone().with_parent_invocation(parent_invocation)
        } else {
            self.clone()
        };
        let mut tool_context =
            crate::ToolContext::from_dispatch(std::sync::Arc::clone(&self.dispatch))
                .runtime_execution_context(runtime_context)
                .prepared_call(&prepared)
                .cancellation_token(self.cancellation_token.clone())
                .runtime_process_id(self.runtime_process_id.clone())
                .parent_invocation(run.parent_invocation.clone())
                .child_execution_trace_hook(child_execution_trace_hook);
        if let Some(process_events) = self.process_event_context.as_ref() {
            tool_context = tool_context.process_events(
                process_events.process_id.clone(),
                std::sync::Arc::clone(&process_events.registry),
                process_events.store.clone(),
                process_events.session_store_factory.clone(),
                process_events.queued_work_driver.clone(),
            );
        }
        let tool_context = tool_context.build();
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
        let _phase = self.named_phase("process.await_handle");
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
        let cancellation = cancellation.unwrap_or_default();
        let deadline = pending
            .pending
            .deadline
            .map(|duration| self.dispatch.clock.now() + duration);
        let outcome = self
            .dispatch
            .effect_controller
            .controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::AwaitEvent { key: pending.key },
                ),
                crate::RuntimeEffectLocalExecutor::await_event_with_clock(
                    cancellation,
                    deadline,
                    std::sync::Arc::clone(&self.dispatch.clock),
                ),
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

    pub async fn call_tool_by_id(
        &self,
        call_id: String,
        tool_id: crate::ToolId,
        args: serde_json::Value,
        index: usize,
    ) -> ToolInvocationReply {
        let executed = self
            .execute_tool_call_by_id(call_id, tool_id, args, index, None, None, None)
            .await;
        let reply = ToolInvocationReply::from_output(executed.completed.output);
        reply.with_record(executed.record)
    }

    pub async fn call_tool_by_id_with_child_execution_trace_hook(
        &self,
        call_id: String,
        tool_id: crate::ToolId,
        args: serde_json::Value,
        index: usize,
        trace_hook: crate::ToolChildExecutionTraceHook,
    ) -> ToolInvocationReply {
        let executed = self
            .execute_tool_call_by_id(call_id, tool_id, args, index, None, None, Some(trace_hook))
            .await;
        let reply = ToolInvocationReply::from_output(executed.completed.output);
        reply.with_record(executed.record)
    }

    pub async fn call_tool_batch(&self, calls: Vec<ToolInvocation>) -> Vec<ToolInvocationReply> {
        if calls.is_empty() {
            return Vec::new();
        }

        let batch_id = deterministic_tool_invocation_batch_id(&calls);
        let mut replies = vec![None; calls.len()];
        let mut prepared_entries = Vec::new();

        for (index, call) in calls.into_iter().enumerate() {
            let Some(manifest) = crate::tool_dispatch::resolve_callable_manifest_by_id(
                self.dispatch.as_ref(),
                &call.tool_id,
            ) else {
                let outcome = ToolDispatchOutcome {
                    record: ToolCallRecord {
                        call_id: Some(call.id.clone()),
                        tool: call.tool_id.to_string(),
                        args: call.args,
                        output: ToolCallOutput::failure(ToolFailure::runtime(
                            ToolFailureClass::Unavailable,
                            "tool_unavailable",
                            format!("Tool id `{}` is unavailable in this session", call.tool_id),
                        )),
                        duration_ms: 0,
                    },
                };
                let completed = self
                    .complete_tool_call(
                        index,
                        call.id,
                        None,
                        outcome,
                        TurnActivityId::new(format!("tool:{}", batch_id)),
                    )
                    .await;
                replies[index] = Some(
                    ToolInvocationReply::from_output(completed.completed.output)
                        .with_record(completed.record),
                );
                continue;
            };

            let pending = crate::sansio::PendingToolCall {
                call_id: call.id.clone(),
                tool_name: manifest.name,
                args: call.args,
                replay: None,
            };
            match self.prepare_tool_call(pending).await {
                ToolPreparationOutcome::Prepared(prepared) => {
                    prepared_entries.push((index, prepared, call.child_execution_trace_hook));
                }
                ToolPreparationOutcome::Completed(outcome) => {
                    let completed = self
                        .complete_tool_call(
                            index,
                            call.id,
                            None,
                            *outcome,
                            TurnActivityId::new(format!("tool:{}", batch_id)),
                        )
                        .await;
                    replies[index] = Some(
                        ToolInvocationReply::from_output(completed.completed.output)
                            .with_record(completed.record),
                    );
                }
            }
        }

        if !prepared_entries.is_empty() {
            let invocation = self.tool_batch_invocation(&batch_id);
            let batch = crate::PreparedToolBatch::new(
                batch_id.clone(),
                prepared_entries
                    .iter()
                    .map(|(_, prepared, _)| prepared.clone())
                    .collect(),
            );
            let child_trace_hooks = prepared_entries
                .iter()
                .filter_map(|(_, prepared, hook)| {
                    hook.clone().map(|hook| (prepared.call_id.clone(), hook))
                })
                .collect();
            let envelope = crate::RuntimeEffectEnvelope::new(
                invocation.clone(),
                crate::RuntimeEffectCommand::ToolBatch { batch },
            );
            let local_executor =
                crate::RuntimeEffectLocalExecutor::tool_batch(self.clone(), child_trace_hooks);
            let raw_outcome = if self.should_execute_child_tool_batch_locally() {
                local_executor.execute(envelope).await
            } else {
                self.dispatch
                    .effect_controller
                    .controller()
                    .execute_effect(envelope, local_executor)
                    .await
            };
            let outcome =
                match raw_outcome.and_then(crate::RuntimeEffectOutcome::into_tool_batch_effect) {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        for (index, prepared, _) in prepared_entries {
                            replies[index] = Some(ToolInvocationReply::error(serde_json::json!(
                                format!("tool batch failed: {err}")
                            )));
                            let _ = prepared;
                        }
                        return replies
                            .into_iter()
                            .map(|reply| reply.expect("every batch reply slot should be filled"))
                            .collect();
                    }
                };
            if outcome.launches.len() != prepared_entries.len() {
                let message = format!(
                    "tool batch returned {} launches for {} prepared calls",
                    outcome.launches.len(),
                    prepared_entries.len()
                );
                for (index, _, _) in prepared_entries {
                    replies[index] = Some(ToolInvocationReply::error(serde_json::json!(message)));
                }
            } else {
                for ((index, prepared, _), launch) in
                    prepared_entries.into_iter().zip(outcome.launches)
                {
                    let call_id = prepared.call_id.clone();
                    let reply = match launch {
                        crate::runtime::ToolCallLaunch::Done { result } => {
                            let record = ToolCallRecord {
                                call_id: Some(result.call_id.clone()),
                                tool: result.tool_name.clone(),
                                args: result.args.clone(),
                                output: result.output.clone(),
                                duration_ms: result.duration_ms,
                            };
                            ToolInvocationReply::from_output(result.output).with_record(record)
                        }
                        crate::runtime::ToolCallLaunch::Pending {
                            key,
                            pending,
                            duration_ms,
                        } => {
                            let dispatch_outcome = self
                                .await_pending_tool_dispatch_outcome(
                                    &call_id,
                                    Some(invocation.clone()),
                                    crate::tool_dispatch::PendingToolDispatchOutcome {
                                        tool_name: prepared.tool_name.clone(),
                                        args: prepared.args.clone(),
                                        key,
                                        pending,
                                        duration_ms,
                                    },
                                    self.cancellation_token.clone(),
                                )
                                .await;
                            let completed = self
                                .complete_tool_call(
                                    index,
                                    call_id.clone(),
                                    prepared.replay.clone(),
                                    dispatch_outcome,
                                    TurnActivityId::new(format!("tool:{call_id}")),
                                )
                                .await;
                            ToolInvocationReply::from_output(completed.completed.output)
                                .with_record(completed.record)
                        }
                    };
                    replies[index] = Some(reply);
                }
            }
        }

        replies
            .into_iter()
            .map(|reply| reply.expect("every batch reply slot should be filled"))
            .collect()
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
