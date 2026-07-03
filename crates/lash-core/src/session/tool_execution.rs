use super::execution_context::RuntimeExecutionContext;
use crate::tool_dispatch::{
    ToolCallLaunch, ToolDispatchOutcome, ToolPreparationOutcome,
    dispatch_granted_prepared_tool_attempt_launch_with_execution_context,
    dispatch_granted_prepared_tool_call_launch_with_execution_context,
    dispatch_prepared_tool_attempt_launch_with_execution_context,
    dispatch_prepared_tool_call_launch_with_execution_context,
    finalize_tool_result_with_execution_context, mark_retry_exhausted,
    prepare_granted_tool_call_with_context, prepare_tool_call_with_context, retry_after_ms,
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
    pub execution_grant: Option<Box<crate::ToolExecutionGrant>>,
    pub child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
}

impl ToolInvocation {
    pub fn new(id: impl Into<String>, tool_id: crate::ToolId, args: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            tool_id,
            args,
            execution_grant: None,
            child_execution_trace_hook: None,
        }
    }

    pub fn with_execution_grant(mut self, grant: crate::ToolExecutionGrant) -> Self {
        self.execution_grant = Some(Box::new(grant));
        self
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
                "execution_grant",
                &self.execution_grant.as_ref().map(|_| "<grant>"),
            )
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

fn runtime_failure_dispatch_outcome(
    call_id: Option<String>,
    tool_name: String,
    args: serde_json::Value,
    code: impl Into<String>,
    message: impl Into<String>,
) -> ToolDispatchOutcome {
    ToolDispatchOutcome {
        record: ToolCallRecord {
            call_id,
            tool: tool_name,
            args,
            output: ToolCallOutput::failure(ToolFailure::runtime(
                ToolFailureClass::Internal,
                code,
                message,
            )),
            duration_ms: 0,
        },
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
                "execution_grant": call.execution_grant.as_ref().map(|grant| serde_json::json!({
                    "tool_id": grant.manifest.id.to_string(),
                    "source_id": grant.source_id.clone(),
                    "execution_binding": grant.execution_binding.clone(),
                })),
            })
        })
        .collect::<Vec<_>>();
    let digest = crate::stable_hash::stable_json_sha256_hex(&identity)
        .unwrap_or_else(|_| format!("len-{}", calls.len()));
    format!("tool-batch:{digest}")
}

struct CoordinatedToolLaunch {
    launch: crate::runtime::ToolCallLaunch,
    triggers: Vec<crate::tool_dispatch::ToolTriggerEffectOutcome>,
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
        if !self
            .dispatch
            .effect_controller
            .controller()
            .supports_concurrent_effects()
        {
            let mut launches = Vec::with_capacity(indexed_tools.len());
            let mut triggers = Vec::new();
            let mut context = self.clone().with_cancellation_token(tool_cancel.clone());
            for (index, child) in indexed_tools {
                if cancellation.is_cancelled() {
                    tool_cancel.cancel();
                    launches.push(cancelled_runtime_tool_call_launch(
                        child.call.call_id,
                        child.call.tool_name,
                        child.call.args,
                        child.call.replay,
                    ));
                    continue;
                }
                let child_execution_trace_hook =
                    child_trace_hooks.get(&child.call.call_id).cloned();
                let outcome = context
                    .execute_prepared_tool_batch_child(
                        child,
                        index,
                        parent_invocation.clone(),
                        child_execution_trace_hook,
                    )
                    .await;
                launches.push(outcome.launch);
                triggers.extend(outcome.triggers);
                context = context.with_cancellation_token(tool_cancel.clone());
            }
            return Ok(crate::ToolBatchEffectOutcome { launches, triggers });
        }
        let child_outcomes = schedule_tool_batch(
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
                        let tool_call = context.execute_prepared_tool_batch_child(
                            child,
                            index,
                            parent_invocation,
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
                                    _ = &mut grace => CoordinatedToolLaunch {
                                        launch: cancelled_runtime_tool_call_launch(
                                            cancelled_tool.call_id,
                                            cancelled_tool.tool_name,
                                            cancelled_tool.args,
                                            cancelled_tool.replay,
                                        ),
                                        triggers: Vec::new(),
                                    },
                                }
                            }
                            outcome = &mut tool_call => outcome,
                        }
                    }
                }
            },
        )
        .await;

        let mut launches = Vec::with_capacity(child_outcomes.len());
        let mut triggers = Vec::new();
        for outcome in child_outcomes {
            launches.push(outcome.launch);
            triggers.extend(outcome.triggers);
        }
        Ok(crate::ToolBatchEffectOutcome { launches, triggers })
    }

    async fn execute_prepared_tool_batch_child(
        &self,
        child: crate::PreparedToolBatchCall,
        index: usize,
        parent_invocation: crate::RuntimeInvocation,
        child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
    ) -> CoordinatedToolLaunch {
        let call_id = child.call.call_id.clone();
        let tool_name = child.call.tool_name.clone();
        let args = child.call.args.clone();
        let replay = child.call.replay.clone();
        let activity_id = TurnActivityId::new(format!("tool:{call_id}"));
        self.emit_tool_call_started(&call_id, &tool_name, args.clone(), activity_id.clone())
            .await;

        let retry_policy = crate::tool_dispatch::resolve_callable_manifest_by_id(
            self.dispatch.as_ref(),
            &child.call.tool_id,
        )
        .map(|manifest| manifest.retry_policy)
        .or_else(|| {
            child
                .execution_grant
                .as_ref()
                .map(|grant| grant.manifest.retry_policy)
        })
        .unwrap_or(crate::ToolRetryPolicy::Never);
        let max_attempts = retry_policy.max_attempts().max(1);
        let mut triggers = Vec::new();

        for attempt in 1..=max_attempts {
            let attempt_invocation =
                self.tool_attempt_invocation(&parent_invocation, &child.replay_suffix, attempt);
            let attempt_outcome = self
                .dispatch
                .effect_controller
                .controller()
                .execute_effect(
                    crate::RuntimeEffectEnvelope::new(
                        attempt_invocation,
                        crate::RuntimeEffectCommand::ToolAttempt {
                            call: child.call.clone(),
                            execution_grant: child.execution_grant.clone(),
                            attempt,
                            max_attempts,
                        },
                    ),
                    crate::RuntimeEffectLocalExecutor::tool_batch(
                        self.clone(),
                        child_execution_trace_hook
                            .clone()
                            .map(|hook| {
                                std::iter::once((child.call.call_id.clone(), hook)).collect()
                            })
                            .unwrap_or_default(),
                    ),
                )
                .await;
            let attempt_outcome = match attempt_outcome {
                Ok(outcome) => match outcome.into_tool_attempt_effect() {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        let completed = self
                            .complete_tool_call(
                                index,
                                call_id.clone(),
                                replay,
                                runtime_failure_dispatch_outcome(
                                    Some(call_id.clone()),
                                    tool_name,
                                    args,
                                    "tool_attempt_failed",
                                    err.to_string(),
                                ),
                                activity_id,
                            )
                            .await;
                        return CoordinatedToolLaunch {
                            launch: crate::runtime::ToolCallLaunch::Done {
                                result: completed.completed,
                            },
                            triggers,
                        };
                    }
                },
                Err(err) => {
                    let completed = self
                        .complete_tool_call(
                            index,
                            call_id.clone(),
                            replay,
                            runtime_failure_dispatch_outcome(
                                Some(call_id.clone()),
                                tool_name,
                                args,
                                "tool_attempt_failed",
                                err.to_string(),
                            ),
                            activity_id,
                        )
                        .await;
                    return CoordinatedToolLaunch {
                        launch: crate::runtime::ToolCallLaunch::Done {
                            result: completed.completed,
                        },
                        triggers,
                    };
                }
            };
            triggers.extend(attempt_outcome.triggers);
            match attempt_outcome.launch {
                crate::ToolAttemptLaunch::Pending {
                    key,
                    pending,
                    duration_ms,
                } => {
                    let dispatch_outcome = self
                        .await_pending_tool_dispatch_outcome_with_suffix(
                            &call_id,
                            Some(parent_invocation.clone()),
                            format!("{}:await", child.replay_suffix),
                            crate::tool_dispatch::PendingToolDispatchOutcome {
                                tool_name: child.call.tool_name.clone(),
                                args: child.call.args.clone(),
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
                            child.call.replay.clone(),
                            dispatch_outcome,
                            activity_id,
                        )
                        .await;
                    return CoordinatedToolLaunch {
                        launch: crate::runtime::ToolCallLaunch::Done {
                            result: completed.completed,
                        },
                        triggers,
                    };
                }
                crate::ToolAttemptLaunch::Done { mut record } => {
                    record.call_id = Some(call_id.clone());
                    let retry_after = retry_after_ms(
                        &ToolResult::from_output(record.output.clone()),
                        retry_policy,
                        attempt - 1,
                    );
                    let Some(retry_after) = retry_after else {
                        let completed = self
                            .complete_tool_call(
                                index,
                                call_id.clone(),
                                child.call.replay.clone(),
                                ToolDispatchOutcome { record },
                                activity_id,
                            )
                            .await;
                        return CoordinatedToolLaunch {
                            launch: crate::runtime::ToolCallLaunch::Done {
                                result: completed.completed,
                            },
                            triggers,
                        };
                    };
                    if attempt >= max_attempts {
                        let exhausted =
                            mark_retry_exhausted(ToolResult::from_output(record.output), attempt);
                        record.output = exhausted.into_done_output().unwrap_or_else(|_| {
                            ToolCallOutput::failure(ToolFailure::runtime(
                                ToolFailureClass::Internal,
                                "tool_retry_exhaustion_failed",
                                "retry exhaustion produced a pending output",
                            ))
                        });
                        let completed = self
                            .complete_tool_call(
                                index,
                                call_id.clone(),
                                child.call.replay.clone(),
                                ToolDispatchOutcome { record },
                                activity_id,
                            )
                            .await;
                        return CoordinatedToolLaunch {
                            launch: crate::runtime::ToolCallLaunch::Done {
                                result: completed.completed,
                            },
                            triggers,
                        };
                    }
                    if retry_after > 0
                        && let Err(err) = self
                            .sleep_before_tool_retry(
                                &parent_invocation,
                                &child.replay_suffix,
                                attempt,
                                retry_after,
                            )
                            .await
                    {
                        let completed = self
                            .complete_tool_call(
                                index,
                                call_id.clone(),
                                child.call.replay.clone(),
                                runtime_failure_dispatch_outcome(
                                    Some(call_id.clone()),
                                    child.call.tool_name.clone(),
                                    child.call.args.clone(),
                                    "tool_retry_sleep_failed",
                                    format!(
                                        "retry sleep for tool `{}` failed after attempt {attempt}: {err}",
                                        child.call.tool_name
                                    ),
                                ),
                                activity_id,
                            )
                            .await;
                        return CoordinatedToolLaunch {
                            launch: crate::runtime::ToolCallLaunch::Done {
                                result: completed.completed,
                            },
                            triggers,
                        };
                    }
                }
            }
        }

        let completed = self
            .complete_tool_call(
                index,
                call_id.clone(),
                child.call.replay,
                runtime_failure_dispatch_outcome(
                    Some(call_id),
                    child.call.tool_name,
                    child.call.args,
                    "tool_retry_loop_failed",
                    "tool retry loop exited without a terminal result",
                ),
                activity_id,
            )
            .await;
        CoordinatedToolLaunch {
            launch: crate::runtime::ToolCallLaunch::Done {
                result: completed.completed,
            },
            triggers,
        }
    }

    async fn emit_tool_call_started(
        &self,
        call_id: &str,
        name: &str,
        args: serde_json::Value,
        activity_id: TurnActivityId,
    ) {
        let _ = self
            .dispatch
            .event_tx
            .send(SessionEvent::ToolCallStart {
                call_id: Some(call_id.to_string()),
                name: name.to_string(),
                args: args.clone(),
            })
            .await;
        self.emit_turn_activity(
            activity_id,
            TurnEvent::ToolCallStarted {
                call_id: Some(call_id.to_string()),
                name: name.to_string(),
                args,
            },
        )
        .await;
    }

    fn tool_attempt_invocation(
        &self,
        parent_invocation: &crate::RuntimeInvocation,
        child_replay_suffix: &str,
        attempt: u32,
    ) -> crate::RuntimeInvocation {
        let suffix = format!("{child_replay_suffix}:attempt:{attempt}");
        let parent_effect_id = parent_invocation.effect_id().unwrap_or("tool-batch");
        crate::runtime::causal::child_effect_invocation(
            parent_invocation,
            format!("{parent_effect_id}:{suffix}"),
            crate::RuntimeEffectKind::ToolAttempt,
            suffix,
        )
    }

    async fn sleep_before_tool_retry(
        &self,
        parent_invocation: &crate::RuntimeInvocation,
        child_replay_suffix: &str,
        attempt: u32,
        retry_after_ms: u64,
    ) -> Result<(), crate::RuntimeEffectControllerError> {
        let suffix = format!("{child_replay_suffix}:attempt:{attempt}:sleep");
        let parent_effect_id = parent_invocation.effect_id().unwrap_or("tool-batch");
        let invocation = crate::runtime::causal::child_effect_invocation(
            parent_invocation,
            format!("{parent_effect_id}:{suffix}"),
            crate::RuntimeEffectKind::Sleep,
            suffix,
        );
        let cancellation = self.cancellation_token.clone().unwrap_or_default();
        let outcome = self
            .dispatch
            .effect_controller
            .controller()
            .execute_effect(
                crate::RuntimeEffectEnvelope::new(
                    invocation,
                    crate::RuntimeEffectCommand::Sleep {
                        duration_ms: retry_after_ms,
                    },
                ),
                crate::RuntimeEffectLocalExecutor::sleep_with_clock(
                    cancellation,
                    std::sync::Arc::clone(&self.dispatch.clock),
                ),
            )
            .await?;
        match outcome {
            crate::RuntimeEffectOutcome::Sleep => Ok(()),
            other => Err(crate::RuntimeEffectControllerError::new(
                "runtime_effect_wrong_outcome",
                format!("expected sleep outcome, got {}", other.kind().as_str()),
            )),
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
                        process_events.awaiter.clone(),
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

    #[expect(
        clippy::too_many_arguments,
        reason = "tool execution carries explicit runtime call metadata"
    )]
    pub(crate) async fn execute_tool_call_by_grant(
        &self,
        call_id: String,
        grant: crate::ToolExecutionGrant,
        args: serde_json::Value,
        index: usize,
        replay: Option<crate::llm::types::ProviderReplayMeta>,
        parent_invocation: Option<crate::RuntimeInvocation>,
        child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
    ) -> CompletedProtocolToolCall {
        let name = grant.manifest.name.clone();
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
        let launch = match prepare_granted_tool_call_with_context(
            &dispatch,
            &grant,
            pending,
            Some(call_id.clone()),
        )
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
                        .tool_execution_binding(grant.execution_binding.clone())
                        .cancellation_token(self.cancellation_token.clone())
                        .runtime_process_id(self.runtime_process_id.clone())
                        .parent_invocation(parent_invocation.clone())
                        .child_execution_trace_hook(child_execution_trace_hook.clone());
                if let Some(process_events) = self.process_event_context.as_ref() {
                    tool_context = tool_context.process_events(
                        process_events.process_id.clone(),
                        std::sync::Arc::clone(&process_events.registry),
                        process_events.awaiter.clone(),
                        process_events.store.clone(),
                        process_events.session_store_factory.clone(),
                        process_events.queued_work_driver.clone(),
                    );
                }
                let tool_context = tool_context.build();
                dispatch_granted_prepared_tool_call_launch_with_execution_context(
                    dispatch_context.as_ref(),
                    &grant,
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

    pub(crate) async fn execute_prepared_tool_attempt_effect(
        &self,
        prepared: crate::PreparedToolCall,
        execution_grant: Option<Box<crate::ToolExecutionGrant>>,
        attempt: u32,
        max_attempts: u32,
        attempt_invocation: crate::RuntimeInvocation,
        child_execution_trace_hook: Option<crate::ToolChildExecutionTraceHook>,
    ) -> Result<crate::ToolAttemptEffectOutcome, crate::RuntimeEffectControllerError> {
        let call_id = prepared.call_id.clone();
        let mut attempt_dispatch = (*self.dispatch).clone();
        attempt_dispatch.parent_invocation = Some(attempt_invocation.clone());
        attempt_dispatch.trigger_outcomes =
            crate::tool_dispatch::ToolTriggerOutcomeBuffer::default();
        let attempt_dispatch = std::sync::Arc::new(attempt_dispatch);
        let mut attempt_context = self.clone();
        attempt_context.dispatch = std::sync::Arc::clone(&attempt_dispatch);
        attempt_context.parent_invocation = Some(attempt_invocation.clone());

        let mut tool_context =
            crate::ToolContext::from_dispatch(std::sync::Arc::clone(&attempt_dispatch))
                .runtime_execution_context(attempt_context.clone())
                .prepared_call(&prepared)
                .cancellation_token(self.cancellation_token.clone())
                .runtime_process_id(self.runtime_process_id.clone())
                .parent_invocation(Some(attempt_invocation))
                .child_execution_trace_hook(child_execution_trace_hook);
        if let Some(process_events) = self.process_event_context.as_ref() {
            tool_context = tool_context.process_events(
                process_events.process_id.clone(),
                std::sync::Arc::clone(&process_events.registry),
                process_events.awaiter.clone(),
                process_events.store.clone(),
                process_events.session_store_factory.clone(),
                process_events.queued_work_driver.clone(),
            );
        }
        let tool_context = tool_context.build();
        let launch = match Box::pin(async {
            if let Some(grant) = execution_grant.as_ref() {
                dispatch_granted_prepared_tool_attempt_launch_with_execution_context(
                    attempt_dispatch.as_ref(),
                    grant,
                    prepared,
                    attempt,
                    max_attempts,
                    None,
                    tool_context,
                )
                .await
            } else {
                dispatch_prepared_tool_attempt_launch_with_execution_context(
                    attempt_dispatch.as_ref(),
                    prepared,
                    attempt,
                    max_attempts,
                    None,
                    tool_context,
                )
                .await
            }
        })
        .await
        {
            ToolCallLaunch::Done(outcome) => {
                let mut record = outcome.record;
                record.call_id = Some(call_id);
                crate::ToolAttemptLaunch::Done { record }
            }
            ToolCallLaunch::Pending(pending) => crate::ToolAttemptLaunch::Pending {
                key: pending.key,
                pending: pending.pending,
                duration_ms: pending.duration_ms,
            },
        };
        let triggers = attempt_context
            .drain_tool_trigger_outcomes()
            .map_err(|err| {
                crate::RuntimeEffectControllerError::new(
                    "tool_trigger_outcome_drain",
                    err.to_string(),
                )
            })?;
        Ok(crate::ToolAttemptEffectOutcome { launch, triggers })
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
        self.await_pending_tool_dispatch_outcome_with_suffix(
            call_id,
            parent_invocation,
            format!("{call_id}:await"),
            pending,
            cancellation,
        )
        .await
    }

    async fn await_pending_tool_dispatch_outcome_with_suffix(
        &self,
        call_id: &str,
        parent_invocation: Option<crate::RuntimeInvocation>,
        replay_suffix: String,
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
            format!("{parent_effect_id}:{replay_suffix}"),
            crate::RuntimeEffectKind::AwaitEvent,
            replay_suffix,
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

    pub async fn call_tool_with_execution_grant(
        &self,
        call_id: String,
        grant: crate::ToolExecutionGrant,
        args: serde_json::Value,
        index: usize,
    ) -> ToolInvocationReply {
        let executed = self
            .execute_tool_call_by_grant(call_id, grant, args, index, None, None, None)
            .await;
        let reply = ToolInvocationReply::from_output(executed.completed.output);
        reply.with_record(executed.record)
    }

    pub async fn call_tool_with_execution_grant_and_child_execution_trace_hook(
        &self,
        call_id: String,
        grant: crate::ToolExecutionGrant,
        args: serde_json::Value,
        index: usize,
        trace_hook: crate::ToolChildExecutionTraceHook,
    ) -> ToolInvocationReply {
        let executed = self
            .execute_tool_call_by_grant(call_id, grant, args, index, None, None, Some(trace_hook))
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
            let preparation = if let Some(grant) = call.execution_grant.as_deref().cloned() {
                let pending = crate::sansio::PendingToolCall {
                    call_id: call.id.clone(),
                    tool_name: grant.manifest.name.clone(),
                    args: call.args,
                    replay: None,
                };
                (
                    Some(grant.clone()),
                    prepare_granted_tool_call_with_context(
                        self.dispatch.as_ref(),
                        &grant,
                        pending,
                        Some(call.id.clone()),
                    )
                    .await,
                )
            } else {
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
                                format!(
                                    "Tool id `{}` is unavailable in this session",
                                    call.tool_id
                                ),
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
                (None, self.prepare_tool_call(pending).await)
            };
            let (execution_grant, preparation) = preparation;
            match preparation {
                ToolPreparationOutcome::Prepared(prepared) => {
                    prepared_entries.push((
                        index,
                        prepared,
                        execution_grant,
                        call.child_execution_trace_hook,
                    ));
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
            let batch = crate::PreparedToolBatch::new_with_grants(
                batch_id.clone(),
                prepared_entries
                    .iter()
                    .map(|(_, prepared, grant, _)| (prepared.clone(), grant.clone()))
                    .collect(),
            );
            let child_trace_hooks = prepared_entries
                .iter()
                .filter_map(|(_, prepared, _, hook)| {
                    hook.clone().map(|hook| (prepared.call_id.clone(), hook))
                })
                .collect();
            let envelope = crate::RuntimeEffectEnvelope::new(
                invocation.clone(),
                crate::RuntimeEffectCommand::ToolBatch { batch },
            );
            let local_executor =
                crate::RuntimeEffectLocalExecutor::tool_batch(self.clone(), child_trace_hooks);
            let raw_outcome = self
                .dispatch
                .effect_controller
                .controller()
                .execute_effect(envelope, local_executor)
                .await;
            let outcome =
                match raw_outcome.and_then(crate::RuntimeEffectOutcome::into_tool_batch_effect) {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        for (index, prepared, _, _) in prepared_entries {
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
                for (index, _, _, _) in prepared_entries {
                    replies[index] = Some(ToolInvocationReply::error(serde_json::json!(message)));
                }
            } else {
                for ((index, prepared, _, _), launch) in
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
