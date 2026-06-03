use super::*;
use crate::tool_dispatch::schedule_tool_batch;

/// Run a single pending tool call through the dispatch context and result
/// projection pipeline, returning the completed call annotated with its
/// original emission index.
///
/// Extracted from `run_tool_calls` so scheduling remains separate from the
/// per-call execution side effects.
async fn run_one_tool_call(
    index: usize,
    prepared_tool: crate::PreparedToolCall,
    invocation: crate::RuntimeInvocation,
    context: crate::RuntimeExecutionContext<'_>,
) -> crate::sansio::CompletedToolCall {
    let executed = context
        .execute_prepared_tool_call(prepared_tool, index, Some(invocation))
        .await;
    debug_assert_eq!(executed.index, index);
    executed.completed
}

fn cancelled_completed_tool_call(
    call_id: String,
    tool_name: String,
    args: serde_json::Value,
    replay: Option<crate::llm::types::ProviderReplayMeta>,
) -> crate::sansio::CompletedToolCall {
    let output =
        crate::ToolCallOutput::cancelled(crate::ToolCancellation::runtime("tool call cancelled"));
    crate::sansio::CompletedToolCall {
        call_id: call_id.clone(),
        tool_name: tool_name.clone(),
        args,
        model_return: crate::ModelToolReturn {
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

impl RuntimeTurnDriver<'_> {
    pub(super) async fn invoke_turn_tool_calls_effect(
        &mut self,
        machine: &mut TurnMachine,
        id: crate::sansio::EffectId,
        calls: Vec<crate::sansio::PendingToolCall>,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Result<Vec<crate::sansio::CompletedToolCall>, RuntimeEffectControllerError> {
        let (tool_event_tx, mut tool_event_rx) = tokio::sync::mpsc::channel::<SessionEvent>(64);
        let runtime_event_tx = event_tx.clone();
        let tool_event_forwarder = tokio::spawn(async move {
            while let Some(event) = tool_event_rx.recv().await {
                send_session_event(&runtime_event_tx, event).await;
            }
        });
        let prepare_context = self
            .execution_context(
                tool_event_tx.clone(),
                Arc::new(crate::ChronologicalProjection::default()),
            )
            .map_err(|err| {
                RuntimeEffectControllerError::new("tool_surface_resolution_failed", err.to_string())
            })?;
        let mut results = Vec::with_capacity(calls.len());
        for (index, call) in calls.into_iter().enumerate() {
            let call_id = call.call_id.clone();
            let replay = call.replay.clone();
            let prepared = match prepare_context.prepare_tool_call(call).await {
                crate::tool_dispatch::ToolPreparationOutcome::Prepared(prepared) => prepared,
                crate::tool_dispatch::ToolPreparationOutcome::Completed(outcome) => {
                    let completed = prepare_context
                        .complete_tool_call(
                            index,
                            call_id.clone(),
                            replay,
                            *outcome,
                            crate::TurnActivityId::new(format!("tool:{call_id}")),
                        )
                        .await
                        .completed;
                    results.push(completed);
                    continue;
                }
            };
            let parent_invocation =
                self.turn_effect_invocation(machine, id, RuntimeEffectKind::ToolCall)?;
            let invocation = crate::runtime::causal::child_tool_effect_invocation(
                &parent_invocation,
                id,
                &prepared.call_id,
            );
            let result = self
                .execute_typed_turn_effect(
                    machine,
                    event_tx,
                    cancel,
                    RuntimeEffectEnvelope::new(
                        invocation,
                        RuntimeEffectCommand::ToolCall { call: prepared },
                    ),
                    RuntimeEffectOutcome::into_tool_call,
                )
                .await?;
            results.push(result);
        }
        drop(prepare_context);
        drop(tool_event_tx);
        let _ = tool_event_forwarder.await;
        Ok(results)
    }

    pub(in crate::runtime) async fn run_tool_calls(
        &mut self,
        pending_tools: Vec<(crate::PreparedToolCall, crate::RuntimeInvocation)>,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Result<Vec<crate::sansio::CompletedToolCall>, crate::RuntimeEffectControllerError> {
        let (tool_event_tx, mut tool_event_rx) = tokio::sync::mpsc::channel::<SessionEvent>(64);
        let (turn_event_tx, mut turn_event_rx) = tokio::sync::mpsc::channel::<TurnActivity>(64);
        let runtime_event_tx = event_tx.clone();
        let tool_event_forwarder = tokio::spawn(async move {
            while let Some(event) = tool_event_rx.recv().await {
                send_session_event(&runtime_event_tx, event).await;
            }
        });
        let runtime_event_tx = event_tx.clone();
        let turn_event_forwarder = tokio::spawn(async move {
            while let Some(event) = turn_event_rx.recv().await {
                let _ = runtime_event_tx.send(RuntimeStreamEvent::Turn(event)).await;
            }
        });
        let context = match self.execution_context(
            tool_event_tx.clone(),
            Arc::new(crate::ChronologicalProjection::default()),
        ) {
            Ok(context) => context.with_turn_event_sender(turn_event_tx.clone()),
            Err(err) => {
                drop(tool_event_tx);
                drop(turn_event_tx);
                let _ = tool_event_forwarder.await;
                let _ = turn_event_forwarder.await;
                return Err(crate::RuntimeEffectControllerError::new(
                    "tool_surface_resolution_failed",
                    err.to_string(),
                ));
            }
        };
        let indexed_tools = pending_tools.into_iter().enumerate().collect::<Vec<_>>();
        let tool_cancel = cancel.child_token();
        let outcomes = schedule_tool_batch(
            indexed_tools,
            |(index, _)| *index,
            |(_, (pending_tool, _))| context.tool_scheduling(&pending_tool.tool_name),
            {
                let context = context.clone();
                let cancel = cancel.clone();
                let tool_cancel = tool_cancel.clone();
                move |(index, (pending_tool, parent_invocation))| {
                    let context = context.clone().with_cancellation_token(tool_cancel.clone());
                    let cancel = cancel.clone();
                    let tool_cancel = tool_cancel.clone();
                    let cancelled_tool = pending_tool.clone();
                    async move {
                        let tool_call =
                            run_one_tool_call(index, pending_tool, parent_invocation, context);
                        tokio::pin!(tool_call);
                        tokio::select! {
                            biased;
                            _ = cancel.cancelled() => {
                                tool_cancel.cancel();
                                let grace = tokio::time::sleep(std::time::Duration::from_millis(50));
                                tokio::pin!(grace);
                                tokio::select! {
                                    biased;
                                    outcome = &mut tool_call => outcome,
                                    _ = &mut grace => cancelled_completed_tool_call(
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

        drop(context);
        drop(tool_event_tx);
        drop(turn_event_tx);
        let _ = tool_event_forwarder.await;
        let _ = turn_event_forwarder.await;
        Ok(outcomes)
    }
}
