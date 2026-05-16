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
    pending_tool: crate::sansio::PendingToolCall,
    context: crate::ModeExecutionContext,
) -> crate::sansio::CompletedToolCall {
    let executed = context
        .execute_tool_call(
            pending_tool.call_id,
            pending_tool.tool_name,
            pending_tool.args,
            index,
            pending_tool.replay,
        )
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
            parts: vec![crate::ModelToolReturnPart::Text(
                "[Tool execution cancelled]\ntool call cancelled".to_string(),
            )],
        },
        output,
        duration_ms: 0,
        replay,
    }
}

fn internal_failure_completed_tool_call(message: String) -> crate::sansio::CompletedToolCall {
    let call_id = uuid::Uuid::new_v4().to_string();
    let tool_name = "unknown".to_string();
    let output = crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
        crate::ToolFailureClass::Internal,
        "tool_task_failed",
        message.clone(),
    ));
    crate::sansio::CompletedToolCall {
        call_id: call_id.clone(),
        tool_name: tool_name.clone(),
        args: serde_json::json!({}),
        model_return: crate::ModelToolReturn {
            call_id,
            tool_name,
            parts: vec![crate::ModelToolReturnPart::Text(format!(
                "[Tool execution failed]\n{message}"
            ))],
        },
        output,
        duration_ms: 0,
        replay: None,
    }
}

impl RuntimeTurnDriver {
    pub(super) async fn run_tool_calls(
        &mut self,
        pending_tools: Vec<crate::sansio::PendingToolCall>,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Vec<crate::sansio::CompletedToolCall> {
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
        let manager = self.session_manager.clone();
        let context = self
            .session
            .mode_execution_context(
                &self.session_id,
                manager,
                tool_event_tx,
                Arc::new(crate::ChronologicalProjection::default()),
                self.mode_extension.clone(),
                self.turn_context.clone(),
            )
            .with_turn_event_sender(turn_event_tx);
        let indexed_tools = pending_tools.into_iter().enumerate().collect::<Vec<_>>();
        let tool_cancel = cancel.child_token();
        let outcomes = schedule_tool_batch(
            indexed_tools,
            |(index, _)| *index,
            |(_, pending_tool)| context.tool_execution_mode(&pending_tool.tool_name),
            {
                let context = context.clone();
                let cancel = cancel.clone();
                let tool_cancel = tool_cancel.clone();
                move |(index, pending_tool)| {
                    let context = context.clone().with_cancellation_token(tool_cancel.clone());
                    let cancel = cancel.clone();
                    let tool_cancel = tool_cancel.clone();
                    let cancelled_tool = pending_tool.clone();
                    async move {
                        let mut task =
                            tokio::spawn(run_one_tool_call(index, pending_tool, context));
                        tokio::select! {
                            biased;
                            _ = cancel.cancelled() => {
                                tool_cancel.cancel();
                                task.abort();
                                cancelled_completed_tool_call(
                                    cancelled_tool.call_id,
                                    cancelled_tool.tool_name,
                                    cancelled_tool.args,
                                    cancelled_tool.replay,
                                )
                            }
                            joined = &mut task => {
                                match joined {
                                    Ok(outcome) => outcome,
                                    Err(err) if err.is_cancelled() => cancelled_completed_tool_call(
                                        cancelled_tool.call_id,
                                        cancelled_tool.tool_name,
                                        cancelled_tool.args,
                                        cancelled_tool.replay,
                                    ),
                                    Err(err) => internal_failure_completed_tool_call(
                                        format!("tool task panicked: {err}"),
                                    ),
                                }
                            }
                        }
                    }
                }
            },
        )
        .await;

        drop(context);
        let _ = tool_event_forwarder.await;
        let _ = turn_event_forwarder.await;
        outcomes
    }
}
