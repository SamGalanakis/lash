use super::*;

/// Run a single pending tool call through the dispatch context and result
/// projection pipeline, returning the completed call annotated with its
/// original emission index.
///
/// Extracted from `run_tool_calls` so the same per-call logic can be used
/// both inside a `JoinSet` (for parallel-safe tools) and in a sequential
/// loop (for [`crate::ToolExecutionMode::Serial`] tools).
async fn run_one_tool_call(
    index: usize,
    pending_tool: crate::sansio::PendingToolCall,
    context: crate::ModeExecutionContext,
) -> (usize, crate::sansio::CompletedToolCall) {
    let executed = context
        .execute_tool_call(
            pending_tool.call_id,
            pending_tool.tool_name,
            pending_tool.args,
            index,
            pending_tool.item_id,
        )
        .await;
    (executed.index, executed.completed)
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
        // Partition pending tool calls by declared [`ToolExecutionMode`]:
        // parallel-safe tools spawn onto a JoinSet to run concurrently, while
        // tools declared `Serial` (apply_patch, exec_command, write_stdin,
        // ...) run one-at-a-time afterwards so they can't
        // interleave with each other or clobber shared state. Order is
        // preserved by the caller-provided `index` and the final sort below.
        let mut parallel_calls: Vec<(usize, crate::sansio::PendingToolCall)> = Vec::new();
        let mut serial_calls: Vec<(usize, crate::sansio::PendingToolCall)> = Vec::new();
        for (index, pending_tool) in pending_tools.into_iter().enumerate() {
            let mode = context.tool_execution_mode(&pending_tool.tool_name);
            match mode {
                crate::ToolExecutionMode::Parallel => parallel_calls.push((index, pending_tool)),
                crate::ToolExecutionMode::Serial => serial_calls.push((index, pending_tool)),
            }
        }

        let mut outcomes: Vec<(usize, crate::sansio::CompletedToolCall)> = Vec::new();

        // Tools get a child of the turn-level cancellation token so they can
        // cooperatively bail out when the turn is cancelled. We also abort the
        // JoinSet below on cancel to force-terminate tasks that don't check.
        let tool_cancel = cancel.child_token();
        let mut join_set = tokio::task::JoinSet::new();
        for (index, pending_tool) in parallel_calls.into_iter() {
            let context = context.clone().with_cancellation_token(tool_cancel.clone());
            join_set.spawn(async move { run_one_tool_call(index, pending_tool, context).await });
        }

        let mut cancelled = false;
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled(), if !cancelled => {
                    // Turn cancellation: signal cooperative shutdown to any
                    // tool that is checking the token, then hard-abort the
                    // JoinSet to ensure we return promptly.
                    tool_cancel.cancel();
                    join_set.abort_all();
                    cancelled = true;
                }
                joined = join_set.join_next() => {
                    let Some(joined) = joined else { break; };
                    match joined {
                        Ok(outcome) => outcomes.push(outcome),
                        Err(e) if e.is_cancelled() => {
                            // Aborted due to turn cancellation — synthesize a
                            // cancellation result so the turn machine receives
                            // a response for every pending call.
                            outcomes.push((
                                usize::MAX,
                                crate::sansio::CompletedToolCall {
                                    call_id: uuid::Uuid::new_v4().to_string(),
                                    tool_name: "unknown".to_string(),
                                    args: serde_json::json!({}),
                                    result: crate::ToolResult::err_fmt("tool call cancelled"),
                                    model_result: crate::ToolResult::err_fmt("tool call cancelled"),
                                    duration_ms: 0,
                                    item_id: None,
                                },
                            ));
                        }
                        Err(e) => outcomes.push((
                            usize::MAX,
                            crate::sansio::CompletedToolCall {
                                call_id: uuid::Uuid::new_v4().to_string(),
                                tool_name: "unknown".to_string(),
                                args: serde_json::json!({}),
                                result: crate::ToolResult::err_fmt(format!(
                                    "tool task panicked: {e}"
                                )),
                                model_result: crate::ToolResult::err_fmt(format!(
                                    "tool task panicked: {e}"
                                )),
                                duration_ms: 0,
                                item_id: None,
                            },
                        )),
                    }
                }
            }
        }

        // Serial tools run sequentially, in original emission order, under the
        // same cancellation token so turn-cancel also stops the serial queue.
        serial_calls.sort_by_key(|(index, _)| *index);
        for (index, pending_tool) in serial_calls.into_iter() {
            if cancelled {
                outcomes.push((
                    index,
                    crate::sansio::CompletedToolCall {
                        call_id: pending_tool.call_id,
                        tool_name: pending_tool.tool_name,
                        args: pending_tool.args,
                        result: crate::ToolResult::err_fmt("tool call cancelled"),
                        model_result: crate::ToolResult::err_fmt("tool call cancelled"),
                        duration_ms: 0,
                        item_id: pending_tool.item_id,
                    },
                ));
                continue;
            }
            let outcome = run_one_tool_call(
                index,
                pending_tool,
                context.clone().with_cancellation_token(tool_cancel.clone()),
            )
            .await;
            outcomes.push(outcome);
        }

        drop(context);
        let _ = tool_event_forwarder.await;
        let _ = turn_event_forwarder.await;
        outcomes.sort_by_key(|(index, _)| *index);
        outcomes.into_iter().map(|(_, outcome)| outcome).collect()
    }
}
