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
    dispatch: Arc<crate::tool_dispatch::ToolDispatchContext>,
    plugins: Arc<crate::PluginSession>,
    projector_manager: Arc<dyn RuntimeSessionHost>,
    event_tx: mpsc::Sender<RuntimeStreamEvent>,
    task_cancel: CancellationToken,
) -> (usize, crate::sansio::CompletedToolCall) {
    let call_id = pending_tool.call_id;
    let tool_name = pending_tool.tool_name;
    let args = pending_tool.args;
    let item_id = pending_tool.item_id;
    let _ = event_tx
        .send(RuntimeStreamEvent::Session(SessionEvent::ToolCallStart {
            call_id: Some(call_id.clone()),
            name: tool_name.clone(),
            args: args.clone(),
        }))
        .await;
    let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel::<SandboxMessage>();
    let progress_event_tx = event_tx.clone();
    let progress_handle = tokio::spawn(async move {
        while let Some(sandbox_msg) = progress_rx.recv().await {
            if sandbox_msg.kind != "final" {
                let _ = progress_event_tx
                    .send(RuntimeStreamEvent::Session(SessionEvent::Message {
                        text: sandbox_msg.text,
                        kind: sandbox_msg.kind,
                    }))
                    .await;
            }
        }
    });
    let tool_context = crate::ToolExecutionContext {
        session_id: dispatch.session_id.clone(),
        host: Arc::clone(&dispatch.host),
        cancellation_token: Some(task_cancel),
        async_task_id: None,
        turn_context: dispatch.turn_context.clone(),
    };
    let outcome = dispatch_tool_call_with_execution_context(
        &dispatch,
        tool_name,
        args,
        Some(&progress_tx),
        tool_context,
    )
    .await;
    drop(progress_tx);
    let _ = progress_handle.await;
    let raw_result = crate::ToolResult {
        success: outcome.record.success,
        result: outcome.record.result.clone(),
        images: outcome.images,
        control: outcome.record.control.clone(),
    };
    let state_result = match plugins
        .project_tool_result(crate::plugin::ToolResultProjectionContext {
            hook: crate::plugin::ToolResultProjectionHook::BeforeState,
            session_id: dispatch.session_id.clone(),
            tool_name: outcome.record.tool.clone(),
            args: outcome.record.args.clone(),
            result: raw_result.clone(),
            duration_ms: outcome.record.duration_ms,
            host: projector_manager.clone(),
        })
        .await
    {
        Ok(projected) => projected,
        Err(err) => crate::ToolResult::err_fmt(err.to_string()),
    };
    let model_result = match plugins
        .project_tool_result(crate::plugin::ToolResultProjectionContext {
            hook: crate::plugin::ToolResultProjectionHook::BeforeModel,
            session_id: dispatch.session_id.clone(),
            tool_name: outcome.record.tool.clone(),
            args: outcome.record.args.clone(),
            result: raw_result.clone(),
            duration_ms: outcome.record.duration_ms,
            host: projector_manager.clone(),
        })
        .await
    {
        Ok(projected) => projected,
        Err(err) => crate::ToolResult::err_fmt(err.to_string()),
    };
    (
        index,
        crate::sansio::CompletedToolCall {
            call_id,
            tool_name: outcome.record.tool,
            args: outcome.record.args,
            state_result,
            model_result,
            duration_ms: outcome.record.duration_ms,
            item_id,
        },
    )
}

impl RuntimeTurnDriver {
    pub(super) async fn run_tool_calls(
        &mut self,
        pending_tools: Vec<crate::sansio::PendingToolCall>,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> Vec<crate::sansio::CompletedToolCall> {
        let (tool_event_tx, mut tool_event_rx) = tokio::sync::mpsc::channel::<SessionEvent>(64);
        let runtime_event_tx = event_tx.clone();
        let tool_event_forwarder = tokio::spawn(async move {
            while let Some(event) = tool_event_rx.recv().await {
                send_session_event(&runtime_event_tx, event).await;
            }
        });
        let plugins = Arc::clone(self.session.plugins());
        let manager = self.session_manager.clone();
        let projector_manager = manager.clone();
        let dispatch = Arc::new(ToolDispatchContext {
            plugins: Arc::clone(&plugins),
            tools: self.session.tools(),
            surface: self
                .session
                .tool_surface(&self.session_id, self.policy.execution_mode.clone()),
            host: manager.clone(),
            session_id: self.session_id.clone(),
            event_tx: tool_event_tx,
            turn_injection_bridge: self.session.turn_injection_bridge().clone(),
            attachment_store: Arc::clone(&self.host.core.attachment_store),
            turn_context: self.turn_context.clone(),
        });
        // Partition pending tool calls by declared [`ToolExecutionMode`]:
        // parallel-safe tools spawn onto a JoinSet to run concurrently, while
        // tools declared `Serial` (apply_patch, exec_command, write_stdin,
        // ...) run one-at-a-time afterwards so they can't
        // interleave with each other or clobber shared state. Order is
        // preserved by the caller-provided `index` and the final sort below.
        let mut parallel_calls: Vec<(usize, crate::sansio::PendingToolCall)> = Vec::new();
        let mut serial_calls: Vec<(usize, crate::sansio::PendingToolCall)> = Vec::new();
        for (index, pending_tool) in pending_tools.into_iter().enumerate() {
            let mode = crate::tool_dispatch::resolve_tool_execution_mode(
                &dispatch,
                &pending_tool.tool_name,
            );
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
            let dispatch = Arc::clone(&dispatch);
            let plugins = Arc::clone(&plugins);
            let projector_manager = projector_manager.clone();
            let event_tx_clone = event_tx.clone();
            let task_cancel = tool_cancel.clone();
            join_set.spawn(async move {
                run_one_tool_call(
                    index,
                    pending_tool,
                    dispatch,
                    plugins,
                    projector_manager,
                    event_tx_clone,
                    task_cancel,
                )
                .await
            });
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
                                    state_result: crate::ToolResult::err_fmt("tool call cancelled"),
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
                                state_result: crate::ToolResult::err_fmt(format!(
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
                        state_result: crate::ToolResult::err_fmt("tool call cancelled"),
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
                Arc::clone(&dispatch),
                Arc::clone(&plugins),
                projector_manager.clone(),
                event_tx.clone(),
                tool_cancel.clone(),
            )
            .await;
            outcomes.push(outcome);
        }

        drop(dispatch);
        let _ = tool_event_forwarder.await;
        outcomes.sort_by_key(|(index, _)| *index);
        outcomes.into_iter().map(|(_, outcome)| outcome).collect()
    }
}
