use super::*;

/// Result of running stream hooks over a visible chunk. Carries both
/// the (possibly rewritten) text and an `abort_requested` flag that the
/// LLM runner uses to break the stream early when a plugin has decided
/// the response is complete (e.g. the RLM mask detecting a closed
/// lashlang fence).
pub(super) struct StreamChunkOutcome {
    pub(super) chunk: String,
    pub(super) abort_requested: bool,
}

async fn send_session_event(event_tx: &mpsc::Sender<RuntimeStreamEvent>, event: SessionEvent) {
    if !event_tx.is_closed() {
        let _ = event_tx.send(RuntimeStreamEvent::Session(event)).await;
    }
}

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
    projector_manager: Arc<dyn SessionManager>,
    event_tx: mpsc::Sender<RuntimeStreamEvent>,
    task_cancel: CancellationToken,
) -> (usize, crate::sansio::CompletedToolCall) {
    let call_id = pending_tool.call_id;
    let tool_name = pending_tool.tool_name;
    let args = pending_tool.args;
    let item_id = pending_tool.item_id;
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
        result: outcome.record.result,
        images: outcome.images,
    };
    let state_result = match plugins
        .project_tool_result(crate::plugin::ToolResultProjectionContext {
            hook: crate::plugin::ToolResultProjectionHook::BeforeState,
            session_id: dispatch.session_id.clone(),
            tool_name: outcome.record.tool.clone(),
            args: outcome.record.args.clone(),
            result: raw_result.clone(),
            duration_ms: outcome.record.duration_ms,
            host: Arc::clone(&projector_manager),
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
            host: Arc::clone(&projector_manager),
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

async fn emit_plugin_surface_events_runtime(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    plugin_id: &str,
    events: Vec<crate::PluginSurfaceEvent>,
) {
    for event in crate::plugin::plugin_surface_session_events(plugin_id, events) {
        send_session_event(event_tx, event).await;
    }
}

pub(super) struct RuntimeTurnDriver {
    pub(super) session: Session,
    pub(super) policy: SessionPolicy,
    pub(super) host: RuntimeHost,
    pub(super) session_id: String,
    pub(super) base_graph: crate::SessionGraph,
    pub(super) tool_calls: Arc<Vec<ToolCallRecord>>,
    pub(super) llm_stream_summaries: HashMap<usize, LlmStreamSummary>,
    pub(super) session_manager: Arc<dyn SessionManager>,
    pub(super) prompt_bridge: HostPromptBridge,
    pub(super) rlm_termination: crate::RlmTermination,
    pub(super) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}

impl RuntimeTurnDriver {
    fn mark_phase_begin(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.begin(phase);
        }
    }

    fn mark_phase_end(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.end(phase);
        }
    }

    pub(super) async fn run(
        &mut self,
        messages: crate::MessageSequence,
        event_tx: mpsc::Sender<RuntimeStreamEvent>,
        cancel: CancellationToken,
        run_offset: usize,
    ) -> (crate::MessageSequence, usize) {
        macro_rules! emit {
            ($event:expr) => {
                send_session_event(&event_tx, $event).await
            };
        }
        let result = async {
            let mut machine = match self
                .prepare_turn_machine(messages, &event_tx, run_offset)
                .await
            {
                Ok(machine) => machine,
                Err(result) => return result,
            };
            loop {
                let Some(effect) = machine.poll_effect() else {
                    break;
                };
                match effect {
                    Effect::Emit(event) => emit!(event),
                    Effect::Done {
                        messages,
                        iteration,
                    } => return (messages, iteration),
                    Effect::LlmCall { id, request } => {
                        if cancel.is_cancelled() {
                            emit!(SessionEvent::Done);
                            return (crate::MessageSequence::default(), run_offset);
                        }
                        let iteration = machine.iteration();
                        let (result, text_streamed) = self
                            .run_llm_call(&mut machine, id, request, iteration, &event_tx, &cancel)
                            .await;
                        machine.handle_response(Response::LlmComplete {
                            id,
                            result,
                            text_streamed,
                        });
                    }
                    Effect::Checkpoint { id, checkpoint } => {
                        match self
                            .run_checkpoint(&mut machine, checkpoint, &event_tx)
                            .await
                        {
                            Ok((messages, transient_messages)) => {
                                machine.handle_response(Response::Checkpoint {
                                    id,
                                    messages,
                                    transient_messages,
                                });
                            }
                            Err(err) => {
                                machine.fail_turn(make_error_event(
                                    "plugin",
                                    Some(&err.code),
                                    err.message,
                                    None,
                                ));
                            }
                        }
                    }
                    Effect::SyncExecutionSurface { id } => {
                        let result = self
                            .session
                            .refresh_tool_surface()
                            .await
                            .map_err(|err| err.to_string());
                        machine.handle_response(Response::ExecutionSurfaceSynced { id, result });
                    }
                    Effect::ToolCalls { id, calls } => {
                        let results = self.run_tool_calls(calls, &event_tx, &cancel).await;
                        Arc::make_mut(&mut self.tool_calls).extend(results.iter().map(|outcome| {
                            ToolCallRecord {
                                call_id: Some(outcome.call_id.clone()),
                                tool: outcome.tool_name.clone(),
                                args: outcome.args.clone(),
                                result: outcome.state_result.result.clone(),
                                success: outcome.state_result.success,
                                duration_ms: outcome.duration_ms,
                            }
                        }));
                        machine.handle_response(Response::ToolResults { id, results });
                    }
                    Effect::Sleep { id, duration } => {
                        tokio::time::sleep(duration).await;
                        machine.handle_response(Response::Timeout { id });
                    }
                    Effect::Log { event } => self.handle_log_event(event),
                    Effect::CancelLlm { .. } => {}
                    Effect::ExecCode { id, code } => {
                        let result = self.run_exec_code(&code, &event_tx).await;
                        let response = match result {
                            Ok(output) => Response::ExecResult {
                                id,
                                result: Ok(output),
                            },
                            Err(error) => Response::ExecResult {
                                id,
                                result: Err(error),
                            },
                        };
                        machine.handle_response(response);
                    }
                }
            }

            (crate::MessageSequence::default(), run_offset)
        }
        .await;
        self.prompt_bridge.clear_sender();
        result
    }

    async fn prepare_turn_machine(
        &mut self,
        messages: crate::MessageSequence,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        run_offset: usize,
    ) -> Result<TurnMachine, (crate::MessageSequence, usize)> {
        macro_rules! emit {
            ($event:expr) => {
                send_session_event(event_tx, $event).await
            };
        }

        let execution_mode = self.policy.execution_mode;
        let mut session_policy = self.runtime_session_policy();
        let model = match self.prepare_provider(&mut session_policy).await {
            Ok(model) => model,
            Err(event) => {
                emit!(event);
                emit!(SessionEvent::Done);
                return Err((messages.clone(), run_offset));
            }
        };
        self.mark_phase_begin(RuntimeTurnPhase::PromptBuild);
        let tool_surface = self.session.tool_surface(&self.session_id, execution_mode);
        let mode_preamble = self.session.mode_preamble(&self.session_id, execution_mode);
        let prompt_state = SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: session_policy.clone(),
            iteration: run_offset,
            ..Default::default()
        };
        let plugin_prompt_contributions = match self
            .session
            .plugins()
            .collect_prompt_contributions(PromptHookContext {
                session_id: self.session_id.clone(),
                host: Arc::clone(&self.session_manager),
                state: crate::SessionReadView::from_graph_projection(
                    &prompt_state,
                    self.base_graph.clone(),
                    messages.shared(),
                    Arc::clone(&self.tool_calls),
                ),
                rlm_termination: self.rlm_termination.clone(),
            })
            .await
        {
            Ok(contributions) => contributions,
            Err(err) => {
                emit!(make_error_event(
                    "plugin_prompt",
                    None,
                    err.to_string(),
                    Some(err.to_string()),
                ));
                emit!(SessionEvent::Done);
                return Err((messages, run_offset));
            }
        };
        let mut all_prompt_contributions = self.session.context_prompt_contributions().to_vec();
        all_prompt_contributions.extend(plugin_prompt_contributions);
        let prepared = crate::build_turn(crate::SansIoTurnInput {
            session_id: self.session_id.clone(),
            run_session_id: session_policy.session_id.clone(),
            model,
            mode: execution_mode,
            messages,
            run_offset,
            mode_preamble,
            tool_surface,
            prompt_template: self.host.core.prompt_template.clone(),
            prompt_contributions: all_prompt_contributions,
            max_turns: session_policy.max_turns,
            model_variant: session_policy.model_variant.clone(),
            emit_llm_debug_log: self.host.core.llm_logger.is_some(),
            rlm_termination: self.rlm_termination.clone(),
            retry_policy: self.host.core.retry_policy.clone(),
        });
        self.policy = session_policy;
        self.mark_phase_end(RuntimeTurnPhase::PromptBuild);
        Ok(prepared.machine)
    }

    async fn run_llm_call(
        &mut self,
        machine: &mut TurnMachine,
        effect_id: crate::sansio::EffectId,
        request: LlmRequest,
        iteration: usize,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        match self.policy.execution_mode {
            ExecutionMode::Standard => {
                self.run_standard_llm_call(request, iteration, event_tx, cancel)
                    .await
            }
            ExecutionMode::Rlm => {
                let _ = machine;
                let _ = effect_id;
                self.run_standard_llm_call(request, iteration, event_tx, cancel)
                    .await
            }
        }
    }

    fn runtime_session_policy(&self) -> SessionPolicy {
        self.policy.clone()
    }

    fn checkpoint_state_view(
        &self,
        messages: Arc<Vec<Message>>,
        iteration: usize,
    ) -> crate::SessionReadView {
        let state = SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: self.policy.clone(),
            session_graph: crate::SessionGraph::default(),
            iteration,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
        };
        crate::SessionReadView::from_graph_projection(
            &state,
            self.base_graph.clone(),
            messages,
            Arc::clone(&self.tool_calls),
        )
    }

    async fn run_checkpoint(
        &mut self,
        machine: &mut TurnMachine,
        checkpoint: CheckpointKind,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    ) -> Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeError> {
        let mut committed = self
            .session
            .turn_injection_bridge()
            .drain()
            .map_err(|err| RuntimeError {
                code: "turn_injection_bridge".to_string(),
                message: err,
            })?;
        let injected = self
            .session
            .turn_input_injection_bridge()
            .drain()
            .map_err(|err| RuntimeError {
                code: "turn_input_injection_bridge".to_string(),
                message: err,
            })?
            .into_iter()
            .map(|item| item.message)
            .collect::<Vec<_>>();
        let plugins = Arc::clone(self.session.plugins());
        let applied = plugins
            .apply_checkpoint(CheckpointHookContext {
                session_id: self.session_id.clone(),
                checkpoint,
                state: self
                    .checkpoint_state_view(machine.materialized_messages(), machine.iteration()),
                host: Arc::clone(&self.session_manager),
            })
            .await
            .map_err(|err| RuntimeError {
                code: "plugin_checkpoint".to_string(),
                message: err.to_string(),
            })?;
        if !injected.is_empty() {
            send_session_event(
                event_tx,
                SessionEvent::InjectedTurnInputAccepted {
                    messages: injected.clone(),
                    checkpoint,
                },
            )
            .await;
        }
        committed.extend(applied.messages);
        emit_session_events(event_tx, applied.events).await;
        if let Some(abort) = applied.abort {
            return Err(RuntimeError {
                code: abort.code,
                message: abort.message,
            });
        }

        if !committed.is_empty() {
            send_session_event(
                event_tx,
                SessionEvent::InjectedMessagesCommitted {
                    messages: committed.clone(),
                    checkpoint,
                },
            )
            .await;
        }

        Ok((committed, injected))
    }

    async fn prepare_provider(
        &mut self,
        policy: &mut SessionPolicy,
    ) -> Result<String, SessionEvent> {
        match policy.provider.ensure_fresh().await {
            Ok(true) => {
                if let Some(path) = self.host.core.credential_store_path.as_ref() {
                    let _ = crate::provider::save_provider(path, &policy.provider);
                }
            }
            Err(e) => {
                return Err(make_error_event(
                    "token_refresh",
                    Some("refresh_failed"),
                    format!(
                        "Token refresh failed: {}. Re-authenticate with /provider and retry.",
                        e
                    ),
                    Some(e.to_string()),
                ));
            }
            _ => {}
        }

        let model = policy.provider.resolve_model(&policy.model);
        match policy.provider.ensure_ready().await {
            Ok(changed) => {
                if changed && let Some(path) = self.host.core.credential_store_path.as_ref() {
                    let _ = crate::provider::save_provider(path, &policy.provider);
                }
            }
            Err(e) => {
                return Err(make_error_event(
                    "llm_provider",
                    e.code.as_deref(),
                    format!(
                        "LLM provider initialization failed: {}. Run /provider to reconfigure credentials, then retry.",
                        e.message
                    ),
                    e.raw,
                ));
            }
        }

        Ok(model)
    }

    async fn transform_assistant_stream_chunk(
        &mut self,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        chunk: String,
    ) -> Result<StreamChunkOutcome, LlmCallError> {
        let original = chunk.clone();
        let transforms = self
            .session
            .plugins()
            .transform_assistant_stream(&self.session_id, chunk, Arc::clone(&self.session_manager))
            .await
            .map_err(|err| LlmCallError {
                message: err.to_string(),
                retryable: false,
                raw: None,
                code: Some("plugin_assistant_stream".to_string()),
                request_body: None,
            })?;
        let mut current = String::new();
        let mut first = true;
        let mut abort_requested = false;
        for emitted in transforms {
            if first {
                first = false;
            }
            current = emitted.value.chunk.clone();
            if emitted.value.abort_stream {
                abort_requested = true;
            }
            emit_plugin_surface_events_runtime(event_tx, &emitted.plugin_id, emitted.value.events)
                .await;
        }
        let chunk = if first { original } else { current };
        Ok(StreamChunkOutcome {
            chunk,
            abort_requested,
        })
    }

    async fn transform_assistant_response(
        &mut self,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        response: LlmResponse,
    ) -> Result<LlmResponse, LlmCallError> {
        let original = response.clone();
        let transforms = self
            .session
            .plugins()
            .transform_assistant_response(
                &self.session_id,
                response,
                Arc::clone(&self.session_manager),
            )
            .await
            .map_err(|err| LlmCallError {
                message: err.to_string(),
                retryable: false,
                raw: None,
                code: Some("plugin_assistant_response".to_string()),
                request_body: None,
            })?;
        let mut current: Option<LlmResponse> = None;
        for emitted in transforms {
            emit_plugin_surface_events_runtime(event_tx, &emitted.plugin_id, emitted.value.events)
                .await;
            current = Some(emitted.value.response);
        }
        Ok(current.unwrap_or(original))
    }

    async fn run_exec_code(
        &mut self,
        code: &str,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    ) -> Result<crate::ExecResponse, String> {
        let (session_event_tx, mut session_event_rx) = mpsc::channel::<SessionEvent>(100);
        let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel::<SandboxMessage>();
        self.session.set_message_sender(msg_tx);
        let event_tx_clone = event_tx.clone();
        let drain_handle = tokio::spawn(async move {
            while let Some(sandbox_msg) = msg_rx.recv().await {
                if sandbox_msg.kind != "final" && !event_tx_clone.is_closed() {
                    let _ = event_tx_clone
                        .send(RuntimeStreamEvent::Session(SessionEvent::Message {
                            text: sandbox_msg.text,
                            kind: sandbox_msg.kind,
                        }))
                        .await;
                }
            }
        });
        let forward_tx = event_tx.clone();
        let forward_handle = tokio::spawn(async move {
            while let Some(event) = session_event_rx.recv().await {
                send_session_event(&forward_tx, event).await;
            }
        });
        let manager = Arc::clone(&self.session_manager);
        let accept_finish = matches!(self.rlm_termination, crate::RlmTermination::Finish { .. });
        let result = self
            .session
            .run_code(
                &self.session_id,
                manager,
                &session_event_tx,
                code,
                accept_finish,
            )
            .await
            .map_err(|e| e.to_string());
        drop(session_event_tx);
        self.session.clear_message_sender();
        let _ = forward_handle.await;
        let _ = drain_handle.await;
        result
    }

    async fn run_standard_llm_call(
        &mut self,
        request: LlmRequest,
        iteration: usize,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        let debug_request = self.host.core.llm_logger.as_ref().map(|_| request.clone());
        let (llm_stream_tx, mut llm_stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
        let llm_request = LlmRequest {
            stream_events: transport_stream_events(&self.policy.provider, Some(llm_stream_tx)),
            ..request
        };

        let mut call_provider = self.policy.provider.clone();
        let mut llm_task = tokio::spawn(async move {
            let result = call_provider.complete(llm_request).await;
            (result, call_provider)
        });

        let mut text_streamed = false;
        let mut streamed_usage = LlmUsage::default();
        let mut streamed_output = StandardStreamFallback::default();
        let mut debug = LlmStreamDebugState::new();
        let mut abort_requested = false;
        let mut stream_state = StandardStreamState {
            text_streamed: &mut text_streamed,
            streamed_usage: &mut streamed_usage,
            streamed_output: &mut streamed_output,
            debug: &mut debug,
            iteration,
            abort_requested: &mut abort_requested,
        };
        let result = loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    llm_task.abort();
                    break Err(LlmCallError {
                        message: "cancelled".to_string(),
                        retryable: false,
                        raw: None,
                        code: Some("cancelled".to_string()),
                        request_body: debug_request.as_ref().map(debug_request_body),
                    });
                }
                Some(stream_event) = llm_stream_rx.recv() => {
                    if let Err(err) = self
                        .forward_standard_stream_event(event_tx, stream_event, &mut stream_state)
                        .await
                    {
                        break Err(err);
                    }
                    if *stream_state.abort_requested {
                        // A plugin stream hook asked us to end the LLM
                        // call now (e.g. RLM mask saw a closed fence).
                        // Drain the channel cheaply for events already
                        // sitting in the buffer, then hand the in-flight
                        // HTTP task + channel to a background "trailing
                        // usage catcher" so the next iteration can start
                        // without waiting on the provider's final
                        // `response.completed` SSE event (which is where
                        // Codex puts the `usage` block). The catcher
                        // appends a late-usage JSONL entry and emits a
                        // `SessionEvent::TokenUsage` delta so persistent
                        // accounting stays accurate without blocking.
                        if let Err(err) = self
                            .drain_standard_stream_queue(event_tx, &mut llm_stream_rx, &mut stream_state)
                            .await
                        {
                            break Err(err);
                        }

                        let mut resp = LlmResponse {
                            deltas: Vec::new(),
                            full_text: streamed_output.full_text(),
                            parts: Vec::new(),
                            usage: streamed_usage.clone(),
                            provider_usage: None,
                            request_body: debug_request.as_ref().map(debug_request_body),
                            http_summary: None,
                        };
                        streamed_output.apply_to_response(&mut resp);
                        let resp = match self.transform_assistant_response(event_tx, resp).await {
                            Ok(resp) => resp,
                            Err(err) => break Err(err),
                        };

                        spawn_trailing_usage_catcher(TrailingUsageCatcher {
                            llm_task,
                            llm_stream_rx,
                            event_tx: event_tx.clone(),
                            llm_logger: self.host.core.llm_logger.clone(),
                            session_id: self.session_id.clone(),
                            iteration,
                            initial_usage: streamed_usage.clone(),
                        });

                        break Ok(resp);
                    }
                }
                join = &mut llm_task => {
                    let (result, provider_after) = match join {
                        Ok(v) => v,
                        Err(e) => break Err(LlmCallError {
                            message: format!("internal task failed: {e}"),
                            retryable: false,
                            raw: None,
                            code: Some("task_join_failed".to_string()),
                            request_body: debug_request.as_ref().map(debug_request_body),
                        }),
                    };
                    self.policy.provider = provider_after;
                    if let Err(err) = self
                        .drain_standard_stream_queue(event_tx, &mut llm_stream_rx, &mut stream_state)
                        .await
                    {
                        break Err(err);
                    }
                    match result {
                        Ok(mut resp) => {
                            if response_usage_is_empty(&resp.usage) {
                                resp.usage = streamed_usage.clone();
                            }
                            streamed_output.apply_to_response(&mut resp);
                            if resp.request_body.is_none() {
                                resp.request_body = debug_request
                                    .as_ref()
                                    .map(debug_request_body);
                            }
                            let resp = match self.transform_assistant_response(event_tx, resp).await {
                                Ok(resp) => resp,
                                Err(err) => break Err(err),
                            };
                            break Ok(resp)
                        }
                        Err(e) => break Err(LlmCallError {
                            message: e.message,
                            retryable: e.retryable,
                            raw: e.raw,
                            code: e.code,
                            request_body: e
                                .request_body
                                .or_else(|| debug_request.as_ref().map(debug_request_body)),
                        }),
                    }
                }
            }
        };

        if let Err(err) = &result {
            tracing::error!(
                session_id = %self.session_id,
                turn = iteration,
                retryable = err.retryable,
                code = ?err.code,
                raw_present = err.raw.is_some(),
                request_body_present = err.request_body.is_some(),
                message = %err.message,
                "llm call failed"
            );
        }
        self.llm_stream_summaries.insert(iteration, debug.summary);
        (result, text_streamed)
    }

    async fn run_tool_calls(
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
        let manager = Arc::clone(&self.session_manager);
        let projector_manager = Arc::clone(&manager);
        let dispatch = Arc::new(ToolDispatchContext {
            plugins: Arc::clone(&plugins),
            tools: self.session.tools(),
            surface: self
                .session
                .tool_surface(&self.session_id, self.policy.execution_mode),
            host: Arc::clone(&manager),
            session_id: self.session_id.clone(),
            event_tx: tool_event_tx,
            turn_injection_bridge: self.session.turn_injection_bridge().clone(),
        });
        // Partition pending tool calls by declared [`ToolExecutionMode`]:
        // parallel-safe tools spawn onto a JoinSet to run concurrently, while
        // tools declared `Serial` (apply_patch, exec_command, write_stdin,
        // followup_task, ...) run one-at-a-time afterwards so they can't
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
            let projector_manager = Arc::clone(&projector_manager);
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
                Arc::clone(&projector_manager),
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

    fn handle_log_event(&mut self, event: crate::sansio::LogEvent) {
        if self.host.core.llm_logger.is_none() {
            return;
        }

        match event {
            crate::sansio::LogEvent::LlmDebug {
                session_id,
                iteration,
                usage,
                provider_usage,
                request_body,
                response_text,
                response_parts,
            } => {
                let stream_summary = self.llm_stream_summaries.remove(&iteration);
                let mut entry = serde_json::json!({
                    "turn": iteration,
                    "ts": chrono::Utc::now().to_rfc3339(),
                    "session_id": session_id,
                    "request": request_body,
                    "response": response_text,
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cached_input_tokens": usage.cached_input_tokens,
                        "reasoning_tokens": usage.reasoning_tokens,
                    }
                });
                if let Some(provider_usage) = provider_usage
                    && let Some(object) = entry.as_object_mut()
                {
                    object.insert("provider_usage".to_string(), provider_usage);
                }
                if let Some(parts) = response_parts
                    && let Some(object) = entry.as_object_mut()
                {
                    object.insert("response_parts".to_string(), parts);
                }
                if let Some(summary) = stream_summary
                    && let Some(object) = entry.as_object_mut()
                {
                    object.insert("stream_summary".to_string(), summary.to_json());
                }
                self.append_llm_debug_entry(entry);
            }
            crate::sansio::LogEvent::LlmError {
                session_id,
                iteration,
                request_body,
                message,
                retryable,
                raw,
                code,
            } => {
                let stream_summary = self.llm_stream_summaries.remove(&iteration);
                let mut entry = serde_json::json!({
                    "kind": "llm_error",
                    "turn": iteration,
                    "ts": chrono::Utc::now().to_rfc3339(),
                    "session_id": session_id,
                    "request": request_body,
                    "error": {
                        "message": message,
                        "retryable": retryable,
                        "code": code,
                    }
                });
                if let Some(raw) = raw
                    && let Some(object) = entry.as_object_mut()
                {
                    object.insert("raw".to_string(), serde_json::Value::String(raw));
                }
                if let Some(summary) = stream_summary
                    && let Some(object) = entry.as_object_mut()
                {
                    object.insert("stream_summary".to_string(), summary.to_json());
                }
                self.append_llm_debug_entry(entry);
            }
        }
    }

    fn append_llm_debug_entry(&self, entry: serde_json::Value) {
        if let Some(logger) = &self.host.core.llm_logger {
            logger.append(&entry);
        }
    }

    fn log_llm_stream_event(&self, debug: &mut LlmStreamDebugState, log: LlmStreamEventLog<'_>) {
        if self.host.core.llm_logger.is_none() {
            return;
        }

        let elapsed_ms = debug.elapsed_ms();
        if matches!(log.event_type, "delta" | "text_part") {
            debug
                .summary
                .record_text_chunk(log.text.visible, elapsed_ms);
        }

        let mut entry = serde_json::json!({
            "kind": "stream_event",
            "turn": log.iteration,
            "ts": chrono::Utc::now().to_rfc3339(),
            "session_id": log.session_id,
            "sequence": debug.next_sequence(),
            "elapsed_ms": elapsed_ms,
            "event": log.event_type,
        });

        if let Some(object) = entry.as_object_mut() {
            if let Some(text) = log.text.raw {
                object.insert(
                    "raw_text".to_string(),
                    serde_json::Value::String(text.to_string()),
                );
                object.insert(
                    "raw_chars".to_string(),
                    serde_json::Value::from(text.chars().count() as u64),
                );
            }
            if let Some(text) = log.text.visible {
                object.insert(
                    "visible_text".to_string(),
                    serde_json::Value::String(text.to_string()),
                );
                object.insert(
                    "visible_chars".to_string(),
                    serde_json::Value::from(text.chars().count() as u64),
                );
            }
            if let Some(usage) = log.usage {
                object.insert(
                    "usage".to_string(),
                    serde_json::json!({
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cached_input_tokens": usage.cached_input_tokens,
                        "reasoning_tokens": usage.reasoning_tokens,
                    }),
                );
            }
            if let Some(tool_call) = log.tool_call {
                object.insert(
                    "call_id".to_string(),
                    serde_json::Value::String(tool_call.call_id.to_string()),
                );
                object.insert(
                    "tool_name".to_string(),
                    serde_json::Value::String(tool_call.tool_name.to_string()),
                );
                object.insert(
                    "input_json".to_string(),
                    serde_json::Value::String(tool_call.input_json.to_string()),
                );
            }
        }

        self.append_llm_debug_entry(entry);
    }

    async fn forward_standard_stream_event(
        &mut self,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        stream_event: LlmStreamEvent,
        state: &mut StandardStreamState<'_>,
    ) -> Result<(), LlmCallError> {
        match stream_event {
            LlmStreamEvent::Delta(delta) => {
                if !delta.is_empty() {
                    *state.text_streamed = true;
                    let raw_delta = self.host.core.llm_logger.as_ref().map(|_| delta.clone());
                    let outcome = self
                        .transform_assistant_stream_chunk(event_tx, delta)
                        .await?;
                    if outcome.abort_requested {
                        *state.abort_requested = true;
                    }
                    let delta = outcome.chunk;
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            session_id: &self.session_id,
                            iteration: state.iteration,
                            event_type: "delta",
                            text: LlmDebugText {
                                raw: raw_delta.as_deref(),
                                visible: Some(&delta),
                            },
                            usage: None,
                            tool_call: None,
                        },
                    );
                    if !delta.is_empty() {
                        state.streamed_output.push_text(&delta);
                        send_session_event(event_tx, SessionEvent::TextDelta { content: delta })
                            .await;
                    }
                }
            }
            LlmStreamEvent::ReasoningDelta(delta) => {
                if !delta.is_empty() {
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            session_id: &self.session_id,
                            iteration: state.iteration,
                            event_type: "reasoning_delta",
                            text: LlmDebugText {
                                raw: None,
                                visible: Some(&delta),
                            },
                            usage: None,
                            tool_call: None,
                        },
                    );
                    // Delta-only streaming path (fix 1.3a display). No
                    // encrypted content yet — that arrives with the full
                    // item on `output_item.done` (fix 1.3b).
                    state
                        .streamed_output
                        .push_reasoning(delta.clone(), None, Vec::new(), None);
                    send_session_event(event_tx, SessionEvent::ReasoningDelta { content: delta })
                        .await;
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::Text { text }) => {
                if !text.is_empty() {
                    *state.text_streamed = true;
                    let raw_text = self.host.core.llm_logger.as_ref().map(|_| text.clone());
                    let outcome = self
                        .transform_assistant_stream_chunk(event_tx, text)
                        .await?;
                    if outcome.abort_requested {
                        *state.abort_requested = true;
                    }
                    let text = outcome.chunk;
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            session_id: &self.session_id,
                            iteration: state.iteration,
                            event_type: "text_part",
                            text: LlmDebugText {
                                raw: raw_text.as_deref(),
                                visible: Some(&text),
                            },
                            usage: None,
                            tool_call: None,
                        },
                    );
                    if !text.is_empty() {
                        state.streamed_output.push_text(&text);
                        send_session_event(event_tx, SessionEvent::TextDelta { content: text })
                            .await;
                    }
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                item_id,
                signature,
            }) => {
                self.log_llm_stream_event(
                    state.debug,
                    LlmStreamEventLog {
                        session_id: &self.session_id,
                        iteration: state.iteration,
                        event_type: "tool_call_part",
                        text: LlmDebugText {
                            raw: None,
                            visible: None,
                        },
                        usage: None,
                        tool_call: Some(LlmDebugToolCall {
                            call_id: &call_id,
                            tool_name: &tool_name,
                            input_json: &input_json,
                        }),
                    },
                );
                state
                    .streamed_output
                    .push_tool_call(call_id, tool_name, input_json, item_id, signature);
            }
            LlmStreamEvent::Part(LlmOutputPart::Reasoning {
                text,
                signature: _,
                redacted: _,
                item_id,
                encrypted_content,
                summary,
            }) => {
                if !text.is_empty() {
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            session_id: &self.session_id,
                            iteration: state.iteration,
                            event_type: "reasoning_part",
                            text: LlmDebugText {
                                raw: None,
                                visible: Some(&text),
                            },
                            usage: None,
                            tool_call: None,
                        },
                    );
                    send_session_event(
                        event_tx,
                        SessionEvent::ReasoningDelta {
                            content: text.clone(),
                        },
                    )
                    .await;
                }
                state
                    .streamed_output
                    .push_reasoning(text, item_id, summary, encrypted_content);
            }
            LlmStreamEvent::Usage(usage) => {
                self.log_llm_stream_event(
                    state.debug,
                    LlmStreamEventLog {
                        session_id: &self.session_id,
                        iteration: state.iteration,
                        event_type: "usage",
                        text: LlmDebugText {
                            raw: None,
                            visible: None,
                        },
                        usage: Some(&usage),
                        tool_call: None,
                    },
                );
                *state.streamed_usage = usage;
            }
        }
        Ok(())
    }

    async fn drain_standard_stream_queue(
        &mut self,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        llm_stream_rx: &mut tokio::sync::mpsc::UnboundedReceiver<LlmStreamEvent>,
        state: &mut StandardStreamState<'_>,
    ) -> Result<(), LlmCallError> {
        while let Ok(stream_event) = llm_stream_rx.try_recv() {
            self.forward_standard_stream_event(event_tx, stream_event, state)
                .await?;
        }
        Ok(())
    }
}

fn response_usage_is_empty(usage: &LlmUsage) -> bool {
    usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_input_tokens == 0
        && usage.reasoning_tokens == 0
}

pub(super) fn llm_response_has_content(response: &LlmResponse) -> bool {
    if !response.full_text.is_empty() {
        return true;
    }
    response.parts.iter().any(|part| match part {
        LlmOutputPart::Text { text } => !text.is_empty(),
        // Reasoning-only responses still count as "has content" so the
        // adapter's stream-fallback buffer is preserved for replay.
        LlmOutputPart::Reasoning { .. } => true,
        LlmOutputPart::ToolCall { .. } => true,
    })
}

fn debug_request_body(req: &LlmRequest) -> String {
    let messages = req
        .messages
        .iter()
        .map(|message| {
            serde_json::json!({
                "role": format!("{:?}", message.role).to_ascii_lowercase(),
                "blocks": message.blocks.len(),
            })
        })
        .collect::<Vec<_>>();
    let tools = req
        .tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "model": req.model,
        "messages": messages,
        "attachments": req.attachments.len(),
        "tools": tools,
        "tool_choice": format!("{:?}", req.tool_choice).to_ascii_lowercase(),
        "model_variant": req.model_variant,
        "session_id": req.session_id,
        "output_spec": match &req.output_spec {
            None => serde_json::Value::Null,
            Some(crate::llm::types::LlmOutputSpec::JsonObject) => {
                serde_json::json!({ "type": "json_object" })
            }
            Some(crate::llm::types::LlmOutputSpec::JsonSchema(schema)) => {
                serde_json::json!({
                    "type": "json_schema",
                    "name": schema.name,
                    "schema": schema.schema,
                    "strict": schema.strict,
                })
            }
        },
        "stream": req.stream_events.is_some(),
    })
    .to_string()
}

/// Parameters for the background task that catches trailing
/// `LlmStreamEvent::Usage` events after an RLM stream-mask abort.
/// Owning the channel + task lets the turn driver return the LLM
/// response immediately while the provider's `response.completed`
/// SSE event is still in flight.
struct TrailingUsageCatcher {
    llm_task: tokio::task::JoinHandle<(
        Result<LlmResponse, crate::llm::transport::LlmTransportError>,
        crate::ProviderHandle,
    )>,
    llm_stream_rx: tokio::sync::mpsc::UnboundedReceiver<LlmStreamEvent>,
    event_tx: mpsc::Sender<RuntimeStreamEvent>,
    llm_logger: Option<Arc<dyn crate::runtime::host::LlmCallLogger>>,
    session_id: String,
    iteration: usize,
    initial_usage: LlmUsage,
}

/// Wait up to 2s for a late `Usage` event from the provider. If it
/// arrives, emit a `SessionEvent::TokenUsage` delta so cumulative
/// accounting stays accurate, and (when an LLM debug log is configured)
/// append a `token_usage_patch` JSONL entry so the benchmark exporter
/// — which sums `usage` blocks across trace lines — picks the delta up.
/// Always aborts the in-flight `llm_task` when done so the HTTP socket
/// closes promptly.
fn spawn_trailing_usage_catcher(args: TrailingUsageCatcher) {
    let TrailingUsageCatcher {
        llm_task,
        mut llm_stream_rx,
        event_tx,
        llm_logger,
        session_id,
        iteration,
        initial_usage,
    } = args;
    tokio::spawn(async move {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(2_000);
        let mut latest = initial_usage.clone();
        let mut saw_usage = false;
        loop {
            match tokio::time::timeout_at(deadline, llm_stream_rx.recv()).await {
                Err(_) | Ok(None) => break,
                Ok(Some(LlmStreamEvent::Usage(usage))) => {
                    latest = usage;
                    saw_usage = true;
                    break;
                }
                Ok(Some(_)) => continue,
            }
        }
        llm_task.abort();
        if !saw_usage || latest == initial_usage {
            return;
        }
        let delta = LlmUsage {
            input_tokens: (latest.input_tokens - initial_usage.input_tokens).max(0),
            output_tokens: (latest.output_tokens - initial_usage.output_tokens).max(0),
            cached_input_tokens: (latest.cached_input_tokens - initial_usage.cached_input_tokens)
                .max(0),
            reasoning_tokens: (latest.reasoning_tokens - initial_usage.reasoning_tokens).max(0),
        };
        if delta == LlmUsage::default() {
            return;
        }
        // Session-level token ledger update. sansio's cumulative tracker
        // adds `usage` into the running total on every TokenUsage event,
        // so a delta-shaped second event correctly bumps totals without
        // resetting them.
        let token_delta = TokenUsage {
            input_tokens: delta.input_tokens,
            output_tokens: delta.output_tokens,
            cached_input_tokens: delta.cached_input_tokens,
            reasoning_tokens: delta.reasoning_tokens,
        };
        let _ = event_tx
            .send(RuntimeStreamEvent::Session(SessionEvent::TokenUsage {
                iteration,
                usage: token_delta.clone(),
                cumulative: token_delta.clone(),
            }))
            .await;
        // Benchmark exporter reads usage from the LLM call log. Append
        // a patch entry there so `usage` sums across trace lines stay
        // correct even when the initial request line landed with zeros.
        if let Some(logger) = llm_logger {
            let entry = serde_json::json!({
                "kind": "token_usage_patch",
                "turn": iteration,
                "ts": chrono::Utc::now().to_rfc3339(),
                "session_id": session_id,
                "usage": {
                    "input_tokens": delta.input_tokens,
                    "output_tokens": delta.output_tokens,
                    "cached_input_tokens": delta.cached_input_tokens,
                    "reasoning_tokens": delta.reasoning_tokens,
                }
            });
            let _ = tokio::task::spawn_blocking(move || {
                logger.append(&entry);
            })
            .await;
        }
    });
}
