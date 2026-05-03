use super::*;

impl LashRuntime {
    fn has_overflow_error(assembled: &AssembledTurn) -> bool {
        assembled.errors.iter().any(|issue| {
            let lower = issue.message.to_lowercase();
            lower.contains("prompt is too long")
                || lower.contains("context_length_exceeded")
                || lower.contains("maximum context length")
                || lower.contains("too many tokens")
                || lower.contains("exceeds the maximum number of tokens")
                || lower.contains("request too large")
        })
    }

    fn max_context_tokens(&self) -> usize {
        self.policy
            .max_context_tokens
            .expect("lash runtime requires explicit max_context_tokens")
    }
    #[doc(hidden)]
    pub fn set_turn_phase_probe(&mut self, probe: Arc<dyn RuntimeTurnPhaseProbe>) {
        self.turn_phase_probe = Some(probe);
    }

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

    async fn handle_pending_prompt(
        prompt: PendingPrompt,
        plugins: Option<&Arc<crate::PluginSession>>,
        manager: &Arc<dyn RuntimeSessionHost>,
        events: &dyn EventSink,
        assembler: &mut TurnAssembler,
    ) {
        if let Some(plugins) = plugins {
            match plugins
                .on_prompt_request(crate::PromptRequestHookContext {
                    session_id: plugins.session_id().to_string(),
                    request: prompt.request.clone(),
                    host: manager.clone(),
                })
                .await
            {
                Ok(emitted) => {
                    for surface in emitted {
                        let surface_events = crate::plugin::plugin_surface_session_events(
                            &surface.plugin_id,
                            vec![surface.value],
                        );
                        for event in surface_events {
                            assembler.push(&event);
                            emit_session_event_to_sink(events, event).await;
                        }
                    }
                }
                Err(err) => {
                    let event = make_error_event(
                        "plugin_prompt_request",
                        None,
                        err.to_string(),
                        Some(err.to_string()),
                    );
                    assembler.push(&event);
                    emit_session_event_to_sink(events, event).await;
                }
            }
        }
        let event = SessionEvent::Prompt {
            request: prompt.request,
            response_tx: prompt.response_tx,
        };
        assembler.push(&event);
        emit_session_event_to_sink(events, event).await;
    }

    /// Run a single turn and stream events to the host sink.
    /// Includes overflow recovery: if the LLM rejects the prompt as too long,
    /// the context is force-compacted and the turn is retried once.
    pub async fn stream_turn(
        &mut self,
        input: TurnInput,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        // Capture lengths instead of Arcs: holding the read-model Arcs across
        // the turn forces every progress-boundary persist to clone the whole
        // read_messages / read_tool_calls Vec via Arc::make_mut.
        let saved_read_model = self.state.read_model();
        let saved_messages_len = saved_read_model.messages.len();
        let saved_tool_calls_len = saved_read_model.tool_calls.len();
        let saved_prompt_usage = self.state.last_prompt_usage.clone();

        let assembled = self
            .stream_turn_inner(input.clone(), events, cancel.clone())
            .await?;

        if !self.overflow_recovery_attempted && Self::has_overflow_error(&assembled) {
            self.overflow_recovery_attempted = true;
            // Restore pre-turn state so the retry appends the user message cleanly.
            // The turn only appends to the active read state, so the original pre-turn
            // state is exactly the prefix of the current read state.
            let current_read_model = self.state.read_model();
            let saved_messages: Vec<_> = current_read_model.messages[..saved_messages_len].to_vec();
            let saved_tool_calls: Vec<_> =
                current_read_model.tool_calls[..saved_tool_calls_len].to_vec();
            self.state
                .replace_active_read_state(&saved_messages, &saved_tool_calls);
            self.state.last_prompt_usage = saved_prompt_usage;
            // Force-compact: strip images, prune, summarize.
            let _ = self
                .rewrite_history(crate::RewriteTrigger::OverflowRecovery)
                .await;
            let retry = self.stream_turn_inner(input, events, cancel).await?;
            self.overflow_recovery_attempted = false;
            return Ok(retry);
        }
        self.overflow_recovery_attempted = false;
        Ok(assembled)
    }

    async fn stream_turn_inner(
        &mut self,
        input: TurnInput,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.refresh_session_graph_from_store().await;
        let previous_prompt_usage = self.state.last_prompt_usage.clone();
        let normalized = match self.normalize_input_items(&input.items, &input.image_blobs) {
            Ok(items) => items,
            Err(e) => {
                self.state.last_prompt_usage = None;
                let mut assembler = TurnAssembler::default();
                let error_event = SessionEvent::Error {
                    message: e.clone(),
                    envelope: Some(crate::session_model::ErrorEnvelope {
                        kind: "input_validation".to_string(),
                        code: Some("invalid_turn_input".to_string()),
                        user_message: e,
                        raw: None,
                    }),
                };
                assembler.push(&error_event);
                emit_session_event_to_sink(events, error_event).await;
                assembler.push(&SessionEvent::Done);
                emit_session_event_to_sink(events, SessionEvent::Done).await;
                return Ok(assembler.finish(
                    self.state.export_state(),
                    false,
                    None,
                    &self.host.core.sanitizer,
                    &self.host.core.termination,
                ));
            }
        };
        if self.host.core.trace_sink.is_some() {
            let mut trace_metadata = std::collections::BTreeMap::new();
            trace_metadata.insert(
                "input_item_count".to_string(),
                serde_json::json!(normalized.len()),
            );
            trace_metadata.insert(
                "has_user_input".to_string(),
                serde_json::json!(input.user_input.is_some()),
            );
            crate::trace::emit_trace(
                &self.host.core.trace_sink,
                &self.host.core.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(self.state.session_id.clone())
                    .for_iteration(self.state.iteration + 1),
                lash_trace::TraceEvent::TurnStarted {
                    metadata: trace_metadata,
                },
            );
        }

        let base_read_model = self.state.read_model();
        let base_messages = base_read_model.messages;
        let base_render_cache = base_read_model.prompt_render_cache;
        let mut turn_delta = Vec::new();
        let mode = input.mode.unwrap_or(RunMode::Normal);
        let mode_msg = match mode {
            RunMode::Normal => None,
        };
        if let Some(content) = mode_msg {
            let sys_id = fresh_message_id();
            turn_delta.push(Message {
                id: sys_id.clone(),
                role: MessageRole::System,
                parts: shared_parts(vec![Part {
                    id: format!("{}.p0", sys_id),
                    kind: PartKind::Text,
                    content,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                }]),
                user_input: None,
                origin: None,
            });
        }

        let user_id = fresh_message_id();
        let mut user_parts: Vec<Part> = Vec::new();
        for item in normalized {
            match item {
                NormalizedItem::Text(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    user_parts.push(Part {
                        id: format!("{}.p{}", user_id, user_parts.len()),
                        kind: PartKind::Text,
                        content: text,
                        attachment: None,
                        tool_call_id: None,
                        tool_name: None,
                        tool_item_id: None,
                        tool_signature: None,
                        prune_state: PruneState::Intact,
                        reasoning_meta: None,
                        response_meta: None,
                    });
                }
                NormalizedItem::Image(reference) => {
                    user_parts.push(Part {
                        id: format!("{}.p{}", user_id, user_parts.len()),
                        kind: PartKind::Image,
                        content: String::new(),
                        attachment: Some(crate::session_model::message::PartAttachment {
                            reference,
                        }),
                        tool_call_id: None,
                        tool_name: None,
                        tool_item_id: None,
                        tool_signature: None,
                        prune_state: PruneState::Intact,
                        reasoning_meta: None,
                        response_meta: None,
                    });
                }
            }
        }
        if user_parts.is_empty() {
            user_parts.push(Part {
                id: format!("{}.p0", user_id),
                kind: PartKind::Text,
                content: String::new(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            });
        }
        reassign_part_ids(&user_id, &mut user_parts);
        turn_delta.push(Message {
            id: user_id.clone(),
            role: MessageRole::User,
            parts: shared_parts(user_parts),
            user_input: input.user_input.clone(),
            origin: None,
        });

        let manager = self
            .runtime_session_manager_for_turn(None, None)
            .map_err(|err| RuntimeError {
                code: "plugin_session_manager".to_string(),
                message: err.to_string(),
            })?;
        let plugin_session = self
            .session
            .as_ref()
            .map(|s| Arc::clone(s.plugins()))
            .ok_or_else(|| RuntimeError {
                code: "context_prepare_turn".to_string(),
                message: "runtime session not available".to_string(),
            })?;
        let turn_ctx = crate::TurnTransformContext {
            session_id: self.state.session_id.clone(),
            state: self.read_snapshot().read_view(),
            prompt_usage: previous_prompt_usage.clone(),
            max_context_tokens: Some(LashRuntime::max_context_tokens(self)),
            host: manager.clone(),
        };
        self.mark_phase_begin(RuntimeTurnPhase::ContextTransform);
        let prepared_context = plugin_session
            .prepare_turn_context(
                &turn_ctx,
                crate::session_model::context::PreparedContext {
                    messages: crate::MessageSequence::from_base_and_delta(
                        base_messages,
                        turn_delta,
                    )
                    .with_base_render_cache(base_render_cache),
                    ..Default::default()
                },
            )
            .await
            .map_err(|err| RuntimeError {
                code: "context_prepare_turn".to_string(),
                message: err.to_string(),
            })?;
        self.mark_phase_end(RuntimeTurnPhase::ContextTransform);
        // Release the read-view's graph clone before the rest of the turn
        // runs. Keeping it alive into `stream_prepared_turn` forces the
        // post-turn `append_active_read_delta` to deep-clone the session
        // graph (Arc::make_mut with refcount > 1).
        drop(turn_ctx);
        let messages = prepared_context.messages;
        if let Some(session) = self.session.as_mut() {
            session.set_context_surface(
                prepared_context.tool_providers,
                prepared_context.prompt_contributions,
                prepared_context.include_base_tools,
            );
        }

        self.state.last_prompt_usage = None;

        self.stream_prepared_turn(
            messages,
            previous_prompt_usage,
            input.mode_turn_options.clone(),
            events,
            cancel,
        )
        .await
    }

    /// Run a single turn and return only the assembled terminal result.
    pub async fn run_turn_assembled(
        &mut self,
        input: TurnInput,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.stream_turn(input, &NoopEventSink, cancel).await
    }

    /// Run a turn using host-prepared message history.
    pub async fn stream_prepared_turn(
        &mut self,
        messages: crate::MessageSequence,
        _previous_prompt_usage: Option<PromptUsage>,
        mode_turn_options: Option<crate::ModeTurnOptions>,
        events: &dyn EventSink,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let prompt_bridge = HostPromptBridge::new();
        let (event_tx, mut event_rx) = mpsc::channel::<RuntimeStreamEvent>(100);
        let child_usage_event_relay = ChildUsageEventRelay::new(event_tx.clone());
        let manager = self
            .runtime_session_manager_for_turn(
                Some(prompt_bridge.clone()),
                Some(child_usage_event_relay.clone()),
            )
            .map_err(|err| RuntimeError {
                code: "plugin_session_manager".to_string(),
                message: err.to_string(),
            })?;
        let (prompt_tx, mut prompt_rx) = tokio::sync::mpsc::unbounded_channel::<PendingPrompt>();
        prompt_bridge.set_sender(prompt_tx);
        let prompt_hook_manager = manager.clone();
        let prompt_plugins = self
            .session
            .as_ref()
            .map(|session| Arc::clone(session.plugins()));
        let plugins = {
            let session = self
                .session
                .as_ref()
                .expect("lash runtime session must be available");
            Arc::clone(session.plugins())
        };
        let capture_text_deltas =
            self.policy.provider.requires_streaming() || plugins.has_assistant_stream_hooks();
        let mut assembler = TurnAssembler::new(capture_text_deltas);
        self.mark_phase_begin(RuntimeTurnPhase::BeforeTurnHooks);
        // Block-scope the pinned future so it (and its captured
        // `SessionReadView` clone of the session graph) drops before the
        // post-turn `append_active_read_delta` mutation. Keeping it alive
        // across the turn forces `Arc::make_mut` to deep-clone
        // `SessionGraphData`.
        let prepared = {
            let prepare_turn = plugins.prepare_turn(PrepareTurnRequest {
                session_id: self.state.session_id.clone(),
                state: self.read_snapshot().read_view(),
                messages,
                host: manager.clone(),
            });
            tokio::pin!(prepare_turn);

            loop {
                tokio::select! {
                    prepared = &mut prepare_turn => {
                        let prepared = prepared.map_err(|err| RuntimeError {
                            code: "plugin_prepare_turn".to_string(),
                            message: err.to_string(),
                        })?;
                        self.mark_phase_end(RuntimeTurnPhase::BeforeTurnHooks);
                        break prepared;
                    }
                    maybe_event = event_rx.recv() => {
                        if let Some(event) = maybe_event {
                            match event {
                                RuntimeStreamEvent::Session(event) => {
                                    assembler.push(&event);
                                    emit_session_event_to_sink(events, event).await;
                                }
                            }
                        }
                    }
                    maybe_prompt = prompt_rx.recv() => {
                        if let Some(prompt) = maybe_prompt {
                            Self::handle_pending_prompt(
                                prompt,
                                prompt_plugins.as_ref(),
                                &prompt_hook_manager,
                                events,
                                &mut assembler,
                            ).await;
                        }
                    }
                }
            }
        };
        for event in &prepared.events {
            assembler.push(event);
        }
        emit_session_events_to_sink(events, prepared.events).await;
        if let Some(abort) = prepared.abort {
            prompt_bridge.clear_sender();
            drop(event_tx);

            let mut turn_pipeline = TurnCommitPipeline::from_state(self.state.clone());
            turn_pipeline.apply_prepared_messages(&prepared.messages);
            let state = turn_pipeline.into_final_state();
            let issue = TurnIssue {
                kind: "plugin".to_string(),
                code: Some(abort.code),
                message: abort.message.clone(),
                raw: None,
            };
            let error_event = SessionEvent::Error {
                message: abort.message,
                envelope: Some(crate::session_model::ErrorEnvelope {
                    kind: "plugin".to_string(),
                    code: issue.code.clone(),
                    user_message: issue.message.clone(),
                    raw: None,
                }),
            };
            assembler.push(&error_event);
            emit_session_event_to_sink(events, error_event).await;
            assembler.push(&SessionEvent::Done);
            emit_session_event_to_sink(events, SessionEvent::Done).await;
            return Ok(assembler.finish(
                state.export_state(),
                cancel.is_cancelled(),
                Some(issue),
                &self.host.core.sanitizer,
                &self.host.core.termination,
            ));
        }
        let mut turn_pipeline = TurnCommitPipeline::from_state(self.state.clone());
        let store = self
            .session
            .as_ref()
            .and_then(|session| session.history_store());
        turn_pipeline
            .prepared_checkpoint(
                store.as_ref().map(|store| store.as_ref()),
                self.policy.clone(),
                self.state.iteration,
                &prepared.messages,
                self.session.as_mut(),
            )
            .await
            .map_err(|err| RuntimeError {
                code: "store_commit_failed".to_string(),
                message: err.to_string(),
            })?;
        let cancel_state = cancel.clone();
        let session = self
            .session
            .take()
            .expect("lash runtime session must be available");
        let mut driver = RuntimeTurnDriver {
            session,
            policy: self.policy.clone(),
            host: self.host.clone(),
            session_id: self.state.session_id.clone(),
            turn_pipeline,
            llm_stream_summaries: HashMap::new(),
            session_manager: manager,
            prompt_bridge,
            mode_turn_options: mode_turn_options
                .clone()
                .unwrap_or_else(|| self.mode_turn_options.clone()),
            turn_phase_probe: self.turn_phase_probe.clone(),
        };
        let run_offset = self.state.iteration;
        let run_task = tokio::spawn(async move {
            let (new_messages, new_iteration) = driver
                .run(prepared.messages, event_tx, cancel, run_offset)
                .await;
            (driver, new_messages, new_iteration)
        });
        tokio::pin!(run_task);

        self.mark_phase_begin(RuntimeTurnPhase::EffectLoop);
        let (driver, new_messages, new_iteration) = loop {
            tokio::select! {
                maybe_event = event_rx.recv() => {
                    if let Some(event) = maybe_event {
                        match event {
                            RuntimeStreamEvent::Session(event) => {
                                assembler.push(&event);
                                emit_session_event_to_sink(events, event).await;
                            }
                        }
                    }
                }
                maybe_prompt = prompt_rx.recv() => {
                    if let Some(prompt) = maybe_prompt {
                        Self::handle_pending_prompt(
                            prompt,
                            prompt_plugins.as_ref(),
                            &prompt_hook_manager,
                            events,
                            &mut assembler,
                        ).await;
                    }
                }
                joined = &mut run_task => {
                    child_usage_event_relay.clear();
                    let joined = match joined {
                        Ok(v) => v,
                        Err(e) => {
                            let issue = TurnIssue {
                                kind: "runtime".to_string(),
                                code: Some("run_task_join_failed".to_string()),
                                message: format!("Runtime turn task failed: {e}"),
                                raw: None,
                            };
                            return Ok(assembler.finish(
                                self.state.export_state(),
                                cancel_state.is_cancelled(),
                                Some(issue),
                                &self.host.core.sanitizer,
                                &self.host.core.termination,
                            ));
                        }
                    };
                    break joined;
                }
            }
        };
        while let Some(event) = event_rx.recv().await {
            match event {
                RuntimeStreamEvent::Session(event) => {
                    assembler.push(&event);
                    emit_session_event_to_sink(events, event).await;
                }
            }
        }
        self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            new_message_count = new_messages.len(),
            tool_call_count = assembler.tool_calls.len(),
            "runtime post-run_task"
        );

        // Drain the shared token ledger (child sessions + direct
        // completions + async OM observers/reflectors). Merge it after
        // restoring the latest progress checkpoint state so in-turn
        // progress commits cannot wipe live child usage.
        let child_ledger = {
            let mut ledger = self.shared_token_ledger.lock().expect("token ledger lock");
            std::mem::take(&mut *ledger)
        };

        let RuntimeTurnDriver {
            session,
            policy,
            mut turn_pipeline,
            ..
        } = driver;
        self.session = Some(session);
        self.policy = policy;
        turn_pipeline.state_mut().policy = self.policy.clone();
        turn_pipeline.state_mut().iteration = new_iteration;
        let mut turn_usage_delta = child_ledger.clone();
        if assembler.token_usage.total() > 0 || assembler.token_usage.cached_input_tokens > 0 {
            let entry = TokenLedgerEntry {
                source: "turn".to_string(),
                model: self.policy.model.clone(),
                usage: assembler.token_usage.clone(),
            };
            turn_usage_delta.push(entry);
        }
        let turn_usage_delta = merge_usage_delta_entries(turn_usage_delta);
        turn_pipeline.finalize_turn_read_state(
            new_messages,
            &assembler.tool_calls,
            cancel_state.is_cancelled(),
        );
        if assembler.token_usage.total() > 0 || assembler.token_usage.cached_input_tokens > 0 {
            turn_pipeline.state_mut().token_usage = assembler.token_usage.clone();
        }

        let last_prompt_usage = assembler
            .last_llm_usage()
            .and_then(|usage| normalize_prompt_usage(self.policy.provider.as_dyn(), usage));
        let finalize_manager = if self.session.is_some() {
            Some(
                self.runtime_session_manager_for_turn(None, None)
                    .map_err(|err| RuntimeError {
                        code: "plugin_session_manager".to_string(),
                        message: err.to_string(),
                    })?,
            )
        } else {
            None
        };
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            state_message_count = turn_pipeline.state_mut().read_model().messages.len(),
            graph_node_count = turn_pipeline.state_mut().session_graph.nodes.len(),
            token_ledger_entries = turn_pipeline.state_mut().token_ledger.len(),
            "runtime before assembler.finish"
        );
        turn_pipeline.state_mut().last_prompt_usage = last_prompt_usage.clone();
        let assembled_state = turn_pipeline.export_state_for_assembly();
        let assembled = assembler.finish(
            assembled_state,
            cancel_state.is_cancelled(),
            None,
            &self.host.core.sanitizer,
            &self.host.core.termination,
        );
        tracing::debug!(
            rss_kb = debug_rss_kb(),
            assembled_message_count = assembled.state.read_model().messages.len(),
            assembled_graph_node_count = assembled.state.session_graph.nodes.len(),
            "runtime after assembler.finish"
        );
        if let Some(session) = self.session.as_ref() {
            let plugins = Arc::clone(session.plugins());
            let manager = finalize_manager.expect("finalize manager should exist with session");
            tracing::debug!(rss_kb = debug_rss_kb(), "runtime before finalize_turn");
            self.mark_phase_begin(RuntimeTurnPhase::FinalizeTurn);
            let finalized = plugins
                .finalize_turn(assembled, manager)
                .await
                .map_err(|err| RuntimeError {
                    code: "plugin_finalize_turn".to_string(),
                    message: err.to_string(),
                })?;
            self.mark_phase_end(RuntimeTurnPhase::FinalizeTurn);
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                finalized_message_count = finalized.turn.state.read_model().messages.len(),
                "runtime after finalize_turn"
            );
            let mut returned_turn = finalized.turn;
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                dynamic_state_present = turn_pipeline.state_mut().dynamic_state_ref.is_some()
                    || turn_pipeline.state_mut().dynamic_state_snapshot.is_some(),
                plugin_snapshot_present = turn_pipeline.state_mut().plugin_snapshot_ref.is_some()
                    || turn_pipeline.state_mut().plugin_snapshot.is_some(),
                "runtime before stamp_runtime_state"
            );
            self.mark_phase_begin(RuntimeTurnPhase::PersistTurn);
            turn_pipeline
                .final_commit(&mut returned_turn, self.session.as_mut(), &turn_usage_delta)
                .await?;
            tracing::debug!(
                rss_kb = debug_rss_kb(),
                persisted_graph_node_count = returned_turn.state.session_graph.nodes.len(),
                persisted_message_count = returned_turn.state.read_model().messages.len(),
                "runtime after stamp_runtime_state"
            );
            emit_session_events_to_sink(events, finalized.events).await;
            self.state = turn_pipeline.into_final_state();
            if let Some(session) = self.session.as_ref()
                && let Ok(host) = self.runtime_session_manager()
            {
                session
                    .plugins()
                    .emit_runtime_event(crate::PluginRuntimeEvent::TurnPersisted(
                        crate::SessionStateChangedContext {
                            session_id: self.state.session_id.clone(),
                            state: RuntimeReadSnapshot::from_exported(&returned_turn.state)
                                .read_view(),
                            host,
                        },
                    ))
                    .await;
            }
            self.mark_phase_end(RuntimeTurnPhase::PersistTurn);
            if self.host.core.trace_sink.is_some() {
                crate::trace::emit_trace(
                    &self.host.core.trace_sink,
                    &self.host.core.trace_context,
                    lash_trace::TraceContext::default()
                        .for_session(returned_turn.state.session_id.clone())
                        .for_iteration(returned_turn.state.iteration),
                    lash_trace::TraceEvent::TurnCompleted {
                        status: format!("{:?}", returned_turn.status).to_ascii_lowercase(),
                        done_reason: format!("{:?}", returned_turn.done_reason)
                            .to_ascii_lowercase(),
                        handoff: returned_turn.handoff_successor_session_id.as_ref().map(
                            |session_id| lash_trace::TraceHandoff {
                                successor_session_id: session_id.clone(),
                            },
                        ),
                    },
                );
            }
            Ok(returned_turn)
        } else {
            self.state.apply_exported_state(&assembled.state);
            if self.host.core.trace_sink.is_some() {
                crate::trace::emit_trace(
                    &self.host.core.trace_sink,
                    &self.host.core.trace_context,
                    lash_trace::TraceContext::default()
                        .for_session(assembled.state.session_id.clone())
                        .for_iteration(assembled.state.iteration),
                    lash_trace::TraceEvent::TurnCompleted {
                        status: format!("{:?}", assembled.status).to_ascii_lowercase(),
                        done_reason: format!("{:?}", assembled.done_reason).to_ascii_lowercase(),
                        handoff: assembled.handoff_successor_session_id.as_ref().map(
                            |session_id| lash_trace::TraceHandoff {
                                successor_session_id: session_id.clone(),
                            },
                        ),
                    },
                );
            }
            Ok(assembled)
        }
    }
    fn normalize_input_items(
        &self,
        items: &[InputItem],
        image_blobs: &HashMap<String, Vec<u8>>,
    ) -> Result<Vec<NormalizedItem>, String> {
        normalize_input_items(
            items,
            image_blobs,
            self.host.core.base_dir.as_path(),
            self.host.core.path_resolver.as_ref(),
            self.host.core.attachment_store.as_ref(),
        )
    }
}
