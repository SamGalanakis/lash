use super::*;

fn trace_fields_from_outcome(
    outcome: &TurnOutcome,
) -> (&'static str, &'static str, Option<lash_trace::TraceHandoff>) {
    match outcome {
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. }) => {
            ("completed", "assistant_message", None)
        }
        TurnOutcome::Finished(TurnFinish::SubmittedValue { .. }) => {
            ("completed", "submitted_value", None)
        }
        TurnOutcome::Finished(TurnFinish::ToolValue { .. }) => ("completed", "tool_value", None),
        TurnOutcome::Handoff { session_id } => (
            "completed",
            "handoff",
            Some(lash_trace::TraceHandoff {
                successor_session_id: session_id.clone(),
            }),
        ),
        TurnOutcome::Stopped(stop) => ("failed", trace_stop_reason(stop), None),
    }
}

fn trace_stop_reason(stop: &TurnStop) -> &'static str {
    match stop {
        TurnStop::Cancelled => "cancelled",
        TurnStop::Incomplete => "incomplete",
        TurnStop::InvalidInput => "invalid_input",
        TurnStop::MaxTurns => "max_turns",
        TurnStop::ToolFailure => "tool_failure",
        TurnStop::ProviderError => "provider_error",
        TurnStop::PluginAbort => "plugin_abort",
        TurnStop::RuntimeError => "runtime_error",
        TurnStop::SubmittedError { .. } => "submitted_error",
        TurnStop::ToolError { .. } => "tool_error",
    }
}

async fn abandon_runtime_turn_lease_best_effort(
    store: Option<&(dyn crate::RuntimePersistence + '_)>,
    lease: Option<&crate::RuntimeTurnLease>,
    context: &str,
) {
    let (Some(store), Some(lease)) = (store, lease) else {
        return;
    };
    if let Err(err) = store.abandon_runtime_turn_lease(lease).await {
        tracing::warn!(
            session_id = %lease.session_id,
            turn_id = %lease.turn_id,
            context,
            "failed to abandon runtime turn lease: {err}"
        );
    }
}

struct LeasedTurnFinish {
    turn_pipeline: TurnCommitPipeline,
    assembler: TurnAssembler,
    new_messages: crate::MessageSequence,
    policy: SessionPolicy,
    turn_index: usize,
    turn_lease: Option<crate::RuntimeTurnLease>,
    queued_work_completions: Vec<crate::QueuedWorkCompletion>,
    trace_turn_id: String,
    abandon_context: &'static str,
}

impl LashRuntime {
    fn max_context_tokens(&self) -> usize {
        self.policy.context_window_tokens()
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

    async fn finish_leased_turn(
        &mut self,
        finish: LeasedTurnFinish,
        events: &dyn EventSink,
        cancel_state: &CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let LeasedTurnFinish {
            mut turn_pipeline,
            assembler,
            new_messages,
            policy,
            turn_index,
            turn_lease,
            queued_work_completions,
            trace_turn_id,
            abandon_context,
        } = finish;
        let store = self
            .session
            .as_ref()
            .and_then(|session| session.history_store());
        self.policy = self.state.policy.clone();
        turn_pipeline.state_mut().policy = self.policy.clone();
        turn_pipeline.state_mut().turn_index = turn_index;

        let mut turn_usage_delta = {
            let mut ledger = self.shared_token_ledger.lock().expect("token ledger lock");
            std::mem::take(&mut *ledger)
        };
        if assembler.token_usage.total() > 0 || assembler.token_usage.cached_input_tokens > 0 {
            turn_usage_delta.push(TokenLedgerEntry {
                source: "turn".to_string(),
                model: policy.model.id.clone(),
                usage: assembler.token_usage.clone(),
            });
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
            .and_then(|usage| normalize_prompt_usage(&policy.provider, usage));
        turn_pipeline.state_mut().last_prompt_usage = last_prompt_usage;
        let assembled_state = turn_pipeline.export_state_for_assembly();
        let assembled = assembler.finish(
            assembled_state,
            cancel_state.is_cancelled(),
            None,
            &self.host.core.termination,
        );

        let Some(session) = self.session.as_ref() else {
            self.state.apply_exported_state(&assembled.state);
            self.emit_completed_turn_trace(&assembled.state, &assembled.outcome, &trace_turn_id);
            return Ok(assembled);
        };

        let plugins = Arc::clone(session.plugins());
        let manager = match self.runtime_session_manager_for_turn(None) {
            Ok(manager) => manager,
            Err(err) => {
                let runtime_err =
                    RuntimeError::new(RuntimeErrorCode::PluginSessionManager, err.to_string());
                let context = format!("{abandon_context}_finalize_manager");
                abandon_runtime_turn_lease_best_effort(
                    store.as_ref().map(|store| store.as_ref()),
                    turn_lease.as_ref(),
                    &context,
                )
                .await;
                return Err(runtime_err);
            }
        };

        self.mark_phase_begin(RuntimeTurnPhase::FinalizeTurn);
        let finalized = match plugins.finalize_turn(assembled, manager).await {
            Ok(finalized) => finalized,
            Err(err) => {
                self.mark_phase_end(RuntimeTurnPhase::FinalizeTurn);
                let runtime_err =
                    RuntimeError::new(RuntimeErrorCode::PluginFinalizeTurn, err.to_string());
                let context = format!("{abandon_context}_finalize_turn");
                abandon_runtime_turn_lease_best_effort(
                    store.as_ref().map(|store| store.as_ref()),
                    turn_lease.as_ref(),
                    &context,
                )
                .await;
                return Err(runtime_err);
            }
        };
        self.mark_phase_end(RuntimeTurnPhase::FinalizeTurn);

        let mut returned_turn = finalized.turn;
        self.mark_phase_begin(RuntimeTurnPhase::PersistTurn);
        if let Err(err) = turn_pipeline
            .final_commit(
                &mut returned_turn,
                self.session.as_mut(),
                &turn_usage_delta,
                turn_lease
                    .as_ref()
                    .map(crate::RuntimeTurnCompletion::from_lease),
                queued_work_completions,
            )
            .await
        {
            self.mark_phase_end(RuntimeTurnPhase::PersistTurn);
            let context = format!("{abandon_context}_final_commit");
            abandon_runtime_turn_lease_best_effort(
                store.as_ref().map(|store| store.as_ref()),
                turn_lease.as_ref(),
                &context,
            )
            .await;
            return Err(err);
        }

        emit_session_events_to_sink(events, finalized.events).await;
        self.state = turn_pipeline.into_final_state();
        self.emit_turn_persisted_event(&returned_turn).await;
        self.mark_phase_end(RuntimeTurnPhase::PersistTurn);

        self.emit_completed_turn_trace(
            &returned_turn.state,
            &returned_turn.outcome,
            &trace_turn_id,
        );
        Ok(returned_turn)
    }

    fn emit_completed_turn_trace(
        &self,
        state: &SessionStateEnvelope,
        outcome: &TurnOutcome,
        trace_turn_id: &str,
    ) {
        if self.host.core.trace_sink.is_none() {
            return;
        }

        let (status, done_reason, handoff) = trace_fields_from_outcome(outcome);
        crate::trace::emit_trace(
            &self.host.core.trace_sink,
            &self.host.core.trace_context,
            lash_trace::TraceContext::default()
                .for_session(state.session_id.clone())
                .for_turn_index(state.turn_index)
                .for_turn(trace_turn_id.to_string()),
            lash_trace::TraceEvent::TurnCompleted {
                status: status.to_string(),
                done_reason: done_reason.to_string(),
                handoff,
            },
        );
    }

    async fn emit_turn_persisted_event(&self, returned_turn: &AssembledTurn) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let Ok(manager) = self.runtime_session_manager() else {
            return;
        };
        let host = manager.clone();

        let direct_completions = manager.direct_completion_client(
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                &self.host.core.effect_controller,
            )),
            None,
            None,
        );

        session
            .plugins()
            .emit_runtime_event(crate::PluginLifecycleEvent::TurnPersisted(
                crate::SessionStateChangedContext {
                    session_id: self.state.session_id.clone(),
                    state: crate::SessionReadView::from_exported_state(&returned_turn.state),
                    host,
                    direct_completions,
                },
            ))
            .await;
    }

    /// Run a single turn and stream events to the host sink.
    pub async fn stream_turn(
        &mut self,
        input: TurnInput,
        opts: TurnOptions<'_>,
    ) -> Result<AssembledTurn, RuntimeError> {
        let turn_id = input
            .trace_turn_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let cancel = opts.cancel.clone();
        let effect_controller = Arc::clone(&self.host.core.effect_controller);
        let effect_scope = opts.resolve_effect_scope(effect_controller.as_ref(), &turn_id)?;
        self.stream_turn_with_effect_scope_inner(
            input,
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            effect_scope,
            cancel,
            None,
        )
        .await
    }

    pub async fn stream_next_queued_work(
        &mut self,
        opts: TurnOptions<'_>,
    ) -> Result<Option<AssembledTurn>, RuntimeError> {
        let Some(store) = self.session.as_ref().and_then(|session| session.history_store()) else {
            return Ok(None);
        };
        let claim = store
            .claim_ready_queued_work(
                &self.state.session_id,
                &self.runtime_scope_id,
                crate::QueuedWorkClaimBoundary::Idle,
                crate::QUEUED_WORK_CLAIM_TTL_MS,
                64,
            )
            .await
            .map_err(|err| RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string()))?;
        let Some(claim) = claim else {
            return Ok(None);
        };
        let mut work = claim.materialize_for_turn();
        let turn_id = work
            .input
            .trace_turn_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        work.input.trace_turn_id = Some(turn_id.clone());
        let cancel = opts.cancel.clone();
        let effect_controller = Arc::clone(&self.host.core.effect_controller);
        let effect_scope = opts.resolve_effect_scope(effect_controller.as_ref(), &turn_id)?;
        self.stream_turn_with_effect_scope_inner(
            work.input,
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            effect_scope,
            cancel,
            Some(claim),
        )
        .await
        .map(Some)
    }

    /// Resume an in-flight (durably checkpointed) turn.
    pub async fn resume_turn(
        &mut self,
        turn_id: &str,
        opts: TurnOptions<'_>,
    ) -> Result<AssembledTurn, RuntimeError> {
        let cancel = opts.cancel.clone();
        let effect_controller = Arc::clone(&self.host.core.effect_controller);
        let effect_scope = opts.resolve_effect_scope(effect_controller.as_ref(), turn_id)?;
        self.resume_turn_inner(
            turn_id,
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            effect_scope,
            cancel,
        )
        .await
    }

    async fn resume_turn_inner(
        &mut self,
        turn_id: &str,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        if turn_id != effect_scope.turn_id() {
            return Err(RuntimeError::new(
                RuntimeErrorCode::EffectScopeTurnIdMismatch,
                format!(
                    "resume turn_id `{turn_id}` does not match effect scope turn_id `{}`",
                    effect_scope.turn_id()
                ),
            ));
        }
        if effect_scope
            .controller()
            .requires_durable_attachment_store()
            && self.host.core.attachment_store.persistence()
                != crate::AttachmentStorePersistence::Durable
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::DurableAttachmentStoreRequired,
                "durable effect controllers require a durable attachment store",
            ));
        }
        self.refresh_session_graph_from_store().await;
        let store = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::RuntimeTurnResumeStoreRequired,
                    "resuming a turn requires a persistent runtime store",
                )
            })?;
        let checkpoint_record = store
            .load_runtime_turn_checkpoint(&self.state.session_id, turn_id)
            .await
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::RuntimeTurnCheckpointLoad, err.to_string())
            })?
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::RuntimeTurnCheckpointMissing,
                    format!("no in-flight runtime turn checkpoint found for `{turn_id}`"),
                )
            })?;
        if checkpoint_record.provider_id != self.policy.provider.kind() {
            return Err(RuntimeError::new(
                RuntimeErrorCode::RuntimeTurnResumeProviderMismatch,
                format!(
                    "checkpoint requires provider `{}`, current runtime has `{}`",
                    checkpoint_record.provider_id,
                    self.policy.provider.kind()
                ),
            ));
        }
        let turn_lease = store
            .claim_runtime_turn_lease(
                &self.state.session_id,
                turn_id,
                &self.runtime_scope_id,
                RUNTIME_TURN_LEASE_TTL_MS,
            )
            .await
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::RuntimeTurnLease, err.to_string())
            })?;
        let (event_tx, mut event_rx) = mpsc::channel::<RuntimeStreamEvent>(100);
        let child_usage_event_relay = ChildUsageEventRelay::new(event_tx.clone());
        let manager = self
            .runtime_session_manager_for_turn_with_lease(
                Some(child_usage_event_relay.clone()),
                Some(turn_lease.clone()),
            )
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::PluginSessionManager, err.to_string())
            })?;
        let mut turn_policy = self.policy.clone();
        turn_policy.model = checkpoint_record.model.clone();
        turn_policy.max_turns = checkpoint_record.machine_config.max_turns;
        self.protocol_turn_options = checkpoint_record.protocol_turn_options.clone();
        let resume_turn_index = checkpoint_record.turn_index;
        let resume_protocol_iteration = checkpoint_record.protocol_iteration;

        let cancel_state = cancel.clone();
        let session = self
            .session
            .take()
            .expect("lash runtime session must be available");
        let mut driver = RuntimeTurnDriver {
            session,
            policy: turn_policy.clone(),
            host: self.host.clone(),
            effect_scope,
            session_id: self.state.session_id.clone(),
            turn_id: turn_id.to_string(),
            turn_index: checkpoint_record.turn_index,
            turn_pipeline: TurnCommitPipeline::from_state(self.state.clone()),
            llm_stream_summaries: HashMap::new(),
            next_llm_ordinal: 0,
            session_manager: manager,
            protocol_turn_options: checkpoint_record.protocol_turn_options.clone(),
            protocol_extension: None,
            turn_context: {
                let mut context = crate::TurnContext::default();
                context.set_prompt_layer(checkpoint_record.turn_prompt_layer.clone());
                context
            },
            turn_causes: Vec::new(),
            pending_queue_claims: Vec::new(),
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            turn_lease: Some(turn_lease.clone()),
            machine_config_snapshot: Some(checkpoint_record.machine_config.clone()),
            turn_phase_probe: self.turn_phase_probe.clone(),
        };
        let restored_machine = match driver.restore_turn_machine(checkpoint_record) {
            Ok(machine) => machine,
            Err(err) => {
                let RuntimeTurnDriver { session, .. } = driver;
                self.session = Some(session);
                abandon_runtime_turn_lease_best_effort(
                    Some(store.as_ref()),
                    Some(&turn_lease),
                    "restore_turn_machine",
                )
                .await;
                return Err(err);
            }
        };
        let mut assembler = TurnAssembler::new();
        self.mark_phase_begin(RuntimeTurnPhase::EffectLoop);
        let run_result = {
            let run_future = driver.run_restored(
                restored_machine,
                event_tx,
                cancel,
                resume_protocol_iteration,
            );
            tokio::pin!(run_future);
            loop {
                tokio::select! {
                    maybe_event = event_rx.recv() => {
                        if let Some(event) = maybe_event {
                            emit_runtime_stream_event_to_sinks(
                                events,
                                turn_events,
                                event,
                                &mut assembler,
                            )
                            .await;
                        }
                    }
                    completed = &mut run_future => {
                        child_usage_event_relay.clear();
                        break completed;
                    }
                }
            }
        };
        let (new_messages, _new_protocol_iteration) = match run_result {
            Ok(result) => result,
            Err(err) => {
                self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
                let RuntimeTurnDriver {
                    session,
                    turn_lease: current_turn_lease,
                    ..
                } = driver;
                self.session = Some(session);
                abandon_runtime_turn_lease_best_effort(
                    Some(store.as_ref()),
                    current_turn_lease.as_ref().or(Some(&turn_lease)),
                    "resume_effect_loop",
                )
                .await;
                return Err(err);
            }
        };
        while let Some(event) = event_rx.recv().await {
            emit_runtime_stream_event_to_sinks(events, turn_events, event, &mut assembler).await;
        }
        self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
        let RuntimeTurnDriver {
            session,
            policy,
            turn_pipeline,
            turn_lease: current_turn_lease,
            pending_queue_claims,
            ..
        } = driver;
        let turn_lease = current_turn_lease.unwrap_or(turn_lease);
        self.session = Some(session);
        self.finish_leased_turn(
            LeasedTurnFinish {
                turn_pipeline,
                assembler,
                new_messages,
                policy,
                turn_index: resume_turn_index,
                turn_lease: Some(turn_lease),
                queued_work_completions: pending_queue_claims
                    .iter()
                    .map(crate::QueuedWorkClaim::completion)
                    .collect(),
                trace_turn_id: turn_id.to_string(),
                abandon_context: "resume",
            },
            events,
            &cancel_state,
        )
        .await
    }

    async fn stream_turn_with_effect_scope_inner(
        &mut self,
        mut input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
        cancel: CancellationToken,
        queued_claim: Option<crate::QueuedWorkClaim>,
    ) -> Result<AssembledTurn, RuntimeError> {
        if let Some(input_turn_id) = input.trace_turn_id.as_deref()
            && input_turn_id != effect_scope.turn_id()
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::EffectScopeTurnIdMismatch,
                format!(
                    "input trace_turn_id `{input_turn_id}` does not match effect scope turn_id `{}`",
                    effect_scope.turn_id()
                ),
            ));
        }
        if effect_scope
            .controller()
            .requires_durable_attachment_store()
            && self.host.core.attachment_store.persistence()
                != crate::AttachmentStorePersistence::Durable
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::DurableAttachmentStoreRequired,
                "durable effect controllers require a durable attachment store",
            ));
        }
        input.trace_turn_id = Some(effect_scope.turn_id().to_string());
        if let Some(execution_session_id) = self
            .active_handoff_leaf(&self.state.session_id)
            .await
            .filter(|session_id| session_id != &self.state.session_id)
        {
            return self
                .stream_turn_on_handoff_successor(
                    execution_session_id,
                    input,
                    events,
                    turn_events,
                    effect_scope,
                    cancel,
                )
                .await;
        }
        self.stream_turn_inner(
            input.clone(),
            events,
            turn_events,
            effect_scope,
            cancel.clone(),
            queued_claim,
        )
        .await
    }

    async fn active_handoff_leaf(&self, session_id: &str) -> Option<String> {
        let continuations = self.active_handoff_continuations.lock().await;
        let mut current = session_id.to_string();
        let mut seen = std::collections::HashSet::new();
        while seen.insert(current.clone()) {
            let Some(next) = continuations.get(&current).cloned() else {
                return (current != session_id).then_some(current);
            };
            current = next;
        }
        None
    }

    async fn stream_turn_on_handoff_successor(
        &mut self,
        execution_session_id: String,
        input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let runtime_handle = {
            let registry = self.managed_sessions.lock().await;
            registry.get(&execution_session_id).cloned()
        }
        .ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::HandoffSuccessorMissing,
                format!("active handoff session `{execution_session_id}` is unavailable"),
            )
        })?;
        let mut runtime = runtime_handle.runtime.lock().await;
        runtime.state.turn_index = self.state.turn_index;
        let turn = runtime
            .stream_turn_inner(input, events, turn_events, effect_scope, cancel, None)
            .await?;
        runtime_handle.publish_from(&runtime);
        self.state.turn_index = turn.state.turn_index;
        Ok(turn)
    }

    /// Stream one logical host turn, following foreground handoffs until a
    /// non-handoff outcome is reached.
    ///
    /// RLM `continue_as` creates a successor session with queued first-turn
    /// input. Hosts that only care about the benchmark/app answer should not
    /// need to special-case that intermediate outcome; this helper activates
    /// each successor and drives its queued first turn with the normal runtime
    /// turn guards.
    pub async fn stream_turn_following_handoffs(
        &mut self,
        input: TurnInput,
        opts: TurnOptions<'_>,
    ) -> Result<FollowedTurn, RuntimeError> {
        let follow_trace_turn_id = input
            .trace_turn_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let cancel = opts.cancel.clone();
        let effect_controller = Arc::clone(&self.host.core.effect_controller);
        let effect_scope =
            opts.resolve_effect_scope(effect_controller.as_ref(), &follow_trace_turn_id)?;
        self.stream_turn_following_handoffs_inner(
            input,
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            effect_scope,
            cancel,
        )
        .await
    }

    async fn stream_turn_following_handoffs_inner(
        &mut self,
        mut input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
        cancel: CancellationToken,
    ) -> Result<FollowedTurn, RuntimeError> {
        if let Some(input_turn_id) = input.trace_turn_id.as_deref()
            && input_turn_id != effect_scope.turn_id()
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::EffectScopeTurnIdMismatch,
                format!(
                    "input trace_turn_id `{input_turn_id}` does not match effect scope turn_id `{}`",
                    effect_scope.turn_id()
                ),
            ));
        }
        let follow_protocol_turn_options = input.protocol_turn_options.clone();
        let follow_turn_context = input.turn_context.clone();
        let follow_trace_turn_id = effect_scope.turn_id().to_string();
        input.trace_turn_id = Some(follow_trace_turn_id.clone());
        let mut turns = Vec::new();
        loop {
            let turn = self
                .stream_turn_with_effect_scope_inner(
                    input,
                    events,
                    turn_events,
                    effect_scope,
                    cancel.clone(),
                    None,
                )
                .await?;
            let successor_session_id = match &turn.outcome {
                TurnOutcome::Handoff { session_id } => Some(session_id.clone()),
                _ => None,
            };
            turns.push(turn);

            let Some(successor_session_id) = successor_session_id else {
                return Ok(FollowedTurn { turns });
            };

            let seed = self
                .pending_first_turn_inputs
                .lock()
                .expect("pending first turn inputs lock")
                .remove(&successor_session_id)
                .ok_or_else(|| {
                    RuntimeError::new(
                        RuntimeErrorCode::HandoffMissingFirstTurn,
                        format!(
                            "handoff session `{successor_session_id}` did not provide a first turn"
                        ),
                    )
                })?;
            input = turn_input_from_plugin_message(seed);
            input.protocol_turn_options = follow_protocol_turn_options.clone();
            input.trace_turn_id = Some(follow_trace_turn_id.clone());
            input.turn_context = follow_turn_context.clone();
            if let Some(successor_handle) = {
                let registry = self.managed_sessions.lock().await;
                registry.get(&successor_session_id).cloned()
            } {
                let mut successor = successor_handle.runtime.lock().await;
                successor.state.turn_index = self.state.turn_index.saturating_sub(1);
                // Keep observers aligned if a handoff successor has not
                // started its own streamed turn yet.
                successor_handle.publish_from(&successor);
            }
        }
    }

    async fn stream_turn_inner(
        &mut self,
        mut input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
        cancel: CancellationToken,
        queued_claim: Option<crate::QueuedWorkClaim>,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.refresh_session_graph_from_store().await;
        let input_trace_turn_id = input.trace_turn_id.clone();
        let queued_turn_work = queued_claim
            .as_ref()
            .map(crate::QueuedWorkClaim::materialize_for_turn);
        if let Some(work) = queued_turn_work.as_ref()
            && input.items.is_empty()
            && input.image_blobs.is_empty()
        {
            input = work.input.clone();
            if input.trace_turn_id.is_none() {
                input.trace_turn_id = input_trace_turn_id;
            }
        }
        if self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
            .is_some()
        {
            ensure_durable_turn_input(&input)?;
        }
        if let Some(extension) = &input.protocol_extension
            && let Some(session) = self.session.as_ref()
        {
            let protocol_session = std::sync::Arc::clone(session.plugins().protocol_session());
            protocol_session
                .validate_turn_extension(extension)
                .await
                .map_err(|err| {
                    RuntimeError::new(RuntimeErrorCode::ProtocolTurnExtension, err.to_string())
                })?;
        }
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
                        terminal_reason: None,
                        user_message: e.clone(),
                        raw: None,
                    }),
                };
                assembler.push(&error_event);
                emit_turn_activity_to_sink(
                    turn_events,
                    TurnActivity::independent(TurnEvent::Error { message: e }),
                )
                .await;
                emit_session_event_to_sink(events, error_event).await;
                let outcome_event = SessionEvent::TurnOutcome {
                    outcome: TurnOutcome::Stopped(TurnStop::InvalidInput),
                };
                assembler.push(&outcome_event);
                emit_session_event_to_sink(events, outcome_event).await;
                assembler.push(&SessionEvent::Done);
                emit_session_event_to_sink(events, SessionEvent::Done).await;
                return Ok(assembler.finish(
                    self.state.export_state(),
                    false,
                    None,
                    &self.host.core.termination,
                ));
            }
        };
        let turn_index = self.state.turn_index + 1;
        let trace_turn_id = input
            .trace_turn_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if self.host.core.trace_sink.is_some() {
            let mut trace_metadata = std::collections::BTreeMap::new();
            trace_metadata.insert(
                "input_item_count".to_string(),
                serde_json::json!(normalized.len()),
            );
            crate::trace::emit_trace(
                &self.host.core.trace_sink,
                &self.host.core.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(self.state.session_id.clone())
                    .for_turn_index(turn_index)
                    .for_turn(trace_turn_id.clone()),
                lash_trace::TraceEvent::TurnStarted {
                    metadata: trace_metadata,
                },
            );
        }

        let base_read_model = self.state.read_model();
        let base_messages = base_read_model.messages;
        let base_render_cache = base_read_model.prompt_render_cache;
        let mut turn_delta = Vec::new();
        let initial_turn_causes = queued_turn_work
            .as_ref()
            .map(|work| work.turn_causes.clone())
            .unwrap_or_default();
        turn_delta.extend(
            initial_turn_causes
                .iter()
                .map(crate::TurnCause::to_event_message),
        );

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
                        tool_replay: None,
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
                        tool_replay: None,
                        prune_state: PruneState::Intact,
                        reasoning_meta: None,
                        response_meta: None,
                    });
                }
            }
        }
        if user_parts.is_empty() && initial_turn_causes.is_empty() {
            user_parts.push(Part {
                id: format!("{}.p0", user_id),
                kind: PartKind::Text,
                content: String::new(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            });
        }
        if !user_parts.is_empty() {
            reassign_part_ids(&user_id, &mut user_parts);
            turn_delta.push(Message {
                id: user_id.clone(),
                role: MessageRole::User,
                parts: shared_parts(user_parts),
                origin: None,
            });
        }

        let manager = self.runtime_session_manager_for_turn(None).map_err(|err| {
            RuntimeError::new(RuntimeErrorCode::PluginSessionManager, err.to_string())
        })?;
        let plugin_session = self
            .session
            .as_ref()
            .map(|s| Arc::clone(s.plugins()))
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::ContextPrepareTurn,
                    "runtime session not available",
                )
            })?;
        let turn_ctx = crate::TurnTransformContext {
            session_id: self.state.session_id.clone(),
            state: self.read_view(),
            prompt_usage: previous_prompt_usage.clone(),
            max_context_tokens: Some(LashRuntime::max_context_tokens(self)),
            host: manager.clone() as Arc<dyn crate::plugin::RuntimeSessionHost>,
            direct_completions: manager.direct_completion_client(
                crate::runtime::RuntimeEffectControllerHandle::shared(Arc::clone(
                    &self.host.core.effect_controller,
                )),
                None,
                None,
            ),
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
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::ContextPrepareTurn, err.to_string())
            })?;
        self.mark_phase_end(RuntimeTurnPhase::ContextTransform);
        // Release the read-view's graph clone before the rest of the turn
        // runs. Keeping it alive into `stream_prepared_turn` forces the
        // post-turn `append_active_read_delta` to deep-clone the session
        // graph (Arc::make_mut with refcount > 1).
        drop(turn_ctx);
        let messages = prepared_context.messages;
        if let Some(session) = self.session.as_mut() {
            session
                .set_context_surface(
                    prepared_context.tool_providers,
                    prepared_context.prompt_contributions,
                    prepared_context.include_base_tools,
                )
                .map_err(|err| {
                    RuntimeError::new(
                        RuntimeErrorCode::Other("session_tool_registry".to_string()),
                        err.to_string(),
                    )
                })?;
        }

        self.state.last_prompt_usage = None;

        self.stream_prepared_turn(
            messages,
            previous_prompt_usage,
            input.protocol_turn_options.clone(),
            input.protocol_extension.clone(),
            input.turn_context.clone(),
            initial_turn_causes,
            trace_turn_id,
            turn_index,
            events,
            turn_events,
            effect_scope,
            cancel,
            queued_claim,
        )
        .await
    }

    /// Run a single turn and return only the assembled terminal result.
    pub async fn run_turn_assembled(
        &mut self,
        input: TurnInput,
        cancel: CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.stream_turn(input, TurnOptions::new(cancel)).await
    }

    /// Run a turn using host-prepared message history.
    #[allow(clippy::too_many_arguments)]
    pub async fn stream_prepared_turn(
        &mut self,
        messages: crate::MessageSequence,
        _previous_prompt_usage: Option<PromptUsage>,
        protocol_turn_options: Option<crate::ProtocolTurnOptions>,
        protocol_extension: Option<crate::ProtocolTurnExtensionHandle>,
        turn_context: crate::TurnContext,
        initial_turn_causes: Vec<crate::TurnCause>,
        trace_turn_id: String,
        turn_index: usize,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
        cancel: CancellationToken,
        initial_queue_claim: Option<crate::QueuedWorkClaim>,
    ) -> Result<AssembledTurn, RuntimeError> {
        let (event_tx, mut event_rx) = mpsc::channel::<RuntimeStreamEvent>(100);
        let child_usage_event_relay = ChildUsageEventRelay::new(event_tx.clone());
        let mut turn_policy = self.policy.clone();
        if let Some(provider) = turn_context.provider().cloned() {
            turn_policy.provider = provider;
        }
        if let Some(model) = turn_context.model_spec() {
            turn_policy.model = model.clone();
        }
        let effective_protocol_turn_options = protocol_turn_options
            .clone()
            .unwrap_or_else(|| self.protocol_turn_options.clone());
        let manager = self
            .runtime_session_manager_for_turn(Some(child_usage_event_relay.clone()))
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::PluginSessionManager, err.to_string())
            })?;
        let plugins = {
            let session = self
                .session
                .as_ref()
                .expect("lash runtime session must be available");
            Arc::clone(session.plugins())
        };
        let mut assembler = TurnAssembler::new();
        self.mark_phase_begin(RuntimeTurnPhase::BeforeTurnHooks);
        // Block-scope the pinned future so it (and its captured
        // `SessionReadView` clone of the session graph) drops before the
        // post-turn `append_active_read_delta` mutation. Keeping it alive
        // across the turn forces `Arc::make_mut` to deep-clone
        // `SessionGraphData`.
        let prepared = {
            let prepare_turn = plugins.prepare_turn(PrepareTurnRequest {
                session_id: self.state.session_id.clone(),
                state: crate::SessionReadView::from_runtime_state(
                    &self.state,
                    turn_policy.clone(),
                    effective_protocol_turn_options.clone(),
                ),
                messages,
                host: manager.clone(),
                turn_context: turn_context.clone(),
            });
            tokio::pin!(prepare_turn);

            loop {
                tokio::select! {
                    prepared = &mut prepare_turn => {
                        let prepared = prepared.map_err(|err| {
                            RuntimeError::new(RuntimeErrorCode::PluginPrepareTurn, err.to_string())
                        })?;
                        self.mark_phase_end(RuntimeTurnPhase::BeforeTurnHooks);
                        break prepared;
                    }
                    maybe_event = event_rx.recv() => {
                        if let Some(event) = maybe_event {
                            emit_runtime_stream_event_to_sinks(
                                events,
                                turn_events,
                                event,
                                &mut assembler,
                            )
                            .await;
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
            drop(event_tx);

            let mut turn_pipeline = TurnCommitPipeline::from_state(self.state.clone());
            turn_pipeline.apply_prepared_messages(&prepared.messages);
            let state = turn_pipeline.into_final_state();
            let issue = TurnIssue {
                kind: "plugin".to_string(),
                code: Some(abort.code),
                terminal_reason: None,
                message: abort.message.clone(),
                raw: None,
            };
            let error_event = SessionEvent::Error {
                message: abort.message,
                envelope: Some(crate::session_model::ErrorEnvelope {
                    kind: "plugin".to_string(),
                    code: issue.code.clone(),
                    terminal_reason: None,
                    user_message: issue.message.clone(),
                    raw: None,
                }),
            };
            assembler.push(&error_event);
            emit_turn_activity_to_sink(
                turn_events,
                TurnActivity::independent(TurnEvent::Error {
                    message: issue.message.clone(),
                }),
            )
            .await;
            emit_session_event_to_sink(events, error_event).await;
            let outcome_event = SessionEvent::TurnOutcome {
                outcome: TurnOutcome::Stopped(TurnStop::PluginAbort),
            };
            assembler.push(&outcome_event);
            emit_session_event_to_sink(events, outcome_event).await;
            assembler.push(&SessionEvent::Done);
            emit_session_event_to_sink(events, SessionEvent::Done).await;
            return Ok(assembler.finish(
                state.export_state(),
                cancel.is_cancelled(),
                Some(issue),
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
                turn_policy.clone(),
                turn_index,
                &prepared.messages,
                self.session.as_mut(),
            )
            .await
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string())
            })?;
        let turn_lease = if let Some(store) = store.as_ref() {
            Some(
                store
                    .claim_runtime_turn_lease(
                        &self.state.session_id,
                        &trace_turn_id,
                        &self.runtime_scope_id,
                        RUNTIME_TURN_LEASE_TTL_MS,
                    )
                    .await
                    .map_err(|err| {
                        RuntimeError::new(RuntimeErrorCode::RuntimeTurnLease, err.to_string())
                    })?,
            )
        } else {
            None
        };
        let manager = self
            .runtime_session_manager_for_turn_with_lease(
                Some(child_usage_event_relay.clone()),
                turn_lease.clone(),
            )
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::PluginSessionManager, err.to_string())
            })?;
        let cancel_state = cancel.clone();
        let session = self
            .session
            .take()
            .expect("lash runtime session must be available");
        let mut driver = RuntimeTurnDriver {
            session,
            policy: turn_policy.clone(),
            host: self.host.clone(),
            effect_scope,
            session_id: self.state.session_id.clone(),
            turn_id: effect_scope.turn_id().to_string(),
            turn_index,
            turn_pipeline,
            llm_stream_summaries: HashMap::new(),
            next_llm_ordinal: 0,
            session_manager: manager,
            protocol_turn_options: effective_protocol_turn_options,
            protocol_extension,
            turn_context,
            turn_causes: initial_turn_causes,
            pending_queue_claims: initial_queue_claim.into_iter().collect(),
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            turn_lease: turn_lease.clone(),
            machine_config_snapshot: None,
            turn_phase_probe: self.turn_phase_probe.clone(),
        };
        let protocol_run_offset = 0;
        self.mark_phase_begin(RuntimeTurnPhase::EffectLoop);
        let run_result = {
            let run_future = driver.run(prepared.messages, event_tx, cancel, protocol_run_offset);
            tokio::pin!(run_future);
            loop {
                tokio::select! {
                    maybe_event = event_rx.recv() => {
                        if let Some(event) = maybe_event {
                            emit_runtime_stream_event_to_sinks(
                                events,
                                turn_events,
                                event,
                                &mut assembler,
                            )
                            .await;
                        }
                    }
                    completed = &mut run_future => {
                        child_usage_event_relay.clear();
                        break completed;
                    }
                }
            }
        };
        let (new_messages, _new_protocol_iteration) = match run_result {
            Ok(result) => result,
            Err(err) => {
                self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
                let RuntimeTurnDriver {
                    session,
                    turn_lease: current_turn_lease,
                    ..
                } = driver;
                self.session = Some(session);
                abandon_runtime_turn_lease_best_effort(
                    store.as_ref().map(|store| store.as_ref()),
                    current_turn_lease.as_ref().or(turn_lease.as_ref()),
                    "effect_loop",
                )
                .await;
                return Err(err);
            }
        };
        while let Some(event) = event_rx.recv().await {
            emit_runtime_stream_event_to_sinks(events, turn_events, event, &mut assembler).await;
        }
        self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
        tracing::debug!(
            new_message_count = new_messages.len(),
            tool_call_count = assembler.tool_calls.len(),
            "runtime post-run_task"
        );

        let RuntimeTurnDriver {
            session,
            policy,
            turn_pipeline,
            turn_lease: current_turn_lease,
            pending_queue_claims,
            ..
        } = driver;
        let turn_lease = current_turn_lease.or(turn_lease);
        self.session = Some(session);
        self.finish_leased_turn(
            LeasedTurnFinish {
                turn_pipeline,
                assembler,
                new_messages,
                policy,
                turn_index,
                turn_lease,
                queued_work_completions: pending_queue_claims
                    .iter()
                    .map(crate::QueuedWorkClaim::completion)
                    .collect(),
                trace_turn_id,
                abandon_context: "fresh",
            },
            events,
            &cancel_state,
        )
        .await
    }
    fn normalize_input_items(
        &self,
        items: &[InputItem],
        image_blobs: &HashMap<String, Vec<u8>>,
    ) -> Result<Vec<NormalizedItem>, String> {
        normalize_input_items(items, image_blobs, self.host.core.attachment_store.as_ref())
    }
}

fn turn_input_from_plugin_message(message: PluginMessage) -> TurnInput {
    let mut items = Vec::new();
    if !message.content.is_empty() {
        items.push(InputItem::Text {
            text: message.content,
        });
    }
    let mut image_blobs = HashMap::new();
    for (index, bytes) in message.images.into_iter().enumerate() {
        let id = format!("handoff-seed-image-{index}");
        image_blobs.insert(id.clone(), bytes);
        items.push(InputItem::ImageRef { id });
    }
    TurnInput {
        items,
        image_blobs,
        protocol_turn_options: None,
        trace_turn_id: None,
        protocol_extension: None,
        turn_context: crate::TurnContext::default(),
    }
}

pub fn ensure_durable_turn_input(input: &TurnInput) -> Result<(), RuntimeError> {
    if input.protocol_extension.is_some() {
        return Err(RuntimeError::new(
            RuntimeErrorCode::DurableTurnLiveProtocolExtension,
            "durable turn resume does not support live protocol_extension inputs; encode resumable data in protocol_turn_options or persisted plugin state",
        ));
    }
    if input.turn_context.has_plugin_inputs() {
        return Err(RuntimeError::new(
            RuntimeErrorCode::DurableTurnLivePluginInput,
            "durable turn resume does not support live TurnContext plugin inputs; encode resumable data in protocol_turn_options or persisted plugin state",
        ));
    }
    Ok(())
}

async fn emit_turn_activity_to_sink(events: &dyn TurnActivitySink, activity: TurnActivity) {
    if !events.is_noop() {
        events.emit(activity).await;
    }
}

async fn emit_runtime_stream_event_to_sinks(
    events: &dyn EventSink,
    turn_events: &dyn TurnActivitySink,
    event: RuntimeStreamEvent,
    assembler: &mut TurnAssembler,
) {
    match event {
        RuntimeStreamEvent::Session(event) => {
            assembler.push(&event);
            emit_session_event_to_sink(events, event).await;
        }
        RuntimeStreamEvent::Turn(activity) => {
            assembler.push_turn_activity(&activity);
            emit_turn_activity_to_sink(turn_events, activity).await;
        }
    }
}
