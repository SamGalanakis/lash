use super::*;

fn trace_fields_from_outcome(
    outcome: &TurnOutcome,
) -> (
    &'static str,
    &'static str,
    Option<lash_trace::TraceAgentFrameSwitch>,
) {
    match outcome {
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. }) => {
            ("completed", "assistant_message", None)
        }
        TurnOutcome::Finished(TurnFinish::SubmittedValue { .. }) => {
            ("completed", "submitted_value", None)
        }
        TurnOutcome::Finished(TurnFinish::ToolValue { .. }) => ("completed", "tool_value", None),
        TurnOutcome::AgentFrameSwitch { frame_id, .. } => (
            "completed",
            "agent_frame_switch",
            Some(lash_trace::TraceAgentFrameSwitch {
                frame_id: frame_id.clone(),
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

fn session_head_refresh_error(err: SessionError) -> RuntimeError {
    RuntimeError::new(
        RuntimeErrorCode::Other("session_head_refresh".to_string()),
        err.to_string(),
    )
}

fn queued_work_payload_type(payload: &crate::QueuedWorkPayload) -> &'static str {
    match payload {
        crate::QueuedWorkPayload::TurnInput { .. } => "turn_input",
        crate::QueuedWorkPayload::ProcessWake { .. } => "process_wake",
        crate::QueuedWorkPayload::SessionCommand { command } => command.kind(),
    }
}

fn queued_work_batch_ids(claim: &crate::QueuedWorkClaim) -> Vec<String> {
    claim
        .batches
        .iter()
        .map(|batch| batch.batch_id.clone())
        .collect()
}

fn turn_phase_id(parent_turn_id: &str, phase: &str) -> String {
    format!("{parent_turn_id}:{phase}")
}

fn scoped_child_turn_controller<'run>(
    scoped_effect_controller: &'run ScopedEffectController<'_>,
    session_id: &str,
    turn_id: &str,
) -> Result<ScopedEffectController<'run>, RuntimeError> {
    ScopedEffectController::borrowed(
        scoped_effect_controller.controller(),
        ExecutionScope::turn(session_id, turn_id),
    )
}

pub(in crate::runtime) fn queued_work_trace_payload(
    boundary: crate::QueuedWorkClaimBoundary,
    claim: &crate::QueuedWorkClaim,
    causes: &[crate::TurnCause],
) -> serde_json::Value {
    serde_json::json!({
        "boundary": boundary,
        "claim_id": claim.claim_id,
        "owner_id": claim.owner_id,
        "batch_ids": queued_work_batch_ids(claim),
        "payload_types": claim.batches.iter()
            .flat_map(|batch| batch.items.iter())
            .map(|item| queued_work_payload_type(&item.payload))
            .collect::<Vec<_>>(),
        "causes": causes,
    })
}

pub(in crate::runtime) fn queued_work_completion_trace_payload(
    completions: &[crate::QueuedWorkCompletion],
) -> serde_json::Value {
    serde_json::json!({
        "claims": completions.iter().map(|completion| {
            serde_json::json!({
                "session_id": completion.session_id,
                "claim_id": completion.claim_id,
                "batch_ids": completion.batch_ids,
            })
        }).collect::<Vec<_>>(),
    })
}

async fn emit_queued_work_started_to_sink(
    events: &dyn TurnActivitySink,
    boundary: crate::QueuedWorkClaimBoundary,
    claim: &crate::QueuedWorkClaim,
    causes: Vec<crate::TurnCause>,
) {
    emit_turn_activity_to_sink(
        events,
        TurnActivity::independent(TurnEvent::QueuedWorkStarted {
            boundary,
            batch_ids: queued_work_batch_ids(claim),
            causes,
        }),
    )
    .await;
}

pub(in crate::runtime) async fn send_queued_work_started_event(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    boundary: crate::QueuedWorkClaimBoundary,
    claim: &crate::QueuedWorkClaim,
    causes: Vec<crate::TurnCause>,
) {
    send_turn_activity(
        event_tx,
        TurnActivityId::fresh(),
        TurnEvent::QueuedWorkStarted {
            boundary,
            batch_ids: queued_work_batch_ids(claim),
            causes,
        },
    )
    .await;
}

struct TurnFinishInput {
    turn_pipeline: TurnBoundary,
    assembler: TurnAssembler,
    new_messages: crate::MessageSequence,
    policy: RuntimeSessionPolicy,
    turn_index: usize,
    queued_work_completions: Vec<crate::QueuedWorkCompletion>,
    trace_turn_id: String,
}

impl LashRuntime {
    fn max_context_tokens(&self) -> usize {
        self.state.effective_policy().context_window_tokens()
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

    async fn finish_turn(
        &mut self,
        finish: TurnFinishInput,
        events: &dyn EventSink,
        scoped_effect_controller: &ScopedEffectController<'_>,
        cancel_state: &CancellationToken,
    ) -> Result<AssembledTurn, RuntimeError> {
        let TurnFinishInput {
            mut turn_pipeline,
            assembler,
            new_messages,
            policy,
            turn_index,
            queued_work_completions,
            trace_turn_id,
        } = finish;
        self.policy = self.state.effective_policy().clone();
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

        turn_pipeline.finalize_turn_read_state(new_messages, cancel_state.is_cancelled());
        if assembler.token_usage.total() > 0 || assembler.token_usage.cached_input_tokens > 0 {
            turn_pipeline.state_mut().token_usage = assembler.token_usage.clone();
        }

        let last_prompt_usage = assembler
            .last_llm_usage()
            .and_then(|usage| normalize_prompt_usage(policy.provider(), usage));
        turn_pipeline.state_mut().last_prompt_usage = last_prompt_usage;
        let assembled_state = turn_pipeline.export_state_for_assembly();
        let assembled = assembler.finish(
            assembled_state,
            cancel_state.is_cancelled(),
            None,
            &self.host.core.control.termination,
        );

        let Some(session) = self.session.as_ref() else {
            self.state.apply_snapshot(&assembled.state);
            self.emit_completed_turn_trace(&assembled.state, &assembled.outcome, &trace_turn_id);
            return Ok(assembled);
        };

        let plugins = Arc::clone(session.plugins());
        let manager = match self.runtime_session_services_for_turn(None) {
            Ok(manager) => manager,
            Err(err) => {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::PluginSessionManager,
                    err.to_string(),
                ));
            }
        };

        self.mark_phase_begin(RuntimeTurnPhase::FinalizeTurn);
        let finalized = match plugins
            .finalize_turn_with_phase_probe(
                assembled,
                manager.state_service(),
                manager.lifecycle_service(),
                manager.graph_service(),
                self.turn_phase_probe.clone(),
            )
            .await
        {
            Ok(finalized) => finalized,
            Err(err) => {
                self.mark_phase_end(RuntimeTurnPhase::FinalizeTurn);
                return Err(RuntimeError::new(
                    RuntimeErrorCode::PluginFinalizeTurn,
                    err.to_string(),
                ));
            }
        };
        self.mark_phase_end(RuntimeTurnPhase::FinalizeTurn);

        let mut returned_turn = finalized.turn;
        self.mark_phase_begin(RuntimeTurnPhase::PersistTurn);
        self.mark_phase_begin(RuntimeTurnPhase::FinalCommit);
        let queued_work_completion_trace = queued_work_completions.clone();
        let pending_attachment_ids = self
            .host
            .core
            .durability
            .attachment_store
            .pending_manifest_commit_ids();
        if let Err(err) = turn_pipeline
            .final_commit(
                &mut returned_turn,
                self.session.as_mut(),
                &turn_usage_delta,
                Some(&trace_turn_id),
                queued_work_completions,
                pending_attachment_ids.clone(),
            )
            .await
        {
            self.mark_phase_end(RuntimeTurnPhase::FinalCommit);
            self.mark_phase_end(RuntimeTurnPhase::PersistTurn);
            return Err(err);
        }
        self.host
            .core
            .durability
            .attachment_store
            .mark_manifest_committed(&pending_attachment_ids);
        self.mark_phase_end(RuntimeTurnPhase::FinalCommit);

        emit_session_events_to_sink(events, finalized.events).await;
        self.state = turn_pipeline.into_final_state();
        if matches!(returned_turn.outcome, TurnOutcome::AgentFrameSwitch { .. })
            && let Some(session) = self.session.as_mut()
        {
            let protocol_session = Arc::clone(session.plugins().protocol_session());
            let session_id = self.state.session_id.clone();
            protocol_session
                .restore_session(
                    crate::plugin::ProtocolSessionContext::new(session, &session_id),
                    &self.state,
                )
                .await
                .map_err(|err| {
                    RuntimeError::new(
                        RuntimeErrorCode::Other("protocol_restore_session".to_string()),
                        err.to_string(),
                    )
                })?;
        }
        if !queued_work_completion_trace.is_empty() {
            crate::trace::emit_trace(
                &self.host.core.tracing.trace_sink,
                &self.host.core.tracing.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(returned_turn.state.session_id.clone())
                    .for_turn_index(returned_turn.state.turn_index)
                    .for_turn(trace_turn_id.clone()),
                lash_trace::TraceEvent::Custom {
                    name: "queued_work.completed".to_string(),
                    payload: queued_work_completion_trace_payload(&queued_work_completion_trace),
                },
            );
        }
        self.mark_phase_begin(RuntimeTurnPhase::PostPersistHooks);
        self.emit_turn_persisted_event(&returned_turn, scoped_effect_controller, &trace_turn_id)
            .await?;
        self.mark_phase_end(RuntimeTurnPhase::PostPersistHooks);
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
        state: &SessionSnapshot,
        outcome: &TurnOutcome,
        trace_turn_id: &str,
    ) {
        if self.host.core.tracing.trace_sink.is_none() {
            return;
        }

        let (status, done_reason, agent_frame_switch) = trace_fields_from_outcome(outcome);
        crate::trace::emit_trace(
            &self.host.core.tracing.trace_sink,
            &self.host.core.tracing.trace_context,
            lash_trace::TraceContext::default()
                .for_session(state.session_id.clone())
                .for_turn_index(state.turn_index)
                .for_turn(trace_turn_id.to_string()),
            lash_trace::TraceEvent::TurnCompleted {
                status: status.to_string(),
                done_reason: done_reason.to_string(),
                agent_frame_switch,
            },
        );
    }

    async fn emit_turn_persisted_event(
        &self,
        returned_turn: &AssembledTurn,
        scoped_effect_controller: &ScopedEffectController<'_>,
        trace_turn_id: &str,
    ) -> Result<(), RuntimeError> {
        let Some(session) = self.session.as_ref() else {
            return Ok(());
        };
        let Ok(manager) = self.runtime_session_services() else {
            return Ok(());
        };
        let phase_turn_id = turn_phase_id(trace_turn_id, "turn-persisted");
        let phase_controller = scoped_child_turn_controller(
            scoped_effect_controller,
            &self.state.session_id,
            &phase_turn_id,
        )?;
        let direct_completions = manager.direct_completion_client(
            RuntimeEffectControllerHandle::borrowed(phase_controller),
            Some(phase_turn_id),
        );

        session
            .plugins()
            .emit_runtime_event_with_phase_probe(
                crate::PluginLifecycleEvent::TurnPersisted(Box::new(
                    crate::SessionStateChangedContext {
                        session_id: self.state.session_id.clone(),
                        state: crate::SessionReadView::from_snapshot(&returned_turn.state),
                        sessions: manager.state_service(),
                        session_graph: manager.graph_service(),
                        direct_completions,
                    },
                )),
                self.turn_phase_probe.clone(),
            )
            .await;
        Ok(())
    }

    /// Run a single turn and stream events to the host sink.
    pub async fn stream_turn(
        &mut self,
        input: TurnInput,
        opts: TurnOptions<'_>,
    ) -> Result<AssembledTurn, RuntimeError> {
        let cancel = opts.cancel.clone();
        let scoped_effect_controller = opts.scoped_effect_controller();
        self.stream_turn_with_scoped_effect_controller_inner(
            input,
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            scoped_effect_controller,
            cancel,
            None,
        )
        .await
    }

    pub async fn stream_next_queued_work(
        &mut self,
        opts: TurnOptions<'_>,
    ) -> Result<Option<AssembledTurn>, RuntimeError> {
        self.stream_queued_work(opts, None).await
    }

    pub async fn stream_selected_queued_work(
        &mut self,
        opts: TurnOptions<'_>,
        batch_ids: &[String],
    ) -> Result<Option<AssembledTurn>, RuntimeError> {
        self.stream_queued_work(opts, Some(batch_ids)).await
    }

    async fn stream_queued_work(
        &mut self,
        opts: TurnOptions<'_>,
        selected_batch_ids: Option<&[String]>,
    ) -> Result<Option<AssembledTurn>, RuntimeError> {
        if self.drain_next_session_command().await?.is_some() {
            return Ok(None);
        }
        let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        else {
            return Ok(None);
        };
        let claim = if let Some(batch_ids) = selected_batch_ids {
            store
                .claim_ready_queued_work_by_batch_ids(
                    &self.state.session_id,
                    &self.runtime_scope_id,
                    crate::QueuedWorkClaimBoundary::Idle,
                    crate::QUEUED_WORK_CLAIM_TTL_MS,
                    batch_ids,
                )
                .await
        } else {
            store
                .claim_ready_queued_work(
                    &self.state.session_id,
                    &self.runtime_scope_id,
                    crate::QueuedWorkClaimBoundary::Idle,
                    crate::QUEUED_WORK_CLAIM_TTL_MS,
                    64,
                )
                .await
        }
        .map_err(|err| RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string()))?;
        let Some(claim) = claim else {
            return Ok(None);
        };
        let mut work = claim.materialize_for_turn();
        let turn_id = work
            .input
            .trace_turn_id
            .clone()
            .or_else(|| Some(opts.execution_scope_id().to_owned()))
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        work.input.trace_turn_id = Some(turn_id.clone());
        let causes = work.turn_causes.clone();
        emit_queued_work_started_to_sink(
            opts.turn_events_or_noop(),
            crate::QueuedWorkClaimBoundary::Idle,
            &claim,
            causes.clone(),
        )
        .await;
        crate::trace::emit_trace(
            &self.host.core.tracing.trace_sink,
            &self.host.core.tracing.trace_context,
            lash_trace::TraceContext::default()
                .for_session(self.state.session_id.clone())
                .for_turn_index(self.state.turn_index + 1)
                .for_turn(turn_id.clone()),
            lash_trace::TraceEvent::Custom {
                name: "queued_work.claimed".to_string(),
                payload: queued_work_trace_payload(
                    crate::QueuedWorkClaimBoundary::Idle,
                    &claim,
                    &causes,
                ),
            },
        );
        let cancel = opts.cancel.clone();
        let scoped_effect_controller = opts.scoped_effect_controller();
        self.stream_turn_with_scoped_effect_controller_inner(
            work.input,
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            scoped_effect_controller,
            cancel,
            Some(claim),
        )
        .await
        .map(Some)
    }

    /// Enforce the durable-first wiring invariant at a turn-scope boundary: when
    /// the host wired a durable effect host, every store reachable from this
    /// scope must also be durable. A durable host running against any ephemeral
    /// store fails loudly here rather than silently degrading.
    ///
    /// Inline controllers (the default tier) impose no requirement, so
    /// inline/in-memory hosts pass unchanged.
    fn ensure_durable_store_facets_for_scope(
        &self,
        scoped_effect_controller: &ScopedEffectController<'_>,
    ) -> Result<(), RuntimeError> {
        if scoped_effect_controller.controller().durability_tier() != crate::DurabilityTier::Durable
        {
            return Ok(());
        }
        if self
            .host
            .core
            .durability
            .attachment_store
            .persistence()
            .durability_tier()
            != crate::DurabilityTier::Durable
        {
            return Err(RuntimeError::durable_store_required(
                crate::DurableStoreFacet::AttachmentStore,
            ));
        }
        if self
            .host
            .core
            .durability
            .lashlang_artifact_store
            .durability_tier()
            != crate::DurabilityTier::Durable
        {
            return Err(RuntimeError::durable_store_required(
                crate::DurableStoreFacet::ArtifactStore,
            ));
        }
        if let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
            && store.durability_tier() != crate::DurabilityTier::Durable
        {
            return Err(RuntimeError::durable_store_required(
                crate::DurableStoreFacet::SessionStore,
            ));
        }
        if let Some(process_registry) = self.host.process_registry.as_ref()
            && process_registry.durability_tier() != crate::DurabilityTier::Durable
        {
            return Err(RuntimeError::durable_store_required(
                crate::DurableStoreFacet::ProcessRegistry,
            ));
        }
        Ok(())
    }

    async fn stream_turn_with_scoped_effect_controller_inner(
        &mut self,
        mut input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
        cancel: CancellationToken,
        queued_claim: Option<crate::QueuedWorkClaim>,
    ) -> Result<AssembledTurn, RuntimeError> {
        if queued_claim.is_none() {
            while self.drain_next_session_command().await?.is_some() {}
        }
        if let Some(input_turn_id) = input.trace_turn_id.as_deref()
            && scoped_effect_controller
                .execution_scope()
                .validates_turn_trace_id()
            && input_turn_id != scoped_effect_controller.scope_id()
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::ExecutionScopeTurnIdMismatch,
                format!(
                    "input trace_turn_id `{input_turn_id}` does not match execution scope id `{}`",
                    scoped_effect_controller.scope_id()
                ),
            ));
        }
        self.ensure_durable_store_facets_for_scope(&scoped_effect_controller)?;
        input
            .trace_turn_id
            .get_or_insert_with(|| scoped_effect_controller.scope_id().to_string());
        self.stream_turn_inner(
            input.clone(),
            events,
            turn_events,
            scoped_effect_controller,
            cancel.clone(),
            queued_claim,
        )
        .await
    }

    /// Stream one logical host turn, following foreground AgentFrame switches
    /// until a terminal outcome is reached.
    ///
    /// RLM `continue_as` creates a new frame in the same session. Hosts that
    /// only care about the benchmark/app answer should not need to special-case
    /// that intermediate outcome; this helper keeps driving the same session
    /// through each frame's task with the normal runtime turn guards.
    pub async fn stream_turn_with_agent_frames(
        &mut self,
        input: TurnInput,
        opts: TurnOptions<'_>,
    ) -> Result<AgentFrameRun, RuntimeError> {
        let cancel = opts.cancel.clone();
        let scoped_effect_controller = opts.scoped_effect_controller();
        self.stream_turn_with_agent_frames_inner(
            input,
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            scoped_effect_controller,
            cancel,
        )
        .await
    }

    async fn stream_turn_with_agent_frames_inner(
        &mut self,
        mut input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
        cancel: CancellationToken,
    ) -> Result<AgentFrameRun, RuntimeError> {
        if let Some(input_turn_id) = input.trace_turn_id.as_deref()
            && scoped_effect_controller
                .execution_scope()
                .validates_turn_trace_id()
            && input_turn_id != scoped_effect_controller.scope_id()
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::ExecutionScopeTurnIdMismatch,
                format!(
                    "input trace_turn_id `{input_turn_id}` does not match execution scope id `{}`",
                    scoped_effect_controller.scope_id()
                ),
            ));
        }
        let follow_protocol_turn_options = input.protocol_turn_options.clone();
        let follow_turn_context = input.turn_context.clone();
        let follow_trace_turn_id = input
            .trace_turn_id
            .clone()
            .unwrap_or_else(|| scoped_effect_controller.scope_id().to_string());
        input
            .trace_turn_id
            .get_or_insert(follow_trace_turn_id.clone());
        let mut turns = Vec::new();
        loop {
            let turn_trace_turn_id = agent_frame_follow_turn_id(&follow_trace_turn_id, turns.len());
            input.trace_turn_id = Some(turn_trace_turn_id.clone());
            let turn_effect_controller = if turns.is_empty() {
                scoped_effect_controller.clone()
            } else {
                ScopedEffectController::borrowed(
                    scoped_effect_controller.controller(),
                    ExecutionScope::turn(&self.state.session_id, &turn_trace_turn_id),
                )?
            };
            let turn = self
                .stream_turn_with_scoped_effect_controller_inner(
                    input,
                    events,
                    turn_events,
                    turn_effect_controller,
                    cancel.clone(),
                    None,
                )
                .await?;
            let switched_frame = match &turn.outcome {
                TurnOutcome::AgentFrameSwitch { frame_id, task } => {
                    Some((frame_id.clone(), task.clone()))
                }
                _ => None,
            };
            turns.push(turn);

            let Some((_frame_id, task)) = switched_frame else {
                return Ok(AgentFrameRun { turns });
            };
            input = turn_input_from_text(task);
            input.protocol_turn_options = follow_protocol_turn_options.clone();
            input.turn_context = follow_turn_context.clone();
        }
    }

    async fn stream_turn_inner(
        &mut self,
        mut input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
        cancel: CancellationToken,
        queued_claim: Option<crate::QueuedWorkClaim>,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.refresh_session_graph_from_store()
            .await
            .map_err(session_head_refresh_error)?;
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
            ensure_durable_effect_input(&input)?;
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
        let normalized = match self
            .normalize_input_items(&input.items, &input.image_blobs)
            .await
        {
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
                    self.state.to_snapshot(),
                    false,
                    None,
                    &self.host.core.control.termination,
                ));
            }
        };
        let turn_index = self.state.turn_index + 1;
        let trace_turn_id = input
            .trace_turn_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if self.host.core.tracing.trace_sink.is_some() {
            let mut trace_metadata = std::collections::BTreeMap::new();
            trace_metadata.insert(
                "input_item_count".to_string(),
                serde_json::json!(normalized.len()),
            );
            crate::trace::emit_trace(
                &self.host.core.tracing.trace_sink,
                &self.host.core.tracing.trace_context,
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

        let manager = self
            .runtime_session_services_for_turn(None)
            .map_err(|err| {
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
        let prepare_phase_turn_id = turn_phase_id(&trace_turn_id, "prepare-turn");
        let prepare_phase_controller = scoped_child_turn_controller(
            &scoped_effect_controller,
            &self.state.session_id,
            &prepare_phase_turn_id,
        )?;
        let turn_ctx = crate::TurnTransformContext {
            session_id: self.state.session_id.clone(),
            state: self.read_view(),
            prompt_usage: previous_prompt_usage.clone(),
            max_context_tokens: Some(LashRuntime::max_context_tokens(self)),
            sessions: manager.state_service(),
            session_lifecycle: manager.lifecycle_service(),
            session_graph: manager.graph_service(),
            scoped_effect_controller: scoped_effect_controller.clone(),
            direct_completions: manager.direct_completion_client(
                RuntimeEffectControllerHandle::borrowed(prepare_phase_controller),
                Some(prepare_phase_turn_id),
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
                self.turn_phase_probe.clone(),
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
                .set_context_overlay(
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
        Box::pin(self.stream_prepared_turn(
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
            scoped_effect_controller,
            cancel,
            queued_claim,
        ))
        .await
    }

    /// Run a single turn and return only the assembled terminal result.
    pub async fn run_turn_assembled(
        &mut self,
        input: TurnInput,
        cancel: CancellationToken,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.stream_turn(input, TurnOptions::new(cancel, scoped_effect_controller))
            .await
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
        scoped_effect_controller: ScopedEffectController<'_>,
        cancel: CancellationToken,
        initial_queue_claim: Option<crate::QueuedWorkClaim>,
    ) -> Result<AssembledTurn, RuntimeError> {
        let (event_tx, mut event_rx) = mpsc::channel::<RuntimeStreamEvent>(100);
        let child_usage_event_relay = ChildUsageEventRelay::new(event_tx.clone());
        let mut turn_policy = self.state.effective_policy().clone();
        let turn_provider_override = turn_context.provider().cloned();
        if let Some(provider) = turn_provider_override.as_ref() {
            turn_policy.provider_id = provider.kind().to_string();
        }
        if let Some(model) = turn_context.model_spec() {
            turn_policy.model = model.clone();
        }
        let session_protocol_turn_options = self.state.effective_protocol_turn_options().clone();
        let effective_protocol_turn_options = protocol_turn_options
            .clone()
            .map(|options| session_protocol_turn_options.merged_with_override(&options))
            .unwrap_or(session_protocol_turn_options);
        let manager = self
            .runtime_session_services_for_turn(Some(child_usage_event_relay.clone()))
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
            let prepare_turn = plugins.prepare_turn_with_phase_probe(
                PrepareTurnRequest {
                    session_id: self.state.session_id.clone(),
                    state: crate::SessionReadView::from_runtime_state(
                        &self.state,
                        turn_policy.clone(),
                        effective_protocol_turn_options.clone(),
                    ),
                    messages,
                    sessions: manager.state_service(),
                    session_lifecycle: manager.lifecycle_service(),
                    session_graph: manager.graph_service(),
                    turn_context: turn_context.clone(),
                },
                self.turn_phase_probe.clone(),
            );
            let mut prepare_turn = Box::pin(prepare_turn);

            loop {
                tokio::select! {
                    prepared = prepare_turn.as_mut() => {
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

            let mut turn_pipeline = TurnBoundary::from_state(self.state.clone());
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
                state.to_snapshot(),
                cancel.is_cancelled(),
                Some(issue),
                &self.host.core.control.termination,
            ));
        }
        let mut turn_pipeline = TurnBoundary::from_state(self.state.clone());
        let store = self
            .session
            .as_ref()
            .and_then(|session| session.history_store());
        // Durable controllers, like Restate, own in-flight replay. Writing
        // progress checkpoints directly to the shared store would make handler
        // replay observe a newer partial turn and change effect replay keys.
        let progress_store = if scoped_effect_controller.controller().durability_tier()
            == crate::DurabilityTier::Durable
        {
            None
        } else {
            store.as_ref().map(|store| store.as_ref())
        };
        turn_pipeline
            .prepared_checkpoint(
                progress_store,
                turn_policy.clone(),
                turn_index,
                &prepared.messages,
                self.session.as_mut(),
            )
            .await
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string())
            })?;
        let resolved_turn_policy = if let Some(provider) = turn_provider_override {
            RuntimeSessionPolicy::from_provider(turn_policy.clone(), provider)
                .map_err(|err| RuntimeError::new("llm_provider", err.to_string()))?
        } else {
            self.host
                .resolve_session_policy(&self.state.session_id, turn_policy.clone())
                .map_err(|err| RuntimeError::new("llm_provider", err.to_string()))?
        };
        let manager = self
            .runtime_session_services_for_turn(Some(child_usage_event_relay.clone()))
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::PluginSessionManager, err.to_string())
            })?;
        let cancel_state = cancel.clone();
        let finish_scoped_effect_controller = scoped_effect_controller.clone();
        let session = self
            .session
            .take()
            .expect("lash runtime session must be available");
        let mut driver = RuntimeTurnDriver {
            session,
            policy: resolved_turn_policy,
            host: self.host.clone(),
            turn_id: scoped_effect_controller.scope_id().to_string(),
            scoped_effect_controller,
            session_id: self.state.session_id.clone(),
            turn_index,
            turn_pipeline,
            llm_stream_summaries: HashMap::new(),
            next_llm_ordinal: 0,
            session_services: manager,
            protocol_turn_options: effective_protocol_turn_options,
            protocol_extension,
            turn_context,
            turn_causes: initial_turn_causes,
            pending_queue_claims: initial_queue_claim.into_iter().collect(),
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            turn_phase_probe: self.turn_phase_probe.clone(),
        };
        let protocol_run_offset = 0;
        self.mark_phase_begin(RuntimeTurnPhase::EffectLoop);
        let run_result = drive_turn_to_completion(
            driver.run(prepared.messages, event_tx, cancel, protocol_run_offset),
            &mut event_rx,
            &mut assembler,
            &child_usage_event_relay,
            events,
            turn_events,
        )
        .await;
        let (new_messages, _new_protocol_iteration) = match run_result {
            Ok(result) => result,
            Err(err) => {
                self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
                let RuntimeTurnDriver { session, .. } = driver;
                self.session = Some(session);
                return Err(err);
            }
        };
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
            pending_queue_claims,
            ..
        } = driver;
        self.session = Some(session);
        self.finish_turn(
            TurnFinishInput {
                turn_pipeline,
                assembler,
                new_messages,
                policy,
                turn_index,
                queued_work_completions: pending_queue_claims
                    .iter()
                    .map(crate::QueuedWorkClaim::completion)
                    .collect(),
                trace_turn_id,
            },
            events,
            &finish_scoped_effect_controller,
            &cancel_state,
        )
        .await
    }
    async fn normalize_input_items(
        &self,
        items: &[InputItem],
        image_blobs: &HashMap<String, Vec<u8>>,
    ) -> Result<Vec<NormalizedItem>, String> {
        normalize_input_items(
            items,
            image_blobs,
            self.host.core.durability.attachment_store.as_ref(),
        )
        .await
    }
}

fn turn_input_from_text(text: String) -> TurnInput {
    TurnInput {
        items: vec![InputItem::Text { text }],
        image_blobs: HashMap::new(),
        protocol_turn_options: None,
        trace_turn_id: None,
        protocol_extension: None,
        turn_context: crate::TurnContext::default(),
    }
}

fn agent_frame_follow_turn_id(root_turn_id: &str, completed_turn_count: usize) -> String {
    if completed_turn_count == 0 {
        root_turn_id.to_string()
    } else {
        format!("{root_turn_id}:agent-frame:{completed_turn_count}")
    }
}

pub fn ensure_durable_effect_input(input: &TurnInput) -> Result<(), RuntimeError> {
    if input.protocol_extension.is_some() {
        return Err(RuntimeError::new(
            RuntimeErrorCode::DurableEffectLiveProtocolExtension,
            "durable effect hosts do not support live protocol_extension inputs; encode replayable data in protocol_turn_options or persisted plugin state",
        ));
    }
    input
        .turn_context
        .live_plugin_inputs()
        .durable_effect_rejection()?;
    Ok(())
}

async fn emit_turn_activity_to_sink(events: &dyn TurnActivitySink, activity: TurnActivity) {
    if !events.is_noop() {
        events.emit(activity).await;
    }
}

/// Pump the turn driver's event channel into the host sinks while the run
/// future executes, then drain any events emitted between completion and the
/// sender dropping.
///
/// Both the fresh and resumed turn entry points construct a
/// `RuntimeTurnDriver`, kick off its run future, and need identical
/// event-pump/drain behavior before tearing the driver down. Only the driver
/// construction and post-run teardown differ, so each caller owns those and
/// shares this loop.
async fn drive_turn_to_completion<F>(
    run_future: F,
    event_rx: &mut mpsc::Receiver<RuntimeStreamEvent>,
    assembler: &mut TurnAssembler,
    child_usage_event_relay: &ChildUsageEventRelay,
    events: &dyn EventSink,
    turn_events: &dyn TurnActivitySink,
) -> Result<(crate::MessageSequence, usize), RuntimeError>
where
    F: std::future::Future<Output = Result<(crate::MessageSequence, usize), RuntimeError>>,
{
    let run_result = {
        let mut run_future = Box::pin(run_future);
        loop {
            tokio::select! {
                maybe_event = event_rx.recv() => {
                    if let Some(event) = maybe_event {
                        emit_runtime_stream_event_to_sinks(
                            events,
                            turn_events,
                            event,
                            assembler,
                        )
                        .await;
                    }
                }
                completed = run_future.as_mut() => {
                    child_usage_event_relay.clear();
                    break completed;
                }
            }
        }
    };
    while let Some(event) = event_rx.recv().await {
        emit_runtime_stream_event_to_sinks(events, turn_events, event, assembler).await;
    }
    run_result
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

#[cfg(test)]
mod tests {
    use super::agent_frame_follow_turn_id;

    #[test]
    fn agent_frame_follow_turn_ids_are_distinct_and_deterministic() {
        assert_eq!(agent_frame_follow_turn_id("root-turn", 0), "root-turn");
        assert_eq!(
            agent_frame_follow_turn_id("root-turn", 1),
            "root-turn:agent-frame:1"
        );
        assert_eq!(
            agent_frame_follow_turn_id("root-turn", 2),
            "root-turn:agent-frame:2"
        );
    }
}
