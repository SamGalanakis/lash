#[cfg(test)]
use super::logical_turn::agent_frame_follow_turn_id;
use super::logical_turn::{
    LogicalTurnClaims, LogicalTurnStart, PhysicalTurnExecution, PreparedLogicalTurn,
};
use super::turn_control::ActiveTurnControl;
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
        TurnOutcome::Finished(TurnFinish::FinalValue { .. }) => ("completed", "final_value", None),
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

#[derive(Clone, Copy)]
pub(super) enum SessionExecutionLeaseReleasePolicy {
    KeepOnAgentFrameSwitch,
}

impl SessionExecutionLeaseReleasePolicy {
    fn should_release(self, outcome: &TurnOutcome) -> bool {
        match self {
            Self::KeepOnAgentFrameSwitch => {
                !matches!(outcome, TurnOutcome::AgentFrameSwitch { .. })
            }
        }
    }
}

fn queued_work_payload_type(payload: &crate::QueuedWorkPayload) -> &'static str {
    match payload {
        crate::QueuedWorkPayload::ProcessWake { .. } => "process_wake",
        crate::QueuedWorkPayload::AgentFrameTask { .. } => "agent_frame_task",
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

/// Measures the whole host-visible turn.
///
/// Opened before the runtime claims the turn (session-execution lease and
/// queued-work/turn-input claims) and stamped onto the assembled turn after
/// the final commit and post-persist hooks complete, so
/// [`ExecutionSummary`](crate::ExecutionSummary) timing covers
/// claim → final commit. Reads only the injected [`Clock`](crate::Clock):
/// `started_at_ms` comes from the wall-clock source and the duration from the
/// monotonic source, so deterministic clocks produce deterministic timing.
#[derive(Clone, Copy)]
pub(super) struct TurnStopwatch {
    started: std::time::Instant,
    started_at_ms: u64,
}

impl TurnStopwatch {
    pub(super) fn start(clock: &dyn crate::Clock) -> Self {
        Self {
            started: clock.now(),
            started_at_ms: clock.timestamp_ms(),
        }
    }

    pub(super) fn stamp(&self, turn: &mut AssembledTurn, clock: &dyn crate::Clock) {
        turn.execution.started_at_ms = self.started_at_ms;
        turn.execution.duration_ms = clock
            .now()
            .saturating_duration_since(self.started)
            .as_millis() as u64;
    }
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
        "owner_id": claim.owner.owner_id,
        "incarnation_id": claim.owner.incarnation_id,
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

pub(in crate::runtime) fn turn_input_completion_trace_payload(
    completions: &[crate::TurnInputCompletion],
) -> serde_json::Value {
    serde_json::json!({
        "claims": completions.iter().map(|completion| {
            serde_json::json!({
                "session_id": completion.session_id,
                "claim_id": completion.claim_id,
                "input_ids": completion.input_ids,
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
    queued_work_claims: Vec<crate::QueuedWorkClaim>,
    turn_input_claims: Vec<crate::TurnInputClaim>,
    trace_turn_id: String,
}

impl LashRuntime {
    fn max_context_tokens(&self) -> usize {
        self.state.effective_policy().context_window_tokens()
    }

    async fn claim_session_execution_lease(
        &self,
        cancel: CancellationToken,
        busy_is_error: bool,
    ) -> Result<Option<SessionExecutionLeaseGuard>, RuntimeError> {
        let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        else {
            return Ok(None);
        };
        match SessionExecutionLeaseGuard::try_acquire(
            store,
            &self.state.session_id,
            &self.runtime_lease_owner,
            self.host.core.control.lease_timings,
            Arc::clone(&self.host.core.clock),
            cancel,
        )
        .await
        .map_err(|err| RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string()))?
        {
            Some(lease) => Ok(Some(lease)),
            None if busy_is_error => Err(RuntimeError::new(
                RuntimeErrorCode::SessionExecutionBusy,
                format!(
                    "session `{}` is already executing on another runtime owner",
                    self.state.session_id
                ),
            )),
            None => Ok(None),
        }
    }

    async fn settle_session_execution_lease<T>(
        &self,
        guard: Option<&SessionExecutionLeaseGuard>,
        result: Result<T, RuntimeError>,
    ) -> Result<T, RuntimeError> {
        match result {
            Ok(value) => {
                if let Some(guard) = guard {
                    guard.release_if_live().await.map_err(|err| {
                        RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string())
                    })?;
                }
                Ok(value)
            }
            Err(err) => {
                if err.code != RuntimeErrorCode::StoreCommitFailed
                    && let Some(guard) = guard
                    && let Err(release_err) = guard.release_if_live().await
                {
                    tracing::warn!(
                        error = %release_err,
                        "failed to release session execution lease after runtime error"
                    );
                }
                Err(err)
            }
        }
    }

    async fn ensure_session_execution_lease_live(
        &self,
        guard: Option<&SessionExecutionLeaseGuard>,
    ) -> Result<(), RuntimeError> {
        let Some(guard) = guard else {
            return Ok(());
        };
        guard.refresh_or_mark_lost().await.map_err(|err| {
            RuntimeError::new(
                RuntimeErrorCode::SessionExecutionLeaseLost,
                format!(
                    "session execution lease for session `{}` was lost before commit: {err}",
                    self.state.session_id
                ),
            )
        })
    }

    // Prompt handback on lease loss. This is no longer load-bearing for
    // correctness: a claim is generation-fenced under the session lease, so once
    // this owner has lost the lease its claims are already superseded and the
    // next acquirer reclaims them by generation regardless (ADR 0029). Abandoning
    // eagerly just lets a peer reclaim the rows without waiting to observe the
    // generation bump.
    async fn abandon_queued_work_claims_after_lease_loss(
        &self,
        err: &RuntimeError,
        claims: &[crate::QueuedWorkClaim],
    ) {
        if err.code != RuntimeErrorCode::SessionExecutionLeaseLost || claims.is_empty() {
            return;
        }
        let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        else {
            return;
        };
        if let Err(abandon_err) = store.abandon_queued_work_claims(claims).await {
            tracing::warn!(
                error = %abandon_err,
                claim_count = claims.len(),
                "failed to abandon queued work claims after session execution lease loss"
            );
        }
    }

    async fn abandon_turn_input_claims_after_lease_loss(
        &self,
        err: &RuntimeError,
        claims: &[crate::TurnInputClaim],
    ) {
        if err.code != RuntimeErrorCode::SessionExecutionLeaseLost || claims.is_empty() {
            return;
        }
        let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        else {
            return;
        };
        if let Err(abandon_err) = store.abandon_turn_input_claims(claims).await {
            tracing::warn!(
                error = %abandon_err,
                claim_count = claims.len(),
                "failed to abandon turn input claims after session execution lease loss"
            );
        }
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

    #[allow(clippy::too_many_arguments)]
    async fn finish_turn(
        &mut self,
        finish: TurnFinishInput,
        events: &dyn EventSink,
        scoped_effect_controller: &ScopedEffectController<'_>,
        cancel_state: &CancellationToken,
        session_execution_lease: Option<&SessionExecutionLeaseGuard>,
        session_execution_lease_release_policy: SessionExecutionLeaseReleasePolicy,
        turn_control: &ActiveTurnControl,
    ) -> Result<PhysicalTurnExecution, RuntimeError> {
        let TurnFinishInput {
            mut turn_pipeline,
            assembler,
            new_messages,
            policy,
            turn_index,
            queued_work_claims,
            turn_input_claims,
            trace_turn_id,
        } = finish;
        self.policy = self.state.effective_policy().clone();
        turn_pipeline.state_mut().policy = self.policy.clone();
        turn_pipeline.state_mut().turn_index = turn_index;

        let mut turn_usage_delta = {
            let mut ledger = self.shared_token_ledger.lock().expect("token ledger lock");
            std::mem::take(&mut *ledger)
        };
        if assembler.token_usage.total() > 0 {
            turn_usage_delta.push(TokenLedgerEntry {
                source: "turn".to_string(),
                model: policy.model.id.clone(),
                usage: assembler.token_usage.clone(),
            });
        }
        let turn_usage_delta = merge_usage_delta_entries(turn_usage_delta);

        if self.session.is_some() {
            self.ensure_session_execution_lease_live(session_execution_lease)
                .await?;
        }
        let assembled_cancelled = matches!(
            assembler.outcome,
            Some(TurnOutcome::Stopped(TurnStop::Cancelled))
        );
        let lease_was_lost = session_execution_lease.is_some_and(|lease| lease.is_lost());
        let cancellation = turn_control
            .settle_before_commit(
                scoped_effect_controller.controller(),
                assembled_cancelled || (cancel_state.is_cancelled() && !lease_was_lost),
            )
            .await?;
        if cancellation.is_some() {
            cancel_state.cancel();
        }
        let interrupted = cancel_state.is_cancelled();

        turn_pipeline.finalize_turn_read_state(new_messages, interrupted);
        if assembler.token_usage.total() > 0 {
            turn_pipeline.state_mut().token_usage = assembler.token_usage.clone();
        }

        let last_prompt_usage = assembler.last_llm_usage().and_then(normalize_prompt_usage);
        turn_pipeline.state_mut().last_prompt_usage = last_prompt_usage;
        let assembled_state = turn_pipeline.export_state_for_assembly();
        let mut assembled = assembler.finish(
            assembled_state,
            interrupted,
            None,
            &self.host.core.control.termination,
        );
        assembled.cancellation = cancellation;

        let Some(session) = self.session.as_ref() else {
            self.state.apply_snapshot(&assembled.state);
            self.emit_completed_turn_trace(&assembled.state, &assembled.outcome, &trace_turn_id);
            publish_terminal_after_commit(
                turn_control,
                scoped_effect_controller.controller(),
                &TurnTerminal::Committed {
                    outcome: assembled.outcome.clone(),
                    cancellation: assembled.cancellation.clone(),
                    session_revision: None,
                },
                &self.state.session_id,
                &trace_turn_id,
            )
            .await;
            return Ok(PhysicalTurnExecution {
                turn: assembled,
                enqueued_queue_batches: Vec::new(),
            });
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
        self.ensure_session_execution_lease_live(session_execution_lease)
            .await?;

        let mut returned_turn = finalized.turn;
        if returned_turn.cancellation.is_some()
            && !matches!(
                returned_turn.outcome,
                TurnOutcome::Stopped(TurnStop::Cancelled)
            )
        {
            returned_turn.outcome = TurnOutcome::Stopped(TurnStop::Cancelled);
        }
        if matches!(
            returned_turn.outcome,
            TurnOutcome::Stopped(TurnStop::Cancelled)
        ) && returned_turn.cancellation.is_none()
        {
            return Err(RuntimeError::new(
                "turn_cancellation_evidence_missing",
                "cancelled turns must carry cancellation evidence",
            ));
        }
        let release_session_execution_lease =
            session_execution_lease_release_policy.should_release(&returned_turn.outcome);
        let commit_effects = LogicalTurnClaims::new(queued_work_claims, turn_input_claims)
            .into_commit_effects(
                &returned_turn.outcome,
                &self.state.session_id,
                &trace_turn_id,
                Some(self.state.effective_protocol_turn_options().clone()),
            );
        self.mark_phase_begin(RuntimeTurnPhase::PersistTurn);
        self.mark_phase_begin(RuntimeTurnPhase::FinalCommit);
        let queued_work_completion_trace = commit_effects.completed_queue_claims.clone();
        let turn_input_completion_trace = commit_effects.completed_turn_input_claims.clone();
        let pending_attachment_ids = self
            .host
            .core
            .durability
            .attachment_store
            .pending_manifest_commit_ids();
        let enqueued_queue_batches = match turn_pipeline
            .final_commit(
                &mut returned_turn,
                self.session.as_mut(),
                &turn_usage_delta,
                Some(&trace_turn_id),
                commit_effects.originating_queue_claims,
                commit_effects.originating_turn_input_claims,
                commit_effects.completed_queue_claims,
                commit_effects.completed_turn_input_claims,
                commit_effects.enqueued_queue_batches,
                cancel_state.is_cancelled().then(|| trace_turn_id.clone()),
                pending_attachment_ids.clone(),
                release_session_execution_lease
                    .then(|| session_execution_lease.map(SessionExecutionLeaseGuard::completion))
                    .flatten(),
            )
            .await
        {
            Ok(batches) => batches,
            Err(err) => {
                self.mark_phase_end(RuntimeTurnPhase::FinalCommit);
                self.mark_phase_end(RuntimeTurnPhase::PersistTurn);
                return Err(err);
            }
        };
        if release_session_execution_lease && let Some(lease) = session_execution_lease {
            lease.mark_released();
        }
        self.host
            .core
            .durability
            .attachment_store
            .mark_manifest_committed(&pending_attachment_ids);
        self.mark_phase_end(RuntimeTurnPhase::FinalCommit);

        emit_session_events_to_sink(events, finalized.events).await;
        self.state = turn_pipeline.into_final_state();
        publish_terminal_after_commit(
            turn_control,
            scoped_effect_controller.controller(),
            &TurnTerminal::Committed {
                outcome: returned_turn.outcome.clone(),
                cancellation: returned_turn.cancellation.clone(),
                session_revision: None,
            },
            &self.state.session_id,
            &trace_turn_id,
        )
        .await;
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
                self.host.core.clock.as_ref(),
            );
        }
        if !turn_input_completion_trace.is_empty() {
            crate::trace::emit_trace(
                &self.host.core.tracing.trace_sink,
                &self.host.core.tracing.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(returned_turn.state.session_id.clone())
                    .for_turn_index(returned_turn.state.turn_index)
                    .for_turn(trace_turn_id.clone()),
                lash_trace::TraceEvent::Custom {
                    name: "turn_input.completed".to_string(),
                    payload: turn_input_completion_trace_payload(&turn_input_completion_trace),
                },
                self.host.core.clock.as_ref(),
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
        Ok(PhysicalTurnExecution {
            turn: returned_turn,
            enqueued_queue_batches,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn finish_cancelled_turn_after_effect_abort(
        &mut self,
        driver: RuntimeTurnDriver<'_>,
        mut assembler: TurnAssembler,
        cancellation_messages: crate::MessageSequence,
        events: &dyn EventSink,
        finish_scoped_effect_controller: &ScopedEffectController<'_>,
        cancel: &CancellationToken,
        session_execution_lease: Option<&SessionExecutionLeaseGuard>,
        session_execution_lease_release_policy: SessionExecutionLeaseReleasePolicy,
        turn_control: &ActiveTurnControl,
        turn_index: usize,
        trace_turn_id: String,
    ) -> Result<PhysicalTurnExecution, RuntimeError> {
        let RuntimeTurnDriver {
            session,
            policy,
            turn_pipeline,
            pending_queue_claims,
            pending_turn_input_claims,
            ..
        } = driver;
        self.session = Some(session);
        let outcome_event = SessionStreamEvent::TurnOutcome {
            outcome: TurnOutcome::Stopped(TurnStop::Cancelled),
        };
        assembler.push(&outcome_event);
        emit_session_event_to_sink(events, outcome_event).await;
        assembler.push(&SessionStreamEvent::Done);
        emit_session_event_to_sink(events, SessionStreamEvent::Done).await;
        self.finish_turn(
            TurnFinishInput {
                turn_pipeline,
                assembler,
                new_messages: cancellation_messages,
                policy,
                turn_index,
                queued_work_claims: pending_queue_claims,
                turn_input_claims: pending_turn_input_claims,
                trace_turn_id,
            },
            events,
            finish_scoped_effect_controller,
            cancel,
            session_execution_lease,
            session_execution_lease_release_policy,
            turn_control,
        )
        .await
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
            self.host.core.clock.as_ref(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn finish_logical_turn_error(
        &mut self,
        message: String,
        trace_turn_id: String,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
        cancel: CancellationToken,
        claims: LogicalTurnClaims,
        session_execution_lease: Option<&SessionExecutionLeaseGuard>,
    ) -> Result<PhysicalTurnExecution, RuntimeError> {
        let turn_control = Arc::new(
            ActiveTurnControl::new(
                scoped_effect_controller.controller(),
                TurnAddress::new(&self.state.session_id, &trace_turn_id),
            )
            .await?,
        );
        let mut assembler = TurnAssembler::default();
        let error_event = SessionStreamEvent::Error {
            message: message.clone(),
            envelope: Some(crate::session_model::ErrorEnvelope {
                kind: "runtime".to_string(),
                code: Some("agent_frame_switch_limit".to_string()),
                terminal_reason: None,
                user_message: message.clone(),
                raw: None,
                retryable: Some(false),
                provider_failure_kind: None,
            }),
        };
        assembler.push(&error_event);
        emit_turn_activity_to_sink(
            turn_events,
            TurnActivity::independent(TurnEvent::Error {
                message: message.clone(),
            }),
        )
        .await;
        emit_session_event_to_sink(events, error_event).await;
        let outcome_event = SessionStreamEvent::TurnOutcome {
            outcome: TurnOutcome::Stopped(TurnStop::RuntimeError),
        };
        assembler.push(&outcome_event);
        emit_session_event_to_sink(events, outcome_event).await;
        assembler.push(&SessionStreamEvent::Done);
        emit_session_event_to_sink(events, SessionStreamEvent::Done).await;

        let messages = crate::MessageSequence::from_base(self.state.read_model().messages);
        let mut turn_pipeline = TurnBoundary::from_state_with_clock(
            self.state.clone(),
            Arc::clone(&self.host.core.clock),
        )
        .with_session_execution_lease(
            session_execution_lease.map(SessionExecutionLeaseGuard::fence),
        );
        turn_pipeline.apply_prepared_messages(&messages);
        self.finish_turn(
            TurnFinishInput {
                turn_pipeline,
                assembler,
                new_messages: messages,
                policy: RuntimeSessionPolicy::new(
                    self.state.effective_policy().clone(),
                    Default::default(),
                ),
                turn_index: self.state.turn_index + 1,
                queued_work_claims: claims.queued,
                turn_input_claims: claims.turn_inputs,
                trace_turn_id,
            },
            events,
            &scoped_effect_controller,
            &cancel,
            session_execution_lease,
            SessionExecutionLeaseReleasePolicy::KeepOnAgentFrameSwitch,
            &turn_control,
        )
        .await
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

    /// Run one logical turn and stream every physical frame to the host sink.
    pub async fn stream_turn(
        &mut self,
        mut input: TurnInput,
        opts: TurnOptions<'_>,
    ) -> Result<AssembledTurn, RuntimeError> {
        if let Some(hint) = opts.local_cancel_origin_hint() {
            input.turn_context.set_local_cancel_origin_hint(hint);
        }
        let stopwatch = TurnStopwatch::start(self.host.core.clock.as_ref());
        let cancel = opts.cancel.clone();
        let session_execution_lease = self
            .claim_session_execution_lease(cancel.clone(), true)
            .await?;
        let scoped_effect_controller = opts.scoped_effect_controller();
        let result = Box::pin(self.drive_logical_turn(
            LogicalTurnStart::Input(input),
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            scoped_effect_controller,
            cancel,
            LogicalTurnClaims::new(Vec::new(), Vec::new()),
            session_execution_lease.as_ref(),
            stopwatch,
        ))
        .await
        .map(|run| {
            run.into_final_turn()
                .expect("logical turn always contains a terminal physical turn")
        });
        self.settle_session_execution_lease(session_execution_lease.as_ref(), result)
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
        let stopwatch = TurnStopwatch::start(self.host.core.clock.as_ref());
        let cancel = opts.cancel.clone();
        let Some(session_execution_lease) = self
            .claim_session_execution_lease(cancel.clone(), false)
            .await?
        else {
            return Ok(None);
        };
        let session_execution_fence = session_execution_lease.fence();
        let Some(store) = self
            .session
            .as_ref()
            .and_then(|session| session.history_store())
        else {
            session_execution_lease
                .release_if_live()
                .await
                .map_err(|err| {
                    RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string())
                })?;
            return Ok(None);
        };
        let drain_commands_before_turn_input = if selected_batch_ids.is_some() {
            true
        } else {
            self.session_commands_precede_pending_turn_input(store.as_ref())
                .await?
        };
        if drain_commands_before_turn_input {
            loop {
                match self
                    .drain_next_session_command(&session_execution_fence)
                    .await
                {
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(err) => {
                        let _ = session_execution_lease.release_if_live().await;
                        return Err(err);
                    }
                }
            }
        }
        if selected_batch_ids.is_none() {
            let input_claim = store
                .claim_next_turn_inputs(
                    &self.state.session_id,
                    &session_execution_fence,
                    &self.runtime_lease_owner,
                    64,
                )
                .await
                .map_err(super::runtime_error_from_store_commit)?;
            if let Some(input_claim) = input_claim {
                let mut input = input_claim.materialize_for_turn();
                if let Some(hint) = opts.local_cancel_origin_hint() {
                    input.turn_context.set_local_cancel_origin_hint(hint);
                }
                let turn_id = input
                    .trace_turn_id
                    .clone()
                    .or_else(|| Some(opts.execution_scope_id().to_owned()))
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                input.trace_turn_id = Some(turn_id.clone());
                crate::trace::emit_trace(
                    &self.host.core.tracing.trace_sink,
                    &self.host.core.tracing.trace_context,
                    lash_trace::TraceContext::default()
                        .for_session(self.state.session_id.clone())
                        .for_turn_index(self.state.turn_index + 1)
                        .for_turn(turn_id.clone()),
                    lash_trace::TraceEvent::Custom {
                        name: "turn_input.claimed".to_string(),
                        payload: serde_json::json!({
                            "claim_id": &input_claim.claim_id,
                            "input_ids": input_claim.inputs.iter().map(|input| input.input_id.clone()).collect::<Vec<_>>(),
                        }),
                    },
                    self.host.core.clock.as_ref(),
                );
                let claim_for_abandon = input_claim.clone();
                let scoped_effect_controller = opts.scoped_effect_controller();
                let result = Box::pin(self.drive_logical_turn(
                    LogicalTurnStart::Input(input),
                    opts.events_or_noop(),
                    opts.turn_events_or_noop(),
                    scoped_effect_controller,
                    cancel,
                    LogicalTurnClaims::new(Vec::new(), vec![input_claim]),
                    Some(&session_execution_lease),
                    stopwatch,
                ))
                .await
                .map(AgentFrameRun::into_final_turn);
                if let Err(err) = &result {
                    self.abandon_turn_input_claims_after_lease_loss(
                        err,
                        std::slice::from_ref(&claim_for_abandon),
                    )
                    .await;
                }
                return self
                    .settle_session_execution_lease(Some(&session_execution_lease), result)
                    .await;
            }
        }
        let claim = if let Some(batch_ids) = selected_batch_ids {
            store
                .claim_ready_queued_work_by_batch_ids(
                    &self.state.session_id,
                    &session_execution_fence,
                    &self.runtime_lease_owner,
                    crate::QueuedWorkClaimBoundary::Idle,
                    batch_ids,
                )
                .await
        } else {
            store
                .claim_ready_queued_work(
                    &self.state.session_id,
                    &session_execution_fence,
                    &self.runtime_lease_owner,
                    crate::QueuedWorkClaimBoundary::Idle,
                    64,
                )
                .await
        }
        .map_err(super::runtime_error_from_store_commit)?;
        let Some(claim) = claim else {
            session_execution_lease
                .release_if_live()
                .await
                .map_err(|err| {
                    RuntimeError::new(RuntimeErrorCode::StoreCommitFailed, err.to_string())
                })?;
            return Ok(None);
        };
        let mut work = claim.materialize_for_turn();
        if let Some(hint) = opts.local_cancel_origin_hint() {
            work.input.turn_context.set_local_cancel_origin_hint(hint);
        }
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
            self.host.core.clock.as_ref(),
        );
        let claim_for_abandon = claim.clone();
        let scoped_effect_controller = opts.scoped_effect_controller();
        let result = Box::pin(self.drive_logical_turn(
            LogicalTurnStart::Input(work.input),
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            scoped_effect_controller,
            cancel,
            LogicalTurnClaims::new(vec![claim], Vec::new()),
            Some(&session_execution_lease),
            stopwatch,
        ))
        .await
        .map(AgentFrameRun::into_final_turn);
        if let Err(err) = &result {
            self.abandon_queued_work_claims_after_lease_loss(
                err,
                std::slice::from_ref(&claim_for_abandon),
            )
            .await;
        }
        self.settle_session_execution_lease(Some(&session_execution_lease), result)
            .await
    }

    async fn session_commands_precede_pending_turn_input(
        &self,
        store: &dyn crate::RuntimePersistence,
    ) -> Result<bool, RuntimeError> {
        let pending_inputs = store
            .list_pending_turn_inputs(&self.state.session_id)
            .await
            .map_err(super::runtime_error_from_store_commit)?;
        let earliest_input = pending_inputs
            .iter()
            .filter(|input| input.state.is_next_turn_pending())
            .min_by_key(|input| (input.enqueued_at_ms, input.enqueue_seq));
        let queued_work = store
            .list_pending_queued_work(&self.state.session_id)
            .await
            .map_err(super::runtime_error_from_store_commit)?;
        let earliest_command = queued_work
            .iter()
            .filter(|batch| batch.is_session_command_work())
            .min_by_key(|batch| (batch.enqueued_at_ms, batch.enqueue_seq));
        Ok(match (earliest_command, earliest_input) {
            (Some(command), Some(input)) => command.enqueued_at_ms < input.enqueued_at_ms,
            (Some(_), None) => true,
            _ => false,
        })
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
            .process_env_store
            .durability_tier()
            != crate::DurabilityTier::Durable
        {
            return Err(RuntimeError::durable_store_required(
                crate::DurableStoreFacet::ProcessEnvStore,
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

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn stream_turn_with_scoped_effect_controller_inner(
        &mut self,
        mut input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
        cancel: CancellationToken,
        queued_claims: Vec<crate::QueuedWorkClaim>,
        turn_input_claims: Vec<crate::TurnInputClaim>,
        materialize_initial_claims: bool,
        session_execution_lease: Option<&SessionExecutionLeaseGuard>,
        session_execution_lease_release_policy: SessionExecutionLeaseReleasePolicy,
    ) -> Result<PhysicalTurnExecution, RuntimeError> {
        if queued_claims.is_empty() && turn_input_claims.is_empty() {
            if let Some(lease) = session_execution_lease {
                while self
                    .drain_next_session_command(&lease.fence())
                    .await?
                    .is_some()
                {}
            } else if self
                .session
                .as_ref()
                .and_then(|session| session.history_store())
                .is_some()
            {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::StoreCommitFailed,
                    "session command drain requires a session execution lease",
                ));
            }
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
            queued_claims,
            turn_input_claims,
            materialize_initial_claims,
            session_execution_lease,
            session_execution_lease_release_policy,
        )
        .await
    }

    /// Stream one logical host turn, following foreground AgentFrame switches
    /// until a terminal outcome is reached.
    ///
    /// A protocol continuation creates a new frame in the same session. Hosts
    /// that only care about the benchmark/app answer should not need to
    /// special-case that intermediate outcome; this helper keeps driving the
    /// same session through each frame's task with the normal runtime turn
    /// guards.
    pub async fn stream_turn_with_agent_frames(
        &mut self,
        mut input: TurnInput,
        opts: TurnOptions<'_>,
    ) -> Result<AgentFrameRun, RuntimeError> {
        if let Some(hint) = opts.local_cancel_origin_hint() {
            input.turn_context.set_local_cancel_origin_hint(hint);
        }
        let stopwatch = TurnStopwatch::start(self.host.core.clock.as_ref());
        let cancel = opts.cancel.clone();
        let session_execution_lease = self
            .claim_session_execution_lease(cancel.clone(), true)
            .await?;
        let scoped_effect_controller = opts.scoped_effect_controller();
        let result = Box::pin(self.drive_logical_turn(
            LogicalTurnStart::Input(input),
            opts.events_or_noop(),
            opts.turn_events_or_noop(),
            scoped_effect_controller,
            cancel,
            LogicalTurnClaims::new(Vec::new(), Vec::new()),
            session_execution_lease.as_ref(),
            stopwatch,
        ))
        .await;
        self.settle_session_execution_lease(session_execution_lease.as_ref(), result)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream_turn_inner(
        &mut self,
        mut input: TurnInput,
        events: &dyn EventSink,
        turn_events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
        cancel: CancellationToken,
        queued_claims: Vec<crate::QueuedWorkClaim>,
        turn_input_claims: Vec<crate::TurnInputClaim>,
        materialize_initial_claims: bool,
        session_execution_lease: Option<&SessionExecutionLeaseGuard>,
        session_execution_lease_release_policy: SessionExecutionLeaseReleasePolicy,
    ) -> Result<PhysicalTurnExecution, RuntimeError> {
        self.refresh_session_graph_from_store()
            .await
            .map_err(session_head_refresh_error)?;
        let input_trace_turn_id = input.trace_turn_id.clone();
        let queued_turn_work = materialize_initial_claims
            .then(|| queued_claims.first())
            .flatten()
            .map(crate::QueuedWorkClaim::materialize_for_turn);
        let pending_turn_input = materialize_initial_claims
            .then(|| turn_input_claims.first())
            .flatten()
            .map(crate::TurnInputClaim::materialize_for_turn);
        if let Some(work) = pending_turn_input.as_ref()
            && input.items.is_empty()
            && input.image_blobs.is_empty()
        {
            input = work.clone();
            if input.trace_turn_id.is_none() {
                input.trace_turn_id = input_trace_turn_id.clone();
            }
        }
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
                let error_event = SessionStreamEvent::Error {
                    message: e.clone(),
                    envelope: Some(crate::session_model::ErrorEnvelope {
                        kind: "input_validation".to_string(),
                        code: Some("invalid_turn_input".to_string()),
                        terminal_reason: None,
                        user_message: e.clone(),
                        raw: None,
                        retryable: Some(false),
                        provider_failure_kind: None,
                    }),
                };
                assembler.push(&error_event);
                emit_turn_activity_to_sink(
                    turn_events,
                    TurnActivity::independent(TurnEvent::Error { message: e }),
                )
                .await;
                emit_session_event_to_sink(events, error_event).await;
                let outcome_event = SessionStreamEvent::TurnOutcome {
                    outcome: TurnOutcome::Stopped(TurnStop::InvalidInput),
                };
                assembler.push(&outcome_event);
                emit_session_event_to_sink(events, outcome_event).await;
                assembler.push(&SessionStreamEvent::Done);
                emit_session_event_to_sink(events, SessionStreamEvent::Done).await;
                let turn_index = self.state.turn_index + 1;
                let trace_turn_id = input
                    .trace_turn_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                let turn_control = ActiveTurnControl::new(
                    scoped_effect_controller.controller(),
                    TurnAddress::new(&self.state.session_id, &trace_turn_id),
                )
                .await?
                .with_local_cancel_origin(input.turn_context.local_cancel_origin_hint());
                let messages = crate::MessageSequence::from_base(self.state.read_model().messages);
                let mut turn_pipeline = TurnBoundary::from_state_with_clock(
                    self.state.clone(),
                    Arc::clone(&self.host.core.clock),
                )
                .with_session_execution_lease(
                    session_execution_lease.map(SessionExecutionLeaseGuard::fence),
                );
                turn_pipeline.apply_prepared_messages(&messages);
                return self
                    .finish_turn(
                        TurnFinishInput {
                            turn_pipeline,
                            assembler,
                            new_messages: messages,
                            policy: RuntimeSessionPolicy::new(
                                self.state.effective_policy().clone(),
                                Default::default(),
                            ),
                            turn_index,
                            queued_work_claims: queued_claims,
                            turn_input_claims,
                            trace_turn_id,
                        },
                        events,
                        &scoped_effect_controller,
                        &cancel,
                        session_execution_lease,
                        session_execution_lease_release_policy,
                        &turn_control,
                    )
                    .await;
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
                self.host.core.clock.as_ref(),
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
        Box::pin(self.stream_prepared_turn_inner(
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
            queued_claims,
            turn_input_claims,
            session_execution_lease,
            session_execution_lease_release_policy,
        ))
        .await
    }

    /// Run one logical turn and return only its assembled terminal result.
    pub async fn run_turn_assembled(
        &mut self,
        input: TurnInput,
        cancel: CancellationToken,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<AssembledTurn, RuntimeError> {
        self.stream_turn(input, TurnOptions::new(cancel, scoped_effect_controller))
            .await
    }

    /// Run one logical turn using host-prepared message history.
    #[allow(clippy::too_many_arguments)]
    pub async fn stream_prepared_turn(
        &mut self,
        messages: crate::MessageSequence,
        previous_prompt_usage: Option<PromptUsage>,
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
        initial_turn_input_claim: Option<crate::TurnInputClaim>,
    ) -> Result<AssembledTurn, RuntimeError> {
        let stopwatch = TurnStopwatch::start(self.host.core.clock.as_ref());
        let session_execution_lease = self
            .claim_session_execution_lease(cancel.clone(), true)
            .await?;
        let result = Box::pin(self.drive_logical_turn(
            LogicalTurnStart::Prepared(PreparedLogicalTurn {
                messages,
                previous_prompt_usage,
                protocol_turn_options,
                protocol_extension,
                turn_context,
                initial_turn_causes,
                trace_turn_id,
                turn_index,
            }),
            events,
            turn_events,
            scoped_effect_controller,
            cancel,
            LogicalTurnClaims::new(
                initial_queue_claim.into_iter().collect(),
                initial_turn_input_claim.into_iter().collect(),
            ),
            session_execution_lease.as_ref(),
            stopwatch,
        ))
        .await
        .map(|run| {
            run.into_final_turn()
                .expect("logical turn always contains a terminal physical turn")
        });
        self.settle_session_execution_lease(session_execution_lease.as_ref(), result)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn stream_prepared_turn_inner(
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
        initial_queue_claims: Vec<crate::QueuedWorkClaim>,
        initial_turn_input_claims: Vec<crate::TurnInputClaim>,
        session_execution_lease: Option<&SessionExecutionLeaseGuard>,
        session_execution_lease_release_policy: SessionExecutionLeaseReleasePolicy,
    ) -> Result<PhysicalTurnExecution, RuntimeError> {
        let turn_control = Arc::new(
            ActiveTurnControl::new(
                scoped_effect_controller.controller(),
                TurnAddress::new(&self.state.session_id, &trace_turn_id),
            )
            .await?
            .with_local_cancel_origin(turn_context.local_cancel_origin_hint()),
        );
        if session_execution_lease.is_none()
            && self
                .session
                .as_ref()
                .and_then(|session| session.history_store())
                .is_some()
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::StoreCommitFailed,
                "prepared turn requires a session execution lease",
            ));
        }
        let session_execution_fence =
            session_execution_lease.map(SessionExecutionLeaseGuard::fence);
        let (event_tx, mut event_rx) = mpsc::channel::<RuntimeStreamEvent>(100);
        let child_usage_event_relay = ChildUsageEventRelay::new(event_tx.clone());
        let mut turn_policy = self.state.effective_policy().clone();
        let turn_provider_override = turn_context.provider().cloned();
        if let Some(provider) = turn_provider_override.as_ref() {
            turn_policy.provider_id = provider.kind().to_string();
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

            let mut turn_pipeline = TurnBoundary::from_state_with_clock(
                self.state.clone(),
                Arc::clone(&self.host.core.clock),
            )
            .with_session_execution_lease(session_execution_fence.clone());
            turn_pipeline.apply_prepared_messages(&prepared.messages);
            let issue = TurnIssue {
                kind: "plugin".to_string(),
                code: Some(abort.code),
                terminal_reason: None,
                message: abort.message.clone(),
                raw: None,
                retryable: None,
                provider_failure_kind: None,
            };
            let error_event = SessionStreamEvent::Error {
                message: abort.message,
                envelope: Some(crate::session_model::ErrorEnvelope {
                    kind: "plugin".to_string(),
                    code: issue.code.clone(),
                    terminal_reason: None,
                    user_message: issue.message.clone(),
                    raw: None,
                    retryable: None,
                    provider_failure_kind: None,
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
            let outcome_event = SessionStreamEvent::TurnOutcome {
                outcome: TurnOutcome::Stopped(TurnStop::PluginAbort),
            };
            assembler.push(&outcome_event);
            emit_session_event_to_sink(events, outcome_event).await;
            assembler.push(&SessionStreamEvent::Done);
            emit_session_event_to_sink(events, SessionStreamEvent::Done).await;
            return self
                .finish_turn(
                    TurnFinishInput {
                        turn_pipeline,
                        assembler,
                        new_messages: prepared.messages,
                        policy: RuntimeSessionPolicy::new(
                            self.state.effective_policy().clone(),
                            Default::default(),
                        ),
                        turn_index,
                        queued_work_claims: initial_queue_claims,
                        turn_input_claims: initial_turn_input_claims,
                        trace_turn_id,
                    },
                    events,
                    &scoped_effect_controller,
                    &cancel,
                    session_execution_lease,
                    session_execution_lease_release_policy,
                    turn_control.as_ref(),
                )
                .await;
        }
        let mut turn_pipeline = TurnBoundary::from_state_with_clock(
            self.state.clone(),
            Arc::clone(&self.host.core.clock),
        )
        .with_session_execution_lease(session_execution_fence.clone());
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
            .map_err(super::runtime_error_from_store_commit)?;
        let resolved_turn_policy = if let Some(provider) = turn_provider_override {
            RuntimeSessionPolicy::from_provider(
                turn_policy.clone(),
                provider.with_clock(Arc::clone(&self.host.core.clock)),
            )
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
        let shared_cancel_controller = match scoped_effect_controller.shared_controller() {
            Some(controller) => Some(controller),
            None if scoped_effect_controller.controller().durability_tier()
                == crate::DurabilityTier::Durable
                && self.host.core.control.effect_host.durability_tier()
                    == crate::DurabilityTier::Durable =>
            {
                self.host
                    .core
                    .control
                    .effect_host
                    .scoped_static(scoped_effect_controller.execution_scope().clone())?
                    .and_then(|scoped| scoped.shared_controller())
            }
            None => None,
        };
        let session = self
            .session
            .take()
            .expect("lash runtime session must be available");
        let mut driver = Box::new(RuntimeTurnDriver {
            session,
            policy: resolved_turn_policy,
            host: self.host.clone(),
            turn_id: scoped_effect_controller.scope_id().to_string(),
            scoped_effect_controller,
            session_id: self.state.session_id.clone(),
            turn_index,
            turn_pipeline,
            llm_stream_summaries: HashMap::new(),
            llm_calls: Vec::new(),
            next_llm_ordinal: 0,
            session_services: manager,
            protocol_turn_options: effective_protocol_turn_options,
            protocol_extension,
            turn_context,
            turn_causes: initial_turn_causes,
            pending_queue_claims: initial_queue_claims,
            pending_turn_input_claims: initial_turn_input_claims,
            checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
            session_execution_lease: session_execution_fence,
            runtime_lease_owner: self.runtime_lease_owner.clone(),
            turn_phase_probe: self.turn_phase_probe.clone(),
        });
        let protocol_run_offset = 0;
        let cancellation_messages = prepared.messages.clone();
        self.mark_phase_begin(RuntimeTurnPhase::EffectLoop);
        let run_result = Box::pin(run_turn_effect_loop(
            &mut driver,
            prepared.messages,
            event_tx,
            cancel.clone(),
            protocol_run_offset,
            Arc::clone(&turn_control),
            shared_cancel_controller,
            finish_scoped_effect_controller.controller(),
            &mut event_rx,
            &mut assembler,
            &child_usage_event_relay,
            events,
            turn_events,
        ))
        .await;
        let (new_messages, _new_protocol_iteration) = match run_result {
            Ok(result) => result,
            Err(err) if cancel.is_cancelled() => {
                if turn_control.evidence().is_none() {
                    turn_control
                        .observe_pending_cancel(finish_scoped_effect_controller.controller())
                        .await?;
                }
                if turn_control.evidence().is_some() {
                    self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
                    return Box::pin(self.finish_cancelled_turn_after_effect_abort(
                        *driver,
                        assembler,
                        cancellation_messages,
                        events,
                        &finish_scoped_effect_controller,
                        &cancel,
                        session_execution_lease,
                        session_execution_lease_release_policy,
                        turn_control.as_ref(),
                        turn_index,
                        trace_turn_id,
                    ))
                    .await;
                }
                self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
                let RuntimeTurnDriver {
                    session,
                    pending_queue_claims,
                    pending_turn_input_claims,
                    ..
                } = *driver;
                self.session = Some(session);
                self.abandon_queued_work_claims_after_lease_loss(&err, &pending_queue_claims)
                    .await;
                self.abandon_turn_input_claims_after_lease_loss(&err, &pending_turn_input_claims)
                    .await;
                return Err(err);
            }
            Err(err) => {
                self.mark_phase_end(RuntimeTurnPhase::EffectLoop);
                let RuntimeTurnDriver {
                    session,
                    pending_queue_claims,
                    pending_turn_input_claims,
                    ..
                } = *driver;
                self.session = Some(session);
                self.abandon_queued_work_claims_after_lease_loss(&err, &pending_queue_claims)
                    .await;
                self.abandon_turn_input_claims_after_lease_loss(&err, &pending_turn_input_claims)
                    .await;
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
            llm_calls,
            pending_queue_claims,
            pending_turn_input_claims,
            ..
        } = *driver;
        self.session = Some(session);
        let pending_queue_claims_for_abandon = pending_queue_claims.clone();
        let pending_turn_input_claims_for_abandon = pending_turn_input_claims.clone();
        let finish_result = Box::pin(self.finish_turn(
            TurnFinishInput {
                turn_pipeline,
                assembler: assembler.with_llm_calls(llm_calls),
                new_messages,
                policy,
                turn_index,
                queued_work_claims: pending_queue_claims,
                turn_input_claims: pending_turn_input_claims,
                trace_turn_id,
            },
            events,
            &finish_scoped_effect_controller,
            &cancel_state,
            session_execution_lease,
            session_execution_lease_release_policy,
            turn_control.as_ref(),
        ))
        .await;
        if let Err(err) = &finish_result {
            self.abandon_queued_work_claims_after_lease_loss(
                err,
                &pending_queue_claims_for_abandon,
            )
            .await;
            self.abandon_turn_input_claims_after_lease_loss(
                err,
                &pending_turn_input_claims_for_abandon,
            )
            .await;
        }
        finish_result
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

async fn publish_terminal_after_commit(
    turn_control: &ActiveTurnControl,
    resolver: &dyn AwaitEventResolver,
    terminal: &TurnTerminal,
    session_id: &str,
    turn_id: &str,
) {
    if let Err(err) = turn_control.publish_terminal(resolver, terminal).await {
        tracing::warn!(
            error = %err,
            session_id,
            turn_id,
            "turn committed but terminal publication failed"
        );
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_turn_effect_loop(
    driver: &mut RuntimeTurnDriver<'_>,
    messages: crate::MessageSequence,
    event_tx: mpsc::Sender<RuntimeStreamEvent>,
    cancellation: CancellationToken,
    protocol_run_offset: usize,
    turn_control: Arc<ActiveTurnControl>,
    shared_cancel_controller: Option<Arc<dyn RuntimeEffectController>>,
    cancel_controller: &dyn RuntimeEffectController,
    event_rx: &mut mpsc::Receiver<RuntimeStreamEvent>,
    assembler: &mut TurnAssembler,
    child_usage_event_relay: &ChildUsageEventRelay,
    events: &dyn EventSink,
    turn_events: &dyn TurnActivitySink,
) -> Result<(crate::MessageSequence, usize), RuntimeError> {
    // The start gate can change the handler's control flow before its first
    // effect, so durable runtimes must observe it through the handler-scoped
    // controller. That controller journals the observation and replays the
    // same answer after an owner crash. The shared controller is intentionally
    // reserved for the concurrent live watcher below: an out-of-band peek here
    // could observe a cancel that arrived after the original attempt and make
    // a replay take a different command path.
    if await_turn_cancellation_start_gate(|| turn_control.observe_pending_cancel(cancel_controller))
        .await?
        .is_some()
    {
        cancellation.cancel();
    }
    let cancel_watcher = shared_cancel_controller.map(|controller| {
        let turn_control = Arc::clone(&turn_control);
        let cancellation = cancellation.clone();
        tokio::spawn(async move {
            if await_turn_cancellation_with_retry(|| {
                turn_control.await_cancel(controller.as_ref(), CancellationToken::new())
            })
            .await
            .is_some()
            {
                cancellation.cancel();
            }
        })
    });
    let run_future = Box::pin(driver.run(
        messages,
        event_tx,
        cancellation.clone(),
        protocol_run_offset,
    ));
    let result = if cancel_watcher.is_some() {
        drive_turn_to_completion(
            run_future,
            event_rx,
            assembler,
            child_usage_event_relay,
            events,
            turn_events,
        )
        .await
    } else {
        drive_turn_to_completion_with_cancel(
            run_future,
            await_turn_cancellation_with_retry(|| {
                turn_control.await_cancel(cancel_controller, CancellationToken::new())
            }),
            cancellation,
            event_rx,
            assembler,
            child_usage_event_relay,
            events,
            turn_events,
        )
        .await
    };
    if let Some(watcher) = cancel_watcher {
        watcher.abort();
    }
    result
}

const TURN_CANCEL_WATCH_RETRY_INITIAL: std::time::Duration = std::time::Duration::from_millis(25);
const TURN_CANCEL_WATCH_RETRY_MAX: std::time::Duration = std::time::Duration::from_secs(1);
/// Keep the journaled turn-start observation bounded so a broken peek cannot
/// pin one Restate invocation forever. Exhaustion fails closed: the error
/// propagates and the turn fails without starting any effect (hosts classify
/// it as non-retryable, so the invocation retires as a failed turn). Transient
/// transport trouble does not reach this bound — a slow journaled peek stays
/// pending inside one attempt; only genuine terminal errors (revoked or
/// unknown keys) burn attempts.
const TURN_CANCEL_START_GATE_ATTEMPTS: usize = 3;

async fn await_turn_cancellation_start_gate<F, C>(
    mut watch: F,
) -> Result<Option<TurnCancellationEvidence>, RuntimeError>
where
    F: FnMut() -> C,
    C: std::future::Future<Output = Result<Option<TurnCancellationEvidence>, RuntimeError>>,
{
    let mut backoff = TURN_CANCEL_WATCH_RETRY_INITIAL;
    for attempt in 1..=TURN_CANCEL_START_GATE_ATTEMPTS {
        match watch().await {
            Ok(observation) => return Ok(observation),
            Err(err) if attempt == TURN_CANCEL_START_GATE_ATTEMPTS => return Err(err),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    attempt,
                    max_attempts = TURN_CANCEL_START_GATE_ATTEMPTS,
                    retry_after_ms = backoff.as_millis(),
                    "turn cancellation start gate failed; retrying before failing the invocation"
                );
                tokio::time::sleep(backoff).await;
                backoff = backoff.saturating_mul(2).min(TURN_CANCEL_WATCH_RETRY_MAX);
            }
        }
    }
    unreachable!("positive start-gate attempt limit")
}

async fn await_turn_cancellation_with_retry<F, C>(mut watch: F) -> Option<TurnCancellationEvidence>
where
    F: FnMut() -> C,
    C: std::future::Future<Output = Result<Option<TurnCancellationEvidence>, RuntimeError>>,
{
    let mut backoff = TURN_CANCEL_WATCH_RETRY_INITIAL;
    loop {
        match watch().await {
            Ok(observation) => return observation,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    retry_after_ms = backoff.as_millis(),
                    "turn cancellation watcher failed; retrying while the turn remains active"
                );
                tokio::time::sleep(backoff).await;
                backoff = backoff.saturating_mul(2).min(TURN_CANCEL_WATCH_RETRY_MAX);
            }
        }
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
                // Some durable adapter futures are not fused. Once turn
                // completion is ready, select it before another ready branch
                // so the loop never polls the completed future again.
                biased;

                completed = run_future.as_mut() => {
                    child_usage_event_relay.clear();
                    break completed;
                }
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
            }
        }
    };
    while let Some(event) = event_rx.recv().await {
        emit_runtime_stream_event_to_sinks(events, turn_events, event, assembler).await;
    }
    run_result
}

#[allow(clippy::too_many_arguments)]
async fn drive_turn_to_completion_with_cancel<F, C>(
    run_future: F,
    cancel_future: C,
    cancellation: CancellationToken,
    event_rx: &mut mpsc::Receiver<RuntimeStreamEvent>,
    assembler: &mut TurnAssembler,
    child_usage_event_relay: &ChildUsageEventRelay,
    events: &dyn EventSink,
    turn_events: &dyn TurnActivitySink,
) -> Result<(crate::MessageSequence, usize), RuntimeError>
where
    F: std::future::Future<Output = Result<(crate::MessageSequence, usize), RuntimeError>>,
    C: std::future::Future<Output = Option<TurnCancellationEvidence>>,
{
    let run_result = {
        let mut run_future = Box::pin(run_future);
        let mut cancel_future = Box::pin(cancel_future);
        let mut cancellation_observed = false;
        loop {
            tokio::select! {
                // Keep the non-fused turn future from being re-polled when
                // cancellation or a stream event becomes ready alongside it.
                biased;

                completed = run_future.as_mut() => {
                    child_usage_event_relay.clear();
                    break completed;
                }
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
                observation = cancel_future.as_mut(), if !cancellation_observed => {
                    cancellation_observed = true;
                    if observation.is_some() {
                        cancellation.cancel();
                    }
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
            emit_turn_activity_to_sink(turn_events, activity).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        ActiveTurnControl, TURN_CANCEL_START_GATE_ATTEMPTS, agent_frame_follow_turn_id,
        await_turn_cancellation_start_gate, await_turn_cancellation_with_retry,
        publish_terminal_after_commit,
    };
    use crate::{
        AwaitEventKey, AwaitEventResolver, Resolution, ResolveOutcome, RuntimeError, TurnAddress,
        TurnCancellationEvidence, TurnFinish, TurnOutcome, TurnTerminal,
    };

    #[derive(Default)]
    struct RejectTerminalPublication {
        attempts: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl AwaitEventResolver for RejectTerminalPublication {
        async fn resolve_await_event(
            &self,
            _key: &AwaitEventKey,
            _resolution: Resolution,
        ) -> Result<ResolveOutcome, RuntimeError> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Err(RuntimeError::new(
                "transient_terminal_publication",
                "terminal backend unavailable",
            ))
        }
    }

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

    #[tokio::test]
    async fn cancellation_watch_retries_transient_errors_until_evidence_arrives() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed_attempts = Arc::clone(&attempts);
        let evidence = await_turn_cancellation_with_retry(move || {
            let attempt = observed_attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if attempt < 2 {
                    Err(RuntimeError::new(
                        "transient_cancel_watch",
                        "temporary ingress failure",
                    ))
                } else {
                    Ok(Some(TurnCancellationEvidence {
                        request_id: "retry-request".to_string(),
                        origin: Some("test-user".to_string()),
                        reason: None,
                    }))
                }
            }
        })
        .await
        .expect("cancellation evidence after retries");

        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(evidence.request_id, "retry-request");
    }

    #[tokio::test]
    async fn cancellation_start_gate_fails_after_bounded_retries() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed_attempts = Arc::clone(&attempts);
        let err = await_turn_cancellation_start_gate(move || {
            observed_attempts.fetch_add(1, Ordering::SeqCst);
            async {
                Err(RuntimeError::new(
                    "cancel_start_gate_unavailable",
                    "temporary ingress failure",
                ))
            }
        })
        .await
        .expect_err("start gate must fail closed after its retry budget");

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            TURN_CANCEL_START_GATE_ATTEMPTS
        );
        assert_eq!(err.code.to_string(), "cancel_start_gate_unavailable");
    }

    #[tokio::test]
    async fn terminal_publication_failure_is_non_fatal_after_commit() {
        let resolver = RejectTerminalPublication::default();
        let control = ActiveTurnControl::new(
            &resolver,
            TurnAddress::new("committed-session", "committed-turn"),
        )
        .await
        .expect("active turn control");
        publish_terminal_after_commit(
            &control,
            &resolver,
            &TurnTerminal::Committed {
                outcome: TurnOutcome::Finished(TurnFinish::AssistantMessage {
                    text: "committed".to_string(),
                }),
                cancellation: None,
                session_revision: Some(1),
            },
            "committed-session",
            "committed-turn",
        )
        .await;
        assert_eq!(resolver.attempts.load(Ordering::SeqCst), 1);
    }
}
