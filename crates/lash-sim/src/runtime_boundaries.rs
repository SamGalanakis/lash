use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lash_core::runtime::{
    QueuedWorkBatchDraft, QueuedWorkClaim, QueuedWorkClaimBoundary, QueuedWorkPayload, RuntimeReplay,
    RuntimeScope, RuntimeSubject,
};
use lash_core::{
    DeliveryPolicy, ExecResponse, ExecutionScope, LeaseOwnerIdentity, LeaseOwnerLiveness, MergeKey,
    PreparedToolCall, RuntimeEffectCommand, RuntimeEffectController, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectLocalExecutor, RuntimeEffectOutcome, RuntimeInvocation,
    RuntimePersistence, SessionExecutionLeaseClaimOutcome, SessionRelation, SessionScope,
    SessionStoreCreateRequest, SessionStoreFactory, SlotPolicy, ToolAttemptLaunch, ToolCallOutput,
    ToolCallRecord, ToolId,
};
use serde_json::{Value, json};

use crate::scheduler::{BoundaryEvent, BoundaryKind};
use crate::trace::value_digest;

const EFFECT_SCOPE_ID: &str = "lash-sim-runtime-boundaries";
const LEASE_TTL_MS: u64 = 30_000;

#[derive(Clone)]
pub enum RuntimeEffectReplayStore {
    Memory,
    SqliteFile(PathBuf),
    Postgres(Arc<lash_postgres_store::PostgresStorage>),
}

impl RuntimeEffectReplayStore {
    pub fn sqlite_file(path: impl Into<PathBuf>) -> Self {
        Self::SqliteFile(path.into())
    }

    pub fn postgres(storage: Arc<lash_postgres_store::PostgresStorage>) -> Self {
        Self::Postgres(storage)
    }

    fn controller_name(&self) -> &'static str {
        match self {
            Self::Memory | Self::SqliteFile(_) => "sqlite_runtime_effect_controller",
            Self::Postgres(_) => "postgres_runtime_effect_controller",
        }
    }
}

impl fmt::Debug for RuntimeEffectReplayStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory => f.write_str("Memory"),
            Self::SqliteFile(path) => f.debug_tuple("SqliteFile").field(path).finish(),
            Self::Postgres(_) => f.write_str("Postgres"),
        }
    }
}

#[derive(Debug)]
pub struct RuntimeBoundaryError {
    message: String,
}

impl RuntimeBoundaryError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RuntimeBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeBoundaryError {}

#[derive(Clone, Debug)]
struct DurableEntry {
    result_digest: String,
    execution_count: usize,
    replay_count: usize,
}

pub struct RuntimeBoundaryHarness {
    store_factory: Arc<dyn SessionStoreFactory>,
    effect_replay_store: RuntimeEffectReplayStore,
    effect_controller: Option<Arc<dyn RuntimeEffectController>>,
    durable_entries: BTreeMap<String, DurableEntry>,
    process_wake_claimed_batches: BTreeSet<String>,
}

impl RuntimeBoundaryHarness {
    pub fn new(
        store_factory: Arc<dyn SessionStoreFactory>,
        effect_replay_store: RuntimeEffectReplayStore,
    ) -> Self {
        Self {
            store_factory,
            effect_replay_store,
            effect_controller: None,
            durable_entries: BTreeMap::new(),
            process_wake_claimed_batches: BTreeSet::new(),
        }
    }

    /// Read a real session-execution-lease fencing token from the store for a
    /// per-session probe scope. Each reading claims the probe lease and then
    /// releases it; because the in-memory store keeps the per-scope fencing
    /// counter across a release, the very next claim acquires a strictly higher
    /// token. The returned token is the ground truth the lease-time-monotonic
    /// oracle checks — unlike a generator-fed tick, it can actually regress if
    /// the lease store's fencing is broken. Deterministic: a single owner on a
    /// dedicated per-session scope, no wall-clock reads.
    pub async fn lease_probe_fencing_token(
        &mut self,
        session: &str,
    ) -> Result<u64, RuntimeBoundaryError> {
        let probe_scope = format!("{session}::lease-probe");
        let store = self.store_for_session(&probe_scope).await?;
        let owner = LeaseOwnerIdentity::opaque(
            "lash-sim-lease-probe",
            format!("{probe_scope}:probe-owner"),
        );
        let lease = match store
            .try_claim_session_execution_lease(&probe_scope, &owner, LEASE_TTL_MS)
            .await
            .map_err(|err| RuntimeBoundaryError::new(format!("claim lease probe failed: {err}")))?
        {
            SessionExecutionLeaseClaimOutcome::Acquired(lease) => lease,
            SessionExecutionLeaseClaimOutcome::Busy { holder } => {
                return Err(RuntimeBoundaryError::new(format!(
                    "lease probe claim unexpectedly busy for `{probe_scope}`; holder fence={}",
                    holder.fencing_token
                )));
            }
        };
        let fencing_token = lease.fencing_token;
        // Release so the next probe re-acquires at a higher fencing token rather
        // than renewing the same one (a same-owner live claim would not advance).
        store
            .release_session_execution_lease(&lease.completion())
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("release lease probe failed: {err}"))
            })?;
        Ok(fencing_token)
    }

    pub async fn deliver(&mut self, event: &BoundaryEvent) -> Result<Value, RuntimeBoundaryError> {
        match event.kind {
            BoundaryKind::Tool => self.complete_tool(event).await,
            BoundaryKind::ExecCode => self.execute_code(event).await,
            BoundaryKind::DurableEffect => self.complete_durable_effect(event).await,
            BoundaryKind::ProcessWake => self.deliver_process_wake(event).await,
            BoundaryKind::Worker => self.run_worker_stale_completion(event).await,
            kind => Err(RuntimeBoundaryError::new(format!(
                "runtime boundary harness does not own {}",
                boundary_kind_name(kind)
            ))),
        }
    }

    pub async fn complete_durable_effect(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, RuntimeBoundaryError> {
        let durable_key = event
            .payload
            .get("durable_key")
            .and_then(Value::as_str)
            .unwrap_or(&event.boundary_id)
            .to_string();
        let requested_result = event
            .payload
            .get("result")
            .cloned()
            .unwrap_or_else(|| json!({"completed": true}));
        let effect_id = event
            .payload
            .get("runtime_effect")
            .and_then(|value| value.get("effect_id"))
            .and_then(Value::as_str)
            .unwrap_or(&event.boundary_id)
            .to_string();
        let envelope = RuntimeEffectEnvelope::new(
            RuntimeInvocation::effect(
                RuntimeScope::new(event.actor_alias.clone()),
                effect_id.clone(),
                RuntimeEffectKind::DurableStep,
                durable_key.clone(),
            ),
            RuntimeEffectCommand::DurableStep {
                step_id: effect_id.clone(),
                input: json!({
                    "durable_key": durable_key,
                    "session": event.actor_alias,
                }),
            },
        );
        let envelope_hash = envelope.stable_hash().map_err(|err| {
            RuntimeBoundaryError::new(format!("durable effect envelope hash failed: {err}"))
        })?;
        let local_calls = Arc::new(AtomicUsize::new(0));
        let local_calls_for_executor = Arc::clone(&local_calls);
        let scripted_result = requested_result.clone();
        let controller = self.ensure_effect_controller().await?;
        let outcome = controller
            .execute_effect(
                envelope.clone(),
                RuntimeEffectLocalExecutor::durable_step(move |_| async move {
                    local_calls_for_executor.fetch_add(1, Ordering::SeqCst);
                    Ok(scripted_result)
                }),
            )
            .await
            .map_err(|err| RuntimeBoundaryError::new(format!("durable effect failed: {err}")))?;
        let RuntimeEffectOutcome::DurableStep { value } = &outcome else {
            return Err(RuntimeBoundaryError::new(
                "durable effect controller returned non-durable-step outcome",
            ));
        };
        let local_execution_count = local_calls.load(Ordering::SeqCst);
        let replayed = local_execution_count == 0;
        let result_digest = value_digest(value);
        let entry = self
            .durable_entries
            .entry(durable_key.clone())
            .or_insert_with(|| DurableEntry {
                result_digest: result_digest.clone(),
                execution_count: usize::from(local_execution_count > 0),
                replay_count: 0,
            });
        if replayed {
            entry.replay_count += 1;
        } else if entry.execution_count == 0 {
            entry.execution_count = 1;
        }
        Ok(json!({
            "durable_key": durable_key,
            "result_digest": entry.result_digest,
            "execution_count": entry.execution_count,
            "replay_count": entry.replay_count,
            "replayed": replayed,
            "runtime_effect": {
                "kind": RuntimeEffectKind::DurableStep.as_str(),
                "effect_id": effect_id,
                "replay_key": envelope.invocation.replay_key(),
                "envelope_hash": envelope_hash,
                "controller": self.effect_replay_store.controller_name(),
                "local_executor_called": local_execution_count > 0,
            },
            "runtime_effect_outcome": outcome,
        }))
    }

    pub async fn complete_tool(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, RuntimeBoundaryError> {
        let tool_name = event
            .payload
            .get("tool")
            .and_then(Value::as_str)
            .unwrap_or("sim_tool")
            .to_string();
        let output = event
            .payload
            .get("output")
            .cloned()
            .unwrap_or_else(|| json!(""));
        let args = json!({
            "boundary_id": event.boundary_id,
            "session": event.actor_alias,
        });
        let call = PreparedToolCall::from_parts(
            event.boundary_id.clone(),
            ToolId::from(format!("tool:{tool_name}")),
            tool_name.clone(),
            args.clone(),
            None,
            json!({"prepared_by": "lash-sim"}),
        );
        let envelope = RuntimeEffectEnvelope::new(
            RuntimeInvocation::effect(
                RuntimeScope::new(event.actor_alias.clone()),
                format!("tool-attempt:{}", event.boundary_id),
                RuntimeEffectKind::ToolAttempt,
                format!("tool/{}/{}", event.actor_alias, event.boundary_id),
            ),
            RuntimeEffectCommand::ToolAttempt {
                call,
                execution_grant: None,
                attempt: 1,
                max_attempts: 1,
            },
        );
        let local_calls = Arc::new(AtomicUsize::new(0));
        let local_calls_for_executor = Arc::clone(&local_calls);
        let output_for_executor = output.clone();
        let tool_name_for_executor = tool_name.clone();
        let call_id_for_executor = event.boundary_id.clone();
        let controller = self.ensure_effect_controller().await?;
        let outcome = controller
            .execute_effect(
                envelope,
                RuntimeEffectLocalExecutor::testing(move |_| async move {
                    local_calls_for_executor.fetch_add(1, Ordering::SeqCst);
                    Ok(RuntimeEffectOutcome::ToolAttempt {
                        launch: ToolAttemptLaunch::Done {
                            record: ToolCallRecord {
                                call_id: Some(call_id_for_executor),
                                tool: tool_name_for_executor,
                                args,
                                output: ToolCallOutput::success(output_for_executor),
                                duration_ms: 0,
                            },
                        },
                        triggers: Vec::new(),
                    })
                }),
            )
            .await
            .map_err(|err| RuntimeBoundaryError::new(format!("tool effect failed: {err}")))?;
        let RuntimeEffectOutcome::ToolAttempt { launch, .. } = outcome else {
            return Err(RuntimeBoundaryError::new(
                "tool controller returned non-tool-attempt outcome",
            ));
        };
        let ToolAttemptLaunch::Done { record } = launch else {
            return Err(RuntimeBoundaryError::new(
                "sim tool boundary unexpectedly returned pending tool launch",
            ));
        };
        let execution_count = local_calls.load(Ordering::SeqCst);
        Ok(json!({
            "session": event.actor_alias,
            "tool_output": output,
            "tool_name": tool_name,
            "tool_call_id": event.boundary_id,
            "execution_count": execution_count,
            "runtime_tool_output": record.output,
            "runtime_tool_record": record,
            "runtime_effect": {
                "kind": RuntimeEffectKind::ToolAttempt.as_str(),
                "controller": self.effect_replay_store.controller_name(),
                "local_executor_called": execution_count > 0,
            },
        }))
    }

    pub async fn execute_code(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, RuntimeBoundaryError> {
        let output = event
            .payload
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let exit_code = event
            .payload
            .get("exit_code")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let code = format!("sim_exec('{}')", event.boundary_id);
        let envelope = RuntimeEffectEnvelope::new(
            RuntimeInvocation::effect(
                RuntimeScope::new(event.actor_alias.clone()),
                format!("exec-code:{}", event.boundary_id),
                RuntimeEffectKind::ExecCode,
                format!("exec/{}/{}", event.actor_alias, event.boundary_id),
            ),
            RuntimeEffectCommand::ExecCode {
                language: "lash-sim-script".to_string(),
                code,
            },
        );
        let response = ExecResponse {
            observations: vec![output.clone()],
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: (exit_code != 0).then(|| format!("exit code {exit_code}")),
            duration_ms: 0,
            terminal_finish: Some(json!({
                "output": output,
                "exit_code": exit_code,
            })),
        };
        let local_calls = Arc::new(AtomicUsize::new(0));
        let local_calls_for_executor = Arc::clone(&local_calls);
        let response_for_executor = response.clone();
        let controller = self.ensure_effect_controller().await?;
        let outcome = controller
            .execute_effect(
                envelope,
                RuntimeEffectLocalExecutor::testing(move |_| async move {
                    local_calls_for_executor.fetch_add(1, Ordering::SeqCst);
                    Ok(RuntimeEffectOutcome::ExecCode {
                        result: Ok(response_for_executor),
                    })
                }),
            )
            .await
            .map_err(|err| RuntimeBoundaryError::new(format!("exec-code effect failed: {err}")))?;
        let execution_count = local_calls.load(Ordering::SeqCst);
        Ok(json!({
            "session": event.actor_alias,
            "exec_output": output,
            "exit_code": exit_code,
            "execution_count": execution_count,
            "runtime_effect_outcome": outcome,
            "runtime_effect": {
                "kind": RuntimeEffectKind::ExecCode.as_str(),
                "controller": self.effect_replay_store.controller_name(),
                "local_executor_called": execution_count > 0,
            },
        }))
    }

    pub async fn deliver_process_wake(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, RuntimeBoundaryError> {
        let session = boundary_session_alias(event);
        let process_id = format!("sim-process-{}", event.boundary_id.replace(':', "-"));
        let dedupe_key = event
            .payload
            .get("dedupe_key")
            .and_then(Value::as_str)
            .unwrap_or(&event.boundary_id)
            .to_string();
        let wake = lash_core::process_wake_delivery(lash_core::ProcessWakeDeliveryRequest {
            target_scope: SessionScope::new(session.clone()),
            process_id: process_id.clone(),
            sequence: 1,
            event_type: "process.wake".to_string(),
            event_invocation: RuntimeInvocation {
                scope: RuntimeScope::new(session.clone()),
                subject: RuntimeSubject::ProcessEvent {
                    process_id: process_id.clone(),
                    sequence: 1,
                    event_type: "process.wake".to_string(),
                },
                caused_by: None,
                replay: Some(RuntimeReplay {
                    key: format!("process:{process_id}:wake:{dedupe_key}"),
                }),
            },
            process_caused_by: None,
            wake: lash_core::ProcessWake {
                input: format!("wake for {session}"),
                dedupe_key: dedupe_key.clone(),
            },
            occurred_at: std::time::UNIX_EPOCH + std::time::Duration::from_millis(event.at),
        })
        .map_err(|err| RuntimeBoundaryError::new(format!("process wake failed: {err}")))?;
        let store = self.store_for_session(&session).await?;
        let owner = LeaseOwnerIdentity::opaque(
            "lash-sim-process-wake-driver",
            format!("{}:process-wake-driver", session),
        );
        let lease = match store
            .try_claim_session_execution_lease(&session, &owner, LEASE_TTL_MS)
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("claim session lease failed: {err}"))
            })? {
            SessionExecutionLeaseClaimOutcome::Acquired(lease) => lease,
            SessionExecutionLeaseClaimOutcome::Busy { holder } => {
                return Ok(json!({
                    "session": session,
                    "process_wake": true,
                    "process_id": process_id,
                    "sequence": wake.sequence,
                    "wake_id": wake.wake_id,
                    "dedupe_key": wake.dedupe_key,
                    "claimed_once": false,
                    "lease_busy": true,
                    "busy_holder": owner_json(&holder.owner),
                    "runtime_process_wake": wake,
                    "runtime_queued_work": {
                        "source_key": dedupe_key,
                        "work_class": "ProcessWake",
                        "enqueued": false,
                        "claimed": false,
                        "claim_id_present": false,
                    },
                }));
            }
        };
        let batch = store
            .enqueue_queued_work(
                QueuedWorkBatchDraft::new(
                    session.clone(),
                    DeliveryPolicy::EarliestSafeBoundary,
                    SlotPolicy::Exclusive,
                    vec![QueuedWorkPayload::process_wake(wake.clone())],
                )
                .with_source_key(dedupe_key.clone())
                .with_merge_key(MergeKey::Never),
            )
            .await
            .map_err(|err| RuntimeBoundaryError::new(format!("enqueue wake failed: {err}")))?;
        let claim = store
            .claim_ready_queued_work_by_batch_ids(
                &session,
                &lease.fence(),
                &owner,
                QueuedWorkClaimBoundary::Idle,
                LEASE_TTL_MS,
                std::slice::from_ref(&batch.batch_id),
            )
            .await
            .map_err(|err| RuntimeBoundaryError::new(format!("claim queued wake failed: {err}")))?;
        store
            .release_session_execution_lease(&lease.completion())
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!(
                    "release process wake session lease failed: {err}"
                ))
            })?;
        let claimed_once = claim.is_some();
        if claimed_once {
            self.process_wake_claimed_batches
                .insert(batch.batch_id.clone());
        }
        Ok(json!({
            "session": session,
            "process_wake": true,
            "process_id": process_id,
            "sequence": wake.sequence,
            "wake_id": wake.wake_id,
            "dedupe_key": wake.dedupe_key,
            "claimed_once": claimed_once,
            "runtime_process_wake": wake,
            "runtime_queued_work": {
                "source_key": batch.source_key,
                "work_class": batch.work_class().map(|class| format!("{class:?}")),
                "claimed": claimed_once,
                "claimed_batch_count": claim.as_ref().map_or(0, |claim| claim.batches.len()),
                "claim_fencing_token": claim.as_ref().map(|claim| claim.fencing_token),
                "batch_id_present": !batch.batch_id.is_empty(),
                "claim_id_present": claim.as_ref().is_some_and(|claim| !claim.claim_id.is_empty()),
            },
        }))
    }

    pub async fn run_worker_stale_completion(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, RuntimeBoundaryError> {
        let session = boundary_session_alias(event);
        let store = self.store_for_session(&session).await?;
        let stale_owner = LeaseOwnerIdentity {
            owner_id: event.actor_alias.clone(),
            incarnation_id: format!("{}:incarnation-001", event.actor_alias),
            liveness: LeaseOwnerLiveness::local_process_for_test(
                "lash-sim-host",
                "lash-sim-boot",
                u32::MAX,
                "dead-process",
            ),
        };
        let live_owner = LeaseOwnerIdentity {
            owner_id: event.actor_alias.clone(),
            incarnation_id: format!("{}:incarnation-002", event.actor_alias),
            liveness: LeaseOwnerLiveness::local_process_for_test(
                "lash-sim-host",
                "lash-sim-boot",
                std::process::id(),
                "live-process",
            ),
        };
        // Worker one (the doomed incarnation) acquires the session execution lease
        // and starts a real unit of worker-owned queued work.
        let stale_lease = match store
            .try_claim_session_execution_lease(&session, &stale_owner, LEASE_TTL_MS)
            .await
            .map_err(|err| RuntimeBoundaryError::new(format!("claim stale lease failed: {err}")))?
        {
            SessionExecutionLeaseClaimOutcome::Acquired(lease) => lease,
            SessionExecutionLeaseClaimOutcome::Busy { .. } => {
                return Ok(json!({
                    "worker_alias": event.actor_alias,
                    "session": session,
                    "initial_owner": owner_json(&stale_owner),
                    "active_owner": owner_json(&live_owner),
                    "stale_completion_rejected": false,
                    "lease_busy": true,
                    "runtime_worker_store": {
                        "session_execution_lease_reclaimed": false,
                        "stale_completion_left_live_lease_renewable": false,
                        "busy_during_in_flight_turn": true,
                    },
                }));
            }
        };
        let work = self
            .start_worker_owned_work(store.as_ref(), &session, &stale_owner, &stale_lease, event.at)
            .await?;

        // Worker one crashes mid-flight: worker two fences it out by reclaiming
        // the session execution lease at a higher fencing token.
        let live_lease = match store
            .reclaim_session_execution_lease(
                &session,
                &live_owner,
                &stale_lease.fence(),
                LEASE_TTL_MS,
            )
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("reclaim worker lease failed: {err}"))
            })? {
            SessionExecutionLeaseClaimOutcome::Acquired(lease) => lease,
            SessionExecutionLeaseClaimOutcome::Busy { holder } => {
                return Err(RuntimeBoundaryError::new(format!(
                    "worker lease reclaim unexpectedly busy for `{session}`; holder={}",
                    holder.owner.owner_id
                )));
            }
        };

        // Worker two resumes the crashed worker's in-flight work under its own
        // lease and rejects the dead owner's stale completion attempt.
        let failover = self
            .resume_crashed_worker_work(store.as_ref(), &session, &live_owner, &live_lease, &work)
            .await?;

        let stale_completion = stale_lease.completion();
        store
            .release_session_execution_lease(&stale_completion)
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("stale worker completion failed: {err}"))
            })?;
        let renewed_live = store
            .renew_session_execution_lease(&live_lease.fence(), LEASE_TTL_MS)
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!(
                    "live worker lease was cleared by stale completion: {err}"
                ))
            })?;
        store
            .release_session_execution_lease(&renewed_live.completion())
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("release live lease failed: {err}"))
            })?;
        Ok(json!({
            "worker_alias": event.actor_alias,
            "session": session,
            "initial_owner": owner_json(&stale_owner),
            "active_owner": owner_json(&live_owner),
            "active_fencing_token": renewed_live.fencing_token,
            "stale_completion_rejected": true,
            "lease_owner_changed": !stale_owner.same_incarnation(&live_owner),
            "runtime_stale_completion": {
                "session_id": stale_completion.session_id,
                "owner": owner_json(&stale_completion.owner),
                "fencing_token": stale_completion.fencing_token,
                "lease_token_present": !stale_completion.lease_token.is_empty(),
            },
            "runtime_active_lease": {
                "session_id": renewed_live.session_id,
                "owner": owner_json(&renewed_live.owner),
                "fencing_token": renewed_live.fencing_token,
                "lease_token_present": !renewed_live.lease_token.is_empty(),
            },
            "runtime_worker_store": {
                "session_execution_lease_reclaimed": true,
                "stale_completion_left_live_lease_renewable": true,
                "worker_owned_work": {
                    "batch_id_present": !work.batch_id.is_empty(),
                    "source_key": work.source_key,
                    "first_owner_claim_fencing_token": work.claim_fencing_token,
                    "first_owner_claimed_work": true,
                    "second_owner_resumed_work": failover.resumed_by_second_owner,
                    "second_owner_claim_fencing_token": failover.resumed_claim_fencing_token,
                    "second_owner_outranks_first": failover.resumed_claim_fencing_token
                        > work.claim_fencing_token,
                    "stale_work_completion_rejected": failover.stale_work_completion_rejected,
                },
            },
        }))
    }

    /// Worker one claims a real unit of queued work under its session execution
    /// lease, modelling a worker-owned turn that is in flight when the worker
    /// crashes.
    async fn start_worker_owned_work(
        &self,
        store: &dyn RuntimePersistence,
        session: &str,
        owner: &LeaseOwnerIdentity,
        lease: &lash_core::SessionExecutionLease,
        occurred_at_ms: u64,
    ) -> Result<WorkerOwnedWork, RuntimeBoundaryError> {
        let source_key = format!("worker-failover/{session}/work");
        let wake = worker_failover_work(session, occurred_at_ms)?;
        let batch = store
            .enqueue_queued_work(
                QueuedWorkBatchDraft::new(
                    session.to_string(),
                    DeliveryPolicy::EarliestSafeBoundary,
                    SlotPolicy::Exclusive,
                    vec![QueuedWorkPayload::process_wake(wake)],
                )
                .with_source_key(source_key.clone())
                .with_merge_key(MergeKey::Never),
            )
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("enqueue worker-owned work failed: {err}"))
            })?;
        let claim = store
            .claim_ready_queued_work_by_batch_ids(
                session,
                &lease.fence(),
                owner,
                QueuedWorkClaimBoundary::Idle,
                LEASE_TTL_MS,
                std::slice::from_ref(&batch.batch_id),
            )
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("first worker claim of work failed: {err}"))
            })?
            .ok_or_else(|| {
                RuntimeBoundaryError::new(
                    "first worker could not claim its own queued work".to_string(),
                )
            })?;
        Ok(WorkerOwnedWork {
            batch_id: batch.batch_id,
            source_key,
            claim_fencing_token: claim.fencing_token,
            claim,
        })
    }

    /// Worker two reclaims the crashed worker's in-flight work, resumes it, and
    /// proves the dead owner's stale work completion is rejected.
    async fn resume_crashed_worker_work(
        &self,
        store: &dyn RuntimePersistence,
        session: &str,
        owner: &LeaseOwnerIdentity,
        lease: &lash_core::SessionExecutionLease,
        work: &WorkerOwnedWork,
    ) -> Result<WorkerFailover, RuntimeBoundaryError> {
        // The crashed worker's claim is released on takeover so the work is
        // reclaimable by the new lease owner.
        store
            .abandon_queued_work_claim(&work.claim)
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("release crashed worker claim failed: {err}"))
            })?;
        let resumed = store
            .claim_ready_queued_work_by_batch_ids(
                session,
                &lease.fence(),
                owner,
                QueuedWorkClaimBoundary::Idle,
                LEASE_TTL_MS,
                std::slice::from_ref(&work.batch_id),
            )
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!("second worker claim of work failed: {err}"))
            })?
            .ok_or_else(|| {
                RuntimeBoundaryError::new(
                    "second worker could not resume the crashed worker's queued work".to_string(),
                )
            })?;
        // The dead owner's late completion (a renewal of its now-superseded
        // claim) must be rejected as expired, never silently accepted.
        let stale_work_completion_rejected = matches!(
            store
                .renew_queued_work_claim(&work.claim, LEASE_TTL_MS)
                .await,
            Err(lash_core::StoreError::QueuedWorkClaimExpired { .. })
        );
        if !stale_work_completion_rejected {
            return Err(RuntimeBoundaryError::new(
                "crashed worker's stale work completion was not rejected after failover".to_string(),
            ));
        }
        Ok(WorkerFailover {
            resumed_by_second_owner: true,
            resumed_claim_fencing_token: resumed.fencing_token,
            stale_work_completion_rejected,
        })
    }

    async fn ensure_effect_controller(
        &mut self,
    ) -> Result<Arc<dyn RuntimeEffectController>, RuntimeBoundaryError> {
        if let Some(controller) = &self.effect_controller {
            return Ok(controller.clone());
        }
        let scope = ExecutionScope::runtime_operation(EFFECT_SCOPE_ID);
        let controller: Arc<dyn RuntimeEffectController> = match &self.effect_replay_store {
            RuntimeEffectReplayStore::Memory => Arc::new(
                lash_sqlite_store::SqliteRuntimeEffectController::memory(scope)
                    .await
                    .map_err(|err| {
                        RuntimeBoundaryError::new(format!(
                            "open in-memory effect replay controller failed: {err}"
                        ))
                    })?,
            ),
            RuntimeEffectReplayStore::SqliteFile(path) => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(|err| {
                        RuntimeBoundaryError::new(format!(
                            "create effect replay directory `{}` failed: {err}",
                            parent.display()
                        ))
                    })?;
                }
                Arc::new(
                    lash_sqlite_store::SqliteRuntimeEffectController::open(path, scope)
                        .await
                        .map_err(|err| {
                            RuntimeBoundaryError::new(format!(
                                "open sqlite effect replay controller `{}` failed: {err}",
                                path.display()
                            ))
                        })?,
                )
            }
            RuntimeEffectReplayStore::Postgres(storage) => {
                Arc::new(storage.runtime_effect_controller(scope))
            }
        };
        self.effect_controller = Some(Arc::clone(&controller));
        Ok(controller)
    }

    async fn store_for_session(
        &self,
        session_id: &str,
    ) -> Result<Arc<dyn RuntimePersistence>, RuntimeBoundaryError> {
        let request = SessionStoreCreateRequest {
            session_id: session_id.to_string(),
            relation: SessionRelation::Root,
            policy: lash_core::SessionPolicy::default(),
        };
        if let Some(store) = self
            .store_factory
            .open_existing_store(&request)
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!(
                    "open existing runtime store for `{session_id}` failed: {err}"
                ))
            })?
        {
            return Ok(store);
        }
        self.store_factory
            .create_store(&request)
            .await
            .map_err(|err| {
                RuntimeBoundaryError::new(format!(
                    "create runtime store for `{session_id}` failed: {err}"
                ))
            })
    }
}

struct WorkerOwnedWork {
    batch_id: String,
    source_key: String,
    claim_fencing_token: u64,
    claim: QueuedWorkClaim,
}

struct WorkerFailover {
    resumed_by_second_owner: bool,
    resumed_claim_fencing_token: u64,
    stale_work_completion_rejected: bool,
}

fn worker_failover_work(
    session: &str,
    occurred_at_ms: u64,
) -> Result<lash_core::ProcessWakeDelivery, RuntimeBoundaryError> {
    let process_id = format!("sim-worker-{session}");
    let dedupe_key = format!("worker-failover/{session}/work");
    lash_core::process_wake_delivery(lash_core::ProcessWakeDeliveryRequest {
        target_scope: SessionScope::new(session.to_string()),
        process_id: process_id.clone(),
        sequence: 1,
        event_type: "process.wake".to_string(),
        event_invocation: RuntimeInvocation {
            scope: RuntimeScope::new(session.to_string()),
            subject: RuntimeSubject::ProcessEvent {
                process_id,
                sequence: 1,
                event_type: "process.wake".to_string(),
            },
            caused_by: None,
            replay: Some(RuntimeReplay {
                key: format!("worker-failover:{session}:work"),
            }),
        },
        process_caused_by: None,
        wake: lash_core::ProcessWake {
            input: format!("worker-owned work for {session}"),
            dedupe_key,
        },
        occurred_at: std::time::UNIX_EPOCH + std::time::Duration::from_millis(occurred_at_ms),
    })
    .map_err(|err| RuntimeBoundaryError::new(format!("build worker-owned work failed: {err}")))
}

fn boundary_session_alias(event: &BoundaryEvent) -> String {
    event
        .payload
        .get("session")
        .and_then(Value::as_str)
        .unwrap_or(&event.actor_alias)
        .to_string()
}

fn owner_json(owner: &LeaseOwnerIdentity) -> Value {
    json!({
        "owner_id": owner.owner_id,
        "incarnation_id": owner.incarnation_id,
    })
}

fn boundary_kind_name(kind: BoundaryKind) -> &'static str {
    match kind {
        BoundaryKind::Ingress => "ingress",
        BoundaryKind::QueuedIngress => "queued_ingress",
        BoundaryKind::Provider => "provider",
        BoundaryKind::ProviderEvent => "provider_event",
        BoundaryKind::Tool => "tool",
        BoundaryKind::ExecCode => "exec_code",
        BoundaryKind::DurableEffect => "durable_effect",
        BoundaryKind::ProcessWake => "process_wake",
        BoundaryKind::Worker => "worker",
        BoundaryKind::Observer => "observer",
        BoundaryKind::Cancellation => "cancellation",
        BoundaryKind::Trigger => "trigger",
        BoundaryKind::BackendFailure => "backend_failure",
        BoundaryKind::ProviderMutation => "provider_mutation",
        BoundaryKind::LeaseTime => "lease_time",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(kind: BoundaryKind, id: &str, payload: Value) -> BoundaryEvent {
        BoundaryEvent::new(id, "session-001", kind, 1, "test", payload)
    }

    fn harness() -> RuntimeBoundaryHarness {
        let factory: Arc<dyn SessionStoreFactory> =
            Arc::new(lash_core::InMemorySessionStoreFactory::new());
        RuntimeBoundaryHarness::new(factory, RuntimeEffectReplayStore::Memory)
    }

    #[tokio::test]
    async fn worker_failover_continuation_oracle_catches_a_store_that_fails_to_fence() {
        // END-TO-END NEGATIVE: drive the REAL worker-stale-completion boundary
        // against a store whose session-execution lease is already held, so the
        // worker's stale-owner claim is Busy and the real lease store can neither
        // fence nor continue the work. The failover-continuation oracle MUST catch
        // this — proving it bites on a real un-fencing path, not just synthetic
        // facts.
        let mut harness = harness();
        let session = "worker-unfenced-session";
        let store = harness.store_for_session(session).await.expect("store");
        let blocker = LeaseOwnerIdentity::opaque("blocker-owner", "blocker-owner:001");
        let blocking = match store
            .try_claim_session_execution_lease(session, &blocker, LEASE_TTL_MS)
            .await
            .expect("blocker claim")
        {
            SessionExecutionLeaseClaimOutcome::Acquired(lease) => lease,
            other => panic!("expected to acquire the blocking lease: {other:?}"),
        };

        let worker_event = BoundaryEvent::new(
            "worker:unfenced:001",
            session,
            BoundaryKind::Worker,
            5,
            "worker.stale-completion-rejected",
            json!({ "session": session }),
        );
        let observed = harness
            .run_worker_stale_completion(&worker_event)
            .await
            .expect("worker boundary observed");

        // The real store could NOT fence: no rejection, no work continuation.
        assert_eq!(
            observed.get("stale_completion_rejected").and_then(Value::as_bool),
            Some(false),
            "a busy store must not report a fenced stale completion: {observed}"
        );
        assert!(
            observed
                .get("runtime_worker_store")
                .and_then(|store| store.get("worker_owned_work"))
                .is_none(),
            "a store that failed to fence must not record worker-owned-work continuation: {observed}"
        );

        let delivered = crate::scheduler::DeliveredBoundary {
            schema: "test".to_string(),
            sequence: 0,
            scheduler: crate::scheduler::SchedulerDeliveryEvidence::default(),
            boundary_id: "worker:unfenced:001".to_string(),
            actor_alias: session.to_string(),
            kind: BoundaryKind::Worker,
            at: 5,
            label: "worker.stale-completion-rejected".to_string(),
            payload: json!({ "session": session }),
            observed,
        };
        let verdict = crate::oracles::worker_failover_continues_work(std::slice::from_ref(&delivered));
        assert!(
            !verdict.is_passed(),
            "the failover-continuation oracle must catch a store that failed to fence: {}",
            verdict.message
        );

        let _ = store
            .release_session_execution_lease(&blocking.completion())
            .await;
    }

    #[tokio::test]
    async fn durable_effect_replays_through_runtime_effect_controller() {
        let mut harness = harness();
        let payload = json!({
            "durable_key": "sleep/session-001/001",
            "result": {"completed": true},
            "runtime_effect": {"effect_id": "effect/sleep/001"},
        });
        let first = harness
            .complete_durable_effect(&event(
                BoundaryKind::DurableEffect,
                "durable:first",
                payload.clone(),
            ))
            .await
            .expect("first durable effect");
        let replay = harness
            .complete_durable_effect(&event(
                BoundaryKind::DurableEffect,
                "durable:replay",
                json!({
                    "durable_key": "sleep/session-001/001",
                    "result": {"completed": false},
                    "runtime_effect": {"effect_id": "effect/sleep/001"},
                }),
            ))
            .await
            .expect("replayed durable effect");

        assert_eq!(first["execution_count"], 1);
        assert_eq!(replay["execution_count"], 1);
        assert_eq!(replay["replay_count"], 1);
        assert_eq!(replay["replayed"], true);
        assert_eq!(first["result_digest"], replay["result_digest"]);
        assert_eq!(
            replay["runtime_effect"]["controller"],
            "sqlite_runtime_effect_controller"
        );
    }

    #[tokio::test]
    async fn process_wake_uses_runtime_queued_work_claim_and_dedupe() {
        let mut harness = harness();
        let payload = json!({
            "session": "session-001",
            "dedupe_key": "wake/session-001/001",
        });
        let first = harness
            .deliver_process_wake(&event(
                BoundaryKind::ProcessWake,
                "wake:first",
                payload.clone(),
            ))
            .await
            .expect("first wake");
        let duplicate = harness
            .deliver_process_wake(&event(BoundaryKind::ProcessWake, "wake:dupe", payload))
            .await
            .expect("duplicate wake");

        assert_eq!(first["claimed_once"], true);
        assert_eq!(duplicate["claimed_once"], false);
        assert_eq!(first["runtime_queued_work"]["claim_id_present"], true);
        assert_eq!(duplicate["runtime_queued_work"]["claim_id_present"], false);
    }

    #[tokio::test]
    async fn tool_boundary_uses_runtime_effect_controller_and_records_output() {
        let mut harness = harness();
        let observed = harness
            .complete_tool(&event(
                BoundaryKind::Tool,
                "tool:001",
                json!({
                    "tool": "lookup",
                    "output": {"answer": "tool data"},
                }),
            ))
            .await
            .expect("tool boundary");

        assert_eq!(observed["execution_count"], 1);
        assert_eq!(
            observed["runtime_effect"]["controller"],
            "sqlite_runtime_effect_controller"
        );
        assert_eq!(observed["runtime_tool_record"]["tool"], "lookup");
        assert!(
            observed["runtime_tool_output"]
                .to_string()
                .contains("tool data")
        );
    }

    #[tokio::test]
    async fn exec_boundary_uses_runtime_effect_controller_and_preserves_exit_data() {
        let mut harness = harness();
        let observed = harness
            .execute_code(&event(
                BoundaryKind::ExecCode,
                "exec:001",
                json!({
                    "output": "exec data",
                    "exit_code": 7,
                }),
            ))
            .await
            .expect("exec boundary");

        assert_eq!(observed["execution_count"], 1);
        assert_eq!(
            observed["runtime_effect"]["controller"],
            "sqlite_runtime_effect_controller"
        );
        assert_eq!(observed["exit_code"], 7);
        assert!(
            observed["runtime_effect_outcome"]
                .to_string()
                .contains("exec data")
        );
    }

    #[tokio::test]
    async fn worker_stale_completion_uses_runtime_session_lease_store() {
        let mut harness = harness();
        let observed = harness
            .run_worker_stale_completion(&event(
                BoundaryKind::Worker,
                "worker-001",
                json!({"session": "session-001"}),
            ))
            .await
            .expect("worker boundary");

        assert_eq!(observed["stale_completion_rejected"], true);
        assert_eq!(
            observed["runtime_worker_store"]["session_execution_lease_reclaimed"],
            true
        );
        assert!(observed["runtime_active_lease"].is_object());
        let work = &observed["runtime_worker_store"]["worker_owned_work"];
        assert_eq!(work["first_owner_claimed_work"], true);
        assert_eq!(work["second_owner_resumed_work"], true);
        assert_eq!(work["second_owner_outranks_first"], true);
        assert_eq!(work["stale_work_completion_rejected"], true);
        assert!(
            work["second_owner_claim_fencing_token"].as_u64().unwrap()
                > work["first_owner_claim_fencing_token"].as_u64().unwrap()
        );
    }
}
