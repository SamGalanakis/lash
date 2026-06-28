use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lash_core::runtime::{
    QueuedWorkBatchDraft, QueuedWorkClaimBoundary, QueuedWorkPayload, RuntimeReplay, RuntimeScope,
    RuntimeSubject,
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
                return Err(RuntimeBoundaryError::new(format!(
                    "process wake session lease busy for `{session}`; holder={}",
                    holder.owner.owner_id
                )));
            }
        };
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
        let stale_lease = match store
            .try_claim_session_execution_lease(&session, &stale_owner, LEASE_TTL_MS)
            .await
            .map_err(|err| RuntimeBoundaryError::new(format!("claim stale lease failed: {err}")))?
        {
            SessionExecutionLeaseClaimOutcome::Acquired(lease) => lease,
            SessionExecutionLeaseClaimOutcome::Busy { holder } => {
                return Err(RuntimeBoundaryError::new(format!(
                    "worker stale lease setup busy for `{session}`; holder={}",
                    holder.owner.owner_id
                )));
            }
        };
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
            },
        }))
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
    }
}
