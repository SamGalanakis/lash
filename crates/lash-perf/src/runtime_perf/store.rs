use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lash::usage::TokenLedgerEntry;
use lash_core::runtime::{
    QueuedWorkBatch, QueuedWorkBatchDraft, QueuedWorkClaim, QueuedWorkClaimBoundary, QueuedWorkItem,
};
use lash_core::store;
use lash_core::store::{
    GraphCommitDelta, PersistedSessionRead, RuntimeCommitResult, SessionCheckpoint, SessionHeadMeta,
};
use lash_core::{
    BlobRef, GcReport, LeaseOwnerIdentity, RuntimeCommit, RuntimePersistence,
    SessionExecutionLease, SessionExecutionLeaseClaimOutcome, SessionExecutionLeaseCompletion,
    SessionExecutionLeaseFence, SessionGraph, SessionNodeRecord, SessionReadScope,
    SessionStoreCreateRequest, SessionStoreFactory, StoreError, VacuumReport, current_epoch_ms,
};

#[derive(Clone)]
struct RuntimePerfQueuedBatch {
    batch: QueuedWorkBatch,
    claim_id: Option<String>,
    claim_token: Option<String>,
    claim_owner: Option<LeaseOwnerIdentity>,
    claim_fencing_token: u64,
    claim_expires_at_ms: u64,
}

#[derive(Clone, Default)]
struct RuntimePerfSessionExecutionLease {
    owner: Option<LeaseOwnerIdentity>,
    lease_token: Option<String>,
    fencing_token: u64,
    claimed_at_epoch_ms: u64,
    expires_at_epoch_ms: u64,
}

#[derive(Clone, Copy)]
enum RuntimePerfQueuedWorkClaimKind {
    LeadingSessionCommand,
    TurnWork {
        boundary: QueuedWorkClaimBoundary,
        max_batches: usize,
    },
}

fn session_fence_equivalent(
    left: &SessionExecutionLeaseFence,
    right: &SessionExecutionLeaseFence,
) -> bool {
    left.session_id == right.session_id
        && left.owner == right.owner
        && left.lease_token == right.lease_token
        && left.fencing_token == right.fencing_token
}

#[derive(Default)]
pub(crate) struct RuntimePerfStore {
    next_blob_id: AtomicU64,
    queued_work_next_seq: AtomicU64,
    session_head_meta: Mutex<Option<SessionHeadMeta>>,
    session_graph: Mutex<SessionGraph>,
    usage_deltas: Mutex<Vec<TokenLedgerEntry>>,
    session_meta: Mutex<Option<store::SessionMeta>>,
    runtime_turn_commits: Mutex<HashMap<(String, String), (String, RuntimeCommitResult)>>,
    session_execution_leases: Mutex<HashMap<String, RuntimePerfSessionExecutionLease>>,
    queued_work: Mutex<Vec<RuntimePerfQueuedBatch>>,
}

impl RuntimePerfStore {
    pub(crate) fn graph_node_count(&self) -> usize {
        self.session_graph
            .lock()
            .expect("lock perf graph")
            .nodes
            .len()
    }

    fn verify_session_execution_lease(
        &self,
        session_id: &str,
        fence: &SessionExecutionLeaseFence,
    ) -> Result<(), StoreError> {
        if fence.session_id != session_id {
            return Err(StoreError::SessionExecutionLeaseExpired {
                session_id: session_id.to_string(),
            });
        }
        let now = current_epoch_ms();
        let leases = self
            .session_execution_leases
            .lock()
            .expect("lock perf session execution leases");
        let Some(current) = leases.get(&fence.session_id) else {
            return Err(StoreError::SessionExecutionLeaseExpired {
                session_id: fence.session_id.clone(),
            });
        };
        if current.owner.as_ref() == Some(&fence.owner)
            && current.lease_token.as_deref() == Some(fence.lease_token.as_str())
            && current.fencing_token == fence.fencing_token
            && current.expires_at_epoch_ms > now
        {
            Ok(())
        } else {
            Err(StoreError::SessionExecutionLeaseExpired {
                session_id: fence.session_id.clone(),
            })
        }
    }

    fn release_session_execution_lease_in_memory(
        &self,
        completion: &SessionExecutionLeaseCompletion,
    ) {
        let mut leases = self
            .session_execution_leases
            .lock()
            .expect("lock perf session execution leases");
        if let Some(current) = leases.get_mut(&completion.session_id)
            && current.owner.as_ref() == Some(&completion.owner)
            && current.lease_token.as_deref() == Some(completion.lease_token.as_str())
            && current.fencing_token == completion.fencing_token
        {
            current.owner = None;
            current.lease_token = None;
            current.claimed_at_epoch_ms = 0;
            current.expires_at_epoch_ms = 0;
        }
    }

    fn queued_batch_work_class(
        batch: &QueuedWorkBatch,
    ) -> Result<lash_core::runtime::QueuedWorkClass, StoreError> {
        batch.work_class().ok_or_else(|| {
            StoreError::Backend(format!(
                "queued-work batch `{}` has mixed or empty payload classes",
                batch.batch_id
            ))
        })
    }

    fn claim_ready_queued_work_perf(
        &self,
        session_id: &str,
        session_execution_lease: &SessionExecutionLeaseFence,
        owner: &LeaseOwnerIdentity,
        lease_ttl_ms: u64,
        kind: RuntimePerfQueuedWorkClaimKind,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        let max_batches = match kind {
            RuntimePerfQueuedWorkClaimKind::LeadingSessionCommand => 1,
            RuntimePerfQueuedWorkClaimKind::TurnWork { max_batches, .. } => max_batches,
        };
        if max_batches == 0 {
            return Ok(None);
        }
        self.verify_session_execution_lease(session_id, session_execution_lease)?;
        let now = current_epoch_ms();
        let mut queued = self.queued_work.lock().expect("lock perf queued work");
        queued.sort_by_key(|entry| entry.batch.enqueue_seq);
        let claim_available = |entry: &RuntimePerfQueuedBatch| {
            entry.claim_token.is_none()
                || entry.claim_expires_at_ms <= now
                || entry
                    .claim_owner
                    .as_ref()
                    .is_some_and(|holder| holder.is_definitely_dead_for_claimant(owner))
        };
        let claimable_indices = queued
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.batch.session_id == session_id
                    && entry.batch.available_at_ms <= now
                    && claim_available(entry)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if claimable_indices.is_empty() {
            return Ok(None);
        }
        let candidates = claimable_indices
            .iter()
            .map(|index| {
                let batch = &queued[*index].batch;
                Ok(store::queued_work::ClaimCandidate {
                    enqueue_seq: batch.enqueue_seq,
                    claim_fencing_token: queued[*index].claim_fencing_token,
                    work_class: Self::queued_batch_work_class(batch)?,
                    delivery_policy: batch.delivery_policy,
                    slot_policy: batch.slot_policy,
                    merge_key: batch.merge_key.clone(),
                })
            })
            .collect::<Result<Vec<_>, StoreError>>()?;
        let selected_len = match kind {
            RuntimePerfQueuedWorkClaimKind::LeadingSessionCommand => {
                store::queued_work::select_leading_session_command(&candidates)
            }
            RuntimePerfQueuedWorkClaimKind::TurnWork {
                boundary,
                max_batches,
            } => store::queued_work::select_turn_work_claim_prefix(
                &candidates,
                boundary,
                max_batches,
            ),
        };
        if selected_len == 0 {
            return Ok(None);
        }
        let first_index = claimable_indices[0];
        let first = queued[first_index].batch.clone();
        let fencing_token = queued[first_index].claim_fencing_token.saturating_add(1);
        let claim_id = format!("perf-qwc:{}:{fencing_token}", first.enqueue_seq);
        let lease_token = format!(
            "{session_id}:{}:{}:{claim_id}:{now}",
            owner.owner_id, owner.incarnation_id
        );
        let expires_at = now.saturating_add(lease_ttl_ms);
        let mut batches = Vec::new();
        for index in claimable_indices.into_iter().take(selected_len) {
            let entry = &mut queued[index];
            entry.claim_id = Some(claim_id.clone());
            entry.claim_token = Some(lease_token.clone());
            entry.claim_owner = Some(owner.clone());
            entry.claim_fencing_token = entry.claim_fencing_token.saturating_add(1);
            entry.claim_expires_at_ms = expires_at;
            batches.push(entry.batch.clone());
        }
        Ok(Some(QueuedWorkClaim {
            session_id: session_id.to_string(),
            claim_id,
            owner: owner.clone(),
            lease_token,
            fencing_token,
            claimed_at_epoch_ms: now,
            expires_at_epoch_ms: expires_at,
            batches,
        }))
    }
}

#[derive(Clone)]
pub(crate) struct RuntimePerfStoreFactory {
    pub(crate) store: Arc<RuntimePerfStore>,
    child_stores: Arc<Mutex<HashMap<String, Arc<RuntimePerfStore>>>>,
}

impl RuntimePerfStoreFactory {
    pub(crate) fn new(store: Arc<RuntimePerfStore>) -> Self {
        Self {
            store,
            child_stores: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl SessionStoreFactory for RuntimePerfStoreFactory {
    async fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        if request.parent_session_id().is_none() {
            return Ok(Arc::clone(&self.store) as Arc<dyn RuntimePersistence>);
        }
        let mut stores = self
            .child_stores
            .lock()
            .expect("lock runtime perf child stores");
        let store = stores
            .entry(request.session_id.clone())
            .or_insert_with(|| Arc::new(RuntimePerfStore::default()));
        Ok(Arc::clone(store) as Arc<dyn RuntimePersistence>)
    }

    async fn delete_session(&self, _session_id: &str) -> Result<(), String> {
        Ok(())
    }
}

lash_core::impl_noop_attachment_manifest!(RuntimePerfStore);

#[async_trait::async_trait]
impl RuntimePersistence for RuntimePerfStore {
    async fn load_session(
        &self,
        scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, store::StoreError> {
        let Some(meta) = self
            .session_head_meta
            .lock()
            .expect("lock perf session head meta")
            .clone()
        else {
            return Ok(None);
        };
        let graph = {
            let stored_graph = self.session_graph.lock().expect("lock perf graph");
            match scope {
                SessionReadScope::FullGraph => stored_graph.clone(),
                SessionReadScope::ActivePath { leaf_node_id } => {
                    if leaf_node_id.is_none() || leaf_node_id == stored_graph.leaf_node_id {
                        let nodes = stored_graph
                            .active_path_nodes()
                            .into_iter()
                            .cloned()
                            .collect();
                        SessionGraph::from_nodes(nodes, stored_graph.leaf_node_id.clone())
                    } else {
                        let mut scoped = stored_graph.clone();
                        scoped.set_leaf_node_id(leaf_node_id);
                        let nodes = scoped.active_path_nodes().into_iter().cloned().collect();
                        SessionGraph::from_nodes(nodes, scoped.leaf_node_id.clone())
                    }
                }
            }
        };
        Ok(Some(PersistedSessionRead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            config: meta.config,
            agent_frames: meta.agent_frames,
            current_agent_frame_id: meta.current_agent_frame_id,
            graph,
            checkpoint_ref: meta.checkpoint_ref,
            checkpoint: None,
            token_ledger: self
                .usage_deltas
                .lock()
                .expect("lock perf usage deltas")
                .clone(),
        }))
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<SessionNodeRecord>, store::StoreError> {
        Ok(self
            .session_graph
            .lock()
            .expect("lock perf graph")
            .find_node(node_id)
            .cloned())
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, store::StoreError> {
        let RuntimeCommit {
            session_id,
            expected_head_revision,
            config,
            agent_frames,
            current_agent_frame_id,
            graph: graph_delta,
            checkpoint,
            usage_deltas,
            turn_commit,
            completed_queue_claims,
            session_execution_lease,
            release_session_execution_lease,
            committed_attachment_ids: _,
        } = commit;
        let mut meta_guard = self
            .session_head_meta
            .lock()
            .expect("lock perf session head meta");
        let actual = meta_guard.as_ref().map_or(0, |meta| meta.head_revision);
        if let Some(completed) = &turn_commit {
            if completed.session_id != session_id {
                return Err(StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
            let key = (completed.session_id.clone(), completed.turn_id.clone());
            if let Some((stored_hash, result)) = self
                .runtime_turn_commits
                .lock()
                .expect("lock perf runtime turn commits")
                .get(&key)
                .cloned()
            {
                if stored_hash == completed.turn_commit_hash {
                    if let Some(completion) = release_session_execution_lease.as_ref() {
                        self.release_session_execution_lease_in_memory(completion);
                    }
                    return Ok(result);
                }
                return Err(StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
        }
        let Some(session_execution_lease) = session_execution_lease.as_ref() else {
            return Err(StoreError::SessionExecutionLeaseExpired {
                session_id: session_id.clone(),
            });
        };
        self.verify_session_execution_lease(&session_id, session_execution_lease)?;
        if expected_head_revision.is_some() && expected_head_revision != Some(actual) {
            return Err(store::StoreError::HeadRevisionConflict {
                expected: expected_head_revision,
                actual,
            });
        }
        let mut graph = self.session_graph.lock().expect("lock perf graph");
        let leaf_node_id = match graph_delta {
            GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
            GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                graph.extend_node_records(nodes);
                leaf_node_id
            }
            GraphCommitDelta::ReplaceFull(next) => {
                let leaf_node_id = next.leaf_node_id.clone();
                *graph = next;
                leaf_node_id
            }
        };
        if !usage_deltas.is_empty() {
            self.usage_deltas
                .lock()
                .expect("lock perf usage deltas")
                .extend(usage_deltas);
        }
        for completed in &completed_queue_claims {
            let mut queued = self.queued_work.lock().expect("lock perf queued work");
            let matches = queued
                .iter()
                .filter(|entry| {
                    entry.batch.session_id == completed.session_id
                        && entry.claim_id.as_deref() == Some(completed.claim_id.as_str())
                        && entry.claim_token.as_deref() == Some(completed.lease_token.as_str())
                        && completed.batch_ids.contains(&entry.batch.batch_id)
                })
                .count();
            if matches != completed.batch_ids.len() {
                return Err(StoreError::QueuedWorkClaimExpired {
                    session_id: completed.session_id.clone(),
                    claim_id: completed.claim_id.clone(),
                });
            }
            queued.retain(|entry| {
                !(entry.batch.session_id == completed.session_id
                    && entry.claim_id.as_deref() == Some(completed.claim_id.as_str())
                    && entry.claim_token.as_deref() == Some(completed.lease_token.as_str())
                    && completed.batch_ids.contains(&entry.batch.batch_id))
            });
        }
        let next_checkpoint_blob_ref = |kind: &str| {
            let id = self.next_blob_id.fetch_add(1, Ordering::Relaxed);
            BlobRef(format!("perf-{kind}-{id}"))
        };
        let manifest = SessionCheckpoint {
            turn_state: checkpoint.turn_state,
            tool_state_ref: if checkpoint.tool_state.is_some() {
                Some(next_checkpoint_blob_ref("tool-state"))
            } else {
                checkpoint.tool_state_ref
            },
            plugin_snapshot_ref: if checkpoint.plugin_snapshot.is_some() {
                Some(next_checkpoint_blob_ref("plugin-snapshot"))
            } else {
                checkpoint.plugin_snapshot_ref
            },
            plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
            execution_state_ref: if checkpoint.execution_state.is_some() {
                Some(next_checkpoint_blob_ref("execution-state"))
            } else {
                checkpoint.execution_state_ref
            },
        };
        let graph_node_count = graph.nodes.len();
        drop(graph);
        let id = self.next_blob_id.fetch_add(1, Ordering::Relaxed);
        let checkpoint_ref = BlobRef(format!("perf-checkpoint-{id}"));
        let head_revision = actual + 1;
        *meta_guard = Some(SessionHeadMeta {
            session_id: session_id.clone(),
            head_revision,
            config,
            agent_frames,
            current_agent_frame_id,
            checkpoint_ref: Some(checkpoint_ref.clone()),
            leaf_node_id,
            graph_node_count,
            token_ledger: Vec::new(),
        });
        let result = RuntimeCommitResult {
            head_revision,
            checkpoint_ref,
            manifest,
        };
        if let Some(completed) = &turn_commit {
            self.runtime_turn_commits
                .lock()
                .expect("lock perf runtime turn commits")
                .insert(
                    (completed.session_id.clone(), completed.turn_id.clone()),
                    (completed.turn_commit_hash.clone(), result.clone()),
                );
        }
        if let Some(completion) = release_session_execution_lease.as_ref() {
            self.release_session_execution_lease_in_memory(completion);
        }
        Ok(result)
    }

    async fn try_claim_session_execution_lease(
        &self,
        session_id: &str,
        owner: &LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLeaseClaimOutcome, StoreError> {
        let now = current_epoch_ms();
        let mut leases = self
            .session_execution_leases
            .lock()
            .expect("lock perf session execution leases");
        let current = leases.entry(session_id.to_string()).or_default();
        if current.lease_token.is_some() && current.expires_at_epoch_ms > now {
            if current
                .owner
                .as_ref()
                .is_some_and(|current_owner| current_owner.same_incarnation(owner))
            {
                current.expires_at_epoch_ms = now.saturating_add(lease_ttl_ms);
                return Ok(SessionExecutionLeaseClaimOutcome::Acquired(
                    SessionExecutionLease {
                        session_id: session_id.to_string(),
                        owner: owner.clone(),
                        lease_token: current.lease_token.clone().expect("live lease token set"),
                        fencing_token: current.fencing_token,
                        claimed_at_epoch_ms: current.claimed_at_epoch_ms,
                        expires_at_epoch_ms: current.expires_at_epoch_ms,
                    },
                ));
            }
            return Ok(SessionExecutionLeaseClaimOutcome::Busy {
                holder: SessionExecutionLease {
                    session_id: session_id.to_string(),
                    owner: current.owner.clone().expect("live lease owner set"),
                    lease_token: current.lease_token.clone().expect("live lease token set"),
                    fencing_token: current.fencing_token,
                    claimed_at_epoch_ms: current.claimed_at_epoch_ms,
                    expires_at_epoch_ms: current.expires_at_epoch_ms,
                },
            });
        }
        current.fencing_token = current.fencing_token.saturating_add(1);
        current.owner = Some(owner.clone());
        current.lease_token = Some(format!(
            "{session_id}:{}:{}:{now}:{}",
            owner.owner_id, owner.incarnation_id, current.fencing_token
        ));
        current.claimed_at_epoch_ms = now;
        current.expires_at_epoch_ms = now.saturating_add(lease_ttl_ms);
        Ok(SessionExecutionLeaseClaimOutcome::Acquired(
            SessionExecutionLease {
                session_id: session_id.to_string(),
                owner: owner.clone(),
                lease_token: current.lease_token.clone().expect("lease token set"),
                fencing_token: current.fencing_token,
                claimed_at_epoch_ms: current.claimed_at_epoch_ms,
                expires_at_epoch_ms: current.expires_at_epoch_ms,
            },
        ))
    }

    async fn reclaim_session_execution_lease(
        &self,
        session_id: &str,
        owner: &LeaseOwnerIdentity,
        observed_holder: &SessionExecutionLeaseFence,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLeaseClaimOutcome, StoreError> {
        let now = current_epoch_ms();
        let mut leases = self
            .session_execution_leases
            .lock()
            .expect("lock perf session execution leases");
        let current = leases.entry(session_id.to_string()).or_default();
        if current.lease_token.is_some() && current.expires_at_epoch_ms > now {
            let holder = SessionExecutionLease {
                session_id: session_id.to_string(),
                owner: current.owner.clone().expect("live lease owner set"),
                lease_token: current.lease_token.clone().expect("live lease token set"),
                fencing_token: current.fencing_token,
                claimed_at_epoch_ms: current.claimed_at_epoch_ms,
                expires_at_epoch_ms: current.expires_at_epoch_ms,
            };
            if !session_fence_equivalent(&holder.fence(), observed_holder)
                || !holder.owner.is_definitely_dead_for_claimant(owner)
            {
                return Ok(SessionExecutionLeaseClaimOutcome::Busy { holder });
            }
        }
        current.fencing_token = current.fencing_token.saturating_add(1);
        current.owner = Some(owner.clone());
        current.lease_token = Some(format!(
            "{session_id}:{}:{}:{now}:{}",
            owner.owner_id, owner.incarnation_id, current.fencing_token
        ));
        current.claimed_at_epoch_ms = now;
        current.expires_at_epoch_ms = now.saturating_add(lease_ttl_ms);
        Ok(SessionExecutionLeaseClaimOutcome::Acquired(
            SessionExecutionLease {
                session_id: session_id.to_string(),
                owner: owner.clone(),
                lease_token: current.lease_token.clone().expect("lease token set"),
                fencing_token: current.fencing_token,
                claimed_at_epoch_ms: current.claimed_at_epoch_ms,
                expires_at_epoch_ms: current.expires_at_epoch_ms,
            },
        ))
    }

    async fn renew_session_execution_lease(
        &self,
        fence: &SessionExecutionLeaseFence,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLease, StoreError> {
        let now = current_epoch_ms();
        let mut leases = self
            .session_execution_leases
            .lock()
            .expect("lock perf session execution leases");
        let Some(current) = leases.get_mut(&fence.session_id) else {
            return Err(StoreError::SessionExecutionLeaseExpired {
                session_id: fence.session_id.clone(),
            });
        };
        if current.owner.as_ref() != Some(&fence.owner)
            || current.lease_token.as_deref() != Some(fence.lease_token.as_str())
            || current.fencing_token != fence.fencing_token
            || current.expires_at_epoch_ms <= now
        {
            return Err(StoreError::SessionExecutionLeaseExpired {
                session_id: fence.session_id.clone(),
            });
        }
        current.expires_at_epoch_ms = now.saturating_add(lease_ttl_ms);
        Ok(SessionExecutionLease {
            session_id: fence.session_id.clone(),
            owner: fence.owner.clone(),
            lease_token: fence.lease_token.clone(),
            fencing_token: fence.fencing_token,
            claimed_at_epoch_ms: current.claimed_at_epoch_ms,
            expires_at_epoch_ms: current.expires_at_epoch_ms,
        })
    }

    async fn release_session_execution_lease(
        &self,
        completion: &SessionExecutionLeaseCompletion,
    ) -> Result<(), StoreError> {
        self.release_session_execution_lease_in_memory(completion);
        Ok(())
    }

    async fn enqueue_queued_work(
        &self,
        batch: QueuedWorkBatchDraft,
    ) -> Result<QueuedWorkBatch, StoreError> {
        let mut queued = self.queued_work.lock().expect("lock perf queued work");
        if let Some(source_key) = batch.source_key.as_deref()
            && let Some(existing) = queued.iter().find(|entry| {
                entry.batch.session_id == batch.session_id
                    && entry.batch.source_key.as_deref() == Some(source_key)
            })
        {
            return Ok(existing.batch.clone());
        }
        let enqueue_seq = self.queued_work_next_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let batch_id = format!("perf-qwb-{enqueue_seq}");
        let items = batch
            .payloads
            .into_iter()
            .enumerate()
            .map(|(index, payload)| QueuedWorkItem {
                item_id: format!("{batch_id}:item:{index}"),
                payload,
            })
            .collect();
        let stored = QueuedWorkBatch {
            batch_id,
            session_id: batch.session_id,
            enqueue_seq,
            source_key: batch.source_key,
            delivery_policy: batch.delivery_policy,
            slot_policy: batch.slot_policy,
            merge_key: batch.merge_key,
            available_at_ms: batch.available_at_ms,
            enqueued_at_ms: current_epoch_ms(),
            items,
        };
        queued.push(RuntimePerfQueuedBatch {
            batch: stored.clone(),
            claim_id: None,
            claim_token: None,
            claim_owner: None,
            claim_fencing_token: 0,
            claim_expires_at_ms: 0,
        });
        queued.sort_by_key(|entry| entry.batch.enqueue_seq);
        Ok(stored)
    }

    async fn claim_leading_ready_session_command(
        &self,
        session_id: &str,
        session_execution_lease: &SessionExecutionLeaseFence,
        owner: &LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        self.claim_ready_queued_work_perf(
            session_id,
            session_execution_lease,
            owner,
            lease_ttl_ms,
            RuntimePerfQueuedWorkClaimKind::LeadingSessionCommand,
        )
    }

    async fn claim_ready_queued_work(
        &self,
        session_id: &str,
        session_execution_lease: &SessionExecutionLeaseFence,
        owner: &LeaseOwnerIdentity,
        boundary: QueuedWorkClaimBoundary,
        lease_ttl_ms: u64,
        max_batches: usize,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        self.claim_ready_queued_work_perf(
            session_id,
            session_execution_lease,
            owner,
            lease_ttl_ms,
            RuntimePerfQueuedWorkClaimKind::TurnWork {
                boundary,
                max_batches,
            },
        )
    }

    async fn renew_queued_work_claim(
        &self,
        claim: &QueuedWorkClaim,
        lease_ttl_ms: u64,
    ) -> Result<QueuedWorkClaim, StoreError> {
        let mut queued = self.queued_work.lock().expect("lock perf queued work");
        let expires_at = current_epoch_ms().saturating_add(lease_ttl_ms);
        let mut changed = 0;
        for entry in queued.iter_mut() {
            if entry.batch.session_id == claim.session_id
                && entry.claim_id.as_deref() == Some(claim.claim_id.as_str())
                && entry.claim_token.as_deref() == Some(claim.lease_token.as_str())
            {
                entry.claim_expires_at_ms = expires_at;
                changed += 1;
            }
        }
        if changed != claim.batches.len() {
            return Err(StoreError::QueuedWorkClaimExpired {
                session_id: claim.session_id.clone(),
                claim_id: claim.claim_id.clone(),
            });
        }
        Ok(QueuedWorkClaim {
            expires_at_epoch_ms: expires_at,
            ..claim.clone()
        })
    }

    async fn abandon_queued_work_claim(&self, claim: &QueuedWorkClaim) -> Result<(), StoreError> {
        let mut queued = self.queued_work.lock().expect("lock perf queued work");
        for entry in queued.iter_mut() {
            if entry.batch.session_id == claim.session_id
                && entry.claim_id.as_deref() == Some(claim.claim_id.as_str())
                && entry.claim_token.as_deref() == Some(claim.lease_token.as_str())
            {
                entry.claim_id = None;
                entry.claim_token = None;
                entry.claim_owner = None;
                entry.claim_expires_at_ms = 0;
            }
        }
        Ok(())
    }

    async fn cancel_queued_work_batch(
        &self,
        session_id: &str,
        batch_id: &str,
    ) -> Result<Option<QueuedWorkBatch>, StoreError> {
        let now = current_epoch_ms();
        let mut queued = self.queued_work.lock().expect("lock perf queued work");
        let Some(index) = queued.iter().position(|entry| {
            entry.batch.session_id == session_id && entry.batch.batch_id == batch_id
        }) else {
            return Ok(None);
        };
        let entry = &queued[index];
        if entry.claim_token.is_some() && entry.claim_expires_at_ms > now {
            return Ok(None);
        }
        Ok(Some(queued.remove(index).batch))
    }

    async fn list_queued_work(&self, session_id: &str) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let mut batches = self
            .queued_work
            .lock()
            .expect("lock perf queued work")
            .iter()
            .filter(|entry| entry.batch.session_id == session_id)
            .map(|entry| entry.batch.clone())
            .collect::<Vec<_>>();
        batches.sort_by_key(|batch| batch.enqueue_seq);
        Ok(batches)
    }

    async fn list_pending_queued_work(
        &self,
        session_id: &str,
    ) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let now = current_epoch_ms();
        let mut batches = self
            .queued_work
            .lock()
            .expect("lock perf queued work")
            .iter()
            .filter(|entry| {
                entry.batch.session_id == session_id
                    && (entry.claim_token.is_none() || entry.claim_expires_at_ms <= now)
            })
            .map(|entry| entry.batch.clone())
            .collect::<Vec<_>>();
        batches.sort_by_key(|batch| batch.enqueue_seq);
        Ok(batches)
    }

    async fn save_session_meta(&self, meta: store::SessionMeta) -> Result<(), store::StoreError> {
        *self.session_meta.lock().expect("lock perf session meta") = Some(meta);
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<store::SessionMeta>, store::StoreError> {
        Ok(self
            .session_meta
            .lock()
            .expect("lock perf session meta")
            .clone())
    }

    async fn tombstone_nodes(&self, _ids: &[String]) -> Result<(), store::StoreError> {
        Ok(())
    }

    async fn vacuum(&self) -> Result<VacuumReport, store::StoreError> {
        Ok(VacuumReport::default())
    }

    async fn gc_unreachable(&self) -> Result<GcReport, store::StoreError> {
        Ok(GcReport::default())
    }
}
