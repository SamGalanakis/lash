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
    BlobRef, DeliveryPolicy, EmbeddedDurableTurnStore, GcReport, RuntimeCommit,
    RuntimeEffectJournalRecord, RuntimePersistence, RuntimeTurnCheckpoint, RuntimeTurnLease,
    SessionGraph, SessionNodeRecord, SessionReadScope, SessionStoreCreateRequest,
    SessionStoreFactory, SlotPolicy, StoreError, VacuumReport, current_epoch_ms,
};

#[derive(Clone)]
struct RuntimePerfQueuedBatch {
    batch: QueuedWorkBatch,
    claim_id: Option<String>,
    claim_token: Option<String>,
    claim_owner_id: Option<String>,
    claim_fencing_token: u64,
    claim_expires_at_ms: u64,
}

#[derive(Default)]
pub(crate) struct RuntimePerfStore {
    next_blob_id: AtomicU64,
    queued_work_next_seq: AtomicU64,
    session_head_meta: Mutex<Option<SessionHeadMeta>>,
    session_graph: Mutex<SessionGraph>,
    usage_deltas: Mutex<Vec<TokenLedgerEntry>>,
    session_meta: Mutex<Option<store::SessionMeta>>,
    runtime_turn_leases: Mutex<HashMap<(String, String), RuntimeTurnLease>>,
    runtime_turn_checkpoints: Mutex<HashMap<(String, String), RuntimeTurnCheckpoint>>,
    runtime_effect_journal: Mutex<HashMap<(String, String, String), RuntimeEffectJournalRecord>>,
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
}

#[derive(Clone)]
pub(crate) struct RuntimePerfStoreFactory {
    pub(crate) store: Arc<RuntimePerfStore>,
}

impl SessionStoreFactory for RuntimePerfStoreFactory {
    fn create_store(
        &self,
        _request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        Ok(Arc::clone(&self.store) as Arc<dyn RuntimePersistence>)
    }

    fn delete_session(&self, _session_id: &str) -> Result<(), String> {
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
            completed_turn,
            completed_queue_claims,
            committed_attachment_ids: _,
        } = commit;
        let mut meta_guard = self
            .session_head_meta
            .lock()
            .expect("lock perf session head meta");
        let actual = meta_guard.as_ref().map_or(0, |meta| meta.head_revision);
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
            session_id,
            head_revision,
            config,
            agent_frames,
            current_agent_frame_id,
            checkpoint_ref: Some(checkpoint_ref.clone()),
            leaf_node_id,
            graph_node_count,
            token_ledger: Vec::new(),
        });
        if let Some(completed) = completed_turn
            && let Some(lease_token) = completed.lease_token.as_deref()
        {
            let key = (completed.session_id.clone(), completed.turn_id.clone());
            let lease_matches = self
                .runtime_turn_leases
                .lock()
                .expect("lock perf runtime turn leases")
                .get(&key)
                .is_some_and(|lease| lease.lease_token == lease_token);
            if lease_matches {
                self.runtime_turn_leases
                    .lock()
                    .expect("lock perf runtime turn leases")
                    .remove(&key);
                self.runtime_turn_checkpoints
                    .lock()
                    .expect("lock perf runtime turn checkpoints")
                    .remove(&key);
                self.runtime_effect_journal
                    .lock()
                    .expect("lock perf runtime effect journal")
                    .retain(|(session_id, turn_id, _), _| {
                        session_id != &completed.session_id || turn_id != &completed.turn_id
                    });
            }
        }
        Ok(RuntimeCommitResult {
            head_revision,
            checkpoint_ref,
            manifest,
        })
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
            claim_owner_id: None,
            claim_fencing_token: 0,
            claim_expires_at_ms: 0,
        });
        queued.sort_by_key(|entry| entry.batch.enqueue_seq);
        Ok(stored)
    }

    async fn claim_ready_queued_work(
        &self,
        session_id: &str,
        owner_id: &str,
        boundary: QueuedWorkClaimBoundary,
        lease_ttl_ms: u64,
        max_batches: usize,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        if max_batches == 0 {
            return Ok(None);
        }
        let now = current_epoch_ms();
        let mut queued = self.queued_work.lock().expect("lock perf queued work");
        queued.sort_by_key(|entry| entry.batch.enqueue_seq);
        let first_index = queued.iter().position(|entry| {
            entry.batch.session_id == session_id
                && entry.batch.available_at_ms <= now
                && (entry.claim_token.is_none() || entry.claim_expires_at_ms <= now)
        });
        let Some(first_index) = first_index else {
            return Ok(None);
        };
        let first = queued[first_index].batch.clone();
        if boundary == QueuedWorkClaimBoundary::ActiveTurnCheckpoint
            && first.delivery_policy != DeliveryPolicy::EarliestSafeBoundary
        {
            return Ok(None);
        }
        let mut indices = vec![first_index];
        if first.slot_policy == SlotPolicy::Join {
            for (index, entry) in queued.iter().enumerate().skip(first_index + 1) {
                if indices.len() >= max_batches {
                    break;
                }
                if entry.batch.session_id != session_id
                    || entry.batch.available_at_ms > now
                    || (entry.claim_token.is_some() && entry.claim_expires_at_ms > now)
                    || entry.batch.slot_policy != SlotPolicy::Join
                    || entry.batch.delivery_policy != first.delivery_policy
                    || entry.batch.merge_key != first.merge_key
                {
                    break;
                }
                indices.push(index);
            }
        }
        let fencing_token = queued[first_index].claim_fencing_token.saturating_add(1);
        let claim_id = format!("perf-qwc:{}:{fencing_token}", first.enqueue_seq);
        let lease_token = format!("{session_id}:{owner_id}:{claim_id}:{now}");
        let expires_at = now.saturating_add(lease_ttl_ms);
        let mut batches = Vec::new();
        for index in indices {
            let entry = &mut queued[index];
            entry.claim_id = Some(claim_id.clone());
            entry.claim_token = Some(lease_token.clone());
            entry.claim_owner_id = Some(owner_id.to_string());
            entry.claim_fencing_token = entry.claim_fencing_token.saturating_add(1);
            entry.claim_expires_at_ms = expires_at;
            batches.push(entry.batch.clone());
        }
        Ok(Some(QueuedWorkClaim {
            session_id: session_id.to_string(),
            claim_id,
            owner_id: owner_id.to_string(),
            lease_token,
            fencing_token,
            claimed_at_epoch_ms: now,
            expires_at_epoch_ms: expires_at,
            batches,
        }))
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
                entry.claim_owner_id = None;
                entry.claim_expires_at_ms = 0;
            }
        }
        Ok(())
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

    fn embedded_durable_turn_store(&self) -> Option<&dyn EmbeddedDurableTurnStore> {
        Some(self)
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

#[async_trait::async_trait]
impl EmbeddedDurableTurnStore for RuntimePerfStore {
    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError> {
        let token = self.next_blob_id.fetch_add(1, Ordering::Relaxed);
        let lease = RuntimeTurnLease {
            schema_version: lash_core::store::RUNTIME_TURN_LEASE_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!("{session_id}:{turn_id}:{owner_id}:{token}"),
            fencing_token: token,
            claimed_at_epoch_ms: 0,
            expires_at_epoch_ms: lease_ttl_ms,
        };
        self.runtime_turn_leases
            .lock()
            .expect("lock perf runtime turn leases")
            .insert((session_id.to_string(), turn_id.to_string()), lease.clone());
        Ok(lease)
    }

    async fn renew_runtime_turn_lease(
        &self,
        lease: &RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError> {
        let key = (lease.session_id.clone(), lease.turn_id.clone());
        let mut leases = self
            .runtime_turn_leases
            .lock()
            .expect("lock perf runtime turn leases");
        leases
            .get(&key)
            .filter(|current| current.lease_token == lease.lease_token)
            .ok_or_else(|| StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        let renewed = RuntimeTurnLease {
            expires_at_epoch_ms: lease.expires_at_epoch_ms.saturating_add(lease_ttl_ms),
            ..lease.clone()
        };
        leases.insert(key, renewed.clone());
        Ok(renewed)
    }

    async fn abandon_runtime_turn_lease(&self, lease: &RuntimeTurnLease) -> Result<(), StoreError> {
        let key = (lease.session_id.clone(), lease.turn_id.clone());
        let mut leases = self
            .runtime_turn_leases
            .lock()
            .expect("lock perf runtime turn leases");
        if leases.get(&key).is_some_and(|current| {
            current.owner_id == lease.owner_id
                && current.lease_token == lease.lease_token
                && current.fencing_token == lease.fencing_token
        }) {
            leases.remove(&key);
        }
        Ok(())
    }

    async fn save_runtime_turn_checkpoint(
        &self,
        lease: &RuntimeTurnLease,
        checkpoint: RuntimeTurnCheckpoint,
    ) -> Result<(), StoreError> {
        let key = (lease.session_id.clone(), lease.turn_id.clone());
        self.runtime_turn_leases
            .lock()
            .expect("lock perf runtime turn leases")
            .get(&key)
            .filter(|current| current.lease_token == lease.lease_token)
            .ok_or_else(|| StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        self.runtime_turn_checkpoints
            .lock()
            .expect("lock perf runtime turn checkpoints")
            .insert(key, checkpoint);
        Ok(())
    }

    async fn load_runtime_turn_checkpoint(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<RuntimeTurnCheckpoint>, StoreError> {
        Ok(self
            .runtime_turn_checkpoints
            .lock()
            .expect("lock perf runtime turn checkpoints")
            .get(&(session_id.to_string(), turn_id.to_string()))
            .cloned())
    }

    async fn save_runtime_effect_outcome(
        &self,
        lease: &RuntimeTurnLease,
        record: RuntimeEffectJournalRecord,
    ) -> Result<(), StoreError> {
        let key = (lease.session_id.clone(), lease.turn_id.clone());
        self.runtime_turn_leases
            .lock()
            .expect("lock perf runtime turn leases")
            .get(&key)
            .filter(|current| current.lease_token == lease.lease_token)
            .ok_or_else(|| StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        self.runtime_effect_journal
            .lock()
            .expect("lock perf runtime effect journal")
            .insert(
                (
                    record.session_id.clone(),
                    record.turn_id.clone(),
                    record.replay_key.clone(),
                ),
                record,
            );
        Ok(())
    }

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        replay_key: &str,
    ) -> Result<Option<RuntimeEffectJournalRecord>, StoreError> {
        Ok(self
            .runtime_effect_journal
            .lock()
            .expect("lock perf runtime effect journal")
            .get(&(
                session_id.to_string(),
                turn_id.to_string(),
                replay_key.to_string(),
            ))
            .cloned())
    }
}
