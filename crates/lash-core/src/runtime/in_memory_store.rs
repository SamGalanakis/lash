//! Public in-memory `RuntimePersistence` + `SessionStoreFactory`.
//!
//! Explicitly-wired ephemeral storage for inline-tier hosts that run background
//! processes without durable backing: a `process` started in a turn (or by a
//! trigger) is executed by the lease-protected worker, which rebuilds its
//! session from the store factory — so even an in-memory host needs a factory.
//! This is the named, opt-in counterpart to a durable factory; there is no
//! silent in-memory default. Holds the same `RuntimePersistence` contract as the
//! SQLite backend (verified by the `runtime_persistence` conformance suite).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::usage::merge_usage_delta_entries;
use super::{SessionStoreCreateRequest, SessionStoreFactory};
use crate::DurabilityTier;
use crate::store::RuntimePersistence;

#[derive(Clone)]
struct InMemoryQueuedBatch {
    batch: crate::QueuedWorkBatch,
    claim_id: Option<String>,
    claim_token: Option<String>,
    claim_owner_id: Option<String>,
    claim_fencing_token: u64,
    claim_expires_at_ms: u64,
}

#[derive(Default)]
pub struct InMemorySessionStore {
    pub(crate) session_head_meta: Mutex<Option<crate::SessionHeadMeta>>,
    pub(crate) session_meta: Mutex<Option<crate::SessionMeta>>,
    pub(crate) session_graph: Mutex<crate::SessionGraph>,
    pub(crate) checkpoint: Mutex<Option<crate::HydratedSessionCheckpoint>>,
    pub(crate) usage_deltas: Mutex<Vec<crate::TokenLedgerEntry>>,
    pub(crate) runtime_commit_count: Mutex<usize>,
    runtime_turn_checkpoints:
        Mutex<std::collections::HashMap<(String, String), crate::RuntimeTurnCheckpoint>>,
    runtime_turn_leases:
        Mutex<std::collections::HashMap<(String, String), crate::RuntimeTurnLease>>,
    runtime_effect_journal: Mutex<
        std::collections::HashMap<(String, String, String), crate::RuntimeEffectJournalRecord>,
    >,
    runtime_turn_commits: Mutex<
        std::collections::HashMap<(String, String), (String, crate::store::RuntimeCommitResult)>,
    >,
    // Save/renew/abandon counters are observed only by lash-core tests; they
    // exist (and are incremented) solely under `cfg(test)` so the shipped store
    // carries no dead test-only state.
    #[cfg(test)]
    runtime_turn_checkpoint_save_count: Mutex<usize>,
    #[cfg(test)]
    runtime_effect_journal_save_count: Mutex<usize>,
    #[cfg(test)]
    runtime_turn_lease_renew_count: Mutex<usize>,
    #[cfg(test)]
    runtime_turn_lease_abandon_count: Mutex<usize>,
    queued_work: Mutex<Vec<InMemoryQueuedBatch>>,
    queued_work_next_seq: Mutex<u64>,
}

crate::impl_noop_attachment_manifest!(InMemorySessionStore);

#[async_trait::async_trait]
impl crate::store::RuntimePersistence for InMemorySessionStore {
    async fn load_session(
        &self,
        scope: crate::store::SessionReadScope,
    ) -> Result<Option<crate::store::PersistedSessionRead>, crate::store::StoreError> {
        let Some(meta) = self.session_head_meta.lock().expect("lock store").clone() else {
            return Ok(None);
        };
        let mut graph = self.session_graph.lock().expect("lock graph").clone();
        if let crate::store::SessionReadScope::ActivePath { leaf_node_id } = scope {
            if let Some(leaf_node_id) = leaf_node_id.or_else(|| meta.leaf_node_id.clone()) {
                graph.set_leaf_node_id(Some(leaf_node_id));
            }
            graph = graph.fork_current_path();
        }
        Ok(Some(crate::store::PersistedSessionRead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            config: meta.config,
            agent_frames: meta.agent_frames,
            current_agent_frame_id: meta.current_agent_frame_id,
            graph,
            checkpoint_ref: meta.checkpoint_ref,
            checkpoint: self.checkpoint.lock().expect("lock checkpoint").clone(),
            token_ledger: merge_usage_delta_entries(
                self.usage_deltas.lock().expect("lock usage deltas").clone(),
            ),
        }))
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<crate::SessionNodeRecord>, crate::store::StoreError> {
        Ok(self
            .session_graph
            .lock()
            .expect("lock graph")
            .find_node(node_id)
            .cloned())
    }

    async fn commit_runtime_state(
        &self,
        commit: crate::store::RuntimeCommit,
    ) -> Result<crate::store::RuntimeCommitResult, crate::store::StoreError> {
        let mut meta = self.session_head_meta.lock().expect("lock store");
        let actual = meta.as_ref().map_or(0, |meta| meta.head_revision);
        if let Some(bound) = meta.as_ref().map(|meta| meta.session_id.clone())
            && bound != commit.session_id
        {
            return Err(crate::store::StoreError::SessionBindingMismatch {
                bound_session_id: bound,
                attempted_session_id: commit.session_id,
            });
        }
        if let Some(completed) = &commit.completed_turn {
            if completed.session_id != commit.session_id {
                return Err(crate::store::StoreError::RuntimeTurnLeaseExpired {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
            let key = (completed.session_id.clone(), completed.turn_id.clone());
            if let Some((stored_hash, result)) = self
                .runtime_turn_commits
                .lock()
                .expect("lock runtime turn commits")
                .get(&key)
                .cloned()
            {
                if stored_hash == completed.turn_commit_hash {
                    return Ok(result);
                }
                return Err(crate::store::StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
        }
        if commit.expected_head_revision.is_some() && commit.expected_head_revision != Some(actual)
        {
            return Err(crate::store::StoreError::HeadRevisionConflict {
                expected: commit.expected_head_revision,
                actual,
            });
        }
        if let Some(completed) = &commit.completed_turn {
            if let Some(lease_token) = completed.lease_token.as_deref() {
                let lease_key = (completed.session_id.clone(), completed.turn_id.clone());
                let lease_matches = self
                    .runtime_turn_leases
                    .lock()
                    .expect("lock runtime turn leases")
                    .get(&lease_key)
                    .is_some_and(|lease| {
                        lease.lease_token == lease_token
                            && lease.expires_at_epoch_ms > current_epoch_ms()
                    });
                if !lease_matches {
                    return Err(crate::store::StoreError::RuntimeTurnLeaseExpired {
                        session_id: completed.session_id.clone(),
                        turn_id: completed.turn_id.clone(),
                    });
                }
            }
        }
        for completed in &commit.completed_queue_claims {
            let mut queued = self.queued_work.lock().expect("lock queued work");
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
                return Err(crate::store::StoreError::QueuedWorkClaimExpired {
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
        let mut graph = self.session_graph.lock().expect("lock graph");
        let leaf_node_id = match &commit.graph {
            crate::store::GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
            crate::store::GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                graph.extend_node_records(nodes.iter().cloned());
                leaf_node_id.clone()
            }
            crate::store::GraphCommitDelta::ReplaceFull(next) => {
                *graph = next.clone();
                next.leaf_node_id.clone()
            }
        };
        self.usage_deltas
            .lock()
            .expect("lock usage deltas")
            .extend(commit.usage_deltas.iter().cloned());
        let checkpoint_ref = crate::BlobRef(format!("recording-checkpoint-{}", actual + 1));
        let manifest = crate::store::SessionCheckpoint {
            turn_state: commit.checkpoint.turn_state.clone(),
            tool_state_ref: commit.checkpoint.tool_state_ref.clone(),
            plugin_snapshot_ref: commit.checkpoint.plugin_snapshot_ref.clone(),
            plugin_snapshot_revision: commit.checkpoint.plugin_snapshot_revision,
            execution_state_ref: commit.checkpoint.execution_state_ref.clone(),
        };
        *self.checkpoint.lock().expect("lock checkpoint") = Some(commit.checkpoint);
        if let Some(completed) = &commit.completed_turn
            && let Some(lease_token) = completed.lease_token.as_deref()
        {
            let lease_key = (completed.session_id.clone(), completed.turn_id.clone());
            let lease_matches = self
                .runtime_turn_leases
                .lock()
                .expect("lock runtime turn leases")
                .get(&lease_key)
                .is_some_and(|lease| lease.lease_token == lease_token);
            if lease_matches {
                self.runtime_turn_checkpoints
                    .lock()
                    .expect("lock runtime turn checkpoints")
                    .remove(&lease_key);
                self.runtime_turn_leases
                    .lock()
                    .expect("lock runtime turn leases")
                    .remove(&lease_key);
                self.runtime_effect_journal
                    .lock()
                    .expect("lock runtime effect journal")
                    .retain(|(session_id, turn_id, _), _| {
                        session_id != &completed.session_id || turn_id != &completed.turn_id
                    });
            }
        }
        let head_revision = actual + 1;
        *meta = Some(crate::SessionHeadMeta {
            session_id: commit.session_id,
            head_revision,
            config: commit.config,
            agent_frames: commit.agent_frames,
            current_agent_frame_id: commit.current_agent_frame_id,
            checkpoint_ref: Some(checkpoint_ref.clone()),
            leaf_node_id,
            graph_node_count: graph.nodes.len(),
            token_ledger: Vec::new(),
        });
        *self
            .runtime_commit_count
            .lock()
            .expect("lock runtime commit count") += 1;
        let result = crate::store::RuntimeCommitResult {
            head_revision,
            checkpoint_ref,
            manifest,
        };
        if let Some(completed) = &commit.completed_turn {
            self.runtime_turn_commits
                .lock()
                .expect("lock runtime turn commits")
                .insert(
                    (completed.session_id.clone(), completed.turn_id.clone()),
                    (completed.turn_commit_hash.clone(), result.clone()),
                );
        }
        Ok(result)
    }

    fn embedded_durable_turn_store(&self) -> Option<&dyn crate::runtime::EmbeddedDurableTurnStore> {
        Some(self)
    }

    async fn enqueue_queued_work(
        &self,
        batch: crate::QueuedWorkBatchDraft,
    ) -> Result<crate::QueuedWorkBatch, crate::store::StoreError> {
        let mut queued = self.queued_work.lock().expect("lock queued work");
        if let Some(source_key) = batch.source_key.as_deref()
            && let Some(existing) = queued.iter().find(|entry| {
                entry.batch.session_id == batch.session_id
                    && entry.batch.source_key.as_deref() == Some(source_key)
            })
        {
            return Ok(existing.batch.clone());
        }
        let mut next_seq = self
            .queued_work_next_seq
            .lock()
            .expect("lock queued work seq");
        *next_seq = next_seq.saturating_add(1);
        let batch_id = format!("recording-qwb-{next_seq}");
        let enqueued_at_ms = current_epoch_ms();
        let payloads = batch.payloads;
        let stored = crate::QueuedWorkBatch {
            batch_id: batch_id.clone(),
            session_id: batch.session_id,
            enqueue_seq: *next_seq,
            source_key: batch.source_key,
            delivery_policy: batch.delivery_policy,
            slot_policy: batch.slot_policy,
            merge_key: batch.merge_key,
            available_at_ms: batch.available_at_ms,
            enqueued_at_ms,
            items: payloads
                .into_iter()
                .enumerate()
                .map(|(index, payload)| crate::QueuedWorkItem {
                    item_id: format!("{batch_id}:item:{index}"),
                    payload,
                })
                .collect(),
        };
        queued.push(InMemoryQueuedBatch {
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
        boundary: crate::QueuedWorkClaimBoundary,
        lease_ttl_ms: u64,
        max_batches: usize,
    ) -> Result<Option<crate::QueuedWorkClaim>, crate::store::StoreError> {
        if max_batches == 0 {
            return Ok(None);
        }
        let now = current_epoch_ms();
        let mut queued = self.queued_work.lock().expect("lock queued work");
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
        if boundary == crate::QueuedWorkClaimBoundary::ActiveTurnCheckpoint
            && first.delivery_policy != crate::DeliveryPolicy::EarliestSafeBoundary
        {
            return Ok(None);
        }
        let mut indices = vec![first_index];
        if first.slot_policy == crate::SlotPolicy::Join {
            for (index, entry) in queued.iter().enumerate().skip(first_index + 1) {
                if indices.len() >= max_batches {
                    break;
                }
                if entry.batch.session_id != session_id
                    || entry.batch.available_at_ms > now
                    || (entry.claim_token.is_some() && entry.claim_expires_at_ms > now)
                    || entry.batch.slot_policy != crate::SlotPolicy::Join
                    || entry.batch.delivery_policy != first.delivery_policy
                    || entry.batch.merge_key != first.merge_key
                {
                    break;
                }
                indices.push(index);
            }
        }
        let fencing_token = queued[first_index].claim_fencing_token.saturating_add(1);
        let claim_id = format!("recording-qwc:{}:{fencing_token}", first.enqueue_seq);
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
        Ok(Some(crate::QueuedWorkClaim {
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
        claim: &crate::QueuedWorkClaim,
        lease_ttl_ms: u64,
    ) -> Result<crate::QueuedWorkClaim, crate::store::StoreError> {
        let mut queued = self.queued_work.lock().expect("lock queued work");
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
            return Err(crate::store::StoreError::QueuedWorkClaimExpired {
                session_id: claim.session_id.clone(),
                claim_id: claim.claim_id.clone(),
            });
        }
        Ok(crate::QueuedWorkClaim {
            expires_at_epoch_ms: expires_at,
            ..claim.clone()
        })
    }

    async fn abandon_queued_work_claim(
        &self,
        claim: &crate::QueuedWorkClaim,
    ) -> Result<(), crate::store::StoreError> {
        let mut queued = self.queued_work.lock().expect("lock queued work");
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

    async fn list_queued_work(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::QueuedWorkBatch>, crate::store::StoreError> {
        let mut batches = self
            .queued_work
            .lock()
            .expect("lock queued work")
            .iter()
            .filter(|entry| entry.batch.session_id == session_id)
            .map(|entry| entry.batch.clone())
            .collect::<Vec<_>>();
        batches.sort_by_key(|batch| batch.enqueue_seq);
        Ok(batches)
    }

    async fn save_session_meta(
        &self,
        meta: crate::store::SessionMeta,
    ) -> Result<(), crate::store::StoreError> {
        *self.session_meta.lock().expect("lock session meta") = Some(meta);
        Ok(())
    }

    async fn load_session_meta(
        &self,
    ) -> Result<Option<crate::store::SessionMeta>, crate::store::StoreError> {
        Ok(self.session_meta.lock().expect("lock session meta").clone())
    }

    async fn tombstone_nodes(&self, _ids: &[String]) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    async fn vacuum(&self) -> Result<crate::store::VacuumReport, crate::store::StoreError> {
        Ok(crate::store::VacuumReport::default())
    }

    async fn gc_unreachable(&self) -> Result<crate::store::GcReport, crate::store::StoreError> {
        Ok(crate::store::GcReport::default())
    }
}

#[async_trait::async_trait]
impl crate::runtime::EmbeddedDurableTurnStore for InMemorySessionStore {
    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<crate::RuntimeTurnLease, crate::store::StoreError> {
        let mut leases = self
            .runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases");
        let key = (session_id.to_string(), turn_id.to_string());
        let next_fencing = match leases.get(&key) {
            Some(active)
                if active.expires_at_epoch_ms > current_epoch_ms()
                    && active.owner_id != owner_id =>
            {
                return Err(crate::store::StoreError::RuntimeTurnLeaseConflict {
                    session_id: session_id.to_string(),
                    turn_id: turn_id.to_string(),
                    owner_id: active.owner_id.clone(),
                    expires_at_epoch_ms: active.expires_at_epoch_ms,
                });
            }
            Some(previous) => previous.fencing_token.saturating_add(1),
            None => 1,
        };
        let lease = crate::RuntimeTurnLease {
            schema_version: crate::RUNTIME_TURN_LEASE_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!("{session_id}:{turn_id}:{owner_id}:{next_fencing}"),
            fencing_token: next_fencing,
            claimed_at_epoch_ms: current_epoch_ms(),
            expires_at_epoch_ms: current_epoch_ms().saturating_add(lease_ttl_ms),
        };
        leases.insert(key, lease.clone());
        Ok(lease)
    }

    async fn renew_runtime_turn_lease(
        &self,
        lease: &crate::RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> Result<crate::RuntimeTurnLease, crate::store::StoreError> {
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .get(&(lease.session_id.clone(), lease.turn_id.clone()))
            .filter(|current| {
                current.lease_token == lease.lease_token
                    && current.expires_at_epoch_ms > current_epoch_ms()
            })
            .ok_or_else(|| crate::store::StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        let renewed = crate::RuntimeTurnLease {
            expires_at_epoch_ms: current_epoch_ms().saturating_add(lease_ttl_ms),
            ..lease.clone()
        };
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .insert(
                (renewed.session_id.clone(), renewed.turn_id.clone()),
                renewed.clone(),
            );
        #[cfg(test)]
        {
            *self
                .runtime_turn_lease_renew_count
                .lock()
                .expect("lock runtime turn lease renew count") += 1;
        }
        Ok(renewed)
    }

    async fn abandon_runtime_turn_lease(
        &self,
        lease: &crate::RuntimeTurnLease,
    ) -> Result<(), crate::store::StoreError> {
        let key = (lease.session_id.clone(), lease.turn_id.clone());
        let mut leases = self
            .runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases");
        if leases.get(&key).is_some_and(|current| {
            current.owner_id == lease.owner_id
                && current.lease_token == lease.lease_token
                && current.fencing_token == lease.fencing_token
        }) {
            leases.remove(&key);
        }
        #[cfg(test)]
        {
            *self
                .runtime_turn_lease_abandon_count
                .lock()
                .expect("lock runtime turn lease abandon count") += 1;
        }
        Ok(())
    }

    async fn save_runtime_turn_checkpoint(
        &self,
        lease: &crate::RuntimeTurnLease,
        checkpoint: crate::RuntimeTurnCheckpoint,
    ) -> Result<(), crate::store::StoreError> {
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .get(&(lease.session_id.clone(), lease.turn_id.clone()))
            .filter(|current| {
                current.lease_token == lease.lease_token
                    && current.expires_at_epoch_ms > current_epoch_ms()
            })
            .ok_or_else(|| crate::store::StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        self.runtime_turn_checkpoints
            .lock()
            .expect("lock runtime turn checkpoints")
            .insert(
                (checkpoint.session_id.clone(), checkpoint.turn_id.clone()),
                checkpoint,
            );
        #[cfg(test)]
        {
            *self
                .runtime_turn_checkpoint_save_count
                .lock()
                .expect("lock runtime turn checkpoint save count") += 1;
        }
        Ok(())
    }

    async fn load_runtime_turn_checkpoint(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<crate::RuntimeTurnCheckpoint>, crate::store::StoreError> {
        Ok(self
            .runtime_turn_checkpoints
            .lock()
            .expect("lock runtime turn checkpoints")
            .get(&(session_id.to_string(), turn_id.to_string()))
            .cloned())
    }

    async fn save_runtime_effect_outcome(
        &self,
        lease: &crate::RuntimeTurnLease,
        record: crate::RuntimeEffectJournalRecord,
    ) -> Result<(), crate::store::StoreError> {
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .get(&(lease.session_id.clone(), lease.turn_id.clone()))
            .filter(|current| {
                current.lease_token == lease.lease_token
                    && current.expires_at_epoch_ms > current_epoch_ms()
            })
            .ok_or_else(|| crate::store::StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        self.runtime_effect_journal
            .lock()
            .expect("lock runtime effect journal")
            .insert(
                (
                    record.session_id.clone(),
                    record.turn_id.clone(),
                    record.replay_key.clone(),
                ),
                record,
            );
        #[cfg(test)]
        {
            *self
                .runtime_effect_journal_save_count
                .lock()
                .expect("lock runtime effect journal save count") += 1;
        }
        Ok(())
    }

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        replay_key: &str,
    ) -> Result<Option<crate::RuntimeEffectJournalRecord>, crate::store::StoreError> {
        Ok(self
            .runtime_effect_journal
            .lock()
            .expect("lock runtime effect journal")
            .get(&(
                session_id.to_string(),
                turn_id.to_string(),
                replay_key.to_string(),
            ))
            .cloned())
    }
}

fn current_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

/// Test-only introspection: call counters and a head-meta seeder used by the
/// lash-core runtime tests. Compiled only under `cfg(test)`, so the shipped
/// embedding surface carries none of it.
#[cfg(test)]
impl InMemorySessionStore {
    pub(crate) async fn save_session_head_meta(&self, meta: crate::SessionHeadMeta) {
        *self.session_head_meta.lock().expect("lock store") = Some(meta);
    }

    pub(crate) fn runtime_turn_checkpoint_count(&self) -> usize {
        self.runtime_turn_checkpoints
            .lock()
            .expect("lock runtime turn checkpoints")
            .len()
    }

    pub(crate) fn runtime_effect_journal_count(&self) -> usize {
        self.runtime_effect_journal
            .lock()
            .expect("lock runtime effect journal")
            .len()
    }

    pub(crate) fn runtime_turn_checkpoint_save_count(&self) -> usize {
        *self
            .runtime_turn_checkpoint_save_count
            .lock()
            .expect("lock runtime turn checkpoint save count")
    }

    pub(crate) fn runtime_effect_journal_save_count(&self) -> usize {
        *self
            .runtime_effect_journal_save_count
            .lock()
            .expect("lock runtime effect journal save count")
    }

    pub(crate) fn runtime_turn_lease_renew_count(&self) -> usize {
        *self
            .runtime_turn_lease_renew_count
            .lock()
            .expect("lock runtime turn lease renew count")
    }

    pub(crate) fn runtime_turn_lease_abandon_count(&self) -> usize {
        *self
            .runtime_turn_lease_abandon_count
            .lock()
            .expect("lock runtime turn lease abandon count")
    }
}

/// Session-id-keyed factory: the same in-memory store is returned for a given
/// session across opens (so a worker rebuild sees the session's state), and a
/// fresh store is created on first use. Inline durability tier.
#[derive(Clone, Default)]
pub struct InMemorySessionStoreFactory {
    stores: Arc<Mutex<HashMap<String, Arc<InMemorySessionStore>>>>,
}

impl InMemorySessionStoreFactory {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SessionStoreFactory for InMemorySessionStoreFactory {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Inline
    }

    fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        let mut stores = self.stores.lock().expect("in-memory store factory");
        let store = stores
            .entry(request.session_id.clone())
            .or_insert_with(|| {
                let store = Arc::new(InMemorySessionStore::default());
                *store.session_meta.lock().expect("lock session meta") = Some(crate::SessionMeta {
                    session_id: request.session_id.clone(),
                    session_name: request.session_id.clone(),
                    created_at: "1970-01-01T00:00:00Z".to_string(),
                    model: request.policy.model.id.clone(),
                    cwd: None,
                    relation: request.relation.clone(),
                });
                store
            })
            .clone();
        Ok(store as Arc<dyn RuntimePersistence>)
    }

    fn delete_session(&self, session_id: &str) -> Result<(), String> {
        self.stores
            .lock()
            .expect("in-memory store factory")
            .remove(session_id);
        Ok(())
    }
}
