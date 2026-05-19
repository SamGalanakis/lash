use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lash::usage::TokenLedgerEntry;
use lash_core::store;
use lash_core::{
    BlobRef, GcReport, GraphCommitDelta, PersistedSessionRead, RuntimeCommit, RuntimeCommitResult,
    RuntimeEffectJournalRecord, RuntimePersistence, RuntimeTurnCheckpoint, RuntimeTurnLease,
    SessionCheckpoint, SessionGraph, SessionHeadMeta, SessionNodeRecord, SessionReadScope,
    SessionStoreCreateRequest, SessionStoreFactory, StoreError, VacuumReport,
};

#[derive(Default)]
pub(crate) struct RuntimePerfStore {
    next_blob_id: AtomicU64,
    session_head_meta: Mutex<Option<SessionHeadMeta>>,
    session_graph: Mutex<SessionGraph>,
    usage_deltas: Mutex<Vec<TokenLedgerEntry>>,
    session_meta: Mutex<Option<store::SessionMeta>>,
    runtime_turn_leases: Mutex<HashMap<(String, String), RuntimeTurnLease>>,
    runtime_turn_checkpoints: Mutex<HashMap<(String, String), RuntimeTurnCheckpoint>>,
    runtime_effect_journal: Mutex<HashMap<(String, String, String), RuntimeEffectJournalRecord>>,
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
}

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
            graph: graph_delta,
            checkpoint,
            usage_deltas,
            completed_turn,
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
            checkpoint_ref: Some(checkpoint_ref.clone()),
            leaf_node_id,
            graph_node_count,
            token_ledger: Vec::new(),
        });
        if let Some(completed) = completed_turn {
            let key = (completed.session_id.clone(), completed.turn_id.clone());
            let lease_matches = self
                .runtime_turn_leases
                .lock()
                .expect("lock perf runtime turn leases")
                .get(&key)
                .is_some_and(|lease| lease.lease_token == completed.lease_token);
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

    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError> {
        let token = self.next_blob_id.fetch_add(1, Ordering::Relaxed);
        let lease = RuntimeTurnLease {
            schema_version: lash_core::RUNTIME_TURN_LEASE_SCHEMA_VERSION,
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
                    record.idempotency_key.clone(),
                ),
                record,
            );
        Ok(())
    }

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<RuntimeEffectJournalRecord>, StoreError> {
        Ok(self
            .runtime_effect_journal
            .lock()
            .expect("lock perf runtime effect journal")
            .get(&(
                session_id.to_string(),
                turn_id.to_string(),
                idempotency_key.to_string(),
            ))
            .cloned())
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
