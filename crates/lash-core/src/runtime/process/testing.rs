#![cfg(any(test, feature = "testing"))]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::plugin::PluginError;

use super::events::{
    ProcessAwaitOutput, ProcessEvent, ProcessEventAppendRequest, ProcessEventAppendResult,
    ProcessTerminalState,
};
use super::model::{
    PROCESS_LEASE_SCHEMA_VERSION, ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant,
    ProcessHandleGrantEntry, ProcessLease, ProcessLeaseCompletion, ProcessRecord,
    ProcessRegistration, ProcessScope, ProcessScopeId, ProcessSessionDeleteReport,
};
use super::registry::ProcessRegistry;
use super::time::current_epoch_ms;
use super::validation::{
    ensure_core_event_types, prepare_process_event_append, process_registration_hash,
    validate_process_registration,
};

/// In-memory process registry for core tests.
pub struct TestLocalProcessRegistry {
    managed: Arc<Mutex<ManagedProcessMap>>,
    grants: Arc<Mutex<ManagedGrantMap>>,
    leases: Arc<Mutex<ManagedLeaseMap>>,
}

impl Default for TestLocalProcessRegistry {
    fn default() -> Self {
        Self {
            managed: Arc::new(Mutex::new(HashMap::new())),
            grants: Arc::new(Mutex::new(HashMap::new())),
            leases: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

type ManagedProcessMap = HashMap<String, ManagedProcessRecord>;
type ManagedGrantMap = HashMap<ProcessScopeId, HashMap<String, ProcessHandleGrant>>;
type ManagedLeaseMap = HashMap<String, ProcessLease>;

struct ManagedProcessRecord {
    record: ProcessRecord,
    events: Vec<ProcessEvent>,
    keyed_events: HashMap<String, (String, ProcessEvent)>,
    acked_wakes: HashSet<u64>,
    notify: Arc<tokio::sync::Notify>,
}

impl TestLocalProcessRegistry {
    async fn insert_process(
        &self,
        mut registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError> {
        ensure_core_event_types(&mut registration);
        validate_process_registration(&registration)?;
        let registration_hash = process_registration_hash(&registration)?;
        let mut managed = self.managed.lock().await;
        if let Some(existing) = managed.get(&registration.id) {
            if existing.record.registration_hash == registration_hash {
                return Ok(existing.record.clone());
            }
            return Err(PluginError::Session(format!(
                "process `{}` registration hash conflict: existing {}, new {}",
                registration.id, existing.record.registration_hash, registration_hash
            )));
        }
        let id = registration.id.clone();
        let record = ProcessRecord::from_prepared_registration(
            registration,
            registration_hash,
            current_epoch_ms(),
        );
        managed.insert(
            id.clone(),
            ManagedProcessRecord {
                record: record.clone(),
                events: Vec::new(),
                keyed_events: HashMap::new(),
                acked_wakes: HashSet::new(),
                notify: Arc::new(tokio::sync::Notify::new()),
            },
        );
        Ok(record)
    }
}

#[async_trait::async_trait]
impl ProcessRegistry for TestLocalProcessRegistry {
    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError> {
        self.insert_process(registration).await
    }

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, PluginError> {
        let mut managed = self.managed.lock().await;
        let Some(record) = managed.get_mut(process_id) else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        record.record.external_ref = Some(external_ref);
        record.record.updated_at_ms = current_epoch_ms();
        Ok(record.record.clone())
    }

    async fn grant_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, PluginError> {
        if self.get_process(process_id).await.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let grant = ProcessHandleGrant {
            session_id: owner_scope.session_id.clone(),
            process_id: process_id.to_string(),
            descriptor,
        };
        self.grants
            .lock()
            .await
            .entry(owner_scope.id())
            .or_default()
            .insert(process_id.to_string(), grant.clone());
        Ok(grant)
    }

    async fn revoke_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
    ) -> Result<(), PluginError> {
        if let Some(session_grants) = self.grants.lock().await.get_mut(&owner_scope.id()) {
            session_grants.remove(process_id);
        }
        Ok(())
    }

    async fn transfer_handle_grants(
        &self,
        from_scope: &ProcessScope,
        to_scope: &ProcessScope,
        process_ids: &[String],
    ) -> Result<(), PluginError> {
        let mut grants = self.grants.lock().await;
        let from_scope_id = from_scope.id();
        let to_scope_id = to_scope.id();
        for process_id in process_ids {
            let grant = grants
                .get_mut(&from_scope_id)
                .and_then(|session_grants| session_grants.remove(process_id))
                .ok_or_else(|| {
                    PluginError::Session(format!(
                        "process handle `{process_id}` is not granted to session `{}`",
                        from_scope.session_id
                    ))
                })?;
            grants.entry(to_scope_id.clone()).or_default().insert(
                process_id.clone(),
                ProcessHandleGrant {
                    session_id: to_scope.session_id.clone(),
                    process_id: process_id.clone(),
                    descriptor: grant.descriptor,
                },
            );
        }
        Ok(())
    }

    async fn list_handle_grants(
        &self,
        owner_scope: &ProcessScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
        let grants = self
            .grants
            .lock()
            .await
            .get(&owner_scope.id())
            .cloned()
            .unwrap_or_default();
        let managed = self.managed.lock().await;
        let mut entries = grants
            .into_values()
            .filter_map(|grant| {
                managed
                    .get(&grant.process_id)
                    .map(|record| (grant, record.record.clone()))
            })
            .collect::<Vec<_>>();
        entries.sort_by(|(left, _), (right, _)| left.process_id.cmp(&right.process_id));
        Ok(entries)
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, PluginError> {
        if self.get_process(process_id).await.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let grants = self.grants.lock().await;
        let mut entries = grants
            .values()
            .filter_map(|session_grants| session_grants.get(process_id).cloned())
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.session_id.cmp(&right.session_id));
        Ok(entries)
    }

    async fn delete_session_process_state(
        &self,
        session_id: &str,
    ) -> Result<ProcessSessionDeleteReport, PluginError> {
        let removed = {
            let mut grants = self.grants.lock().await;
            let mut removed = Vec::new();
            grants.retain(|_, session_grants| {
                if session_grants
                    .values()
                    .next()
                    .is_some_and(|grant| grant.session_id == session_id)
                {
                    removed.extend(session_grants.drain().map(|(_, grant)| grant));
                    false
                } else {
                    true
                }
            });
            removed
        };
        let managed = self.managed.lock().await;
        let grants = self.grants.lock().await;
        let mut cancel_process_ids = Vec::new();
        let mut preserved_process_ids = Vec::new();
        for grant in &removed {
            let Some(record) = managed.get(&grant.process_id) else {
                continue;
            };
            if record.record.is_terminal() {
                continue;
            }
            let still_granted = grants
                .values()
                .any(|session_grants| session_grants.contains_key(&grant.process_id));
            if still_granted {
                preserved_process_ids.push(grant.process_id.clone());
            } else {
                cancel_process_ids.push(grant.process_id.clone());
            }
        }
        cancel_process_ids.sort();
        cancel_process_ids.dedup();
        preserved_process_ids.sort();
        preserved_process_ids.dedup();
        Ok(ProcessSessionDeleteReport {
            session_id: session_id.to_string(),
            revoked_handle_count: removed.len(),
            deleted_wake_count: 0,
            cancel_process_ids,
            preserved_process_ids,
        })
    }

    async fn append_event(
        &self,
        process_id: &str,
        request: ProcessEventAppendRequest,
    ) -> Result<ProcessEventAppendResult, PluginError> {
        let mut managed = self.managed.lock().await;
        let Some(record) = managed.get_mut(process_id) else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        let replay_lookup = request
            .replay
            .as_ref()
            .and_then(|replay| record.keyed_events.get(replay.key.as_str()))
            .map(|(hash, event)| (hash.clone(), event.clone()));
        let sequence = record.events.len() as u64 + 1;
        let prepared = prepare_process_event_append(
            &record.record,
            request,
            sequence,
            replay_lookup,
            current_epoch_ms(),
        )?;
        if prepared.replayed {
            return Ok(ProcessEventAppendResult {
                event: prepared.event,
                wake_delivery: prepared.wake_delivery,
            });
        }
        let event = prepared.event;
        if let Some(terminal) = prepared.terminal_update.clone() {
            record.record.terminal = Some(terminal);
        }
        record.record.updated_at_ms = prepared.occurred_at_ms;
        record.events.push(event.clone());
        if let Some(replay) = event.invocation.replay.clone() {
            record
                .keyed_events
                .insert(replay.key, (prepared.payload_hash, event.clone()));
        }
        let wake_delivery = prepared.wake_delivery;
        record.notify.notify_waiters();
        Ok(ProcessEventAppendResult {
            event,
            wake_delivery,
        })
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        let managed = self.managed.lock().await;
        let Some(record) = managed.get(process_id) else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        Ok(record
            .events
            .iter()
            .filter(|event| event.sequence > after_sequence)
            .cloned()
            .collect())
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        let managed = self.managed.lock().await;
        let Some(record) = managed.get(process_id) else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        Ok(record
            .events
            .iter()
            .filter(|event| event.sequence > after_sequence)
            .filter(|event| event.semantics.wake.is_some())
            .filter(|event| !record.acked_wakes.contains(&event.sequence))
            .cloned()
            .collect())
    }

    async fn wait_event_after(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, PluginError> {
        loop {
            let notify = {
                let managed = self.managed.lock().await;
                let Some(record) = managed.get(process_id) else {
                    return Err(PluginError::Session(format!(
                        "unknown process `{process_id}`"
                    )));
                };
                if let Some(event) = record
                    .events
                    .iter()
                    .find(|event| event.sequence > after_sequence && event.event_type == event_type)
                    .cloned()
                {
                    return Ok(event);
                }
                Arc::clone(&record.notify)
            };
            notify.notified().await;
        }
    }

    async fn await_process(&self, process_id: &str) -> Result<ProcessAwaitOutput, PluginError> {
        loop {
            let notify = {
                let managed = self.managed.lock().await;
                let Some(record) = managed.get(process_id) else {
                    return Err(PluginError::Session(format!(
                        "unknown process `{process_id}`"
                    )));
                };
                if let Some(terminal) = record
                    .events
                    .iter()
                    .find_map(|event| event.semantics.terminal.clone())
                {
                    return Ok(terminal.await_output);
                }
                Arc::clone(&record.notify)
            };
            notify.notified().await;
        }
    }

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, PluginError> {
        let event_type = match await_output.terminal_state() {
            ProcessTerminalState::Completed => "process.completed",
            ProcessTerminalState::Failed => "process.failed",
            ProcessTerminalState::Cancelled => "process.cancelled",
        };
        self.append_event(
            process_id,
            ProcessEventAppendRequest::new(
                event_type,
                serde_json::json!({ "await_output": await_output }),
            )
            .with_replay_key(format!("process:{process_id}:terminal:{event_type}")),
        )
        .await?;
        self.get_process(process_id).await.ok_or_else(|| {
            PluginError::Session(format!(
                "unknown process `{process_id}` after terminal event"
            ))
        })
    }

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord> {
        let managed = self.managed.lock().await;
        managed.get(process_id).map(|record| record.record.clone())
    }

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError> {
        let mut managed = self.managed.lock().await;
        let Some(record) = managed.get_mut(process_id) else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        record.acked_wakes.insert(sequence);
        Ok(())
    }

    async fn list_non_terminal(&self) -> Result<Vec<ProcessRecord>, PluginError> {
        let managed = self.managed.lock().await;
        let mut records: Vec<ProcessRecord> = managed
            .values()
            .filter(|record| !record.record.is_terminal())
            .map(|record| record.record.clone())
            .collect();
        records.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(records)
    }

    async fn claim_process_lease(
        &self,
        process_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, PluginError> {
        let mut leases = self.leases.lock().await;
        let now = current_epoch_ms();
        if let Some(current) = leases.get(process_id)
            && !current.owner_id.is_empty()
            && current.expires_at_epoch_ms > now
            && current.owner_id != owner_id
        {
            return Err(process_lease_conflict(process_id, current));
        }
        // The fencing token increases monotonically even across completion: a
        // released lease retains its `fencing_token` so a re-claim never reuses
        // a stale writer's token (mirrors `SqliteProcessRegistry`).
        let fencing_token = leases
            .get(process_id)
            .map_or(0, |current| current.fencing_token)
            .saturating_add(1);
        let lease = ProcessLease {
            schema_version: PROCESS_LEASE_SCHEMA_VERSION,
            process_id: process_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!("{process_id}:{owner_id}:{now}:{fencing_token}"),
            fencing_token,
            claimed_at_epoch_ms: now,
            expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
        };
        leases.insert(process_id.to_string(), lease.clone());
        Ok(lease)
    }

    async fn renew_process_lease(
        &self,
        lease: &ProcessLease,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, PluginError> {
        let mut leases = self.leases.lock().await;
        let now = current_epoch_ms();
        let live = leases.get(&lease.process_id).filter(|current| {
            !current.owner_id.is_empty()
                && current.lease_token == lease.lease_token
                && current.expires_at_epoch_ms > now
        });
        if live.is_none() {
            return Err(process_lease_expired(&lease.process_id));
        }
        let renewed = ProcessLease {
            expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
            ..lease.clone()
        };
        leases.insert(lease.process_id.clone(), renewed.clone());
        Ok(renewed)
    }

    async fn complete_process_lease(
        &self,
        completion: &ProcessLeaseCompletion,
    ) -> Result<(), PluginError> {
        let mut leases = self.leases.lock().await;
        // Release (don't drop) the lease, fenced by the completion token, so a
        // stale completion cannot release a newer owner's lease and the
        // `fencing_token` is preserved for the next claim.
        if let Some(current) = leases.get_mut(&completion.process_id)
            && current.lease_token == completion.lease_token
        {
            current.owner_id = String::new();
            current.lease_token = String::new();
            current.claimed_at_epoch_ms = 0;
            current.expires_at_epoch_ms = 0;
        }
        Ok(())
    }
}

/// Loud, stable error for a fenced process-lease claim. Mirrors
/// [`StoreError::RuntimeTurnLeaseConflict`](crate::StoreError) on the
/// `PluginError` channel the [`ProcessRegistry`] trait returns.
fn process_lease_conflict(process_id: &str, current: &ProcessLease) -> PluginError {
    PluginError::Session(format!(
        "process `{process_id}` is already leased by `{}` until {}",
        current.owner_id, current.expires_at_epoch_ms
    ))
}

/// Loud, stable error for a superseded or expired process lease. Mirrors
/// [`StoreError::RuntimeTurnLeaseExpired`](crate::StoreError).
fn process_lease_expired(process_id: &str) -> PluginError {
    PluginError::Session(format!(
        "process lease for `{process_id}` is missing or expired"
    ))
}
