//! Atomic, lease-fenced terminal process completion.

use super::process_registry::{process_lease_expired, tx_outcome};
use super::*;

pub(super) async fn complete_process_with_lease(
    registry: &SqliteProcessRegistry,
    lease: &ProcessLease,
    await_output: ProcessAwaitOutput,
) -> Result<ProcessRecord, lash_core::PluginError> {
    let lease = lease.clone();
    registry.conn
        .write_flow(move |tx| {
            Ok(tx_outcome((|| {
                let process_id = lease.process_id.as_str();
                let mut record = SqliteProcessRegistry::load_process_conn(tx, process_id)?.ok_or_else(|| {
                    lash_core::PluginError::Session(format!(
                        "unknown process `{process_id}`"
                    ))
                })?;
                let event_type = match await_output.terminal_state() {
                    lash_core::ProcessTerminalState::Completed => "process.completed",
                    lash_core::ProcessTerminalState::Failed => "process.failed",
                    lash_core::ProcessTerminalState::Cancelled => "process.cancelled",
                    lash_core::ProcessTerminalState::Abandoned => "process.abandoned",
                };
                let request = ProcessEventAppendRequest::new(
                    event_type,
                    serde_json::json!({ "await_output": await_output }),
                )
                .with_replay_key(format!("process:{process_id}:terminal:{event_type}"));
                let replay_lookup = request
                    .replay
                    .as_ref()
                    .map(|replay| {
                        SqliteProcessRegistry::load_event_by_key_conn(tx, process_id, replay.key.as_str())
                    })
                    .transpose()?
                    .flatten();
                let sequence = tx
                    .query_row(
                        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM process_events WHERE process_id = ?1",
                        params![process_id],
                        |row| row.get::<_, i64>(0),
                    )
                    .map_err(process_sqlite_error)? as u64;
                let now = current_epoch_ms();

                // A successful prior terminal append is replay-idempotent even
                // though that transaction already cleared the lease.
                let prepared = prepare_process_event_append(
                    &record,
                    request,
                    sequence,
                    replay_lookup,
                    now,
                )?;
                if matches!(prepared, lash_core::ProcessEventAppendPlan::Replay { .. }) {
                    return Ok(record);
                }

                let current = SqliteProcessRegistry::load_process_lease_conn(tx, process_id)?;
                if !guard_lease(current.as_ref(), &lease.lease_token, now)
                    || !current.as_ref().is_some_and(|current| {
                        current.owner.same_incarnation(&lease.owner)
                            && current.fencing_token == lease.fencing_token
                    })
                {
                    return Err(process_lease_expired(process_id));
                }

                let lash_core::ProcessEventAppendPlan::Insert {
                    event,
                    payload_hash,
                    status_update,
                    occurred_at_ms,
                    ..
                } = prepared
                else {
                    unreachable!("replay returned above")
                };
                tx.execute(
                    "INSERT INTO process_events (
                        process_id, sequence, event_type, payload_hash, idempotency_key,
                        occurred_at_ms, event_json
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        process_id,
                        sequence as i64,
                        event.event_type.as_str(),
                        payload_hash.as_str(),
                        event.invocation.replay_key(),
                        occurred_at_ms as i64,
                        process_encode_json(&event)?,
                    ],
                )
                .map_err(process_sqlite_error)?;
                if let Some(status) = status_update {
                    lash_core::apply_process_status_projection(
                        &mut record,
                        status,
                        occurred_at_ms,
                    );
                }
                SqliteProcessRegistry::save_process_conn(tx, &record)?;
                tx.execute(
                    "UPDATE process_leases
                     SET lease_owner_id = NULL,
                         lease_owner_incarnation_id = NULL,
                         lease_owner_liveness_json = NULL,
                         lease_token = NULL,
                         lease_claimed_at_ms = 0,
                         lease_expires_at_ms = 0
                     WHERE process_id = ?1
                       AND lease_token = ?2
                       AND lease_fencing_token = ?3",
                    params![process_id, lease.lease_token, lease.fencing_token as i64],
                )
                .map_err(process_sqlite_error)?;
                Ok(record)
            })()))
        })
        .await
        .map_err(process_sqlite_error)?
}
