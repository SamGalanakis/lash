use super::*;

pub(crate) fn load_runtime_turn_lease_from_conn(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> rusqlite::Result<Option<RuntimeTurnLease>> {
    conn.query_row(
        "SELECT lease_owner_id, lease_token, lease_fencing_token, lease_claimed_at_ms, lease_expires_at_ms
         FROM runtime_turn_checkpoints
         WHERE session_id = ?1 AND turn_id = ?2",
        params![session_id, turn_id],
        |row| {
            let owner_id: Option<String> = row.get(0)?;
            let lease_token: Option<String> = row.get(1)?;
            let Some(owner_id) = owner_id else {
                return Ok(None);
            };
            let Some(lease_token) = lease_token else {
                return Ok(None);
            };
            Ok(Some(RuntimeTurnLease {
                schema_version: RUNTIME_TURN_LEASE_SCHEMA_VERSION,
                session_id: session_id.to_string(),
                turn_id: turn_id.to_string(),
                owner_id,
                lease_token,
                fencing_token: row.get::<_, i64>(2)? as u64,
                claimed_at_epoch_ms: row.get::<_, i64>(3)? as u64,
                expires_at_epoch_ms: row.get::<_, i64>(4)? as u64,
            }))
        },
    )
    .optional()
    .map(|lease| lease.flatten())
}

pub(crate) fn ensure_runtime_turn_lease_conn(
    conn: &Connection,
    lease: &RuntimeTurnLease,
) -> Result<(), StoreError> {
    let now = current_epoch_ms();
    let Some(current) = load_runtime_turn_lease_from_conn(conn, &lease.session_id, &lease.turn_id)
        .map_err(sqlite_error)?
    else {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: lease.session_id.clone(),
            turn_id: lease.turn_id.clone(),
        });
    };
    if current.lease_token != lease.lease_token || current.expires_at_epoch_ms <= now {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: lease.session_id.clone(),
            turn_id: lease.turn_id.clone(),
        });
    }
    Ok(())
}

pub(crate) fn ensure_runtime_turn_completion_conn(
    conn: &Connection,
    completed: &lash_core::store::RuntimeTurnCompletion,
) -> Result<(), StoreError> {
    let now = current_epoch_ms();
    let Some(current) =
        load_runtime_turn_lease_from_conn(conn, &completed.session_id, &completed.turn_id)
            .map_err(sqlite_error)?
    else {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: completed.session_id.clone(),
            turn_id: completed.turn_id.clone(),
        });
    };
    if current.lease_token != completed.lease_token || current.expires_at_epoch_ms <= now {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: completed.session_id.clone(),
            turn_id: completed.turn_id.clone(),
        });
    }
    Ok(())
}
