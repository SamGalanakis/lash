use super::*;

/// Shared lease-fencing predicate for both turn leases and process leases.
///
/// A lease holder may keep operating only while the *currently stored* lease
/// still carries the holder's `lease_token` and has not expired. The
/// `current` argument is whatever lease (if any) is presently persisted for
/// the key; `expected_token` / `now` are the holder's claimed token and the
/// current clock. Returns `true` when the holder still owns a live lease.
///
/// This is the one place the "stored token matches and is unexpired" rule
/// lives — `claim_runtime_turn_lease`, `claim_process_lease`,
/// `renew_*`, and the commit-time completion checks all defer to it instead of
/// re-deriving the comparison (it previously appeared verbatim in both lease
/// families, with a comment noting one "mirrors" the other).
pub(crate) fn guard_lease<L: LeaseFence>(
    current: Option<&L>,
    expected_token: &str,
    now: u64,
) -> bool {
    match current {
        Some(current) => {
            current.lease_token() == expected_token && current.expires_at_epoch_ms() > now
        }
        None => false,
    }
}

/// Minimal view of a persisted lease that [`guard_lease`] needs. Implemented by
/// both [`RuntimeTurnLease`] and [`ProcessLease`] so the fencing predicate is
/// written once.
pub(crate) trait LeaseFence {
    fn lease_token(&self) -> &str;
    fn expires_at_epoch_ms(&self) -> u64;
}

impl LeaseFence for RuntimeTurnLease {
    fn lease_token(&self) -> &str {
        &self.lease_token
    }
    fn expires_at_epoch_ms(&self) -> u64 {
        self.expires_at_epoch_ms
    }
}

impl LeaseFence for ProcessLease {
    fn lease_token(&self) -> &str {
        &self.lease_token
    }
    fn expires_at_epoch_ms(&self) -> u64 {
        self.expires_at_epoch_ms
    }
}

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
    let current = load_runtime_turn_lease_from_conn(conn, &lease.session_id, &lease.turn_id)
        .map_err(sqlite_error)?;
    if !guard_lease(current.as_ref(), &lease.lease_token, now) {
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
    let current =
        load_runtime_turn_lease_from_conn(conn, &completed.session_id, &completed.turn_id)
            .map_err(sqlite_error)?;
    if !guard_lease(current.as_ref(), &completed.lease_token, now) {
        return Err(StoreError::RuntimeTurnLeaseExpired {
            session_id: completed.session_id.clone(),
            turn_id: completed.turn_id.clone(),
        });
    }
    Ok(())
}
