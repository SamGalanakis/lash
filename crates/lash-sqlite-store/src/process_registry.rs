//! SQLite-backed [`ProcessRegistry`] (`SqliteProcessRegistry`).
//!
//! First-party SQLite implementation of the public async process-registry
//! surface. Every DB body is a *synchronous* rusqlite closure handed
//! to [`SqliteConnection::call`] (reads) or [`SqliteConnection::write_flow`]
//! (read-then-write).
//!
//! ## Why `write_flow`, not `write`
//!
//! The registry's transactional methods produce a [`lash_core::PluginError`],
//! not a `rusqlite::Error`. `SqliteConnection::write` rolls back only when the
//! closure returns `Err(rusqlite::Error)`, so a logical `PluginError` (e.g. a
//! registration-hash conflict) would otherwise *commit* the partial work. Each
//! such method therefore runs its synchronous body returning
//! `Result<T, PluginError>` and maps it to a [`TxOutcome`]: `Ok` ⇒
//! `Commit(Ok(value))`, `Err` ⇒ `Rollback(Err(error))`. That preserves the
//! prior behaviour of rolling back on every error while still carrying the
//! `PluginError` back to the caller. The outer `rusqlite::Error` channel only
//! carries genuine SQLite/connection failures, mapped via `process_sqlite_error`.
//!
//! The `*_conn` helpers are synchronous and take a `&rusqlite::Connection` so
//! they compose inside either closure — including from within a `&Transaction`,
//! which derefs to `&Connection`.

use super::*;

fn process_status_label(record: &ProcessRecord) -> &'static str {
    record.status.label()
}

impl SqliteProcessRegistry {
    pub async fn open(path: &Path) -> tokio_rusqlite::Result<Self> {
        let conn = SqliteConnection::open(path).await?;
        ensure_process_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::File).await?;
        Ok(Self {
            conn,
            notify: tokio::sync::Notify::new(),
        })
    }

    pub async fn memory() -> tokio_rusqlite::Result<Self> {
        let conn = SqliteConnection::open_in_memory().await?;
        ensure_process_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::Memory).await?;
        Ok(Self {
            conn,
            notify: tokio::sync::Notify::new(),
        })
    }

    fn load_process_conn(
        conn: &Connection,
        process_id: &str,
    ) -> Result<Option<ProcessRecord>, lash_core::PluginError> {
        let json: Option<String> = conn
            .query_row(
                "SELECT record_json FROM processes WHERE process_id = ?1",
                params![process_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(process_sqlite_error)?;
        json.map(|json| serde_json::from_str(&json).map_err(process_decode_error))
            .transpose()
    }

    fn save_process_conn(
        conn: &Connection,
        record: &ProcessRecord,
    ) -> Result<(), lash_core::PluginError> {
        conn.execute(
            "UPDATE processes
             SET updated_at_ms = ?2, status = ?3, record_json = ?4
             WHERE process_id = ?1",
            params![
                record.id.as_str(),
                record.updated_at_ms as i64,
                process_status_label(record),
                process_encode_json(record)?
            ],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }

    fn load_event_by_key_conn(
        conn: &Connection,
        process_id: &str,
        replay_key: &str,
    ) -> Result<Option<(String, ProcessEvent)>, lash_core::PluginError> {
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT payload_hash, event_json
                 FROM process_events
                 WHERE process_id = ?1 AND idempotency_key = ?2",
                params![process_id, replay_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(process_sqlite_error)?;
        row.map(|(hash, json)| {
            serde_json::from_str(&json)
                .map(|event| (hash, event))
                .map_err(process_decode_error)
        })
        .transpose()
    }

    fn load_process_lease_conn(
        conn: &Connection,
        process_id: &str,
    ) -> Result<Option<ProcessLease>, lash_core::PluginError> {
        conn.query_row(
            "SELECT lease_owner_id, lease_token, lease_fencing_token,
                    lease_claimed_at_ms, lease_expires_at_ms
             FROM process_leases
             WHERE process_id = ?1",
            params![process_id],
            |row| {
                let owner_id: Option<String> = row.get(0)?;
                let lease_token: Option<String> = row.get(1)?;
                let (Some(owner_id), Some(lease_token)) = (owner_id, lease_token) else {
                    return Ok(None);
                };
                Ok(Some(ProcessLease {
                    schema_version: PROCESS_LEASE_SCHEMA_VERSION,
                    process_id: process_id.to_string(),
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
        .map_err(process_sqlite_error)
    }

    fn list_grants_for_scope_conn(
        conn: &Connection,
        session_scope: &SessionScope,
        live_only: bool,
    ) -> Result<Vec<ProcessHandleGrantEntry>, lash_core::PluginError> {
        let session_scope_id = session_scope.id();
        let status_clause = if live_only {
            "AND p.status = 'running'"
        } else {
            ""
        };
        let mut stmt = conn
            .prepare(&format!(
                "SELECT g.process_id, g.descriptor_json, p.record_json
                 FROM process_handle_grants g
                 JOIN processes p ON p.process_id = g.process_id
                 WHERE g.scope_id = ?1 {status_clause}
                 ORDER BY g.process_id ASC"
            ))
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![session_scope_id.as_str()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(process_sqlite_error)?;
        let mut entries = Vec::new();
        for row in rows {
            let (process_id, descriptor_json, record_json) = row.map_err(process_sqlite_error)?;
            let descriptor: ProcessHandleDescriptor =
                serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
            let record: ProcessRecord =
                serde_json::from_str(&record_json).map_err(process_decode_error)?;
            entries.push((
                ProcessHandleGrant {
                    session_id: session_scope.session_id.clone(),
                    process_id,
                    descriptor,
                },
                record,
            ));
        }
        Ok(entries)
    }
}

/// Map a `Result<T, PluginError>` produced by a synchronous transaction body to
/// a [`TxOutcome`]: commit on success, roll back on logical error. Both arms
/// carry the inner `Result` back so the caller recovers the value or the
/// `PluginError` after the transaction resolves.
fn tx_outcome<T>(
    result: Result<T, lash_core::PluginError>,
) -> TxOutcome<Result<T, lash_core::PluginError>> {
    match result {
        Ok(value) => TxOutcome::Commit(Ok(value)),
        Err(err) => TxOutcome::Rollback(Err(err)),
    }
}

#[async_trait::async_trait]
impl ProcessRegistry for SqliteProcessRegistry {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let (registration, registration_hash) = prepare_process_registration(registration)?;
        let record = self
            .conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    if let Some(existing) = Self::load_process_conn(tx, &registration.id)? {
                        if existing.registration_hash == registration_hash {
                            return Ok(existing);
                        }
                        return Err(lash_core::PluginError::Session(format!(
                            "process `{}` registration hash conflict: existing {}, new {}",
                            registration.id, existing.registration_hash, registration_hash
                        )));
                    }
                    let now = current_epoch_ms();
                    let record = ProcessRecord::from_prepared_registration(
                        registration,
                        registration_hash,
                        now,
                    );
                    let originator_scope_id = record.originator_scope_id();
                    tx.execute(
                        "INSERT INTO processes (
                            process_id, registration_hash, owner_scope_id,
                            created_at_ms, updated_at_ms, status, record_json
                         )
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            record.id.as_str(),
                            record.registration_hash.as_str(),
                            originator_scope_id.as_str(),
                            record.created_at_ms as i64,
                            record.updated_at_ms as i64,
                            process_status_label(&record),
                            process_encode_json(&record)?,
                        ],
                    )
                    .map_err(process_sqlite_error)?;
                    Ok(record)
                })()))
            })
            .await
            .map_err(process_sqlite_error)??;
        self.notify.notify_waiters();
        Ok(record)
    }

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let process_id = process_id.to_string();
        let record = self
            .conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    let mut record =
                        Self::load_process_conn(tx, &process_id)?.ok_or_else(|| {
                            lash_core::PluginError::Session(format!(
                                "unknown process `{process_id}`"
                            ))
                        })?;
                    record.external_ref = Some(external_ref);
                    record.updated_at_ms = current_epoch_ms();
                    Self::save_process_conn(tx, &record)?;
                    Ok(record)
                })()))
            })
            .await
            .map_err(process_sqlite_error)??;
        self.notify.notify_waiters();
        Ok(record)
    }

    async fn grant_handle(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, lash_core::PluginError> {
        let session_scope = session_scope.clone();
        let process_id = process_id.to_string();
        self.conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    let session_scope_id = session_scope.id();
                    if Self::load_process_conn(tx, &process_id)?.is_none() {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown process `{process_id}`"
                        )));
                    }
                    tx.execute(
                        "INSERT INTO process_handle_grants (session_id, scope_id, process_id, descriptor_json)
                         VALUES (?1, ?2, ?3, ?4)
                         ON CONFLICT(scope_id, process_id) DO UPDATE SET
                            session_id = excluded.session_id,
                            descriptor_json = excluded.descriptor_json",
                        params![
                            session_scope.session_id.as_str(),
                            session_scope_id.as_str(),
                            process_id.as_str(),
                            process_encode_json(&descriptor)?
                        ],
                    )
                    .map_err(process_sqlite_error)?;
                    Ok(ProcessHandleGrant {
                        session_id: session_scope.session_id.clone(),
                        process_id: process_id.clone(),
                        descriptor,
                    })
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn revoke_handle(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
    ) -> Result<(), lash_core::PluginError> {
        let session_scope_id = session_scope.id().as_str().to_string();
        let process_id = process_id.to_string();
        self.conn
            .call(move |conn| {
                conn.execute(
                    "DELETE FROM process_handle_grants WHERE scope_id = ?1 AND process_id = ?2",
                    params![session_scope_id, process_id],
                )
            })
            .await
            .map_err(process_sqlite_error)?;
        Ok(())
    }

    async fn transfer_handle_grants(
        &self,
        from_scope: &SessionScope,
        to_scope: &SessionScope,
        process_ids: &[String],
    ) -> Result<(), lash_core::PluginError> {
        let from_scope = from_scope.clone();
        let to_scope = to_scope.clone();
        let process_ids = process_ids.to_vec();
        self.conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    let from_scope_id = from_scope.id();
                    let to_scope_id = to_scope.id();
                    for process_id in &process_ids {
                        let descriptor_json: Option<String> = tx
                            .query_row(
                                "SELECT descriptor_json
                                 FROM process_handle_grants
                                 WHERE scope_id = ?1 AND process_id = ?2",
                                params![from_scope_id.as_str(), process_id.as_str()],
                                |row| row.get(0),
                            )
                            .optional()
                            .map_err(process_sqlite_error)?;
                        let Some(descriptor_json) = descriptor_json else {
                            return Err(lash_core::PluginError::Session(format!(
                                "process handle `{process_id}` is not granted to session `{}`",
                                from_scope.session_id
                            )));
                        };
                        tx.execute(
                            "DELETE FROM process_handle_grants
                             WHERE scope_id = ?1 AND process_id = ?2",
                            params![from_scope_id.as_str(), process_id.as_str()],
                        )
                        .map_err(process_sqlite_error)?;
                        tx.execute(
                            "INSERT INTO process_handle_grants (session_id, scope_id, process_id, descriptor_json)
                             VALUES (?1, ?2, ?3, ?4)
                             ON CONFLICT(scope_id, process_id) DO UPDATE SET
                                session_id = excluded.session_id,
                                descriptor_json = excluded.descriptor_json",
                            params![
                                to_scope.session_id.as_str(),
                                to_scope_id.as_str(),
                                process_id.as_str(),
                                descriptor_json
                            ],
                        )
                        .map_err(process_sqlite_error)?;
                    }
                    Ok(())
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn list_handle_grants(
        &self,
        session_scope: &SessionScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, lash_core::PluginError> {
        let session_scope = session_scope.clone();
        self.conn
            .call(move |conn| {
                Ok(Self::list_grants_for_scope_conn(
                    conn,
                    &session_scope,
                    false,
                ))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn list_live_handle_grants(
        &self,
        session_scope: &SessionScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, lash_core::PluginError> {
        let session_scope = session_scope.clone();
        self.conn
            .call(move |conn| Ok(Self::list_grants_for_scope_conn(conn, &session_scope, true)))
            .await
            .map_err(process_sqlite_error)?
    }

    async fn has_handle_grant(
        &self,
        session_scope: &SessionScope,
        process_id: &str,
    ) -> Result<bool, lash_core::PluginError> {
        let session_scope_id = session_scope.id().as_str().to_string();
        let process_id = process_id.to_string();
        self.conn
            .call(move |conn| {
                let exists = conn
                    .query_row(
                        "SELECT 1
                         FROM process_handle_grants g
                         JOIN processes p ON p.process_id = g.process_id
                         WHERE g.scope_id = ?1 AND g.process_id = ?2
                         LIMIT 1",
                        params![session_scope_id, process_id],
                        |_| Ok(()),
                    )
                    .optional()?
                    .is_some();
                Ok(exists)
            })
            .await
            .map_err(process_sqlite_error)
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, lash_core::PluginError> {
        let process_id = process_id.to_string();
        self.conn
            .call(move |conn| {
                Ok((|| {
                    if Self::load_process_conn(conn, &process_id)?.is_none() {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown process `{process_id}`"
                        )));
                    }
                    let mut stmt = conn
                        .prepare(
                            "SELECT session_id, descriptor_json
                             FROM process_handle_grants
                             WHERE process_id = ?1
                             ORDER BY session_id ASC, scope_id ASC",
                        )
                        .map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map(params![process_id], |row| {
                            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                        })
                        .map_err(process_sqlite_error)?;
                    let mut grants = Vec::new();
                    for row in rows {
                        let (session_id, descriptor_json) = row.map_err(process_sqlite_error)?;
                        let descriptor: ProcessHandleDescriptor =
                            serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
                        grants.push(ProcessHandleGrant {
                            session_id,
                            process_id: process_id.clone(),
                            descriptor,
                        });
                    }
                    Ok(grants)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn delete_session_process_state(
        &self,
        session_id: &str,
    ) -> Result<lash_core::ProcessSessionDeleteReport, lash_core::PluginError> {
        let session_id_owned = session_id.to_string();
        let (
            revoked_handle_count,
            deleted_wake_count,
            mut orphaned_process_ids,
            mut preserved_process_ids,
        ) = self
            .conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    let session_id = session_id_owned;
                    let removed = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT g.process_id, p.record_json
                                 FROM process_handle_grants g
                                 JOIN processes p ON p.process_id = g.process_id
                                 WHERE g.session_id = ?1
                                 ORDER BY g.process_id ASC",
                            )
                            .map_err(process_sqlite_error)?;
                        let rows = stmt
                            .query_map(params![session_id], |row| {
                                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                            })
                            .map_err(process_sqlite_error)?;
                        let mut removed = Vec::new();
                        for row in rows {
                            let (process_id, record_json) = row.map_err(process_sqlite_error)?;
                            let record: ProcessRecord =
                                serde_json::from_str(&record_json).map_err(process_decode_error)?;
                            removed.push((process_id, record));
                        }
                        removed
                    };

                    // Wake acknowledgements are process-scoped consumed-event markers.
                    // Session deletion removes materialized session-addressed deliveries
                    // through the session store; clearing these rows would re-expose
                    // already-consumed wakes to surviving grants or future host readers.
                    let deleted_wake_count = 0;
                    let revoked_handle_count = tx
                        .execute(
                            "DELETE FROM process_handle_grants WHERE session_id = ?1",
                            params![session_id],
                        )
                        .map_err(process_sqlite_error)?;
                    let mut orphaned_process_ids = Vec::new();
                    let mut preserved_process_ids = Vec::new();
                    for (process_id, record) in removed {
                        if record.is_terminal() {
                            continue;
                        }
                        let remaining_grants: i64 = tx
                            .query_row(
                                "SELECT COUNT(*) FROM process_handle_grants WHERE process_id = ?1",
                                params![process_id],
                                |row| row.get(0),
                            )
                            .map_err(process_sqlite_error)?;
                        if remaining_grants == 0 {
                            orphaned_process_ids.push(process_id);
                        } else {
                            preserved_process_ids.push(process_id);
                        }
                    }
                    let wake_targeted = {
                        let mut stmt = tx
                            .prepare("SELECT process_id, record_json FROM processes ORDER BY process_id ASC")
                            .map_err(process_sqlite_error)?;
                        let rows = stmt
                            .query_map([], |row| {
                                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                            })
                            .map_err(process_sqlite_error)?;
                        let mut records = Vec::new();
                        for row in rows {
                            let (process_id, record_json) = row.map_err(process_sqlite_error)?;
                            let record: ProcessRecord =
                                serde_json::from_str(&record_json).map_err(process_decode_error)?;
                            records.push((process_id, record));
                        }
                        records
                    };
                    for (_process_id, mut record) in wake_targeted {
                        if record.clear_wake_target_for_session(&session_id) {
                            Self::save_process_conn(tx, &record)?;
                        }
                    }
                    Ok((
                        revoked_handle_count,
                        deleted_wake_count,
                        orphaned_process_ids,
                        preserved_process_ids,
                    ))
                })()))
            })
            .await
            .map_err(process_sqlite_error)??;
        orphaned_process_ids.sort();
        orphaned_process_ids.dedup();
        preserved_process_ids.sort();
        preserved_process_ids.dedup();
        Ok(lash_core::ProcessSessionDeleteReport {
            session_id: session_id.to_string(),
            revoked_handle_count,
            deleted_wake_count,
            orphaned_process_ids,
            preserved_process_ids,
        })
    }

    async fn append_event(
        &self,
        process_id: &str,
        request: ProcessEventAppendRequest,
    ) -> Result<ProcessEventAppendResult, lash_core::PluginError> {
        let process_id = process_id.to_string();
        let (result, appended) = self
            .conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    let mut record =
                        Self::load_process_conn(tx, &process_id)?.ok_or_else(|| {
                            lash_core::PluginError::Session(format!(
                                "unknown process `{process_id}`"
                            ))
                        })?;
                    let replay_lookup = if let Some(replay_key) =
                        request.replay.as_ref().map(|replay| replay.key.as_str())
                    {
                        Self::load_event_by_key_conn(tx, &process_id, replay_key)?
                    } else {
                        None
                    };
                    let sequence = tx
                        .query_row(
                            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM process_events WHERE process_id = ?1",
                            params![process_id],
                            |row| row.get::<_, i64>(0),
                        )
                        .map_err(process_sqlite_error)? as u64;
                    let occurred_at_ms = current_epoch_ms();
                    let prepared = prepare_process_event_append(
                        &record,
                        request,
                        sequence,
                        replay_lookup,
                        occurred_at_ms,
                    )?;
                    if prepared.replayed {
                        return Ok((
                            ProcessEventAppendResult {
                                event: prepared.event,
                                wake_delivery: prepared.wake_delivery,
                            },
                            false,
                        ));
                    }
                    let event = prepared.event;
                    tx.execute(
                        "INSERT INTO process_events (
                            process_id, sequence, event_type, payload_hash, idempotency_key,
                            occurred_at_ms, event_json
                         )
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            process_id,
                            sequence as i64,
                            event.event_type.as_str(),
                            prepared.payload_hash.as_str(),
                            event.invocation.replay_key(),
                            prepared.occurred_at_ms as i64,
                            process_encode_json(&event)?,
                        ],
                    )
                    .map_err(process_sqlite_error)?;
                    if let Some(status) = prepared.status_update.clone() {
                        record.status = status;
                        if record.status.is_terminal() {
                            record.wait = None;
                        }
                    }
                    record.updated_at_ms = prepared.occurred_at_ms;
                    Self::save_process_conn(tx, &record)?;
                    Ok((
                        ProcessEventAppendResult {
                            event,
                            wake_delivery: prepared.wake_delivery,
                        },
                        true,
                    ))
                })()))
            })
            .await
            .map_err(process_sqlite_error)??;
        if appended {
            self.notify.notify_waiters();
        }
        Ok(result)
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        let process_id = process_id.to_string();
        self.conn
            .call(move |conn| {
                Ok((|| {
                    if Self::load_process_conn(conn, &process_id)?.is_none() {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown process `{process_id}`"
                        )));
                    }
                    let mut stmt = conn
                        .prepare(
                            "SELECT event_json FROM process_events
                             WHERE process_id = ?1 AND sequence > ?2
                             ORDER BY sequence ASC",
                        )
                        .map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map(params![process_id, after_sequence as i64], |row| {
                            row.get::<_, String>(0)
                        })
                        .map_err(process_sqlite_error)?;
                    let mut events = Vec::new();
                    for row in rows {
                        events.push(
                            serde_json::from_str(&row.map_err(process_sqlite_error)?)
                                .map_err(process_decode_error)?,
                        );
                    }
                    Ok(events)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn count_events_through(
        &self,
        process_id: &str,
        event_type: &str,
        up_to_sequence: u64,
    ) -> Result<u64, lash_core::PluginError> {
        let process_id = process_id.to_string();
        let event_type = event_type.to_string();
        self.conn
            .call(move |conn| {
                Ok((|| {
                    if Self::load_process_conn(conn, &process_id)?.is_none() {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown process `{process_id}`"
                        )));
                    }
                    conn.query_row(
                        "SELECT COUNT(*) FROM process_events
                         WHERE process_id = ?1 AND event_type = ?2 AND sequence <= ?3",
                        params![process_id, event_type, up_to_sequence as i64],
                        |row| row.get::<_, i64>(0),
                    )
                    .map(|count| count as u64)
                    .map_err(process_sqlite_error)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn recent_events(
        &self,
        process_id: &str,
        limit: usize,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        let process_id = process_id.to_string();
        self.conn
            .call(move |conn| {
                Ok((|| {
                    if Self::load_process_conn(conn, &process_id)?.is_none() {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown process `{process_id}`"
                        )));
                    }
                    let mut stmt = conn
                        .prepare(
                            "SELECT event_json FROM process_events
                             WHERE process_id = ?1
                             ORDER BY sequence DESC
                             LIMIT ?2",
                        )
                        .map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map(params![process_id, limit as i64], |row| {
                            row.get::<_, String>(0)
                        })
                        .map_err(process_sqlite_error)?;
                    let mut events: Vec<ProcessEvent> = Vec::new();
                    for row in rows {
                        events.push(
                            serde_json::from_str(&row.map_err(process_sqlite_error)?)
                                .map_err(process_decode_error)?,
                        );
                    }
                    events.reverse();
                    Ok(events)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        let acked: std::collections::HashSet<u64> = {
            let process_id = process_id.to_string();
            self.conn
                .call(move |conn| {
                    Ok(
                        (|| -> Result<std::collections::HashSet<u64>, lash_core::PluginError> {
                            let mut stmt = conn
                                .prepare(
                                    "SELECT sequence FROM process_wake_acks WHERE process_id = ?1",
                                )
                                .map_err(process_sqlite_error)?;
                            let rows = stmt
                                .query_map(params![process_id], |row| row.get::<_, i64>(0))
                                .map_err(process_sqlite_error)?;
                            let mut set = std::collections::HashSet::new();
                            for row in rows {
                                set.insert(row.map_err(process_sqlite_error)? as u64);
                            }
                            Ok(set)
                        })(),
                    )
                })
                .await
                .map_err(process_sqlite_error)??
        };
        Ok(self
            .events_after(process_id, after_sequence)
            .await?
            .into_iter()
            .filter(|event| event.semantics.wake.is_some() && !acked.contains(&event.sequence))
            .collect())
    }

    async fn wait_event_after(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, lash_core::PluginError> {
        loop {
            if let Some(event) = self
                .events_after(process_id, after_sequence)
                .await?
                .into_iter()
                .find(|event| event.event_type == event_type)
            {
                return Ok(event);
            }
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
        }
    }

    async fn await_process(
        &self,
        process_id: &str,
    ) -> Result<ProcessAwaitOutput, lash_core::PluginError> {
        loop {
            let record = self.get_process(process_id).await.ok_or_else(|| {
                lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
            })?;
            if let Some(await_output) = record.status.await_output() {
                return Ok(await_output.clone());
            }
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
        }
    }

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let event_type = match await_output.terminal_state() {
            lash_core::ProcessTerminalState::Completed => "process.completed",
            lash_core::ProcessTerminalState::Failed => "process.failed",
            lash_core::ProcessTerminalState::Cancelled => "process.cancelled",
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
            lash_core::PluginError::Session(format!(
                "unknown process `{process_id}` after terminal event"
            ))
        })
    }

    async fn set_process_wait(
        &self,
        process_id: &str,
        wait: lash_core::WaitState,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let process_id = process_id.to_string();
        self.conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    let mut record =
                        Self::load_process_conn(tx, &process_id)?.ok_or_else(|| {
                            lash_core::PluginError::Session(format!(
                                "unknown process `{process_id}`"
                            ))
                        })?;
                    if record.is_terminal() {
                        return Err(lash_core::PluginError::Session(format!(
                            "terminal process `{process_id}` cannot enter a wait state"
                        )));
                    }
                    record.wait = Some(wait);
                    record.updated_at_ms = current_epoch_ms();
                    Self::save_process_conn(tx, &record)?;
                    Ok(record)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn clear_process_wait(
        &self,
        process_id: &str,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let process_id = process_id.to_string();
        self.conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    let mut record =
                        Self::load_process_conn(tx, &process_id)?.ok_or_else(|| {
                            lash_core::PluginError::Session(format!(
                                "unknown process `{process_id}`"
                            ))
                        })?;
                    record.wait = None;
                    record.updated_at_ms = current_epoch_ms();
                    Self::save_process_conn(tx, &record)?;
                    Ok(record)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord> {
        let process_id = process_id.to_string();
        self.conn
            .call(move |conn| Ok(Self::load_process_conn(conn, &process_id).ok().flatten()))
            .await
            .ok()
            .flatten()
    }

    async fn list_processes(
        &self,
        filter: &lash_core::ProcessListFilter,
    ) -> Result<Vec<ProcessRecord>, lash_core::PluginError> {
        let filter = filter.clone();
        self.conn
            .call(move |conn| {
                Ok((|| {
                    let mut stmt = conn
                        .prepare(
                            "SELECT record_json FROM processes
                             ORDER BY process_id ASC",
                        )
                        .map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map([], |row| row.get::<_, String>(0))
                        .map_err(process_sqlite_error)?;
                    let mut records = Vec::new();
                    for row in rows {
                        let record: ProcessRecord =
                            serde_json::from_str(&row.map_err(process_sqlite_error)?)
                                .map_err(process_decode_error)?;
                        if filter.matches_record(&record) {
                            records.push(record);
                        }
                    }
                    Ok(records)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn ack_wake(
        &self,
        process_id: &str,
        sequence: u64,
    ) -> Result<(), lash_core::PluginError> {
        let process_id = process_id.to_string();
        self.conn
            .call(move |conn| {
                Ok((|| {
                    if Self::load_process_conn(conn, &process_id)?.is_none() {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown process `{process_id}`"
                        )));
                    }
                    conn.execute(
                        "INSERT OR IGNORE INTO process_wake_acks (process_id, sequence) VALUES (?1, ?2)",
                        params![process_id, sequence as i64],
                    )
                    .map_err(process_sqlite_error)?;
                    Ok(())
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn list_non_terminal(&self) -> Result<Vec<ProcessRecord>, lash_core::PluginError> {
        self.conn
            .call(move |conn| {
                Ok((|| {
                    let mut stmt = conn
                        .prepare(
                            "SELECT record_json FROM processes
                             WHERE status = 'running'
                             ORDER BY process_id ASC",
                        )
                        .map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map([], |row| row.get::<_, String>(0))
                        .map_err(process_sqlite_error)?;
                    let mut records = Vec::new();
                    for row in rows {
                        let record: ProcessRecord =
                            serde_json::from_str(&row.map_err(process_sqlite_error)?)
                                .map_err(process_decode_error)?;
                        records.push(record);
                    }
                    Ok(records)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn claim_process_lease(
        &self,
        process_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, lash_core::PluginError> {
        let process_id = process_id.to_string();
        let owner_id = owner_id.to_string();
        self.conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    if Self::load_process_conn(tx, &process_id)?.is_none() {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown process `{process_id}`"
                        )));
                    }
                    let now = current_epoch_ms();
                    let current = Self::load_process_lease_conn(tx, &process_id)?;
                    if let Some(current) = current.as_ref()
                        && current.expires_at_epoch_ms > now
                        && current.owner_id != owner_id
                    {
                        return Err(process_lease_conflict(&process_id, current));
                    }
                    // Read the raw fencing token directly: a completed/abandoned
                    // lease nulls the owner/token columns but retains the
                    // monotonically-increasing `lease_fencing_token`, so a
                    // re-claim never reuses a stale writer's token.
                    let fencing_token: u64 = tx
                        .query_row(
                            "SELECT lease_fencing_token FROM process_leases WHERE process_id = ?1",
                            params![process_id],
                            |row| row.get::<_, i64>(0),
                        )
                        .optional()
                        .map_err(process_sqlite_error)?
                        .unwrap_or(0) as u64
                        + 1;
                    let lease = ProcessLease {
                        schema_version: PROCESS_LEASE_SCHEMA_VERSION,
                        process_id: process_id.clone(),
                        owner_id: owner_id.clone(),
                        lease_token: format!(
                            "{:x}",
                            Sha256::digest(
                                format!("{process_id}:{owner_id}:{now}:{fencing_token}").as_bytes()
                            )
                        ),
                        fencing_token,
                        claimed_at_epoch_ms: now,
                        expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
                    };
                    tx.execute(
                        "INSERT INTO process_leases (
                            process_id, lease_owner_id, lease_token, lease_fencing_token,
                            lease_claimed_at_ms, lease_expires_at_ms
                         )
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                         ON CONFLICT(process_id) DO UPDATE SET
                            lease_owner_id = excluded.lease_owner_id,
                            lease_token = excluded.lease_token,
                            lease_fencing_token = excluded.lease_fencing_token,
                            lease_claimed_at_ms = excluded.lease_claimed_at_ms,
                            lease_expires_at_ms = excluded.lease_expires_at_ms",
                        params![
                            lease.process_id.as_str(),
                            lease.owner_id.as_str(),
                            lease.lease_token.as_str(),
                            lease.fencing_token as i64,
                            lease.claimed_at_epoch_ms as i64,
                            lease.expires_at_epoch_ms as i64,
                        ],
                    )
                    .map_err(process_sqlite_error)?;
                    Ok(lease)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn renew_process_lease(
        &self,
        lease: &ProcessLease,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, lash_core::PluginError> {
        let lease = lease.clone();
        self.conn
            .write_flow(move |tx| {
                Ok(tx_outcome((|| {
                    let now = current_epoch_ms();
                    let current = Self::load_process_lease_conn(tx, &lease.process_id)?;
                    if !guard_lease(current.as_ref(), &lease.lease_token, now) {
                        return Err(process_lease_expired(&lease.process_id));
                    }
                    let renewed = ProcessLease {
                        expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
                        ..lease.clone()
                    };
                    tx.execute(
                        "UPDATE process_leases
                         SET lease_expires_at_ms = ?2
                         WHERE process_id = ?1 AND lease_token = ?3",
                        params![
                            renewed.process_id.as_str(),
                            renewed.expires_at_epoch_ms as i64,
                            renewed.lease_token.as_str(),
                        ],
                    )
                    .map_err(process_sqlite_error)?;
                    Ok(renewed)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn complete_process_lease(
        &self,
        completion: &ProcessLeaseCompletion,
    ) -> Result<(), lash_core::PluginError> {
        let process_id = completion.process_id.clone();
        let lease_token = completion.lease_token.clone();
        self.conn
            .call(move |conn| {
                conn.execute(
                    "UPDATE process_leases
                     SET lease_owner_id = NULL,
                         lease_token = NULL,
                         lease_claimed_at_ms = 0,
                         lease_expires_at_ms = 0
                     WHERE process_id = ?1 AND lease_token = ?2",
                    params![process_id, lease_token],
                )
            })
            .await
            .map_err(process_sqlite_error)?;
        Ok(())
    }
}

/// Loud, stable error for a fenced process-lease claim on the `PluginError`
/// channel the [`ProcessRegistry`] trait returns.
fn process_lease_conflict(process_id: &str, current: &ProcessLease) -> lash_core::PluginError {
    lash_core::PluginError::Session(format!(
        "process `{process_id}` is already leased by `{}` until {}",
        current.owner_id, current.expires_at_epoch_ms
    ))
}

/// Loud, stable error for a superseded or expired process lease.
fn process_lease_expired(process_id: &str) -> lash_core::PluginError {
    lash_core::PluginError::Session(format!(
        "process lease for `{process_id}` is missing or expired"
    ))
}
