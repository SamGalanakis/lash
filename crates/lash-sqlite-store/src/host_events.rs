//! SQLite-backed runtime host-event store.
//!
//! This is the durable peer of [`SqliteProcessRegistry`]: it stores trigger
//! subscriptions and append-only host-event occurrences at deployment scope,
//! outside any session database.

use super::*;

pub struct SqliteHostEventStore {
    conn: SqliteConnection,
}

impl SqliteHostEventStore {
    pub async fn open(path: &Path) -> tokio_rusqlite::Result<Self> {
        let conn = SqliteConnection::open(path).await?;
        ensure_host_event_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::File).await?;
        Ok(Self { conn })
    }

    pub async fn memory() -> tokio_rusqlite::Result<Self> {
        let conn = SqliteConnection::open_in_memory().await?;
        ensure_host_event_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::Memory).await?;
        Ok(Self { conn })
    }

    fn encode_json<T: serde::Serialize>(value: &T) -> Result<String, lash_core::PluginError> {
        serde_json::to_string(value).map_err(|err| {
            lash_core::PluginError::Session(format!("failed to encode host event row: {err}"))
        })
    }

    fn decode_subscription(
        json: String,
    ) -> Result<lash_core::TriggerSubscriptionRecord, lash_core::PluginError> {
        serde_json::from_str(&json).map_err(|err| {
            lash_core::PluginError::Session(format!(
                "failed to decode host event subscription row: {err}"
            ))
        })
    }

    fn decode_occurrence(
        json: String,
    ) -> Result<lash_core::HostEventOccurrenceRecord, lash_core::PluginError> {
        serde_json::from_str(&json).map_err(|err| {
            lash_core::PluginError::Session(format!(
                "failed to decode host event occurrence row: {err}"
            ))
        })
    }
}

fn host_event_tx_outcome<T>(
    result: Result<T, lash_core::PluginError>,
) -> TxOutcome<Result<T, lash_core::PluginError>> {
    match result {
        Ok(value) => TxOutcome::Commit(Ok(value)),
        Err(err) => TxOutcome::Rollback(Err(err)),
    }
}

#[async_trait::async_trait]
impl lash_core::HostEventStore for SqliteHostEventStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn register_subscription(
        &self,
        draft: lash_core::TriggerSubscriptionDraft,
    ) -> Result<lash_core::TriggerSubscriptionRecord, lash_core::PluginError> {
        self.conn
            .write_flow(move |tx| {
                Ok(host_event_tx_outcome((|| {
                    tx.execute("INSERT INTO host_event_subscription_seq DEFAULT VALUES", [])
                        .map_err(process_sqlite_error)?;
                    let seq = tx.last_insert_rowid();
                    let handle = format!("trigger:{seq}");
                    let subscription_id = format!("subscription:{seq}");
                    let now = current_epoch_ms();
                    let record = lash_core::TriggerSubscriptionRecord {
                        subscription_id: subscription_id.clone(),
                        registrant: draft.registrant,
                        env_ref: draft.env_ref,
                        wake_target: draft.wake_target,
                        handle,
                        name: draft.name,
                        source_type: draft.source_type,
                        source_key: draft.source_key,
                        source: draft.source,
                        event_ty: draft.event_ty,
                        module_ref: draft.module_ref,
                        required_surface_ref: draft.required_surface_ref,
                        process_ref: draft.process_ref,
                        process_name: draft.process_name,
                        input_template: draft.input_template,
                        enabled: true,
                        created_at_ms: now,
                        updated_at_ms: now,
                    };
                    tx.execute(
                        "INSERT INTO host_event_trigger_subscriptions (
                            subscription_id, registrant_scope_id, handle, source_type, source_key,
                            enabled, created_at_ms, updated_at_ms, record_json
                         )
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                        params![
                            record.subscription_id.as_str(),
                            record.registrant_scope_id().as_str(),
                            record.handle.as_str(),
                            record.source_type.as_str(),
                            record.source_key.as_str(),
                            i64::from(record.enabled),
                            record.created_at_ms as i64,
                            record.updated_at_ms as i64,
                            Self::encode_json(&record)?,
                        ],
                    )
                    .map_err(process_sqlite_error)?;
                    Ok(record)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn list_subscriptions(
        &self,
        filter: lash_core::TriggerSubscriptionFilter,
    ) -> Result<Vec<lash_core::TriggerSubscriptionRecord>, lash_core::PluginError> {
        self.conn
            .call(move |conn| {
                Ok((|| {
                    let mut sql =
                        "SELECT record_json FROM host_event_trigger_subscriptions WHERE 1 = 1"
                            .to_string();
                    let mut values = Vec::<rusqlite::types::Value>::new();
                    if let Some(handle) = filter.handle.as_ref() {
                        sql.push_str(" AND handle = ?");
                        values.push(handle.clone().into());
                    }
                    if let Some(name) = filter.name.as_ref() {
                        sql.push_str(" AND json_extract(record_json, '$.name') = ?");
                        values.push(name.clone().into());
                    }
                    if let Some(source_type) = filter.source_type.as_ref() {
                        sql.push_str(" AND source_type = ?");
                        values.push(source_type.clone().into());
                    }
                    if let Some(source_key) = filter.source_key.as_ref() {
                        sql.push_str(" AND source_key = ?");
                        values.push(source_key.clone().into());
                    }
                    if let Some(enabled) = filter.enabled {
                        sql.push_str(" AND enabled = ?");
                        values.push(i64::from(enabled).into());
                    }
                    sql.push_str(" ORDER BY registrant_scope_id ASC, handle ASC");
                    let mut stmt = conn.prepare(&sql).map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map(rusqlite::params_from_iter(values.iter()), |row| {
                            row.get::<_, String>(0)
                        })
                        .map_err(process_sqlite_error)?;
                    let mut records = Vec::new();
                    for row in rows {
                        let record = Self::decode_subscription(row.map_err(process_sqlite_error)?)?;
                        if filter.matches(&record) {
                            records.push(record);
                        }
                    }
                    Ok(records)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn cancel_subscription(
        &self,
        session_id: &str,
        handle: &str,
    ) -> Result<bool, lash_core::PluginError> {
        let session_id = session_id.to_string();
        let handle = handle.to_string();
        self.conn
            .write_flow(move |tx| {
                Ok(host_event_tx_outcome((|| {
                    let json = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT record_json
                                 FROM host_event_trigger_subscriptions
                                 WHERE handle = ?1
                                 ORDER BY registrant_scope_id ASC",
                            )
                            .map_err(process_sqlite_error)?;
                        let rows = stmt
                            .query_map(params![handle.as_str()], |row| row.get::<_, String>(0))
                            .map_err(process_sqlite_error)?;
                        let mut matched = None;
                        for row in rows {
                            let json = row.map_err(process_sqlite_error)?;
                            let record = Self::decode_subscription(json.clone())?;
                            if record.registrant_session_id() == Some(session_id.as_str()) {
                                matched = Some(json);
                                break;
                            }
                        }
                        matched
                    };
                    let Some(json) = json else {
                        return Ok(false);
                    };
                    let mut record = Self::decode_subscription(json)?;
                    let changed = record.enabled;
                    record.enabled = false;
                    record.updated_at_ms = current_epoch_ms();
                    tx.execute(
                        "UPDATE host_event_trigger_subscriptions
                         SET enabled = ?3, updated_at_ms = ?4, record_json = ?5
                         WHERE registrant_scope_id = ?1 AND handle = ?2",
                        params![
                            record.registrant_scope_id().as_str(),
                            handle.as_str(),
                            i64::from(record.enabled),
                            record.updated_at_ms as i64,
                            Self::encode_json(&record)?,
                        ],
                    )
                    .map_err(process_sqlite_error)?;
                    Ok(changed)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn delete_session_subscriptions(
        &self,
        session_id: &str,
    ) -> Result<usize, lash_core::PluginError> {
        let session_id = session_id.to_string();
        self.conn
            .write_flow(move |tx| {
                Ok(host_event_tx_outcome((|| {
                    let rows = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT subscription_id, record_json
                                 FROM host_event_trigger_subscriptions
                                 ORDER BY registrant_scope_id ASC, handle ASC",
                            )
                            .map_err(process_sqlite_error)?;
                        let rows = stmt
                            .query_map([], |row| {
                                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                            })
                            .map_err(process_sqlite_error)?;
                        let mut rows_out = Vec::new();
                        for row in rows {
                            rows_out.push(row.map_err(process_sqlite_error)?);
                        }
                        rows_out
                    };
                    let mut deleted = 0usize;
                    for (subscription_id, json) in rows {
                        let record = Self::decode_subscription(json)?;
                        if record.registrant_session_id() != Some(session_id.as_str()) {
                            continue;
                        }
                        tx.execute(
                            "DELETE FROM host_event_trigger_subscriptions WHERE subscription_id = ?1",
                            params![subscription_id.as_str()],
                        )
                        .map_err(process_sqlite_error)?;
                        deleted = deleted.saturating_add(1);
                    }
                    Ok(deleted)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn record_occurrence(
        &self,
        request: lash_core::HostEventOccurrenceRequest,
    ) -> Result<lash_core::HostEventOccurrenceRecord, lash_core::PluginError> {
        lash_core::validate_host_event_occurrence_request(&request)?;
        let request_hash = lash_core::host_event_occurrence_request_hash(&request)?;
        let occurrence_id = lash_core::deterministic_occurrence_id(&request)?;
        self.conn
            .write_flow(move |tx| {
                Ok(host_event_tx_outcome((|| {
                    let existing: Option<(String, String)> = tx
                        .query_row(
                            "SELECT request_hash, record_json
                             FROM host_event_occurrences
                             WHERE idempotency_key = ?1",
                            params![request.idempotency_key.as_str()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        )
                        .optional()
                        .map_err(process_sqlite_error)?;
                    if let Some((existing_hash, existing_json)) = existing {
                        if existing_hash != request_hash {
                            return Err(lash_core::PluginError::Session(format!(
                                "host event occurrence idempotency conflict for `{}`",
                                request.idempotency_key
                            )));
                        }
                        return Self::decode_occurrence(existing_json);
                    }
                    let record = lash_core::HostEventOccurrenceRecord {
                        occurrence_id: occurrence_id.clone(),
                        source_type: request.source_type,
                        source_key: request.source_key,
                        payload: request.payload,
                        idempotency_key: request.idempotency_key,
                        source: request.source,
                        occurred_at_ms: current_epoch_ms(),
                    };
                    tx.execute(
                        "INSERT INTO host_event_occurrences (
                            occurrence_id, idempotency_key, request_hash, source_type,
                            source_key, occurred_at_ms, record_json
                         )
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            record.occurrence_id.as_str(),
                            record.idempotency_key.as_str(),
                            request_hash.as_str(),
                            record.source_type.as_str(),
                            record.source_key.as_str(),
                            record.occurred_at_ms as i64,
                            Self::encode_json(&record)?,
                        ],
                    )
                    .map_err(process_sqlite_error)?;
                    Ok(record)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn reserve_matching_deliveries(
        &self,
        occurrence_id: &str,
    ) -> Result<Vec<lash_core::TriggerDeliveryReservation>, lash_core::PluginError> {
        let occurrence_id = occurrence_id.to_string();
        self.conn
            .write_flow(move |tx| {
                Ok(host_event_tx_outcome((|| {
                    let occurrence_json: Option<String> = tx
                        .query_row(
                            "SELECT record_json
                             FROM host_event_occurrences
                             WHERE occurrence_id = ?1",
                            params![occurrence_id.as_str()],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(process_sqlite_error)?;
                    let Some(occurrence_json) = occurrence_json else {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown host event occurrence `{occurrence_id}`"
                        )));
                    };
                    let occurrence = Self::decode_occurrence(occurrence_json)?;
                    let subscriptions = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT record_json
                                 FROM host_event_trigger_subscriptions
                                 WHERE enabled = 1 AND source_type = ?1 AND source_key = ?2
                                 ORDER BY registrant_scope_id ASC, handle ASC",
                            )
                            .map_err(process_sqlite_error)?;
                        let rows = stmt
                            .query_map(
                                params![
                                    occurrence.source_type.as_str(),
                                    occurrence.source_key.as_str()
                                ],
                                |row| row.get::<_, String>(0),
                            )
                            .map_err(process_sqlite_error)?;
                        let mut subscriptions = Vec::new();
                        for row in rows {
                            subscriptions.push(Self::decode_subscription(
                                row.map_err(process_sqlite_error)?,
                            )?);
                        }
                        subscriptions
                    };
                    let mut reservations = Vec::new();
                    for subscription in subscriptions {
                        let process_id = lash_core::deterministic_delivery_process_id(
                            &occurrence.occurrence_id,
                            &subscription.subscription_id,
                        )?;
                        let inserted = tx
                            .execute(
                                "INSERT OR IGNORE INTO host_event_deliveries (
                                    occurrence_id, subscription_id, process_id, created_at_ms
                                 )
                                 VALUES (?1, ?2, ?3, ?4)",
                                params![
                                    occurrence.occurrence_id.as_str(),
                                    subscription.subscription_id.as_str(),
                                    process_id.as_str(),
                                    current_epoch_ms() as i64,
                                ],
                            )
                            .map_err(process_sqlite_error)?;
                        if inserted == 0 {
                            continue;
                        }
                        reservations.push(lash_core::TriggerDeliveryReservation {
                            occurrence: occurrence.clone(),
                            subscription,
                            process_id,
                        });
                    }
                    Ok(reservations)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }
}
