//! SQLite-backed runtime trigger store.
//!
//! This is the durable peer of [`SqliteProcessRegistry`]: it stores trigger
//! subscriptions and append-only trigger occurrences at deployment scope,
//! outside any session database.

use super::*;

pub struct SqliteTriggerStore {
    conn: SqliteConnection,
    clock: Arc<dyn lash_core::Clock>,
}

impl SqliteTriggerStore {
    pub async fn open(path: &Path) -> tokio_rusqlite::Result<Self> {
        Self::open_with_clock(path, Arc::new(lash_core::SystemClock)).await
    }

    pub async fn open_with_clock(
        path: &Path,
        clock: Arc<dyn lash_core::Clock>,
    ) -> tokio_rusqlite::Result<Self> {
        let conn = SqliteConnection::open(path).await?;
        ensure_trigger_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::File).await?;
        Ok(Self { conn, clock })
    }

    pub async fn memory() -> tokio_rusqlite::Result<Self> {
        Self::memory_with_clock(Arc::new(lash_core::SystemClock)).await
    }

    pub async fn memory_with_clock(
        clock: Arc<dyn lash_core::Clock>,
    ) -> tokio_rusqlite::Result<Self> {
        let conn = SqliteConnection::open_in_memory().await?;
        ensure_trigger_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::Memory).await?;
        Ok(Self { conn, clock })
    }

    fn encode_json<T: serde::Serialize>(value: &T) -> Result<String, lash_core::PluginError> {
        serde_json::to_string(value).map_err(|err| {
            lash_core::PluginError::Session(format!("failed to encode trigger row: {err}"))
        })
    }

    fn decode_subscription(
        json: String,
    ) -> Result<lash_core::TriggerSubscriptionRecord, lash_core::PluginError> {
        serde_json::from_str(&json).map_err(|err| {
            lash_core::PluginError::Session(format!(
                "failed to decode trigger subscription row: {err}"
            ))
        })
    }

    fn decode_occurrence(
        json: String,
    ) -> Result<lash_core::TriggerOccurrenceRecord, lash_core::PluginError> {
        serde_json::from_str(&json).map_err(|err| {
            lash_core::PluginError::Session(format!(
                "failed to decode trigger occurrence row: {err}"
            ))
        })
    }

    fn decode_delivery(
        occurrence_json: String,
        subscription_json: String,
        process_id: String,
        created_at_ms: i64,
        reservation_status: lash_core::TriggerDeliveryReservationStatus,
    ) -> Result<lash_core::TriggerDeliveryReservation, lash_core::PluginError> {
        Ok(lash_core::TriggerDeliveryReservation {
            occurrence: Self::decode_occurrence(occurrence_json)?,
            subscription: Self::decode_subscription(subscription_json)?,
            process_id,
            created_at_ms: created_at_ms as u64,
            reservation_status,
        })
    }

    async fn list_deliveries_where(
        &self,
        where_clause: &'static str,
        value: String,
    ) -> Result<Vec<lash_core::TriggerDeliveryReservation>, lash_core::PluginError> {
        self.conn
            .call(move |conn| {
                Ok((|| {
                    let sql = format!(
                        "SELECT d.process_id, d.created_at_ms, o.record_json, s.record_json
                         FROM trigger_deliveries d
                         JOIN trigger_occurrences o ON o.occurrence_id = d.occurrence_id
                         JOIN trigger_subscriptions s ON s.subscription_id = d.subscription_id
                         WHERE {where_clause}
                         ORDER BY d.created_at_ms ASC, d.occurrence_id ASC, d.subscription_id ASC"
                    );
                    let mut stmt = conn.prepare(&sql).map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map(params![value.as_str()], |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, i64>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, String>(3)?,
                            ))
                        })
                        .map_err(process_sqlite_error)?;
                    let mut deliveries = Vec::new();
                    for row in rows {
                        let (process_id, created_at_ms, occurrence_json, subscription_json) =
                            row.map_err(process_sqlite_error)?;
                        deliveries.push(Self::decode_delivery(
                            occurrence_json,
                            subscription_json,
                            process_id,
                            created_at_ms,
                            lash_core::TriggerDeliveryReservationStatus::AlreadyReserved,
                        )?);
                    }
                    Ok(deliveries)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }
}

fn trigger_tx_outcome<T>(
    result: Result<T, lash_core::PluginError>,
) -> TxOutcome<Result<T, lash_core::PluginError>> {
    match result {
        Ok(value) => TxOutcome::Commit(Ok(value)),
        Err(err) => TxOutcome::Rollback(Err(err)),
    }
}

#[async_trait::async_trait]
impl lash_core::TriggerStore for SqliteTriggerStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn register_subscription(
        &self,
        draft: lash_core::TriggerSubscriptionDraft,
    ) -> Result<lash_core::TriggerSubscriptionRecord, lash_core::PluginError> {
        draft.validate()?;
        let now = self.clock.timestamp_ms();
        self.conn
            .write_flow(move |tx| {
                Ok(trigger_tx_outcome((|| {
                    tx.execute("INSERT INTO trigger_subscription_seq DEFAULT VALUES", [])
                        .map_err(process_sqlite_error)?;
                    let seq = tx.last_insert_rowid();
                    let handle = format!("trigger:{seq}");
                    let subscription_id = format!("subscription:{seq}");
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
                        payload_schema: draft.payload_schema,
                        target: draft.target,
                        target_identity: draft.target_identity,
                        event_types: draft.event_types,
                        input_template: draft.input_template,
                        target_label: draft.target_label,
                        enabled: true,
                        created_at_ms: now,
                        updated_at_ms: now,
                    };
                    tx.execute(
                        "INSERT INTO trigger_subscriptions (
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
                        "SELECT subscription_id, record_json FROM trigger_subscriptions WHERE 1 = 1"
                            .to_string();
                    let mut values = Vec::<rusqlite::types::Value>::new();
                    if let Some(registrant_scope_id) = filter.effective_registrant_scope_id() {
                        sql.push_str(" AND registrant_scope_id = ?");
                        values.push(registrant_scope_id.into());
                    }
                    if let Some(handle) = filter.handle.as_ref() {
                        sql.push_str(" AND handle = ?");
                        values.push(handle.clone().into());
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
                            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                        })
                        .map_err(process_sqlite_error)?;
                    let mut records = Vec::new();
                    for row in rows {
                        let (subscription_id, json) = row.map_err(process_sqlite_error)?;
                        let record = match Self::decode_subscription(json) {
                            Ok(record) => record,
                            Err(err) => {
                                tracing::warn!(
                                    error = %err,
                                    subscription_id,
                                    "skipping malformed trigger subscription during listing"
                                );
                                continue;
                            }
                        };
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
        registrant_scope_id: &str,
        handle: &str,
    ) -> Result<bool, lash_core::PluginError> {
        self.set_subscription_enabled(registrant_scope_id, handle, false)
            .await
    }

    async fn set_subscription_enabled(
        &self,
        registrant_scope_id: &str,
        handle: &str,
        enabled: bool,
    ) -> Result<bool, lash_core::PluginError> {
        let registrant_scope_id = registrant_scope_id.to_string();
        let handle = handle.to_string();
        let updated_at_ms = self.clock.timestamp_ms();
        self.conn
            .write_flow(move |tx| {
                Ok(trigger_tx_outcome((|| {
                    let selected: Option<(String, i64, String)> = tx
                        .query_row(
                            "SELECT subscription_id, enabled, record_json
                             FROM trigger_subscriptions
                             WHERE registrant_scope_id = ?1 AND handle = ?2",
                            params![registrant_scope_id.as_str(), handle.as_str()],
                            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                        )
                        .optional()
                        .map_err(process_sqlite_error)?;
                    let Some((subscription_id, stored_enabled, json)) = selected else {
                        return Ok(false);
                    };
                    let changed = (stored_enabled != 0) != enabled;
                    match Self::decode_subscription(json) {
                        Ok(mut record) => {
                            record.enabled = enabled;
                            record.updated_at_ms = updated_at_ms;
                            tx.execute(
                                "UPDATE trigger_subscriptions
                                 SET enabled = ?3, updated_at_ms = ?4, record_json = ?5
                                 WHERE subscription_id = ?1 AND handle = ?2",
                                params![
                                    subscription_id.as_str(),
                                    handle.as_str(),
                                    i64::from(record.enabled),
                                    record.updated_at_ms as i64,
                                    Self::encode_json(&record)?,
                                ],
                            )
                            .map_err(process_sqlite_error)?;
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                subscription_id,
                                handle,
                                "disabling malformed trigger subscription without rewriting record JSON"
                            );
                            tx.execute(
                                "UPDATE trigger_subscriptions
                                 SET enabled = ?3, updated_at_ms = ?4
                                 WHERE subscription_id = ?1 AND handle = ?2",
                                params![
                                    subscription_id.as_str(),
                                    handle.as_str(),
                                    i64::from(enabled),
                                    updated_at_ms as i64,
                                ],
                            )
                            .map_err(process_sqlite_error)?;
                        }
                    }
                    Ok(changed)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn delete_subscription(
        &self,
        registrant_scope_id: &str,
        handle: &str,
    ) -> Result<bool, lash_core::PluginError> {
        let registrant_scope_id = registrant_scope_id.to_string();
        let handle = handle.to_string();
        self.conn
            .write_flow(move |tx| {
                Ok(trigger_tx_outcome((|| {
                    let deleted = tx
                        .execute(
                            "DELETE FROM trigger_subscriptions
                             WHERE registrant_scope_id = ?1 AND handle = ?2",
                            params![registrant_scope_id.as_str(), handle.as_str()],
                        )
                        .map_err(process_sqlite_error)?;
                    Ok(deleted != 0)
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
                Ok(trigger_tx_outcome((|| {
                    let mut stmt = tx
                        .prepare("SELECT subscription_id, record_json FROM trigger_subscriptions")
                        .map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map([], |row| {
                            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                        })
                        .map_err(process_sqlite_error)?;
                    let mut subscription_ids = Vec::new();
                    for row in rows {
                        let (subscription_id, json) = row.map_err(process_sqlite_error)?;
                        let record = match Self::decode_subscription(json) {
                            Ok(record) => record,
                            Err(err) => {
                                tracing::warn!(
                                    error = %err,
                                    subscription_id,
                                    "skipping malformed trigger subscription during session delete"
                                );
                                continue;
                            }
                        };
                        if record.registrant_session_id() == Some(session_id.as_str()) {
                            subscription_ids.push(subscription_id);
                        }
                    }
                    drop(stmt);
                    let mut deleted = 0usize;
                    for subscription_id in subscription_ids {
                        deleted = deleted.saturating_add(
                            tx.execute(
                                "DELETE FROM trigger_subscriptions WHERE subscription_id = ?1",
                                params![subscription_id.as_str()],
                            )
                            .map_err(process_sqlite_error)?,
                        );
                    }
                    Ok(deleted)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn record_occurrence(
        &self,
        request: lash_core::TriggerOccurrenceRequest,
    ) -> Result<lash_core::TriggerOccurrenceRecord, lash_core::PluginError> {
        lash_core::validate_trigger_occurrence_request(&request)?;
        let request_hash = lash_core::trigger_occurrence_request_hash(&request)?;
        let occurrence_id = lash_core::deterministic_occurrence_id(&request)?;
        let occurred_at_ms = self.clock.timestamp_ms();
        self.conn
            .write_flow(move |tx| {
                Ok(trigger_tx_outcome((|| {
                    let existing: Option<(String, String)> = tx
                        .query_row(
                            "SELECT request_hash, record_json
                             FROM trigger_occurrences
                             WHERE idempotency_key = ?1",
                            params![request.idempotency_key.as_str()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        )
                        .optional()
                        .map_err(process_sqlite_error)?;
                    if let Some((existing_hash, existing_json)) = existing {
                        if existing_hash != request_hash {
                            return Err(lash_core::PluginError::Session(format!(
                                "trigger occurrence idempotency conflict for `{}`",
                                request.idempotency_key
                            )));
                        }
                        return Self::decode_occurrence(existing_json);
                    }
                    let record = lash_core::TriggerOccurrenceRecord {
                        occurrence_id: occurrence_id.clone(),
                        source_type: request.source_type,
                        source_key: request.source_key,
                        payload: request.payload,
                        idempotency_key: request.idempotency_key,
                        source: request.source,
                        session_id: request.session_id,
                        occurred_at_ms,
                    };
                    tx.execute(
                        "INSERT INTO trigger_occurrences (
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

    async fn list_occurrences(
        &self,
        filter: lash_core::TriggerOccurrenceFilter,
    ) -> Result<Vec<lash_core::TriggerOccurrenceRecord>, lash_core::PluginError> {
        self.conn
            .call(move |conn| {
                Ok((|| {
                    let mut sql =
                        "SELECT occurrence_id, record_json FROM trigger_occurrences WHERE 1 = 1"
                            .to_string();
                    let mut values = Vec::<rusqlite::types::Value>::new();
                    if let Some(source_type) = filter.source_type.as_ref() {
                        sql.push_str(" AND source_type = ?");
                        values.push(source_type.clone().into());
                    }
                    if let Some(source_key) = filter.source_key.as_ref() {
                        sql.push_str(" AND source_key = ?");
                        values.push(source_key.clone().into());
                    }
                    if let Some(start_ms) = filter.occurred_at_start_ms {
                        sql.push_str(" AND occurred_at_ms >= ?");
                        values.push((start_ms as i64).into());
                    }
                    if let Some(end_ms) = filter.occurred_at_end_ms {
                        sql.push_str(" AND occurred_at_ms < ?");
                        values.push((end_ms as i64).into());
                    }
                    sql.push_str(" ORDER BY occurred_at_ms ASC, occurrence_id ASC");
                    let mut stmt = conn.prepare(&sql).map_err(process_sqlite_error)?;
                    let rows = stmt
                        .query_map(rusqlite::params_from_iter(values.iter()), |row| {
                            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                        })
                        .map_err(process_sqlite_error)?;
                    let mut records = Vec::new();
                    for row in rows {
                        let (occurrence_id, json) = row.map_err(process_sqlite_error)?;
                        match Self::decode_occurrence(json) {
                            Ok(record) => records.push(record),
                            Err(err) => tracing::warn!(
                                error = %err,
                                occurrence_id,
                                "skipping malformed trigger occurrence during listing"
                            ),
                        }
                    }
                    Ok(records)
                })())
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn reserve_matching_deliveries(
        &self,
        occurrence_id: &str,
    ) -> Result<Vec<lash_core::TriggerDeliveryReservation>, lash_core::PluginError> {
        let occurrence_id = occurrence_id.to_string();
        let created_at_ms = self.clock.timestamp_ms();
        self.conn
            .write_flow(move |tx| {
                Ok(trigger_tx_outcome((|| {
                    let occurrence_json: Option<String> = tx
                        .query_row(
                            "SELECT record_json
                             FROM trigger_occurrences
                             WHERE occurrence_id = ?1",
                            params![occurrence_id.as_str()],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(process_sqlite_error)?;
                    let Some(occurrence_json) = occurrence_json else {
                        return Err(lash_core::PluginError::Session(format!(
                            "unknown trigger occurrence `{occurrence_id}`"
                        )));
                    };
                    let occurrence = Self::decode_occurrence(occurrence_json)?;
                    let subscriptions = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT subscription_id, record_json
                                 FROM trigger_subscriptions
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
                                |row| {
                                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                                },
                            )
                            .map_err(process_sqlite_error)?;
                        let mut subscriptions = Vec::new();
                        for row in rows {
                            let (subscription_id, json) = row.map_err(process_sqlite_error)?;
                            match Self::decode_subscription(json) {
                                Ok(subscription)
                                    if occurrence.session_id.as_deref().is_none_or(|session_id| {
                                        subscription.registrant_session_id() == Some(session_id)
                                    }) =>
                                {
                                    subscriptions.push(subscription);
                                }
                                Ok(_) => {}
                                Err(err) => tracing::warn!(
                                    error = %err,
                                    subscription_id,
                                    occurrence_id = %occurrence.occurrence_id,
                                    "skipping malformed trigger subscription during delivery reservation"
                                ),
                            }
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
                                "INSERT OR IGNORE INTO trigger_deliveries (
                                    occurrence_id, subscription_id, process_id, created_at_ms
                                 )
                                 VALUES (?1, ?2, ?3, ?4)",
                                params![
                                    occurrence.occurrence_id.as_str(),
                                    subscription.subscription_id.as_str(),
                                    process_id.as_str(),
                                    created_at_ms as i64,
                                ],
                            )
                            .map_err(process_sqlite_error)?;
                        let stored_created_at_ms: i64 = tx
                            .query_row(
                                "SELECT created_at_ms FROM trigger_deliveries
                                 WHERE occurrence_id = ?1 AND subscription_id = ?2",
                                params![
                                    occurrence.occurrence_id.as_str(),
                                    subscription.subscription_id.as_str()
                                ],
                                |row| row.get(0),
                            )
                            .map_err(process_sqlite_error)?;
                        reservations.push(lash_core::TriggerDeliveryReservation {
                            occurrence: occurrence.clone(),
                            subscription,
                            process_id,
                            created_at_ms: stored_created_at_ms as u64,
                            reservation_status: if inserted == 0 {
                                lash_core::TriggerDeliveryReservationStatus::AlreadyReserved
                            } else {
                                lash_core::TriggerDeliveryReservationStatus::Reserved
                            },
                        });
                    }
                    Ok(reservations)
                })()))
            })
            .await
            .map_err(process_sqlite_error)?
    }

    async fn list_deliveries_by_occurrence_id(
        &self,
        occurrence_id: &str,
    ) -> Result<Vec<lash_core::TriggerDeliveryReservation>, lash_core::PluginError> {
        self.list_deliveries_where("d.occurrence_id = ?1", occurrence_id.to_string())
            .await
    }

    async fn list_deliveries_by_subscription_id(
        &self,
        subscription_id: &str,
    ) -> Result<Vec<lash_core::TriggerDeliveryReservation>, lash_core::PluginError> {
        self.list_deliveries_where("d.subscription_id = ?1", subscription_id.to_string())
            .await
    }

    async fn list_deliveries_by_process_id(
        &self,
        process_id: &str,
    ) -> Result<Vec<lash_core::TriggerDeliveryReservation>, lash_core::PluginError> {
        self.list_deliveries_where("d.process_id = ?1", process_id.to_string())
            .await
    }
}
