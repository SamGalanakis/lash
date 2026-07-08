#[async_trait::async_trait]
impl TriggerStore for PostgresTriggerStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn register_subscription(
        &self,
        draft: TriggerSubscriptionDraft,
    ) -> Result<TriggerSubscriptionRecord, PluginError> {
        draft.validate()?;
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let seq: i64 = sqlx::query_scalar("SELECT nextval('lash_trigger_subscription_seq')")
            .fetch_one(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?;
        let handle = format!("trigger:{seq}");
        let subscription_id = format!("subscription:{seq}");
        let now = current_epoch_ms();
        let record = TriggerSubscriptionRecord {
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
        sqlx::query(
            "INSERT INTO lash_trigger_subscriptions (
                subscription_id, registrant_scope_id, handle, source_type, source_key,
                enabled, created_at_ms, updated_at_ms, record_json
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(&record.subscription_id)
        .bind(record.registrant_scope_id())
        .bind(&record.handle)
        .bind(&record.source_type)
        .bind(&record.source_key)
        .bind(record.enabled)
        .bind(record.created_at_ms as i64)
        .bind(record.updated_at_ms as i64)
        .bind(serde_json::to_string(&record).map_err(process_decode_error)?)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(record)
    }

    async fn list_subscriptions(
        &self,
        filter: TriggerSubscriptionFilter,
    ) -> Result<Vec<TriggerSubscriptionRecord>, PluginError> {
        let mut query = sqlx::QueryBuilder::<sqlx::Postgres>::new(
            "SELECT subscription_id, record_json FROM lash_trigger_subscriptions WHERE TRUE",
        );
        if let Some(registrant_scope_id) = filter.effective_registrant_scope_id() {
            query
                .push(" AND registrant_scope_id = ")
                .push_bind(registrant_scope_id);
        }
        if let Some(handle) = filter.handle.as_ref() {
            query.push(" AND handle = ").push_bind(handle);
        }
        if let Some(source_type) = filter.source_type.as_ref() {
            query.push(" AND source_type = ").push_bind(source_type);
        }
        if let Some(source_key) = filter.source_key.as_ref() {
            query.push(" AND source_key = ").push_bind(source_key);
        }
        if let Some(enabled) = filter.enabled {
            query.push(" AND enabled = ").push_bind(enabled);
        }
        query.push(" ORDER BY registrant_scope_id ASC, handle ASC");
        let rows = query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(plugin_sqlx_error)?;
        let mut records = Vec::new();
        for row in rows {
            let subscription_id: String = row.get(0);
            let json: String = row.get(1);
            let record: TriggerSubscriptionRecord = match serde_json::from_str(&json) {
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
    }

    async fn cancel_subscription(
        &self,
        registrant_scope_id: &str,
        handle: &str,
    ) -> Result<bool, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let row = sqlx::query(
            "SELECT enabled, record_json FROM lash_trigger_subscriptions
             WHERE registrant_scope_id = $1 AND handle = $2
             FOR UPDATE",
        )
        .bind(registrant_scope_id)
        .bind(handle)
        .fetch_optional(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let Some(row) = row else {
            tx.commit().await.map_err(plugin_sqlx_error)?;
            return Ok(false);
        };
        let changed: bool = row.get(0);
        let json: String = row.get(1);
        let updated_at_ms = current_epoch_ms();
        match serde_json::from_str::<TriggerSubscriptionRecord>(&json) {
            Ok(mut record) => {
                record.enabled = false;
                record.updated_at_ms = updated_at_ms;
                sqlx::query(
                    "UPDATE lash_trigger_subscriptions
                     SET enabled = $3, updated_at_ms = $4, record_json = $5
                     WHERE registrant_scope_id = $1 AND handle = $2",
                )
                .bind(registrant_scope_id)
                .bind(handle)
                .bind(record.enabled)
                .bind(record.updated_at_ms as i64)
                .bind(serde_json::to_string(&record).map_err(process_decode_error)?)
                .execute(&mut *tx)
                .await
                .map_err(plugin_sqlx_error)?;
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    registrant_scope_id,
                    handle,
                    "disabling malformed trigger subscription without rewriting record JSON"
                );
                sqlx::query(
                    "UPDATE lash_trigger_subscriptions
                     SET enabled = FALSE, updated_at_ms = $3
                     WHERE registrant_scope_id = $1 AND handle = $2",
                )
                .bind(registrant_scope_id)
                .bind(handle)
                .bind(updated_at_ms as i64)
                .execute(&mut *tx)
                .await
                .map_err(plugin_sqlx_error)?;
            }
        }
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(changed)
    }

    async fn delete_session_subscriptions(&self, session_id: &str) -> Result<usize, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let root_scope_id = format!("session:{session_id}");
        let frame_scope_pattern = format!("{}/frame:%", escape_postgres_like(&root_scope_id));
        let deleted = sqlx::query(
            "DELETE FROM lash_trigger_subscriptions
             WHERE registrant_scope_id = $1
                OR registrant_scope_id LIKE $2 ESCAPE '\\'",
        )
        .bind(&root_scope_id)
        .bind(&frame_scope_pattern)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?
        .rows_affected() as usize;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(deleted)
    }

    async fn record_occurrence(
        &self,
        request: TriggerOccurrenceRequest,
    ) -> Result<TriggerOccurrenceRecord, PluginError> {
        lash_core::validate_trigger_occurrence_request(&request)?;
        let request_hash = lash_core::trigger_occurrence_request_hash(&request)?;
        let occurrence_id = lash_core::deterministic_occurrence_id(&request)?;
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let existing = sqlx::query(
            "SELECT request_hash, record_json
             FROM lash_trigger_occurrences
             WHERE idempotency_key = $1
             FOR UPDATE",
        )
        .bind(&request.idempotency_key)
        .fetch_optional(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        if let Some(row) = existing {
            let existing_hash: String = row.get(0);
            let existing_json: String = row.get(1);
            if existing_hash != request_hash {
                return Err(PluginError::Session(format!(
                    "trigger occurrence idempotency conflict for `{}`",
                    request.idempotency_key
                )));
            }
            let record = serde_json::from_str(&existing_json).map_err(process_decode_error)?;
            tx.commit().await.map_err(plugin_sqlx_error)?;
            return Ok(record);
        }
        let record = TriggerOccurrenceRecord {
            occurrence_id: occurrence_id.clone(),
            source_type: request.source_type,
            source_key: request.source_key,
            payload: request.payload,
            idempotency_key: request.idempotency_key.clone(),
            source: request.source,
            occurred_at_ms: current_epoch_ms(),
        };
        sqlx::query(
            "INSERT INTO lash_trigger_occurrences (
                occurrence_id, idempotency_key, request_hash, source_type, source_key,
                occurred_at_ms, record_json
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(&record.occurrence_id)
        .bind(&record.idempotency_key)
        .bind(&request_hash)
        .bind(&record.source_type)
        .bind(&record.source_key)
        .bind(record.occurred_at_ms as i64)
        .bind(serde_json::to_string(&record).map_err(process_decode_error)?)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(record)
    }

    async fn list_occurrences(
        &self,
        filter: lash_core::TriggerOccurrenceFilter,
    ) -> Result<Vec<TriggerOccurrenceRecord>, PluginError> {
        let mut query = sqlx::QueryBuilder::<sqlx::Postgres>::new(
            "SELECT occurrence_id, record_json FROM lash_trigger_occurrences WHERE TRUE",
        );
        if let Some(source_type) = filter.source_type.as_ref() {
            query.push(" AND source_type = ").push_bind(source_type);
        }
        if let Some(source_key) = filter.source_key.as_ref() {
            query.push(" AND source_key = ").push_bind(source_key);
        }
        if let Some(start_ms) = filter.occurred_at_start_ms {
            query
                .push(" AND occurred_at_ms >= ")
                .push_bind(start_ms as i64);
        }
        if let Some(end_ms) = filter.occurred_at_end_ms {
            query
                .push(" AND occurred_at_ms < ")
                .push_bind(end_ms as i64);
        }
        query.push(" ORDER BY occurred_at_ms ASC, occurrence_id ASC");
        let rows = query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(plugin_sqlx_error)?;
        let mut records = Vec::new();
        for row in rows {
            let occurrence_id: String = row.get(0);
            let json: String = row.get(1);
            match serde_json::from_str(&json) {
                Ok(record) => records.push(record),
                Err(err) => tracing::warn!(
                    error = %err,
                    occurrence_id,
                    "skipping malformed trigger occurrence during listing"
                ),
            }
        }
        Ok(records)
    }

    async fn reserve_matching_deliveries(
        &self,
        occurrence_id: &str,
    ) -> Result<Vec<TriggerDeliveryReservation>, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let occurrence_json: Option<String> = sqlx::query_scalar(
            "SELECT record_json FROM lash_trigger_occurrences WHERE occurrence_id = $1",
        )
        .bind(occurrence_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let Some(occurrence_json) = occurrence_json else {
            return Err(PluginError::Session(format!(
                "unknown trigger occurrence `{occurrence_id}`"
            )));
        };
        let occurrence: TriggerOccurrenceRecord =
            serde_json::from_str(&occurrence_json).map_err(process_decode_error)?;
        let rows = sqlx::query(
            "SELECT subscription_id, record_json FROM lash_trigger_subscriptions
             WHERE enabled = TRUE AND source_type = $1 AND source_key = $2
             ORDER BY registrant_scope_id ASC, handle ASC",
        )
        .bind(&occurrence.source_type)
        .bind(&occurrence.source_key)
        .fetch_all(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut deliveries = Vec::new();
        for row in rows {
            let subscription_id: String = row.get(0);
            let json: String = row.get(1);
            let subscription: TriggerSubscriptionRecord = match serde_json::from_str(&json) {
                Ok(subscription) => subscription,
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        subscription_id,
                        occurrence_id = %occurrence.occurrence_id,
                        "skipping malformed trigger subscription during delivery reservation"
                    );
                    continue;
                }
            };
            let process_id = lash_core::deterministic_delivery_process_id(
                &occurrence.occurrence_id,
                &subscription.subscription_id,
            )?;
            let created_at_ms = current_epoch_ms();
            let inserted = sqlx::query(
                "INSERT INTO lash_trigger_deliveries (
                    occurrence_id, subscription_id, process_id, created_at_ms
                 )
                 VALUES ($1, $2, $3, $4)
                 ON CONFLICT DO NOTHING",
            )
            .bind(&occurrence.occurrence_id)
            .bind(&subscription.subscription_id)
            .bind(&process_id)
            .bind(created_at_ms as i64)
            .execute(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?
            .rows_affected();
            let stored_created_at_ms: i64 = sqlx::query_scalar(
                "SELECT created_at_ms FROM lash_trigger_deliveries
                 WHERE occurrence_id = $1 AND subscription_id = $2",
            )
            .bind(&occurrence.occurrence_id)
            .bind(&subscription.subscription_id)
            .fetch_one(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?;
            deliveries.push(TriggerDeliveryReservation {
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
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(deliveries)
    }

    async fn list_deliveries_by_occurrence_id(
        &self,
        occurrence_id: &str,
    ) -> Result<Vec<TriggerDeliveryReservation>, PluginError> {
        list_deliveries_where(&self.pool, "d.occurrence_id = $1", occurrence_id.to_string()).await
    }

    async fn list_deliveries_by_subscription_id(
        &self,
        subscription_id: &str,
    ) -> Result<Vec<TriggerDeliveryReservation>, PluginError> {
        list_deliveries_where(&self.pool, "d.subscription_id = $1", subscription_id.to_string())
            .await
    }

    async fn list_deliveries_by_process_id(
        &self,
        process_id: &str,
    ) -> Result<Vec<TriggerDeliveryReservation>, PluginError> {
        list_deliveries_where(&self.pool, "d.process_id = $1", process_id.to_string()).await
    }
}

fn escape_postgres_like(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        if matches!(ch, '\\' | '%' | '_') {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

async fn list_deliveries_where(
    pool: &sqlx::PgPool,
    where_clause: &'static str,
    value: String,
) -> Result<Vec<TriggerDeliveryReservation>, PluginError> {
    let sql = format!(
        "SELECT d.process_id, d.created_at_ms, o.record_json, s.record_json
         FROM lash_trigger_deliveries d
         JOIN lash_trigger_occurrences o ON o.occurrence_id = d.occurrence_id
         JOIN lash_trigger_subscriptions s ON s.subscription_id = d.subscription_id
         WHERE {where_clause}
         ORDER BY d.created_at_ms ASC, d.occurrence_id ASC, d.subscription_id ASC"
    );
    let rows = sqlx::query(&sql)
        .bind(value)
        .fetch_all(pool)
        .await
        .map_err(plugin_sqlx_error)?;
    let mut deliveries = Vec::new();
    for row in rows {
        let process_id: String = row.get(0);
        let created_at_ms: i64 = row.get(1);
        let occurrence_json: String = row.get(2);
        let subscription_json: String = row.get(3);
        deliveries.push(TriggerDeliveryReservation {
            occurrence: serde_json::from_str(&occurrence_json).map_err(process_decode_error)?,
            subscription: serde_json::from_str(&subscription_json).map_err(process_decode_error)?,
            process_id,
            created_at_ms: created_at_ms as u64,
            reservation_status: lash_core::TriggerDeliveryReservationStatus::AlreadyReserved,
        });
    }
    Ok(deliveries)
}
