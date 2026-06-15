#[async_trait::async_trait]
impl TriggerStore for PostgresTriggerStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn register_subscription(
        &self,
        draft: TriggerSubscriptionDraft,
    ) -> Result<TriggerSubscriptionRecord, PluginError> {
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
            event_ty: draft.event_ty,
            module_ref: draft.module_ref,
            host_requirements_ref: draft.host_requirements_ref,
            process_ref: draft.process_ref,
            process_name: draft.process_name,
            input_template: draft.input_template,
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
        let rows = sqlx::query(
            "SELECT record_json FROM lash_trigger_subscriptions
             ORDER BY registrant_scope_id ASC, handle ASC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut records = Vec::new();
        for row in rows {
            let json: String = row.get(0);
            let record: TriggerSubscriptionRecord =
                serde_json::from_str(&json).map_err(process_decode_error)?;
            if filter.matches(&record) {
                records.push(record);
            }
        }
        Ok(records)
    }

    async fn cancel_subscription(
        &self,
        session_id: &str,
        handle: &str,
    ) -> Result<bool, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let rows = sqlx::query(
            "SELECT record_json FROM lash_trigger_subscriptions
             WHERE handle = $1
             ORDER BY registrant_scope_id ASC
             FOR UPDATE",
        )
        .bind(handle)
        .fetch_all(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut json = None;
        for row in rows {
            let candidate: String = row.get(0);
            let record: TriggerSubscriptionRecord =
                serde_json::from_str(&candidate).map_err(process_decode_error)?;
            if record.registrant_session_id() == Some(session_id) {
                json = Some(candidate);
                break;
            }
        }
        let Some(json) = json else {
            tx.commit().await.map_err(plugin_sqlx_error)?;
            return Ok(false);
        };
        let mut record: TriggerSubscriptionRecord =
            serde_json::from_str(&json).map_err(process_decode_error)?;
        let changed = record.enabled;
        record.enabled = false;
        record.updated_at_ms = current_epoch_ms();
        sqlx::query(
            "UPDATE lash_trigger_subscriptions
             SET enabled = $3, updated_at_ms = $4, record_json = $5
             WHERE registrant_scope_id = $1 AND handle = $2",
        )
        .bind(record.registrant_scope_id())
        .bind(handle)
        .bind(record.enabled)
        .bind(record.updated_at_ms as i64)
        .bind(serde_json::to_string(&record).map_err(process_decode_error)?)
        .execute(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(changed)
    }

    async fn delete_session_subscriptions(&self, session_id: &str) -> Result<usize, PluginError> {
        let mut tx = self.pool.begin().await.map_err(plugin_sqlx_error)?;
        let rows = sqlx::query(
            "SELECT subscription_id, record_json
             FROM lash_trigger_subscriptions
             ORDER BY registrant_scope_id ASC, handle ASC
             FOR UPDATE",
        )
        .fetch_all(&mut *tx)
        .await
        .map_err(plugin_sqlx_error)?;
        let mut deleted = 0usize;
        for row in rows {
            let subscription_id: String = row.get(0);
            let json: String = row.get(1);
            let record: TriggerSubscriptionRecord =
                serde_json::from_str(&json).map_err(process_decode_error)?;
            if record.registrant_session_id() != Some(session_id) {
                continue;
            }
            sqlx::query("DELETE FROM lash_trigger_subscriptions WHERE subscription_id = $1")
                .bind(&subscription_id)
                .execute(&mut *tx)
                .await
                .map_err(plugin_sqlx_error)?;
            deleted = deleted.saturating_add(1);
        }
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
            "SELECT record_json FROM lash_trigger_subscriptions
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
            let json: String = row.get(0);
            let subscription: TriggerSubscriptionRecord =
                serde_json::from_str(&json).map_err(process_decode_error)?;
            let process_id = lash_core::deterministic_delivery_process_id(
                &occurrence.occurrence_id,
                &subscription.subscription_id,
            )?;
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
            .bind(current_epoch_ms() as i64)
            .execute(&mut *tx)
            .await
            .map_err(plugin_sqlx_error)?
            .rows_affected();
            if inserted == 0 {
                continue;
            }
            deliveries.push(TriggerDeliveryReservation {
                occurrence: occurrence.clone(),
                subscription,
                process_id,
            });
        }
        tx.commit().await.map_err(plugin_sqlx_error)?;
        Ok(deliveries)
    }
}
