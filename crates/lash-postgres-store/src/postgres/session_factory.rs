#[async_trait::async_trait]
impl SessionStoreFactory for PostgresSessionStoreFactory {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        let store = PostgresSessionStore {
            pool: self.pool.clone(),
            session_id: Some(request.session_id.clone()),
            bound_session: Arc::new(OnceLock::new()),
        };
        if store
            .load_session_meta()
            .await
            .map_err(|err| err.to_string())?
            .is_none()
        {
            store
                .save_session_meta(SessionMeta {
                    session_id: request.session_id.clone(),
                    session_name: request.session_id.clone(),
                    created_at: current_timestamp_string(),
                    model: request.policy.model.id.clone(),
                    cwd: std::env::current_dir()
                        .ok()
                        .and_then(|path| path.to_str().map(str::to_string)),
                    relation: request.relation.clone(),
                })
                .await
                .map_err(|err| err.to_string())?;
        }
        Ok(Arc::new(store))
    }

    async fn open_existing_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Option<Arc<dyn RuntimePersistence>>, String> {
        let store = PostgresSessionStore {
            pool: self.pool.clone(),
            session_id: Some(request.session_id.clone()),
            bound_session: Arc::new(OnceLock::new()),
        };
        if store
            .load_session_meta()
            .await
            .map_err(|err| err.to_string())?
            .is_some()
        {
            Ok(Some(Arc::new(store)))
        } else {
            Ok(None)
        }
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let mut tx = self.pool.begin().await.map_err(|err| err.to_string())?;
        for sql in [
            "DELETE FROM lash_queued_work_items WHERE batch_id IN (SELECT batch_id FROM lash_queued_work_batches WHERE session_id = $1)",
            "DELETE FROM lash_queued_work_batches WHERE session_id = $1",
            "DELETE FROM lash_usage_deltas WHERE session_id = $1",
            "DELETE FROM lash_graph_nodes WHERE session_id = $1",
            "DELETE FROM lash_runtime_turn_commits WHERE session_id = $1",
            "DELETE FROM lash_session_meta WHERE session_id = $1",
            "DELETE FROM lash_sessions WHERE session_id = $1",
            "DELETE FROM lash_attachment_manifest WHERE session_id = $1",
        ] {
            sqlx::query(sql)
                .bind(session_id)
                .execute(&mut *tx)
                .await
                .map_err(|err| err.to_string())?;
        }
        tx.commit().await.map_err(|err| err.to_string())
    }
}

#[derive(Clone, Debug)]
struct QueuedBatchRow {
    enqueue_seq: u64,
    batch_id: String,
    session_id: String,
    source_key: Option<String>,
    delivery_policy: DeliveryPolicy,
    slot_policy: SlotPolicy,
    merge_key: MergeKey,
    available_at_ms: u64,
    enqueued_at_ms: u64,
    claim_fencing_token: u64,
}

fn queued_batch_row(row: PgRow) -> Result<QueuedBatchRow, StoreError> {
    let delivery_policy =
        DeliveryPolicy::from_wire_str(row.get::<String, _>("delivery_policy").as_str())
            .ok_or_else(|| {
                StoreError::Backend("invalid queued work delivery policy".to_string())
            })?;
    let slot_policy = SlotPolicy::from_wire_str(row.get::<String, _>("slot_policy").as_str())
        .ok_or_else(|| StoreError::Backend("invalid queued work slot policy".to_string()))?;
    let merge_json: String = row.get("merge_key_json");
    Ok(QueuedBatchRow {
        enqueue_seq: row.get::<i64, _>("enqueue_seq") as u64,
        batch_id: row.get("batch_id"),
        session_id: row.get("session_id"),
        source_key: row.get("source_key"),
        delivery_policy,
        slot_policy,
        merge_key: store_decode_json(&merge_json, "queued work merge key")?,
        available_at_ms: row.get::<i64, _>("available_at_ms") as u64,
        enqueued_at_ms: row.get::<i64, _>("enqueued_at_ms") as u64,
        claim_fencing_token: row.get::<i64, _>("claim_fencing_token") as u64,
    })
}

async fn load_queued_batch(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    batch_id: &str,
) -> Result<Option<QueuedWorkBatch>, StoreError> {
    let row = sqlx::query(
        "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                claim_fencing_token
         FROM lash_queued_work_batches
         WHERE batch_id = $1",
    )
    .bind(batch_id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let row = queued_batch_row(row)?;
    queued_work_batch_from_row(tx, row).await.map(Some)
}

async fn queued_work_batch_from_row(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    row: QueuedBatchRow,
) -> Result<QueuedWorkBatch, StoreError> {
    let item_rows = sqlx::query(
        "SELECT item_id, payload_json
         FROM lash_queued_work_items
         WHERE batch_id = $1
         ORDER BY item_index ASC",
    )
    .bind(&row.batch_id)
    .fetch_all(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    let mut items = Vec::new();
    for item in item_rows {
        let payload_json: String = item.get(1);
        items.push(QueuedWorkItem {
            item_id: item.get(0),
            payload: store_decode_json(&payload_json, "queued work payload")?,
        });
    }
    Ok(QueuedWorkBatch {
        batch_id: row.batch_id,
        session_id: row.session_id,
        enqueue_seq: row.enqueue_seq,
        source_key: row.source_key,
        delivery_policy: row.delivery_policy,
        slot_policy: row.slot_policy,
        merge_key: row.merge_key,
        available_at_ms: row.available_at_ms,
        enqueued_at_ms: row.enqueued_at_ms,
        items,
    })
}

async fn ensure_queued_work_completion_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    completed: &QueuedWorkCompletion,
) -> Result<(), StoreError> {
    for batch_id in &completed.batch_ids {
        let exists: Option<i64> = sqlx::query_scalar(
            "SELECT 1::BIGINT FROM lash_queued_work_batches
             WHERE session_id = $1
               AND batch_id = $2
               AND claim_id = $3
               AND claim_token = $4
             LIMIT 1",
        )
        .bind(&completed.session_id)
        .bind(batch_id)
        .bind(&completed.claim_id)
        .bind(&completed.lease_token)
        .fetch_optional(&mut **tx)
        .await
        .map_err(store_sqlx_error)?;
        if exists.is_none() {
            return Err(StoreError::QueuedWorkClaimExpired {
                session_id: completed.session_id.clone(),
                claim_id: completed.claim_id.clone(),
            });
        }
    }
    Ok(())
}
