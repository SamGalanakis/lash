use super::*;

pub(crate) fn decode_delivery_policy(value: String) -> Result<DeliveryPolicy, StoreError> {
    DeliveryPolicy::from_wire_str(&value).ok_or_else(|| {
        StoreError::Backend(format!("unknown queued-work delivery policy `{value}`"))
    })
}

pub(crate) fn decode_slot_policy(value: String) -> Result<SlotPolicy, StoreError> {
    SlotPolicy::from_wire_str(&value)
        .ok_or_else(|| StoreError::Backend(format!("unknown queued-work slot policy `{value}`")))
}

pub(crate) fn decode_merge_key(value: String) -> Result<MergeKey, StoreError> {
    serde_json::from_str(&value).map_err(|err| {
        StoreError::Backend(format!("failed to decode queued-work merge key: {err}"))
    })
}

pub(crate) fn decode_queued_payload(value: String) -> Result<QueuedWorkPayload, StoreError> {
    serde_json::from_str(&value)
        .map_err(|err| StoreError::Backend(format!("failed to decode queued-work payload: {err}")))
}

pub(crate) async fn queued_work_batch_from_conn(
    conn: &Connection,
    row: QueuedBatchRow,
) -> Result<QueuedWorkBatch, StoreError> {
    let item_rows = collect_rows(
        conn,
        "SELECT item_id, payload_json
         FROM queued_work_items
         WHERE batch_id = ?1
         ORDER BY item_index ASC",
        params![row.batch_id.as_str()],
    )
    .await
    .map_err(turso_error)?;
    let mut items = Vec::new();
    for item_row in item_rows {
        let item_id = row_string(&item_row, 0).map_err(turso_error)?;
        let payload_json = row_string(&item_row, 1).map_err(turso_error)?;
        items.push(QueuedWorkItem {
            item_id,
            payload: decode_queued_payload(payload_json)?,
        });
    }
    Ok(QueuedWorkBatch {
        batch_id: row.batch_id,
        session_id: row.session_id,
        enqueue_seq: row.enqueue_seq,
        source_key: row.source_key,
        delivery_policy: decode_delivery_policy(row.delivery_policy)?,
        slot_policy: decode_slot_policy(row.slot_policy)?,
        merge_key: decode_merge_key(row.merge_key_json)?,
        available_at_ms: row.available_at_ms,
        enqueued_at_ms: row.enqueued_at_ms,
        items,
    })
}

pub(crate) struct QueuedBatchRow {
    pub(crate) enqueue_seq: u64,
    pub(crate) batch_id: String,
    pub(crate) session_id: String,
    pub(crate) source_key: Option<String>,
    pub(crate) delivery_policy: String,
    pub(crate) slot_policy: String,
    pub(crate) merge_key_json: String,
    pub(crate) available_at_ms: u64,
    pub(crate) enqueued_at_ms: u64,
    pub(crate) claim_fencing_token: u64,
}

pub(crate) fn queued_batch_row_from_sql(row: &Row) -> turso::Result<QueuedBatchRow> {
    Ok(QueuedBatchRow {
        enqueue_seq: row_i64(row, 0)? as u64,
        batch_id: row_string(row, 1)?,
        session_id: row_string(row, 2)?,
        source_key: row_optional_string(row, 3)?,
        delivery_policy: row_string(row, 4)?,
        slot_policy: row_string(row, 5)?,
        merge_key_json: row_string(row, 6)?,
        available_at_ms: row_i64(row, 7)? as u64,
        enqueued_at_ms: row_i64(row, 8)? as u64,
        claim_fencing_token: row_i64(row, 9)? as u64,
    })
}

pub(crate) async fn load_queued_batch_by_id_conn(
    conn: &Connection,
    batch_id: &str,
) -> Result<Option<QueuedWorkBatch>, StoreError> {
    let row = optional_row(
        conn,
        "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                claim_fencing_token
         FROM queued_work_batches
         WHERE batch_id = ?1",
        params![batch_id],
    )
    .await
    .map_err(turso_error)?
    .map(|row| queued_batch_row_from_sql(&row).map_err(turso_error))
    .transpose()?;
    match row {
        Some(row) => queued_work_batch_from_conn(conn, row).await.map(Some),
        None => Ok(None),
    }
}

pub(crate) async fn ensure_queued_work_completion_conn(
    conn: &Connection,
    completed: &QueuedWorkCompletion,
) -> Result<(), StoreError> {
    let row = required_row(
        conn,
        "SELECT COUNT(*)
         FROM queued_work_batches
         WHERE session_id = ?1
           AND claim_id = ?2
           AND claim_token = ?3",
        params![
            completed.session_id.as_str(),
            completed.claim_id.as_str(),
            completed.lease_token.as_str()
        ],
    )
    .await
    .map_err(turso_error)?;
    let count = row_i64(&row, 0).map_err(turso_error)? as usize;
    if count != completed.batch_ids.len() {
        return Err(StoreError::QueuedWorkClaimExpired {
            session_id: completed.session_id.clone(),
            claim_id: completed.claim_id.clone(),
        });
    }
    Ok(())
}
