fn process_status_label(record: &ProcessRecord) -> &'static str {
    record.status.label()
}

async fn load_process_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    process_id: &str,
) -> Result<Option<ProcessRecord>, PluginError> {
    let json: Option<String> = sqlx::query_scalar(
        "SELECT record_json
             FROM lash_processes
             WHERE process_id = $1
             FOR UPDATE",
    )
    .bind(process_id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(plugin_sqlx_error)?;
    json.map(|json| serde_json::from_str(&json).map_err(process_decode_error))
        .transpose()
}

async fn load_process(
    pool: &PgPool,
    process_id: &str,
) -> Result<Option<ProcessRecord>, PluginError> {
    let json: Option<String> =
        sqlx::query_scalar("SELECT record_json FROM lash_processes WHERE process_id = $1")
            .bind(process_id)
            .fetch_optional(pool)
            .await
            .map_err(plugin_sqlx_error)?;
    json.map(|json| serde_json::from_str(&json).map_err(process_decode_error))
        .transpose()
}

async fn save_process_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    record: &ProcessRecord,
) -> Result<(), PluginError> {
    sqlx::query(
        "UPDATE lash_processes
         SET updated_at_ms = $2, status = $3, record_json = $4
         WHERE process_id = $1",
    )
    .bind(&record.id)
    .bind(record.updated_at_ms as i64)
    .bind(process_status_label(record))
    .bind(serde_json::to_string(record).map_err(process_decode_error)?)
    .execute(&mut **tx)
    .await
    .map_err(plugin_sqlx_error)?;
    Ok(())
}

async fn load_event_by_key_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    process_id: &str,
    replay_key: &str,
) -> Result<Option<(String, ProcessEvent)>, PluginError> {
    let row = sqlx::query(
        "SELECT payload_hash, event_json
         FROM lash_process_events
         WHERE process_id = $1 AND idempotency_key = $2",
    )
    .bind(process_id)
    .bind(replay_key)
    .fetch_optional(&mut **tx)
    .await
    .map_err(plugin_sqlx_error)?;
    row.map(|row| {
        let hash: String = row.get(0);
        let json: String = row.get(1);
        serde_json::from_str(&json)
            .map(|event| (hash, event))
            .map_err(process_decode_error)
    })
    .transpose()
}

async fn load_process_lease_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    process_id: &str,
) -> Result<Option<ProcessLease>, PluginError> {
    let row = sqlx::query(
        "SELECT lease_owner_id, lease_token, lease_fencing_token,
                lease_claimed_at_ms, lease_expires_at_ms
         FROM lash_process_leases
         WHERE process_id = $1",
    )
    .bind(process_id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(plugin_sqlx_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let owner_id: Option<String> = row.get(0);
    let lease_token: Option<String> = row.get(1);
    let (Some(owner_id), Some(lease_token)) = (owner_id, lease_token) else {
        return Ok(None);
    };
    Ok(Some(ProcessLease {
        schema_version: PROCESS_LEASE_SCHEMA_VERSION,
        process_id: process_id.to_string(),
        owner_id,
        lease_token,
        fencing_token: row.get::<i64, _>(2) as u64,
        claimed_at_epoch_ms: row.get::<i64, _>(3) as u64,
        expires_at_epoch_ms: row.get::<i64, _>(4) as u64,
    }))
}

fn process_lease_conflict(process_id: &str, current: &ProcessLease) -> PluginError {
    PluginError::Session(format!(
        "process `{process_id}` is already leased by `{}` until {}",
        current.owner_id, current.expires_at_epoch_ms
    ))
}

fn process_lease_expired(process_id: &str) -> PluginError {
    PluginError::Session(format!(
        "process lease for `{process_id}` is missing or expired"
    ))
}

fn guard_lease(current: Option<&ProcessLease>, lease_token: &str, now: u64) -> bool {
    current
        .map(|current| current.lease_token == lease_token && current.expires_at_epoch_ms > now)
        .unwrap_or(false)
}

async fn list_grants_for_scope(
    pool: &PgPool,
    session_scope: &SessionScope,
    live_only: bool,
) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
    let status_clause = if live_only {
        "AND p.status = 'running'"
    } else {
        ""
    };
    let sql = format!(
        "SELECT g.process_id, g.descriptor_json, p.record_json
         FROM lash_process_handle_grants g
         JOIN lash_processes p ON p.process_id = g.process_id
         WHERE g.scope_id = $1 {status_clause}
         ORDER BY g.process_id ASC"
    );
    let rows = sqlx::query(&sql)
        .bind(session_scope.id().as_str())
        .fetch_all(pool)
        .await
        .map_err(plugin_sqlx_error)?;
    let mut entries = Vec::new();
    for row in rows {
        let process_id: String = row.get(0);
        let descriptor_json: String = row.get(1);
        let record_json: String = row.get(2);
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
