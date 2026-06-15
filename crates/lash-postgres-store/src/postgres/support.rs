fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn current_timestamp_string() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", now.as_secs())
}

fn store_sqlx_error(err: sqlx::Error) -> StoreError {
    StoreError::Backend(err.to_string())
}

/// Postgres SQLSTATEs that signal transient write contention rather than a hard
/// failure: serialization failure, deadlock, and lock-acquisition timeout. On the
/// session head these all mean "a concurrent committer got there first" — i.e. a
/// revision conflict the caller should reload-and-retry, not a backend error.
fn is_contention_error(err: &sqlx::Error) -> bool {
    matches!(
        err.as_database_error().and_then(|db| db.code()).as_deref(),
        Some("40001" | "40P01" | "55P03")
    )
}

fn plugin_sqlx_error(err: sqlx::Error) -> PluginError {
    PluginError::Session(err.to_string())
}

fn process_decode_error(err: serde_json::Error) -> PluginError {
    PluginError::Session(format!("failed to decode process registry row: {err}"))
}

fn store_decode_json<T: serde::de::DeserializeOwned>(
    json: &str,
    what: &str,
) -> Result<T, StoreError> {
    serde_json::from_str(json)
        .map_err(|err| StoreError::Backend(format!("failed to decode {what}: {err}")))
}

fn encode_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("persisted state should serialize")
}

fn encode_msgpack<T: serde::Serialize>(value: &T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1024);
    rmp_serde::encode::write_named(&mut buf, value).expect("value should serialize");
    buf
}

fn decode_msgpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    rmp_serde::from_slice(bytes).ok()
}

fn block_on_detached<T: Send + 'static>(
    future: impl std::future::Future<Output = T> + Send + 'static,
) -> T {
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("postgres manifest runtime")
            .block_on(future)
    })
    .join()
    .expect("postgres manifest thread")
}

fn merge_token_ledger_entries(entries: Vec<TokenLedgerEntry>) -> Vec<TokenLedgerEntry> {
    let mut merged = Vec::<TokenLedgerEntry>::new();
    for entry in entries {
        if entry.usage.total() == 0 {
            continue;
        }
        if let Some(existing) = merged
            .iter_mut()
            .find(|existing| existing.source == entry.source && existing.model == entry.model)
        {
            existing.usage.input_tokens += entry.usage.input_tokens;
            existing.usage.output_tokens += entry.usage.output_tokens;
            existing.usage.cached_input_tokens += entry.usage.cached_input_tokens;
            existing.usage.reasoning_tokens += entry.usage.reasoning_tokens;
        } else {
            merged.push(entry);
        }
    }
    merged
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SessionCheckpointEnvelope {
    manifest: SessionCheckpoint,
    tool_state: Option<lash_core::ToolState>,
    plugin_snapshot: Option<lash_core::PluginSessionSnapshot>,
    execution_state: Option<Vec<u8>>,
}

async fn put_blob_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    content: &[u8],
) -> Result<BlobRef, StoreError> {
    let hash = format!("{:x}", Sha256::digest(content));
    sqlx::query(
        "INSERT INTO lash_blobs (hash, content)
         VALUES ($1, $2)
         ON CONFLICT (hash) DO NOTHING",
    )
    .bind(&hash)
    .bind(content)
    .execute(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    Ok(BlobRef(hash))
}

async fn get_blob(pool: &PgPool, blob_ref: &BlobRef) -> Option<Vec<u8>> {
    sqlx::query_scalar::<_, Vec<u8>>("SELECT content FROM lash_blobs WHERE hash = $1")
        .bind(blob_ref.as_str())
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
}

async fn put_checkpoint_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    checkpoint: &HydratedSessionCheckpoint,
) -> Result<(BlobRef, SessionCheckpoint), StoreError> {
    let manifest = SessionCheckpoint {
        turn_state: checkpoint.turn_state.clone(),
        tool_state_ref: checkpoint.tool_state_ref.clone(),
        plugin_snapshot_ref: checkpoint.plugin_snapshot_ref.clone(),
        plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
        execution_state_ref: checkpoint.execution_state_ref.clone(),
    };
    let envelope = SessionCheckpointEnvelope {
        manifest: manifest.clone(),
        tool_state: checkpoint.tool_state.clone(),
        plugin_snapshot: checkpoint.plugin_snapshot.clone(),
        execution_state: checkpoint.execution_state.clone(),
    };
    let bytes = encode_msgpack(&envelope);
    let checkpoint_ref = put_blob_tx(tx, &bytes).await?;
    Ok((checkpoint_ref, manifest))
}

async fn get_checkpoint(pool: &PgPool, blob_ref: &BlobRef) -> Option<HydratedSessionCheckpoint> {
    let bytes = get_blob(pool, blob_ref).await?;
    let envelope: SessionCheckpointEnvelope = decode_msgpack(&bytes)?;
    Some(HydratedSessionCheckpoint {
        turn_state: envelope.manifest.turn_state,
        tool_state_ref: envelope.manifest.tool_state_ref,
        tool_state: envelope.tool_state,
        plugin_snapshot_ref: envelope.manifest.plugin_snapshot_ref,
        plugin_snapshot: envelope.plugin_snapshot,
        plugin_snapshot_revision: envelope.manifest.plugin_snapshot_revision,
        execution_state_ref: envelope.manifest.execution_state_ref,
        execution_state: envelope.execution_state,
    })
}

async fn load_session_head_meta_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    session_id: &str,
    for_update: bool,
) -> Result<Option<SessionHeadMeta>, StoreError> {
    let sql = if for_update {
        "SELECT head_json, head_revision FROM lash_sessions WHERE session_id = $1 FOR UPDATE"
    } else {
        "SELECT head_json, head_revision FROM lash_sessions WHERE session_id = $1"
    };
    let row = sqlx::query(sql)
        .bind(session_id)
        .fetch_optional(&mut **tx)
        .await
        .map_err(store_sqlx_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let head_json: String = row.get(0);
    let head_revision: i64 = row.get(1);
    let mut meta: SessionHeadMeta = store_decode_json(&head_json, "session head")?;
    meta.head_revision = head_revision as u64;
    Ok(Some(meta))
}

async fn load_usage_deltas(pool: &PgPool, session_id: &str) -> Vec<TokenLedgerEntry> {
    let rows = sqlx::query(
        "SELECT entry_json FROM lash_usage_deltas WHERE session_id = $1 ORDER BY seq ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    rows.into_iter()
        .filter_map(|row| {
            let json: String = row.get(0);
            serde_json::from_str(&json).ok()
        })
        .collect()
}

async fn load_graph(
    pool: &PgPool,
    session_id: &str,
    leaf_node_id: Option<String>,
    active_path: bool,
) -> Result<lash_core::SessionGraph, StoreError> {
    let rows = sqlx::query(
        "SELECT node_json FROM lash_graph_nodes
         WHERE session_id = $1 AND tombstoned = FALSE
         ORDER BY seq ASC",
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .map_err(store_sqlx_error)?;
    let mut nodes = Vec::<SessionNodeRecord>::new();
    for row in rows {
        let json: String = row.get(0);
        nodes.push(store_decode_json(&json, "session graph node")?);
    }
    if active_path && let Some(leaf) = leaf_node_id.clone() {
        let wanted = active_path_node_ids(&nodes, &leaf);
        nodes.retain(|node| wanted.contains(&node.node_id));
    }
    Ok(lash_core::SessionGraph::from_nodes(nodes, leaf_node_id))
}

fn active_path_node_ids(nodes: &[SessionNodeRecord], leaf_node_id: &str) -> HashSet<String> {
    let mut parent_by_id = std::collections::BTreeMap::new();
    for node in nodes {
        parent_by_id.insert(node.node_id.clone(), node.parent_node_id.clone());
    }
    let mut wanted = HashSet::new();
    let mut cursor = Some(leaf_node_id.to_string());
    while let Some(node_id) = cursor {
        if !wanted.insert(node_id.clone()) {
            break;
        }
        cursor = parent_by_id.get(&node_id).cloned().flatten();
    }
    wanted
}

async fn commit_attachment_refs_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    session_id: &str,
    attachment_ids: &[AttachmentId],
) -> Result<(), StoreError> {
    if attachment_ids.is_empty() {
        return Ok(());
    }
    let now = current_epoch_ms() as i64;
    for id in attachment_ids {
        sqlx::query(
            "UPDATE lash_attachment_manifest
             SET committed_at_ms = COALESCE(committed_at_ms, $1)
             WHERE attachment_id = $2 AND session_id = $3",
        )
        .bind(now)
        .bind(id.as_str())
        .bind(session_id)
        .execute(&mut **tx)
        .await
        .map_err(store_sqlx_error)?;
    }
    Ok(())
}
