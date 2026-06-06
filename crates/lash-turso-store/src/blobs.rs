//! Content-addressed blob, artifact, checkpoint, and usage-ledger storage on [`Store`].

use super::*;

impl Store {
    pub(crate) async fn insert_artifact_blob_conn(
        conn: &Connection,
        descriptor: BlobArtifactDescriptor,
        content: &[u8],
        profile: BuiltinBlobProfile,
    ) -> turso::Result<BlobRef> {
        let hash = format!("{:x}", Sha256::digest(content));
        let stored = encode_artifact_blob(&descriptor, profile, content);
        conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
            params![hash.clone(), stored],
        )
        .await?;
        Ok(BlobRef(hash))
    }

    pub(crate) async fn put_typed_artifact_blob_conn<T: serde::Serialize>(
        conn: &Connection,
        descriptor: BlobArtifactDescriptor,
        value: &T,
        profile: BuiltinBlobProfile,
    ) -> turso::Result<BlobRef> {
        let bytes = encode_msgpack(value);
        Self::insert_artifact_blob_conn(conn, descriptor, &bytes, profile).await
    }

    pub(crate) async fn put_checkpoint_conn(
        conn: &Connection,
        checkpoint: &HydratedSessionCheckpoint,
        profile: BuiltinBlobProfile,
    ) -> turso::Result<StoredSessionCheckpoint> {
        let tool_state_ref = if let Some(snapshot) = checkpoint.tool_state.as_ref() {
            Some(
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::tool_state_snapshot(),
                    snapshot,
                    profile,
                )
                .await?,
            )
        } else {
            checkpoint.tool_state_ref.clone()
        };
        let plugin_snapshot_ref = if let Some(snapshot) = checkpoint.plugin_snapshot.as_ref() {
            Some(
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::plugin_session_snapshot(),
                    snapshot,
                    profile,
                )
                .await?,
            )
        } else {
            checkpoint.plugin_snapshot_ref.clone()
        };
        let execution_state_ref = if let Some(snapshot) = checkpoint.execution_state.as_ref() {
            Some(
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::execution_state_snapshot(),
                    snapshot,
                    profile,
                )
                .await?,
            )
        } else {
            checkpoint.execution_state_ref.clone()
        };
        let manifest = SessionCheckpoint {
            turn_state: checkpoint.turn_state.clone(),
            tool_state_ref,
            plugin_snapshot_ref,
            plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
            execution_state_ref,
        };
        let checkpoint_ref = Self::put_typed_artifact_blob_conn(
            conn,
            BlobArtifactDescriptor::checkpoint_manifest(),
            &manifest,
            profile,
        )
        .await?;
        Ok(StoredSessionCheckpoint {
            checkpoint_ref,
            manifest,
        })
    }

    pub(crate) async fn get_blob_conn(conn: &Connection, blob_ref: &BlobRef) -> Option<Vec<u8>> {
        let row = optional_row(
            conn,
            "SELECT content FROM blobs WHERE hash = ?1",
            params![blob_ref.as_str()],
        )
        .await
        .ok()??;
        let bytes = row_blob(&row, 0).ok()?;
        decode_artifact_blob(&bytes).or(Some(bytes))
    }

    pub(crate) async fn get_typed_blob_conn<T: serde::de::DeserializeOwned>(
        conn: &Connection,
        blob_ref: &BlobRef,
    ) -> Option<T> {
        let bytes = Self::get_blob_conn(conn, blob_ref).await?;
        decode_msgpack(&bytes)
    }

    pub(crate) async fn get_checkpoint_conn(
        conn: &Connection,
        blob_ref: &BlobRef,
    ) -> Option<HydratedSessionCheckpoint> {
        let record: SessionCheckpoint = Self::get_typed_blob_conn(conn, blob_ref).await?;
        Some(HydratedSessionCheckpoint {
            turn_state: record.turn_state,
            tool_state_ref: record.tool_state_ref.clone(),
            tool_state: match record.tool_state_ref.as_ref() {
                Some(blob_ref) => Self::get_typed_blob_conn(conn, blob_ref).await,
                None => None,
            },
            plugin_snapshot_ref: record.plugin_snapshot_ref.clone(),
            plugin_snapshot: match record.plugin_snapshot_ref.as_ref() {
                Some(blob_ref) => Self::get_typed_blob_conn(conn, blob_ref).await,
                None => None,
            },
            plugin_snapshot_revision: record.plugin_snapshot_revision,
            execution_state_ref: record.execution_state_ref.clone(),
            execution_state: match record.execution_state_ref.as_ref() {
                Some(blob_ref) => Self::get_typed_blob_conn(conn, blob_ref).await,
                None => None,
            },
        })
    }

    pub(crate) async fn load_usage_deltas_conn(
        conn: &Connection,
    ) -> Vec<lash_core::TokenLedgerEntry> {
        let rows = match collect_rows(
            conn,
            "SELECT source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             FROM usage_deltas ORDER BY seq ASC",
            (),
        )
        .await
        {
            Ok(rows) => rows,
            Err(_) => return Vec::new(),
        };
        rows.into_iter()
            .filter_map(|row| {
                Some(lash_core::TokenLedgerEntry {
                    source: row_string(&row, 0).ok()?,
                    model: row_string(&row, 1).ok()?,
                    usage: lash_core::TokenUsage {
                        input_tokens: row_i64(&row, 2).ok()?,
                        output_tokens: row_i64(&row, 3).ok()?,
                        cached_input_tokens: row_i64(&row, 4).ok()?,
                        reasoning_tokens: row_i64(&row, 5).ok()?,
                    },
                })
            })
            .collect()
    }

    pub async fn put_blob(&self, content: &[u8]) -> BlobRef {
        let hash = format!("{:x}", Sha256::digest(content));
        let conn = self.conn.lock().await;
        if let Err(err) = conn
            .execute(
                "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
                params![hash.clone(), content.to_vec()],
            )
            .await
        {
            tracing::warn!(error = %err, hash, "failed to persist checkpoint blob");
        }
        BlobRef(hash)
    }

    pub async fn put_artifact_blob(
        &self,
        descriptor: BlobArtifactDescriptor,
        content: &[u8],
    ) -> BlobRef {
        let hash = format!("{:x}", Sha256::digest(content));
        let stored = encode_artifact_blob(&descriptor, self.options.blob_profile, content);
        let conn = self.conn.lock().await;
        if let Err(err) = conn
            .execute(
                "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
                params![hash.clone(), stored],
            )
            .await
        {
            tracing::warn!(error = %err, hash, "failed to persist artifact blob");
        }
        BlobRef(hash)
    }

    pub async fn get_blob(&self, blob_ref: &BlobRef) -> Option<Vec<u8>> {
        let conn = self.conn.lock().await;
        Self::get_blob_conn(&conn, blob_ref).await
    }

    pub async fn put_typed_blob<T: serde::Serialize>(&self, value: &T) -> BlobRef {
        let bytes = encode_msgpack(value);
        self.put_blob(&bytes).await
    }

    pub async fn put_typed_artifact_blob<T: serde::Serialize>(
        &self,
        descriptor: BlobArtifactDescriptor,
        value: &T,
    ) -> BlobRef {
        let bytes = encode_msgpack(value);
        self.put_artifact_blob(descriptor, &bytes).await
    }

    pub async fn get_typed_blob<T: serde::de::DeserializeOwned>(
        &self,
        blob_ref: &BlobRef,
    ) -> Option<T> {
        let conn = self.conn.lock().await;
        Self::get_typed_blob_conn(&conn, blob_ref).await
    }

    pub async fn put_checkpoint(
        &self,
        checkpoint: &HydratedSessionCheckpoint,
    ) -> StoredSessionCheckpoint {
        let conn = self.conn.lock().await;
        Self::put_checkpoint_conn(&conn, checkpoint, self.options.blob_profile)
            .await
            .expect("checkpoint blob should persist")
    }

    pub async fn get_checkpoint(&self, blob_ref: &BlobRef) -> Option<HydratedSessionCheckpoint> {
        let conn = self.conn.lock().await;
        Self::get_checkpoint_conn(&conn, blob_ref).await
    }

    pub async fn append_usage_deltas(&self, entries: &[lash_core::TokenLedgerEntry]) {
        if entries.is_empty() {
            return;
        }
        let conn = self.conn.lock().await;
        if let Err(err) = conn.execute("BEGIN IMMEDIATE", ()).await {
            tracing::warn!(error = %err, "failed to begin usage delta transaction");
            return;
        }
        let result = async {
            for entry in entries {
                conn.execute(
                    "INSERT INTO usage_deltas (
                        source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        entry.source.clone(),
                        entry.model.clone(),
                        entry.usage.input_tokens,
                        entry.usage.output_tokens,
                        entry.usage.cached_input_tokens,
                        entry.usage.reasoning_tokens,
                    ],
                )
                .await?;
            }
            Ok::<(), turso::Error>(())
        }
        .await;
        match result {
            Ok(()) => {
                if let Err(err) = conn.execute("COMMIT", ()).await {
                    tracing::warn!(error = %err, "failed to commit usage deltas");
                }
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                tracing::warn!(error = %err, "failed to persist usage deltas");
            }
        }
    }

    pub async fn load_usage_deltas(&self) -> Vec<lash_core::TokenLedgerEntry> {
        let conn = self.conn.lock().await;
        Self::load_usage_deltas_conn(&conn).await
    }
}
