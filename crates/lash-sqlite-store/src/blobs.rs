//! Content-addressed blob, artifact, checkpoint, and usage-ledger storage on [`Store`].

use super::*;

impl Store {
    pub(crate) fn insert_artifact_blob_conn(
        conn: &Connection,
        descriptor: BlobArtifactDescriptor,
        content: &[u8],
        profile: BuiltinBlobProfile,
    ) -> rusqlite::Result<BlobRef> {
        let hash = format!("{:x}", Sha256::digest(content));
        let stored = encode_artifact_blob(&descriptor, profile, content);
        conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
            params![hash, stored],
        )?;
        Ok(BlobRef(hash))
    }

    pub(crate) fn put_typed_artifact_blob_conn<T: serde::Serialize>(
        conn: &Connection,
        descriptor: BlobArtifactDescriptor,
        value: &T,
        profile: BuiltinBlobProfile,
    ) -> rusqlite::Result<BlobRef> {
        let bytes = encode_msgpack(value);
        Self::insert_artifact_blob_conn(conn, descriptor, &bytes, profile)
    }

    pub(crate) fn put_checkpoint_conn(
        conn: &Connection,
        checkpoint: &HydratedSessionCheckpoint,
        profile: BuiltinBlobProfile,
    ) -> rusqlite::Result<StoredSessionCheckpoint> {
        let tool_state_ref = checkpoint
            .tool_state
            .as_ref()
            .map(|snapshot| {
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::tool_state_snapshot(),
                    snapshot,
                    profile,
                )
            })
            .transpose()?
            .or_else(|| checkpoint.tool_state_ref.clone());
        let plugin_snapshot_ref = checkpoint
            .plugin_snapshot
            .as_ref()
            .map(|snapshot| {
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::plugin_session_snapshot(),
                    snapshot,
                    profile,
                )
            })
            .transpose()?
            .or_else(|| checkpoint.plugin_snapshot_ref.clone());
        let execution_state_ref = checkpoint
            .execution_state
            .as_ref()
            .map(|snapshot| {
                Self::put_typed_artifact_blob_conn(
                    conn,
                    BlobArtifactDescriptor::execution_state_snapshot(),
                    snapshot,
                    profile,
                )
            })
            .transpose()?
            .or_else(|| checkpoint.execution_state_ref.clone());
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
        )?;
        Ok(StoredSessionCheckpoint {
            checkpoint_ref,
            manifest,
        })
    }

    pub(crate) fn get_blob_conn(conn: &Connection, blob_ref: &BlobRef) -> Option<Vec<u8>> {
        let bytes: Vec<u8> = conn
            .query_row(
                "SELECT content FROM blobs WHERE hash = ?1",
                params![blob_ref.as_str()],
                |row| row.get(0),
            )
            .ok()?;
        decode_artifact_blob(&bytes).or(Some(bytes))
    }

    pub(crate) fn get_typed_blob_conn<T: serde::de::DeserializeOwned>(
        conn: &Connection,
        blob_ref: &BlobRef,
    ) -> Option<T> {
        let bytes = Self::get_blob_conn(conn, blob_ref)?;
        decode_msgpack(&bytes)
    }

    pub(crate) fn get_checkpoint_conn(
        conn: &Connection,
        blob_ref: &BlobRef,
    ) -> Option<HydratedSessionCheckpoint> {
        let record: SessionCheckpoint = Self::get_typed_blob_conn(conn, blob_ref)?;
        Some(HydratedSessionCheckpoint {
            turn_state: record.turn_state,
            tool_state_ref: record.tool_state_ref.clone(),
            tool_state: record
                .tool_state_ref
                .as_ref()
                .and_then(|blob_ref| Self::get_typed_blob_conn(conn, blob_ref)),
            plugin_snapshot_ref: record.plugin_snapshot_ref.clone(),
            plugin_snapshot: record
                .plugin_snapshot_ref
                .as_ref()
                .and_then(|blob_ref| Self::get_typed_blob_conn(conn, blob_ref)),
            plugin_snapshot_revision: record.plugin_snapshot_revision,
            execution_state_ref: record.execution_state_ref.clone(),
            execution_state: record
                .execution_state_ref
                .as_ref()
                .and_then(|blob_ref| Self::get_typed_blob_conn(conn, blob_ref)),
        })
    }

    pub(crate) fn load_usage_deltas_conn(conn: &Connection) -> Vec<lash_core::TokenLedgerEntry> {
        let mut stmt = match conn.prepare(
            "SELECT source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
             FROM usage_deltas ORDER BY seq ASC",
        ) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        let rows = match stmt.query_map([], |row| {
            Ok(lash_core::TokenLedgerEntry {
                source: row.get(0)?,
                model: row.get(1)?,
                usage: lash_core::TokenUsage {
                    input_tokens: row.get(2)?,
                    output_tokens: row.get(3)?,
                    cached_input_tokens: row.get(4)?,
                    reasoning_tokens: row.get(5)?,
                },
            })
        }) {
            Ok(rows) => rows,
            Err(_) => return Vec::new(),
        };
        rows.filter_map(Result::ok).collect()
    }

    pub fn put_blob(&self, content: &[u8]) -> BlobRef {
        let hash = format!("{:x}", Sha256::digest(content));
        let conn = lock_conn(&self.conn);
        if let Err(err) = conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
            params![hash, content],
        ) {
            tracing::warn!(error = %err, hash, "failed to persist checkpoint blob");
        }
        BlobRef(hash)
    }

    pub fn put_artifact_blob(&self, descriptor: BlobArtifactDescriptor, content: &[u8]) -> BlobRef {
        let hash = format!("{:x}", Sha256::digest(content));
        let stored = encode_artifact_blob(&descriptor, self.options.blob_profile, content);
        let conn = lock_conn(&self.conn);
        if let Err(err) = conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, content) VALUES (?1, ?2)",
            params![hash, stored],
        ) {
            tracing::warn!(error = %err, hash, "failed to persist artifact blob");
        }
        BlobRef(hash)
    }

    pub fn get_blob(&self, blob_ref: &BlobRef) -> Option<Vec<u8>> {
        let conn = lock_conn(&self.conn);
        Self::get_blob_conn(&conn, blob_ref)
    }

    pub fn put_typed_blob<T: serde::Serialize>(&self, value: &T) -> BlobRef {
        let bytes = encode_msgpack(value);
        self.put_blob(&bytes)
    }

    pub fn put_typed_artifact_blob<T: serde::Serialize>(
        &self,
        descriptor: BlobArtifactDescriptor,
        value: &T,
    ) -> BlobRef {
        let bytes = encode_msgpack(value);
        self.put_artifact_blob(descriptor, &bytes)
    }

    pub fn get_typed_blob<T: serde::de::DeserializeOwned>(&self, blob_ref: &BlobRef) -> Option<T> {
        let conn = lock_conn(&self.conn);
        Self::get_typed_blob_conn(&conn, blob_ref)
    }

    pub fn put_checkpoint(
        &self,
        checkpoint: &HydratedSessionCheckpoint,
    ) -> StoredSessionCheckpoint {
        let tool_state_ref = checkpoint
            .tool_state
            .as_ref()
            .map(|snapshot| {
                self.put_typed_artifact_blob(
                    BlobArtifactDescriptor::tool_state_snapshot(),
                    snapshot,
                )
            })
            .or_else(|| checkpoint.tool_state_ref.clone());
        let plugin_snapshot_ref = checkpoint
            .plugin_snapshot
            .as_ref()
            .map(|snapshot| {
                self.put_typed_artifact_blob(
                    BlobArtifactDescriptor::plugin_session_snapshot(),
                    snapshot,
                )
            })
            .or_else(|| checkpoint.plugin_snapshot_ref.clone());
        let execution_state_ref = checkpoint
            .execution_state
            .as_ref()
            .map(|snapshot| {
                self.put_typed_artifact_blob(
                    BlobArtifactDescriptor::execution_state_snapshot(),
                    snapshot,
                )
            })
            .or_else(|| checkpoint.execution_state_ref.clone());
        let manifest = SessionCheckpoint {
            turn_state: checkpoint.turn_state.clone(),
            tool_state_ref,
            plugin_snapshot_ref,
            plugin_snapshot_revision: checkpoint.plugin_snapshot_revision,
            execution_state_ref,
        };
        let checkpoint_ref =
            self.put_typed_artifact_blob(BlobArtifactDescriptor::checkpoint_manifest(), &manifest);
        StoredSessionCheckpoint {
            checkpoint_ref,
            manifest,
        }
    }

    pub fn get_checkpoint(&self, blob_ref: &BlobRef) -> Option<HydratedSessionCheckpoint> {
        let conn = lock_conn(&self.conn);
        Self::get_checkpoint_conn(&conn, blob_ref)
    }

    pub fn append_usage_deltas(&self, entries: &[lash_core::TokenLedgerEntry]) {
        if entries.is_empty() {
            return;
        }
        let result = self.with_write_tx(|tx| {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO usage_deltas (
                        source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .map_err(sqlite_error)?;
            for entry in entries {
                stmt.execute(params![
                    entry.source,
                    entry.model,
                    entry.usage.input_tokens,
                    entry.usage.output_tokens,
                    entry.usage.cached_input_tokens,
                    entry.usage.reasoning_tokens,
                ])
                .map_err(sqlite_error)?;
            }
            Ok(())
        });
        if let Err(err) = result {
            tracing::warn!(error = %err, "failed to persist usage deltas");
        }
    }

    pub fn load_usage_deltas(&self) -> Vec<lash_core::TokenLedgerEntry> {
        let conn = lock_conn(&self.conn);
        Self::load_usage_deltas_conn(&conn)
    }
}
