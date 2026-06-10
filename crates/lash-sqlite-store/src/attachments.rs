//! The lashlang module-artifact store and the attachment write-ahead manifest.
//!
//! Both traits in this module have synchronous-looking call sites in their
//! consumers but bridge to the async [`SqliteConnection`] underneath:
//!
//! * [`lashlang::LashlangArtifactStore`] is itself an `#[async_trait]`, so its
//!   methods `.await` the connection wrapper directly (matching the the prior store
//!   store's async surface byte-for-byte).
//! * [`AttachmentManifest`] is a *synchronous* trait. Its bodies therefore wrap
//!   the async store work in [`block_on_store`], exactly as the prior store did.
//!
//! Every DB body is a synchronous rusqlite closure handed to `conn.call`
//! (reads) or `conn.write` (read-then-write); only the wrapper call is awaited.

use super::*;

#[async_trait::async_trait]
impl lashlang::LashlangArtifactStore for Store {
    fn durability_tier(&self) -> lashlang::DurabilityTier {
        lashlang::DurabilityTier::Durable
    }

    async fn put_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), lashlang::ArtifactStoreError> {
        let bytes = artifact
            .to_store_bytes()
            .map_err(|err| lashlang::ArtifactStoreError::Encode(err.to_string()))?;
        let blob_profile = self.options.blob_profile;
        let artifact_ref = artifact.module_ref.as_str().to_string();
        self.conn
            .write(move |tx| {
                let blob_ref = Self::insert_artifact_blob_conn(
                    tx,
                    BlobArtifactDescriptor::lashlang_module(),
                    &bytes,
                    blob_profile,
                )?;
                tx.execute(
                    "INSERT OR REPLACE INTO artifact_refs (artifact_ref, blob_ref) VALUES (?1, ?2)",
                    params![artifact_ref, blob_ref.as_str()],
                )?;
                Ok(())
            })
            .await
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        self.artifact_cache
            .lock()
            .map_err(|_| {
                lashlang::ArtifactStoreError::Backend("artifact cache lock poisoned".to_string())
            })?
            .insert(artifact.module_ref.clone(), Arc::new(artifact.clone()));
        Ok(())
    }

    async fn get_module_artifact(
        &self,
        module_ref: &lashlang::ModuleRef,
    ) -> Result<Option<Arc<lashlang::ModuleArtifact>>, lashlang::ArtifactStoreError> {
        if let Some(artifact) = self
            .artifact_cache
            .lock()
            .map_err(|_| {
                lashlang::ArtifactStoreError::Backend("artifact cache lock poisoned".to_string())
            })?
            .get(module_ref)
            .cloned()
        {
            return Ok(Some(artifact));
        }

        let artifact_ref = module_ref.as_str().to_string();
        // `Option<Option<Vec<u8>>>`: the outer `None` means no `artifact_refs`
        // row exists (return `Ok(None)`); the inner `None` means the row points
        // at a missing blob (a hard error, matching the prior store).
        let resolved = self
            .conn
            .call(move |conn| {
                let blob_ref: Option<String> = conn
                    .query_row(
                        "SELECT blob_ref FROM artifact_refs WHERE artifact_ref = ?1",
                        params![artifact_ref],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?;
                let Some(blob_ref) = blob_ref else {
                    return Ok(None);
                };
                Ok(Some(Self::get_blob_conn(conn, &BlobRef(blob_ref))))
            })
            .await
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        let Some(blob) = resolved else {
            return Ok(None);
        };
        let bytes = blob.ok_or_else(|| {
            lashlang::ArtifactStoreError::Backend(format!(
                "lashlang module artifact `{}` points at a missing blob",
                module_ref
            ))
        })?;
        let artifact = Arc::new(
            lashlang::ModuleArtifact::from_store_bytes(&bytes)
                .map_err(lashlang::ArtifactStoreError::from)?,
        );
        self.artifact_cache
            .lock()
            .map_err(|_| {
                lashlang::ArtifactStoreError::Backend("artifact cache lock poisoned".to_string())
            })?
            .insert(module_ref.clone(), artifact.clone());
        Ok(Some(artifact))
    }

    async fn put_artifact_bytes(
        &self,
        artifact_ref: &str,
        descriptor: &str,
        bytes: &[u8],
    ) -> Result<(), lashlang::ArtifactStoreError> {
        let blob_profile = self.options.blob_profile;
        let artifact_ref = artifact_ref.to_string();
        let descriptor = match descriptor {
            "process_execution_env" => BlobArtifactDescriptor::process_execution_env(),
            _ => BlobArtifactDescriptor::new(PersistedArtifactKind::GenericBlob, Vec::new()),
        };
        let bytes = bytes.to_vec();
        self.conn
            .write(move |tx| {
                let blob_ref =
                    Self::insert_artifact_blob_conn(tx, descriptor, &bytes, blob_profile)?;
                tx.execute(
                    "INSERT OR REPLACE INTO artifact_refs (artifact_ref, blob_ref) VALUES (?1, ?2)",
                    params![artifact_ref, blob_ref.as_str()],
                )?;
                Ok(())
            })
            .await
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))
    }

    async fn get_artifact_bytes(
        &self,
        artifact_ref: &str,
    ) -> Result<Option<Vec<u8>>, lashlang::ArtifactStoreError> {
        let artifact_ref = artifact_ref.to_string();
        let diagnostic_ref = artifact_ref.clone();
        let resolved = self
            .conn
            .call(move |conn| {
                let blob_ref: Option<String> = conn
                    .query_row(
                        "SELECT blob_ref FROM artifact_refs WHERE artifact_ref = ?1",
                        params![artifact_ref],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?;
                let Some(blob_ref) = blob_ref else {
                    return Ok(None);
                };
                Ok(Some(Self::get_blob_conn(conn, &BlobRef(blob_ref))))
            })
            .await
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        let Some(blob) = resolved else {
            return Ok(None);
        };
        let bytes = blob.ok_or_else(|| {
            lashlang::ArtifactStoreError::Backend(format!(
                "artifact `{diagnostic_ref}` points at a missing blob"
            ))
        })?;
        Ok(Some(bytes))
    }
}

impl AttachmentManifest for Store {
    fn record_intent(&self, intent: AttachmentIntent) -> Result<(), StoreError> {
        block_on_store(async {
            let attachment_id = intent.attachment_id.as_str().to_string();
            let session_id = intent.session_id.as_str().to_string();
            let canonical_uri = intent.canonical_uri.as_str().to_string();
            let intent_at_ms = intent.intent_at_epoch_ms as i64;
            self.conn
                .call(move |conn| {
                    conn.execute(
                        "INSERT INTO attachment_manifest
                            (attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms)
                         VALUES (?1, ?2, ?3, ?4, NULL)
                         ON CONFLICT(attachment_id) DO NOTHING",
                        params![attachment_id, session_id, canonical_uri, intent_at_ms],
                    )
                })
                .await
                .map_err(sqlite_error)?;
            Ok(())
        })
    }

    fn commit_refs(
        &self,
        session_id: &str,
        attachment_ids: &[AttachmentId],
    ) -> Result<(), StoreError> {
        if attachment_ids.is_empty() {
            return Ok(());
        }
        block_on_store(async {
            let session_id = session_id.to_string();
            let attachment_ids: Vec<String> = attachment_ids
                .iter()
                .map(|id| id.as_str().to_string())
                .collect();
            self.conn
                .write(move |tx| {
                    let now = current_epoch_ms() as i64;
                    let mut stmt = tx.prepare(
                        "UPDATE attachment_manifest
                         SET committed_at_ms = COALESCE(committed_at_ms, ?1)
                         WHERE attachment_id = ?2 AND session_id = ?3",
                    )?;
                    for id in &attachment_ids {
                        stmt.execute(params![now, id, session_id])?;
                    }
                    Ok(())
                })
                .await
                .map_err(sqlite_error)?;
            Ok(())
        })
    }

    fn list_uncommitted(
        &self,
        older_than_epoch_ms: u64,
    ) -> Result<Vec<AttachmentManifestEntry>, StoreError> {
        block_on_store(async {
            let older_than = older_than_epoch_ms as i64;
            self.conn
                .call(move |conn| {
                    let mut stmt = conn.prepare(
                        "SELECT attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms
                         FROM attachment_manifest
                         WHERE committed_at_ms IS NULL AND intent_at_ms <= ?1
                         ORDER BY intent_at_ms ASC",
                    )?;
                    let rows = stmt.query_map(params![older_than], |row| {
                        let id: String = row.get(0)?;
                        let session_id: String = row.get(1)?;
                        let canonical_uri: String = row.get(2)?;
                        let intent_at_ms: i64 = row.get(3)?;
                        let committed_at_ms: Option<i64> = row.get(4)?;
                        Ok(AttachmentManifestEntry {
                            attachment_id: AttachmentId::new(id),
                            session_id,
                            canonical_uri,
                            intent_at_epoch_ms: intent_at_ms as u64,
                            committed_at_epoch_ms: committed_at_ms.map(|v| v as u64),
                        })
                    })?;
                    Ok(rows.filter_map(Result::ok).collect())
                })
                .await
                .map_err(sqlite_error)
        })
    }

    fn forget(&self, attachment_id: &AttachmentId) -> Result<(), StoreError> {
        block_on_store(async {
            let attachment_id = attachment_id.as_str().to_string();
            self.conn
                .call(move |conn| {
                    conn.execute(
                        "DELETE FROM attachment_manifest WHERE attachment_id = ?1",
                        params![attachment_id],
                    )
                })
                .await
                .map_err(sqlite_error)?;
            Ok(())
        })
    }
}
