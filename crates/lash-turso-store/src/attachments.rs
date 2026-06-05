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
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ()).await.map_err(|err| {
            lashlang::ArtifactStoreError::Backend(format!("begin artifact write: {err}"))
        })?;
        let result = async {
            let blob_ref = Self::insert_artifact_blob_conn(
                &conn,
                BlobArtifactDescriptor::lashlang_module(),
                &bytes,
                blob_profile,
            )
            .await
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
            conn.execute(
                "INSERT OR REPLACE INTO artifact_refs (artifact_ref, blob_ref) VALUES (?1, ?2)",
                params![artifact.module_ref.as_str(), blob_ref.as_str()],
            )
            .await
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
            Ok::<(), lashlang::ArtifactStoreError>(())
        }
        .await;
        match result {
            Ok(()) => {
                conn.execute("COMMIT", ()).await.map_err(|err| {
                    lashlang::ArtifactStoreError::Backend(format!("commit artifact write: {err}"))
                })?;
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(err);
            }
        }
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

        let conn = self.conn.lock().await;
        let blob_ref = optional_row(
            &conn,
            "SELECT blob_ref FROM artifact_refs WHERE artifact_ref = ?1",
            params![module_ref.as_str()],
        )
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?
        .map(|row| {
            row_string(&row, 0)
                .map(BlobRef)
                .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))
        })
        .transpose()?;
        let Some(blob_ref) = blob_ref else {
            return Ok(None);
        };
        let bytes = Self::get_blob_conn(&conn, &blob_ref).await.ok_or_else(|| {
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
}

impl AttachmentManifest for Store {
    fn record_intent(&self, intent: AttachmentIntent) -> Result<(), StoreError> {
        block_on_store(async {
            let conn = self.conn.lock().await;
            conn.execute(
                "INSERT INTO attachment_manifest
                    (attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms)
                 VALUES (?1, ?2, ?3, ?4, NULL)
                 ON CONFLICT(attachment_id) DO NOTHING",
                params![
                    intent.attachment_id.as_str(),
                    intent.session_id.as_str(),
                    intent.canonical_uri.as_str(),
                    intent.intent_at_epoch_ms as i64,
                ],
            )
            .await
            .map_err(turso_error)?;
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
            let conn = self.conn.lock().await;
            conn.execute("BEGIN IMMEDIATE", ())
                .await
                .map_err(turso_error)?;
            let result = async {
                let now = current_epoch_ms() as i64;
                for id in attachment_ids {
                    conn.execute(
                        "UPDATE attachment_manifest
                         SET committed_at_ms = COALESCE(committed_at_ms, ?1)
                         WHERE attachment_id = ?2 AND session_id = ?3",
                        params![now, id.as_str(), session_id],
                    )
                    .await
                    .map_err(turso_error)?;
                }
                Ok::<(), StoreError>(())
            }
            .await;
            match result {
                Ok(()) => {
                    conn.execute("COMMIT", ()).await.map_err(turso_error)?;
                    Ok(())
                }
                Err(err) => {
                    let _ = conn.execute("ROLLBACK", ()).await;
                    Err(err)
                }
            }
        })
    }

    fn list_uncommitted(
        &self,
        older_than_epoch_ms: u64,
    ) -> Result<Vec<AttachmentManifestEntry>, StoreError> {
        block_on_store(async {
            let conn = self.conn.lock().await;
            let rows = collect_rows(
                &conn,
                "SELECT attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms
                 FROM attachment_manifest
                 WHERE committed_at_ms IS NULL AND intent_at_ms <= ?1
                 ORDER BY intent_at_ms ASC",
                params![older_than_epoch_ms as i64],
            )
            .await
            .map_err(turso_error)?;
            let mut out = Vec::new();
            for row in rows {
                out.push(AttachmentManifestEntry {
                    attachment_id: AttachmentId::new(row_string(&row, 0).map_err(turso_error)?),
                    session_id: row_string(&row, 1).map_err(turso_error)?,
                    canonical_uri: row_string(&row, 2).map_err(turso_error)?,
                    intent_at_epoch_ms: row_i64(&row, 3).map_err(turso_error)? as u64,
                    committed_at_epoch_ms: row_optional_i64(&row, 4)
                        .map_err(turso_error)?
                        .map(|value| value as u64),
                });
            }
            Ok(out)
        })
    }

    fn forget(&self, attachment_id: &AttachmentId) -> Result<(), StoreError> {
        block_on_store(async {
            let conn = self.conn.lock().await;
            conn.execute(
                "DELETE FROM attachment_manifest WHERE attachment_id = ?1",
                params![attachment_id.as_str()],
            )
            .await
            .map_err(turso_error)?;
            Ok(())
        })
    }
}
