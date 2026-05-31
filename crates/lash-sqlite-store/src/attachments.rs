use super::*;

impl lashlang::LashlangArtifactStore for Store {
    fn durability_tier(&self) -> lashlang::DurabilityTier {
        lashlang::DurabilityTier::Durable
    }

    fn put_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), lashlang::ArtifactStoreError> {
        let bytes = artifact
            .to_store_bytes()
            .map_err(|err| lashlang::ArtifactStoreError::Encode(err.to_string()))?;
        let blob_profile = self.options.blob_profile;
        let mut conn = lock_conn(&self.conn);
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        let blob_ref = Self::insert_artifact_blob_conn(
            &tx,
            BlobArtifactDescriptor::lashlang_module(),
            &bytes,
            blob_profile,
        )
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        tx.execute(
            "INSERT OR REPLACE INTO artifact_refs (artifact_ref, blob_ref) VALUES (?1, ?2)",
            params![artifact.module_ref.as_str(), blob_ref.as_str()],
        )
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        tx.commit()
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        Ok(())
    }

    fn get_module_artifact(
        &self,
        module_ref: &lashlang::ModuleRef,
    ) -> Result<Option<lashlang::ModuleArtifact>, lashlang::ArtifactStoreError> {
        let conn = lock_conn(&self.conn);
        let blob_ref = conn
            .query_row(
                "SELECT blob_ref FROM artifact_refs WHERE artifact_ref = ?1",
                params![module_ref.as_str()],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?;
        let Some(blob_ref) = blob_ref else {
            return Ok(None);
        };
        let bytes = Self::get_blob_conn(&conn, &BlobRef(blob_ref)).ok_or_else(|| {
            lashlang::ArtifactStoreError::Backend(format!(
                "lashlang module artifact `{}` points at a missing blob",
                module_ref
            ))
        })?;
        lashlang::ModuleArtifact::from_store_bytes(&bytes)
            .map(Some)
            .map_err(lashlang::ArtifactStoreError::from)
    }
}

impl AttachmentManifest for Store {
    fn record_intent(&self, intent: AttachmentIntent) -> Result<(), StoreError> {
        let conn = lock_conn(&self.conn);
        conn.execute(
            "INSERT INTO attachment_manifest
                (attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms)
             VALUES (?1, ?2, ?3, ?4, NULL)
             ON CONFLICT(attachment_id) DO NOTHING",
            params![
                intent.attachment_id.as_str(),
                intent.session_id,
                intent.canonical_uri,
                intent.intent_at_epoch_ms as i64,
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    fn commit_refs(
        &self,
        session_id: &str,
        attachment_ids: &[AttachmentId],
    ) -> Result<(), StoreError> {
        if attachment_ids.is_empty() {
            return Ok(());
        }
        self.with_write_tx(|tx| {
            let now = current_epoch_ms() as i64;
            let mut stmt = tx
                .prepare(
                    "UPDATE attachment_manifest
                     SET committed_at_ms = COALESCE(committed_at_ms, ?1)
                     WHERE attachment_id = ?2 AND session_id = ?3",
                )
                .map_err(sqlite_error)?;
            for id in attachment_ids {
                stmt.execute(params![now, id.as_str(), session_id])
                    .map_err(sqlite_error)?;
            }
            Ok(())
        })
    }

    fn list_uncommitted(
        &self,
        older_than_epoch_ms: u64,
    ) -> Result<Vec<AttachmentManifestEntry>, StoreError> {
        self.with_read(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms
                     FROM attachment_manifest
                     WHERE committed_at_ms IS NULL AND intent_at_ms <= ?1
                     ORDER BY intent_at_ms ASC",
                )
                .map_err(sqlite_error)?;
            let rows = stmt
                .query_map(params![older_than_epoch_ms as i64], |row| {
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
                })
                .map_err(sqlite_error)?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(sqlite_error)?);
            }
            Ok(out)
        })
    }

    fn forget(&self, attachment_id: &AttachmentId) -> Result<(), StoreError> {
        let conn = lock_conn(&self.conn);
        conn.execute(
            "DELETE FROM attachment_manifest WHERE attachment_id = ?1",
            params![attachment_id.as_str()],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }
}
