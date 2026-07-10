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

/// Logical keyspaces multiplexed onto the `artifact_refs` pointer table. Each
/// namespace owns its own half of the `(namespace, artifact_ref)` composite
/// primary key. The `blobs` table is content-addressed, but the `artifact_refs`
/// pointer is *not*: without the namespace column, a module ref that collides
/// with a process-execution-env ref would rewrite the same pointer row under
/// `INSERT OR REPLACE`, so content-addressing alone does not keep the namespaces
/// disjoint. The composite key does.
pub(crate) const MODULE_ARTIFACT_NAMESPACE: &str = "lashlang_module";
pub(crate) const RAW_ARTIFACT_NAMESPACE: &str = "lashlang_artifact";
pub(crate) const PROCESS_ENV_NAMESPACE: &str = "process_execution_env";

impl Store {
    async fn put_artifact_ref_blob(
        &self,
        namespace: &'static str,
        artifact_ref: String,
        descriptor: BlobArtifactDescriptor,
        bytes: Vec<u8>,
    ) -> Result<(), StoreError> {
        let blob_profile = self.options.blob_profile;
        self.conn
            .write(move |tx| {
                let blob_ref =
                    Self::insert_artifact_blob_conn(tx, descriptor, &bytes, blob_profile)?;
                tx.execute(
                    "INSERT OR REPLACE INTO artifact_refs (namespace, artifact_ref, blob_ref)
                     VALUES (?1, ?2, ?3)",
                    params![namespace, artifact_ref, blob_ref.as_str()],
                )?;
                Ok(())
            })
            .await
            .map_err(sqlite_error)
    }

    async fn get_artifact_ref_blob(
        &self,
        namespace: &'static str,
        artifact_ref: String,
        missing_diagnostic: String,
    ) -> Result<Option<Vec<u8>>, StoreError> {
        let resolved = self
            .conn
            .call(move |conn| {
                let blob_ref: Option<String> = conn
                    .query_row(
                        "SELECT blob_ref FROM artifact_refs
                         WHERE namespace = ?1 AND artifact_ref = ?2",
                        params![namespace, artifact_ref],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?;
                let Some(blob_ref) = blob_ref else {
                    return Ok(None);
                };
                Ok(Some(Self::get_blob_conn(conn, &BlobRef(blob_ref))))
            })
            .await
            .map_err(sqlite_error)?;
        let Some(blob) = resolved else {
            return Ok(None);
        };
        blob.ok_or_else(|| {
            StoreError::Backend(format!("{missing_diagnostic} points at a missing blob"))
        })
        .map(Some)
    }
}

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
        let artifact_ref = artifact.module_ref.as_str().to_string();
        self.put_artifact_ref_blob(
            MODULE_ARTIFACT_NAMESPACE,
            artifact_ref,
            BlobArtifactDescriptor::lashlang_module(),
            bytes,
        )
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
        let Some(bytes) = self
            .get_artifact_ref_blob(
                MODULE_ARTIFACT_NAMESPACE,
                artifact_ref,
                format!("lashlang module artifact `{module_ref}`"),
            )
            .await
            .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))?
        else {
            return Ok(None);
        };
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
        let artifact_ref = artifact_ref.to_string();
        let descriptor = match descriptor {
            "process_execution_env" => BlobArtifactDescriptor::process_execution_env(),
            _ => BlobArtifactDescriptor::new(PersistedArtifactKind::GenericBlob, Vec::new()),
        };
        self.put_artifact_ref_blob(
            RAW_ARTIFACT_NAMESPACE,
            artifact_ref,
            descriptor,
            bytes.to_vec(),
        )
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))
    }

    async fn get_artifact_bytes(
        &self,
        artifact_ref: &str,
    ) -> Result<Option<Vec<u8>>, lashlang::ArtifactStoreError> {
        let artifact_ref = artifact_ref.to_string();
        self.get_artifact_ref_blob(
            RAW_ARTIFACT_NAMESPACE,
            artifact_ref.clone(),
            format!("artifact `{artifact_ref}`"),
        )
        .await
        .map_err(|err| lashlang::ArtifactStoreError::Backend(err.to_string()))
    }
}

#[async_trait::async_trait]
impl lash_core::ProcessExecutionEnvStore for Store {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn put_process_execution_env(
        &self,
        env_ref: &lash_core::ProcessExecutionEnvRef,
        bytes: &[u8],
    ) -> Result<(), lash_core::PluginError> {
        let artifact_ref = env_ref.as_str().to_string();
        self.put_artifact_ref_blob(
            PROCESS_ENV_NAMESPACE,
            artifact_ref,
            BlobArtifactDescriptor::process_execution_env(),
            bytes.to_vec(),
        )
        .await
        .map_err(|err| lash_core::PluginError::Session(err.to_string()))
    }

    async fn get_process_execution_env(
        &self,
        env_ref: &lash_core::ProcessExecutionEnvRef,
    ) -> Result<Option<Vec<u8>>, lash_core::PluginError> {
        let artifact_ref = env_ref.as_str().to_string();
        self.get_artifact_ref_blob(
            PROCESS_ENV_NAMESPACE,
            artifact_ref.clone(),
            format!("process execution env `{artifact_ref}`"),
        )
        .await
        .map_err(|err| lash_core::PluginError::Session(err.to_string()))
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
                    // Re-recording an intent REFRESHES `intent_at_ms` (the
                    // `DO UPDATE SET intent_at_ms` below). A long-lived retry loop
                    // re-`put`ting the same content id keeps bumping the timestamp
                    // forward, so the crash-orphan reconciliation — which deletes
                    // uncommitted intents at/before the grace cutoff — never
                    // collects a blob a session is still actively retrying.
                    conn.execute(
                        "INSERT INTO attachment_manifest
                            (attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms)
                         VALUES (?1, ?2, ?3, ?4, NULL)
                         ON CONFLICT(session_id, attachment_id) DO UPDATE SET
                            canonical_uri = excluded.canonical_uri,
                            intent_at_ms = excluded.intent_at_ms",
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
            let now = self.clock.timestamp_ms() as i64;
            self.conn
                .write(move |tx| {
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

    fn forget_aged_uncommitted_intents(
        &self,
        intent_grace_cutoff_epoch_ms: u64,
    ) -> Result<(), StoreError> {
        block_on_store(async {
            let cutoff = intent_grace_cutoff_epoch_ms as i64;
            self.conn
                .write(move |tx| {
                    // One conditional DELETE inside the write transaction — the age
                    // predicate lives in the statement, not in a prior read. A
                    // concurrent `record_intent` that refreshes `intent_at_ms` past
                    // the cutoff no longer matches, so a freshly-refreshed live
                    // intent is never deleted (no list-then-forget race).
                    tx.execute(
                        "DELETE FROM attachment_manifest
                         WHERE committed_at_ms IS NULL AND intent_at_ms <= ?1",
                        params![cutoff],
                    )?;
                    Ok(())
                })
                .await
                .map_err(sqlite_error)?;
            Ok(())
        })
    }

    fn has_live_ref_for_id(
        &self,
        attachment_id: &AttachmentId,
        intent_grace_cutoff_epoch_ms: u64,
    ) -> Result<bool, StoreError> {
        block_on_store(async {
            let attachment_id = attachment_id.as_str().to_string();
            let cutoff = intent_grace_cutoff_epoch_ms as i64;
            self.conn
                .call(move |conn| {
                    // A GC-live ref: a committed row, or an uncommitted intent
                    // younger than the cutoff. Aged orphan intents do not count.
                    conn.query_row(
                        "SELECT 1 FROM attachment_manifest
                         WHERE attachment_id = ?1
                           AND (committed_at_ms IS NOT NULL OR intent_at_ms > ?2)
                         LIMIT 1",
                        params![attachment_id, cutoff],
                        |_| Ok(()),
                    )
                    .optional()
                    .map(|found| found.is_some())
                })
                .await
                .map_err(sqlite_error)
        })
    }

    fn forget(&self, session_id: &str, attachment_id: &AttachmentId) -> Result<(), StoreError> {
        block_on_store(async {
            let session_id = session_id.to_string();
            let attachment_id = attachment_id.as_str().to_string();
            self.conn
                .call(move |conn| {
                    conn.execute(
                        "DELETE FROM attachment_manifest
                         WHERE session_id = ?1 AND attachment_id = ?2",
                        params![session_id, attachment_id],
                    )
                })
                .await
                .map_err(sqlite_error)?;
            Ok(())
        })
    }

    fn holds_ref(
        &self,
        session_id: &str,
        attachment_id: &AttachmentId,
    ) -> Result<bool, StoreError> {
        block_on_store(async {
            let session_id = session_id.to_string();
            let attachment_id = attachment_id.as_str().to_string();
            self.conn
                .call(move |conn| {
                    conn.query_row(
                        "SELECT 1 FROM attachment_manifest
                         WHERE session_id = ?1 AND attachment_id = ?2",
                        params![session_id, attachment_id],
                        |_| Ok(()),
                    )
                    .optional()
                    .map(|found| found.is_some())
                })
                .await
                .map_err(sqlite_error)
        })
    }

    fn list_all_refs(&self) -> Result<Vec<AttachmentId>, StoreError> {
        block_on_store(async {
            self.conn
                .call(move |conn| {
                    let mut stmt =
                        conn.prepare("SELECT DISTINCT attachment_id FROM attachment_manifest")?;
                    let rows = stmt.query_map([], |row| {
                        let id: String = row.get(0)?;
                        Ok(AttachmentId::new(id))
                    })?;
                    Ok(rows.filter_map(Result::ok).collect())
                })
                .await
                .map_err(sqlite_error)
        })
    }
}
