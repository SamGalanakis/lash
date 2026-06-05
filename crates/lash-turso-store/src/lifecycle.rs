//! [`Store`] open/memory lifecycle plus session head/meta accessors.

use super::*;

impl Store {
    pub async fn open(path: &Path) -> turso::Result<Self> {
        Self::open_with_options(path, StoreOptions::default()).await
    }

    pub async fn open_with_options(path: &Path, options: StoreOptions) -> turso::Result<Self> {
        let _schema_guard = file_schema_open_guard().await;
        let path = path.to_string_lossy().into_owned();
        let db = turso::Builder::new_local(&path).build().await?;
        let conn = db.connect()?;
        conn.busy_timeout(TURSO_BUSY_TIMEOUT)?;
        ensure_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::File).await?;
        Ok(Self {
            _db: db,
            conn: tokio::sync::Mutex::new(conn),
            artifact_cache: Mutex::new(BTreeMap::new()),
            options,
            commit_count: AtomicU64::new(0),
        })
    }

    /// Turso does not expose a read-only local open mode yet. This method keeps
    /// the export/resume call sites explicit while opening the local database
    /// with normal Turso semantics.
    pub async fn open_readonly(path: &Path) -> turso::Result<Self> {
        Self::open(path).await
    }

    pub async fn load_picker_info(&self) -> Option<SessionPickerInfo> {
        let conn = self.conn.lock().await;
        let row = optional_row(
            &conn,
            "SELECT session_id, cwd, relation_json
             FROM session_meta WHERE singleton = 1",
            (),
        )
        .await
        .ok()??;
        let session_id = row_string(&row, 0).ok()?;
        let cwd = row_optional_string(&row, 1).ok()?;
        let relation = row_optional_string(&row, 2)
            .ok()
            .flatten()
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();

        let head_json = optional_row(
            &conn,
            "SELECT head_json FROM session_head WHERE singleton = 1",
            (),
        )
        .await
        .ok()
        .flatten()
        .and_then(|row| row_string(&row, 0).ok())
        .unwrap_or_else(|| "{}".to_string());
        let head_meta = serde_json::from_str::<SessionHeadMeta>(&head_json).unwrap_or_default();
        let graph = Self::load_session_graph_from_conn(&conn, head_meta.leaf_node_id).await;

        Some(SessionPickerInfo {
            session_id,
            cwd,
            relation,
            first_user_message: graph.first_user_message(),
            user_message_count: graph.user_message_count(),
        })
    }

    pub async fn memory() -> turso::Result<Self> {
        Self::memory_with_options(StoreOptions {
            blob_profile: BuiltinBlobProfile::LowLatency,
            gc_policy: StoreGcPolicy::default(),
        })
        .await
    }

    pub async fn memory_with_options(options: StoreOptions) -> turso::Result<Self> {
        let db = turso::Builder::new_local(":memory:").build().await?;
        let conn = db.connect()?;
        conn.busy_timeout(TURSO_BUSY_TIMEOUT)?;
        ensure_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::Memory).await?;
        Ok(Self {
            _db: db,
            conn: tokio::sync::Mutex::new(conn),
            artifact_cache: Mutex::new(BTreeMap::new()),
            options,
            commit_count: AtomicU64::new(0),
        })
    }

    pub async fn save_session_head_meta(&self, meta: SessionHeadMeta) {
        let conn = self.conn.lock().await;
        let head_json = encode_json(&meta);
        if let Err(err) = conn
            .execute(
                "INSERT OR REPLACE INTO session_head (singleton, session_id, head_json, head_revision)
                 VALUES (1, ?1, ?2, ?3)",
                params![meta.session_id, head_json, meta.head_revision as i64],
            )
            .await
        {
            tracing::warn!(error = %err, "failed to persist session head");
        }
    }

    pub async fn load_session_head_meta(&self) -> Option<SessionHeadMeta> {
        let conn = self.conn.lock().await;
        load_session_head_meta_from_conn(&conn).await
    }

    pub async fn save_session_head(&self, head: SessionHead) {
        self.replace_session_graph(&head.graph).await;
        self.save_session_head_meta(session_head_meta(&head)).await;
    }

    pub async fn load_session_head(&self) -> Option<SessionHead> {
        let meta = self.load_session_head_meta().await?;
        let mut graph = self.load_session_graph().await;
        graph.set_leaf_node_id(meta.leaf_node_id.clone());
        Some(SessionHead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            agent_frames: meta.agent_frames,
            current_agent_frame_id: meta.current_agent_frame_id,
            graph,
            config: meta.config,
            checkpoint_ref: meta.checkpoint_ref,
            token_ledger: merge_token_ledger_entries(self.load_usage_deltas().await),
        })
    }

    pub async fn head_copy_from_store(&self, source: &Store) {
        if let Some(head) = source.load_session_head().await {
            if let Some(checkpoint_ref) = &head.checkpoint_ref
                && let Some(record) = source
                    .get_typed_blob::<SessionCheckpoint>(checkpoint_ref)
                    .await
            {
                for blob_ref in [
                    record.tool_state_ref.as_ref(),
                    record.plugin_snapshot_ref.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    if let Some(blob) = source.get_blob(blob_ref).await {
                        let descriptor = match record
                            .tool_state_ref
                            .as_ref()
                            .filter(|candidate| *candidate == blob_ref)
                        {
                            Some(_) => BlobArtifactDescriptor::tool_state_snapshot(),
                            None => BlobArtifactDescriptor::plugin_session_snapshot(),
                        };
                        let _ = self.put_artifact_blob(descriptor, &blob).await;
                    }
                }
                if let Some(blob) = source.get_blob(checkpoint_ref).await {
                    let _ = self
                        .put_artifact_blob(BlobArtifactDescriptor::checkpoint_manifest(), &blob)
                        .await;
                }
            }
            self.replace_session_graph(&head.graph).await;
            self.save_session_head_meta(session_head_meta(&head)).await;
        }
    }

    pub async fn save_session_meta(&self, meta: SessionMeta) {
        let conn = self.conn.lock().await;
        let relation_json = serde_json::to_string(&meta.relation).ok();
        let session_id_for_log = meta.session_id.clone();
        if let Err(err) = conn
            .execute(
                "INSERT OR REPLACE INTO session_meta
                 (singleton, session_id, session_name, created_at, model, cwd, relation_json)
                 VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    meta.session_id.as_str(),
                    meta.session_name.as_str(),
                    meta.created_at.as_str(),
                    meta.model.as_str(),
                    meta.cwd.as_deref(),
                    relation_json.as_deref()
                ],
            )
            .await
        {
            tracing::warn!(
                error = %err,
                session_id = session_id_for_log,
                "failed to persist session metadata"
            );
        }
    }

    pub async fn load_session_meta(&self) -> Option<SessionMeta> {
        let conn = self.conn.lock().await;
        load_session_meta_from_conn(&conn).await
    }
}
