impl AttachmentManifest for PostgresSessionStore {
    fn record_intent(&self, intent: AttachmentIntent) -> Result<(), StoreError> {
        let pool = self.pool.clone();
        block_on_detached(async move {
            sqlx::query(
                "INSERT INTO lash_attachment_manifest (
                    attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms
                 )
                 VALUES ($1, $2, $3, $4, NULL)
                 ON CONFLICT (attachment_id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    canonical_uri = EXCLUDED.canonical_uri,
                    intent_at_ms = EXCLUDED.intent_at_ms",
            )
            .bind(intent.attachment_id.as_str())
            .bind(intent.session_id)
            .bind(intent.canonical_uri)
            .bind(intent.intent_at_epoch_ms as i64)
            .execute(&pool)
            .await
            .map(|_| ())
            .map_err(store_sqlx_error)
        })
    }

    fn commit_refs(
        &self,
        session_id: &str,
        attachment_ids: &[AttachmentId],
    ) -> Result<(), StoreError> {
        let pool = self.pool.clone();
        let session_id = session_id.to_string();
        let attachment_ids = attachment_ids.to_vec();
        block_on_detached(async move {
            let mut tx = pool.begin().await.map_err(store_sqlx_error)?;
            commit_attachment_refs_tx(&mut tx, &session_id, &attachment_ids).await?;
            tx.commit().await.map_err(store_sqlx_error)
        })
    }

    fn list_uncommitted(
        &self,
        older_than_epoch_ms: u64,
    ) -> Result<Vec<AttachmentManifestEntry>, StoreError> {
        let pool = self.pool.clone();
        block_on_detached(async move {
            let rows = sqlx::query(
                "SELECT attachment_id, session_id, canonical_uri, intent_at_ms, committed_at_ms
                 FROM lash_attachment_manifest
                 WHERE committed_at_ms IS NULL AND intent_at_ms <= $1
                 ORDER BY attachment_id ASC",
            )
            .bind(older_than_epoch_ms as i64)
            .fetch_all(&pool)
            .await
            .map_err(store_sqlx_error)?;
            Ok(rows
                .into_iter()
                .map(|row| AttachmentManifestEntry {
                    attachment_id: AttachmentId::new(row.get::<String, _>(0)),
                    session_id: row.get(1),
                    canonical_uri: row.get(2),
                    intent_at_epoch_ms: row.get::<i64, _>(3) as u64,
                    committed_at_epoch_ms: row.get::<Option<i64>, _>(4).map(|value| value as u64),
                })
                .collect())
        })
    }

    fn forget(&self, attachment_id: &AttachmentId) -> Result<(), StoreError> {
        let pool = self.pool.clone();
        let attachment_id = attachment_id.to_string();
        block_on_detached(async move {
            sqlx::query("DELETE FROM lash_attachment_manifest WHERE attachment_id = $1")
                .bind(attachment_id)
                .execute(&pool)
                .await
                .map(|_| ())
                .map_err(store_sqlx_error)
        })
    }
}
