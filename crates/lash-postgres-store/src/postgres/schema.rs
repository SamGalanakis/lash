async fn ensure_schema(pool: &PgPool) -> Result<(), StoreError> {
    let mut tx = pool.begin().await.map_err(store_sqlx_error)?;
    tx.execute("SELECT pg_advisory_xact_lock(715421, 907001)")
        .await
        .map_err(store_sqlx_error)?;
    tx.execute(
        r#"
        CREATE TABLE IF NOT EXISTS lash_schema_versions (
            component TEXT PRIMARY KEY,
            version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_blobs (
            hash TEXT PRIMARY KEY,
            content BYTEA NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_sessions (
            session_id TEXT PRIMARY KEY,
            head_revision BIGINT NOT NULL DEFAULT 0,
            head_json TEXT NOT NULL,
            checkpoint_ref TEXT
        );

        CREATE TABLE IF NOT EXISTS lash_graph_nodes (
            session_id TEXT NOT NULL,
            seq BIGSERIAL,
            node_id TEXT NOT NULL,
            node_json TEXT NOT NULL,
            tombstoned BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY (session_id, node_id)
        );
        CREATE INDEX IF NOT EXISTS idx_lash_graph_nodes_seq
            ON lash_graph_nodes(session_id, seq);

        CREATE TABLE IF NOT EXISTS lash_usage_deltas (
            seq BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            entry_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_session_meta (
            session_id TEXT PRIMARY KEY,
            meta_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_runtime_turn_commits (
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            turn_commit_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            committed_at_ms BIGINT NOT NULL,
            PRIMARY KEY (session_id, turn_id)
        );

        CREATE TABLE IF NOT EXISTS lash_session_execution_leases (
            session_id TEXT PRIMARY KEY,
            lease_owner_id TEXT,
            lease_token TEXT,
            lease_fencing_token BIGINT NOT NULL DEFAULT 0,
            lease_claimed_at_ms BIGINT NOT NULL DEFAULT 0,
            lease_expires_at_ms BIGINT NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS lash_queued_work_batches (
            enqueue_seq BIGSERIAL PRIMARY KEY,
            batch_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            source_key TEXT,
            delivery_policy TEXT NOT NULL,
            slot_policy TEXT NOT NULL,
            merge_key_json TEXT NOT NULL,
            available_at_ms BIGINT NOT NULL,
            enqueued_at_ms BIGINT NOT NULL,
            claim_id TEXT,
            claim_owner_id TEXT,
            claim_token TEXT,
            claim_fencing_token BIGINT NOT NULL DEFAULT 0,
            claim_claimed_at_ms BIGINT NOT NULL DEFAULT 0,
            claim_expires_at_ms BIGINT NOT NULL DEFAULT 0,
            UNIQUE (session_id, source_key)
        );
        CREATE INDEX IF NOT EXISTS idx_lash_queued_work_ready
            ON lash_queued_work_batches(session_id, available_at_ms, enqueue_seq);

        CREATE TABLE IF NOT EXISTS lash_queued_work_items (
            batch_id TEXT NOT NULL REFERENCES lash_queued_work_batches(batch_id) ON DELETE CASCADE,
            item_index INTEGER NOT NULL,
            item_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (batch_id, item_index)
        );

        CREATE TABLE IF NOT EXISTS lash_attachment_manifest (
            attachment_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            canonical_uri TEXT NOT NULL,
            intent_at_ms BIGINT NOT NULL,
            committed_at_ms BIGINT
        );
        CREATE INDEX IF NOT EXISTS idx_lash_attachment_manifest_uncommitted
            ON lash_attachment_manifest(committed_at_ms)
            WHERE committed_at_ms IS NULL;

        CREATE TABLE IF NOT EXISTS lash_processes (
            process_id TEXT PRIMARY KEY,
            registration_hash TEXT NOT NULL,
            owner_scope_id TEXT NOT NULL,
            created_at_ms BIGINT NOT NULL,
            updated_at_ms BIGINT NOT NULL,
            status TEXT NOT NULL,
            record_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lash_processes_status
            ON lash_processes(status);

        CREATE TABLE IF NOT EXISTS lash_process_events (
            process_id TEXT NOT NULL REFERENCES lash_processes(process_id) ON DELETE CASCADE,
            sequence BIGINT NOT NULL,
            event_type TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            idempotency_key TEXT,
            occurred_at_ms BIGINT NOT NULL,
            event_json TEXT NOT NULL,
            PRIMARY KEY (process_id, sequence)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_lash_process_events_key
            ON lash_process_events(process_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL;

        CREATE TABLE IF NOT EXISTS lash_process_wake_acks (
            process_id TEXT NOT NULL REFERENCES lash_processes(process_id) ON DELETE CASCADE,
            sequence BIGINT NOT NULL,
            PRIMARY KEY (process_id, sequence)
        );

        CREATE TABLE IF NOT EXISTS lash_process_handle_grants (
            session_id TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            process_id TEXT NOT NULL REFERENCES lash_processes(process_id) ON DELETE CASCADE,
            descriptor_json TEXT NOT NULL,
            PRIMARY KEY (scope_id, process_id)
        );
        CREATE INDEX IF NOT EXISTS idx_lash_process_handle_grants_session
            ON lash_process_handle_grants(session_id);
        CREATE INDEX IF NOT EXISTS idx_lash_process_handle_grants_process
            ON lash_process_handle_grants(process_id);

        CREATE TABLE IF NOT EXISTS lash_process_leases (
            process_id TEXT PRIMARY KEY REFERENCES lash_processes(process_id) ON DELETE CASCADE,
            lease_owner_id TEXT,
            lease_token TEXT,
            lease_fencing_token BIGINT NOT NULL DEFAULT 0,
            lease_claimed_at_ms BIGINT NOT NULL DEFAULT 0,
            lease_expires_at_ms BIGINT NOT NULL DEFAULT 0
        );

        CREATE SEQUENCE IF NOT EXISTS lash_trigger_subscription_seq;
        CREATE TABLE IF NOT EXISTS lash_trigger_subscriptions (
            subscription_id TEXT PRIMARY KEY,
            registrant_scope_id TEXT NOT NULL,
            handle TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_key TEXT NOT NULL,
            enabled BOOLEAN NOT NULL,
            created_at_ms BIGINT NOT NULL,
            updated_at_ms BIGINT NOT NULL,
            record_json TEXT NOT NULL,
            UNIQUE(registrant_scope_id, handle)
        );
        CREATE INDEX IF NOT EXISTS idx_lash_trigger_subscriptions_registrant
            ON lash_trigger_subscriptions(registrant_scope_id, handle);
        CREATE INDEX IF NOT EXISTS idx_lash_trigger_subscriptions_source
            ON lash_trigger_subscriptions(source_type, source_key, enabled);

        CREATE TABLE IF NOT EXISTS lash_trigger_occurrences (
            occurrence_id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            request_hash TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_key TEXT NOT NULL,
            occurred_at_ms BIGINT NOT NULL,
            record_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lash_trigger_deliveries (
            occurrence_id TEXT NOT NULL REFERENCES lash_trigger_occurrences(occurrence_id) ON DELETE CASCADE,
            subscription_id TEXT NOT NULL REFERENCES lash_trigger_subscriptions(subscription_id) ON DELETE CASCADE,
            process_id TEXT NOT NULL,
            created_at_ms BIGINT NOT NULL,
            PRIMARY KEY (occurrence_id, subscription_id)
        );

        CREATE TABLE IF NOT EXISTS lash_lashlang_artifacts (
            module_ref TEXT PRIMARY KEY,
            artifact_bytes BYTEA NOT NULL
        );
        "#,
    )
    .await
    .map_err(store_sqlx_error)?;

    let existing: Option<i32> =
        sqlx::query_scalar("SELECT version FROM lash_schema_versions WHERE component = $1")
            .bind(SCHEMA_COMPONENT)
            .fetch_optional(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
    match existing {
        Some(version) if version == SCHEMA_VERSION => {}
        Some(version) => {
            return Err(StoreError::Backend(format!(
                "Postgres schema component `{SCHEMA_COMPONENT}` has version {version}, expected {SCHEMA_VERSION}"
            )));
        }
        None => {
            sqlx::query("INSERT INTO lash_schema_versions (component, version) VALUES ($1, $2)")
                .bind(SCHEMA_COMPONENT)
                .bind(SCHEMA_VERSION)
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
        }
    }
    tx.commit().await.map_err(store_sqlx_error)
}
