use super::*;

#[async_trait::async_trait]
impl RuntimePersistence for Store {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn load_session(
        &self,
        scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, StoreError> {
        let conn = self.conn.lock().await;
        let Some(meta) = try_load_session_head_meta_from_conn(&conn).await? else {
            return Ok(None);
        };
        let leaf_node_id = match &scope {
            SessionReadScope::FullGraph => meta.leaf_node_id.clone(),
            SessionReadScope::ActivePath { leaf_node_id } => {
                leaf_node_id.clone().or_else(|| meta.leaf_node_id.clone())
            }
        };
        let mut graph = match scope {
            SessionReadScope::FullGraph => {
                Self::load_session_graph_from_conn(&conn, meta.leaf_node_id.clone()).await
            }
            SessionReadScope::ActivePath { .. } => {
                Self::load_active_path_session_graph_from_conn(&conn, leaf_node_id.clone())
                    .await
                    .map_err(turso_error)?
            }
        };
        graph.set_leaf_node_id(leaf_node_id);
        let checkpoint = match meta.checkpoint_ref.as_ref() {
            Some(blob_ref) => Self::get_checkpoint_conn(&conn, blob_ref).await,
            None => None,
        };
        Ok(Some(PersistedSessionRead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            config: meta.config,
            agent_frames: meta.agent_frames,
            current_agent_frame_id: meta.current_agent_frame_id,
            graph,
            checkpoint_ref: meta.checkpoint_ref,
            checkpoint,
            token_ledger: merge_token_ledger_entries(Self::load_usage_deltas_conn(&conn).await),
        }))
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<lash_core::SessionNodeRecord>, StoreError> {
        let conn = self.conn.lock().await;
        let row = optional_row(
            &conn,
            "SELECT node_json FROM graph_nodes WHERE node_id = ?1 AND tombstoned = 0",
            params![node_id],
        )
        .await
        .map_err(turso_error)?;
        Ok(row.and_then(|row| {
            let json = row_string(&row, 0).ok()?;
            serde_json::from_str(&json).ok()
        }))
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let blob_profile = self.options.blob_profile;
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(turso_error)?;
        let result = async {
            let existing = try_load_session_head_meta_from_conn(&conn).await?;
            if let Some(bound_session_id) = existing.as_ref().map(|meta| meta.session_id.as_str())
                && bound_session_id != commit.session_id
            {
                return Err(StoreError::SessionBindingMismatch {
                    bound_session_id: bound_session_id.to_string(),
                    attempted_session_id: commit.session_id.clone(),
                });
            }
            if let Some(completed) = &commit.turn_commit {
                if completed.session_id != commit.session_id {
                    return Err(StoreError::RuntimeTurnCommitConflict {
                        session_id: completed.session_id.clone(),
                        turn_id: completed.turn_id.clone(),
                    });
                }
                let prior = optional_row(
                    &conn,
                    "SELECT turn_commit_hash, result_json FROM runtime_turn_commits
                     WHERE session_id = ?1 AND turn_id = ?2",
                    params![completed.session_id.as_str(), completed.turn_id.as_str()],
                )
                .await
                .map_err(turso_error)?;
                if let Some(row) = prior {
                    let turn_commit_hash = row_string(&row, 0).map_err(turso_error)?;
                    let result_json = row_string(&row, 1).map_err(turso_error)?;
                    if turn_commit_hash == completed.turn_commit_hash {
                        let result: RuntimeCommitResult =
                            serde_json::from_str(&result_json).map_err(|err| {
                                StoreError::Backend(format!(
                                    "failed to decode runtime turn commit result: {err}"
                                ))
                            })?;
                        return Ok(result);
                    }
                    return Err(StoreError::RuntimeTurnCommitConflict {
                        session_id: completed.session_id.clone(),
                        turn_id: completed.turn_id.clone(),
                    });
                }
            }
            let actual_revision = existing.as_ref().map_or(0, |meta| meta.head_revision);
            if commit.expected_head_revision.is_some()
                && commit.expected_head_revision != Some(actual_revision)
            {
                return Err(StoreError::HeadRevisionConflict {
                    expected: commit.expected_head_revision,
                    actual: actual_revision,
                });
            }
            for completed in &commit.completed_queue_claims {
                if completed.session_id != commit.session_id {
                    return Err(StoreError::QueuedWorkClaimExpired {
                        session_id: completed.session_id.clone(),
                        claim_id: completed.claim_id.clone(),
                    });
                }
                ensure_queued_work_completion_conn(&conn, completed).await?;
            }

            let stored_checkpoint =
                Self::put_checkpoint_conn(&conn, &commit.checkpoint, blob_profile)
                    .await
                    .map_err(turso_error)?;

            for entry in &commit.usage_deltas {
                conn.execute(
                    "INSERT INTO usage_deltas (
                        source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        entry.source.as_str(),
                        entry.model.as_str(),
                        entry.usage.input_tokens,
                        entry.usage.output_tokens,
                        entry.usage.cached_input_tokens,
                        entry.usage.reasoning_tokens,
                    ],
                )
                .await
                .map_err(turso_error)?;
            }

            let leaf_node_id = match &commit.graph {
                GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
                GraphCommitDelta::Append {
                    nodes,
                    leaf_node_id,
                } => {
                    for node in nodes {
                        let node_json = encode_json(node);
                        conn.execute(
                            "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                            params![node.node_id.as_str(), node_json],
                        )
                        .await
                        .map_err(turso_error)?;
                    }
                    leaf_node_id.clone()
                }
                GraphCommitDelta::ReplaceFull(graph) => {
                    conn.execute("DELETE FROM graph_nodes", ())
                        .await
                        .map_err(turso_error)?;
                    for node in &graph.nodes {
                        let node_json = encode_json(node);
                        conn.execute(
                            "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                            params![node.node_id.as_str(), node_json],
                        )
                        .await
                        .map_err(turso_error)?;
                    }
                    graph.leaf_node_id.clone()
                }
            };
            let row = required_row(
                &conn,
                "SELECT COUNT(*) FROM graph_nodes WHERE tombstoned = 0",
                (),
            )
            .await
            .map_err(turso_error)?;
            let graph_node_count = row_i64(&row, 0).map_err(turso_error)? as usize;
            let next_revision = actual_revision + 1;
            let meta = SessionHeadMeta {
                session_id: commit.session_id.clone(),
                head_revision: next_revision,
                config: commit.config.clone(),
                agent_frames: commit.agent_frames.clone(),
                current_agent_frame_id: commit.current_agent_frame_id.clone(),
                checkpoint_ref: Some(stored_checkpoint.checkpoint_ref.clone()),
                leaf_node_id,
                graph_node_count,
                token_ledger: Vec::new(),
            };
            conn.execute(
                "INSERT OR REPLACE INTO session_head (singleton, session_id, head_json, head_revision)
                 VALUES (1, ?1, ?2, ?3)",
                params![
                    meta.session_id.as_str(),
                    encode_json(&meta),
                    meta.head_revision as i64
                ],
            )
            .await
            .map_err(turso_error)?;
            for completed in &commit.completed_queue_claims {
                for batch_id in &completed.batch_ids {
                    conn.execute(
                        "DELETE FROM queued_work_batches
                         WHERE session_id = ?1
                           AND batch_id = ?2
                           AND claim_id = ?3
                           AND claim_token = ?4",
                        params![
                            completed.session_id.as_str(),
                            batch_id.as_str(),
                            completed.claim_id.as_str(),
                            completed.lease_token.as_str()
                        ],
                    )
                    .await
                    .map_err(turso_error)?;
                }
            }
            if !commit.committed_attachment_ids.is_empty() {
                let now = current_epoch_ms() as i64;
                for id in &commit.committed_attachment_ids {
                    conn.execute(
                        "UPDATE attachment_manifest
                         SET committed_at_ms = COALESCE(committed_at_ms, ?1)
                         WHERE attachment_id = ?2 AND session_id = ?3",
                        params![now, id.as_str(), commit.session_id.as_str()],
                    )
                    .await
                    .map_err(turso_error)?;
                }
            }
            let result = RuntimeCommitResult {
                head_revision: next_revision,
                checkpoint_ref: stored_checkpoint.checkpoint_ref,
                manifest: stored_checkpoint.manifest,
            };
            if let Some(completed) = &commit.turn_commit {
                conn.execute(
                    "INSERT INTO runtime_turn_commits (
                        session_id, turn_id, turn_commit_hash, result_json, committed_at_ms
                     )
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        completed.session_id.as_str(),
                        completed.turn_id.as_str(),
                        completed.turn_commit_hash.as_str(),
                        encode_json(&result),
                        current_epoch_ms() as i64
                    ],
                )
                .await
                .map_err(turso_error)?;
            }
            Ok(result)
        }
        .await;
        match result {
            Ok(result) => {
                conn.execute("COMMIT", ()).await.map_err(turso_error)?;
                drop(conn);
                self.maybe_auto_gc().await;
                Ok(result)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn enqueue_queued_work(
        &self,
        batch: QueuedWorkBatchDraft,
    ) -> Result<QueuedWorkBatch, StoreError> {
        let nonce = self.commit_count.fetch_add(1, AtomicOrdering::Relaxed);
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(turso_error)?;
        let result = async {
            if let Some(source_key) = batch.source_key.as_deref() {
                let existing_id = optional_row(
                    &conn,
                    "SELECT batch_id
                     FROM queued_work_batches
                     WHERE session_id = ?1 AND source_key = ?2",
                    params![batch.session_id.as_str(), source_key],
                )
                .await
                .map_err(turso_error)?
                .map(|row| row_string(&row, 0).map_err(turso_error))
                .transpose()?;
                if let Some(batch_id) = existing_id {
                    let existing = load_queued_batch_by_id_conn(&conn, &batch_id)
                        .await?
                        .ok_or_else(|| {
                            StoreError::Backend("queued work source row disappeared".to_string())
                        })?;
                    return Ok(existing);
                }
            }
            let now = current_epoch_ms();
            let batch_id = format!(
                "qwb:{:x}",
                Sha256::digest(
                    format!("{}:{:?}:{now}:{nonce}", batch.session_id, batch.source_key).as_bytes()
                )
            );
            conn.execute(
                "INSERT INTO queued_work_batches (
                    batch_id, session_id, source_key, delivery_policy, slot_policy,
                    merge_key_json, available_at_ms, enqueued_at_ms
                 )
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    batch_id.as_str(),
                    batch.session_id.as_str(),
                    batch.source_key.as_deref(),
                    batch.delivery_policy.as_str(),
                    batch.slot_policy.as_str(),
                    encode_json(&batch.merge_key),
                    batch.available_at_ms as i64,
                    now as i64,
                ],
            )
            .await
            .map_err(turso_error)?;
            for (index, payload) in batch.payloads.iter().enumerate() {
                let item_id = format!("{batch_id}:item:{index}");
                conn.execute(
                    "INSERT INTO queued_work_items (batch_id, item_index, item_id, payload_json)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![
                        batch_id.as_str(),
                        index as i64,
                        item_id,
                        encode_json(payload)
                    ],
                )
                .await
                .map_err(turso_error)?;
            }
            load_queued_batch_by_id_conn(&conn, &batch_id)
                .await?
                .ok_or_else(|| StoreError::Backend("queued work insert disappeared".to_string()))
        }
        .await;
        match result {
            Ok(batch) => {
                conn.execute("COMMIT", ()).await.map_err(turso_error)?;
                Ok(batch)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn claim_ready_queued_work(
        &self,
        session_id: &str,
        owner_id: &str,
        boundary: QueuedWorkClaimBoundary,
        lease_ttl_ms: u64,
        max_batches: usize,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        if max_batches == 0 {
            return Ok(None);
        }
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(turso_error)?;
        let result = async {
            let now = current_epoch_ms();
            let candidate_rows = collect_rows(
                &conn,
                "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                        slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                        claim_fencing_token
                 FROM queued_work_batches
                 WHERE session_id = ?1
                   AND available_at_ms <= ?2
                   AND (claim_token IS NULL OR claim_expires_at_ms <= ?2)
                 ORDER BY enqueue_seq ASC
                 LIMIT ?3",
                params![
                    session_id,
                    now as i64,
                    (max_batches as i64).saturating_add(32)
                ],
            )
            .await
            .map_err(turso_error)?
            .into_iter()
            .map(|row| queued_batch_row_from_sql(&row).map_err(turso_error))
            .collect::<Result<Vec<_>, _>>()?;
            let Some(first_row) = candidate_rows.first() else {
                return Ok(TxOutcome::Commit(None));
            };
            let first_delivery = decode_delivery_policy(first_row.delivery_policy.clone())?;
            if boundary == QueuedWorkClaimBoundary::ActiveTurnCheckpoint
                && first_delivery != DeliveryPolicy::EarliestSafeBoundary
            {
                return Ok(TxOutcome::Commit(None));
            }
            let first_slot = decode_slot_policy(first_row.slot_policy.clone())?;
            let first_merge_key = decode_merge_key(first_row.merge_key_json.clone())?;
            let mut selected = Vec::new();
            for row in candidate_rows {
                if selected.len() >= max_batches {
                    break;
                }
                let delivery = decode_delivery_policy(row.delivery_policy.clone())?;
                let slot = decode_slot_policy(row.slot_policy.clone())?;
                let merge_key = decode_merge_key(row.merge_key_json.clone())?;
                if selected.is_empty() {
                    selected.push(row);
                    if first_slot == SlotPolicy::Exclusive {
                        break;
                    }
                    continue;
                }
                if first_slot != SlotPolicy::Join
                    || slot != SlotPolicy::Join
                    || delivery != first_delivery
                    || merge_key != first_merge_key
                {
                    break;
                }
                selected.push(row);
            }
            let Some(first) = selected.first() else {
                return Ok(TxOutcome::Commit(None));
            };
            let fencing_token = first.claim_fencing_token.saturating_add(1);
            let claim_id = format!("qwc:{}:{fencing_token}", first.enqueue_seq);
            let lease_token = format!(
                "{:x}",
                Sha256::digest(format!("{session_id}:{owner_id}:{claim_id}:{now}").as_bytes())
            );
            let expires_at = now.saturating_add(lease_ttl_ms);
            for row in &selected {
                let claimed = conn
                    .execute(
                        "UPDATE queued_work_batches
                         SET claim_id = ?3,
                             claim_owner_id = ?4,
                             claim_token = ?5,
                             claim_fencing_token = claim_fencing_token + 1,
                             claim_claimed_at_ms = ?6,
                             claim_expires_at_ms = ?7
                         WHERE session_id = ?1
                           AND batch_id = ?2
                           AND (claim_token IS NULL OR claim_expires_at_ms <= ?6)",
                        params![
                            session_id,
                            row.batch_id.as_str(),
                            claim_id.as_str(),
                            owner_id,
                            lease_token.as_str(),
                            now as i64,
                            expires_at as i64
                        ],
                    )
                    .await
                    .map_err(turso_error)?;
                if claimed == 0 {
                    return Ok(TxOutcome::Rollback(None));
                }
            }
            let mut batches = Vec::new();
            for row in selected {
                batches.push(queued_work_batch_from_conn(&conn, row).await?);
            }
            Ok(TxOutcome::Commit(Some(QueuedWorkClaim {
                session_id: session_id.to_string(),
                claim_id,
                owner_id: owner_id.to_string(),
                lease_token,
                fencing_token,
                claimed_at_epoch_ms: now,
                expires_at_epoch_ms: expires_at,
                batches,
            })))
        }
        .await;
        match result {
            Ok(TxOutcome::Commit(value)) => {
                conn.execute("COMMIT", ()).await.map_err(turso_error)?;
                Ok(value)
            }
            Ok(TxOutcome::Rollback(value)) => {
                conn.execute("ROLLBACK", ()).await.map_err(turso_error)?;
                Ok(value)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn renew_queued_work_claim(
        &self,
        claim: &QueuedWorkClaim,
        lease_ttl_ms: u64,
    ) -> Result<QueuedWorkClaim, StoreError> {
        let conn = self.conn.lock().await;
        let now = current_epoch_ms();
        let expires_at = now.saturating_add(lease_ttl_ms);
        let changed = conn
            .execute(
                "UPDATE queued_work_batches
                 SET claim_expires_at_ms = ?4
                 WHERE session_id = ?1 AND claim_id = ?2 AND claim_token = ?3",
                params![
                    claim.session_id.as_str(),
                    claim.claim_id.as_str(),
                    claim.lease_token.as_str(),
                    expires_at as i64
                ],
            )
            .await
            .map_err(turso_error)?;
        if changed as usize != claim.batches.len() {
            return Err(StoreError::QueuedWorkClaimExpired {
                session_id: claim.session_id.clone(),
                claim_id: claim.claim_id.clone(),
            });
        }
        Ok(QueuedWorkClaim {
            expires_at_epoch_ms: expires_at,
            ..claim.clone()
        })
    }

    async fn abandon_queued_work_claim(&self, claim: &QueuedWorkClaim) -> Result<(), StoreError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE queued_work_batches
             SET claim_id = NULL,
                 claim_owner_id = NULL,
                 claim_token = NULL,
                 claim_claimed_at_ms = 0,
                 claim_expires_at_ms = 0
             WHERE session_id = ?1 AND claim_id = ?2 AND claim_token = ?3",
            params![
                claim.session_id.as_str(),
                claim.claim_id.as_str(),
                claim.lease_token.as_str()
            ],
        )
        .await
        .map_err(turso_error)?;
        Ok(())
    }

    async fn cancel_queued_work_batch(
        &self,
        session_id: &str,
        batch_id: &str,
    ) -> Result<Option<QueuedWorkBatch>, StoreError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(turso_error)?;
        let result = async {
            let row = optional_row(
                &conn,
                "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                        slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                        claim_fencing_token
                 FROM queued_work_batches
                 WHERE session_id = ?1
                   AND batch_id = ?2
                   AND (claim_token IS NULL OR claim_expires_at_ms <= ?3)",
                params![session_id, batch_id, current_epoch_ms() as i64],
            )
            .await
            .map_err(turso_error)?
            .map(|row| queued_batch_row_from_sql(&row).map_err(turso_error))
            .transpose()?;
            let Some(row) = row else {
                return Ok(None);
            };
            let batch = queued_work_batch_from_conn(&conn, row).await?;
            conn.execute(
                "DELETE FROM queued_work_batches
                 WHERE session_id = ?1
                   AND batch_id = ?2
                   AND (claim_token IS NULL OR claim_expires_at_ms <= ?3)",
                params![session_id, batch_id, current_epoch_ms() as i64],
            )
            .await
            .map_err(turso_error)?;
            Ok(Some(batch))
        }
        .await;
        match result {
            Ok(batch) => {
                conn.execute("COMMIT", ()).await.map_err(turso_error)?;
                Ok(batch)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn list_queued_work(&self, session_id: &str) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let conn = self.conn.lock().await;
        let rows = collect_rows(
            &conn,
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token
             FROM queued_work_batches
             WHERE session_id = ?1
             ORDER BY enqueue_seq ASC",
            params![session_id],
        )
        .await
        .map_err(turso_error)?
        .into_iter()
        .map(|row| queued_batch_row_from_sql(&row).map_err(turso_error))
        .collect::<Result<Vec<_>, _>>()?;
        let mut batches = Vec::new();
        for row in rows {
            batches.push(queued_work_batch_from_conn(&conn, row).await?);
        }
        Ok(batches)
    }

    async fn list_pending_queued_work(
        &self,
        session_id: &str,
    ) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let conn = self.conn.lock().await;
        let now = current_epoch_ms();
        let rows = collect_rows(
            &conn,
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token
             FROM queued_work_batches
             WHERE session_id = ?1
               AND (claim_token IS NULL OR claim_expires_at_ms <= ?2)
             ORDER BY enqueue_seq ASC",
            params![session_id, now as i64],
        )
        .await
        .map_err(turso_error)?
        .into_iter()
        .map(|row| queued_batch_row_from_sql(&row).map_err(turso_error))
        .collect::<Result<Vec<_>, _>>()?;
        let mut batches = Vec::new();
        for row in rows {
            batches.push(queued_work_batch_from_conn(&conn, row).await?);
        }
        Ok(batches)
    }

    async fn save_session_meta(&self, meta: SessionMeta) -> Result<(), StoreError> {
        Store::save_session_meta(self, meta).await;
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<SessionMeta>, StoreError> {
        Ok(Store::load_session_meta(self).await)
    }

    async fn tombstone_nodes(&self, ids: &[String]) -> Result<(), StoreError> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(turso_error)?;
        let result = async {
            for id in ids {
                conn.execute(
                    "UPDATE graph_nodes SET tombstoned = 1 WHERE node_id = ?1",
                    params![id.as_str()],
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
    }

    async fn vacuum(&self) -> Result<VacuumReport, StoreError> {
        let conn = self.conn.lock().await;
        let removed = conn
            .execute("DELETE FROM graph_nodes WHERE tombstoned = 1", ())
            .await
            .map_err(turso_error)?;
        Ok(VacuumReport {
            removed_node_count: removed as usize,
        })
    }

    async fn gc_unreachable(&self) -> Result<GcReport, StoreError> {
        Ok(Store::gc_unreachable(self).await)
    }
}
