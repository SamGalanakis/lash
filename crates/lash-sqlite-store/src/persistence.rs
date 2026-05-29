use super::*;

#[async_trait::async_trait]
impl RuntimePersistence for Store {
    async fn load_session(
        &self,
        scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let Some(meta) = try_load_session_head_meta_from_conn(&conn)? else {
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
                Self::load_session_graph_from_conn(&conn, meta.leaf_node_id.clone())
            }
            SessionReadScope::ActivePath { .. } => {
                Self::load_active_path_session_graph_from_conn(&conn, leaf_node_id.clone())
                    .map_err(sqlite_error)?
            }
        };
        graph.set_leaf_node_id(leaf_node_id);
        let checkpoint = meta
            .checkpoint_ref
            .as_ref()
            .and_then(|blob_ref| Self::get_checkpoint_conn(&conn, blob_ref));
        Ok(Some(PersistedSessionRead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            config: meta.config,
            agent_frames: meta.agent_frames,
            current_agent_frame_id: meta.current_agent_frame_id,
            graph,
            checkpoint_ref: meta.checkpoint_ref,
            checkpoint,
            token_ledger: merge_token_ledger_entries(Self::load_usage_deltas_conn(&conn)),
        }))
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<lash_core::SessionNodeRecord>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let row: Option<String> = conn
            .query_row(
                "SELECT node_json FROM graph_nodes WHERE node_id = ?1 AND tombstoned = 0",
                params![node_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_error)?;
        Ok(row.and_then(|json| serde_json::from_str(&json).ok()))
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let result = {
            let mut conn = self.conn.lock().unwrap();
            let tx = conn.transaction().map_err(sqlite_error)?;
            let existing = try_load_session_head_meta_from_conn(&tx)?;
            if let Some(bound_session_id) = existing.as_ref().map(|meta| meta.session_id.as_str())
                && bound_session_id != commit.session_id
            {
                return Err(StoreError::SessionBindingMismatch {
                    bound_session_id: bound_session_id.to_string(),
                    attempted_session_id: commit.session_id,
                });
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
            if let Some(completed) = &commit.completed_turn {
                if completed.session_id != commit.session_id {
                    return Err(StoreError::RuntimeTurnLeaseExpired {
                        session_id: completed.session_id.clone(),
                        turn_id: completed.turn_id.clone(),
                    });
                }
                ensure_runtime_turn_completion_conn(&tx, completed)?;
            }
            for completed in &commit.completed_queue_claims {
                if completed.session_id != commit.session_id {
                    return Err(StoreError::QueuedWorkClaimExpired {
                        session_id: completed.session_id.clone(),
                        claim_id: completed.claim_id.clone(),
                    });
                }
                ensure_queued_work_completion_conn(&tx, completed)?;
            }

            let stored_checkpoint =
                Self::put_checkpoint_conn(&tx, &commit.checkpoint, self.options.blob_profile)
                    .map_err(sqlite_error)?;

            if !commit.usage_deltas.is_empty() {
                {
                    let mut stmt = tx
                        .prepare(
                            "INSERT INTO usage_deltas (
                                source, model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
                            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        )
                        .map_err(sqlite_error)?;
                    for entry in &commit.usage_deltas {
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
                }
            }

            let leaf_node_id = match &commit.graph {
                GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
                GraphCommitDelta::Append {
                    nodes,
                    leaf_node_id,
                } => {
                    for node in nodes {
                        let node_json = encode_json(node);
                        tx.execute(
                            "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                            params![node.node_id, node_json],
                        )
                        .map_err(sqlite_error)?;
                    }
                    leaf_node_id.clone()
                }
                GraphCommitDelta::ReplaceFull(graph) => {
                    tx.execute("DELETE FROM graph_nodes", [])
                        .map_err(sqlite_error)?;
                    for node in &graph.nodes {
                        let node_json = encode_json(node);
                        tx.execute(
                            "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                            params![node.node_id, node_json],
                        )
                        .map_err(sqlite_error)?;
                    }
                    graph.leaf_node_id.clone()
                }
            };
            let graph_node_count: usize = tx
                .query_row(
                    "SELECT COUNT(*) FROM graph_nodes WHERE tombstoned = 0",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(sqlite_error)? as usize;
            let next_revision = actual_revision + 1;
            let meta = SessionHeadMeta {
                session_id: commit.session_id.clone(),
                head_revision: next_revision,
                config: commit.config,
                agent_frames: commit.agent_frames,
                current_agent_frame_id: commit.current_agent_frame_id,
                checkpoint_ref: Some(stored_checkpoint.checkpoint_ref.clone()),
                leaf_node_id,
                graph_node_count,
                token_ledger: Vec::new(),
            };
            tx.execute(
                "INSERT OR REPLACE INTO session_head (singleton, session_id, head_json, head_revision)
                 VALUES (1, ?1, ?2, ?3)",
                params![
                    meta.session_id,
                    encode_json(&meta),
                    meta.head_revision as i64
                ],
            )
            .map_err(sqlite_error)?;
            if let Some(completed) = &commit.completed_turn {
                tx.execute(
                    "DELETE FROM runtime_effect_journal
                     WHERE session_id = ?1 AND turn_id = ?2
                       AND EXISTS (
                         SELECT 1 FROM runtime_turn_checkpoints
                         WHERE session_id = ?1 AND turn_id = ?2 AND lease_token = ?3
                       )",
                    params![
                        completed.session_id,
                        completed.turn_id,
                        completed.lease_token
                    ],
                )
                .map_err(sqlite_error)?;
                tx.execute(
                    "DELETE FROM runtime_turn_checkpoints
                     WHERE session_id = ?1 AND turn_id = ?2 AND lease_token = ?3",
                    params![
                        completed.session_id,
                        completed.turn_id,
                        completed.lease_token
                    ],
                )
                .map_err(sqlite_error)?;
            }
            for completed in &commit.completed_queue_claims {
                for batch_id in &completed.batch_ids {
                    tx.execute(
                        "DELETE FROM queued_work_batches
                         WHERE session_id = ?1
                           AND batch_id = ?2
                           AND claim_id = ?3
                           AND claim_token = ?4",
                        params![
                            completed.session_id,
                            batch_id,
                            completed.claim_id,
                            completed.lease_token
                        ],
                    )
                    .map_err(sqlite_error)?;
                }
            }
            if !commit.committed_attachment_ids.is_empty() {
                let now = current_epoch_ms() as i64;
                let mut stmt = tx
                    .prepare(
                        "UPDATE attachment_manifest
                         SET committed_at_ms = COALESCE(committed_at_ms, ?1)
                         WHERE attachment_id = ?2 AND session_id = ?3",
                    )
                    .map_err(sqlite_error)?;
                for id in &commit.committed_attachment_ids {
                    stmt.execute(params![now, id.as_str(), commit.session_id])
                        .map_err(sqlite_error)?;
                }
            }
            tx.commit().map_err(sqlite_error)?;
            RuntimeCommitResult {
                head_revision: next_revision,
                checkpoint_ref: stored_checkpoint.checkpoint_ref,
                manifest: stored_checkpoint.manifest,
            }
        };
        self.maybe_auto_gc();
        Ok(result)
    }

    async fn enqueue_queued_work(
        &self,
        batch: QueuedWorkBatchDraft,
    ) -> Result<QueuedWorkBatch, StoreError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sqlite_error)?;
        if let Some(source_key) = batch.source_key.as_deref() {
            let existing_id: Option<String> = tx
                .query_row(
                    "SELECT batch_id
                     FROM queued_work_batches
                     WHERE session_id = ?1 AND source_key = ?2",
                    params![batch.session_id, source_key],
                    |row| row.get(0),
                )
                .optional()
                .map_err(sqlite_error)?;
            if let Some(batch_id) = existing_id {
                let existing = load_queued_batch_by_id_conn(&tx, &batch_id)?.ok_or_else(|| {
                    StoreError::Backend("queued work source row disappeared".to_string())
                })?;
                tx.commit().map_err(sqlite_error)?;
                return Ok(existing);
            }
        }
        let now = current_epoch_ms();
        let nonce = self.commit_count.fetch_add(1, AtomicOrdering::Relaxed);
        let batch_id = format!(
            "qwb:{:x}",
            Sha256::digest(
                format!("{}:{:?}:{now}:{nonce}", batch.session_id, batch.source_key).as_bytes()
            )
        );
        tx.execute(
            "INSERT INTO queued_work_batches (
                batch_id, session_id, source_key, delivery_policy, slot_policy,
                merge_key_json, available_at_ms, enqueued_at_ms
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                &batch_id,
                &batch.session_id,
                batch.source_key.as_deref(),
                batch.delivery_policy.as_str(),
                batch.slot_policy.as_str(),
                encode_json(&batch.merge_key),
                batch.available_at_ms as i64,
                now as i64,
            ],
        )
        .map_err(sqlite_error)?;
        for (index, payload) in batch.payloads.iter().enumerate() {
            let item_id = format!("{batch_id}:item:{index}");
            tx.execute(
                "INSERT INTO queued_work_items (batch_id, item_index, item_id, payload_json)
                 VALUES (?1, ?2, ?3, ?4)",
                params![&batch_id, index as i64, item_id, encode_json(payload)],
            )
            .map_err(sqlite_error)?;
        }
        let enqueued = load_queued_batch_by_id_conn(&tx, &batch_id)?
            .ok_or_else(|| StoreError::Backend("queued work insert disappeared".to_string()))?;
        tx.commit().map_err(sqlite_error)?;
        Ok(enqueued)
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
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sqlite_error)?;
        let now = current_epoch_ms();
        let candidate_rows = {
            let mut stmt = tx
                .prepare(
                    "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                            slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                            claim_fencing_token
                     FROM queued_work_batches
                     WHERE session_id = ?1
                       AND available_at_ms <= ?2
                       AND (claim_token IS NULL OR claim_expires_at_ms <= ?2)
                     ORDER BY enqueue_seq ASC
                     LIMIT ?3",
                )
                .map_err(sqlite_error)?;
            let rows = stmt
                .query_map(
                    params![
                        session_id,
                        now as i64,
                        (max_batches as i64).saturating_add(32)
                    ],
                    queued_batch_row_from_sql,
                )
                .map_err(sqlite_error)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
        };
        let Some(first_row) = candidate_rows.first() else {
            tx.commit().map_err(sqlite_error)?;
            return Ok(None);
        };
        let first_delivery = decode_delivery_policy(first_row.delivery_policy.clone())?;
        if boundary == QueuedWorkClaimBoundary::ActiveTurnCheckpoint
            && first_delivery != DeliveryPolicy::EarliestSafeBoundary
        {
            tx.commit().map_err(sqlite_error)?;
            return Ok(None);
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
            tx.commit().map_err(sqlite_error)?;
            return Ok(None);
        };
        let fencing_token = first.claim_fencing_token.saturating_add(1);
        let claim_id = format!("qwc:{}:{fencing_token}", first.enqueue_seq);
        let lease_token = format!(
            "{:x}",
            Sha256::digest(format!("{session_id}:{owner_id}:{claim_id}:{now}").as_bytes())
        );
        let expires_at = now.saturating_add(lease_ttl_ms);
        for row in &selected {
            tx.execute(
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
                    row.batch_id,
                    claim_id,
                    owner_id,
                    lease_token,
                    now as i64,
                    expires_at as i64
                ],
            )
            .map_err(sqlite_error)?;
        }
        let mut batches = Vec::new();
        for row in selected {
            batches.push(queued_work_batch_from_conn(&tx, row)?);
        }
        tx.commit().map_err(sqlite_error)?;
        Ok(Some(QueuedWorkClaim {
            session_id: session_id.to_string(),
            claim_id,
            owner_id: owner_id.to_string(),
            lease_token,
            fencing_token,
            claimed_at_epoch_ms: now,
            expires_at_epoch_ms: expires_at,
            batches,
        }))
    }

    async fn renew_queued_work_claim(
        &self,
        claim: &QueuedWorkClaim,
        lease_ttl_ms: u64,
    ) -> Result<QueuedWorkClaim, StoreError> {
        let conn = self.conn.lock().unwrap();
        let now = current_epoch_ms();
        let expires_at = now.saturating_add(lease_ttl_ms);
        let changed = conn
            .execute(
                "UPDATE queued_work_batches
                 SET claim_expires_at_ms = ?4
                 WHERE session_id = ?1 AND claim_id = ?2 AND claim_token = ?3",
                params![
                    claim.session_id,
                    claim.claim_id,
                    claim.lease_token,
                    expires_at as i64
                ],
            )
            .map_err(sqlite_error)?;
        if changed != claim.batches.len() {
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
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE queued_work_batches
             SET claim_id = NULL,
                 claim_owner_id = NULL,
                 claim_token = NULL,
                 claim_claimed_at_ms = 0,
                 claim_expires_at_ms = 0
             WHERE session_id = ?1 AND claim_id = ?2 AND claim_token = ?3",
            params![claim.session_id, claim.claim_id, claim.lease_token],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn list_queued_work(&self, session_id: &str) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let rows = {
            let mut stmt = conn
                .prepare(
                    "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                            slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                            claim_fencing_token
                     FROM queued_work_batches
                     WHERE session_id = ?1
                     ORDER BY enqueue_seq ASC",
                )
                .map_err(sqlite_error)?;
            let rows = stmt
                .query_map(params![session_id], queued_batch_row_from_sql)
                .map_err(sqlite_error)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
        };
        rows.into_iter()
            .map(|row| queued_work_batch_from_conn(&conn, row))
            .collect()
    }

    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sqlite_error)?;
        let now = current_epoch_ms();
        let current =
            load_runtime_turn_lease_from_conn(&tx, session_id, turn_id).map_err(sqlite_error)?;
        if let Some(current) = current
            && current.expires_at_epoch_ms > now
            && current.owner_id != owner_id
        {
            return Err(StoreError::RuntimeTurnLeaseConflict {
                session_id: session_id.to_string(),
                turn_id: turn_id.to_string(),
                owner_id: current.owner_id,
                expires_at_epoch_ms: current.expires_at_epoch_ms,
            });
        }
        let fencing_token: u64 = tx
            .query_row(
                "SELECT lease_fencing_token FROM runtime_turn_checkpoints
                 WHERE session_id = ?1 AND turn_id = ?2",
                params![session_id, turn_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(sqlite_error)?
            .unwrap_or(0) as u64
            + 1;
        let lease = RuntimeTurnLease {
            schema_version: RUNTIME_TURN_LEASE_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!(
                "{:x}",
                Sha256::digest(
                    format!("{session_id}:{turn_id}:{owner_id}:{now}:{fencing_token}").as_bytes()
                )
            ),
            fencing_token,
            claimed_at_epoch_ms: now,
            expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
        };
        tx.execute(
            "INSERT INTO runtime_turn_checkpoints (
                session_id, turn_id, lease_owner_id, lease_token, lease_fencing_token,
                lease_claimed_at_ms, lease_expires_at_ms, updated_at_ms
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(session_id, turn_id) DO UPDATE SET
                lease_owner_id = excluded.lease_owner_id,
                lease_token = excluded.lease_token,
                lease_fencing_token = excluded.lease_fencing_token,
                lease_claimed_at_ms = excluded.lease_claimed_at_ms,
                lease_expires_at_ms = excluded.lease_expires_at_ms,
                updated_at_ms = excluded.updated_at_ms",
            params![
                lease.session_id,
                lease.turn_id,
                lease.owner_id,
                lease.lease_token,
                lease.fencing_token as i64,
                lease.claimed_at_epoch_ms as i64,
                lease.expires_at_epoch_ms as i64,
                now as i64
            ],
        )
        .map_err(sqlite_error)?;
        tx.commit().map_err(sqlite_error)?;
        Ok(lease)
    }

    async fn renew_runtime_turn_lease(
        &self,
        lease: &RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> Result<RuntimeTurnLease, StoreError> {
        let conn = self.conn.lock().unwrap();
        ensure_runtime_turn_lease_conn(&conn, lease)?;
        let now = current_epoch_ms();
        let renewed = RuntimeTurnLease {
            expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
            ..lease.clone()
        };
        conn.execute(
            "UPDATE runtime_turn_checkpoints
             SET lease_expires_at_ms = ?3
             WHERE session_id = ?1 AND turn_id = ?2 AND lease_token = ?4",
            params![
                renewed.session_id,
                renewed.turn_id,
                renewed.expires_at_epoch_ms as i64,
                renewed.lease_token
            ],
        )
        .map_err(sqlite_error)?;
        Ok(renewed)
    }

    async fn abandon_runtime_turn_lease(&self, lease: &RuntimeTurnLease) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE runtime_turn_checkpoints
             SET lease_owner_id = NULL,
                 lease_token = NULL,
                 lease_claimed_at_ms = 0,
                 lease_expires_at_ms = 0,
                 updated_at_ms = ?6
             WHERE session_id = ?1
               AND turn_id = ?2
               AND lease_owner_id = ?3
               AND lease_token = ?4
               AND lease_fencing_token = ?5",
            params![
                lease.session_id,
                lease.turn_id,
                lease.owner_id,
                lease.lease_token,
                lease.fencing_token as i64,
                current_epoch_ms() as i64
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn save_runtime_turn_checkpoint(
        &self,
        lease: &RuntimeTurnLease,
        checkpoint: RuntimeTurnCheckpoint,
    ) -> Result<(), StoreError> {
        if checkpoint.session_id != lease.session_id || checkpoint.turn_id != lease.turn_id {
            return Err(StoreError::RuntimeTurnLeaseExpired {
                session_id: checkpoint.session_id,
                turn_id: checkpoint.turn_id,
            });
        }
        let conn = self.conn.lock().unwrap();
        ensure_runtime_turn_lease_conn(&conn, lease)?;
        let actual_hash = lash_core::runtime_turn_checkpoint_hash(&checkpoint.checkpoint)?;
        if actual_hash != checkpoint.checkpoint_hash {
            return Err(StoreError::RuntimeTurnCheckpointHashMismatch {
                session_id: checkpoint.session_id,
                turn_id: checkpoint.turn_id,
            });
        }
        let checkpoint_json = encode_json(&checkpoint);
        conn.execute(
            "UPDATE runtime_turn_checkpoints
             SET checkpoint_json = ?3, checkpoint_hash = ?4, updated_at_ms = ?5
             WHERE session_id = ?1 AND turn_id = ?2 AND lease_token = ?6",
            params![
                checkpoint.session_id,
                checkpoint.turn_id,
                checkpoint_json,
                checkpoint.checkpoint_hash,
                checkpoint.updated_at_epoch_ms as i64,
                lease.lease_token
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn load_runtime_turn_checkpoint(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<RuntimeTurnCheckpoint>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let checkpoint_json: Option<String> = conn
            .query_row(
                "SELECT checkpoint_json FROM runtime_turn_checkpoints
                 WHERE session_id = ?1 AND turn_id = ?2 AND checkpoint_json IS NOT NULL",
                params![session_id, turn_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_error)?;
        let Some(checkpoint_json) = checkpoint_json else {
            return Ok(None);
        };
        let checkpoint: RuntimeTurnCheckpoint =
            serde_json::from_str(&checkpoint_json).map_err(|err| {
                StoreError::Backend(format!("failed to decode runtime turn checkpoint: {err}"))
            })?;
        ensure_supported_schema_version(
            "RuntimeTurnCheckpoint",
            checkpoint.schema_version,
            RUNTIME_TURN_CHECKPOINT_SCHEMA_VERSION,
        )?;
        let actual_hash = lash_core::runtime_turn_checkpoint_hash(&checkpoint.checkpoint)?;
        if checkpoint.checkpoint_hash != actual_hash {
            return Err(StoreError::RuntimeTurnCheckpointHashMismatch {
                session_id: session_id.to_string(),
                turn_id: turn_id.to_string(),
            });
        }
        Ok(Some(checkpoint))
    }

    async fn save_runtime_effect_outcome(
        &self,
        lease: &RuntimeTurnLease,
        record: RuntimeEffectJournalRecord,
    ) -> Result<(), StoreError> {
        if record.session_id != lease.session_id || record.turn_id != lease.turn_id {
            return Err(StoreError::RuntimeTurnLeaseExpired {
                session_id: record.session_id,
                turn_id: record.turn_id,
            });
        }
        let conn = self.conn.lock().unwrap();
        ensure_runtime_turn_lease_conn(&conn, lease)?;
        conn.execute(
            "INSERT INTO runtime_effect_journal (
                session_id, turn_id, idempotency_key, envelope_hash, effect_kind,
                outcome_json, created_at_ms
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(session_id, turn_id, idempotency_key) DO UPDATE SET
                envelope_hash = excluded.envelope_hash,
                effect_kind = excluded.effect_kind,
                outcome_json = excluded.outcome_json,
                created_at_ms = excluded.created_at_ms",
            params![
                record.session_id,
                record.turn_id,
                record.replay_key,
                record.envelope_hash,
                record.effect_kind.as_str(),
                encode_json(&record),
                record.created_at_epoch_ms as i64
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        replay_key: &str,
    ) -> Result<Option<RuntimeEffectJournalRecord>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let row: Option<String> = conn
            .query_row(
                "SELECT outcome_json FROM runtime_effect_journal
                 WHERE session_id = ?1 AND turn_id = ?2 AND idempotency_key = ?3",
                params![session_id, turn_id, replay_key],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_error)?;
        let Some(json) = row else {
            return Ok(None);
        };
        let record: RuntimeEffectJournalRecord = serde_json::from_str(&json).map_err(|err| {
            StoreError::Backend(format!("failed to decode runtime effect journal: {err}"))
        })?;
        ensure_supported_schema_version(
            "RuntimeEffectJournalRecord",
            record.schema_version,
            RUNTIME_EFFECT_JOURNAL_SCHEMA_VERSION,
        )?;
        Ok(Some(record))
    }

    async fn save_session_meta(&self, meta: SessionMeta) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        let relation_json = serde_json::to_string(&meta.relation)
            .map_err(|err| StoreError::Backend(err.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO session_meta
             (singleton, session_id, session_name, created_at, model, cwd, relation_json)
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                meta.session_id,
                meta.session_name,
                meta.created_at,
                meta.model,
                meta.cwd,
                relation_json
            ],
        )
        .map_err(sqlite_error)?;
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<SessionMeta>, StoreError> {
        let conn = self.conn.lock().unwrap();
        Ok(load_session_meta_from_conn(&conn))
    }

    async fn tombstone_nodes(&self, ids: &[String]) -> Result<(), StoreError> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(sqlite_error)?;
        for id in ids {
            tx.execute(
                "UPDATE graph_nodes SET tombstoned = 1 WHERE node_id = ?1",
                params![id],
            )
            .map_err(sqlite_error)?;
        }
        tx.commit().map_err(sqlite_error)?;
        Ok(())
    }

    async fn vacuum(&self) -> Result<VacuumReport, StoreError> {
        let conn = self.conn.lock().unwrap();
        let removed = conn
            .execute("DELETE FROM graph_nodes WHERE tombstoned = 1", [])
            .map_err(sqlite_error)?;
        Ok(VacuumReport {
            removed_node_count: removed,
        })
    }

    async fn gc_unreachable(&self) -> Result<GcReport, StoreError> {
        Ok(Self::gc_unreachable(self))
    }
}
