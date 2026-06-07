//! The [`RuntimePersistence`] trait implementation for [`Store`].
//!
//! This is the tokio-rusqlite port of the prior store's `persistence.rs`. The
//! public surface is byte-for-byte the prior store async trait: identical method
//! names and signatures, so consumers swap backends with a path rename only.
//!
//! The translation rules (see `conn.rs`, `lifecycle.rs`, `blobs.rs`):
//!
//! * Pure reads run through `self.conn.call(move |conn| { ... })`.
//! * Read-then-write paths run through `self.conn.write(move |tx| { ... })`
//!   (`BEGIN IMMEDIATE`, commit on `Ok`, rollback on `Err`) — this is the
//!   cross-process write-lock guard.
//! * Paths that may abandon partially-applied writes (the queued-work claim)
//!   run through `self.conn.write_flow`, deciding commit vs rollback via
//!   [`TxOutcome`].
//! * The shared `*_conn` helpers (`try_load_session_head_meta_from_conn`,
//!   `Self::put_checkpoint_conn`, `Self::load_usage_deltas_conn`,
//!   `Self::load_session_graph_from_conn`, the queued-work helpers, …) are
//!   synchronous and take a `&rusqlite::Connection`, so they are reused from
//!   inside these closures (a `&Transaction` derefs to `&Connection`).
//! * Closures must be `'static` + `Send`: every borrow of `self`/caller data is
//!   cloned into an owned value before being moved in.

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
        self.conn
            .call(move |conn| {
                let outcome: Result<Option<PersistedSessionRead>, StoreError> = (|| {
                    let Some(meta) = try_load_session_head_meta_from_conn(conn)? else {
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
                            Self::load_session_graph_from_conn(conn, meta.leaf_node_id.clone())
                        }
                        SessionReadScope::ActivePath { .. } => {
                            Self::load_active_path_session_graph_from_conn(
                                conn,
                                leaf_node_id.clone(),
                            )
                            .map_err(sqlite_error)?
                        }
                    };
                    graph.set_leaf_node_id(leaf_node_id);
                    let checkpoint = meta
                        .checkpoint_ref
                        .as_ref()
                        .and_then(|blob_ref| Self::get_checkpoint_conn(conn, blob_ref));
                    Ok(Some(PersistedSessionRead {
                        session_id: meta.session_id,
                        head_revision: meta.head_revision,
                        config: meta.config,
                        agent_frames: meta.agent_frames,
                        current_agent_frame_id: meta.current_agent_frame_id,
                        graph,
                        checkpoint_ref: meta.checkpoint_ref,
                        checkpoint,
                        token_ledger: merge_token_ledger_entries(Self::load_usage_deltas_conn(
                            conn,
                        )),
                    }))
                })(
                );
                Ok(outcome)
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<lash_core::SessionNodeRecord>, StoreError> {
        let node_id = node_id.to_string();
        let row: Option<String> = self
            .conn
            .call(move |conn| {
                conn.query_row(
                    "SELECT node_json FROM graph_nodes WHERE node_id = ?1 AND tombstoned = 0",
                    params![node_id],
                    |row| row.get(0),
                )
                .optional()
            })
            .await
            .map_err(sqlite_error)?;
        Ok(row.and_then(|json| serde_json::from_str(&json).ok()))
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let blob_profile = self.options.blob_profile;
        let result = self
            .conn
            .write_flow(move |tx| {
                let outcome: Result<RuntimeCommitResult, StoreError> = (|| {
                    let existing = try_load_session_head_meta_from_conn(tx)?;
                    if let Some(bound_session_id) =
                        existing.as_ref().map(|meta| meta.session_id.as_str())
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
                        let prior: Option<(String, String)> = tx
                            .query_row(
                                "SELECT turn_commit_hash, result_json FROM runtime_turn_commits
                                 WHERE session_id = ?1 AND turn_id = ?2",
                                params![completed.session_id, completed.turn_id],
                                |row| Ok((row.get(0)?, row.get(1)?)),
                            )
                            .optional()
                            .map_err(sqlite_error)?;
                        if let Some((turn_commit_hash, result_json)) = prior {
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
                        ensure_queued_work_completion_conn(tx, completed)?;
                    }

                    let stored_checkpoint =
                        Self::put_checkpoint_conn(tx, &commit.checkpoint, blob_profile)
                            .map_err(sqlite_error)?;

                    if !commit.usage_deltas.is_empty() {
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
                        config: commit.config.clone(),
                        agent_frames: commit.agent_frames.clone(),
                        current_agent_frame_id: commit.current_agent_frame_id.clone(),
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
                    let result = RuntimeCommitResult {
                        head_revision: next_revision,
                        checkpoint_ref: stored_checkpoint.checkpoint_ref,
                        manifest: stored_checkpoint.manifest,
                    };
                    if let Some(completed) = &commit.turn_commit {
                        tx.execute(
                            "INSERT INTO runtime_turn_commits (
                                session_id, turn_id, turn_commit_hash, result_json, committed_at_ms
                             )
                             VALUES (?1, ?2, ?3, ?4, ?5)",
                            params![
                                completed.session_id,
                                completed.turn_id,
                                completed.turn_commit_hash,
                                encode_json(&result),
                                current_epoch_ms() as i64
                            ],
                        )
                        .map_err(sqlite_error)?;
                    }
                    Ok(result)
                })();
                // Roll back on a `StoreError` so a failure after the first
                // write (e.g. a head-revision conflict surfaced mid-commit, or a
                // backend write error) does not leave the partial transaction
                // committed, while still carrying the typed error to the caller.
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)??;
        self.maybe_auto_gc().await;
        Ok(result)
    }

    async fn enqueue_queued_work(
        &self,
        batch: QueuedWorkBatchDraft,
    ) -> Result<QueuedWorkBatch, StoreError> {
        let nonce = self.commit_count.fetch_add(1, AtomicOrdering::Relaxed);
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<QueuedWorkBatch, StoreError> = (|| {
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
                            let existing = load_queued_batch_by_id_conn(tx, &batch_id)?
                                .ok_or_else(|| {
                                    StoreError::Backend(
                                        "queued work source row disappeared".to_string(),
                                    )
                                })?;
                            return Ok(existing);
                        }
                    }
                    let now = current_epoch_ms();
                    let batch_id = format!(
                        "qwb:{:x}",
                        Sha256::digest(
                            format!("{}:{:?}:{now}:{nonce}", batch.session_id, batch.source_key)
                                .as_bytes()
                        )
                    );
                    tx.execute(
                        "INSERT INTO queued_work_batches (
                            batch_id, session_id, source_key, delivery_policy, slot_policy,
                            merge_key_json, available_at_ms, enqueued_at_ms
                         )
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                        params![
                            batch_id,
                            batch.session_id,
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
                            params![batch_id, index as i64, item_id, encode_json(payload)],
                        )
                        .map_err(sqlite_error)?;
                    }
                    load_queued_batch_by_id_conn(tx, &batch_id)?.ok_or_else(|| {
                        StoreError::Backend("queued work insert disappeared".to_string())
                    })
                })();
                // Roll back the partially-inserted batch/items on a
                // `StoreError` while still returning the typed error.
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
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
        let session_id = session_id.to_string();
        let owner_id = owner_id.to_string();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<TxOutcome<Option<QueuedWorkClaim>>, StoreError> = (|| {
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
                        Sha256::digest(
                            format!("{session_id}:{owner_id}:{claim_id}:{now}").as_bytes()
                        )
                    );
                    let expires_at = now.saturating_add(lease_ttl_ms);
                    for row in &selected {
                        // Under `BEGIN IMMEDIATE` this connection already holds
                        // the write lock, but the row could still have been
                        // claimed by an earlier committed writer (its
                        // `claim_token` set and not yet expired). The `WHERE`
                        // clause filters those out, so a 0-row update means we
                        // lost the race for this batch: treat the whole claim as
                        // not-won rather than returning a claim that doesn't
                        // actually own the row.
                        let claimed = tx
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
                                    row.batch_id,
                                    claim_id,
                                    owner_id,
                                    lease_token,
                                    now as i64,
                                    expires_at as i64
                                ],
                            )
                            .map_err(sqlite_error)?;
                        if claimed == 0 {
                            // Lost the race for this batch. Roll back any sibling
                            // rows we already claimed in this transaction so we
                            // never return a half-owned claim.
                            return Ok(TxOutcome::Rollback(None));
                        }
                    }
                    let mut batches = Vec::new();
                    for row in selected {
                        batches.push(queued_work_batch_from_conn(tx, row)?);
                    }
                    Ok(TxOutcome::Commit(Some(QueuedWorkClaim {
                        session_id: session_id.clone(),
                        claim_id,
                        owner_id: owner_id.clone(),
                        lease_token,
                        fencing_token,
                        claimed_at_epoch_ms: now,
                        expires_at_epoch_ms: expires_at,
                        batches,
                    })))
                })();
                // Lower a `StoreError` into the rollback arm so the closure body
                // can keep using `?` while still propagating the error to the
                // caller. Encode it as a `Result` carried out of the flow.
                match outcome {
                    Ok(TxOutcome::Commit(value)) => Ok(TxOutcome::Commit(Ok(value))),
                    Ok(TxOutcome::Rollback(value)) => Ok(TxOutcome::Rollback(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn renew_queued_work_claim(
        &self,
        claim: &QueuedWorkClaim,
        lease_ttl_ms: u64,
    ) -> Result<QueuedWorkClaim, StoreError> {
        let now = current_epoch_ms();
        let expires_at = now.saturating_add(lease_ttl_ms);
        let session_id = claim.session_id.clone();
        let claim_id = claim.claim_id.clone();
        let lease_token = claim.lease_token.clone();
        let batch_count = claim.batches.len();
        let changed = self
            .conn
            .write(move |tx| {
                tx.execute(
                    "UPDATE queued_work_batches
                     SET claim_expires_at_ms = ?4
                     WHERE session_id = ?1 AND claim_id = ?2 AND claim_token = ?3",
                    params![session_id, claim_id, lease_token, expires_at as i64],
                )
            })
            .await
            .map_err(sqlite_error)?;
        if changed != batch_count {
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
        let session_id = claim.session_id.clone();
        let claim_id = claim.claim_id.clone();
        let lease_token = claim.lease_token.clone();
        self.conn
            .write(move |tx| {
                tx.execute(
                    "UPDATE queued_work_batches
                     SET claim_id = NULL,
                         claim_owner_id = NULL,
                         claim_token = NULL,
                         claim_claimed_at_ms = 0,
                         claim_expires_at_ms = 0
                     WHERE session_id = ?1 AND claim_id = ?2 AND claim_token = ?3",
                    params![session_id, claim_id, lease_token],
                )
            })
            .await
            .map_err(sqlite_error)?;
        Ok(())
    }

    async fn cancel_queued_work_batch(
        &self,
        session_id: &str,
        batch_id: &str,
    ) -> Result<Option<QueuedWorkBatch>, StoreError> {
        let session_id = session_id.to_string();
        let batch_id = batch_id.to_string();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<Option<QueuedWorkBatch>, StoreError> = (|| {
                    let now = current_epoch_ms() as i64;
                    let row = tx
                        .query_row(
                            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                                    claim_fencing_token
                             FROM queued_work_batches
                             WHERE session_id = ?1
                               AND batch_id = ?2
                               AND (claim_token IS NULL OR claim_expires_at_ms <= ?3)",
                            params![session_id, batch_id, now],
                            queued_batch_row_from_sql,
                        )
                        .optional()
                        .map_err(sqlite_error)?;
                    let Some(row) = row else {
                        return Ok(None);
                    };
                    let batch = queued_work_batch_from_conn(tx, row)?;
                    tx.execute(
                        "DELETE FROM queued_work_batches
                         WHERE session_id = ?1
                           AND batch_id = ?2
                           AND (claim_token IS NULL OR claim_expires_at_ms <= ?3)",
                        params![session_id, batch_id, now],
                    )
                    .map_err(sqlite_error)?;
                    Ok(Some(batch))
                })();
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn list_queued_work(&self, session_id: &str) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let session_id = session_id.to_string();
        self.conn
            .call(move |conn| {
                let outcome: Result<Vec<QueuedWorkBatch>, StoreError> = (|| {
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
                        .map(|row| queued_work_batch_from_conn(conn, row))
                        .collect()
                })();
                Ok(outcome)
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn list_pending_queued_work(
        &self,
        session_id: &str,
    ) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let session_id = session_id.to_string();
        self.conn
            .call(move |conn| {
                let outcome: Result<Vec<QueuedWorkBatch>, StoreError> = (|| {
                    let now = current_epoch_ms();
                    let rows = {
                        let mut stmt = conn
                            .prepare(
                                "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                                        slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                                        claim_fencing_token
                                 FROM queued_work_batches
                                 WHERE session_id = ?1
                                   AND (claim_token IS NULL OR claim_expires_at_ms <= ?2)
                                 ORDER BY enqueue_seq ASC",
                            )
                            .map_err(sqlite_error)?;
                        let rows = stmt
                            .query_map(
                                params![session_id, now as i64],
                                queued_batch_row_from_sql,
                            )
                            .map_err(sqlite_error)?;
                        rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
                    };
                    rows.into_iter()
                        .map(|row| queued_work_batch_from_conn(conn, row))
                        .collect()
                })();
                Ok(outcome)
            })
            .await
            .map_err(sqlite_error)?
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
        let ids = ids.to_vec();
        self.conn
            .write(move |tx| {
                let mut stmt =
                    tx.prepare("UPDATE graph_nodes SET tombstoned = 1 WHERE node_id = ?1")?;
                for id in &ids {
                    stmt.execute(params![id])?;
                }
                Ok(())
            })
            .await
            .map_err(sqlite_error)
    }

    async fn vacuum(&self) -> Result<VacuumReport, StoreError> {
        let removed = self
            .conn
            .write(move |tx| tx.execute("DELETE FROM graph_nodes WHERE tombstoned = 1", []))
            .await
            .map_err(sqlite_error)?;
        Ok(VacuumReport {
            removed_node_count: removed,
        })
    }

    async fn gc_unreachable(&self) -> Result<GcReport, StoreError> {
        Ok(Store::gc_unreachable(self).await)
    }
}
