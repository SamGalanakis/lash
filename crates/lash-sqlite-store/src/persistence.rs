//! The [`RuntimePersistence`] capability-segment implementations for
//! [`Store`]: [`SessionCommitStore`], [`SessionExecutionLeaseStore`],
//! [`QueuedWorkStore`], [`TurnInputStore`], and [`StoreMaintenance`].
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
impl SessionCommitStore for Store {
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
                        .map(|blob_ref| Self::get_checkpoint_conn(conn, blob_ref))
                        .transpose()?
                        .flatten();
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
                                if let Some(completion) =
                                    commit.release_session_execution_lease.as_ref()
                                {
                                    release_session_execution_lease_conn(tx, completion)?;
                                }
                                return Ok(result);
                            }
                            return Err(StoreError::RuntimeTurnCommitConflict {
                                session_id: completed.session_id.clone(),
                                turn_id: completed.turn_id.clone(),
                            });
                        }
                    }
                    let Some(session_execution_lease) = commit.session_execution_lease.as_ref()
                    else {
                        return Err(StoreError::SessionExecutionLeaseExpired {
                            session_id: commit.session_id.clone(),
                        });
                    };
                    ensure_session_execution_lease_conn(
                        tx,
                        &commit.session_id,
                        session_execution_lease,
                    )?;
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
                    for completed in &commit.completed_turn_input_claims {
                        if completed.session_id != commit.session_id {
                            return Err(StoreError::TurnInputClaimExpired {
                                session_id: completed.session_id.clone(),
                                claim_id: completed.claim_id.clone(),
                            });
                        }
                        let owned_rows: usize = tx
                            .query_row(
                                "SELECT COUNT(*)
                                 FROM pending_turn_inputs
                                 WHERE session_id = ?1
                                   AND claim_id = ?2
                                   AND claim_token = ?3",
                                params![
                                    completed.session_id,
                                    completed.claim_id,
                                    completed.lease_token
                                ],
                                |row| row.get::<_, i64>(0),
                            )
                            .map_err(sqlite_error)? as usize;
                        ensure_turn_input_completion_owns_all_inputs(completed, owned_rows)?;
                    }

                    let stored_checkpoint =
                        Self::put_checkpoint_conn(tx, &commit.checkpoint, blob_profile)
                            .map_err(sqlite_error)?;

                    if !commit.usage_deltas.is_empty() {
                        let mut stmt = tx
                            .prepare(
                                "INSERT INTO usage_deltas (
                                    source, model, input_tokens, output_tokens, cache_read_input_tokens, cache_write_input_tokens, reasoning_output_tokens
                                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                            )
                            .map_err(sqlite_error)?;
                        for entry in &commit.usage_deltas {
                            stmt.execute(params![
                                entry.source,
                                entry.model,
                                entry.usage.input_tokens,
                                entry.usage.output_tokens,
                                entry.usage.cache_read_input_tokens,
                                entry.usage.cache_write_input_tokens,
                                entry.usage.reasoning_output_tokens,
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
                        schema_version: lash_core::store::SESSION_HEAD_META_SCHEMA_VERSION,
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
                    for completed in &commit.completed_turn_input_claims {
                        for input_id in &completed.input_ids {
                            tx.execute(
                                "UPDATE pending_turn_inputs
                                 SET state = ?5,
                                     claim_id = NULL,
                                     claim_owner_id = NULL,
                                     claim_owner_incarnation_id = NULL,
                                     claim_owner_liveness_json = NULL,
                                     claim_token = NULL,
                                     claim_claimed_at_ms = 0,
                                     claim_expires_at_ms = 0
                                 WHERE session_id = ?1
                                   AND input_id = ?2
                                   AND claim_id = ?3
                                   AND claim_token = ?4",
                                params![
                                    completed.session_id,
                                    input_id,
                                    completed.claim_id,
                                    completed.lease_token,
                                    lash_core::TurnInputState::Completed.as_str(),
                                ],
                            )
                            .map_err(sqlite_error)?;
                        }
                    }
                    if let Some(turn_id) = commit.interrupted_turn_input_turn_id.as_deref() {
                        let input_ids = {
                            let mut stmt = tx
                                .prepare(
                                    "SELECT input_id, ingress_json
                                     FROM pending_turn_inputs
                                     WHERE session_id = ?1 AND state = ?2",
                                )
                                .map_err(sqlite_error)?;
                            let rows = stmt
                                .query_map(
                                    params![
                                        commit.session_id,
                                        lash_core::TurnInputState::PendingActive.as_str()
                                    ],
                                    |row| {
                                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                                    },
                                )
                                .map_err(sqlite_error)?;
                            let mut input_ids = Vec::new();
                            for row in rows {
                                let (input_id, ingress_json) = row.map_err(sqlite_error)?;
                                let ingress = decode_turn_input_ingress(ingress_json)?;
                                if ingress
                                    .active_turn_id()
                                    .is_some_and(|active| active == turn_id)
                                {
                                    input_ids.push(input_id);
                                }
                            }
                            input_ids
                        };
                        let next_turn_ingress = encode_json(&lash_core::TurnInputIngress::NextTurn);
                        let mut stmt = tx
                            .prepare(
                                "UPDATE pending_turn_inputs
                                 SET state = ?3,
                                     ingress_json = ?4,
                                     claim_id = NULL,
                                     claim_owner_id = NULL,
                                     claim_owner_incarnation_id = NULL,
                                     claim_owner_liveness_json = NULL,
                                     claim_token = NULL,
                                     claim_claimed_at_ms = 0,
                                     claim_expires_at_ms = 0
                                 WHERE session_id = ?1 AND input_id = ?2",
                            )
                            .map_err(sqlite_error)?;
                        for input_id in input_ids {
                            stmt.execute(params![
                                commit.session_id,
                                input_id,
                                lash_core::TurnInputState::DeferredNextTurn.as_str(),
                                next_turn_ingress
                            ])
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
                    if let Some(completion) = commit.release_session_execution_lease.as_ref() {
                        release_session_execution_lease_conn(tx, completion)?;
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

    async fn save_session_meta(&self, meta: SessionMeta) -> Result<(), StoreError> {
        Store::save_session_meta(self, meta).await;
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<SessionMeta>, StoreError> {
        Ok(Store::load_session_meta(self).await)
    }
}

#[async_trait::async_trait]
impl SessionExecutionLeaseStore for Store {
    async fn try_claim_session_execution_lease(
        &self,
        session_id: &str,
        owner: &LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLeaseClaimOutcome, StoreError> {
        let session_id = session_id.to_string();
        let owner = owner.clone();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<SessionExecutionLeaseClaimOutcome, StoreError> = (|| {
                    let now = current_epoch_ms();
                    let current = load_session_execution_lease_row_conn(tx, &session_id)?;
                    if current.as_ref().is_some_and(|lease| {
                        lease.lease_token.is_some() && lease.expires_at_ms > now
                    }) {
                        let current = current.expect("checked current lease is present");
                        if current
                            .owner
                            .as_ref()
                            .is_some_and(|current_owner| current_owner.same_incarnation(&owner))
                        {
                            let expires_at = now.saturating_add(lease_ttl_ms);
                            tx.execute(
                                "UPDATE session_execution_leases
                                 SET lease_expires_at_ms = ?2
                                 WHERE session_id = ?1",
                                params![session_id, expires_at as i64],
                            )
                            .map_err(sqlite_error)?;
                            return Ok(SessionExecutionLeaseClaimOutcome::Acquired(
                                SessionExecutionLease {
                                    session_id,
                                    owner,
                                    lease_token: current.lease_token.expect("live lease token set"),
                                    fencing_token: current.fencing_token,
                                    claimed_at_epoch_ms: current.claimed_at_ms,
                                    expires_at_epoch_ms: expires_at,
                                },
                            ));
                        }
                        return Ok(SessionExecutionLeaseClaimOutcome::Busy {
                            holder: row_to_session_execution_lease(&session_id, current)?,
                        });
                    }
                    Ok(SessionExecutionLeaseClaimOutcome::Acquired(
                        acquire_session_execution_lease_conn(
                            tx,
                            &session_id,
                            &owner,
                            current.as_ref().map_or(0, |lease| lease.fencing_token),
                            now,
                            lease_ttl_ms,
                        )?,
                    ))
                })(
                );
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn reclaim_session_execution_lease(
        &self,
        session_id: &str,
        owner: &LeaseOwnerIdentity,
        observed_holder: &SessionExecutionLeaseFence,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLeaseClaimOutcome, StoreError> {
        let session_id = session_id.to_string();
        let owner = owner.clone();
        let observed_holder = observed_holder.clone();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<SessionExecutionLeaseClaimOutcome, StoreError> = (|| {
                    let now = current_epoch_ms();
                    let current = load_session_execution_lease_row_conn(tx, &session_id)?;
                    let Some(current) = current else {
                        return Ok(SessionExecutionLeaseClaimOutcome::Acquired(
                            acquire_session_execution_lease_conn(
                                tx,
                                &session_id,
                                &owner,
                                0,
                                now,
                                lease_ttl_ms,
                            )?,
                        ));
                    };
                    if current.lease_token.is_none() || current.expires_at_ms <= now {
                        return Ok(SessionExecutionLeaseClaimOutcome::Acquired(
                            acquire_session_execution_lease_conn(
                                tx,
                                &session_id,
                                &owner,
                                current.fencing_token,
                                now,
                                lease_ttl_ms,
                            )?,
                        ));
                    }
                    let holder = row_to_session_execution_lease(&session_id, current)?;
                    if observed_holder.session_id == session_id
                        && holder.owner.same_incarnation(&observed_holder.owner)
                        && holder.lease_token == observed_holder.lease_token
                        && holder.fencing_token == observed_holder.fencing_token
                        && holder.owner.is_definitely_dead_for_claimant(&owner)
                    {
                        let fencing_token = holder.fencing_token.saturating_add(1);
                        let lease_token = format!(
                            "{}:{}:{}:{now}:{fencing_token}",
                            session_id, owner.owner_id, owner.incarnation_id
                        );
                        let expires_at = now.saturating_add(lease_ttl_ms);
                        let liveness_json = encode_liveness(&owner.liveness)?;
                        let changed = tx
                            .execute(
                                "UPDATE session_execution_leases
                                 SET lease_owner_id = ?1,
                                     lease_owner_incarnation_id = ?2,
                                     lease_owner_liveness_json = ?3,
                                     lease_token = ?4,
                                     lease_fencing_token = ?5,
                                     lease_claimed_at_ms = ?6,
                                     lease_expires_at_ms = ?7
                                 WHERE session_id = ?8
                                   AND lease_owner_id = ?9
                                   AND lease_owner_incarnation_id = ?10
                                   AND lease_token = ?11
                                   AND lease_fencing_token = ?12",
                                params![
                                    owner.owner_id,
                                    owner.incarnation_id,
                                    liveness_json,
                                    lease_token,
                                    fencing_token as i64,
                                    now as i64,
                                    expires_at as i64,
                                    session_id,
                                    observed_holder.owner.owner_id,
                                    observed_holder.owner.incarnation_id,
                                    observed_holder.lease_token,
                                    observed_holder.fencing_token as i64,
                                ],
                            )
                            .map_err(sqlite_error)?;
                        if changed == 1 {
                            return Ok(SessionExecutionLeaseClaimOutcome::Acquired(
                                SessionExecutionLease {
                                    session_id,
                                    owner,
                                    lease_token,
                                    fencing_token,
                                    claimed_at_epoch_ms: now,
                                    expires_at_epoch_ms: expires_at,
                                },
                            ));
                        }
                        let current = load_session_execution_lease_row_conn(tx, &session_id)?;
                        if current.as_ref().is_some_and(|lease| {
                            lease.lease_token.is_some() && lease.expires_at_ms > now
                        }) {
                            let current = current.expect("checked current lease is present");
                            return Ok(SessionExecutionLeaseClaimOutcome::Busy {
                                holder: row_to_session_execution_lease(&session_id, current)?,
                            });
                        }
                        let previous_fencing_token =
                            current.as_ref().map_or(0, |lease| lease.fencing_token);
                        return Ok(SessionExecutionLeaseClaimOutcome::Acquired(
                            acquire_session_execution_lease_conn(
                                tx,
                                &session_id,
                                &owner,
                                previous_fencing_token,
                                now,
                                lease_ttl_ms,
                            )?,
                        ));
                    }
                    Ok(SessionExecutionLeaseClaimOutcome::Busy { holder })
                })(
                );
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn renew_session_execution_lease(
        &self,
        fence: &SessionExecutionLeaseFence,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLease, StoreError> {
        let fence = fence.clone();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<SessionExecutionLease, StoreError> = (|| {
                    let now = current_epoch_ms();
                    let current = load_session_execution_lease_row_conn(tx, &fence.session_id)?;
                    let Some(current) = current else {
                        return Err(StoreError::SessionExecutionLeaseExpired {
                            session_id: fence.session_id.clone(),
                        });
                    };
                    if !current
                        .owner
                        .as_ref()
                        .is_some_and(|owner| owner.same_incarnation(&fence.owner))
                        || current.lease_token.as_deref() != Some(fence.lease_token.as_str())
                        || current.fencing_token != fence.fencing_token
                        || current.expires_at_ms <= now
                    {
                        return Err(StoreError::SessionExecutionLeaseExpired {
                            session_id: fence.session_id.clone(),
                        });
                    }
                    let expires_at = now.saturating_add(lease_ttl_ms);
                    tx.execute(
                        "UPDATE session_execution_leases
                         SET lease_expires_at_ms = ?5
                         WHERE session_id = ?1
                           AND lease_owner_id = ?2
                           AND lease_owner_incarnation_id = ?3
                           AND lease_token = ?4
                           AND lease_fencing_token = ?6",
                        params![
                            fence.session_id,
                            fence.owner.owner_id,
                            fence.owner.incarnation_id,
                            fence.lease_token,
                            expires_at as i64,
                            fence.fencing_token as i64
                        ],
                    )
                    .map_err(sqlite_error)?;
                    Ok(SessionExecutionLease {
                        session_id: fence.session_id,
                        owner: fence.owner,
                        lease_token: fence.lease_token,
                        fencing_token: fence.fencing_token,
                        claimed_at_epoch_ms: current.claimed_at_ms,
                        expires_at_epoch_ms: expires_at,
                    })
                })();
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn release_session_execution_lease(
        &self,
        completion: &SessionExecutionLeaseCompletion,
    ) -> Result<(), StoreError> {
        let completion = completion.clone();
        self.conn
            .write_flow(move |tx| {
                let outcome = release_session_execution_lease_conn(tx, &completion);
                match outcome {
                    Ok(()) => Ok(TxOutcome::Commit(Ok(()))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }
}

#[async_trait::async_trait]
impl QueuedWorkStore for Store {
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
                    let batch_id =
                        derive_batch_id(&batch.session_id, batch.source_key.as_deref(), now, Some(nonce));
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

    async fn claim_leading_ready_session_command(
        &self,
        session_id: &str,
        session_execution_lease: &SessionExecutionLeaseFence,
        owner: &LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        let session_id = session_id.to_string();
        let session_execution_lease = session_execution_lease.clone();
        let owner = owner.clone();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<TxOutcome<Option<QueuedWorkClaim>>, StoreError> = (|| {
                    ensure_session_execution_lease_conn(
                        tx,
                        &session_id,
                        &session_execution_lease,
                    )?;
                    let now = current_epoch_ms();
                    let candidate_rows = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                                        slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                                        claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                                        claim_owner_liveness_json, claim_token, claim_expires_at_ms
                                 FROM queued_work_batches
                                 WHERE session_id = ?1
                                   AND available_at_ms <= ?2
                                 ORDER BY enqueue_seq ASC
                                 LIMIT ?3",
                            )
                            .map_err(sqlite_error)?;
                        let rows = stmt
                            .query_map(
                                params![session_id, now as i64, claim_scan_limit(1)],
                                queued_batch_row_from_sql,
                            )
                            .map_err(sqlite_error)?;
                        rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
                    };
                    let candidate_rows = candidate_rows
                        .into_iter()
                        .filter(|row| {
                            row.claim_token.is_none()
                                || row.claim_expires_at_ms <= now
                                || row
                                    .claim_owner
                                    .as_ref()
                                    .is_some_and(|holder| holder.is_definitely_dead_for_claimant(&owner))
                        })
                        .collect::<Vec<_>>();
                    let candidate_batches = candidate_rows
                        .iter()
                        .map(|row| queued_work_batch_from_conn(tx, row.clone()))
                        .collect::<Result<Vec<_>, StoreError>>()?;
                    let candidates = candidate_rows
                        .iter()
                        .zip(candidate_batches.iter())
                        .map(|(row, batch)| {
                            Ok(ClaimCandidate {
                                enqueue_seq: row.enqueue_seq,
                                claim_fencing_token: row.claim_fencing_token,
                                work_class: batch.work_class().ok_or_else(|| {
                                    StoreError::Backend(format!(
                                        "queued-work batch `{}` has mixed or empty payload classes",
                                        batch.batch_id
                                    ))
                                })?,
                                delivery_policy: decode_delivery_policy(
                                    row.delivery_policy.clone(),
                                )?,
                                slot_policy: decode_slot_policy(row.slot_policy.clone())?,
                                merge_key: decode_merge_key(row.merge_key_json.clone())?,
                            })
                        })
                        .collect::<Result<Vec<_>, StoreError>>()?;
                    let selected_len = select_leading_session_command(&candidates);
                    if selected_len == 0 {
                        return Ok(TxOutcome::Commit(None));
                    }
                    let mut selected = candidate_rows;
                    selected.truncate(selected_len);
                    let mut selected_batches = candidate_batches;
                    selected_batches.truncate(selected_len);
                    let lease = QueuedWorkClaimLease::derive(
                        &candidates[0],
                        &session_id,
                        &owner,
                        now,
                        lease_ttl_ms,
                    );
                    let liveness_json = encode_liveness(&owner.liveness)?;
                    for row in &selected {
                        let claimed = tx
                            .execute(
                                "UPDATE queued_work_batches
                                 SET claim_id = ?3,
                                     claim_owner_id = ?4,
                                     claim_owner_incarnation_id = ?5,
                                     claim_owner_liveness_json = ?6,
                                     claim_token = ?7,
                                     claim_fencing_token = claim_fencing_token + 1,
                                     claim_claimed_at_ms = ?8,
                                     claim_expires_at_ms = ?9
                                 WHERE session_id = ?1
                                   AND batch_id = ?2
                                   AND (
                                        claim_token IS NULL
                                        OR claim_expires_at_ms <= ?8
                                        OR (
                                            claim_token = ?10
                                            AND claim_owner_id = ?11
                                            AND claim_owner_incarnation_id = ?12
                                        )
                                   )",
                                params![
                                    session_id,
                                    row.batch_id,
                                    lease.claim_id,
                                    owner.owner_id.as_str(),
                                    owner.incarnation_id.as_str(),
                                    liveness_json.as_str(),
                                    lease.lease_token,
                                    now as i64,
                                    lease.expires_at_epoch_ms as i64,
                                    row.claim_token,
                                    row.claim_owner.as_ref().map(|owner| owner.owner_id.as_str()),
                                    row.claim_owner
                                        .as_ref()
                                        .map(|owner| owner.incarnation_id.as_str())
                                ],
                            )
                            .map_err(sqlite_error)?;
                        if claimed == 0 {
                            return Ok(TxOutcome::Rollback(None));
                        }
                    }
                    Ok(TxOutcome::Commit(Some(QueuedWorkClaim {
                        session_id: session_id.clone(),
                        claim_id: lease.claim_id,
                        owner: owner.clone(),
                        lease_token: lease.lease_token,
                        fencing_token: lease.fencing_token,
                        claimed_at_epoch_ms: lease.claimed_at_epoch_ms,
                        expires_at_epoch_ms: lease.expires_at_epoch_ms,
                        batches: selected_batches,
                    })))
                })();
                match outcome {
                    Ok(TxOutcome::Commit(value)) => Ok(TxOutcome::Commit(Ok(value))),
                    Ok(TxOutcome::Rollback(value)) => Ok(TxOutcome::Rollback(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn claim_ready_queued_work(
        &self,
        session_id: &str,
        session_execution_lease: &SessionExecutionLeaseFence,
        owner: &LeaseOwnerIdentity,
        boundary: QueuedWorkClaimBoundary,
        lease_ttl_ms: u64,
        max_batches: usize,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        if max_batches == 0 {
            return Ok(None);
        }
        let session_id = session_id.to_string();
        let session_execution_lease = session_execution_lease.clone();
        let owner = owner.clone();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<TxOutcome<Option<QueuedWorkClaim>>, StoreError> = (|| {
                    ensure_session_execution_lease_conn(
                        tx,
                        &session_id,
                        &session_execution_lease,
                    )?;
                    let now = current_epoch_ms();
                    let candidate_rows = {
                        let mut stmt = tx
                            .prepare(
                                "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                                        slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                                        claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                                        claim_owner_liveness_json, claim_token, claim_expires_at_ms
                                 FROM queued_work_batches
                                 WHERE session_id = ?1
                                   AND available_at_ms <= ?2
                                 ORDER BY enqueue_seq ASC
                                 LIMIT ?3",
                            )
                            .map_err(sqlite_error)?;
                        let rows = stmt
                            .query_map(
                                params![session_id, now as i64, claim_scan_limit(max_batches)],
                                queued_batch_row_from_sql,
                            )
                            .map_err(sqlite_error)?;
                        rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
                    };
                    let candidate_rows = candidate_rows
                        .into_iter()
                        .filter(|row| {
                            row.claim_token.is_none()
                                || row.claim_expires_at_ms <= now
                                || row
                                    .claim_owner
                                    .as_ref()
                                    .is_some_and(|holder| holder.is_definitely_dead_for_claimant(&owner))
                        })
                        .collect::<Vec<_>>();
                    let candidate_batches = candidate_rows
                        .iter()
                        .map(|row| queued_work_batch_from_conn(tx, row.clone()))
                        .collect::<Result<Vec<_>, StoreError>>()?;
                    let candidates = candidate_rows
                        .iter()
                        .zip(candidate_batches.iter())
                        .map(|(row, batch)| {
                            Ok(ClaimCandidate {
                                enqueue_seq: row.enqueue_seq,
                                claim_fencing_token: row.claim_fencing_token,
                                work_class: batch.work_class().ok_or_else(|| {
                                    StoreError::Backend(format!(
                                        "queued-work batch `{}` has mixed or empty payload classes",
                                        batch.batch_id
                                    ))
                                })?,
                                delivery_policy: decode_delivery_policy(
                                    row.delivery_policy.clone(),
                                )?,
                                slot_policy: decode_slot_policy(row.slot_policy.clone())?,
                                merge_key: decode_merge_key(row.merge_key_json.clone())?,
                            })
                        })
                        .collect::<Result<Vec<_>, StoreError>>()?;
                    let selected_len =
                        select_turn_work_claim_prefix(&candidates, boundary, max_batches);
                    if selected_len == 0 {
                        return Ok(TxOutcome::Commit(None));
                    }
                    let mut selected = candidate_rows;
                    selected.truncate(selected_len);
                    let mut selected_batches = candidate_batches;
                    selected_batches.truncate(selected_len);
                    let lease = QueuedWorkClaimLease::derive(
                        &candidates[0],
                        &session_id,
                        &owner,
                        now,
                        lease_ttl_ms,
                    );
                    let liveness_json = encode_liveness(&owner.liveness)?;
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
                                     claim_owner_incarnation_id = ?5,
                                     claim_owner_liveness_json = ?6,
                                     claim_token = ?7,
                                     claim_fencing_token = claim_fencing_token + 1,
                                     claim_claimed_at_ms = ?8,
                                     claim_expires_at_ms = ?9
                                 WHERE session_id = ?1
                                   AND batch_id = ?2
                                   AND (
                                        claim_token IS NULL
                                        OR claim_expires_at_ms <= ?8
                                        OR (
                                            claim_token = ?10
                                            AND claim_owner_id = ?11
                                            AND claim_owner_incarnation_id = ?12
                                        )
                                   )",
                                params![
                                    session_id,
                                    row.batch_id,
                                    lease.claim_id,
                                    owner.owner_id.as_str(),
                                    owner.incarnation_id.as_str(),
                                    liveness_json.as_str(),
                                    lease.lease_token,
                                    now as i64,
                                    lease.expires_at_epoch_ms as i64,
                                    row.claim_token,
                                    row.claim_owner.as_ref().map(|owner| owner.owner_id.as_str()),
                                    row.claim_owner
                                        .as_ref()
                                        .map(|owner| owner.incarnation_id.as_str())
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
                    Ok(TxOutcome::Commit(Some(QueuedWorkClaim {
                        session_id: session_id.clone(),
                        claim_id: lease.claim_id,
                        owner: owner.clone(),
                        lease_token: lease.lease_token,
                        fencing_token: lease.fencing_token,
                        claimed_at_epoch_ms: lease.claimed_at_epoch_ms,
                        expires_at_epoch_ms: lease.expires_at_epoch_ms,
                        batches: selected_batches,
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
        renewed_claim(claim, changed, expires_at)
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
                         claim_owner_incarnation_id = NULL,
                         claim_owner_liveness_json = NULL,
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
                                    claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                                    claim_owner_liveness_json, claim_token, claim_expires_at_ms
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
                                        claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                                        claim_owner_liveness_json, claim_token, claim_expires_at_ms
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
                                        claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                                        claim_owner_liveness_json, claim_token, claim_expires_at_ms
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
}

#[async_trait::async_trait]
impl TurnInputStore for Store {
    async fn enqueue_pending_turn_input(
        &self,
        draft: lash_core::PendingTurnInputDraft,
    ) -> Result<lash_core::PendingTurnInput, StoreError> {
        let nonce = self.commit_count.fetch_add(1, AtomicOrdering::Relaxed);
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<lash_core::PendingTurnInput, StoreError> = (|| {
                    if let Some(source_key) = draft.source_key.as_deref() {
                        let existing_id: Option<String> = tx
                            .query_row(
                                "SELECT input_id
                                 FROM pending_turn_inputs
                                 WHERE session_id = ?1 AND source_key = ?2",
                                params![draft.session_id, source_key],
                                |row| row.get(0),
                            )
                            .optional()
                            .map_err(sqlite_error)?;
                        if let Some(input_id) = existing_id {
                            let existing = load_pending_turn_input_by_id_conn(
                                tx,
                                &draft.session_id,
                                &input_id,
                            )?
                            .ok_or_else(|| {
                                StoreError::Backend(
                                    "pending turn input source row disappeared".to_string(),
                                )
                            })?;
                            if !draft.submitted_content_matches(&existing).map_err(|err| {
                                StoreError::Backend(format!(
                                    "failed to compare pending turn input submission: {err}"
                                ))
                            })? {
                                return Err(StoreError::PendingTurnInputSourceKeyConflict {
                                    session_id: draft.session_id.clone(),
                                    source_key: source_key.to_string(),
                                    existing_input_id: existing.input_id.clone(),
                                });
                            }
                            return Ok(existing);
                        }
                    }
                    let now = current_epoch_ms();
                    let input_id = draft.input_id.clone().unwrap_or_else(|| {
                        derive_pending_turn_input_id(
                            &draft.session_id,
                            draft.source_key.as_deref(),
                            now,
                            nonce,
                        )
                    });
                    let state = match draft.ingress {
                        lash_core::TurnInputIngress::ActiveTurn { .. } => {
                            lash_core::TurnInputState::PendingActive
                        }
                        lash_core::TurnInputIngress::NextTurn => {
                            lash_core::TurnInputState::DeferredNextTurn
                        }
                    };
                    tx.execute(
                        "INSERT INTO pending_turn_inputs (
                            input_id, session_id, source_key, ingress_json, state,
                            input_json, enqueued_at_ms
                         )
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            input_id,
                            draft.session_id,
                            draft.source_key.as_deref(),
                            encode_json(&draft.ingress),
                            state.as_str(),
                            encode_json(&draft.input),
                            now as i64,
                        ],
                    )
                    .map_err(sqlite_error)?;
                    load_pending_turn_input_by_id_conn(tx, &draft.session_id, &input_id)?
                        .ok_or_else(|| {
                            StoreError::Backend("pending turn input insert disappeared".to_string())
                        })
                })();
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn list_pending_turn_inputs(
        &self,
        session_id: &str,
    ) -> Result<Vec<lash_core::PendingTurnInput>, StoreError> {
        let session_id = session_id.to_string();
        self.conn
            .call(move |conn| {
                let outcome: Result<Vec<lash_core::PendingTurnInput>, StoreError> = (|| {
                    let now = current_epoch_ms();
                    let rows = {
                        let mut stmt = conn
                            .prepare(
                                "SELECT enqueue_seq, input_id, session_id, source_key, ingress_json,
                                        state, input_json, enqueued_at_ms, claim_id, claim_fencing_token,
                                        claim_owner_id, claim_owner_incarnation_id,
                                        claim_owner_liveness_json, claim_token, claim_expires_at_ms
                                 FROM pending_turn_inputs
                                 WHERE session_id = ?1
                                   AND state IN (?2, ?3)
                                   AND (claim_token IS NULL OR claim_expires_at_ms <= ?4)
                                 ORDER BY enqueue_seq ASC",
                            )
                            .map_err(sqlite_error)?;
                        let rows = stmt
                            .query_map(
                                params![
                                    session_id,
                                    lash_core::TurnInputState::PendingActive.as_str(),
                                    lash_core::TurnInputState::DeferredNextTurn.as_str(),
                                    now as i64
                                ],
                                pending_turn_input_row_from_sql,
                            )
                            .map_err(sqlite_error)?;
                        rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
                    };
                    rows.into_iter().map(pending_turn_input_from_row).collect()
                })(
                );
                Ok(outcome)
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn cancel_pending_turn_inputs(
        &self,
        session_id: &str,
        targets: &[lash_core::PendingTurnInputCancelTarget],
    ) -> Result<Vec<lash_core::PendingTurnInputCancelResult>, StoreError> {
        let session_id = session_id.to_string();
        let targets = targets.to_vec();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<Vec<lash_core::PendingTurnInputCancelResult>, StoreError> =
                    (|| {
                        let now = current_epoch_ms();
                        let mut results = Vec::with_capacity(targets.len());
                        for target in targets {
                            let outcome = match load_pending_turn_input_row_by_target_conn(
                                tx,
                                &session_id,
                                &target,
                            )? {
                                Some(row) => cancel_pending_turn_input_row_conn(tx, row, now)?,
                                None => lash_core::PendingTurnInputCancelOutcome::NotFound,
                            };
                            results
                                .push(lash_core::PendingTurnInputCancelResult { target, outcome });
                        }
                        Ok(results)
                    })();
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn cancel_pending_turn_input_suffix(
        &self,
        session_id: &str,
        anchor: &lash_core::PendingTurnInputCancelTarget,
    ) -> Result<lash_core::PendingTurnInputSuffixCancelOutcome, StoreError> {
        let session_id = session_id.to_string();
        let anchor = anchor.clone();
        self.conn
            .write_flow(move |tx| {
                let outcome: Result<lash_core::PendingTurnInputSuffixCancelOutcome, StoreError> =
                    (|| {
                        let now = current_epoch_ms();
                        let Some(anchor_row) =
                            load_pending_turn_input_row_by_target_conn(tx, &session_id, &anchor)?
                        else {
                            return Ok(
                                lash_core::PendingTurnInputSuffixCancelOutcome::AnchorNotFound {
                                    anchor,
                                },
                            );
                        };
                        let rows = {
                            let mut stmt = tx
                                .prepare(
                                    "SELECT enqueue_seq, input_id, session_id, source_key, ingress_json,
                                            state, input_json, enqueued_at_ms, claim_id, claim_fencing_token,
                                            claim_owner_id, claim_owner_incarnation_id,
                                            claim_owner_liveness_json, claim_token, claim_expires_at_ms
                                     FROM pending_turn_inputs
                                     WHERE session_id = ?1 AND enqueue_seq >= ?2
                                     ORDER BY enqueue_seq ASC",
                                )
                                .map_err(sqlite_error)?;
                            let rows = stmt
                                .query_map(
                                    params![session_id, anchor_row.enqueue_seq as i64],
                                    pending_turn_input_row_from_sql,
                                )
                                .map_err(sqlite_error)?;
                            rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
                        };
                        let mut outcomes = Vec::with_capacity(rows.len());
                        for row in rows {
                            outcomes.push(cancel_pending_turn_input_row_conn(tx, row, now)?);
                        }
                        Ok(lash_core::PendingTurnInputSuffixCancelOutcome::Outcomes {
                            anchor,
                            outcomes,
                        })
                    })();
                match outcome {
                    Ok(value) => Ok(TxOutcome::Commit(Ok(value))),
                    Err(err) => Ok(TxOutcome::Rollback(Err(err))),
                }
            })
            .await
            .map_err(sqlite_error)?
    }

    async fn claim_active_turn_inputs(
        &self,
        session_id: &str,
        session_execution_lease: &SessionExecutionLeaseFence,
        owner: &LeaseOwnerIdentity,
        turn_id: &str,
        checkpoint: lash_core::CheckpointKind,
        lease_ttl_ms: u64,
        max_inputs: usize,
    ) -> Result<Option<lash_core::TurnInputClaim>, StoreError> {
        claim_pending_turn_inputs_sqlite(
            &self.conn,
            session_id,
            session_execution_lease,
            owner,
            lease_ttl_ms,
            max_inputs,
            lash_core::TurnInputClaimMode::ActiveTurn {
                turn_id: turn_id.to_string(),
                checkpoint,
            },
        )
        .await
    }

    async fn claim_next_turn_inputs(
        &self,
        session_id: &str,
        session_execution_lease: &SessionExecutionLeaseFence,
        owner: &LeaseOwnerIdentity,
        lease_ttl_ms: u64,
        max_inputs: usize,
    ) -> Result<Option<lash_core::TurnInputClaim>, StoreError> {
        claim_pending_turn_inputs_sqlite(
            &self.conn,
            session_id,
            session_execution_lease,
            owner,
            lease_ttl_ms,
            max_inputs,
            lash_core::TurnInputClaimMode::NextTurn,
        )
        .await
    }

    async fn abandon_turn_input_claim(
        &self,
        claim: &lash_core::TurnInputClaim,
    ) -> Result<(), StoreError> {
        let session_id = claim.session_id.clone();
        let claim_id = claim.claim_id.clone();
        let lease_token = claim.lease_token.clone();
        let restored_state = match claim.mode {
            lash_core::TurnInputClaimMode::ActiveTurn { .. } => {
                lash_core::TurnInputState::PendingActive
            }
            lash_core::TurnInputClaimMode::NextTurn => lash_core::TurnInputState::DeferredNextTurn,
        };
        self.conn
            .write(move |tx| {
                tx.execute(
                    "UPDATE pending_turn_inputs
                     SET state = CASE
                             WHEN state = ?4 THEN ?5
                             ELSE state
                         END,
                         claim_id = NULL,
                         claim_owner_id = NULL,
                         claim_owner_incarnation_id = NULL,
                         claim_owner_liveness_json = NULL,
                         claim_token = NULL,
                         claim_claimed_at_ms = 0,
                         claim_expires_at_ms = 0
                     WHERE session_id = ?1 AND claim_id = ?2 AND claim_token = ?3",
                    params![
                        session_id,
                        claim_id,
                        lease_token,
                        lash_core::TurnInputState::Accepted.as_str(),
                        restored_state.as_str(),
                    ],
                )
            })
            .await
            .map_err(sqlite_error)?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl StoreMaintenance for Store {
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
        let (removed_node_count, removed_pending_turn_input_tombstone_count) = self
            .conn
            .write(move |tx| {
                let removed_node_count =
                    tx.execute("DELETE FROM graph_nodes WHERE tombstoned = 1", [])?;
                let removed_pending_turn_input_tombstone_count = tx.execute(
                    "DELETE FROM pending_turn_inputs
                     WHERE state IN (?1, ?2)",
                    params![
                        lash_core::TurnInputState::Cancelled.as_str(),
                        lash_core::TurnInputState::Completed.as_str()
                    ],
                )?;
                Ok((
                    removed_node_count,
                    removed_pending_turn_input_tombstone_count,
                ))
            })
            .await
            .map_err(sqlite_error)?;
        Ok(VacuumReport {
            removed_node_count,
            removed_pending_turn_input_tombstone_count,
        })
    }

    async fn gc_unreachable(&self) -> Result<GcReport, StoreError> {
        Ok(Store::gc_unreachable(self).await)
    }
}

fn derive_pending_turn_input_id(
    session_id: &str,
    source_key: Option<&str>,
    now_epoch_ms: u64,
    nonce: u64,
) -> String {
    format!(
        "ti:{:x}",
        Sha256::digest(format!("{session_id}:{source_key:?}:{now_epoch_ms}:{nonce}").as_bytes())
    )
}

fn cancel_pending_turn_input_row_conn(
    conn: &Connection,
    row: PendingTurnInputRow,
    now_epoch_ms: u64,
) -> Result<lash_core::PendingTurnInputCancelOutcome, StoreError> {
    let mut input = pending_turn_input_from_row(row.clone())?;
    match input.state {
        lash_core::TurnInputState::Cancelled => Ok(
            lash_core::PendingTurnInputCancelOutcome::AlreadyCancelled(input),
        ),
        lash_core::TurnInputState::Completed => Ok(
            lash_core::PendingTurnInputCancelOutcome::AlreadyCompleted(input),
        ),
        lash_core::TurnInputState::Accepted => {
            Ok(lash_core::PendingTurnInputCancelOutcome::AlreadyClaimed {
                claim: pending_turn_input_claim_diagnostics_from_row(&row, input.state),
                input,
            })
        }
        lash_core::TurnInputState::PendingActive | lash_core::TurnInputState::DeferredNextTurn => {
            let live_claim = row.claim_token.is_some() && row.claim_expires_at_ms > now_epoch_ms;
            if live_claim {
                return Ok(lash_core::PendingTurnInputCancelOutcome::AlreadyClaimed {
                    claim: pending_turn_input_claim_diagnostics_from_row(&row, input.state),
                    input,
                });
            }
            conn.execute(
                "UPDATE pending_turn_inputs
                 SET state = ?3,
                     claim_id = NULL,
                     claim_owner_id = NULL,
                     claim_owner_incarnation_id = NULL,
                     claim_owner_liveness_json = NULL,
                     claim_token = NULL,
                     claim_claimed_at_ms = 0,
                     claim_expires_at_ms = 0
                 WHERE session_id = ?1 AND input_id = ?2",
                params![
                    row.session_id,
                    row.input_id,
                    lash_core::TurnInputState::Cancelled.as_str(),
                ],
            )
            .map_err(sqlite_error)?;
            input.state = lash_core::TurnInputState::Cancelled;
            Ok(lash_core::PendingTurnInputCancelOutcome::Cancelled(input))
        }
    }
}

async fn claim_pending_turn_inputs_sqlite(
    conn: &SqliteConnection,
    session_id: &str,
    session_execution_lease: &SessionExecutionLeaseFence,
    owner: &LeaseOwnerIdentity,
    lease_ttl_ms: u64,
    max_inputs: usize,
    mode: lash_core::TurnInputClaimMode,
) -> Result<Option<lash_core::TurnInputClaim>, StoreError> {
    if max_inputs == 0 {
        return Ok(None);
    }
    let session_id = session_id.to_string();
    let session_execution_lease = session_execution_lease.clone();
    let owner = owner.clone();
    conn.write_flow(move |tx| {
        let outcome: Result<TxOutcome<Option<lash_core::TurnInputClaim>>, StoreError> = (|| {
            ensure_session_execution_lease_conn(tx, &session_id, &session_execution_lease)?;
            let now = current_epoch_ms();
            let wanted_state = match &mode {
                lash_core::TurnInputClaimMode::ActiveTurn { .. } => {
                    lash_core::TurnInputState::PendingActive
                }
                lash_core::TurnInputClaimMode::NextTurn => {
                    lash_core::TurnInputState::DeferredNextTurn
                }
            };
            let candidate_rows = {
                let mut stmt = tx
                    .prepare(
                        "SELECT enqueue_seq, input_id, session_id, source_key, ingress_json,
                                state, input_json, enqueued_at_ms, claim_id, claim_fencing_token,
                                claim_owner_id, claim_owner_incarnation_id,
                                claim_owner_liveness_json, claim_token, claim_expires_at_ms
                         FROM pending_turn_inputs
                         WHERE session_id = ?1 AND state = ?2
                         ORDER BY enqueue_seq ASC
                         LIMIT ?3",
                    )
                    .map_err(sqlite_error)?;
                let rows = stmt
                    .query_map(
                        params![
                            session_id,
                            wanted_state.as_str(),
                            (max_inputs as i64).saturating_add(32)
                        ],
                        pending_turn_input_row_from_sql,
                    )
                    .map_err(sqlite_error)?;
                rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
            };
            let mut selected = Vec::new();
            for row in candidate_rows {
                let claim_available = row.claim_token.is_none()
                    || row.claim_expires_at_ms <= now
                    || row
                        .claim_owner
                        .as_ref()
                        .is_some_and(|holder| holder.is_definitely_dead_for_claimant(&owner));
                if !claim_available {
                    continue;
                }
                let input = pending_turn_input_from_row(row.clone())?;
                let matches_mode = match &mode {
                    lash_core::TurnInputClaimMode::ActiveTurn {
                        turn_id,
                        checkpoint,
                    } => {
                        input
                            .ingress
                            .active_turn_id()
                            .is_some_and(|active| active == turn_id)
                            && input.ingress.admits_checkpoint(*checkpoint)
                    }
                    lash_core::TurnInputClaimMode::NextTurn => input.state.is_next_turn_pending(),
                };
                if matches_mode {
                    selected.push((row, input));
                    if selected.len() >= max_inputs {
                        break;
                    }
                }
            }
            let Some((head, _)) = selected.first() else {
                return Ok(TxOutcome::Commit(None));
            };
            let lease = TurnInputClaimLease::derive(head, &session_id, &owner, now, lease_ttl_ms);
            let liveness_json = encode_liveness(&owner.liveness)?;
            let state_after_claim = match &mode {
                lash_core::TurnInputClaimMode::ActiveTurn { .. } => {
                    lash_core::TurnInputState::Accepted
                }
                lash_core::TurnInputClaimMode::NextTurn => {
                    lash_core::TurnInputState::DeferredNextTurn
                }
            };
            let mut inputs = Vec::new();
            for (row, mut input) in selected {
                let claimed = tx
                    .execute(
                        "UPDATE pending_turn_inputs
                         SET state = ?3,
                             claim_id = ?4,
                             claim_owner_id = ?5,
                             claim_owner_incarnation_id = ?6,
                             claim_owner_liveness_json = ?7,
                             claim_token = ?8,
                             claim_fencing_token = claim_fencing_token + 1,
                             claim_claimed_at_ms = ?9,
                             claim_expires_at_ms = ?10
                         WHERE session_id = ?1
                           AND input_id = ?2
                           AND (
                                claim_token IS NULL
                                OR claim_expires_at_ms <= ?9
                                OR (
                                    claim_token = ?11
                                    AND claim_owner_id = ?12
                                    AND claim_owner_incarnation_id = ?13
                                )
                           )",
                        params![
                            session_id,
                            row.input_id,
                            state_after_claim.as_str(),
                            lease.claim_id,
                            owner.owner_id.as_str(),
                            owner.incarnation_id.as_str(),
                            liveness_json.as_str(),
                            lease.lease_token,
                            now as i64,
                            lease.expires_at_epoch_ms as i64,
                            row.claim_token,
                            row.claim_owner
                                .as_ref()
                                .map(|owner| owner.owner_id.as_str()),
                            row.claim_owner
                                .as_ref()
                                .map(|owner| owner.incarnation_id.as_str())
                        ],
                    )
                    .map_err(sqlite_error)?;
                if claimed == 0 {
                    return Ok(TxOutcome::Rollback(None));
                }
                input.state = state_after_claim;
                inputs.push(input);
            }
            Ok(TxOutcome::Commit(Some(lash_core::TurnInputClaim {
                session_id: session_id.clone(),
                claim_id: lease.claim_id,
                owner: owner.clone(),
                lease_token: lease.lease_token,
                fencing_token: lease.fencing_token,
                claimed_at_epoch_ms: lease.claimed_at_epoch_ms,
                expires_at_epoch_ms: lease.expires_at_epoch_ms,
                mode,
                inputs,
            })))
        })(
        );
        match outcome {
            Ok(TxOutcome::Commit(value)) => Ok(TxOutcome::Commit(Ok(value))),
            Ok(TxOutcome::Rollback(value)) => Ok(TxOutcome::Rollback(Ok(value))),
            Err(err) => Ok(TxOutcome::Rollback(Err(err))),
        }
    })
    .await
    .map_err(sqlite_error)?
}

struct SessionExecutionLeaseRow {
    owner: Option<LeaseOwnerIdentity>,
    lease_token: Option<String>,
    fencing_token: u64,
    claimed_at_ms: u64,
    expires_at_ms: u64,
}

fn load_session_execution_lease_row_conn(
    conn: &Connection,
    session_id: &str,
) -> Result<Option<SessionExecutionLeaseRow>, StoreError> {
    let row = conn
        .query_row(
            "SELECT lease_owner_id, lease_token, lease_fencing_token,
                    lease_claimed_at_ms, lease_expires_at_ms,
                    lease_owner_incarnation_id, lease_owner_liveness_json
             FROM session_execution_leases
             WHERE session_id = ?1",
            params![session_id],
            |row| {
                let owner_id: Option<String> = row.get(0)?;
                let incarnation_id: Option<String> = row.get(5)?;
                let liveness_json: Option<String> = row.get(6)?;
                Ok(SessionExecutionLeaseRow {
                    owner: lease_owner_from_columns(owner_id, incarnation_id, liveness_json),
                    lease_token: row.get(1)?,
                    fencing_token: row.get::<_, i64>(2)? as u64,
                    claimed_at_ms: row.get::<_, i64>(3)? as u64,
                    expires_at_ms: row.get::<_, i64>(4)? as u64,
                })
            },
        )
        .optional()
        .map_err(sqlite_error)?;
    Ok(row)
}

fn lease_owner_from_columns(
    owner_id: Option<String>,
    incarnation_id: Option<String>,
    liveness_json: Option<String>,
) -> Option<LeaseOwnerIdentity> {
    owner_id.map(|owner_id| LeaseOwnerIdentity {
        incarnation_id: incarnation_id.unwrap_or_else(|| owner_id.clone()),
        owner_id,
        liveness: liveness_json
            .as_deref()
            .and_then(|json| serde_json::from_str(json).ok())
            .unwrap_or(LeaseOwnerLiveness::Opaque),
    })
}

fn encode_liveness(liveness: &LeaseOwnerLiveness) -> Result<String, StoreError> {
    serde_json::to_string(liveness)
        .map_err(|err| StoreError::Backend(format!("failed to encode lease liveness: {err}")))
}

fn row_to_session_execution_lease(
    session_id: &str,
    row: SessionExecutionLeaseRow,
) -> Result<SessionExecutionLease, StoreError> {
    Ok(SessionExecutionLease {
        session_id: session_id.to_string(),
        owner: row
            .owner
            .ok_or_else(|| StoreError::Backend("live session lease missing owner".to_string()))?,
        lease_token: row.lease_token.ok_or_else(|| {
            StoreError::Backend("live session lease missing lease token".to_string())
        })?,
        fencing_token: row.fencing_token,
        claimed_at_epoch_ms: row.claimed_at_ms,
        expires_at_epoch_ms: row.expires_at_ms,
    })
}

fn acquire_session_execution_lease_conn(
    conn: &Connection,
    session_id: &str,
    owner: &LeaseOwnerIdentity,
    previous_fencing_token: u64,
    now: u64,
    lease_ttl_ms: u64,
) -> Result<SessionExecutionLease, StoreError> {
    let fencing_token = previous_fencing_token.saturating_add(1);
    let lease_token = format!(
        "{}:{}:{}:{now}:{fencing_token}",
        session_id, owner.owner_id, owner.incarnation_id
    );
    let expires_at = now.saturating_add(lease_ttl_ms);
    let liveness_json = encode_liveness(&owner.liveness)?;
    conn.execute(
        "INSERT INTO session_execution_leases (
            session_id, lease_owner_id, lease_owner_incarnation_id, lease_owner_liveness_json,
            lease_token, lease_fencing_token, lease_claimed_at_ms, lease_expires_at_ms
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(session_id) DO UPDATE SET
            lease_owner_id = excluded.lease_owner_id,
            lease_owner_incarnation_id = excluded.lease_owner_incarnation_id,
            lease_owner_liveness_json = excluded.lease_owner_liveness_json,
            lease_token = excluded.lease_token,
            lease_fencing_token = excluded.lease_fencing_token,
            lease_claimed_at_ms = excluded.lease_claimed_at_ms,
            lease_expires_at_ms = excluded.lease_expires_at_ms",
        params![
            session_id,
            owner.owner_id,
            owner.incarnation_id,
            liveness_json,
            lease_token,
            fencing_token as i64,
            now as i64,
            expires_at as i64
        ],
    )
    .map_err(sqlite_error)?;
    Ok(SessionExecutionLease {
        session_id: session_id.to_string(),
        owner: owner.clone(),
        lease_token,
        fencing_token,
        claimed_at_epoch_ms: now,
        expires_at_epoch_ms: expires_at,
    })
}

fn ensure_session_execution_lease_conn(
    conn: &Connection,
    session_id: &str,
    fence: &SessionExecutionLeaseFence,
) -> Result<(), StoreError> {
    if fence.session_id != session_id {
        return Err(StoreError::SessionExecutionLeaseExpired {
            session_id: session_id.to_string(),
        });
    }
    let now = current_epoch_ms();
    let current = load_session_execution_lease_row_conn(conn, session_id)?;
    let Some(current) = current else {
        return Err(StoreError::SessionExecutionLeaseExpired {
            session_id: session_id.to_string(),
        });
    };
    if current
        .owner
        .as_ref()
        .is_some_and(|owner| owner.same_incarnation(&fence.owner))
        && current.lease_token.as_deref() == Some(fence.lease_token.as_str())
        && current.fencing_token == fence.fencing_token
        && current.expires_at_ms > now
    {
        Ok(())
    } else {
        Err(StoreError::SessionExecutionLeaseExpired {
            session_id: session_id.to_string(),
        })
    }
}

fn release_session_execution_lease_conn(
    conn: &Connection,
    completion: &SessionExecutionLeaseCompletion,
) -> Result<(), StoreError> {
    conn.execute(
        "UPDATE session_execution_leases
         SET lease_owner_id = NULL,
             lease_owner_incarnation_id = NULL,
             lease_owner_liveness_json = NULL,
             lease_token = NULL,
             lease_claimed_at_ms = 0,
             lease_expires_at_ms = 0
         WHERE session_id = ?1
           AND lease_owner_id = ?2
           AND lease_owner_incarnation_id = ?3
           AND lease_token = ?4
           AND lease_fencing_token = ?5",
        params![
            completion.session_id,
            completion.owner.owner_id,
            completion.owner.incarnation_id,
            completion.lease_token,
            completion.fencing_token as i64
        ],
    )
    .map_err(sqlite_error)?;
    Ok(())
}
