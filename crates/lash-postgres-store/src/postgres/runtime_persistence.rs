#[async_trait::async_trait]
impl RuntimePersistence for PostgresSessionStore {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn load_session(
        &self,
        scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, StoreError> {
        let Some(session_id) = self.selected_session_id().await? else {
            return Ok(None);
        };
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let Some(meta) = load_session_head_meta_tx(&mut tx, &session_id, false).await? else {
            return Ok(None);
        };
        tx.commit().await.map_err(store_sqlx_error)?;
        let leaf_node_id = match &scope {
            SessionReadScope::FullGraph => meta.leaf_node_id.clone(),
            SessionReadScope::ActivePath { leaf_node_id } => {
                leaf_node_id.clone().or_else(|| meta.leaf_node_id.clone())
            }
        };
        let graph = load_graph(
            &self.pool,
            &session_id,
            leaf_node_id.clone(),
            matches!(scope, SessionReadScope::ActivePath { .. }),
        )
        .await?;
        let checkpoint = match meta.checkpoint_ref.as_ref() {
            Some(blob_ref) => get_checkpoint(&self.pool, blob_ref).await?,
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
            token_ledger: merge_token_ledger_entries(
                load_usage_deltas(&self.pool, &session_id).await,
            ),
        }))
    }

    async fn load_node(&self, node_id: &str) -> Result<Option<SessionNodeRecord>, StoreError> {
        let json: Option<String> = if let Some(session_id) = &self.session_id {
            sqlx::query_scalar(
                "SELECT node_json FROM lash_graph_nodes
                 WHERE session_id = $1 AND node_id = $2 AND tombstoned = FALSE",
            )
            .bind(session_id)
            .bind(node_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(store_sqlx_error)?
        } else {
            sqlx::query_scalar(
                "SELECT node_json FROM lash_graph_nodes
                 WHERE node_id = $1 AND tombstoned = FALSE
                 ORDER BY session_id ASC
                 LIMIT 1",
            )
            .bind(node_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(store_sqlx_error)?
        };
        json.map(|json| store_decode_json(&json, "session graph node"))
            .transpose()
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        // Read the head WITHOUT a lock. The session execution lease is the
        // primary cross-runner serialization point; the conditional CAS write
        // below is the stale-writer backstop, so no pessimistic `FOR UPDATE`
        // lock is held across the rest of this transaction.
        let existing = load_session_head_meta_tx(&mut tx, &commit.session_id, false).await?;
        if let Some(bound_session_id) = existing.as_ref().map(|meta| meta.session_id.as_str())
            && bound_session_id != commit.session_id
        {
            return Err(StoreError::SessionBindingMismatch {
                bound_session_id: bound_session_id.to_string(),
                attempted_session_id: commit.session_id,
            });
        }
        // A session-store handle commits to exactly one session. An explicit
        // binding (`self.session_id`) is authoritative; otherwise the handle binds
        // to the first session it commits and rejects any other thereafter.
        let effective_binding = self
            .session_id
            .clone()
            .or_else(|| self.bound_session.get().cloned());
        if let Some(bound_session_id) = &effective_binding
            && commit.session_id != *bound_session_id
        {
            return Err(StoreError::SessionBindingMismatch {
                bound_session_id: bound_session_id.clone(),
                attempted_session_id: commit.session_id,
            });
        }
        if self.session_id.is_none() {
            let _ = self.bound_session.set(commit.session_id.clone());
        }
        if let Some(completed) = &commit.turn_commit {
            if completed.session_id != commit.session_id {
                return Err(StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
            let prior = sqlx::query(
                "SELECT turn_commit_hash, result_json
                 FROM lash_runtime_turn_commits
                 WHERE session_id = $1 AND turn_id = $2",
            )
            .bind(&completed.session_id)
            .bind(&completed.turn_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
            if let Some(row) = prior {
                let hash: String = row.get(0);
                let result_json: String = row.get(1);
                if hash == completed.turn_commit_hash {
                    let result = store_decode_json(&result_json, "runtime turn commit result")?;
                    if let Some(completion) = commit.release_session_execution_lease.as_ref() {
                        release_session_execution_lease_tx(&mut tx, completion).await?;
                    }
                    tx.commit().await.map_err(store_sqlx_error)?;
                    return Ok(result);
                }
                return Err(StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
        }
        let Some(session_execution_lease) = commit.session_execution_lease.as_ref() else {
            return Err(StoreError::SessionExecutionLeaseExpired {
                session_id: commit.session_id.clone(),
            });
        };
        ensure_session_execution_lease_tx(&mut tx, &commit.session_id, session_execution_lease)
            .await?;
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
            ensure_queued_work_completion_tx(&mut tx, completed).await?;
        }
        for completed in &commit.completed_turn_input_claims {
            if completed.session_id != commit.session_id {
                return Err(StoreError::TurnInputClaimExpired {
                    session_id: completed.session_id.clone(),
                    claim_id: completed.claim_id.clone(),
                });
            }
            ensure_turn_input_completion_tx(&mut tx, completed).await?;
        }
        let (checkpoint_ref, manifest) = put_checkpoint_tx(&mut tx, &commit.checkpoint).await?;
        for entry in &commit.usage_deltas {
            sqlx::query("INSERT INTO lash_usage_deltas (session_id, entry_json) VALUES ($1, $2)")
                .bind(&commit.session_id)
                .bind(encode_json(entry))
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
        }
        let leaf_node_id = match &commit.graph {
            GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
            GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                for node in nodes {
                    sqlx::query(
                        "INSERT INTO lash_graph_nodes (session_id, node_id, node_json)
                         VALUES ($1, $2, $3)
                         ON CONFLICT (session_id, node_id) DO UPDATE SET
                            node_json = EXCLUDED.node_json,
                            tombstoned = FALSE",
                    )
                    .bind(&commit.session_id)
                    .bind(&node.node_id)
                    .bind(encode_json(node))
                    .execute(&mut *tx)
                    .await
                    .map_err(store_sqlx_error)?;
                }
                leaf_node_id.clone()
            }
            GraphCommitDelta::ReplaceFull(graph) => {
                sqlx::query("DELETE FROM lash_graph_nodes WHERE session_id = $1")
                    .bind(&commit.session_id)
                    .execute(&mut *tx)
                    .await
                    .map_err(store_sqlx_error)?;
                for node in &graph.nodes {
                    sqlx::query(
                        "INSERT INTO lash_graph_nodes (session_id, node_id, node_json)
                         VALUES ($1, $2, $3)",
                    )
                    .bind(&commit.session_id)
                    .bind(&node.node_id)
                    .bind(encode_json(node))
                    .execute(&mut *tx)
                    .await
                    .map_err(store_sqlx_error)?;
                }
                graph.leaf_node_id.clone()
            }
        };
        let graph_node_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM lash_graph_nodes WHERE session_id = $1 AND tombstoned = FALSE",
        )
        .bind(&commit.session_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let next_revision = actual_revision + 1;
        let meta = SessionHeadMeta {
            schema_version: lash_core::store::SESSION_HEAD_META_SCHEMA_VERSION,
            session_id: commit.session_id.clone(),
            head_revision: next_revision,
            config: commit.config.clone(),
            agent_frames: commit.agent_frames.clone(),
            current_agent_frame_id: commit.current_agent_frame_id.clone(),
            checkpoint_ref: Some(checkpoint_ref.clone()),
            leaf_node_id,
            graph_node_count: graph_node_count as usize,
            token_ledger: Vec::new(),
        };
        // Optimistic CAS on the head revision. The `WHERE head_revision = $5`
        // guard makes the write succeed only if no concurrent committer moved the
        // head since our unlocked read above. A brand-new session inserts (no
        // conflict); an existing one updates only when the revision still matches.
        let head_write = sqlx::query(
            "INSERT INTO lash_sessions (session_id, head_revision, head_json, checkpoint_ref)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (session_id) DO UPDATE SET
                head_revision = EXCLUDED.head_revision,
                head_json = EXCLUDED.head_json,
                checkpoint_ref = EXCLUDED.checkpoint_ref
             WHERE lash_sessions.head_revision = $5",
        )
        .bind(&commit.session_id)
        .bind(next_revision as i64)
        .bind(encode_json(&meta))
        .bind(checkpoint_ref.as_str())
        .bind(actual_revision as i64)
        .execute(&mut *tx)
        .await;
        let head_write = match head_write {
            Ok(result) => result,
            Err(err) if is_contention_error(&err) => {
                // The head row is contended by a concurrent committer (lock
                // timeout / serialization failure / deadlock). That is a conflict,
                // not an opaque backend error: surface it so the caller reloads
                // and retries. The tx is now aborted; returning drops it.
                return Err(StoreError::HeadRevisionConflict {
                    expected: commit.expected_head_revision.or(Some(actual_revision)),
                    actual: actual_revision,
                });
            }
            Err(err) => return Err(store_sqlx_error(err)),
        };
        if head_write.rows_affected() == 0 {
            // A concurrent commit won the race: the head no longer matches the
            // revision we read. Re-read the now-current revision for an accurate
            // report, then drop `tx` (auto-rollback), discarding this attempt's
            // node/usage writes; the caller reloads and retries.
            let actual_now = sqlx::query_scalar::<_, i64>(
                "SELECT head_revision FROM lash_sessions WHERE session_id = $1",
            )
            .bind(&commit.session_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(store_sqlx_error)?
            .map_or(actual_revision, |revision| revision as u64);
            return Err(StoreError::HeadRevisionConflict {
                expected: commit.expected_head_revision.or(Some(actual_revision)),
                actual: actual_now,
            });
        }
        for completed in &commit.completed_queue_claims {
            for batch_id in &completed.batch_ids {
                sqlx::query(
                    "DELETE FROM lash_queued_work_batches
                     WHERE session_id = $1 AND batch_id = $2 AND claim_id = $3 AND claim_token = $4",
                )
                .bind(&completed.session_id)
                .bind(batch_id)
                .bind(&completed.claim_id)
                .bind(&completed.lease_token)
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
            }
        }
        for completed in &commit.completed_turn_input_claims {
            for input_id in &completed.input_ids {
                sqlx::query(
                    "UPDATE lash_pending_turn_inputs
                     SET state = $5,
                         claim_id = NULL,
                         claim_owner_id = NULL,
                         claim_owner_incarnation_id = NULL,
                         claim_owner_liveness_json = NULL,
                         claim_token = NULL,
                         claim_claimed_at_ms = 0,
                         claim_expires_at_ms = 0
                     WHERE session_id = $1 AND input_id = $2 AND claim_id = $3 AND claim_token = $4",
                )
                .bind(&completed.session_id)
                .bind(input_id)
                .bind(&completed.claim_id)
                .bind(&completed.lease_token)
                .bind(lash_core::TurnInputState::Completed.as_str())
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
            }
        }
        if let Some(turn_id) = commit.interrupted_turn_input_turn_id.as_deref() {
            let rows = sqlx::query(
                "SELECT enqueue_seq, input_id, session_id, source_key, ingress_json,
                        state, input_json, enqueued_at_ms, claim_id, claim_fencing_token,
                        claim_owner_id, claim_owner_incarnation_id,
                        claim_owner_liveness_json, claim_token, claim_expires_at_ms
                 FROM lash_pending_turn_inputs
                 WHERE session_id = $1 AND state = $2
                 ORDER BY enqueue_seq ASC
                 FOR UPDATE",
            )
            .bind(&commit.session_id)
            .bind(lash_core::TurnInputState::PendingActive.as_str())
            .fetch_all(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
            let mut input_ids = Vec::new();
            for row in rows {
                let input = pending_turn_input_from_row(pending_turn_input_row(row)?)?;
                if input
                    .ingress
                    .active_turn_id()
                    .is_some_and(|active| active == turn_id)
                {
                    input_ids.push(input.input_id);
                }
            }
            for input_id in input_ids {
                sqlx::query(
                    "UPDATE lash_pending_turn_inputs
                     SET state = $3,
                         ingress_json = $4,
                         claim_id = NULL,
                         claim_owner_id = NULL,
                         claim_owner_incarnation_id = NULL,
                         claim_owner_liveness_json = NULL,
                         claim_token = NULL,
                         claim_claimed_at_ms = 0,
                         claim_expires_at_ms = 0
                     WHERE session_id = $1 AND input_id = $2",
                )
                .bind(&commit.session_id)
                .bind(input_id)
                .bind(lash_core::TurnInputState::DeferredNextTurn.as_str())
                .bind(encode_json(&lash_core::TurnInputIngress::NextTurn))
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
            }
        }
        commit_attachment_refs_tx(
            &mut tx,
            &commit.session_id,
            &commit.committed_attachment_ids,
        )
        .await?;
        let result = RuntimeCommitResult {
            head_revision: next_revision,
            checkpoint_ref,
            manifest,
        };
        if let Some(completed) = &commit.turn_commit {
            sqlx::query(
                "INSERT INTO lash_runtime_turn_commits (
                    session_id, turn_id, turn_commit_hash, result_json, committed_at_ms
                 )
                 VALUES ($1, $2, $3, $4, $5)",
            )
            .bind(&completed.session_id)
            .bind(&completed.turn_id)
            .bind(&completed.turn_commit_hash)
            .bind(encode_json(&result))
            .bind(current_epoch_ms() as i64)
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
        }
        if let Some(completion) = commit.release_session_execution_lease.as_ref() {
            release_session_execution_lease_tx(&mut tx, completion).await?;
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(result)
    }

    async fn try_claim_session_execution_lease(
        &self,
        session_id: &str,
        owner: &LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLeaseClaimOutcome, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        lock_session_execution_lease_tx(&mut tx, session_id).await?;
        let now = current_epoch_ms();
        let current = load_session_execution_lease_tx(&mut tx, session_id).await?;
        if current
            .as_ref()
            .is_some_and(|lease| lease.lease_token.is_some() && lease.expires_at_ms > now)
        {
            let current = current.expect("checked current lease is present");
            if current
                .owner
                .as_ref()
                .is_some_and(|current_owner| current_owner.same_incarnation(owner))
            {
                let expires_at = now.saturating_add(lease_ttl_ms);
                sqlx::query(
                    "UPDATE lash_session_execution_leases
                     SET lease_expires_at_ms = $2
                     WHERE session_id = $1",
                )
                .bind(session_id)
                .bind(expires_at as i64)
                .execute(&mut *tx)
                .await
                .map_err(store_sqlx_error)?;
                tx.commit().await.map_err(store_sqlx_error)?;
                return Ok(SessionExecutionLeaseClaimOutcome::Acquired(
                    SessionExecutionLease {
                        session_id: session_id.to_string(),
                        owner: owner.clone(),
                        lease_token: current.lease_token.expect("live lease token set"),
                        fencing_token: current.fencing_token,
                        claimed_at_epoch_ms: current.claimed_at_ms,
                        expires_at_epoch_ms: expires_at,
                    },
                ));
            }
            let holder = row_to_session_execution_lease(session_id, current)?;
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(SessionExecutionLeaseClaimOutcome::Busy { holder });
        }
        let previous_fencing_token = current.as_ref().map_or(0, |lease| lease.fencing_token);
        let lease = acquire_session_execution_lease_tx(
            &mut tx,
            session_id,
            owner,
            previous_fencing_token,
            now,
            lease_ttl_ms,
        )
        .await?;
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(SessionExecutionLeaseClaimOutcome::Acquired(lease))
    }

    async fn reclaim_session_execution_lease(
        &self,
        session_id: &str,
        owner: &LeaseOwnerIdentity,
        observed_holder: &SessionExecutionLeaseFence,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLeaseClaimOutcome, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        lock_session_execution_lease_tx(&mut tx, session_id).await?;
        let now = current_epoch_ms();
        let current = load_session_execution_lease_tx(&mut tx, session_id).await?;
        let Some(current) = current else {
            let lease = acquire_session_execution_lease_tx(
                &mut tx,
                session_id,
                owner,
                0,
                now,
                lease_ttl_ms,
            )
            .await?;
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(SessionExecutionLeaseClaimOutcome::Acquired(lease));
        };
        if current.lease_token.is_none() || current.expires_at_ms <= now {
            let lease = acquire_session_execution_lease_tx(
                &mut tx,
                session_id,
                owner,
                current.fencing_token,
                now,
                lease_ttl_ms,
            )
            .await?;
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(SessionExecutionLeaseClaimOutcome::Acquired(lease));
        }
        let holder = row_to_session_execution_lease(session_id, current)?;
        if observed_holder.session_id == session_id
            && holder.owner.same_incarnation(&observed_holder.owner)
            && holder.lease_token == observed_holder.lease_token
            && holder.fencing_token == observed_holder.fencing_token
            && holder.owner.is_definitely_dead_for_claimant(owner)
        {
            let fencing_token = holder.fencing_token.saturating_add(1);
            let lease_token = format!(
                "{}:{}:{}:{now}:{fencing_token}",
                session_id, owner.owner_id, owner.incarnation_id
            );
            let expires_at = now.saturating_add(lease_ttl_ms);
            let liveness_json = encode_liveness(&owner.liveness)?;
            let changed = sqlx::query(
                "UPDATE lash_session_execution_leases
                 SET lease_owner_id = $1,
                     lease_owner_incarnation_id = $2,
                     lease_owner_liveness_json = $3,
                     lease_token = $4,
                     lease_fencing_token = $5,
                     lease_claimed_at_ms = $6,
                     lease_expires_at_ms = $7
                 WHERE session_id = $8
                   AND lease_owner_id = $9
                   AND lease_owner_incarnation_id = $10
                   AND lease_token = $11
                   AND lease_fencing_token = $12",
            )
            .bind(&owner.owner_id)
            .bind(&owner.incarnation_id)
            .bind(&liveness_json)
            .bind(&lease_token)
            .bind(fencing_token as i64)
            .bind(now as i64)
            .bind(expires_at as i64)
            .bind(session_id)
            .bind(&observed_holder.owner.owner_id)
            .bind(&observed_holder.owner.incarnation_id)
            .bind(&observed_holder.lease_token)
            .bind(observed_holder.fencing_token as i64)
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?
            .rows_affected();
            if changed == 1 {
                let lease = SessionExecutionLease {
                    session_id: session_id.to_string(),
                    owner: owner.clone(),
                    lease_token,
                    fencing_token,
                    claimed_at_epoch_ms: now,
                    expires_at_epoch_ms: expires_at,
                };
                tx.commit().await.map_err(store_sqlx_error)?;
                return Ok(SessionExecutionLeaseClaimOutcome::Acquired(lease));
            }
            let current = load_session_execution_lease_tx(&mut tx, session_id).await?;
            if current
                .as_ref()
                .is_some_and(|lease| lease.lease_token.is_some() && lease.expires_at_ms > now)
            {
                let current = current.expect("checked current lease is present");
                let holder = row_to_session_execution_lease(session_id, current)?;
                tx.commit().await.map_err(store_sqlx_error)?;
                return Ok(SessionExecutionLeaseClaimOutcome::Busy { holder });
            }
            let previous_fencing_token = current.as_ref().map_or(0, |lease| lease.fencing_token);
            let lease = acquire_session_execution_lease_tx(
                &mut tx,
                session_id,
                owner,
                previous_fencing_token,
                now,
                lease_ttl_ms,
            )
            .await?;
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(SessionExecutionLeaseClaimOutcome::Acquired(lease));
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(SessionExecutionLeaseClaimOutcome::Busy { holder })
    }

    async fn renew_session_execution_lease(
        &self,
        fence: &SessionExecutionLeaseFence,
        lease_ttl_ms: u64,
    ) -> Result<SessionExecutionLease, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let now = current_epoch_ms();
        let current = load_session_execution_lease_tx(&mut tx, &fence.session_id).await?;
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
        sqlx::query(
            "UPDATE lash_session_execution_leases
             SET lease_expires_at_ms = $5
             WHERE session_id = $1
               AND lease_owner_id = $2
               AND lease_owner_incarnation_id = $3
               AND lease_token = $4
               AND lease_fencing_token = $6",
        )
        .bind(&fence.session_id)
        .bind(&fence.owner.owner_id)
        .bind(&fence.owner.incarnation_id)
        .bind(&fence.lease_token)
        .bind(expires_at as i64)
        .bind(fence.fencing_token as i64)
        .execute(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(SessionExecutionLease {
            session_id: fence.session_id.clone(),
            owner: fence.owner.clone(),
            lease_token: fence.lease_token.clone(),
            fencing_token: fence.fencing_token,
            claimed_at_epoch_ms: current.claimed_at_ms,
            expires_at_epoch_ms: expires_at,
        })
    }

    async fn release_session_execution_lease(
        &self,
        completion: &SessionExecutionLeaseCompletion,
    ) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        release_session_execution_lease_tx(&mut tx, completion).await?;
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(())
    }

    async fn enqueue_queued_work(
        &self,
        batch: QueuedWorkBatchDraft,
    ) -> Result<QueuedWorkBatch, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        if let Some(source_key) = batch.source_key.as_deref() {
            let existing_id: Option<String> = sqlx::query_scalar(
                "SELECT batch_id FROM lash_queued_work_batches
                 WHERE session_id = $1 AND source_key = $2",
            )
            .bind(&batch.session_id)
            .bind(source_key)
            .fetch_optional(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
            if let Some(batch_id) = existing_id {
                let existing = load_queued_batch(&mut tx, &batch_id)
                    .await?
                    .ok_or_else(|| {
                        StoreError::Backend("queued work source row disappeared".to_string())
                    })?;
                tx.commit().await.map_err(store_sqlx_error)?;
                return Ok(existing);
            }
        }
        let now = current_epoch_ms();
        let batch_id = derive_batch_id(&batch.session_id, batch.source_key.as_deref(), now, None);
        let row = sqlx::query_scalar::<_, i64>(
            "INSERT INTO lash_queued_work_batches (
                batch_id, session_id, source_key, delivery_policy, slot_policy,
                merge_key_json, available_at_ms, enqueued_at_ms
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             RETURNING enqueue_seq",
        )
        .bind(&batch_id)
        .bind(&batch.session_id)
        .bind(&batch.source_key)
        .bind(batch.delivery_policy.as_str())
        .bind(batch.slot_policy.as_str())
        .bind(encode_json(&batch.merge_key))
        .bind(batch.available_at_ms as i64)
        .bind(now as i64)
        .fetch_one(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        for (index, payload) in batch.payloads.iter().enumerate() {
            let item_id = format!("{batch_id}:item:{index}");
            sqlx::query(
                "INSERT INTO lash_queued_work_items (batch_id, item_index, item_id, payload_json)
                 VALUES ($1, $2, $3, $4)",
            )
            .bind(&batch_id)
            .bind(index as i32)
            .bind(item_id)
            .bind(encode_json(payload))
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
        }
        let queued = load_queued_batch(&mut tx, &batch_id)
            .await?
            .ok_or_else(|| StoreError::Backend("queued work insert disappeared".to_string()))?;
        debug_assert_eq!(queued.enqueue_seq, row as u64);
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(queued)
    }

    async fn claim_leading_ready_session_command(
        &self,
        session_id: &str,
        session_execution_lease: &SessionExecutionLeaseFence,
        owner: &LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<Option<QueuedWorkClaim>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        ensure_session_execution_lease_tx(&mut tx, session_id, session_execution_lease).await?;
        let now = current_epoch_ms();
        let rows = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                    claim_owner_liveness_json, claim_token, claim_expires_at_ms
             FROM lash_queued_work_batches
             WHERE session_id = $1
               AND available_at_ms <= $2
             ORDER BY enqueue_seq ASC
             LIMIT $3
             FOR UPDATE SKIP LOCKED",
        )
        .bind(session_id)
        .bind(now as i64)
        .bind(claim_scan_limit(1))
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let mut selected = Vec::new();
        for row in rows {
            let row = queued_batch_row(row)?;
            if row.claim_token.is_none()
                || row.claim_expires_at_ms <= now
                || row
                    .claim_owner
                    .as_ref()
                    .is_some_and(|holder| holder.is_definitely_dead_for_claimant(owner))
            {
                selected.push(row);
            }
        }
        let mut selected_batches = Vec::new();
        for row in &selected {
            selected_batches.push(queued_work_batch_from_row(&mut tx, row.clone()).await?);
        }
        let candidates = selected
            .iter()
            .zip(selected_batches.iter())
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
                    delivery_policy: row.delivery_policy,
                    slot_policy: row.slot_policy,
                    merge_key: row.merge_key.clone(),
                })
            })
            .collect::<Result<Vec<_>, StoreError>>()?;
        let selected_len = select_leading_session_command(&candidates);
        if selected_len == 0 {
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(None);
        }
        selected.truncate(selected_len);
        selected_batches.truncate(selected_len);
        let lease =
            QueuedWorkClaimLease::derive(&candidates[0], session_id, owner, now, lease_ttl_ms);
        let liveness_json = encode_liveness(&owner.liveness)?;
        for row in &selected {
            let changed = sqlx::query(
                "UPDATE lash_queued_work_batches
                 SET claim_id = $3,
                     claim_owner_id = $4,
                     claim_owner_incarnation_id = $5,
                     claim_owner_liveness_json = $6,
                     claim_token = $7,
                     claim_fencing_token = claim_fencing_token + 1,
                     claim_claimed_at_ms = $8,
                     claim_expires_at_ms = $9
                 WHERE session_id = $1
                   AND batch_id = $2
                   AND (
                        claim_token IS NULL
                        OR claim_expires_at_ms <= $8
                        OR (
                            claim_token = $10
                            AND claim_owner_id = $11
                            AND claim_owner_incarnation_id = $12
                        )
                   )",
            )
            .bind(session_id)
            .bind(&row.batch_id)
            .bind(&lease.claim_id)
            .bind(&owner.owner_id)
            .bind(&owner.incarnation_id)
            .bind(&liveness_json)
            .bind(&lease.lease_token)
            .bind(now as i64)
            .bind(lease.expires_at_epoch_ms as i64)
            .bind(&row.claim_token)
            .bind(
                row.claim_owner
                    .as_ref()
                    .map(|owner| owner.owner_id.as_str()),
            )
            .bind(
                row.claim_owner
                    .as_ref()
                    .map(|owner| owner.incarnation_id.as_str()),
            )
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?
            .rows_affected();
            if changed == 0 {
                tx.rollback().await.map_err(store_sqlx_error)?;
                return Ok(None);
            }
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(Some(QueuedWorkClaim {
            session_id: session_id.to_string(),
            claim_id: lease.claim_id,
            owner: owner.clone(),
            lease_token: lease.lease_token,
            fencing_token: lease.fencing_token,
            claimed_at_epoch_ms: lease.claimed_at_epoch_ms,
            expires_at_epoch_ms: lease.expires_at_epoch_ms,
            batches: selected_batches,
        }))
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
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        ensure_session_execution_lease_tx(&mut tx, session_id, session_execution_lease).await?;
        let now = current_epoch_ms();
        let rows = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                    claim_owner_liveness_json, claim_token, claim_expires_at_ms
             FROM lash_queued_work_batches
             WHERE session_id = $1
               AND available_at_ms <= $2
             ORDER BY enqueue_seq ASC
             LIMIT $3
             FOR UPDATE SKIP LOCKED",
        )
        .bind(session_id)
        .bind(now as i64)
        .bind(claim_scan_limit(max_batches))
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let mut selected = Vec::new();
        for row in rows {
            let row = queued_batch_row(row)?;
            if row.claim_token.is_none()
                || row.claim_expires_at_ms <= now
                || row
                    .claim_owner
                    .as_ref()
                    .is_some_and(|holder| holder.is_definitely_dead_for_claimant(owner))
            {
                selected.push(row);
            }
        }
        let mut selected_batches = Vec::new();
        for row in &selected {
            selected_batches.push(queued_work_batch_from_row(&mut tx, row.clone()).await?);
        }
        let candidates = selected
            .iter()
            .zip(selected_batches.iter())
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
                    delivery_policy: row.delivery_policy,
                    slot_policy: row.slot_policy,
                    merge_key: row.merge_key.clone(),
                })
            })
            .collect::<Result<Vec<_>, StoreError>>()?;
        let selected_len = select_turn_work_claim_prefix(&candidates, boundary, max_batches);
        if selected_len == 0 {
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(None);
        }
        selected.truncate(selected_len);
        selected_batches.truncate(selected_len);
        let lease =
            QueuedWorkClaimLease::derive(&candidates[0], session_id, owner, now, lease_ttl_ms);
        let liveness_json = encode_liveness(&owner.liveness)?;
        for row in &selected {
            let changed = sqlx::query(
                "UPDATE lash_queued_work_batches
                 SET claim_id = $3,
                     claim_owner_id = $4,
                     claim_owner_incarnation_id = $5,
                     claim_owner_liveness_json = $6,
                     claim_token = $7,
                     claim_fencing_token = claim_fencing_token + 1,
                     claim_claimed_at_ms = $8,
                     claim_expires_at_ms = $9
                 WHERE session_id = $1
                   AND batch_id = $2
                   AND (
                        claim_token IS NULL
                        OR claim_expires_at_ms <= $8
                        OR (
                            claim_token = $10
                            AND claim_owner_id = $11
                            AND claim_owner_incarnation_id = $12
                        )
                   )",
            )
            .bind(session_id)
            .bind(&row.batch_id)
            .bind(&lease.claim_id)
            .bind(&owner.owner_id)
            .bind(&owner.incarnation_id)
            .bind(&liveness_json)
            .bind(&lease.lease_token)
            .bind(now as i64)
            .bind(lease.expires_at_epoch_ms as i64)
            .bind(&row.claim_token)
            .bind(
                row.claim_owner
                    .as_ref()
                    .map(|owner| owner.owner_id.as_str()),
            )
            .bind(
                row.claim_owner
                    .as_ref()
                    .map(|owner| owner.incarnation_id.as_str()),
            )
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?
            .rows_affected();
            if changed == 0 {
                tx.rollback().await.map_err(store_sqlx_error)?;
                return Ok(None);
            }
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(Some(QueuedWorkClaim {
            session_id: session_id.to_string(),
            claim_id: lease.claim_id,
            owner: owner.clone(),
            lease_token: lease.lease_token,
            fencing_token: lease.fencing_token,
            claimed_at_epoch_ms: lease.claimed_at_epoch_ms,
            expires_at_epoch_ms: lease.expires_at_epoch_ms,
            batches: selected_batches,
        }))
    }

    async fn renew_queued_work_claim(
        &self,
        claim: &QueuedWorkClaim,
        lease_ttl_ms: u64,
    ) -> Result<QueuedWorkClaim, StoreError> {
        let expires_at = current_epoch_ms().saturating_add(lease_ttl_ms);
        let changed = sqlx::query(
            "UPDATE lash_queued_work_batches
             SET claim_expires_at_ms = $4
             WHERE session_id = $1 AND claim_id = $2 AND claim_token = $3",
        )
        .bind(&claim.session_id)
        .bind(&claim.claim_id)
        .bind(&claim.lease_token)
        .bind(expires_at as i64)
        .execute(&self.pool)
        .await
        .map_err(store_sqlx_error)?
        .rows_affected();
        renewed_claim(claim, changed as usize, expires_at)
    }

    async fn abandon_queued_work_claim(&self, claim: &QueuedWorkClaim) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE lash_queued_work_batches
             SET claim_id = NULL,
                 claim_owner_id = NULL,
                 claim_owner_incarnation_id = NULL,
                 claim_owner_liveness_json = NULL,
                 claim_token = NULL,
                 claim_claimed_at_ms = 0,
                 claim_expires_at_ms = 0
             WHERE session_id = $1 AND claim_id = $2 AND claim_token = $3",
        )
        .bind(&claim.session_id)
        .bind(&claim.claim_id)
        .bind(&claim.lease_token)
        .execute(&self.pool)
        .await
        .map_err(store_sqlx_error)?;
        Ok(())
    }

    async fn cancel_queued_work_batch(
        &self,
        session_id: &str,
        batch_id: &str,
    ) -> Result<Option<QueuedWorkBatch>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let now = current_epoch_ms();
        let row = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                    claim_owner_liveness_json, claim_token, claim_expires_at_ms
             FROM lash_queued_work_batches
             WHERE session_id = $1
               AND batch_id = $2
               AND (claim_token IS NULL OR claim_expires_at_ms <= $3)
             FOR UPDATE",
        )
        .bind(session_id)
        .bind(batch_id)
        .bind(now as i64)
        .fetch_optional(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let Some(row) = row else {
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(None);
        };
        let batch = queued_work_batch_from_row(&mut tx, queued_batch_row(row)?).await?;
        sqlx::query("DELETE FROM lash_queued_work_batches WHERE batch_id = $1")
            .bind(batch_id)
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(Some(batch))
    }

    async fn list_queued_work(&self, session_id: &str) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let rows = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                    claim_owner_liveness_json, claim_token, claim_expires_at_ms
             FROM lash_queued_work_batches
             WHERE session_id = $1
             ORDER BY enqueue_seq ASC",
        )
        .bind(session_id)
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let mut batches = Vec::new();
        for row in rows {
            batches.push(queued_work_batch_from_row(&mut tx, queued_batch_row(row)?).await?);
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(batches)
    }

    async fn list_pending_queued_work(
        &self,
        session_id: &str,
    ) -> Result<Vec<QueuedWorkBatch>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let now = current_epoch_ms();
        let rows = sqlx::query(
            "SELECT enqueue_seq, batch_id, session_id, source_key, delivery_policy,
                    slot_policy, merge_key_json, available_at_ms, enqueued_at_ms,
                    claim_fencing_token, claim_owner_id, claim_owner_incarnation_id,
                    claim_owner_liveness_json, claim_token, claim_expires_at_ms
             FROM lash_queued_work_batches
             WHERE session_id = $1
               AND (claim_token IS NULL OR claim_expires_at_ms <= $2)
             ORDER BY enqueue_seq ASC",
        )
        .bind(session_id)
        .bind(now as i64)
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let mut batches = Vec::new();
        for row in rows {
            batches.push(queued_work_batch_from_row(&mut tx, queued_batch_row(row)?).await?);
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(batches)
    }

    async fn enqueue_pending_turn_input(
        &self,
        draft: lash_core::PendingTurnInputDraft,
    ) -> Result<lash_core::PendingTurnInput, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let now = current_epoch_ms();
        let input_id = draft.input_id.clone().unwrap_or_else(|| {
            derive_pending_turn_input_id(&draft.session_id, draft.source_key.as_deref(), now)
        });
        let state = match draft.ingress {
            lash_core::TurnInputIngress::ActiveTurn { .. } => {
                lash_core::TurnInputState::PendingActive
            }
            lash_core::TurnInputIngress::NextTurn => lash_core::TurnInputState::DeferredNextTurn,
        };
        let ingress_json = encode_json(&draft.ingress);
        let input_json = encode_json(&draft.input);
        let input = if let Some(source_key) = draft.source_key.as_deref() {
            let row = sqlx::query(
                "INSERT INTO lash_pending_turn_inputs (
                    input_id, session_id, source_key, ingress_json, state, input_json, enqueued_at_ms
                 )
                 VALUES ($1, $2, $3, $4, $5, $6, $7)
                 ON CONFLICT (session_id, source_key) DO UPDATE
                 SET source_key = lash_pending_turn_inputs.source_key
                 RETURNING enqueue_seq, input_id, session_id, source_key, ingress_json,
                           state, input_json, enqueued_at_ms, claim_id, claim_fencing_token,
                           claim_owner_id, claim_owner_incarnation_id,
                           claim_owner_liveness_json, claim_token, claim_expires_at_ms",
            )
            .bind(&input_id)
            .bind(&draft.session_id)
            .bind(source_key)
            .bind(&ingress_json)
            .bind(state.as_str())
            .bind(&input_json)
            .bind(now as i64)
            .fetch_one(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
            let input = pending_turn_input_from_row(pending_turn_input_row(row)?)?;
            if !draft.submitted_content_matches(&input).map_err(|err| {
                StoreError::Backend(format!(
                    "failed to compare pending turn input submission: {err}"
                ))
            })? {
                return Err(StoreError::PendingTurnInputSourceKeyConflict {
                    session_id: draft.session_id.clone(),
                    source_key: source_key.to_string(),
                    existing_input_id: input.input_id.clone(),
                });
            }
            input
        } else {
            sqlx::query(
                "INSERT INTO lash_pending_turn_inputs (
                    input_id, session_id, source_key, ingress_json, state, input_json, enqueued_at_ms
                 )
                 VALUES ($1, $2, $3, $4, $5, $6, $7)",
            )
            .bind(&input_id)
            .bind(&draft.session_id)
            .bind(&draft.source_key)
            .bind(&ingress_json)
            .bind(state.as_str())
            .bind(&input_json)
            .bind(now as i64)
            .execute(&mut *tx)
            .await
            .map_err(store_sqlx_error)?;
            load_pending_turn_input(&mut tx, &draft.session_id, &input_id)
                .await?
                .ok_or_else(|| {
                    StoreError::Backend("pending turn input insert disappeared".to_string())
                })?
        };
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(input)
    }

    async fn list_pending_turn_inputs(
        &self,
        session_id: &str,
    ) -> Result<Vec<lash_core::PendingTurnInput>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let now = current_epoch_ms();
        let rows = sqlx::query(
            "SELECT enqueue_seq, input_id, session_id, source_key, ingress_json,
                    state, input_json, enqueued_at_ms, claim_id, claim_fencing_token,
                    claim_owner_id, claim_owner_incarnation_id,
                    claim_owner_liveness_json, claim_token, claim_expires_at_ms
             FROM lash_pending_turn_inputs
             WHERE session_id = $1
               AND state IN ($2, $3)
               AND (claim_token IS NULL OR claim_expires_at_ms <= $4)
             ORDER BY enqueue_seq ASC",
        )
        .bind(session_id)
        .bind(lash_core::TurnInputState::PendingActive.as_str())
        .bind(lash_core::TurnInputState::DeferredNextTurn.as_str())
        .bind(now as i64)
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let inputs = rows
            .into_iter()
            .map(pending_turn_input_row)
            .map(|row| row.and_then(pending_turn_input_from_row))
            .collect::<Result<Vec<_>, StoreError>>()?;
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(inputs)
    }

    async fn cancel_pending_turn_input(
        &self,
        session_id: &str,
        input_id: &str,
    ) -> Result<lash_core::PendingTurnInputCancelOutcome, StoreError> {
        let target = lash_core::PendingTurnInputCancelTarget::input_id(input_id);
        let targets = vec![target];
        let mut outcomes = self.cancel_pending_turn_inputs(session_id, &targets).await?;
        Ok(outcomes
            .pop()
            .map(|result| result.outcome)
            .unwrap_or(lash_core::PendingTurnInputCancelOutcome::NotFound))
    }

    async fn cancel_pending_turn_inputs(
        &self,
        session_id: &str,
        targets: &[lash_core::PendingTurnInputCancelTarget],
    ) -> Result<Vec<lash_core::PendingTurnInputCancelResult>, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let targets = targets.to_vec();
        let now = current_epoch_ms();
        let mut results = Vec::with_capacity(targets.len());
        for target in targets {
            let outcome =
                match load_pending_turn_input_row_by_target_tx(&mut tx, session_id, &target, true)
                    .await?
                {
                    Some(row) => cancel_pending_turn_input_row_tx(&mut tx, row, now).await?,
                    None => lash_core::PendingTurnInputCancelOutcome::NotFound,
                };
            results.push(lash_core::PendingTurnInputCancelResult { target, outcome });
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(results)
    }

    async fn cancel_pending_turn_input_suffix(
        &self,
        session_id: &str,
        anchor: &lash_core::PendingTurnInputCancelTarget,
    ) -> Result<lash_core::PendingTurnInputSuffixCancelOutcome, StoreError> {
        let mut tx = self.pool.begin().await.map_err(store_sqlx_error)?;
        let anchor = anchor.clone();
        let now = current_epoch_ms();
        let Some(anchor_row) =
            load_pending_turn_input_row_by_target_tx(&mut tx, session_id, &anchor, true).await?
        else {
            tx.commit().await.map_err(store_sqlx_error)?;
            return Ok(
                lash_core::PendingTurnInputSuffixCancelOutcome::AnchorNotFound { anchor },
            );
        };
        let rows = sqlx::query(
            "SELECT enqueue_seq, input_id, session_id, source_key, ingress_json,
                    state, input_json, enqueued_at_ms, claim_id, claim_fencing_token,
                    claim_owner_id, claim_owner_incarnation_id,
                    claim_owner_liveness_json, claim_token, claim_expires_at_ms
             FROM lash_pending_turn_inputs
             WHERE session_id = $1 AND enqueue_seq >= $2
             ORDER BY enqueue_seq ASC
             FOR UPDATE",
        )
        .bind(session_id)
        .bind(anchor_row.enqueue_seq as i64)
        .fetch_all(&mut *tx)
        .await
        .map_err(store_sqlx_error)?;
        let mut outcomes = Vec::with_capacity(rows.len());
        for row in rows {
            outcomes.push(
                cancel_pending_turn_input_row_tx(&mut tx, pending_turn_input_row(row)?, now)
                    .await?,
            );
        }
        tx.commit().await.map_err(store_sqlx_error)?;
        Ok(lash_core::PendingTurnInputSuffixCancelOutcome::Outcomes {
            anchor,
            outcomes,
        })
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
        claim_pending_turn_inputs_postgres(
            &self.pool,
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
        claim_pending_turn_inputs_postgres(
            &self.pool,
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
        let restored_state = match claim.mode {
            lash_core::TurnInputClaimMode::ActiveTurn { .. } => {
                lash_core::TurnInputState::PendingActive
            }
            lash_core::TurnInputClaimMode::NextTurn => {
                lash_core::TurnInputState::DeferredNextTurn
            }
        };
        sqlx::query(
            "UPDATE lash_pending_turn_inputs
             SET state = CASE
                     WHEN state = $4 THEN $5
                     ELSE state
                 END,
                 claim_id = NULL,
                 claim_owner_id = NULL,
                 claim_owner_incarnation_id = NULL,
                 claim_owner_liveness_json = NULL,
                 claim_token = NULL,
                 claim_claimed_at_ms = 0,
                 claim_expires_at_ms = 0
             WHERE session_id = $1 AND claim_id = $2 AND claim_token = $3",
        )
        .bind(&claim.session_id)
        .bind(&claim.claim_id)
        .bind(&claim.lease_token)
        .bind(lash_core::TurnInputState::Accepted.as_str())
        .bind(restored_state.as_str())
        .execute(&self.pool)
        .await
        .map_err(store_sqlx_error)?;
        Ok(())
    }

    async fn save_session_meta(&self, meta: SessionMeta) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO lash_session_meta (session_id, meta_json)
             VALUES ($1, $2)
             ON CONFLICT (session_id) DO UPDATE SET meta_json = EXCLUDED.meta_json",
        )
        .bind(&meta.session_id)
        .bind(encode_json(&meta))
        .execute(&self.pool)
        .await
        .map_err(store_sqlx_error)?;
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<SessionMeta>, StoreError> {
        let json: Option<String> = if let Some(session_id) = &self.session_id {
            sqlx::query_scalar("SELECT meta_json FROM lash_session_meta WHERE session_id = $1")
                .bind(session_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(store_sqlx_error)?
        } else {
            sqlx::query_scalar(
                "SELECT meta_json FROM lash_session_meta ORDER BY session_id ASC LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await
            .map_err(store_sqlx_error)?
        };
        json.map(|json| store_decode_json(&json, "session meta"))
            .transpose()
    }

    async fn tombstone_nodes(&self, ids: &[String]) -> Result<(), StoreError> {
        for id in ids {
            if let Some(session_id) = &self.session_id {
                sqlx::query(
                    "UPDATE lash_graph_nodes
                     SET tombstoned = TRUE
                     WHERE session_id = $1 AND node_id = $2",
                )
                .bind(session_id)
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?;
            } else {
                sqlx::query(
                    "UPDATE lash_graph_nodes
                     SET tombstoned = TRUE
                     WHERE node_id = $1",
                )
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?;
            }
        }
        Ok(())
    }

    async fn vacuum(&self) -> Result<VacuumReport, StoreError> {
        let removed_node_count = if let Some(session_id) = &self.session_id {
            sqlx::query("DELETE FROM lash_graph_nodes WHERE session_id = $1 AND tombstoned = TRUE")
                .bind(session_id)
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?
                .rows_affected()
        } else {
            sqlx::query("DELETE FROM lash_graph_nodes WHERE tombstoned = TRUE")
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?
                .rows_affected()
        };
        let removed_pending_turn_input_tombstone_count = if let Some(session_id) = &self.session_id
        {
            sqlx::query(
                "DELETE FROM lash_pending_turn_inputs
                 WHERE session_id = $1 AND state IN ($2, $3)",
            )
            .bind(session_id)
            .bind(lash_core::TurnInputState::Cancelled.as_str())
            .bind(lash_core::TurnInputState::Completed.as_str())
            .execute(&self.pool)
            .await
            .map_err(store_sqlx_error)?
            .rows_affected()
        } else {
            sqlx::query("DELETE FROM lash_pending_turn_inputs WHERE state IN ($1, $2)")
                .bind(lash_core::TurnInputState::Cancelled.as_str())
                .bind(lash_core::TurnInputState::Completed.as_str())
                .execute(&self.pool)
                .await
                .map_err(store_sqlx_error)?
                .rows_affected()
        };
        Ok(VacuumReport {
            removed_node_count: removed_node_count as usize,
            removed_pending_turn_input_tombstone_count:
                removed_pending_turn_input_tombstone_count as usize,
        })
    }

    async fn gc_unreachable(&self) -> Result<GcReport, StoreError> {
        Ok(GcReport::default())
    }
}

fn derive_pending_turn_input_id(
    session_id: &str,
    source_key: Option<&str>,
    now_epoch_ms: u64,
) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!(
        "ti:{:x}",
        Sha256::digest(format!("{session_id}:{source_key:?}:{now_epoch_ms}:{nanos}").as_bytes())
    )
}

async fn claim_pending_turn_inputs_postgres(
    pool: &PgPool,
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
    let mut tx = pool.begin().await.map_err(store_sqlx_error)?;
    ensure_session_execution_lease_tx(&mut tx, session_id, session_execution_lease).await?;
    let now = current_epoch_ms();
    let wanted_state = match &mode {
        lash_core::TurnInputClaimMode::ActiveTurn { .. } => {
            lash_core::TurnInputState::PendingActive
        }
        lash_core::TurnInputClaimMode::NextTurn => lash_core::TurnInputState::DeferredNextTurn,
    };
    let rows = sqlx::query(
        "SELECT enqueue_seq, input_id, session_id, source_key, ingress_json,
                state, input_json, enqueued_at_ms, claim_id, claim_fencing_token,
                claim_owner_id, claim_owner_incarnation_id,
                claim_owner_liveness_json, claim_token, claim_expires_at_ms
         FROM lash_pending_turn_inputs
         WHERE session_id = $1 AND state = $2
         ORDER BY enqueue_seq ASC
         LIMIT $3
         FOR UPDATE SKIP LOCKED",
    )
    .bind(session_id)
    .bind(wanted_state.as_str())
    .bind(claim_scan_limit(max_inputs))
    .fetch_all(&mut *tx)
    .await
    .map_err(store_sqlx_error)?;
    let mut selected = Vec::new();
    for row in rows {
        let row = pending_turn_input_row(row)?;
        let claim_available = row.claim_token.is_none()
            || row.claim_expires_at_ms <= now
            || row
                .claim_owner
                .as_ref()
                .is_some_and(|holder| holder.is_definitely_dead_for_claimant(owner));
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
        tx.commit().await.map_err(store_sqlx_error)?;
        return Ok(None);
    };
    let lease = TurnInputClaimLease::derive(head, session_id, owner, now, lease_ttl_ms);
    let liveness_json = encode_liveness(&owner.liveness)?;
    let state_after_claim = match &mode {
        lash_core::TurnInputClaimMode::ActiveTurn { .. } => {
            lash_core::TurnInputState::Accepted
        }
        lash_core::TurnInputClaimMode::NextTurn => lash_core::TurnInputState::DeferredNextTurn,
    };
    let mut inputs = Vec::new();
    for (row, mut input) in selected {
        let changed = sqlx::query(
            "UPDATE lash_pending_turn_inputs
             SET state = $3,
                 claim_id = $4,
                 claim_owner_id = $5,
                 claim_owner_incarnation_id = $6,
                 claim_owner_liveness_json = $7,
                 claim_token = $8,
                 claim_fencing_token = claim_fencing_token + 1,
                 claim_claimed_at_ms = $9,
                 claim_expires_at_ms = $10
             WHERE session_id = $1
               AND input_id = $2
               AND (
                    claim_token IS NULL
                    OR claim_expires_at_ms <= $9
                    OR (
                        claim_token = $11
                        AND claim_owner_id = $12
                        AND claim_owner_incarnation_id = $13
                    )
               )",
        )
        .bind(session_id)
        .bind(&row.input_id)
        .bind(state_after_claim.as_str())
        .bind(&lease.claim_id)
        .bind(&owner.owner_id)
        .bind(&owner.incarnation_id)
        .bind(&liveness_json)
        .bind(&lease.lease_token)
        .bind(now as i64)
        .bind(lease.expires_at_epoch_ms as i64)
        .bind(&row.claim_token)
        .bind(
            row.claim_owner
                .as_ref()
                .map(|owner| owner.owner_id.as_str()),
        )
        .bind(
            row.claim_owner
                .as_ref()
                .map(|owner| owner.incarnation_id.as_str()),
        )
        .execute(&mut *tx)
        .await
        .map_err(store_sqlx_error)?
        .rows_affected();
        if changed == 0 {
            tx.rollback().await.map_err(store_sqlx_error)?;
            return Ok(None);
        }
        input.state = state_after_claim;
        inputs.push(input);
    }
    tx.commit().await.map_err(store_sqlx_error)?;
    Ok(Some(lash_core::TurnInputClaim {
        session_id: session_id.to_string(),
        claim_id: lease.claim_id,
        owner: owner.clone(),
        lease_token: lease.lease_token,
        fencing_token: lease.fencing_token,
        claimed_at_epoch_ms: lease.claimed_at_epoch_ms,
        expires_at_epoch_ms: lease.expires_at_epoch_ms,
        mode,
        inputs,
    }))
}

struct SessionExecutionLeaseRow {
    owner: Option<LeaseOwnerIdentity>,
    lease_token: Option<String>,
    fencing_token: u64,
    claimed_at_ms: u64,
    expires_at_ms: u64,
}

async fn load_session_execution_lease_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    session_id: &str,
) -> Result<Option<SessionExecutionLeaseRow>, StoreError> {
    let row = sqlx::query(
        "SELECT lease_owner_id, lease_token, lease_fencing_token,
                lease_claimed_at_ms, lease_expires_at_ms,
                lease_owner_incarnation_id, lease_owner_liveness_json
         FROM lash_session_execution_leases
         WHERE session_id = $1
         FOR UPDATE",
    )
    .bind(session_id)
    .fetch_optional(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    Ok(row.map(|row| SessionExecutionLeaseRow {
        owner: lease_owner_from_columns(row.get(0), row.get(5), row.get(6)),
        lease_token: row.get(1),
        fencing_token: row.get::<i64, _>(2) as u64,
        claimed_at_ms: row.get::<i64, _>(3) as u64,
        expires_at_ms: row.get::<i64, _>(4) as u64,
    }))
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

/// Serialize concurrent session-execution-lease claims for one session.
///
/// `try_claim`/`reclaim` read the current lease and then conditionally
/// `acquire` it. That check-then-act is not atomic under Postgres READ
/// COMMITTED, so two concurrent first claims can both observe no live lease and
/// both `ON CONFLICT DO UPDATE`, leaving two acquired winners. A
/// transaction-scoped advisory lock keyed by the session id makes the sequence
/// mutually exclusive per session; Postgres releases it automatically when the
/// transaction ends. (SQLite and the in-memory store serialize writers
/// globally, so they do not need this.)
async fn lock_session_execution_lease_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    session_id: &str,
) -> Result<(), StoreError> {
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0::bigint))")
        .bind(session_id)
        .execute(&mut **tx)
        .await
        .map_err(store_sqlx_error)?;
    Ok(())
}

async fn acquire_session_execution_lease_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
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
    sqlx::query(
        "INSERT INTO lash_session_execution_leases (
            session_id, lease_owner_id, lease_owner_incarnation_id, lease_owner_liveness_json,
            lease_token, lease_fencing_token, lease_claimed_at_ms, lease_expires_at_ms
         )
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         ON CONFLICT (session_id) DO UPDATE SET
            lease_owner_id = EXCLUDED.lease_owner_id,
            lease_owner_incarnation_id = EXCLUDED.lease_owner_incarnation_id,
            lease_owner_liveness_json = EXCLUDED.lease_owner_liveness_json,
            lease_token = EXCLUDED.lease_token,
            lease_fencing_token = EXCLUDED.lease_fencing_token,
            lease_claimed_at_ms = EXCLUDED.lease_claimed_at_ms,
            lease_expires_at_ms = EXCLUDED.lease_expires_at_ms",
    )
    .bind(session_id)
    .bind(&owner.owner_id)
    .bind(&owner.incarnation_id)
    .bind(&liveness_json)
    .bind(&lease_token)
    .bind(fencing_token as i64)
    .bind(now as i64)
    .bind(expires_at as i64)
    .execute(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    Ok(SessionExecutionLease {
        session_id: session_id.to_string(),
        owner: owner.clone(),
        lease_token,
        fencing_token,
        claimed_at_epoch_ms: now,
        expires_at_epoch_ms: expires_at,
    })
}

async fn ensure_session_execution_lease_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    session_id: &str,
    fence: &SessionExecutionLeaseFence,
) -> Result<(), StoreError> {
    if fence.session_id != session_id {
        return Err(StoreError::SessionExecutionLeaseExpired {
            session_id: session_id.to_string(),
        });
    }
    let now = current_epoch_ms();
    let current = load_session_execution_lease_tx(tx, session_id).await?;
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

async fn release_session_execution_lease_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    completion: &SessionExecutionLeaseCompletion,
) -> Result<(), StoreError> {
    sqlx::query(
        "UPDATE lash_session_execution_leases
         SET lease_owner_id = NULL,
             lease_owner_incarnation_id = NULL,
             lease_owner_liveness_json = NULL,
             lease_token = NULL,
             lease_claimed_at_ms = 0,
             lease_expires_at_ms = 0
         WHERE session_id = $1
           AND lease_owner_id = $2
           AND lease_owner_incarnation_id = $3
           AND lease_token = $4
           AND lease_fencing_token = $5",
    )
    .bind(&completion.session_id)
    .bind(&completion.owner.owner_id)
    .bind(&completion.owner.incarnation_id)
    .bind(&completion.lease_token)
    .bind(completion.fencing_token as i64)
    .execute(&mut **tx)
    .await
    .map_err(store_sqlx_error)?;
    Ok(())
}
