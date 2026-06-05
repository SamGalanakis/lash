use super::*;

fn process_status_label(record: &ProcessRecord) -> &'static str {
    record.status.label()
}

impl TursoProcessRegistry {
    pub async fn open(path: &Path) -> turso::Result<Self> {
        let _schema_guard = file_schema_open_guard().await;
        let path = path.to_string_lossy().into_owned();
        let db = turso::Builder::new_local(&path).build().await?;
        let conn = db.connect()?;
        conn.busy_timeout(TURSO_BUSY_TIMEOUT)?;
        ensure_process_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::File).await?;
        Ok(Self {
            _db: db,
            conn: tokio::sync::Mutex::new(conn),
            notify: tokio::sync::Notify::new(),
        })
    }

    pub async fn memory() -> turso::Result<Self> {
        let db = turso::Builder::new_local(":memory:").build().await?;
        let conn = db.connect()?;
        conn.busy_timeout(TURSO_BUSY_TIMEOUT)?;
        ensure_process_schema(&conn).await?;
        apply_pragmas(&conn, StoreBacking::Memory).await?;
        Ok(Self {
            _db: db,
            conn: tokio::sync::Mutex::new(conn),
            notify: tokio::sync::Notify::new(),
        })
    }

    async fn load_process_conn(
        conn: &Connection,
        process_id: &str,
    ) -> Result<Option<ProcessRecord>, lash_core::PluginError> {
        let row = optional_row(
            conn,
            "SELECT record_json FROM processes WHERE process_id = ?1",
            params![process_id],
        )
        .await
        .map_err(process_turso_error)?;
        row.map(|row| {
            serde_json::from_str(&row_string(&row, 0).map_err(process_turso_error)?)
                .map_err(process_decode_error)
        })
        .transpose()
    }

    async fn save_process_conn(
        conn: &Connection,
        record: &ProcessRecord,
    ) -> Result<(), lash_core::PluginError> {
        conn.execute(
            "UPDATE processes
             SET updated_at_ms = ?2, status = ?3, record_json = ?4
             WHERE process_id = ?1",
            params![
                record.id.as_str(),
                record.updated_at_ms as i64,
                process_status_label(record),
                process_encode_json(record)?,
            ],
        )
        .await
        .map_err(process_turso_error)?;
        Ok(())
    }

    async fn load_event_by_key_conn(
        conn: &Connection,
        process_id: &str,
        replay_key: &str,
    ) -> Result<Option<(String, ProcessEvent)>, lash_core::PluginError> {
        let row = optional_row(
            conn,
            "SELECT payload_hash, event_json
             FROM process_events
             WHERE process_id = ?1 AND idempotency_key = ?2",
            params![process_id, replay_key],
        )
        .await
        .map_err(process_turso_error)?;
        row.map(|row| {
            let hash = row_string(&row, 0).map_err(process_turso_error)?;
            let json = row_string(&row, 1).map_err(process_turso_error)?;
            serde_json::from_str(&json)
                .map(|event| (hash, event))
                .map_err(process_decode_error)
        })
        .transpose()
    }

    async fn load_process_lease_conn(
        conn: &Connection,
        process_id: &str,
    ) -> Result<Option<ProcessLease>, lash_core::PluginError> {
        let row = optional_row(
            conn,
            "SELECT lease_owner_id, lease_token, lease_fencing_token,
                    lease_claimed_at_ms, lease_expires_at_ms
             FROM process_leases
             WHERE process_id = ?1",
            params![process_id],
        )
        .await
        .map_err(process_turso_error)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let owner_id = row_optional_string(&row, 0).map_err(process_turso_error)?;
        let lease_token = row_optional_string(&row, 1).map_err(process_turso_error)?;
        let (Some(owner_id), Some(lease_token)) = (owner_id, lease_token) else {
            return Ok(None);
        };
        Ok(Some(ProcessLease {
            schema_version: PROCESS_LEASE_SCHEMA_VERSION,
            process_id: process_id.to_string(),
            owner_id,
            lease_token,
            fencing_token: row_i64(&row, 2).map_err(process_turso_error)? as u64,
            claimed_at_epoch_ms: row_i64(&row, 3).map_err(process_turso_error)? as u64,
            expires_at_epoch_ms: row_i64(&row, 4).map_err(process_turso_error)? as u64,
        }))
    }

    async fn list_grants_for_scope(
        conn: &Connection,
        owner_scope: &ProcessScope,
        live_only: bool,
    ) -> Result<Vec<ProcessHandleGrantEntry>, lash_core::PluginError> {
        let owner_scope_id = owner_scope.id();
        let status_clause = if live_only {
            "AND p.status = 'running'"
        } else {
            ""
        };
        let rows = collect_rows(
            conn,
            &format!(
                "SELECT g.process_id, g.descriptor_json, p.record_json
                 FROM process_handle_grants g
                 JOIN processes p ON p.process_id = g.process_id
                 WHERE g.scope_id = ?1 {status_clause}
                 ORDER BY g.process_id ASC"
            ),
            params![owner_scope_id.as_str()],
        )
        .await
        .map_err(process_turso_error)?;
        rows.into_iter()
            .map(|row| {
                let process_id = row_string(&row, 0).map_err(process_turso_error)?;
                let descriptor_json = row_string(&row, 1).map_err(process_turso_error)?;
                let record_json = row_string(&row, 2).map_err(process_turso_error)?;
                let descriptor: ProcessHandleDescriptor =
                    serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
                let record: ProcessRecord =
                    serde_json::from_str(&record_json).map_err(process_decode_error)?;
                Ok((
                    ProcessHandleGrant {
                        session_id: owner_scope.session_id.clone(),
                        process_id,
                        descriptor,
                    },
                    record,
                ))
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl ProcessRegistry for TursoProcessRegistry {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let (registration, registration_hash) = prepare_process_registration(registration)?;
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(process_turso_error)?;
        let result = async {
            if let Some(existing) = Self::load_process_conn(&conn, &registration.id).await? {
                if existing.registration_hash == registration_hash {
                    return Ok(existing);
                }
                return Err(lash_core::PluginError::Session(format!(
                    "process `{}` registration hash conflict: existing {}, new {}",
                    registration.id, existing.registration_hash, registration_hash
                )));
            }
            let now = current_epoch_ms();
            let record =
                ProcessRecord::from_prepared_registration(registration, registration_hash, now);
            let owner_scope_id = record.owner_scope_id();
            conn.execute(
                "INSERT INTO processes (
                    process_id, registration_hash, owner_scope_id, host_profile_id,
                    created_at_ms, updated_at_ms, status, record_json
                 )
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    record.id.as_str(),
                    record.registration_hash.as_str(),
                    owner_scope_id.as_str(),
                    record.host_profile_id(),
                    record.created_at_ms as i64,
                    record.updated_at_ms as i64,
                    process_status_label(&record),
                    process_encode_json(&record)?,
                ],
            )
            .await
            .map_err(process_turso_error)?;
            Ok(record)
        }
        .await;
        match result {
            Ok(record) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(process_turso_error)?;
                Ok(record)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(process_turso_error)?;
        let result = async {
            let mut record = Self::load_process_conn(&conn, process_id)
                .await?
                .ok_or_else(|| {
                    lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
                })?;
            record.external_ref = Some(external_ref);
            record.updated_at_ms = current_epoch_ms();
            Self::save_process_conn(&conn, &record).await?;
            Ok(record)
        }
        .await;
        match result {
            Ok(record) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(process_turso_error)?;
                Ok(record)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn grant_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, lash_core::PluginError> {
        let owner_scope_id = owner_scope.id();
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(process_turso_error)?;
        let result = async {
            if Self::load_process_conn(&conn, process_id).await?.is_none() {
                return Err(lash_core::PluginError::Session(format!(
                    "unknown process `{process_id}`"
                )));
            }
            conn.execute(
                "INSERT INTO process_handle_grants (session_id, scope_id, process_id, descriptor_json)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(scope_id, process_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    descriptor_json = excluded.descriptor_json",
                params![
                    owner_scope.session_id.as_str(),
                    owner_scope_id.as_str(),
                    process_id,
                    process_encode_json(&descriptor)?,
                ],
            )
            .await
            .map_err(process_turso_error)?;
            Ok(ProcessHandleGrant {
                session_id: owner_scope.session_id.clone(),
                process_id: process_id.to_string(),
                descriptor,
            })
        }
        .await;
        match result {
            Ok(grant) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(process_turso_error)?;
                Ok(grant)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn revoke_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
    ) -> Result<(), lash_core::PluginError> {
        let conn = self.conn.lock().await;
        let owner_scope_id = owner_scope.id();
        conn.execute(
            "DELETE FROM process_handle_grants WHERE scope_id = ?1 AND process_id = ?2",
            params![owner_scope_id.as_str(), process_id],
        )
        .await
        .map_err(process_turso_error)?;
        Ok(())
    }

    async fn transfer_handle_grants(
        &self,
        from_scope: &ProcessScope,
        to_scope: &ProcessScope,
        process_ids: &[String],
    ) -> Result<(), lash_core::PluginError> {
        let from_scope_id = from_scope.id();
        let to_scope_id = to_scope.id();
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(process_turso_error)?;
        let result = async {
            for process_id in process_ids {
                let descriptor_json = optional_row(
                    &conn,
                    "SELECT descriptor_json
                     FROM process_handle_grants
                     WHERE scope_id = ?1 AND process_id = ?2",
                    params![from_scope_id.as_str(), process_id.as_str()],
                )
                .await
                .map_err(process_turso_error)?
                .map(|row| row_string(&row, 0).map_err(process_turso_error))
                .transpose()?;
                let Some(descriptor_json) = descriptor_json else {
                    return Err(lash_core::PluginError::Session(format!(
                        "process handle `{process_id}` is not granted to session `{}`",
                        from_scope.session_id
                    )));
                };
                conn.execute(
                    "DELETE FROM process_handle_grants
                     WHERE scope_id = ?1 AND process_id = ?2",
                    params![from_scope_id.as_str(), process_id.as_str()],
                )
                .await
                .map_err(process_turso_error)?;
                conn.execute(
                    "INSERT INTO process_handle_grants (session_id, scope_id, process_id, descriptor_json)
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(scope_id, process_id) DO UPDATE SET
                        session_id = excluded.session_id,
                        descriptor_json = excluded.descriptor_json",
                    params![
                        to_scope.session_id.as_str(),
                        to_scope_id.as_str(),
                        process_id.as_str(),
                        descriptor_json,
                    ],
                )
                .await
                .map_err(process_turso_error)?;
            }
            Ok(())
        }
        .await;
        match result {
            Ok(()) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(process_turso_error)?;
                Ok(())
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn list_handle_grants(
        &self,
        owner_scope: &ProcessScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        Self::list_grants_for_scope(&conn, owner_scope, false).await
    }

    async fn list_live_handle_grants(
        &self,
        owner_scope: &ProcessScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        Self::list_grants_for_scope(&conn, owner_scope, true).await
    }

    async fn has_handle_grant(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
    ) -> Result<bool, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        let owner_scope_id = owner_scope.id();
        let exists = optional_row(
            &conn,
            "SELECT 1
             FROM process_handle_grants g
             JOIN processes p ON p.process_id = g.process_id
             WHERE g.scope_id = ?1 AND g.process_id = ?2
             LIMIT 1",
            params![owner_scope_id.as_str(), process_id],
        )
        .await
        .map_err(process_turso_error)?
        .is_some();
        Ok(exists)
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        if Self::load_process_conn(&conn, process_id).await?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let rows = collect_rows(
            &conn,
            "SELECT session_id, descriptor_json
             FROM process_handle_grants
             WHERE process_id = ?1
             ORDER BY session_id ASC, scope_id ASC",
            params![process_id],
        )
        .await
        .map_err(process_turso_error)?;
        rows.into_iter()
            .map(|row| {
                let session_id = row_string(&row, 0).map_err(process_turso_error)?;
                let descriptor_json = row_string(&row, 1).map_err(process_turso_error)?;
                let descriptor: ProcessHandleDescriptor =
                    serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
                Ok(ProcessHandleGrant {
                    session_id,
                    process_id: process_id.to_string(),
                    descriptor,
                })
            })
            .collect()
    }

    async fn delete_session_process_state(
        &self,
        session_id: &str,
    ) -> Result<lash_core::ProcessSessionDeleteReport, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(process_turso_error)?;
        let result = async {
            let rows = collect_rows(
                &conn,
                "SELECT g.process_id, p.record_json
                 FROM process_handle_grants g
                 JOIN processes p ON p.process_id = g.process_id
                 WHERE g.session_id = ?1
                 ORDER BY g.process_id ASC",
                params![session_id],
            )
            .await
            .map_err(process_turso_error)?;
            let mut removed = Vec::new();
            for row in rows {
                let process_id = row_string(&row, 0).map_err(process_turso_error)?;
                let record_json = row_string(&row, 1).map_err(process_turso_error)?;
                let record: ProcessRecord =
                    serde_json::from_str(&record_json).map_err(process_decode_error)?;
                removed.push((process_id, record));
            }

            let revoked_handle_count = conn
                .execute(
                    "DELETE FROM process_handle_grants WHERE session_id = ?1",
                    params![session_id],
                )
                .await
                .map_err(process_turso_error)? as usize;
            let mut cancel_process_ids = Vec::new();
            let mut preserved_process_ids = Vec::new();
            for (process_id, record) in removed {
                if record.is_terminal() {
                    continue;
                }
                let remaining_grants = required_row(
                    &conn,
                    "SELECT COUNT(*) FROM process_handle_grants WHERE process_id = ?1",
                    params![process_id.as_str()],
                )
                .await
                .map_err(process_turso_error)
                .and_then(|row| row_i64(&row, 0).map_err(process_turso_error))?;
                if remaining_grants == 0 {
                    cancel_process_ids.push(process_id);
                } else {
                    preserved_process_ids.push(process_id);
                }
            }
            Ok((
                revoked_handle_count,
                cancel_process_ids,
                preserved_process_ids,
            ))
        }
        .await;
        let (revoked_handle_count, mut cancel_process_ids, mut preserved_process_ids) = match result
        {
            Ok(value) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(process_turso_error)?;
                value
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(err);
            }
        };
        cancel_process_ids.sort();
        cancel_process_ids.dedup();
        preserved_process_ids.sort();
        preserved_process_ids.dedup();
        Ok(lash_core::ProcessSessionDeleteReport {
            session_id: session_id.to_string(),
            revoked_handle_count,
            deleted_wake_count: 0,
            cancel_process_ids,
            preserved_process_ids,
        })
    }

    async fn append_event(
        &self,
        process_id: &str,
        request: ProcessEventAppendRequest,
    ) -> Result<ProcessEventAppendResult, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(process_turso_error)?;
        let result = async {
            let mut record = Self::load_process_conn(&conn, process_id)
                .await?
                .ok_or_else(|| {
                    lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
                })?;
            let replay_lookup = if let Some(replay_key) =
                request.replay.as_ref().map(|replay| replay.key.as_str())
            {
                Self::load_event_by_key_conn(&conn, process_id, replay_key).await?
            } else {
                None
            };
            let sequence = required_row(
                &conn,
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM process_events WHERE process_id = ?1",
                params![process_id],
            )
            .await
            .map_err(process_turso_error)
            .and_then(|row| row_i64(&row, 0).map_err(process_turso_error))?
                as u64;
            let occurred_at_ms = current_epoch_ms();
            let prepared = prepare_process_event_append(
                &record,
                request,
                sequence,
                replay_lookup,
                occurred_at_ms,
            )?;
            if prepared.replayed {
                return Ok((
                    ProcessEventAppendResult {
                        event: prepared.event,
                        wake_delivery: prepared.wake_delivery,
                    },
                    false,
                ));
            }
            let event = prepared.event;
            conn.execute(
                "INSERT INTO process_events (
                    process_id, sequence, event_type, payload_hash, idempotency_key,
                    occurred_at_ms, event_json
                 )
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    process_id,
                    sequence as i64,
                    event.event_type.as_str(),
                    prepared.payload_hash.as_str(),
                    event.invocation.replay_key(),
                    prepared.occurred_at_ms as i64,
                    process_encode_json(&event)?,
                ],
            )
            .await
            .map_err(process_turso_error)?;
            if let Some(status) = prepared.status_update.clone() {
                record.status = status;
            }
            record.updated_at_ms = prepared.occurred_at_ms;
            Self::save_process_conn(&conn, &record).await?;
            Ok((
                ProcessEventAppendResult {
                    event,
                    wake_delivery: prepared.wake_delivery,
                },
                true,
            ))
        }
        .await;
        match result {
            Ok((result, appended)) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(process_turso_error)?;
                drop(conn);
                if appended {
                    self.notify.notify_waiters();
                }
                Ok(result)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        if Self::load_process_conn(&conn, process_id).await?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let rows = collect_rows(
            &conn,
            "SELECT event_json FROM process_events
             WHERE process_id = ?1 AND sequence > ?2
             ORDER BY sequence ASC",
            params![process_id, after_sequence as i64],
        )
        .await
        .map_err(process_turso_error)?;
        rows.into_iter()
            .map(|row| {
                let json = row_string(&row, 0).map_err(process_turso_error)?;
                serde_json::from_str(&json).map_err(process_decode_error)
            })
            .collect()
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        let acked: std::collections::HashSet<u64> = {
            let conn = self.conn.lock().await;
            let rows = collect_rows(
                &conn,
                "SELECT sequence FROM process_wake_acks WHERE process_id = ?1",
                params![process_id],
            )
            .await
            .map_err(process_turso_error)?;
            rows.into_iter()
                .map(|row| row_i64(&row, 0).map(|value| value as u64))
                .collect::<turso::Result<_>>()
                .map_err(process_turso_error)?
        };
        Ok(self
            .events_after(process_id, after_sequence)
            .await?
            .into_iter()
            .filter(|event| event.semantics.wake.is_some() && !acked.contains(&event.sequence))
            .collect())
    }

    async fn wait_event_after(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, lash_core::PluginError> {
        loop {
            if let Some(event) = self
                .events_after(process_id, after_sequence)
                .await?
                .into_iter()
                .find(|event| event.event_type == event_type)
            {
                return Ok(event);
            }
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
        }
    }

    async fn await_process(
        &self,
        process_id: &str,
    ) -> Result<ProcessAwaitOutput, lash_core::PluginError> {
        loop {
            let record = self.get_process(process_id).await.ok_or_else(|| {
                lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
            })?;
            if let Some(await_output) = record.status.await_output() {
                return Ok(await_output.clone());
            }
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
        }
    }

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let event_type = match await_output.terminal_state() {
            lash_core::ProcessTerminalState::Completed => "process.completed",
            lash_core::ProcessTerminalState::Failed => "process.failed",
            lash_core::ProcessTerminalState::Cancelled => "process.cancelled",
        };
        self.append_event(
            process_id,
            ProcessEventAppendRequest::new(
                event_type,
                serde_json::json!({ "await_output": await_output }),
            )
            .with_replay_key(format!("process:{process_id}:terminal:{event_type}")),
        )
        .await?;
        self.get_process(process_id).await.ok_or_else(|| {
            lash_core::PluginError::Session(format!(
                "unknown process `{process_id}` after terminal event"
            ))
        })
    }

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord> {
        let conn = self.conn.lock().await;
        Self::load_process_conn(&conn, process_id)
            .await
            .ok()
            .flatten()
    }

    async fn ack_wake(
        &self,
        process_id: &str,
        sequence: u64,
    ) -> Result<(), lash_core::PluginError> {
        let conn = self.conn.lock().await;
        if Self::load_process_conn(&conn, process_id).await?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        conn.execute(
            "INSERT OR IGNORE INTO process_wake_acks (process_id, sequence) VALUES (?1, ?2)",
            params![process_id, sequence as i64],
        )
        .await
        .map_err(process_turso_error)?;
        Ok(())
    }

    async fn list_non_terminal(&self) -> Result<Vec<ProcessRecord>, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        let rows = collect_rows(
            &conn,
            "SELECT record_json FROM processes
             WHERE status = 'running'
             ORDER BY process_id ASC",
            (),
        )
        .await
        .map_err(process_turso_error)?;
        rows.into_iter()
            .map(|row| {
                let json = row_string(&row, 0).map_err(process_turso_error)?;
                serde_json::from_str(&json).map_err(process_decode_error)
            })
            .collect()
    }

    async fn claim_process_lease(
        &self,
        process_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(process_turso_error)?;
        let result = async {
            if Self::load_process_conn(&conn, process_id).await?.is_none() {
                return Err(lash_core::PluginError::Session(format!(
                    "unknown process `{process_id}`"
                )));
            }
            let now = current_epoch_ms();
            let current = Self::load_process_lease_conn(&conn, process_id).await?;
            if let Some(current) = current.as_ref()
                && current.expires_at_epoch_ms > now
                && current.owner_id != owner_id
            {
                return Err(process_lease_conflict(process_id, current));
            }
            let fencing_token = optional_row(
                &conn,
                "SELECT lease_fencing_token FROM process_leases WHERE process_id = ?1",
                params![process_id],
            )
            .await
            .map_err(process_turso_error)?
            .map(|row| row_i64(&row, 0).map(|value| value as u64))
            .transpose()
            .map_err(process_turso_error)?
            .unwrap_or(0)
                + 1;
            let lease = ProcessLease {
                schema_version: PROCESS_LEASE_SCHEMA_VERSION,
                process_id: process_id.to_string(),
                owner_id: owner_id.to_string(),
                lease_token: format!(
                    "{:x}",
                    Sha256::digest(
                        format!("{process_id}:{owner_id}:{now}:{fencing_token}").as_bytes()
                    )
                ),
                fencing_token,
                claimed_at_epoch_ms: now,
                expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
            };
            conn.execute(
                "INSERT INTO process_leases (
                    process_id, lease_owner_id, lease_token, lease_fencing_token,
                    lease_claimed_at_ms, lease_expires_at_ms
                 )
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(process_id) DO UPDATE SET
                    lease_owner_id = excluded.lease_owner_id,
                    lease_token = excluded.lease_token,
                    lease_fencing_token = excluded.lease_fencing_token,
                    lease_claimed_at_ms = excluded.lease_claimed_at_ms,
                    lease_expires_at_ms = excluded.lease_expires_at_ms",
                params![
                    lease.process_id.as_str(),
                    lease.owner_id.as_str(),
                    lease.lease_token.as_str(),
                    lease.fencing_token as i64,
                    lease.claimed_at_epoch_ms as i64,
                    lease.expires_at_epoch_ms as i64,
                ],
            )
            .await
            .map_err(process_turso_error)?;
            Ok(lease)
        }
        .await;
        match result {
            Ok(lease) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(process_turso_error)?;
                Ok(lease)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn renew_process_lease(
        &self,
        lease: &ProcessLease,
        lease_ttl_ms: u64,
    ) -> Result<ProcessLease, lash_core::PluginError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(process_turso_error)?;
        let result = async {
            let now = current_epoch_ms();
            let current = Self::load_process_lease_conn(&conn, &lease.process_id).await?;
            if !guard_lease(current.as_ref(), &lease.lease_token, now) {
                return Err(process_lease_expired(&lease.process_id));
            }
            let renewed = ProcessLease {
                expires_at_epoch_ms: now.saturating_add(lease_ttl_ms),
                ..lease.clone()
            };
            conn.execute(
                "UPDATE process_leases
                 SET lease_expires_at_ms = ?2
                 WHERE process_id = ?1 AND lease_token = ?3",
                params![
                    renewed.process_id.as_str(),
                    renewed.expires_at_epoch_ms as i64,
                    renewed.lease_token.as_str(),
                ],
            )
            .await
            .map_err(process_turso_error)?;
            Ok(renewed)
        }
        .await;
        match result {
            Ok(lease) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(process_turso_error)?;
                Ok(lease)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }

    async fn complete_process_lease(
        &self,
        completion: &ProcessLeaseCompletion,
    ) -> Result<(), lash_core::PluginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE process_leases
             SET lease_owner_id = NULL,
                 lease_token = NULL,
                 lease_claimed_at_ms = 0,
                 lease_expires_at_ms = 0
             WHERE process_id = ?1 AND lease_token = ?2",
            params![
                completion.process_id.as_str(),
                completion.lease_token.as_str()
            ],
        )
        .await
        .map_err(process_turso_error)?;
        Ok(())
    }
}

fn process_lease_conflict(process_id: &str, current: &ProcessLease) -> lash_core::PluginError {
    lash_core::PluginError::Session(format!(
        "process `{process_id}` is already leased by `{}` until {}",
        current.owner_id, current.expires_at_epoch_ms
    ))
}

fn process_lease_expired(process_id: &str) -> lash_core::PluginError {
    lash_core::PluginError::Session(format!(
        "process lease for `{process_id}` is missing or expired"
    ))
}
