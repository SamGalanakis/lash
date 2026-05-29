use super::*;

impl SqliteProcessRegistry {
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        apply_pragmas(&conn, StoreBacking::File)?;
        ensure_process_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            notify: tokio::sync::Notify::new(),
        })
    }

    pub fn memory() -> rusqlite::Result<Self> {
        let conn = Connection::open_in_memory()?;
        apply_pragmas(&conn, StoreBacking::Memory)?;
        ensure_process_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            notify: tokio::sync::Notify::new(),
        })
    }

    fn load_process_conn(
        conn: &Connection,
        process_id: &str,
    ) -> Result<Option<ProcessRecord>, lash_core::PluginError> {
        let json: Option<String> = conn
            .query_row(
                "SELECT record_json FROM processes WHERE process_id = ?1",
                params![process_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(process_sqlite_error)?;
        json.map(|json| serde_json::from_str(&json).map_err(process_decode_error))
            .transpose()
    }

    fn save_process_conn(
        conn: &Connection,
        record: &ProcessRecord,
    ) -> Result<(), lash_core::PluginError> {
        conn.execute(
            "UPDATE processes
             SET updated_at_ms = ?2, record_json = ?3
             WHERE process_id = ?1",
            params![
                &record.id,
                record.updated_at_ms as i64,
                process_encode_json(record)?
            ],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }

    fn load_event_by_key_conn(
        conn: &Connection,
        process_id: &str,
        replay_key: &str,
    ) -> Result<Option<(String, ProcessEvent)>, lash_core::PluginError> {
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT payload_hash, event_json
                 FROM process_events
                 WHERE process_id = ?1 AND idempotency_key = ?2",
                params![process_id, replay_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(process_sqlite_error)?;
        row.map(|(hash, json)| {
            serde_json::from_str(&json)
                .map(|event| (hash, event))
                .map_err(process_decode_error)
        })
        .transpose()
    }
}

#[async_trait::async_trait]
impl ProcessRegistry for SqliteProcessRegistry {
    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let (registration, registration_hash) = prepare_process_registration(registration)?;
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        if let Some(existing) = Self::load_process_conn(&tx, &registration.id)? {
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
        tx.execute(
            "INSERT INTO processes (
                process_id, registration_hash, owner_scope_id, host_profile_id,
                created_at_ms, updated_at_ms, record_json
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &record.id,
                &record.registration_hash,
                owner_scope_id.as_str(),
                record.host_profile_id(),
                record.created_at_ms as i64,
                record.updated_at_ms as i64,
                process_encode_json(&record)?,
            ],
        )
        .map_err(process_sqlite_error)?;
        tx.commit().map_err(process_sqlite_error)?;
        Ok(record)
    }

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, lash_core::PluginError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        let mut record = Self::load_process_conn(&tx, process_id)?.ok_or_else(|| {
            lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
        })?;
        record.external_ref = Some(external_ref);
        record.updated_at_ms = current_epoch_ms();
        Self::save_process_conn(&tx, &record)?;
        tx.commit().map_err(process_sqlite_error)?;
        Ok(record)
    }

    async fn grant_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        if Self::load_process_conn(&conn, process_id)?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let owner_scope_id = owner_scope.id();
        conn.execute(
            "INSERT INTO process_handle_grants (session_id, scope_id, process_id, descriptor_json)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(scope_id, process_id) DO UPDATE SET
                session_id = excluded.session_id,
                descriptor_json = excluded.descriptor_json",
            params![
                &owner_scope.session_id,
                owner_scope_id.as_str(),
                process_id,
                process_encode_json(&descriptor)?
            ],
        )
        .map_err(process_sqlite_error)?;
        Ok(ProcessHandleGrant {
            session_id: owner_scope.session_id.clone(),
            process_id: process_id.to_string(),
            descriptor,
        })
    }

    async fn revoke_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
    ) -> Result<(), lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        let owner_scope_id = owner_scope.id();
        conn.execute(
            "DELETE FROM process_handle_grants WHERE scope_id = ?1 AND process_id = ?2",
            params![owner_scope_id.as_str(), process_id],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }

    async fn transfer_handle_grants(
        &self,
        from_scope: &ProcessScope,
        to_scope: &ProcessScope,
        process_ids: &[String],
    ) -> Result<(), lash_core::PluginError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        let from_scope_id = from_scope.id();
        let to_scope_id = to_scope.id();
        for process_id in process_ids {
            let descriptor_json: Option<String> = tx
                .query_row(
                    "SELECT descriptor_json
                     FROM process_handle_grants
                     WHERE scope_id = ?1 AND process_id = ?2",
                    params![from_scope_id.as_str(), process_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(process_sqlite_error)?;
            let Some(descriptor_json) = descriptor_json else {
                return Err(lash_core::PluginError::Session(format!(
                    "process handle `{process_id}` is not granted to session `{}`",
                    from_scope.session_id
                )));
            };
            tx.execute(
                "DELETE FROM process_handle_grants
                 WHERE scope_id = ?1 AND process_id = ?2",
                params![from_scope_id.as_str(), process_id],
            )
            .map_err(process_sqlite_error)?;
            tx.execute(
                "INSERT INTO process_handle_grants (session_id, scope_id, process_id, descriptor_json)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(scope_id, process_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    descriptor_json = excluded.descriptor_json",
                params![
                    &to_scope.session_id,
                    to_scope_id.as_str(),
                    process_id,
                    descriptor_json
                ],
            )
            .map_err(process_sqlite_error)?;
        }
        tx.commit().map_err(process_sqlite_error)?;
        Ok(())
    }

    async fn list_handle_grants(
        &self,
        owner_scope: &ProcessScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        let owner_scope_id = owner_scope.id();
        let mut stmt = conn
            .prepare(
                "SELECT g.process_id, g.descriptor_json, p.record_json
                 FROM process_handle_grants g
                 JOIN processes p ON p.process_id = g.process_id
                 WHERE g.scope_id = ?1
                 ORDER BY g.process_id ASC",
            )
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![owner_scope_id.as_str()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(process_sqlite_error)?;
        let mut entries = Vec::new();
        for row in rows {
            let (process_id, descriptor_json, record_json) = row.map_err(process_sqlite_error)?;
            let descriptor: ProcessHandleDescriptor =
                serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
            let record: ProcessRecord =
                serde_json::from_str(&record_json).map_err(process_decode_error)?;
            entries.push((
                ProcessHandleGrant {
                    session_id: owner_scope.session_id.clone(),
                    process_id,
                    descriptor,
                },
                record,
            ));
        }
        Ok(entries)
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        if Self::load_process_conn(&conn, process_id)?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let mut stmt = conn
            .prepare(
                "SELECT session_id, descriptor_json
                 FROM process_handle_grants
                 WHERE process_id = ?1
                 ORDER BY session_id ASC, scope_id ASC",
            )
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![process_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(process_sqlite_error)?;
        let mut grants = Vec::new();
        for row in rows {
            let (session_id, descriptor_json) = row.map_err(process_sqlite_error)?;
            let descriptor: ProcessHandleDescriptor =
                serde_json::from_str(&descriptor_json).map_err(process_decode_error)?;
            grants.push(ProcessHandleGrant {
                session_id,
                process_id: process_id.to_string(),
                descriptor,
            });
        }
        Ok(grants)
    }

    async fn delete_session_process_state(
        &self,
        session_id: &str,
    ) -> Result<lash_core::ProcessSessionDeleteReport, lash_core::PluginError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        let mut stmt = tx
            .prepare(
                "SELECT g.process_id, p.record_json
                 FROM process_handle_grants g
                 JOIN processes p ON p.process_id = g.process_id
                 WHERE g.session_id = ?1
                 ORDER BY g.process_id ASC",
            )
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![session_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(process_sqlite_error)?;
        let mut removed = Vec::new();
        for row in rows {
            let (process_id, record_json) = row.map_err(process_sqlite_error)?;
            let record: ProcessRecord =
                serde_json::from_str(&record_json).map_err(process_decode_error)?;
            removed.push((process_id, record));
        }
        drop(stmt);

        let revoked_handle_count = tx
            .execute(
                "DELETE FROM process_handle_grants WHERE session_id = ?1",
                params![session_id],
            )
            .map_err(process_sqlite_error)?;
        let mut cancel_process_ids = Vec::new();
        let mut preserved_process_ids = Vec::new();
        for (process_id, record) in removed {
            if record.is_terminal() {
                continue;
            }
            let remaining_grants: i64 = tx
                .query_row(
                    "SELECT COUNT(*) FROM process_handle_grants WHERE process_id = ?1",
                    params![process_id],
                    |row| row.get(0),
                )
                .map_err(process_sqlite_error)?;
            if remaining_grants == 0 {
                cancel_process_ids.push(process_id);
            } else {
                preserved_process_ids.push(process_id);
            }
        }
        tx.commit().map_err(process_sqlite_error)?;
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
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(process_sqlite_error)?;
        let mut record = Self::load_process_conn(&tx, process_id)?.ok_or_else(|| {
            lash_core::PluginError::Session(format!("unknown process `{process_id}`"))
        })?;
        let replay_lookup =
            if let Some(replay_key) = request.replay.as_ref().map(|replay| replay.key.as_str()) {
                Self::load_event_by_key_conn(&tx, process_id, replay_key)?
            } else {
                None
            };
        let sequence = tx
            .query_row(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM process_events WHERE process_id = ?1",
                params![process_id],
                |row| row.get::<_, i64>(0),
            )
            .map_err(process_sqlite_error)? as u64;
        let occurred_at_ms = current_epoch_ms();
        let prepared = prepare_process_event_append(
            &record,
            request,
            sequence,
            replay_lookup,
            occurred_at_ms,
        )?;
        if prepared.replayed {
            return Ok(ProcessEventAppendResult {
                event: prepared.event,
                wake_delivery: prepared.wake_delivery,
            });
        }
        let event = prepared.event;
        tx.execute(
            "INSERT INTO process_events (
                process_id, sequence, event_type, payload_hash, idempotency_key,
                occurred_at_ms, event_json
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                process_id,
                sequence as i64,
                &event.event_type,
                &prepared.payload_hash,
                event.invocation.replay_key(),
                prepared.occurred_at_ms as i64,
                process_encode_json(&event)?,
            ],
        )
        .map_err(process_sqlite_error)?;
        if let Some(terminal) = prepared.terminal_update.clone() {
            record.terminal = Some(terminal);
        }
        record.updated_at_ms = prepared.occurred_at_ms;
        Self::save_process_conn(&tx, &record)?;
        let wake_delivery = prepared.wake_delivery;
        tx.commit().map_err(process_sqlite_error)?;
        self.notify.notify_waiters();
        Ok(ProcessEventAppendResult {
            event,
            wake_delivery,
        })
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        if Self::load_process_conn(&conn, process_id)?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let mut stmt = conn
            .prepare(
                "SELECT event_json FROM process_events
                 WHERE process_id = ?1 AND sequence > ?2
                 ORDER BY sequence ASC",
            )
            .map_err(process_sqlite_error)?;
        let rows = stmt
            .query_map(params![process_id, after_sequence as i64], |row| {
                row.get::<_, String>(0)
            })
            .map_err(process_sqlite_error)?;
        let mut events = Vec::new();
        for row in rows {
            events.push(
                serde_json::from_str(&row.map_err(process_sqlite_error)?)
                    .map_err(process_decode_error)?,
            );
        }
        Ok(events)
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, lash_core::PluginError> {
        let acked: std::collections::HashSet<u64> = {
            let conn = self.conn.lock().unwrap();
            let mut stmt = conn
                .prepare("SELECT sequence FROM process_wake_acks WHERE process_id = ?1")
                .map_err(process_sqlite_error)?;
            let rows = stmt
                .query_map(params![process_id], |row| row.get::<_, i64>(0))
                .map_err(process_sqlite_error)?;
            let mut set = std::collections::HashSet::new();
            for row in rows {
                set.insert(row.map_err(process_sqlite_error)? as u64);
            }
            set
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
            tokio::time::sleep(Duration::from_millis(50)).await;
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
            if let Some(terminal) = record.terminal {
                return Ok(terminal.await_output);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
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
        let conn = self.conn.lock().ok()?;
        Self::load_process_conn(&conn, process_id).ok().flatten()
    }

    async fn ack_wake(
        &self,
        process_id: &str,
        sequence: u64,
    ) -> Result<(), lash_core::PluginError> {
        let conn = self.conn.lock().unwrap();
        if Self::load_process_conn(&conn, process_id)?.is_none() {
            return Err(lash_core::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        conn.execute(
            "INSERT OR IGNORE INTO process_wake_acks (process_id, sequence) VALUES (?1, ?2)",
            params![process_id, sequence as i64],
        )
        .map_err(process_sqlite_error)?;
        Ok(())
    }
}
