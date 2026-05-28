use super::*;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

impl ProcessCapability {
    pub(in crate::runtime::session_manager) fn process_scope(
        &self,
        session_id: &str,
    ) -> crate::ProcessScope {
        crate::ProcessScope::new(session_id)
    }

    pub(in crate::runtime::session_manager) fn process_scope_id(
        &self,
        session_id: &str,
    ) -> crate::ProcessScopeId {
        self.process_scope(session_id).id()
    }

    pub(in crate::runtime::session_manager) async fn start_process(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        runner: Arc<dyn crate::runtime::effect::ProcessRunner>,
        session_id: &str,
        registration: crate::ProcessRegistration,
        options: crate::ProcessStartOptions,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        self.ensure_known_process_session(current, managed, session_id)
            .await?;
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "processes are unavailable in this runtime".to_string(),
            ));
        };
        self.mark_current_process_sync_needed(current, session_id);
        let creator_scope = self.process_scope(session_id);
        let registration = registration.with_provenance(
            creator_scope.clone(),
            current.host.core.host_profile_id.clone(),
        );
        let wake_target_scope = options
            .wake_session_id
            .as_deref()
            .map(|session_id| self.process_scope(session_id))
            .unwrap_or_else(|| creator_scope.clone());
        let execution_context = options
            .execution_context(&scope)
            .with_wake_target_scope(wake_target_scope);
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Start {
                    registration,
                    grant: options
                        .descriptor
                        .map(|descriptor| crate::ProcessStartGrant {
                            owner_scope: creator_scope,
                            descriptor,
                        }),
                    execution_context: Box::new(execution_context),
                },
                Some(runner),
                scope.effect_metadata,
                scope.effect_controller,
                scope.turn_lease,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::Start { record } => Ok(record),
            _ => Err(crate::PluginError::Session(
                "process start returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn await_process(
        &self,
        current: &CurrentSessionCapability,
        process_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessAwaitOutput, crate::PluginError> {
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Await {
                    process_id: process_id.to_string(),
                },
                None,
                scope.effect_metadata,
                scope.effect_controller,
                scope.turn_lease,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::Await { output } => Ok(output),
            _ => Err(crate::PluginError::Session(
                "process await returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn list_process_handles(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<Vec<crate::ProcessHandleGrantEntry>, crate::PluginError> {
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::List {
                    owner_scope: self.process_scope(session_id),
                },
                None,
                scope.effect_metadata,
                scope.effect_controller,
                scope.turn_lease,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::List { entries } => Ok(entries),
            _ => Err(crate::PluginError::Session(
                "process list returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn cancel_process(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        process_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        if registry.get_process(process_id).await.is_none() {
            return Err(crate::PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let _ = (managed, host, session_id);
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Cancel {
                    process_id: process_id.to_string(),
                    reason: Some("requested by host".to_string()),
                },
                None,
                scope.effect_metadata,
                scope.effect_controller,
                scope.turn_lease,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::Cancel { record } => Ok(record),
            _ => Err(crate::PluginError::Session(
                "process cancel returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn cancel_all_processes(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        let tasks = self
            .list_process_handles(current, session_id, scope.clone())
            .await?;
        let mut cancelled = Vec::new();
        for (grant, record) in tasks {
            if record.is_terminal() {
                continue;
            }
            cancelled.push(
                self.cancel_process(
                    current,
                    managed,
                    Arc::clone(&host),
                    session_id,
                    &grant.process_id,
                    scope.clone(),
                )
                .await?,
            );
        }
        Ok(cancelled)
    }

    pub(in crate::runtime::session_manager) async fn validate_process_handles_visible(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        if handle_ids.is_empty() {
            return Ok(());
        }
        let requested = handle_ids.iter().cloned().collect::<HashSet<_>>();
        let mut visible = HashSet::new();
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        let owner_scope = self.process_scope(session_id);
        for (grant, _record) in registry.list_handle_grants(&owner_scope).await? {
            visible.insert(grant.process_id);
        }
        if let Some(missing) = requested.iter().find(|id| !visible.contains(*id)) {
            return Err(crate::PluginError::Session(format!(
                "process handle `{missing}` is not live or visible in this session"
            )));
        }
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn transfer_process_handles(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        from_session_id: &str,
        to_session_id: &str,
        process_ids: Vec<String>,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<(), crate::PluginError> {
        if process_ids.is_empty() {
            return Ok(());
        }
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        let outcome = self
            .execute_process_effect(
                current,
                Arc::clone(registry),
                crate::ProcessCommand::Transfer {
                    from_scope: self.process_scope(from_session_id),
                    to_scope: self.process_scope(to_session_id),
                    process_ids,
                },
                None,
                scope.effect_metadata,
                scope.effect_controller,
                scope.turn_lease,
            )
            .await?;
        match outcome {
            crate::ProcessEffectOutcome::Transfer => Ok(()),
            _ => Err(crate::PluginError::Session(
                "process transfer returned the wrong outcome".to_string(),
            )),
        }
    }

    pub(in crate::runtime::session_manager) async fn cancel_unreferenced_process_handles(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        keep_process_ids: Vec<String>,
        scope: crate::ProcessOpScope<'_>,
    ) -> Result<Vec<crate::ProcessRecord>, crate::PluginError> {
        let keep = keep_process_ids.iter().cloned().collect::<HashSet<_>>();
        let Some(registry) = &current.host.process_registry else {
            return Err(crate::PluginError::Session(
                "process registry is unavailable in this runtime".to_string(),
            ));
        };
        let owner_scope = self.process_scope(session_id);
        let tasks = registry.list_handle_grants(&owner_scope).await?;
        let mut cancelled = Vec::new();
        for (grant, record) in tasks {
            if keep.contains(&grant.process_id) {
                continue;
            }
            registry
                .revoke_handle(&owner_scope, &grant.process_id)
                .await?;
            if record.is_terminal() {
                continue;
            }
            if !registry
                .handle_grants_for_process(&grant.process_id)
                .await?
                .is_empty()
            {
                continue;
            }
            cancelled.push(
                self.cancel_process(
                    current,
                    _managed,
                    Arc::clone(&host),
                    session_id,
                    &grant.process_id,
                    scope.clone(),
                )
                .await?,
            );
        }
        Ok(cancelled)
    }

    async fn ensure_known_process_session(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
    ) -> Result<(), crate::PluginError> {
        if session_id == current.session_id
            || managed.registry.lock().await.contains_key(session_id)
        {
            return Ok(());
        }
        Err(crate::PluginError::Session(format!(
            "unknown session `{session_id}`"
        )))
    }

    fn mark_current_process_sync_needed(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) {
        if session_id == current.session_id {
            self.sync_needed.store(true, Ordering::Release);
        }
    }

    async fn execute_process_effect(
        &self,
        current: &CurrentSessionCapability,
        registry: Arc<dyn crate::ProcessRegistry>,
        command: crate::ProcessCommand,
        runner: Option<Arc<dyn crate::runtime::effect::ProcessRunner>>,
        parent_metadata: Option<crate::EffectInvocationMetadata>,
        effect_controller: Option<&dyn crate::RuntimeEffectController>,
        scope_turn_lease: Option<crate::RuntimeTurnLease>,
    ) -> Result<crate::ProcessEffectOutcome, crate::PluginError> {
        let effect_id = command.effect_id();
        let metadata = self.process_effect_metadata(current, &effect_id, parent_metadata)?;
        let envelope = crate::RuntimeEffectEnvelope::new(
            metadata,
            crate::RuntimeEffectCommand::Process { command },
        );
        let controller = effect_controller.unwrap_or(current.host.core.effect_controller.as_ref());
        let turn_lease = scope_turn_lease.as_ref().or(current.turn_lease.as_ref());
        // Process registry/workflow idempotency is the replay boundary when a
        // process control call is issued outside an active turn lease.
        let journal_store = turn_lease.and(current.store.as_ref().map(|store| store.as_ref()));
        let outcome = crate::runtime::effect::execute_effect_with_journal(
            journal_store,
            turn_lease,
            controller,
            envelope,
            crate::RuntimeEffectLocalExecutor::process(registry, runner),
        )
        .await?;
        outcome.into_process().map_err(crate::PluginError::from)
    }

    fn process_effect_metadata(
        &self,
        current: &CurrentSessionCapability,
        effect_id: &str,
        parent_metadata: Option<crate::EffectInvocationMetadata>,
    ) -> Result<crate::EffectInvocationMetadata, crate::PluginError> {
        if let Some(parent) = parent_metadata {
            let Some(turn_id) = parent.turn_id.clone() else {
                return Err(crate::PluginError::Session(format!(
                    "process effect `{effect_id}` requires active turn metadata"
                )));
            };
            return Ok(crate::EffectInvocationMetadata {
                session_id: current.session_id.clone(),
                origin: parent.origin,
                turn_id: Some(turn_id),
                turn_index: parent.turn_index,
                protocol_iteration: parent.protocol_iteration,
                effect_id: effect_id.to_string(),
                effect_kind: crate::RuntimeEffectKind::Process,
                idempotency_key: format!("{}:{effect_id}", parent.idempotency_key),
                turn_checkpoint_hash: parent.turn_checkpoint_hash,
            });
        }
        Ok(crate::EffectInvocationMetadata {
            session_id: current.session_id.clone(),
            origin: crate::EffectOrigin::Turn,
            turn_id: None,
            turn_index: None,
            protocol_iteration: None,
            effect_id: effect_id.to_string(),
            effect_kind: crate::RuntimeEffectKind::Process,
            idempotency_key: format!("{}:{effect_id}", current.session_id),
            turn_checkpoint_hash: None,
        })
    }
}
