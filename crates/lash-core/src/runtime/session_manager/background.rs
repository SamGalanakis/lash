use super::*;
use std::collections::HashSet;
use std::sync::atomic::Ordering;

impl BackgroundTaskCapability {
    pub(in crate::runtime::session_manager) fn background_scope_key(
        &self,
        session_id: &str,
    ) -> String {
        format!("{}:{session_id}", self.runtime_scope_id)
    }

    pub(in crate::runtime::session_manager) async fn spawn_hidden_task(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
        label: &str,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        self.ensure_known_background_session(current, managed, session_id)
            .await?;
        let Some(registry) = &current.host.background_task_registry else {
            return Err(crate::PluginError::Session(
                "background tasks are unavailable in this runtime".to_string(),
            ));
        };
        self.mark_current_background_sync_needed(current, session_id);
        let task_id = format!("observer:{}:{}", label, uuid::Uuid::new_v4());
        let registration = crate::BackgroundTaskRegistration::new(
            task_id,
            crate::BackgroundTaskKind::Observer,
            label.to_string(),
            crate::BackgroundTaskScope {
                session_id: self.background_scope_key(session_id),
            },
            crate::BackgroundTaskInput::External {
                metadata: serde_json::json!({ "label": label }),
            },
        );
        registry.register(registration.clone()).await?;
        let task_id_for_error = registration.id.clone();
        match current
            .host
            .core
            .effect_controller
            .start_background_task(
                Arc::clone(registry),
                registration,
                crate::BackgroundTaskLocalExecutor::new(|_| async move {
                    match task.await {
                        Ok(()) => crate::BackgroundTaskCompletion {
                            state: crate::BackgroundTaskState::Completed,
                            summary: Some("observer completed".to_string()),
                            output: None,
                        },
                        Err(err) => crate::BackgroundTaskCompletion {
                            state: crate::BackgroundTaskState::Failed,
                            summary: Some(err.to_string()),
                            output: Some(crate::ToolCallOutput::failure(
                                crate::ToolFailure::runtime(
                                    crate::ToolFailureClass::Internal,
                                    "observer_failed",
                                    err.to_string(),
                                ),
                            )),
                        },
                    }
                }),
            )
            .await
        {
            Ok(_) => Ok(()),
            Err(err) => {
                let _ = registry
                    .complete(
                        &task_id_for_error,
                        crate::BackgroundTaskCompletion {
                            state: crate::BackgroundTaskState::Failed,
                            summary: Some(err.to_string()),
                            output: Some(crate::ToolCallOutput::failure(
                                crate::ToolFailure::runtime(
                                    crate::ToolFailureClass::Internal,
                                    "background_task_start_failed",
                                    err.to_string(),
                                ),
                            )),
                        },
                    )
                    .await;
                Err(err)
            }
        }
    }

    pub(in crate::runtime::session_manager) async fn await_hidden_tasks(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) -> Result<(), crate::PluginError> {
        let Some(registry) = &current.host.background_task_registry else {
            return Ok(());
        };
        let scope = self.background_scope_key(session_id);
        loop {
            let tasks = registry
                .list(crate::BackgroundTaskFilter {
                    session_id: Some(scope.clone()),
                    kind: Some(crate::BackgroundTaskKind::Observer),
                    include_terminal: false,
                })
                .await;
            if tasks.is_empty() {
                return Ok(());
            }
            for task in tasks {
                let completion = registry.await_completion(&task.id).await?;
                match completion.state {
                    crate::BackgroundTaskState::Completed => {}
                    crate::BackgroundTaskState::Cancelled => {
                        return Err(crate::PluginError::Session(format!(
                            "hidden background task `{}` was cancelled",
                            task.id
                        )));
                    }
                    crate::BackgroundTaskState::Failed => {
                        return Err(crate::PluginError::Session(
                            completion.summary.unwrap_or_else(|| {
                                format!("hidden background task `{}` failed", task.id)
                            }),
                        ));
                    }
                    _ => {}
                }
            }
        }
    }

    pub(in crate::runtime::session_manager) async fn start_background_task(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
        mut registration: crate::BackgroundTaskRegistration,
        executor: crate::BackgroundTaskLocalExecutor,
    ) -> Result<crate::BackgroundTaskRecord, crate::PluginError> {
        self.ensure_known_background_session(current, managed, session_id)
            .await?;
        let Some(registry) = &current.host.background_task_registry else {
            return Err(crate::PluginError::Session(
                "background tasks are unavailable in this runtime".to_string(),
            ));
        };
        registration.scope = crate::BackgroundTaskScope {
            session_id: self.background_scope_key(session_id),
        };
        self.mark_current_background_sync_needed(current, session_id);
        registry.register(registration.clone()).await?;
        let task_id_for_error = registration.id.clone();
        match current
            .host
            .core
            .effect_controller
            .start_background_task(Arc::clone(registry), registration, executor)
            .await
        {
            Ok(record) => Ok(record),
            Err(err) => {
                let _ = registry
                    .complete(
                        &task_id_for_error,
                        crate::BackgroundTaskCompletion {
                            state: crate::BackgroundTaskState::Failed,
                            summary: Some(err.to_string()),
                            output: Some(crate::ToolCallOutput::failure(
                                crate::ToolFailure::runtime(
                                    crate::ToolFailureClass::Internal,
                                    "background_task_start_failed",
                                    err.to_string(),
                                ),
                            )),
                        },
                    )
                    .await;
                Err(err)
            }
        }
    }

    pub(in crate::runtime::session_manager) async fn await_background_task(
        &self,
        current: &CurrentSessionCapability,
        task_id: &str,
    ) -> Result<crate::BackgroundTaskCompletion, crate::PluginError> {
        let Some(registry) = &current.host.background_task_registry else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        registry.await_completion(task_id).await
    }

    pub(in crate::runtime::session_manager) async fn complete_background_task(
        &self,
        current: &CurrentSessionCapability,
        task_id: &str,
        completion: crate::BackgroundTaskCompletion,
    ) -> Result<crate::BackgroundTaskRecord, crate::PluginError> {
        let Some(registry) = &current.host.background_task_registry else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        registry.complete(task_id, completion).await
    }

    pub(in crate::runtime::session_manager) async fn list_background_tasks(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) -> Result<Vec<crate::BackgroundTaskRecord>, crate::PluginError> {
        let Some(registry) = &current.host.background_task_registry else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        Ok(registry
            .list(crate::BackgroundTaskFilter {
                session_id: Some(self.background_scope_key(session_id)),
                kind: None,
                include_terminal: true,
            })
            .await)
    }

    pub(in crate::runtime::session_manager) async fn cancel_background_task(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        task_id: &str,
    ) -> Result<crate::BackgroundTaskRecord, crate::PluginError> {
        let Some(registry) = &current.host.background_task_registry else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        let Some(_status) = registry.get(task_id).await else {
            return Err(crate::PluginError::Session(format!(
                "unknown background task `{task_id}`"
            )));
        };
        let _ = (managed, host, session_id);
        current
            .host
            .core
            .effect_controller
            .request_background_task_cancel(
                Arc::clone(registry),
                task_id,
                Some("requested by host".to_string()),
            )
            .await
    }

    pub(in crate::runtime::session_manager) async fn cancel_all_background_tasks(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
    ) -> Result<Vec<crate::BackgroundTaskRecord>, crate::PluginError> {
        let tasks = self.list_background_tasks(current, session_id).await?;
        let mut cancelled = Vec::new();
        for task in tasks {
            if task.state.is_terminal() {
                continue;
            }
            cancelled.push(
                self.cancel_background_task(
                    current,
                    managed,
                    Arc::clone(&host),
                    session_id,
                    &task.id,
                )
                .await?,
            );
        }
        Ok(cancelled)
    }

    pub(in crate::runtime::session_manager) async fn validate_async_handles_visible(
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
        if let Some(registry) = &current.host.background_task_registry {
            let scope_key = self.background_scope_key(session_id);
            for task in registry
                .list(crate::BackgroundTaskFilter {
                    session_id: Some(scope_key),
                    kind: None,
                    include_terminal: false,
                })
                .await
            {
                if !task.state.is_terminal() {
                    visible.insert(task.id);
                }
            }
        }
        if let Some(missing) = requested.iter().find(|id| !visible.contains(*id)) {
            return Err(crate::PluginError::Session(format!(
                "async handle `{missing}` is not live or visible in this session"
            )));
        }
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn transfer_async_handles(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        from_session_id: &str,
        to_session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        if handle_ids.is_empty() {
            return Ok(());
        }
        if let Some(registry) = &current.host.background_task_registry {
            let to_scope = crate::BackgroundTaskScope {
                session_id: self.background_scope_key(to_session_id),
            };
            for handle_id in handle_ids {
                let task = registry.get(handle_id).await.ok_or_else(|| {
                    crate::PluginError::Session(format!(
                        "async handle `{handle_id}` is not live or visible in this session"
                    ))
                })?;
                if task.scope.session_id != self.background_scope_key(from_session_id) {
                    return Err(crate::PluginError::Session(format!(
                        "async handle `{handle_id}` is not owned by session `{from_session_id}`"
                    )));
                }
                registry.transfer(handle_id, to_scope.clone()).await?;
            }
        }
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn cancel_unreferenced_async_handles(
        &self,
        current: &CurrentSessionCapability,
        _managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        keep_handle_ids: &[String],
    ) -> Result<Vec<crate::BackgroundTaskRecord>, crate::PluginError> {
        let keep = keep_handle_ids.iter().cloned().collect::<HashSet<_>>();
        let tasks = self.list_background_tasks(current, session_id).await?;
        let mut cancelled = Vec::new();
        for task in tasks {
            if task.state.is_terminal() || keep.contains(&task.id) {
                continue;
            }
            cancelled.push(
                self.cancel_background_task(
                    current,
                    _managed,
                    Arc::clone(&host),
                    session_id,
                    &task.id,
                )
                .await?,
            );
        }
        Ok(cancelled)
    }

    async fn ensure_known_background_session(
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

    fn mark_current_background_sync_needed(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) {
        if session_id == current.session_id {
            self.sync_needed.store(true, Ordering::Release);
        }
    }
}
