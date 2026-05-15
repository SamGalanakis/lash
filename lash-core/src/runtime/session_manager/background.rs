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
        let Some(executor) = &current.host.background_task_host else {
            return Err(crate::PluginError::Session(
                "background tasks are unavailable in this runtime".to_string(),
            ));
        };
        self.mark_current_background_sync_needed(current, session_id);
        executor
            .spawn_hidden(&self.background_scope_key(session_id), label, task)
            .await
    }

    pub(in crate::runtime::session_manager) async fn await_hidden_tasks(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) -> Result<(), crate::PluginError> {
        let Some(executor) = &current.host.background_task_host else {
            return Ok(());
        };
        executor
            .await_hidden(&self.background_scope_key(session_id))
            .await
    }

    pub(in crate::runtime::session_manager) async fn spawn_managed_task(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
        spec: crate::BackgroundTaskRegistration,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        self.ensure_known_background_session(current, managed, session_id)
            .await?;
        let Some(executor) = &current.host.background_task_host else {
            return Err(crate::PluginError::Session(
                "managed background tasks are unavailable in this runtime".to_string(),
            ));
        };
        self.mark_current_background_sync_needed(current, session_id);
        executor
            .spawn_managed(&self.background_scope_key(session_id), spec, task)
            .await
    }

    pub(in crate::runtime::session_manager) async fn cancel_managed_task(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        task_id: &str,
    ) -> Result<(), crate::PluginError> {
        let Some(executor) = &current.host.background_task_host else {
            return Ok(());
        };
        executor
            .cancel_managed(&self.background_scope_key(session_id), task_id)
            .await
    }

    pub(in crate::runtime::session_manager) async fn register_background_task(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        spec: crate::BackgroundTaskRegistration,
        cancel: Option<crate::LocalBackgroundTaskCancel>,
    ) -> Result<(), crate::PluginError> {
        let Some(executor) = &current.host.background_task_host else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        executor
            .register_external(&self.background_scope_key(session_id), spec, cancel)
            .await
    }

    pub(in crate::runtime::session_manager) async fn unregister_background_task(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        task_id: &str,
    ) {
        let Some(executor) = &current.host.background_task_host else {
            return;
        };
        executor
            .unregister_external(&self.background_scope_key(session_id), task_id)
            .await;
    }

    pub(in crate::runtime::session_manager) async fn complete_background_task(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        task_id: &str,
        state: crate::BackgroundTaskState,
    ) {
        let Some(executor) = &current.host.background_task_host else {
            return;
        };
        executor
            .mark_terminal(&self.background_scope_key(session_id), task_id, state)
            .await;
    }

    pub(in crate::runtime::session_manager) async fn transition_background_task_live_state(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        task_id: &str,
        state: crate::BackgroundTaskState,
    ) {
        let Some(executor) = &current.host.background_task_host else {
            return;
        };
        executor
            .mark_live_state(&self.background_scope_key(session_id), task_id, state)
            .await;
    }

    pub(in crate::runtime::session_manager) async fn list_background_tasks(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) -> Result<Vec<crate::BackgroundTaskRecord>, crate::PluginError> {
        let Some(executor) = &current.host.background_task_host else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        Ok(executor
            .list_managed(&self.background_scope_key(session_id))
            .await)
    }

    pub(in crate::runtime::session_manager) async fn cancel_background_task(
        &self,
        current: &CurrentSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        task_id: &str,
    ) -> Result<crate::BackgroundTaskRecord, crate::PluginError> {
        let Some(executor) = &current.host.background_task_host else {
            return Err(crate::PluginError::Session(
                "background task registry is unavailable in this runtime".to_string(),
            ));
        };
        let scope_key = self.background_scope_key(session_id);
        let Some(status) = executor.get_managed(&scope_key, task_id).await else {
            return Err(crate::PluginError::Session(format!(
                "unknown background task `{task_id}`"
            )));
        };
        let _ = executor
            .request_cancel(task_id, Some("requested by host".to_string()))
            .await;
        match status.kind {
            crate::BackgroundTaskKind::Monitor => {
                let monitor_id = task_id
                    .strip_prefix("monitor:")
                    .unwrap_or(task_id)
                    .to_string();
                self.stop_monitor(current, host, session_id, &monitor_id)
                    .await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::BackgroundTaskState::Cancelled)
                    .await;
            }
            crate::BackgroundTaskKind::Subagent => {
                executor.cancel_managed(&scope_key, task_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::BackgroundTaskState::Cancelled)
                    .await;
            }
            _ => {
                executor.cancel_managed(&scope_key, task_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::BackgroundTaskState::Cancelled)
                    .await;
            }
        }
        let updated = executor
            .get_managed(&scope_key, task_id)
            .await
            .unwrap_or(status);
        Ok(updated)
    }

    pub(in crate::runtime::session_manager) async fn cancel_all_background_tasks(
        &self,
        current: &CurrentSessionCapability,
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
                self.cancel_background_task(current, Arc::clone(&host), session_id, &task.id)
                    .await?,
            );
        }
        Ok(cancelled)
    }

    pub(in crate::runtime::session_manager) async fn validate_async_handles_visible(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        if handle_ids.is_empty() {
            return Ok(());
        }
        let requested = handle_ids.iter().cloned().collect::<HashSet<_>>();
        let mut visible = HashSet::new();
        if let Some(map) = self
            .async_handle_map_for_session(current, managed, session_id)
            .await
        {
            visible.extend(crate::session::async_handles::live_session_async_handle_ids(&map));
        }
        if let Some(executor) = &current.host.background_task_host {
            let scope_key = self.background_scope_key(session_id);
            for task in executor.list_managed(&scope_key).await {
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
        managed: &ManagedSessionCapability,
        from_session_id: &str,
        to_session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), crate::PluginError> {
        if handle_ids.is_empty() {
            return Ok(());
        }
        let ids = handle_ids.iter().cloned().collect::<HashSet<_>>();
        let from_map = self
            .async_handle_map_for_session(current, managed, from_session_id)
            .await;
        let to_map = self
            .async_handle_map_for_session(current, managed, to_session_id)
            .await;
        if let (Some(from_map), Some(to_map)) = (from_map, to_map) {
            crate::session::async_handles::transfer_session_async_handles(&from_map, &to_map, &ids)
                .map_err(crate::PluginError::Session)?;
        }
        if let Some(executor) = &current.host.background_task_host {
            executor
                .transfer_managed(
                    &self.background_scope_key(from_session_id),
                    &self.background_scope_key(to_session_id),
                    handle_ids,
                )
                .await?;
        }
        Ok(())
    }

    pub(in crate::runtime::session_manager) async fn cancel_unreferenced_async_handles(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        host: Arc<dyn crate::plugin::RuntimeSessionHost>,
        session_id: &str,
        keep_handle_ids: &[String],
    ) -> Result<Vec<crate::BackgroundTaskRecord>, crate::PluginError> {
        let keep = keep_handle_ids.iter().cloned().collect::<HashSet<_>>();
        if let Some(map) = self
            .async_handle_map_for_session(current, managed, session_id)
            .await
        {
            crate::session::async_handles::cancel_unreferenced_session_async_handles(&map, &keep)
                .await
                .map_err(crate::PluginError::Session)?;
        }
        let tasks = self.list_background_tasks(current, session_id).await?;
        let mut cancelled = Vec::new();
        for task in tasks {
            if task.state.is_terminal() || keep.contains(&task.id) {
                continue;
            }
            cancelled.push(
                self.cancel_background_task(current, Arc::clone(&host), session_id, &task.id)
                    .await?,
            );
        }
        Ok(cancelled)
    }

    async fn async_handle_map_for_session(
        &self,
        current: &CurrentSessionCapability,
        managed: &ManagedSessionCapability,
        session_id: &str,
    ) -> Option<crate::session::AsyncToolHandleMap> {
        if session_id == current.session_id {
            return current.async_tool_handles.clone();
        }
        let runtime_handle = {
            let registry = managed.registry.lock().await;
            registry.get(session_id).cloned()
        }?;
        let runtime = runtime_handle.runtime.lock().await;
        runtime
            .session
            .as_ref()
            .map(|session| session.async_tool_handle_map())
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
