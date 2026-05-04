use super::*;
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
        let Some(executor) = &current.host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "session tasks are unavailable in this runtime".to_string(),
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
        let Some(executor) = &current.host.session_task_executor else {
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
        spec: crate::ManagedTaskSpec,
        task: crate::plugin::PluginSessionTask,
    ) -> Result<(), crate::PluginError> {
        self.ensure_known_background_session(current, managed, session_id)
            .await?;
        let Some(executor) = &current.host.session_task_executor else {
            return Err(crate::PluginError::Session(
                "managed session tasks are unavailable in this runtime".to_string(),
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
        let Some(executor) = &current.host.session_task_executor else {
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
        spec: crate::ManagedTaskSpec,
        cancel: Option<crate::ManagedTaskCancel>,
    ) -> Result<(), crate::PluginError> {
        let Some(executor) = &current.host.session_task_executor else {
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
        let Some(executor) = &current.host.session_task_executor else {
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
        run_state: crate::ManagedRunState,
    ) {
        let Some(executor) = &current.host.session_task_executor else {
            return;
        };
        executor
            .mark_terminal(&self.background_scope_key(session_id), task_id, run_state)
            .await;
    }

    pub(in crate::runtime::session_manager) async fn transition_background_task_live_state(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
        task_id: &str,
        run_state: crate::ManagedRunState,
    ) {
        let Some(executor) = &current.host.session_task_executor else {
            return;
        };
        executor
            .mark_live_state(&self.background_scope_key(session_id), task_id, run_state)
            .await;
    }

    pub(in crate::runtime::session_manager) async fn list_background_tasks(
        &self,
        current: &CurrentSessionCapability,
        session_id: &str,
    ) -> Result<Vec<crate::ManagedTaskStatus>, crate::PluginError> {
        let Some(executor) = &current.host.session_task_executor else {
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
    ) -> Result<crate::ManagedTaskStatus, crate::PluginError> {
        let Some(executor) = &current.host.session_task_executor else {
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
        match status.kind {
            crate::ManagedTaskKind::Monitor => {
                let monitor_id = task_id
                    .strip_prefix("monitor:")
                    .unwrap_or(task_id)
                    .to_string();
                self.stop_monitor(current, host, session_id, &monitor_id)
                    .await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
            crate::ManagedTaskKind::Subagent => {
                executor.cancel_managed(&scope_key, task_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
            _ => {
                executor.cancel_managed(&scope_key, task_id).await?;
                executor
                    .mark_terminal(&scope_key, task_id, crate::ManagedRunState::Cancelled)
                    .await;
            }
        }
        let updated = executor
            .get_managed(&scope_key, task_id)
            .await
            .unwrap_or(status);
        Ok(updated)
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
