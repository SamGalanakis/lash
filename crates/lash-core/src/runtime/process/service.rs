use crate::plugin::PluginError;

use super::events::ProcessAwaitOutput;
use super::model::{
    ProcessHandleGrantEntry, ProcessOpScope, ProcessRecord, ProcessRegistration,
    ProcessStartOptions,
};

#[async_trait::async_trait]
pub trait ProcessService: Send + Sync {
    async fn start(
        &self,
        session_id: &str,
        registration: ProcessRegistration,
        options: ProcessStartOptions,
        scope: ProcessOpScope<'_>,
    ) -> Result<ProcessRecord, PluginError>;

    async fn await_process(
        &self,
        process_id: &str,
        scope: ProcessOpScope<'_>,
    ) -> Result<ProcessAwaitOutput, PluginError>;

    async fn list_visible(
        &self,
        session_id: &str,
        scope: ProcessOpScope<'_>,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError>;

    async fn validate_visible(
        &self,
        session_id: &str,
        process_ids: &[String],
    ) -> Result<(), PluginError>;

    async fn cancel(
        &self,
        session_id: &str,
        process_id: &str,
        scope: ProcessOpScope<'_>,
    ) -> Result<ProcessRecord, PluginError>;

    async fn cancel_all(
        &self,
        session_id: &str,
        scope: ProcessOpScope<'_>,
    ) -> Result<Vec<ProcessRecord>, PluginError>;

    async fn transfer(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        process_ids: Vec<String>,
        scope: ProcessOpScope<'_>,
    ) -> Result<(), PluginError>;

    async fn cancel_unreferenced(
        &self,
        session_id: &str,
        keep_process_ids: Vec<String>,
        scope: ProcessOpScope<'_>,
    ) -> Result<Vec<ProcessRecord>, PluginError>;
}

pub struct UnavailableProcessService;

#[async_trait::async_trait]
impl ProcessService for UnavailableProcessService {
    async fn start(
        &self,
        _session_id: &str,
        _registration: ProcessRegistration,
        _options: ProcessStartOptions,
        _scope: ProcessOpScope<'_>,
    ) -> Result<ProcessRecord, PluginError> {
        Err(PluginError::Session(
            "processes are unavailable in this runtime".to_string(),
        ))
    }

    async fn await_process(
        &self,
        _process_id: &str,
        _scope: ProcessOpScope<'_>,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        Err(PluginError::Session(
            "process awaiting is unavailable in this runtime".to_string(),
        ))
    }

    async fn list_visible(
        &self,
        _session_id: &str,
        _scope: ProcessOpScope<'_>,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
        Err(PluginError::Session(
            "process registry is unavailable in this runtime".to_string(),
        ))
    }

    async fn validate_visible(
        &self,
        _session_id: &str,
        _process_ids: &[String],
    ) -> Result<(), PluginError> {
        Err(PluginError::Session(
            "process handle validation is unavailable in this runtime".to_string(),
        ))
    }

    async fn cancel(
        &self,
        _session_id: &str,
        _process_id: &str,
        _scope: ProcessOpScope<'_>,
    ) -> Result<ProcessRecord, PluginError> {
        Err(PluginError::Session(
            "process registry is unavailable in this runtime".to_string(),
        ))
    }

    async fn cancel_all(
        &self,
        session_id: &str,
        scope: ProcessOpScope<'_>,
    ) -> Result<Vec<ProcessRecord>, PluginError> {
        let entries = self.list_visible(session_id, scope.clone()).await?;
        let mut cancelled = Vec::new();
        for (grant, record) in entries {
            if record.is_terminal() {
                continue;
            }
            cancelled.push(
                self.cancel(session_id, &grant.process_id, scope.clone())
                    .await?,
            );
        }
        Ok(cancelled)
    }

    async fn transfer(
        &self,
        _from_session_id: &str,
        _to_session_id: &str,
        process_ids: Vec<String>,
        _scope: ProcessOpScope<'_>,
    ) -> Result<(), PluginError> {
        if process_ids.is_empty() {
            return Ok(());
        }
        Err(PluginError::Session(
            "process handle transfer is unavailable in this runtime".to_string(),
        ))
    }

    async fn cancel_unreferenced(
        &self,
        _session_id: &str,
        _keep_process_ids: Vec<String>,
        _scope: ProcessOpScope<'_>,
    ) -> Result<Vec<ProcessRecord>, PluginError> {
        Err(PluginError::Session(
            "process handle cleanup is unavailable in this runtime".to_string(),
        ))
    }
}
