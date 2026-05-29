use crate::plugin::PluginError;

use super::events::{
    ProcessAwaitOutput, ProcessEvent, ProcessEventAppendRequest, ProcessEventAppendResult,
};
use super::model::{
    ProcessExternalRef, ProcessHandleDescriptor, ProcessHandleGrant, ProcessHandleGrantEntry,
    ProcessRecord, ProcessRegistration, ProcessScope, ProcessSessionDeleteReport,
};

/// Durability-neutral process registry.
#[async_trait::async_trait]
pub trait ProcessRegistry: Send + Sync {
    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError>;

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, PluginError>;

    async fn grant_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, PluginError>;

    async fn revoke_handle(
        &self,
        owner_scope: &ProcessScope,
        process_id: &str,
    ) -> Result<(), PluginError>;

    async fn transfer_handle_grants(
        &self,
        from_scope: &ProcessScope,
        to_scope: &ProcessScope,
        process_ids: &[String],
    ) -> Result<(), PluginError>;

    async fn list_handle_grants(
        &self,
        owner_scope: &ProcessScope,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError>;

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, PluginError>;

    async fn delete_session_process_state(
        &self,
        session_id: &str,
    ) -> Result<ProcessSessionDeleteReport, PluginError>;

    async fn append_event(
        &self,
        process_id: &str,
        request: ProcessEventAppendRequest,
    ) -> Result<ProcessEventAppendResult, PluginError>;

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError>;

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError>;

    async fn wait_event_after(
        &self,
        process_id: &str,
        event_type: &str,
        after_sequence: u64,
    ) -> Result<ProcessEvent, PluginError>;

    async fn await_process(&self, process_id: &str) -> Result<ProcessAwaitOutput, PluginError>;

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, PluginError>;

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord>;

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError>;
}
