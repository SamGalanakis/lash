use crate::plugin::PluginError;

use super::ToolProcessEventContext;

#[derive(Clone)]
pub struct ToolProcessEventControl {
    pub(super) context: Option<ToolProcessEventContext>,
}

impl ToolProcessEventControl {
    pub async fn emit(
        &self,
        event_type: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<crate::ProcessEvent, PluginError> {
        self.emit_request(crate::ProcessEventAppendRequest::new(event_type, payload))
            .await
    }

    pub async fn emit_request(
        &self,
        request: crate::ProcessEventAppendRequest,
    ) -> Result<crate::ProcessEvent, PluginError> {
        let Some(process) = self.context.as_ref() else {
            return Err(PluginError::Session(
                "process event emission is unavailable outside a durable process".to_string(),
            ));
        };
        process
            .registry
            .append_event(
                &process.process_id,
                request.with_optional_wake_target_scope(process.wake_target_scope.clone()),
            )
            .await
    }
}
