use crate::plugin::PluginError;

use super::ToolProcessEventContext;

pub(crate) async fn enqueue_wake_delivery(
    store: Option<&dyn crate::RuntimePersistence>,
    wake_delivery: Option<crate::ProcessWakeDelivery>,
) -> Result<(), PluginError> {
    let Some(wake_delivery) = wake_delivery else {
        return Ok(());
    };
    let Some(store) = store else {
        return Err(PluginError::Session(format!(
            "process wake for session `{}` requires a runtime persistence store",
            wake_delivery.target_session_id
        )));
    };
    store
        .enqueue_queued_work(crate::process_wake_batch_draft(wake_delivery))
        .await
        .map(|_| ())
        .map_err(|err| PluginError::Session(err.to_string()))
}

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
        let result = process
            .registry
            .append_event(
                &process.process_id,
                request.with_optional_wake_target_scope(process.wake_target_scope.clone()),
            )
            .await?;
        enqueue_wake_delivery(process.store.as_deref(), result.wake_delivery).await?;
        Ok(result.event)
    }
}
