use crate::{HostEventEmitReport, PluginError};

use super::ToolContext;

#[derive(Clone)]
pub struct ToolHostEventControl<'run> {
    pub(super) context: ToolContext<'run>,
}

impl<'run> ToolHostEventControl<'run> {
    pub async fn emit(
        &self,
        resource_type: impl AsRef<str>,
        alias: impl AsRef<str>,
        event: impl AsRef<str>,
        payload: serde_json::Value,
    ) -> Result<HostEventEmitReport, PluginError> {
        let Some(dispatch) = self.context.runtime_dispatch.clone() else {
            return Err(PluginError::Session(
                "host event emission is unavailable outside runtime tool execution".to_string(),
            ));
        };
        let resource_type = resource_type.as_ref();
        let alias = alias.as_ref();
        let event = event.as_ref();
        crate::session::triggers::validate_host_event(
            dispatch.plugins.as_ref(),
            resource_type,
            alias,
            event,
            &payload,
        )?;
        let source_type = crate::host_event_source_type(alias, event);
        let idempotency_key = tool_event_scope_id(&self.context, &source_type, &payload)?;
        let source_key = crate::empty_host_event_source_key(&source_type)?;
        let router = dispatch.host_event_router.as_ref().ok_or_else(|| {
            PluginError::Session("host event store is unavailable in this runtime".to_string())
        })?;
        let report = router
            .emit(
                crate::HostEventOccurrenceRequest::new(
                    &source_type,
                    &source_key,
                    payload.clone(),
                    idempotency_key,
                )
                .with_source(serde_json::json!({})),
                dispatch.effect_controller.controller(),
            )
            .await?;
        dispatch
            .host_event_outcomes
            .enqueue(crate::tool_dispatch::ToolHostEventEffectOutcome {
                resource_type: resource_type.to_string(),
                alias: alias.to_string(),
                event: event.to_string(),
                source_type,
                source_key,
                occurrence_id: report.occurrence_id.clone(),
                payload: payload.clone(),
                started_process_ids: report.started_process_ids.clone(),
            })
            .map_err(PluginError::Session)?;
        Ok(report)
    }
}

fn tool_event_scope_id(
    context: &ToolContext<'_>,
    source_type: &str,
    payload: &serde_json::Value,
) -> Result<String, PluginError> {
    if let Some(tool_call_id) = context.tool_call_id() {
        return Ok(format!("tool:{tool_call_id}:{source_type}"));
    }
    let digest = crate::stable_hash::stable_json_sha256_hex(&(source_type, payload))
        .map_err(|err| PluginError::Session(format!("hash tool host event: {err}")))?;
    Ok(format!("tool:{source_type}:sha256:{digest}"))
}
