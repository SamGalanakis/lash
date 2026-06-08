use crate::plugin::PluginError;

pub(crate) fn validate_host_event(
    plugins: &crate::PluginSession,
    resource_type: &str,
    alias: &str,
    event: &str,
    payload: &serde_json::Value,
) -> Result<(), PluginError> {
    let declared = plugins
        .host_events()
        .get(resource_type, alias, event)
        .ok_or_else(|| {
            PluginError::Session(format!(
                "unknown host event `{resource_type}.{alias}.{event}`"
            ))
        })?;
    crate::host_events::validate_payload(payload, declared.payload_type().ty()).map_err(|message| {
        PluginError::Session(format!(
            "invalid payload for host event `{resource_type}.{alias}.{event}`: {message}"
        ))
    })
}
