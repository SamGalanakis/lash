use std::time::SystemTime;

use crate::plugin::PluginError;

use super::events::{ProcessWake, ProcessWakeDelivery};
use super::model::{ProcessId, ProcessScope};
use super::time::epoch_ms_from_system_time;

pub fn process_wake_delivery(
    target_scope: ProcessScope,
    process_id: ProcessId,
    sequence: u64,
    wake: ProcessWake,
    occurred_at: SystemTime,
) -> Result<ProcessWakeDelivery, PluginError> {
    let target_scope_id = target_scope.id();
    let wake_id = crate::stable_hash::stable_json_sha256_hex(&(
        target_scope_id.as_str(),
        wake.dedupe_key.as_str(),
    ))
    .map_err(|err| {
        PluginError::Session(format!(
            "failed to hash wake delivery for process `{process_id}`: {err}"
        ))
    })?;
    Ok(ProcessWakeDelivery {
        wake_id: format!("wake:{wake_id}"),
        target_session_id: target_scope.session_id,
        target_scope_id,
        process_id,
        sequence,
        dedupe_key: wake.dedupe_key,
        input: wake.input,
        created_at_ms: epoch_ms_from_system_time(occurred_at),
    })
}
