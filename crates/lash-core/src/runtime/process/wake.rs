use std::time::SystemTime;

use crate::plugin::PluginError;

use super::events::{ProcessWake, ProcessWakeDelivery};
use super::model::{ProcessId, ProcessScope};
use super::time::epoch_ms_from_system_time;

/// Extracts the model-facing wake input from a process wake event payload.
pub fn process_wake_input_from_event_payload(payload: &serde_json::Value) -> String {
    payload
        .pointer("/text")
        .or_else(|| payload.pointer("/value"))
        .map(wake_payload_value_to_string)
        .unwrap_or_else(|| payload.to_string())
}

/// Renders a durable process wake as model-visible chronological context.
pub fn process_wake_turn_text(wake: &ProcessWakeDelivery) -> String {
    format!(
        "Background process wake\nProcess: {}\nEvent: process.wake #{}\nWake input:\n{}",
        wake.process_id, wake.sequence, wake.input
    )
}

pub fn process_wake_turn_cause(wake: &ProcessWakeDelivery) -> crate::TurnCause {
    crate::TurnCause {
        id: wake.wake_id.clone(),
        event_type: "process.wake".to_string(),
        origin: crate::MessageOrigin::Process {
            process_id: wake.process_id.clone(),
            event_type: "process.wake".to_string(),
            sequence: wake.sequence,
            wake_id: Some(wake.wake_id.clone()),
        },
        text: process_wake_turn_text(wake),
    }
}

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

fn wake_payload_value_to_string(value: &serde_json::Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}
