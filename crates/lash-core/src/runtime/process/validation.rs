use std::collections::HashSet;

use crate::plugin::PluginError;

use super::events::{
    ProcessEvent, ProcessEventAppendRequest, ProcessEventSemanticsSpec, ProcessTerminalState,
    ProcessWakeDelivery, default_process_event_types,
};
use super::materialization::materialize_process_event_semantics;
use super::model::{ProcessRecord, ProcessRegistration, ProcessStatus};
use super::time::{epoch_ms_from_system_time, system_time_from_epoch_ms};
use super::wake::{ProcessWakeDeliveryRequest, process_wake_delivery};

#[derive(Clone, Debug)]
pub struct PreparedProcessEventAppend {
    pub event: ProcessEvent,
    pub payload_hash: String,
    pub status_update: Option<ProcessStatus>,
    pub wake_delivery: Option<ProcessWakeDelivery>,
    pub occurred_at_ms: u64,
    pub replayed: bool,
}

pub fn prepare_process_event_append(
    record: &ProcessRecord,
    request: ProcessEventAppendRequest,
    sequence: u64,
    replay_lookup: Option<(String, ProcessEvent)>,
    occurred_at_ms: u64,
) -> Result<PreparedProcessEventAppend, PluginError> {
    let process_id = record.id.as_str();
    let payload_hash = process_event_payload_hash(&request.event_type, &request.payload)?;
    if let Some(replay_key) = request.replay.as_ref().map(|replay| replay.key.as_str())
        && let Some((existing_hash, existing)) = replay_lookup
    {
        if existing_hash == payload_hash {
            let status_update = existing.semantics.terminal.clone().and_then(|terminal| {
                (!record.is_terminal()).then(|| ProcessStatus::from_terminal(terminal))
            });
            let occurred_at_ms = epoch_ms_from_system_time(existing.occurred_at);
            let wake_delivery = prepare_wake_delivery(
                process_id,
                record,
                existing.sequence,
                existing.event_type.clone(),
                existing.invocation.clone(),
                existing.occurred_at,
                existing.semantics.wake.clone(),
                request
                    .wake_target_scope
                    .clone()
                    .or_else(|| record.wake_target.clone()),
            )?;
            return Ok(PreparedProcessEventAppend {
                event: existing,
                payload_hash,
                status_update,
                wake_delivery,
                occurred_at_ms,
                replayed: true,
            });
        }
        return Err(PluginError::Session(format!(
            "process `{process_id}` event replay key `{replay_key}` conflicts with an existing event"
        )));
    }
    let declared = record
        .event_types
        .iter()
        .find(|declared| declared.name == request.event_type)
        .ok_or_else(|| {
            PluginError::Session(format!(
                "process `{process_id}` emitted undeclared event type `{}`",
                request.event_type
            ))
        })?;
    require_event_replay(process_id, &request, &declared.semantics)?;
    declared
        .payload_schema
        .validate(&request.payload)
        .map_err(|err| {
            PluginError::Session(format!("invalid `{}` payload: {err}", request.event_type))
        })?;
    let semantics = materialize_process_event_semantics(
        process_id,
        sequence,
        &request.payload,
        &declared.semantics,
    )?;
    if semantics.terminal.is_some() && record.is_terminal() {
        return Err(PluginError::Session(format!(
            "process `{process_id}` is already terminal"
        )));
    }
    let occurred_at = system_time_from_epoch_ms(occurred_at_ms);
    let event = ProcessEvent {
        process_id: process_id.to_string(),
        sequence,
        event_type: request.event_type,
        payload: request.payload,
        invocation: crate::runtime::causal::process_event_invocation(
            process_id,
            sequence,
            declared.name.as_str(),
            request.replay,
        ),
        semantics: semantics.clone(),
        occurred_at,
    };
    let wake_delivery = prepare_wake_delivery(
        process_id,
        record,
        event.sequence,
        event.event_type.clone(),
        event.invocation.clone(),
        event.occurred_at,
        semantics.wake.clone(),
        request
            .wake_target_scope
            .or_else(|| record.wake_target.clone()),
    )?;
    Ok(PreparedProcessEventAppend {
        event,
        payload_hash,
        status_update: semantics.terminal.map(ProcessStatus::from_terminal),
        wake_delivery,
        occurred_at_ms,
        replayed: false,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "wake delivery mirrors the persisted event plus its optional materialized wake"
)]
fn prepare_wake_delivery(
    process_id: &str,
    record: &ProcessRecord,
    sequence: u64,
    event_type: String,
    event_invocation: crate::RuntimeInvocation,
    occurred_at: std::time::SystemTime,
    wake: Option<super::events::ProcessWake>,
    wake_target_scope: Option<super::model::SessionScope>,
) -> Result<Option<ProcessWakeDelivery>, PluginError> {
    let Some(wake) = wake else {
        return Ok(None);
    };
    let Some(target_scope) = wake_target_scope else {
        return Ok(None);
    };
    process_wake_delivery(ProcessWakeDeliveryRequest {
        target_scope,
        process_id: process_id.to_string(),
        sequence,
        event_type,
        event_invocation,
        process_caused_by: record.provenance.caused_by.clone(),
        wake,
        occurred_at,
    })
    .map(Some)
}

pub fn prepare_process_registration(
    mut registration: ProcessRegistration,
) -> Result<(ProcessRegistration, String), PluginError> {
    ensure_core_event_types(&mut registration);
    validate_process_registration(&registration)?;
    let registration_hash = process_registration_hash(&registration)?;
    Ok((registration, registration_hash))
}

pub fn process_registration_hash(
    registration: &ProcessRegistration,
) -> Result<String, PluginError> {
    crate::stable_hash::stable_json_sha256_hex(registration).map_err(|err| {
        PluginError::Session(format!(
            "failed to hash process `{}` registration: {err}",
            registration.id
        ))
    })
}

pub fn process_event_payload_hash(
    event_type: &str,
    payload: &serde_json::Value,
) -> Result<String, PluginError> {
    crate::stable_hash::stable_json_sha256_hex(&(event_type, payload)).map_err(|err| {
        PluginError::Session(format!(
            "failed to hash `{event_type}` process event: {err}"
        ))
    })
}

pub fn require_event_replay(
    process_id: &str,
    request: &ProcessEventAppendRequest,
    spec: &ProcessEventSemanticsSpec,
) -> Result<(), PluginError> {
    let requires_key =
        spec.terminal.is_some() || request.event_type.as_str() == "process.cancel_requested";
    if requires_key
        && request
            .replay
            .as_ref()
            .is_none_or(|replay| replay.key.is_empty())
    {
        return Err(PluginError::Session(format!(
            "process `{process_id}` event `{}` requires a deterministic replay key",
            request.event_type
        )));
    }
    Ok(())
}

pub(super) fn ensure_core_event_types(registration: &mut ProcessRegistration) {
    let mut existing = registration
        .event_types
        .iter()
        .map(|event_type| event_type.name.clone())
        .collect::<HashSet<_>>();
    for event_type in default_process_event_types() {
        if existing.insert(event_type.name.clone()) {
            registration.event_types.push(event_type);
        }
    }
}

pub(super) fn validate_process_registration(
    registration: &ProcessRegistration,
) -> Result<(), PluginError> {
    if registration.id.trim().is_empty() {
        return Err(PluginError::Session(
            "process id must be a non-empty string".to_string(),
        ));
    }
    if registration.provenance.host_profile_id.trim().is_empty() {
        return Err(PluginError::Session(format!(
            "process `{}` host profile id must be non-empty",
            registration.id
        )));
    }
    match registration.input.as_ref() {
        super::model::ProcessInput::ToolCall { .. }
        | super::model::ProcessInput::LashlangProcess { .. } => {
            if registration.env_ref.is_none() {
                return Err(PluginError::Session(format!(
                    "process `{}` requires a captured execution env",
                    registration.id
                )));
            }
        }
        super::model::ProcessInput::External { .. }
        | super::model::ProcessInput::SessionTurn { .. } => {
            if registration.env_ref.is_some() {
                return Err(PluginError::Session(format!(
                    "process `{}` must not capture an execution env for this input kind",
                    registration.id
                )));
            }
        }
    }
    let mut names = HashSet::new();
    for event_type in &registration.event_types {
        if event_type.name.trim().is_empty() {
            return Err(PluginError::Session(format!(
                "process `{}` declares an empty event type",
                registration.id
            )));
        }
        if !names.insert(event_type.name.as_str()) {
            return Err(PluginError::Session(format!(
                "process `{}` declares duplicate event type `{}`",
                registration.id, event_type.name
            )));
        }
        if let Some(terminal) = &event_type.semantics.terminal
            && terminal.state != ProcessTerminalState::Completed
            && terminal.await_output.is_none()
        {
            return Err(PluginError::Session(format!(
                "terminal event `{}` for process `{}` must declare await output",
                event_type.name, registration.id
            )));
        }
    }
    Ok(())
}
