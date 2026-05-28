use std::collections::HashSet;

use crate::plugin::PluginError;

use super::events::{
    ProcessEventAppendRequest, ProcessEventSemanticsSpec, ProcessTerminalState,
    default_process_event_types,
};
use super::model::ProcessRegistration;

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

pub fn require_event_idempotency(
    process_id: &str,
    request: &ProcessEventAppendRequest,
    spec: &ProcessEventSemanticsSpec,
) -> Result<(), PluginError> {
    let requires_key =
        spec.terminal.is_some() || request.event_type.as_str() == "process.cancel_requested";
    if requires_key && request.idempotency_key.as_deref().is_none_or(str::is_empty) {
        return Err(PluginError::Session(format!(
            "process `{process_id}` event `{}` requires a deterministic idempotency key",
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
    if let Some(scope) = &registration.created_by_scope {
        if scope.is_empty() {
            return Err(PluginError::Session(format!(
                "process `{}` creator scope must include a session id",
                registration.id
            )));
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
