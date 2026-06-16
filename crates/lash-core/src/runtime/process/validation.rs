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

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProcessEnvValidationRuntime {
    pub(crate) process_registry_available: bool,
}

pub(crate) async fn validate_lashlang_process_execution_env(
    artifact_store: &dyn lashlang::LashlangArtifactStore,
    plugin_host: &crate::PluginHost,
    session_id: &str,
    runtime: ProcessEnvValidationRuntime,
    input: &super::model::ProcessInput,
    env_spec: &super::model::ProcessExecutionEnvSpec,
) -> Result<(), PluginError> {
    let super::model::ProcessInput::LashlangProcess {
        module_ref,
        process_ref,
        host_requirements_ref,
        process_name,
        ..
    } = input
    else {
        return Ok(());
    };

    let artifact = artifact_store
        .get_module_artifact(module_ref)
        .await
        .map_err(|err| {
            PluginError::Session(format!(
                "failed to load lashlang module artifact `{module_ref}` while validating process environment: {err}"
            ))
        })?
        .ok_or_else(|| {
            PluginError::Session(format!(
                "missing lashlang module artifact `{module_ref}` while validating process environment"
            ))
        })?;
    if artifact.host_requirements_ref != *host_requirements_ref {
        return Err(PluginError::Session(format!(
            "lashlang process `{process_name}` requested host requirements {}, artifact has {}",
            host_requirements_ref, artifact.host_requirements_ref
        )));
    }
    if artifact.process_ref(process_name) != Some(process_ref) {
        return Err(PluginError::Session(format!(
            "lashlang module `{module_ref}` does not export process `{process_name}` as requested ref {:?}",
            process_ref
        )));
    }

    let host = plugin_host.clone().with_lashlang_abilities(
        crate::runtime::builder::lashlang_abilities_for_process_registry(
            plugin_host.lashlang_abilities(),
            runtime.process_registry_available,
        ),
    );
    let plugins = host
        .isolated_registry()
        .build_session_with_parent(
            session_id.to_string(),
            None,
            None,
            crate::plugin::SessionAuthorityContext {
                plugin_options: env_spec.plugin_options.clone(),
                ..Default::default()
            },
        )
        .map_err(|err| {
            PluginError::Session(format!(
                "failed to rebuild process environment plugin options for `{process_name}`: {err}"
            ))
        })?;
    let tool_catalog = plugins.resolved_tool_catalog(session_id)?;
    let lashlang_abilities = crate::runtime::builder::lashlang_abilities_for_process_registry(
        plugins.lashlang_abilities(),
        runtime.process_registry_available,
    );
    let current_environment = crate::session::lashlang_host_environment_from_tool_catalog(
        &tool_catalog,
        lashlang_abilities,
        plugins.lashlang_language_features(),
        plugins.lashlang_resources(),
    );
    lashlang_host_environment_satisfies_requirements(
        &artifact.host_requirements,
        &current_environment,
    )
    .map_err(|err| {
        PluginError::Session(format!(
            "lashlang process `{process_name}` is incompatible with captured process environment: {err}"
        ))
    })
}

pub(crate) fn lashlang_host_environment_satisfies_requirements(
    required: &lashlang::HostRequirements,
    current: &lashlang::LashlangHostEnvironment,
) -> Result<(), String> {
    let abilities = required.abilities;
    let current_abilities = current.abilities;
    if abilities.processes && !current_abilities.processes {
        return Err("processes are not available".to_string());
    }
    if abilities.sleep && !current_abilities.sleep {
        return Err("sleep is not available".to_string());
    }
    if abilities.process_signals && !current_abilities.process_signals {
        return Err("process signals are not available".to_string());
    }
    if abilities.triggers && !current_abilities.triggers {
        return Err("triggers are not available".to_string());
    }
    if required.language_features.label_annotations && !current.language_features.label_annotations
    {
        return Err("label annotations are not available".to_string());
    }

    for (_, module) in required.resources.module_instances() {
        let current_module = current
            .resources
            .resolve_module_path(&module.path)
            .ok_or_else(|| format!("module `{}` is not available", module.alias))?;
        if current_module.resource_type != module.resource_type {
            return Err(format!(
                "module `{}` has type `{}`, expected `{}`",
                module.alias, current_module.resource_type, module.resource_type
            ));
        }
        for (operation, required_binding) in &module.operations {
            match current.resources.resolve_module_operation(
                &module.resource_type,
                &module.alias,
                operation,
            ) {
                Some(current_binding) if current_binding == required_binding => {}
                Some(current_binding) => {
                    return Err(format!(
                        "module `{}` operation `{operation}` resolves to `{}`, expected `{}`",
                        module.alias,
                        current_binding.host_operation,
                        required_binding.host_operation
                    ));
                }
                None => {
                    return Err(format!(
                        "module `{}` does not expose operation `{operation}`",
                        module.alias
                    ));
                }
            }
        }
    }

    for (resource_type, required_type) in required.resources.resource_types() {
        if !current.resources.has_resource_type(resource_type) {
            return Err(format!("resource type `{resource_type}` is not available"));
        }
        for (operation, required_binding) in &required_type.operations {
            let current_binding = current
                .resources
                .resolve_operation(resource_type, operation)
                .ok_or_else(|| {
                    format!(
                        "resource type `{resource_type}` does not expose operation `{operation}`"
                    )
                })?;
            if current_binding.input_ty != required_binding.input_ty {
                return Err(format!(
                    "resource type `{resource_type}` operation `{operation}` has incompatible input type"
                ));
            }
            if current_binding.output_ty != required_binding.output_ty {
                return Err(format!(
                    "resource type `{resource_type}` operation `{operation}` has incompatible output type"
                ));
            }
        }
    }
    for (name, required_data_type) in required.resources.named_data_types() {
        let current_data_type = current
            .resources
            .resolve_named_data_type(name)
            .ok_or_else(|| format!("host data type `{name}` is not available"))?;
        if current_data_type != required_data_type {
            return Err(format!(
                "host data type `{name}` has incompatible structure"
            ));
        }
    }
    for (path, required_binding) in required.resources.value_constructors() {
        let current_binding = current
            .resources
            .resolve_value_constructor(&path.split('.').collect::<Vec<_>>())
            .ok_or_else(|| format!("value constructor `{path}` is not available"))?;
        if current_binding.input_ty != required_binding.input_ty {
            return Err(format!(
                "value constructor `{path}` has incompatible input type"
            ));
        }
        if current_binding.output_ty != required_binding.output_ty {
            return Err(format!(
                "value constructor `{path}` has incompatible output type"
            ));
        }
    }
    for (source_ty, required_binding) in required.resources.trigger_sources() {
        let current_binding = current
            .resources
            .resolve_trigger_source(source_ty)
            .ok_or_else(|| format!("trigger source type `{source_ty}` is not available"))?;
        if current_binding != required_binding {
            return Err(format!(
                "trigger source type `{source_ty}` has incompatible event type"
            ));
        }
    }

    Ok(())
}

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
