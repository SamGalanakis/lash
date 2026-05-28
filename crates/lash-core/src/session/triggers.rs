use std::sync::Arc;

use crate::plugin::PluginError;

pub(crate) async fn emit_host_event(
    session_id: &str,
    plugins: Arc<crate::PluginSession>,
    artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
    processes: Arc<dyn crate::ProcessService>,
    surface: lashlang::LashlangSurface,
    resource_type: &str,
    alias: &str,
    event: &str,
    payload: serde_json::Value,
) -> Result<crate::HostEventEmitReport, PluginError> {
    let declared = plugins
        .host_events()
        .get(resource_type, alias, event)
        .ok_or_else(|| {
            PluginError::Session(format!(
                "unknown host event `{resource_type}.{alias}.{event}`"
            ))
        })?;
    validate_payload(&payload, &declared.payload_ty).map_err(|message| {
        PluginError::Session(format!(
            "invalid payload for host event `{resource_type}.{alias}.{event}`: {message}"
        ))
    })?;

    let triggers = plugins.installed_lashlang_triggers()?;
    if triggers.is_empty() {
        return Ok(crate::HostEventEmitReport::empty());
    }

    let mut started_process_ids = Vec::new();
    for installed in triggers {
        let program = lashlang::parse(&installed.source).map_err(|err| {
            PluginError::Session(format!(
                "parse installed trigger `{}`: {}",
                installed.name,
                lashlang::format_parse_diagnostic(&installed.source, &err)
            ))
        })?;
        let linked = lashlang::LinkedModule::link(program, surface.clone()).map_err(|err| {
            PluginError::Session(format!(
                "link installed trigger `{}`: {}",
                installed.name,
                lashlang::format_link_diagnostic(&installed.source, &err)
            ))
        })?;
        let Some(trigger) =
            linked
                .artifact
                .canonical_ir
                .declarations
                .iter()
                .find_map(|declaration| match declaration {
                    lashlang::Declaration::Trigger(trigger)
                        if trigger.name.as_str() == installed.name.as_str() =>
                    {
                        Some(trigger)
                    }
                    _ => None,
                })
        else {
            return Err(PluginError::Session(format!(
                "installed trigger `{}` is missing from its module",
                installed.name
            )));
        };
        if !trigger_matches(trigger, resource_type, alias, event) {
            continue;
        }
        artifact_store
            .put_module_artifact(&linked.artifact)
            .map_err(|err| {
                PluginError::Session(format!(
                    "store installed trigger `{}` module artifact: {err}",
                    installed.name
                ))
            })?;
        let args = trigger_process_args(trigger, resource_type, alias, payload.clone())?;
        let process_ref = linked
            .artifact
            .process_ref(trigger.process_name.as_str())
            .cloned()
            .ok_or_else(|| {
                PluginError::Session(format!(
                    "trigger `{}` target process `{}` is not exported",
                    trigger.name, trigger.process_name
                ))
            })?;
        let process_id = format!("process:{}", uuid::Uuid::new_v4());
        let args = match serde_json::to_value(lashlang::Value::Record(Arc::new(args)))
            .map_err(|err| PluginError::Session(format!("serialize trigger process args: {err}")))?
        {
            serde_json::Value::Object(map) => map,
            _ => {
                return Err(PluginError::Session(
                    "trigger process args must serialize as an object".to_string(),
                ));
            }
        };
        let registration = crate::ProcessRegistration::new(
            process_id.clone(),
            crate::ProcessInput::LashlangProcess {
                module_ref: linked.module_ref.clone(),
                process_ref,
                required_surface_ref: linked.required_surface_ref.clone(),
                process_name: trigger.process_name.to_string(),
                args,
            },
        )
        .with_extra_event_types(crate::lashlang_process_event_types());
        processes
            .start(
                session_id,
                registration,
                crate::ProcessStartOptions::new()
                    .with_wake_session_id(session_id.to_string())
                    .with_descriptor(crate::ProcessHandleDescriptor::new(
                        Some("lashlang"),
                        Some(trigger.process_name.as_str()),
                    )),
                crate::ProcessOpScope::new(),
            )
            .await?;
        started_process_ids.push(process_id);
    }

    Ok(crate::HostEventEmitReport {
        started_process_ids,
    })
}

fn trigger_matches(
    trigger: &lashlang::TriggerDecl,
    resource_type: &str,
    alias: &str,
    event: &str,
) -> bool {
    match &trigger.source {
        lashlang::TriggerSource::Binding {
            resource,
            event: trigger_event,
        } => {
            resource.resource_type.as_str() == resource_type
                && resource.alias.as_str() == alias
                && trigger_event.as_str() == event
        }
        lashlang::TriggerSource::Each {
            resource_type: trigger_resource_type,
            event: trigger_event,
            ..
        } => trigger_resource_type.as_str() == resource_type && trigger_event.as_str() == event,
    }
}

fn trigger_process_args(
    trigger: &lashlang::TriggerDecl,
    emitted_resource_type: &str,
    emitted_alias: &str,
    payload: serde_json::Value,
) -> Result<lashlang::Record, PluginError> {
    let mut args = lashlang::Record::with_capacity(trigger.args.len());
    for (name, binding) in &trigger.args {
        let value = match binding {
            lashlang::TriggerArg::EventBinding(binding)
                if binding.as_str() == trigger.event_binding.as_str() =>
            {
                lashlang::from_json(payload.clone())
            }
            lashlang::TriggerArg::ResourceBinding(binding) => {
                let lashlang::TriggerSource::Each {
                    resource_binding, ..
                } = &trigger.source
                else {
                    return Err(PluginError::Session(format!(
                        "trigger `{}` argument `{}` references unsupported binding `{}`",
                        trigger.name, name, binding
                    )));
                };
                if binding != resource_binding {
                    return Err(PluginError::Session(format!(
                        "trigger `{}` argument `{}` references unknown binding `{}`",
                        trigger.name, name, binding
                    )));
                }
                lashlang::Value::Resource(lashlang::ResourceHandle::new(
                    emitted_resource_type,
                    emitted_alias,
                ))
            }
            lashlang::TriggerArg::ResourceRef(resource) => {
                lashlang::Value::Resource(lashlang::ResourceHandle::new(
                    resource.resource_type.as_str(),
                    resource.alias.as_str(),
                ))
            }
            lashlang::TriggerArg::EventBinding(binding) => {
                return Err(PluginError::Session(format!(
                    "trigger `{}` argument `{}` references unknown binding `{}`",
                    trigger.name, name, binding
                )));
            }
        };
        args.insert(name.to_string(), value);
    }
    Ok(args)
}

fn validate_payload(value: &serde_json::Value, ty: &lashlang::TypeExpr) -> Result<(), String> {
    if json_matches_type(value, ty) {
        Ok(())
    } else {
        Err(format!("expected {}", lashlang::format_type_expr(ty)))
    }
}

fn json_matches_type(value: &serde_json::Value, ty: &lashlang::TypeExpr) -> bool {
    match ty {
        lashlang::TypeExpr::Any | lashlang::TypeExpr::Ref(_) => true,
        lashlang::TypeExpr::Str => value.is_string(),
        lashlang::TypeExpr::Int => value.as_i64().is_some() || value.as_u64().is_some(),
        lashlang::TypeExpr::Float => value.is_number(),
        lashlang::TypeExpr::Bool => value.is_boolean(),
        lashlang::TypeExpr::Dict => value.is_object(),
        lashlang::TypeExpr::Null => value.is_null(),
        lashlang::TypeExpr::Enum(values) => value
            .as_str()
            .is_some_and(|value| values.iter().any(|candidate| candidate.as_str() == value)),
        lashlang::TypeExpr::List(item) => value.as_array().is_some_and(|items| {
            items
                .iter()
                .all(|item_value| json_matches_type(item_value, item))
        }),
        lashlang::TypeExpr::Object(fields) => {
            let Some(map) = value.as_object() else {
                return false;
            };
            fields
                .iter()
                .all(|field| match map.get(field.name.as_str()) {
                    Some(field_value) => json_matches_type(field_value, &field.ty),
                    None => field.optional,
                })
        }
        lashlang::TypeExpr::Union(items) => items.iter().any(|item| json_matches_type(value, item)),
    }
}
