use std::sync::Arc;

use crate::plugin::{PluginError, SessionTriggerMatcher, SessionTriggerRoute};

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
    validate_payload(payload, &declared.payload_ty).map_err(|message| {
        PluginError::Session(format!(
            "invalid payload for host event `{resource_type}.{alias}.{event}`: {message}"
        ))
    })
}

pub(crate) async fn emit_host_event(
    session_id: &str,
    plugins: Arc<crate::PluginSession>,
    processes: Arc<dyn crate::ProcessService>,
    host_event_invocation: crate::RuntimeInvocation,
    resource_type: &str,
    alias: &str,
    event: &str,
    payload: serde_json::Value,
) -> Result<crate::HostEventEmitReport, PluginError> {
    let routes = plugins.installed_lashlang_trigger_routes()?;
    if routes.is_empty() {
        return Ok(crate::HostEventEmitReport::empty());
    }

    let mut started_process_ids = Vec::new();
    for route in routes {
        if !route_matches(&route, resource_type, alias, event) {
            continue;
        }
        let args = trigger_process_args(&route, resource_type, alias, payload.clone())?;
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
                module_ref: route.module_ref.clone(),
                process_ref: route.process_ref.clone(),
                required_surface_ref: route.required_surface_ref.clone(),
                process_name: route.process_name.clone(),
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
                        Some(route.process_name.as_str()),
                    )),
                crate::ProcessOpScope::new()
                    .with_parent_invocation(Some(host_event_invocation.clone())),
            )
            .await?;
        started_process_ids.push(process_id);
    }

    Ok(crate::HostEventEmitReport {
        started_process_ids,
    })
}

fn route_matches(
    route: &SessionTriggerRoute,
    resource_type: &str,
    alias: &str,
    event: &str,
) -> bool {
    match &route.matcher {
        SessionTriggerMatcher::Resource {
            resource_type: trigger_resource_type,
            alias: trigger_alias,
            event: trigger_event,
        } => {
            trigger_resource_type == resource_type
                && trigger_alias == alias
                && trigger_event == event
        }
        SessionTriggerMatcher::AnyResource {
            resource_type: trigger_resource_type,
            event: trigger_event,
            ..
        } => trigger_resource_type == resource_type && trigger_event == event,
    }
}

fn trigger_process_args(
    route: &SessionTriggerRoute,
    emitted_resource_type: &str,
    emitted_alias: &str,
    payload: serde_json::Value,
) -> Result<lashlang::Record, PluginError> {
    let mut args = lashlang::Record::with_capacity(route.args.len());
    for arg in &route.args {
        let value = match &arg.value {
            lashlang::TriggerArg::EventBinding(event_binding)
                if event_binding.as_str() == route.event_binding.as_str() =>
            {
                lashlang::from_json(payload.clone())
            }
            lashlang::TriggerArg::ResourceBinding(resource_binding_name) => {
                let resource_binding = match &route.matcher {
                    SessionTriggerMatcher::AnyResource {
                        resource_binding, ..
                    } => resource_binding,
                    SessionTriggerMatcher::Resource { .. } => {
                        return Err(PluginError::Session(format!(
                            "trigger `{}` argument `{}` references unsupported binding `{}`",
                            route.name, arg.name, resource_binding_name
                        )));
                    }
                };
                if resource_binding_name.as_str() != resource_binding.as_str() {
                    return Err(PluginError::Session(format!(
                        "trigger `{}` argument `{}` references unknown binding `{}`",
                        route.name, arg.name, resource_binding_name
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
            lashlang::TriggerArg::EventBinding(event_binding) => {
                return Err(PluginError::Session(format!(
                    "trigger `{}` argument `{}` references unknown binding `{}`",
                    route.name, arg.name, event_binding
                )));
            }
        };
        args.insert(arg.name.clone(), value);
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
