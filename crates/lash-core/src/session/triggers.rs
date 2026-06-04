use std::sync::Arc;

use crate::plugin::{PluginError, SessionTriggerRegistry};

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
    validate_payload(payload, declared.payload_type().ty()).map_err(|message| {
        PluginError::Session(format!(
            "invalid payload for host event `{resource_type}.{alias}.{event}`: {message}"
        ))
    })
}

pub struct TriggerActivationService<'a> {
    session_id: String,
    registry: Arc<SessionTriggerRegistry>,
    processes: Arc<dyn crate::ProcessService>,
    scoped_effect_controller: crate::ScopedEffectController<'a>,
}

impl<'a> TriggerActivationService<'a> {
    pub(crate) fn new(
        session_id: String,
        registry: Arc<SessionTriggerRegistry>,
        processes: Arc<dyn crate::ProcessService>,
        scoped_effect_controller: crate::ScopedEffectController<'a>,
    ) -> Self {
        Self {
            session_id,
            registry,
            processes,
            scoped_effect_controller,
        }
    }

    pub async fn activate(
        &self,
        handle: impl AsRef<str>,
        event_payload: serde_json::Value,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> Result<Option<String>, PluginError> {
        let Some(route) = self.registry.route(handle.as_ref())? else {
            return Ok(None);
        };
        if !route.enabled {
            return Ok(None);
        }
        self.start_route(route, event_payload, parent_invocation)
            .await
    }

    pub async fn activate_source_type(
        &self,
        source_type: impl AsRef<str>,
        event_payload: serde_json::Value,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> Result<Vec<String>, PluginError> {
        let routes = self
            .registry
            .activation_routes_by_source_type(source_type.as_ref())?;
        let mut started_process_ids = Vec::new();
        for route in routes {
            if !route.enabled {
                continue;
            }
            if let Some(process_id) = self
                .start_route(route, event_payload.clone(), parent_invocation.clone())
                .await?
            {
                started_process_ids.push(process_id);
            }
        }
        Ok(started_process_ids)
    }

    async fn start_route(
        &self,
        route: crate::plugin::SessionTriggerRoute,
        event_payload: serde_json::Value,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> Result<Option<String>, PluginError> {
        validate_payload(&event_payload, &route.event_ty).map_err(|message| {
            PluginError::Session(format!(
                "invalid payload for trigger `{}`: {message}",
                route.handle
            ))
        })?;
        let mut args = lashlang::Record::default();
        for (input_name, input) in route.input_template.entries() {
            let value = match input {
                lashlang::TriggerInputBinding::Event => event_payload.clone(),
                lashlang::TriggerInputBinding::Fixed { value } => value.clone(),
            };
            args.insert(input_name.to_string(), lashlang::from_json(value));
        }
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
        let process_id = format!("process:{}", uuid::Uuid::new_v4());
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
        let scoped_effect_controller = crate::ScopedEffectController::borrowed(
            self.scoped_effect_controller.controller(),
            crate::EffectScope::host_event(
                &self.session_id,
                format!(
                    "{}:{}",
                    self.scoped_effect_controller.scope_id(),
                    route.handle
                ),
            ),
        )
        .map_err(|err| PluginError::Session(err.to_string()))?;
        self.processes
            .start(
                &self.session_id,
                registration,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(
                        Some("lashlang"),
                        Some(route.process_name.as_str()),
                    ),
                ),
                crate::ProcessOpScope::new(scoped_effect_controller)
                    .with_parent_invocation(parent_invocation),
            )
            .await?;
        Ok(Some(process_id))
    }
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
        lashlang::TypeExpr::Any => true,
        lashlang::TypeExpr::Ref(_) => false,
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
        lashlang::TypeExpr::Process { .. } | lashlang::TypeExpr::TriggerHandle(_) => {
            value.is_object()
        }
    }
}
