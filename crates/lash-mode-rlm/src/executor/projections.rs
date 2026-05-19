use std::sync::Arc;

use lash_core::ModeExecutionContext;
use lashlang::{
    ProjectedBindings, ProjectedFuture, ProjectedHostValue, ProjectedReadRequest,
    ProjectedReadResponse, ProjectedValue, State as FlowState, Value as FlowValue,
};

use crate::projected_bindings::{
    ProjectionRef, ProjectionResolver, RLM_TURN_INPUT_PLUGIN_ID, RlmProjectedBindings,
    RlmProjectionExtension,
};
use crate::projection::{RlmHistoryProjection, rlm_history_projection};
use crate::projection_codec::json_to_flow_value;

pub(super) async fn projected_bindings(
    ctx: &ModeExecutionContext<'_>,
    session_bindings: RlmProjectedBindings,
    projection_resolver: Arc<dyn ProjectionResolver>,
) -> Result<ProjectedBindings, String> {
    let mut bindings = ProjectedBindings::new();
    bindings
        .try_insert(
            "history",
            ProjectedValue::custom(
                "history",
                Arc::new(HistoryProjectedValue {
                    projection: Arc::new(rlm_history_projection(
                        ctx.chronological_projection().as_ref(),
                    )),
                }),
            ),
        )
        .map_err(|err| format!("`{}` is reserved as an RLM built-in binding", err.name()))?;
    insert_projected_bindings(
        &mut bindings,
        session_bindings,
        Arc::clone(&projection_resolver),
    )
    .await?;
    if let Some(extension) = ctx
        .turn_context()
        .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
    {
        insert_projected_bindings(
            &mut bindings,
            extension.bindings.clone(),
            projection_resolver,
        )
        .await?;
    }
    Ok(bindings)
}

async fn insert_projected_bindings(
    target: &mut ProjectedBindings,
    bindings: RlmProjectedBindings,
    projection_resolver: Arc<dyn ProjectionResolver>,
) -> Result<(), String> {
    let host_bindings = bindings
        .into_projected_bindings(projection_resolver)
        .await
        .map_err(|err| err.to_string())?;
    for name in host_bindings.names().collect::<Vec<_>>() {
        let value = host_bindings
            .get(&name)
            .expect("name came from projected bindings");
        target.try_insert(name, value).map_err(|err| {
            format!(
                "`{}` is already bound as an RLM projected binding",
                err.name()
            )
        })?;
    }
    Ok(())
}

pub(super) async fn rehydrate_projected_globals(
    rlm: &mut FlowState,
    projection_resolver: Arc<dyn ProjectionResolver>,
) -> Result<(), String> {
    let mut snapshot = rlm.snapshot();
    let mut changed = false;
    let keys = snapshot
        .globals
        .keys()
        .map(str::to_string)
        .collect::<Vec<_>>();
    for key in keys {
        if let Some(value) = snapshot.globals.get_mut(&key) {
            changed |= rehydrate_projected_value(value, Arc::clone(&projection_resolver)).await?;
        }
    }
    if changed {
        *rlm = FlowState::from_snapshot(snapshot);
    }
    Ok(())
}

fn rehydrate_projected_value<'a>(
    value: &'a mut FlowValue,
    projection_resolver: Arc<dyn ProjectionResolver>,
) -> ProjectedFuture<'a, Result<bool, String>> {
    Box::pin(async move {
        match value {
            FlowValue::Projected(projected) => {
                let Some(ref_json) = projected.projection_ref().cloned() else {
                    return Ok(false);
                };
                let name = projected.name().to_string();
                let reference = serde_json::from_value::<ProjectionRef>(ref_json.clone())
                    .map_err(|err| format!("invalid projection ref for `{name}`: {err}"))?;
                let resolved = projection_resolver
                    .resolve_projection(&reference)
                    .await
                    .map_err(|err| err.to_string())?;
                *value = FlowValue::Projected(ProjectedValue::custom_with_projection_ref(
                    name, resolved, ref_json,
                ));
                Ok(true)
            }
            FlowValue::List(values) => {
                let mut changed = false;
                let mut restored = values.iter().cloned().collect::<Vec<_>>();
                for value in restored.iter_mut() {
                    changed |=
                        rehydrate_projected_value(value, Arc::clone(&projection_resolver)).await?;
                }
                if changed {
                    *value = FlowValue::List(restored.into());
                }
                Ok(changed)
            }
            FlowValue::Record(record) => {
                let mut changed = false;
                let record = Arc::make_mut(record);
                let keys = record.keys().map(str::to_string).collect::<Vec<_>>();
                for key in keys {
                    if let Some(value) = record.get_mut(&key) {
                        changed |=
                            rehydrate_projected_value(value, Arc::clone(&projection_resolver))
                                .await?;
                    }
                }
                Ok(changed)
            }
            FlowValue::Null
            | FlowValue::Bool(_)
            | FlowValue::Number(_)
            | FlowValue::String(_)
            | FlowValue::Image(_) => Ok(false),
        }
    })
}

struct HistoryProjectedValue {
    projection: Arc<RlmHistoryProjection>,
}

impl ProjectedHostValue for HistoryProjectedValue {
    fn type_name(&self) -> &str {
        "list"
    }

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            match request {
                ProjectedReadRequest::Len => ProjectedReadResponse::Len(self.projection.len()),
                ProjectedReadRequest::Index(index) => {
                    let Ok(Some(index)) = projected_index(&index, self.projection.len()) else {
                        return ProjectedReadResponse::Missing;
                    };
                    self.projection
                        .item(index)
                        .and_then(|item| serde_json::to_value(item).ok())
                        .map(json_to_flow_value)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Render => ProjectedReadResponse::Text(
                    serde_json::to_string(self.projection.history())
                        .unwrap_or_else(|_| "[]".to_string()),
                ),
                ProjectedReadRequest::Materialize => {
                    ProjectedReadResponse::Value(json_to_flow_value(self.projection.value()))
                }
                _ => ProjectedReadResponse::Missing,
            }
        })
    }
}

pub(super) fn projected_index(index: &FlowValue, len: usize) -> Result<Option<usize>, ()> {
    let FlowValue::Number(index) = index else {
        return Err(());
    };
    if !index.is_finite() || index.fract() != 0.0 {
        return Err(());
    }
    let len = len as isize;
    let index = *index as isize;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Ok(None);
    }
    Ok(Some(normalized as usize))
}

pub(super) fn prune_reserved_projected_bindings(rlm: &mut FlowState) {
    prune_protected_bindings(rlm, &std::collections::BTreeSet::new());
}

pub(super) fn prune_protected_bindings(
    rlm: &mut FlowState,
    protected_names: &std::collections::BTreeSet<String>,
) {
    prune_projected_binding_names(
        rlm,
        std::iter::once("history").chain(protected_names.iter().map(String::as_str)),
    );
}

pub(super) fn prune_projected_binding_names<'a>(
    rlm: &mut FlowState,
    names: impl IntoIterator<Item = &'a str>,
) {
    let mut snapshot = rlm.snapshot();
    for key in names {
        snapshot.globals.remove(key);
    }
    *rlm = FlowState::from_snapshot(snapshot);
}
