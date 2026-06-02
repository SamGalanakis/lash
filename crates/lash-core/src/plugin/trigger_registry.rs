use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta,
    SessionPlugin, SnapshotReader, SnapshotWriter,
};

pub(crate) const SESSION_TRIGGER_PLUGIN_ID: &str = "lash.session_triggers";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TriggerSourceType(String);

impl TriggerSourceType {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for TriggerSourceType {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for TriggerSourceType {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl AsRef<str> for TriggerSourceType {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for TriggerSourceType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerRegistration {
    pub handle: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source_type: TriggerSourceType,
    pub source: serde_json::Value,
    pub target: TriggerTargetSummary,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriggerTargetSummary {
    pub process_name: String,
    pub input_name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SessionTriggerRoute {
    pub handle: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source_type: String,
    pub source: serde_json::Value,
    pub event_ty: lashlang::TypeExpr,
    pub module_ref: lashlang::ModuleRef,
    pub required_surface_ref: lashlang::RequiredSurfaceRef,
    pub process_ref: lashlang::ProcessRef,
    pub process_name: String,
    pub input_name: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl From<&SessionTriggerRoute> for TriggerRegistration {
    fn from(route: &SessionTriggerRoute) -> Self {
        Self {
            handle: route.handle.clone(),
            name: route.name.clone(),
            source_type: TriggerSourceType::new(route.source_type.clone()),
            source: route.source.clone(),
            target: TriggerTargetSummary {
                process_name: route.process_name.clone(),
                input_name: route.input_name.clone(),
            },
            enabled: route.enabled,
        }
    }
}

fn default_enabled() -> bool {
    true
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct SessionTriggerRegistryState {
    #[serde(default)]
    revision: u64,
    #[serde(default)]
    next_id: u64,
    #[serde(default)]
    routes: BTreeMap<String, SessionTriggerRoute>,
}

#[derive(Default)]
pub(crate) struct SessionTriggerRegistry {
    state: Mutex<SessionTriggerRegistryState>,
}

impl SessionTriggerRegistry {
    pub(crate) fn register_route(
        &self,
        request: serde_json::Value,
        resources: &lashlang::ResourceCatalog,
        artifact_store: &dyn lashlang::LashlangArtifactStore,
    ) -> Result<SessionTriggerRoute, PluginError> {
        let request = lashlang::TriggerRegistrationRequest::decode(&request)
            .map_err(|err| PluginError::Session(err.to_string()))?;
        let source_type = request.source.source_type.clone();
        let source = request.source.to_host_value();
        let event_ty = lashlang::event_type_for_source(resources, &source_type)
            .map_err(|err| PluginError::Session(err.to_string()))?;
        let target = request.target;
        let input_name = validate_target_process(&target, &event_ty, artifact_store)?;

        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        state.next_id = state.next_id.saturating_add(1);
        let handle = format!("trigger:{}", state.next_id);
        let route = SessionTriggerRoute {
            handle: handle.clone(),
            name: request.name,
            source_type,
            source,
            event_ty,
            module_ref: target.module_ref,
            required_surface_ref: target.required_surface_ref,
            process_ref: target.process_ref,
            process_name: target.process_name,
            input_name,
            enabled: true,
        };
        state.routes.insert(handle, route.clone());
        state.revision = state.revision.saturating_add(1);
        Ok(route)
    }

    pub(crate) fn list(
        &self,
        request: serde_json::Value,
    ) -> Result<Vec<TriggerRegistration>, PluginError> {
        let request = lashlang::TriggerListRequest::decode(&request)
            .map_err(|err| PluginError::Session(err.to_string()))?;
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        Ok(state
            .routes
            .values()
            .filter(|route| {
                request.target.matches(
                    &route.module_ref,
                    &route.required_surface_ref,
                    &route.process_ref,
                    &route.process_name,
                )
            })
            .map(TriggerRegistration::from)
            .collect())
    }

    pub(crate) fn list_all(&self) -> Result<Vec<TriggerRegistration>, PluginError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        Ok(state
            .routes
            .values()
            .map(TriggerRegistration::from)
            .collect())
    }

    pub(crate) fn routes_by_source_type(
        &self,
        source_type: &TriggerSourceType,
    ) -> Result<Vec<TriggerRegistration>, PluginError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        Ok(state
            .routes
            .values()
            .filter(|route| route.source_type == source_type.as_str())
            .map(TriggerRegistration::from)
            .collect())
    }

    pub(crate) fn activation_routes_by_source_type(
        &self,
        source_type: &str,
    ) -> Result<Vec<SessionTriggerRoute>, PluginError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        Ok(state
            .routes
            .values()
            .filter(|route| route.source_type == source_type)
            .cloned()
            .collect())
    }

    pub(crate) fn cancel(&self, request: serde_json::Value) -> Result<bool, PluginError> {
        let request = lashlang::TriggerCancelRequest::decode(&request)
            .map_err(|err| PluginError::Session(err.to_string()))?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        let Some(route) = state.routes.get_mut(&request.handle) else {
            return Ok(false);
        };
        let changed = route.enabled;
        route.enabled = false;
        if changed {
            state.revision = state.revision.saturating_add(1);
        }
        Ok(changed)
    }

    pub(crate) fn route(&self, handle: &str) -> Result<Option<SessionTriggerRoute>, PluginError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        Ok(state.routes.get(handle).cloned())
    }

    fn snapshot_state(&self) -> Result<SessionTriggerRegistryState, PluginError> {
        self.state
            .lock()
            .map(|state| state.clone())
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))
    }

    fn restore_state(&self, state: SessionTriggerRegistryState) -> Result<(), PluginError> {
        let mut current = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        *current = state;
        Ok(())
    }

    fn revision(&self) -> u64 {
        self.state
            .lock()
            .map(|state| state.revision)
            .unwrap_or_default()
    }
}

pub(crate) struct SessionTriggerPluginFactory;

impl PluginFactory for SessionTriggerPluginFactory {
    fn id(&self) -> &'static str {
        SESSION_TRIGGER_PLUGIN_ID
    }

    fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
        let mut resources = lashlang::ResourceCatalog::new();
        lashlang::add_trigger_resource_operations(&mut resources);
        resources
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(SessionTriggerPlugin {
            registry: Arc::new(SessionTriggerRegistry::default()),
        }))
    }
}

struct SessionTriggerPlugin {
    registry: Arc<SessionTriggerRegistry>,
}

impl SessionPlugin for SessionTriggerPlugin {
    fn id(&self) -> &'static str {
        SESSION_TRIGGER_PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.triggers().registry(Arc::clone(&self.registry))
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            revision: self.snapshot_revision(),
            state: Some(
                serde_json::to_value(self.registry.snapshot_state()?).map_err(|err| {
                    PluginError::Session(format!(
                        "failed to encode trigger registry snapshot: {err}"
                    ))
                })?,
            ),
        })
    }

    fn snapshot_revision(&self) -> u64 {
        self.registry.revision()
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let Some(value) = meta.state.clone() else {
            return self
                .registry
                .restore_state(SessionTriggerRegistryState::default());
        };
        let state: SessionTriggerRegistryState = serde_json::from_value(value).map_err(|err| {
            PluginError::Session(format!("failed to decode trigger registry snapshot: {err}"))
        })?;
        self.registry.restore_state(state)
    }
}

pub(super) fn trigger_handle_json(handle: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "trigger_handle",
        "id": handle,
    })
}

fn validate_target_process(
    target: &lashlang::TriggerTargetIdentity,
    event_ty: &lashlang::TypeExpr,
    artifact_store: &dyn lashlang::LashlangArtifactStore,
) -> Result<String, PluginError> {
    let artifact = artifact_store
        .get_module_artifact(&target.module_ref)
        .map_err(|err| PluginError::Session(format!("load trigger target artifact: {err}")))?
        .ok_or_else(|| {
            PluginError::Session(format!(
                "missing trigger target artifact `{}`",
                target.module_ref
            ))
        })?;
    let validation = lashlang::validate_trigger_target(target, event_ty, &artifact)
        .map_err(|err| PluginError::Session(err.to_string()))?;
    Ok(validation.input_name)
}
