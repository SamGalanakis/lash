use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta,
    SessionPlugin, SnapshotReader, SnapshotWriter,
};
use crate::SessionTriggerInstallReport;

pub(crate) const SESSION_TRIGGER_PLUGIN_ID: &str = "lash.session_triggers";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SessionTriggerRoute {
    pub(crate) name: String,
    pub(crate) matcher: SessionTriggerMatcher,
    pub(crate) module_ref: lashlang::ModuleRef,
    pub(crate) required_surface_ref: lashlang::RequiredSurfaceRef,
    pub(crate) process_ref: lashlang::ProcessRef,
    pub(crate) process_name: String,
    pub(crate) event_binding: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) args: Vec<SessionTriggerArgBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum SessionTriggerMatcher {
    Resource {
        resource_type: String,
        alias: String,
        event: String,
    },
    AnyResource {
        resource_type: String,
        event: String,
        resource_binding: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SessionTriggerArgBinding {
    pub(crate) name: String,
    pub(crate) value: lashlang::TriggerArg,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct SessionTriggerRegistryState {
    #[serde(default)]
    revision: u64,
    #[serde(default)]
    routes: BTreeMap<String, SessionTriggerRoute>,
}

#[derive(Default)]
pub(crate) struct SessionTriggerRegistry {
    state: Mutex<SessionTriggerRegistryState>,
}

impl SessionTriggerRegistry {
    pub(crate) fn install_lashlang_source(
        &self,
        source: &str,
        surface: lashlang::LashlangSurface,
        artifact_store: &dyn lashlang::LashlangArtifactStore,
    ) -> Result<SessionTriggerInstallReport, PluginError> {
        let program = lashlang::parse(source).map_err(|err| {
            PluginError::Registration(format!(
                "parse trigger lashlang module: {}",
                lashlang::format_parse_diagnostic(source, &err)
            ))
        })?;
        validate_trigger_module_shape(&program)?;
        let linked = lashlang::LinkedModule::link(program, surface).map_err(|err| {
            PluginError::Registration(format!(
                "link trigger lashlang module: {}",
                lashlang::format_link_diagnostic(source, &err)
            ))
        })?;
        self.install_linked_lashlang_source(source, &linked, artifact_store)
    }

    pub(crate) fn install_linked_lashlang_source(
        &self,
        source: &str,
        linked: &lashlang::LinkedModule,
        artifact_store: &dyn lashlang::LashlangArtifactStore,
    ) -> Result<SessionTriggerInstallReport, PluginError> {
        validate_trigger_module_shape(&linked.artifact.canonical_ir)?;
        artifact_store
            .put_module_artifact(&linked.artifact)
            .map_err(|err| {
                PluginError::Registration(format!("store trigger module artifact: {err}"))
            })?;
        let source_sha256 = source_sha256(source);
        let routes = linked
            .artifact
            .canonical_ir
            .declarations
            .iter()
            .filter_map(|declaration| match declaration {
                lashlang::Declaration::Trigger(trigger) => Some(trigger),
                _ => None,
            })
            .map(|trigger| route_from_trigger(trigger, linked, source_sha256.clone()))
            .collect::<Result<Vec<_>, _>>()?;

        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        let mut report = SessionTriggerInstallReport::default();
        let mut changed = false;
        for route in routes {
            let name = route.name.clone();
            match state.routes.get(&name) {
                Some(existing) if existing == &route => report.unchanged.push(name),
                Some(_) => {
                    state.routes.insert(name.clone(), route);
                    report.replaced.push(name);
                    changed = true;
                }
                None => {
                    state.routes.insert(name.clone(), route);
                    report.installed.push(name);
                    changed = true;
                }
            }
        }
        if changed {
            state.revision = state.revision.saturating_add(1);
        }
        Ok(report)
    }

    pub(crate) fn installed_routes(&self) -> Result<Vec<SessionTriggerRoute>, PluginError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        Ok(state.routes.values().cloned().collect())
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
        let state = serde_json::from_value(value).map_err(|err| {
            PluginError::Session(format!("failed to decode trigger registry snapshot: {err}"))
        })?;
        self.registry.restore_state(state)
    }
}

fn validate_trigger_module_shape(program: &lashlang::Program) -> Result<(), PluginError> {
    let mut trigger_count = 0usize;
    for declaration in &program.declarations {
        match declaration {
            lashlang::Declaration::Type(_) | lashlang::Declaration::Process(_) => {}
            lashlang::Declaration::Trigger(_) => trigger_count += 1,
            lashlang::Declaration::Schedule(_) => {
                return Err(PluginError::Registration(
                    "trigger modules cannot declare schedules".to_string(),
                ));
            }
        }
    }
    if trigger_count == 0 {
        return Err(PluginError::Registration(
            "trigger module must declare at least one trigger".to_string(),
        ));
    }
    Ok(())
}

fn route_from_trigger(
    trigger: &lashlang::TriggerDecl,
    linked: &lashlang::LinkedModule,
    source_sha256: Option<String>,
) -> Result<SessionTriggerRoute, PluginError> {
    let process_ref = linked
        .artifact
        .process_ref(trigger.process_name.as_str())
        .cloned()
        .ok_or_else(|| {
            PluginError::Registration(format!(
                "trigger `{}` target process `{}` is not exported",
                trigger.name, trigger.process_name
            ))
        })?;
    Ok(SessionTriggerRoute {
        name: trigger.name.to_string(),
        matcher: matcher_from_trigger_source(&trigger.source),
        module_ref: linked.module_ref.clone(),
        required_surface_ref: linked.required_surface_ref.clone(),
        process_ref,
        process_name: trigger.process_name.to_string(),
        event_binding: trigger.event_binding.to_string(),
        args: trigger
            .args
            .iter()
            .map(|(name, value)| SessionTriggerArgBinding {
                name: name.to_string(),
                value: value.clone(),
            })
            .collect(),
        source_sha256,
    })
}

fn matcher_from_trigger_source(source: &lashlang::TriggerSource) -> SessionTriggerMatcher {
    match source {
        lashlang::TriggerSource::Binding { resource, event } => SessionTriggerMatcher::Resource {
            resource_type: resource.resource_type.to_string(),
            alias: resource.alias.to_string(),
            event: event.to_string(),
        },
        lashlang::TriggerSource::Each {
            resource_type,
            event,
            resource_binding,
        } => SessionTriggerMatcher::AnyResource {
            resource_type: resource_type.to_string(),
            event: event.to_string(),
            resource_binding: resource_binding.to_string(),
        },
    }
}

fn source_sha256(source: &str) -> Option<String> {
    if source.is_empty() {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    Some(format!("{:x}", hasher.finalize()))
}
