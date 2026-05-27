use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta,
    SessionPlugin, SnapshotReader, SnapshotWriter,
};
use crate::SessionTriggerInstallReport;

pub(crate) const SESSION_TRIGGER_PLUGIN_ID: &str = "lash.session_triggers";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct InstalledSessionTrigger {
    pub(crate) name: String,
    pub(crate) source: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct SessionTriggerRegistryState {
    #[serde(default)]
    revision: u64,
    #[serde(default)]
    triggers: BTreeMap<String, String>,
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
    ) -> Result<SessionTriggerInstallReport, PluginError> {
        let program = lashlang::parse(source).map_err(|err| {
            PluginError::Registration(format!(
                "parse trigger lashlang module: {}",
                lashlang::format_parse_diagnostic(source, &err)
            ))
        })?;
        validate_trigger_module_shape(source, &program)?;
        let linked = lashlang::LinkedModule::link(program, surface).map_err(|err| {
            PluginError::Registration(format!(
                "link trigger lashlang module: {}",
                lashlang::format_link_diagnostic(source, &err)
            ))
        })?;
        let trigger_names = linked
            .artifact
            .canonical_ir
            .declarations
            .iter()
            .filter_map(|declaration| match declaration {
                lashlang::Declaration::Trigger(trigger) => Some(trigger.name.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();

        let source = source.to_string();
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        let mut report = SessionTriggerInstallReport::default();
        let mut changed = false;
        for name in trigger_names {
            match state.triggers.get(&name) {
                Some(existing) if existing == &source => report.unchanged.push(name),
                Some(_) => {
                    state.triggers.insert(name.clone(), source.clone());
                    report.replaced.push(name);
                    changed = true;
                }
                None => {
                    state.triggers.insert(name.clone(), source.clone());
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

    pub(crate) fn installed_triggers(&self) -> Result<Vec<InstalledSessionTrigger>, PluginError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Session("trigger registry lock poisoned".to_string()))?;
        Ok(state
            .triggers
            .iter()
            .map(|(name, source)| InstalledSessionTrigger {
                name: name.clone(),
                source: source.clone(),
            })
            .collect())
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

fn validate_trigger_module_shape(
    source: &str,
    program: &lashlang::Program,
) -> Result<(), PluginError> {
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
    if has_foreground_expressions(program) {
        return Err(PluginError::Registration(format!(
            "trigger module must be declaration-only; remove foreground expressions from:\n{source}"
        )));
    }
    Ok(())
}

fn has_foreground_expressions(program: &lashlang::Program) -> bool {
    match &program.main {
        lashlang::Expr::Block(expressions) => !expressions.is_empty(),
        _ => true,
    }
}
