use std::sync::Arc;

use super::{PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin};

pub(crate) const TRIGGER_RESOURCE_PLUGIN_ID: &str = "lash.triggers";

pub(crate) struct TriggerResourcePluginFactory;

impl PluginFactory for TriggerResourcePluginFactory {
    fn id(&self) -> &'static str {
        TRIGGER_RESOURCE_PLUGIN_ID
    }

    fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
        let mut resources = lashlang::ResourceCatalog::new();
        lashlang::add_trigger_resource_operations(&mut resources);
        resources
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(TriggerResourcePlugin))
    }
}

struct TriggerResourcePlugin;

impl SessionPlugin for TriggerResourcePlugin {
    fn id(&self) -> &'static str {
        TRIGGER_RESOURCE_PLUGIN_ID
    }

    fn register(&self, _reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        Ok(())
    }
}

pub(crate) fn trigger_handle_json(handle: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "trigger_handle",
        "id": handle,
    })
}

pub(crate) async fn validate_target_process(
    target: &lashlang::TriggerTargetIdentity,
    event_ty: &lashlang::NamedDataType,
    inputs: &lashlang::TriggerInputTemplate,
    artifact_store: &dyn lashlang::LashlangArtifactStore,
) -> Result<lashlang::TriggerTargetValidation, PluginError> {
    let artifact = artifact_store
        .get_module_artifact(&target.module_ref)
        .await
        .map_err(|err| PluginError::Session(format!("load trigger target artifact: {err}")))?
        .ok_or_else(|| {
            PluginError::Session(format!(
                "missing trigger target artifact `{}`",
                target.module_ref
            ))
        })?;
    let validation = lashlang::validate_trigger_target(target, event_ty, inputs, &artifact)
        .map_err(|err| PluginError::Session(err.to_string()))?;
    Ok(validation)
}
