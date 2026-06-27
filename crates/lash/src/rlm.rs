use crate::support::*;

#[cfg(feature = "rlm")]
pub trait RlmTurnBuilderExt: Sized {
    fn require_finish(self) -> Result<Self>;
    fn require_finish_schema(self, schema: serde_json::Value) -> Result<Self>;
    fn allow_prose_or_finish(self) -> Result<Self>;
}

#[cfg(feature = "rlm")]
impl RlmTurnBuilderExt for TurnBuilder {
    fn require_finish(self) -> Result<Self> {
        rlm_termination(
            self,
            lash_rlm_types::RlmTermination::FinishRequired { schema: None },
        )
    }

    fn require_finish_schema(self, schema: serde_json::Value) -> Result<Self> {
        rlm_termination(
            self,
            lash_rlm_types::RlmTermination::FinishRequired {
                schema: Some(schema),
            },
        )
    }

    fn allow_prose_or_finish(self) -> Result<Self> {
        rlm_termination(self, lash_rlm_types::RlmTermination::Natural)
    }
}

#[cfg(feature = "rlm")]
pub trait RlmSessionBuilderExt: Sized {
    fn final_answer_format(self, format: lash_rlm_types::RlmFinalAnswerFormat) -> Self;
}

#[cfg(feature = "rlm")]
impl RlmSessionBuilderExt for RlmSessionBuilder {
    fn final_answer_format(mut self, format: lash_rlm_types::RlmFinalAnswerFormat) -> Self {
        self.rlm_final_answer_format = Some(format);
        self
    }
}

pub use lash_lashlang_runtime::{
    CataloguePreviewEntry, CataloguePreviewOptions, DEFAULT_CATALOGUE_PREVIEW_CALL_NAME_LIMIT,
    DEFAULT_CATALOGUE_PREVIEW_MODULE_LIMIT, LASHLANG_ENGINE_KIND, LASHLANG_SURFACE_EXTENSION_ID,
    LASHLANG_TOOL_BINDING_KEY, LashlangAbilities, LashlangHostCatalog, LashlangHostEnvironment,
    LashlangLanguageFeatures, LashlangProcessEngine, LashlangProcessInput, LashlangSurface,
    LashlangSurfaceContribution, LashlangToolBinding, RemoteToolGrantLashlangExt,
    ToolDefinitionLashlangExt, ToolManifestLashlangExt, catalogue_preview_contribution,
    catalogue_preview_contribution_for_entries,
    catalogue_preview_contribution_for_entries_with_options,
    catalogue_preview_contribution_for_manifests, catalogue_preview_contribution_with_options,
    catalogue_preview_entries_from_catalog_records, catalogue_preview_entries_from_manifests,
    catalogue_preview_entry_from_catalog_record, catalogue_preview_entry_from_manifest,
    lashlang_process_event_types, lashlang_process_signal_event_types,
};
pub use lash_protocol_rlm::{
    NamedDataType, RlmProtocolPluginConfig, TypeExpr, TypeField, format_type_expr,
};
pub use lash_rlm_types::RlmFinalAnswerFormat;

#[cfg(feature = "rlm")]
pub struct LashlangCompileSurfaceRequest {
    pub session_id: String,
    pub execution_env_spec: ProcessExecutionEnvSpec,
    pub extra_plugin_factories: Vec<Arc<dyn PluginFactory>>,
}

#[cfg(feature = "rlm")]
impl LashlangCompileSurfaceRequest {
    pub fn new(session_id: impl Into<String>, execution_env_spec: ProcessExecutionEnvSpec) -> Self {
        Self {
            session_id: session_id.into(),
            execution_env_spec,
            extra_plugin_factories: Vec::new(),
        }
    }

    pub fn plugin(mut self, plugin: Arc<dyn PluginFactory>) -> Self {
        self.extra_plugin_factories.push(plugin);
        self
    }

    pub fn plugins(mut self, plugins: impl IntoIterator<Item = Arc<dyn PluginFactory>>) -> Self {
        self.extra_plugin_factories.extend(plugins);
        self
    }
}

#[cfg(feature = "rlm")]
pub struct LashlangModuleCompileRequest {
    pub session_id: String,
    pub source: String,
    pub execution_env_spec: ProcessExecutionEnvSpec,
    pub extra_plugin_factories: Vec<Arc<dyn PluginFactory>>,
}

#[cfg(feature = "rlm")]
impl LashlangModuleCompileRequest {
    pub fn new(
        session_id: impl Into<String>,
        source: impl Into<String>,
        execution_env_spec: ProcessExecutionEnvSpec,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            source: source.into(),
            execution_env_spec,
            extra_plugin_factories: Vec::new(),
        }
    }

    pub fn plugin(mut self, plugin: Arc<dyn PluginFactory>) -> Self {
        self.extra_plugin_factories.push(plugin);
        self
    }

    pub fn plugins(mut self, plugins: impl IntoIterator<Item = Arc<dyn PluginFactory>>) -> Self {
        self.extra_plugin_factories.extend(plugins);
        self
    }
}

#[cfg(feature = "rlm")]
pub type LashlangModuleCompileError = lashlang::ModuleCompileError;

#[cfg(feature = "rlm")]
pub type ModuleCompileOutput = lashlang::ModuleCompileOutput;

#[cfg(feature = "rlm")]
pub struct LashlangCompileSurface {
    pub host_environment: lash_lashlang_runtime::LashlangHostEnvironment,
    pub tool_catalog: Arc<lash_core::ToolCatalog>,
    pub surface: lash_lashlang_runtime::LashlangSurface,
}

#[cfg(feature = "rlm")]
pub(crate) fn rlm_protocol_config(
    config: lash_protocol_rlm::RlmProtocolPluginConfig,
    process_lifecycle: bool,
) -> lash_protocol_rlm::RlmProtocolPluginConfig {
    let language_features = config.lashlang_language_features.with_label_annotations();
    let mut config = config.with_lashlang_language_features(language_features);
    if process_lifecycle {
        config.lashlang_abilities = config
            .lashlang_abilities
            .with_sleep()
            .with_processes()
            .with_process_signals();
    }
    config
}

#[cfg(feature = "rlm")]
pub(crate) fn rlm_lashlang_surface(
    config: &lash_protocol_rlm::RlmProtocolPluginConfig,
    process_lifecycle: bool,
) -> lash_lashlang_runtime::LashlangSurface {
    let surface = lash_lashlang_runtime::LashlangSurface::new(
        config.lashlang_abilities,
        config.lashlang_language_features,
        lash_lashlang_runtime::LashlangHostCatalog::new(),
    );
    if process_lifecycle {
        surface.for_process_registry(true)
    } else {
        surface
    }
}

#[cfg(feature = "rlm")]
fn rlm_termination(
    mut builder: TurnBuilder,
    termination: lash_rlm_types::RlmTermination,
) -> Result<TurnBuilder> {
    let override_options = ProtocolTurnOptions::typed(lash_rlm_types::RlmCreateExtras {
        termination,
        final_answer_format: None,
    })?;
    let options = builder
        .protocol_turn_options
        .as_ref()
        .map(|current| current.merged_with_override(&override_options))
        .unwrap_or(override_options);
    builder.protocol_turn_options = Some(options);
    Ok(builder)
}
