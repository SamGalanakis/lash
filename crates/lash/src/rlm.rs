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

/// Sugar for setting the RLM final-answer format on a plain [`SessionBuilder`].
///
/// This writes `RlmCreateExtras` into the builder's generic open-time plugin
/// options bag (the same seam every plugin uses), merging with any RLM options
/// already present so it never clobbers a termination the host set through the
/// same key.
#[cfg(feature = "rlm")]
pub trait RlmSessionBuilderExt: Sized {
    fn final_answer_format(self, format: lash_rlm_types::RlmFinalAnswerFormat) -> Self;
}

#[cfg(feature = "rlm")]
impl RlmSessionBuilderExt for SessionBuilder {
    fn final_answer_format(mut self, format: lash_rlm_types::RlmFinalAnswerFormat) -> Self {
        let mut extras = self
            .plugin_options
            .decode::<lash_rlm_types::RlmCreateExtras>(lash_protocol_rlm::RLM_PROTOCOL_PLUGIN_ID)
            .ok()
            .flatten()
            .unwrap_or_default();
        extras.final_answer_format = Some(format);
        // The value round-trips through the same typed encoder used elsewhere;
        // `RlmCreateExtras` always serializes, so this cannot fail.
        self = self
            .plugin_option(lash_protocol_rlm::RLM_PROTOCOL_PLUGIN_ID, extras)
            .expect("encode RLM create extras");
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

/// The Lashlang compile APIs are operations over an
/// [`RlmProtocolPluginFactory`](lash_protocol_rlm::RlmProtocolPluginFactory) and
/// a plugin host; they live in `lash-protocol-rlm` and are re-exported here.
#[cfg(feature = "rlm")]
pub use lash_protocol_rlm::{
    LashlangCompileSurface, LashlangCompileSurfaceRequest, LashlangModuleCompileError,
    LashlangModuleCompileRequest, ModuleCompileOutput,
};

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
