use crate::support::*;

#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ModeId(String);

impl ModeId {
    pub fn new(mode: impl Into<String>) -> Self {
        Self(mode.into())
    }

    pub fn standard() -> Self {
        Self("standard".to_string())
    }

    #[cfg(feature = "rlm")]
    pub fn rlm() -> Self {
        Self("rlm".to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Semantic mode preset installed on a [`LashCore`].
#[derive(Clone)]
pub struct ModePreset {
    pub(crate) mode_id: ModeId,
    pub(crate) kind: ModePresetKind,
}

#[derive(Clone)]
pub(crate) enum ModePresetKind {
    Factory(Arc<dyn PluginFactory>),
    #[cfg(feature = "rlm")]
    Rlm {
        config: lash_protocol_rlm::RlmProtocolPluginConfig,
        projection_resolver: Arc<dyn lash_protocol_rlm::ProjectionResolver>,
    },
}

impl ModePreset {
    pub fn standard() -> Self {
        Self {
            mode_id: ModeId::standard(),
            kind: ModePresetKind::Factory(Arc::new(
                lash_protocol_standard::StandardProtocolPluginFactory::new(),
            )),
        }
    }

    #[cfg(feature = "rlm")]
    pub fn rlm() -> Self {
        Self::rlm_with_config(lash_protocol_rlm::RlmProtocolPluginConfig::default())
    }

    #[cfg(feature = "rlm")]
    pub fn rlm_with_config(config: lash_protocol_rlm::RlmProtocolPluginConfig) -> Self {
        Self {
            mode_id: ModeId::rlm(),
            kind: ModePresetKind::Rlm {
                config: rlm_preset_config(config),
                projection_resolver: Arc::new(lash_protocol_rlm::ProjectionRegistry::default()),
            },
        }
    }

    #[cfg(feature = "rlm")]
    pub fn rlm_with_projection_resolver(
        projection_resolver: Arc<dyn lash_protocol_rlm::ProjectionResolver>,
    ) -> Self {
        Self::rlm_with_config_and_projection_resolver(
            lash_protocol_rlm::RlmProtocolPluginConfig::default(),
            projection_resolver,
        )
    }

    #[cfg(feature = "rlm")]
    pub fn rlm_with_config_and_projection_resolver(
        config: lash_protocol_rlm::RlmProtocolPluginConfig,
        projection_resolver: Arc<dyn lash_protocol_rlm::ProjectionResolver>,
    ) -> Self {
        Self {
            mode_id: ModeId::rlm(),
            kind: ModePresetKind::Rlm {
                config: rlm_preset_config(config),
                projection_resolver,
            },
        }
    }

    pub fn mode_id(&self) -> &ModeId {
        &self.mode_id
    }

    #[cfg(feature = "rlm")]
    pub(crate) fn needs_lashlang_runtime(&self) -> bool {
        matches!(self.kind, ModePresetKind::Rlm { .. })
    }

    pub(crate) fn factory(
        &self,
        #[cfg(feature = "rlm")] lashlang_artifact_store: Option<
            Arc<dyn lash_lashlang_runtime::LashlangArtifactStore>,
        >,
        #[cfg(not(feature = "rlm"))] _lashlang_artifact_store: Option<()>,
        #[cfg(feature = "rlm")] lashlang_execution_sink: Option<Arc<dyn lash_trace::TraceSink>>,
        #[cfg(not(feature = "rlm"))] _lashlang_execution_sink: Option<()>,
        #[cfg(feature = "rlm")] trace_context: lash_trace::TraceContext,
        #[cfg(not(feature = "rlm"))] _trace_context: (),
        process_lifecycle: bool,
    ) -> Result<Arc<dyn PluginFactory>> {
        #[cfg(not(feature = "rlm"))]
        let _ = process_lifecycle;
        match &self.kind {
            ModePresetKind::Factory(factory) => Ok(Arc::clone(factory)),
            #[cfg(feature = "rlm")]
            ModePresetKind::Rlm {
                config,
                projection_resolver,
            } => {
                let mut config = config.clone();
                if process_lifecycle {
                    config.lashlang_abilities = config
                        .lashlang_abilities
                        .with_sleep()
                        .with_processes()
                        .with_process_signals();
                }
                let artifact_store =
                    lashlang_artifact_store.ok_or(EmbedError::MissingLashlangArtifactStore)?;
                Ok(Arc::new(
                    lash_protocol_rlm::RlmProtocolPluginFactory::new(config)
                        .with_projection_resolver(Arc::clone(projection_resolver))
                        .with_lashlang_artifact_store(artifact_store)
                        .with_lashlang_execution_trace(lashlang_execution_sink, trace_context),
                ))
            }
        }
    }

    #[cfg(feature = "rlm")]
    pub(crate) fn lashlang_surface(
        &self,
        process_lifecycle: bool,
    ) -> Option<lash_lashlang_runtime::LashlangSurface> {
        let ModePresetKind::Rlm { config, .. } = &self.kind else {
            return None;
        };
        let mut surface = lash_lashlang_runtime::LashlangSurface::new(
            config.lashlang_abilities,
            config.lashlang_language_features,
            lash_lashlang_runtime::LashlangHostCatalog::new(),
        );
        if process_lifecycle {
            surface = surface.for_process_registry(true);
        }
        Some(surface)
    }
}

#[cfg(feature = "rlm")]
pub trait RlmTurnBuilderExt: Sized {
    fn require_submit(self) -> Result<Self>;
    fn require_submit_schema(self, schema: serde_json::Value) -> Result<Self>;
    fn allow_prose_or_submit(self) -> Result<Self>;
}

#[cfg(feature = "rlm")]
impl RlmTurnBuilderExt for TurnBuilder {
    fn require_submit(self) -> Result<Self> {
        rlm_termination(
            self,
            lash_rlm_types::RlmTermination::SubmitRequired { schema: None },
        )
    }

    fn require_submit_schema(self, schema: serde_json::Value) -> Result<Self> {
        rlm_termination(
            self,
            lash_rlm_types::RlmTermination::SubmitRequired {
                schema: Some(schema),
            },
        )
    }

    fn allow_prose_or_submit(self) -> Result<Self> {
        rlm_termination(self, lash_rlm_types::RlmTermination::ProseOrSubmit)
    }
}

#[cfg(feature = "rlm")]
pub trait RlmSessionBuilderExt: Sized {
    fn final_answer_format(self, format: lash_rlm_types::RlmFinalAnswerFormat) -> Self;
}

#[cfg(feature = "rlm")]
impl RlmSessionBuilderExt for SessionBuilder {
    fn final_answer_format(mut self, format: lash_rlm_types::RlmFinalAnswerFormat) -> Self {
        self.rlm_final_answer_format = Some(format);
        self
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

#[cfg(feature = "rlm")]
fn rlm_preset_config(
    config: lash_protocol_rlm::RlmProtocolPluginConfig,
) -> lash_protocol_rlm::RlmProtocolPluginConfig {
    let language_features = config.lashlang_language_features.with_label_annotations();
    config.with_lashlang_language_features(language_features)
}
