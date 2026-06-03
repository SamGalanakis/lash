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
    pub(crate) factory: Arc<dyn PluginFactory>,
}

impl ModePreset {
    pub fn standard() -> Self {
        Self {
            mode_id: ModeId::standard(),
            factory: Arc::new(lash_protocol_standard::StandardProtocolPluginFactory::new()),
        }
    }

    pub fn rlm() -> Self {
        Self {
            mode_id: ModeId::rlm(),
            factory: Arc::new(lash_protocol_rlm::RlmProtocolPluginFactory::new(
                rlm_preset_config(lash_protocol_rlm::RlmProtocolPluginConfig::default()),
            )),
        }
    }

    pub fn rlm_with_config(config: lash_protocol_rlm::RlmProtocolPluginConfig) -> Self {
        Self {
            mode_id: ModeId::rlm(),
            factory: Arc::new(lash_protocol_rlm::RlmProtocolPluginFactory::new(
                rlm_preset_config(config),
            )),
        }
    }

    pub fn rlm_with_projection_resolver(
        projection_resolver: Arc<dyn lash_protocol_rlm::ProjectionResolver>,
    ) -> Self {
        Self::rlm_with_config_and_projection_resolver(
            lash_protocol_rlm::RlmProtocolPluginConfig::default(),
            projection_resolver,
        )
    }

    pub fn rlm_with_config_and_projection_resolver(
        config: lash_protocol_rlm::RlmProtocolPluginConfig,
        projection_resolver: Arc<dyn lash_protocol_rlm::ProjectionResolver>,
    ) -> Self {
        Self {
            mode_id: ModeId::rlm(),
            factory: Arc::new(
                lash_protocol_rlm::RlmProtocolPluginFactory::new(rlm_preset_config(config))
                    .with_projection_resolver(projection_resolver),
            ),
        }
    }

    pub fn mode_id(&self) -> &ModeId {
        &self.mode_id
    }
}

fn rlm_preset_config(
    config: lash_protocol_rlm::RlmProtocolPluginConfig,
) -> lash_protocol_rlm::RlmProtocolPluginConfig {
    let language_features = config.lashlang_language_features.with_label_annotations();
    config.with_lashlang_language_features(language_features)
}
