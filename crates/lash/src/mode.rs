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

    pub(crate) fn execution_mode(&self) -> ExecutionMode {
        ExecutionMode::new(self.0.clone())
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
    pub(crate) standard_context_approach: Option<StandardContextApproach>,
}

impl ModePreset {
    pub fn standard() -> Self {
        Self {
            mode_id: ModeId::standard(),
            factory: Arc::new(lash_mode_standard::BuiltinStandardModePluginFactory::new()),
            standard_context_approach: Some(StandardContextApproach::default()),
        }
    }

    pub fn rlm() -> Self {
        Self {
            mode_id: ModeId::rlm(),
            factory: Arc::new(lash_mode_rlm::BuiltinRlmModePluginFactory::default()),
            standard_context_approach: None,
        }
    }

    pub fn rlm_with_config(config: lash_mode_rlm::RlmModePluginConfig) -> Self {
        Self {
            mode_id: ModeId::rlm(),
            factory: Arc::new(lash_mode_rlm::BuiltinRlmModePluginFactory::new(config)),
            standard_context_approach: None,
        }
    }

    pub fn rlm_with_projection_resolver(
        projection_resolver: Arc<dyn lash_mode_rlm::ProjectionResolver>,
    ) -> Self {
        Self::rlm_with_config_and_projection_resolver(
            lash_mode_rlm::RlmModePluginConfig::default(),
            projection_resolver,
        )
    }

    pub fn rlm_with_config_and_projection_resolver(
        config: lash_mode_rlm::RlmModePluginConfig,
        projection_resolver: Arc<dyn lash_mode_rlm::ProjectionResolver>,
    ) -> Self {
        Self {
            mode_id: ModeId::rlm(),
            factory: Arc::new(
                lash_mode_rlm::BuiltinRlmModePluginFactory::new(config)
                    .with_projection_resolver(projection_resolver),
            ),
            standard_context_approach: None,
        }
    }

    pub fn with_standard_context_approach(
        mut self,
        standard_context_approach: Option<StandardContextApproach>,
    ) -> Self {
        self.standard_context_approach = standard_context_approach;
        self
    }

    pub fn mode_id(&self) -> &ModeId {
        &self.mode_id
    }
}
