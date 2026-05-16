use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardContextApproachKind {
    RollingHistory,
    ObservationalMemory,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StandardContextApproach {
    RollingHistory(RollingHistoryConfig),
    ObservationalMemory(ObservationalMemoryConfig),
}

impl Default for StandardContextApproach {
    fn default() -> Self {
        Self::RollingHistory(RollingHistoryConfig)
    }
}

impl StandardContextApproach {
    pub fn kind(&self) -> StandardContextApproachKind {
        match self {
            Self::RollingHistory(_) => StandardContextApproachKind::RollingHistory,
            Self::ObservationalMemory(_) => StandardContextApproachKind::ObservationalMemory,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollingHistoryConfig;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationalMemoryConfig {
    pub observation_message_tokens: usize,
    pub observation_buffer_tokens: usize,
    pub observation_block_after_tokens: usize,
    pub observation_max_tokens_per_batch: usize,
    pub previous_observer_tokens: usize,
    pub reflection_observation_tokens: usize,
    #[serde(default = "default_reflection_buffer_activation_bps")]
    pub reflection_buffer_activation_bps: u16,
    pub reflection_block_after_tokens: usize,
}

impl Default for ObservationalMemoryConfig {
    fn default() -> Self {
        Self {
            observation_message_tokens: 30_000,
            observation_buffer_tokens: 6_000,
            observation_block_after_tokens: 36_000,
            observation_max_tokens_per_batch: 10_000,
            previous_observer_tokens: 2_000,
            reflection_observation_tokens: 40_000,
            reflection_buffer_activation_bps: default_reflection_buffer_activation_bps(),
            reflection_block_after_tokens: 48_000,
        }
    }
}

impl ObservationalMemoryConfig {
    pub fn observation_buffer_interval_tokens(&self) -> usize {
        self.observation_buffer_tokens
    }

    pub fn observation_retention_tokens(&self) -> usize {
        self.observation_buffer_tokens
    }

    pub fn reflection_buffer_activation_tokens(&self) -> usize {
        self.reflection_observation_tokens
            .saturating_mul(self.reflection_buffer_activation_bps as usize)
            / 10_000
    }
}

const fn default_reflection_buffer_activation_bps() -> u16 {
    5_000
}
