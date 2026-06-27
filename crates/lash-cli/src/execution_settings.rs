//! Execution-mode, RLM termination, and standard-context-approach settings:
//! parsing, labels, and the observational-memory tuning-flag overrides.

use lash_standard_plugins::{StandardContextApproach, StandardContextApproachKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionMode {
    Standard,
    Rlm,
}

impl ExecutionMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Rlm => "rlm",
        }
    }

    pub(crate) fn is_standard(self) -> bool {
        matches!(self, Self::Standard)
    }

    pub(crate) fn is_rlm(self) -> bool {
        matches!(self, Self::Rlm)
    }
}

pub(crate) fn parse_execution_mode(input: &str) -> Result<ExecutionMode, String> {
    match input.trim().to_ascii_lowercase().as_str() {
        "" => Err("Execution mode cannot be empty.".to_string()),
        "rlm" => Ok(ExecutionMode::Rlm),
        "standard" | "tools" => Ok(ExecutionMode::Standard),
        other => Err(format!(
            "Unknown execution mode `{other}`. Expected `rlm` or `standard`."
        )),
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RlmTerminationMode {
    #[value(name = "natural")]
    Natural,
    #[value(name = "finish-required")]
    FinishRequired,
}

impl RlmTerminationMode {
    pub(crate) fn as_rlm_termination(self) -> lash_rlm_types::RlmTermination {
        match self {
            Self::Natural => lash_rlm_types::RlmTermination::Natural,
            Self::FinishRequired => lash_rlm_types::RlmTermination::FinishRequired { schema: None },
        }
    }
}

pub(crate) fn default_rlm_termination_for_mode(mode: ExecutionMode) -> Option<RlmTerminationMode> {
    mode.is_rlm().then_some(RlmTerminationMode::Natural)
}

pub(crate) fn parse_standard_context_approach(
    input: &str,
) -> Result<StandardContextApproach, String> {
    match input.trim().to_ascii_lowercase().as_str() {
        "" => Err("Context approach cannot be empty.".to_string()),
        "rolling" | "rolling-history" | "rolling_history" => {
            Ok(StandardContextApproach::RollingHistory(Default::default()))
        }
        "om" | "observational" | "observational-memory" | "observational_memory" => Ok(
            StandardContextApproach::ObservationalMemory(Default::default()),
        ),
        other => Err(format!(
            "Unknown context approach `{other}`. Expected `rolling_history` or `observational_memory`."
        )),
    }
}

/// The `--om-*` observational-memory tuning flags, gathered once from the CLI
/// arguments so "are any set?" and "apply them" share a single source.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OmOverrides {
    pub(crate) observation_message_tokens: Option<usize>,
    pub(crate) observation_buffer_tokens: Option<usize>,
    pub(crate) observation_block_after_tokens: Option<usize>,
    pub(crate) observation_max_tokens_per_batch: Option<usize>,
    pub(crate) previous_observer_tokens: Option<usize>,
    pub(crate) reflection_observation_tokens: Option<usize>,
    pub(crate) reflection_buffer_activation_percent: Option<u16>,
    pub(crate) reflection_block_after_tokens: Option<usize>,
}

impl OmOverrides {
    pub(crate) fn from_args(args: &crate::Args) -> Self {
        Self {
            observation_message_tokens: args.om_observation_message_tokens,
            observation_buffer_tokens: args.om_observation_buffer_tokens,
            observation_block_after_tokens: args.om_observation_block_after_tokens,
            observation_max_tokens_per_batch: args.om_observation_max_tokens_per_batch,
            previous_observer_tokens: args.om_previous_observer_tokens,
            reflection_observation_tokens: args.om_reflection_observation_tokens,
            reflection_buffer_activation_percent: args.om_reflection_buffer_activation_percent,
            reflection_block_after_tokens: args.om_reflection_block_after_tokens,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.observation_message_tokens.is_none()
            && self.observation_buffer_tokens.is_none()
            && self.observation_block_after_tokens.is_none()
            && self.observation_max_tokens_per_batch.is_none()
            && self.previous_observer_tokens.is_none()
            && self.reflection_observation_tokens.is_none()
            && self.reflection_buffer_activation_percent.is_none()
            && self.reflection_block_after_tokens.is_none()
    }

    pub(crate) fn apply(
        self,
        mut approach: StandardContextApproach,
    ) -> Result<StandardContextApproach, String> {
        if self.is_empty() {
            return Ok(approach);
        }

        let StandardContextApproach::ObservationalMemory(config) = &mut approach else {
            return Err(
                "OM tuning flags require `--context-approach observational_memory`.".to_string(),
            );
        };

        if let Some(value) = self.observation_message_tokens {
            if value == 0 {
                return Err("`--om-observation-message-tokens` must be greater than 0.".to_string());
            }
            config.observation_message_tokens = value;
        }
        if let Some(value) = self.observation_buffer_tokens {
            config.observation_buffer_tokens = value;
        }
        if let Some(value) = self.observation_block_after_tokens {
            if value == 0 {
                return Err(
                    "`--om-observation-block-after-tokens` must be greater than 0.".to_string(),
                );
            }
            config.observation_block_after_tokens = value;
        }
        if let Some(value) = self.observation_max_tokens_per_batch {
            if value == 0 {
                return Err(
                    "`--om-observation-max-tokens-per-batch` must be greater than 0.".to_string(),
                );
            }
            config.observation_max_tokens_per_batch = value;
        }
        if let Some(value) = self.previous_observer_tokens {
            config.previous_observer_tokens = value;
        }
        if let Some(value) = self.reflection_observation_tokens {
            if value == 0 {
                return Err(
                    "`--om-reflection-observation-tokens` must be greater than 0.".to_string(),
                );
            }
            config.reflection_observation_tokens = value;
        }
        if let Some(value) = self.reflection_buffer_activation_percent {
            if value > 100 {
                return Err(
                    "`--om-reflection-buffer-activation-percent` must be between 0 and 100."
                        .to_string(),
                );
            }
            config.reflection_buffer_activation_bps = value.saturating_mul(100);
        }
        if let Some(value) = self.reflection_block_after_tokens {
            if value == 0 {
                return Err(
                    "`--om-reflection-block-after-tokens` must be greater than 0.".to_string(),
                );
            }
            config.reflection_block_after_tokens = value;
        }

        if config.observation_buffer_tokens >= config.observation_block_after_tokens {
            return Err(
                "`--om-observation-buffer-tokens` must be smaller than `--om-observation-block-after-tokens`."
                    .to_string(),
            );
        }
        if config.observation_buffer_interval_tokens() >= config.observation_message_tokens {
            return Err(
                "`--om-observation-buffer-tokens` must be smaller than `--om-observation-message-tokens`."
                    .to_string(),
            );
        }
        if config.reflection_buffer_activation_bps > 10_000 {
            return Err(
                "`--om-reflection-buffer-activation-percent` must be between 0 and 100."
                    .to_string(),
            );
        }

        Ok(approach)
    }
}

pub(crate) fn execution_mode_usage() -> &'static str {
    "<rlm|standard>"
}

pub(crate) fn ensure_supported_execution_mode(
    mode: ExecutionMode,
) -> Result<ExecutionMode, String> {
    Ok(mode)
}

pub(crate) fn execution_mode_label(mode: &ExecutionMode) -> &str {
    mode.as_str()
}

pub(crate) fn standard_context_approach_label(approach: &StandardContextApproach) -> &'static str {
    match approach.kind() {
        StandardContextApproachKind::RollingHistory => "rolling_history",
        StandardContextApproachKind::ObservationalMemory => "observational_memory",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observational_memory_overrides_apply_to_standard_context_approach() {
        let approach = OmOverrides {
            observation_message_tokens: Some(45_000),
            observation_buffer_tokens: Some(8_000),
            observation_max_tokens_per_batch: Some(12_000),
            previous_observer_tokens: Some(3_000),
            reflection_buffer_activation_percent: Some(60),
            ..OmOverrides::default()
        }
        .apply(StandardContextApproach::ObservationalMemory(
            Default::default(),
        ))
        .expect("overrides");
        let StandardContextApproach::ObservationalMemory(config) = approach else {
            panic!("expected observational_memory");
        };
        assert_eq!(config.observation_message_tokens, 45_000);
        assert_eq!(config.observation_buffer_tokens, 8_000);
        assert_eq!(config.observation_max_tokens_per_batch, 12_000);
        assert_eq!(config.previous_observer_tokens, 3_000);
        assert_eq!(config.reflection_buffer_activation_bps, 6_000);
    }

    #[test]
    fn observational_memory_overrides_require_om_standard_context_approach() {
        let err = OmOverrides {
            observation_message_tokens: Some(45_000),
            ..OmOverrides::default()
        }
        .apply(StandardContextApproach::RollingHistory(Default::default()))
        .expect_err("expected validation error");
        assert!(err.contains("observational_memory"));
    }

    #[test]
    fn rlm_termination_mode_maps_to_protocol_termination() {
        assert_eq!(
            RlmTerminationMode::Natural.as_rlm_termination(),
            lash_rlm_types::RlmTermination::Natural
        );
        assert!(matches!(
            RlmTerminationMode::FinishRequired.as_rlm_termination(),
            lash_rlm_types::RlmTermination::FinishRequired { schema: None }
        ));
    }

    #[test]
    fn rlm_termination_defaults_only_for_rlm_mode() {
        assert_eq!(
            default_rlm_termination_for_mode(ExecutionMode::Rlm),
            Some(RlmTerminationMode::Natural)
        );
        assert_eq!(
            default_rlm_termination_for_mode(ExecutionMode::Standard),
            None
        );
    }
}
