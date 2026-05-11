use lash_core::{
    ExecutionMode, ObservationalMemoryConfig, RollingHistoryConfig, StandardContextApproach,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RuntimePerfScenario {
    Standard,
    Rlm,
    RlmToolCalls,
    RlmGlobals,
    ObservationalMemory,
    OpenAiCompatStream,
    EmbedStandard,
    EmbedRlm,
}

impl RuntimePerfScenario {
    pub(crate) const DEFAULTS: [Self; 8] = [
        Self::Standard,
        Self::Rlm,
        Self::RlmToolCalls,
        Self::RlmGlobals,
        Self::ObservationalMemory,
        Self::OpenAiCompatStream,
        Self::EmbedStandard,
        Self::EmbedRlm,
    ];
    pub(crate) const KNOWN: [Self; 8] = [
        Self::Standard,
        Self::Rlm,
        Self::RlmToolCalls,
        Self::RlmGlobals,
        Self::ObservationalMemory,
        Self::OpenAiCompatStream,
        Self::EmbedStandard,
        Self::EmbedRlm,
    ];

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "standard" => Some(Self::Standard),
            "rlm" => Some(Self::Rlm),
            "rlm_tool_calls" => Some(Self::RlmToolCalls),
            "rlm_globals" => Some(Self::RlmGlobals),
            "observational_memory" => Some(Self::ObservationalMemory),
            "openai_compat_stream" => Some(Self::OpenAiCompatStream),
            "embed_standard" => Some(Self::EmbedStandard),
            "embed_rlm" => Some(Self::EmbedRlm),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Rlm => "rlm",
            Self::RlmToolCalls => "rlm_tool_calls",
            Self::RlmGlobals => "rlm_globals",
            Self::ObservationalMemory => "observational_memory",
            Self::OpenAiCompatStream => "openai_compat_stream",
            Self::EmbedStandard => "embed_standard",
            Self::EmbedRlm => "embed_rlm",
        }
    }

    pub(crate) fn execution_mode(self) -> ExecutionMode {
        match self {
            Self::Standard
            | Self::ObservationalMemory
            | Self::OpenAiCompatStream
            | Self::EmbedStandard => ExecutionMode::standard(),
            Self::Rlm | Self::RlmToolCalls | Self::RlmGlobals | Self::EmbedRlm => {
                ExecutionMode::new("rlm")
            }
        }
    }

    pub(crate) fn standard_context_approach(self) -> Option<StandardContextApproach> {
        match self.execution_mode() {
            mode if mode != ExecutionMode::standard() => None,
            _ => Some(match self {
                Self::ObservationalMemory => StandardContextApproach::ObservationalMemory(
                    ObservationalMemoryConfig::default(),
                ),
                _ => StandardContextApproach::RollingHistory(RollingHistoryConfig),
            }),
        }
    }
}
