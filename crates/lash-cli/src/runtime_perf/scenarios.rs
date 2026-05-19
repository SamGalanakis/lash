use lash_core::{
    ExecutionMode, ObservationalMemoryConfig, RollingHistoryConfig, StandardContextApproach,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RuntimePerfScenario {
    Standard,
    Rlm,
    StandardToolCalls,
    RlmToolCalls,
    RlmAsyncHandles,
    RlmLlmQuery,
    RlmToolRetry,
    RlmGlobals,
    RlmLargeToolSurface,
    ObservationalMemory,
    OpenAiCompatStream,
    EmbedStandard,
    EmbedRlm,
    ScopedEffectController,
    StoreReopen,
    TurnCheckpoint,
}

impl RuntimePerfScenario {
    pub(crate) const DEFAULTS: [Self; 16] = [
        Self::Standard,
        Self::Rlm,
        Self::StandardToolCalls,
        Self::RlmToolCalls,
        Self::RlmAsyncHandles,
        Self::RlmLlmQuery,
        Self::RlmToolRetry,
        Self::RlmGlobals,
        Self::RlmLargeToolSurface,
        Self::ObservationalMemory,
        Self::OpenAiCompatStream,
        Self::EmbedStandard,
        Self::EmbedRlm,
        Self::ScopedEffectController,
        Self::StoreReopen,
        Self::TurnCheckpoint,
    ];
    pub(crate) const KNOWN: [Self; 16] = [
        Self::Standard,
        Self::Rlm,
        Self::StandardToolCalls,
        Self::RlmToolCalls,
        Self::RlmAsyncHandles,
        Self::RlmLlmQuery,
        Self::RlmToolRetry,
        Self::RlmGlobals,
        Self::RlmLargeToolSurface,
        Self::ObservationalMemory,
        Self::OpenAiCompatStream,
        Self::EmbedStandard,
        Self::EmbedRlm,
        Self::ScopedEffectController,
        Self::StoreReopen,
        Self::TurnCheckpoint,
    ];

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "standard" => Some(Self::Standard),
            "rlm" => Some(Self::Rlm),
            "standard_tool_calls" => Some(Self::StandardToolCalls),
            "rlm_tool_calls" => Some(Self::RlmToolCalls),
            "rlm_async_handles" => Some(Self::RlmAsyncHandles),
            "rlm_llm_query" => Some(Self::RlmLlmQuery),
            "rlm_tool_retry" => Some(Self::RlmToolRetry),
            "rlm_globals" => Some(Self::RlmGlobals),
            "rlm_large_tool_surface" => Some(Self::RlmLargeToolSurface),
            "observational_memory" => Some(Self::ObservationalMemory),
            "openai_compat_stream" => Some(Self::OpenAiCompatStream),
            "embed_standard" => Some(Self::EmbedStandard),
            "embed_rlm" => Some(Self::EmbedRlm),
            "scoped_effect_controller" => Some(Self::ScopedEffectController),
            "store_reopen" => Some(Self::StoreReopen),
            "turn_checkpoint" => Some(Self::TurnCheckpoint),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Rlm => "rlm",
            Self::StandardToolCalls => "standard_tool_calls",
            Self::RlmToolCalls => "rlm_tool_calls",
            Self::RlmAsyncHandles => "rlm_async_handles",
            Self::RlmLlmQuery => "rlm_llm_query",
            Self::RlmToolRetry => "rlm_tool_retry",
            Self::RlmGlobals => "rlm_globals",
            Self::RlmLargeToolSurface => "rlm_large_tool_surface",
            Self::ObservationalMemory => "observational_memory",
            Self::OpenAiCompatStream => "openai_compat_stream",
            Self::EmbedStandard => "embed_standard",
            Self::EmbedRlm => "embed_rlm",
            Self::ScopedEffectController => "scoped_effect_controller",
            Self::StoreReopen => "store_reopen",
            Self::TurnCheckpoint => "turn_checkpoint",
        }
    }

    pub(crate) fn execution_mode(self) -> ExecutionMode {
        match self {
            Self::Standard
            | Self::StandardToolCalls
            | Self::ObservationalMemory
            | Self::OpenAiCompatStream
            | Self::EmbedStandard
            | Self::ScopedEffectController
            | Self::StoreReopen
            | Self::TurnCheckpoint => ExecutionMode::standard(),
            Self::Rlm
            | Self::RlmToolCalls
            | Self::RlmAsyncHandles
            | Self::RlmLlmQuery
            | Self::RlmToolRetry
            | Self::RlmGlobals
            | Self::RlmLargeToolSurface
            | Self::EmbedRlm => ExecutionMode::new("rlm"),
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
