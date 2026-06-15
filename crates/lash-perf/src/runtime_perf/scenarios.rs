use lash::ModeId;
use lash_standard_plugins::{
    ObservationalMemoryConfig, RollingHistoryConfig, StandardContextApproach,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RuntimePerfScenario {
    Standard,
    Rlm,
    StandardToolCalls,
    StandardAsyncToolCompletion,
    RlmToolCalls,
    RlmProcessHandles,
    RlmLlmQuery,
    RlmGlobals,
    RlmLargeToolCatalog,
    ObservationalMemory,
    ObservationalMemoryMaintenance,
    OpenAiCompatStream,
    StandardShellOutput,
    ToolDiscoverySearch,
    OpenAiResponsesSseParse,
    DirectLlmClient,
    ProcessListStress,
    EmbedStandard,
    EmbedRlm,
    ScopedEffectController,
    StoreReopen,
    SqliteStoreReopen,
    TurnCheckpoint,
    LiveReplayPressure,
    TraceJsonlStandard,
    TraceJsonlExtended,
}

impl RuntimePerfScenario {
    pub(crate) const DEFAULTS: [Self; 26] = [
        Self::Standard,
        Self::Rlm,
        Self::StandardToolCalls,
        Self::StandardAsyncToolCompletion,
        Self::RlmToolCalls,
        Self::RlmProcessHandles,
        Self::RlmLlmQuery,
        Self::RlmGlobals,
        Self::RlmLargeToolCatalog,
        Self::ObservationalMemory,
        Self::ObservationalMemoryMaintenance,
        Self::OpenAiCompatStream,
        Self::StandardShellOutput,
        Self::ToolDiscoverySearch,
        Self::OpenAiResponsesSseParse,
        Self::DirectLlmClient,
        Self::ProcessListStress,
        Self::EmbedStandard,
        Self::EmbedRlm,
        Self::ScopedEffectController,
        Self::StoreReopen,
        Self::SqliteStoreReopen,
        Self::TurnCheckpoint,
        Self::LiveReplayPressure,
        Self::TraceJsonlStandard,
        Self::TraceJsonlExtended,
    ];
    pub(crate) const KNOWN: [Self; 26] = [
        Self::Standard,
        Self::Rlm,
        Self::StandardToolCalls,
        Self::StandardAsyncToolCompletion,
        Self::RlmToolCalls,
        Self::RlmProcessHandles,
        Self::RlmLlmQuery,
        Self::RlmGlobals,
        Self::RlmLargeToolCatalog,
        Self::ObservationalMemory,
        Self::ObservationalMemoryMaintenance,
        Self::OpenAiCompatStream,
        Self::StandardShellOutput,
        Self::ToolDiscoverySearch,
        Self::OpenAiResponsesSseParse,
        Self::DirectLlmClient,
        Self::ProcessListStress,
        Self::EmbedStandard,
        Self::EmbedRlm,
        Self::ScopedEffectController,
        Self::StoreReopen,
        Self::SqliteStoreReopen,
        Self::TurnCheckpoint,
        Self::LiveReplayPressure,
        Self::TraceJsonlStandard,
        Self::TraceJsonlExtended,
    ];

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "standard" => Some(Self::Standard),
            "rlm" => Some(Self::Rlm),
            "standard_tool_calls" => Some(Self::StandardToolCalls),
            "standard_async_tool_completion" => Some(Self::StandardAsyncToolCompletion),
            "rlm_tool_calls" => Some(Self::RlmToolCalls),
            "rlm_process_handles" => Some(Self::RlmProcessHandles),
            "rlm_llm_query" => Some(Self::RlmLlmQuery),
            "rlm_globals" => Some(Self::RlmGlobals),
            "rlm_large_tool_catalog" => Some(Self::RlmLargeToolCatalog),
            "observational_memory" => Some(Self::ObservationalMemory),
            "observational_memory_maintenance" => Some(Self::ObservationalMemoryMaintenance),
            "openai_compat_stream" => Some(Self::OpenAiCompatStream),
            "standard_shell_output" => Some(Self::StandardShellOutput),
            "tool_discovery_search" => Some(Self::ToolDiscoverySearch),
            "openai_responses_sse_parse" => Some(Self::OpenAiResponsesSseParse),
            "direct_llm_client" => Some(Self::DirectLlmClient),
            "process_list_stress" => Some(Self::ProcessListStress),
            "embed_standard" => Some(Self::EmbedStandard),
            "embed_rlm" => Some(Self::EmbedRlm),
            "scoped_effect_controller" => Some(Self::ScopedEffectController),
            "store_reopen" => Some(Self::StoreReopen),
            "sqlite_store_reopen" => Some(Self::SqliteStoreReopen),
            "turn_checkpoint" => Some(Self::TurnCheckpoint),
            "live_replay_pressure" => Some(Self::LiveReplayPressure),
            "trace_jsonl_standard" => Some(Self::TraceJsonlStandard),
            "trace_jsonl_extended" => Some(Self::TraceJsonlExtended),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Rlm => "rlm",
            Self::StandardToolCalls => "standard_tool_calls",
            Self::StandardAsyncToolCompletion => "standard_async_tool_completion",
            Self::RlmToolCalls => "rlm_tool_calls",
            Self::RlmProcessHandles => "rlm_process_handles",
            Self::RlmLlmQuery => "rlm_llm_query",
            Self::RlmGlobals => "rlm_globals",
            Self::RlmLargeToolCatalog => "rlm_large_tool_catalog",
            Self::ObservationalMemory => "observational_memory",
            Self::ObservationalMemoryMaintenance => "observational_memory_maintenance",
            Self::OpenAiCompatStream => "openai_compat_stream",
            Self::StandardShellOutput => "standard_shell_output",
            Self::ToolDiscoverySearch => "tool_discovery_search",
            Self::OpenAiResponsesSseParse => "openai_responses_sse_parse",
            Self::DirectLlmClient => "direct_llm_client",
            Self::ProcessListStress => "process_list_stress",
            Self::EmbedStandard => "embed_standard",
            Self::EmbedRlm => "embed_rlm",
            Self::ScopedEffectController => "scoped_effect_controller",
            Self::StoreReopen => "store_reopen",
            Self::SqliteStoreReopen => "sqlite_store_reopen",
            Self::TurnCheckpoint => "turn_checkpoint",
            Self::LiveReplayPressure => "live_replay_pressure",
            Self::TraceJsonlStandard => "trace_jsonl_standard",
            Self::TraceJsonlExtended => "trace_jsonl_extended",
        }
    }

    pub(crate) fn execution_mode(self) -> ModeId {
        match self {
            Self::Standard
            | Self::StandardToolCalls
            | Self::StandardAsyncToolCompletion
            | Self::ObservationalMemory
            | Self::ObservationalMemoryMaintenance
            | Self::OpenAiCompatStream
            | Self::StandardShellOutput
            | Self::ToolDiscoverySearch
            | Self::OpenAiResponsesSseParse
            | Self::DirectLlmClient
            | Self::ProcessListStress
            | Self::EmbedStandard
            | Self::ScopedEffectController
            | Self::StoreReopen
            | Self::SqliteStoreReopen
            | Self::TurnCheckpoint
            | Self::LiveReplayPressure
            | Self::TraceJsonlStandard => ModeId::standard(),
            Self::Rlm
            | Self::RlmToolCalls
            | Self::RlmProcessHandles
            | Self::RlmLlmQuery
            | Self::RlmGlobals
            | Self::RlmLargeToolCatalog
            | Self::EmbedRlm
            | Self::TraceJsonlExtended => ModeId::rlm(),
        }
    }

    pub(crate) fn standard_context_approach(self) -> Option<StandardContextApproach> {
        match self.execution_mode() {
            mode if mode != ModeId::standard() => None,
            _ => Some(match self {
                Self::ObservationalMemory => StandardContextApproach::ObservationalMemory(
                    ObservationalMemoryConfig::default(),
                ),
                Self::ObservationalMemoryMaintenance => {
                    StandardContextApproach::ObservationalMemory(ObservationalMemoryConfig {
                        observation_message_tokens: 30_000,
                        observation_buffer_tokens: 128,
                        observation_block_after_tokens: 60_000,
                        observation_max_tokens_per_batch: 128,
                        previous_observer_tokens: 256,
                        reflection_observation_tokens: 40_000,
                        reflection_buffer_activation_bps: 5_000,
                        reflection_block_after_tokens: 60_000,
                    })
                }
                _ => StandardContextApproach::RollingHistory(RollingHistoryConfig),
            }),
        }
    }
}
