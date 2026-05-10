//! App-facing embedding facade for Lash.
//!
//! `lash-embed` is intentionally a small layer above the lower-level
//! `lash` runtime crates. Host applications own providers, persistence,
//! app state, HTTP protocols, auth, and frontend streaming; this crate
//! owns only the ergonomic core/session/turn API.

pub mod control;
mod core;
mod error;
mod mode;
mod plugin_binding;
mod session;
mod support;
#[cfg(test)]
mod tests;
pub mod turn;

pub use crate::core::{AdvancedLashCoreBuilder, LashCore, LashCoreBuilder};
pub use crate::error::{EmbedError, Result};
pub use crate::mode::{ModeId, ModePreset, ModelSelection};
pub use crate::plugin_binding::PluginBinding;
pub use crate::session::{
    LashSession, ObservableSession, QueueInputBuilder, SessionBuilder, SessionConfigPatch,
};
pub use crate::turn::{
    TurnActivityFanout, TurnBuilder, TurnOutput, TurnResult, TurnStream, message_role, message_text,
};
pub use lash::{TurnActivity, TurnActivityId, TurnEvent, TurnInput};

pub mod prelude {
    pub use crate::{
        AdvancedLashCoreBuilder, EmbedError, LashCore, LashCoreBuilder, LashSession, ModeId,
        ModePreset, ObservableSession, PluginBinding, Result, SessionBuilder, TurnActivity,
        TurnBuilder, TurnEvent, TurnInput, TurnOutput, TurnResult, TurnStream,
    };
    pub use lash::TurnActivitySink;
}

pub mod tools {
    pub use crate::ToolState;
    pub use lash::{
        AckWakeArgs, McpServerConfig, MonitorAckWakeOp, MonitorEmptyArgs, MonitorRegisterSpecsOp,
        MonitorRunState, MonitorSnapshot, MonitorSpec, MonitorStartOp, MonitorStatus,
        MonitorStatusOp, MonitorStopOp, MonitorTakeUpdatesOp, MonitorUpdateBatch,
        RegisterSpecsArgs, StartMonitorArgs, StopMonitorArgs, ToolAvailability, ToolDefinition,
        ToolProvider, ToolResult, ToolResultView, ToolSourceHandle,
    };
}

pub mod persistence {
    pub use lash::{
        AttachmentStore, PersistedSessionState, RuntimePersistence, SessionReadView,
        SessionStoreCreateRequest, SessionStoreFactory, SessionUsageReport,
    };
}

pub mod plugins {
    pub use crate::plugin_binding::PluginBinding;
    pub use lash::{PluginError, PluginFactory, PluginMessage, PluginSpec};
}

pub mod modes {
    pub use crate::mode::{ModeId, ModePreset};
}

pub mod advanced {
    pub use crate::AdvancedLashCoreBuilder;
    pub use lash::{
        AssembledTurn, EventSink, ExecutionMode, ManagedTaskStatus, ModeSessionExtensionHandle,
        ModeTurnOptions, PluginMessage, Residency, RewriteTrigger, RuntimeSessionHost,
        SessionCreateRequest, SessionHandle, SessionTurnHandle, StandardContextApproach,
        TerminationPolicy,
    };
}

pub mod prompt {
    pub use lash::{
        PromptBuiltin, PromptContribution, PromptContributionGate, PromptLayer, PromptSlot,
        PromptTemplate, PromptTemplateEntry, PromptTemplateSection, default_prompt_template,
    };
}

pub mod tracing {
    pub use lash_trace::{TraceContext, TraceLevel, TraceSink};
}

pub type ToolState = lash::ToolState;
