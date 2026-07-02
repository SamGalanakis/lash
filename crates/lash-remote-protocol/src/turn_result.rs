//! Turn result envelopes: the turn result itself, outcomes, stops, assistant
//! output, usage/execution summaries, tool-call summaries, issues, and causal
//! references.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ensure_protocol_version;
use crate::llm::{RemoteLlmTerminalReason, RemoteProviderFailureKind};
use crate::registry_errors::{RemoteProtocolError, require_non_empty};
use crate::usage_activity::{RemoteTokenLedgerEntry, RemoteTurnActivity, RemoteUsage};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnResult {
    pub protocol_version: u32,
    pub session_id: String,
    pub turn_id: String,
    pub status: RemoteTurnStatus,
    pub outcome: RemoteTurnOutcome,
    pub assistant_output: RemoteAssistantOutput,
    #[serde(default)]
    pub usage: RemoteTurnUsageSummary,
    #[serde(default)]
    pub execution: RemoteExecutionSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<RemoteToolCallSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<RemoteTurnIssue>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activities: Vec<RemoteTurnActivity>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl RemoteTurnResult {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteTurnResult", "session_id", &self.session_id)?;
        require_non_empty("RemoteTurnResult", "turn_id", &self.turn_id)?;
        for activity in &self.activities {
            if activity.protocol_version != self.protocol_version {
                return Err(RemoteProtocolError::MismatchedNestedProtocolVersion {
                    parent: "RemoteTurnResult",
                    child: "activities",
                    parent_version: self.protocol_version,
                    child_version: activity.protocol_version,
                });
            }
            activity.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteCausalRef {
    Turn {
        session_id: String,
        turn_id: String,
    },
    Effect {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        effect_id: String,
    },
    ToolCall {
        session_id: String,
        call_id: String,
    },
    Process {
        process_id: String,
    },
    ProcessEvent {
        process_id: String,
        sequence: u64,
    },
    TriggerOccurrence {
        occurrence_id: String,
    },
    SessionNode {
        session_id: String,
        node_id: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteTurnStatus {
    #[default]
    Completed,
    Failed,
    Cancelled,
    InProgress,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteTurnOutcome {
    Finished { finish: RemoteTurnFinish },
    AgentFrameSwitch { frame_id: String, task: String },
    Stopped { stop: RemoteTurnStop },
}

impl Default for RemoteTurnOutcome {
    fn default() -> Self {
        Self::Stopped {
            stop: RemoteTurnStop::Incomplete,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteTurnFinish {
    AssistantMessage {
        text: String,
    },
    FinalValue {
        value: serde_json::Value,
    },
    ToolValue {
        tool_name: String,
        value: serde_json::Value,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteTurnStop {
    Cancelled,
    Incomplete,
    InvalidInput,
    MaxTurns,
    ToolFailure,
    ProviderError,
    PluginAbort,
    RuntimeError,
    SubmittedError {
        value: serde_json::Value,
    },
    ToolError {
        tool_name: String,
        value: serde_json::Value,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteAssistantOutput {
    #[serde(default)]
    pub safe_text: String,
    #[serde(default)]
    pub raw_text: String,
    #[serde(default)]
    pub state: RemoteAssistantOutputState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteAssistantOutputState {
    #[default]
    Usable,
    EmptyOutput,
    TracebackOnly,
    RecoveredFromError,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnUsageSummary {
    #[serde(default)]
    pub parent: RemoteUsage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<RemoteTokenLedgerEntry>,
    #[serde(default)]
    pub total: RemoteUsage,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteExecutionSummary {
    #[serde(default)]
    pub had_tool_calls: bool,
    #[serde(default)]
    pub had_code_execution: bool,
    /// Wall-clock turn start (epoch milliseconds), measured from turn claim.
    /// `0` when the producer predates the field.
    #[serde(default)]
    pub started_at_ms: u64,
    /// Whole-turn duration in milliseconds (claim → final commit). `0` when
    /// the producer predates the field.
    #[serde(default)]
    pub duration_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteToolCallSummary {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub tool_name: String,
    #[serde(default)]
    pub args: serde_json::Value,
    pub outcome: RemoteToolCallOutcome,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", content = "payload", rename_all = "snake_case")]
pub enum RemoteToolCallOutcome {
    Success(serde_json::Value),
    Failure(serde_json::Value),
    Cancelled(serde_json::Value),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteTurnIssue {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_reason: Option<RemoteLlmTerminalReason>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    /// Typed retryability signal; `None` when the source did not know.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    /// Typed provider-failure classification, present only for classified
    /// LLM provider/transport failures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_failure_kind: Option<RemoteProviderFailureKind>,
}
