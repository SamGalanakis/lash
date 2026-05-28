use std::collections::BTreeMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::model::{ProcessId, ProcessScope, ProcessScopeId};
use super::validation::process_event_payload_hash;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventType {
    pub name: String,
    pub payload_schema: crate::LashSchema,
    pub semantics: ProcessEventSemanticsSpec,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventSemanticsSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<ProcessTerminalSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake: Option<ProcessWakeSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessTerminalSpec {
    pub state: ProcessTerminalState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub await_output: Option<ProcessValueSelector>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessWakeSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<ProcessValueSelector>,
    pub input: ProcessValueSelector,
    #[serde(default)]
    pub dedupe_key: ProcessWakeDedupeKey,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessWakeDedupeKey {
    #[default]
    EventIdentity,
    Selector(ProcessValueSelector),
    Const(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessValueSelector {
    Payload,
    Pointer(String),
    Const(serde_json::Value),
    Template {
        template: String,
        #[serde(default)]
        fields: BTreeMap<String, ProcessValueSelector>,
    },
    Present(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventSemantics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<ProcessTerminalSemantics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake: Option<ProcessWake>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessTerminalState {
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessTerminalSemantics {
    pub state: ProcessTerminalState,
    pub await_output: ProcessAwaitOutput,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessAwaitOutput {
    Success {
        value: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<crate::ToolControl>,
    },
    Failure {
        class: crate::ToolFailureClass,
        code: String,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<crate::ToolControl>,
    },
    Cancelled {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<crate::ToolControl>,
    },
}

impl ProcessAwaitOutput {
    pub fn terminal_state(&self) -> ProcessTerminalState {
        match self {
            Self::Success { .. } => ProcessTerminalState::Completed,
            Self::Failure { .. } => ProcessTerminalState::Failed,
            Self::Cancelled { .. } => ProcessTerminalState::Cancelled,
        }
    }

    pub fn from_tool_output(output: crate::ToolCallOutput) -> Self {
        let control = output.control;
        match output.outcome {
            crate::ToolCallOutcome::Success(value) => Self::Success {
                value: value.to_json_value(),
                control,
            },
            crate::ToolCallOutcome::Failure(failure) => Self::Failure {
                class: failure.class,
                code: failure.code,
                message: failure.message,
                raw: failure.raw.map(|value| value.to_json_value()),
                control,
            },
            crate::ToolCallOutcome::Cancelled(cancellation) => Self::Cancelled {
                message: cancellation.message,
                raw: cancellation.raw.map(|value| value.to_json_value()),
                control,
            },
        }
    }

    pub fn into_tool_output(self) -> crate::ToolCallOutput {
        match self {
            Self::Success { value, control } => {
                let mut output = crate::ToolCallOutput::success(value);
                output.control = control;
                output
            }
            Self::Failure {
                class,
                code,
                message,
                raw,
                control,
            } => {
                let mut failure = crate::ToolFailure::tool(class, code, message);
                failure.raw = raw.map(crate::ToolValue::from);
                let mut output = crate::ToolCallOutput::failure(failure);
                output.control = control;
                output
            }
            Self::Cancelled {
                message,
                raw,
                control,
            } => {
                let mut cancellation = crate::ToolCancellation::runtime(message);
                cancellation.raw = raw.map(crate::ToolValue::from);
                let mut output = crate::ToolCallOutput::cancelled(cancellation);
                output.control = control;
                output
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessWake {
    pub input: String,
    pub dedupe_key: String,
}

pub fn lashlang_process_event_types() -> Vec<ProcessEventType> {
    vec![
        ProcessEventType {
            name: "process.yield".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: ProcessEventSemanticsSpec::default(),
        },
        ProcessEventType {
            name: "process.wake".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: ProcessEventSemanticsSpec {
                wake: Some(ProcessWakeSpec {
                    when: None,
                    input: ProcessValueSelector::Pointer("/text".to_string()),
                    dedupe_key: ProcessWakeDedupeKey::EventIdentity,
                }),
                ..ProcessEventSemanticsSpec::default()
            },
        },
        ProcessEventType {
            name: "process.signal".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: ProcessEventSemanticsSpec::default(),
        },
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessEvent {
    pub process_id: ProcessId,
    pub sequence: u64,
    pub event_type: String,
    pub payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    pub semantics: ProcessEventSemantics,
    pub occurred_at: SystemTime,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventAppendRequest {
    pub event_type: String,
    pub payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_target_scope: Option<ProcessScope>,
}

impl ProcessEventAppendRequest {
    pub fn new(event_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            event_type: event_type.into(),
            payload,
            idempotency_key: None,
            wake_target_scope: None,
        }
    }

    pub fn with_idempotency_key(mut self, idempotency_key: impl Into<String>) -> Self {
        self.idempotency_key = Some(idempotency_key.into());
        self
    }

    pub fn with_optional_idempotency_key(mut self, idempotency_key: Option<String>) -> Self {
        self.idempotency_key = idempotency_key;
        self
    }

    pub fn with_wake_target_scope(mut self, scope: ProcessScope) -> Self {
        self.wake_target_scope = Some(scope);
        self
    }

    pub fn with_optional_wake_target_scope(mut self, scope: Option<ProcessScope>) -> Self {
        self.wake_target_scope = scope;
        self
    }

    pub fn cancel_requested(process_id: &str, reason: Option<String>) -> Self {
        let payload = serde_json::json!({
            "reason": reason,
        });
        let idempotency_key = process_event_payload_hash("process.cancel_requested", &payload)
            .unwrap_or_else(|_| format!("process:{process_id}:cancel_requested"));
        Self::new("process.cancel_requested", payload).with_idempotency_key(format!(
            "process:{process_id}:cancel_requested:{idempotency_key}"
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessWakeDelivery {
    pub wake_id: String,
    pub target_session_id: String,
    pub target_scope_id: ProcessScopeId,
    pub process_id: ProcessId,
    pub sequence: u64,
    pub dedupe_key: String,
    pub input: String,
    pub created_at_ms: u64,
}

pub(super) fn default_process_event_types() -> Vec<ProcessEventType> {
    vec![
        ProcessEventType {
            name: "process.cancel_requested".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: ProcessEventSemanticsSpec::default(),
        },
        terminal_event_type("process.completed", ProcessTerminalState::Completed),
        terminal_event_type("process.failed", ProcessTerminalState::Failed),
        terminal_event_type("process.cancelled", ProcessTerminalState::Cancelled),
    ]
}

fn terminal_event_type(name: &str, state: ProcessTerminalState) -> ProcessEventType {
    ProcessEventType {
        name: name.to_string(),
        payload_schema: crate::LashSchema::any(),
        semantics: ProcessEventSemanticsSpec {
            terminal: Some(ProcessTerminalSpec {
                state,
                await_output: Some(ProcessValueSelector::Pointer("/await_output".to_string())),
            }),
            ..ProcessEventSemanticsSpec::default()
        },
    }
}
