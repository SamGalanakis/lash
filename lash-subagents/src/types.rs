//! Request, response, and event types exposed by the subagent host API.
//!
//! These are pure data definitions with no behavior beyond parsing a few
//! string-valued enums. Keeping them in a dedicated module keeps
//! `host.rs` focused on trait definitions and the `local.rs`
//! orchestrator focused on behavior.

use serde::Serialize;
use serde_json::Value;

use lash::{SessionCreateRequest, TurnInput};

#[derive(Clone, Debug)]
pub struct SessionAgentInfo {
    pub path: String,
    /// Registered capability name this agent was spawned with. `None` for
    /// the implicit `/root` agent which is created on demand and has no
    /// spawn-time capability binding.
    pub capability: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SpawnAgentRequest {
    pub task_name: String,
    pub task: String,
    /// Name of the capability the spawn was made with. The capability
    /// itself is resolved by the caller via the registry; the host just
    /// records the name for diagnostics, surfacing in the agent tree, and
    /// the tool-surface deny path.
    pub capability: String,
    pub create_request: SessionCreateRequest,
    pub turn_input: TurnInput,
}

#[derive(Clone, Debug)]
pub struct SendMessageRequest {
    pub target: String,
    pub message: String,
    pub delivery: DeliveryMode,
}

#[derive(Clone, Debug)]
pub struct FollowupTaskRequest {
    pub target: String,
    pub task: String,
    pub turn_input: TurnInput,
    pub delivery: DeliveryMode,
}

#[derive(Clone, Debug)]
pub struct WaitAgentRequest {
    pub targets: Vec<String>,
    pub until: WaitUntil,
    pub timeout_ms: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct CloseAgentRequest {
    pub target: String,
}

#[derive(Clone, Debug)]
pub struct ListAgentsRequest {
    pub path_prefix: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpawnAgentResponse {
    pub task_name: String,
    pub target: String,
    pub task_id: String,
    pub session_id: String,
    pub run_state: String,
    pub capability: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    /// Populated when the requested `task_name` was normalized (e.g.
    /// hyphens/spaces converted to underscores, uppercase lowered).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_name_note: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SendMessageResponse {
    pub from: String,
    pub to: String,
    pub message_id: String,
    pub delivery: String,
    pub status: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FollowupTaskResponse {
    pub target: String,
    pub task_id: String,
    pub delivery: String,
    pub status: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentResponse {
    pub timed_out: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<WaitAgentCompletion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<WaitAgentMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closed: Option<WaitAgentClosed>,
    pub events: Vec<WaitAgentEvent>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CloseAgentResponse {
    pub closed: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ListAgentsResponse {
    pub agents: Vec<AgentSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AgentSummary {
    pub target: String,
    pub task_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
    pub agent_state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_task: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_task_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_task_state: Option<String>,
    pub queued_tasks: usize,
    pub queued_messages: usize,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMode {
    NextPossible,
    Interrupt,
    NextTurn,
}

impl DeliveryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NextPossible => "next_possible",
            Self::Interrupt => "interrupt",
            Self::NextTurn => "next_turn",
        }
    }

    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("next_possible") {
            "next_possible" => Ok(Self::NextPossible),
            "interrupt" => Ok(Self::Interrupt),
            "next_turn" => Ok(Self::NextTurn),
            other => Err(format!(
                "invalid delivery: expected `next_possible`, `interrupt`, or `next_turn`, got `{other}`"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitUntil {
    TaskCompleted,
    Terminal,
    Message,
    AnyResult,
    AnyEvent,
}

impl WaitUntil {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("task_completed") {
            "task_completed" => Ok(Self::TaskCompleted),
            "terminal" => Ok(Self::Terminal),
            "message" => Ok(Self::Message),
            "any_result" => Ok(Self::AnyResult),
            "any_event" => Ok(Self::AnyEvent),
            other => Err(format!(
                "invalid until: expected `task_completed`, `terminal`, `message`, `any_result`, or `any_event`, got `{other}`"
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentSessionSummary {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    pub task: String,
    pub iterations: usize,
    pub tool_calls: usize,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    pub token_usage: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentCompletion {
    pub target: String,
    pub task: String,
    pub status: String,
    pub result: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub session: WaitAgentSessionSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentMessage {
    pub from: String,
    pub to: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentClosed {
    pub target: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WaitAgentEvent {
    TaskStarted {
        target: String,
        task: String,
        session_id: String,
        capability: String,
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        model_variant: Option<String>,
    },
    Message {
        from: String,
        to: String,
        message: String,
    },
    TaskCompleted {
        target: String,
        task: String,
        status: String,
        result: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        session: WaitAgentSessionSummary,
    },
    AgentClosed {
        target: String,
    },
}
