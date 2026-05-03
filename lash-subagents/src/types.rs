//! Request, response, and event types exposed by the subagent host API.

use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

use lash::{SessionCreateRequest, TurnInput};

#[derive(Clone, Debug)]
pub struct SpawnAgentRequest {
    pub agent_name: String,
    pub task: String,
    pub capability: String,
    pub hidden_tools: BTreeSet<String>,
    pub create_request: SessionCreateRequest,
    pub turn_input: TurnInput,
}

#[derive(Clone, Debug)]
pub struct SendMessageRequest {
    pub agent_name: String,
    pub message: String,
    pub delivery: DeliveryMode,
}

#[derive(Clone, Debug)]
pub struct FollowupTaskRequest {
    pub agent_name: String,
    pub task: String,
    pub turn_input: TurnInput,
    pub delivery: DeliveryMode,
}

#[derive(Clone, Debug)]
pub struct WaitAgentRequest {
    pub agents: Vec<String>,
    pub until: WaitUntil,
    pub timeout_ms: Option<u64>,
    pub all: bool,
}

#[derive(Clone, Debug)]
pub struct CloseAgentRequest {
    pub agent_name: String,
}

#[derive(Clone, Debug)]
pub struct ListAgentsRequest {}

#[derive(Clone, Debug, Serialize)]
pub struct SpawnAgentResponse {
    pub agent_name: String,
    pub task_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_name_note: Option<String>,
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
    pub agent_name: String,
    pub task_id: String,
    pub delivery: String,
    pub status: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentResponse {
    pub timed_out: bool,
    pub completed: BTreeMap<String, WaitAgentCompletion>,
    pub pending: BTreeMap<String, WaitAgentPending>,
    pub messages: BTreeMap<String, Vec<WaitAgentMessage>>,
    pub closed: BTreeMap<String, WaitAgentClosed>,
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
    pub agent_name: String,
    pub task_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
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
}

impl WaitUntil {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("task_completed") {
            "task_completed" => Ok(Self::TaskCompleted),
            "terminal" => Ok(Self::Terminal),
            "message" => Ok(Self::Message),
            "any_result" => Ok(Self::AnyResult),
            other => Err(format!(
                "invalid until: expected `task_completed`, `terminal`, `message`, or `any_result`, got `{other}`"
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentCompletion {
    pub agent_name: String,
    pub task: String,
    pub status: String,
    pub result: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentPending {
    pub agent_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    pub status: String,
}

#[derive(Clone, Debug)]
pub struct AgentMetadata {
    pub session_id: String,
    pub parent_session_id: Option<String>,
    pub capability: Option<String>,
    pub run_state: String,
    pub model: String,
    pub model_variant: Option<String>,
    pub last_iterations: Option<usize>,
    pub last_tool_calls: Option<usize>,
    pub last_token_usage: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentMessage {
    pub from_agent: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentClosed {
    pub agent_name: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WaitAgentEvent {
    TaskStarted {
        agent_name: String,
        #[serde(skip)]
        parent_session_id: String,
        task: String,
        session_id: String,
    },
    Message {
        from_agent: String,
        to_agent: String,
        #[serde(skip)]
        parent_session_id: String,
        message: String,
    },
    TaskCompleted {
        agent_name: String,
        #[serde(skip)]
        parent_session_id: String,
        task: String,
        status: String,
        result: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    AgentClosed {
        agent_name: String,
        #[serde(skip)]
        parent_session_id: String,
    },
}
