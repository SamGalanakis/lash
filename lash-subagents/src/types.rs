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

#[derive(Clone, Debug, Serialize)]
pub struct SpawnAgentResponse {
    pub agent_name: String,
    pub task_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_name_note: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentResponse {
    pub timed_out: bool,
    pub completed: BTreeMap<String, WaitAgentCompletion>,
    pub pending: BTreeMap<String, WaitAgentPending>,
    pub closed: BTreeMap<String, WaitAgentClosed>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CloseAgentResponse {
    pub closed: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitUntil {
    TaskCompleted,
    Terminal,
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
