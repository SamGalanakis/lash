//! Internal state and queueing primitives for `LocalSubagentHost`.
//!
//! This module owns the data structures the orchestrator holds behind
//! its mutex (`HostState`, `AgentTree`, `AgentRecord`, queued-turn
//! variants) plus the per-event and per-completion helpers that the
//! orchestrator uses to populate the event queue.
//!
//! The ordering invariant enforced by [`queue_event`] is load-bearing:
//! when a new `TaskStarted` event is queued for an agent, any stale
//! `TaskCompleted` or `AgentClosed` events still sitting in the queue
//! for that same target are purged. Without this, a later wait for a
//! restarted task could drain the PRIOR task's completion and return
//! immediately. The regression test
//! `queue_event_task_started_purges_stale_completion_for_same_target`
//! in `local.rs` pins this behavior.
//!
//! All fields are `pub(crate)` because the orchestrator in `local.rs`
//! reads and mutates them directly while holding `state.lock()`. They
//! are never exposed outside the crate.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use lash::{AssembledTurn, TurnFinish, TurnInput, TurnOutcome, TurnStop};
use serde_json::{Value, json};

use crate::types::WaitAgentEvent;

#[derive(Default)]
pub(crate) struct HostState {
    pub(crate) trees: HashMap<String, AgentTree>,
    pub(crate) session_agents: HashMap<String, AgentLocator>,
    pub(crate) children_by_parent_session: HashMap<String, BTreeMap<String, String>>,
}

pub(crate) struct AgentTree {
    pub(crate) agents: BTreeMap<String, AgentRecord>,
    pub(crate) events: VecDeque<WaitAgentEvent>,
    pub(crate) notify: Arc<tokio::sync::Notify>,
}

impl Default for AgentTree {
    fn default() -> Self {
        Self {
            agents: BTreeMap::new(),
            events: VecDeque::new(),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }
}

#[derive(Clone)]
pub(crate) struct AgentLocator {
    pub(crate) root_session_id: String,
    pub(crate) path: String,
    pub(crate) depth: u8,
}

pub(crate) struct AgentRecord {
    pub(crate) session_id: String,
    pub(crate) agent_name: String,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) capability: Option<String>,
    pub(crate) model: String,
    pub(crate) model_variant: Option<String>,
    pub(crate) active_turn: Option<ActiveTurn>,
    pub(crate) queued_turns: VecDeque<QueuedTurn>,
    pub(crate) closing: bool,
    pub(crate) last_task_state: Option<String>,
    /// Per-completion stats from the most recently finished turn. Stashed
    /// on the record so the activity projector can render the dock without
    /// the wire shape carrying the data.
    pub(crate) last_iterations: Option<usize>,
    pub(crate) last_tool_calls: Option<usize>,
    pub(crate) last_token_usage: Option<Value>,
    /// Session that spawned this agent; used to complete the agent's entry
    /// in that session's background task registry when the agent exits.
    pub(crate) owner_session_id: String,
}

pub(crate) struct ActiveTurn {
    pub(crate) kind: ActiveTurnKind,
    pub(crate) turn_id: String,
}

pub(crate) enum ActiveTurnKind {
    Task { task: String },
}

pub(crate) enum QueuedTurn {
    Task(QueuedTask),
}

pub(crate) struct QueuedTask {
    pub(crate) task: String,
    pub(crate) turn_input: TurnInput,
}

pub(crate) struct PreparedTurnLaunch {
    pub(crate) session_id: String,
    pub(crate) kind: ActiveTurnKind,
    pub(crate) turn_input: TurnInput,
    pub(crate) notify: Arc<tokio::sync::Notify>,
}

/// Push an event onto a tree's queue and wake waiters.
///
/// When the event is a `TaskStarted` for a given target, any stale
/// `TaskCompleted` or `AgentClosed` events for that same target are
/// purged first. See the module doc for why this matters.
pub(crate) fn queue_event(tree: &mut AgentTree, event: WaitAgentEvent) {
    if let WaitAgentEvent::TaskStarted {
        agent_name,
        parent_session_id,
        ..
    } = &event
    {
        let agent_name = agent_name.clone();
        let parent_session_id = parent_session_id.clone();
        tree.events.retain(|existing| match existing {
            WaitAgentEvent::TaskCompleted {
                agent_name: other_name,
                parent_session_id: other_parent,
                ..
            }
            | WaitAgentEvent::AgentClosed {
                agent_name: other_name,
                parent_session_id: other_parent,
            } => other_name != &agent_name || other_parent != &parent_session_id,
            _ => true,
        });
    }
    tree.events.push_back(event);
    tree.notify.notify_waiters();
}

pub(crate) fn task_result_value(turn: &AssembledTurn) -> Value {
    match &turn.outcome {
        TurnOutcome::Finished(TurnFinish::Value { value, .. }) => return value.clone(),
        TurnOutcome::Finished(TurnFinish::AssistantMessage { text }) => {
            if !text.trim().is_empty() {
                return json!(text.trim().to_string());
            }
        }
        TurnOutcome::Stopped(TurnStop::TerminalError { value, .. }) => return value.clone(),
        TurnOutcome::Handoff { session_id } => return json!({ "session_id": session_id }),
        TurnOutcome::Stopped(_) => {}
    }
    if !turn.assistant_output.safe_text.trim().is_empty() {
        return json!(turn.assistant_output.safe_text.trim().to_string());
    }
    json!(turn.assistant_output.raw_text.trim().to_string())
}

pub(crate) fn task_completion_event(
    agent: &mut AgentRecord,
    task: String,
    outcome: &Result<AssembledTurn, lash::PluginError>,
) -> WaitAgentEvent {
    match outcome {
        Ok(turn) => {
            let status = turn_outcome_label(&turn.outcome);
            agent.last_task_state = Some(status.to_string());
            agent.last_iterations = Some(turn.state.turn_index);
            agent.last_tool_calls = Some(turn.state.read_view().tool_calls().len());
            agent.last_token_usage = Some(json!({
                "input_tokens": turn.token_usage.input_tokens,
                "output_tokens": turn.token_usage.output_tokens,
                "cached_input_tokens": turn.token_usage.cached_input_tokens,
                "reasoning_tokens": turn.token_usage.reasoning_tokens,
                "total_tokens": turn.token_usage.total(),
            }));
            WaitAgentEvent::TaskCompleted {
                agent_name: agent.agent_name.clone(),
                parent_session_id: agent.parent_session_id.clone().unwrap_or_default(),
                task,
                status: status.to_string(),
                result: task_result_value(turn),
                error: terminal_error_message(&turn.outcome),
            }
        }
        Err(_) => {
            agent.last_task_state = Some("failed".to_string());
            agent.last_iterations = Some(0);
            agent.last_tool_calls = Some(0);
            agent.last_token_usage = Some(json!({
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            }));
            WaitAgentEvent::TaskCompleted {
                agent_name: agent.agent_name.clone(),
                parent_session_id: agent.parent_session_id.clone().unwrap_or_default(),
                task,
                status: "failed".to_string(),
                result: Value::Null,
                error: Some("Subagent failed while executing its task.".to_string()),
            }
        }
    }
}

pub(crate) fn turn_outcome_label(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. } => "completed",
        TurnOutcome::Stopped(TurnStop::Cancelled) => "interrupted",
        TurnOutcome::Stopped(_) => "failed",
    }
}

pub(crate) fn terminal_error_message(outcome: &TurnOutcome) -> Option<String> {
    if let TurnOutcome::Stopped(TurnStop::TerminalError { value, .. }) = outcome {
        return Some(
            value
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("Subagent reported an error.")
                .to_string(),
        );
    }
    None
}

#[cfg(test)]
mod tests {
    use lash::testing::mock_assembled_turn;

    use super::*;

    #[test]
    fn task_result_value_prefers_submission_outcome() {
        let mut turn = mock_assembled_turn("child", "fallback");
        turn.outcome = TurnOutcome::Finished(TurnFinish::Value {
            source: lash::TerminalOutputSource::RlmSubmit,
            value: json!({ "answer": "ok" }),
        });
        assert_eq!(task_result_value(&turn), json!({ "answer": "ok" }));
    }
}
