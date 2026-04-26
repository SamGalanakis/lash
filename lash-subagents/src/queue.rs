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
//! for that same target are purged. Without this, a `wait_agent` call
//! issued right after a `followup_task` on an idle agent would drain
//! the PRIOR task's completion and return immediately, skipping the
//! follow-up. The regression test
//! `queue_event_task_started_purges_stale_completion_for_same_target`
//! in `local.rs` pins this behavior.
//!
//! All fields are `pub(crate)` because the orchestrator in `local.rs`
//! reads and mutates them directly while holding `state.lock()`. They
//! are never exposed outside the crate.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use lash::{AssembledTurn, InputItem, TurnInput, TurnStatus};
use serde_json::{Value, json};

use crate::types::{WaitAgentEvent, WaitAgentSessionSummary};

#[derive(Default)]
pub(crate) struct HostState {
    pub(crate) trees: HashMap<String, AgentTree>,
    pub(crate) session_agents: HashMap<String, AgentLocator>,
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
}

#[derive(Clone)]
pub(crate) struct QueuedMessage {
    pub(crate) from: String,
    pub(crate) message: String,
}

pub(crate) struct AgentRecord {
    pub(crate) session_id: String,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) parent_path: Option<String>,
    pub(crate) capability: Option<String>,
    pub(crate) model: String,
    pub(crate) model_variant: Option<String>,
    pub(crate) active_turn: Option<ActiveTurn>,
    pub(crate) queued_turns: VecDeque<QueuedTurn>,
    pub(crate) closing: bool,
    pub(crate) last_task_state: Option<String>,
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
    Message { from: String },
}

pub(crate) enum QueuedTurn {
    Task(QueuedTask),
    Message(QueuedMessage),
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
    if let WaitAgentEvent::TaskStarted { target, .. } = &event {
        let target = target.clone();
        tree.events.retain(|existing| match existing {
            WaitAgentEvent::TaskCompleted {
                target: other_target,
                ..
            }
            | WaitAgentEvent::AgentClosed {
                target: other_target,
            } => other_target != &target,
            _ => true,
        });
    }
    tree.events.push_back(event);
    tree.notify.notify_waiters();
}

pub(crate) fn build_session_summary(
    agent: &AgentRecord,
    task: &str,
    turn: &AssembledTurn,
) -> WaitAgentSessionSummary {
    WaitAgentSessionSummary {
        id: agent.session_id.clone(),
        parent_session_id: agent.parent_session_id.clone(),
        task: task.to_string(),
        iterations: turn.state.iteration,
        tool_calls: turn.state.projected_tool_calls().len(),
        model: turn.state.policy.model.clone(),
        model_variant: turn.state.policy.model_variant.clone(),
        token_usage: json!({
            "input_tokens": turn.token_usage.input_tokens,
            "output_tokens": turn.token_usage.output_tokens,
            "cached_input_tokens": turn.token_usage.cached_input_tokens,
            "reasoning_tokens": turn.token_usage.reasoning_tokens,
            "total_tokens": turn.token_usage.total(),
        }),
    }
}

pub(crate) fn task_result_value(turn: &AssembledTurn) -> Value {
    if let Some(value) = &turn.typed_finish {
        return value.clone();
    }
    if !turn.assistant_output.safe_text.trim().is_empty() {
        return json!(turn.assistant_output.safe_text.trim().to_string());
    }
    json!(turn.assistant_output.raw_text.trim().to_string())
}

pub(crate) fn task_result_text(turn: &AssembledTurn) -> String {
    match task_result_value(turn) {
        Value::String(text) => text,
        value => serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
    }
}

pub(crate) fn task_completion_event(
    agent: &mut AgentRecord,
    path: &str,
    task: String,
    outcome: &Result<AssembledTurn, lash::PluginError>,
) -> WaitAgentEvent {
    match outcome {
        Ok(turn) => {
            let status = turn_status_label(&turn.status);
            agent.last_task_state = Some(status.to_string());
            let session = build_session_summary(agent, &task, turn);
            WaitAgentEvent::TaskCompleted {
                target: path.to_string(),
                task,
                status: status.to_string(),
                result: task_result_value(turn),
                error: None,
                session,
            }
        }
        Err(_) => {
            agent.last_task_state = Some("failed".to_string());
            WaitAgentEvent::TaskCompleted {
                target: path.to_string(),
                task: task.clone(),
                status: "failed".to_string(),
                result: Value::Null,
                error: Some("Subagent failed while executing its task.".to_string()),
                session: WaitAgentSessionSummary {
                    id: agent.session_id.clone(),
                    parent_session_id: agent.parent_session_id.clone(),
                    task,
                    iterations: 0,
                    tool_calls: 0,
                    model: agent.model.clone(),
                    model_variant: agent.model_variant.clone(),
                    token_usage: json!({
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_input_tokens": 0,
                        "reasoning_tokens": 0,
                        "total_tokens": 0,
                    }),
                },
            }
        }
    }
}

pub(crate) fn message_response_event(
    path: &str,
    to: String,
    outcome: &Result<AssembledTurn, lash::PluginError>,
) -> WaitAgentEvent {
    let message = match outcome {
        Ok(turn) => task_result_text(turn),
        Err(_) => "Subagent failed while handling the message.".to_string(),
    };
    WaitAgentEvent::Message {
        from: path.to_string(),
        to,
        message,
    }
}

pub(crate) fn message_turn_input(from: &str, message: &str) -> TurnInput {
    TurnInput {
        items: vec![InputItem::Text {
            text: format!(
                "## Message from {from}\n\n{message}\n\nRespond to this message directly. If no reply is needed, briefly acknowledge it."
            ),
        }],
        image_blobs: HashMap::new(),
        user_input: None,
        mode: None,
        mode_turn_options: Some(lash::ModeTurnOptions::rlm(
            lash_rlm_types::RlmTermination::ProseWithoutFence,
        )),
    }
}

pub(crate) fn turn_status_label(status: &TurnStatus) -> &'static str {
    match status {
        TurnStatus::Completed => "completed",
        TurnStatus::Interrupted => "interrupted",
        TurnStatus::Failed => "failed",
    }
}
