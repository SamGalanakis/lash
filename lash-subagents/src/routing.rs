//! Pure helpers for subagent name normalization, event filtering, wait
//! response construction, and snapshot truncation.

use std::collections::{BTreeMap, HashSet};

use crate::types::{
    WaitAgentClosed, WaitAgentCompletion, WaitAgentEvent, WaitAgentMessage, WaitAgentPending,
    WaitAgentResponse, WaitUntil,
};

pub(crate) fn event_matches(
    event: &WaitAgentEvent,
    parent_session_id: &str,
    agents: &[String],
) -> bool {
    if agents.is_empty() {
        return false;
    }
    agents.iter().any(|name| match event {
        WaitAgentEvent::TaskStarted {
            agent_name,
            parent_session_id: event_parent,
            ..
        }
        | WaitAgentEvent::TaskCompleted {
            agent_name,
            parent_session_id: event_parent,
            ..
        }
        | WaitAgentEvent::AgentClosed {
            agent_name,
            parent_session_id: event_parent,
        } => event_parent == parent_session_id && agent_name == name,
        WaitAgentEvent::Message {
            from_agent,
            parent_session_id: event_parent,
            ..
        } => event_parent == parent_session_id && from_agent == name,
    })
}

pub(crate) fn wait_until_satisfied(events: &[WaitAgentEvent], until: WaitUntil) -> bool {
    events
        .iter()
        .any(|event| event_visible_for_until(event, until))
}

pub(crate) fn event_visible_for_until(event: &WaitAgentEvent, until: WaitUntil) -> bool {
    match until {
        WaitUntil::TaskCompleted => matches!(event, WaitAgentEvent::TaskCompleted { .. }),
        WaitUntil::Terminal => {
            matches!(
                event,
                WaitAgentEvent::TaskCompleted { .. } | WaitAgentEvent::AgentClosed { .. }
            )
        }
        WaitUntil::Message => matches!(event, WaitAgentEvent::Message { .. }),
        WaitUntil::AnyResult => {
            matches!(
                event,
                WaitAgentEvent::Message { .. }
                    | WaitAgentEvent::TaskCompleted { .. }
                    | WaitAgentEvent::AgentClosed { .. }
            )
        }
    }
}

pub(crate) fn wait_response(
    timed_out: bool,
    events: Vec<WaitAgentEvent>,
    pending: BTreeMap<String, WaitAgentPending>,
) -> WaitAgentResponse {
    let mut completed = BTreeMap::new();
    let mut messages: BTreeMap<String, Vec<WaitAgentMessage>> = BTreeMap::new();
    let mut closed = BTreeMap::new();
    for event in events {
        match event {
            WaitAgentEvent::TaskCompleted {
                agent_name,
                parent_session_id: _,
                task,
                status,
                result,
                error,
            } => {
                completed.insert(
                    agent_name.clone(),
                    WaitAgentCompletion {
                        agent_name,
                        task,
                        status,
                        result,
                        error,
                    },
                );
            }
            WaitAgentEvent::Message {
                from_agent,
                parent_session_id: _,
                message,
                ..
            } => {
                messages
                    .entry(from_agent.clone())
                    .or_default()
                    .push(WaitAgentMessage {
                        from_agent,
                        message,
                    });
            }
            WaitAgentEvent::AgentClosed {
                agent_name,
                parent_session_id: _,
            } => {
                closed.insert(agent_name.clone(), WaitAgentClosed { agent_name });
            }
            WaitAgentEvent::TaskStarted { .. } => {}
        }
    }
    WaitAgentResponse {
        timed_out,
        completed,
        pending,
        messages,
        closed,
    }
}

pub(crate) fn completed_agents(events: &[WaitAgentEvent]) -> HashSet<String> {
    events
        .iter()
        .filter_map(|event| {
            if let WaitAgentEvent::TaskCompleted { agent_name, .. }
            | WaitAgentEvent::AgentClosed { agent_name, .. } = event
            {
                Some(agent_name.clone())
            } else {
                None
            }
        })
        .collect()
}

pub(crate) fn normalize_agent_name(value: &str) -> Result<String, String> {
    validate_segment(value)
}

pub(crate) fn validate_segment(segment: &str) -> Result<String, String> {
    let mut out = String::with_capacity(segment.len());
    let mut prev_was_sep = false;
    for ch in segment.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_was_sep = false;
        } else if !out.is_empty() && !prev_was_sep {
            out.push('_');
            prev_was_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        Err(format!(
            "agent_name `{segment}` has no usable characters - use letters or digits"
        ))
    } else {
        Ok(out)
    }
}

pub fn truncate_snapshot_to_recent_turns(
    mut snapshot: lash::SessionSnapshot,
    turns: usize,
) -> lash::SessionSnapshot {
    if turns == 0 {
        return snapshot;
    }

    let read_model = snapshot.read_model();
    let messages = read_model.messages.as_slice();
    let user_turn_starts = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| matches!(message.role, lash::MessageRole::User))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let Some(&start) = user_turn_starts.get(user_turn_starts.len().saturating_sub(turns)) else {
        return snapshot;
    };
    let kept_messages = messages[start..].to_vec();
    let referenced = kept_messages
        .iter()
        .flat_map(|message| message.parts.iter())
        .filter_map(|part| part.tool_call_id.clone())
        .collect::<HashSet<_>>();
    let kept_tool_calls = read_model
        .tool_calls
        .iter()
        .filter(|tool_call| {
            tool_call
                .call_id
                .as_ref()
                .is_some_and(|call_id| referenced.contains(call_id))
        })
        .cloned()
        .collect::<Vec<_>>();
    snapshot.replace_active_read_state(&kept_messages, &kept_tool_calls);
    snapshot
}
