//! Event routing and path utilities for the subagent host.
//!
//! Responsibilities:
//!   * Translate parent/child paths between absolute and relative forms,
//!     validate segment shape, and slugify names.
//!   * Decide which events in the per-tree queue a given `wait_agent`
//!     caller is allowed to see, based on the caller's current path,
//!     target filters, and `until` mode.
//!   * Build the final `WaitAgentResponse` from a collected event batch.
//!   * Trim a session snapshot down to the N most recent user turns
//!     when forking a child session with bounded history.
//!
//! Everything here is a pure function — no `Arc`, no lock acquisition,
//! no async. Keeping it separate makes the live orchestrator in
//! `local.rs` easier to read and lets the routing rules be tested
//! without spinning up a `LocalSubagentHost`.

use std::collections::HashSet;

use crate::types::{
    WaitAgentClosed, WaitAgentCompletion, WaitAgentEvent, WaitAgentMessage, WaitAgentResponse,
    WaitUntil,
};

pub(crate) fn event_matches(
    event: &WaitAgentEvent,
    current_path: &str,
    targets: &[String],
) -> bool {
    if targets.is_empty() {
        return match event {
            WaitAgentEvent::TaskStarted { target, .. }
            | WaitAgentEvent::TaskCompleted { target, .. }
            | WaitAgentEvent::AgentClosed { target } => is_same_or_descendant(target, current_path),
            WaitAgentEvent::Message { from, to, .. } => {
                to == current_path || is_descendant(from, current_path)
            }
        };
    }

    targets.iter().any(|target| match event {
        WaitAgentEvent::TaskStarted {
            target: event_target,
            ..
        }
        | WaitAgentEvent::TaskCompleted {
            target: event_target,
            ..
        }
        | WaitAgentEvent::AgentClosed {
            target: event_target,
        } => event_target == target,
        WaitAgentEvent::Message { from, .. } => from == target,
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
        WaitUntil::AnyEvent => true,
    }
}

pub(crate) fn wait_response(timed_out: bool, events: Vec<WaitAgentEvent>) -> WaitAgentResponse {
    let completion = events.iter().find_map(|event| {
        if let WaitAgentEvent::TaskCompleted {
            target,
            task,
            status,
            result,
            error,
            session,
        } = event
        {
            Some(WaitAgentCompletion {
                target: target.clone(),
                task: task.clone(),
                status: status.clone(),
                result: result.clone(),
                error: error.clone(),
                session: session.clone(),
            })
        } else {
            None
        }
    });
    let message = events.iter().find_map(|event| {
        if let WaitAgentEvent::Message { from, to, message } = event {
            Some(WaitAgentMessage {
                from: from.clone(),
                to: to.clone(),
                message: message.clone(),
            })
        } else {
            None
        }
    });
    let closed = events.iter().find_map(|event| {
        if let WaitAgentEvent::AgentClosed { target } = event {
            Some(WaitAgentClosed {
                target: target.clone(),
            })
        } else {
            None
        }
    });
    WaitAgentResponse {
        timed_out,
        completion,
        message,
        closed,
        events,
    }
}

pub(crate) fn normalize_relative_path(value: &str) -> Result<String, String> {
    let segments = value
        .split('/')
        .filter(|segment| !segment.is_empty())
        .map(validate_segment)
        .collect::<Result<Vec<_>, _>>()?;
    if segments.is_empty() {
        return Err("path must not be empty".to_string());
    }
    Ok(segments.join("/"))
}

pub(crate) fn normalize_absolute_path(value: &str) -> Result<String, String> {
    Ok(format!("/{}", normalize_relative_path(value)?))
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
            "task name segment `{segment}` has no usable characters — use letters or digits"
        ))
    } else {
        Ok(out)
    }
}

pub(crate) fn join_path(parent: &str, relative: &str) -> String {
    if parent == "/root" {
        format!("/root/{relative}")
    } else {
        format!("{parent}/{relative}")
    }
}

pub(crate) fn is_same_or_descendant(path: &str, prefix: &str) -> bool {
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

pub(crate) fn is_descendant(path: &str, prefix: &str) -> bool {
    path.starts_with(&format!("{prefix}/"))
}

pub fn truncate_snapshot_to_recent_turns(
    mut snapshot: lash::SessionSnapshot,
    turns: usize,
) -> lash::SessionSnapshot {
    if turns == 0 {
        return snapshot;
    }

    let messages = snapshot.project_conversation_messages();
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
    let kept_tool_calls = snapshot
        .project_tool_calls()
        .into_iter()
        .filter(|tool_call| {
            tool_call
                .call_id
                .as_ref()
                .is_some_and(|call_id| referenced.contains(call_id))
        })
        .collect::<Vec<_>>();
    snapshot.replace_projection(&kept_messages, &kept_tool_calls);
    snapshot
}
