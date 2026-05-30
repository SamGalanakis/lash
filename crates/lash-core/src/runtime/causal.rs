use serde::Serialize;

use crate::sansio::EffectId;
use crate::{
    CausalRef, RuntimeEffectControllerError, RuntimeEffectKind, RuntimeInvocation, RuntimeReplay,
    RuntimeScope, RuntimeSubject,
};

pub(crate) fn turn_effect_invocation(
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    protocol_iteration: usize,
    effect_id: EffectId,
    effect_kind: RuntimeEffectKind,
    checkpoint_hash: Option<String>,
) -> RuntimeInvocation {
    RuntimeInvocation::effect(
        RuntimeScope::for_turn(session_id, turn_id, turn_index, protocol_iteration),
        effect_id.0.to_string(),
        effect_kind,
        turn_effect_replay_key(
            session_id,
            turn_id,
            turn_index,
            protocol_iteration,
            effect_kind,
            effect_id,
        ),
        checkpoint_hash,
    )
}

fn turn_effect_replay_key(
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    protocol_iteration: usize,
    kind: RuntimeEffectKind,
    effect_id: EffectId,
) -> String {
    format!(
        "{session_id}:{turn_id}:{turn_index}:{protocol_iteration}:{}:{}",
        kind.as_str(),
        effect_id.0
    )
}

pub(crate) fn child_effect_invocation(
    parent: &RuntimeInvocation,
    effect_id: impl Into<String>,
    kind: RuntimeEffectKind,
    replay_suffix: impl AsRef<str>,
) -> RuntimeInvocation {
    let replay_base = parent
        .replay_key()
        .or_else(|| parent.effect_id())
        .unwrap_or("effect");
    RuntimeInvocation {
        scope: parent.scope.clone(),
        subject: RuntimeSubject::Effect {
            effect_id: effect_id.into(),
            kind,
        },
        caused_by: parent.causal_ref(),
        replay: Some(RuntimeReplay {
            key: format!("{replay_base}:{}", replay_suffix.as_ref()),
        }),
        checkpoint_hash: parent.checkpoint_hash.clone(),
    }
}

pub(crate) fn child_tool_effect_invocation(
    parent: &RuntimeInvocation,
    parent_effect_id: EffectId,
    call_id: &str,
) -> RuntimeInvocation {
    child_effect_invocation(
        parent,
        format!("{}:{call_id}", parent_effect_id.0),
        RuntimeEffectKind::ToolCall,
        call_id,
    )
}

pub(crate) fn tool_retry_sleep_invocation(
    parent: &RuntimeInvocation,
    tool_name: &str,
    attempt: u32,
) -> RuntimeInvocation {
    let parent_effect_id = parent.effect_id().unwrap_or("effect");
    child_effect_invocation(
        parent,
        format!("{parent_effect_id}:{tool_name}:attempt:{attempt}:sleep"),
        RuntimeEffectKind::Sleep,
        format!("{tool_name}:attempt:{attempt}:sleep"),
    )
}

pub(crate) fn lashlang_sleep_invocation(
    session_id: &str,
    parent: Option<&RuntimeInvocation>,
    scope: &str,
    sequence: u64,
) -> RuntimeInvocation {
    let suffix = format!("lashlang:{scope}:sleep:{sequence}");
    if let Some(parent) = parent {
        let parent_effect_id = parent.effect_id().unwrap_or("effect");
        return child_effect_invocation(
            parent,
            format!("{parent_effect_id}:{suffix}"),
            RuntimeEffectKind::Sleep,
            suffix,
        );
    }
    RuntimeInvocation::effect(
        RuntimeScope::new(session_id),
        suffix.clone(),
        RuntimeEffectKind::Sleep,
        suffix,
        None,
    )
}

pub(crate) fn process_effect_invocation(
    session_id: &str,
    parent: Option<RuntimeInvocation>,
    effect_id: &str,
) -> RuntimeInvocation {
    if let Some(parent) = parent {
        let scope = if let Some(turn_id) = parent.scope.turn_id.clone() {
            RuntimeScope {
                session_id: session_id.to_string(),
                turn_id: Some(turn_id),
                turn_index: parent.scope.turn_index,
                protocol_iteration: parent.scope.protocol_iteration,
            }
        } else {
            RuntimeScope::new(session_id)
        };
        let replay_base = parent.replay_key().unwrap_or("process");
        return RuntimeInvocation {
            scope,
            subject: RuntimeSubject::Effect {
                effect_id: effect_id.to_string(),
                kind: RuntimeEffectKind::Process,
            },
            caused_by: parent.causal_ref(),
            replay: Some(RuntimeReplay {
                key: format!("{replay_base}:{effect_id}"),
            }),
            checkpoint_hash: parent.checkpoint_hash,
        };
    }
    RuntimeInvocation::effect(
        RuntimeScope::new(session_id),
        effect_id.to_string(),
        RuntimeEffectKind::Process,
        format!("{session_id}:{effect_id}"),
        None,
    )
}

pub fn process_event_invocation(
    owner_session_id: &str,
    process_id: &str,
    sequence: u64,
    event_type: &str,
    replay: Option<RuntimeReplay>,
) -> RuntimeInvocation {
    RuntimeInvocation {
        scope: RuntimeScope::new(owner_session_id),
        subject: RuntimeSubject::ProcessEvent {
            process_id: process_id.to_string(),
            sequence,
            event_type: event_type.to_string(),
        },
        caused_by: Some(CausalRef::Process {
            process_id: process_id.to_string(),
        }),
        replay,
        checkpoint_hash: None,
    }
}

pub(crate) fn session_node_invocation(
    session_id: &str,
    node_id: impl Into<String>,
) -> RuntimeInvocation {
    RuntimeInvocation {
        scope: RuntimeScope::new(session_id),
        subject: RuntimeSubject::SessionNode {
            node_id: node_id.into(),
        },
        caused_by: None,
        replay: None,
        checkpoint_hash: None,
    }
}

pub(crate) fn direct_effect_invocation(
    session_id: &str,
    usage_source: &str,
    replay_discriminator: String,
    turn_id: Option<&str>,
    caused_by: Option<CausalRef>,
) -> RuntimeInvocation {
    let replay_key = match turn_id.filter(|value| !value.is_empty()) {
        Some(turn_id) => {
            format!("{session_id}:{turn_id}:direct:{usage_source}:{replay_discriminator}")
        }
        None => format!("{session_id}:direct:{usage_source}:{replay_discriminator}"),
    };
    RuntimeInvocation::effect(
        RuntimeScope {
            session_id: session_id.to_string(),
            turn_id: turn_id.map(str::to_string),
            turn_index: None,
            protocol_iteration: None,
        },
        replay_discriminator,
        RuntimeEffectKind::Direct,
        replay_key,
        None,
    )
    .with_caused_by(caused_by)
}

pub(crate) fn direct_request_discriminator<T>(
    request: &T,
    explicit_replay: Option<&RuntimeReplay>,
    caused_by: Option<&CausalRef>,
) -> Result<String, RuntimeEffectControllerError>
where
    T: Serialize,
{
    let cause_discriminator = caused_by
        .map(causal_replay_discriminator)
        .unwrap_or_default();
    if let Some(replay) = explicit_replay.filter(|replay| !replay.key.is_empty()) {
        return Ok(format!("{cause_discriminator}request:{}", replay.key));
    }
    let digest = crate::stable_hash::stable_json_sha256_hex(request).map_err(|err| {
        RuntimeEffectControllerError::new(
            "runtime_effect_discriminator",
            format!("failed to serialize runtime effect discriminator: {err}"),
        )
    })?;
    Ok(format!("{cause_discriminator}sha256:{digest}"))
}

fn causal_replay_discriminator(caused_by: &CausalRef) -> String {
    match caused_by {
        CausalRef::Turn {
            session_id,
            turn_id,
        } => format!("cause:turn:{session_id}:{turn_id}:"),
        CausalRef::Effect {
            session_id,
            turn_id,
            effect_id,
        } => {
            let turn = turn_id.as_deref().unwrap_or("");
            format!("cause:effect:{session_id}:{turn}:{effect_id}:")
        }
        CausalRef::ToolCall {
            session_id,
            call_id,
        } => format!("cause:tool_call:{session_id}:{call_id}:"),
        CausalRef::Process { process_id } => format!("cause:process:{process_id}:"),
        CausalRef::ProcessEvent {
            process_id,
            sequence,
        } => format!("cause:process_event:{process_id}:{sequence}:"),
        CausalRef::SessionNode {
            session_id,
            node_id,
        } => format!("cause:session_node:{session_id}:{node_id}:"),
    }
}
