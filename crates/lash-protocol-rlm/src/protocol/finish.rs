use lash_core::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, fresh_message_id, shared_parts,
};
use serde_json::Value;

pub(crate) fn turn_limit_final_message(message_id: String, max_turns: usize) -> Message {
    Message {
        id: message_id.clone(),
        role: MessageRole::System,
        parts: shared_parts(vec![Part {
            id: format!("{message_id}.p0"),
            kind: PartKind::Text,
            content: format!(
                "Turn limit reached ({max_turns}). You MUST reply in plain prose now containing:\n\
                1. Summary of what you accomplished\n\
                2. List of remaining tasks not yet completed\n\
                3. Recommended next steps\n\
                Do NOT emit a lashlang code fence, invoke resource operations, or call submit/continue_as."
            ),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}

pub(super) fn internal_assistant_prose_message(content: String) -> Message {
    prose_message(
        content,
        Some(lash_core::MessageOrigin::Plugin {
            plugin_id: "rlm_protocol".to_string(),
            transient: false,
        }),
    )
}

fn prose_message(content: String, origin: Option<lash_core::MessageOrigin>) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::Assistant,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Prose,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin,
    }
}

pub(super) fn submit_required_reminder_message(requires_schema: bool) -> Message {
    let id = fresh_message_id();
    let content = if requires_schema {
        "Deliver the final answer from a fenced ```lashlang block by calling `submit <value>` with a value matching the required output schema. Plain text outside a fence is not delivered."
    } else {
        "Deliver the final answer from a fenced ```lashlang block by calling `submit <value>`. Plain text outside a fence is not delivered."
    };
    Message {
        id: id.clone(),
        role: MessageRole::System,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: content.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: Some(lash_core::MessageOrigin::Plugin {
            plugin_id: "rlm_protocol".to_string(),
            transient: false,
        }),
    }
}

pub(super) fn submit_schema_mismatch_message(error_text: &str) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::System,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: format!(
                "The `submit` value didn't match the required output schema:\n{error_text}\n\nFix the value and call `submit <corrected>` from another fenced ```lashlang block."
            ),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: Some(lash_core::MessageOrigin::Plugin {
            plugin_id: "rlm_protocol".to_string(),
            transient: false,
        }),
    }
}

pub(super) fn validate_finish_value(value: &Value, schema: &Value) -> Result<(), String> {
    let compiled = jsonschema::JSONSchema::compile(schema)
        .map_err(|err| format!("required output schema is invalid: {err}"))?;
    if let Err(errors) = compiled.validate(value) {
        let message = errors
            .map(|err| err.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(message);
    }
    Ok(())
}
