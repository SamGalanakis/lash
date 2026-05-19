use serde::Serialize;

use crate::sansio::EffectId;

use super::controller::RuntimeEffectControllerError;
use super::envelope::{EffectInvocationMetadata, EffectOrigin, RuntimeEffectKind};

pub(crate) fn turn_idempotency_key(
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    mode_iteration: usize,
    kind: RuntimeEffectKind,
    effect_id: EffectId,
) -> String {
    format!(
        "{session_id}:{turn_id}:{turn_index}:{mode_iteration}:{}:{}",
        kind.as_str(),
        effect_id.0
    )
}

pub(crate) fn direct_effect_metadata(
    session_id: &str,
    usage_source: &str,
    effect_kind: RuntimeEffectKind,
    idempotency_discriminator: String,
    turn_id: Option<&str>,
) -> EffectInvocationMetadata {
    let origin = match effect_kind {
        RuntimeEffectKind::DirectCompletion => EffectOrigin::DirectCompletion {
            usage_source: usage_source.to_string(),
        },
        RuntimeEffectKind::DirectLlmCompletion => EffectOrigin::DirectLlmCompletion {
            usage_source: usage_source.to_string(),
        },
        _ => unreachable!("direct invocation requires a direct effect kind"),
    };
    let idempotency_key = match turn_id.filter(|value| !value.is_empty()) {
        Some(turn_id) => format!(
            "{session_id}:{turn_id}:direct:{}:{usage_source}:{idempotency_discriminator}",
            effect_kind.as_str()
        ),
        None => format!(
            "{session_id}:direct:{}:{usage_source}:{idempotency_discriminator}",
            effect_kind.as_str()
        ),
    };
    EffectInvocationMetadata {
        session_id: session_id.to_string(),
        origin,
        turn_id: turn_id.map(str::to_string),
        turn_index: None,
        mode_iteration: None,
        effect_id: idempotency_discriminator,
        effect_kind,
        idempotency_key,
        turn_checkpoint_hash: None,
    }
}

pub(crate) fn tool_retry_sleep_metadata(
    parent: &EffectInvocationMetadata,
    tool_name: &str,
    attempt: u32,
) -> EffectInvocationMetadata {
    let effect_id = format!("{}:{tool_name}:attempt:{attempt}:sleep", parent.effect_id);
    let idempotency_key = format!(
        "{}:{tool_name}:attempt:{attempt}:sleep",
        parent.idempotency_key
    );
    EffectInvocationMetadata {
        session_id: parent.session_id.clone(),
        origin: parent.origin.clone(),
        turn_id: parent.turn_id.clone(),
        turn_index: parent.turn_index,
        mode_iteration: parent.mode_iteration,
        effect_id,
        effect_kind: RuntimeEffectKind::Sleep,
        idempotency_key,
        turn_checkpoint_hash: parent.turn_checkpoint_hash.clone(),
    }
}

pub(crate) fn direct_request_discriminator<T>(
    request: &T,
    explicit_key: Option<&str>,
    parent_tool_call_id: Option<&str>,
) -> Result<String, RuntimeEffectControllerError>
where
    T: Serialize,
{
    if let Some(explicit_key) = explicit_key.filter(|key| !key.is_empty()) {
        return Ok(match parent_tool_call_id.filter(|id| !id.is_empty()) {
            Some(parent) => format!("tool:{parent}:request:{explicit_key}"),
            None => format!("request:{explicit_key}"),
        });
    }
    let digest = crate::stable_hash::stable_json_sha256_hex(request).map_err(|err| {
        RuntimeEffectControllerError::new(
            "runtime_effect_discriminator",
            format!("failed to serialize runtime effect discriminator: {err}"),
        )
    })?;
    Ok(match parent_tool_call_id.filter(|id| !id.is_empty()) {
        Some(parent) => format!("tool:{parent}:sha256:{digest}"),
        None => format!("sha256:{digest}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_effect_metadata_preserves_metadata_shape() {
        let metadata = direct_effect_metadata(
            "s",
            "tool",
            RuntimeEffectKind::DirectCompletion,
            "request:k".to_string(),
            None,
        );

        assert_eq!(metadata.session_id, "s");
        assert_eq!(
            metadata.origin,
            EffectOrigin::DirectCompletion {
                usage_source: "tool".to_string()
            }
        );
        assert!(
            metadata
                .idempotency_key
                .starts_with("s:direct:direct_completion:tool:request:k")
        );
        assert!(metadata.turn_checkpoint_hash.is_none());
    }

    #[test]
    fn tool_retry_sleep_metadata_preserves_parent_checkpoint_digest() {
        let mut parent = direct_effect_metadata(
            "s",
            "tool",
            RuntimeEffectKind::DirectCompletion,
            "request:k".to_string(),
            Some("turn"),
        );
        parent.turn_checkpoint_hash = Some("a".repeat(64));

        let sleep = tool_retry_sleep_metadata(&parent, "probe", 2);

        assert_eq!(sleep.effect_kind, RuntimeEffectKind::Sleep);
        assert_eq!(sleep.turn_checkpoint_hash, parent.turn_checkpoint_hash);
        assert!(sleep.idempotency_key.ends_with(":probe:attempt:2:sleep"));
    }
}
