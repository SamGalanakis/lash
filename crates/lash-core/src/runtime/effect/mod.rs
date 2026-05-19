mod controller;
mod direct;
mod envelope;
mod identity;
mod inline;
mod journal;
mod local;
mod spec;
mod trace;

pub use controller::{
    RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectControllerScope,
};
pub use envelope::{
    EffectInvocationMetadata, EffectOrigin, RuntimeEffectCommand, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectOutcome,
};
pub use inline::InlineRuntimeEffectController;
pub use local::{
    BackgroundTaskLocalExecutor, LocalBackgroundCancelPolicy, RuntimeEffectLocalExecutor,
};
pub use spec::{DirectRequestSpec, LlmAttachmentSpec, LlmRequestSpec};

pub(crate) use controller::RuntimeEffectControllerHandle;
pub(crate) use direct::{apply_direct_completion_outcome, apply_direct_llm_completion_outcome};
pub(crate) use identity::{
    direct_effect_metadata, direct_request_discriminator, tool_retry_sleep_metadata,
    turn_idempotency_key,
};
pub(crate) use journal::{execute_effect_with_journal, renew_runtime_turn_lease_for_effect};
pub(crate) use trace::{
    LlmTraceFailure, emit_llm_trace_completed, emit_llm_trace_failed, emit_llm_trace_started,
    token_usage_from_llm,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DirectRequest;
    use crate::LlmRequest as CoreLlmRequest;
    use crate::llm::types::{
        LlmAttachment, LlmEventSender, LlmMessage, LlmProviderTraceSender, LlmToolChoice,
    };
    use std::sync::Arc;

    #[test]
    fn runtime_effect_envelope_and_request_specs_round_trip_without_live_fields() {
        let attachment_store = crate::InMemoryAttachmentStore::new();
        let llm_request = CoreLlmRequest {
            model: "model".to_string(),
            messages: vec![LlmMessage::text(crate::llm::types::LlmRole::User, "hello")],
            attachments: vec![LlmAttachment::bytes("image/png", vec![1, 2, 3, 4])],
            tools: Arc::new(Vec::new()),
            tool_choice: LlmToolChoice::None,
            model_variant: Some("fast".to_string()),
            session_id: Some("session".to_string()),
            output_spec: None,
            stream_events: Some(LlmEventSender::new(|_| {})),
            provider_trace: Some(LlmProviderTraceSender::new(|_| {})),
        };
        let spec = LlmRequestSpec::from_request(&llm_request, &attachment_store).expect("llm spec");
        let encoded = serde_json::to_string(&spec).expect("serialize llm spec");
        assert!(!encoded.contains("stream_events"));
        assert!(!encoded.contains("provider_trace"));
        assert!(!encoded.contains("\"data\""));
        assert!(encoded.contains(crate::attachments::content_id(&[1, 2, 3, 4]).as_str()));
        let decoded: LlmRequestSpec = serde_json::from_str(&encoded).expect("decode llm spec");
        let live = decoded.into_request(None, None);
        assert_eq!(live.model, "model");
        assert!(live.attachments[0].data.is_empty());
        assert!(live.attachments[0].reference.is_some());
        assert!(live.stream_events.is_none());
        assert!(live.provider_trace.is_none());

        let direct_request = DirectRequest::text("model", "direct");
        let direct_spec = DirectRequestSpec::from_request(&direct_request, &attachment_store)
            .expect("direct spec");
        let metadata = direct_effect_metadata(
            "session",
            "test",
            RuntimeEffectKind::DirectCompletion,
            "request:direct".to_string(),
            Some("turn"),
        );
        let envelope = RuntimeEffectEnvelope::new(
            metadata,
            RuntimeEffectCommand::DirectCompletion {
                request: direct_spec,
                normalized_request: LlmRequestSpec::from_request(&llm_request, &attachment_store)
                    .expect("normalized spec"),
                model: "model".to_string(),
                usage_source: "test".to_string(),
            },
        );
        let hash = envelope.stable_hash().expect("stable hash");
        assert!(!hash.is_empty());
        let encoded = serde_json::to_string(&envelope).expect("serialize envelope");
        let decoded: RuntimeEffectEnvelope =
            serde_json::from_str(&encoded).expect("decode envelope");
        assert_eq!(
            decoded.metadata.idempotency_key,
            envelope.metadata.idempotency_key
        );
        assert_eq!(decoded.command.kind(), RuntimeEffectKind::DirectCompletion);
    }
}
