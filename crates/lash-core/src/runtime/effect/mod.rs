mod envelope;
mod executor;
mod journal;

pub use envelope::{
    DirectRequestSpec, EffectInvocationMetadata, EffectOrigin, LlmAttachmentSpec, LlmRequestSpec,
    ProcessCommand, ProcessEffectOutcome, RuntimeEffectCommand, RuntimeEffectEnvelope,
    RuntimeEffectKind, RuntimeEffectOutcome,
};
pub use executor::{
    InlineRuntimeEffectController, RuntimeEffectController, RuntimeEffectControllerError,
    RuntimeEffectControllerScope, RuntimeEffectLocalExecutor,
};

pub(crate) use executor::{ProcessRunner, RuntimeEffectControllerHandle};
pub(crate) use journal::{
    LlmTraceFailure, apply_direct_completion_outcome, apply_direct_llm_completion_outcome,
    direct_effect_metadata, direct_request_discriminator, emit_llm_trace_completed,
    emit_llm_trace_failed, emit_llm_trace_started, execute_effect_with_journal,
    renew_runtime_turn_lease_for_effect, token_usage_from_llm, tool_retry_sleep_metadata,
    turn_idempotency_key,
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
            generation: crate::GenerationOptions::default(),
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
                request: Box::new(direct_spec),
                normalized_request: Box::new(
                    LlmRequestSpec::from_request(&llm_request, &attachment_store)
                        .expect("normalized spec"),
                ),
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

    #[test]
    fn process_effect_envelope_round_trips_prepared_tool_call() {
        let registration = crate::ProcessRegistration::new(
            "call-123",
            crate::ProcessInput::ToolCall {
                call: crate::PreparedToolCall {
                    call_id: "call-123".to_string(),
                    tool_name: "echo".to_string(),
                    args: serde_json::json!({"value": "hi"}),
                    replay: None,
                    prepared_payload: serde_json::json!({"context": "prepared"}),
                },
            },
        );
        let metadata = EffectInvocationMetadata {
            session_id: "session".to_string(),
            origin: EffectOrigin::Turn,
            turn_id: Some("turn".to_string()),
            turn_index: Some(0),
            mode_iteration: Some(0),
            effect_id: "process:start:call-123".to_string(),
            effect_kind: RuntimeEffectKind::Process,
            idempotency_key: "session:turn:process:start:call-123".to_string(),
            turn_checkpoint_hash: Some("0".repeat(64)),
        };
        let envelope = RuntimeEffectEnvelope::new(
            metadata,
            RuntimeEffectCommand::Process {
                command: ProcessCommand::Start {
                    registration,
                    grant: None,
                    execution_context: Box::new(crate::ProcessExecutionContext::default()),
                },
            },
        );

        let hash = envelope.stable_hash().expect("hash");
        let decoded: RuntimeEffectEnvelope =
            serde_json::from_str(&serde_json::to_string(&envelope).expect("serialize"))
                .expect("decode");

        assert_eq!(decoded.command.kind(), RuntimeEffectKind::Process);
        assert_eq!(decoded.stable_hash().expect("decoded hash"), hash);
        let RuntimeEffectCommand::Process {
            command:
                ProcessCommand::Start {
                    registration,
                    grant: None,
                    execution_context,
                },
        } = decoded.command
        else {
            panic!("wrong process command");
        };
        assert!(execution_context.is_empty());
        let crate::ProcessInput::ToolCall { call } = registration.input.as_ref() else {
            panic!("wrong process input");
        };
        assert_eq!(call.call_id, "call-123");
        assert_eq!(call.tool_name, "echo");
        assert_eq!(call.args, serde_json::json!({"value": "hi"}));
        assert_eq!(
            call.prepared_payload,
            serde_json::json!({"context": "prepared"})
        );
    }
}
