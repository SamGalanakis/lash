mod envelope;
mod executor;
mod outcome;

pub use envelope::{
    LlmAttachmentSpec, LlmRequestSpec, ProcessCommand, ProcessEffectOutcome, RuntimeEffectCommand,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectOutcome, RuntimeInvocation,
    RuntimeReplay, RuntimeScope, RuntimeSubject,
};
pub use executor::{
    EffectHost, EffectScope, InlineEffectHost, InlineRuntimeEffectController,
    RuntimeEffectController, RuntimeEffectControllerError, RuntimeEffectLocalExecutor,
    ScopedEffectController,
};
pub use lash_sansio::CausalRef;

pub(crate) use executor::{ProcessRunner, RuntimeEffectControllerHandle};
pub(crate) use outcome::{
    LlmTraceFailure, apply_direct_outcome, emit_llm_trace_completed, emit_llm_trace_failed,
    emit_llm_trace_started, token_usage_from_llm,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LlmRequest as CoreLlmRequest;
    use crate::llm::types::{
        LlmAttachment, LlmEventSender, LlmMessage, LlmProviderTraceSender, LlmToolChoice,
    };
    use std::sync::Arc;

    #[tokio::test]
    async fn runtime_effect_envelope_and_request_specs_round_trip_without_live_fields() {
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
        let spec = LlmRequestSpec::from_request(&llm_request, &attachment_store)
            .await
            .expect("llm spec");
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

        let invocation = crate::runtime::causal::direct_effect_invocation(
            "session",
            "test",
            "request:direct".to_string(),
            Some("turn"),
            None,
        );
        let envelope = RuntimeEffectEnvelope::new(
            invocation,
            RuntimeEffectCommand::Direct {
                request: Box::new(
                    LlmRequestSpec::from_request(&llm_request, &attachment_store)
                        .await
                        .expect("normalized spec"),
                ),
                usage_source: "test".to_string(),
            },
        );
        let hash = envelope.stable_hash().expect("stable hash");
        assert!(!hash.is_empty());
        let encoded = serde_json::to_string(&envelope).expect("serialize envelope");
        let decoded: RuntimeEffectEnvelope =
            serde_json::from_str(&encoded).expect("decode envelope");
        assert_eq!(
            decoded.invocation.replay_key(),
            envelope.invocation.replay_key()
        );
        assert_eq!(decoded.command.kind(), RuntimeEffectKind::Direct);
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
        let invocation = RuntimeInvocation::effect(
            RuntimeScope::for_turn("session", "turn", 0, 0),
            "process:start:call-123",
            RuntimeEffectKind::Process,
            "session:turn:process:start:call-123",
        );
        let envelope = RuntimeEffectEnvelope::new(
            invocation,
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
