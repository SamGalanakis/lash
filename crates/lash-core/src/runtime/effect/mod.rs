mod envelope;
mod executor;
mod outcome;

pub use envelope::{
    LlmAttachmentSpec, LlmRequestSpec, ProcessCommand, ProcessEffectOutcome, RuntimeEffectCommand,
    RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeEffectOutcome, RuntimeInvocation,
    RuntimeReplay, RuntimeScope, RuntimeSubject, ToolAttemptEffectOutcome, ToolAttemptLaunch,
    ToolBatchEffectOutcome, ToolCallLaunch,
};
pub use executor::{
    AwaitEventKey, AwaitEventWaitIdentity, EffectHost, ExecutionScope, ExternalCompletionError,
    InlineEffectHost, InlineRuntimeEffectController, Resolution, ResolveOutcome,
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
            scope: Some(crate::LlmRequestScope::new(
                "session",
                "session:frame:test",
                "session:turn:test:llm:0",
            )),
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
                    tool_id: crate::ToolId::from("tool:echo"),
                    tool_name: "echo".to_string(),
                    args: serde_json::json!({"value": "hi"}),
                    replay: None,
                    prepared_payload: serde_json::json!({"context": "prepared"}),
                },
            },
            crate::ProcessProvenance::host(),
        );
        let invocation = RuntimeInvocation::effect(
            RuntimeScope::for_turn("session", "turn", 0, 0),
            "process:start:call-123",
            RuntimeEffectKind::Process,
            "session:turn:process:start:call-123",
        );
        let envelope = RuntimeEffectEnvelope::new(
            invocation,
            RuntimeEffectCommand::process(ProcessCommand::Start {
                registration,
                grant: None,
                execution_context: Box::new(crate::ProcessExecutionContext::default()),
            }),
        );

        let hash = envelope.stable_hash().expect("hash");
        let decoded: RuntimeEffectEnvelope =
            serde_json::from_str(&serde_json::to_string(&envelope).expect("serialize"))
                .expect("decode");

        assert_eq!(decoded.command.kind(), RuntimeEffectKind::Process);
        assert_eq!(decoded.stable_hash().expect("decoded hash"), hash);
        let RuntimeEffectCommand::Process { command } = decoded.command else {
            panic!("wrong process command");
        };
        let ProcessCommand::Start {
            registration,
            grant: None,
            execution_context,
        } = *command
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

    fn prepared_tool_call(call_id: &str, tool_name: &str) -> crate::PreparedToolCall {
        crate::PreparedToolCall {
            call_id: call_id.to_string(),
            tool_id: crate::ToolId::from(format!("tool:{tool_name}")),
            tool_name: tool_name.to_string(),
            args: serde_json::json!({"value": call_id}),
            replay: None,
            prepared_payload: serde_json::json!({"prepared": true}),
        }
    }

    #[test]
    fn tool_batch_effect_envelope_round_trips_and_hashes_stably() {
        let batch = crate::PreparedToolBatch::new(
            "batch-123",
            vec![
                prepared_tool_call("call-1", "echo"),
                prepared_tool_call("call-2", "lookup"),
            ],
        );
        let invocation = RuntimeInvocation::effect(
            RuntimeScope::for_turn("session", "turn", 0, 0),
            "tool-batch:batch-123",
            RuntimeEffectKind::ToolBatch,
            "session:turn:tool-batch:batch-123",
        );
        let envelope = RuntimeEffectEnvelope::new(
            invocation,
            RuntimeEffectCommand::ToolBatch {
                batch: batch.clone(),
            },
        );

        let hash = envelope.stable_hash().expect("hash");
        let decoded: RuntimeEffectEnvelope =
            serde_json::from_str(&serde_json::to_string(&envelope).expect("serialize"))
                .expect("decode");

        assert_eq!(decoded.command.kind(), RuntimeEffectKind::ToolBatch);
        assert_eq!(decoded.stable_hash().expect("decoded hash"), hash);
        let RuntimeEffectCommand::ToolBatch {
            batch: decoded_batch,
        } = decoded.command
        else {
            panic!("wrong command");
        };
        assert_eq!(decoded_batch.batch_id, batch.batch_id);
        assert_eq!(decoded_batch.calls.len(), 2);
        assert_eq!(decoded_batch.calls[0].call.call_id, "call-1");
        assert_eq!(decoded_batch.calls[0].replay_suffix, "child:0:call-1");
        assert_eq!(decoded_batch.calls[1].call.call_id, "call-2");
        assert_eq!(decoded_batch.calls[1].replay_suffix, "child:1:call-2");
    }

    #[test]
    fn tool_batch_outcome_rejects_wrong_effect_kind() {
        let error = RuntimeEffectOutcome::ToolAttempt {
            launch: ToolAttemptLaunch::Done {
                record: crate::ToolCallRecord {
                    call_id: Some("call-1".to_string()),
                    tool: "echo".to_string(),
                    args: serde_json::json!({"value": "call-1"}),
                    output: crate::ToolCallOutput::success(serde_json::json!({"done": "call-1"})),
                    duration_ms: 7,
                },
            },
            triggers: Vec::new(),
        }
        .into_tool_batch_effect()
        .expect_err("tool attempt is not a tool batch outcome");

        assert_eq!(error.code, "runtime_effect_wrong_outcome");
        assert!(error.message.contains("expected tool_batch outcome"));
        assert!(error.message.contains("got tool_attempt"));
    }

    #[tokio::test]
    async fn await_event_key_is_stable_for_scope_and_wait_identity() {
        let host = InlineEffectHost::default();
        let scope = ExecutionScope::turn("session", "turn");
        let wait = AwaitEventWaitIdentity::tool_completion("call");

        let first = host
            .await_event_key(&scope, wait.clone())
            .await
            .expect("first key");
        let second = host
            .await_event_key(&scope, wait)
            .await
            .expect("second key");

        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn duplicate_await_event_resolution_reports_existing_terminal() {
        let host = InlineEffectHost::default();
        let scope = ExecutionScope::turn("session-dupe", "turn-dupe");
        let key = host
            .await_event_key(&scope, AwaitEventWaitIdentity::tool_completion("call-dupe"))
            .await
            .expect("key");
        let resolution = Resolution::Ok(serde_json::json!({"done": true}));

        let first = host
            .resolve_await_event(&key, resolution.clone())
            .await
            .expect("first resolve");
        let second = host
            .resolve_await_event(&key, Resolution::Ok(serde_json::json!({"ignored": true})))
            .await
            .expect("duplicate resolve");

        assert_eq!(first, ResolveOutcome::Accepted);
        assert_eq!(
            second,
            ResolveOutcome::AlreadyResolved {
                terminal: resolution
            }
        );
    }
}
