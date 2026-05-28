use std::sync::Arc;

use lash_core::sansio::{self, ChatContextProjector, ProtocolDriverHandle, Response};
use lash_core::{Effect, TurnMachine, TurnMachineConfig};
use lash_protocol_rlm::RlmDriver;
use lash_protocol_standard::StandardDriver;
use lash_rlm_types::{RlmProtocolEvent, RlmTermination, RlmTrajectoryEntry};
use lash_sansio::llm::types::{LlmOutputPart, LlmRequest, LlmResponse};
use lash_sansio::{CheckpointKind, Message, MessageRole, Part, PartKind, PruneState, SessionEvent};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestProtocol {
    Standard,
    Rlm,
}

fn test_config(protocol: TestProtocol) -> TurnMachineConfig {
    test_config_with_termination(protocol, RlmTermination::default())
}

fn test_config_with_termination(
    protocol: TestProtocol,
    rlm_termination: RlmTermination,
) -> TurnMachineConfig {
    test_config_with_protocol_turn_options(
        protocol,
        lash_core::ProtocolTurnOptions::typed(rlm_termination).expect("valid rlm turn options"),
    )
}

fn test_config_with_protocol_turn_options(
    protocol: TestProtocol,
    termination: lash_core::ProtocolTurnOptions,
) -> TurnMachineConfig {
    let protocol_driver: Arc<dyn ProtocolDriverHandle<lash_core::HostTurnProtocol>> = match protocol
    {
        TestProtocol::Standard => Arc::new(StandardDriver),
        TestProtocol::Rlm => Arc::new(RlmDriver),
    };
    TurnMachineConfig {
        protocol_driver,
        projector: Arc::new(ChatContextProjector),
        sync_execution_surface: protocol == TestProtocol::Rlm,
        model: "test-model".to_string(),
        max_turns: None,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        run_session_id: None,
        autonomous: false,
        tool_specs: Vec::new().into(),
        system_prompt: std::sync::Arc::from(""),
        session_id: "test".to_string(),
        emit_llm_trace: false,
        termination,
        turn_limit_final_message: Arc::new(test_turn_limit_final_message),
    }
}

fn test_turn_limit_final_message(message_id: String, max_turns: usize) -> Message {
    Message {
        id: message_id.clone(),
        role: MessageRole::System,
        parts: lash_sansio::shared_parts(vec![Part {
            id: format!("{message_id}.p0"),
            kind: PartKind::Error,
            content: format!("Turn limit reached ({max_turns}) before a final test response."),
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

fn user_message(content: &str) -> Message {
    Message {
        id: "m0".to_string(),
        role: MessageRole::User,
        parts: vec![Part {
            id: "m0.p0".to_string(),
            kind: PartKind::Text,
            content: content.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]
        .into(),
        origin: None,
    }
}

fn drain_effects(machine: &mut TurnMachine) -> Vec<Effect> {
    let mut effects = Vec::new();
    while let Some(effect) = machine.poll_effect() {
        if let Effect::SyncExecutionSurface { id, .. } = effect {
            effects.push(effect);
            machine.handle_response(Response::ExecutionSurfaceSynced {
                id,
                result: Ok(Some(sansio::ExecutionSurfaceSync {
                    system_prompt: std::sync::Arc::from(""),
                    tool_specs: Arc::new(Vec::new()),
                })),
            });
            continue;
        }
        effects.push(effect);
    }
    effects
}

fn find_llm_call(effects: &[Effect]) -> Option<&sansio::EffectId> {
    effects.iter().find_map(|e| match e {
        Effect::LlmCall { id, .. } => Some(id),
        _ => None,
    })
}

fn find_llm_request(effects: &[Effect]) -> Option<&LlmRequest> {
    effects
        .iter()
        .find_map(|e| match e {
            Effect::LlmCall { request, .. } => Some(request),
            _ => None,
        })
        .map(|request| request.as_ref())
}

fn find_checkpoint(effects: &[Effect]) -> Option<(sansio::EffectId, CheckpointKind)> {
    effects.iter().find_map(|e| match e {
        Effect::Checkpoint { id, checkpoint } => Some((*id, *checkpoint)),
        _ => None,
    })
}

fn find_done(effects: &[Effect]) -> Option<(&lash_sansio::MessageSequence, usize)> {
    effects.iter().find_map(|e| match e {
        Effect::Done {
            messages,
            event_delta: _,
            protocol_iteration,
        } => Some((messages, *protocol_iteration)),
        _ => None,
    })
}

fn roundtrip_turn_checkpoint(
    checkpoint: lash_sansio::TurnCheckpoint<lash_core::HostTurnProtocol>,
) -> lash_sansio::TurnCheckpoint<lash_core::HostTurnProtocol> {
    let encoded = serde_json::to_string(&checkpoint).expect("serialize checkpoint");
    serde_json::from_str(&encoded).expect("deserialize checkpoint")
}

fn machine_trajectory(machine: &TurnMachine) -> Vec<RlmTrajectoryEntry> {
    machine
        .events()
        .iter()
        .filter_map(|event| match event {
            lash_core::SessionEventRecord::Protocol(event) => {
                match lash_protocol_rlm::decode_rlm_protocol_event(event) {
                    Some(RlmProtocolEvent::RlmTrajectoryEntry(entry)) => Some(entry),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect()
}

fn effects_include_runtime_error(effects: &[Effect], message_fragment: &str) -> bool {
    let has_error = effects.iter().any(|effect| {
        matches!(
            effect,
            Effect::Emit(SessionEvent::Error { message, .. })
                if message.contains(message_fragment)
        )
    });
    let has_runtime_outcome = effects.iter().any(|effect| {
        matches!(
            effect,
            Effect::Emit(SessionEvent::TurnOutcome {
                outcome: lash_sansio::TurnOutcome::Stopped(lash_sansio::TurnStop::RuntimeError)
            })
        )
    });
    has_error && has_runtime_outcome
}

fn rewrite_first_rlm_driver_state_owner(value: &mut serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            if map.get("plugin_id").and_then(serde_json::Value::as_str)
                == Some(lash_protocol_rlm::RLM_PROTOCOL_PLUGIN_ID)
            {
                map.insert(
                    "plugin_id".to_string(),
                    serde_json::Value::String("other_protocol".to_string()),
                );
                return true;
            }
            map.values_mut().any(rewrite_first_rlm_driver_state_owner)
        }
        serde_json::Value::Array(values) => {
            values.iter_mut().any(rewrite_first_rlm_driver_state_owner)
        }
        _ => false,
    }
}

#[test]
fn standard_prose_only_response_emits_done() {
    let config = test_config(TestProtocol::Standard);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello there!".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello there!".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

#[test]
fn standard_tool_calls_produce_effects_and_loop() {
    let config = test_config(TestProtocol::Standard);
    let msgs = vec![user_message("read file")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![
                LlmOutputPart::Text {
                    text: "Let me read that.".to_string(),
                    response_meta: None,
                },
                LlmOutputPart::ToolCall {
                    call_id: "tc1".to_string(),
                    tool_name: "read_file".to_string(),
                    input_json: r#"{"path":"foo.txt"}"#.to_string(),
                    replay: None,
                },
            ],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let tool_effect = effects.iter().find_map(|e| match e {
        Effect::ToolCalls { id, calls } => calls.first().map(|call| {
            (
                *id,
                call.call_id.clone(),
                call.tool_name.clone(),
                call.args.clone(),
            )
        }),
        _ => None,
    });
    let (tool_id, call_id, tool_name, args) = tool_effect.expect("tool calls effect");
    assert_eq!(args, serde_json::json!({"path":"foo.txt"}));

    machine.handle_response(Response::ToolResults {
        id: tool_id,
        results: vec![sansio::CompletedToolCall {
            call_id: call_id.clone(),
            tool_name: tool_name.clone(),
            args,
            output: lash_sansio::ToolCallOutput::success(serde_json::json!("file contents")),
            model_return: lash_sansio::ModelToolReturn {
                call_id: call_id.clone(),
                tool_name: tool_name.clone(),
                parts: vec![lash_sansio::ModelToolReturnPart::Text(
                    "file contents".to_string(),
                )],
            },
            duration_ms: 10,
            replay: None,
        }],
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn standard_checkpoint_redrives_parallel_tool_batch_without_losing_calls() {
    let config = test_config(TestProtocol::Standard);
    let msgs = vec![user_message("read and search")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![
                LlmOutputPart::ToolCall {
                    call_id: "tc-read".to_string(),
                    tool_name: "read_file".to_string(),
                    input_json: r#"{"path":"foo.txt"}"#.to_string(),
                    replay: None,
                },
                LlmOutputPart::ToolCall {
                    call_id: "tc-grep".to_string(),
                    tool_name: "grep".to_string(),
                    input_json: r#"{"pattern":"needle"}"#.to_string(),
                    replay: None,
                },
            ],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (tool_id, calls) = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ToolCalls { id, calls } => Some((*id, calls.clone())),
            _ => None,
        })
        .expect("tool batch");
    assert_eq!(calls.len(), 2);

    let checkpoint = roundtrip_turn_checkpoint(machine.checkpoint());
    let mut restored =
        TurnMachine::restore_from_checkpoint(test_config(TestProtocol::Standard), checkpoint);
    let effects = drain_effects(&mut restored);
    let (restored_tool_id, restored_calls) = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ToolCalls { id, calls } => Some((*id, calls)),
            _ => None,
        })
        .expect("restored tool batch");
    assert_eq!(restored_tool_id, tool_id);
    assert_eq!(restored_calls.len(), 2);
    assert_eq!(restored_calls[0].call_id, "tc-read");
    assert_eq!(restored_calls[1].tool_name, "grep");
}

#[test]
fn standard_checkpoint_after_tool_control_finish_preserves_terminal_outcome() {
    let config = test_config(TestProtocol::Standard);
    let msgs = vec![user_message("finish through a tool")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "tc-finish".to_string(),
                tool_name: "submit_result".to_string(),
                input_json: "{}".to_string(),
                replay: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let tool_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ToolCalls { id, .. } => Some(*id),
            _ => None,
        })
        .expect("tool call");
    machine.handle_response(Response::ToolResults {
        id: tool_id,
        results: vec![sansio::CompletedToolCall {
            call_id: "tc-finish".to_string(),
            tool_name: "submit_result".to_string(),
            args: serde_json::json!({}),
            output: lash_sansio::ToolCallOutput::success(serde_json::json!({"ok": true}))
                .with_control(lash_core::ToolControl::Finish {
                    value: serde_json::json!({"ok": true}).into(),
                }),
            model_return: lash_sansio::ModelToolReturn {
                call_id: "tc-finish".to_string(),
                tool_name: "submit_result".to_string(),
                parts: vec![lash_sansio::ModelToolReturnPart::Text("done".to_string())],
            },
            duration_ms: 1,
            replay: None,
        }],
    });

    let checkpoint = roundtrip_turn_checkpoint(machine.checkpoint());
    let mut restored =
        TurnMachine::restore_from_checkpoint(test_config(TestProtocol::Standard), checkpoint);
    let effects = drain_effects(&mut restored);
    assert!(find_llm_call(&effects).is_none());
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::Emit(SessionEvent::TurnOutcome {
            outcome: lash_sansio::TurnOutcome::Finished(
                lash_sansio::TurnFinish::ToolValue { tool_name, value }
            )
        }) if tool_name == "submit_result" && *value == serde_json::json!({"ok": true})
    )));
    assert!(find_done(&effects).is_some());
}

#[test]
fn standard_empty_final_after_tool_result_finishes_without_error() {
    let config = test_config(TestProtocol::Standard);
    let msgs = vec![user_message("update the plan and do nothing else")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "tc1".to_string(),
                tool_name: "update_plan".to_string(),
                input_json: r#"{"plan":[{"step":"done","status":"completed"}]}"#.to_string(),
                replay: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let tool_id = effects
        .iter()
        .find_map(|e| match e {
            Effect::ToolCalls { id, .. } => Some(*id),
            _ => None,
        })
        .expect("tool call");
    machine.handle_response(Response::ToolResults {
        id: tool_id,
        results: vec![sansio::CompletedToolCall {
            call_id: "tc1".to_string(),
            tool_name: "update_plan".to_string(),
            args: serde_json::json!({"plan":[{"step":"done","status":"completed"}]}),
            output: lash_sansio::ToolCallOutput::success(serde_json::json!("Plan updated")),
            model_return: lash_sansio::ModelToolReturn {
                call_id: "tc1".to_string(),
                tool_name: "update_plan".to_string(),
                parts: vec![lash_sansio::ModelToolReturnPart::Text(
                    "Plan updated".to_string(),
                )],
            },
            duration_ms: 1,
            replay: None,
        }],
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("after-work checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("follow-up llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse::default()),
    });

    let effects = drain_effects(&mut machine);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Emit(SessionEvent::Error { .. })))
    );
    let (checkpoint_id, checkpoint) =
        find_checkpoint(&effects).expect("before-completion checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Emit(SessionEvent::Error { .. })))
    );
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

#[test]
fn standard_max_turns_stops_iteration() {
    let mut config = test_config(TestProtocol::Standard);
    config.max_turns = Some(1);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "tc1".to_string(),
                tool_name: "test".to_string(),
                input_json: "{}".to_string(),
                replay: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let tool_id = effects
        .iter()
        .find_map(|e| match e {
            Effect::ToolCalls { id, .. } => Some(*id),
            _ => None,
        })
        .expect("tool call");
    machine.handle_response(Response::ToolResults {
        id: tool_id,
        results: vec![sansio::CompletedToolCall {
            call_id: "tc1".to_string(),
            tool_name: "test".to_string(),
            args: serde_json::json!({}),
            output: lash_sansio::ToolCallOutput::success(serde_json::json!("ok")),
            model_return: lash_sansio::ModelToolReturn {
                call_id: "tc1".to_string(),
                tool_name: "test".to_string(),
                parts: vec![lash_sansio::ModelToolReturnPart::Text("ok".to_string())],
            },
            duration_ms: 1,
            replay: None,
        }],
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_prose_only_response_requests_submit_by_default() {
    let config = test_config(TestProtocol::Rlm);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello there!".to_string(),
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    assert!(machine.messages().iter().any(|message| {
        message.role == MessageRole::System
            && message.parts.iter().any(|part| {
                part.content.contains("Deliver the final answer") && part.content.contains("submit")
            })
    }));
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn typed_rlm_prose_only_response_requests_submit() {
    let config = test_config_with_termination(
        TestProtocol::Rlm,
        RlmTermination::SubmitRequired { schema: None },
    );
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello there!".to_string(),
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    assert!(machine.messages().iter().any(|message| {
        message.role == MessageRole::System
            && message.parts.iter().any(|part| {
                part.content.contains("Deliver the final answer")
                    && part.content.contains("submit")
                    && !part.content.contains("required output schema")
            })
    }));
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn submit_required_rlm_prose_at_max_turns_stops_without_retry_prompt() {
    let mut config = test_config_with_termination(
        TestProtocol::Rlm,
        RlmTermination::SubmitRequired { schema: None },
    );
    config.max_turns = Some(1);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "plain prose cannot finish submit-required RLM".to_string(),
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_none());
    assert!(find_checkpoint(&effects).is_none());
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::Emit(SessionEvent::TurnOutcome {
            outcome: lash_core::TurnOutcome::Stopped(lash_core::TurnStop::MaxTurns)
        })
    )));
    assert!(find_done(&effects).is_some());
    assert!(!machine.messages().iter().any(|message| {
        message.parts.iter().any(|part| {
            part.content.contains("Deliver the final answer") && part.content.contains("submit")
        })
    }));
}

#[test]
fn submit_required_rlm_exec_error_at_max_turns_stops_without_retry() {
    let mut config = test_config_with_termination(
        TestProtocol::Rlm,
        RlmTermination::SubmitRequired { schema: None },
    );
    config.max_turns = Some(1);
    let msgs = vec![user_message("run bad code")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nmissing_name\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nmissing_name\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: Some("unknown variable `missing_name`".to_string()),
            duration_ms: 1,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_none());
    assert!(find_checkpoint(&effects).is_none());
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::Emit(SessionEvent::TurnOutcome {
            outcome: lash_core::TurnOutcome::Stopped(lash_core::TurnStop::MaxTurns)
        })
    )));
    assert!(find_done(&effects).is_some());
    let trajectory = machine_trajectory(&machine);
    assert!(
        trajectory
            .last()
            .and_then(|entry| entry.error.as_deref())
            .is_some_and(|error| error.contains("missing_name"))
    );
}

#[test]
fn prose_or_submit_response_finishes_with_assistant_message() {
    let config = test_config_with_termination(TestProtocol::Rlm, RlmTermination::ProseOrSubmit);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello there!".to_string(),
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(!effects.iter().any(|effect| {
        matches!(
            effect,
            Effect::Emit(lash_sansio::SessionEvent::Message { kind, .. }) if kind == "final"
        )
    }));
    assert!(
        !effects.iter().any(|effect| matches!(
            effect,
            Effect::Progress { event_delta, .. }
                if event_delta.iter().any(|event| matches!(
                    event,
                    lash_sansio::SessionEventRecord::Conversation(record)
                        if record.to_message().role == MessageRole::Assistant
                ))
        )),
        "RLM prose finalization must not write a protocol-owned assistant message"
    );
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(effects.iter().any(|effect| {
        matches!(
            effect,
            Effect::Emit(lash_sansio::SessionEvent::TurnOutcome {
                outcome: lash_sansio::TurnOutcome::Finished(
                    lash_sansio::TurnFinish::AssistantMessage { text }
                )
            }) if text == "Hello there!"
        )
    }));
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_fenced_lashlang_block_runs_exec_and_continues() {
    let config = test_config(TestProtocol::Rlm);
    let msgs = vec![user_message("run some code")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    let request = find_llm_request(&effects).expect("request");
    assert!(request.tools.is_empty());

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Quick check.\n\n```lashlang\nprint \"hi\"\n```\n".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Quick check.\n\n```lashlang\nprint \"hi\"\n```\n".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_effect = effects.iter().find_map(|e| match e {
        Effect::ExecCode { id, code } => Some((*id, code.clone())),
        _ => None,
    });
    assert_eq!(
        exec_effect.as_ref().map(|(_, code)| code.as_str()),
        Some("print \"hi\"")
    );

    machine.handle_response(Response::ExecResult {
        id: exec_effect.expect("exec").0,
        result: Ok(lash_sansio::ExecResponse {
            output: "hi\n".to_string(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    let trajectory = machine_trajectory(&machine);
    let entry = trajectory.last().expect("rlm trajectory entry");
    assert_eq!(entry.code, "print \"hi\"");
    // Raw lashlang stdout (rare) is folded into the trajectory's
    // `output: Vec<String>` as a single anonymous entry alongside any
    // `print` results.
    assert_eq!(entry.output, vec!["hi\n".to_string()]);
    assert!(entry.final_output.is_none());

    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn rlm_empty_turn_options_use_submit_required_default() {
    let config = test_config_with_protocol_turn_options(
        TestProtocol::Rlm,
        lash_core::ProtocolTurnOptions::empty(),
    );
    let msgs = vec![user_message("submit")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nsubmit \"done\"\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nsubmit \"done\"\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!("done")),
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::Emit(SessionEvent::TurnOutcome {
            outcome: lash_sansio::TurnOutcome::Finished(
                lash_sansio::TurnFinish::SubmittedValue { value }
            )
        }) if value == &serde_json::json!("done")
    )));
    assert!(find_done(&effects).is_some());
}

#[test]
fn malformed_rlm_turn_options_fail_before_llm() {
    let config = test_config_with_protocol_turn_options(
        TestProtocol::Rlm,
        lash_core::ProtocolTurnOptions {
            payload: serde_json::json!({ "kind": "unknown" }),
        },
    );
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);

    assert!(find_llm_call(&effects).is_none());
    assert!(effects_include_runtime_error(
        &effects,
        "invalid RLM turn options"
    ));
    assert!(find_done(&effects).is_some());
}

#[test]
fn null_rlm_turn_options_fail_before_llm() {
    let config = test_config_with_protocol_turn_options(
        TestProtocol::Rlm,
        lash_core::ProtocolTurnOptions {
            payload: serde_json::Value::Null,
        },
    );
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);

    assert!(find_llm_call(&effects).is_none());
    assert!(effects_include_runtime_error(
        &effects,
        "invalid RLM turn options"
    ));
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_driver_state_with_wrong_plugin_id_fails_loudly() {
    let config = test_config(TestProtocol::Rlm);
    let msgs = vec![user_message("run some code")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);
    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());

    let mut checkpoint = serde_json::to_value(machine.checkpoint()).expect("checkpoint serializes");
    assert!(
        rewrite_first_rlm_driver_state_owner(&mut checkpoint),
        "checkpoint should contain RLM driver state"
    );
    let checkpoint = serde_json::from_value(checkpoint).expect("checkpoint deserializes");
    let mut restored =
        TurnMachine::restore_from_checkpoint(test_config(TestProtocol::Rlm), checkpoint);

    let effects = drain_effects(&mut restored);
    let llm_id = *find_llm_call(&effects).expect("restored llm call");
    restored.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nprint \"hi\"\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nprint \"hi\"\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut restored);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::ExecCode { .. })),
        "invalid driver state must not reach code execution"
    );
    assert!(effects_include_runtime_error(
        &effects,
        "driver state belongs to plugin"
    ));
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_checkpoint_redrives_pending_exec_code_with_driver_state() {
    let config = test_config(TestProtocol::Rlm);
    let msgs = vec![user_message("run some code")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Reason first.\n```lashlang\nprint \"hi\"\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Reason first.\n```lashlang\nprint \"hi\"\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (exec_id, code) = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, code } => Some((*id, code.clone())),
            _ => None,
        })
        .expect("exec effect");
    assert_eq!(code, "print \"hi\"");

    let checkpoint = roundtrip_turn_checkpoint(machine.checkpoint());
    let mut restored =
        TurnMachine::restore_from_checkpoint(test_config(TestProtocol::Rlm), checkpoint);
    let effects = drain_effects(&mut restored);
    let (restored_exec_id, restored_code) = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, code } => Some((*id, code.clone())),
            _ => None,
        })
        .expect("restored exec effect");
    assert_eq!(restored_exec_id, exec_id);
    assert_eq!(restored_code, "print \"hi\"");

    restored.handle_response(Response::ExecResult {
        id: restored_exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: "hi\n".to_string(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut restored);
    let trajectory = machine_trajectory(&restored);
    let entry = trajectory.last().expect("rlm trajectory entry");
    assert_eq!(entry.code, "print \"hi\"");
    assert!(entry.reasoning.contains("Reason first."));
    assert_eq!(entry.output, vec!["hi\n".to_string()]);
    let (_, checkpoint) = find_checkpoint(&effects).expect("after-work checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
}

#[test]
fn rlm_checkpoint_after_exec_fanout_tool_outputs_preserves_structured_outcomes() {
    let config = test_config(TestProtocol::Rlm);
    let msgs = vec![user_message("run fanout tools")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nok = await tools.ok({})\nfail = await tools.fail({})\nstop = await tools.stop({})\nresults = { a: ok, b: fail, c: stop }\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nok = await tools.ok({})\nfail = await tools.fail({})\nstop = await tools.stop({})\nresults = { a: ok, b: fail, c: stop }\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: vec!["fanout done".to_string()],
            observation_truncation: Vec::new(),
            tool_calls: vec![
                lash_core::ToolCallRecord {
                    call_id: Some("fanout-ok".to_string()),
                    tool: "ok".to_string(),
                    args: serde_json::json!({}),
                    output: lash_core::ToolCallOutput::success(serde_json::json!("ok")),
                    duration_ms: 1,
                },
                lash_core::ToolCallRecord {
                    call_id: Some("fanout-fail".to_string()),
                    tool: "fail".to_string(),
                    args: serde_json::json!({}),
                    output: lash_core::ToolCallOutput::failure(lash_core::ToolFailure::tool(
                        lash_core::ToolFailureClass::Execution,
                        "tool_failed",
                        "failed but captured",
                    )),
                    duration_ms: 2,
                },
                lash_core::ToolCallRecord {
                    call_id: Some("fanout-cancel".to_string()),
                    tool: "stop".to_string(),
                    args: serde_json::json!({}),
                    output: lash_core::ToolCallOutput::cancelled(
                        lash_core::ToolCancellation::runtime("cancelled sibling"),
                    ),
                    duration_ms: 3,
                },
            ],
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 3,
            terminal_finish: None,
        }),
    });

    let checkpoint = roundtrip_turn_checkpoint(machine.checkpoint());
    let mut restored =
        TurnMachine::restore_from_checkpoint(test_config(TestProtocol::Rlm), checkpoint);
    let effects = drain_effects(&mut restored);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Emit(SessionEvent::ToolCall { .. })))
    );
    let trajectory = machine_trajectory(&restored);
    let entry = trajectory.last().expect("rlm trajectory entry");
    assert_eq!(
        entry.tool_call_ids,
        vec![
            "fanout-ok".to_string(),
            "fanout-fail".to_string(),
            "fanout-cancel".to_string()
        ]
    );
    assert_eq!(entry.output, vec!["fanout done".to_string()]);
    let (_, checkpoint) = find_checkpoint(&effects).expect("after-work checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
}

#[test]
fn rlm_exec_result_stores_tool_call_ids_without_replayed_tool_events() {
    let config = test_config(TestProtocol::Rlm);
    let msgs = vec![user_message("run a tool")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nx = await tools.read_file({ path: \"foo\" })?\n```"
                .to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nx = await tools.read_file({ path: \"foo\" })?\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: vec![lash_core::ToolCallRecord {
                call_id: Some("rlm-call-1".to_string()),
                tool: "read_file".to_string(),
                args: serde_json::json!({"path": "foo"}),
                output: lash_core::ToolCallOutput::success(serde_json::json!("contents")),
                duration_ms: 7,
            }],
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 7,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Emit(SessionEvent::ToolCall { .. })))
    );
    let trajectory = machine_trajectory(&machine);
    let entry = trajectory.last().expect("rlm trajectory entry");
    assert_eq!(entry.tool_call_ids, vec!["rlm-call-1".to_string()]);
}

#[test]
fn rlm_exec_any_tool_control_handoff_is_terminal() {
    let config = test_config(TestProtocol::Rlm);
    let msgs = vec![user_message("run a custom handoff tool")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nx = await tools.custom_handoff({})?\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nx = await tools.custom_handoff({})?\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: vec![lash_core::ToolCallRecord {
                call_id: Some("custom-call-1".to_string()),
                tool: "custom_handoff".to_string(),
                args: serde_json::json!({}),
                output: lash_core::ToolCallOutput::success(serde_json::json!({"ok": true}))
                    .with_control(lash_core::ToolControl::Handoff {
                        session_id: "successor-session".to_string(),
                    }),
                duration_ms: 3,
            }],
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 3,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Emit(SessionEvent::ToolCall { .. })))
    );
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::Emit(SessionEvent::TurnOutcome {
            outcome: lash_core::TurnOutcome::Handoff { session_id }
        }) if session_id == "successor-session"
    )));
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_exec_any_tool_control_fail_is_terminal_error() {
    let config = test_config(TestProtocol::Rlm);
    let msgs = vec![user_message("run a custom failure tool")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nx = await tools.custom_fail({})?\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nx = await tools.custom_fail({})?\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: vec![lash_core::ToolCallRecord {
                call_id: Some("custom-call-1".to_string()),
                tool: "custom_fail".to_string(),
                args: serde_json::json!({}),
                output: lash_core::ToolCallOutput::success(serde_json::json!({"ok": true}))
                    .with_control(lash_core::ToolControl::Fail {
                        failure: lash_core::ToolFailure::tool(
                            lash_core::ToolFailureClass::Execution,
                            "custom_fail",
                            "no valid result",
                        ),
                    }),
                duration_ms: 3,
            }],
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 3,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Emit(SessionEvent::ToolCall { .. })))
    );
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(effects.iter().any(|effect| matches!(
        effect,
        Effect::Emit(SessionEvent::TurnOutcome {
            outcome: lash_core::TurnOutcome::Stopped(lash_core::TurnStop::ToolError {
                tool_name: name,
                value,
            })
        }) if name == "custom_fail" && value.get("message") == Some(&serde_json::json!("no valid result"))
    )));
    assert!(find_done(&effects).is_some());
}

#[test]
fn typed_rlm_finish_emits_turn_outcome_and_done() {
    let config = test_config_with_termination(
        TestProtocol::Rlm,
        RlmTermination::SubmitRequired { schema: None },
    );
    let msgs = vec![user_message("return typed data")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nsubmit { ok: true }\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nsubmit { ok: true }\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!({ "ok": true })),
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(
        !effects.iter().any(|effect| matches!(
            effect,
            Effect::Emit(lash_sansio::SessionEvent::Message { kind, .. }) if kind == "final"
        )),
        "RLM submit should surface through SubmittedValue, not a duplicate final message"
    );
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(effects.iter().any(|e| matches!(
        e,
        Effect::Emit(lash_sansio::SessionEvent::TurnOutcome {
            outcome: lash_sansio::TurnOutcome::Finished(
                lash_sansio::TurnFinish::SubmittedValue { value }
            )
        }) if *value == serde_json::json!({ "ok": true })
    )));
    assert!(find_done(&effects).is_some());
}

#[test]
fn prose_or_submit_allows_submit_value() {
    let config = test_config_with_termination(TestProtocol::Rlm, RlmTermination::ProseOrSubmit);
    let msgs = vec![user_message("return typed data")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nsubmit { ok: true }\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nsubmit { ok: true }\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!({ "ok": true })),
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(effects.iter().any(|effect| {
        matches!(
            effect,
            Effect::Emit(lash_sansio::SessionEvent::TurnOutcome {
                outcome: lash_sansio::TurnOutcome::Finished(
                    lash_sansio::TurnFinish::SubmittedValue { value }
                )
            }) if *value == serde_json::json!({ "ok": true })
        )
    }));
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_reasoning_part_is_preserved_in_trajectory() {
    let config = test_config_with_termination(
        TestProtocol::Rlm,
        RlmTermination::SubmitRequired { schema: None },
    );
    let msgs = vec![user_message("say hi")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: true,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nsubmit \"Hi.\"\n```".to_string(),
            parts: vec![
                LlmOutputPart::Reasoning {
                    text: "I'll answer directly.".to_string(),
                    replay: None,
                },
                LlmOutputPart::Text {
                    text: "```lashlang\nsubmit \"Hi.\"\n```".to_string(),
                    response_meta: None,
                },
            ],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!("Hi.")),
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(
        effects
            .iter()
            .any(|effect| { matches!(effect, Effect::Checkpoint { .. }) })
    );
    assert!(
        !effects.iter().any(|effect| matches!(
            effect,
            Effect::Emit(lash_sansio::SessionEvent::Message { kind, .. }) if kind == "final"
        )),
        "RLM submit should surface through SubmittedValue, not a duplicate final message"
    );
    let trajectory = machine_trajectory(&machine);
    let entry = trajectory.last().expect("trajectory entry");
    assert!(entry.reasoning.contains("I'll answer directly."));
    assert!(entry.reasoning.contains("```lashlang"));
    assert_eq!(entry.final_output, Some(serde_json::json!("Hi.")));
}

#[test]
fn typed_rlm_schema_mismatch_loops_with_feedback() {
    let config = test_config_with_termination(
        TestProtocol::Rlm,
        RlmTermination::SubmitRequired {
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "ok": { "type": "boolean" }
                },
                "required": ["ok"]
            })),
        },
    );
    let msgs = vec![user_message("return typed data")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nsubmit { missing: true }\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nsubmit { missing: true }\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!({ "missing": true })),
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    assert!(machine.messages().iter().any(|message| {
        message.role == MessageRole::System
            && message.parts.iter().any(|part| {
                part.content
                    .contains("didn't match the required output schema")
            })
    }));
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        delivery: sansio::CheckpointDelivery::default(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn typed_rlm_schema_mismatch_checks_any_of() {
    let config = test_config_with_termination(
        TestProtocol::Rlm,
        RlmTermination::SubmitRequired {
            schema: Some(serde_json::json!({
                "anyOf": [
                    { "type": "string" },
                    { "type": "integer" }
                ]
            })),
        },
    );
    let msgs = vec![user_message("return typed data")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nsubmit true\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nsubmit true\n```".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec_id = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, .. } => Some(*id),
            _ => None,
        })
        .expect("exec effect");
    machine.handle_response(Response::ExecResult {
        id: exec_id,
        result: Ok(lash_sansio::ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!(true)),
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_checkpoint(&effects).is_some());
    assert!(machine.messages().iter().any(|message| {
        message.role == MessageRole::System
            && message.parts.iter().any(|part| {
                part.content
                    .contains("didn't match the required output schema")
            })
    }));
}
