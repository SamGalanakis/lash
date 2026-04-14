use super::*;
use crate::ExecutionMode;
use crate::session_model::{Message, MessageRole, MessageSequence, Part, PartKind, PruneState};

fn test_config(mode: ExecutionMode) -> TurnMachineConfig {
    TurnMachineConfig {
        turn_protocol: match mode {
            ExecutionMode::Standard => TurnProtocol::Standard,
            ExecutionMode::Rlm => TurnProtocol::Rlm,
        },
        sync_execution_surface: matches!(mode, ExecutionMode::Rlm),
        model: "test-model".to_string(),
        max_turns: None,
        model_variant: None,
        run_session_id: None,
        tool_specs: Vec::new().into(),
        system_prompt: String::new(),
        session_id: "test".to_string(),
        emit_llm_debug_log: false,
        rlm_termination: RlmTermination::default(),
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
            prune_state: PruneState::Intact,
        }],
        user_input: None,
        origin: None,
    }
}

/// Collect effects until a specific variant is found or exhausted.
fn drain_effects(machine: &mut TurnMachine) -> Vec<Effect> {
    let mut effects = Vec::new();
    while let Some(effect) = machine.poll_effect() {
        if let Effect::SyncExecutionSurface { id } = effect {
            effects.push(effect);
            machine.handle_response(Response::ExecutionSurfaceSynced { id, result: Ok(()) });
            continue;
        }
        effects.push(effect);
    }
    effects
}

fn find_llm_call(effects: &[Effect]) -> Option<&EffectId> {
    effects.iter().find_map(|e| match e {
        Effect::LlmCall { id, .. } => Some(id),
        _ => None,
    })
}

fn find_llm_request(effects: &[Effect]) -> Option<&LlmRequest> {
    effects.iter().find_map(|e| match e {
        Effect::LlmCall { request, .. } => Some(request),
        _ => None,
    })
}

fn find_done(effects: &[Effect]) -> Option<(&MessageSequence, usize)> {
    effects.iter().find_map(|e| match e {
        Effect::Done {
            messages,
            iteration,
        } => Some((messages, *iteration)),
        _ => None,
    })
}

fn find_retry_status(effects: &[Effect]) -> Option<(u64, usize, usize, String)> {
    effects.iter().find_map(|e| match e {
        Effect::Emit(SessionEvent::RetryStatus {
            wait_seconds,
            attempt,
            max_attempts,
            reason,
            envelope: _,
        }) => Some((*wait_seconds, *attempt, *max_attempts, reason.clone())),
        _ => None,
    })
}

fn find_retry_envelope(effects: &[Effect]) -> Option<crate::session_model::ErrorEnvelope> {
    effects.iter().find_map(|e| match e {
        Effect::Emit(SessionEvent::RetryStatus {
            envelope: Some(envelope),
            ..
        }) => Some(envelope.clone()),
        _ => None,
    })
}

fn has_error_event(effects: &[Effect], needle: &str) -> bool {
    effects.iter().any(|e| match e {
        Effect::Emit(SessionEvent::Error { message, .. }) => message.contains(needle),
        _ => false,
    })
}

fn find_checkpoint(effects: &[Effect]) -> Option<(EffectId, CheckpointKind)> {
    effects.iter().find_map(|e| match e {
        Effect::Checkpoint { id, checkpoint } => Some((*id, *checkpoint)),
        _ => None,
    })
}

fn find_llm_debug(effects: &[Effect]) -> Option<(TokenUsage, String, Option<Value>)> {
    effects.iter().find_map(|e| match e {
        Effect::Log {
            event:
                LogEvent::LlmDebug {
                    usage,
                    response_text,
                    response_parts,
                    ..
                },
        } => Some((usage.clone(), response_text.clone(), response_parts.clone())),
        _ => None,
    })
}

// ─── Standard tests ───

#[test]
fn standard_prose_only_response_emits_done() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

    // Respond with prose-only
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello there!".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello there!".to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

#[test]
fn standard_multiple_text_parts_preserve_block_boundaries() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![
                LlmOutputPart::Text {
                    text: "What’s working:".to_string(),
                },
                LlmOutputPart::Text {
                    text: "- one\n- two".to_string(),
                },
            ],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    let (messages, _) = find_done(&effects).expect("done");
    let assistant = messages
        .iter()
        .find(|message| message.role == MessageRole::Assistant)
        .expect("assistant message");
    let prose = assistant
        .parts
        .iter()
        .find(|part| matches!(part.kind, PartKind::Prose))
        .map(|part| part.content.as_str())
        .expect("prose part");

    assert_eq!(prose, "What’s working:\n\n- one\n- two");
}

#[test]
fn standard_multiple_text_parts_do_not_accumulate_extra_blank_lines() {
    let mut combined = "Heading\n".to_string();
    append_assistant_text_part(&mut combined, "\nBody");
    assert_eq!(combined, "Heading\n\nBody");
}

#[test]
fn llm_request_includes_image_prompt_parts_for_attached_images() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![Message {
        id: "m0".to_string(),
        role: MessageRole::User,
        parts: vec![
            Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Image,
                content: String::new(),
                attachment: Some(PartAttachment {
                    mime: "image/png".to_string(),
                    url: data_url_for_bytes("image/png", &[1, 2, 3]),
                    filename: None,
                }),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            },
            Part {
                id: "m0.p1".to_string(),
                kind: PartKind::Text,
                content: "explain this".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            },
        ],
        user_input: None,
        origin: None,
    }];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let request = effects
        .into_iter()
        .find_map(|effect| match effect {
            Effect::LlmCall { request, .. } => Some(request),
            _ => None,
        })
        .expect("llm call");

    assert_eq!(request.attachments.len(), 1);
    // Images and text now always flow through ordered request messages.
    assert!(
        request
            .messages
            .iter()
            .any(|msg| msg.kind == "image" && msg.image_idx == 0)
    );
    assert!(
        request
            .messages
            .iter()
            .any(|msg| msg.kind == "text" && msg.content.contains("explain this"))
    );
}

#[test]
fn standard_tool_calls_produce_effects_and_loop() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("read file")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

    // Respond with tool call
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![
                LlmOutputPart::Text {
                    text: "Let me read that.".to_string(),
                },
                LlmOutputPart::ToolCall {
                    call_id: "tc1".to_string(),
                    tool_name: "read_file".to_string(),
                    input_json: r#"{"path":"foo.txt"}"#.to_string(),
                },
            ],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    // Should have ToolCalls effect
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
    assert!(tool_effect.is_some());
    let (tool_id, call_id, tool_name, args) = tool_effect.unwrap();
    assert_eq!(args, serde_json::json!({"path":"foo.txt"}));

    // Feed tool result
    machine.handle_response(Response::ToolResults {
        id: tool_id,
        results: vec![CompletedToolCall {
            call_id,
            tool_name,
            args,
            state_result: crate::ToolResult::ok(serde_json::json!("file contents")),
            model_result: crate::ToolResult::ok(serde_json::json!("file contents")),
            duration_ms: 10,
        }],
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    let llm_id2 = find_llm_call(&effects);
    assert!(
        llm_id2.is_some(),
        "should emit another LlmCall after tool results"
    );

    // Respond with prose to end
    machine.handle_response(Response::LlmComplete {
        id: *llm_id2.unwrap(),
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Done.".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Done.".to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
}

#[test]
fn standard_tool_results_preserve_original_args() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("what time is it")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "tc1".to_string(),
                tool_name: "exec_command".to_string(),
                input_json: r#"{"cmd":"date","workdir":"/tmp"}"#.to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (tool_id, call_id, tool_name, args) = effects
        .iter()
        .find_map(|e| match e {
            Effect::ToolCalls { id, calls } => calls.first().map(|call| {
                (
                    *id,
                    call.call_id.clone(),
                    call.tool_name.clone(),
                    call.args.clone(),
                )
            }),
            _ => None,
        })
        .expect("should emit ToolCalls effect");
    assert_eq!(args, serde_json::json!({"cmd":"date","workdir":"/tmp"}));

    machine.handle_response(Response::ToolResults {
        id: tool_id,
        results: vec![CompletedToolCall {
            call_id,
            tool_name,
            args: args.clone(),
            state_result: crate::ToolResult::ok(serde_json::json!({
                "output": "ok",
                "exit_code": 0,
                "timed_out": false,
                "duration_ms": 1
            })),
            model_result: crate::ToolResult::ok(serde_json::json!({
                "output": "ok",
                "exit_code": 0,
                "timed_out": false,
                "duration_ms": 1
            })),
            duration_ms: 1,
        }],
    });

    let effects = drain_effects(&mut machine);
    let tool_event = effects
        .iter()
        .find_map(|e| match e {
            Effect::Emit(SessionEvent::ToolCall { args, .. }) => Some(args.clone()),
            _ => None,
        })
        .expect("should emit ToolCall event");
    assert_eq!(
        tool_event,
        serde_json::json!({"cmd":"date","workdir":"/tmp"})
    );
}

#[test]
fn standard_retryable_error_sleeps_then_retries() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).unwrap();

    // Retryable error
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Err(LlmCallError {
            message: "rate limited".to_string(),
            retryable: true,
            raw: None,
            code: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    let retry = find_retry_status(&effects).expect("retry status");
    assert_eq!(retry.1, 2);
    assert_eq!(retry.2, LLM_MAX_RETRIES + 1);
    let envelope = find_retry_envelope(&effects).expect("retry envelope");
    assert_eq!(envelope.kind, "llm_provider");
    assert_eq!(envelope.user_message, "LLM error: rate limited");
    let sleep_effect = effects.iter().find_map(|e| match e {
        Effect::Sleep { id, .. } => Some(*id),
        _ => None,
    });
    assert!(sleep_effect.is_some(), "should emit Sleep for retry");

    // Feed timeout
    machine.handle_response(Response::Timeout {
        id: sleep_effect.unwrap(),
    });

    // Should get new LlmCall
    let effects = drain_effects(&mut machine);
    assert!(find_retry_status(&effects).is_none());
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn standard_retryable_error_exhaustion_emits_error_and_done() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let mut effects = drain_effects(&mut machine);
    let mut llm_id = *find_llm_call(&effects).expect("initial llm call");

    for expected_attempt in 2..=(LLM_MAX_RETRIES + 1) {
        machine.handle_response(Response::LlmComplete {
            id: llm_id,
            text_streamed: false,
            result: Err(LlmCallError {
                message: "provider unavailable".to_string(),
                retryable: true,
                raw: None,
                code: Some("http_500".to_string()),
            }),
        });

        effects = drain_effects(&mut machine);

        let retry = find_retry_status(&effects).expect("retry status");
        assert_eq!(retry.1, expected_attempt);
        let envelope = find_retry_envelope(&effects).expect("retry envelope");
        assert_eq!(envelope.code.as_deref(), Some("http_500"));
        let sleep_id = effects
            .iter()
            .find_map(|e| match e {
                Effect::Sleep { id, .. } => Some(*id),
                _ => None,
            })
            .expect("sleep effect");
        machine.handle_response(Response::Timeout { id: sleep_id });
        effects = drain_effects(&mut machine);
        llm_id = *find_llm_call(&effects).expect("retried llm call");
    }

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Err(LlmCallError {
            message: "provider unavailable".to_string(),
            retryable: true,
            raw: None,
            code: Some("http_500".to_string()),
        }),
    });

    effects = drain_effects(&mut machine);
    assert!(has_error_event(&effects, "LLM error: provider unavailable"));
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

#[test]
fn standard_fatal_error_emits_done() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).unwrap();

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Err(LlmCallError {
            message: "auth failed".to_string(),
            retryable: false,
            raw: None,
            code: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

#[test]
fn standard_empty_response_emits_error() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).unwrap();

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse::default()),
    });

    let effects = drain_effects(&mut machine);
    let has_error = effects
        .iter()
        .any(|e| matches!(e, Effect::Emit(SessionEvent::Error { .. })));
    assert!(has_error);
    assert!(find_done(&effects).is_some());
}

#[test]
fn standard_max_turns_stops_iteration() {
    let mut config = test_config(ExecutionMode::Standard);
    config.max_turns = Some(1);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).unwrap();

    // Respond with tool call to trigger iteration
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "tc1".to_string(),
                tool_name: "test".to_string(),
                input_json: "{}".to_string(),
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
        .unwrap();

    machine.handle_response(Response::ToolResults {
        id: tool_id,
        results: vec![CompletedToolCall {
            call_id: "tc1".to_string(),
            tool_name: "test".to_string(),
            args: serde_json::json!({}),
            state_result: crate::ToolResult::ok(serde_json::json!("ok")),
            model_result: crate::ToolResult::ok(serde_json::json!("ok")),
            duration_ms: 1,
        }],
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

// ─── RLM tests ───

#[test]
fn rlm_prose_only_response_emits_done() {
    let config = test_config(ExecutionMode::Rlm);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).unwrap();

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello there!".to_string(),
            deltas: vec!["Hello there!".to_string()],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

#[test]
fn standard_checkpoint_messages_continue_turn_before_completion() {
    let config = test_config(ExecutionMode::Standard);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("should emit LlmCall");

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Hello there!".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello there!".to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);

    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: vec![PluginMessage::text(MessageRole::User, "one more thing")],
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
    assert!(machine.messages().iter().any(|message| {
        message.role == MessageRole::User
            && message
                .parts
                .iter()
                .any(|part| part.content == "one more thing")
    }));
}

#[test]
fn rlm_fenced_lashlang_block_runs_exec_and_continues() {
    let config = test_config(ExecutionMode::Rlm);
    let msgs = vec![user_message("run some code")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    let request = find_llm_request(&effects).expect("request");
    assert!(
        request.tools.is_empty(),
        "RLM mode no longer advertises native tool specs",
    );

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Quick check.\n\n```lashlang\nobserve \"hi\"\n```\n".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Quick check.\n\n```lashlang\nobserve \"hi\"\n```\n".to_string(),
            }],
            usage: LlmUsage {
                input_tokens: 100,
                output_tokens: 50,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
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
        Some("observe \"hi\"")
    );

    machine.handle_response(Response::ExecResult {
        id: exec_effect.expect("exec").0,
        result: Ok(crate::ExecResponse {
            output: "hi\n".to_string(),
            observations: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn rlm_prose_after_exec_finishes_turn() {
    let config = test_config(ExecutionMode::Rlm);
    let msgs = vec![user_message("run code then summarize")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).unwrap();

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "Let me check.\n\n```lashlang\nobserve \"hi\"\n```\n".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Let me check.\n\n```lashlang\nobserve \"hi\"\n```\n".to_string(),
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
        result: Ok(crate::ExecResponse {
            output: "hi\n".to_string(),
            observations: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            error: None,
            duration_ms: 5,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    let next_llm_id = *find_llm_call(&effects).expect("next llm call");
    machine.handle_response(Response::LlmComplete {
        id: next_llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "All done.".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "All done.".to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("completion checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_takes_first_fenced_block_when_multiple_present() {
    let config = test_config(ExecutionMode::Rlm);
    let msgs = vec![user_message("run some code")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).unwrap();

    let response_text = "Step 1.\n\n```lashlang\nobserve \"first\"\n```\n\nStep 2.\n\n```lashlang\nobserve \"second\"\n```\n";
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: response_text.to_string(),
            parts: vec![LlmOutputPart::Text {
                text: response_text.to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let exec = effects
        .iter()
        .find_map(|e| match e {
            Effect::ExecCode { code, .. } => Some(code.clone()),
            _ => None,
        })
        .expect("exec effect");
    assert_eq!(exec, "observe \"first\"");
}

#[test]
fn rlm_debug_log_preserves_text_only_responses() {
    let mut config = test_config(ExecutionMode::Rlm);
    config.emit_llm_debug_log = true;
    let msgs = vec![user_message("run code")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    let response_text = "Working on it.\n\n```lashlang\nobserve \"hi\"\n```\n";
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: response_text.to_string(),
            parts: vec![LlmOutputPart::Text {
                text: response_text.to_string(),
            }],
            usage: LlmUsage {
                input_tokens: 321,
                output_tokens: 123,
                cached_input_tokens: 45,
                reasoning_tokens: 67,
            },
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (debug_usage, debug_text, _debug_parts) = find_llm_debug(&effects).expect("llm debug");
    assert_eq!(debug_usage.input_tokens, 321);
    assert_eq!(debug_usage.output_tokens, 123);
    assert_eq!(debug_usage.cached_input_tokens, 45);
    assert_eq!(debug_usage.reasoning_tokens, 67);
    assert_eq!(debug_text, response_text);
}

#[test]
fn llm_debug_log_preserves_tool_call_only_responses() {
    let mut config = test_config(ExecutionMode::Standard);
    config.emit_llm_debug_log = true;
    let msgs = vec![user_message("call a tool")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");

    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "tc1".to_string(),
                tool_name: "read_file".to_string(),
                input_json: r#"{"path":"foo.txt"}"#.to_string(),
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (usage, response_text, response_parts) = find_llm_debug(&effects).expect("llm debug");
    assert_eq!(usage.total(), 0);
    assert!(response_text.is_empty());
    assert_eq!(
        response_parts,
        Some(Value::Array(vec![serde_json::json!({
            "type": "tool_call",
            "call_id": "tc1",
            "tool_name": "read_file",
            "input_json": r#"{"path":"foo.txt"}"#,
        })]))
    );
}

#[test]
fn fence_extractor_handles_lashlang_and_repl_aliases() {
    let lashlang = "Plan.\n\n```lashlang\nfoo = 1\n```\n";
    let extracted = super::extract_first_lashlang_fence(lashlang).expect("fence");
    assert_eq!(extracted.code, "foo = 1");
    assert!(!extracted.had_extra_fences);

    let rlm_alias = "```rlm\nbar = 2\n```";
    let extracted = super::extract_first_lashlang_fence(rlm_alias).expect("fence");
    assert_eq!(extracted.code, "bar = 2");
}

#[test]
fn fence_extractor_returns_none_for_pure_prose() {
    assert!(super::extract_first_lashlang_fence("All done!").is_none());
    // Other languages don't count.
    assert!(super::extract_first_lashlang_fence("```python\nprint('x')\n```").is_none(),);
    // Inline triple-backticks in prose don't fool it either.
    assert!(super::extract_first_lashlang_fence("see ```lashlang inline``` here").is_none(),);
}

#[test]
fn fence_extractor_flags_extra_blocks() {
    let two_blocks = "first\n\n```lashlang\na = 1\n```\n\nbridge\n\n```lashlang\nb = 2\n```\n";
    let extracted = super::extract_first_lashlang_fence(two_blocks).expect("fence");
    assert_eq!(extracted.code, "a = 1");
    assert!(extracted.had_extra_fences);
}

#[test]
fn finish_value_validator_accepts_matching_record() {
    let value = serde_json::json!({"answer": "yes", "confidence": 0.92});
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["answer", "confidence"]
    });
    assert!(super::validate_finish_value(&value, &schema).is_ok());
}

#[test]
fn finish_value_validator_flags_missing_required_field() {
    let value = serde_json::json!({"answer": "yes"});
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["answer", "confidence"]
    });
    let err = super::validate_finish_value(&value, &schema).expect_err("missing field");
    assert!(err.contains("missing required field `confidence`"), "{err}");
}

#[test]
fn finish_value_validator_flags_type_mismatch() {
    let value = serde_json::json!({"answer": "yes", "confidence": "high"});
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["answer", "confidence"]
    });
    let err = super::validate_finish_value(&value, &schema).expect_err("type mismatch");
    assert!(err.contains("confidence"), "{err}");
    assert!(err.contains("number"), "{err}");
}

#[test]
fn finish_value_validator_recurses_into_array_items() {
    let value = serde_json::json!({"items": ["a", 2, "c"]});
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["items"]
    });
    let err = super::validate_finish_value(&value, &schema).expect_err("inner type mismatch");
    assert!(err.contains("items[1]"), "{err}");
}

#[test]
fn fence_extractor_takes_unterminated_remainder_when_no_close() {
    let unterminated = "loading...\n\n```lashlang\nx = 1\nobserve x\n";
    let extracted = super::extract_first_lashlang_fence(unterminated).expect("fence");
    assert_eq!(extracted.code, "x = 1\nobserve x");
    assert!(!extracted.had_extra_fences);
}
