use std::sync::Arc;

use lash::sansio::{self, ChatContextProjector, ProtocolDriverHandle, Response};
use lash::{Effect, ExecutionMode, TurnMachine, TurnMachineConfig};
use lash_mode_rlm::RlmDriver;
use lash_mode_standard::StandardDriver;
use lash_rlm_types::{RlmModeEvent, RlmTermination, RlmTrajectoryEntry};
use lash_sansio::llm::types::{LlmOutputPart, LlmRequest, LlmResponse};
use lash_sansio::{
    CheckpointKind, Message, MessageRole, Part, PartKind, PruneState, RetryPolicy, SessionEvent,
};

fn test_config(mode: ExecutionMode) -> TurnMachineConfig {
    test_config_with_termination(mode, RlmTermination::default())
}

fn test_config_with_termination(
    mode: ExecutionMode,
    rlm_termination: RlmTermination,
) -> TurnMachineConfig {
    let protocol_driver: Arc<dyn ProtocolDriverHandle<lash::HostModeProtocol>> = match &mode {
        mode if *mode == ExecutionMode::standard() => Arc::new(StandardDriver),
        mode if *mode == ExecutionMode::new("rlm") || *mode == ExecutionMode::new("rlmpure") => {
            Arc::new(RlmDriver)
        }
        _ => Arc::new(StandardDriver),
    };
    TurnMachineConfig {
        protocol_driver,
        projector: Arc::new(ChatContextProjector),
        sync_execution_surface: mode == ExecutionMode::new("rlm")
            || mode == ExecutionMode::new("rlmpure"),
        model: "test-model".to_string(),
        max_turns: None,
        model_variant: None,
        run_session_id: None,
        autonomous: false,
        tool_specs: Vec::new().into(),
        system_prompt: std::sync::Arc::new(String::new()),
        session_id: "test".to_string(),
        emit_llm_debug_log: false,
        termination: lash::ModeTurnOptions::rlm(rlm_termination),
        retry_policy: RetryPolicy::default(),
        initial_events: Arc::new(Vec::new()),
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
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
        }],
        user_input: None,
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
                    system_prompt: std::sync::Arc::new(String::new()),
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
    effects.iter().find_map(|e| match e {
        Effect::LlmCall { request, .. } => Some(request),
        _ => None,
    })
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
            events: _,
            iteration,
        } => Some((messages, *iteration)),
        _ => None,
    })
}

fn machine_trajectory(machine: &TurnMachine) -> Vec<RlmTrajectoryEntry> {
    machine
        .events()
        .iter()
        .filter_map(|event| match event {
            lash::SessionEventRecord::Mode(event) => match event.rlm_event() {
                Some(RlmModeEvent::RlmTrajectoryEntry(entry)) => Some(entry),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

#[test]
fn standard_prose_only_response_emits_done() {
    let config = test_config(ExecutionMode::standard());
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
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
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
    assert!(machine.is_done());
}

#[test]
fn standard_tool_calls_produce_effects_and_loop() {
    let config = test_config(ExecutionMode::standard());
    let msgs = vec![user_message("read file")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
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
                    item_id: None,
                    signature: None,
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
            call_id,
            tool_name,
            args,
            state_result: lash_sansio::ToolResult::ok(serde_json::json!("file contents")),
            model_result: lash_sansio::ToolResult::ok(serde_json::json!("file contents")),
            duration_ms: 10,
            item_id: None,
        }],
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn standard_empty_final_after_tool_result_finishes_without_error() {
    let config = test_config(ExecutionMode::standard());
    let msgs = vec![user_message("update the plan and do nothing else")];
    let mut machine = TurnMachine::new(config, msgs, 0);

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
                item_id: None,
                signature: None,
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
            state_result: lash_sansio::ToolResult::ok(serde_json::json!("Plan updated")),
            model_result: lash_sansio::ToolResult::ok(serde_json::json!("Plan updated")),
            duration_ms: 1,
            item_id: None,
        }],
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("after-work checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
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
        messages: Vec::new(),
        transient_messages: Vec::new(),
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
    let mut config = test_config(ExecutionMode::standard());
    config.max_turns = Some(1);
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

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
                item_id: None,
                signature: None,
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
            state_result: lash_sansio::ToolResult::ok(serde_json::json!("ok")),
            model_result: lash_sansio::ToolResult::ok(serde_json::json!("ok")),
            duration_ms: 1,
            item_id: None,
        }],
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_prose_only_response_emits_done() {
    let config = test_config(ExecutionMode::new("rlm"));
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

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
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
}

#[test]
fn typed_rlm_prose_only_response_requests_submit() {
    let config = test_config_with_termination(
        ExecutionMode::new("rlm"),
        RlmTermination::Finish { schema: None },
    );
    let msgs = vec![user_message("hello")];
    let mut machine = TurnMachine::new(config, msgs, 0);

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
        message.role == MessageRole::User
            && message.parts.iter().any(|part| {
                part.content.contains("output schema is required")
                    && part.content.contains("submit")
            })
    }));
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn rlm_fenced_lashlang_block_runs_exec_and_continues() {
    let config = test_config(ExecutionMode::new("rlm"));
    let msgs = vec![user_message("run some code")];
    let mut machine = TurnMachine::new(config, msgs, 0);

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
            tool_calls: Vec::new(),
            images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: None,
        }),
    });

    let effects = drain_effects(&mut machine);
    let trajectory = machine_trajectory(&machine);
    let entry = trajectory.last().expect("rlm trajectory entry");
    assert_eq!(entry.code, "print \"hi\"");
    assert_eq!(entry.output, "hi\n");
    assert!(entry.final_output.is_none());

    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}

#[test]
fn typed_rlm_finish_emits_typed_finish_and_done() {
    let config = test_config_with_termination(
        ExecutionMode::new("rlm"),
        RlmTermination::Finish { schema: None },
    );
    let msgs = vec![user_message("return typed data")];
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nsubmit { ok: true }\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nsubmit { ok: true }\n```".to_string(),
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
            tool_calls: Vec::new(),
            images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!({ "ok": true })),
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(effects.iter().any(|e| matches!(
        e,
        Effect::Emit(lash_sansio::SessionEvent::Message { text, kind })
            if text.contains("\"ok\": true") && kind == "final"
    )));
    assert!(effects.iter().any(|e| matches!(
        e,
        Effect::Emit(lash_sansio::SessionEvent::TypedFinish { value })
            if *value == serde_json::json!({ "ok": true })
    )));
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::BeforeCompletion);
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_done(&effects).is_some());
}

#[test]
fn rlm_reasoning_part_is_preserved_in_trajectory() {
    let config = test_config_with_termination(
        ExecutionMode::new("rlmpure"),
        RlmTermination::Finish { schema: None },
    );
    let msgs = vec![user_message("say hi")];
    let mut machine = TurnMachine::new(config, msgs, 0);

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
                    signature: None,
                    redacted: false,
                    item_id: None,
                    encrypted_content: None,
                    summary: Vec::new(),
                },
                LlmOutputPart::Text {
                    text: "```lashlang\nsubmit \"Hi.\"\n```".to_string(),
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
            tool_calls: Vec::new(),
            images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!("Hi.")),
        }),
    });

    let effects = drain_effects(&mut machine);
    assert!(effects.iter().any(|effect| {
        matches!(
            effect,
            Effect::Emit(lash_sansio::SessionEvent::Message { text, kind })
                if text == "Hi." && kind == "final"
        )
    }));
    let trajectory = machine_trajectory(&machine);
    let entry = trajectory.last().expect("trajectory entry");
    assert!(entry.reasoning.contains("I'll answer directly."));
    assert!(entry.reasoning.contains("```lashlang"));
    assert_eq!(entry.final_output, Some(serde_json::json!("Hi.")));
}

#[test]
fn typed_rlm_schema_mismatch_loops_with_feedback() {
    let config = test_config_with_termination(
        ExecutionMode::new("rlm"),
        RlmTermination::Finish {
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
    let mut machine = TurnMachine::new(config, msgs, 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: "```lashlang\nsubmit { missing: true }\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nsubmit { missing: true }\n```".to_string(),
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
            tool_calls: Vec::new(),
            images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!({ "missing": true })),
        }),
    });

    let effects = drain_effects(&mut machine);
    let (checkpoint_id, checkpoint) = find_checkpoint(&effects).expect("checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
    assert!(machine.messages().iter().any(|message| {
        message.role == MessageRole::User
            && message.parts.iter().any(|part| {
                part.content
                    .contains("didn't match the required output schema")
            })
    }));
    machine.handle_response(Response::Checkpoint {
        id: checkpoint_id,
        messages: Vec::new(),
        transient_messages: Vec::new(),
    });

    let effects = drain_effects(&mut machine);
    assert!(find_llm_call(&effects).is_some());
}
