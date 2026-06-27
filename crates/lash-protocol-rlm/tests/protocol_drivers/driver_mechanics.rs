use super::support::*;

// Focused RLM driver mechanics: malformed options, driver-state ownership, and checkpoint restore internals.

// === Focused RLM White-Box Tests ===
//
// These keep direct `TurnMachine` access because they validate malformed turn
// options, driver-state ownership, and checkpoint restore internals rather
// than reusable protocol scenario behavior.
#[test]
fn malformed_rlm_turn_options_fail_before_llm() {
    let config = test_config_with_protocol_turn_options(
        lash_core::ProtocolTurnOptions::from_payload(serde_json::json!({
            "termination": { "kind": "unknown" }
        })),
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
        lash_core::ProtocolTurnOptions::from_payload(serde_json::Value::Null),
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
    let config = test_config();
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
    let mut restored = TurnMachine::restore_from_checkpoint(test_config(), checkpoint);

    let effects = drain_effects(&mut restored);
    let llm_id = *find_llm_call(&effects).expect("restored llm call");
    restored.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: lashlang_block("print \"hi\""),
            parts: vec![LlmOutputPart::Text {
                text: lashlang_block("print \"hi\""),
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
    let config = test_config();
    let msgs = vec![user_message("run some code")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: lashlang_block_with_prose("Reason first.", "print \"hi\""),
            parts: vec![LlmOutputPart::Text {
                text: lashlang_block_with_prose("Reason first.", "print \"hi\""),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    });

    let effects = drain_effects(&mut machine);
    let (exec_id, code) = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, code, .. } => Some((*id, code.clone())),
            _ => None,
        })
        .expect("exec effect");
    assert_eq!(code, "print \"hi\"");

    let checkpoint = roundtrip_turn_checkpoint(machine.checkpoint());
    let mut restored = TurnMachine::restore_from_checkpoint(test_config(), checkpoint);
    let effects = drain_effects(&mut restored);
    let (restored_exec_id, restored_code) = effects
        .iter()
        .find_map(|effect| match effect {
            Effect::ExecCode { id, code, .. } => Some((*id, code.clone())),
            _ => None,
        })
        .expect("restored exec effect");
    assert_eq!(restored_exec_id, exec_id);
    assert_eq!(restored_code, "print \"hi\"");

    restored.handle_response(Response::ExecResult {
        id: restored_exec_id,
        result: Ok(lash_sansio::ExecResponse {
            observations: vec!["hi\n".to_string()],
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
    assert_eq!(assistant_visible_texts(&restored), vec!["Reason first."]);
    assert_eq!(entry.output, vec!["hi\n".to_string()]);
    let (_, checkpoint) = find_checkpoint(&effects).expect("after-work checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
}

#[test]
fn rlm_checkpoint_after_exec_fanout_tool_outputs_preserves_structured_outcomes() {
    let config = test_config();
    let msgs = vec![user_message("run fanout tools")];
    let mut machine = TurnMachine::new(config, msgs, Arc::new(Vec::new()), 0);

    let effects = drain_effects(&mut machine);
    let llm_id = *find_llm_call(&effects).expect("llm call");
    machine.handle_response(Response::LlmComplete {
        id: llm_id,
        text_streamed: false,
        result: Ok(LlmResponse {
            full_text: lashlang_block("ok = await tools.ok({})\nfail = await tools.fail({})\nstop = await tools.stop({})\nresults = { a: ok, b: fail, c: stop }"),
            parts: vec![LlmOutputPart::Text {
                text: lashlang_block("ok = await tools.ok({})\nfail = await tools.fail({})\nstop = await tools.stop({})\nresults = { a: ok, b: fail, c: stop }"),
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
    let mut restored = TurnMachine::restore_from_checkpoint(test_config(), checkpoint);
    let effects = drain_effects(&mut restored);
    assert!(
        !effects
            .iter()
            .any(|effect| matches!(effect, Effect::Emit(SessionEvent::ToolCall { .. })))
    );
    let trajectory = machine_trajectory(&restored);
    let entry = trajectory.last().expect("rlm trajectory entry");
    assert!(
        serde_json::to_value(entry)
            .unwrap()
            .get("tool_call_ids")
            .is_none()
    );
    assert_eq!(entry.output, vec!["fanout done".to_string()]);
    let (_, checkpoint) = find_checkpoint(&effects).expect("after-work checkpoint");
    assert_eq!(checkpoint, CheckpointKind::AfterWork);
}
