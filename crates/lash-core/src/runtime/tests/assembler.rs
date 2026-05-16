use super::*;

#[test]
fn assembler_uses_last_semantic_assistant_prose_group() {
    let mut assembler = TurnAssembler::default();
    assembler.push_turn_activity(&TurnActivity::new(
        TurnActivityId::new("assistant:first"),
        TurnEvent::AssistantProseDelta {
            text: "first".to_string(),
        },
    ));
    assembler.push_turn_activity(&TurnActivity::new(
        TurnActivityId::new("assistant:second"),
        TurnEvent::AssistantProseDelta {
            text: "second".to_string(),
        },
    ));
    assembler.push(&SessionEvent::Done);

    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );

    assert_eq!(
        out.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage {
            text: "second".to_string()
        })
    );
    assert_eq!(out.assistant_output.safe_text, "second");
}

#[test]
fn assembler_coalesces_semantic_assistant_prose_with_same_correlation_id() {
    let mut assembler = TurnAssembler::default();
    let correlation_id = TurnActivityId::new("assistant:one");
    assembler.push_turn_activity(&TurnActivity::new(
        correlation_id.clone(),
        TurnEvent::AssistantProseDelta {
            text: "hel".to_string(),
        },
    ));
    assembler.push_turn_activity(&TurnActivity::new(
        correlation_id,
        TurnEvent::AssistantProseDelta {
            text: "lo".to_string(),
        },
    ));
    assembler.push(&SessionEvent::Done);

    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );

    assert_eq!(
        out.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage {
            text: "hello".to_string()
        })
    );
    assert_eq!(out.assistant_output.safe_text, "hello");
}

#[test]
fn assembler_rewrites_assistant_message_outcome_to_last_semantic_prose_group() {
    let mut assembler = TurnAssembler::default();
    assembler.push_turn_activity(&TurnActivity::new(
        TurnActivityId::new("assistant:first"),
        TurnEvent::AssistantProseDelta {
            text: "first".to_string(),
        },
    ));
    assembler.push_turn_activity(&TurnActivity::new(
        TurnActivityId::new("assistant:second"),
        TurnEvent::AssistantProseDelta {
            text: "second".to_string(),
        },
    ));
    assembler.push(&SessionEvent::TurnOutcome {
        outcome: TurnOutcome::Finished(TurnFinish::AssistantMessage {
            text: "first\n\nsecond".to_string(),
        }),
    });
    assembler.push(&SessionEvent::Done);

    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );

    assert_eq!(
        out.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage {
            text: "second".to_string()
        })
    );
}

#[test]
fn assembler_uses_submitted_value_for_assistant_output_when_semantic_prose_streamed() {
    let mut assembler = TurnAssembler::default();
    assembler.push_turn_activity(&TurnActivity::new(
        TurnActivityId::new("assistant:before-submit"),
        TurnEvent::AssistantProseDelta {
            text: "thinking before submit".to_string(),
        },
    ));
    assembler.push(&SessionEvent::TurnOutcome {
        outcome: TurnOutcome::Finished(TurnFinish::SubmittedValue {
            value: serde_json::json!({ "ok": true }),
        }),
    });
    assembler.push(&SessionEvent::Done);

    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );

    assert_eq!(
        out.outcome,
        TurnOutcome::Finished(TurnFinish::SubmittedValue {
            value: serde_json::json!({ "ok": true })
        })
    );
    assert_eq!(out.assistant_output.safe_text, "{\n  \"ok\": true\n}");
}

#[test]
fn assembler_keeps_tool_value_when_semantic_prose_streamed() {
    let mut assembler = TurnAssembler::default();
    assembler.push_turn_activity(&TurnActivity::new(
        TurnActivityId::new("assistant:before-tool"),
        TurnEvent::AssistantProseDelta {
            text: "thinking before tool".to_string(),
        },
    ));
    assembler.push(&SessionEvent::TurnOutcome {
        outcome: TurnOutcome::Finished(TurnFinish::ToolValue {
            tool_name: "finish".to_string(),
            value: serde_json::json!("done"),
        }),
    });
    assembler.push(&SessionEvent::Done);

    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );

    assert_eq!(
        out.outcome,
        TurnOutcome::Finished(TurnFinish::ToolValue {
            tool_name: "finish".to_string(),
            value: serde_json::json!("done")
        })
    );
}

#[test]
fn interrupted_assembler_does_not_finish_with_semantic_assistant_prose() {
    let mut assembler = TurnAssembler::default();
    assembler.push_turn_activity(&TurnActivity::new(
        TurnActivityId::new("assistant:partial"),
        TurnEvent::AssistantProseDelta {
            text: "partial answer".to_string(),
        },
    ));

    let out = assembler.finish(
        default_state().export_state(),
        true,
        None,
        &TerminationPolicy::default(),
    );

    assert_eq!(out.outcome, TurnOutcome::Stopped(TurnStop::Cancelled));
    assert_eq!(out.assistant_output.safe_text, "partial answer");
}

#[test]
fn assembler_falls_back_to_last_assistant_message_when_stream_output_is_empty() {
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Prose,
                content: "stored".to_string(),
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
        },
    );
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        state.export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(out.assistant_output.safe_text, "stored");
    assert_eq!(out.assistant_output.raw_text, "stored");
    assert_eq!(out.assistant_output.state, OutputState::Usable);
}

#[test]
fn interrupted_assembler_does_not_reuse_assistant_before_latest_user_message() {
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "a0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: "a0.p0".to_string(),
                kind: PartKind::Prose,
                content: "previous assistant answer".to_string(),
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
        },
    );
    append_message(
        &mut state,
        Message {
            id: "u1".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "u1.p0".to_string(),
                kind: PartKind::Text,
                content: "new prompt".to_string(),
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
        },
    );

    let out = TurnAssembler::default().finish(
        state.export_state(),
        true,
        None,
        &TerminationPolicy::default(),
    );

    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::Cancelled)
    ));
    assert!(out.assistant_output.safe_text.is_empty());
    assert!(out.assistant_output.raw_text.is_empty());
}

#[test]
fn assembler_prefers_state_output_when_streamed_text_is_a_truncated_prefix() {
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Prose,
                content: "You graduated with a degree in Business Administration.".to_string(),
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
        },
    );
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::TextDelta {
        content: "You graduated with a degree in Business".to_string(),
    });
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        state.export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );
    assert_eq!(
        out.assistant_output.safe_text,
        "You graduated with a degree in Business Administration."
    );
    assert_eq!(
        out.assistant_output.raw_text,
        "You graduated with a degree in Business Administration."
    );
    assert_eq!(out.assistant_output.state, OutputState::Usable);
}

#[test]
fn assembler_state_output_excludes_tool_call_payload() {
    // Regression: codex commits an assistant message containing a prose
    // part followed by a tool-call part whose `content` is the raw JSON
    // arguments. On interrupt the assembler falls back to the last
    // assistant message's parts; concatenating EVERY part's content
    // leaks the tool-call JSON into safe_text and the UI then renders it
    // as a literal AssistantText block. Only Text/Prose/Image parts
    // should appear in safe_text.
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![
                Part {
                    id: "m0.p0".to_string(),
                    kind: PartKind::Prose,
                    content: "Searching for the relevant code.".to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                },
                Part {
                    id: "m0.p1".to_string(),
                    kind: PartKind::ToolCall,
                    content:
                        "{\"tool_calls\":[{\"tool\":\"grep\",\"parameters\":{\"query\":\"x\"}}]}"
                            .to_string(),
                    attachment: None,
                    tool_call_id: Some("tc1".to_string()),
                    tool_name: Some("batch".to_string()),
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                },
            ]
            .into(),
            origin: None,
        },
    );
    let assembler = TurnAssembler::default();
    let out = assembler.finish(
        state.export_state(),
        true,
        None,
        &TerminationPolicy::default(),
    );
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::Cancelled)
    ));
    assert_eq!(
        out.assistant_output.safe_text,
        "Searching for the relevant code."
    );
    assert!(!out.assistant_output.raw_text.contains("tool_calls"));
}

#[test]
fn assembler_marks_tool_failure() {
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::ToolCall {
        call_id: Some("tc1".to_string()),
        name: "x".to_string(),
        args: serde_json::json!({}),
        output: crate::ToolCallOutput::failure(crate::ToolFailure::tool(
            crate::ToolFailureClass::Execution,
            "tool_error",
            serde_json::json!({"error": true}).to_string(),
        )),
        duration_ms: 1,
    });
    assembler.push(&SessionEvent::Error {
        message: "tool failed".to_string(),
        envelope: None,
    });
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );
    assert!(matches!(&out.outcome, TurnOutcome::Stopped(_)));
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::ToolFailure)
    ));
    assert_eq!(out.tool_calls.len(), 1);
}

#[test]
fn assembler_marks_missing_done_as_failure() {
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::TextDelta {
        content: "partial".to_string(),
    });
    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );
    assert!(matches!(&out.outcome, TurnOutcome::Stopped(_)));
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::RuntimeError)
    ));
}

#[test]
fn assembler_detects_max_turn_message() {
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::System,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Text,
                content: "Turn limit reached (5).".to_string(),
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
        },
    );
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        state.export_state(),
        false,
        None,
        &TerminationPolicy::default(),
    );
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::MaxTurns)
    ));
}

#[test]
fn output_state_empty_output() {
    assert_eq!(classify_output_state("", "", &[]), OutputState::EmptyOutput);
}

#[test]
fn output_state_traceback_only() {
    let raw = "Runtime error: Traceback (most recent call last):\nFile \"rlm_1.py\", line 2, in <module>\nNameError: name 'now' is not defined";
    assert_eq!(
        classify_output_state(raw, "", &[]),
        OutputState::TracebackOnly
    );
}

#[test]
fn output_state_recovered_from_error() {
    let issues = vec![TurnIssue {
        kind: "runtime".to_string(),
        code: Some("example".to_string()),
        terminal_reason: None,
        message: "something failed".to_string(),
        raw: None,
    }];
    assert_eq!(
        classify_output_state("raw", "usable", &issues),
        OutputState::RecoveredFromError
    );
}

#[test]
fn normalize_items_merges_adjacent_text_items() {
    let items = vec![
        InputItem::Text {
            text: "before ".to_string(),
        },
        InputItem::Text {
            text: "[file: host-prepared.txt]".to_string(),
        },
    ];
    let out = normalize_input_items(
        &items,
        &HashMap::new(),
        &crate::InMemoryAttachmentStore::new(),
    )
    .expect("normalized");
    assert_eq!(out.len(), 1);
    match &out[0] {
        NormalizedItem::Text(text) => {
            assert_eq!(text, "before [file: host-prepared.txt]");
        }
        _ => panic!("expected merged text item"),
    }
}
