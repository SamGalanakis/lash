use super::*;

#[test]
fn finish_turn_from_read_view_rebuilds_current_turn_from_authoritative_state() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline
        .push(UiTimelineItem::SystemMessage("Local note".into()));
    let turn = PreparedTurn::new("What exists now?".into(), Vec::new());
    app.push_prepared_user_input(&turn);
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "I looked at the actual librarian prompt".into(),
    });

    let messages = vec![
        text_message("u1", MessageRole::User, "What exists now?"),
        text_message(
            "a1",
            MessageRole::Assistant,
            "I looked at the actual librarian prompt, the graph tool constraints.\n\n## What exists now",
        ),
    ];
    let events = events_from_messages(&messages);
    app.finish_turn_from_read_view(&test_read_view(&events, &messages, &[]));

    assert!(!app.running);
    assert!(app.timeline.iter().any(|block| {
        matches!(block, UiTimelineItem::SystemMessage(text) if text == "Local note")
    }));
    let last_block = app
        .timeline
        .iter()
        .rev()
        .find_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .expect("assistant block");
    assert_eq!(
        last_block,
        "I looked at the actual librarian prompt, the graph tool constraints.\n\n## What exists now"
    );
}

#[test]
fn finish_turn_from_read_view_preserves_projected_turns_after_repeated_input() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let first_turn = PreparedTurn::new("hi".into(), Vec::new());
    app.push_prepared_user_input(&first_turn);
    app.timeline
        .push(UiTimelineItem::AssistantText("Hi.".into()));
    app.start_turn();

    let messages = vec![
        text_message("u1", MessageRole::User, "hi"),
        text_message("a1", MessageRole::Assistant, "Hi."),
        text_message("u2", MessageRole::User, "hi"),
        text_message("a2", MessageRole::Assistant, "Hi."),
        text_message("u3", MessageRole::User, "hi"),
        text_message("a3", MessageRole::Assistant, "Hi."),
    ];
    let events = events_from_messages(&messages);

    app.finish_turn_from_read_view(&test_read_view(&events, &messages, &[]));

    let user_inputs = app
        .timeline
        .iter()
        .filter(|block| matches!(block, UiTimelineItem::UserInput(_)))
        .count();
    let assistant_texts = app
        .timeline
        .iter()
        .filter(|block| matches!(block, UiTimelineItem::AssistantText(_)))
        .count();
    assert_eq!(user_inputs, 3);
    assert_eq!(assistant_texts, 3);
}

#[test]
fn finish_turn_from_read_view_does_not_duplicate_assistant_text_after_tool_activity() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("Fix it".into(), Vec::new());
    app.push_prepared_user_input(&turn);
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "I found and fixed the 500.".into(),
    });
    app.handle_session_event(SessionEvent::ToolCall {
        name: "update_plan".into(),
        args: serde_json::json!({
            "plan": [
                {"step": "Inspect proxy", "status": "completed"},
                {"step": "Patch request body handling", "status": "completed"},
                {"step": "Verify auth fallback", "status": "completed"},
            ]
        }),
        output: lash_core::ToolCallOutput::success(serde_json::json!({"ok": true})),
        duration_ms: 12,
        call_id: Some("tc-plan".into()),
    });

    let messages = vec![
        text_message("u1", MessageRole::User, "Fix it"),
        text_message("a1", MessageRole::Assistant, "I found and fixed the 500."),
    ];
    let tool_calls = vec![ToolCallRecord {
        call_id: Some("tc-plan".into()),
        tool: "update_plan".into(),
        args: serde_json::json!({
            "plan": [
                {"step": "Inspect proxy", "status": "completed"},
                {"step": "Patch request body handling", "status": "completed"},
                {"step": "Verify auth fallback", "status": "completed"},
            ]
        }),
        output: lash_core::ToolCallOutput::success(serde_json::json!({"ok": true})),
        duration_ms: 12,
    }];
    let events = events_from_messages(&messages);
    app.finish_turn_from_read_view(&test_read_view(&events, &messages, &tool_calls));

    let assistant_texts: Vec<&str> = app
        .timeline
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(assistant_texts, vec!["I found and fixed the 500."]);
}

#[test]
fn finish_turn_from_read_view_uses_authoritative_reasoning_and_text() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("Write a poem".into(), Vec::new());
    app.push_prepared_user_input(&turn);
    app.start_turn();
    app.handle_session_event(SessionEvent::ReasoningDelta {
        content: "Crafting a cool poem".into(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: "Neon rain on midnight street.".into(),
    });
    app.handle_session_event(SessionEvent::ReasoningDelta {
        content: "I see the user wants a cool poem.".into(),
    });

    let message = Message {
        id: "a1".into(),
        role: MessageRole::Assistant,
        parts: vec![
            part(
                "a1.r",
                PartKind::Reasoning,
                "Crafting a cool poem\n\nI see the user wants a cool poem.",
            ),
            part("a1.t", PartKind::Text, "Neon rain on midnight street."),
        ]
        .into(),
        origin: None,
    };
    let messages = vec![
        text_message("u1", MessageRole::User, "Write a poem"),
        message,
    ];
    let events = events_from_messages(&messages);
    app.finish_turn_from_read_view(&test_read_view(&events, &messages, &[]));

    let assistant_texts: Vec<&str> = app
        .timeline
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(assistant_texts, vec!["Neon rain on midnight street."]);

    let reasoning_blocks: Vec<&str> = app
        .timeline
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::AssistantReasoning(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning_blocks.len(), 1);
    assert!(reasoning_blocks[0].contains("Crafting a cool poem"));
    assert!(reasoning_blocks[0].contains("I see the user wants a cool poem."));
}

#[test]
fn read_view_timeline_places_reasoning_before_text() {
    let message = Message {
        id: "a1".into(),
        role: MessageRole::Assistant,
        parts: vec![
            part("a1.t", PartKind::Text, "Visible answer."),
            part("a1.r", PartKind::Reasoning, "Late summary."),
        ]
        .into(),
        origin: None,
    };

    let events = events_from_messages(std::slice::from_ref(&message));
    let blocks =
        timeline_items_from_test_read_view(&events, &[message], &[], &UiProjectionState::default());
    let variants: Vec<&str> = blocks.iter().map(other_variant_name).collect();
    assert_eq!(variants, vec!["AssistantReasoning", "AssistantText"]);
}

#[test]
fn read_view_timeline_round_trips_to_existing_display_blocks() {
    let user = text_message("u1", MessageRole::User, "Summarize this");
    let assistant = text_message("a1", MessageRole::Assistant, "Summary.");
    let events = events_from_messages(&[user.clone(), assistant.clone()]);
    let ui_state = UiProjectionState {
        live_reasoning_text: Some("Checking final wording.".to_string()),
        live_assistant_text: Some("Summary.\n\nOne more sentence.".to_string()),
        ..UiProjectionState::default()
    };

    let timeline = timeline_from_test_read_view(&events, &[user, assistant], &[], &ui_state);
    let blocks = timeline_items_from_test_read_view(
        &events,
        &[
            text_message("u1", MessageRole::User, "Summarize this"),
            text_message("a1", MessageRole::Assistant, "Summary."),
        ],
        &[],
        &ui_state,
    );

    assert_eq!(timeline.items(), blocks.as_slice());
    let variants = timeline
        .items()
        .iter()
        .map(|item| match item {
            projection::UiTimelineItem::TurnStart(_) => "TurnStart",
            projection::UiTimelineItem::UserInput(_) => "UserInput",
            projection::UiTimelineItem::AssistantText(_) => "AssistantText",
            projection::UiTimelineItem::AssistantReasoning(_) => "AssistantReasoning",
            projection::UiTimelineItem::Activity(_) => "Activity",
            projection::UiTimelineItem::ShellOutput { .. } => "ShellOutput",
            projection::UiTimelineItem::Error(_) => "Error",
            projection::UiTimelineItem::SystemMessage(_) => "SystemMessage",
            projection::UiTimelineItem::PluginPanel(_) => "PluginPanel",
            projection::UiTimelineItem::LashlangCode(_) => "LashlangCode",
            projection::UiTimelineItem::Splash => "Splash",
        })
        .collect::<Vec<_>>();
    assert_eq!(
        variants,
        vec![
            "TurnStart",
            "UserInput",
            "AssistantText",
            "AssistantReasoning",
            "AssistantText",
        ]
    );
}

#[test]
fn rlm_trajectory_reasoning_projects_as_assistant_reasoning() {
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "I'll reply directly.\n\n```lashlang\nsubmit \"hi\"\n```".to_string(),
        code: "submit \"hi\"".to_string(),
        output: Vec::new(),
        tool_call_ids: Vec::new(),
        images: Vec::new(),
        error: None,
        final_output: None,
    };
    let events = vec![lash_core::SessionEventRecord::Mode(
        lash_mode_rlm::rlm_mode_event(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry)),
    )];

    let blocks =
        timeline_items_from_test_read_view(&events, &[], &[], &UiProjectionState::default());
    let variants: Vec<&str> = blocks.iter().map(other_variant_name).collect();
    assert_eq!(variants, vec!["AssistantReasoning", "LashlangCode"]);

    let reasoning = blocks.iter().find_map(|block| match block {
        UiTimelineItem::AssistantReasoning(text) => Some(text.as_str()),
        _ => None,
    });
    assert_eq!(reasoning, Some("I'll reply directly."));
}

#[test]
fn rlm_trajectory_final_output_projects_as_assistant_text() {
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "I'll reply directly.\n\n```lashlang\nsubmit \"Hi!\"\n```".to_string(),
        code: "submit \"Hi!\"".to_string(),
        output: Vec::new(),
        tool_call_ids: Vec::new(),
        images: Vec::new(),
        error: None,
        final_output: Some(serde_json::json!("Hi!")),
    };
    let events = vec![lash_core::SessionEventRecord::Mode(
        lash_mode_rlm::rlm_mode_event(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry)),
    )];

    let blocks =
        timeline_items_from_test_read_view(&events, &[], &[], &UiProjectionState::default());
    let variants: Vec<&str> = blocks.iter().map(other_variant_name).collect();
    assert_eq!(
        variants,
        vec!["AssistantReasoning", "LashlangCode", "AssistantText"]
    );
    let answer = blocks.iter().find_map(|block| match block {
        UiTimelineItem::AssistantText(text) => Some(text.as_str()),
        _ => None,
    });
    assert_eq!(answer, Some("Hi!"));
}

#[test]
fn rlm_final_answer_projects_after_reasoning_and_lashlang_code() {
    let user = text_message("u1", MessageRole::User, "hi");
    let assistant = plugin_text_message("a1", MessageRole::Assistant, "mode_rlm", "Hi!");
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "I'll answer directly.\n\n```lashlang\nsubmit \"Hi!\"\n```".to_string(),
        code: "submit \"Hi!\"".to_string(),
        output: Vec::new(),
        tool_call_ids: Vec::new(),
        images: Vec::new(),
        error: None,
        final_output: Some(serde_json::json!("Hi!")),
    };
    let events = vec![
        conversation_event(user.clone()),
        lash_core::SessionEventRecord::Mode(lash_mode_rlm::rlm_mode_event(
            lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry),
        )),
        conversation_event(assistant.clone()),
    ];

    let blocks = timeline_items_from_test_read_view(
        &events,
        &[user, assistant],
        &[],
        &UiProjectionState::default(),
    );
    let variants: Vec<&str> = blocks.iter().map(other_variant_name).collect();
    assert_eq!(
        variants,
        vec![
            "TurnStart",
            "UserInput",
            "AssistantReasoning",
            "LashlangCode",
            "AssistantText",
        ]
    );
    let assistant_texts: Vec<&str> = blocks
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(assistant_texts, vec!["Hi!"]);
}

#[test]
fn finish_turn_replaces_live_submitted_value_with_projection() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("What time is it".into(), Vec::new());
    app.push_prepared_user_input(&turn);
    app.start_turn();
    app.handle_turn_activity(TurnActivity::independent(TurnEvent::SubmittedValue {
        value: serde_json::json!("Current system time: Fri May 15 11:35:52 PM CEST 2026"),
    }));

    let user = text_message("u1", MessageRole::User, "What time is it");
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning:
            "```lashlang\nsubmit \"Current system time: Fri May 15 11:35:52 PM CEST 2026\"\n```"
                .to_string(),
        code: "submit \"Current system time: Fri May 15 11:35:52 PM CEST 2026\"".to_string(),
        output: Vec::new(),
        tool_call_ids: Vec::new(),
        images: Vec::new(),
        error: None,
        final_output: Some(serde_json::json!(
            "Current system time: Fri May 15 11:35:52 PM CEST 2026"
        )),
    };
    let events = vec![
        conversation_event(user.clone()),
        lash_core::SessionEventRecord::Mode(lash_mode_rlm::rlm_mode_event(
            lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry),
        )),
    ];

    app.finish_turn_from_read_view(&test_read_view(&events, &[user], &[]));

    let assistant_texts: Vec<&str> = app
        .timeline
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        assistant_texts,
        vec!["Current system time: Fri May 15 11:35:52 PM CEST 2026"]
    );
}

#[test]
fn submitted_value_turn_event_projects_as_assistant_text() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_turn_activity(TurnActivity::independent(TurnEvent::SubmittedValue {
        value: serde_json::json!("**done**"),
    }));

    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::AssistantText(text)) if text == "**done**"
    ));
}

#[test]
fn rlm_trajectory_projects_tool_calls_after_own_reasoning() {
    let record = ToolCallRecord {
        call_id: Some("call-date-utc".to_string()),
        tool: "exec_command".to_string(),
        args: serde_json::json!({ "cmd": "date -u" }),
        output: lash_core::ToolCallOutput::success(serde_json::json!({
            "output": "2026-04-25 20:05:57 UTC\n",
            "exit_code": 0
        })),
        duration_ms: 12,
    };
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "I'll inspect the environment.".to_string(),
        code: "now = (call exec_command { cmd: \"date -u\" })?\nprint now".to_string(),
        output: vec!["2026-04-25 20:05:57 UTC".to_string()],
        tool_call_ids: vec!["call-date-utc".to_string()],
        images: Vec::new(),
        error: None,
        final_output: None,
    };
    let events = vec![lash_core::SessionEventRecord::Mode(
        lash_mode_rlm::rlm_mode_event(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry)),
    )];

    let blocks =
        timeline_items_from_test_read_view(&events, &[], &[record], &UiProjectionState::default());
    let variants: Vec<&str> = blocks.iter().map(other_variant_name).collect();
    assert_eq!(
        variants,
        vec!["AssistantReasoning", "LashlangCode", "Activity"]
    );
    assert!(matches!(
        blocks.get(2),
        Some(UiTimelineItem::Activity(activity)) if activity.call.tool_name == "exec_command"
    ));
}

#[test]
fn rlm_trajectory_owns_matching_raw_tool_call_projection() {
    let record = ToolCallRecord {
        call_id: Some("call-date".to_string()),
        tool: "exec_command".to_string(),
        args: serde_json::json!({ "cmd": "date" }),
        output: lash_core::ToolCallOutput::success(serde_json::json!({
            "output": "Mon May 11 01:51:25 PM CEST 2026\n",
            "exit_code": 0
        })),
        duration_ms: 5,
    };
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "I'll check the system time.".to_string(),
        code: "now = (call exec_command { cmd: \"date\" })?\nsubmit trim(now.output)".to_string(),
        output: Vec::new(),
        tool_call_ids: vec!["call-date".to_string()],
        images: Vec::new(),
        error: None,
        final_output: Some(serde_json::json!("Mon May 11 01:51:25 PM CEST 2026")),
    };
    let events = vec![lash_core::SessionEventRecord::Mode(
        lash_mode_rlm::rlm_mode_event(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry)),
    )];

    let blocks =
        timeline_items_from_test_read_view(&events, &[], &[record], &UiProjectionState::default());
    let variants: Vec<&str> = blocks.iter().map(other_variant_name).collect();
    assert_eq!(
        variants,
        vec![
            "AssistantReasoning",
            "LashlangCode",
            "Activity",
            "AssistantText"
        ]
    );
    assert_eq!(
        blocks
            .iter()
            .filter(|block| matches!(block, UiTimelineItem::Activity(_)))
            .count(),
        1
    );
}

#[test]
fn rlm_trajectory_steps_project_chronologically_with_tool_results() {
    let first_record = ToolCallRecord {
        call_id: Some("call-date-utc".to_string()),
        tool: "exec_command".to_string(),
        args: serde_json::json!({ "cmd": "date -u" }),
        output: lash_core::ToolCallOutput::success(
            serde_json::json!({ "output": "time\n", "exit_code": 0 }),
        ),
        duration_ms: 3,
    };
    let first = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "First check the time.".to_string(),
        code: "now = (call exec_command { cmd: \"date -u\" })?\nprint now".to_string(),
        output: vec!["time".to_string()],
        tool_call_ids: vec!["call-date-utc".to_string()],
        images: Vec::new(),
        error: None,
        final_output: None,
    };
    let second_record = ToolCallRecord {
        call_id: Some("call-ls".to_string()),
        tool: "ls".to_string(),
        args: serde_json::json!({ "path": "." }),
        output: lash_core::ToolCallOutput::success(
            serde_json::json!({ "entries": ["Cargo.toml"] }),
        ),
        duration_ms: 4,
    };
    let second = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_1".to_string(),
        mode_iteration: 1,
        reasoning: "Then check files.".to_string(),
        code: "files = (call ls { path: \".\" })?\nprint files".to_string(),
        output: vec!["files".to_string()],
        tool_call_ids: vec!["call-ls".to_string()],
        images: Vec::new(),
        error: None,
        final_output: None,
    };
    let assistant = text_message("a1", MessageRole::Assistant, "Done.");
    let events = vec![
        lash_core::SessionEventRecord::Mode(lash_mode_rlm::rlm_mode_event(
            lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(first),
        )),
        lash_core::SessionEventRecord::Mode(lash_mode_rlm::rlm_mode_event(
            lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(second),
        )),
        conversation_event(assistant.clone()),
    ];

    let blocks = timeline_items_from_test_read_view(
        &events,
        &[assistant],
        &[first_record, second_record],
        &UiProjectionState::default(),
    );
    let variants: Vec<&str> = blocks.iter().map(other_variant_name).collect();
    assert_eq!(
        variants,
        vec![
            "AssistantReasoning",
            "LashlangCode",
            "Activity",
            "AssistantReasoning",
            "LashlangCode",
            "Activity",
            "AssistantText",
        ]
    );
}

#[test]
fn finish_turn_from_read_view_uses_authoritative_transcript_even_when_streamed_text_differs() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("Shorten it".into(), Vec::new());
    app.push_prepared_user_input(&turn);
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "Visible streamed text".into(),
    });

    let messages = vec![
        text_message("u1", MessageRole::User, "Shorten it"),
        text_message("a1", MessageRole::Assistant, "Visible"),
    ];
    let events = events_from_messages(&messages);
    app.finish_turn_from_read_view(&test_read_view(&events, &messages, &[]));

    let last_block = app
        .timeline
        .iter()
        .rev()
        .find_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .expect("assistant block");
    assert_eq!(last_block, "Visible");
}

#[test]
fn interrupted_read_view_preserves_partial_assistant_text() {
    let blocks = interrupted_blocks_from_test_read_view(
        &[],
        &[],
        &[],
        &UiProjectionState {
            live_assistant_text: Some("Partial streamed answer".to_string()),
            ..UiProjectionState::default()
        },
        "Cancelled.",
    );

    assert!(matches!(
        blocks.first(),
        Some(UiTimelineItem::AssistantText(text)) if text == "Partial streamed answer"
    ));
    assert!(matches!(
        blocks.last(),
        Some(UiTimelineItem::SystemMessage(msg)) if msg == "Cancelled."
    ));
}

#[test]
fn interrupted_read_view_does_not_duplicate_already_committed_prose() {
    // Codex emits multiple prose chunks intermixed with tool calls. Each
    // prose chunk is committed as a separate assistant message, and on
    // interrupt the runtime hands the UI the entire concatenation as
    // `live_assistant_text`. The projection must not re-render that
    // concat as an additional block on top of the already-rendered ones.
    let messages = vec![
        text_message("m0", MessageRole::User, "go"),
        text_message("m1", MessageRole::Assistant, "first prose"),
        text_message("m2", MessageRole::Assistant, "second prose"),
    ];
    let events = events_from_messages(&messages);
    let blocks = interrupted_blocks_from_test_read_view(
        &events,
        &messages,
        &[],
        &UiProjectionState {
            live_assistant_text: Some("first prose\n\nsecond prose".into()),
            ..UiProjectionState::default()
        },
        "Cancelled.",
    );

    let assistant_texts: Vec<&str> = blocks
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(assistant_texts, vec!["first prose", "second prose"]);
}

#[test]
fn interrupted_read_view_appends_only_uncommitted_tail() {
    // If the streamed text contains everything in the committed messages
    // PLUS a trailing chunk the model was mid-stream on when the abort
    // landed, only that trailing chunk should be appended as a new block.
    let messages = vec![text_message("m0", MessageRole::Assistant, "first prose")];
    let events = events_from_messages(&messages);
    let blocks = interrupted_blocks_from_test_read_view(
        &events,
        &messages,
        &[],
        &UiProjectionState {
            live_assistant_text: Some("first prose\n\nmid stream tail".into()),
            ..UiProjectionState::default()
        },
        "Cancelled.",
    );

    let assistant_texts: Vec<&str> = blocks
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(assistant_texts, vec!["first prose", "mid stream tail"]);
}

#[test]
fn interrupted_assistant_tail_ignores_visible_blocks_already_on_screen() {
    let blocks = vec![
        UiTimelineItem::TurnStart(Turn::user(false)),
        UiTimelineItem::UserInput("ship it".into()),
        UiTimelineItem::AssistantText(
            "Still running - cargo test in progress. Waiting for completion.".into(),
        ),
        UiTimelineItem::AssistantText(
            "It looks like there's a rendering issue stripping my variable names. Let me use a different approach.".into(),
        ),
    ];

    let tail = interrupted_assistant_tail(
        &blocks,
        "Still running - cargo test in progress. Waiting for completion.\n\nIt looks like there's a rendering issue stripping my variable names. Let me use a different approach.\n\nLet me check the current state first.",
    );

    assert_eq!(
        tail.as_deref(),
        Some("Let me check the current state first.")
    );
}

#[test]
fn interrupted_read_view_hides_rlm_execution_result_user_message() {
    let result_message = Message {
        id: "m1".to_string(),
        role: MessageRole::User,
        parts: vec![Part {
            // Legacy RLM exec results were stored as user text with
            // `tool_call_id` + `tool_name` preserved. Read-model rendering should
            // hide them without rendering a fake execute_lashlang
            // activity.
            id: "m1.p0".to_string(),
            kind: PartKind::Text,
            content: "[Lashlang execution result]\n\nobservations:\nraw dump".to_string(),
            attachment: None,
            tool_call_id: Some("rlm_exec_0".to_string()),
            tool_name: Some("execute_lashlang".to_string()),
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]
        .into(),
        origin: None,
    };
    let tool_calls = vec![ToolCallRecord {
        call_id: Some("rlm_exec_0".to_string()),
        tool: "execute_lashlang".to_string(),
        args: serde_json::json!({ "code": "print inspect" }),
        output: lash_core::ToolCallOutput::success(serde_json::json!({
            "observations": "raw dump",
            "tool_calls": []
        })),
        duration_ms: 0,
    }];
    let messages = vec![text_message("m0", MessageRole::User, "go"), result_message];
    let events = events_from_messages(&messages);
    let blocks = interrupted_blocks_from_test_read_view(
        &events,
        &messages,
        &tool_calls,
        &UiProjectionState::default(),
        "Cancelled.",
    );

    assert!(blocks.iter().all(|block| {
        !matches!(block, UiTimelineItem::UserInput(text) if text.contains("[Lashlang execution result]"))
    }));
    assert!(blocks.iter().all(|block| {
        !matches!(
            block,
            UiTimelineItem::Activity(activity) if activity.call.tool_name == "execute_lashlang"
        )
    }));
}

#[test]
fn ui_projection_state_omits_transient_live_turn() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.set_status("retrying", Some("in 5s".into()), true);
    if let Some(turn) = app.live_turn.as_mut() {
        turn.has_visible_output = true;
    }

    let persisted = serde_json::to_value(app.ui_projection_state()).expect("serialize ui");
    assert!(persisted.get("live_turn").is_none());
}
