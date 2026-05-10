use super::*;
use crate::editor::LARGE_PASTE_CHAR_THRESHOLD;
use async_trait::async_trait;
use lash::{Part, PruneState, ToolResultView, TurnEvent};
use lash_tui_extensions::{
    SlashCommandSpec, TuiExtension, TuiExtensionContext, TuiExtensions, TuiHostEffect,
};
use std::sync::Arc;

fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
    Message {
        id: id.to_string(),
        role,
        parts: vec![part(&format!("{id}.p0"), PartKind::Text, content)].into(),
        origin: None,
    }
}

fn part(id: &str, kind: PartKind, content: &str) -> Part {
    Part {
        id: id.to_string(),
        kind,
        content: content.to_string(),
        attachment: None,
        tool_call_id: None,
        tool_name: None,
        tool_item_id: None,
        tool_signature: None,
        prune_state: PruneState::Intact,
        reasoning_meta: None,
        response_meta: None,
    }
}

fn conversation_event(message: Message) -> lash::SessionEventRecord {
    lash::SessionEventRecord::Conversation(lash::ConversationRecord::from_message(message))
}

fn events_from_messages(messages: &[Message]) -> Vec<lash::SessionEventRecord> {
    messages.iter().cloned().map(conversation_event).collect()
}

#[test]
fn background_subagent_terminal_state_is_transient_and_freezes_duration() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let started_at = std::time::SystemTime::now() - std::time::Duration::from_secs(125);
    app.update_background_tasks(vec![lash::ManagedTaskStatus {
        id: "subagent:smoke".into(),
        kind: lash::ManagedTaskKind::Subagent,
        producer: "subagent".into(),
        run_state: lash::ManagedRunState::Running,
        started_at,
    }]);
    assert_eq!(app.background_tasks.len(), 1);
    assert_eq!(app.background_tasks[0].terminal_duration, None);

    app.update_background_tasks(vec![lash::ManagedTaskStatus {
        id: "subagent:smoke".into(),
        kind: lash::ManagedTaskKind::Subagent,
        producer: "subagent".into(),
        run_state: lash::ManagedRunState::Completed,
        started_at,
    }]);

    assert_eq!(app.background_tasks.len(), 1);
    assert_eq!(
        app.background_tasks[0].run_state,
        lash::ManagedRunState::Completed
    );
    assert!(app.background_tasks[0].transient_until.is_some());
    assert!(
        app.background_tasks[0]
            .terminal_duration
            .is_some_and(|duration| duration.as_secs() >= 125)
    );
}

fn test_read_view(
    events: &[lash::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
) -> lash::SessionReadView {
    lash::SessionReadView::from_derived_message_view(
        lash::SessionStateEnvelope::default(),
        std::sync::Arc::new(events.to_vec()),
        std::sync::Arc::new(messages.to_vec()),
        std::sync::Arc::new(tool_calls.to_vec()),
    )
}

fn timeline_items_from_test_read_view(
    events: &[lash::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
) -> Vec<UiTimelineItem> {
    let read_view = test_read_view(events, messages, tool_calls);
    timeline_from_read_view(&read_view, ui_state)
        .items()
        .to_vec()
}

fn timeline_from_test_read_view(
    events: &[lash::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
) -> UiTimeline {
    let read_view = test_read_view(events, messages, tool_calls);
    timeline_from_read_view(&read_view, ui_state)
}

fn interrupted_blocks_from_test_read_view(
    events: &[lash::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
    status_message: impl Into<String>,
) -> Vec<UiTimelineItem> {
    let read_view = test_read_view(events, messages, tool_calls);
    interrupted_blocks_from_read_view(&read_view, ui_state, status_message)
        .items()
        .to_vec()
}

fn other_variant_name(block: &UiTimelineItem) -> &'static str {
    match block {
        UiTimelineItem::TurnStart(_) => "TurnStart",
        UiTimelineItem::UserInput(_) => "UserInput",
        UiTimelineItem::AssistantText(_) => "AssistantText",
        UiTimelineItem::AssistantReasoning(_) => "AssistantReasoning",
        UiTimelineItem::Activity(_) => "Activity",
        UiTimelineItem::ShellOutput { .. } => "ShellOutput",
        UiTimelineItem::Error(_) => "Error",
        UiTimelineItem::SystemMessage(_) => "SystemMessage",
        UiTimelineItem::PluginPanel(_) => "PluginPanel",
        UiTimelineItem::LashlangCode(_) => "LashlangCode",
        UiTimelineItem::Splash => "Splash",
    }
}

#[test]
fn text_delta_accumulates_raw() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "\n\nfirst\n".into(),
    });
    assert_eq!(
        app.live_assistant.normalized_text().as_deref(),
        Some("first")
    );

    app.handle_session_event(SessionEvent::TextDelta {
        content: "\n\n\nsecond\n".into(),
    });
    assert_eq!(
        app.live_assistant.normalized_text().as_deref(),
        Some("first\n\nsecond")
    );
}

#[test]
fn text_delta_code_fence_preserved() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "text\n\n```python\n".into(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: "# comment\n".into(),
    });
    // The newline between ```python and # comment must be preserved
    assert!(
        app.live_assistant
            .normalized_text()
            .is_some_and(|text| text.contains("```python\n# comment"))
    );
}

#[test]
fn text_delta_stays_in_live_assistant_until_committed() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::TextDelta {
        content: "Draft answer".into(),
    });

    assert!(matches!(app.timeline.last(), Some(UiTimelineItem::Splash)));
    assert_eq!(
        app.live_assistant_normalized_text().as_deref(),
        Some("Draft answer")
    );
}

#[test]
fn ui_extension_commands_appear_in_editor_suggestions() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let ui_extensions = lash_tui_extensions::TuiExtensions::builtin().expect("ui extensions");
    app.set_ui_extensions(Arc::new(ui_extensions));
    app.set_input("/pl".into());

    app.update_suggestions();

    assert!(app.suggestions().iter().any(|s| s.name == "/plan"));
}

#[test]
fn ui_extension_argument_suggestions_complete_second_token() {
    struct DemoTuiExtension;

    const DEMO_COMMANDS: &[SlashCommandSpec] = &[SlashCommandSpec {
        name: "/demo",
        aliases: &[],
        usage: "/demo [help|off]",
        description: "Demo command",
        argument_hint: Some("[help|off]"),
        argument_options: &["help", "off"],
        takes_argument: true,
        allow_while_running: true,
        action: "demo",
    }];

    #[async_trait]
    impl TuiExtension for DemoTuiExtension {
        fn id(&self) -> &'static str {
            "demo_ui"
        }

        fn commands(&self) -> &'static [SlashCommandSpec] {
            DEMO_COMMANDS
        }

        async fn invoke_action(
            &self,
            _action: &str,
            _arg: Option<&str>,
            _ctx: TuiExtensionContext<'_>,
        ) -> Result<Vec<TuiHostEffect>, String> {
            Ok(Vec::new())
        }
    }

    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let ui_extensions =
        TuiExtensions::new(vec![Arc::new(DemoTuiExtension)]).expect("ui extensions");
    app.set_ui_extensions(Arc::new(ui_extensions));
    app.set_input("/demo h".into());
    app.editor.cursor_pos = app.input().len();

    app.update_suggestions();

    assert_eq!(
        app.suggestions().first().map(|value| value.name.as_str()),
        Some("help")
    );
    app.complete_suggestion();

    assert_eq!(app.input(), "/demo help");
}

#[test]
fn final_message_never_replaces_visible_streamed_text_with_shorter_text() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "Visible streamed text".into(),
    });

    app.handle_session_event(SessionEvent::Message {
        text: "Visible".into(),
        kind: "final".into(),
    });

    assert_eq!(
        app.live_assistant.normalized_text().as_deref(),
        Some("Visible streamed text")
    );
    assert!(
        !app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::AssistantText(_)))
    );
}

#[test]
fn text_delta_updates_live_token_estimate() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        mode_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: "abcd".into(),
    });
    assert_eq!(app.live_output_tokens_estimate, 1);
    app.handle_session_event(SessionEvent::TextDelta {
        content: "efgh".into(),
    });
    assert_eq!(app.live_output_tokens_estimate, 2);
}

#[test]
fn final_message_event_renders_in_live_assistant_lane() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::Message {
        text: "final output".into(),
        kind: "final".into(),
    });
    assert_eq!(
        app.live_assistant.normalized_text().as_deref(),
        Some("final output")
    );
    assert!(
        !app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::AssistantText(_)))
    );
}

#[test]
fn first_text_delta_switches_thinking_to_responding() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        mode_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: "hello".into(),
    });
    assert_eq!(
        app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
        Some("responding")
    );
    assert_eq!(
        app.live_turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        None
    );
}

#[test]
fn llm_request_sets_plain_thinking_status() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        mode_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    assert_eq!(
        app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
        Some("thinking")
    );
    assert_eq!(
        app.live_turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        None
    );
}

#[test]
fn llm_request_flushes_intermediate_stream_text() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "Let me continue testing.".into(),
    });
    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc1".into()),
        name: "read_file".into(),
        args: serde_json::json!({"path":"src/main.rs"}),
        result: serde_json::json!("ok"),
        success: true,
        duration_ms: 1,
    });
    app.handle_session_event(SessionEvent::LlmRequest {
        mode_iteration: 1,
        message_count: 0,
        tool_list: String::new(),
    });

    assert!(app.live_assistant.normalized_text().is_none());
    assert!(app.timeline.iter().any(|block| {
        matches!(block, UiTimelineItem::AssistantText(text) if text == "Let me continue testing.")
    }));
}

#[test]
fn tool_call_flushes_intermediate_stream_text_immediately() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();

    app.handle_session_event(SessionEvent::TextDelta {
        content: "I’m checking the rendering path first.".into(),
    });
    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc2".into()),
        name: "read_file".into(),
        args: serde_json::json!({"path":"lash-cli/src/app/mod.rs"}),
        result: serde_json::json!("ok"),
        success: true,
        duration_ms: 1,
    });

    assert!(app.live_assistant.normalized_text().is_none());
    assert!(matches!(
        app.timeline.first(),
        Some(UiTimelineItem::AssistantText(text))
            if text == "I’m checking the rendering path first."
    ));
    assert!(matches!(
        app.timeline.get(1),
        Some(UiTimelineItem::Activity(_))
    ));
}

#[test]
fn token_usage_resets_live_token_estimate() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "abcdefgh".into(),
    });
    assert!(app.live_output_tokens_estimate > 0);
    app.handle_session_event(SessionEvent::TokenUsage {
        mode_iteration: 0,
        usage: TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cached_input_tokens: 0,
            reasoning_tokens: 2,
        },
        cumulative: TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cached_input_tokens: 0,
            reasoning_tokens: 2,
        },
    });
    assert_eq!(app.live_output_tokens_estimate, 0);
    assert_eq!(app.last_response_usage.input_tokens, 10);
    assert_eq!(app.last_response_usage.reasoning_tokens, 2);
}

#[test]
fn input_only_streamed_usage_keeps_live_output_estimate() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "abcdefgh".into(),
    });
    let live_estimate = app.live_output_tokens_estimate;
    assert!(live_estimate > 0);
    app.handle_session_event(SessionEvent::TokenUsage {
        mode_iteration: 0,
        usage: TokenUsage {
            input_tokens: 10,
            output_tokens: 0,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        },
        cumulative: TokenUsage {
            input_tokens: 10,
            output_tokens: 0,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        },
    });
    assert_eq!(app.live_output_tokens_estimate, live_estimate);
    assert_eq!(app.token_usage.input_tokens, 10);
    assert_eq!(app.last_response_usage.input_tokens, 10);
}

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
        result: serde_json::json!({"ok": true}),
        success: true,
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
        result: serde_json::json!({"ok": true}),
        success: true,
        duration_ms: 12,
        control: None,
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
        tool_calls: Vec::new(),
        images: Vec::new(),
        error: None,
        final_output: None,
    };
    let events = vec![lash::SessionEventRecord::Mode(
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
fn rlm_trajectory_final_output_does_not_project_visible_answer_without_conversation() {
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "I'll reply directly.\n\n```lashlang\nsubmit \"Hi!\"\n```".to_string(),
        code: "submit \"Hi!\"".to_string(),
        output: Vec::new(),
        tool_calls: Vec::new(),
        images: Vec::new(),
        error: None,
        final_output: Some(serde_json::json!("Hi!")),
    };
    let events = vec![lash::SessionEventRecord::Mode(
        lash_mode_rlm::rlm_mode_event(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry)),
    )];

    let blocks =
        timeline_items_from_test_read_view(&events, &[], &[], &UiProjectionState::default());
    let variants: Vec<&str> = blocks.iter().map(other_variant_name).collect();
    assert_eq!(variants, vec!["AssistantReasoning", "LashlangCode"]);
}

#[test]
fn rlm_final_answer_projects_after_reasoning_and_lashlang_code() {
    let user = text_message("u1", MessageRole::User, "hi");
    let assistant = text_message("a1", MessageRole::Assistant, "Hi!");
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "I'll answer directly.\n\n```lashlang\nsubmit \"Hi!\"\n```".to_string(),
        code: "submit \"Hi!\"".to_string(),
        output: Vec::new(),
        tool_calls: Vec::new(),
        images: Vec::new(),
        error: None,
        final_output: Some(serde_json::json!("Hi!")),
    };
    let events = vec![
        conversation_event(user.clone()),
        lash::SessionEventRecord::Mode(lash_mode_rlm::rlm_mode_event(
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
}

#[test]
fn rlm_trajectory_projects_tool_calls_after_own_reasoning() {
    let entry = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "I'll inspect the environment.".to_string(),
        code: "now = (call exec_command { cmd: \"date -u\" })?\nprint now".to_string(),
        output: vec!["2026-04-25 20:05:57 UTC".to_string()],
        tool_calls: vec![ToolCallRecord {
            call_id: None,
            tool: "exec_command".to_string(),
            args: serde_json::json!({ "cmd": "date -u" }),
            result: serde_json::json!({
                "output": "2026-04-25 20:05:57 UTC\n",
                "exit_code": 0
            }),
            success: true,
            duration_ms: 12,
            control: None,
        }],
        images: Vec::new(),
        error: None,
        final_output: None,
    };
    let events = vec![lash::SessionEventRecord::Mode(
        lash_mode_rlm::rlm_mode_event(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry)),
    )];

    let blocks =
        timeline_items_from_test_read_view(&events, &[], &[], &UiProjectionState::default());
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
fn rlm_trajectory_steps_project_chronologically_with_tool_results() {
    let first = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_0".to_string(),
        mode_iteration: 0,
        reasoning: "First check the time.".to_string(),
        code: "now = (call exec_command { cmd: \"date -u\" })?\nprint now".to_string(),
        output: vec!["time".to_string()],
        tool_calls: vec![ToolCallRecord {
            call_id: None,
            tool: "exec_command".to_string(),
            args: serde_json::json!({ "cmd": "date -u" }),
            result: serde_json::json!({ "output": "time\n", "exit_code": 0 }),
            success: true,
            duration_ms: 3,
            control: None,
        }],
        images: Vec::new(),
        error: None,
        final_output: None,
    };
    let second = lash_rlm_types::RlmTrajectoryEntry {
        id: "rlm_step_1".to_string(),
        mode_iteration: 1,
        reasoning: "Then check files.".to_string(),
        code: "files = (call ls { path: \".\" })?\nprint files".to_string(),
        output: vec!["files".to_string()],
        tool_calls: vec![ToolCallRecord {
            call_id: None,
            tool: "ls".to_string(),
            args: serde_json::json!({ "path": "." }),
            result: serde_json::json!({ "entries": ["Cargo.toml"] }),
            success: true,
            duration_ms: 4,
            control: None,
        }],
        images: Vec::new(),
        error: None,
        final_output: None,
    };
    let assistant = text_message("a1", MessageRole::Assistant, "Done.");
    let events = vec![
        lash::SessionEventRecord::Mode(lash_mode_rlm::rlm_mode_event(
            lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(first),
        )),
        lash::SessionEventRecord::Mode(lash_mode_rlm::rlm_mode_event(
            lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(second),
        )),
        conversation_event(assistant.clone()),
    ];

    let blocks = timeline_items_from_test_read_view(
        &events,
        &[assistant],
        &[],
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
fn tool_output_renders_during_generic_running_turn() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::Message {
        text: "started git status --short\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(
        app.live_tool_output.lines,
        vec!["started git status --short".to_string()]
    );
}

#[test]
fn tool_output_carriage_return_rewrites_partial_line() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "Compiling alpha".into(),
        kind: "tool_output".into(),
    });
    assert_eq!(app.live_tool_output.partial, "Compiling alpha");

    app.handle_session_event(SessionEvent::Message {
        text: "\rCompiling beta".into(),
        kind: "tool_output".into(),
    });

    assert!(app.live_tool_output.lines.is_empty());
    assert_eq!(app.live_tool_output.partial, "Compiling beta");
}

#[test]
fn tool_output_crlf_commits_current_line() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "started cargo check\r\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(
        app.live_tool_output.lines,
        vec!["started cargo check".to_string()]
    );
    assert!(app.live_tool_output.partial.is_empty());
}

#[test]
fn tool_output_strips_ansi_escape_sequences_from_live_preview() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "\u{1b}[33mwarning\u{1b}[0m: check this\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(
        app.live_tool_output.lines,
        vec!["warning: check this".to_string()]
    );
    assert!(app.live_tool_output.partial.is_empty());
}

#[test]
fn tool_output_strips_osc_escape_sequences_from_live_preview() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "\u{1b}]11;?\u{1b}\\".into(),
        kind: "tool_output".into(),
    });
    app.handle_session_event(SessionEvent::Message {
        text: "done\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(app.live_tool_output.lines, vec!["done".to_string()]);
    assert!(app.live_tool_output.partial.is_empty());
}

#[test]
fn tool_output_tabs_collapse_to_single_spaces_in_live_preview() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "hash\trefs/tags/v0.2.29\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(
        app.live_tool_output.lines,
        vec!["hash refs/tags/v0.2.29".to_string()]
    );
    assert!(app.live_tool_output.partial.is_empty());
}

#[test]
fn tool_output_does_not_change_total_content_height() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline = vec![UiTimelineItem::UserInput("inspect this".into())].into();
    app.start_turn();

    let baseline = app.total_content_height(32, 8);
    app.handle_session_event(SessionEvent::Message {
        text: "started git status --short\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(app.total_content_height(32, 8), baseline);
}

#[test]
fn update_plan_panel_lights_up_plan_dock() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    assert!(
        app.plan_dock.is_none(),
        "dock starts empty before any update_plan event"
    );

    app.handle_session_event(SessionEvent::PluginEvent {
        plugin_id: "update_plan".into(),
        event: lash::PluginSurfaceEvent::PanelUpsert {
            key: "plan".into(),
            title: "PLAN".into(),
            content: "- [x] Inspect\n- [~] Patch layout\n- [ ] Run tests\n".into(),
        },
    });

    let dock = app.plan_dock.as_ref().expect("plan dock populated");
    assert_eq!(dock.title, "PLAN");
    assert_eq!(dock.items.len(), 3);
    assert_eq!(dock.items[0].status, PlanDockItemStatus::Done);
    assert_eq!(dock.items[1].status, PlanDockItemStatus::Active);
    assert_eq!(dock.items[2].status, PlanDockItemStatus::Pending);
    // Event must NOT land as an inline UiTimelineItem::PluginPanel.
    assert!(
        !app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::PluginPanel(_))),
        "update_plan panels route to the dock, not the scroll history"
    );
}

#[test]
fn rlm_budget_warning_uses_status_not_user_message() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline = vec![UiTimelineItem::AssistantText("working".into())].into();

    app.handle_session_event(SessionEvent::PluginEvent {
        plugin_id: "mode_rlm".into(),
        event: lash::PluginSurfaceEvent::Status {
            key: "rlm_context_budget_warning".into(),
            label: "context budget".into(),
            detail: Some("120292 tokens used; warn at 100000; choose handoff path".into()),
            transient_ms: Some(8_000),
        },
    });

    assert!(
        app.timeline
            .iter()
            .all(|block| !matches!(block, UiTimelineItem::UserInput(_))),
        "runtime budget warning must not be rendered as user input"
    );
    assert!(app.live_turn.as_ref().is_some_and(|turn| {
        turn.status_text == "context budget"
            && turn
                .status_detail
                .as_deref()
                .is_some_and(|detail| detail.contains("choose handoff path"))
    }));
}

#[test]
fn plugin_panel_events_upsert_and_clear_blocks() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::PluginEvent {
        plugin_id: "demo".into(),
        event: lash::PluginSurfaceEvent::PanelUpsert {
            key: "panel:1".into(),
            title: "TASK BOARD".into(),
            content: "1. Inspect\n2. Patch".into(),
        },
    });
    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::PluginPanel(panel)) if panel.title == "TASK BOARD"
    ));
    assert!(
        app.live_turn
            .as_ref()
            .is_some_and(|turn| turn.has_visible_output)
    );

    app.handle_session_event(SessionEvent::PluginEvent {
        plugin_id: "demo".into(),
        event: lash::PluginSurfaceEvent::PanelClear {
            key: "panel:1".into(),
        },
    });
    assert!(
        !app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::PluginPanel(_)))
    );
}

#[test]
fn plan_exit_tool_queues_follow_up_turn() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let ui_extensions =
        lash_tui_extensions::TuiExtensions::builtin().expect("builtin ui extensions");
    crate::apply_ui_host_effects(
        &mut app,
        ui_extensions.effects_for_turn_event(&TurnEvent::ToolCallCompleted {
            call_id: Some("tc-plan-exit".into()),
            name: "plan_exit".into(),
            args: serde_json::json!({}),
            result: ToolResultView {
                raw: serde_json::json!({
                "approved": true,
                "confirmation_display": "Start implementing now\n\nNote: safe slice first",
                "plan_path": ".lash/plans/session.md",
                "execution_mode": "current_session",
                "next_turn_input": "Execute the plan in `.lash/plans/session.md`."
                }),
                for_model: serde_json::json!({}),
                for_state: serde_json::json!({}),
            },
            success: true,
            duration_ms: 5,
        }),
    );

    let (queued, was_pending) = app.take_next_queued_turn().expect("queued turn");
    assert!(!was_pending);
    assert_eq!(
        queued.display_text,
        "Start implementing now\n\nNote: safe slice first"
    );
    assert_eq!(
        queued.effective_text,
        "Execute the plan in `.lash/plans/session.md`."
    );
}

#[test]
fn plan_exit_tool_call_consumes_pending_prompt_response() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let (tx, _rx) = std::sync::mpsc::channel();
    app.show_prompt(PromptState {
        request: lash::PromptRequest::single(
            "Review the plan",
            vec!["Start implementing now".into(), "Keep planning".into()],
        )
        .with_optional_note(),
        focus: crate::overlay::PromptFocus::Text,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: "safe slice first".into(),
        reply_cursor: "safe slice first".len(),
        response_tx: tx,
    });

    let response = app.take_prompt_response();
    assert_eq!(
        response.as_deref(),
        Some("1. Start implementing now\n\nNote: safe slice first")
    );
    assert!(
        app.timeline
            .iter()
            .all(|block| !matches!(block, UiTimelineItem::UserInput(_)))
    );

    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc-plan-exit".into()),
        name: "plan_exit".into(),
        args: serde_json::json!({}),
        result: serde_json::json!({
            "approved": true,
            "confirmation_display": "Start implementing now\n\nNote: safe slice first",
            "execution_mode": "current_session",
            "next_turn_input": "Execute the plan in `.lash/plans/session.md`."
        }),
        success: true,
        duration_ms: 5,
    });

    assert!(app.timeline.iter().all(|block| {
        !matches!(block, UiTimelineItem::UserInput(text) if text.contains("Start implementing now"))
    }));
}

#[test]
fn plan_exit_fresh_context_tool_does_not_queue_ui_turn_or_switch() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let ui_extensions =
        lash_tui_extensions::TuiExtensions::builtin().expect("builtin ui extensions");
    crate::apply_ui_host_effects(
        &mut app,
        ui_extensions.effects_for_turn_event(&TurnEvent::ToolCallCompleted {
            call_id: Some("tc-plan-exit-fresh".into()),
            name: "plan_exit".into(),
            args: serde_json::json!({}),
            result: ToolResultView {
                raw: serde_json::json!({
                "approved": true,
                "execution_mode": "fresh_context",
                "session_id": "new-plan-session",
                }),
                for_model: serde_json::json!({}),
                for_state: serde_json::json!({}),
            },
            success: true,
            duration_ms: 5,
        }),
    );

    assert!(app.take_next_queued_turn().is_none());
    assert!(
        app.timeline
            .iter()
            .all(|block| !matches!(block, UiTimelineItem::UserInput(_))),
        "fresh-context handoff is a runtime transition, not a UI-queued turn"
    );
}

#[test]
fn cancelled_error_renders_as_system_message() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.note_manual_interrupt_requested();
    app.handle_session_event(SessionEvent::Error {
        message: "LLM error: cancelled".into(),
        envelope: Some(lash::session_model::ErrorEnvelope {
            kind: "llm_provider".into(),
            code: Some("cancelled".into()),
            user_message: "LLM error: cancelled".into(),
            raw: None,
        }),
    });

    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::SystemMessage(msg)) if msg == "Manually interrupted."
    ));
    assert!(!app.running);
    assert!(app.live_turn.is_none());
}

#[test]
fn cancelled_error_without_manual_request_still_stops_immediately() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::Error {
        message: "LLM error: cancelled".into(),
        envelope: Some(lash::session_model::ErrorEnvelope {
            kind: "llm_provider".into(),
            code: Some("cancelled".into()),
            user_message: "LLM error: cancelled".into(),
            raw: None,
        }),
    });

    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::SystemMessage(msg)) if msg == "Cancelled."
    ));
    assert!(!app.running);
    assert!(app.live_turn.is_none());
}

#[test]
fn repeated_cancelled_errors_do_not_duplicate_system_message() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    let cancelled = SessionEvent::Error {
        message: "LLM error: cancelled".into(),
        envelope: Some(lash::session_model::ErrorEnvelope {
            kind: "llm_provider".into(),
            code: Some("cancelled".into()),
            user_message: "LLM error: cancelled".into(),
            raw: None,
        }),
    };

    app.handle_session_event(cancelled.clone());
    app.handle_session_event(cancelled);

    let cancelled_blocks = app
        .timeline
        .iter()
        .filter(|block| {
            matches!(
                block,
                UiTimelineItem::SystemMessage(msg) if msg == "Cancelled."
            )
        })
        .count();
    assert_eq!(cancelled_blocks, 1);
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
            tool_item_id: None,
            tool_signature: None,
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
        result: serde_json::json!({
            "observations": "raw dump",
            "tool_calls": []
        }),
        success: true,
        duration_ms: 0,
        control: None,
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
fn non_manual_error_sets_transient_status() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        mode_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::Error {
        message: "LLM error: Claude request failed with 500".into(),
        envelope: Some(lash::session_model::ErrorEnvelope {
            kind: "llm_provider".into(),
            code: Some("http_500".into()),
            user_message: "LLM error: Claude request failed with 500".into(),
            raw: None,
        }),
    });
    app.handle_session_event(SessionEvent::Done);

    assert_eq!(
        app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
        Some("error")
    );
    assert_eq!(
        app.live_turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        Some("LLM error: Claude request failed with 500")
    );
}

#[test]
fn retry_status_stays_visible_when_retry_request_starts() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        mode_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::RetryStatus {
        wait_seconds: 2,
        attempt: 2,
        max_attempts: 4,
        reason: "Codex returned non-SSE body but it could not be read".into(),
        envelope: None,
    });

    assert_eq!(
        app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
        Some("retrying")
    );
    assert_eq!(
        app.live_turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        Some("in 2s · attempt 2/4 · Codex returned non-SSE body but it could not be read")
    );

    app.handle_session_event(SessionEvent::LlmRequest {
        mode_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });

    assert_eq!(
        app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
        Some("retrying")
    );
    assert_eq!(
        app.live_turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        Some("attempt 2/4 · Codex returned non-SSE body but it could not be read")
    );
}

#[test]
fn transient_status_expires_on_tick() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        mode_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::Error {
        message: "runtime error".into(),
        envelope: None,
    });
    app.handle_session_event(SessionEvent::Done);

    if let Some(turn) = app.live_turn.as_mut() {
        turn.transient_until = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    }
    app.on_tick();
    assert!(app.live_turn.is_none());
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

#[test]
fn queued_turns_are_fifo_and_skip_pending_injections() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.queue_turn(PreparedTurn::new("queued-1".into(), Vec::new()));
    app.queue_turn(PreparedTurn::new("queued-2".into(), Vec::new()));
    app.queue_pending_steer(PreparedTurn::new("next-1".into(), Vec::new()));
    app.queue_pending_steer(PreparedTurn::new("next-2".into(), Vec::new()));

    let order: Vec<(String, bool)> = std::iter::from_fn(|| app.take_next_queued_turn())
        .map(|(turn, was_pending)| (turn.display_text, was_pending))
        .collect();

    assert_eq!(
        order,
        vec![("queued-1".into(), false), ("queued-2".into(), false),]
    );
    assert_eq!(app.pending_steers.len(), 2);
}

#[test]
fn take_last_queued_turn_restores_explicit_queue_only() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.queue_pending_steer(PreparedTurn::new("next".into(), Vec::new()));
    app.queue_turn(PreparedTurn::new("queued".into(), vec![vec![1, 2, 3]]));

    let (turn, was_pending) = app.take_last_queued_turn().expect("queued turn");
    assert_eq!(turn.display_text, "queued");
    assert_eq!(turn.images.len(), 1);
    assert_eq!(turn.images[0].id, 1);
    assert_eq!(turn.images[0].png_bytes, vec![1, 2, 3]);
    assert!(!was_pending);

    assert!(app.take_last_queued_turn().is_none());
    assert_eq!(app.pending_steers.len(), 1);
}

#[test]
fn wake_session_effect_uses_hidden_monitor_queue() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());

    crate::apply_ui_host_effects(
        &mut app,
        vec![TuiHostEffect::WakeSession {
            input: "Monitor event \"build\": done".into(),
        }],
    );

    assert!(!app.has_queued_messages());
    assert_eq!(
        app.take_pending_monitor_wakes(),
        vec!["Monitor event \"build\": done".to_string()]
    );
}

#[test]
fn acknowledged_monitor_wakes_do_not_requeue() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.queue_monitor_wake("Monitor event \"build\": done".into());
    let wakes = app.take_pending_monitor_wakes();
    app.mark_monitor_wakes_in_flight(&wakes);

    app.acknowledge_monitor_wakes(&[PluginMessage::text(
        MessageRole::System,
        "Monitor event \"build\": done",
    )]);
    app.recycle_unaccepted_monitor_wakes();

    assert!(app.take_pending_monitor_wakes().is_empty());
}

#[test]
fn accepted_injected_turn_input_renders_matching_pending_steer() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("follow up".into(), Vec::new());
    app.queue_pending_steer(turn.clone());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash::AcceptedInjectedTurnInput {
            id: None,
            message: PluginMessage::text(MessageRole::User, "follow up"),
        }],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    assert!(app.pending_steers.is_empty());
    assert!(
        app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::UserInput(text) if text == "follow up"))
    );
}

#[test]
fn accepted_injected_turn_input_matches_by_runtime_content_even_when_display_text_differs() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let mut turn = PreparedTurn::new("/localref lash for context if needed".into(), Vec::new());
    turn.effective_text =
        "/localref lash for context if needed\n\n<skill>\n<name>localref</name>\nbody\n</skill>"
            .into();
    turn.input_metadata.effective_text = turn.effective_text.clone();
    app.queue_pending_steer(turn.clone());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash::AcceptedInjectedTurnInput {
            id: None,
            message: PluginMessage::text(
                MessageRole::User,
                "/localref lash for context if needed\n\n<skill>\n<name>localref</name>\nbody\n</skill>",
            ),
        }],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    assert!(app.pending_steers.is_empty());
    assert!(app.timeline.iter().any(|block| matches!(
        block,
        UiTimelineItem::UserInput(text) if text == "/localref lash for context if needed"
    )));
}

#[test]
fn accepted_injected_turn_input_without_pending_match_still_renders_once() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash::AcceptedInjectedTurnInput {
            id: None,
            message: PluginMessage {
                role: MessageRole::User,
                content: "runtime content".into(),
                parts: Vec::new(),
                images: Vec::new(),
            },
        }],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    app.handle_session_event(SessionEvent::InjectedMessagesCommitted {
        messages: vec![PluginMessage {
            role: MessageRole::User,
            content: "runtime content".into(),
            parts: Vec::new(),
            images: Vec::new(),
        }],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    let matching_blocks = app
        .timeline
        .iter()
        .filter(
            |block| matches!(block, UiTimelineItem::UserInput(text) if text == "runtime content"),
        )
        .count();
    assert_eq!(matching_blocks, 1);
}

#[test]
fn accepted_injected_turn_input_removes_matching_pending_steer_without_popping_wrong_one() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.queue_pending_steer(PreparedTurn::new("first queued steer".into(), Vec::new()));
    app.queue_pending_steer(PreparedTurn::new(
        "uhh do not switch nvm".into(),
        Vec::new(),
    ));

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash::AcceptedInjectedTurnInput {
            id: None,
            message: PluginMessage::text(MessageRole::User, "uhh do not switch nvm"),
        }],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    assert_eq!(app.pending_steers.len(), 1);
    assert_eq!(app.pending_steers[0].display_text, "first queued steer");
}

#[test]
fn injected_messages_committed_do_not_duplicate_user_input_after_assistant_work() {
    // Regression: when the runtime fires `InjectedMessagesCommitted` after the
    // assistant has already streamed text and tool calls (codex prose-between-
    // tool-calls flow), the existing UserInput block is no longer at the
    // tail of `app.timeline` — so the old `blocks.last()` dedup let a duplicate
    // through. Dedup must scan the whole current turn.
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("Why are you still dillydallying".into(), Vec::new());
    app.push_prepared_user_input(&turn);
    app.queue_pending_steer(turn.clone());

    // Assistant streams some prose, then runs a tool — this pushes an
    // AssistantText and an Activity block in front of the original UserInput.
    app.handle_session_event(SessionEvent::TextDelta {
        content: "You're right.".into(),
    });
    app.handle_session_event(SessionEvent::ToolCall {
        name: "fs_read".into(),
        args: serde_json::json!({"path": "lash/src/plugin.rs"}),
        result: serde_json::json!({"output": "..."}),
        success: true,
        duration_ms: 12,
        call_id: None,
    });

    // Now the late commit arrives.
    app.handle_session_event(SessionEvent::InjectedMessagesCommitted {
        messages: vec![PluginMessage::text(
            MessageRole::User,
            "Why are you still dillydallying",
        )],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    let matching_blocks = app
        .timeline
        .iter()
        .filter(|block| {
            matches!(
                block,
                UiTimelineItem::UserInput(text) if text == "Why are you still dillydallying"
            )
        })
        .count();
    assert_eq!(matching_blocks, 1);
}

#[test]
fn injected_messages_committed_do_not_duplicate_existing_visible_user_input() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new(
        "(I want future migrations to work though!)".into(),
        Vec::new(),
    );
    app.push_prepared_user_input(&turn);
    app.queue_pending_steer(turn.clone());

    app.handle_session_event(SessionEvent::InjectedMessagesCommitted {
        messages: vec![PluginMessage::text(
            MessageRole::User,
            "(I want future migrations to work though!)",
        )],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    let matching_blocks = app
        .timeline
        .iter()
        .filter(|block| {
            matches!(
                block,
                UiTimelineItem::UserInput(text)
                    if text == "(I want future migrations to work though!)"
            )
        })
        .count();
    assert_eq!(matching_blocks, 1);
    assert!(app.pending_steers.is_empty());
}

#[test]
fn queued_injection_stays_out_of_history_until_committed() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("follow up now".into(), Vec::new());

    app.queue_pending_steer(turn.clone());

    assert_eq!(app.pending_steers.len(), 1);
    assert!(!matches!(
        app.timeline.last(),
        Some(UiTimelineItem::UserInput(_))
    ));
}

#[test]
fn regular_queued_turn_stays_out_of_history_until_dispatched() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("queued text".into(), Vec::new());
    app.queue_turn(turn);

    assert_eq!(app.queued_turns.len(), 1);
    assert!(!matches!(
        app.timeline.last(),
        Some(UiTimelineItem::UserInput(_))
    ));
}

#[test]
fn history_up_restores_last_queued_turn_before_history() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.editor.input_history = vec!["older turn".into()];
    app.queue_turn(PreparedTurn::new("queued text".into(), vec![vec![1, 2, 3]]));

    app.history_up();

    assert_eq!(app.input(), "queued text");
    assert_eq!(app.editor.pending_images.len(), 1);
    assert_eq!(app.editor.pending_images[0].id, 1);
    assert_eq!(app.editor.pending_images[0].png_bytes, vec![1, 2, 3]);
    assert!(app.queued_turns.is_empty());
    assert_eq!(app.editor.input_history_idx, None);
}

#[test]
fn restore_prepared_turn_clears_history_selection() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.editor.input_history = vec!["older turn".into()];
    app.editor.input_history_idx = Some(0);
    app.start_input_selection(1);
    app.update_input_selection(4);
    app.finish_input_selection();

    app.restore_prepared_turn(PreparedTurn::new("queued text".into(), Vec::new()));

    assert_eq!(app.editor.input_history_idx, None);
    assert!(!app.has_input_selection());
}

#[test]
fn backspace_deletes_image_marker_atomically() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("hello [Image #2] world".into());
    app.editor.cursor_pos = "hello [Image #2]".len();
    app.editor.pending_images = vec![PendingImage {
        id: 2,
        png_bytes: vec![1, 2, 3],
    }];

    app.backspace();

    assert_eq!(app.input(), "hello  world");
    assert!(app.editor.pending_images.is_empty());
    assert_eq!(app.cursor_pos(), "hello ".len());
}

#[test]
fn next_image_marker_id_tracks_highest_visible_marker() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("[Image #2] [Image #5]".into());
    app.editor.pending_images = vec![PendingImage {
        id: 2,
        png_bytes: vec![1, 2, 3],
    }];

    assert_eq!(app.next_image_marker_id(), 6);
}

#[test]
fn add_pending_image_uses_highest_marker_plus_one() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("before [Image #4] after".into());
    app.editor.pending_images = vec![PendingImage {
        id: 2,
        png_bytes: vec![9],
    }];

    let id = app.add_pending_image(vec![1, 2, 3]);

    assert_eq!(id, 5);
    assert_eq!(app.editor.pending_images.last().map(|img| img.id), Some(5));
}

#[test]
fn complete_pending_image_only_attaches_when_marker_still_exists() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("before [Image #3] after".into());
    app.begin_pending_image(3);

    assert!(app.complete_pending_image(3, vec![1, 2, 3]));
    assert_eq!(app.editor.pending_images.len(), 1);
    assert_eq!(app.editor.pending_images[0].id, 3);

    app.editor.input.clear();
    app.begin_pending_image(4);
    assert!(!app.complete_pending_image(4, vec![9]));
    assert!(app.editor.pending_images.iter().all(|image| image.id != 4));
}

#[test]
fn fail_pending_image_removes_marker_and_inflight_state() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("before [Image #7] after".into());
    app.editor.cursor_pos = app.input().len();
    app.begin_pending_image(7);

    assert!(app.fail_pending_image(7));
    assert_eq!(app.input(), "before  after");
    assert!(!app.editor.inflight_image_ids.contains(&7));
}

#[test]
fn pending_image_jobs_only_count_visible_markers() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.begin_pending_image(2);
    assert!(!app.has_pending_image_jobs());

    app.set_input("[Image #2]".into());
    app.editor.cursor_pos = app.input().len();
    assert!(app.has_pending_image_jobs());

    app.backspace();
    assert!(!app.has_pending_image_jobs());
}

#[test]
fn try_take_prepared_turn_waits_for_visible_inflight_images() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("[Image #1]".into());
    app.begin_pending_image(1);

    assert!(app.try_take_prepared_turn().is_none());
    assert_eq!(app.input(), "[Image #1]");
    assert!(app.has_pending_image_jobs());
}

#[test]
fn take_prompt_response_renders_visible_user_block() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let (tx, rx) = std::sync::mpsc::channel();
    app.show_prompt(PromptState {
        request: lash::PromptRequest::freeform("Pick one"),
        focus: crate::overlay::PromptFocus::Text,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: "red".into(),
        reply_cursor: 3,
        response_tx: tx,
    });

    let response = app.take_prompt_response();

    assert_eq!(response.as_deref(), Some("red"));
    assert_eq!(
        rx.recv().expect("response"),
        lash::PromptResponse::Text {
            text: "red".to_string(),
        }
    );
    assert!(app.prompt_state().is_none());
    assert!(app.dirty);
    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::UserInput(text)) if text == "red"
    ));
}

#[test]
fn dismiss_prompt_marks_ui_dirty() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let (tx, rx) = std::sync::mpsc::channel();
    app.show_prompt(PromptState {
        request: lash::PromptRequest::single("Pick one", vec!["red".into()]),
        focus: crate::overlay::PromptFocus::Options,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: String::new(),
        reply_cursor: 0,
        response_tx: tx,
    });

    app.dismiss_prompt();

    assert_eq!(
        rx.recv().expect("response"),
        lash::PromptResponse::Single {
            selection: String::new(),
            note: None,
        }
    );
    assert!(app.prompt_state().is_none());
    assert!(app.dirty);
}

#[test]
fn keep_latest_user_block_visible_shows_prompt_start_before_first_token() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();
    for idx in 0..6 {
        app.timeline
            .push(UiTimelineItem::AssistantText(format!("history {idx}")));
    }
    app.timeline.push(UiTimelineItem::UserInput(
        [
            "first line",
            "second line",
            "third line",
            "fourth line",
            "fifth line",
            "sixth line",
        ]
        .join("\n"),
    ));
    app.start_turn();
    app.follow_mode = FollowOutputMode::Contextual;

    let width = 32usize;
    let viewport_height = 4usize;
    app.ensure_height_cache_pub(width, viewport_height);
    app.scroll_offset = usize::MAX;

    app.keep_latest_user_block_visible();

    let last_idx = app.timeline.len() - 1;
    let max_scroll = app
        .total_content_height(width, viewport_height)
        .saturating_sub(viewport_height);
    let expected =
        app.contextual_follow_offset(app.block_content_start_offset(last_idx), max_scroll);
    assert_eq!(app.scroll_offset, expected);
}

#[test]
fn keep_latest_user_block_visible_keeps_short_prompt_bottom_aligned() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();
    for idx in 0..6 {
        app.timeline
            .push(UiTimelineItem::AssistantText(format!("history {idx}")));
    }
    app.timeline
        .push(UiTimelineItem::UserInput("short prompt".into()));
    app.running = true;
    app.follow_mode = FollowOutputMode::Contextual;

    let width = 32usize;
    let viewport_height = 4usize;
    app.ensure_height_cache_pub(width, viewport_height);
    app.scroll_offset = usize::MAX;

    app.keep_latest_user_block_visible();

    let cache = app.height_cache_snapshot().to_vec();
    let last_idx = app.timeline.len() - 1;
    let block_end = cache[last_idx];
    assert_eq!(app.scroll_offset, block_end.saturating_sub(viewport_height));
}

#[test]
fn splash_collapses_to_compact_scrollback_height_once_history_exists() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline
        .push(UiTimelineItem::UserInput("short prompt".into()));

    let width = 32usize;
    let viewport_height = 12usize;
    app.ensure_height_cache_pub(width, viewport_height);

    let cache = app.height_cache_snapshot().to_vec();
    assert_eq!(cache[0], SPLASH_SCROLLBACK_HEIGHT);
}

#[test]
fn dismiss_splash_removes_empty_state_before_history_content() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.dismiss_splash();

    assert!(app.timeline.is_empty());

    app.timeline.push(UiTimelineItem::UserInput("hello".into()));
    app.dismiss_splash();

    assert_eq!(app.timeline.len(), 1);
    assert!(matches!(app.timeline[0], UiTimelineItem::UserInput(_)));
}

#[test]
fn refresh_follow_output_anchor_tracks_bottom_when_idle() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();

    app.timeline
        .push(UiTimelineItem::UserInput("Message 1".into()));
    app.timeline.push(UiTimelineItem::AssistantText(
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5".into(),
    ));

    let width = 80;
    app.follow_mode = FollowOutputMode::Bottom;

    app.refresh_follow_output_anchor(width, 3);
    let small_bottom = app.total_content_height(width, 3).saturating_sub(3);
    assert_eq!(app.scroll_offset, small_bottom);

    app.refresh_follow_output_anchor(width, 6);
    let large_bottom = app.total_content_height(width, 6).saturating_sub(6);
    assert_eq!(app.scroll_offset, large_bottom);
}

#[test]
fn refresh_follow_output_anchor_reveals_output_start_once_then_follows_tail() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();

    app.timeline
        .push(UiTimelineItem::UserInput("Message 1".into()));
    app.timeline.push(UiTimelineItem::AssistantText(
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5".into(),
    ));
    app.start_turn();
    app.follow_mode = FollowOutputMode::Contextual;
    if let Some(turn) = app.live_turn.as_mut() {
        turn.has_visible_output = true;
        turn.output_start_anchor_pending = true;
    }

    let width = 80;
    let viewport_height = 3;

    app.refresh_follow_output_anchor(width, viewport_height);
    let first_anchor = app.scroll_offset;
    let max_scroll = app
        .total_content_height(width, viewport_height)
        .saturating_sub(viewport_height);
    assert!(first_anchor < max_scroll);
    assert_eq!(app.follow_mode, FollowOutputMode::Bottom);

    app.refresh_follow_output_anchor(width, viewport_height);
    assert_eq!(app.scroll_offset, max_scroll);
}

#[test]
fn resume_follow_output_reenables_bottom_following() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();
    app.timeline.push(UiTimelineItem::UserInput("hello".into()));
    app.timeline
        .push(UiTimelineItem::AssistantText("world".into()));
    app.follow_mode = FollowOutputMode::Paused;
    app.scroll_offset = 3;

    app.resume_follow_output();

    assert_eq!(app.follow_mode, FollowOutputMode::Bottom);
    assert_eq!(app.scroll_offset, usize::MAX);
}

#[test]
fn scroll_up_from_follow_output_detaches_from_bottom_anchor() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();
    app.timeline.push(UiTimelineItem::UserInput("hello".into()));
    app.timeline.push(UiTimelineItem::AssistantText(
        (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    ));
    let width = 24;
    let viewport_height = 5;
    app.follow_mode = FollowOutputMode::Bottom;
    app.ensure_height_cache_pub(width, viewport_height);
    app.refresh_follow_output_anchor(width, viewport_height);

    let bottom = app.scroll_offset;
    app.scroll_up(2);

    assert_eq!(app.follow_mode, FollowOutputMode::Paused);
    assert_eq!(app.scroll_offset, bottom.saturating_sub(2));
}

#[test]
fn scroll_down_to_bottom_reenables_tail_follow_instead_of_contextual_anchor() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();
    app.timeline
        .push(UiTimelineItem::AssistantText("older history".into()));
    app.timeline
        .push(UiTimelineItem::UserInput("prompt".into()));
    app.start_turn();
    app.follow_mode = FollowOutputMode::Contextual;

    app.handle_session_event(SessionEvent::TextDelta {
        content: (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    });

    let width = 24;
    let viewport_height = 5;
    app.refresh_follow_output_anchor(width, viewport_height);
    let contextual_anchor = app.scroll_offset;

    app.scroll_up(2);
    assert_eq!(app.follow_mode, FollowOutputMode::Paused);

    app.scroll_down(usize::MAX / 2, viewport_height, width);
    assert_eq!(app.follow_mode, FollowOutputMode::Bottom);

    let max_scroll = app
        .total_content_height(width, viewport_height)
        .saturating_sub(viewport_height);
    assert!(
        contextual_anchor < max_scroll,
        "test requires contextual anchor above the tail"
    );

    app.refresh_follow_output_anchor(width, viewport_height);

    assert_eq!(app.scroll_offset, max_scroll);
}

#[test]
fn text_delta_does_not_force_scroll_when_follow_output_is_paused() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();
    app.timeline
        .push(UiTimelineItem::UserInput("prompt".into()));
    app.start_turn();
    app.follow_mode = FollowOutputMode::Paused;
    app.scroll_offset = 3;

    app.handle_session_event(SessionEvent::TextDelta {
        content: "streamed output".into(),
    });

    assert_eq!(app.scroll_offset, 3);
    assert_eq!(app.follow_mode, FollowOutputMode::Paused);
}

#[test]
fn text_delta_reveals_message_start_before_switching_to_tail_follow() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();
    app.timeline
        .push(UiTimelineItem::UserInput("prompt".into()));
    app.start_turn();
    app.follow_mode = FollowOutputMode::Contextual;

    app.handle_session_event(SessionEvent::TextDelta {
        content: (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    });

    let width = 24;
    let viewport_height = 5;
    app.refresh_follow_output_anchor(width, viewport_height);

    let max_scroll = app
        .total_content_height(width, viewport_height)
        .saturating_sub(viewport_height);
    let expected = app.contextual_follow_offset(
        app.latest_turn_output_start_offset()
            .expect("live assistant start offset"),
        max_scroll,
    );

    assert_eq!(app.scroll_offset, expected);
    assert!(
        app.scroll_offset < max_scroll,
        "follow mode should anchor above the tail for tall streamed output"
    );
    assert_eq!(app.follow_mode, FollowOutputMode::Bottom);

    app.refresh_follow_output_anchor(width, viewport_height);
    assert_eq!(app.scroll_offset, max_scroll);
}

#[test]
fn refresh_follow_output_anchor_repositions_waiting_prompt_on_resize() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.push(UiTimelineItem::UserInput(
        "A long prompt that should stay visible while we are waiting for first token output".into(),
    ));
    app.running = true;
    app.follow_mode = FollowOutputMode::Contextual;

    let width = 24;
    app.refresh_follow_output_anchor(width, 3);
    let initial_offset = app.scroll_offset;

    app.refresh_follow_output_anchor(width, 6);
    assert!(
        app.scroll_offset <= initial_offset,
        "larger viewport should not push the waiting prompt further down"
    );
}

#[test]
fn handle_tool_call_merges_contiguous_exploration_activity() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();

    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc3".into()),
        name: "grep".into(),
        args: serde_json::json!({"query": "ctx"}),
        result: serde_json::json!("match"),
        success: true,
        duration_ms: 10,
    });
    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc4".into()),
        name: "read_file".into(),
        args: serde_json::json!({"path": "lash-cli/src/render/mod.rs"}),
        result: serde_json::json!("==> lash-cli/src/render/mod.rs <==\nline"),
        success: true,
        duration_ms: 5,
    });

    assert_eq!(app.timeline.len(), 1);
    match &app.timeline[0] {
        UiTimelineItem::Activity(activity) => {
            assert_eq!(activity.call.kind, ActivityKind::Exploration);
            // Multi-op explorations render as `Explored` + an op list.
            // No tag, no step counter — the list is always visible at
            // the default expand level so the count would duplicate it.
            assert_eq!(activity.call.tag, None);
            assert_eq!(activity.call.summary, "Explored");
            assert!(activity.children.is_empty());
            assert_eq!(
                activity.result.detail_lines,
                vec![
                    "Search \"ctx\"".to_string(),
                    "Read lash-cli/src/render/mod.rs".to_string(),
                ]
            );
        }
        other => panic!(
            "expected activity block, got {:?}",
            other_variant_name(other)
        ),
    }
}

#[test]
fn handle_tool_call_merges_contiguous_edit_activity() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();

    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc5".into()),
        name: "apply_patch".into(),
        args: serde_json::json!({}),
        result: serde_json::json!({
            "summary": "Applied patch to 1 file",
            "added": 1,
            "removed": 1,
            "files": [{
                "path": "a.rs",
                "status": "modified",
                "added": 1,
                "removed": 1,
                "diff": "--- a/a.rs\n+++ b/a.rs\n@@ -1,1 +1,1 @@\n-old\n+new"
            }]
        }),
        success: true,
        duration_ms: 7,
    });
    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc6".into()),
        name: "apply_patch".into(),
        args: serde_json::json!({}),
        result: serde_json::json!({
            "summary": "Applied patch to 1 file",
            "added": 2,
            "removed": 0,
            "files": [{
                "path": "b.rs",
                "status": "added",
                "added": 2,
                "removed": 0,
                "diff": "--- a/b.rs\n+++ b/b.rs\n@@ -0,0 +1,2 @@\n+fn one() {}\n+fn two() {}"
            }]
        }),
        success: true,
        duration_ms: 5,
    });

    assert_eq!(app.timeline.len(), 1);
    match &app.timeline[0] {
        UiTimelineItem::Activity(activity) => {
            assert_eq!(activity.call.kind, ActivityKind::Edit);
            assert_eq!(activity.call.summary, "Edited 2 files (+3 -1)");
            assert_eq!(activity.duration_ms, 12);
            assert!(activity.children.is_empty());
        }
        other => panic!(
            "expected activity block, got {:?}",
            other_variant_name(other)
        ),
    }
}

#[test]
fn insert_text_inserts_literal_payload_at_cursor() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("startend".into());
    app.editor.cursor_pos = "start".len();

    app.insert_text("\nplain pasted text\n");

    assert_eq!(app.input(), "start\nplain pasted text\nend");
    assert_eq!(app.cursor_pos(), "start\nplain pasted text\n".len());
}

#[test]
fn insert_pasted_text_large_uses_placeholder_and_prepare_expands() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 5);

    app.insert_pasted_text(&large);

    let placeholder = format!("[Pasted Content {} chars]", large.chars().count());
    assert_eq!(app.input(), placeholder);
    assert_eq!(app.editor.pending_large_pastes.len(), 1);

    let prepared = app.take_prepared_turn();
    assert_eq!(prepared.display_text, placeholder);
    assert_eq!(prepared.effective_text, large);
    assert_eq!(prepared.large_pastes.len(), 1);
    assert!(app.input().is_empty());
    assert!(app.editor.pending_large_pastes.is_empty());
}

#[test]
fn backspace_deletes_large_paste_placeholder_atomically() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 2);

    app.insert_pasted_text(&large);
    let placeholder = app.input().to_string();
    app.editor.cursor_pos = placeholder.len();

    app.backspace();

    assert!(app.input().is_empty());
    assert!(app.editor.pending_large_pastes.is_empty());
}

#[test]
fn repeated_same_size_large_pastes_get_numbered_placeholders() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 4);
    let base = format!("[Pasted Content {} chars]", large.chars().count());

    app.insert_pasted_text(&large);
    app.insert_pasted_text(&large);

    assert_eq!(app.input(), format!("{base}{base} #2"));
    assert_eq!(app.editor.pending_large_pastes.len(), 2);
    assert_eq!(app.editor.pending_large_pastes[0].placeholder, base);
    assert_eq!(
        app.editor.pending_large_pastes[1].placeholder,
        format!("{base} #2")
    );
}

#[test]
fn prepared_turn_history_text_annotates_only_pasted_content_inline() {
    let large = format!("alpha {}\nomega", "x".repeat(480));
    let char_count = large.chars().count();
    let placeholder = format!("[Pasted Content {char_count} chars]");
    let turn = PreparedTurn::prepare_with_large_pastes(
        format!("before {placeholder} after"),
        Vec::new(),
        &SkillCatalog::default(),
        vec![LargePaste {
            placeholder: placeholder.clone(),
            content: large,
        }],
    );

    let history = turn.history_text();
    assert!(history.starts_with("before "));
    assert!(history.ends_with(" after"));
    assert!(history.contains(&placeholder));
    assert!(!history.contains('\n'));
}

#[test]
fn prepared_turn_display_text_keeps_same_large_paste_placeholder_as_input() {
    let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 9);
    let placeholder = format!("[Pasted Content {} chars]", large.chars().count());

    let turn = PreparedTurn::prepare_with_large_pastes(
        format!("before {placeholder} after"),
        Vec::new(),
        &SkillCatalog::default(),
        vec![LargePaste {
            placeholder: placeholder.clone(),
            content: large.clone(),
        }],
    );

    assert_eq!(turn.display_text, format!("before {placeholder} after"));
    assert_eq!(turn.history_text(), format!("before {placeholder} after"));
    assert_eq!(turn.effective_text, format!("before {large} after"));
}

#[test]
fn prepared_turn_history_text_keeps_long_user_text_without_middle_truncation() {
    let text = format!(
        "I would like to add some kind of all knowing knowledge graph to figments. {} user facing documentation.",
        "x".repeat(400)
    );
    let turn = PreparedTurn::prepare(text.clone(), Vec::new(), &SkillCatalog::default());

    let history = turn.history_text();
    assert_eq!(history, text);
    assert!(!history.contains("chars hidden"));
}

#[test]
fn prompt_insert_text_inserts_literal_payload_at_cursor() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.show_prompt(PromptState {
        request: lash::PromptRequest::freeform("Question?"),
        focus: crate::overlay::PromptFocus::Text,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: "startend".into(),
        reply_cursor: "start".len(),
        response_tx: std::sync::mpsc::channel().0,
    });

    app.prompt_insert_text("\nplain pasted text\n");

    let prompt = app.prompt_state().expect("prompt");
    assert_eq!(prompt.reply_text, "start\nplain pasted text\nend");
    assert_eq!(prompt.reply_cursor, "start\nplain pasted text\n".len());
}

#[test]
fn prompt_toggle_current_option_ignores_freeform_prompts() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.show_prompt(PromptState {
        request: lash::PromptRequest::freeform("Question?"),
        focus: crate::overlay::PromptFocus::Text,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: String::new(),
        reply_cursor: 0,
        response_tx: std::sync::mpsc::channel().0,
    });

    app.prompt_toggle_current_option();

    assert!(
        app.prompt_state().expect("prompt").is_freeform(),
        "freeform prompts should stay in text-entry mode"
    );
}

#[test]
fn prompt_toggle_note_focus_switches_between_choices_and_note() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.show_prompt(PromptState {
        request: lash::PromptRequest::single("Pick one", vec!["red".into(), "blue".into()])
            .with_optional_note(),
        focus: crate::overlay::PromptFocus::Options,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: String::new(),
        reply_cursor: 0,
        response_tx: std::sync::mpsc::channel().0,
    });

    assert!(app.prompt_supports_note());
    assert!(!app.is_prompt_text_entry());

    app.prompt_toggle_note_focus();
    assert!(app.is_prompt_text_entry());

    app.prompt_toggle_note_focus();
    assert!(!app.is_prompt_text_entry());
}

#[test]
fn take_prompt_response_defers_option_prompt_display() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let (tx, rx) = std::sync::mpsc::channel();
    app.show_prompt(PromptState {
        request: lash::PromptRequest::single("Pick one", vec!["red".into(), "blue".into()])
            .with_optional_note(),
        focus: crate::overlay::PromptFocus::Text,
        cursor: 1,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: "ship the blue path".into(),
        reply_cursor: "ship the blue path".len(),
        response_tx: tx,
    });

    let response = app.take_prompt_response();

    assert_eq!(
        response.as_deref(),
        Some("2. blue\n\nNote: ship the blue path")
    );
    assert_eq!(
        rx.recv().expect("response"),
        lash::PromptResponse::Single {
            selection: "blue".to_string(),
            note: Some("ship the blue path".to_string()),
        }
    );
    assert!(
        !app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::UserInput(_)))
    );
}

#[test]
fn option_prompt_response_falls_back_to_user_block_without_inline_panel() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let (tx, _rx) = std::sync::mpsc::channel();
    app.show_prompt(PromptState {
        request: lash::PromptRequest::single("Pick one", vec!["red".into(), "blue".into()]),
        focus: crate::overlay::PromptFocus::Options,
        cursor: 0,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: String::new(),
        reply_cursor: 0,
        response_tx: tx,
    });

    let response = app.take_prompt_response();
    assert_eq!(response.as_deref(), Some("1. red"));

    app.handle_session_event(SessionEvent::ToolCall {
        call_id: None,
        name: "search_tools".into(),
        args: serde_json::json!({ "query": "queue" }),
        result: serde_json::json!([]),
        success: true,
        duration_ms: 1,
    });

    assert!(
        app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::UserInput(text) if text == "1. red"))
    );
}

#[test]
fn option_prompt_response_is_rendered_inline_by_question_panel_artifact() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let (tx, _rx) = std::sync::mpsc::channel();
    app.show_prompt(PromptState {
        request: lash::PromptRequest::single("Pick one", vec!["red".into(), "blue".into()])
            .with_optional_note(),
        focus: crate::overlay::PromptFocus::Text,
        cursor: 1,
        scroll_offset: 0,
        selected: Default::default(),
        reply_text: "ship the blue path".into(),
        reply_cursor: "ship the blue path".len(),
        response_tx: tx,
    });

    let response = app.take_prompt_response();
    assert_eq!(
        response.as_deref(),
        Some("2. blue\n\nNote: ship the blue path")
    );

    app.handle_session_event(SessionEvent::ToolCall {
        call_id: None,
        name: "ask".into(),
        args: serde_json::json!({
            "question": "Pick one",
            "options": ["red", "blue"]
        }),
        result: serde_json::json!({
            "kind": "single",
            "selection": "blue",
            "note": "ship the blue path"
        }),
        success: true,
        duration_ms: 1,
    });

    assert!(
        !app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::UserInput(_)))
    );
    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::Activity(activity))
            if matches!(
                activity.result.artifact.as_ref(),
                Some(ActivityArtifact::QuestionPanel(panel))
                    if panel.options.len() == 2
                        && !panel.options[0].selected
                        && panel.options[1].selected
                        && panel.note.as_deref() == Some("ship the blue path")
            )
    ));
}

#[test]
fn live_batch_tool_call_expands_children_without_parent_batch_block() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.clear();

    app.handle_session_event(SessionEvent::ToolCall {
        call_id: None,
        name: "batch".into(),
        args: serde_json::json!({
            "tool_uses": [
                {
                    "recipient_name": "functions.read_file",
                    "parameters": {"path": "README.md"}
                },
                {
                    "recipient_name": "functions.grep",
                    "parameters": {"query": "OpenAI", "path": "README.md"}
                }
            ]
        }),
        result: serde_json::json!([
            {"success": true, "result": "README body", "duration_ms": 8},
            {"success": true, "result": "match", "duration_ms": 13}
        ]),
        success: true,
        duration_ms: 21,
    });

    let activities = app
        .timeline
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::Activity(activity) => Some(activity),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(activities.len(), 1);
    assert_eq!(activities[0].call.kind, ActivityKind::Exploration);
    assert_eq!(activities[0].call.summary, "Explored");
    assert!(
        activities
            .iter()
            .all(|activity| activity.call.tool_name != "batch" && activity.call.summary != "batch")
    );
}

#[test]
fn at_completion_finds_nested_file_via_fuzzy() {
    use std::fs;
    use tempfile::TempDir;

    let dir = TempDir::new().expect("tempdir");
    fs::create_dir_all(dir.path().join("lash-cli/src/interactive")).unwrap();
    fs::create_dir_all(dir.path().join("lash-cli/src")).unwrap();
    fs::create_dir_all(dir.path().join("other/dir")).unwrap();
    fs::write(
        dir.path()
            .join("lash-cli/src/interactive/input_handling.rs"),
        "",
    )
    .unwrap();
    fs::write(dir.path().join("lash-cli/src/editor.rs"), "").unwrap();
    fs::write(dir.path().join("other/dir/unrelated.rs"), "").unwrap();

    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let index = lash_file_index::FileIndex::for_root_blocking(dir.path().to_path_buf());
    app.install_file_index(index);

    app.set_input("@input_handling".into());
    app.editor.cursor_pos = app.input().len();
    app.update_suggestions();

    assert_eq!(
        app.suggestion_kind(),
        crate::editor::SuggestionKind::Path,
        "expected Path suggestion kind, got {:?}",
        app.suggestion_kind()
    );
    let suggestions: Vec<&str> = app.suggestions().iter().map(|s| s.name.as_str()).collect();
    assert!(
        suggestions.contains(&"lash-cli/src/interactive/input_handling.rs"),
        "expected fuzzy match to surface input_handling.rs; got {suggestions:?}"
    );
}

#[test]
fn at_completion_with_no_index_disables_popup() {
    // Wholehog: without an installed FileIndex, `@`-completion is silent.
    // No fallback to the old prefix-only directory listing.
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("@anything".into());
    app.editor.cursor_pos = app.input().len();
    app.update_suggestions();

    assert!(
        app.suggestions().is_empty(),
        "expected no suggestions without installed index, got {:?}",
        app.suggestions()
    );
    assert_eq!(app.suggestion_kind(), crate::editor::SuggestionKind::None);
}

#[test]
fn at_completion_threads_match_indices_for_highlighting() {
    // Match indices from nucleo are plumbed through editor::Suggestion so the
    // popup can bold matched characters. Without them the highlight-on-match
    // feature is dead code.
    use std::fs;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join("src")).unwrap();
    fs::write(dir.path().join("src/handler.rs"), "").unwrap();

    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let index = lash_file_index::FileIndex::for_root_blocking(dir.path().to_path_buf());
    app.install_file_index(index);

    app.set_input("@handler".into());
    app.editor.cursor_pos = app.input().len();
    app.update_suggestions();

    let suggestion = app
        .suggestions()
        .iter()
        .find(|s| s.name == "src/handler.rs")
        .expect("expected src/handler.rs in suggestions");
    assert!(
        !suggestion.match_indices.is_empty(),
        "expected non-empty match_indices for fuzzy match, got {:?}",
        suggestion.match_indices
    );
    // Indices reference char offsets within `name`, so all must be in range.
    let name_len = suggestion.name.chars().count() as u32;
    assert!(
        suggestion.match_indices.iter().all(|&i| i < name_len),
        "match_indices out of bounds for {:?}: {:?}",
        suggestion.name,
        suggestion.match_indices
    );
}
