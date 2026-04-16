use super::*;
use crate::editor::LARGE_PASTE_CHAR_THRESHOLD;
use async_trait::async_trait;
use lash_ui::{SlashCommandSpec, UiContext, UiExtension, UiExtensions, UiHostEffect};
use std::sync::Arc;
fn other_variant_name(block: &DisplayBlock) -> &'static str {
    match block {
        DisplayBlock::TurnStart(_) => "TurnStart",
        DisplayBlock::UserInput(_) => "UserInput",
        DisplayBlock::AssistantText(_) => "AssistantText",
        DisplayBlock::Activity(_) => "Activity",
        DisplayBlock::ShellOutput { .. } => "ShellOutput",
        DisplayBlock::Error(_) => "Error",
        DisplayBlock::SystemMessage(_) => "SystemMessage",
        DisplayBlock::PlanContent(_) => "PlanContent",
        DisplayBlock::PluginPanel(_) => "PluginPanel",
        DisplayBlock::Splash => "Splash",
    }
}

#[test]
fn renders_plan_content_from_update_plan_args() {
    let content = render_plan_content_from_args(&serde_json::json!({
        "explanation": "Found the renderer.",
        "plan": [
            {"step":"Inspect UI", "status":"completed"},
            {"step":"Patch layout", "status":"in_progress"}
        ]
    }))
    .expect("plan content");
    assert!(content.contains("Found the renderer."));
    assert!(content.contains("\u{2713} Inspect UI"));
    assert!(content.contains("\u{25b8} Patch layout"));
    assert!(!content.contains("1."));
}

#[test]
fn text_delta_accumulates_raw() {
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "\n\nfirst\n".into(),
    });
    // Raw accumulation — no normalization until flush
    assert_eq!(
        app.live_assistant
            .as_ref()
            .map(|view| view.raw_text.as_str()),
        Some("\n\nfirst\n")
    );

    app.handle_session_event(SessionEvent::TextDelta {
        content: "\n\n\nsecond\n".into(),
    });
    assert_eq!(
        app.live_assistant
            .as_ref()
            .map(|view| view.raw_text.as_str()),
        Some("\n\nfirst\n\n\n\nsecond\n")
    );
}

#[test]
fn text_delta_code_fence_preserved() {
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "text\n\n```python\n".into(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: "# comment\n".into(),
    });
    // The newline between ```python and # comment must be preserved
    assert!(
        app.live_assistant
            .as_ref()
            .is_some_and(|view| view.raw_text.contains("```python\n# comment"))
    );
}

#[test]
fn text_delta_stays_in_live_assistant_until_committed() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::TextDelta {
        content: "Draft answer".into(),
    });

    assert!(matches!(app.blocks.last(), Some(DisplayBlock::Splash)));
    assert_eq!(
        app.live_assistant_normalized_text().as_deref(),
        Some("Draft answer")
    );
}

#[test]
fn ui_extension_commands_appear_in_editor_suggestions() {
    let mut app = App::new("test-model".into(), "test".into());
    let ui_extensions = lash_ui::UiExtensions::builtin().expect("ui extensions");
    app.set_ui_extensions(Arc::new(ui_extensions));
    app.set_input("/pl".into());

    app.update_suggestions();

    assert!(app.suggestions().iter().any(|(name, _)| name == "/plan"));
}

#[test]
fn ui_extension_argument_suggestions_complete_second_token() {
    struct DemoUiExtension;

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
    impl UiExtension for DemoUiExtension {
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
            _ctx: UiContext<'_>,
        ) -> Result<Vec<UiHostEffect>, String> {
            Ok(Vec::new())
        }
    }

    let mut app = App::new("test-model".into(), "test".into());
    let ui_extensions = UiExtensions::new(vec![Arc::new(DemoUiExtension)]).expect("ui extensions");
    app.set_ui_extensions(Arc::new(ui_extensions));
    app.set_input("/demo h".into());
    app.editor.cursor_pos = app.input().len();

    app.update_suggestions();

    assert_eq!(
        app.suggestions().first().map(|value| value.0.as_str()),
        Some("help")
    );
    app.complete_suggestion();

    assert_eq!(app.input(), "/demo help");
}

#[test]
fn final_message_never_replaces_visible_streamed_text_with_shorter_text() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "Visible streamed text".into(),
    });

    app.handle_session_event(SessionEvent::Message {
        text: "Visible".into(),
        kind: "final".into(),
    });

    assert!(matches!(
        app.blocks.last(),
        Some(DisplayBlock::AssistantText(text)) if text == "Visible streamed text"
    ));
}

#[test]
fn text_delta_updates_live_token_estimate() {
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        iteration: 0,
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
fn late_text_deltas_after_stop_turn_extend_last_assistant_block() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "I".into(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: "’m".into(),
    });

    app.stop_turn();

    app.handle_session_event(SessionEvent::TextDelta {
        content: " an".into(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: " AI".into(),
    });

    let assistant_blocks: Vec<&str> = app
        .blocks
        .iter()
        .filter_map(|block| match block {
            DisplayBlock::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(assistant_blocks, vec!["I’m an AI"]);
}

#[test]
fn first_text_delta_switches_thinking_to_responding() {
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        iteration: 0,
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
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        iteration: 0,
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
    let mut app = App::new("test-model".into(), "test".into());
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
        iteration: 1,
        message_count: 0,
        tool_list: String::new(),
    });

    assert!(app.live_assistant.is_none());
    assert!(app.blocks.iter().any(|block| {
        matches!(block, DisplayBlock::AssistantText(text) if text == "Let me continue testing.")
    }));
}

#[test]
fn tool_call_flushes_intermediate_stream_text_immediately() {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();

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

    assert!(app.live_assistant.is_none());
    assert!(matches!(
        app.blocks.first(),
        Some(DisplayBlock::AssistantText(text))
            if text == "I’m checking the rendering path first."
    ));
    assert!(matches!(app.blocks.get(1), Some(DisplayBlock::Activity(_))));
}

#[test]
fn token_usage_resets_live_token_estimate() {
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "abcdefgh".into(),
    });
    assert!(app.live_output_tokens_estimate > 0);
    app.handle_session_event(SessionEvent::TokenUsage {
        iteration: 0,
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
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "abcdefgh".into(),
    });
    let live_estimate = app.live_output_tokens_estimate;
    assert!(live_estimate > 0);
    app.handle_session_event(SessionEvent::TokenUsage {
        iteration: 0,
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
fn final_message_event_is_rendered() {
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::Message {
        text: "final output".into(),
        kind: "final".into(),
    });
    assert!(matches!(
        app.blocks.last(),
        Some(DisplayBlock::AssistantText(text)) if text == "final output"
    ));
}

#[test]
fn finish_turn_for_resume_reconciles_authoritative_assistant_text() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "I looked at the actual librarian prompt".into(),
    });
    app.stop_turn();

    let persisted = app.finish_turn_for_resume_with_output(Some(
        "I looked at the actual librarian prompt, the graph tool constraints.\n\n## What exists now",
    ));

    let last_block = app
        .blocks
        .iter()
        .rev()
        .find_map(|block| match block {
            DisplayBlock::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .expect("assistant block");
    assert_eq!(
        last_block,
        "I looked at the actual librarian prompt, the graph tool constraints.\n\n## What exists now"
    );
    assert!(persisted.plugin_panels.is_empty());
}

#[test]
fn finish_turn_for_resume_does_not_append_shorter_authoritative_text() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "Visible streamed text".into(),
    });
    app.stop_turn();

    let persisted = app.finish_turn_for_resume_with_output(Some("Visible"));

    let last_block = app
        .blocks
        .iter()
        .rev()
        .find_map(|block| match block {
            DisplayBlock::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .expect("assistant block");
    assert_eq!(last_block, "Visible streamed text");
    assert!(persisted.plugin_panels.is_empty());
}

#[test]
fn late_text_deltas_after_authoritative_final_output_are_ignored() {
    let mut app = App::new("test-model".into(), "test".into());
    let final_text = "Use this minimal set:\n\n- `code`\n- `feature`\n- `issue`\n- `decision`\n\nThat’s probably the sweet spot.";
    app.start_turn();
    app.handle_session_event(SessionEvent::TextDelta {
        content: "Use this minimal set:\n\n- `code`\n- `feature`\n".into(),
    });

    let _persisted = app.finish_turn_for_resume_with_output(Some(final_text));

    app.handle_session_event(SessionEvent::TextDelta {
        content: "Yeah — **`feature` is nicer than `topic`** if you want the graph to stay product-shaped.\n\nMy take:\n\n- **`topic` is safer**".into(),
    });

    let assistant_blocks: Vec<&str> = app
        .blocks
        .iter()
        .filter_map(|block| match block {
            DisplayBlock::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(assistant_blocks, vec![final_text]);
}

#[test]
fn tool_output_renders_during_generic_running_turn() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::Message {
        text: "started git status --short\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(
        app.streaming_output,
        vec!["started git status --short".to_string()]
    );
}

#[test]
fn tool_output_carriage_return_rewrites_partial_line() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "Compiling alpha".into(),
        kind: "tool_output".into(),
    });
    assert_eq!(app.streaming_output_partial, "Compiling alpha");

    app.handle_session_event(SessionEvent::Message {
        text: "\rCompiling beta".into(),
        kind: "tool_output".into(),
    });

    assert!(app.streaming_output.is_empty());
    assert_eq!(app.streaming_output_partial, "Compiling beta");
}

#[test]
fn tool_output_crlf_commits_current_line() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "started cargo check\r\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(
        app.streaming_output,
        vec!["started cargo check".to_string()]
    );
    assert!(app.streaming_output_partial.is_empty());
}

#[test]
fn tool_output_strips_ansi_escape_sequences_from_live_preview() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "\u{1b}[33mwarning\u{1b}[0m: check this\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(
        app.streaming_output,
        vec!["warning: check this".to_string()]
    );
    assert!(app.streaming_output_partial.is_empty());
}

#[test]
fn tool_output_strips_osc_escape_sequences_from_live_preview() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "\u{1b}]11;?\u{1b}\\".into(),
        kind: "tool_output".into(),
    });
    app.handle_session_event(SessionEvent::Message {
        text: "done\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(app.streaming_output, vec!["done".to_string()]);
    assert!(app.streaming_output_partial.is_empty());
}

#[test]
fn tool_output_tabs_collapse_to_single_spaces_in_live_preview() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();

    app.handle_session_event(SessionEvent::Message {
        text: "hash\trefs/tags/v0.2.29\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(
        app.streaming_output,
        vec!["hash refs/tags/v0.2.29".to_string()]
    );
    assert!(app.streaming_output_partial.is_empty());
}

#[test]
fn tool_output_does_not_change_total_content_height() {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks = vec![DisplayBlock::UserInput("inspect this".into())];
    app.start_turn();

    let baseline = app.total_content_height(32, 8);
    app.handle_session_event(SessionEvent::Message {
        text: "started git status --short\n".into(),
        kind: "tool_output".into(),
    });

    assert_eq!(app.total_content_height(32, 8), baseline);
}

#[test]
fn finish_turn_for_resume_preserves_streaming_output_snapshot() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::Message {
        text: "started git status --short\n".into(),
        kind: "tool_output".into(),
    });

    let persisted = app.finish_turn_for_resume_with_output(None);

    assert!(!app.running);
    assert!(app.streaming_output.is_empty());
    assert_eq!(
        persisted.streaming_output,
        vec!["started git status --short".to_string()]
    );
}

#[test]
fn plugin_panel_events_upsert_and_clear_blocks() {
    let mut app = App::new("test-model".into(), "test".into());
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
        app.blocks.last(),
        Some(DisplayBlock::PluginPanel(panel)) if panel.title == "TASK BOARD"
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
        !app.blocks
            .iter()
            .any(|block| matches!(block, DisplayBlock::PluginPanel(_)))
    );
}

#[test]
fn plan_exit_tool_does_not_queue_follow_up_turn() {
    let mut app = App::new("test-model".into(), "test".into());
    let ui_extensions = lash_ui::UiExtensions::builtin().expect("builtin ui extensions");
    crate::apply_ui_host_effects(
        &mut app,
        ui_extensions.effects_for_session_event(&SessionEvent::ToolCall {
            call_id: Some("tc-plan-exit".into()),
            name: "plan_exit".into(),
            args: serde_json::json!({}),
            result: serde_json::json!({
                "approved": true,
                "plan_path": ".lash/plans/session.md",
                "next_turn_input": "Execute the plan in `.lash/plans/session.md`."
            }),
            success: true,
            duration_ms: 5,
        }),
    );

    assert!(app.take_next_queued_turn().is_none());
}

#[test]
fn plan_exit_fresh_context_tool_does_not_queue_follow_up_turn() {
    let mut app = App::new("test-model".into(), "test".into());
    let ui_extensions = lash_ui::UiExtensions::builtin().expect("builtin ui extensions");
    crate::apply_ui_host_effects(
        &mut app,
        ui_extensions.effects_for_session_event(&SessionEvent::ToolCall {
            call_id: Some("tc-plan-exit-fresh".into()),
            name: "plan_exit".into(),
            args: serde_json::json!({}),
            result: serde_json::json!({
                "approved": true,
                "execution_mode": "fresh_context",
                "session_id": "new-plan-session",
                "fresh_context_input": "Do a full, faithful implementation of the plan found at: .lash/plans/session.md"
            }),
            success: true,
            duration_ms: 5,
        }),
    );

    assert!(app.take_next_queued_turn().is_none());
}

#[test]
fn cancelled_error_renders_as_system_message() {
    let mut app = App::new("test-model".into(), "test".into());
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
        app.blocks.last(),
        Some(DisplayBlock::SystemMessage(msg)) if msg == "Manually interrupted."
    ));
    assert!(!app.running);
    assert!(app.live_turn.is_none());
}

#[test]
fn cancelled_error_without_manual_request_still_stops_immediately() {
    let mut app = App::new("test-model".into(), "test".into());
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
        app.blocks.last(),
        Some(DisplayBlock::SystemMessage(msg)) if msg == "Cancelled."
    ));
    assert!(!app.running);
    assert!(app.live_turn.is_none());
}

#[test]
fn repeated_cancelled_errors_do_not_duplicate_system_message() {
    let mut app = App::new("test-model".into(), "test".into());
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
        .blocks
        .iter()
        .filter(|block| {
            matches!(
                block,
                DisplayBlock::SystemMessage(msg) if msg == "Cancelled."
            )
        })
        .count();
    assert_eq!(cancelled_blocks, 1);
}

#[test]
fn non_manual_error_sets_transient_status() {
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        iteration: 0,
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
fn transient_status_expires_on_tick() {
    let mut app = App::new("test-model".into(), "test".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        iteration: 0,
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
fn ui_resume_state_omits_transient_live_turn() {
    let mut app = App::new("test-model".into(), "test".into());
    app.start_turn();
    app.set_status("retrying", Some("in 5s".into()), true);
    if let Some(turn) = app.live_turn.as_mut() {
        turn.has_visible_output = true;
    }

    let persisted = serde_json::to_value(app.ui_resume_state()).expect("serialize ui");
    assert!(persisted.get("live_turn").is_none());
}

#[test]
fn queued_turns_are_fifo_and_skip_pending_injections() {
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
fn accepted_injected_turn_input_renders_matching_pending_steer() {
    let mut app = App::new("test-model".into(), "test".into());
    let turn = PreparedTurn::new("follow up".into(), Vec::new());
    app.queue_pending_steer(turn.clone());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        messages: vec![PluginMessage::text(MessageRole::User, "follow up")],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    assert!(app.pending_steers.is_empty());
    assert!(
        app.blocks
            .iter()
            .any(|block| matches!(block, DisplayBlock::UserInput(text) if text == "follow up"))
    );
}

#[test]
fn accepted_injected_turn_input_matches_by_runtime_content_even_when_display_text_differs() {
    let mut app = App::new("test-model".into(), "test".into());
    let mut turn = PreparedTurn::new("/localref lash for context if needed".into(), Vec::new());
    turn.input_provenance.transforms = vec![lash::UserInputTransform::SkillBlockAppend {
        skill_name: "localref".into(),
        skill_path: "/tmp/localref/SKILL.md".into(),
    }];
    app.queue_pending_steer(turn.clone());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        messages: vec![PluginMessage::text(
            MessageRole::User,
            "/localref lash for context if needed\n\n<skill>\n<name>localref</name>\nbody\n</skill>",
        )],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    assert!(app.pending_steers.is_empty());
    assert!(app.blocks.iter().any(|block| matches!(
        block,
        DisplayBlock::UserInput(text) if text == "/localref lash for context if needed"
    )));
}

#[test]
fn accepted_injected_turn_input_without_pending_match_still_renders_once() {
    let mut app = App::new("test-model".into(), "test".into());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        messages: vec![PluginMessage {
            role: MessageRole::User,
            content: "runtime content".into(),
            parts: Vec::new(),
            images: Vec::new(),
            user_input: Some(lash::UserInputProvenance {
                display_text: "visible text".into(),
                effective_text: "runtime content".into(),
                transforms: Vec::new(),
            }),
        }],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    app.handle_session_event(SessionEvent::InjectedMessagesCommitted {
        messages: vec![PluginMessage {
            role: MessageRole::User,
            content: "runtime content".into(),
            parts: Vec::new(),
            images: Vec::new(),
            user_input: Some(lash::UserInputProvenance {
                display_text: "visible text".into(),
                effective_text: "runtime content".into(),
                transforms: Vec::new(),
            }),
        }],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    let matching_blocks = app
        .blocks
        .iter()
        .filter(|block| matches!(block, DisplayBlock::UserInput(text) if text == "visible text"))
        .count();
    assert_eq!(matching_blocks, 1);
}

#[test]
fn accepted_injected_turn_input_removes_matching_pending_steer_without_popping_wrong_one() {
    let mut app = App::new("test-model".into(), "test".into());
    app.queue_pending_steer(PreparedTurn::new("first queued steer".into(), Vec::new()));
    app.queue_pending_steer(PreparedTurn::new(
        "uhh do not switch nvm".into(),
        Vec::new(),
    ));

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        messages: vec![PluginMessage::text(
            MessageRole::User,
            "uhh do not switch nvm",
        )],
        checkpoint: lash::CheckpointKind::AfterWork,
    });

    assert_eq!(app.pending_steers.len(), 1);
    assert_eq!(app.pending_steers[0].display_text, "first queued steer");
}

#[test]
fn injected_messages_committed_do_not_duplicate_existing_visible_user_input() {
    let mut app = App::new("test-model".into(), "test".into());
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
        .blocks
        .iter()
        .filter(|block| {
            matches!(
                block,
                DisplayBlock::UserInput(text)
                    if text == "(I want future migrations to work though!)"
            )
        })
        .count();
    assert_eq!(matching_blocks, 1);
    assert!(app.pending_steers.is_empty());
}

#[test]
fn queued_injection_stays_out_of_history_until_committed() {
    let mut app = App::new("test-model".into(), "test".into());
    let turn = PreparedTurn::new("follow up now".into(), Vec::new());

    app.queue_pending_steer(turn.clone());

    assert_eq!(app.pending_steers.len(), 1);
    assert!(!matches!(
        app.blocks.last(),
        Some(DisplayBlock::UserInput(_))
    ));
}

#[test]
fn regular_queued_turn_stays_out_of_history_until_dispatched() {
    let mut app = App::new("test-model".into(), "test".into());
    let turn = PreparedTurn::new("queued text".into(), Vec::new());
    app.queue_turn(turn);

    assert_eq!(app.queued_turns.len(), 1);
    assert!(!matches!(
        app.blocks.last(),
        Some(DisplayBlock::UserInput(_))
    ));
}

#[test]
fn history_up_restores_last_queued_turn_before_history() {
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
    app.set_input("[Image #2] [Image #5]".into());
    app.editor.pending_images = vec![PendingImage {
        id: 2,
        png_bytes: vec![1, 2, 3],
    }];

    assert_eq!(app.next_image_marker_id(), 6);
}

#[test]
fn add_pending_image_uses_highest_marker_plus_one() {
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
    app.set_input("before [Image #7] after".into());
    app.editor.cursor_pos = app.input().len();
    app.begin_pending_image(7);

    assert!(app.fail_pending_image(7));
    assert_eq!(app.input(), "before  after");
    assert!(!app.editor.inflight_image_ids.contains(&7));
}

#[test]
fn pending_image_jobs_only_count_visible_markers() {
    let mut app = App::new("test-model".into(), "test".into());
    app.begin_pending_image(2);
    assert!(!app.has_pending_image_jobs());

    app.set_input("[Image #2]".into());
    app.editor.cursor_pos = app.input().len();
    assert!(app.has_pending_image_jobs());

    app.backspace();
    assert!(!app.has_pending_image_jobs());
}

#[test]
fn take_prompt_response_renders_visible_user_block() {
    let mut app = App::new("test-model".into(), "test".into());
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
        app.blocks.last(),
        Some(DisplayBlock::UserInput(text)) if text == "red"
    ));
}

#[test]
fn dismiss_prompt_marks_ui_dirty() {
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();
    for idx in 0..6 {
        app.blocks
            .push(DisplayBlock::AssistantText(format!("history {idx}")));
    }
    app.blocks.push(DisplayBlock::UserInput(
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

    let last_idx = app.blocks.len() - 1;
    let max_scroll = app
        .total_content_height(width, viewport_height)
        .saturating_sub(viewport_height);
    let expected =
        app.contextual_follow_offset(app.block_content_start_offset(last_idx), max_scroll);
    assert_eq!(app.scroll_offset, expected);
}

#[test]
fn keep_latest_user_block_visible_keeps_short_prompt_bottom_aligned() {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();
    for idx in 0..6 {
        app.blocks
            .push(DisplayBlock::AssistantText(format!("history {idx}")));
    }
    app.blocks
        .push(DisplayBlock::UserInput("short prompt".into()));
    app.running = true;
    app.follow_mode = FollowOutputMode::Contextual;

    let width = 32usize;
    let viewport_height = 4usize;
    app.ensure_height_cache_pub(width, viewport_height);
    app.scroll_offset = usize::MAX;

    app.keep_latest_user_block_visible();

    let cache = app.height_cache_snapshot().to_vec();
    let last_idx = app.blocks.len() - 1;
    let block_end = cache[last_idx];
    assert_eq!(app.scroll_offset, block_end.saturating_sub(viewport_height));
}

#[test]
fn splash_collapses_to_compact_scrollback_height_once_history_exists() {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks
        .push(DisplayBlock::UserInput("short prompt".into()));

    let width = 32usize;
    let viewport_height = 12usize;
    app.ensure_height_cache_pub(width, viewport_height);

    let cache = app.height_cache_snapshot().to_vec();
    assert_eq!(cache[0], SPLASH_SCROLLBACK_HEIGHT);
}

#[test]
fn dismiss_splash_removes_empty_state_before_history_content() {
    let mut app = App::new("test-model".into(), "test".into());
    app.dismiss_splash();

    assert!(app.blocks.is_empty());

    app.blocks.push(DisplayBlock::UserInput("hello".into()));
    app.dismiss_splash();

    assert_eq!(app.blocks.len(), 1);
    assert!(matches!(app.blocks[0], DisplayBlock::UserInput(_)));
}

#[test]
fn refresh_follow_output_anchor_tracks_bottom_when_idle() {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();

    app.blocks.push(DisplayBlock::UserInput("Message 1".into()));
    app.blocks.push(DisplayBlock::AssistantText(
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();

    app.blocks.push(DisplayBlock::UserInput("Message 1".into()));
    app.blocks.push(DisplayBlock::AssistantText(
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();
    app.blocks.push(DisplayBlock::UserInput("hello".into()));
    app.blocks.push(DisplayBlock::AssistantText("world".into()));
    app.follow_mode = FollowOutputMode::Paused;
    app.scroll_offset = 3;

    app.resume_follow_output();

    assert_eq!(app.follow_mode, FollowOutputMode::Bottom);
    assert_eq!(app.scroll_offset, usize::MAX);
}

#[test]
fn scroll_up_from_follow_output_detaches_from_bottom_anchor() {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();
    app.blocks.push(DisplayBlock::UserInput("hello".into()));
    app.blocks.push(DisplayBlock::AssistantText(
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();
    app.blocks
        .push(DisplayBlock::AssistantText("older history".into()));
    app.blocks.push(DisplayBlock::UserInput("prompt".into()));
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();
    app.blocks.push(DisplayBlock::UserInput("prompt".into()));
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();
    app.blocks.push(DisplayBlock::UserInput("prompt".into()));
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.push(DisplayBlock::UserInput(
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();

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

    assert_eq!(app.blocks.len(), 1);
    match &app.blocks[0] {
        DisplayBlock::Activity(activity) => {
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
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks.clear();

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

    assert_eq!(app.blocks.len(), 1);
    match &app.blocks[0] {
        DisplayBlock::Activity(activity) => {
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
    let mut app = App::new("test-model".into(), "test".into());
    app.set_input("startend".into());
    app.editor.cursor_pos = "start".len();

    app.insert_text("\nplain pasted text\n");

    assert_eq!(app.input(), "start\nplain pasted text\nend");
    assert_eq!(app.cursor_pos(), "start\nplain pasted text\n".len());
}

#[test]
fn insert_pasted_text_large_uses_placeholder_and_prepare_expands() {
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
fn wait_overlay_resume_returns_resume_early_token_without_chat_echo() {
    let mut app = App::new("test-model".into(), "test".into());
    let (response_tx, response_rx) = std::sync::mpsc::channel();
    app.show_wait(WaitState::from_request(
        lash::PromptRequest::freeform("Pausing briefly before continuing.").with_wait(5),
        response_tx,
    ));

    app.resume_wait();

    assert!(!app.has_wait());
    assert!(!app.blocks.iter().any(
        |block| matches!(block, DisplayBlock::UserInput(text) if text.contains("Pausing briefly"))
    ));
    assert_eq!(
        response_rx.recv().expect("response"),
        lash::PromptResponse::Text {
            text: lash::WAIT_PROMPT_RESUME_EARLY_TOKEN.to_string()
        }
    );
}

#[test]
fn wait_overlay_reports_countdown_and_timeout_response() {
    let mut app = App::new("test-model".into(), "test".into());
    let (response_tx, response_rx) = std::sync::mpsc::channel();
    app.show_wait(WaitState::from_request(
        lash::PromptRequest::freeform("Pausing briefly before continuing.").with_wait(0),
        response_tx,
    ));

    assert_eq!(app.wait_remaining_seconds(), Some(0));
    assert!(app.wait_timed_out());

    app.timeout_wait();

    assert!(!app.has_wait());
    assert_eq!(
        response_rx.recv().expect("response"),
        lash::PromptResponse::Text {
            text: lash::WAIT_PROMPT_TIMEOUT_TOKEN.to_string()
        }
    );
}

#[test]
fn wait_overlay_starts_with_requested_remaining_seconds() {
    let mut app = App::new("test-model".into(), "test".into());
    app.show_wait(WaitState::from_request(
        lash::PromptRequest::freeform("Pausing briefly before continuing.").with_wait(5),
        std::sync::mpsc::channel().0,
    ));

    assert_eq!(app.wait_remaining_seconds(), Some(5));
    assert!(!app.wait_timed_out());
}

#[test]
fn prompt_toggle_current_option_ignores_freeform_prompts() {
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
    let mut app = App::new("test-model".into(), "test".into());
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
        !app.blocks
            .iter()
            .any(|block| matches!(block, DisplayBlock::UserInput(_)))
    );
}

#[test]
fn option_prompt_response_falls_back_to_user_block_without_inline_panel() {
    let mut app = App::new("test-model".into(), "test".into());
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
        app.blocks
            .iter()
            .any(|block| matches!(block, DisplayBlock::UserInput(text) if text == "1. red"))
    );
}

#[test]
fn option_prompt_response_is_rendered_inline_by_question_panel_artifact() {
    let mut app = App::new("test-model".into(), "test".into());
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
        !app.blocks
            .iter()
            .any(|block| matches!(block, DisplayBlock::UserInput(_)))
    );
    assert!(matches!(
        app.blocks.last(),
        Some(DisplayBlock::Activity(activity))
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
