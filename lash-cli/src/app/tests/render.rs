use super::*;

#[test]
fn background_subagent_terminal_state_is_transient_and_freezes_duration() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let created_at = std::time::SystemTime::now() - std::time::Duration::from_secs(125);
    let mut running = lash_core::BackgroundTaskRecord::local_session(
        "test-session-id",
        "subagent:smoke",
        lash_core::BackgroundTaskKind::Subagent,
        "subagent",
        lash_core::BackgroundTaskState::Running,
    );
    running.created_at = created_at;
    app.update_background_tasks(vec![running]);
    assert_eq!(app.background_tasks.len(), 1);
    assert_eq!(app.background_tasks[0].terminal_duration, None);

    let mut completed = lash_core::BackgroundTaskRecord::local_session(
        "test-session-id",
        "subagent:smoke",
        lash_core::BackgroundTaskKind::Subagent,
        "subagent",
        lash_core::BackgroundTaskState::Completed,
    );
    completed.created_at = created_at;
    app.update_background_tasks(vec![completed]);

    assert_eq!(app.background_tasks.len(), 1);
    assert_eq!(
        app.background_tasks[0].state,
        lash_core::BackgroundTaskState::Completed
    );
    assert!(app.background_tasks[0].transient_until.is_some());
    assert!(
        app.background_tasks[0]
            .terminal_duration
            .is_some_and(|duration| duration.as_secs() >= 125)
    );
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
        output: lash_core::ToolCallOutput::success(serde_json::json!("ok")),
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
        output: lash_core::ToolCallOutput::success(serde_json::json!("ok")),
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
        event: lash_core::PluginSurfaceEvent::PanelUpsert {
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
        event: lash_core::PluginSurfaceEvent::Status {
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
        event: lash_core::PluginSurfaceEvent::PanelUpsert {
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
        event: lash_core::PluginSurfaceEvent::PanelClear {
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
fn cancelled_error_renders_as_system_message() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.note_manual_interrupt_requested();
    app.handle_session_event(SessionEvent::Error {
        message: "LLM error: cancelled".into(),
        envelope: Some(lash_core::session_model::ErrorEnvelope {
            kind: "llm_provider".into(),
            code: Some("cancelled".into()),
            terminal_reason: None,
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
        envelope: Some(lash_core::session_model::ErrorEnvelope {
            kind: "llm_provider".into(),
            code: Some("cancelled".into()),
            terminal_reason: None,
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
        envelope: Some(lash_core::session_model::ErrorEnvelope {
            kind: "llm_provider".into(),
            code: Some("cancelled".into()),
            terminal_reason: None,
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
        output: lash_core::ToolCallOutput::success(serde_json::json!("match")),
        duration_ms: 10,
    });
    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc4".into()),
        name: "read_file".into(),
        args: serde_json::json!({"path": "lash-cli/src/render/mod.rs"}),
        output: lash_core::ToolCallOutput::success(serde_json::json!(
            "==> lash-cli/src/render/mod.rs <==\nline"
        )),
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
        output: lash_core::ToolCallOutput::success(serde_json::json!({
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
        })),
        duration_ms: 7,
    });
    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc6".into()),
        name: "apply_patch".into(),
        args: serde_json::json!({}),
        output: lash_core::ToolCallOutput::success(serde_json::json!({
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
        })),
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
        output: lash_core::ToolCallOutput::success(serde_json::json!([
            {"success": true, "result": "README body", "duration_ms": 8},
            {"success": true, "result": "match", "duration_ms": 13}
        ])),
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
