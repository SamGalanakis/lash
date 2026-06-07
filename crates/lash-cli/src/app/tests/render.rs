use super::*;

#[test]
fn background_subagent_status_is_transient_and_freezes_duration() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let running = lash_core::ProcessHandleSummary {
        handle_type: "process".to_string(),
        id: "subagent:smoke".to_string(),
        process_id: "subagent:smoke".to_string(),
        descriptor: lash_core::ProcessHandleDescriptor::new(Some("subagent"), Some("smoke")),
        definition: None,
        status: lash_core::ProcessLifecycleStatus::Running,
    };
    app.update_processes(vec![running]);
    app.processes[0].first_seen = std::time::Instant::now() - std::time::Duration::from_secs(125);
    assert_eq!(app.processes.len(), 1);
    assert_eq!(app.processes[0].status_duration, None);

    let completed = lash_core::ProcessHandleSummary {
        handle_type: "process".to_string(),
        id: "subagent:smoke".to_string(),
        process_id: "subagent:smoke".to_string(),
        descriptor: lash_core::ProcessHandleDescriptor::new(Some("subagent"), Some("smoke")),
        definition: None,
        status: lash_core::ProcessLifecycleStatus::Completed,
    };
    app.update_processes(vec![completed]);

    assert_eq!(app.processes.len(), 1);
    assert_eq!(
        app.processes[0].status.terminal_state(),
        Some(lash_core::ProcessTerminalState::Completed)
    );
    assert!(app.processes[0].transient_until.is_some());
    assert!(
        app.processes[0]
            .status_duration
            .is_some_and(|duration| duration.as_secs() >= 125)
    );
}

#[test]
fn process_selection_tracks_visible_process_snapshot() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let running = lash_core::ProcessHandleSummary::new(
        "process-1",
        lash_core::ProcessHandleDescriptor::new(Some("lashlang"), Some("responder")),
        lash_core::ProcessLifecycleStatus::Running,
    )
    .with_definition(Some(lash_core::ProcessDefinitionSummary {
        name: "responder".into(),
    }));

    app.update_processes(vec![running.clone()]);
    assert!(app.select_next_process());
    assert_eq!(
        app.selected_process()
            .map(|process| process.process_id.as_str()),
        Some("process-1")
    );
    let overview = app
        .selected_process_overview_state()
        .expect("process overview");
    assert_eq!(overview.title, "Process responder");
    assert!(
        overview
            .rows
            .iter()
            .any(|(label, value)| { label == "definition" && value == "responder" })
    );

    app.update_processes(vec![running]);
    assert_eq!(
        app.selected_process()
            .map(|process| process.process_id.as_str()),
        Some("process-1")
    );

    app.update_processes(Vec::new());
    assert!(app.selected_process().is_none());
}

#[test]
fn selected_process_row_is_focusable_and_renders_definition() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.update_processes(vec![
        lash_core::ProcessHandleSummary::new(
            "process-1",
            lash_core::ProcessHandleDescriptor::new(Some("lashlang"), Some("responder")),
            lash_core::ProcessLifecycleStatus::Running,
        )
        .with_definition(Some(lash_core::ProcessDefinitionSummary {
            name: "responder".into(),
        })),
    ]);
    app.select_next_process();

    let lines = crate::render::process_lines_snapshot(&app, 80).expect("process lines");
    let text: Vec<String> = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect();

    assert!(text.iter().any(|line| {
        line.contains("▶ SELECTED") && line.contains("running · responder · responder")
    }));
    let selected_row = lines
        .iter()
        .find(|line| {
            line.spans
                .iter()
                .any(|span| span.content.as_ref().contains("SELECTED"))
        })
        .expect("selected process row");
    assert!(
        selected_row
            .spans
            .iter()
            .any(|span| span.style.bg == Some(crate::theme::SELECTION_BG)),
        "selected process row should carry a visible background highlight"
    );
}

#[test]
fn process_rows_are_input_chrome_not_history_content() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let baseline_height = app.total_content_height(80, 20);
    app.update_processes(vec![lash_core::ProcessHandleSummary::new(
        "process-1",
        lash_core::ProcessHandleDescriptor::new(Some("lashlang"), Some("responder")),
        lash_core::ProcessLifecycleStatus::Running,
    )]);

    assert_eq!(crate::render::process_dock_height(&app, 80), 2);
    assert_eq!(app.total_content_height(80, 20), baseline_height);

    let areas = crate::render::chrome_areas(&app, 80, 24);
    assert_eq!(areas.process.height, 2);
    assert_eq!(areas.process.y, areas.input.y + areas.input.height);
}

#[test]
fn stop_turn_marks_idle_redraw_dirty() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.dirty = false;

    app.stop_turn();

    assert_eq!(app.run_state, CliRunState::Idle);
    assert!(app.dirty);
}

#[test]
fn text_delta_accumulates_raw() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "\n\nfirst\n".into(),
    });
    assert_eq!(
        app.live.assistant.normalized_text().as_deref(),
        Some("first")
    );

    app.handle_session_event(SessionEvent::TextDelta {
        content: "\n\n\nsecond\n".into(),
    });
    assert_eq!(
        app.live.assistant.normalized_text().as_deref(),
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
        app.live
            .assistant
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

    assert!(app.timeline.is_empty());
    assert_eq!(
        app.live_assistant_normalized_text().as_deref(),
        Some("Draft answer")
    );
}

#[test]
fn text_delta_updates_live_token_estimate() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        protocol_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: "abcd".into(),
    });
    assert_eq!(app.usage.live_output_tokens_estimate, 1);
    app.handle_session_event(SessionEvent::TextDelta {
        content: "efgh".into(),
    });
    assert_eq!(app.usage.live_output_tokens_estimate, 2);
}

#[test]
fn first_text_delta_switches_thinking_to_responding() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        protocol_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::TextDelta {
        content: "hello".into(),
    });
    assert_eq!(
        app.live.turn.as_ref().map(|turn| turn.run_state),
        Some(CliRunState::Responding)
    );
    assert_eq!(
        app.live
            .turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        None
    );
}

#[test]
fn llm_request_sets_plain_thinking_status() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        protocol_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    assert_eq!(
        app.live.turn.as_ref().map(|turn| turn.run_state),
        Some(CliRunState::Thinking)
    );
    assert_eq!(
        app.live
            .turn
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
        protocol_iteration: 1,
        message_count: 0,
        tool_list: String::new(),
    });

    assert!(app.live.assistant.normalized_text().is_none());
    assert!(app.timeline.iter().any(|block| {
        matches!(block, UiTimelineItem::AssistantText(text) if text == "Let me continue testing.")
    }));
}

#[test]
fn tool_call_flushes_intermediate_stream_text_immediately() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);

    app.handle_session_event(SessionEvent::TextDelta {
        content: "I’m checking the rendering path first.".into(),
    });
    app.handle_session_event(SessionEvent::ToolCall {
        call_id: Some("tc2".into()),
        name: "read_file".into(),
        args: serde_json::json!({"path":"crates/lash-cli/src/app/mod.rs"}),
        output: lash_core::ToolCallOutput::success(serde_json::json!("ok")),
        duration_ms: 1,
    });

    assert!(app.live.assistant.normalized_text().is_none());
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
    assert!(app.usage.live_output_tokens_estimate > 0);
    app.handle_session_event(SessionEvent::TokenUsage {
        protocol_iteration: 0,
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
    assert_eq!(app.usage.live_output_tokens_estimate, 0);
    assert_eq!(app.usage.last_response_usage.input_tokens, 10);
    assert_eq!(app.usage.last_response_usage.reasoning_tokens, 2);
}

#[test]
fn input_only_streamed_usage_keeps_live_output_estimate() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::TextDelta {
        content: "abcdefgh".into(),
    });
    let live_estimate = app.usage.live_output_tokens_estimate;
    assert!(live_estimate > 0);
    app.handle_session_event(SessionEvent::TokenUsage {
        protocol_iteration: 0,
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
    assert_eq!(app.usage.live_output_tokens_estimate, live_estimate);
    assert_eq!(app.usage.token_usage.input_tokens, 10);
    assert_eq!(app.usage.last_response_usage.input_tokens, 10);
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
        app.live.tool_output.lines,
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
    assert_eq!(app.live.tool_output.partial, "Compiling alpha");

    app.handle_session_event(SessionEvent::Message {
        text: "\rCompiling beta".into(),
        kind: "tool_output".into(),
    });

    assert!(app.live.tool_output.lines.is_empty());
    assert_eq!(app.live.tool_output.partial, "Compiling beta");
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
        app.live.tool_output.lines,
        vec!["started cargo check".to_string()]
    );
    assert!(app.live.tool_output.partial.is_empty());
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
        app.live.tool_output.lines,
        vec!["warning: check this".to_string()]
    );
    assert!(app.live.tool_output.partial.is_empty());
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

    assert_eq!(app.live.tool_output.lines, vec!["done".to_string()]);
    assert!(app.live.tool_output.partial.is_empty());
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
        app.live.tool_output.lines,
        vec!["hash refs/tags/v0.2.29".to_string()]
    );
    assert!(app.live.tool_output.partial.is_empty());
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
        event: lash_core::PluginRuntimeEvent::Custom {
            name: "update_plan.snapshot".into(),
            payload: serde_json::json!({
                "generation": 1,
                "plan": [
                    {"step": "Inspect", "status": "completed"},
                    {"step": "Patch layout", "status": "in_progress"},
                    {"step": "Run tests", "status": "pending"}
                ]
            }),
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
        plugin_id: "rlm_protocol".into(),
        event: lash_core::PluginRuntimeEvent::Status {
            key: "rlm_context_budget_warning".into(),
            label: "context budget".into(),
            detail: Some("120292 tokens used; warn at 100000; choose frame switch path".into()),
        },
    });

    assert!(
        app.timeline
            .iter()
            .all(|block| !matches!(block, UiTimelineItem::UserInput(_))),
        "runtime budget warning must not be rendered as user input"
    );
    assert!(app.live.turn.as_ref().is_some_and(|turn| {
        turn.run_state == CliRunState::RunningTool
            && turn.status_detail.as_deref().is_some_and(|detail| {
                detail.contains("context budget") && detail.contains("choose frame switch path")
            })
    }));
}

#[test]
fn plan_protocol_state_events_upsert_and_clear_blocks() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.start_turn();
    app.handle_session_event(SessionEvent::PluginEvent {
        plugin_id: "plan_mode".into(),
        event: lash_core::PluginRuntimeEvent::Custom {
            name: "plan_mode.state".into(),
            payload: serde_json::json!({
                "session_id": "test-session-id",
                "enabled": true,
                "plan_path": ".lash/plans/test-session-id.md"
            }),
        },
    });
    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::PluginPanel(panel))
            if panel.title == "PLAN"
                && panel.content.contains(".lash/plans/test-session-id.md")
    ));
    assert!(
        app.live
            .turn
            .as_ref()
            .is_some_and(|turn| turn.has_visible_output)
    );

    app.handle_session_event(SessionEvent::PluginEvent {
        plugin_id: "plan_mode".into(),
        event: lash_core::PluginRuntimeEvent::Custom {
            name: "plan_mode.state".into(),
            payload: serde_json::json!({
                "session_id": "test-session-id",
                "enabled": false,
                "plan_path": null
            }),
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
    assert_eq!(app.run_state, CliRunState::Idle);
    assert!(app.live.turn.is_none());
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
    assert_eq!(app.run_state, CliRunState::Idle);
    assert!(app.live.turn.is_none());
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
    app.timeline.truncate(0);
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
    app.follow_mode = FollowOutputMode::PinnedTurnStart;

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
    app.timeline.truncate(0);
    for idx in 0..6 {
        app.timeline
            .push(UiTimelineItem::AssistantText(format!("history {idx}")));
    }
    app.timeline
        .push(UiTimelineItem::UserInput("short prompt".into()));
    app.start_turn();
    app.follow_mode = FollowOutputMode::PinnedTurnStart;

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
fn empty_state_is_not_persisted_in_scrollback_once_history_exists() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline
        .push(UiTimelineItem::UserInput("short prompt".into()));

    let width = 32usize;
    let viewport_height = 12usize;
    app.ensure_height_cache_pub(width, viewport_height);

    let cache = app.height_cache_snapshot().to_vec();
    assert_eq!(app.timeline.len(), 1);
    assert_eq!(cache[0], 1);
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
fn refresh_scroll_position_tracks_bottom_when_idle() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);

    app.timeline
        .push(UiTimelineItem::UserInput("Message 1".into()));
    app.timeline.push(UiTimelineItem::AssistantText(
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5".into(),
    ));

    let width = 80;
    app.follow_mode = FollowOutputMode::PinnedBottom;

    app.refresh_scroll_position(width, 3);
    let small_bottom = app.total_content_height(width, 3).saturating_sub(3);
    assert_eq!(app.scroll_offset, small_bottom);

    app.refresh_scroll_position(width, 6);
    let large_bottom = app.total_content_height(width, 6).saturating_sub(6);
    assert_eq!(app.scroll_offset, large_bottom);
}

#[test]
fn refresh_scroll_position_reveals_output_start_once_then_follows_tail() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);

    app.timeline
        .push(UiTimelineItem::UserInput("Message 1".into()));
    app.timeline.push(UiTimelineItem::AssistantText(
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5".into(),
    ));
    app.start_turn();
    app.follow_mode = FollowOutputMode::PinnedTurnStart;
    if let Some(turn) = app.live.turn.as_mut() {
        turn.has_visible_output = true;
        turn.output_start_anchor_pending = true;
    }

    let width = 80;
    let viewport_height = 3;

    app.refresh_scroll_position(width, viewport_height);
    let first_anchor = app.scroll_offset;
    let max_scroll = app
        .total_content_height(width, viewport_height)
        .saturating_sub(viewport_height);
    assert!(first_anchor < max_scroll);
    assert_eq!(app.follow_mode, FollowOutputMode::PinnedBottom);

    app.refresh_scroll_position(width, viewport_height);
    assert_eq!(app.scroll_offset, max_scroll);
}

#[test]
fn resume_follow_output_reenables_bottom_following() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);
    app.timeline.push(UiTimelineItem::UserInput("hello".into()));
    app.timeline
        .push(UiTimelineItem::AssistantText("world".into()));
    app.follow_mode = FollowOutputMode::Manual;
    app.scroll_offset = 3;

    app.resume_follow_output();

    assert_eq!(app.follow_mode, FollowOutputMode::PinnedBottom);
    assert_eq!(app.scroll_offset, usize::MAX);
}

#[test]
fn scroll_up_from_follow_output_detaches_from_bottom_anchor() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);
    app.timeline.push(UiTimelineItem::UserInput("hello".into()));
    app.timeline.push(UiTimelineItem::AssistantText(
        (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    ));
    let width = 24;
    let viewport_height = 5;
    app.follow_mode = FollowOutputMode::PinnedBottom;
    app.ensure_height_cache_pub(width, viewport_height);
    app.refresh_scroll_position(width, viewport_height);

    let bottom = app.scroll_offset;
    app.scroll_up(2);

    assert_eq!(app.follow_mode, FollowOutputMode::Manual);
    assert_eq!(app.scroll_offset, bottom.saturating_sub(2));
}

#[test]
fn scroll_down_to_bottom_reenables_tail_follow_instead_of_contextual_anchor() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);
    app.timeline
        .push(UiTimelineItem::AssistantText("older history".into()));
    app.timeline
        .push(UiTimelineItem::UserInput("prompt".into()));
    app.start_turn();
    app.follow_mode = FollowOutputMode::PinnedTurnStart;

    app.handle_session_event(SessionEvent::TextDelta {
        content: (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    });

    let width = 24;
    let viewport_height = 5;
    app.refresh_scroll_position(width, viewport_height);
    let contextual_anchor = app.scroll_offset;

    app.scroll_up(2);
    assert_eq!(app.follow_mode, FollowOutputMode::Manual);

    app.scroll_down(usize::MAX / 2, viewport_height, width);
    assert_eq!(app.follow_mode, FollowOutputMode::PinnedBottom);

    let max_scroll = app
        .total_content_height(width, viewport_height)
        .saturating_sub(viewport_height);
    assert!(
        contextual_anchor < max_scroll,
        "test requires contextual anchor above the tail"
    );

    app.refresh_scroll_position(width, viewport_height);

    assert_eq!(app.scroll_offset, max_scroll);
}

#[test]
fn text_delta_does_not_force_scroll_when_follow_output_is_manual() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);
    app.timeline
        .push(UiTimelineItem::UserInput("prompt".into()));
    app.start_turn();
    app.follow_mode = FollowOutputMode::Manual;
    app.scroll_offset = 3;

    app.handle_session_event(SessionEvent::TextDelta {
        content: "streamed output".into(),
    });

    assert_eq!(app.scroll_offset, 3);
    assert_eq!(app.follow_mode, FollowOutputMode::Manual);
}

#[test]
fn manual_scroll_keeps_explicit_offset_when_prior_block_height_changes() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);
    app.timeline.push(UiTimelineItem::AssistantText(
        (0..8)
            .map(|idx| format!("old line {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    ));
    app.timeline.push(UiTimelineItem::AssistantText(
        (0..8)
            .map(|idx| format!("anchored line {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    ));

    let width = 80;
    let viewport_height = 4;
    app.follow_mode = FollowOutputMode::PinnedBottom;
    app.ensure_height_cache_pub(width, viewport_height);
    app.refresh_scroll_position(width, viewport_height);

    let target = app.height_cache_snapshot()[0] + 2;
    let bottom = app.scroll_offset;
    app.scroll_up(bottom.saturating_sub(target));
    assert_eq!(app.follow_mode, FollowOutputMode::Manual);
    assert_eq!(app.scroll_offset, target);

    if let Some(UiTimelineItem::AssistantText(text)) = app.timeline.iter_mut().next() {
        text.insert_str(0, "inserted line 0\ninserted line 1\ninserted line 2\n");
    }
    app.invalidate_height_cache_from(0);
    app.refresh_scroll_position(width, viewport_height);

    assert_eq!(app.scroll_offset, target);
    assert_eq!(app.follow_mode, FollowOutputMode::Manual);
}

#[test]
fn text_delta_reveals_message_start_before_switching_to_tail_follow() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);
    app.timeline
        .push(UiTimelineItem::UserInput("prompt".into()));
    app.start_turn();
    app.follow_mode = FollowOutputMode::PinnedTurnStart;

    app.handle_session_event(SessionEvent::TextDelta {
        content: (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    });

    let width = 24;
    let viewport_height = 5;
    app.refresh_scroll_position(width, viewport_height);

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
    assert_eq!(app.follow_mode, FollowOutputMode::PinnedBottom);

    app.refresh_scroll_position(width, viewport_height);
    assert_eq!(app.scroll_offset, max_scroll);
}

#[test]
fn refresh_scroll_position_repositions_waiting_prompt_on_resize() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.push(UiTimelineItem::UserInput(
        "A long prompt that should stay visible while we are waiting for first token output".into(),
    ));
    app.start_turn();
    app.follow_mode = FollowOutputMode::PinnedTurnStart;

    let width = 24;
    app.refresh_scroll_position(width, 3);
    let initial_offset = app.scroll_offset;

    app.refresh_scroll_position(width, 6);
    assert!(
        app.scroll_offset <= initial_offset,
        "larger viewport should not push the waiting prompt further down"
    );
}

#[test]
fn handle_tool_call_merges_contiguous_exploration_activity() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.truncate(0);

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
        args: serde_json::json!({"path": "crates/lash-cli/src/render/mod.rs"}),
        output: lash_core::ToolCallOutput::success(serde_json::json!(
            "==> crates/lash-cli/src/render/mod.rs <==\nline"
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
                    "Read crates/lash-cli/src/render/mod.rs".to_string(),
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
    app.timeline.truncate(0);

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
    app.timeline.truncate(0);

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
