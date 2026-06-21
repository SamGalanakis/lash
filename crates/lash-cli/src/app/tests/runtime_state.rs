use super::*;

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
            _ctx: TuiExtensionContext,
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
fn plan_exit_tool_does_not_create_local_shadow_queue() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let ui_extensions =
        lash_tui_extensions::TuiExtensions::builtin().expect("builtin ui extensions");
    crate::ui_effects::apply_ui_host_effects(
        &mut app,
        ui_extensions.effects_for_turn_event(&TurnEvent::ToolCallCompleted {
            call_id: Some("tc-plan-exit".into()),
            name: "plan_exit".into(),
            args: serde_json::json!({}),
            output: lash_core::ToolCallOutput::success(serde_json::json!({
                "approved": true,
                "confirmation_display": "Start implementing now\n\nNote: safe slice first",
                "plan_path": ".lash/plans/session.md",
                "execution_mode": "current_session",
                "next_turn_input": "Execute the plan in `.lash/plans/session.md`."
            })),
            duration_ms: 5,
        }),
    );

    assert!(app.queued_work_snapshot().is_empty());
    assert!(
        app.timeline
            .iter()
            .any(|block| matches!(block, UiTimelineItem::SystemMessage(message) if message.contains("live runtime queue")))
    );
}

#[test]
fn plan_exit_tool_call_consumes_pending_prompt_response() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let (tx, _rx) = std::sync::mpsc::channel();
    app.show_prompt(PromptState {
        request: crate::prompt_model::PromptRequest::single(
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
        output: lash_core::ToolCallOutput::success(serde_json::json!({
            "approved": true,
            "confirmation_display": "Start implementing now\n\nNote: safe slice first",
            "execution_mode": "current_session",
            "next_turn_input": "Execute the plan in `.lash/plans/session.md`."
        })),
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
    crate::ui_effects::apply_ui_host_effects(
        &mut app,
        ui_extensions.effects_for_turn_event(&TurnEvent::ToolCallCompleted {
            call_id: Some("tc-plan-exit-fresh".into()),
            name: "plan_exit".into(),
            args: serde_json::json!({}),
            output: lash_core::ToolCallOutput::success(serde_json::json!({
                "approved": true,
                "execution_mode": "fresh_context",
                "session_id": "new-plan-session",
            })),
            duration_ms: 5,
        }),
    );

    assert!(app.queued_work_snapshot().is_empty());
    assert!(
        app.timeline
            .iter()
            .all(|block| !matches!(block, UiTimelineItem::UserInput(_))),
        "fresh-context switch is a runtime transition, not a UI-queued turn"
    );
}

#[test]
fn non_manual_error_sets_transient_status() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        protocol_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::Error {
        message: "LLM error: Claude request failed with 500".into(),
        envelope: Some(lash_core::session_model::ErrorEnvelope {
            kind: "llm_provider".into(),
            code: Some("http_500".into()),
            terminal_reason: None,
            user_message: "LLM error: Claude request failed with 500".into(),
            raw: None,
        }),
    });
    app.handle_session_event(SessionEvent::Done);

    assert_eq!(
        app.live.turn.as_ref().map(|turn| turn.run_state),
        Some(CliRunState::Error)
    );
    assert_eq!(
        app.live
            .turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        Some("LLM error: Claude request failed with 500")
    );
}

#[test]
fn retry_status_stays_visible_when_retry_request_starts() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        protocol_iteration: 0,
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
        app.live.turn.as_ref().map(|turn| turn.run_state),
        Some(CliRunState::Waiting)
    );
    assert_eq!(
        app.live
            .turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        Some("in 2s · attempt 2/4 · Codex returned non-SSE body but it could not be read")
    );

    app.handle_session_event(SessionEvent::LlmRequest {
        protocol_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });

    assert_eq!(
        app.live.turn.as_ref().map(|turn| turn.run_state),
        Some(CliRunState::Waiting)
    );
    assert_eq!(
        app.live
            .turn
            .as_ref()
            .and_then(|turn| turn.status_detail.as_deref()),
        Some("attempt 2/4 · Codex returned non-SSE body but it could not be read")
    );
}

#[test]
fn transient_status_expires_on_tick() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.handle_session_event(SessionEvent::LlmRequest {
        protocol_iteration: 0,
        message_count: 0,
        tool_list: String::new(),
    });
    app.handle_session_event(SessionEvent::Error {
        message: "runtime error".into(),
        envelope: None,
    });
    app.handle_session_event(SessionEvent::Done);

    if let Some(turn) = app.live.turn.as_mut() {
        turn.transient_until = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    }
    app.on_tick();
    assert!(app.live.turn.is_none());
}

#[test]
fn durable_queue_snapshot_preserves_order_and_cache_is_display_only() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let first = PreparedTurn::new("queued-1".into(), Vec::new());
    let second = PreparedTurn::new("queued-2".into(), Vec::new());
    app.test_seed_queued_turn_snapshot(
        first.clone(),
        lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
        lash_core::SlotPolicy::Exclusive,
    );
    app.test_seed_queued_turn_snapshot(
        second.clone(),
        lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
        lash_core::SlotPolicy::Exclusive,
    );

    let previews = app
        .queued_work_snapshot()
        .iter()
        .filter_map(|batch| app.prepared_turn_for_queued_batch(batch))
        .map(|turn| turn.display_text)
        .collect::<Vec<_>>();

    assert_eq!(previews, vec!["queued-1", "queued-2"]);
    assert_eq!(app.queues.draft_presentations.len(), 2);
}

#[test]
fn queued_work_started_removes_claimed_batch_from_preview() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let first_id = app.test_seed_queued_turn_snapshot(
        PreparedTurn::new("processing now".into(), Vec::new()),
        lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
        lash_core::SlotPolicy::Exclusive,
    );
    let second_id = app.test_seed_queued_turn_snapshot(
        PreparedTurn::new("still pending".into(), Vec::new()),
        lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
        lash_core::SlotPolicy::Exclusive,
    );

    app.handle_turn_activity(lash_core::TurnActivity::independent(
        TurnEvent::QueuedWorkStarted {
            boundary: lash_core::runtime::QueuedWorkClaimBoundary::Idle,
            batch_ids: vec![first_id],
            causes: Vec::new(),
        },
    ));

    assert_eq!(
        app.queued_work_snapshot()
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        vec![second_id.as_str()]
    );
}

#[test]
fn take_prepared_turn_for_queued_batch_restores_cached_images() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("queued".into(), vec![vec![1, 2, 3]]);
    app.test_seed_queued_turn_snapshot(
        turn,
        lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
        lash_core::SlotPolicy::Exclusive,
    );

    let batch = app.queued_work_snapshot()[0].clone();
    let restored = app
        .take_prepared_turn_for_queued_batch(&batch)
        .expect("queued presentation");

    assert_eq!(restored.display_text, "queued");
    assert_eq!(restored.images.len(), 1);
    assert_eq!(restored.images[0].id, 1);
    assert_eq!(restored.images[0].png_bytes, vec![1, 2, 3]);
    assert!(app.queues.draft_presentations.is_empty());
}

#[test]
fn accepted_injected_turn_input_renders_matching_draft_presentation() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("follow up".into(), Vec::new());
    app.cache_draft_presentation(turn.clone());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash_core::AcceptedInjectedTurnInput {
            id: None,
            message: PluginMessage::text(MessageRole::User, "follow up"),
        }],
        checkpoint: lash_core::CheckpointKind::AfterWork,
    });

    assert!(app.queues.draft_presentations.is_empty());
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
    app.cache_draft_presentation(turn.clone());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash_core::AcceptedInjectedTurnInput {
            id: None,
            message: PluginMessage::text(
                MessageRole::User,
                "/localref lash for context if needed\n\n<skill>\n<name>localref</name>\nbody\n</skill>",
            ),
        }],
        checkpoint: lash_core::CheckpointKind::AfterWork,
    });

    assert!(app.queues.draft_presentations.is_empty());
    assert!(app.timeline.iter().any(|block| matches!(
        block,
        UiTimelineItem::UserInput(text) if text == "/localref lash for context if needed"
    )));
}

#[test]
fn accepted_injected_turn_input_prefers_draft_id_over_content_match() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let first = PreparedTurn::new("first queued steer".into(), Vec::new());
    let accepted = PreparedTurn::new("visible accepted draft".into(), Vec::new());
    let accepted_id = accepted.draft_id.clone();
    app.cache_draft_presentation(first.clone());
    app.cache_draft_presentation(accepted.clone());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash_core::AcceptedInjectedTurnInput {
            id: Some(accepted_id),
            message: PluginMessage::text(MessageRole::User, "runtime accepted content"),
        }],
        checkpoint: lash_core::CheckpointKind::AfterWork,
    });

    assert_eq!(app.queues.draft_presentations.len(), 1);
    assert!(app.queues.draft_presentations.contains_key(&first.draft_id));
    assert!(app.timeline.iter().any(|block| matches!(
        block,
        UiTimelineItem::UserInput(text) if text == "visible accepted draft"
    )));
    assert!(!app.timeline.iter().any(|block| matches!(
        block,
        UiTimelineItem::UserInput(text) if text == "runtime accepted content"
    )));
}

#[test]
fn live_turn_requires_visible_user_input_before_enter_injection() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());

    app.start_turn();

    assert!(!app.can_inject_into_active_turn());
    assert_eq!(
        app.route_turn_submission(true),
        TurnSubmissionRoute::QueueNextFullTurn
    );

    app.mark_live_turn_user_input_visible();

    assert!(app.can_inject_into_active_turn());
    assert_eq!(
        app.route_turn_submission(true),
        TurnSubmissionRoute::InjectActiveTurn
    );
}

#[test]
fn turn_submission_route_sends_immediately_when_idle() {
    let app = App::new("test-model".into(), "test".into(), "test-session-id".into());

    assert_eq!(
        app.route_turn_submission(true),
        TurnSubmissionRoute::SendNow
    );
}

#[test]
fn turn_submission_route_blocks_when_runtime_is_switching() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());

    assert_eq!(
        app.route_turn_submission(false),
        TurnSubmissionRoute::BlockedSessionSwitch
    );

    app.start_turn();
    app.mark_live_turn_user_input_visible();

    assert_eq!(
        app.route_turn_submission(false),
        TurnSubmissionRoute::BlockedSessionSwitch
    );
}

#[test]
fn accepted_injected_turn_input_without_pending_match_still_renders_once() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash_core::AcceptedInjectedTurnInput {
            id: None,
            message: PluginMessage {
                role: MessageRole::User,
                content: "runtime content".into(),
                origin: None,
                parts: Vec::new(),
                images: Vec::new(),
            },
        }],
        checkpoint: lash_core::CheckpointKind::AfterWork,
    });

    app.handle_session_event(SessionEvent::InjectedMessagesCommitted {
        messages: vec![PluginMessage {
            role: MessageRole::User,
            content: "runtime content".into(),
            origin: None,
            parts: Vec::new(),
            images: Vec::new(),
        }],
        checkpoint: lash_core::CheckpointKind::AfterWork,
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
fn accepted_injected_turn_input_removes_matching_draft_without_popping_wrong_one() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let first = PreparedTurn::new("first queued steer".into(), Vec::new());
    app.cache_draft_presentation(first.clone());
    app.cache_draft_presentation(PreparedTurn::new(
        "uhh do not switch nvm".into(),
        Vec::new(),
    ));

    app.handle_session_event(SessionEvent::InjectedTurnInputAccepted {
        inputs: vec![lash_core::AcceptedInjectedTurnInput {
            id: None,
            message: PluginMessage::text(MessageRole::User, "uhh do not switch nvm"),
        }],
        checkpoint: lash_core::CheckpointKind::AfterWork,
    });

    assert_eq!(app.queues.draft_presentations.len(), 1);
    assert!(app.queues.draft_presentations.contains_key(&first.draft_id));
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
    app.cache_draft_presentation(turn.clone());

    // Assistant streams some prose, then runs a tool — this pushes an
    // AssistantText and an Activity block in front of the original UserInput.
    app.handle_session_event(SessionEvent::TextDelta {
        content: "You're right.".into(),
    });
    app.handle_session_event(SessionEvent::ToolCall {
        name: "fs_read".into(),
        args: serde_json::json!({"path": "crates/lash/src/plugin.rs"}),
        output: lash_core::ToolCallOutput::success(serde_json::json!({"output": "..."})),
        duration_ms: 12,
        call_id: None,
    });

    // Now the late commit arrives.
    app.handle_session_event(SessionEvent::InjectedMessagesCommitted {
        messages: vec![PluginMessage::text(
            MessageRole::User,
            "Why are you still dillydallying",
        )],
        checkpoint: lash_core::CheckpointKind::AfterWork,
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
    app.cache_draft_presentation(turn.clone());

    app.handle_session_event(SessionEvent::InjectedMessagesCommitted {
        messages: vec![PluginMessage::text(
            MessageRole::User,
            "(I want future migrations to work though!)",
        )],
        checkpoint: lash_core::CheckpointKind::AfterWork,
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
    assert!(app.queues.draft_presentations.is_empty());
}

#[test]
fn queued_injection_stays_out_of_history_until_committed() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("follow up now".into(), Vec::new());

    app.test_seed_queued_turn_snapshot(
        turn,
        lash_core::DeliveryPolicy::EarliestSafeBoundary,
        lash_core::SlotPolicy::Join,
    );

    assert_eq!(app.queued_work_snapshot().len(), 1);
    assert!(!matches!(
        app.timeline.last(),
        Some(UiTimelineItem::UserInput(_))
    ));
}

#[test]
fn idle_earliest_boundary_preview_is_next_full_turn_not_this_turn() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.test_seed_queued_turn_snapshot(
        PreparedTurn::new("follow up now".into(), Vec::new()),
        lash_core::DeliveryPolicy::EarliestSafeBoundary,
        lash_core::SlotPolicy::Join,
    );

    let preview = crate::render::queue_preview_lines_snapshot(&app, 80)
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(!preview.contains("Will send in this turn"));
    assert!(preview.contains("Queued for next turn"));
    assert!(preview.contains("follow up now"));
}

#[test]
fn regular_queued_turn_stays_out_of_history_until_dispatched() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let turn = PreparedTurn::new("queued text".into(), Vec::new());
    app.test_seed_queued_turn_snapshot(
        turn,
        lash_core::DeliveryPolicy::AfterCurrentTurnCommit,
        lash_core::SlotPolicy::Exclusive,
    );

    assert_eq!(app.queued_work_snapshot().len(), 1);
    assert!(!matches!(
        app.timeline.last(),
        Some(UiTimelineItem::UserInput(_))
    ));
}

#[test]
fn suppressed_ready_batch_does_not_render_queue_preview() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let batch_id = app.test_seed_queued_turn_snapshot(
        PreparedTurn::new("send immediately".into(), Vec::new()),
        lash_core::DeliveryPolicy::EarliestSafeBoundary,
        lash_core::SlotPolicy::Join,
    );

    assert!(
        crate::render::queue_preview_lines_snapshot(&app, 80)
            .iter()
            .any(|line| line
                .spans
                .iter()
                .any(|span| span.content.contains("send immediately")))
    );

    app.suppress_queue_preview_batches([batch_id.as_str()]);

    assert!(crate::render::queue_preview_lines_snapshot(&app, 80).is_empty());
    assert!(!app.has_queued_messages());
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
fn text_selection_contract_tracks_output_and_input_selection() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.selection.anchor = (1, 0);
    app.selection.end = (4, 0);
    app.selection.visible = true;
    app.set_input("input selection".into());
    app.start_input_selection(0);
    app.update_input_selection(5);
    app.finish_input_selection();

    assert!(app.has_output_selection());
    assert!(app.has_visible_output_selection());
    assert!(app.has_text_selection());
    assert!(app.has_visible_text_selection());

    app.clear_text_selection();

    assert!(!app.has_output_selection());
    assert!(!app.has_input_selection());
    assert!(app.dirty);
}

#[test]
fn selection_mouse_up_suppression_is_one_shot() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());

    assert!(!app.take_suppressed_mouse_up_after_selection_clear());
    app.suppress_mouse_up_after_selection_clear();

    assert!(app.take_suppressed_mouse_up_after_selection_clear());
    assert!(!app.take_suppressed_mouse_up_after_selection_clear());
}

#[test]
fn toast_expires_on_tick() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.show_toast("Copied to clipboard", ToastKind::Info);
    app.toast.as_mut().expect("toast").expires_at =
        std::time::Instant::now() - std::time::Duration::from_secs(1);

    app.on_tick();

    assert!(app.toast.is_none());
    assert!(app.dirty);
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

    app.editor.backspace();

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

    app.editor.backspace();
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
        request: crate::prompt_model::PromptRequest::freeform("Pick one"),
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
        crate::prompt_model::PromptResponse::Text {
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
        request: crate::prompt_model::PromptRequest::single("Pick one", vec!["red".into()]),
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
        crate::prompt_model::PromptResponse::Single {
            selection: String::new(),
            note: None,
        }
    );
    assert!(app.prompt_state().is_none());
    assert!(app.dirty);
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

    app.editor.backspace();

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
        request: crate::prompt_model::PromptRequest::freeform("Question?"),
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
        request: crate::prompt_model::PromptRequest::freeform("Question?"),
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
        request: crate::prompt_model::PromptRequest::single(
            "Pick one",
            vec!["red".into(), "blue".into()],
        )
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
        request: crate::prompt_model::PromptRequest::single(
            "Pick one",
            vec!["red".into(), "blue".into()],
        )
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
        crate::prompt_model::PromptResponse::Single {
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
        request: crate::prompt_model::PromptRequest::single(
            "Pick one",
            vec!["red".into(), "blue".into()],
        ),
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
        output: lash_core::ToolCallOutput::success(serde_json::json!([])),
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
        request: crate::prompt_model::PromptRequest::single(
            "Pick one",
            vec!["red".into(), "blue".into()],
        )
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
        output: lash_core::ToolCallOutput::success(serde_json::json!({
            "kind": "single",
            "selection": "blue",
            "note": "ship the blue path"
        })),
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
fn at_completion_finds_nested_file_via_fuzzy() {
    use std::fs;
    use tempfile::TempDir;

    let dir = TempDir::new().expect("tempdir");
    fs::create_dir_all(dir.path().join("crates/lash-cli/src/interactive")).unwrap();
    fs::create_dir_all(dir.path().join("crates/lash-cli/src")).unwrap();
    fs::create_dir_all(dir.path().join("other/dir")).unwrap();
    fs::write(
        dir.path()
            .join("crates/lash-cli/src/interactive/input_handling.rs"),
        "",
    )
    .unwrap();
    fs::write(dir.path().join("crates/lash-cli/src/editor.rs"), "").unwrap();
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
        suggestions.contains(&"crates/lash-cli/src/interactive/input_handling.rs"),
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
