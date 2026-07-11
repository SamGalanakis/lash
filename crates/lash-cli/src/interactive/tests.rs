use super::*;
use std::sync::Arc;

use crate::app::App;
use crate::clipboard::{ClipboardEnv, osc52_allowed_by_env, osc52_sequence_for};
use crate::editor::LARGE_PASTE_CHAR_THRESHOLD;
use lash::{TurnActivity, TurnActivityId};
use lash_core::SessionPolicy;

#[tokio::test]
async fn observation_activity_coalescer_batches_text_before_structural_event() {
    let mut pump = crate::event::AppEventPump::new();
    let mut coalescer = super::helpers::TurnActivityCoalescer::new(7, pump.sender());

    coalescer
        .emit(TurnActivity::new(
            TurnActivityId::new("assistant"),
            TurnEvent::AssistantProseDelta {
                text: "Hello".to_string(),
            },
        ))
        .await;
    coalescer
        .emit(TurnActivity::new(
            TurnActivityId::new("assistant"),
            TurnEvent::AssistantProseDelta {
                text: ", world".to_string(),
            },
        ))
        .await;
    coalescer
        .emit(TurnActivity::independent(TurnEvent::ModelRequestStarted {
            protocol_iteration: 0,
        }))
        .await;

    let first = pump.recv().await.expect("text event").event;
    let AppEvent::Session {
        stream_id: 7,
        activity,
    } = first
    else {
        panic!("expected coalesced text delta");
    };
    let TurnActivity {
        event: TurnEvent::AssistantProseDelta { text },
        ..
    } = *activity
    else {
        panic!("expected coalesced text delta");
    };
    assert_eq!(text, "Hello, world");

    let second = pump.recv().await.expect("structural event").event;
    let AppEvent::Session {
        stream_id: 7,
        activity,
    } = second
    else {
        panic!("expected structural event");
    };
    assert!(matches!(
        activity.event,
        TurnEvent::ModelRequestStarted { .. }
    ));
}

#[tokio::test]
async fn cli_turn_runner_streams_to_ui_through_session_observation() {
    let session = observation_test_session(
        "cli-observation-bridge",
        Some("Hello from observation"),
        None,
    )
    .await;
    let mut pump = crate::event::AppEventPump::new();
    let stream_id = 42;

    super::helpers::SessionObservationBridge::spawn(&session, stream_id, pump.sender());
    let (_cancel, return_rx) = crate::turn_runner::spawn_session_turn(
        session,
        lash_core::TurnInput::text("hello"),
        stream_id,
    );

    let mut saw_text = false;
    let mut saw_finished = false;
    while !saw_text || !saw_finished {
        let event = tokio::time::timeout(std::time::Duration::from_secs(2), pump.recv())
            .await
            .expect("observation event")
            .expect("app event")
            .event;
        match event {
            AppEvent::Session {
                stream_id: observed,
                activity,
            } => {
                assert_eq!(observed, stream_id);
                if let TurnEvent::AssistantProseDelta { text } = activity.event {
                    assert_eq!(text, "Hello from observation");
                    saw_text = true;
                }
            }
            AppEvent::SessionObservationFinished {
                stream_id: observed,
            } => {
                assert_eq!(observed, stream_id);
                saw_finished = true;
            }
            _ => {}
        }
    }

    let done = return_rx.await.expect("turn result");
    assert_eq!(done.stream_id, stream_id);
    assert!(matches!(
        done.result.outcome,
        lash_core::TurnOutcome::Finished(_)
    ));
}

#[tokio::test]
async fn session_observation_bridge_surfaces_gap_and_requests_refresh() {
    let session =
        observation_test_session("cli-observation-gap", Some("gap response"), Some(1)).await;
    let stale_cursor = session.observe().current_observation().cursor;
    session
        .turn(lash_core::TurnInput::text("first"))
        .run()
        .await
        .expect("first turn");
    session
        .turn(lash_core::TurnInput::text("second"))
        .run()
        .await
        .expect("second turn");

    let mut pump = crate::event::AppEventPump::new();
    super::helpers::SessionObservationBridge::spawn_from_cursor(
        &session,
        stale_cursor,
        77,
        pump.sender(),
    );

    let mut saw_diagnostic = false;
    let mut saw_pending_input_refresh = false;
    let mut saw_ui_refresh = false;
    for _ in 0..8 {
        let event = tokio::time::timeout(std::time::Duration::from_secs(2), pump.recv())
            .await
            .expect("gap bridge event")
            .expect("app event")
            .event;
        match event {
            AppEvent::SystemMessage { message } => {
                saw_diagnostic |=
                    message.contains("Live session observation skipped buffered events");
            }
            AppEvent::RequestPendingTurnInputSnapshot => saw_pending_input_refresh = true,
            AppEvent::RequestUiSnapshot => saw_ui_refresh = true,
            _ => {}
        }
        if saw_diagnostic && saw_pending_input_refresh && saw_ui_refresh {
            break;
        }
    }
    assert!(saw_diagnostic, "expected live replay gap diagnostic");
    assert!(
        saw_pending_input_refresh,
        "expected pending-input snapshot refresh request"
    );
    assert!(saw_ui_refresh, "expected UI snapshot refresh request");
}

#[tokio::test]
async fn session_observation_bridge_surfaces_process_changes() {
    let session = observation_test_session("cli-process-observation", Some("unused"), None).await;
    let mut pump = crate::event::AppEventPump::new();
    let stream_id = 91;
    let process_id = "bridge-process";

    super::helpers::SessionObservationBridge::spawn(&session, stream_id, pump.sender());
    session
        .processes()
        .start(
            lash_core::ProcessStartRequest::external(
                process_id,
                lash_core::ProcessOriginator::host(),
                serde_json::Value::Null,
            )
            .with_grant(Some(lash_core::ProcessStartGrant {
                session_scope: lash_core::SessionScope::new("bridge-process-descriptor"),
                descriptor: lash_core::ProcessHandleDescriptor::new(
                    Some("test"),
                    Some("bridge process"),
                ),
            })),
            inline_scope(lash_core::ExecutionScope::process(process_id)),
        )
        .await
        .expect("start process");
    session
        .processes()
        .cancel(
            process_id,
            inline_scope(lash_core::ExecutionScope::process(process_id)),
        )
        .await
        .expect("cancel process");

    let mut saw_started = false;
    let mut saw_cancelled = false;
    let mut ui_refreshes = 0;
    for _ in 0..12 {
        let event = tokio::time::timeout(std::time::Duration::from_secs(2), pump.recv())
            .await
            .expect("process bridge event")
            .expect("app event")
            .event;
        match event {
            AppEvent::ProcessChanged {
                stream_id: observed,
                kind,
                process_ids,
            } => {
                assert_eq!(observed, stream_id);
                assert_eq!(process_ids, vec![process_id.to_string()]);
                match kind {
                    lash_core::SessionProcessEventKind::Started => saw_started = true,
                    lash_core::SessionProcessEventKind::Cancelled => saw_cancelled = true,
                }
            }
            AppEvent::RequestUiSnapshot => ui_refreshes += 1,
            _ => {}
        }
        if saw_started && saw_cancelled && ui_refreshes >= 2 {
            break;
        }
    }
    assert!(saw_started, "expected process start event");
    assert!(saw_cancelled, "expected process cancellation event");
    assert!(
        ui_refreshes >= 2,
        "expected snapshot reconciliation requests for process changes"
    );
}

async fn observation_test_session(
    session_id: &str,
    response: Option<&'static str>,
    live_replay_capacity: Option<usize>,
) -> LashSession {
    let provider = lash_core::testing::TestProvider::builder()
        .kind("observation-test")
        .complete(move |_| async move {
            Ok(lash_core::LlmResponse {
                full_text: response.unwrap_or_default().to_string(),
                parts: Vec::new(),
                usage: Default::default(),
                terminal_reason: lash_core::LlmTerminalReason::Stop,
                terminal_diagnostic: None,
                provider_usage: None,
                request_body: None,
                http_summary: None,
                execution_evidence: None,
            })
        })
        .build()
        .into_handle();
    let mut builder = lash::LashCore::standard_builder()
        .provider(provider)
        .model(
            lash_core::ModelSpec::from_token_limits(
                "mock-model",
                Default::default(),
                200_000,
                None,
            )
            .expect("model spec"),
        )
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .process_registry(Arc::new(lash_core::TestLocalProcessRegistry::default()))
        .process_env_store(Arc::new(
            lash::persistence::InMemoryProcessExecutionEnvStore::new(),
        ))
        .store_factory(Arc::new(
            lash::persistence::InMemorySessionStoreFactory::new(),
        ));
    if let Some(capacity) = live_replay_capacity {
        builder =
            builder.live_replay_store(Arc::new(lash_core::InMemoryLiveReplayStore::with_bounds(
                capacity,
                std::time::Duration::from_secs(120),
            )));
    }
    let core = builder.build().expect("core");
    core.session(session_id).open().await.expect("session")
}

fn inline_scope(scope: lash_core::ExecutionScope) -> lash_core::ScopedEffectController<'static> {
    lash_core::ScopedEffectController::shared(
        Arc::new(lash_core::InlineRuntimeEffectController),
        scope,
    )
    .expect("inline execution scope")
}

#[test]
fn completed_runtime_stream_rejects_late_activity() {
    assert!(session_activity_is_current(7, 7, true));
    assert!(!session_activity_is_current(6, 7, true));
    assert!(!session_activity_is_current(7, 7, false));
}

#[test]
fn late_final_value_after_reconciliation_is_not_rendered_again() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline = vec![
        UiTimelineItem::Splash,
        UiTimelineItem::TurnStart(crate::app::Turn::user(false)),
        UiTimelineItem::UserInput("hi".into()),
        UiTimelineItem::LashlangCode("\nsubmit \"Hi! How can I help?\"".into()),
        UiTimelineItem::AssistantText("Hi! How can I help?".into()),
    ]
    .into();

    if session_activity_is_current(1, 1, false) {
        app.handle_turn_activity(TurnActivity::independent(TurnEvent::FinalValue {
            value: serde_json::json!("Hi! How can I help?"),
        }));
    }

    let assistant_texts: Vec<&str> = app
        .timeline
        .iter()
        .filter_map(|block| match block {
            UiTimelineItem::AssistantText(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(assistant_texts, vec!["Hi! How can I help?"]);
}

#[test]
fn enter_on_command_suggestion_resolves_selected_slash_command() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.set_input("/in".into());
    app.update_suggestions();

    let (text, parsed) =
        super::input_handling::selected_slash_command_suggestion(&app, app.ui_extensions())
            .expect("selected slash command");

    assert_eq!(text, "/info");
    assert!(matches!(
        parsed,
        super::commands::ParsedSlashCommand::Builtin(crate::command::Command::Info)
    ));
}

#[test]
fn enter_on_skill_suggestion_stays_text_completion() {
    let root =
        std::env::temp_dir().join(format!("lash-interactive-skill-{}", uuid::Uuid::new_v4()));
    let skill_dir = root.join("localref");
    std::fs::create_dir_all(&skill_dir).expect("skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: localref\ndescription: Clone a local reference\n---\n\nbody\n",
    )
    .expect("skill file");

    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.skills = crate::SkillCatalog::from_dirs(std::slice::from_ref(&root));
    app.set_input("/loc".into());
    app.update_suggestions();

    assert!(
        app.suggestions()
            .iter()
            .any(|suggestion| suggestion.name == "/localref")
    );
    assert!(
        super::input_handling::selected_slash_command_suggestion(&app, app.ui_extensions())
            .is_none()
    );

    app.complete_suggestion();
    assert_eq!(app.input(), "/localref ");
}

#[test]
fn active_turn_slash_command_block_message_is_command_not_queue_language() {
    let message =
        super::input_handling::slash_command_blocked_while_working_message("/resume previous");

    assert!(message.contains("Cannot run `/resume` while Lash is working"));
    assert!(!message.to_ascii_lowercase().contains("queue"));
}

#[test]
fn command_palette_items_include_current_settings_and_theme_choices() {
    crate::theme::set_active_theme(crate::config::ThemeName::Lash);
    let provider = lash_core::testing::TestProvider::builder()
        .kind("codex")
        .build()
        .into_handle();
    let mut app = App::new("gpt-5.5".into(), "test".into(), "test-session-id".into());
    app.set_model_variant(Some("xhigh".to_string()));
    app.set_execution_mode_label(&crate::execution_settings::ExecutionMode::Rlm);

    let items = super::input_handling::command_palette_items(&app, &provider);

    assert!(items.iter().any(|item| {
        item.title == "Theme: Lash"
            && item.current
            && matches!(
                item.action,
                crate::overlay::CommandPaletteAction::Theme(crate::config::ThemeName::Lash)
            )
    }));
    assert!(items.iter().any(|item| {
        item.title == "Theme: System"
            && matches!(
                item.action,
                crate::overlay::CommandPaletteAction::Theme(crate::config::ThemeName::System)
            )
    }));
    assert!(items.iter().any(|item| {
        item.title == "Model"
            && item.description.contains("gpt-5.5")
            && matches!(
                item.action,
                crate::overlay::CommandPaletteAction::InsertDraft(ref draft) if draft == "/model "
            )
    }));
    assert!(items.iter().any(|item| {
        item.title == "Variant: xhigh"
            && item.current
            && matches!(
                item.action,
                crate::overlay::CommandPaletteAction::Builtin(crate::command::Command::Variant(Some(ref variant)))
                    if variant == "xhigh"
            )
    }));
    assert!(items.iter().any(|item| {
        item.title == "Provider"
            && item.description.contains("Codex")
            && matches!(
                item.action,
                crate::overlay::CommandPaletteAction::Builtin(
                    crate::command::Command::ChangeProvider
                )
            )
    }));
}

#[test]
fn manual_interrupt_keeps_durable_queue_snapshot_out_of_history() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.push(UiTimelineItem::UserInput(
        "(I want future migrations to work though!)".into(),
    ));
    app.test_seed_queued_turn_snapshot(
        PreparedTurn::new("next queued thing".into(), Vec::new()),
        lash_core::TurnInputIngress::NextTurn,
    );

    assert!(app.has_queued_messages());
    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::UserInput(text)) if text == "(I want future migrations to work though!)"
    ));
}

#[test]
fn selection_is_preserved_for_modified_key_chords() {
    let key = crossterm::event::KeyEvent {
        code: crossterm::event::KeyCode::Char('c'),
        modifiers: crossterm::event::KeyModifiers::SHIFT,
        kind: crossterm::event::KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };

    assert!(should_preserve_selection_for_key(key));
}

#[test]
fn selection_is_cleared_for_plain_typing_keys() {
    let key = crossterm::event::KeyEvent {
        code: crossterm::event::KeyCode::Char('x'),
        modifiers: crossterm::event::KeyModifiers::NONE,
        kind: crossterm::event::KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };

    assert!(!should_preserve_selection_for_key(key));
}

#[test]
fn copy_shortcut_accepts_ctrl_shift_c() {
    let key = crossterm::event::KeyEvent {
        code: crossterm::event::KeyCode::Char('c'),
        modifiers: crossterm::event::KeyModifiers::CONTROL | crossterm::event::KeyModifiers::SHIFT,
        kind: crossterm::event::KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };

    assert!(is_copy_shortcut(key));
}

#[test]
fn copy_shortcut_rejects_plain_ctrl_c() {
    let key = crossterm::event::KeyEvent {
        code: crossterm::event::KeyCode::Char('c'),
        modifiers: crossterm::event::KeyModifiers::CONTROL,
        kind: crossterm::event::KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };

    assert!(!is_copy_shortcut(key));
}

#[test]
fn cleared_session_state_preserves_model_spec() {
    let state = cleared_session_state(SessionPolicy {
        model: lash_core::ModelSpec::from_token_limits(
            "mock-model",
            Default::default(),
            123_456,
            None,
        )
        .expect("valid model spec"),
        ..SessionPolicy::default()
    });

    assert_eq!(state.policy.model.context_window_tokens(), 123_456);
}

#[test]
fn osc52_sequence_wraps_for_tmux() {
    let seq = osc52_sequence_for(
        "YWJj",
        &ClipboardEnv {
            tmux: true,
            ..ClipboardEnv::default()
        },
    );
    assert!(seq.starts_with("\x1bPtmux;\x1b\x1b]52;c;YWJj\x07"));
    assert!(seq.ends_with("\x1b\\"));
}

#[test]
fn osc52_sequence_wraps_for_screen() {
    let seq = osc52_sequence_for(
        "YWJj",
        &ClipboardEnv {
            term: "screen-256color".to_string(),
            ..ClipboardEnv::default()
        },
    );
    assert_eq!(seq, "\x1bP\x1b]52;c;YWJj\x07\x1b\\");
}

#[test]
fn osc52_allowed_when_ssh_present() {
    assert!(osc52_allowed_by_env(&ClipboardEnv {
        ssh_tty: true,
        ..ClipboardEnv::default()
    }));
}

#[test]
fn osc52_disallowed_for_unknown_terminal_without_ssh() {
    assert!(!osc52_allowed_by_env(&ClipboardEnv {
        term: "dumb".to_string(),
        ..ClipboardEnv::default()
    }));
}

#[test]
fn pasted_text_input_uses_large_paste_placeholder_path() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 5);

    app.insert_pasted_text(&large);
    app.update_suggestions();

    assert_eq!(app.editor.pending_large_pastes.len(), 1);
    assert!(app.input().contains("[Pasted Content"));
}

#[test]
fn word_cursor_actions_move_across_word_boundaries() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.insert_text("alpha beta_gamma-delta  omega");

    app.editor.move_cursor_word_left();
    assert_eq!(&app.input()[app.cursor_pos()..], "omega");

    app.editor.move_cursor_word_right();
    assert_eq!(app.cursor_pos(), app.input().len());
}
