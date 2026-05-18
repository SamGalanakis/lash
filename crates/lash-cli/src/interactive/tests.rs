use super::helpers::TurnActivityBridge;
use super::*;
use crate::app::App;
use crate::clipboard::{ClipboardEnv, osc52_allowed_by_env, osc52_sequence_for};
use crate::editor::LARGE_PASTE_CHAR_THRESHOLD;
use crate::ui_action::{UiAction, UiActionContext, UiActionOutcome, apply_ui_action};
use lash::{TurnActivity, TurnActivityId, TurnActivitySink};
use lash_core::SessionPolicy;

async fn monitor_test_session() -> lash::LashSession {
    lash::LashCore::standard()
        .max_context_tokens(200_000)
        .build()
        .expect("core")
        .session("test-session-id")
        .open()
        .await
        .expect("session")
}

#[tokio::test]
async fn runtime_event_bridge_coalesces_text_before_structural_event() {
    let mut pump = crate::event::AppEventPump::new();
    let sink = TurnActivityBridge::spawn(7, pump.sender());

    sink.emit(TurnActivity::new(
        TurnActivityId::new("assistant"),
        TurnEvent::AssistantProseDelta {
            text: "Hello".to_string(),
        },
    ))
    .await;
    sink.emit(TurnActivity::new(
        TurnActivityId::new("assistant"),
        TurnEvent::AssistantProseDelta {
            text: ", world".to_string(),
        },
    ))
    .await;
    sink.emit(TurnActivity::independent(TurnEvent::ModelRequestStarted {
        mode_iteration: 0,
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

#[test]
fn completed_runtime_stream_rejects_late_activity() {
    assert!(session_activity_is_current(7, 7, true));
    assert!(!session_activity_is_current(6, 7, true));
    assert!(!session_activity_is_current(7, 7, false));
}

#[test]
fn late_submitted_value_after_reconciliation_is_not_rendered_again() {
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
        app.handle_turn_activity(TurnActivity::independent(TurnEvent::SubmittedValue {
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
fn promote_pending_steers_to_queue_preserves_order() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.queue_pending_steer(PreparedTurn::new("after tool 1".into(), Vec::new()));
    app.queue_pending_steer(PreparedTurn::new("after tool 2".into(), Vec::new()));

    let mut ui_trace = None;
    promote_pending_steers_to_queue(&mut app, &mut ui_trace);

    assert!(app.pending_steers.is_empty());
    let queued: Vec<String> = std::iter::from_fn(|| app.take_next_queued_turn())
        .map(|(turn, _)| turn.display_text)
        .collect();
    assert_eq!(queued, vec!["after tool 1", "after tool 2"]);
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
fn manual_interrupt_prefers_queued_followup_over_interrupted_reprojection() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline.push(UiTimelineItem::UserInput(
        "(I want future migrations to work though!)".into(),
    ));
    app.queue_pending_steer(PreparedTurn::new("next queued thing".into(), Vec::new()));

    let mut ui_trace = None;
    promote_pending_steers_to_queue(&mut app, &mut ui_trace);

    assert!(app.pending_steers.is_empty());
    assert!(app.has_queued_messages());
    assert!(matches!(
        app.timeline.last(),
        Some(UiTimelineItem::UserInput(text)) if text == "(I want future migrations to work though!)"
    ));
}

#[tokio::test]
async fn pending_monitor_wakes_inject_as_hidden_system_messages() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let session = monitor_test_session().await;
    app.queue_monitor_wake("Monitor event \"build\": done".into());

    let injected = enqueue_pending_monitor_wakes(&mut app, &session)
        .await
        .expect("inject wakes");
    assert_eq!(injected, 1);
    assert!(!app.has_pending_monitor_wakes());

    app.recycle_unaccepted_monitor_wakes();
    assert_eq!(
        app.take_pending_monitor_wakes(),
        vec!["Monitor event \"build\": done".to_string()]
    );
}

#[tokio::test]
async fn accepted_monitor_wake_is_not_requeued_after_bridge_delivery() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let session = monitor_test_session().await;
    app.queue_monitor_wake("Monitor event \"build\": done".into());

    let injected = enqueue_pending_monitor_wakes(&mut app, &session)
        .await
        .expect("inject wakes");
    assert_eq!(injected, 1);
    let messages = vec![super::helpers::monitor_wake_message(
        "Monitor event \"build\": done",
    )];

    app.acknowledge_monitor_wakes(&messages);
    app.recycle_unaccepted_monitor_wakes();

    assert!(app.take_pending_monitor_wakes().is_empty());
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
fn copy_shortcut_accepts_ctrl_c_even_when_shift_modifier_is_present() {
    let key = crossterm::event::KeyEvent {
        code: crossterm::event::KeyCode::Char('c'),
        modifiers: crossterm::event::KeyModifiers::CONTROL | crossterm::event::KeyModifiers::SHIFT,
        kind: crossterm::event::KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };

    assert!(is_copy_shortcut(key));
}

#[test]
fn copy_shortcut_accepts_plain_ctrl_c_for_selected_text_precedence() {
    let key = crossterm::event::KeyEvent {
        code: crossterm::event::KeyCode::Char('c'),
        modifiers: crossterm::event::KeyModifiers::CONTROL,
        kind: crossterm::event::KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };

    assert!(is_copy_shortcut(key));
}

#[test]
fn cleared_session_state_preserves_max_context_tokens() {
    let state = cleared_session_state(SessionPolicy {
        max_context_tokens: Some(123_456),
        ..SessionPolicy::default()
    });

    assert_eq!(state.policy.max_context_tokens, Some(123_456));
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
fn pasted_text_ui_action_uses_large_paste_placeholder_path() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 5);

    let outcome = apply_ui_action(
        &mut app,
        UiAction::InputInsertPastedText(large),
        UiActionContext {
            viewport_width: 80,
            viewport_height: 24,
            prompt_max_scroll: 0,
        },
    );

    assert_eq!(outcome, UiActionOutcome::None);
    assert_eq!(app.editor.pending_large_pastes.len(), 1);
    assert!(app.input().contains("[Pasted Content"));
}

#[test]
fn word_cursor_actions_move_across_word_boundaries() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.insert_text("alpha beta_gamma-delta  omega");

    let outcome = apply_ui_action(
        &mut app,
        UiAction::MoveCursorWordLeft,
        UiActionContext {
            viewport_width: 80,
            viewport_height: 24,
            prompt_max_scroll: 0,
        },
    );
    assert_eq!(outcome, UiActionOutcome::None);
    assert_eq!(&app.input()[app.cursor_pos()..], "omega");

    let outcome = apply_ui_action(
        &mut app,
        UiAction::MoveCursorWordRight,
        UiActionContext {
            viewport_width: 80,
            viewport_height: 24,
            prompt_max_scroll: 0,
        },
    );
    assert_eq!(outcome, UiActionOutcome::None);
    assert_eq!(app.cursor_pos(), app.input().len());
}
