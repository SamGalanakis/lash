use super::*;
use crate::app::App;
use crate::clipboard::{ClipboardEnv, osc52_allowed_by_env, osc52_sequence_for};
use crate::editor::LARGE_PASTE_CHAR_THRESHOLD;
use crate::ui_action::{UiAction, UiActionContext, UiActionOutcome, apply_ui_action};
use lash::{InjectedTurnInput, MessageRole, TurnInputInjectionBridge};

#[test]
fn pending_text_delta_buffer_coalesces_adjacent_chunks() {
    let mut pending = PendingTextDeltaBuffer::default();
    pending.push("Hello".to_string());
    pending.push(", world".to_string());

    match pending.take_event() {
        Some(SessionEvent::TextDelta { content }) => assert_eq!(content, "Hello, world"),
        other => panic!("expected coalesced text delta, got {other:?}"),
    }
}

#[test]
fn pending_text_delta_buffer_uses_first_chunk_deadline() {
    let mut pending = PendingTextDeltaBuffer::default();
    let now = tokio::time::Instant::now();
    pending.push_at("first".to_string(), now);
    pending.push_at(
        " second".to_string(),
        now + std::time::Duration::from_millis(10),
    );

    let deadline = pending.flush_deadline().expect("deadline");
    assert_eq!(deadline, now + TEXT_DELTA_REDRAW_INTERVAL);
    assert!(!pending.should_flush(now + std::time::Duration::from_millis(32)));
    assert!(pending.should_flush(now + std::time::Duration::from_millis(33)));
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
fn manual_interrupt_prefers_queued_followup_over_interrupted_reprojection() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.blocks.push(DisplayBlock::UserInput(
        "(I want future migrations to work though!)".into(),
    ));
    app.queue_pending_steer(PreparedTurn::new("next queued thing".into(), Vec::new()));

    let mut ui_trace = None;
    promote_pending_steers_to_queue(&mut app, &mut ui_trace);

    assert!(app.pending_steers.is_empty());
    assert!(app.has_queued_messages());
    assert!(matches!(
        app.blocks.last(),
        Some(DisplayBlock::UserInput(text)) if text == "(I want future migrations to work though!)"
    ));
}

#[test]
fn pending_monitor_wakes_inject_as_hidden_system_messages() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let bridge = TurnInputInjectionBridge::new();
    app.queue_monitor_wake("Monitor event \"build\": done".into());

    let injected = enqueue_pending_monitor_wakes(&mut app, &bridge).expect("inject wakes");
    assert_eq!(injected, 1);
    assert!(!app.has_pending_monitor_wakes());

    let drained = bridge.drain().expect("drain bridge");
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].message.role, MessageRole::System);
    assert_eq!(drained[0].message.content, "Monitor event \"build\": done");

    app.recycle_unaccepted_monitor_wakes();
    assert_eq!(
        app.take_pending_monitor_wakes(),
        vec!["Monitor event \"build\": done".to_string()]
    );
}

#[test]
fn accepted_monitor_wake_is_not_requeued_after_bridge_delivery() {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    let bridge = TurnInputInjectionBridge::new();
    app.queue_monitor_wake("Monitor event \"build\": done".into());

    let injected = enqueue_pending_monitor_wakes(&mut app, &bridge).expect("inject wakes");
    assert_eq!(injected, 1);
    let drained = bridge.drain().expect("drain bridge");
    let messages = drained
        .into_iter()
        .map(|InjectedTurnInput { message }| message)
        .collect::<Vec<_>>();

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
