use super::*;

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
    let mut app = App::new("test-model".into(), "test".into());
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
