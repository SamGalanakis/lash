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
