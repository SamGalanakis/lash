async fn check(session: lash::LashSession, events: &dyn lash::TurnActivitySink) {
    let _ = session
        .turn(lash::TurnInput::text("hello"))
        .stream(events)
        .await;
}

fn main() {}
