async fn check(session: lash::LashSession) {
    let _ = session.turn(lash::TurnInput::text("hello")).run().await;
}

fn main() {}
