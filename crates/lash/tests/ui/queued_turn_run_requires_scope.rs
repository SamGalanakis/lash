async fn check(session: lash::LashSession) {
    let _ = session.next_queued_turn().run().await;
}

fn main() {}
