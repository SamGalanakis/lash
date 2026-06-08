async fn check(session: lash::LashSession, request: lash_core::SessionTurnRequest<'_>) {
    let _ = session.control().children().start_turn(request).await;
}

fn main() {}
