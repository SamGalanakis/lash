async fn check(session: lash::LashSession, request: lash_core::SessionTurnRequest<'_>) {
    let _ = session.admin().children().start_turn(request).await;
}

fn main() {}
