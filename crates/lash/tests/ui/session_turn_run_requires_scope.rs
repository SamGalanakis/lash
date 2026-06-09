async fn check(
    session: lash::LashSession,
    scope: lash::runtime::ScopedEffectController<'_>,
) {
    let _ = session.turn(lash::TurnInput::text("hello")).run(scope).await;
}

fn main() {}
