async fn check(core: lash::LashCore) {
    let _ = core.delete_session("session-id").await;
}

fn main() {}
