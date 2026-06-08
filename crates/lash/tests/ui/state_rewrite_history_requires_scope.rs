async fn check(session: lash::LashSession, trigger: lash::runtime::RewriteTrigger) {
    let _ = session.control().state().rewrite_history(trigger).await;
}

fn main() {}
