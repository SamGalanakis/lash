async fn check(control: lash_embed::control::SessionControl) {
    let _ = control.tool_state().await;
}

fn main() {}
