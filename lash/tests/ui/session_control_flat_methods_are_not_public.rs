async fn check(control: lash::control::SessionControl) {
    let _ = control.tool_state().await;
}

fn main() {}
