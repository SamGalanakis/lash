async fn check(control: lash::admin::SessionAdmin) {
    let _ = control.tool_state().await;
}

fn main() {}
