async fn check(core: lash::LashCore) {
    let source_key = lash::empty_host_event_source_key("ui.button.pressed").unwrap();
    let _ = core
        .host_events()
        .emit(lash::HostEventOccurrenceRequest::new(
            "ui.button.pressed",
            source_key,
            serde_json::json!({ "pressed": true }),
            "button-press-1",
        ))
        .await;
}

fn main() {}
