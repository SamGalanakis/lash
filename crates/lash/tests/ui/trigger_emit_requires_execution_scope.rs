async fn check(core: lash::LashCore) {
    let source_key = lash::triggers::empty_trigger_source_key("ui.button.pressed").unwrap();
    let _ = core
        .triggers()
        .emit(lash::triggers::TriggerOccurrenceRequest::new(
            "ui.button.pressed",
            source_key,
            serde_json::json!({ "pressed": true }),
            "button-press-1",
        ))
        .await;
}

fn main() {}
