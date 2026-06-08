fn main() {
    let input = lash::remote::RemoteTurnInput::text("hello");
    let request = lash::remote::RemoteTurnRequest {
        protocol_version: lash::remote::REMOTE_PROTOCOL_VERSION,
        session_id: "session".to_string(),
        turn_id: "turn".to_string(),
        idempotency_key: Some("session:turn".to_string()),
        input,
        tool_grants: Vec::new(),
        model_intent: Some(lash::remote::RemoteModelIntent::new("gpt-test")),
        activity_cursor: None,
        metadata: std::collections::HashMap::new(),
    };

    request.validate().unwrap();

    let host_event = lash::remote::RemoteHostEventOccurrenceRequest::new(
        "ui.button.pressed",
        "source-key",
        serde_json::json!({ "button": "Blue" }),
        "button-blue-1",
    );
    host_event.validate().unwrap();

    let filter = lash::remote::RemoteTriggerSubscriptionFilter::for_source_type(
        "ui.button.pressed",
    );
    filter.validate().unwrap();

    let report = lash::remote::RemoteHostEventEmitReport {
        protocol_version: lash::remote::REMOTE_PROTOCOL_VERSION,
        occurrence_id: "occurrence:1".to_string(),
        started_process_ids: Vec::new(),
    };
    report.validate().unwrap();

    let _cause = lash::remote::RemoteCausalRef::HostEvent {
        occurrence_id: "occurrence:1".to_string(),
    };
}
