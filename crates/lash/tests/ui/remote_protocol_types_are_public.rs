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
}
