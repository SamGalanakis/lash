//! Compiled sources for the Rust snippets on `docs/remote-protocol.html`.

fn remote_turn_request(
    chat_id: String,
    turn_id: String,
    idempotency_key: String,
    trace_turn_id: String,
) -> anyhow::Result<()> {
    // docs:start:remote-turn-request
    use lash::remote::{
        REMOTE_PROTOCOL_VERSION, RemoteInputItem, RemoteTurnInput, RemoteTurnRequest,
    };

    let request = RemoteTurnRequest {
        protocol_version: REMOTE_PROTOCOL_VERSION,
        session_id: chat_id.clone(),
        turn_id: turn_id.clone(),
        idempotency_key: Some(idempotency_key.clone()),
        input: RemoteTurnInput {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            items: vec![RemoteInputItem::Text {
                text: "Summarize this task.".to_string(),
            }],
            image_blobs_base64: Default::default(),
            protocol_turn_options: None,
            trace_turn_id: Some(trace_turn_id.clone()),
            prompt_layer: None,
        },
        tool_grants: Vec::new(),
        model_intent: None,
        metadata: Default::default(),
    };

    request.validate()?;
    // docs:end:remote-turn-request
    Ok(())
}
