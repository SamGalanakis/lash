use super::*;

#[test]
fn stream_accumulator_merges_adjacent_display_reasoning_chunks() {
    let mut accumulator = StandardStreamAccumulator::default();
    accumulator.push_reasoning("I'll".to_string(), None, Vec::new(), None);
    accumulator.push_reasoning(" check".to_string(), None, Vec::new(), None);
    accumulator.push_reasoning(" the time.".to_string(), None, Vec::new(), None);

    assert_eq!(accumulator.parts.len(), 1);
    assert!(matches!(
        &accumulator.parts[0],
        LlmOutputPart::Reasoning { text, .. } if text == "I'll check the time."
    ));
}

#[test]
fn stream_accumulator_enriches_reasoning_delta_with_later_roundtrip_payload() {
    let mut accumulator = StandardStreamAccumulator::default();
    accumulator.push_reasoning("I'll check the time.".to_string(), None, Vec::new(), None);
    accumulator.push_reasoning(
        "I'll check the time.".to_string(),
        Some("rs_1".to_string()),
        vec!["I'll check the time.".to_string()],
        Some("encrypted".to_string()),
    );

    assert_eq!(accumulator.parts.len(), 1);
    assert!(matches!(
        &accumulator.parts[0],
        LlmOutputPart::Reasoning {
            text,
            item_id: Some(item_id),
            encrypted_content: Some(encrypted),
            ..
        } if text == "I'll check the time." && item_id == "rs_1" && encrypted == "encrypted"
    ));
}

#[test]
fn stream_accumulator_preserves_reasoning_when_final_response_has_tool_call() {
    let mut accumulator = StandardStreamAccumulator::default();
    accumulator.push_reasoning("I'll check the time.".to_string(), None, Vec::new(), None);
    accumulator.push_tool_call(
        "call_1".to_string(),
        "exec_command".to_string(),
        "{\"cmd\":\"date\"}".to_string(),
        Some("item_1".to_string()),
        Some("sig".to_string()),
    );

    let mut response = LlmResponse {
        full_text: String::new(),
        parts: vec![LlmOutputPart::ToolCall {
            call_id: "call_1".to_string(),
            tool_name: "exec_command".to_string(),
            input_json: "{\"cmd\":\"date\"}".to_string(),
            item_id: Some("item_1".to_string()),
            signature: Some("sig".to_string()),
        }],
        ..Default::default()
    };

    accumulator.apply_to_response(&mut response);

    assert_eq!(response.parts.len(), 2);
    assert!(matches!(
        &response.parts[0],
        LlmOutputPart::Reasoning { text, .. } if text == "I'll check the time."
    ));
    assert!(matches!(
        &response.parts[1],
        LlmOutputPart::ToolCall { tool_name, .. } if tool_name == "exec_command"
    ));
}

#[test]
fn stream_accumulator_does_not_duplicate_complete_final_response() {
    let mut accumulator = StandardStreamAccumulator::default();
    accumulator.push_reasoning("I'll answer.".to_string(), None, Vec::new(), None);
    accumulator.push_text("Done.");

    let mut response = LlmResponse {
        full_text: "Done.".to_string(),
        parts: vec![
            LlmOutputPart::Reasoning {
                text: "I'll answer.".to_string(),
                signature: None,
                redacted: false,
                item_id: None,
                encrypted_content: None,
                summary: Vec::new(),
            },
            LlmOutputPart::Text {
                text: "Done.".to_string(),
                response_meta: None,
            },
        ],
        ..Default::default()
    };

    accumulator.apply_to_response(&mut response);

    assert_eq!(response.parts.len(), 2);
    assert!(matches!(
        &response.parts[0],
        LlmOutputPart::Reasoning { text, .. } if text == "I'll answer."
    ));
    assert!(matches!(
        &response.parts[1],
        LlmOutputPart::Text { text, .. } if text == "Done."
    ));
}
