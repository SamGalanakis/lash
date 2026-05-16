use lash_core::{Message, MessageRole, Part, PartKind, SessionGraph};

use crate::constants::{ACTIVE_STATE_PLUGIN_TYPE, BUFFERED_OBSERVATION_PLUGIN_TYPE};
use crate::graph_state::{
    build_graph_state, prefix_len_covering_tokens, retained_message_tokens_by_message_id,
};
use crate::model::MessageNode;
use crate::prompts::parse_memory_output;

fn user_message(id: &str, content: &str) -> MessageNode {
    MessageNode {
        timestamp: "2026-04-14T10:00:00Z".to_string(),
        message: Message {
            id: id.to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: lash_core::PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: None,
        },
    }
}

#[test]
fn build_graph_state_resets_buffers_after_active_state() {
    let mut graph = SessionGraph::default();
    graph.append_message(user_message("m1", "hello").message);
    graph.append_plugin(
        BUFFERED_OBSERVATION_PLUGIN_TYPE,
        serde_json::json!({
            "observed_through_message_id": "m1",
            "observations": "old buffered",
            "observation_tokens": 10
        }),
    );
    graph.append_plugin(
        ACTIVE_STATE_PLUGIN_TYPE,
        serde_json::json!({
            "observed_through_message_id": "m1",
            "observations": "active memory"
        }),
    );
    graph.append_message(user_message("m2", "need help").message);
    graph.append_plugin(
        BUFFERED_OBSERVATION_PLUGIN_TYPE,
        serde_json::json!({
            "observed_through_message_id": "m2",
            "observations": "new buffered",
            "observation_tokens": 20
        }),
    );

    let state = build_graph_state(&graph);
    assert_eq!(
        state.active.as_ref().map(|item| item.observations.as_str()),
        Some("active memory")
    );
    assert_eq!(state.buffered_observations.len(), 1);
    assert_eq!(
        state.buffered_observations[0].observations,
        "new buffered".to_string()
    );
}

#[test]
fn retained_message_tokens_tracks_suffix_after_message() {
    let messages = vec![
        user_message("m1", &"a".repeat(4000)),
        user_message("m2", &"b".repeat(4000)),
        user_message("m3", &"c".repeat(4000)),
    ];
    let retained = retained_message_tokens_by_message_id(&messages);
    assert_eq!(retained.get("m3").copied(), Some(0));
    assert!(retained.get("m2").copied().unwrap_or_default() > 0);
    assert!(retained.get("m1").copied().unwrap_or_default() > retained["m2"]);
}

#[test]
fn prefix_len_covering_tokens_handles_partial_prefix() {
    let messages = vec![
        user_message("m1", &"a".repeat(4000)),
        user_message("m2", &"b".repeat(4000)),
        user_message("m3", &"c".repeat(4000)),
    ];
    let prefix = prefix_len_covering_tokens(&messages, 2000).expect("prefix");
    assert_eq!(prefix, 2);
}

#[test]
fn parse_memory_output_extracts_xml_sections() {
    let parsed = parse_memory_output(
        "<observations>\nDate: Apr 12, 2026\n* 🔴 Test\n</observations>\n<current-task>\nWork\n</current-task>\n<suggested-response>\nContinue\n</suggested-response>",
    );
    assert!(parsed.observations.contains("Test"));
    assert_eq!(parsed.current_task.as_deref(), Some("Work"));
    assert_eq!(parsed.suggested_response.as_deref(), Some("Continue"));
}
