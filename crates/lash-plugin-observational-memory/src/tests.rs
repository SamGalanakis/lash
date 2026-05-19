use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lash_core::plugin::{PluginError, PluginLifecycleEvent, PluginSessionTask, RuntimeSessionHost};
use lash_core::{
    AppendSessionNodesRequest, AppendSessionNodesResult, DirectCompletion, DirectCompletionClient,
    DirectRequest, Message, MessageRole, ObservationalMemoryConfig, Part, PartKind,
    SessionAppendNode, SessionGraph, SessionReadView, SessionStateChangedContext,
    SessionStateEnvelope,
};

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

#[derive(Default)]
struct RecordingHost {
    spawned: Mutex<Vec<(String, String)>>,
    append_requests: Mutex<Vec<(String, AppendSessionNodesRequest)>>,
    run_hidden_tasks_immediately: bool,
}

#[async_trait]
impl RuntimeSessionHost for RecordingHost {
    async fn spawn_hidden_task(
        &self,
        session_id: &str,
        label: &str,
        task: PluginSessionTask,
    ) -> Result<(), PluginError> {
        self.spawned
            .lock()
            .expect("spawned lock")
            .push((session_id.to_string(), label.to_string()));
        if self.run_hidden_tasks_immediately {
            task.await?;
        }
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        session_id: &str,
        request: AppendSessionNodesRequest,
    ) -> Result<AppendSessionNodesResult, PluginError> {
        let node_ids = request
            .nodes
            .iter()
            .enumerate()
            .map(|(index, _)| format!("appended-{index}"))
            .collect::<Vec<_>>();
        let leaf_node_id = node_ids
            .last()
            .cloned()
            .or_else(|| request.requires_ancestor_node_id.clone())
            .unwrap_or_else(|| "empty-append".to_string());
        self.append_requests
            .lock()
            .expect("append requests lock")
            .push((session_id.to_string(), request));
        Ok(AppendSessionNodesResult::Appended {
            node_ids,
            leaf_node_id,
        })
    }
}

fn post_persist_context(
    session_id: &str,
    graph: SessionGraph,
    host: Arc<dyn RuntimeSessionHost>,
) -> SessionStateChangedContext {
    post_persist_context_with_completion(session_id, graph, host, String::new())
}

fn post_persist_context_with_completion(
    session_id: &str,
    graph: SessionGraph,
    host: Arc<dyn RuntimeSessionHost>,
    completion_text: String,
) -> SessionStateChangedContext {
    SessionStateChangedContext {
        session_id: session_id.to_string(),
        state: SessionReadView::from_exported_state(&SessionStateEnvelope {
            session_id: session_id.to_string(),
            session_graph: graph,
            policy: lash_core::testing::mock_session_policy(),
            ..Default::default()
        }),
        host,
        direct_completions: DirectCompletionClient::from_fn(
            move |_request: DirectRequest, _usage_source: String| {
                let completion_text = completion_text.clone();
                Ok(DirectCompletion {
                    text: completion_text,
                    usage: Default::default(),
                })
            },
        ),
    }
}

#[tokio::test]
async fn maintenance_uses_post_persist_leaf_as_append_cas_ancestor() {
    let host = Arc::new(RecordingHost {
        run_hidden_tasks_immediately: true,
        ..Default::default()
    });
    let config = ObservationalMemoryConfig {
        observation_buffer_tokens: 1,
        observation_max_tokens_per_batch: 1,
        ..Default::default()
    };
    let hook = crate::observational_memory_event_hook(config);

    let mut graph = SessionGraph::default();
    graph.append_message(user_message("committed-message", &"x".repeat(64)).message);
    let committed_leaf = graph
        .leaf_node_id
        .clone()
        .expect("committed graph should have a leaf");
    let completion = "<observations>\nDate: May 19, 2026\n- User needs the post-persist graph as the CAS base.\n</observations>\n<current-task>\nVerify OM append ancestry.\n</current-task>\n<suggested-response>\nContinue.\n</suggested-response>"
        .to_string();

    hook(PluginLifecycleEvent::TurnPersisted(
        post_persist_context_with_completion("session", graph, host.clone(), completion),
    ))
    .await
    .expect("turn persisted hook");

    let append_requests = host.append_requests.lock().expect("append requests lock");
    assert_eq!(append_requests.len(), 1);
    let (session_id, request) = &append_requests[0];
    assert_eq!(session_id, "session");
    assert_eq!(
        request.requires_ancestor_node_id.as_deref(),
        Some(committed_leaf.as_str())
    );
    assert_eq!(request.nodes.len(), 1);
    let SessionAppendNode::Plugin { plugin_type, body } = &request.nodes[0] else {
        panic!("expected OM maintenance to append a plugin node");
    };
    assert_eq!(plugin_type, BUFFERED_OBSERVATION_PLUGIN_TYPE);
    assert_eq!(
        body.get("observed_through_message_id")
            .and_then(|value| value.as_str()),
        Some("committed-message")
    );
    assert!(
        body.get("observations")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .contains("post-persist graph")
    );
}

#[tokio::test]
async fn maintenance_hook_only_spawns_from_post_persisted_graph() {
    let host = Arc::new(RecordingHost::default());
    let config = ObservationalMemoryConfig {
        observation_buffer_tokens: 1,
        ..Default::default()
    };
    let hook = crate::observational_memory_event_hook(config);

    hook(PluginLifecycleEvent::TurnFinalized(Arc::new(
        lash_core::testing::mock_assembled_turn("session", "done"),
    )))
    .await
    .expect("turn finalized hook");
    assert!(
        host.spawned.lock().expect("spawned lock").is_empty(),
        "pre-persist turn finalization must not schedule OM maintenance"
    );

    let mut graph = SessionGraph::default();
    graph.append_message(user_message("post-persist-message", "x".repeat(64).as_str()).message);
    hook(PluginLifecycleEvent::TurnPersisted(post_persist_context(
        "session",
        graph,
        host.clone(),
    )))
    .await
    .expect("turn persisted hook");

    assert_eq!(
        host.spawned.lock().expect("spawned lock").as_slice(),
        &[(
            "session".to_string(),
            crate::OBSERVATIONAL_MEMORY_PLUGIN_ID.to_string()
        )]
    );
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
