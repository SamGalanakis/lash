use std::time::Duration;

use serde_json::Value;
use workflow_graph_roundtrip::{
    AppState, EditableValue, RunEvent, RunStatus, RunTiming, WorkflowDocument,
};

#[tokio::test]
async fn project_mutate_save_and_run_streams_correlated_events() {
    let state = AppState::with_run_timing(RunTiming {
        sleep_cap: Duration::from_millis(2),
        signal_delay: Duration::from_millis(2),
    })
    .expect("default workflow");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    let server = tokio::spawn(workflow_graph_roundtrip::serve(listener, state));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let response = client
        .get(format!("{base}/workflow"))
        .send()
        .await
        .expect("GET /workflow");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let projected_json: Value = response.json().await.expect("workflow JSON");
    assert!(!contains_key(&projected_json, "position"));
    let mut document: WorkflowDocument =
        serde_json::from_value(projected_json).expect("workflow contract");
    assert_eq!(document.version, 1);

    let message = document
        .nodes
        .iter_mut()
        .find(|node| node.data.operation.as_deref() == Some("show_message"))
        .expect("show_message node");
    assert_eq!(message.data.name_source, "derived");
    message.data.fields.insert(
        "text".to_string(),
        EditableValue::String("Edited through the graph API".to_string()),
    );

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("POST /workflow");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: WorkflowDocument = response.json().await.expect("saved workflow JSON");
    assert_eq!(saved.version, 2);
    assert!(saved.source.contains("Edited through the graph API"));
    let edited_node_id = saved
        .nodes
        .iter()
        .find(|node| {
            node.data.fields.get("text")
                == Some(&EditableValue::String(
                    "Edited through the graph API".to_string(),
                ))
        })
        .expect("saved edited node")
        .id
        .clone();

    let response = client
        .post(format!("{base}/run"))
        .send()
        .await
        .expect("POST /run");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let stream = response.text().await.expect("complete SSE stream");
    let events = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<RunEvent>(data).expect("run event JSON"))
        .collect::<Vec<_>>();

    assert!(!events.is_empty());
    let run_id = &events[0].run_id;
    assert!(events.iter().all(|event| event.run_id == *run_id));
    assert!(events.iter().all(|event| event.workflow_version == 2));
    assert!(
        events
            .iter()
            .all(|event| { saved.nodes.iter().any(|node| node.id == event.node_id) })
    );
    assert!(
        events
            .iter()
            .any(|event| event.status == RunStatus::Waiting)
    );
    assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
    assert!(events.iter().any(|event| {
        event.node_id == edited_node_id
            && event.status == RunStatus::Succeeded
            && event
                .display_delta
                .messages_appended
                .iter()
                .any(|message| message == "Edited through the graph API")
    }));
    assert!(
        events
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence)
    );

    server.abort();
}

fn contains_key(value: &Value, key: &str) -> bool {
    match value {
        Value::Array(values) => values.iter().any(|value| contains_key(value, key)),
        Value::Object(map) => {
            map.contains_key(key) || map.values().any(|value| contains_key(value, key))
        }
        _ => false,
    }
}
