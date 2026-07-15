use std::time::Duration;

use serde_json::Value;
use workflow_graph_roundtrip::{
    AppState, EditableValue, RunEvent, RunStatus, RunTiming, WorkflowCatalogEntry, WorkflowDocument,
};

#[tokio::test]
async fn lists_selects_projects_and_runs_built_in_workflows() {
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
        .get(format!("{base}/workflows"))
        .send()
        .await
        .expect("GET /workflows");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let catalog: Vec<WorkflowCatalogEntry> = response.json().await.expect("workflow catalog JSON");
    assert_eq!(
        catalog
            .iter()
            .map(|entry| (entry.id.as_str(), entry.name.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("onboarding", "Onboarding"),
            ("traffic-lights", "Traffic Lights"),
            ("branching-approval", "Branching Approval"),
            ("counter-loop", "Counter Loop"),
        ]
    );
    assert!(catalog.iter().all(|entry| !entry.description.is_empty()));

    for (index, entry) in catalog.iter().enumerate() {
        let response = client
            .post(format!("{base}/workflow/select"))
            .json(&serde_json::json!({ "id": entry.id }))
            .send()
            .await
            .expect("POST /workflow/select");
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let document: WorkflowDocument = response.json().await.expect("selected workflow JSON");
        assert_eq!(document.version, index as u64 + 2);
        assert!(!document.nodes.is_empty());
        assert!(!contains_key(
            &serde_json::to_value(&document).expect("workflow document value"),
            "position"
        ));

        let projected = lashlang::workflow_graph_from_source(&document.source)
            .expect("catalog source should project");
        let rendered =
            lashlang::workflow_graph_to_source(&projected).expect("catalog graph should render");
        assert_eq!(rendered, document.source);

        if entry.id == "counter-loop" {
            assert!(document.nodes.iter().any(|node| node.node_type == "opaque"));
            assert!(document.nodes.iter().any(|node| {
                node.node_type == "container" && node.data.title.starts_with("for ")
            }));
        }
    }

    let response = client
        .post(format!("{base}/workflow/select"))
        .json(&serde_json::json!({ "id": "branching-approval" }))
        .send()
        .await
        .expect("select branching approval");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let selected: WorkflowDocument = response.json().await.expect("selected workflow JSON");
    assert_eq!(selected.version, 6);

    let response = client
        .post(format!("{base}/run"))
        .send()
        .await
        .expect("POST /run");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let stream = response.text().await.expect("complete SSE stream");
    let events = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<RunEvent>(data).expect("run event JSON"))
        .collect::<Vec<_>>();

    assert!(!events.is_empty());
    let run_id = &events[0].run_id;
    assert!(events.iter().all(|event| event.run_id == *run_id));
    assert!(events.iter().all(|event| event.workflow_version == 6));
    assert!(
        events
            .iter()
            .all(|event| { selected.nodes.iter().any(|node| node.id == event.node_id) })
    );
    assert!(
        events
            .iter()
            .any(|event| event.status == RunStatus::Waiting)
    );
    assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
    assert!(
        events
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence)
    );
    let terminal_id = selected
        .nodes
        .iter()
        .find(|node| node.node_type == "terminal")
        .expect("terminal node")
        .id
        .as_str();
    let final_event = events.last().expect("final run event");
    assert_eq!(final_event.node_id, terminal_id);
    assert_eq!(final_event.status, RunStatus::Succeeded);
    assert_eq!(
        final_event
            .display
            .statuses
            .get("approval")
            .map(String::as_str),
        Some("approved")
    );
    assert!(
        final_event
            .display
            .messages
            .iter()
            .any(|message| message == "Request approved")
    );

    server.abort();
}

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
