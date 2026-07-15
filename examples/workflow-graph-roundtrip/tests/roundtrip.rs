use std::collections::BTreeSet;
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

    let mut run_ids = BTreeSet::new();
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
        assert_eq!(
            lashlang::parse(&rendered).expect("rendered catalog source"),
            lashlang::parse(&document.source).expect("canonical catalog source")
        );
        assert_eq!(
            lashlang::workflow_graph_from_source(&rendered).expect("reproject catalog source"),
            projected
        );

        if entry.id == "counter-loop" {
            assert!(!document.nodes.iter().any(|node| node.node_type == "opaque"));
            assert!(
                document
                    .nodes
                    .iter()
                    .any(|node| node.node_type == "state_update")
            );
            assert!(document.nodes.iter().any(|node| {
                node.node_type == "container"
                    && node.data.title == "while"
                    && node.data.children.iter().any(|child| child.slot == "body")
            }));
            assert!(document.nodes.iter().any(|node| {
                node.node_type == "container" && node.data.title.starts_with("for ")
            }));
            assert!(projected.nodes().any(|node| {
                matches!(
                    &node.kind,
                    lashlang::WorkflowNodeKind::StateUpdate { target, .. }
                        if target.contains('.')
                )
            }));
        }

        let events = run_workflow(&client, &base).await;
        assert!(!events.is_empty(), "{} should emit run events", entry.id);
        let run_id = events[0].run_id.clone();
        assert!(run_ids.insert(run_id.clone()), "each run id must be fresh");
        assert!(events.iter().all(|event| event.run_id == run_id));
        assert!(
            events
                .iter()
                .all(|event| event.workflow_version == document.version)
        );
        assert!(
            events
                .iter()
                .all(|event| { document.nodes.iter().any(|node| node.id == event.node_id) })
        );
        assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
        assert!(
            events
                .windows(2)
                .all(|pair| pair[0].sequence < pair[1].sequence)
        );
        let terminal_id = document
            .nodes
            .iter()
            .find(|node| node.node_type == "terminal")
            .expect("catalog workflow terminal node")
            .id
            .as_str();
        let final_event = events.last().expect("final run event");
        let final_title = document
            .nodes
            .iter()
            .find(|node| node.id == final_event.node_id)
            .map(|node| node.data.title.as_str());
        assert_eq!(
            final_event.node_id, terminal_id,
            "{} final event was {final_title:?}",
            entry.id
        );
        assert_eq!(final_event.status, RunStatus::Succeeded);

        if entry.id == "branching-approval" {
            assert!(
                events
                    .iter()
                    .any(|event| event.status == RunStatus::Waiting)
            );
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
        }
    }
    assert_eq!(run_ids.len(), catalog.len());

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

    let events = run_workflow(&client, &base).await;

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

#[tokio::test]
async fn edited_counter_loop_condition_saves_reprojects_and_runs() {
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

    let mut document: WorkflowDocument = client
        .post(format!("{base}/workflow/select"))
        .json(&serde_json::json!({ "id": "counter-loop" }))
        .send()
        .await
        .expect("select counter-loop")
        .json()
        .await
        .expect("counter-loop document");
    let while_node = document
        .nodes
        .iter_mut()
        .find(|node| node.node_type == "container" && node.data.title == "while")
        .expect("counter-loop while node");
    assert_eq!(
        while_node.data.condition.as_deref(),
        Some("(state.count < 3)")
    );
    while_node.data.condition = Some("(state.count < 1)".to_string());

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save edited counter-loop");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: WorkflowDocument = response.json().await.expect("saved counter-loop");
    assert!(saved.source.contains("while (state.count < 1)"));
    assert!(saved.nodes.iter().any(|node| {
        node.node_type == "container" && node.data.condition.as_deref() == Some("(state.count < 1)")
    }));

    let events = run_workflow(&client, &base).await;
    assert!(!events.is_empty());
    assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
    let final_event = events.last().expect("terminal event");
    assert_eq!(final_event.status, RunStatus::Succeeded);
    assert_eq!(
        final_event.display.lists.get("counts"),
        Some(&vec!["0".to_string()])
    );

    server.abort();
}

#[tokio::test]
async fn delete_node_edit_round_trips_and_runs_the_saved_graph() {
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

    let mut document: WorkflowDocument = client
        .get(format!("{base}/workflow"))
        .send()
        .await
        .expect("GET /workflow")
        .json()
        .await
        .expect("workflow document");
    let original_node_count = document.nodes.len();
    let deleted_id = document
        .nodes
        .iter()
        .find(|node| {
            node.data.operation.as_deref() == Some("set_light")
                && node.data.fields.get("name")
                    == Some(&EditableValue::String("complete".to_string()))
        })
        .expect("complete light node")
        .id
        .clone();
    document.nodes.retain(|node| node.id != deleted_id);
    document
        .edges
        .retain(|edge| edge.source != deleted_id && edge.target != deleted_id);
    document.roots.main.retain(|node_id| node_id != &deleted_id);
    document
        .roots
        .processes
        .retain(|node_id| node_id != &deleted_id);
    for node in &mut document.nodes {
        for child in &mut node.data.children {
            child.node_ids.retain(|node_id| node_id != &deleted_id);
        }
    }

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("POST deleted workflow node");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: WorkflowDocument = response.json().await.expect("saved workflow JSON");
    assert_eq!(saved.version, 2);
    assert_eq!(saved.nodes.len(), original_node_count - 1);
    assert!(!saved.source.contains("name: \"complete\""));
    let projected = lashlang::workflow_graph_from_source(&saved.source)
        .expect("deleted workflow source should reproject");
    assert_eq!(
        lashlang::workflow_graph_to_source(&projected).expect("deleted graph should render"),
        saved.source
    );

    let events = run_workflow(&client, &base).await;
    assert!(!events.is_empty());
    assert!(events.iter().all(|event| event.workflow_version == 2));
    assert!(
        events
            .iter()
            .all(|event| saved.nodes.iter().any(|node| node.id == event.node_id))
    );
    assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
    assert_eq!(
        events.last().expect("terminal event").status,
        RunStatus::Succeeded
    );

    server.abort();
}

#[tokio::test]
async fn invalid_graph_post_returns_typed_unprocessable_entity() {
    let state = AppState::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    let server = tokio::spawn(workflow_graph_roundtrip::serve(listener, state));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let mut document: WorkflowDocument = client
        .get(format!("{base}/workflow"))
        .send()
        .await
        .expect("GET /workflow")
        .json()
        .await
        .expect("workflow document");
    document.schema_version += 1;

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("POST invalid workflow graph");
    assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value = response.json().await.expect("typed error response");
    assert_eq!(body["error"]["code"], "unsupported_schema_version");
    assert_eq!(
        body["error"]["details"]["found"],
        lashlang::WORKFLOW_GRAPH_SCHEMA_VERSION + 1
    );
    assert_eq!(
        body["error"]["details"]["expected"],
        lashlang::WORKFLOW_GRAPH_SCHEMA_VERSION
    );

    document.schema_version = lashlang::WORKFLOW_GRAPH_SCHEMA_VERSION;
    let while_node = document
        .nodes
        .iter_mut()
        .find(|node| node.node_type == "container" && node.data.title == "while")
        .expect("default workflow while container");
    let original_condition = while_node.data.condition.clone();
    while_node.data.condition = Some("count <".to_string());
    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("POST invalid while condition");
    assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value = response.json().await.expect("typed expression response");
    assert_eq!(body["error"]["code"], "invalid_expression");
    assert_eq!(body["error"]["details"]["field"], "condition");

    let while_node = document
        .nodes
        .iter_mut()
        .find(|node| node.node_type == "container" && node.data.title == "while")
        .expect("default workflow while container");
    while_node.data.condition = original_condition;
    while_node
        .data
        .children
        .retain(|child| child.slot != "body");
    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("POST while without body");
    assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value = response.json().await.expect("typed missing-child response");
    assert_eq!(body["error"]["code"], "missing_required_child");
    assert_eq!(body["error"]["details"]["child"], "body");

    let mut document: WorkflowDocument = client
        .post(format!("{base}/workflow/select"))
        .json(&serde_json::json!({ "id": "counter-loop" }))
        .send()
        .await
        .expect("select counter-loop")
        .json()
        .await
        .expect("counter-loop document");
    let invalid_target = "state.count + 1";
    assert!(
        lashlang::parse_expression(invalid_target).is_ok(),
        "invalid assignment target should remain a valid expression"
    );
    let state_update = document
        .nodes
        .iter_mut()
        .find(|node| node.node_type == "state_update")
        .expect("counter-loop state update");
    state_update.data.target = Some(invalid_target.to_string());
    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("POST invalid state update target");
    assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value = response
        .json()
        .await
        .expect("typed assignment-target response");
    assert_eq!(body["error"]["code"], "invalid_assignment_target");
    assert_eq!(body["error"]["details"]["field"], "target");

    server.abort();
}

async fn run_workflow(client: &reqwest::Client, base: &str) -> Vec<RunEvent> {
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
    stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<RunEvent>(data).expect("run event JSON"))
        .collect()
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
