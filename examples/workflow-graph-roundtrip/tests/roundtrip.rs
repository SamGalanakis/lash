use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde_json::Value;
use workflow_graph_roundtrip::{
    AppState, ChildGroup, EditableValue, FlowNode, NodeData, RunEvent, RunStatus, RunTiming,
    SaveWorkflowResponse, WorkflowCatalogEntry, WorkflowDocument,
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
            ("blank", "Blank workflow"),
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
async fn expression_valued_call_fields_save_reproject_and_reject_malformed_edits() {
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

    let mut document = select_workflow(&client, &base, "counter-loop").await;
    let expected_progress = lashlang::canonical_expression_source(
        &lashlang::parse_expression("state.count * 20 + 20").expect("progress expression"),
    )
    .expect("canonical progress expression");
    let expected_item = lashlang::canonical_expression_source(
        &lashlang::parse_expression("state.count").expect("item expression"),
    )
    .expect("canonical item expression");
    assert!(document.nodes.iter().any(|node| {
        node.data.operation.as_deref() == Some("add_item")
            && node.data.fields.get("item") == Some(&EditableValue::Expr(expected_item.clone()))
    }));
    let progress = document
        .nodes
        .iter_mut()
        .find(|node| {
            node.data.operation.as_deref() == Some("set_progress")
                && node.data.fields.get("pct")
                    == Some(&EditableValue::Expr(expected_progress.clone()))
        })
        .expect("expression-valued progress node");
    progress.data.fields.insert(
        "pct".to_string(),
        EditableValue::Expr("state.count * 10 + 10".to_string()),
    );

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save expression-valued field");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let mut saved: WorkflowDocument = response.json().await.expect("saved workflow");
    let edited_progress = lashlang::canonical_expression_source(
        &lashlang::parse_expression("state.count * 10 + 10").expect("edited expression"),
    )
    .expect("canonical edited expression");
    assert!(saved.nodes.iter().any(|node| {
        node.data.operation.as_deref() == Some("set_progress")
            && node.data.fields.get("pct") == Some(&EditableValue::Expr(edited_progress.clone()))
    }));
    assert!(saved.source.contains(&edited_progress));

    let malformed = saved
        .nodes
        .iter_mut()
        .find(|node| {
            node.data.operation.as_deref() == Some("set_progress")
                && node.data.fields.get("pct")
                    == Some(&EditableValue::Expr(edited_progress.clone()))
        })
        .expect("edited progress node");
    malformed.data.fields.insert(
        "pct".to_string(),
        EditableValue::Expr("state.count +".to_string()),
    );
    let response = client
        .post(format!("{base}/workflow"))
        .json(&saved)
        .send()
        .await
        .expect("save malformed expression-valued field");
    assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value = response.json().await.expect("typed expression error");
    assert_eq!(body["error"]["code"], "invalid_expression");
    assert_eq!(body["error"]["details"]["field"], "fields.pct");

    server.abort();
}

#[tokio::test]
async fn edited_if_condition_and_for_iterable_save_reproject_and_run() {
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

    let mut if_document: WorkflowDocument = client
        .post(format!("{base}/workflow/select"))
        .json(&serde_json::json!({ "id": "branching-approval" }))
        .send()
        .await
        .expect("select branching-approval")
        .json()
        .await
        .expect("branching-approval document");
    let if_node = if_document
        .nodes
        .iter_mut()
        .find(|node| {
            node.node_type == "container" && node.data.condition.as_deref() == Some("true")
        })
        .expect("branching-approval literal if node");
    if_node.data.condition = Some("false".to_string());

    let response = client
        .post(format!("{base}/workflow"))
        .json(&if_document)
        .send()
        .await
        .expect("save edited branching-approval");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved_if: WorkflowDocument = response.json().await.expect("saved branching-approval");
    assert!(saved_if.source.contains("if false"));
    assert!(saved_if.nodes.iter().any(|node| {
        node.node_type == "container" && node.data.condition.as_deref() == Some("false")
    }));
    let projected_if = lashlang::workflow_graph_from_source(&saved_if.source)
        .expect("edited branching-approval source should reproject");
    assert_eq!(
        lashlang::workflow_graph_to_source(&projected_if)
            .expect("edited branching-approval graph should render"),
        saved_if.source
    );

    let if_events = run_workflow(&client, &base).await;
    assert!(!if_events.is_empty());
    assert!(
        if_events
            .iter()
            .all(|event| event.workflow_version == saved_if.version)
    );
    assert!(
        if_events
            .iter()
            .all(|event| { saved_if.nodes.iter().any(|node| node.id == event.node_id) })
    );
    assert!(
        !if_events
            .iter()
            .any(|event| event.status == RunStatus::Failed)
    );
    let final_if_event = if_events.last().expect("branching terminal event");
    assert_eq!(final_if_event.status, RunStatus::Succeeded);
    assert!(
        final_if_event
            .display
            .messages
            .iter()
            .any(|message| message == "Approval needs review")
    );

    let mut for_document: WorkflowDocument = client
        .post(format!("{base}/workflow/select"))
        .json(&serde_json::json!({ "id": "counter-loop" }))
        .send()
        .await
        .expect("select counter-loop")
        .json()
        .await
        .expect("counter-loop document");
    let for_node = for_document
        .nodes
        .iter_mut()
        .find(|node| node.node_type == "container" && node.data.iterable.is_some())
        .expect("counter-loop for node");
    assert_eq!(for_node.data.iterable.as_deref(), Some("[70, 85, 100]"));
    for_node.data.iterable = Some("[15]".to_string());

    let response = client
        .post(format!("{base}/workflow"))
        .json(&for_document)
        .send()
        .await
        .expect("save edited counter-loop for iterable");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved_for: WorkflowDocument = response
        .json()
        .await
        .expect("saved counter-loop for iterable");
    assert!(saved_for.source.contains("for pct in [15]"));
    assert!(saved_for.nodes.iter().any(|node| {
        node.node_type == "container" && node.data.iterable.as_deref() == Some("[15]")
    }));
    let projected_for = lashlang::workflow_graph_from_source(&saved_for.source)
        .expect("edited counter-loop source should reproject");
    assert_eq!(
        lashlang::workflow_graph_to_source(&projected_for)
            .expect("edited counter-loop graph should render"),
        saved_for.source
    );

    let for_events = run_workflow(&client, &base).await;
    assert!(!for_events.is_empty());
    assert!(
        for_events
            .iter()
            .all(|event| event.workflow_version == saved_for.version)
    );
    assert!(
        for_events
            .iter()
            .all(|event| { saved_for.nodes.iter().any(|node| node.id == event.node_id) })
    );
    assert!(
        !for_events
            .iter()
            .any(|event| event.status == RunStatus::Failed)
    );
    let final_for_event = for_events.last().expect("counter-loop terminal event");
    assert_eq!(final_for_event.status, RunStatus::Succeeded);
    assert_eq!(final_for_event.display.progress, 15.0);

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
async fn new_call_node_saves_reprojects_and_runs_with_canonical_correlation() {
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

    let mut document = select_workflow(&client, &base, "blank").await;
    let mut call = new_flow_node("new:call", "call", None, "Set status");
    call.data.expression =
        Some("await display.set_status({ key: \"k\", value: \"v\" })?".to_string());
    append_process_node(&mut document, call);
    let posted_ids = document
        .nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save workflow with new call");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved_response: SaveWorkflowResponse = response.json().await.expect("saved workflow");
    let saved = saved_response.document;
    assert_eq!(saved_response.id_map.len(), posted_ids.len());
    assert!(posted_ids.iter().all(|posted_id| {
        saved_response
            .id_map
            .get(posted_id)
            .is_some_and(|reprojected_id| saved.nodes.iter().any(|node| &node.id == reprojected_id))
    }));
    assert!(
        saved_response
            .id_map
            .get("new:call")
            .is_some_and(|reprojected_id| {
                saved.nodes.iter().any(|node| {
                    &node.id == reprojected_id
                        && node.data.operation.as_deref() == Some("set_status")
                })
            })
    );
    assert!(
        saved
            .source
            .contains("await display.set_status({ key: \"k\", value: \"v\" })?")
    );
    let call_id = saved
        .nodes
        .iter()
        .find(|node| node.data.operation.as_deref() == Some("set_status"))
        .expect("reprojected call node")
        .id
        .clone();
    assert_ne!(call_id, "new:call");

    let events = run_workflow(&client, &base).await;
    assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
    assert!(events.iter().any(|event| {
        event.node_id == call_id
            && event.status == RunStatus::Succeeded
            && event.display.statuses.get("k").map(String::as_str) == Some("v")
    }));
    assert!(
        events
            .iter()
            .all(|event| saved.nodes.iter().any(|node| node.id == event.node_id))
    );

    server.abort();
}

#[tokio::test]
async fn new_if_while_and_for_containers_save_reproject_and_run() {
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

    let mut document = select_workflow(&client, &base, "blank").await;

    let mut if_child = new_flow_node("new:if-call", "call", None, "If message");
    if_child.parent_id = Some("new:if".to_string());
    if_child.data.expression =
        Some("await display.show_message({ text: \"Created if body\" })?".to_string());
    let mut if_node = new_flow_node("new:if", "container", Some("if"), "if");
    if_node.data.condition = Some("true".to_string());
    if_node.data.children = vec![
        ChildGroup {
            slot: "then".to_string(),
            scope: "container:new-if:then".to_string(),
            node_ids: vec![if_child.id.clone()],
        },
        ChildGroup {
            slot: "else".to_string(),
            scope: "container:new-if:else".to_string(),
            node_ids: Vec::new(),
        },
    ];
    append_process_node(&mut document, if_node);
    document.nodes.push(if_child);

    let mut while_child = new_flow_node("new:while-call", "call", None, "While message");
    while_child.parent_id = Some("new:while".to_string());
    while_child.data.expression =
        Some("await display.show_message({ text: \"Created while body\" })?".to_string());
    let mut while_node = new_flow_node("new:while", "container", Some("while"), "while");
    while_node.data.condition = Some("false".to_string());
    while_node.data.children = vec![ChildGroup {
        slot: "body".to_string(),
        scope: "container:new-while:body".to_string(),
        node_ids: vec![while_child.id.clone()],
    }];
    append_process_node(&mut document, while_node);
    document.nodes.push(while_child);

    let mut for_child = new_flow_node("new:for-call", "call", None, "For status");
    for_child.parent_id = Some("new:for".to_string());
    for_child.data.expression =
        Some("await display.set_status({ key: \"loop\", value: \"ran\" })?".to_string());
    let mut for_node = new_flow_node("new:for", "container", Some("for"), "for item");
    for_node.data.binding = Some("item".to_string());
    for_node.data.iterable = Some("[1]".to_string());
    for_node.data.children = vec![ChildGroup {
        slot: "body".to_string(),
        scope: "container:new-for:body".to_string(),
        node_ids: vec![for_child.id.clone()],
    }];
    append_process_node(&mut document, for_node);
    document.nodes.push(for_child);

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save workflow with new containers");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: WorkflowDocument = response.json().await.expect("saved workflow");
    assert!(saved.source.contains("if true"));
    assert!(saved.source.contains("while false"));
    assert!(saved.source.contains("for item in [1]"));
    for subkind in ["if", "while", "for"] {
        assert!(saved.nodes.iter().any(|node| {
            node.data.kind == "container" && node.data.subkind.as_deref() == Some(subkind)
        }));
    }

    let events = run_workflow(&client, &base).await;
    assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
    let final_event = events.last().expect("terminal event");
    assert_eq!(final_event.status, RunStatus::Succeeded);
    assert_eq!(
        final_event.display.statuses.get("loop").map(String::as_str),
        Some("ran")
    );
    assert!(
        final_event
            .display
            .messages
            .iter()
            .any(|message| message == "Created if body")
    );

    server.abort();
}

#[tokio::test]
async fn new_statement_containers_allow_empty_bodies() {
    let state = AppState::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    let server = tokio::spawn(workflow_graph_roundtrip::serve(listener, state));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let mut document = select_workflow(&client, &base, "blank").await;
    let mut if_node = new_flow_node("new:empty-if", "container", Some("if"), "if");
    if_node.data.condition = Some("true".to_string());
    append_process_node(&mut document, if_node);
    let mut while_node = new_flow_node("new:empty-while", "container", Some("while"), "while");
    while_node.data.condition = Some("false".to_string());
    append_process_node(&mut document, while_node);
    let mut for_node = new_flow_node("new:empty-for", "container", Some("for"), "for item");
    for_node.data.binding = Some("item".to_string());
    for_node.data.iterable = Some("[]".to_string());
    append_process_node(&mut document, for_node);

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save empty statement containers");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: WorkflowDocument = response.json().await.expect("saved workflow");
    assert!(saved.source.contains("if true {}"));
    assert!(saved.source.contains("while false {}"));
    assert!(saved.source.contains("for item in [] {}"));

    let events = run_workflow(&client, &base).await;
    assert_eq!(
        events.last().expect("terminal event").status,
        RunStatus::Succeeded
    );

    server.abort();
}

#[tokio::test]
async fn blank_workflow_grows_by_two_nodes_then_saves_and_runs() {
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

    let mut document = select_workflow(&client, &base, "blank").await;
    assert_eq!(document.source, "process blank() {\n  finish 0\n}\n");

    let mut status = new_flow_node("new:blank-status", "call", None, "Blank status");
    status.data.expression =
        Some("await display.set_status({ key: \"blank\", value: \"built\" })?".to_string());
    append_process_node(&mut document, status);
    let mut message = new_flow_node("new:blank-message", "call", None, "Blank message");
    message.data.expression =
        Some("await display.show_message({ text: \"Built from blank\" })?".to_string());
    append_process_node(&mut document, message);

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save grown blank workflow");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: WorkflowDocument = response.json().await.expect("saved workflow");
    assert!(saved.source.contains("display.set_status"));
    assert!(saved.source.contains("display.show_message"));

    let events = run_workflow(&client, &base).await;
    let final_event = events.last().expect("terminal event");
    assert_eq!(final_event.status, RunStatus::Succeeded);
    assert_eq!(
        final_event
            .display
            .statuses
            .get("blank")
            .map(String::as_str),
        Some("built")
    );
    assert!(
        final_event
            .display
            .messages
            .iter()
            .any(|message| message == "Built from blank")
    );

    server.abort();
}

#[tokio::test]
async fn reordered_process_statements_save_reproject_and_run_in_node_id_order() {
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

    let mut document = select_workflow(&client, &base, "blank").await;
    let mut first = new_flow_node("new:first", "call", None, "First message");
    first.data.expression = Some("await display.show_message({ text: \"First\" })?".to_string());
    append_process_node(&mut document, first);
    let mut second = new_flow_node("new:second", "call", None, "Second message");
    second.data.expression = Some("await display.show_message({ text: \"Second\" })?".to_string());
    append_process_node(&mut document, second);

    let process = document
        .nodes
        .iter_mut()
        .find(|node| node.node_type == "process")
        .expect("process container");
    let body = process
        .data
        .children
        .iter_mut()
        .find(|child| child.slot == "body")
        .expect("process body");
    let first_index = body
        .node_ids
        .iter()
        .position(|id| id == "new:first")
        .expect("first node membership");
    let second_index = body
        .node_ids
        .iter()
        .position(|id| id == "new:second")
        .expect("second node membership");
    body.node_ids.swap(first_index, second_index);

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save reordered workflow");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: SaveWorkflowResponse = response.json().await.expect("saved reordered workflow");
    let second_source_index = saved
        .document
        .source
        .find("text: \"Second\"")
        .expect("second statement source");
    let first_source_index = saved
        .document
        .source
        .find("text: \"First\"")
        .expect("first statement source");
    assert!(second_source_index < first_source_index);

    let events = run_workflow(&client, &base).await;
    assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
    assert_eq!(
        events.last().expect("terminal event").display.messages,
        vec!["Second".to_string(), "First".to_string()]
    );

    server.abort();
}

#[tokio::test]
async fn moved_statement_between_scopes_saves_reprojects_and_runs_in_new_scope() {
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

    let mut document = select_workflow(&client, &base, "branching-approval").await;
    let process_id = document
        .nodes
        .iter()
        .find(|node| node.node_type == "process")
        .expect("process container")
        .id
        .clone();
    let moved_id = document
        .nodes
        .iter()
        .find(|node| {
            node.data.fields.get("text")
                == Some(&EditableValue::String("Approval needs review".to_string()))
        })
        .expect("inner else-branch message")
        .id
        .clone();
    let original_parent = document
        .nodes
        .iter()
        .find(|node| node.id == moved_id)
        .and_then(|node| node.parent_id.clone())
        .expect("nested message parent");
    assert_ne!(original_parent, process_id);

    for node in &mut document.nodes {
        for child in &mut node.data.children {
            child.node_ids.retain(|id| id != &moved_id);
        }
    }
    let terminal_id = document
        .nodes
        .iter()
        .find(|node| node.node_type == "terminal")
        .expect("terminal node")
        .id
        .clone();
    let process = document
        .nodes
        .iter_mut()
        .find(|node| node.id == process_id)
        .expect("process container");
    let body = process
        .data
        .children
        .iter_mut()
        .find(|child| child.slot == "body")
        .expect("process body");
    let terminal_index = body
        .node_ids
        .iter()
        .position(|id| id == &terminal_id)
        .expect("terminal membership");
    body.node_ids.insert(terminal_index, moved_id.clone());
    document
        .nodes
        .iter_mut()
        .find(|node| node.id == moved_id)
        .expect("moved message node")
        .parent_id = Some(process_id);

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save moved workflow");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: SaveWorkflowResponse = response.json().await.expect("saved moved workflow");
    let canonical_moved_id = saved
        .id_map
        .get(&moved_id)
        .expect("moved node ID reconciliation");
    let canonical_process = saved
        .document
        .nodes
        .iter()
        .find(|node| node.node_type == "process")
        .expect("canonical process");
    assert_eq!(
        saved
            .document
            .nodes
            .iter()
            .find(|node| &node.id == canonical_moved_id)
            .and_then(|node| node.parent_id.as_ref()),
        Some(&canonical_process.id)
    );
    assert!(canonical_process.data.children.iter().any(|child| {
        child.slot == "body" && child.node_ids.iter().any(|id| id == canonical_moved_id)
    }));
    let highlight_index = saved
        .document
        .source
        .find("target: \"result\"")
        .expect("result highlight source");
    let moved_index = saved
        .document
        .source
        .find("text: \"Approval needs review\"")
        .expect("moved message source");
    assert!(highlight_index < moved_index);

    let events = run_workflow(&client, &base).await;
    assert!(!events.iter().any(|event| event.status == RunStatus::Failed));
    assert!(
        events
            .last()
            .expect("terminal event")
            .display
            .messages
            .iter()
            .any(|message| message == "Approval needs review")
    );

    server.abort();
}

#[tokio::test]
async fn new_call_without_receiver_expression_returns_typed_error() {
    let state = AppState::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    let server = tokio::spawn(workflow_graph_roundtrip::serve(listener, state));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let mut document = select_workflow(&client, &base, "blank").await;
    let mut call = new_flow_node("new:invalid-call", "call", None, "Invalid call");
    call.data.expression = Some("1 + 2".to_string());
    append_process_node(&mut document, call);

    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save invalid new call");
    assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value = response.json().await.expect("typed error response");
    assert_eq!(body["error"]["code"], "invalid_expression");
    assert_eq!(body["error"]["details"]["nodeId"], "new:invalid-call");
    assert_eq!(body["error"]["details"]["field"], "expression");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("needs a receiver call"))
    );

    let mut document = select_workflow(&client, &base, "blank").await;
    let comprehension = new_flow_node(
        "new:empty-comprehension",
        "container",
        Some("comprehension"),
        "list comprehension",
    );
    append_process_node(&mut document, comprehension);
    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("save empty comprehension");
    assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
    let body: Value = response.json().await.expect("typed error response");
    assert_eq!(body["error"]["code"], "invalid_node_payload");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("container body cannot be empty"))
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

async fn select_workflow(client: &reqwest::Client, base: &str, id: &str) -> WorkflowDocument {
    let response = client
        .post(format!("{base}/workflow/select"))
        .json(&serde_json::json!({ "id": id }))
        .send()
        .await
        .expect("POST /workflow/select");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    response.json().await.expect("selected workflow document")
}

fn new_flow_node(id: &str, kind: &str, subkind: Option<&str>, title: &str) -> FlowNode {
    FlowNode {
        id: id.to_string(),
        node_type: kind.to_string(),
        parent_id: None,
        data: NodeData {
            kind: kind.to_string(),
            subkind: subkind.map(str::to_string),
            title: title.to_string(),
            description: None,
            name_source: "derived".to_string(),
            operation: None,
            effect: None,
            fields: BTreeMap::new(),
            binding: None,
            target: None,
            expression: None,
            condition: None,
            iterable: None,
            clauses: Vec::new(),
            source: None,
            children: Vec::new(),
        },
    }
}

fn append_process_node(document: &mut WorkflowDocument, mut node: FlowNode) {
    let process_index = document
        .nodes
        .iter()
        .position(|candidate| candidate.node_type == "process")
        .expect("process container");
    let process_id = document.nodes[process_index].id.clone();
    let terminal_id = document
        .nodes
        .iter()
        .find_map(|candidate| (candidate.node_type == "terminal").then(|| candidate.id.clone()));
    let body = document.nodes[process_index]
        .data
        .children
        .iter_mut()
        .find(|child| child.slot == "body")
        .expect("process body");
    let insert_at = terminal_id
        .as_ref()
        .and_then(|terminal_id| {
            body.node_ids
                .iter()
                .position(|node_id| node_id == terminal_id)
        })
        .unwrap_or(body.node_ids.len());
    body.node_ids.insert(insert_at, node.id.clone());
    node.parent_id = Some(process_id);
    document.nodes.push(node);
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
