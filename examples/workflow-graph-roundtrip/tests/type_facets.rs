use serde_json::Value;
use workflow_graph_roundtrip::{AppState, SaveWorkflowResponse, WorkflowDocument};

#[tokio::test]
async fn type_facets_are_projected_and_client_echoes_are_ignored_on_save() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    let server = tokio::spawn(workflow_graph_roundtrip::serve(
        listener,
        AppState::default(),
    ));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");
    let response = client
        .post(format!("{base}/project"))
        .json(&serde_json::json!({
            "source": r#"
                process typed(name: str) {
                  result = await display.show_message({ text: name })?
                  for item in "not a list" { seen = item }
                  finish result
                }
            "#
        }))
        .send()
        .await
        .expect("POST typed source");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: Value = response.json().await.expect("typed project JSON");
    let mut document: WorkflowDocument =
        serde_json::from_value(body["document"].clone()).expect("typed document");
    assert_eq!(
        document.facet_schema_version,
        Some(lashlang::WORKFLOW_TYPE_FACET_SCHEMA_VERSION)
    );
    let call = document
        .nodes
        .iter()
        .find(|node| node.data.operation.as_deref() == Some("show_message"))
        .expect("typed display call");
    assert!(
        call.data
            .expected_arg_types
            .iter()
            .any(|argument| { argument.slot == "arg[0].text" && argument.expected_type == "str" })
    );
    let loop_node = document
        .nodes
        .iter()
        .find(|node| node.data.subkind.as_deref() == Some("for"))
        .expect("non-list loop");
    assert!(
        loop_node
            .data
            .available_vars
            .iter()
            .any(|variable| { variable.name == "result" && variable.variable_type == "null" })
    );
    assert!(loop_node.data.diagnostics.iter().any(|diagnostic| {
        diagnostic.kind == "incompatible_iteration_target"
            && diagnostic.message.contains("expected a list")
    }));

    let canonical_source = document.source.clone();
    document.facet_schema_version = Some(999);
    for node in &mut document.nodes {
        for variable in &mut node.data.available_vars {
            variable.variable_type = "forged".to_string();
        }
        for argument in &mut node.data.expected_arg_types {
            argument.expected_type = "forged".to_string();
        }
        for diagnostic in &mut node.data.diagnostics {
            diagnostic.message = "forged".to_string();
        }
    }
    let response = client
        .post(format!("{base}/workflow"))
        .json(&document)
        .send()
        .await
        .expect("POST document with echoed facets");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let saved: SaveWorkflowResponse = response.json().await.expect("saved typed document");
    assert_eq!(saved.document.source, canonical_source);
    assert_eq!(
        saved.document.facet_schema_version,
        Some(lashlang::WORKFLOW_TYPE_FACET_SCHEMA_VERSION)
    );
    assert!(saved.document.nodes.iter().all(|node| {
        node.data
            .available_vars
            .iter()
            .all(|variable| variable.variable_type != "forged")
            && node
                .data
                .expected_arg_types
                .iter()
                .all(|argument| argument.expected_type != "forged")
            && node
                .data
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.message != "forged")
    }));

    server.abort();
}
