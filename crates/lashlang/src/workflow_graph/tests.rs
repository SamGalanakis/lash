use super::*;

const REPRESENTATIVE: &str = r#"type Input = { enabled: bool, values: list[num] }

@label(title: "Run child", description: "A labeled process")
process child(input: Input) signals { refresh: null } -> number {
  total = 0
  for value in input.values {
    @label(title: "Pause")
    sleep for 1
  }
  signal = wait_signal("refresh")
  finish total
}

items = [value * 2 for value in [1, 2, 3] if value > 1]
if true {
  @label(title: "Print items", description: "Label metadata survives")
  print items
} else {}
finish items
"#;

#[test]
fn canonical_get_put_and_put_get() {
    let canonical = canonical_program_source(&parse(REPRESENTATIVE).unwrap()).unwrap();
    let graph = workflow_graph_from_source(&canonical).unwrap();
    let rendered = workflow_graph_to_source(&graph).unwrap();
    assert_eq!(rendered, canonical);
    assert_eq!(workflow_graph_from_source(&rendered).unwrap(), graph);
    assert_eq!(parse(&rendered).unwrap(), parse(&canonical).unwrap());
}

#[test]
fn canonicalization_discards_comments_but_preserves_labels() {
    let graph = workflow_graph_from_source(
        "# comment\n@label(title: \"Named\", description: \"Kept\")\nvalue = 1\n",
    )
    .unwrap();
    let rendered = workflow_graph_to_source(&graph).unwrap();
    assert!(!rendered.contains("comment"));
    assert!(rendered.contains("@label(title: \"Named\", description: \"Kept\")"));
}

#[test]
fn invalid_graphs_are_refused() {
    let mut graph = workflow_graph_from_source("value = 1\nfinish value\n").unwrap();
    graph.main.edges.push(WorkflowEdge {
        id: "dangling".to_string(),
        from: graph.main.nodes[0].id.clone(),
        to: WorkflowNodeId("missing".to_string()),
        kind: WorkflowEdgeKind::Sequence,
    });
    assert!(matches!(
        workflow_graph_to_source(&graph),
        Err(GraphRenderError::UnknownNodeReference { .. })
    ));

    let mut graph = workflow_graph_from_source("if true { value = 1 }\n").unwrap();
    let WorkflowNodeKind::Container(WorkflowContainer::If { then_graph, .. }) =
        &mut graph.main.nodes[0].kind
    else {
        panic!("expected if container")
    };
    *then_graph = None;
    assert!(matches!(
        workflow_graph_to_source(&graph),
        Err(GraphRenderError::MissingRequiredChild {
            child: "then_graph",
            ..
        })
    ));
}

#[test]
fn while_and_path_assignment_are_opaque() {
    let source = "state = { count: 0 }\nstate.count = 1\nwhile state.count < 3 { state.count = state.count + 1 }\nfinish state\n";
    let graph = workflow_graph_from_source(source).unwrap();
    assert!(matches!(
        graph.main.nodes[1].kind,
        WorkflowNodeKind::Opaque { .. }
    ));
    assert!(matches!(
        graph.main.nodes[2].kind,
        WorkflowNodeKind::Opaque { .. }
    ));
    assert_eq!(
        graph.main.nodes[2].outputs,
        vec![VariableVersion {
            variable: "state".to_string(),
            version: 3,
        }]
    );
    assert!(graph.main.edges.iter().any(|edge| {
        edge.from == graph.main.nodes[2].id
            && edge.to == graph.main.nodes[3].id
            && matches!(
                edge.kind,
                WorkflowEdgeKind::DataDependency {
                    ref variable,
                    version: 3
                } if variable == "state"
            )
    }));
    let rendered = workflow_graph_to_source(&graph).unwrap();
    let canonical = canonical_program_source(&parse(source).unwrap()).unwrap();
    assert_eq!(parse(&rendered).unwrap(), parse(&canonical).unwrap());
}

#[test]
fn iteration_carried_reassignment_is_opaque() {
    let graph = workflow_graph_from_source(
        "total = 0\nfor value in [1, 2] { total = total + value }\nfinish total\n",
    )
    .unwrap();
    assert!(matches!(
        graph.main.nodes[1].kind,
        WorkflowNodeKind::Opaque { .. }
    ));
    assert_eq!(graph.main.nodes[1].outputs[0].version, 2);
    assert!(graph.main.edges.iter().any(|edge| {
        edge.from == graph.main.nodes[1].id
            && edge.to == graph.main.nodes[2].id
            && matches!(
                edge.kind,
                WorkflowEdgeKind::DataDependency { version: 2, .. }
            )
    }));
    assert!(workflow_graph_to_source(&graph).is_ok());
}

#[test]
fn structured_loop_control_round_trips_without_opaque_parse_context() {
    let source = "for value in [1, 2] {\n  if value == 1 { continue } else { break }\n}\n";
    let canonical = canonical_program_source(&parse(source).unwrap()).unwrap();
    let graph = workflow_graph_from_source(&canonical).unwrap();
    assert_eq!(workflow_graph_to_source(&graph).unwrap(), canonical);
}
