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
fn expression_if_and_direct_else_if_obey_all_lens_laws() {
    let source = r#"choice = (true ? 1 : (false ? 2 : 3))
if choice == 1 {
  print "one"
} else if choice == 2 {
  print "two"
} else {
  print "other"
}
finish choice
"#;
    let canonical = canonical_program_source(&parse(source).unwrap()).unwrap();
    let graph = workflow_graph_from_source(&canonical).unwrap();

    let WorkflowNodeKind::Container(WorkflowContainer::If {
        then_is_block,
        else_is_block,
        ..
    }) = &graph.main.nodes[0].kind
    else {
        panic!("expected expression-if container")
    };
    assert!(!then_is_block);
    assert!(!else_is_block);

    let WorkflowNodeKind::Container(WorkflowContainer::If {
        then_is_block,
        else_is_block,
        else_graph,
        ..
    }) = &graph.main.nodes[1].kind
    else {
        panic!("expected statement-if container")
    };
    assert!(*then_is_block);
    assert!(!else_is_block);
    assert!(matches!(
        else_graph.as_deref().unwrap().nodes.as_slice(),
        [WorkflowNode {
            kind: WorkflowNodeKind::Container(WorkflowContainer::If {
                then_is_block: true,
                ..
            }),
            ..
        }]
    ));

    let rendered = workflow_graph_to_source(&graph).unwrap();
    assert_eq!(rendered, canonical, "GetPut and canonical fixpoint");
    assert_eq!(parse(&rendered).unwrap(), parse(&canonical).unwrap());
    assert_eq!(
        workflow_graph_from_source(&rendered).unwrap(),
        graph,
        "PutGet"
    );
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
fn while_and_path_assignment_are_structured_and_typed() {
    let source = "state = { count: 0 }\nstate.count = 1\nwhile state.count < 3 { state.count = state.count + 1 }\nfinish state\n";
    let graph = workflow_graph_from_source(source).unwrap();
    assert!(matches!(
        graph.main.nodes[1].kind,
        WorkflowNodeKind::StateUpdate { .. }
    ));
    let WorkflowNodeKind::Container(WorkflowContainer::While { body, .. }) =
        &graph.main.nodes[2].kind
    else {
        panic!("expected while container")
    };
    assert!(matches!(
        body.as_deref().unwrap().nodes[0].kind,
        WorkflowNodeKind::StateUpdate { .. }
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
    assert_eq!(rendered, canonical);
    assert_eq!(workflow_graph_from_source(&rendered).unwrap(), graph);
}

#[test]
fn iteration_carried_reassignment_is_structured_state_update() {
    let graph = workflow_graph_from_source(
        "total = 0\nfor value in [1, 2] { total = total + value }\nfinish total\n",
    )
    .unwrap();
    let WorkflowNodeKind::Container(WorkflowContainer::For { body, .. }) =
        &graph.main.nodes[1].kind
    else {
        panic!("expected for container")
    };
    assert!(matches!(
        body.as_deref().unwrap().nodes[0].kind,
        WorkflowNodeKind::StateUpdate { .. }
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
fn loops_publish_path_and_loop_introduced_writes_once() {
    let source = r#"state = { count: 0 }
for item in [1, 2] {
  state.count = item
  introduced = item
  item = item + 1
}
finish [state, introduced]
"#;
    let graph = workflow_graph_from_source(source).unwrap();
    let loop_node = &graph.main.nodes[1];
    assert_eq!(
        loop_node.outputs,
        vec![
            VariableVersion {
                variable: "introduced".to_string(),
                version: 1,
            },
            VariableVersion {
                variable: "state".to_string(),
                version: 2,
            },
        ]
    );
    assert!(
        !loop_node
            .outputs
            .iter()
            .any(|output| output.variable == "item")
    );
    let WorkflowNodeKind::Container(WorkflowContainer::For { body, .. }) = &loop_node.kind else {
        panic!("expected for container")
    };
    assert!(matches!(
        body.as_deref().unwrap().nodes[0].kind,
        WorkflowNodeKind::StateUpdate { .. }
    ));
    assert!(matches!(
        body.as_deref().unwrap().nodes[2].kind,
        WorkflowNodeKind::StateUpdate { .. }
    ));
    assert_lens_laws(source);
}

#[test]
fn scoped_loop_bindings_do_not_depend_on_or_replace_outer_versions() {
    let source = r#"item = 99
items = [1, 2]
for item in items {
  print item
}
selected = [item for item in items if item > 0]
finish item
"#;
    let graph = workflow_graph_from_source(source).unwrap();
    let outer_item = &graph.main.nodes[0];
    let WorkflowNodeKind::Container(WorkflowContainer::For { body, .. }) =
        &graph.main.nodes[2].kind
    else {
        panic!("expected for container")
    };
    let body_node = &body.as_deref().unwrap().nodes[0];
    assert!(!body.as_deref().unwrap().edges.iter().any(|edge| {
        edge.from == outer_item.id
            && edge.to == body_node.id
            && matches!(
                edge.kind,
                WorkflowEdgeKind::DataDependency { ref variable, .. } if variable == "item"
            )
    }));
    assert!(!graph.main.edges.iter().any(|edge| {
        edge.from == outer_item.id
            && (edge.to == graph.main.nodes[2].id || edge.to == graph.main.nodes[3].id)
            && matches!(
                edge.kind,
                WorkflowEdgeKind::DataDependency { ref variable, .. } if variable == "item"
            )
    }));
    assert!(graph.main.edges.iter().any(|edge| {
        edge.from == outer_item.id
            && edge.to == graph.main.nodes[4].id
            && matches!(
                edge.kind,
                WorkflowEdgeKind::DataDependency { ref variable, version: 1 }
                    if variable == "item"
            )
    }));
    assert_lens_laws(source);
}

#[test]
fn reference_type_literals_and_effectful_composites_are_typed() {
    let source = r#"type Item = { name: str }

process child() {
  finish 1
}

schema = Type { item: Item }
runs = [start child(), start child()]
tupled = (await runs[0], await runs[1])
listed = [await runs[0], await runs[1]]
recorded = { value: await runs[0] }
built = len(await runs)
binary = (await runs[0] + 1)
unary = not await runs[0]
field = (await runs[0]).value
indexed = (await runs)[0]
unusual = (await runs[0])??
finish unusual
"#;
    let graph = workflow_graph_from_source(source).unwrap();
    assert!(matches!(
        graph.main.nodes[0].kind,
        WorkflowNodeKind::Data {
            expression: Expr::TypeLiteral(_),
            ..
        }
    ));
    assert!(
        graph.main.nodes[1..=10]
            .iter()
            .all(|node| matches!(node.kind, WorkflowNodeKind::Computation { .. }))
    );
    assert!(
        !graph
            .nodes()
            .any(|node| matches!(node.kind, WorkflowNodeKind::Opaque { .. }))
    );
    assert_lens_laws(source);
}

#[test]
fn while_collects_condition_sites_without_duplicating_body_sites() {
    let source = r#"@label(title: "Loop guard")
while await tools.ready({})? {
  await tools.tick({})?
}
"#;
    let graph = workflow_graph_from_source(source).unwrap();
    let node = &graph.main.nodes[0];
    let WorkflowNodeKind::Container(WorkflowContainer::While { body, .. }) = &node.kind else {
        panic!("expected while container")
    };
    assert_eq!(
        node.execution_sites
            .iter()
            .map(|site| site.label.as_str())
            .collect::<Vec<_>>(),
        vec!["Loop guard", "ready"]
    );
    assert_eq!(
        body.as_deref().unwrap().nodes[0].execution_sites[0].label,
        "tick"
    );
    assert_lens_laws(source);
}

fn assert_lens_laws(source: &str) {
    let canonical = canonical_program_source(&parse(source).unwrap()).unwrap();
    let graph = workflow_graph_from_source(&canonical).unwrap();
    let rendered = workflow_graph_to_source(&graph).unwrap();
    assert_eq!(rendered, canonical, "GetPut");
    assert_eq!(parse(&rendered).unwrap(), parse(&canonical).unwrap());
    assert_eq!(
        workflow_graph_from_source(&rendered).unwrap(),
        graph,
        "PutGet"
    );
}

#[test]
fn structured_loop_control_round_trips_without_opaque_parse_context() {
    let source = "for value in [1, 2] {\n  if value == 1 { continue } else { break }\n}\n";
    let canonical = canonical_program_source(&parse(source).unwrap()).unwrap();
    let graph = workflow_graph_from_source(&canonical).unwrap();
    assert_eq!(workflow_graph_to_source(&graph).unwrap(), canonical);
}
