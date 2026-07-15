use lashlang::{
    canonical_program_source, parse, workflow_graph_from_source, workflow_graph_to_source,
};

#[test]
fn source_graph_source_round_trip() {
    let source = r#"
@label(title: "Fetch account", description: "Labels name graph nodes")
account = await crm.get_account({ id: "acct-1" })?
finish account
"#;
    let canonical = canonical_program_source(&parse(source).expect("example source parses"))
        .expect("example source has a canonical form");
    let graph = workflow_graph_from_source(&canonical).expect("source projects to a graph");
    let rendered = workflow_graph_to_source(&graph).expect("graph renders to source");

    assert_eq!(rendered, canonical);
    assert_eq!(parse(&rendered).unwrap(), parse(&canonical).unwrap());
    assert_eq!(workflow_graph_from_source(&rendered).unwrap(), graph);
}
