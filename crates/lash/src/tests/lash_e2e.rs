use super::*;
use std::collections::{BTreeSet, VecDeque};

#[derive(Default)]
struct ExpectedContracts {
    labeled_resource_titles: Vec<&'static str>,
    labeled_node_titles: Vec<&'static str>,
    completed_process_entries: Vec<&'static str>,
    min_completed_child_session_exec_graphs: usize,
    min_completed_process_graphs: usize,
}

struct LashE2eCase {
    name: &'static str,
    session_id: &'static str,
    scripted_provider_responses: Vec<&'static str>,
    root_prompt: &'static str,
    expected_submitted_value: Option<serde_json::Value>,
    tool_provider: Option<Arc<dyn ToolProvider>>,
    install_subagents: bool,
    max_turns: Option<usize>,
    expected_contracts: ExpectedContracts,
}

struct LashE2eRun {
    turn_output: Option<TurnResult>,
    streamed_events: Vec<TurnActivity>,
    graph_snapshots: Vec<crate::tracing::TraceLashlangGraph>,
    prompt_captures: Vec<LlmRequest>,
    final_process_list: Vec<lash_core::ProcessHandleSummary>,
}

#[derive(Debug)]
struct GraphContract {
    graphs: Vec<GraphFact>,
    child_links: Vec<ChildLinkFact>,
}

#[allow(dead_code)]
#[derive(Debug)]
struct GraphFact {
    graph_key: String,
    session_id: String,
    turn_id: Option<String>,
    subject_kind: String,
    subject_id: String,
    entry_kind: String,
    entry_name: String,
    status: crate::tracing::TraceLashlangStatus,
    nodes: Vec<NodeFact>,
}

#[allow(dead_code)]
#[derive(Debug)]
struct NodeFact {
    graph_key: String,
    kind: String,
    label: String,
    label_title: Option<String>,
    status: crate::tracing::TraceLashlangNodeStatus,
    has_error: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
struct ChildLinkFact {
    parent_graph_key: String,
    parent_entry_name: String,
    parent_node_kind: Option<String>,
    parent_node_label_title: Option<String>,
    child_graph_key: String,
    child_entry_name: Option<String>,
}

impl GraphContract {
    fn from_graphs(graphs: &[crate::tracing::TraceLashlangGraph]) -> Self {
        let mut facts = Vec::new();
        let mut links = Vec::new();
        for graph in graphs {
            let (subject_kind, subject_id) = match &graph.subject {
                crate::tracing::TraceRuntimeSubject::Effect { effect_id, kind } => {
                    (format!("effect:{kind}"), effect_id.clone())
                }
                crate::tracing::TraceRuntimeSubject::Process { process_id } => {
                    ("process".to_string(), process_id.clone())
                }
            };
            facts.push(GraphFact {
                graph_key: graph.graph_key.clone(),
                session_id: graph.scope.session_id.clone(),
                turn_id: graph.scope.turn_id.clone(),
                subject_kind,
                subject_id,
                entry_kind: graph.entry_kind.clone(),
                entry_name: graph.entry_name.clone(),
                status: graph.status,
                nodes: graph
                    .nodes
                    .iter()
                    .map(|node| NodeFact {
                        graph_key: graph.graph_key.clone(),
                        kind: node.kind.clone(),
                        label: node.label.clone(),
                        label_title: node
                            .label_metadata
                            .as_ref()
                            .map(|label| label.title.clone()),
                        status: node.status,
                        has_error: node.latest_error.is_some(),
                    })
                    .collect(),
            });
            for child in &graph.children {
                let parent = graph
                    .nodes
                    .iter()
                    .find(|node| node.id == child.parent_node_id);
                links.push(ChildLinkFact {
                    parent_graph_key: child.parent_graph_key.clone(),
                    parent_entry_name: graph.entry_name.clone(),
                    parent_node_kind: parent.map(|node| node.kind.clone()),
                    parent_node_label_title: parent
                        .and_then(|node| node.label_metadata.as_ref())
                        .map(|label| label.title.clone()),
                    child_graph_key: child.child_graph_key.clone(),
                    child_entry_name: child.child_entry_name.clone(),
                });
            }
        }
        Self {
            graphs: facts,
            child_links: links,
        }
    }

    fn graph_keys(&self) -> BTreeSet<&str> {
        self.graphs
            .iter()
            .map(|graph| graph.graph_key.as_str())
            .collect()
    }

    fn nodes(&self) -> impl Iterator<Item = &NodeFact> {
        self.graphs.iter().flat_map(|graph| graph.nodes.iter())
    }
}

#[test]
fn lash_e2e_foreground_labeled_tool_call() -> Result<()> {
    run_async_test_on_large_stack("lash-e2e-foreground-tool", || async {
        let case = LashE2eCase {
            name: "foreground labeled tool call",
            session_id: "lash-e2e-foreground-tool",
            scripted_provider_responses: vec![
                r#"```lashlang
@label(title: "Lookup app state")
value = await tools.app_lookup({})?
submit value
```"#,
            ],
            root_prompt: "Call the app lookup tool and submit its value.",
            expected_submitted_value: Some(serde_json::json!({ "ok": true })),
            tool_provider: Some(Arc::new(AppTools)),
            install_subagents: false,
            max_turns: None,
            expected_contracts: ExpectedContracts {
                labeled_resource_titles: vec!["Lookup app state"],
                ..ExpectedContracts::default()
            },
        };

        let run = run_turn_case(case).await?;
        assert_eq!(run.prompt_captures.len(), 1);
        Ok(())
    })
}

#[test]
fn lash_e2e_started_process_labeled_tool_call() -> Result<()> {
    run_async_test_on_large_stack("lash-e2e-started-process-tool", || async {
        run_turn_case(LashE2eCase {
            name: "started process labeled tool call",
            session_id: "lash-e2e-process-tool",
            scripted_provider_responses: vec![
                r#"```lashlang
process lookup(tools: Tools) {
  @label(title: "Lookup app state in process")
  value = await tools.app_lookup({})?
  finish value
}
handle = start lookup(tools: tools)
result = (await handle)?
submit result
```"#,
            ],
            root_prompt: "Start a process that calls the app lookup tool.",
            expected_submitted_value: Some(serde_json::json!({ "ok": true })),
            tool_provider: Some(Arc::new(AppTools)),
            install_subagents: false,
            max_turns: None,
            expected_contracts: ExpectedContracts {
                labeled_resource_titles: vec!["Lookup app state in process"],
                completed_process_entries: vec!["lookup"],
                min_completed_process_graphs: 1,
                ..ExpectedContracts::default()
            },
        })
        .await?;
        Ok(())
    })
}

#[test]
fn lash_e2e_started_process_labeled_subagent_spawn() -> Result<()> {
    run_async_test_on_large_stack("lash-e2e-started-process-subagent", || async {
        run_turn_case(LashE2eCase {
            name: "started process labeled subagent spawn",
            session_id: "lash-e2e-process-subagent",
            scripted_provider_responses: vec![
                r#"```lashlang
process spawn_child() {
  @label(title: "Spawn subagent with web search")
  result = await agents.spawn({
    capability: "default",
    task: "Submit `{ len: len(chunk) }` using the seeded `chunk` variable.",
    seed: { chunk: ["a", "b"] },
    output: Type { len: int }
  })?
  finish result
}
handle = start spawn_child()
result = (await handle)?
submit result
```"#,
                "```lashlang\nsubmit { len: len(chunk) }\n```",
            ],
            root_prompt: "Run a Lashlang process that spawns a subagent and returns its value.",
            expected_submitted_value: Some(serde_json::json!({ "len": 2 })),
            tool_provider: None,
            install_subagents: true,
            max_turns: None,
            expected_contracts: ExpectedContracts {
                labeled_resource_titles: vec!["Spawn subagent with web search"],
                completed_process_entries: vec!["spawn_child"],
                min_completed_child_session_exec_graphs: 1,
                min_completed_process_graphs: 1,
                ..ExpectedContracts::default()
            },
        })
        .await?;
        Ok(())
    })
}

#[test]
fn lash_e2e_nested_process_start_await() -> Result<()> {
    run_async_test_on_large_stack("lash-e2e-nested-process", || async {
        run_turn_case(LashE2eCase {
            name: "nested process start await",
            session_id: "lash-e2e-nested-process",
            scripted_provider_responses: vec![
                r#"```lashlang
process child() {
  finish { child: "done" }
}
process parent() {
  @label(title: "Start nested child process")
  handle = start child()
  result = (await handle)?
  finish { parent: result.child }
}
handle = start parent()
result = (await handle)?
submit result
```"#,
            ],
            root_prompt: "Start a parent process that starts and awaits a child process.",
            expected_submitted_value: Some(serde_json::json!({ "parent": "done" })),
            tool_provider: None,
            install_subagents: false,
            max_turns: None,
            expected_contracts: ExpectedContracts {
                labeled_node_titles: vec!["Start nested child process"],
                completed_process_entries: vec!["parent", "child"],
                min_completed_process_graphs: 2,
                ..ExpectedContracts::default()
            },
        })
        .await?;
        Ok(())
    })
}

#[test]
fn lash_e2e_session_turn_process_child() -> Result<()> {
    run_async_test_on_large_stack("lash-e2e-session-turn-process", || async {
        run_session_turn_process_case().await
    })
}

#[test]
fn lash_e2e_failed_child_preserves_failure_graph() -> Result<()> {
    run_async_test_on_large_stack("lash-e2e-failed-child", || async {
        let run = run_turn_case_without_success_assertions(LashE2eCase {
            name: "failed child preserves failure graph",
            session_id: "lash-e2e-failed-child",
            scripted_provider_responses: vec![
                r#"```lashlang
@label(title: "Spawn failing subagent")
result = await agents.spawn({
  capability: "default",
  task: "Fail with reason child boom.",
  seed: {},
  output: Type { reason: str }
})?
submit result
```"#,
                "```lashlang\nawait tools.submit_error({ reason: \"child boom\" })?\n```",
            ],
            root_prompt: "Spawn a child that fails and preserve its execution graph.",
            expected_submitted_value: None,
            tool_provider: None,
            install_subagents: true,
            max_turns: Some(1),
            expected_contracts: ExpectedContracts::default(),
        })
        .await?;

        assert_failed_code_block_present(&run.streamed_events);
        assert_no_forbidden_error_text(&run.streamed_events);
        assert_no_false_submitted_success(&run);
        assert_all_processes_terminal(&run.final_process_list);
        let contract = GraphContract::from_graphs(&run.graph_snapshots);
        assert_labeled_resource_operation(
            &contract,
            "Spawn failing subagent",
            crate::tracing::TraceLashlangNodeStatus::Failed,
        );
        assert_no_duplicate_label_step(&contract, "Spawn failing subagent");
        assert_graph_lineage_connected(&contract, &run.final_process_list);
        assert_subagent_bridge_exec_graphs(&run, crate::tracing::TraceLashlangStatus::Completed);
        Ok(())
    })
}

#[test]
fn lash_e2e_parallel_spawn_and_join() -> Result<()> {
    run_async_test_on_large_stack("lash-e2e-parallel-spawn-join", || async {
        run_turn_case(LashE2eCase {
            name: "parallel process spawn and join",
            session_id: "lash-e2e-parallel-processes",
            scripted_provider_responses: vec![
                r#"```lashlang
process child(value: str) {
  finish value
}
@label(title: "Start left process")
left = start child(value: "left")
@label(title: "Start right process")
right = start child(value: "right")
left_value = (await left)?
right_value = (await right)?
submit { joined: [left_value, right_value] }
```"#,
            ],
            root_prompt: "Start two processes, await both, and submit their joined result.",
            expected_submitted_value: Some(serde_json::json!({ "joined": ["left", "right"] })),
            tool_provider: None,
            install_subagents: false,
            max_turns: None,
            expected_contracts: ExpectedContracts {
                labeled_node_titles: vec!["Start left process", "Start right process"],
                completed_process_entries: vec!["child"],
                min_completed_process_graphs: 2,
                ..ExpectedContracts::default()
            },
        })
        .await?;
        Ok(())
    })
}

async fn run_turn_case(case: LashE2eCase) -> Result<LashE2eRun> {
    let run = run_turn_case_without_success_assertions(case).await?;
    assert_successful_turn_case(&run);
    Ok(run)
}

async fn run_turn_case_without_success_assertions(case: LashE2eCase) -> Result<LashE2eRun> {
    let graph_store = Arc::new(crate::tracing::TraceLashlangGraphStore::default());
    let process_registry = Arc::new(TestLocalProcessRegistry::default());
    let prompt_captures = Arc::new(StdMutex::new(Vec::new()));
    let provider = scripted_provider(
        case.scripted_provider_responses.clone(),
        Arc::clone(&prompt_captures),
    );
    let mut builder = LashCore::rlm()
        .provider(provider)
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::clone(&process_registry) as Arc<dyn ProcessRegistry>)
        .lashlang_execution_sink(Some(
            Arc::clone(&graph_store) as Arc<dyn crate::tracing::TraceSink>
        ));
    if let Some(tools) = case.tool_provider.clone() {
        builder = builder.tools(tools);
    }
    if case.install_subagents {
        builder = builder.plugin(subagents_plugin());
    }
    if let Some(max_turns) = case.max_turns {
        builder = builder.max_turns(max_turns);
    }
    let core = builder.build()?;
    let session = core.session(case.session_id).open().await?;
    let events = Arc::new(RecordingEvents::default());

    let turn_output = session
        .turn(TurnInput::text(case.root_prompt))
        .stream(events.as_ref())
        .await?;
    session.process_control().await_all().await?;
    let run = LashE2eRun {
        turn_output: Some(turn_output),
        streamed_events: events.snapshot().await,
        graph_snapshots: graph_store.graphs(),
        prompt_captures: prompt_captures.lock().expect("prompt captures").clone(),
        final_process_list: session.process_control().list_all().await?,
    };

    if let Some(expected) = &case.expected_submitted_value {
        let Some(output) = run.turn_output.as_ref() else {
            panic!("{} did not run a turn", case.name);
        };
        assert_eq!(
            output.submitted_value(),
            Some(expected),
            "{} submitted value mismatch",
            case.name
        );
    }

    let contract = GraphContract::from_graphs(&run.graph_snapshots);
    for title in case.expected_contracts.labeled_resource_titles {
        assert_labeled_resource_operation(
            &contract,
            title,
            crate::tracing::TraceLashlangNodeStatus::Completed,
        );
        assert_no_duplicate_label_step(&contract, title);
    }
    for title in case.expected_contracts.labeled_node_titles {
        assert_labeled_node(
            &contract,
            title,
            crate::tracing::TraceLashlangNodeStatus::Completed,
        );
        assert_no_duplicate_label_step(&contract, title);
    }
    for entry_name in case.expected_contracts.completed_process_entries {
        assert_completed_process_graph(&contract, entry_name);
    }
    assert_min_completed_process_graphs(
        &contract,
        case.expected_contracts.min_completed_process_graphs,
    );
    assert_min_completed_child_session_exec_graphs(
        &run,
        case.session_id,
        case.expected_contracts
            .min_completed_child_session_exec_graphs,
    );
    Ok(run)
}

async fn run_session_turn_process_case() -> Result<()> {
    let session_id = "lash-e2e-session-turn-root";
    let child_session_id = "lash-e2e-session-turn-child";
    let process_id = "lash-e2e-session-turn-process";
    let graph_store = Arc::new(crate::tracing::TraceLashlangGraphStore::default());
    let process_registry = Arc::new(TestLocalProcessRegistry::default());
    let prompt_captures = Arc::new(StdMutex::new(Vec::new()));
    let provider = scripted_provider(
        vec!["```lashlang\nsubmit { child: \"done\", scoped: true }\n```"],
        Arc::clone(&prompt_captures),
    );
    let core = LashCore::rlm()
        .provider(provider)
        .model(mock_model_spec())
        .plugin(subagents_plugin())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::clone(&process_registry) as Arc<dyn ProcessRegistry>)
        .lashlang_execution_sink(Some(
            Arc::clone(&graph_store) as Arc<dyn crate::tracing::TraceSink>
        ))
        .build()?;
    let session = core.session(session_id).open().await?;
    let child_policy = lash_core::SessionPolicy {
        model: mock_model_spec(),
        max_turns: Some(2),
        ..lash_core::SessionPolicy::default()
    };
    let create_request = lash_core::SessionCreateRequest::child(
        session_id,
        lash_core::SessionStartPoint::Empty,
        child_policy,
        lash_core::PluginOptions::default(),
        "e2e-session-turn",
    )
    .with_session_id(child_session_id);

    let handle = session
        .process_control()
        .start(lash_core::ProcessStartRequest::new(
            process_id,
            lash_core::ProcessInput::SessionTurn {
                create_request: Box::new(create_request),
                turn_input: Box::new(TurnInput::text("run child session turn")),
                output_contract: lash_core::ToolOutputContract::Static,
            },
            lash_core::ProcessHandleDescriptor::new(Some("session_turn"), Some("child turn")),
        ))
        .await?;
    assert_eq!(handle.process_id, process_id);
    session.process_control().await_all().await?;

    let await_output = process_registry.await_process(process_id).await?;
    let lash_core::ProcessAwaitOutput::Success { value, .. } = await_output else {
        panic!("session-turn process did not succeed");
    };
    assert_eq!(
        value.get("child_session_id"),
        Some(&serde_json::json!(child_session_id))
    );
    let turn: lash_core::AssembledTurn = value
        .get("turn")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .expect("session-turn output should decode")
        .expect("session-turn output should contain a turn");
    assert_eq!(
        turn.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue {
            value: serde_json::json!({ "child": "done", "scoped": true })
        })
    );

    let run = LashE2eRun {
        turn_output: None,
        streamed_events: Vec::new(),
        graph_snapshots: graph_store.graphs(),
        prompt_captures: prompt_captures.lock().expect("prompt captures").clone(),
        final_process_list: session.process_control().list_all().await?,
    };
    assert_eq!(run.prompt_captures.len(), 1);
    assert_all_processes_terminal(&run.final_process_list);
    assert_session_turn_child_graph(&run, child_session_id, process_id);
    Ok(())
}

fn scripted_provider(
    responses: Vec<&'static str>,
    prompt_captures: Arc<StdMutex<Vec<LlmRequest>>>,
) -> ProviderHandle {
    let responses = Arc::new(TokioMutex::new(VecDeque::from(
        responses
            .into_iter()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>(),
    )));
    crate::testing::TestProvider::builder()
        .kind("lash-e2e")
        .complete(move |request| {
            let responses = Arc::clone(&responses);
            let prompt_captures = Arc::clone(&prompt_captures);
            async move {
                prompt_captures
                    .lock()
                    .expect("prompt captures")
                    .push(request.clone());
                let text = responses
                    .lock()
                    .await
                    .pop_front()
                    .unwrap_or_else(|| panic!("no scripted e2e provider response left"));
                Ok(LlmResponse {
                    full_text: text.clone(),
                    parts: vec![LlmOutputPart::Text {
                        text,
                        response_meta: None,
                    }],
                    ..LlmResponse::default()
                })
            }
        })
        .build()
        .into_handle()
}

fn subagents_plugin() -> Arc<dyn PluginFactory> {
    Arc::new(lash_subagents::SubagentsPluginFactory::new(Arc::new(
        lash_subagents::CapabilityRegistry::new().with(Arc::new(
            lash_subagents::StaticCapability::new("default", SessionSpec::inherit()),
        )),
    )))
}

fn assert_successful_turn_case(run: &LashE2eRun) {
    assert_no_unexpected_turn_errors(&run.streamed_events);
    assert_successful_lash_code_path(&run.streamed_events);
    assert_all_processes_terminal(&run.final_process_list);
    let output = run.turn_output.as_ref().expect("turn output");
    assert!(
        output.is_success(),
        "turn should have succeeded: {:?}",
        output.outcome
    );
    let contract = GraphContract::from_graphs(&run.graph_snapshots);
    assert_foreground_exec_graph_completed(run);
    assert_graph_lineage_connected(&contract, &run.final_process_list);
    assert_subagent_bridge_exec_graphs(run, crate::tracing::TraceLashlangStatus::Completed);
}

fn assert_no_unexpected_turn_errors(events: &[TurnActivity]) {
    assert_no_forbidden_error_text(events);
    assert!(
        !events.iter().any(|activity| matches!(
            &activity.event,
            TurnEvent::Error { .. } | TurnEvent::CodeBlockCompleted { success: false, .. }
        )),
        "unexpected failed turn event: {events:#?}"
    );
}

fn assert_no_forbidden_error_text(events: &[TurnActivity]) {
    let forbidden = [
        "Invalid process handle",
        "missing __handle__",
        "deployment effect-host fallback",
        "missing scoped controller",
    ];
    for activity in events {
        let text = format!("{:?}", activity.event);
        for needle in forbidden {
            assert!(
                !text.contains(needle),
                "unexpected error text `{needle}` in event: {activity:#?}"
            );
        }
    }
}

fn assert_successful_lash_code_path(events: &[TurnActivity]) {
    let code_started = events
        .iter()
        .position(|activity| {
            matches!(
                &activity.event,
                TurnEvent::CodeBlockStarted { language, .. } if language == "lashlang"
            )
        })
        .unwrap_or_else(|| panic!("missing Lashlang code start event: {events:#?}"));
    let code_completed = events
        .iter()
        .rposition(|activity| {
            matches!(
                &activity.event,
                TurnEvent::CodeBlockCompleted { language, success: true, .. } if language == "lashlang"
            )
        })
        .unwrap_or_else(|| panic!("missing successful Lashlang code completion: {events:#?}"));
    let terminal_output = events
        .iter()
        .position(|activity| {
            matches!(
                &activity.event,
                TurnEvent::SubmittedValue { .. } | TurnEvent::ToolValue { .. }
            )
        })
        .unwrap_or_else(|| panic!("missing terminal output event: {events:#?}"));
    assert!(code_started < code_completed);
    assert!(code_completed < terminal_output);
    assert!(
        !events[code_completed + 1..].iter().any(|activity| {
            matches!(
                &activity.event,
                TurnEvent::ToolCallStarted { .. } | TurnEvent::ToolCallCompleted { .. }
            )
        }),
        "tool events should not be emitted after code completion: {events:#?}"
    );
}

fn assert_failed_code_block_present(events: &[TurnActivity]) {
    assert!(
        events.iter().any(|activity| {
            matches!(
                &activity.event,
                TurnEvent::CodeBlockCompleted {
                    success: false,
                    error: Some(_),
                    ..
                }
            )
        }),
        "missing failed code block completion: {events:#?}"
    );
}

fn assert_no_false_submitted_success(run: &LashE2eRun) {
    let output = run.turn_output.as_ref().expect("turn output");
    assert!(
        output.submitted_value().is_none(),
        "failure scenario produced a submitted value: {:?}",
        output.submitted_value()
    );
    assert!(
        !run.streamed_events
            .iter()
            .any(|activity| matches!(&activity.event, TurnEvent::SubmittedValue { .. })),
        "failure scenario emitted submitted success: {:#?}",
        run.streamed_events
    );
}

fn assert_all_processes_terminal(processes: &[lash_core::ProcessHandleSummary]) {
    assert!(
        processes.iter().all(|process| process.status.is_terminal()),
        "expected all visible process handles terminal: {processes:#?}"
    );
}

fn assert_foreground_exec_graph_completed(run: &LashE2eRun) {
    let output = run.turn_output.as_ref().expect("turn output");
    let session_id = &output.state.session_id;
    let graph = run
        .graph_snapshots
        .iter()
        .find(|graph| {
            graph.scope.session_id == *session_id
                && matches!(
                    &graph.subject,
                    crate::tracing::TraceRuntimeSubject::Effect { kind, .. } if kind == "exec_code"
                )
        })
        .unwrap_or_else(|| {
            panic!(
                "missing foreground exec graph for {session_id}: {:#?}",
                GraphContract::from_graphs(&run.graph_snapshots)
            )
        });
    assert_eq!(
        graph.status,
        crate::tracing::TraceLashlangStatus::Completed,
        "foreground exec graph did not complete: {graph:#?}"
    );
}

fn assert_graph_lineage_connected(
    contract: &GraphContract,
    processes: &[lash_core::ProcessHandleSummary],
) {
    let graph_keys = contract.graph_keys();
    let process_ids = processes
        .iter()
        .map(|process| process.process_id.as_str())
        .collect::<BTreeSet<_>>();
    for link in &contract.child_links {
        let linked_graph_exists = graph_keys.contains(link.child_graph_key.as_str());
        let linked_process_exists = link
            .child_graph_key
            .strip_prefix("process:")
            .is_some_and(|process_id| process_ids.contains(process_id));
        assert!(
            linked_graph_exists || linked_process_exists,
            "child link points nowhere: {link:#?}\ncontract={contract:#?}\nprocesses={processes:#?}"
        );
    }
}

fn assert_labeled_resource_operation(
    contract: &GraphContract,
    title: &str,
    expected_status: crate::tracing::TraceLashlangNodeStatus,
) {
    let node = contract
        .nodes()
        .find(|node| {
            node.kind == "resource_operation"
                && node.label_title.as_deref() == Some(title)
                && node.status == expected_status
        })
        .unwrap_or_else(|| {
            panic!(
                "missing labeled resource operation `{title}` with status {expected_status:?}: {contract:#?}"
            );
        });
    assert_eq!(
        node.status, expected_status,
        "labeled resource operation `{title}` had wrong status: {node:#?}"
    );
    if expected_status == crate::tracing::TraceLashlangNodeStatus::Failed {
        assert!(
            node.has_error,
            "failed labeled resource operation should retain node error: {node:#?}"
        );
    }
}

fn assert_labeled_node(
    contract: &GraphContract,
    title: &str,
    expected_status: crate::tracing::TraceLashlangNodeStatus,
) {
    let node = contract
        .nodes()
        .find(|node| node.label_title.as_deref() == Some(title) && node.status == expected_status)
        .unwrap_or_else(|| {
            panic!("missing labeled node `{title}` with status {expected_status:?}: {contract:#?}")
        });
    assert_eq!(
        node.status, expected_status,
        "labeled node `{title}` had wrong status: {node:#?}"
    );
}

fn assert_no_duplicate_label_step(contract: &GraphContract, title: &str) {
    assert!(
        !contract
            .nodes()
            .any(|node| node.kind == "step" && node.label == title),
        "label `{title}` produced a duplicate standalone step: {contract:#?}"
    );
}

fn assert_completed_process_graph(contract: &GraphContract, entry_name: &str) {
    assert!(
        contract.graphs.iter().any(|graph| {
            graph.entry_kind == "process"
                && graph.entry_name == entry_name
                && graph.subject_kind == "process"
                && graph.status == crate::tracing::TraceLashlangStatus::Completed
        }),
        "missing completed process graph `{entry_name}`: {contract:#?}"
    );
}

fn assert_min_completed_process_graphs(contract: &GraphContract, expected_min: usize) {
    if expected_min == 0 {
        return;
    }
    let count = contract
        .graphs
        .iter()
        .filter(|graph| {
            graph.entry_kind == "process"
                && graph.subject_kind == "process"
                && graph.status == crate::tracing::TraceLashlangStatus::Completed
        })
        .count();
    assert!(
        count >= expected_min,
        "expected at least {expected_min} completed process graphs, got {count}: {contract:#?}"
    );
}

fn assert_min_completed_child_session_exec_graphs(
    run: &LashE2eRun,
    root_session_id: &str,
    expected_min: usize,
) {
    if expected_min == 0 {
        return;
    }
    let count = run
        .graph_snapshots
        .iter()
        .filter(|graph| {
            graph.scope.session_id != root_session_id
                && matches!(
                    &graph.subject,
                    crate::tracing::TraceRuntimeSubject::Effect { kind, .. } if kind == "exec_code"
                )
                && graph.status == crate::tracing::TraceLashlangStatus::Completed
        })
        .count();
    assert!(
        count >= expected_min,
        "expected at least {expected_min} child-session exec graphs, got {count}: {:#?}",
        GraphContract::from_graphs(&run.graph_snapshots)
    );
}

fn assert_subagent_bridge_exec_graphs(
    run: &LashE2eRun,
    expected_status: crate::tracing::TraceLashlangStatus,
) {
    let subagent_process_ids = run
        .final_process_list
        .iter()
        .filter(|process| {
            process.descriptor.kind.as_deref() == Some("subagent")
                || process.process_id.starts_with("process:subagent:")
        })
        .map(|process| process.process_id.as_str())
        .collect::<Vec<_>>();
    if subagent_process_ids.is_empty() {
        return;
    }
    for process_id in subagent_process_ids {
        assert!(
            run.graph_snapshots.iter().any(|graph| {
                graph.scope.turn_id.as_deref() == Some(process_id)
                    && matches!(
                        &graph.subject,
                        crate::tracing::TraceRuntimeSubject::Effect { kind, .. } if kind == "exec_code"
                    )
                    && graph.status == expected_status
            }),
            "missing {expected_status:?} child-session exec graph for subagent process {process_id}: {:#?}",
            GraphContract::from_graphs(&run.graph_snapshots)
        );
    }
}

fn assert_session_turn_child_graph(run: &LashE2eRun, child_session_id: &str, process_id: &str) {
    let graph = run
        .graph_snapshots
        .iter()
        .find(|graph| {
            graph.scope.session_id == child_session_id
                && graph.scope.turn_id.as_deref() == Some(process_id)
                && matches!(
                    &graph.subject,
                    crate::tracing::TraceRuntimeSubject::Effect { kind, .. } if kind == "exec_code"
                )
        })
        .unwrap_or_else(|| {
            panic!(
                "missing scoped session-turn child exec graph: {:#?}",
                GraphContract::from_graphs(&run.graph_snapshots)
            )
        });
    assert_eq!(graph.status, crate::tracing::TraceLashlangStatus::Completed);
}
