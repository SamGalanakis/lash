use super::super::*;
use super::contracts::{
    GraphContract, assert_all_processes_terminal, assert_completed_process_graph,
    assert_labeled_node, assert_labeled_resource_operation,
    assert_min_completed_child_session_exec_graphs, assert_min_completed_process_graphs,
    assert_no_duplicate_label_step, assert_session_turn_child_graph, assert_successful_turn_case,
};
use std::collections::VecDeque;

#[derive(Default)]
pub(super) struct ExpectedContracts {
    pub(super) labeled_resource_titles: Vec<&'static str>,
    pub(super) labeled_node_titles: Vec<&'static str>,
    pub(super) completed_process_entries: Vec<&'static str>,
    pub(super) min_completed_child_session_exec_graphs: usize,
    pub(super) min_completed_process_graphs: usize,
}

pub(super) struct LashE2eCase {
    pub(super) name: &'static str,
    pub(super) session_id: &'static str,
    pub(super) scripted_provider_responses: Vec<&'static str>,
    pub(super) root_prompt: &'static str,
    pub(super) expected_submitted_value: Option<serde_json::Value>,
    pub(super) tool_provider: Option<Arc<dyn ToolProvider>>,
    pub(super) install_subagents: bool,
    pub(super) max_turns: Option<usize>,
    pub(super) expected_contracts: ExpectedContracts,
}

pub(super) struct LashE2eRun {
    pub(super) turn_output: Option<TurnResult>,
    pub(super) streamed_events: Vec<TurnActivity>,
    pub(super) graph_snapshots: Vec<crate::tracing::TraceLashlangGraph>,
    pub(super) prompt_captures: Vec<LlmRequest>,
    pub(super) final_process_list: Vec<lash_core::ProcessHandleSummary>,
}

pub(super) async fn run_turn_case(case: LashE2eCase) -> Result<LashE2eRun> {
    let run = run_turn_case_without_success_assertions(case).await?;
    assert_successful_turn_case(&run);
    Ok(run)
}

pub(super) async fn run_turn_case_without_success_assertions(
    case: LashE2eCase,
) -> Result<LashE2eRun> {
    let graph_store = Arc::new(crate::tracing::TraceLashlangGraphStore::default());
    let process_registry = Arc::new(TestLocalProcessRegistry::default());
    let prompt_captures = Arc::new(StdMutex::new(Vec::new()));
    let provider = scripted_provider(
        case.scripted_provider_responses.clone(),
        Arc::clone(&prompt_captures),
    );
    let mut builder = explicit_ephemeral_facets(LashCore::rlm())
        .provider(provider)
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::clone(&process_registry) as Arc<dyn ProcessRegistry>)
        .lashlang_execution_sink(Arc::clone(&graph_store) as Arc<dyn crate::tracing::TraceSink>);
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
        .stream_to(events.as_ref())
        .await?;
    session.processes().await_all().await?;
    let prompt_captures_snapshot = prompt_captures.lock().expect("prompt captures").clone();
    let final_process_list = all_host_process_summaries(&core).await?;
    assert_remote_process_dto_surface(&core, process_registry.as_ref(), case.session_id).await;
    assert_remote_process_summaries_round_trip(&final_process_list);
    let run = LashE2eRun {
        turn_output: Some(turn_output),
        streamed_events: events.snapshot().await,
        graph_snapshots: graph_store.graphs(),
        prompt_captures: prompt_captures_snapshot,
        final_process_list,
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

async fn all_host_process_summaries(
    core: &LashCore,
) -> Result<Vec<lash_core::ProcessHandleSummary>> {
    let processes = core
        .processes()
        .list(&lash_core::ProcessListFilter {
            definition: None,
            status: lash_core::ProcessStatusFilter::Any,
            waiting: None,
        })
        .await?;
    Ok(processes
        .into_iter()
        .map(observed_process_summary)
        .collect())
}

fn observed_process_summary(
    process: lash_core::ObservedProcess,
) -> lash_core::ProcessHandleSummary {
    lash_core::ProcessHandleSummary::new(
        process.process_id,
        lash_core::ProcessHandleDescriptor::new(Some(process.kind), Some(process.label)),
        process.lifecycle,
    )
    .with_definition(process.input.definition())
}

async fn assert_remote_process_dto_surface(
    core: &LashCore,
    registry: &dyn lash_core::ProcessRegistry,
    session_id: &str,
) {
    let filter = lash_core::ProcessListFilter {
        definition: None,
        status: lash_core::ProcessStatusFilter::Any,
        waiting: None,
    };

    let observed = core
        .processes()
        .list(&filter)
        .await
        .expect("list observed processes for remote DTO round trip");
    let remote_list = lash_remote_protocol::RemoteProcessListResponse::try_from(observed.clone())
        .expect("observed process list should convert to remote DTO");
    remote_list
        .validate()
        .expect("remote observed process list should validate");
    let round_trip_observed: Vec<lash_core::ObservedProcess> = remote_list
        .try_into()
        .expect("remote observed process list should convert back");
    let observed_ids = observed
        .iter()
        .map(|process| process.process_id.as_str())
        .collect::<Vec<_>>();
    let round_trip_ids = round_trip_observed
        .iter()
        .map(|process| process.process_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(round_trip_ids, observed_ids);

    let snapshot = core
        .processes()
        .session_snapshot(session_id)
        .await
        .expect("capture process work snapshot for remote DTO round trip");
    let remote_snapshot = lash_remote_protocol::RemoteProcessWorkSnapshot::try_from(snapshot)
        .expect("process work snapshot should convert to remote DTO");
    remote_snapshot
        .validate()
        .expect("remote process work snapshot should validate");
    let round_trip_snapshot: lash_core::ProcessWorkSnapshot = remote_snapshot
        .try_into()
        .expect("remote process work snapshot should convert back");
    assert_eq!(round_trip_snapshot.session_id, session_id);

    let records = registry
        .list_processes(&filter)
        .await
        .expect("list process records for remote DTO round trip");
    for record in records {
        let process_id = record.id.clone();
        let remote_record = lash_remote_protocol::RemoteProcessRecord::try_from(record)
            .expect("process record should convert to remote DTO");
        remote_record
            .validate("LashE2eProcessRecord")
            .expect("remote process record should validate");
        let round_trip_record: lash_core::ProcessRecord = remote_record
            .try_into()
            .expect("remote process record should convert back");
        assert_eq!(round_trip_record.id, process_id);

        let events = registry
            .recent_events(&process_id, 32)
            .await
            .expect("load process event tail for remote DTO round trip");
        let expected_tail = events
            .iter()
            .map(|event| (event.sequence, event.event_type.clone()))
            .collect::<Vec<_>>();
        let remote_events =
            lash_remote_protocol::RemoteProcessEventsResponse::from((process_id.clone(), events));
        remote_events
            .validate()
            .expect("remote process event tail should validate");
        let (round_trip_process_id, round_trip_events): (String, Vec<lash_core::ProcessEvent>) =
            remote_events
                .try_into()
                .expect("remote process event tail should convert back");
        let round_trip_tail = round_trip_events
            .iter()
            .map(|event| (event.sequence, event.event_type.clone()))
            .collect::<Vec<_>>();
        assert_eq!(round_trip_process_id, process_id);
        assert_eq!(round_trip_tail, expected_tail);
    }
}

fn assert_remote_process_summaries_round_trip(summaries: &[lash_core::ProcessHandleSummary]) {
    for summary in summaries {
        let remote = lash_remote_protocol::RemoteProcessSummary::from(summary.clone());
        remote
            .validate("LashE2eProcessSummary")
            .expect("remote process summary should validate");
        let round_trip =
            lash_core::ProcessHandleSummary::try_from(remote).expect("remote summary round trip");
        assert_eq!(&round_trip, summary);
    }
}

pub(super) async fn run_session_turn_process_case() -> Result<()> {
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
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(provider)
        .model(mock_model_spec())
        .plugin(subagents_plugin())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::clone(&process_registry) as Arc<dyn ProcessRegistry>)
        .lashlang_execution_sink(Arc::clone(&graph_store) as Arc<dyn crate::tracing::TraceSink>)
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
        .processes()
        .start(
            lash_core::ProcessStartRequest::new(
                process_id,
                lash_core::ProcessInput::SessionTurn {
                    create_request: Box::new(create_request),
                    turn_input: Box::new(TurnInput::text("run child session turn")),
                    output_contract: lash_core::ToolOutputContract::Static,
                },
                lash_core::ProcessOriginator::host(),
            )
            .with_grant(Some(lash_core::ProcessStartGrant {
                session_scope: lash_core::SessionScope::new("request-descriptor"),
                descriptor: lash_core::ProcessHandleDescriptor::new(
                    Some("session_turn"),
                    Some("child turn"),
                ),
            })),
            inline_scope(lash_core::ExecutionScope::process(process_id)),
        )
        .await?;
    assert_eq!(handle.process_id, process_id);
    session.processes().await_all().await?;

    let await_output = process_registry.await_process(process_id).await?;
    let lash_core::ProcessAwaitOutput::Success { value, .. } = await_output else {
        panic!("session-turn process did not succeed: {await_output:#?}");
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

    let prompt_captures_snapshot = prompt_captures.lock().expect("prompt captures").clone();
    let final_process_list = all_host_process_summaries(&core).await?;
    assert_remote_process_dto_surface(&core, process_registry.as_ref(), session_id).await;
    assert_remote_process_summaries_round_trip(&final_process_list);
    let run = LashE2eRun {
        turn_output: None,
        streamed_events: Vec::new(),
        graph_snapshots: graph_store.graphs(),
        prompt_captures: prompt_captures_snapshot,
        final_process_list,
    };
    assert_eq!(run.prompt_captures.len(), 1);
    assert_all_processes_terminal(&run.final_process_list);
    assert_session_turn_child_graph(&run, child_session_id, process_id);
    Ok(())
}

pub(super) async fn run_durable_input_request_case() -> Result<()> {
    let session_id = "lash-e2e-durable-input-request";
    let graph_store = Arc::new(crate::tracing::TraceLashlangGraphStore::default());
    let process_registry = Arc::new(TestLocalProcessRegistry::default());
    let prompt_captures = Arc::new(StdMutex::new(Vec::new()));
    let provider = scripted_provider(
        vec![
            r#"```lashlang
process request_answer(tools: Tools) {
  result = await tools.mock_input_request({ question: "Need input?" })?
  finish result
}
handle = start request_answer(tools: tools)
result = (await handle)?
submit result.answer
```"#,
            "```lashlang\nsubmit { recovered: true }\n```",
        ],
        Arc::clone(&prompt_captures),
    );
    let (key_tx, key_rx) = oneshot::channel();
    let tools = Arc::new(DurableInputTools::new(key_tx));
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(provider)
        .model(mock_model_spec())
        .tools(Arc::clone(&tools) as Arc<dyn ToolProvider>)
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::clone(&process_registry) as Arc<dyn ProcessRegistry>)
        .lashlang_execution_sink(Arc::clone(&graph_store) as Arc<dyn crate::tracing::TraceSink>)
        .build()?;
    let session = core.session(session_id).open().await?;
    let events = Arc::new(RecordingEvents::default());
    let turn_session = session.clone();
    let turn_events = Arc::clone(&events);
    let mut turn = tokio::spawn(async move {
        turn_session
            .turn(TurnInput::text(
                "Start a process that asks for durable input.",
            ))
            .stream_to(turn_events.as_ref())
            .await
    });

    let key_result = tokio::time::timeout(std::time::Duration::from_secs(1), key_rx)
        .await
        .expect("durable input tool should publish await key")
        .expect("durable input key sender should stay alive");
    let key = match key_result {
        Ok(key) => key,
        Err(err) => {
            panic!(
                "durable input tool failed before awaiting external input: {err}; events: {:#?}",
                events.snapshot().await
            )
        }
    };
    assert!(
        matches!(
            key.wait,
            lash_core::AwaitEventWaitIdentity::Custom { ref key }
                if key == "mock-input-request:request-1"
        ),
        "durable input tool should use a custom await key: {:?}",
        key.wait
    );
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    if let Ok(joined) = tokio::time::timeout(std::time::Duration::from_millis(20), &mut turn).await
    {
        let result = joined.expect("turn task completed before durable input resolution");
        panic!(
            "turn completed before the durable input request was resolved: {result:#?}; events: {:#?}",
            events.snapshot().await
        );
    }

    let answer = serde_json::json!({ "answer": "approved" });
    let outcome = core
        .completions()
        .resolve(key, lash_core::Resolution::Ok(answer))
        .await?;
    assert_eq!(outcome, lash_core::ResolveOutcome::Accepted);

    let turn_output = turn.await.expect("turn task")?;
    session.processes().await_all().await?;
    assert!(matches!(
        turn_output.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue { .. })
    ));
    assert_eq!(
        turn_output.submitted_value(),
        Some(&serde_json::json!("approved"))
    );
    assert_eq!(tools.step_count(), 2);

    let final_process_list = all_host_process_summaries(&core).await?;
    assert_remote_process_dto_surface(&core, process_registry.as_ref(), session_id).await;
    assert_remote_process_summaries_round_trip(&final_process_list);
    assert_eq!(
        final_process_list.len(),
        1,
        "durable input request should not start a child process"
    );
    assert_all_processes_terminal(&final_process_list);
    let process_id = final_process_list[0].process_id.clone();
    let process_events = core.processes().events(&process_id, 0).await?;
    assert!(
        process_events.iter().any(|event| {
            event.event_type == "process.yield"
                && event.payload.get("type")
                    == Some(&serde_json::json!("work.input_request.opened"))
                && event.payload.get("answer").is_none()
                && event.payload.get("request_id") == Some(&serde_json::json!("request-1"))
        }),
        "durable input request event was not appended: {process_events:#?}"
    );
    assert!(
        process_events
            .iter()
            .all(|event| event.event_type != "process.waiting"),
        "durable input request should not rely on wait_signal: {process_events:#?}"
    );
    assert_eq!(prompt_captures.lock().expect("prompt captures").len(), 1);
    let contract = GraphContract::from_graphs(&graph_store.graphs());
    assert_min_completed_process_graphs(&contract, 1);
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
