use super::super::*;
use super::contracts::{
    GraphContract, assert_all_processes_terminal, assert_failed_code_block_present,
    assert_graph_lineage_connected, assert_labeled_resource_operation,
    assert_no_duplicate_label_step, assert_no_false_submitted_success,
    assert_no_forbidden_error_text, assert_subagent_bridge_exec_graphs,
};
use super::harness::{
    ExpectedContracts, LashE2eCase, run_durable_input_request_case, run_session_turn_process_case,
    run_turn_case, run_turn_case_without_success_assertions,
};

#[test]
fn lash_e2e_foreground_labeled_tool_call() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-foreground-tool", || async {
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
    run_async_test_on_stack_budget("lash-e2e-started-process-tool", || async {
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
fn lash_e2e_process_durable_input_request_tool() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-durable-input-request", || async {
        run_durable_input_request_case().await
    })
}

#[test]
fn lash_e2e_started_process_labeled_subagent_spawn() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-started-process-subagent", || async {
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
    run_async_test_on_stack_budget("lash-e2e-nested-process", || async {
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
    run_async_test_on_stack_budget("lash-e2e-session-turn-process", || async {
        run_session_turn_process_case().await
    })
}

#[test]
fn lash_e2e_failed_child_preserves_failure_graph() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-failed-child", || async {
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
    run_async_test_on_stack_budget("lash-e2e-parallel-spawn-join", || async {
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
