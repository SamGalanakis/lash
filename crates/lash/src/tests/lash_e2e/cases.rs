use super::super::*;
use super::contracts::{
    GraphContract, assert_all_processes_terminal, assert_failed_code_block_present,
    assert_graph_lineage_connected, assert_labeled_resource_operation,
    assert_no_duplicate_label_step, assert_no_false_finishted_success,
    assert_no_forbidden_error_text, assert_subagent_bridge_exec_graphs,
};
use super::harness::{
    ExpectedContracts, LashE2eCase, lashlang_block, run_durable_input_request_case,
    run_session_turn_process_case, run_turn_case, run_turn_case_without_success_assertions,
};
use std::collections::BTreeSet;

#[test]
fn lash_e2e_foreground_labeled_tool_call() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-foreground-tool", || async {
        let case = LashE2eCase {
            name: "foreground labeled tool call",
            session_id: "lash-e2e-foreground-tool",
            scripted_provider_responses: vec![lashlang_block(
                r#"
@label(title: "Lookup app state")
value = await tools.app_lookup({})?
finish value"#,
            )],
            root_prompt: "Call the app lookup tool and finish its value.",
            expected_final_value: Some(serde_json::json!({ "ok": true })),
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
            scripted_provider_responses: vec![lashlang_block(
                r#"
process lookup(tools: Tools) {
  @label(title: "Lookup app state in process")
  value = await tools.app_lookup({})?
  finish value
}
handle = start lookup(tools: tools)
result = (await handle)?
finish result"#,
            )],
            root_prompt: "Start a process that calls the app lookup tool.",
            expected_final_value: Some(serde_json::json!({ "ok": true })),
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
fn lash_e2e_shell_nonzero_and_pipeline_results_are_data() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-shell-results-are-data", || async {
        run_turn_case(LashE2eCase {
            name: "shell nonzero and pipeline results are data",
            session_id: "lash-e2e-shell-results-are-data",
            scripted_provider_responses: vec![lashlang_block(
                r#"
pipe = await shell.exec({ cmd: "yes line | head -n 3", login: false })?
missing = await shell.exec({ cmd: "test -f /tmp/lash-e2e-definitely-missing-file", login: false })?
finish {
  pipe_exit: pipe.exit_code,
  pipe_output: pipe.output,
  missing_exit: missing.exit_code,
  missing_status: missing.status
}"#,
            )],
            root_prompt: "Run shell commands and report their result metadata.",
            expected_final_value: Some(serde_json::json!({
                "pipe_exit": 0,
                "pipe_output": "line\nline\nline\n",
                "missing_exit": 1,
                "missing_status": "completed"
            })),
            tool_provider: Some(Arc::new(lash_tools::shell::shell_provider(
                lash_tools::shell::StandardShell::new(),
            ))),
            install_subagents: false,
            max_turns: None,
            expected_contracts: ExpectedContracts::default(),
        })
        .await?;
        Ok(())
    })
}

#[test]
fn lash_e2e_shell_output_survives_print_projection_in_variable() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-shell-output-variable", || async {
        run_turn_case(LashE2eCase {
            name: "shell output survives print projection in variable",
            session_id: "lash-e2e-shell-output-variable",
            scripted_provider_responses: vec![
                lashlang_block(
                    r#"
big = await shell.exec({ cmd: "yes x | head -c 60000", login: false })?
print big.output"#,
                ),
                lashlang_block(
                    r#"
finish {
  chars: len(big.output),
  tail: slice(big.output, 59996, null),
  has_full_output_path: big.full_output_path == null ? false : len(big.full_output_path) > 0
}"#,
                ),
            ],
            root_prompt: "Run a large shell command, inspect it, then report retained metadata.",
            expected_final_value: Some(serde_json::json!({
                "chars": 60000,
                "tail": "x\nx\n",
                "has_full_output_path": true
            })),
            tool_provider: Some(Arc::new(lash_tools::shell::shell_provider(
                lash_tools::shell::StandardShell::new(),
            ))),
            install_subagents: false,
            max_turns: None,
            expected_contracts: ExpectedContracts::default(),
        })
        .await?;
        Ok(())
    })
}

#[test]
fn lash_e2e_started_process_labeled_subagent_spawn() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-started-process-subagent", || async {
        run_turn_case(LashE2eCase {
            name: "started process labeled subagent spawn",
            session_id: "lash-e2e-process-subagent",
            scripted_provider_responses: vec![
                lashlang_block(
                    r#"
process spawn_child() {
  @label(title: "Spawn subagent with web search")
  result = await agents.spawn({
    capability: "default",
    task: "Finish `{ len: len(chunk) }` using the seeded `chunk` variable.",
    seed: { chunk: ["a", "b"] },
    output: Type { len: int }
  })?
  finish result
}
handle = start spawn_child()
result = (await handle)?
finish result"#,
                ),
                lashlang_block("finish { len: len(chunk) }"),
            ],
            root_prompt: "Run a Lashlang process that spawns a subagent and returns its value.",
            expected_final_value: Some(serde_json::json!({ "len": 2 })),
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
        let run = run_turn_case(LashE2eCase {
            name: "nested process start await",
            session_id: "lash-e2e-nested-process",
            scripted_provider_responses: vec![lashlang_block(
                r#"
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
finish result"#,
            )],
            root_prompt: "Start a parent process that starts and awaits a child process.",
            expected_final_value: Some(serde_json::json!({ "parent": "done" })),
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
        assert_lashlang_process_ids_unique_for_labels(&run.final_process_list, ["parent", "child"]);
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
                lashlang_block(
                    r#"
@label(title: "Spawn failing subagent")
result = await agents.spawn({
  capability: "default",
  task: "Fail with reason child boom.",
  seed: {},
  output: Type { reason: str }
})?
finish result"#,
                ),
                lashlang_block(r#"await task.fail({ reason: "child boom" })?"#),
            ],
            root_prompt: "Spawn a child that fails and preserve its execution graph.",
            expected_final_value: None,
            tool_provider: None,
            install_subagents: true,
            max_turns: Some(1),
            expected_contracts: ExpectedContracts::default(),
        })
        .await?;

        assert_failed_code_block_present(&run.streamed_events);
        assert_no_forbidden_error_text(&run.streamed_events);
        assert_no_false_finishted_success(&run);
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
        let run = run_turn_case(LashE2eCase {
            name: "parallel process spawn and join",
            session_id: "lash-e2e-parallel-processes",
            scripted_provider_responses: vec![lashlang_block(
                r#"
process child(value: str) {
  finish value
}
@label(title: "Start left process")
left = start child(value: "left")
@label(title: "Start right process")
right = start child(value: "right")
left_value = (await left)?
right_value = (await right)?
finish { joined: [left_value, right_value] }"#,
            )],
            root_prompt: "Start two processes, await both, and finish their joined result.",
            expected_final_value: Some(serde_json::json!({ "joined": ["left", "right"] })),
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
        assert_lashlang_process_ids_unique_for_labels(&run.final_process_list, ["child", "child"]);
        Ok(())
    })
}

#[test]
fn lash_e2e_tuple_values_finish_as_json_arrays() -> Result<()> {
    run_async_test_on_stack_budget("lash-e2e-tuple-values", || async {
        run_turn_case(LashE2eCase {
            name: "tuple values finish as json arrays",
            session_id: "lash-e2e-tuple-values",
            scripted_provider_responses: vec![lashlang_block(
                r#"
pair = "left", "right"
tail = slice(pair, 1, null)
seen = []
for item in pair {
  seen = push(seen, item)
}
finish {
  first: pair[0],
  tail: tail,
  seen: seen,
  tuple: pair,
  nested: { pair: pair }
}"#,
            )],
            root_prompt: "Use tuple values and finish the derived result.",
            expected_final_value: Some(serde_json::json!({
                "first": "left",
                "tail": ["right"],
                "seen": ["left", "right"],
                "tuple": ["left", "right"],
                "nested": { "pair": ["left", "right"] }
            })),
            tool_provider: None,
            install_subagents: false,
            max_turns: None,
            expected_contracts: ExpectedContracts::default(),
        })
        .await?;
        Ok(())
    })
}

fn assert_lashlang_process_ids_unique_for_labels<const N: usize>(
    processes: &[lash_core::ProcessHandleSummary],
    expected_labels: [&str; N],
) {
    let mut ids = BTreeSet::new();
    let mut labels = Vec::new();
    for process in processes {
        let Some(kind) = process.descriptor.kind.as_deref() else {
            continue;
        };
        if kind != lash_lashlang_runtime::LASHLANG_ENGINE_KIND {
            continue;
        }
        assert!(
            process.process_id.starts_with("process:lashlang:sha256:"),
            "lashlang process `{}` did not use a deterministic process id",
            process.process_id
        );
        assert!(
            ids.insert(process.process_id.as_str()),
            "duplicate lashlang process id `{}`",
            process.process_id
        );
        labels.push(process.descriptor.label.as_deref().unwrap_or("<missing>"));
    }
    labels.sort_unstable();
    let mut expected = expected_labels;
    expected.sort_unstable();
    assert_eq!(labels, expected);
}
