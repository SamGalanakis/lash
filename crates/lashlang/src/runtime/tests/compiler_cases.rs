#[test]
fn label_on_await_assignment_attaches_to_await_instruction() {
    let program = crate::parse(
        r#"
        @label(title: "Wait for child")
        result = await handle
        submit result
        "#,
    )
    .expect("program should parse");
    let surface = runtime_test_environment().with_language_features(
        crate::LashlangLanguageFeatures::default().with_label_annotations(),
    );
    let linked = crate::LinkedModule::link(program, surface).expect("program should link");
    let compiled = crate::compile_linked(&linked);
    let await_instruction = compiled
        .chunk
        .code
        .iter()
        .position(|instruction| matches!(instruction, Instruction::AwaitHandle))
        .expect("await handle instruction");

    assert!(
        compiled
            .chunk
            .lashlang_execution_sites
            .get(await_instruction)
            .and_then(Option::as_ref)
            .is_some(),
        "label should attach to the awaited effect instruction"
    );
    assert!(
        !compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::ObserveStep)),
        "label on awaited assignment should not emit a standalone observe step"
    );
}

#[test]
fn labeled_process_resource_operation_site_matches_static_graph_node() {
    let program = crate::parse(
        r#"
        process search_test() {
          @label(title: "Spawn subagent with web search")
          result = await tools.echo({ value: { ok: true } })?
          wake result
          finish result
        }
        "#,
    )
    .expect("program should parse");
    let surface = runtime_test_environment().with_language_features(
        crate::LashlangLanguageFeatures::default().with_label_annotations(),
    );
    let linked = crate::LinkedModule::link(program, surface).expect("program should link");
    let compiled =
        crate::compile_linked_process(&linked, "search_test").expect("process should compile");
    let resource_call = compiled
        .chunk
        .code
        .iter()
        .position(|instruction| matches!(instruction, Instruction::ResourceCallUnwrap { .. }))
        .expect("resource call unwrap instruction");
    let site = compiled
        .chunk
        .lashlang_execution_sites
        .get(resource_call)
        .and_then(Option::as_ref)
        .expect("resource call execution site");

    let process_ref = linked
        .artifact
        .process_ref("search_test")
        .expect("search_test process ref")
        .clone();
    let map = crate::map_lashlang_process(
        &linked.artifact,
        &process_ref,
        crate::LashlangMapOptions::default(),
    )
    .expect("process graph");
    let static_node =
        map.nodes
            .iter()
            .find(|node| {
                node.kind == "resource_operation"
                    && node.label_metadata.as_ref().is_some_and(|label| {
                        label.title.as_str() == "Spawn subagent with web search"
                    })
            })
            .unwrap_or_else(|| panic!("missing labeled resource operation node: {:?}", map.nodes));

    assert_eq!(site.node_kind, "resource_operation");
    assert_eq!(site.node_id, static_node.id);
    assert!(
        !compiled
            .chunk
            .lashlang_execution_sites
            .iter()
            .flatten()
            .any(|site| {
                site.node_kind == "step" && site.label == "Spawn subagent with web search"
            }),
        "labeled resource operation should not emit a parallel step site"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn generic_iterator_loops_cover_range_list_keys_nested_control_and_mutation() {
    let source = r#"
        counts = {}
        items = ["a", "b", "a"]
        seen = []
        total = 0
        for i in range(0, 5) {
          if i == 1 {
            continue
          }
          if i == 4 {
            break
          }
          total = total + i
        }
        for item in items {
          counts[item] = counts[item] + 1
          seen = push(seen, format("{}:{}", item, counts[item]))
        }
        pairs = []
        for key in keys(counts) {
          for n in range(0, counts[key]) {
            pairs = pairs + [format("{}{}", key, n)]
          }
        }
        submit { total: total, counts: counts, seen: seen, pairs: pairs }
    "#;
    let compiled = compile_source(source).expect("program should compile");
    let begin_iterators = compiled
        .chunk
        .code
        .iter()
        .filter(|instruction| {
            matches!(
                instruction,
                Instruction::BeginIter(_) | Instruction::BeginRangeIter { .. }
            )
        })
        .count();
    assert!(
        begin_iterators >= 4,
        "every `for` loop should compile to the generic iterator bytecode, got {begin_iterators}"
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &Host)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(Value::Record(record)) = outcome else {
        panic!("expected record result");
    };
    assert_eq!(record["total"], Value::Number(5.0));
    let counts = record["counts"].as_record().expect("counts record");
    assert_eq!(counts["a"], Value::Number(2.0));
    assert_eq!(counts["b"], Value::Number(1.0));
    let Value::List(seen) = &record["seen"] else {
        panic!("seen should be a list");
    };
    assert_eq!(seen.len(), 3);
    let Value::List(pairs) = &record["pairs"] else {
        panic!("pairs should be a list");
    };
    assert_eq!(pairs.len(), 3);
}

#[test]
fn effectful_loop_bodies_compile_to_generic_iterator_bytecode() {
    let program = crate::parse(
        r#"
        items = [1, 2]
        for item in items {
          print item
        }
        submit null
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);
    assert!(
        compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::BeginIter(_))),
        "effectful loops should use the generic iterator bytecode"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn constant_propagation_does_not_cross_control_flow_boundaries() {
    let value = exec(
        r#"
        x = 1
        if false {
          x = 2
        }
        y = x + 1
        submit y
        "#,
    )
    .await
    .expect("program should succeed");

    assert_eq!(value, Value::Number(2.0));
}

#[tokio::test(flavor = "current_thread")]
async fn reusable_execution_scratch_preserves_results_across_runs() {
    let program = crate::parse(
        r#"
        items = [1, 2, 3]
        total = 0
        for item in items {
          total = total + item
        }
        submit total
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);
    let mut scratch = ExecutionScratch::new();

    for _ in 0..3 {
        let mut state = State::new();
        let outcome = execute_compiled_with_scratch(&compiled, &mut state, &Host, &mut scratch)
            .await
            .expect("program should run");
        assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(6.0)));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn continuation_and_undefined_variable_are_reported() {
    let outcome = exec_outcome("x = 1")
        .await
        .expect("missing submit should continue");
    assert_eq!(outcome, ExecutionOutcome::Continued);

    let value = exec("submit").await.expect("bare submit should succeed");
    assert_eq!(value, Value::Null);

    let err = exec("submit x")
        .await
        .expect_err("undefined variable should fail");
    assert_eq!(
        err,
        RuntimeError::UndefinedVariable {
            name: "x".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn condition_and_iteration_errors_are_reported() {
    let value = exec("if 1 { submit 1 } else { submit 2 }")
        .await
        .expect("numeric truthiness should be accepted");
    assert_eq!(value, Value::Number(1.0));

    let value = exec("if \"\" { submit 1 } else { submit 2 }")
        .await
        .expect("empty string should be falsy");
    assert_eq!(value, Value::Number(2.0));

    let err = exec("for x in 1 { submit x }")
        .await
        .expect_err("non-list iteration should fail");
    assert_eq!(err, RuntimeError::NonListIteration);
}

#[tokio::test(flavor = "current_thread")]
async fn stmt_call_and_tool_results_cover_success_and_error() {
    exec("await tools.echo({ value: 1 }) submit 1")
        .await
        .expect("statement module operation should succeed");
    let missing = exec("bad = await tools.missing({}) submit bad")
        .await
        .expect("missing module operation should be wrapped");
    assert_eq!(
        missing.as_record().expect("result should be a record")["ok"],
        Value::Bool(false)
    );

    let value = exec(
        "ok = await tools.echo({ value: 7 }) bad = await tools.err({}) submit { ok: ok, bad: bad }",
    )
    .await
    .expect("module operation program should succeed");
    let record = value.as_record().expect("expected record");
    assert_eq!(record["ok"].as_record().unwrap()["ok"], Value::Bool(true));
    assert_eq!(record["bad"].as_record().unwrap()["ok"], Value::Bool(false));
}

#[tokio::test(flavor = "current_thread")]
async fn result_unwrap_extracts_success_and_preserves_manual_handling() {
    let value = exec("submit (await tools.echo({ value: 7 })?)")
        .await
        .expect("unwrap should succeed");
    assert_eq!(value, Value::Number(7.0));

    let value = exec(
        r#"
        result = await tools.err({})
        submit result.ok ? result.error : "unexpected"
        "#,
    )
    .await
    .expect("manual wrapper handling should still work");
    assert_eq!(value, Value::String("unexpected".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn unparenthesized_await_module_operation_unwrap_skips_handle_await() {
    let compiled = compile_source("value = await tools.echo({ value: 7 })?\nsubmit value")
        .expect("program should compile");
    assert_resource_call_unwrap_without_handle_await(&compiled);

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &RejectingAwaitHost)
        .await
        .expect("program should run");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(7.0)));
}

#[tokio::test(flavor = "current_thread")]
async fn parenthesized_await_module_operation_unwrap_skips_handle_await() {
    let compiled = compile_source("value = (await tools.echo({ value: 7 }))?\nsubmit value")
        .expect("program should compile");
    assert_resource_call_unwrap_without_handle_await(&compiled);

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &RejectingAwaitHost)
        .await
        .expect("program should run");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(7.0)));
}

#[tokio::test(flavor = "current_thread")]
async fn labeled_await_module_operation_unwrap_skips_handle_await() {
    let compiled = compile_labeled_source(
        r#"@label(title: "Echo")
value = await tools.echo({ value: { answer: "ok" } })?
submit value"#,
    );
    assert_resource_call_unwrap_without_handle_await(&compiled);

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &RejectingAwaitHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finished outcome");
    };
    let record = value
        .as_record()
        .expect("finished value should be a record");
    assert_eq!(record["answer"], Value::String("ok".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn labeled_parenthesized_await_module_operation_unwrap_skips_handle_await() {
    let compiled = compile_labeled_source(
        r#"@label(title: "Echo")
value = (await tools.echo({ value: { answer: "ok" } }))?
submit value"#,
    );
    assert_resource_call_unwrap_without_handle_await(&compiled);

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &RejectingAwaitHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finished outcome");
    };
    let record = value
        .as_record()
        .expect("finished value should be a record");
    assert_eq!(record["answer"], Value::String("ok".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn labeled_process_await_module_operation_unwrap_skips_handle_await() {
    let compiled = compile_labeled_process_source(
        r#"
process echo_from_process() {
  @label(title: "Echo")
  value = await tools.echo({ value: { answer: "ok" } })?
  finish value
}
"#,
        "echo_from_process",
    );
    assert_resource_call_unwrap_without_handle_await(&compiled);

    let mut state = State::new();
    let outcome = execute_compiled_process(&compiled, &mut state, &RejectingAwaitHost)
        .await
        .expect("process should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finished outcome");
    };
    let record = value
        .as_record()
        .expect("finished value should be a record");
    assert_eq!(record["answer"], Value::String("ok".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn direct_module_operation_unwrap_skips_observable_wrapper() {
    let compiled =
        compile_source("submit (await tools.echo({ value: 7 })?)").expect("program should compile");
    assert_resource_call_unwrap_without_handle_await(&compiled);
    assert!(
        !compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::ResultUnwrap))
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &RejectingAwaitHost)
        .await
        .expect("program should run");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(7.0)));

    let err = exec("submit (await tools.err({})?)")
        .await
        .expect_err("failed unwrap should abort");
    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "`?` unwrapped failed module operation: boom".to_string(),
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn result_unwrap_reports_failed_and_malformed_wrappers() {
    let err = exec("submit (await tools.err({})?)")
        .await
        .expect_err("failed module operation unwrap should abort");
    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "`?` unwrapped failed module operation: boom".to_string(),
        }
    );

    let err = exec("submit 1?")
        .await
        .expect_err("non-wrapper should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`?` expected a tool result wrapper, got number".to_string(),
        }
    );

    let err = exec("submit { ok: true }?")
        .await
        .expect_err("missing value should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`?` found a successful tool result wrapper missing `value`".to_string(),
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn field_index_unary_and_boolean_paths_are_covered() {
    let value = exec(
        r#"
        rec = { nested: { name: "lash" } }
        xs = ["a", "b"]
        ok = false and missing
        alt = true or missing
        submit [rec.nested.name, xs[1], "abc"[2], -1, not false, !false, ok, alt]
        "#,
    )
    .await
    .expect("program should succeed");

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::String("lash".to_string().into()),
                Value::String("b".to_string().into()),
                Value::String("c".to_string().into()),
                Value::Number(-1.0),
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(true),
            ]
            .into()
        )
    );

    let value = exec("submit true and false")
        .await
        .expect("and path should succeed");
    assert_eq!(value, Value::Bool(false));

    let value = exec("submit false or true")
        .await
        .expect("or path should succeed");
    assert_eq!(value, Value::Bool(true));
}

#[tokio::test(flavor = "current_thread")]
async fn field_index_and_type_errors_are_covered() {
    let err = exec("n = 1 submit n.name")
        .await
        .expect_err("field access should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("rec = {} submit rec.name")
        .await
        .expect("missing field should yield null");
    assert_eq!(value, Value::Null);

    let err = exec("submit 1[0]")
        .await
        .expect_err("bad index target should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("submit [1][2]")
        .await
        .expect("list oob should yield null");
    assert_eq!(value, Value::Null);

    let value = exec("submit \"a\"[2]")
        .await
        .expect("string oob should yield null");
    assert_eq!(value, Value::Null);

    let err = exec("submit [1][1.5]")
        .await
        .expect_err("fractional index should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("submit [1][-1]")
        .await
        .expect("negative index should resolve from the end");
    assert_eq!(value, Value::Number(1.0));

    let value = exec("submit not 1")
        .await
        .expect("not should use truthiness");
    assert_eq!(value, Value::Bool(false));

    let value = exec("submit not 0").await.expect("zero should be falsy");
    assert_eq!(value, Value::Bool(true));

    let value = exec("rec = { ok: false } submit len(rec.value.items)")
        .await
        .expect("null chain should work");
    assert_eq!(value, Value::Number(0.0));
}

#[tokio::test(flavor = "current_thread")]
async fn arithmetic_and_compare_errors_are_covered() {
    assert_eq!(
        exec("submit 7 - 2").await.expect("subtract should succeed"),
        Value::Number(5.0)
    );
    assert_eq!(
        exec("submit 3 * 4").await.expect("multiply should succeed"),
        Value::Number(12.0)
    );
    assert_eq!(
        exec("submit 8 / 2").await.expect("divide should succeed"),
        Value::Number(4.0)
    );
    assert_eq!(
        exec("submit 8 % 3").await.expect("modulo should succeed"),
        Value::Number(2.0)
    );
    assert_eq!(
        exec("submit 1 != 2")
            .await
            .expect("not equal should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 1 <= 2")
            .await
            .expect("less-equal should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 2 > 1").await.expect("greater should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 2 >= 1")
            .await
            .expect("greater-equal should succeed"),
        Value::Bool(true)
    );

    let value = exec("submit [1,2] + [3]")
        .await
        .expect("list concat should succeed");
    assert_eq!(
        value,
        Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)].into())
    );

    let value = exec("submit \"a\" + \"b\"")
        .await
        .expect("string add should succeed");
    assert_eq!(value, Value::String("ab".to_string().into()));

    let value = exec("submit \"a\" + 1")
        .await
        .expect("string coercion should succeed");
    assert_eq!(value, Value::String("a1".to_string().into()));

    let value = exec("submit 1 + \"b\"")
        .await
        .expect("string coercion should succeed");
    assert_eq!(value, Value::String("1b".to_string().into()));

    let value = exec("submit 1 + true")
        .await
        .expect("bool should coerce for addition");
    assert_eq!(value, Value::Number(2.0));

    let value = exec("submit null + 2")
        .await
        .expect("null should coerce for addition");
    assert_eq!(value, Value::Number(2.0));

    let value = exec("submit \"2\" * 3")
        .await
        .expect("numeric strings should coerce");
    assert_eq!(value, Value::Number(6.0));

    let value = exec("submit \"2\" < 10")
        .await
        .expect("numeric strings should compare");
    assert_eq!(value, Value::Bool(true));

    let err = exec("submit {} + 1")
        .await
        .expect_err("records should still fail arithmetic");
    assert!(matches!(err, RuntimeError::TypeError { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_success_matrix_is_covered() {
    let value = exec(
        r#"
        rec = { a: 1, b: 2 }
        base = [1, 2]
        submit {
          len_s: len("ab"),
          len_l: len([1,2,3]),
          len_r: len(rec),
          len_n: len(null),
          empty_n: empty(null),
          empty_s: empty(""),
          empty_l: empty([]),
          empty_r: empty({}),
          keys_n: keys(null),
          values_n: values(null),
          keys: keys(rec),
          values: values(rec),
          contains_s: contains("abc", "b"),
          contains_num: contains("123", 2),
          contains_l: contains([1,2,3], 2),
          contains_r: contains({ foo: 1, bar: 2 }, "foo"),
          contains_n: contains(null, 2),
          find_hit: find("alpha beta", "beta"),
          find_from: find("banana", "na", 3),
          find_missing: find("alpha", "z"),
          starts: starts_with("lash", "la"),
          starts_num: starts_with(123, 12),
          ends: ends_with("lash", "sh"),
          split: split(101, 0),
          join: join(["a",2,true], "-"),
          trim: trim(101),
          slice_s: slice("abcd", 1, 3),
          slice_end_s: slice("abcd", 2, null),
          slice_back_s: slice("abcd", 3, 1),
          slice_from_start_s: slice("abcd", null, 2),
          slice_l: slice([1,2,3,4], 1, 3),
          slice_end_l: slice([1,2,3,4], 2, null),
          slice_back_l: slice([1,2,3,4], 3, 1),
          slice_from_start_l: slice([1,2,3,4], null, 2),
          to_s: to_string({ a: 1 }),
          to_i_n: to_int(3.9),
          to_i_s: to_int("4"),
          to_i_b: to_int(true),
          to_f_n: to_float(1),
          to_f_s: to_float("2.5"),
          to_f_nl: to_float(null),
          fmt: format("x={},y={}", 1, true),
          range_end: range(3),
          range_pair: range(-2, 2),
          range_step: range(0, 5, 2),
          range_step_down: range(5, 0, -2),
          range_empty: range(2, 2),
          ceil_div: ceil_div(10, 3),
          floor_div: floor_div(-10, 3),
          pushed: push(base, 3),
          base_after_push: base,
          valid: validate(
            {
              name: "pkg",
              version: "1.0.0",
              deps: [{ name: "dep", optional: true }],
              extra: "preserved"
            },
            Type {
              name: str,
              version: str,
              deps: list[Type { name: str, optional: bool? }]
            }
          )
        }
        "#,
    )
    .await
    .expect("builtins should succeed");

    let record = value.as_record().expect("expected record");
    assert_eq!(record["len_s"], Value::Number(2.0));
    assert_eq!(record["len_n"], Value::Number(0.0));
    assert_eq!(record["contains_num"], Value::Bool(true));
    assert_eq!(record["contains_l"], Value::Bool(true));
    assert_eq!(record["contains_r"], Value::Bool(true));
    assert_eq!(record["contains_n"], Value::Bool(false));
    assert_eq!(record["find_hit"], Value::Number(6.0));
    assert_eq!(record["find_from"], Value::Number(4.0));
    assert_eq!(record["find_missing"], Value::Null);
    assert_eq!(record["keys_n"], Value::List(Vec::new().into()));
    assert_eq!(record["values_n"], Value::List(Vec::new().into()));
    assert_eq!(record["starts_num"], Value::Bool(true));
    assert_eq!(
        record["split"],
        Value::List(
            vec![
                Value::String("1".to_string().into()),
                Value::String("1".to_string().into())
            ]
            .into()
        )
    );
    assert_eq!(record["join"], Value::String("a-2-true".to_string().into()));
    assert_eq!(record["trim"], Value::String("101".to_string().into()));
    assert_eq!(record["slice_s"], Value::String("bc".to_string().into()));
    assert_eq!(
        record["slice_end_s"],
        Value::String("cd".to_string().into())
    );
    assert_eq!(record["slice_back_s"], Value::String(String::new().into()));
    assert_eq!(
        record["slice_from_start_s"],
        Value::String("ab".to_string().into())
    );
    assert_eq!(
        record["slice_end_l"],
        Value::List(vec![Value::Number(3.0), Value::Number(4.0)].into())
    );
    assert_eq!(record["slice_back_l"], Value::List(Vec::new().into()));
    assert_eq!(
        record["slice_from_start_l"],
        Value::List(vec![Value::Number(1.0), Value::Number(2.0)].into())
    );
    assert_eq!(record["to_i_n"], Value::Number(3.0));
    assert_eq!(record["to_i_b"], Value::Number(1.0));
    assert_eq!(record["to_f_s"], Value::Number(2.5));
    assert_eq!(record["to_f_nl"], Value::Number(0.0));
    assert_eq!(
        record["fmt"],
        Value::String("x=1,y=true".to_string().into())
    );
    assert_eq!(
        record["range_end"],
        Value::List(vec![Value::Number(0.0), Value::Number(1.0), Value::Number(2.0)].into())
    );
    assert_eq!(
        record["range_pair"],
        Value::List(
            vec![
                Value::Number(-2.0),
                Value::Number(-1.0),
                Value::Number(0.0),
                Value::Number(1.0)
            ]
            .into()
        )
    );
    assert_eq!(record["range_empty"], Value::List(Vec::new().into()));
    assert_eq!(
        record["range_step"],
        Value::List(vec![Value::Number(0.0), Value::Number(2.0), Value::Number(4.0)].into())
    );
    assert_eq!(
        record["range_step_down"],
        Value::List(vec![Value::Number(5.0), Value::Number(3.0), Value::Number(1.0)].into())
    );
    assert_eq!(record["ceil_div"], Value::Number(4.0));
    assert_eq!(record["floor_div"], Value::Number(-4.0));
    assert_eq!(
        record["pushed"],
        Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)].into())
    );
    assert_eq!(
        record["base_after_push"],
        Value::List(vec![Value::Number(1.0), Value::Number(2.0)].into())
    );
    let valid = record["valid"].as_record().expect("validated record");
    assert_eq!(valid["name"], Value::String("pkg".to_string().into()));
    assert_eq!(
        valid["extra"],
        Value::String("preserved".to_string().into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn grep_text_returns_documented_line_records() {
    let value = exec(
        r#"
        matches = grep_text("alpha\nbeta match\r\ngamma match\n", "match")
        submit matches
        "#,
    )
    .await
    .expect("grep_text should succeed");

    let Value::List(matches) = value else {
        panic!("expected list");
    };
    assert_eq!(matches.len(), 2);
    let first = matches[0].as_record().expect("first match record");
    assert_eq!(first["line"], Value::Number(2.0));
    assert_eq!(first["text"], Value::String("beta match".into()));
    assert_eq!(first["match"], Value::String("match".into()));
    assert_eq!(first["start"], Value::Number(5.0));
    assert_eq!(first["end"], Value::Number(10.0));
}

#[tokio::test(flavor = "current_thread")]
async fn find_uses_character_offsets() {
    let value = exec(
        r#"
        submit {
          first: find("éclair café", "café"),
          from_end: find("abc", "", 3),
          beyond_end: find("abc", "a", 4)
        }
        "#,
    )
    .await
    .expect("find should succeed");

    let record = value.as_record().expect("record");
    assert_eq!(record["first"], Value::Number(7.0));
    assert_eq!(record["from_end"], Value::Number(3.0));
    assert_eq!(record["beyond_end"], Value::Null);
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_error_matrix_is_covered() {
    let cases = [
        ("submit len(true)", "len"),
        ("submit empty(true)", "empty"),
        ("submit keys([])", "keys"),
        ("submit values([])", "values"),
        ("submit contains(1, 2)", "contains"),
        ("submit find(\"a\")", "find"),
        ("submit find(\"a\", \"a\", -1)", "find"),
        ("submit grep_text({}, \"a\")", "grep_text"),
        ("submit grep_text(\"a\", \"\")", "grep_text"),
        ("submit starts_with({}, \"a\")", "starts_with"),
        ("submit ends_with({}, \"a\")", "ends_with"),
        ("submit split({}, \",\")", "split"),
        ("submit join(1, \",\")", "join"),
        ("submit trim({})", "trim"),
        ("submit slice(1, 0, 1)", "slice"),
        ("submit to_int({})", "to_int"),
        ("submit to_int(\"x\")", "to_int"),
        ("submit to_float({})", "to_float"),
        ("submit to_float(\"x\")", "to_float"),
        ("submit json_parse(\"{\")", "json_parse"),
        ("submit format()", "format"),
        ("submit format({})", "format"),
        ("submit format(\"{1}\", \"x\")", "format"),
        ("submit format(\"{}\", \"x\", \"y\")", "format"),
        ("submit format(\"{} {1}\", \"x\", \"y\")", "format"),
        ("submit format(\"{x}\")", "format"),
        ("submit format(\"{\")", "format"),
        ("submit format(\"}\")", "format"),
        (
            "submit validate({ name: \"pkg\" }, { type: \"object\" })",
            "validate",
        ),
        ("submit range()", "range"),
        ("submit range(1, 2, 3, 4)", "range"),
        ("submit range(\"3\")", "range"),
        ("submit range(1.5)", "range"),
        ("submit range(0, 5, 0)", "range"),
        ("submit range(0, 1000001)", "range"),
        ("submit range(1000001, 0, -1)", "range"),
        ("submit ceil_div()", "ceil_div"),
        ("submit ceil_div(1.5, 1)", "ceil_div"),
        ("submit ceil_div(1, 0)", "ceil_div"),
        ("submit floor_div(\"1\", 1)", "floor_div"),
        ("submit push(1, 2)", "push"),
        ("submit push([1])", "push"),
        ("submit no_such_builtin()", "no_such_builtin"),
    ];

    for (source, _) in cases {
        let err = exec(source).await.expect_err("builtin should fail");
        assert!(matches!(
            err,
            RuntimeError::TypeError { .. }
                | RuntimeError::ValueError { .. }
                | RuntimeError::UnknownBuiltin { .. }
        ));
    }

    let err = exec("submit len()")
        .await
        .expect_err("arity error should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn validate_reports_precise_shape_errors() {
    let cases = [
        (
            "submit validate({ name: \"pkg\" }, Type { name: str, version: str })",
            "validation failed: $: missing required field `version`",
        ),
        (
            r#"submit validate({ packages: [{ name: "pkg", version: 1 }] }, Type { packages: list[Type { name: str, version: str }] })"#,
            "validation failed: $.packages[0].version: expected string, got number",
        ),
        (
            r#"submit validate({ status: "maybe" }, Type { status: enum["ok", "err"] })"#,
            "validation failed: $.status: expected one of [ok, err], got maybe",
        ),
        (
            r#"submit validate({ count: 1.5 }, Type { count: int })"#,
            "validation failed: $.count: expected integer, got number",
        ),
    ];

    for (source, expected) in cases {
        let err = exec(source)
            .await
            .expect_err("validate should reject bad value");
        assert_eq!(
            err,
            RuntimeError::ValueError {
                message: expected.to_string()
            }
        );
    }

    let err = exec("submit validate({ name: \"pkg\" }, { type: \"object\" })")
        .await
        .expect_err("raw schema records should be rejected");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`validate` requires a Type literal as the second argument".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn validate_union_accepts_any_variant() {
    // `str | null` must accept both a string and a null.
    let out = exec(r#"submit validate({ email: "a@b" }, Type { email: str | null })"#)
        .await
        .expect("string-branch validate should succeed");
    assert_eq!(
        out,
        Value::Record(Arc::new({
            let mut rec = record_with_capacity(1);
            rec.insert("email".into(), Value::String("a@b".into()));
            rec
        }))
    );

    let out = exec(r#"submit validate({ email: null }, Type { email: str | null })"#)
        .await
        .expect("null-branch validate should succeed");
    let Value::Record(rec) = &out else {
        panic!("expected record");
    };
    assert!(matches!(rec.get("email"), Some(Value::Null)));
}

#[tokio::test(flavor = "current_thread")]
async fn validate_union_rejects_value_matching_no_variant() {
    let err = exec(r#"submit validate({ email: 42 }, Type { email: str | null })"#)
        .await
        .expect_err("number should not match str | null");
    let RuntimeError::ValueError { message } = err else {
        panic!("expected ValueError");
    };
    assert!(
        message.contains("$.email"),
        "error should point at the failing field: {message}",
    );
}

#[tokio::test(flavor = "current_thread")]
async fn validate_static_and_dynamic_type_paths_share_error_text() {
    let static_err = exec(r#"submit validate({ email: 42 }, Type { email: str | null })"#)
        .await
        .expect_err("static Type literal should reject number");
    let dynamic_err = exec(
        r#"
Schema = await tools.echo({ value: Type { email: str | null } })?
submit validate({ email: 42 }, Schema)
"#,
    )
    .await
    .expect_err("runtime Type value should reject number");

    assert_eq!(static_err, dynamic_err);
    assert_eq!(
        dynamic_err,
        RuntimeError::ValueError {
            message: "validation failed: $.email: expected one of [string, null], got number"
                .to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn validate_object_type_accepts_image_descriptors() {
    let program = crate::parse(
        r#"
submit validate(img, Type {
  type: str,
  id: str,
  label: str,
  size: int,
  width: int | null,
  height: int | null
})
"#,
    )
    .expect("program should parse");
    let mut state = State::new();
    state.globals.insert_str(
        "img",
        Value::Image(ImageValue::new("img-1", "chart.png", 1234, Some(640), None)),
    );

    let outcome = execute_program(&program, &mut state, &Host)
        .await
        .expect("image descriptor validation should succeed");
    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::Image(ImageValue::new(
            "img-1",
            "chart.png",
            1234,
            Some(640),
            None
        )))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn helper_functions_are_covered_directly() {
    assert!(expect_arg_count("x", &[Value::Null], 1).is_ok());
    assert!(expect_arg_count("x", &[], 1).is_err());
    assert_eq!(as_number(&Value::Number(1.0)).expect("number"), 1.0);
    assert_eq!(as_number(&Value::Bool(true)).expect("bool"), 1.0);
    assert_eq!(as_number(&Value::Null).expect("null"), 0.0);
    assert_eq!(
        as_number(&Value::String("2.5".to_string().into())).expect("numeric"),
        2.5
    );
    assert_eq!(
        coerce_string(&Value::String("x".to_string().into())).expect("string"),
        "x"
    );
    assert_eq!(coerce_string(&Value::Bool(true)).expect("bool"), "true");
    assert_eq!(as_offset(&Value::Number(-1.0)).expect("offset"), -1);
    assert_eq!(as_slice_bound(&Value::Null).expect("null bound"), None);
    assert_eq!(
        as_slice_bound(&Value::Number(2.0)).expect("numeric bound"),
        Some(2)
    );
    assert_eq!(
        as_slice_bound(&Value::Number(-2.0)).expect("negative numeric bound"),
        Some(-2)
    );
    assert_eq!(slice_string("héllo", Some(1), Some(4)), "éll");
    assert_eq!(slice_string("abcdef", Some(-2), None), "ef");
    assert_eq!(slice_string("abcdef", None, Some(-1)), "abcde");
    assert_eq!(slice_string("abcdef", Some(-5), Some(-2)), "bcd");
    assert_eq!(slice_string("abc", Some(1), None), "bc");
    assert_eq!(slice_string("abc", Some(3), Some(1)), "");
    assert_eq!(clamp_slice_bounds(Some(1), Some(3), 4), Some((1, 3)));
    assert_eq!(clamp_slice_bounds(Some(1), None, 4), Some((1, 4)));
    assert_eq!(clamp_slice_bounds(None, Some(2), 4), Some((0, 2)));
    assert_eq!(clamp_slice_bounds(Some(-2), None, 4), Some((2, 4)));
    assert_eq!(clamp_slice_bounds(None, Some(-1), 4), Some((0, 3)));
    assert_eq!(clamp_slice_bounds(Some(-10), Some(10), 4), Some((0, 4)));
    assert_eq!(clamp_slice_bounds(Some(3), Some(1), 4), None);
    assert_eq!(
        resolve_index(&Value::Number(-1.0), 3).expect("resolved"),
        Some(2)
    );
    assert_eq!(
        resolve_index(&Value::Number(-4.0), 3).expect("resolved"),
        None
    );

    assert_eq!(
        compare_numbers(Value::Number(1.0), Value::Number(2.0), |a, b| a < b).expect("compare"),
        Value::Bool(true)
    );
    assert_eq!(
        compare_ordered(
            Value::String("abc".to_string().into()),
            Value::String("def".to_string().into()),
            |a, b| a < b,
            |a, b| a < b,
        )
        .expect("string compare"),
        Value::Bool(true)
    );
    assert_eq!(
        add_values(Value::Number(1.0), Value::Number(2.0)).expect("add"),
        Value::Number(3.0)
    );
    assert_eq!(
        add_values(Value::String("a".to_string().into()), Value::Bool(true)).expect("concat"),
        Value::String("atrue".to_string().into())
    );
    assert_eq!(
        add_values(Value::Bool(true), Value::Number(2.0)).expect("numeric coercion"),
        Value::Number(3.0)
    );
    assert_eq!(
        success(Value::Number(1.0)).as_record().unwrap()["ok"],
        Value::Bool(true)
    );
    assert_eq!(
        error_value("x".to_string()).as_record().unwrap()["error"],
        Value::String("x".to_string().into())
    );
    assert_eq!(stringify_value(&Value::Null).expect("stringify"), "null");
    assert_eq!(
        stringify_value(&Value::Number(1.0)).expect("stringify"),
        "1"
    );
    assert_eq!(
        stringify_value(&Value::List(
            vec![Value::Number(1.0), Value::Number(2.0)].into()
        ))
        .expect("stringify"),
        "[1,2]"
    );
    let mut appended = String::from("prefix:");
    append_stringified_value(&mut appended, &Value::Bool(true)).expect("append stringify");
    assert_eq!(appended, "prefix:true");
    assert_eq!(
        apply_format("a{}b", &[Value::Number(1.0)]).expect("format"),
        "a1b"
    );
    let compiled_one_arg = compile_format_template("a{}b", 1);
    let one_arg = compiled_one_arg
        .one_arg
        .as_ref()
        .expect("single placeholder template should keep its direct shape");
    assert_eq!(one_arg.prefix.as_deref(), Some("a"));
    assert_eq!(one_arg.suffix.as_deref(), Some("b"));
    assert_eq!(
        execute_compiled_format_one_number_compact_direct(&compiled_one_arg, 42.0)
            .expect("compiled one-number format")
            .as_str(),
        "a42b"
    );
    assert!(
        compile_format_template("{}:{}", 2).one_arg.is_none(),
        "multi-arg templates should keep the generic compiled format path"
    );
    assert_eq!(
        apply_format(
            "b={1} a={0}",
            &[Value::String("x".into()), Value::String("y".into())]
        )
        .expect("indexed format"),
        "b=y a=x"
    );
    assert_eq!(
        apply_format("{{{}}}", &[Value::Number(1.0)]).expect("escaped braces"),
        "{1}"
    );
    assert_eq!(
        apply_format("{999999999999999999999999999999999999}", &[])
            .expect_err("overflow slot should fail"),
        RuntimeError::ValueError {
            message: "bad format slot `999999999999999999999999999999999999`".to_string()
        }
    );
    assert_eq!(
        apply_format("{x}", &[]).expect_err("invalid placeholder should fail"),
        RuntimeError::ValueError {
            message: "invalid format placeholder".to_string()
        }
    );
    assert_eq!(
        apply_format(
            "{} {1}",
            &[Value::String("x".into()), Value::String("y".into())]
        )
        .expect_err("mixed placeholder styles should fail"),
        RuntimeError::ValueError {
            message: "can't mix `{}` and indexed format placeholders".to_string()
        }
    );
    assert_eq!(
        apply_format("{", &[]).expect_err("unmatched open brace should fail"),
        RuntimeError::ValueError {
            message: "unmatched `{` in format string".to_string()
        }
    );
    assert_eq!(
        apply_format("}", &[]).expect_err("unmatched close brace should fail"),
        RuntimeError::ValueError {
            message: "unmatched `}` in format string".to_string()
        }
    );
    assert_eq!(
        apply_format("plain", &[Value::Number(1.0)]).expect_err("unused arg should fail"),
        RuntimeError::ValueError {
            message: "format argument `0` is unused".to_string()
        }
    );
    assert_eq!(
        value_type_name(&Value::Record(Record::default().into())),
        "record"
    );
    assert_eq!(
        RuntimeError::UndefinedVariable {
            name: "x".to_string()
        }
        .to_string(),
        "unknown name `x`"
    );
    assert_eq!(
        RuntimeError::TypeError {
            message: "can't index record".to_string()
        }
        .to_string(),
        "can't index record"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn json_helpers_cover_special_paths() {
    let json = to_json(&Value::Number(f64::NAN));
    assert_eq!(json, serde_json::Value::Null);
    assert_eq!(to_json(&Value::Null), serde_json::Value::Null);
    assert_eq!(to_json(&Value::Bool(true)), serde_json::Value::Bool(true));
    assert_eq!(
        to_json(&Value::String("x".to_string().into())),
        serde_json::Value::String("x".to_string())
    );
    assert_eq!(
        to_json(&Value::List(vec![Value::Number(1.0)].into())),
        serde_json::json!([1])
    );
    assert_eq!(
        to_json(&Value::Record({
            let mut record = Record::default();
            record.insert("a".to_string(), Value::Number(1.0));
            record.into()
        })),
        serde_json::json!({"a": 1})
    );

    let value = from_json(serde_json::json!({
        "a": [1, true, null, "x"]
    }));
    let record = value.as_record().expect("expected record");
    assert!(matches!(record["a"], Value::List(_)));
}
