use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Host;

impl ToolHost for Host {
    async fn call(&self, name: String, args: Record) -> Result<Value, ToolHostError> {
        match name.as_str() {
            "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
            "err" => Err(ToolHostError::new("boom")),
            "panic" => panic!("boom"),
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }
}

async fn exec(source: &str) -> Result<Value, RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    match execute_program(&program, &mut state, &Host).await? {
        ExecutionOutcome::Finished(value) => Ok(value),
        ExecutionOutcome::Continued => panic!("expected `submit` in test program"),
    }
}

async fn exec_outcome(source: &str) -> Result<ExecutionOutcome, RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    execute_program(&program, &mut state, &Host).await
}

#[tokio::test(flavor = "current_thread")]
async fn value_helpers_and_display_cover_all_variants() {
    let mut record = Record::default();
    record.insert("k".to_string(), Value::Number(1.0));

    assert_eq!(Value::Null.to_string(), "null");
    assert_eq!(Value::Bool(true).to_string(), "true");
    assert_eq!(Value::Number(1.5).to_string(), "1.5");
    assert_eq!(Value::String("x".to_string().into()).to_string(), "x");
    assert_eq!(
        Value::List(vec![Value::Bool(true)].into()).to_string(),
        "[true]"
    );
    assert_eq!(
        Value::Record(record.clone().into()).as_record().unwrap()["k"],
        Value::Number(1.0)
    );
    assert!(Value::String("x".to_string().into()).as_record().is_none());
    assert!(Value::Record(record.into()).to_string().contains("\"k\":1"));
}

#[tokio::test(flavor = "current_thread")]
async fn compiler_folds_constant_list_and_record_literals() {
    let program = crate::parse(
        r#"
        items = [{ label: "a", weight: 1 }, { label: "b", weight: 2 }]
        submit items
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);

    assert!(
        compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::PushConst(_)))
    );
    assert!(
        !compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::BuildRecord(_)))
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &Host)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(Value::List(items)) = outcome else {
        panic!("expected folded list result");
    };
    assert_eq!(items.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn compiler_propagates_safe_straight_line_constants() {
    let program = crate::parse(
        r#"
        items = [1, 2, 3]
        indexes = range(0, len(items))
        extended = push(indexes, len(items))
        submit extended
        "#,
    )
    .expect("program should parse");
    let compiled = compile_program(&program);

    assert!(
        !compiled.chunk.code.iter().any(|instruction| matches!(
            instruction,
            Instruction::CallBuiltin {
                builtin: Builtin::Len | Builtin::Range | Builtin::Push,
                ..
            }
        )),
        "straight-line constant builtins should fold out of the runtime instruction stream"
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &Host)
        .await
        .expect("program should run");
    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::List(
            vec![
                Value::Number(0.0),
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0),
            ]
            .into()
        ))
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
    exec("call echo { value: 1 } submit 1")
        .await
        .expect("statement call should succeed");
    let missing = exec("bad = call missing {} submit bad")
        .await
        .expect("missing tool should be wrapped");
    assert_eq!(
        missing.as_record().expect("result should be a record")["ok"],
        Value::Bool(false)
    );

    let value = exec("ok = call echo { value: 7 } bad = call err {} submit { ok: ok, bad: bad }")
        .await
        .expect("tool call program should succeed");
    let record = value.as_record().expect("expected record");
    assert_eq!(record["ok"].as_record().unwrap()["ok"], Value::Bool(true));
    assert_eq!(record["bad"].as_record().unwrap()["ok"], Value::Bool(false));
}

#[tokio::test(flavor = "current_thread")]
async fn result_unwrap_extracts_success_and_preserves_manual_handling() {
    let value = exec("submit (call echo { value: 7 })?")
        .await
        .expect("unwrap should succeed");
    assert_eq!(value, Value::Number(7.0));

    let value = exec(
        r#"
        result = call err {}
        submit result.ok ? result.error : "unexpected"
        "#,
    )
    .await
    .expect("manual wrapper handling should still work");
    assert_eq!(value, Value::String("unexpected".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn direct_tool_call_unwrap_skips_observable_wrapper() {
    let program = crate::parse("submit (call echo { value: 7 })?").expect("program should parse");
    let compiled = compile_program(&program);
    assert!(
        compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::CallToolUnwrap { .. }))
    );
    assert!(
        !compiled
            .chunk
            .code
            .iter()
            .any(|instruction| matches!(instruction, Instruction::ResultUnwrap))
    );

    let mut state = State::new();
    let outcome = execute_compiled(&compiled, &mut state, &Host)
        .await
        .expect("program should run");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(7.0)));

    let err = exec("submit (call err {})?")
        .await
        .expect_err("failed unwrap should abort");
    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "`?` unwrapped failed tool result: boom".to_string(),
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn result_unwrap_reports_failed_and_malformed_wrappers() {
    let err = exec("submit (call err {})?")
        .await
        .expect_err("failed tool unwrap should abort");
    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "`?` unwrapped failed tool result: boom".to_string(),
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
          range_empty: range(2, 2),
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
async fn builtin_error_matrix_is_covered() {
    let cases = [
        ("submit len(true)", "len"),
        ("submit empty(true)", "empty"),
        ("submit keys([])", "keys"),
        ("submit values([])", "values"),
        ("submit contains(1, 2)", "contains"),
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
        ("submit range(1, 2, 3)", "range"),
        ("submit range(\"3\")", "range"),
        ("submit range(1.5)", "range"),
        ("submit range(0, 1000001)", "range"),
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

fn test_image() -> Value {
    Value::Image(ImageValue::new(
        "img-1",
        "chart.png",
        1234,
        Some(640),
        Some(480),
    ))
}

async fn exec_with_global(name: &str, value: Value, source: &str) -> Result<Value, RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    state.globals.insert(name.to_string(), value);
    match execute_program(&program, &mut state, &Host).await? {
        ExecutionOutcome::Finished(value) => Ok(value),
        ExecutionOutcome::Continued => panic!("expected `submit` in test program"),
    }
}

struct TestProjectedValue {
    values: Vec<Value>,
    get_count: AtomicUsize,
    materialize_count: AtomicUsize,
    render_count: AtomicUsize,
}

impl TestProjectedValue {
    fn new(values: Vec<Value>) -> Arc<Self> {
        Arc::new(Self {
            values,
            get_count: AtomicUsize::new(0),
            materialize_count: AtomicUsize::new(0),
            render_count: AtomicUsize::new(0),
        })
    }
}

impl ProjectedHostValue for TestProjectedValue {
    fn type_name(&self) -> &'static str {
        "list"
    }

    fn len(&self) -> ProjectedFuture<'_, Option<usize>> {
        Box::pin(async { Some(self.values.len()) })
    }

    fn get_index(&self, index: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async move {
            let Value::Number(index) = index else {
                return ProjectedRead::Missing;
            };
            if !index.is_finite() || index.fract() != 0.0 {
                return ProjectedRead::Missing;
            }
            let len = self.values.len() as isize;
            let index = index as isize;
            let index = if index < 0 { len + index } else { index };
            if index < 0 || index >= len {
                return ProjectedRead::Missing;
            }
            self.get_count.fetch_add(1, Ordering::SeqCst);
            self.values
                .get(index as usize)
                .cloned()
                .map(ProjectedRead::Value)
                .unwrap_or(ProjectedRead::Missing)
        })
    }

    fn render(&self) -> ProjectedFuture<'_, String> {
        Box::pin(async {
            self.render_count.fetch_add(1, Ordering::SeqCst);
            "<projected list>".to_string()
        })
    }

    fn materialize(&self) -> ProjectedFuture<'_, Value> {
        Box::pin(async {
            self.materialize_count.fetch_add(1, Ordering::SeqCst);
            Value::List(self.values.clone().into())
        })
    }
}

fn projected_list_bindings(name: &str, list: Arc<TestProjectedValue>) -> ProjectedBindings {
    let mut projected = ProjectedBindings::new();
    projected.insert(name, ProjectedValue::custom(name.to_string(), list));
    projected
}

async fn exec_with_projected(
    source: &str,
    projected: &ProjectedBindings,
) -> Result<(Value, State), RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    let outcome = execute_compiled_with_projected_bindings(
        &compile_program(&program),
        &mut state,
        &Host,
        projected,
    )
    .await?;
    match outcome {
        ExecutionOutcome::Finished(value) => Ok((value, state)),
        ExecutionOutcome::Continued => panic!("expected `submit` in test program"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn projected_list_len_and_index_are_lazy() {
    let list = TestProjectedValue::new(vec![Value::String("first".into()), Value::Number(2.0)]);
    let projected = projected_list_bindings("history", Arc::clone(&list));

    let (value, _) = exec_with_projected(
        "submit { n: len(history), first: history[0], missing: history[9] }",
        &projected,
    )
    .await
    .expect("projected read");

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["n"], Value::Number(2.0));
    assert_eq!(record["first"], Value::String("first".into()));
    assert_eq!(record["missing"], Value::Null);
    assert_eq!(list.get_count.load(Ordering::SeqCst), 1);
    assert_eq!(list.materialize_count.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn projected_bindings_are_read_only_and_not_snapshotted() {
    let list = TestProjectedValue::new(vec![Value::String("entry".into())]);
    let projected = projected_list_bindings("history", Arc::clone(&list));

    let err = exec_with_projected("history = []\nsubmit history", &projected)
        .await
        .expect_err("projected root assignment should fail");
    assert!(err.to_string().contains("read-only projected binding"));

    let (_, state) = exec_with_projected("alias = history\nsubmit alias[0]", &projected)
        .await
        .expect("alias should materialize");
    assert!(state.snapshot().globals.get("history").is_none());
    assert!(matches!(
        state.snapshot().globals.get("alias"),
        Some(Value::List(_))
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn projected_children_can_be_lazy_inside_ordinary_records() {
    let body = TestProjectedValue::new(vec![Value::String("lazy markdown".into())]);
    let mut record = Record::default();
    record.insert("title".to_string(), Value::String("Rules".into()));
    record.insert(
        "body".to_string(),
        Value::Projected(ProjectedValue::custom("body", body.clone())),
    );
    let mut projected = ProjectedBindings::new();
    projected.insert(
        "rules",
        ProjectedValue::scalar("rules", Value::Record(Arc::new(record))),
    );

    let (value, _) = exec_with_projected(
        "submit { title: rules.title, first_body_item: rules.body[0] }",
        &projected,
    )
    .await
    .expect("projected child read");

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["title"], Value::String("Rules".into()));
    assert_eq!(
        record["first_body_item"],
        Value::String("lazy markdown".into())
    );
    assert_eq!(body.get_count.load(Ordering::SeqCst), 1);
    assert_eq!(body.materialize_count.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn print_projected_uses_render_and_submit_materializes() {
    let list = TestProjectedValue::new(vec![Value::String("entry".into())]);
    let projected = projected_list_bindings("history", Arc::clone(&list));

    let (value, _) = exec_with_projected("print history\nsubmit history", &projected)
        .await
        .expect("projected print and submit");
    let _ = to_json(&value);

    assert_eq!(list.render_count.load(Ordering::SeqCst), 1);
    assert_eq!(list.materialize_count.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn image_values_expose_read_only_metadata_fields() {
    let value = exec_with_global(
        "img",
        test_image(),
        "submit [img.id, img.label, img.size, img.width, img.height, img.missing]",
    )
    .await
    .expect("image fields should read");

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::String("img-1".into()),
                Value::String("chart.png".into()),
                Value::Number(1234.0),
                Value::Number(640.0),
                Value::Number(480.0),
                Value::Null,
            ]
            .into()
        )
    );
}

#[tokio::test(flavor = "current_thread")]
async fn image_values_serialize_as_descriptors() {
    let image = test_image();
    assert_eq!(
        to_json(&image),
        serde_json::json!({
            "type": "image",
            "id": "img-1",
            "label": "chart.png",
            "size": 1234,
            "width": 640,
            "height": 480
        })
    );
    assert_eq!(
        stringify_value(&image).expect("stringify image"),
        r#"{"height":480,"id":"img-1","label":"chart.png","size":1234,"type":"image","width":640}"#
    );
    assert_eq!(
        exec_with_global("img", image.clone(), "submit img")
            .await
            .expect("submit image"),
        image
    );
}

#[tokio::test(flavor = "current_thread")]
async fn image_values_are_immutable_and_len_is_unsupported() {
    let err = exec_with_global("img", test_image(), "img.label = \"other\"\nsubmit img")
        .await
        .expect_err("image field assignment should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "can't assign image fields; images are immutable".to_string()
        }
    );

    let err = exec_with_global("img", test_image(), "submit len(img)")
        .await
        .expect_err("len image should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`len` requires a string, list, record, or null; use `.size` for images"
                .to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn false_if_branch_and_finish_inside_loop_are_covered() {
    let value = exec(
        r#"
        if false {
          out = 1
        } else {
          out = 2
        }
        submit out
        "#,
    )
    .await
    .expect("else branch should succeed");
    assert_eq!(value, Value::Number(2.0));

    let value = exec(
        r#"
        for x in [1, 2] {
          submit x
        }
        submit 0
        "#,
    )
    .await
    .expect("submit inside loop should bubble out");
    assert_eq!(value, Value::Number(1.0));
}

#[tokio::test(flavor = "current_thread")]
async fn parallel_branch_panics_are_reported_as_runtime_errors() {
    let err = exec(
        r#"
        parallel {
          crash = call panic {}
        }
        submit 1
        "#,
    )
    .await
    .expect_err("parallel panic should be reported");

    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "parallel branch panicked".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parallel_tool_calls_use_host_batch_when_available() {
    struct BatchHost {
        calls: AtomicUsize,
        batches: AtomicUsize,
    }

    impl ToolHost for BatchHost {
        async fn call(&self, _name: String, _args: Record) -> Result<Value, ToolHostError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(ToolHostError::new("single call should not be used"))
        }

        async fn call_batch(&self, calls: Vec<ToolHostCall>) -> Vec<Result<Value, ToolHostError>> {
            self.batches.fetch_add(1, Ordering::Relaxed);
            calls
                .into_iter()
                .map(|call| match call.name.as_str() {
                    "echo" => Ok(call.args.get("value").cloned().unwrap_or(Value::Null)),
                    other => Err(ToolHostError::new(format!("unknown tool: {other}"))),
                })
                .collect()
        }
    }

    let host = BatchHost {
        calls: AtomicUsize::new(0),
        batches: AtomicUsize::new(0),
    };
    let program = crate::parse(
        r#"
        result = parallel {
          left: call echo { value: "a" }
          right: call echo { value: "b" }
        }
        submit [result.left?, result.right?]
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &host)
        .await
        .expect("program should run");

    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::List(
            vec![Value::String("a".into()), Value::String("b".into())].into()
        ))
    );
    assert_eq!(host.calls.load(Ordering::Relaxed), 0);
    assert_eq!(host.batches.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn truthiness_covers_scalar_and_container_values() {
    assert!(!is_truthy(&Value::Null));
    assert!(!is_truthy(&Value::Bool(false)));
    assert!(!is_truthy(&Value::Number(0.0)));
    assert!(!is_truthy(&Value::String(String::new().into())));
    assert!(is_truthy(&Value::Bool(true)));
    assert!(is_truthy(&Value::Number(1.0)));
    assert!(is_truthy(&Value::List(Vec::new().into())));
    assert!(is_truthy(&Value::Record(Record::default().into())));
}

struct AsyncHost;

impl ToolHost for AsyncHost {
    async fn call(&self, name: String, args: Record) -> Result<Value, ToolHostError> {
        Host.call(name, args).await
    }

    async fn start_call(&self, name: String, args: Record) -> Result<Value, ToolHostError> {
        let mut record = Record::default();
        record.insert("__handle__".to_string(), Value::String("task".into()));
        record.insert("tool".to_string(), Value::String(name.into()));
        record.insert(
            "value".to_string(),
            args.get("value").cloned().unwrap_or(Value::Null),
        );
        Ok(Value::Record(Arc::new(record)))
    }

    async fn await_handle(&self, handle: Value) -> Result<Value, ToolHostError> {
        let record = handle
            .as_record()
            .ok_or_else(|| ToolHostError::new("expected handle record"))?;
        Ok(record.get("value").cloned().unwrap_or(Value::Null))
    }

    async fn cancel_handle(&self, _handle: Value) -> Result<Value, ToolHostError> {
        Ok(Value::Null)
    }
}

#[tokio::test(flavor = "current_thread")]
async fn async_tool_handles_can_be_started_awaited_and_cancelled() {
    let program = crate::parse(
        r#"
        handle = start call echo { value: "done" }
        result = await handle
        cancel handle
        submit result
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let record = value
        .as_record()
        .expect("await should return wrapped result");
    assert_eq!(record["ok"], Value::Bool(true));
    assert_eq!(record["value"], Value::String("done".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn sync_steps_resume_correctly_after_tool_effects() {
    let value = exec(
        r#"
        before = 20 + 2
        echoed = (call echo { value: before })?
        after = echoed + 1
        submit [before, echoed, after]
        "#,
    )
    .await
    .expect("program should run");

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::Number(22.0),
                Value::Number(22.0),
                Value::Number(23.0)
            ]
            .into()
        )
    );
}

#[tokio::test(flavor = "current_thread")]
async fn traced_async_tool_errors_keep_original_instruction_span() {
    let source = r#"
        before = 1
        value = (call err {})?
        submit value
        "#;
    let program = crate::parse(source).expect("program should parse");
    let compiled = compile_program(&program);
    let mut state = State::new();
    let failure = execute_compiled_traced(&compiled, &mut state, &Host)
        .await
        .expect_err("unwrapped tool error should fail");
    let message = crate::format_runtime_diagnostic(source, &failure.error, failure.span);

    assert!(
        message.contains("`?` unwrapped failed tool result: boom"),
        "{message}"
    );
    assert!(message.contains("--> line 3, column 9"), "{message}");
    assert!(message.contains("value = (call err {})?"), "{message}");
}

#[tokio::test(flavor = "current_thread")]
async fn profiled_tool_effect_keeps_sync_instruction_counts() {
    let source = r#"
        before = 20 + 2
        echoed = (call echo { value: before })?
        after = echoed + 1
        submit after
        "#;
    let program = crate::parse(source).expect("program should parse");
    let compiled = compile_program(&program);
    let mut state = State::new();
    let (_outcome, report) = profile_compiled(&compiled, &mut state, &Host)
        .await
        .expect("profile should succeed");
    let count = |name| {
        report
            .instruction_stats()
            .iter()
            .find(|stat| stat.name == name)
            .map_or(0, |stat| stat.count)
    };

    assert!(count("call_tool") > 0, "{:?}", report.instruction_stats());
    assert!(count("binary") > 0, "{:?}", report.instruction_stats());
    assert!(count("load_name") > 0, "{:?}", report.instruction_stats());
    assert!(count("store_name") >= 3, "{:?}", report.instruction_stats());
}

#[tokio::test(flavor = "current_thread")]
async fn await_unknown_handle_surfaces_runtime_error() {
    let program = crate::parse(
        r#"
        result = await 1
        submit result
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let record = value
        .as_record()
        .expect("await should return wrapped error");
    assert_eq!(record["ok"], Value::Bool(false));
    assert_eq!(
        record["error"],
        Value::String("expected handle record".into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn await_list_of_handles_returns_results_in_order() {
    let program = crate::parse(
        r#"
        handles = [
          start call echo { value: "first" },
          start call echo { value: "second" },
          start call echo { value: "third" }
        ]
        results = await handles
        submit results
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let Value::List(results) = value else {
        panic!("await list should return a list");
    };
    assert_eq!(results.len(), 3);
    for (result, expected) in results.iter().zip(["first", "second", "third"]) {
        let record = result
            .as_record()
            .expect("await should return wrapped result");
        assert_eq!(record["ok"], Value::Bool(true));
        assert_eq!(record["value"], Value::String(expected.into()));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn await_list_preserves_per_item_errors() {
    let program = crate::parse(
        r#"
        handles = [start call echo { value: "done" }, 1]
        results = await handles
        submit results
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let Value::List(results) = value else {
        panic!("await list should return a list");
    };
    let ok = results[0]
        .as_record()
        .expect("first result should be wrapped");
    assert_eq!(ok["ok"], Value::Bool(true));
    assert_eq!(ok["value"], Value::String("done".into()));

    let err = results[1]
        .as_record()
        .expect("second result should be wrapped");
    assert_eq!(err["ok"], Value::Bool(false));
    assert_eq!(err["error"], Value::String("expected handle record".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn await_record_of_handles_returns_record_of_wrappers() {
    let program = crate::parse(
        r#"
        handles = {
          first: start call echo { value: "one" },
          second: start call echo { value: "two" },
        }
        results = await handles
        submit [results.first?, results.second?]
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    assert_eq!(
        value,
        Value::List(vec![Value::String("one".into()), Value::String("two".into())].into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn result_unwrap_extracts_awaited_handles_and_parallel_results() {
    let program = crate::parse(
        r#"
        handle = start call echo { value: "done" }
        result = (await handle)?
        submit result
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    assert_eq!(value, Value::String("done".into()));

    let program = crate::parse(
        r#"
        results = parallel {
          call echo { value: "left" }
          call echo { value: "right" }
        }
        submit [(results[0])?, (results[1])?]
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost)
        .await
        .expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    assert_eq!(
        value,
        Value::List(vec![Value::String("left".into()), Value::String("right".into()),].into())
    );
}

// ------------------------------------------------------------------
//  Type literals: syntactic signatures with enum, list, nested, ref,
//  optional fields. See the top-level README for the full grammar.
// ------------------------------------------------------------------

/// Extract the inner JSON Schema wrapped by a `$lash_type` value.
fn unwrap_schema(value: &Value) -> &Record {
    crate::runtime::unwrap_type_value(value)
        .and_then(Value::as_record)
        .expect("Type value must unwrap to a schema record")
}

#[tokio::test(flavor = "current_thread")]
async fn type_scalar_schemas_const_fold_to_json_schema() {
    for (src, expected) in [
        ("submit Type { v: str }", "string"),
        ("submit Type { v: int }", "integer"),
        ("submit Type { v: float }", "number"),
        ("submit Type { v: bool }", "boolean"),
        ("submit Type { v: dict }", "object"),
    ] {
        let value = exec(src).await.expect("should succeed");
        let schema = unwrap_schema(&value);
        assert_eq!(schema["type"], Value::String("object".into()));
        let props = schema["properties"]
            .as_record()
            .expect("properties must be record");
        let v = props["v"].as_record().expect("field schema");
        assert_eq!(v["type"], Value::String(expected.into()));
        assert_eq!(
            schema["additionalProperties"],
            Value::Bool(false),
            "additionalProperties must be false for {src}",
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn type_any_is_empty_schema() {
    let value = exec("submit Type { v: any }")
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let props = schema["properties"].as_record().expect("properties");
    let v = props["v"].as_record().expect("field schema");
    assert!(v.is_empty(), "any must be an empty JSON Schema");
}

#[tokio::test(flavor = "current_thread")]
async fn type_enum_produces_string_with_enum_array() {
    let value = exec(r#"submit Type { status: enum["ok", "err", "pending"] }"#)
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let status = schema["properties"].as_record().unwrap()["status"]
        .as_record()
        .expect("enum field schema");
    assert_eq!(status["type"], Value::String("string".into()));
    let Value::List(values) = &status["enum"] else {
        panic!("enum must be a list");
    };
    let strings: Vec<_> = values.iter().collect();
    assert_eq!(strings.len(), 3);
    assert_eq!(strings[0], &Value::String("ok".into()));
    assert_eq!(strings[2], &Value::String("pending".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn type_list_schema_wraps_inner_type_as_items() {
    let value = exec("submit Type { tags: list[str] }")
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let tags = schema["properties"].as_record().unwrap()["tags"]
        .as_record()
        .expect("list field schema");
    assert_eq!(tags["type"], Value::String("array".into()));
    let items = tags["items"].as_record().expect("items schema");
    assert_eq!(items["type"], Value::String("string".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn type_list_of_enum_preserves_nested_shape() {
    let value = exec(r#"submit Type { labels: list[enum["a", "b"]] }"#)
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let labels = schema["properties"].as_record().unwrap()["labels"]
        .as_record()
        .expect("list schema");
    let items = labels["items"].as_record().expect("enum item schema");
    assert_eq!(items["type"], Value::String("string".into()));
    assert!(matches!(items["enum"], Value::List(_)));
}

#[tokio::test(flavor = "current_thread")]
async fn type_nested_object_is_full_subschema() {
    let value = exec(
        r#"
        submit Type {
          title: str,
          meta: Type {
            pages: int,
            published: int
          }
        }
        "#,
    )
    .await
    .expect("should succeed");
    let schema = unwrap_schema(&value);
    let meta = schema["properties"].as_record().unwrap()["meta"]
        .as_record()
        .expect("nested object schema");
    assert_eq!(meta["type"], Value::String("object".into()));
    let sub_props = meta["properties"].as_record().unwrap();
    assert_eq!(
        sub_props["pages"].as_record().unwrap()["type"],
        Value::String("integer".into())
    );
    let required = match &meta["required"] {
        Value::List(items) => items,
        _ => panic!("required must be list"),
    };
    assert_eq!(required.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn type_optional_field_drops_from_required() {
    let value = exec("submit Type { a: str, b: int? }")
        .await
        .expect("should succeed");
    let schema = unwrap_schema(&value);
    let required = match &schema["required"] {
        Value::List(items) => items,
        _ => panic!("required must be list"),
    };
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], Value::String("a".into()));
    // Optional field still appears in properties (just not required).
    let props = schema["properties"].as_record().unwrap();
    assert!(props.get("b").is_some());
}

#[tokio::test(flavor = "current_thread")]
async fn type_ref_resolves_previously_defined_type() {
    let src = r#"
        Inner = Type { count: int }
        Outer = Type { name: str, nested: Inner }
        submit Outer
    "#;
    let value = exec(src).await.expect("should succeed");
    let schema = unwrap_schema(&value);
    let nested = schema["properties"].as_record().unwrap()["nested"]
        .as_record()
        .expect("nested resolved schema");
    assert_eq!(nested["type"], Value::String("object".into()));
    let nested_props = nested["properties"].as_record().unwrap();
    assert_eq!(
        nested_props["count"].as_record().unwrap()["type"],
        Value::String("integer".into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn type_ref_to_non_type_value_is_type_error() {
    let err = exec(
        r#"
        Inner = { count: 5 }
        Outer = Type { nested: Inner }
        submit Outer
        "#,
    )
    .await
    .expect_err("should fail: Inner is not a Type value");
    assert!(
        matches!(err, RuntimeError::TypeError { .. }),
        "expected TypeError, got {err:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn type_ref_with_undefined_name_is_undefined_variable() {
    let err = exec("submit Type { nested: MissingType }")
        .await
        .expect_err("unknown ref should fail");
    assert_eq!(
        err,
        RuntimeError::UndefinedVariable {
            name: "MissingType".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn compile_stats_count_const_folded_and_dynamic_literals() {
    let src = r#"
        Inner = Type { n: int }
        A = Type { x: str }
        B = Type { nested: Inner }
        submit B
    "#;
    let program = crate::parse(src).expect("should parse");
    let compiled = crate::compile_program(&program);
    let stats = compiled.compile_stats();
    assert_eq!(stats.type_literals_total, 3);
    assert_eq!(
        stats.type_literals_const_folded, 2,
        "Inner and A are constant"
    );
    assert_eq!(stats.type_literals_dynamic, 1, "B references Inner");
    assert_eq!(stats.type_ref_sites, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn profile_report_surfaces_resolve_type_ref_counts() {
    let src = r#"
        Inner = Type { n: int }
        Outer = Type { nested: Inner }
        limit = (call echo { value: 1 })?
        numbers = push(range(limit), limit)
        checked = validate({ nested: { n: numbers[0] } }, Outer)
        submit checked
    "#;
    let program = crate::parse(src).expect("should parse");
    let compiled = crate::compile_program(&program);
    let mut state = State::new();
    let (_outcome, report) = crate::profile_compiled(&compiled, &mut state, &Host)
        .await
        .expect("profile should succeed");
    let names: Vec<_> = report.instruction_stats().iter().map(|s| s.name).collect();
    assert!(
        names.contains(&"resolve_type_ref"),
        "profile should track resolve_type_ref: {names:?}"
    );
    assert!(
        names.contains(&"wrap_type_literal"),
        "profile should track wrap_type_literal: {names:?}"
    );
    let builtin_names: Vec<_> = report.builtin_stats().iter().map(|s| s.name).collect();
    assert!(
        builtin_names.contains(&"validate"),
        "profile should track validate: {builtin_names:?}"
    );
    assert!(
        builtin_names.contains(&"range"),
        "profile should track range: {builtin_names:?}"
    );
    assert!(
        builtin_names.contains(&"push"),
        "profile should track push: {builtin_names:?}"
    );
    assert_eq!(report.compile_stats().type_literals_total, 2);
}

#[tokio::test(flavor = "current_thread")]
async fn type_literal_inside_tool_call_args_passes_through_as_record() {
    struct CaptureHost {
        captured: std::sync::Mutex<Option<Value>>,
    }
    impl ToolHost for CaptureHost {
        async fn call(&self, name: String, args: Record) -> Result<Value, ToolHostError> {
            if name == "spawn" {
                let schema = args
                    .get("output")
                    .cloned()
                    .expect("output arg must be present");
                *self.captured.lock().unwrap() = Some(schema);
                return Ok(Value::Null);
            }
            Err(ToolHostError::new(format!("unknown: {name}")))
        }
    }
    let host = CaptureHost {
        captured: std::sync::Mutex::new(None),
    };
    let program = crate::parse(
        r#"
        Shape = Type { name: str, tags: list[str] }
        call spawn { output: Shape }
        submit null
        "#,
    )
    .expect("should parse");
    let mut state = State::new();
    crate::execute_program(&program, &mut state, &host)
        .await
        .expect("should run");

    let captured = host.captured.lock().unwrap().clone().expect("captured");
    let inner = crate::runtime::unwrap_type_value(&captured).expect("has $lash_type");
    let schema = inner.as_record().expect("schema record");
    assert_eq!(schema["type"], Value::String("object".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn duplicate_field_name_is_parse_error() {
    let err = crate::parse("x = Type { a: str, a: int }").expect_err("duplicate field");
    let message = format!("{err}");
    assert!(message.contains("duplicate field"), "{message}");
}

#[tokio::test(flavor = "current_thread")]
async fn empty_enum_is_parse_error() {
    let err = crate::parse("x = Type { status: enum[] }").expect_err("empty enum");
    let message = format!("{err}");
    assert!(message.contains("enum"), "{message}");
}

#[tokio::test(flavor = "current_thread")]
async fn unknown_type_constructor_becomes_ref_not_error_at_parse() {
    // Unknown identifiers in type position are treated as refs; runtime
    // resolution is what errors out.
    let program = crate::parse("submit Type { x: Unknown }").expect("should parse as ref");
    assert!(matches!(program.statements.last(), Some(Stmt::Submit(_))));
}

#[tokio::test(flavor = "current_thread")]
async fn lash_type_wrapper_survives_round_trip_through_json() {
    let value = exec("submit Type { n: int }")
        .await
        .expect("should succeed");
    // to_json + from_json must preserve the Type-ness.
    let json = crate::runtime::to_json(&value);
    let recovered = crate::runtime::from_json(json);
    let schema = crate::runtime::unwrap_type_value(&recovered)
        .and_then(Value::as_record)
        .expect("round-trip must preserve wrapper");
    assert_eq!(schema["type"], Value::String("object".into()));
}
