use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Host;

impl ToolHost for Host {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        match name {
            "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
            "err" => Err(ToolHostError::new("boom")),
            "panic" => panic!("boom"),
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }
}

fn exec(source: &str) -> Result<Value, RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    match execute_program(&program, &mut state, &Host)? {
        ExecutionOutcome::Finished(value) => Ok(value),
        ExecutionOutcome::Continued => panic!("expected `submit` in test program"),
    }
}

fn exec_outcome(source: &str) -> Result<ExecutionOutcome, RuntimeError> {
    let program = crate::parse(source).expect("program should parse");
    let mut state = State::new();
    execute_program(&program, &mut state, &Host)
}

#[test]
fn value_helpers_and_display_cover_all_variants() {
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

#[test]
fn compiler_folds_constant_list_and_record_literals() {
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
    let outcome = execute_compiled(&compiled, &mut state, &Host).expect("program should run");
    let ExecutionOutcome::Finished(Value::List(items)) = outcome else {
        panic!("expected folded list result");
    };
    assert_eq!(items.len(), 2);
}

#[test]
fn compiler_propagates_safe_straight_line_constants() {
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
    let outcome = execute_compiled(&compiled, &mut state, &Host).expect("program should run");
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

#[test]
fn constant_propagation_does_not_cross_control_flow_boundaries() {
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
    .expect("program should succeed");

    assert_eq!(value, Value::Number(2.0));
}

#[test]
fn reusable_execution_scratch_preserves_results_across_runs() {
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
            .expect("program should run");
        assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(6.0)));
    }
}

#[test]
fn continuation_and_undefined_variable_are_reported() {
    let outcome = exec_outcome("x = 1").expect("missing submit should continue");
    assert_eq!(outcome, ExecutionOutcome::Continued);

    let value = exec("submit").expect("bare submit should succeed");
    assert_eq!(value, Value::Null);

    let err = exec("submit x").expect_err("undefined variable should fail");
    assert_eq!(
        err,
        RuntimeError::UndefinedVariable {
            name: "x".to_string()
        }
    );
}

#[test]
fn condition_and_iteration_errors_are_reported() {
    let value =
        exec("if 1 { submit 1 } else { submit 2 }").expect("numeric truthiness should be accepted");
    assert_eq!(value, Value::Number(1.0));

    let value =
        exec("if \"\" { submit 1 } else { submit 2 }").expect("empty string should be falsy");
    assert_eq!(value, Value::Number(2.0));

    let err = exec("for x in 1 { submit x }").expect_err("non-list iteration should fail");
    assert_eq!(err, RuntimeError::NonListIteration);
}

#[test]
fn stmt_call_and_tool_results_cover_success_and_error() {
    exec("call echo { value: 1 } submit 1").expect("statement call should succeed");
    let missing = exec("bad = call missing {} submit bad").expect("missing tool should be wrapped");
    assert_eq!(
        missing.as_record().expect("result should be a record")["ok"],
        Value::Bool(false)
    );

    let value = exec("ok = call echo { value: 7 } bad = call err {} submit { ok: ok, bad: bad }")
        .expect("tool call program should succeed");
    let record = value.as_record().expect("expected record");
    assert_eq!(record["ok"].as_record().unwrap()["ok"], Value::Bool(true));
    assert_eq!(record["bad"].as_record().unwrap()["ok"], Value::Bool(false));
}

#[test]
fn result_unwrap_extracts_success_and_preserves_manual_handling() {
    let value = exec("submit (call echo { value: 7 })?").expect("unwrap should succeed");
    assert_eq!(value, Value::Number(7.0));

    let value = exec(
        r#"
        result = call err {}
        submit result.ok ? result.error : "unexpected"
        "#,
    )
    .expect("manual wrapper handling should still work");
    assert_eq!(value, Value::String("unexpected".into()));
}

#[test]
fn direct_tool_call_unwrap_skips_observable_wrapper() {
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
    let outcome = execute_compiled(&compiled, &mut state, &Host).expect("program should run");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(7.0)));

    let err = exec("submit (call err {})?").expect_err("failed unwrap should abort");
    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "`?` unwrapped failed tool result: boom".to_string(),
        }
    );
}

#[test]
fn result_unwrap_reports_failed_and_malformed_wrappers() {
    let err = exec("submit (call err {})?").expect_err("failed tool unwrap should abort");
    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "`?` unwrapped failed tool result: boom".to_string(),
        }
    );

    let err = exec("submit 1?").expect_err("non-wrapper should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`?` expected a tool result wrapper, got number".to_string(),
        }
    );

    let err = exec("submit { ok: true }?").expect_err("missing value should fail");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`?` found a successful tool result wrapper missing `value`".to_string(),
        }
    );
}

#[test]
fn field_index_unary_and_boolean_paths_are_covered() {
    let value = exec(
        r#"
        rec = { nested: { name: "lash" } }
        xs = ["a", "b"]
        ok = false and missing
        alt = true or missing
        submit [rec.nested.name, xs[1], "abc"[2], -1, not false, !false, ok, alt]
        "#,
    )
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

    let value = exec("submit true and false").expect("and path should succeed");
    assert_eq!(value, Value::Bool(false));

    let value = exec("submit false or true").expect("or path should succeed");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn field_index_and_type_errors_are_covered() {
    let err = exec("n = 1 submit n.name").expect_err("field access should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("rec = {} submit rec.name").expect("missing field should yield null");
    assert_eq!(value, Value::Null);

    let err = exec("submit 1[0]").expect_err("bad index target should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("submit [1][2]").expect("list oob should yield null");
    assert_eq!(value, Value::Null);

    let value = exec("submit \"a\"[2]").expect("string oob should yield null");
    assert_eq!(value, Value::Null);

    let err = exec("submit [1][1.5]").expect_err("fractional index should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("submit [1][-1]").expect("negative index should resolve from the end");
    assert_eq!(value, Value::Number(1.0));

    let value = exec("submit not 1").expect("not should use truthiness");
    assert_eq!(value, Value::Bool(false));

    let value = exec("submit not 0").expect("zero should be falsy");
    assert_eq!(value, Value::Bool(true));

    let value =
        exec("rec = { ok: false } submit len(rec.value.items)").expect("null chain should work");
    assert_eq!(value, Value::Number(0.0));
}

#[test]
fn arithmetic_and_compare_errors_are_covered() {
    assert_eq!(
        exec("submit 7 - 2").expect("subtract should succeed"),
        Value::Number(5.0)
    );
    assert_eq!(
        exec("submit 3 * 4").expect("multiply should succeed"),
        Value::Number(12.0)
    );
    assert_eq!(
        exec("submit 8 / 2").expect("divide should succeed"),
        Value::Number(4.0)
    );
    assert_eq!(
        exec("submit 8 % 3").expect("modulo should succeed"),
        Value::Number(2.0)
    );
    assert_eq!(
        exec("submit 1 != 2").expect("not equal should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 1 <= 2").expect("less-equal should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 2 > 1").expect("greater should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("submit 2 >= 1").expect("greater-equal should succeed"),
        Value::Bool(true)
    );

    let value = exec("submit [1,2] + [3]").expect("list concat should succeed");
    assert_eq!(
        value,
        Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)].into())
    );

    let value = exec("submit \"a\" + \"b\"").expect("string add should succeed");
    assert_eq!(value, Value::String("ab".to_string().into()));

    let value = exec("submit \"a\" + 1").expect("string coercion should succeed");
    assert_eq!(value, Value::String("a1".to_string().into()));

    let value = exec("submit 1 + \"b\"").expect("string coercion should succeed");
    assert_eq!(value, Value::String("1b".to_string().into()));

    let value = exec("submit 1 + true").expect("bool should coerce for addition");
    assert_eq!(value, Value::Number(2.0));

    let value = exec("submit null + 2").expect("null should coerce for addition");
    assert_eq!(value, Value::Number(2.0));

    let value = exec("submit \"2\" * 3").expect("numeric strings should coerce");
    assert_eq!(value, Value::Number(6.0));

    let value = exec("submit \"2\" < 10").expect("numeric strings should compare");
    assert_eq!(value, Value::Bool(true));

    let err = exec("submit {} + 1").expect_err("records should still fail arithmetic");
    assert!(matches!(err, RuntimeError::TypeError { .. }));
}

#[test]
fn builtin_success_matrix_is_covered() {
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

#[test]
fn builtin_error_matrix_is_covered() {
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
        let err = exec(source).expect_err("builtin should fail");
        assert!(matches!(
            err,
            RuntimeError::TypeError { .. }
                | RuntimeError::ValueError { .. }
                | RuntimeError::UnknownBuiltin { .. }
        ));
    }

    let err = exec("submit len()").expect_err("arity error should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));
}

#[test]
fn validate_reports_precise_shape_errors() {
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
        let err = exec(source).expect_err("validate should reject bad value");
        assert_eq!(
            err,
            RuntimeError::ValueError {
                message: expected.to_string()
            }
        );
    }

    let err = exec("submit validate({ name: \"pkg\" }, { type: \"object\" })")
        .expect_err("raw schema records should be rejected");
    assert_eq!(
        err,
        RuntimeError::TypeError {
            message: "`validate` requires a Type literal as the second argument".to_string()
        }
    );
}

#[test]
fn validate_union_accepts_any_variant() {
    // `str | null` must accept both a string and a null.
    let out = exec(r#"submit validate({ email: "a@b" }, Type { email: str | null })"#)
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
        .expect("null-branch validate should succeed");
    let Value::Record(rec) = &out else {
        panic!("expected record");
    };
    assert!(matches!(rec.get("email"), Some(Value::Null)));
}

#[test]
fn validate_union_rejects_value_matching_no_variant() {
    let err = exec(r#"submit validate({ email: 42 }, Type { email: str | null })"#)
        .expect_err("number should not match str | null");
    let RuntimeError::ValueError { message } = err else {
        panic!("expected ValueError");
    };
    assert!(
        message.contains("$.email"),
        "error should point at the failing field: {message}",
    );
}

#[test]
fn helper_functions_are_covered_directly() {
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

#[test]
fn json_helpers_cover_special_paths() {
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

#[test]
fn false_if_branch_and_finish_inside_loop_are_covered() {
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
    .expect("submit inside loop should bubble out");
    assert_eq!(value, Value::Number(1.0));
}

#[test]
fn parallel_branch_panics_are_reported_as_runtime_errors() {
    let err = exec(
        r#"
        parallel {
          crash = call panic {}
        }
        submit 1
        "#,
    )
    .expect_err("parallel panic should be reported");

    assert_eq!(
        err,
        RuntimeError::ValueError {
            message: "parallel branch panicked".to_string()
        }
    );
}

#[test]
fn parallel_tool_calls_use_host_batch_when_available() {
    struct BatchHost {
        calls: AtomicUsize,
        batches: AtomicUsize,
    }

    impl ToolHost for BatchHost {
        fn call(&self, _name: &str, _args: &Record) -> Result<Value, ToolHostError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Err(ToolHostError::new("single call should not be used"))
        }

        fn call_batch(
            &self,
            calls: &[(&str, &Record)],
            push_result: &mut dyn FnMut(Result<Value, ToolHostError>),
        ) -> bool {
            self.batches.fetch_add(1, Ordering::Relaxed);
            for (name, args) in calls {
                push_result(match *name {
                    "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
                    other => Err(ToolHostError::new(format!("unknown tool: {other}"))),
                });
            }
            true
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
    let outcome = execute_program(&program, &mut state, &host).expect("program should run");

    assert_eq!(
        outcome,
        ExecutionOutcome::Finished(Value::List(
            vec![Value::String("a".into()), Value::String("b".into())].into()
        ))
    );
    assert_eq!(host.calls.load(Ordering::Relaxed), 0);
    assert_eq!(host.batches.load(Ordering::Relaxed), 1);
}

#[test]
fn truthiness_covers_scalar_and_container_values() {
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
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        Host.call(name, args)
    }

    fn start_call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        let mut record = Record::default();
        record.insert("__handle__".to_string(), Value::String("task".into()));
        record.insert("tool".to_string(), Value::String(name.to_string().into()));
        record.insert(
            "value".to_string(),
            args.get("value").cloned().unwrap_or(Value::Null),
        );
        Ok(Value::Record(Arc::new(record)))
    }

    fn await_handle(&self, handle: &Value) -> Result<Value, ToolHostError> {
        let record = handle
            .as_record()
            .ok_or_else(|| ToolHostError::new("expected handle record"))?;
        Ok(record.get("value").cloned().unwrap_or(Value::Null))
    }

    fn cancel_handle(&self, _handle: &Value) -> Result<Value, ToolHostError> {
        Ok(Value::Null)
    }
}

#[test]
fn async_tool_handles_can_be_started_awaited_and_cancelled() {
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
    let outcome = execute_program(&program, &mut state, &AsyncHost).expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let record = value
        .as_record()
        .expect("await should return wrapped result");
    assert_eq!(record["ok"], Value::Bool(true));
    assert_eq!(record["value"], Value::String("done".into()));
}

#[test]
fn await_unknown_handle_surfaces_runtime_error() {
    let program = crate::parse(
        r#"
        result = await 1
        submit result
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost).expect("program should run");
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

#[test]
fn await_list_of_handles_returns_results_in_order() {
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
    let outcome = execute_program(&program, &mut state, &AsyncHost).expect("program should run");
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

#[test]
fn await_list_preserves_per_item_errors() {
    let program = crate::parse(
        r#"
        handles = [start call echo { value: "done" }, 1]
        results = await handles
        submit results
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost).expect("program should run");
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

#[test]
fn await_record_of_handles_returns_record_of_wrappers() {
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
    let outcome = execute_program(&program, &mut state, &AsyncHost).expect("program should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    assert_eq!(
        value,
        Value::List(vec![Value::String("one".into()), Value::String("two".into())].into())
    );
}

#[test]
fn result_unwrap_extracts_awaited_handles_and_parallel_results() {
    let program = crate::parse(
        r#"
        handle = start call echo { value: "done" }
        result = (await handle)?
        submit result
        "#,
    )
    .expect("program should parse");
    let mut state = State::new();
    let outcome = execute_program(&program, &mut state, &AsyncHost).expect("program should run");
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
    let outcome = execute_program(&program, &mut state, &AsyncHost).expect("program should run");
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

#[test]
fn type_scalar_schemas_const_fold_to_json_schema() {
    for (src, expected) in [
        ("submit Type { v: str }", "string"),
        ("submit Type { v: int }", "integer"),
        ("submit Type { v: float }", "number"),
        ("submit Type { v: bool }", "boolean"),
        ("submit Type { v: dict }", "object"),
    ] {
        let value = exec(src).expect("should succeed");
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

#[test]
fn type_any_is_empty_schema() {
    let value = exec("submit Type { v: any }").expect("should succeed");
    let schema = unwrap_schema(&value);
    let props = schema["properties"].as_record().expect("properties");
    let v = props["v"].as_record().expect("field schema");
    assert!(v.is_empty(), "any must be an empty JSON Schema");
}

#[test]
fn type_enum_produces_string_with_enum_array() {
    let value =
        exec(r#"submit Type { status: enum["ok", "err", "pending"] }"#).expect("should succeed");
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

#[test]
fn type_list_schema_wraps_inner_type_as_items() {
    let value = exec("submit Type { tags: list[str] }").expect("should succeed");
    let schema = unwrap_schema(&value);
    let tags = schema["properties"].as_record().unwrap()["tags"]
        .as_record()
        .expect("list field schema");
    assert_eq!(tags["type"], Value::String("array".into()));
    let items = tags["items"].as_record().expect("items schema");
    assert_eq!(items["type"], Value::String("string".into()));
}

#[test]
fn type_list_of_enum_preserves_nested_shape() {
    let value = exec(r#"submit Type { labels: list[enum["a", "b"]] }"#).expect("should succeed");
    let schema = unwrap_schema(&value);
    let labels = schema["properties"].as_record().unwrap()["labels"]
        .as_record()
        .expect("list schema");
    let items = labels["items"].as_record().expect("enum item schema");
    assert_eq!(items["type"], Value::String("string".into()));
    assert!(matches!(items["enum"], Value::List(_)));
}

#[test]
fn type_nested_object_is_full_subschema() {
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

#[test]
fn type_optional_field_drops_from_required() {
    let value = exec("submit Type { a: str, b: int? }").expect("should succeed");
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

#[test]
fn type_ref_resolves_previously_defined_type() {
    let src = r#"
        Inner = Type { count: int }
        Outer = Type { name: str, nested: Inner }
        submit Outer
    "#;
    let value = exec(src).expect("should succeed");
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

#[test]
fn type_ref_to_non_type_value_is_type_error() {
    let err = exec(
        r#"
        Inner = { count: 5 }
        Outer = Type { nested: Inner }
        submit Outer
        "#,
    )
    .expect_err("should fail: Inner is not a Type value");
    assert!(
        matches!(err, RuntimeError::TypeError { .. }),
        "expected TypeError, got {err:?}"
    );
}

#[test]
fn type_ref_with_undefined_name_is_undefined_variable() {
    let err = exec("submit Type { nested: MissingType }").expect_err("unknown ref should fail");
    assert_eq!(
        err,
        RuntimeError::UndefinedVariable {
            name: "MissingType".to_string()
        }
    );
}

#[test]
fn compile_stats_count_const_folded_and_dynamic_literals() {
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

#[test]
fn profile_report_surfaces_resolve_type_ref_counts() {
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
    let (_outcome, report) =
        crate::profile_compiled(&compiled, &mut state, &Host).expect("profile should succeed");
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

#[test]
fn type_literal_inside_tool_call_args_passes_through_as_record() {
    struct CaptureHost {
        captured: std::sync::Mutex<Option<Value>>,
    }
    impl ToolHost for CaptureHost {
        fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
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
    crate::execute_program(&program, &mut state, &host).expect("should run");

    let captured = host.captured.lock().unwrap().clone().expect("captured");
    let inner = crate::runtime::unwrap_type_value(&captured).expect("has $lash_type");
    let schema = inner.as_record().expect("schema record");
    assert_eq!(schema["type"], Value::String("object".into()));
}

#[test]
fn duplicate_field_name_is_parse_error() {
    let err = crate::parse("x = Type { a: str, a: int }").expect_err("duplicate field");
    let message = format!("{err}");
    assert!(message.contains("duplicate field"), "{message}");
}

#[test]
fn empty_enum_is_parse_error() {
    let err = crate::parse("x = Type { status: enum[] }").expect_err("empty enum");
    let message = format!("{err}");
    assert!(message.contains("enum"), "{message}");
}

#[test]
fn unknown_type_constructor_becomes_ref_not_error_at_parse() {
    // Unknown identifiers in type position are treated as refs; runtime
    // resolution is what errors out.
    let program = crate::parse("submit Type { x: Unknown }").expect("should parse as ref");
    assert!(matches!(program.statements.last(), Some(Stmt::Submit(_))));
}

#[test]
fn lash_type_wrapper_survives_round_trip_through_json() {
    let value = exec("submit Type { n: int }").expect("should succeed");
    // to_json + from_json must preserve the Type-ness.
    let json = crate::runtime::to_json(&value);
    let recovered = crate::runtime::from_json(json);
    let schema = crate::runtime::unwrap_type_value(&recovered)
        .and_then(Value::as_record)
        .expect("round-trip must preserve wrapper");
    assert_eq!(schema["type"], Value::String("object".into()));
}
