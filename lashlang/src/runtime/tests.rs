use super::*;

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
        ExecutionOutcome::Continued => panic!("expected `finish` in test program"),
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
fn continuation_and_undefined_variable_are_reported() {
    let outcome = exec_outcome("x = 1").expect("missing finish should continue");
    assert_eq!(outcome, ExecutionOutcome::Continued);

    let value = exec("finish").expect("bare finish should succeed");
    assert_eq!(value, Value::Null);

    let err = exec("finish x").expect_err("undefined variable should fail");
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
        exec("if 1 { finish 1 } else { finish 2 }").expect("numeric truthiness should be accepted");
    assert_eq!(value, Value::Number(1.0));

    let value =
        exec("if \"\" { finish 1 } else { finish 2 }").expect("empty string should be falsy");
    assert_eq!(value, Value::Number(2.0));

    let err = exec("for x in 1 { finish x }").expect_err("non-list iteration should fail");
    assert_eq!(err, RuntimeError::NonListIteration);
}

#[test]
fn stmt_call_and_tool_results_cover_success_and_error() {
    exec("call echo { value: 1 } finish 1").expect("statement call should succeed");
    let missing = exec("bad = call missing {} finish bad").expect("missing tool should be wrapped");
    assert_eq!(
        missing.as_record().expect("result should be a record")["ok"],
        Value::Bool(false)
    );

    let value = exec("ok = call echo { value: 7 } bad = call err {} finish { ok: ok, bad: bad }")
        .expect("tool call program should succeed");
    let record = value.as_record().expect("expected record");
    assert_eq!(record["ok"].as_record().unwrap()["ok"], Value::Bool(true));
    assert_eq!(record["bad"].as_record().unwrap()["ok"], Value::Bool(false));
}

#[test]
fn field_index_unary_and_boolean_paths_are_covered() {
    let value = exec(
        r#"
        rec = { nested: { name: "lash" } }
        xs = ["a", "b"]
        ok = false and missing
        alt = true or missing
        finish [rec.nested.name, xs[1], "abc"[2], -1, not false, !false, ok, alt]
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

    let value = exec("finish true and false").expect("and path should succeed");
    assert_eq!(value, Value::Bool(false));

    let value = exec("finish false or true").expect("or path should succeed");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn field_index_and_type_errors_are_covered() {
    let err = exec("n = 1 finish n.name").expect_err("field access should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("rec = {} finish rec.name").expect("missing field should yield null");
    assert_eq!(value, Value::Null);

    let err = exec("finish 1[0]").expect_err("bad index target should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("finish [1][2]").expect("list oob should yield null");
    assert_eq!(value, Value::Null);

    let value = exec("finish \"a\"[2]").expect("string oob should yield null");
    assert_eq!(value, Value::Null);

    let err = exec("finish [1][1.5]").expect_err("fractional index should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));

    let value = exec("finish [1][-1]").expect("negative index should resolve from the end");
    assert_eq!(value, Value::Number(1.0));

    let value = exec("finish not 1").expect("not should use truthiness");
    assert_eq!(value, Value::Bool(false));

    let value = exec("finish not 0").expect("zero should be falsy");
    assert_eq!(value, Value::Bool(true));

    let value =
        exec("rec = { ok: false } finish len(rec.value.items)").expect("null chain should work");
    assert_eq!(value, Value::Number(0.0));
}

#[test]
fn arithmetic_and_compare_errors_are_covered() {
    assert_eq!(
        exec("finish 7 - 2").expect("subtract should succeed"),
        Value::Number(5.0)
    );
    assert_eq!(
        exec("finish 3 * 4").expect("multiply should succeed"),
        Value::Number(12.0)
    );
    assert_eq!(
        exec("finish 8 / 2").expect("divide should succeed"),
        Value::Number(4.0)
    );
    assert_eq!(
        exec("finish 8 % 3").expect("modulo should succeed"),
        Value::Number(2.0)
    );
    assert_eq!(
        exec("finish 1 != 2").expect("not equal should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("finish 1 <= 2").expect("less-equal should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("finish 2 > 1").expect("greater should succeed"),
        Value::Bool(true)
    );
    assert_eq!(
        exec("finish 2 >= 1").expect("greater-equal should succeed"),
        Value::Bool(true)
    );

    let value = exec("finish [1,2] + [3]").expect("list concat should succeed");
    assert_eq!(
        value,
        Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)].into())
    );

    let value = exec("finish \"a\" + \"b\"").expect("string add should succeed");
    assert_eq!(value, Value::String("ab".to_string().into()));

    let value = exec("finish \"a\" + 1").expect("string coercion should succeed");
    assert_eq!(value, Value::String("a1".to_string().into()));

    let value = exec("finish 1 + \"b\"").expect("string coercion should succeed");
    assert_eq!(value, Value::String("1b".to_string().into()));

    let value = exec("finish 1 + true").expect("bool should coerce for addition");
    assert_eq!(value, Value::Number(2.0));

    let value = exec("finish null + 2").expect("null should coerce for addition");
    assert_eq!(value, Value::Number(2.0));

    let value = exec("finish \"2\" * 3").expect("numeric strings should coerce");
    assert_eq!(value, Value::Number(6.0));

    let value = exec("finish \"2\" < 10").expect("numeric strings should compare");
    assert_eq!(value, Value::Bool(true));

    let err = exec("finish {} + 1").expect_err("records should still fail arithmetic");
    assert!(matches!(err, RuntimeError::TypeError { .. }));
}

#[test]
fn builtin_success_matrix_is_covered() {
    let value = exec(
        r#"
        rec = { a: 1, b: 2 }
        finish {
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
          fmt: format("x={},y={}", 1, true)
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
}

#[test]
fn builtin_error_matrix_is_covered() {
    let cases = [
        ("finish len(true)", "len"),
        ("finish empty(true)", "empty"),
        ("finish keys([])", "keys"),
        ("finish values([])", "values"),
        ("finish contains(1, 2)", "contains"),
        ("finish starts_with({}, \"a\")", "starts_with"),
        ("finish ends_with({}, \"a\")", "ends_with"),
        ("finish split({}, \",\")", "split"),
        ("finish join(1, \",\")", "join"),
        ("finish trim({})", "trim"),
        ("finish slice(1, 0, 1)", "slice"),
        ("finish to_int({})", "to_int"),
        ("finish to_int(\"x\")", "to_int"),
        ("finish to_float({})", "to_float"),
        ("finish to_float(\"x\")", "to_float"),
        ("finish json_parse(\"{\")", "json_parse"),
        ("finish format()", "format"),
        ("finish format({})", "format"),
        ("finish format(\"{1}\", \"x\")", "format"),
        ("finish format(\"{}\", \"x\", \"y\")", "format"),
        ("finish format(\"{} {1}\", \"x\", \"y\")", "format"),
        ("finish format(\"{x}\")", "format"),
        ("finish format(\"{\")", "format"),
        ("finish format(\"}\")", "format"),
        ("finish no_such_builtin()", "no_such_builtin"),
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

    let err = exec("finish len()").expect_err("arity error should fail");
    assert!(matches!(err, RuntimeError::TypeError { .. }));
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
        finish out
        "#,
    )
    .expect("else branch should succeed");
    assert_eq!(value, Value::Number(2.0));

    let value = exec(
        r#"
        for x in [1, 2] {
          finish x
        }
        finish 0
        "#,
    )
    .expect("finish inside loop should bubble out");
    assert_eq!(value, Value::Number(1.0));
}

#[test]
fn parallel_branch_panics_are_reported_as_runtime_errors() {
    let err = exec(
        r#"
        parallel {
          crash = call panic {}
        }
        finish 1
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
