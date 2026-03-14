use lashlang::{
    ExecuteError, ExecutionOutcome, Record, RuntimeError, State, ToolHost, ToolHostError, Value,
    execute, parse,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
struct TestHost {
    files: HashMap<String, String>,
    globs: HashMap<String, Vec<String>>,
    observations: std::sync::Mutex<Vec<Value>>,
    active: AtomicUsize,
    max_active: AtomicUsize,
}

impl TestHost {
    fn with_file(mut self, path: &str, content: &str) -> Self {
        self.files.insert(path.to_string(), content.to_string());
        self
    }
}

impl ToolHost for TestHost {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        match name {
            "read_file" => {
                let path = expect_string(args, "path")?;
                match self.files.get(path) {
                    Some(content) => Ok(Value::String(content.clone().into())),
                    None => Err(ToolHostError::new(format!("missing file: {path}"))),
                }
            }
            "glob" => {
                let pattern = expect_string(args, "pattern")?;
                let values: Vec<_> = self
                    .globs
                    .get(pattern)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|value| Value::String(value.into()))
                    .collect();
                Ok(Value::List(values.into()))
            }
            "sleep_echo" => {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                loop {
                    let max = self.max_active.load(Ordering::SeqCst);
                    if active <= max {
                        break;
                    }
                    if self
                        .max_active
                        .compare_exchange(max, active, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                    {
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
                self.active.fetch_sub(1, Ordering::SeqCst);
                Ok(Value::String(
                    expect_string(args, "value")?.to_string().into(),
                ))
            }
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }

    fn observe(&self, value: &Value) -> Result<(), ToolHostError> {
        self.observations
            .lock()
            .expect("observation mutex")
            .push(value.clone());
        Ok(())
    }
}

fn finished(outcome: ExecutionOutcome) -> Value {
    match outcome {
        ExecutionOutcome::Finished(value) => value,
        ExecutionOutcome::Continued => panic!("expected `finish`"),
    }
}

fn runtime_error(source: &str) -> RuntimeError {
    let host = TestHost::default();
    let mut state = State::new();
    match execute(source, &mut state, &host).expect_err("execution should fail") {
        ExecuteError::Runtime(error) => error,
        ExecuteError::Parse(error) => panic!("expected runtime error, got parse error: {error:?}"),
    }
}

#[test]
fn parser_handles_precedence_and_parallel() {
    let program = parse(
        r#"
        total = 1 + 2 * 3
        parallel {
          left = call glob { pattern: "src/*.rs" }
          right = call read_file { path: "src/lib.rs" }
        }
        finish total
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 3);
}

#[test]
fn parser_accepts_double_slash_comments() {
    let program = parse(
        r#"
        // setup
        total = 1 + 2
        // finish
        finish total
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[test]
fn parser_accepts_comment_only_program() {
    let program = parse(
        r#"
        // comment one
        // comment two
        "#,
    )
    .expect("program should parse");

    assert!(program.statements.is_empty());
}

#[test]
fn parser_accepts_inline_trailing_comments_in_blocks() {
    let program = parse(
        r#"
        if true { // enter block
          value = 1 // assign
        } else { // fallback
          value = 2
        }
        finish value // done
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[test]
fn parser_accepts_else_if_chains() {
    let program = parse(
        r#"
        if false {
          answer = 1
        } else if true {
          answer = 2
        } else {
          answer = 3
        }
        finish answer
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[test]
fn parser_allows_parallel_in_expression_position() {
    let program = parse(
        r#"
        results = parallel {
          left = call glob { pattern: "src/*.rs" }
          right = call read_file { path: "src/lib.rs" }
        }
        finish results
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[test]
fn parser_allows_bare_expression_statements() {
    let program = parse(
        r#"
        "branch_a"
        finish "done"
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
    assert!(matches!(program.statements[0], lashlang::Stmt::Expr(_)));
}

#[test]
fn executes_programs_with_double_slash_comments() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        // Create some values first
        total = 6 / 2
        // Return the result
        finish total
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::Number(3.0));
}

#[test]
fn executes_inline_trailing_comments_inside_blocks() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        if true { // choose this branch
          total = 1 + 2 // add
        } else {
          total = 0
        }
        finish total // final answer
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::Number(3.0));
}

#[test]
fn double_slash_inside_strings_is_not_a_comment() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        url = "https://example.com/a//b"
        finish url
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::String("https://example.com/a//b".to_string().into())
    );
}

#[test]
fn parser_accepts_ternary_in_call_arguments() {
    let program = parse(
        r#"
        result = format("{}", true ? "yes" : "no")
        finish result
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[test]
fn executes_arithmetic_strings_and_finish() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        total = 1 + 2 * 3
        msg = format("total={}", total)
        finish msg
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("total=7".to_string().into()));
    assert_eq!(state.globals()["total"], Value::Number(7.0));
}

#[test]
fn executes_if_for_and_list_concat() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        nums = [1, 2, 3, 4]
        sum = 0
        labels = []
        for n in nums {
          sum = sum + n
          labels = labels + [format("n={}", n)]
        }
        if sum == 10 {
          result = join(labels, ",")
        } else {
          result = "bad"
        }
        finish result
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("n=1,n=2,n=3,n=4".to_string().into()));
}

#[test]
fn ternary_selects_the_correct_branch() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        truthy = true ? "left" : "right"
        falsy = false ? "left" : "right"
        finish format("{}:{}", truthy, falsy)
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("left:right".to_string().into()));
}

#[test]
fn ternary_is_right_associative() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        result = false ? 1 : true ? 2 : 3
        finish result
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::Number(2.0));
}

#[test]
fn ternary_has_lower_precedence_than_boolean_ops() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        result = false or true ? "yes" : "no"
        finish result
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("yes".to_string().into()));
}

#[test]
fn ternary_short_circuits_unselected_branch() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        yes = true ? "ok" : missing_name
        no = false ? missing_name : "ok"
        finish format("{}:{}", yes, no)
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("ok:ok".to_string().into()));
}

#[test]
fn unary_bang_aliases_not() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        a = !false
        b = !true
        finish [a, b]
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::List(vec![Value::Bool(true), Value::Bool(false)].into())
    );
}

#[test]
fn symbolic_boolean_aliases_match_word_operators() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        a = true && false
        b = false || true
        c = !false && (false || true)
        finish [a, b, c]
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::List(vec![Value::Bool(false), Value::Bool(true), Value::Bool(true)].into())
    );
}

#[test]
fn conditions_and_ternary_use_bounded_truthiness() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        a = 1 ? "yes" : "no"
        b = "" ? "yes" : "no"
        c = !0
        d = ![]
        finish [a, b, c, d]
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::String("yes".to_string().into()),
                Value::String("no".to_string().into()),
                Value::Bool(true),
                Value::Bool(false),
            ]
            .into()
        )
    );
}

#[test]
fn string_concatenation_stringifies_non_string_side() {
    let host = TestHost::default().with_file("src/lib.rs", "pub fn main() {}");
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        found = call read_file { path: "src/lib.rs" }
        finish "status=" + found.ok + " value=" + found.value
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::String("status=true value=pub fn main() {}".to_string().into())
    );
}

#[test]
fn arithmetic_and_string_builtins_coerce_scalars() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        total = true + 2
        scaled = "3" * 2
        joined = join(["a", 2, true], "-")
        split_num = split(101, 0)
        prefix = starts_with(123, 12)
        finish {
          total: total,
          scaled: scaled,
          joined: joined,
          split_num: split_num,
          prefix: prefix
        }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let record = value.as_record().expect("expected record");
    assert_eq!(record["total"], Value::Number(3.0));
    assert_eq!(record["scaled"], Value::Number(6.0));
    assert_eq!(
        record["joined"],
        Value::String("a-2-true".to_string().into())
    );
    assert_eq!(
        record["split_num"],
        Value::List(
            vec![
                Value::String("1".to_string().into()),
                Value::String("1".to_string().into())
            ]
            .into()
        )
    );
    assert_eq!(record["prefix"], Value::Bool(true));
}

#[test]
fn to_string_stringifies_records() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        finish to_string({ ok: true, count: 2 })
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::String("{\"count\":2,\"ok\":true}".to_string().into())
    );
}

#[test]
fn observe_captures_intermediate_values_without_ending_execution() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        item = { ok: true, count: 2 }
        observe item
        observe "step done"
        finish "final"
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("final".to_string().into()));
    let observed = host.observations.lock().expect("observation mutex");
    assert_eq!(observed.len(), 2);
    assert_eq!(
        observed[0],
        Value::Record({
            let mut record = Record::default();
            record.insert("ok".to_string(), Value::Bool(true));
            record.insert("count".to_string(), Value::Number(2.0));
            record.into()
        })
    );
    assert_eq!(observed[1], Value::String("step done".to_string().into()));
}

#[test]
fn execution_can_continue_without_finish() {
    let host = TestHost::default();
    let mut state = State::new();

    let outcome = execute(
        r#"
        counter = 1
        observe counter
        "#,
        &mut state,
        &host,
    )
    .expect("execution should succeed");

    assert_eq!(outcome, ExecutionOutcome::Continued);
    assert_eq!(state.globals()["counter"], Value::Number(1.0));
    let observed = host.observations.lock().expect("observation mutex");
    assert_eq!(observed.as_slice(), &[Value::Number(1.0)]);
}

#[test]
fn ternary_fixes_tool_result_formatting_pattern() {
    let host = TestHost::default().with_file("src/lib.rs", "pub fn main() {}");
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        found = call read_file { path: "src/lib.rs" }
        missing = call read_file { path: "src/missing.rs" }
        summary = format(
          "found={} missing={}",
          found.ok ? "ok" : format("failed: {}", found.error),
          missing.ok ? "ok" : format("failed: {}", missing.error)
        )
        finish summary
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::String(text) = value else {
        panic!("expected string");
    };
    assert!(text.contains("found=ok"));
    assert!(text.contains("missing=failed:"));
}

#[test]
fn format_supports_indexed_reordering() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        finish format("b={1} a={0}", "x", "y")
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("b=y a=x".to_string().into()));
}

#[test]
fn format_without_placeholders_returns_literal_string() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        finish format("plain")
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("plain".to_string().into()));
}

#[test]
fn format_supports_escaped_braces() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        finish format("{{{}}}", 1)
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("{1}".to_string().into()));
}

#[test]
fn format_rejects_mixed_placeholder_styles_end_to_end() {
    let error = runtime_error(
        r#"
        finish format("{} {1}", "x", "y")
        "#,
    );

    assert_eq!(
        error,
        RuntimeError::ValueError {
            message: "can't mix `{}` and indexed format placeholders".to_string()
        }
    );
}

#[test]
fn format_rejects_unused_args_end_to_end() {
    let error = runtime_error(
        r#"
        finish format("plain", 1)
        "#,
    );

    assert_eq!(
        error,
        RuntimeError::ValueError {
            message: "format argument `0` is unused".to_string()
        }
    );
}

#[test]
fn format_rejects_unmatched_braces_end_to_end() {
    let open_error = runtime_error(
        r#"
        finish format("{")
        "#,
    );
    assert_eq!(
        open_error,
        RuntimeError::ValueError {
            message: "unmatched `{` in format string".to_string()
        }
    );

    let close_error = runtime_error(
        r#"
        finish format("}")
        "#,
    );
    assert_eq!(
        close_error,
        RuntimeError::ValueError {
            message: "unmatched `}` in format string".to_string()
        }
    );
}

#[test]
fn tool_calls_return_result_records() {
    let host = TestHost::default().with_file("src/lib.rs", "pub fn main() {}");
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        found = call read_file { path: "src/lib.rs" }
        missing = call read_file { path: "src/missing.rs" }
        finish { found: found, missing: missing }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(
        record["found"].as_record().unwrap()["ok"],
        Value::Bool(true)
    );
    assert_eq!(
        record["missing"].as_record().unwrap()["ok"],
        Value::Bool(false)
    );
}

#[test]
fn parallel_executes_concurrently_and_merges_distinct_bindings() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        parallel {
          left = call sleep_echo { value: "a" }
          right = call sleep_echo { value: "b" }
        }
        finish { left: left, right: right }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(
        record["left"].as_record().unwrap()["value"],
        Value::String("a".to_string().into())
    );
    assert_eq!(
        record["right"].as_record().unwrap()["value"],
        Value::String("b".to_string().into())
    );
    assert!(host.max_active.load(Ordering::SeqCst) >= 2);
}

#[test]
fn parallel_expression_returns_branch_results_in_order() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        results = parallel {
          left = call sleep_echo { value: "a" }
          right = call sleep_echo { value: "b" }
        }
        finish { results: results, left: left, right: right }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    let Value::List(results) = &record["results"] else {
        panic!("expected result list");
    };
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].as_record().unwrap()["value"],
        Value::String("a".to_string().into())
    );
    assert_eq!(
        results[1].as_record().unwrap()["value"],
        Value::String("b".to_string().into())
    );
    assert_eq!(
        record["left"].as_record().unwrap()["value"],
        Value::String("a".to_string().into())
    );
    assert_eq!(
        record["right"].as_record().unwrap()["value"],
        Value::String("b".to_string().into())
    );
    assert!(host.max_active.load(Ordering::SeqCst) >= 2);
}

#[test]
fn parallel_expression_accepts_bare_expression_branches() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        results = parallel {
          "branch_a"
          40 + 2
          len([1,2,3])
        }
        finish results
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::String("branch_a".to_string().into()),
                Value::Number(42.0),
                Value::Number(3.0),
            ]
            .into()
        )
    );
}

#[test]
fn slice_null_bounds_default_to_start_or_end() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        values = [10, 20, 30, 40, 50]
        finish {
          list_tail: slice(values, 3, null),
          list_head: slice(values, null, 2),
          string_tail: slice("abcdef", 4, null),
          string_head: slice("abcdef", null, 2)
        }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(
        record["list_tail"],
        Value::List(vec![Value::Number(40.0), Value::Number(50.0)].into())
    );
    assert_eq!(
        record["list_head"],
        Value::List(vec![Value::Number(10.0), Value::Number(20.0)].into())
    );
    assert_eq!(
        record["string_tail"],
        Value::String("ef".to_string().into())
    );
    assert_eq!(
        record["string_head"],
        Value::String("ab".to_string().into())
    );
}

#[test]
fn negative_indices_and_record_contains_are_supported() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        values = [10, 20, 30]
        text = "abc"
        finish {
          tail: values[-1],
          before_tail: values[-2],
          oob: values[-4],
          last_char: text[-1],
          record_has_key: contains({ foo: 1, bar: 2 }, "foo"),
          record_missing_key: contains({ foo: 1, bar: 2 }, "baz")
        }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["tail"], Value::Number(30.0));
    assert_eq!(record["before_tail"], Value::Number(20.0));
    assert_eq!(record["oob"], Value::Null);
    assert_eq!(record["last_char"], Value::String("c".to_string().into()));
    assert_eq!(record["record_has_key"], Value::Bool(true));
    assert_eq!(record["record_missing_key"], Value::Bool(false));
}

#[test]
fn else_if_chains_execute_without_extra_braces() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        score = 7
        if score > 10 {
          label = "large"
        } else if score > 5 {
          label = "medium"
        } else {
          label = "small"
        }
        finish label
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("medium".to_string().into()));
}

#[test]
fn slice_supports_negative_bounds() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        values = [10, 20, 30, 40, 50]
        finish {
          list_tail: slice(values, -2, null),
          list_without_last: slice(values, null, -1),
          list_middle: slice(values, -4, -1),
          string_tail: slice("abcdef", -2, null),
          string_without_last: slice("abcdef", null, -1),
          string_middle: slice("abcdef", -5, -2)
        }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(
        record["list_tail"],
        Value::List(vec![Value::Number(40.0), Value::Number(50.0)].into())
    );
    assert_eq!(
        record["list_without_last"],
        Value::List(
            vec![
                Value::Number(10.0),
                Value::Number(20.0),
                Value::Number(30.0),
                Value::Number(40.0),
            ]
            .into()
        )
    );
    assert_eq!(
        record["list_middle"],
        Value::List(
            vec![
                Value::Number(20.0),
                Value::Number(30.0),
                Value::Number(40.0),
            ]
            .into()
        )
    );
    assert_eq!(
        record["string_tail"],
        Value::String("ef".to_string().into())
    );
    assert_eq!(
        record["string_without_last"],
        Value::String("abcde".to_string().into())
    );
    assert_eq!(
        record["string_middle"],
        Value::String("bcd".to_string().into())
    );
}

#[test]
fn string_comparisons_are_lexicographic() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        finish {
          lt: "abc" < "def",
          gt: "xyz" > "abc",
          le: "abc" <= "abc",
          ge: "xyz" >= "abc"
        }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["lt"], Value::Bool(true));
    assert_eq!(record["gt"], Value::Bool(true));
    assert_eq!(record["le"], Value::Bool(true));
    assert_eq!(record["ge"], Value::Bool(true));
}

#[test]
fn stringification_preserves_integer_format_inside_containers() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        finish {
          list_text: to_string([1, 2]),
          record_text: to_string({ a: 1, b: 2.5 })
        }
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(
        record["list_text"],
        Value::String("[1,2]".to_string().into())
    );
    assert_eq!(
        record["record_text"],
        Value::String("{\"a\":1,\"b\":2.5}".to_string().into())
    );
}

#[test]
fn parallel_rejects_conflicting_assignments() {
    let host = TestHost::default();
    let mut state = State::new();

    let error = execute(
        r#"
        parallel {
          result = call sleep_echo { value: "a" }
          result = call sleep_echo { value: "b" }
        }
        finish result
        "#,
        &mut state,
        &host,
    )
    .expect_err("execution should fail");

    assert!(matches!(
        error,
        lashlang::ExecuteError::Runtime(RuntimeError::ParallelConflict { .. })
    ));
}

#[test]
fn snapshot_round_trip_preserves_repl_like_state() {
    let host = TestHost::default();
    let mut state = State::new();

    finished(
        execute(
            r#"
        counter = 1
        finish counter
        "#,
            &mut state,
            &host,
        )
        .expect("first execution should succeed"),
    );

    let snapshot = state.snapshot();
    let encoded = serde_json::to_vec(&snapshot).expect("snapshot should serialize");
    let decoded = serde_json::from_slice(&encoded).expect("snapshot should deserialize");
    let mut restored = State::from_snapshot(decoded);

    let value = finished(
        execute(
            r#"
        counter = counter + 1
        finish counter
        "#,
            &mut restored,
            &host,
        )
        .expect("restored execution should succeed"),
    );

    assert_eq!(value, Value::Number(2.0));
}

#[test]
fn json_and_record_helpers_work() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        obj = json_parse("{\"path\":\"src/lib.rs\",\"line\":7}")
        finish format("{}:{}", obj.path, obj.line)
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("src/lib.rs:7".to_string().into()));
}

#[test]
fn finish_inside_parallel_is_rejected() {
    let host = TestHost::default();
    let mut state = State::new();

    let error = execute(
        r#"
        parallel {
          finish "nope"
        }
        finish "done"
        "#,
        &mut state,
        &host,
    )
    .expect_err("execution should fail");

    assert!(matches!(
        error,
        lashlang::ExecuteError::Runtime(RuntimeError::FinishInsideParallel)
    ));
}

#[test]
fn parse_errors_are_surface_level_and_precise() {
    let error = parse(
        r#"
        if true {
          answer = 1
        "#,
    )
    .expect_err("parse should fail");

    match error {
        lashlang::ParseError::Expected { expected, .. } => assert_eq!(expected, "`}`"),
        other => panic!("unexpected parse error: {other:?}"),
    }
}

fn expect_string<'a>(args: &'a Record, key: &str) -> Result<&'a str, ToolHostError> {
    match args.get(key) {
        Some(Value::String(value)) => Ok(value),
        _ => Err(ToolHostError::new(format!("missing string arg: {key}"))),
    }
}
