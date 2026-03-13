use lashlang::{
    ExecutionOutcome, Record, RuntimeError, State, ToolHost, ToolHostError, Value, execute, parse,
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
                Ok(Value::String(expect_string(args, "value")?.to_string().into()))
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
fn parser_accepts_ternary_in_call_arguments() {
    let program = parse(
        r#"
        result = format("{0}", true ? "yes" : "no")
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
        msg = format("total={0}", total)
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
          labels = labels + [format("n={0}", n)]
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
        finish format("{0}:{1}", truthy, falsy)
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
        finish format("{0}:{1}", yes, no)
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
    assert_eq!(record["joined"], Value::String("a-2-true".to_string().into()));
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
fn format_with_single_value_stringifies_it() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        finish format({ ok: true, count: 2 })
        "#,
            &mut state,
            &host,
        )
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::String("{\"count\":2.0,\"ok\":true}".to_string().into())
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
          "found={0} missing={1}",
          found.ok ? "ok" : format("failed: {0}", found.error),
          missing.ok ? "ok" : format("failed: {0}", missing.error)
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
        finish format("{0}:{1}", obj.path, obj.line)
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
