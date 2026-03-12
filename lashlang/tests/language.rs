use lashlang::{Record, RuntimeError, State, ToolHost, ToolHostError, Value, execute, parse};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
struct TestHost {
    files: HashMap<String, String>,
    globs: HashMap<String, Vec<String>>,
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
                    Some(content) => Ok(Value::String(content.clone())),
                    None => Err(ToolHostError::new(format!("missing file: {path}"))),
                }
            }
            "glob" => {
                let pattern = expect_string(args, "pattern")?;
                let values = self
                    .globs
                    .get(pattern)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(Value::String)
                    .collect();
                Ok(Value::List(values))
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
                Ok(Value::String(expect_string(args, "value")?.to_string()))
            }
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
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
fn executes_arithmetic_strings_and_finish() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = execute(
        r#"
        total = 1 + 2 * 3
        msg = format("total={0}", total)
        finish msg
        "#,
        &mut state,
        &host,
    )
    .expect("execution should succeed");

    assert_eq!(value, Value::String("total=7".to_string()));
    assert_eq!(state.globals()["total"], Value::Number(7.0));
}

#[test]
fn executes_if_for_and_list_concat() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = execute(
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
    .expect("execution should succeed");

    assert_eq!(value, Value::String("n=1,n=2,n=3,n=4".to_string()));
}

#[test]
fn tool_calls_return_result_records() {
    let host = TestHost::default().with_file("src/lib.rs", "pub fn main() {}");
    let mut state = State::new();

    let value = execute(
        r#"
        found = call read_file { path: "src/lib.rs" }
        missing = call read_file { path: "src/missing.rs" }
        finish { found: found, missing: missing }
        "#,
        &mut state,
        &host,
    )
    .expect("execution should succeed");

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

    let value = execute(
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
    .expect("execution should succeed");

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(
        record["left"].as_record().unwrap()["value"],
        Value::String("a".to_string())
    );
    assert_eq!(
        record["right"].as_record().unwrap()["value"],
        Value::String("b".to_string())
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

    execute(
        r#"
        counter = 1
        finish counter
        "#,
        &mut state,
        &host,
    )
    .expect("first execution should succeed");

    let snapshot = state.snapshot();
    let encoded = serde_json::to_vec(&snapshot).expect("snapshot should serialize");
    let decoded = serde_json::from_slice(&encoded).expect("snapshot should deserialize");
    let mut restored = State::from_snapshot(decoded);

    let value = execute(
        r#"
        counter = counter + 1
        finish counter
        "#,
        &mut restored,
        &host,
    )
    .expect("restored execution should succeed");

    assert_eq!(value, Value::Number(2.0));
}

#[test]
fn json_and_record_helpers_work() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = execute(
        r#"
        obj = json_parse("{\"path\":\"src/lib.rs\",\"line\":7}")
        finish format("{0}:{1}", obj.path, obj.line)
        "#,
        &mut state,
        &host,
    )
    .expect("execution should succeed");

    assert_eq!(value, Value::String("src/lib.rs:7".to_string()));
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
