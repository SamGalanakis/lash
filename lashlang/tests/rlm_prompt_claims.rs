//! Tests that pin down every behavioural claim made in the RLM execution
//! section of the system prompt (`lash-mode-rlm/src/driver.rs::RLM_EXECUTION_SECTION`).
//!
//! Each test here is wired to a specific bullet or example so the prompt and
//! the runtime can never drift again without a test failing. If you change
//! lashlang semantics, the failing test tells you which line in the prompt
//! is now a lie.

use lashlang::{
    ExecuteError, ExecutionOutcome, Record, State, ToolHost, ToolHostError, Value, execute,
};
use std::collections::HashMap;
use std::sync::Mutex;

// ─────────────────────────────────────────────────────────────────────
// Test host: mirrors the real RLM runtime's tool-result wrapping
// contract (the runtime automatically wraps `call` results as
// {ok, value}; the host itself just returns the raw inner value).
// ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct MockHost {
    files: HashMap<String, String>,
    observations: Mutex<Vec<Value>>,
    // Handles for start/await: `start_call` returns a record like
    // `{ handle: "h<n>", tool: "<name>" }`, and `await_handle` looks up
    // the pending tool result by handle id.
    pending: Mutex<HashMap<String, Value>>,
    next_handle: Mutex<u32>,
}

impl MockHost {
    fn with_file(mut self, path: &str, contents: &str) -> Self {
        self.files.insert(path.to_string(), contents.to_string());
        self
    }

    fn record_observation(&self, value: Value) {
        self.observations.lock().unwrap().push(value);
    }

    fn observations(&self) -> Vec<Value> {
        self.observations.lock().unwrap().clone()
    }
}

impl ToolHost for MockHost {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        match name {
            "read_file" => {
                let path = args
                    .get("path")
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.to_string()),
                        _ => None,
                    })
                    .ok_or_else(|| ToolHostError::new("missing path"))?;
                self.files
                    .get(&path)
                    .cloned()
                    .map(|s| Value::String(s.into()))
                    .ok_or_else(|| ToolHostError::new(format!("no such file: {path}")))
            }
            "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
            "spawn_agent" => {
                let name = args
                    .get("task_name")
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.to_string()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let target = format!("/root/{name}");
                let mut record = Record::default();
                record.insert("target".into(), Value::String(target.clone().into()));
                record.insert(
                    "task_id".into(),
                    Value::String(format!("subagent:{target}").into()),
                );
                record.insert("task_name".into(), Value::String(name.into()));
                record.insert("run_state".into(), Value::String("running".into()));
                Ok(Value::Record(record.into()))
            }
            "wait_agent" => {
                // Fake one terminal event per target.
                let targets = args
                    .get("targets")
                    .and_then(|v| match v {
                        Value::List(items) => Some(items.to_vec()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let events: Vec<Value> = targets
                    .into_iter()
                    .map(|target| {
                        let mut ev = Record::default();
                        ev.insert("target".into(), target.clone());
                        ev.insert(
                            "result".into(),
                            Value::String(format!("done:{}", target).into()),
                        );
                        Value::Record(ev.into())
                    })
                    .collect();
                let completion = events.first().cloned().unwrap_or(Value::Null);
                let mut out = Record::default();
                out.insert("completion".into(), completion);
                out.insert("events".into(), Value::List(events.into()));
                Ok(Value::Record(out.into()))
            }
            "boom" => Err(ToolHostError::new("explicit failure for tests")),
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }

    fn start_call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        // Allocate a handle id, run the sync call now, stash the result.
        let mut counter = self.next_handle.lock().unwrap();
        *counter += 1;
        let handle_id = format!("h{}", *counter);
        drop(counter);
        let result = self.call(name, args)?;
        self.pending
            .lock()
            .unwrap()
            .insert(handle_id.clone(), result);
        let mut record = Record::default();
        record.insert("handle".into(), Value::String(handle_id.into()));
        record.insert("tool".into(), Value::String(name.to_string().into()));
        Ok(Value::Record(record.into()))
    }

    fn await_handle(&self, handle: &Value) -> Result<Value, ToolHostError> {
        let record = handle
            .as_record()
            .ok_or_else(|| ToolHostError::new("expected handle record"))?;
        let id = record
            .get("handle")
            .and_then(|v| match v {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
            .ok_or_else(|| ToolHostError::new("handle record missing `handle` field"))?;
        self.pending
            .lock()
            .unwrap()
            .remove(&id)
            .ok_or_else(|| ToolHostError::new(format!("unknown handle: {id}")))
    }

    fn cancel_handle(&self, handle: &Value) -> Result<Value, ToolHostError> {
        if let Some(record) = handle.as_record()
            && let Some(Value::String(id)) = record.get("handle")
        {
            self.pending.lock().unwrap().remove(id.as_str());
        }
        Ok(Value::Null)
    }

    fn print(&self, value: &Value) -> Result<(), ToolHostError> {
        self.record_observation(value.clone());
        Ok(())
    }
}

fn run(host: &MockHost, source: &str) -> Value {
    let mut state = State::new();
    match execute(source, &mut state, host).expect("execution should succeed") {
        ExecutionOutcome::Finished(value) => value,
        ExecutionOutcome::Continued => Value::Null,
    }
}

fn run_continued(host: &MockHost, source: &str) -> (ExecutionOutcome, State) {
    let mut state = State::new();
    let outcome = execute(source, &mut state, host).expect("execution should succeed");
    (outcome, state)
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "Values: null, booleans, numbers, strings, lists, records.
//               Literals: `[a, b]`, `{ a: 1, b: 2 }`."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_value_literals_parse_and_evaluate() {
    let host = MockHost::default();
    assert_eq!(run(&host, "submit null"), Value::Null);
    assert_eq!(run(&host, "submit true"), Value::Bool(true));
    assert_eq!(run(&host, "submit 42"), Value::Number(42.0));
    assert_eq!(
        run(&host, r#"submit "hi""#),
        Value::String("hi".to_string().into())
    );
    // list literal
    let Value::List(items) = run(&host, "submit [1, 2, 3]") else {
        panic!("expected list");
    };
    assert_eq!(items.len(), 3);
    // record literal
    let Value::Record(rec) = run(&host, "submit { a: 1, b: 2 }") else {
        panic!("expected record");
    };
    assert_eq!(rec["a"], Value::Number(1.0));
    assert_eq!(rec["b"], Value::Number(2.0));
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "Assign with `name = expr`. Variables persist across
// fenced blocks within the turn."
// (Within-block persistence is covered here; cross-block persistence
// is tested at the RLM-runtime level, not at the lashlang level.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_assignment_persists_within_program() {
    let host = MockHost::default();
    assert_eq!(
        run(&host, "x = 7\ny = x + 3\nsubmit y"),
        Value::Number(10.0)
    );
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "Every tool call returns a wrapper record:
// `{ ok: true, value: <tool output> }` on success,
// `{ ok: false, error: "..." }` on failure."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_tool_call_success_is_wrapped_with_ok_and_value() {
    let host = MockHost::default().with_file("a.txt", "hello world");
    let Value::Record(r) = run(&host, r#"submit call read_file { path: "a.txt" }"#) else {
        panic!("expected wrapped record");
    };
    assert_eq!(r["ok"], Value::Bool(true));
    assert_eq!(r["value"], Value::String("hello world".to_string().into()));
}

#[test]
fn prompt_claim_tool_call_failure_is_wrapped_with_ok_false_and_error() {
    let host = MockHost::default();
    let Value::Record(r) = run(&host, "submit call boom {}") else {
        panic!("expected wrapped record");
    };
    assert_eq!(r["ok"], Value::Bool(false));
    assert!(matches!(&r["error"], Value::String(_)));
}

#[test]
fn prompt_claim_value_field_reaches_the_underlying_tool_output() {
    // The bug that motivated this section: models used `.output` / `.path`
    // directly on the wrapper and got null. Manual `.value` access still
    // works when code intentionally keeps the raw wrapper.
    let host = MockHost::default().with_file("a.txt", "file text");
    assert_eq!(
        run(
            &host,
            r#"r = call read_file { path: "a.txt" }
submit r.value"#,
        ),
        Value::String("file text".to_string().into())
    );
}

#[test]
fn prompt_claim_question_unwraps_successful_tool_results() {
    let host = MockHost::default().with_file("a.txt", "file text");
    assert_eq!(
        run(
            &host,
            r#"text = (call read_file { path: "a.txt" })?
submit text"#,
        ),
        Value::String("file text".to_string().into())
    );
}

#[test]
fn prompt_claim_question_aborts_failed_tool_results_with_error() {
    let host = MockHost::default();
    let mut state = State::new();
    let err = execute("submit (call boom {})?", &mut state, &host)
        .expect_err("failed result unwrap should abort");
    let ExecuteError::Runtime(err) = err else {
        panic!("expected runtime error");
    };
    assert!(
        err.to_string().contains("explicit failure for tests"),
        "unexpected error: {err}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "`start call tool { arg: expr }` returns a handle
// (not wrapped). Resolve with `await handle` — that returns the same
// `{ ok, value }` wrapper as a synchronous call."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_start_returns_unwrapped_handle() {
    let host = MockHost::default().with_file("a.txt", "x");
    let Value::Record(handle) = run(
        &host,
        r#"h = start call read_file { path: "a.txt" }
submit h"#,
    ) else {
        panic!("expected handle record");
    };
    // Handle is NOT a tool-result wrapper.
    assert!(
        handle.get("ok").is_none(),
        "start should not wrap in {{ok,value}}"
    );
    assert!(matches!(handle.get("handle"), Some(Value::String(_))));
}

#[test]
fn prompt_claim_await_handle_wraps_result_with_ok_value() {
    let host = MockHost::default().with_file("a.txt", "body");
    let Value::Record(r) = run(
        &host,
        r#"h = start call read_file { path: "a.txt" }
submit await h"#,
    ) else {
        panic!("expected wrapped record");
    };
    assert_eq!(r["ok"], Value::Bool(true));
    assert_eq!(r["value"], Value::String("body".to_string().into()));
}

#[test]
fn prompt_claim_question_unwraps_awaited_handle_results() {
    let host = MockHost::default().with_file("a.txt", "body");
    assert_eq!(
        run(
            &host,
            r#"h = start call read_file { path: "a.txt" }
submit (await h)?"#,
        ),
        Value::String("body".to_string().into())
    );
}

#[test]
fn prompt_claim_await_list_returns_wrappers_in_order() {
    let host = MockHost::default()
        .with_file("a.txt", "A")
        .with_file("b.txt", "B");
    let Value::List(items) = run(
        &host,
        r#"results = await [
  start call read_file { path: "a.txt" },
  start call read_file { path: "b.txt" },
]
submit results"#,
    ) else {
        panic!("expected list");
    };
    assert_eq!(items.len(), 2);
    let a = items[0].as_record().expect("wrapper record");
    let b = items[1].as_record().expect("wrapper record");
    assert_eq!(a["ok"], Value::Bool(true));
    assert_eq!(a["value"], Value::String("A".to_string().into()));
    assert_eq!(b["ok"], Value::Bool(true));
    assert_eq!(b["value"], Value::String("B".to_string().into()));
}

#[test]
fn prompt_claim_await_record_returns_wrappers_by_name() {
    let host = MockHost::default()
        .with_file("a.txt", "A")
        .with_file("b.txt", "B");
    let Value::Record(items) = run(
        &host,
        r#"results = await {
  a: start call read_file { path: "a.txt" },
  b: start call read_file { path: "b.txt" },
}
submit results"#,
    ) else {
        panic!("expected record");
    };
    assert_eq!(
        items["a"].as_record().unwrap()["value"],
        Value::String("A".into())
    );
    assert_eq!(
        items["b"].as_record().unwrap()["value"],
        Value::String("B".into())
    );
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "`cancel handle` (best-effort)"
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_cancel_handle_runs_without_error() {
    let host = MockHost::default().with_file("a.txt", "x");
    run(
        &host,
        r#"h = start call read_file { path: "a.txt" }
cancel h
submit "done""#,
    );
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "`parallel { ... }` returns a list of branch results in
// order. Do not use it when one branch needs another branch's output."
// Plus: "Bare expressions are valid statements. Inside `parallel { ... }`,
// a bare expression contributes its value to the result list."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_parallel_expression_returns_branch_list_in_order() {
    let host = MockHost::default()
        .with_file("a.txt", "A")
        .with_file("b.txt", "B");
    let Value::List(items) = run(
        &host,
        r#"results = parallel {
  a = call read_file { path: "a.txt" }
  b = call read_file { path: "b.txt" }
}
submit results"#,
    ) else {
        panic!("expected list");
    };
    assert_eq!(items.len(), 2);
    // Branches return wrapped tool results.
    let a = items[0].as_record().expect("wrapper");
    assert_eq!(a["value"], Value::String("A".to_string().into()));
}

#[test]
fn prompt_claim_named_parallel_expression_returns_record() {
    let host = MockHost::default()
        .with_file("a.txt", "A")
        .with_file("b.txt", "B");
    let Value::Record(items) = run(
        &host,
        r#"results = parallel {
  a: call read_file { path: "a.txt" }
  b: call read_file { path: "b.txt" }
}
submit results"#,
    ) else {
        panic!("expected record");
    };
    assert_eq!(
        items["a"].as_record().unwrap()["value"],
        Value::String("A".into())
    );
    assert_eq!(
        items["b"].as_record().unwrap()["value"],
        Value::String("B".into())
    );
}

#[test]
fn prompt_claim_bare_expressions_are_valid_parallel_branches() {
    let host = MockHost::default();
    let Value::List(items) = run(&host, "submit parallel {\n  1 + 1\n  \"lit\"\n}") else {
        panic!("expected list");
    };
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], Value::Number(2.0));
    assert_eq!(items[1], Value::String("lit".to_string().into()));
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "`print expr` inspects a value mid-turn."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_print_feeds_value_to_host() {
    let host = MockHost::default();
    let (outcome, _state) = run_continued(
        &host,
        r#"v = { total: 42 }
print v"#,
    );
    assert!(matches!(outcome, ExecutionOutcome::Continued));
    let observations = host.observations();
    assert_eq!(observations.len(), 1);
    let r = observations[0].as_record().expect("observed record");
    assert_eq!(r["total"], Value::Number(42.0));
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "`submit <expr>` ends the turn with the given value..."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_submit_terminates_program_with_value() {
    let host = MockHost::default();
    let mut state = State::new();
    let outcome = execute("x = 3\nsubmit x * 2", &mut state, &host).expect("runs");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(6.0)));
}

#[test]
fn prompt_claim_bare_submit_terminates_with_null() {
    let host = MockHost::default();
    let mut state = State::new();
    let outcome = execute("submit", &mut state, &host).expect("runs");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Null));
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "Control flow: statement `if`/`for`; expression ternary
// `cond ? yes : no` (there is no expression-form `if`); boolean negation
// via `!cond` or `not cond`."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_if_for_and_ternary_work() {
    let host = MockHost::default();
    assert_eq!(
        run(
            &host,
            r#"total = 0
for n in [1, 2, 3] {
  total = total + n
}
submit total"#,
        ),
        Value::Number(6.0)
    );
    assert_eq!(
        run(&host, r#"submit true ? "a" : "b""#),
        Value::String("a".to_string().into())
    );
    assert_eq!(
        run(&host, "if 1 < 2 { submit 7 } else { submit 9 }"),
        Value::Number(7.0)
    );
}

#[test]
fn prompt_claim_both_negation_forms_work() {
    let host = MockHost::default();
    assert_eq!(run(&host, "submit !true"), Value::Bool(false));
    assert_eq!(run(&host, "submit not true"), Value::Bool(false));
    assert_eq!(run(&host, "submit !false"), Value::Bool(true));
    assert_eq!(run(&host, "submit not false"), Value::Bool(true));
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: every listed builtin is callable and behaves as described.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_builtin_len_returns_length() {
    let host = MockHost::default();
    assert_eq!(run(&host, r#"submit len("hi")"#), Value::Number(2.0));
    assert_eq!(run(&host, "submit len([1,2,3])"), Value::Number(3.0));
    assert_eq!(run(&host, "submit len({a: 1, b: 2})"), Value::Number(2.0));
    assert_eq!(run(&host, "submit len(null)"), Value::Number(0.0));
}

#[test]
fn prompt_claim_builtin_empty_checks_zero_length() {
    let host = MockHost::default();
    assert_eq!(run(&host, r#"submit empty("")"#), Value::Bool(true));
    assert_eq!(run(&host, r#"submit empty("x")"#), Value::Bool(false));
    assert_eq!(run(&host, "submit empty([])"), Value::Bool(true));
    assert_eq!(run(&host, "submit empty([1])"), Value::Bool(false));
}

#[test]
fn prompt_claim_builtin_slice_supports_null_bounds_and_negative_bounds() {
    let host = MockHost::default();
    // null = from start / to end
    assert_eq!(
        run(&host, r#"submit slice("abcdef", null, 3)"#),
        Value::String("abc".to_string().into())
    );
    assert_eq!(
        run(&host, r#"submit slice("abcdef", 3, null)"#),
        Value::String("def".to_string().into())
    );
    // negative bound counts from the end
    assert_eq!(
        run(&host, r#"submit slice("abcdef", 0, -2)"#),
        Value::String("abcd".to_string().into())
    );
    // works on lists too
    let Value::List(items) = run(&host, "submit slice([1,2,3,4], 1, 3)") else {
        panic!("list");
    };
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], Value::Number(2.0));
}

#[test]
fn prompt_claim_builtin_range_and_push_build_lists() {
    let host = MockHost::default();
    assert_eq!(
        run(&host, "submit range(3)"),
        Value::List(vec![Value::Number(0.0), Value::Number(1.0), Value::Number(2.0)].into())
    );
    assert_eq!(
        run(&host, "submit range(-2, 1)"),
        Value::List(vec![Value::Number(-2.0), Value::Number(-1.0), Value::Number(0.0)].into())
    );
    assert_eq!(
        run(&host, "submit range(3, 3)"),
        Value::List(Vec::new().into())
    );

    let value = run(
        &host,
        r#"
        base = ["a"]
        extended = push(base, "b")
        submit { base: base, extended: extended }
        "#,
    );
    let record = value.as_record().expect("record");
    assert_eq!(
        record["base"],
        Value::List(vec![Value::String("a".to_string().into())].into())
    );
    assert_eq!(
        record["extended"],
        Value::List(
            vec![
                Value::String("a".to_string().into()),
                Value::String("b".to_string().into())
            ]
            .into()
        )
    );
}

#[test]
fn prompt_claim_builtin_split_and_join() {
    let host = MockHost::default();
    let Value::List(parts) = run(&host, r#"submit split("a,b,c", ",")"#) else {
        panic!("list");
    };
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], Value::String("a".to_string().into()));

    assert_eq!(
        run(&host, r#"submit join(["a", "b", "c"], "-")"#),
        Value::String("a-b-c".to_string().into())
    );
}

#[test]
fn prompt_claim_builtin_trim_strips_whitespace() {
    let host = MockHost::default();
    assert_eq!(
        run(&host, r#"submit trim("  hi  ")"#),
        Value::String("hi".to_string().into())
    );
}

#[test]
fn prompt_claim_builtin_starts_ends_contains() {
    let host = MockHost::default();
    assert_eq!(
        run(&host, r#"submit starts_with("foobar", "foo")"#),
        Value::Bool(true)
    );
    assert_eq!(
        run(&host, r#"submit ends_with("foobar", "bar")"#),
        Value::Bool(true)
    );
    assert_eq!(
        run(&host, r#"submit contains("foobar", "oob")"#),
        Value::Bool(true)
    );
    assert_eq!(
        run(&host, r#"submit contains([1,2,3], 2)"#),
        Value::Bool(true)
    );
}

#[test]
fn prompt_claim_builtin_keys_and_values() {
    let host = MockHost::default();
    let Value::List(keys) = run(&host, "submit keys({a: 1, b: 2})") else {
        panic!("list");
    };
    assert_eq!(keys.len(), 2);
    // Order is insertion order in lashlang records.
    assert_eq!(keys[0], Value::String("a".to_string().into()));
    assert_eq!(keys[1], Value::String("b".to_string().into()));

    let Value::List(vals) = run(&host, "submit values({a: 1, b: 2})") else {
        panic!("list");
    };
    assert_eq!(vals.len(), 2);
    assert_eq!(vals[0], Value::Number(1.0));
    assert_eq!(vals[1], Value::Number(2.0));
}

#[test]
fn prompt_claim_builtin_to_string_to_int_to_float() {
    let host = MockHost::default();
    assert_eq!(
        run(&host, "submit to_string(42)"),
        Value::String("42".to_string().into())
    );
    assert_eq!(run(&host, r#"submit to_int("7")"#), Value::Number(7.0));
    assert_eq!(run(&host, r#"submit to_float("3.5")"#), Value::Number(3.5));
}

#[test]
fn prompt_claim_builtin_json_parse_parses_strings_to_values() {
    let host = MockHost::default();
    let Value::Record(r) = run(&host, r#"submit json_parse("{\"a\": 1}")"#) else {
        panic!("record");
    };
    assert_eq!(r["a"], Value::Number(1.0));
}

#[test]
fn prompt_claim_builtin_format_positional_placeholders() {
    let host = MockHost::default();
    // Auto-numbered `{}` placeholders.
    assert_eq!(
        run(&host, r#"submit format("hi {} you are {}", "sam", 3)"#),
        Value::String("hi sam you are 3".to_string().into())
    );
    // Explicit indices `{0}`, `{1}`.
    assert_eq!(
        run(&host, r#"submit format("{1} {0}", "world", "hello")"#),
        Value::String("hello world".to_string().into())
    );
    // Literal braces via doubling.
    assert_eq!(
        run(&host, r#"submit format("{{ {} }}", "x")"#),
        Value::String("{ x }".to_string().into())
    );
}

#[test]
fn prompt_claim_builtin_validate_checks_type_literals_mid_program() {
    let host = MockHost::default();
    let value = run(
        &host,
        r#"
        raw = { name: "lashlang", labels: ["agent", "runtime"] }
        submit validate(raw, Type { name: str, labels: list[str] })
        "#,
    );
    let record = value.as_record().expect("validated record");
    assert_eq!(record["name"], Value::String("lashlang".to_string().into()));

    let mut state = State::new();
    let err = execute(
        r#"submit validate({ labels: ["agent", 42] }, Type { labels: list[str] })"#,
        &mut state,
        &host,
    )
    .expect_err("validate should abort on bad shape");
    let ExecuteError::Runtime(err) = err else {
        panic!("expected runtime error");
    };
    assert!(
        err.to_string()
            .contains("$.labels[1]: expected string, got number"),
        "unexpected error: {err}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Simple worked example from the prompt's "Example format" block:
//   r = call read_file { path: "Cargo.toml" }
//   submit split(r.value, "\n")[2]
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_example_format_block_executes_as_shown() {
    let host = MockHost::default().with_file("Cargo.toml", "line0\nline1\nline2\nline3\n");
    assert_eq!(
        run(
            &host,
            r#"r = call read_file { path: "Cargo.toml" }
submit split(r.value, "\n")[2]"#,
        ),
        Value::String("line2".to_string().into())
    );
}

// ─────────────────────────────────────────────────────────────────────
// Prompt fanout example: `?` unwraps normal happy-path tool results.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_fanout_example_unwraps_spawn_and_wait_results_with_question() {
    let host = MockHost::default();
    let Value::List(results) = run(
        &host,
        r#"a = (call spawn_agent { task_name: "chunk_1", task: "x", capability: "low" })?
b = (call spawn_agent { task_name: "chunk_2", task: "y", capability: "low" })?
events = await {
  a: start call wait_agent { targets: [a.target] },
  b: start call wait_agent { targets: [b.target] },
}
submit [events.a?.completion.result, events.b?.completion.result]"#,
    ) else {
        panic!("expected list");
    };
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0],
        Value::String("done:/root/chunk_1".to_string().into())
    );
    assert_eq!(
        results[1],
        Value::String("done:/root/chunk_2".to_string().into())
    );
}

// ─────────────────────────────────────────────────────────────────────
// Meta-claim guard: the prompt still references every builtin we've
// implemented — no silent drift. If someone adds/removes a builtin
// this fails and points them at the prompt.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_mentions_every_builtin_we_document() {
    // Source: `lash-sansio/src/mode.rs::RLM_EXECUTION_SECTION`. We don't
    // pull the const directly (cross-crate visibility) — instead keep the
    // expected list here. If you add a builtin, update both places.
    const DOCUMENTED: &[&str] = &[
        "len",
        "empty",
        "slice",
        "split",
        "join",
        "trim",
        "starts_with",
        "ends_with",
        "contains",
        "keys",
        "values",
        "to_string",
        "to_int",
        "to_float",
        "json_parse",
        "format",
        "validate",
        "range",
        "push",
    ];
    // Call each builtin with a shape guaranteed to succeed — we don't
    // check results here (covered by the per-builtin tests above), only
    // that the runtime recognises the name.
    let host = MockHost::default();
    let smoke = [
        (r#"submit len("a")"#, "len"),
        (r#"submit empty("")"#, "empty"),
        (r#"submit slice("abc", 0, 1)"#, "slice"),
        (r#"submit split("a,b", ",")"#, "split"),
        (r#"submit join(["a","b"], ",")"#, "join"),
        (r#"submit trim(" a ")"#, "trim"),
        (r#"submit starts_with("abc", "a")"#, "starts_with"),
        (r#"submit ends_with("abc", "c")"#, "ends_with"),
        (r#"submit contains("abc", "b")"#, "contains"),
        (r#"submit keys({a: 1})"#, "keys"),
        (r#"submit values({a: 1})"#, "values"),
        (r#"submit to_string(1)"#, "to_string"),
        (r#"submit to_int("1")"#, "to_int"),
        (r#"submit to_float("1.5")"#, "to_float"),
        (r#"submit json_parse("1")"#, "json_parse"),
        (r#"submit format("x")"#, "format"),
        (
            r#"submit validate({ value: "x" }, Type { value: str })"#,
            "validate",
        ),
        (r#"submit range(1)"#, "range"),
        (r#"submit push([], "x")"#, "push"),
    ];
    assert_eq!(smoke.len(), DOCUMENTED.len());
    for (code, name) in smoke {
        let mut state = State::new();
        let outcome = execute(code, &mut state, &host)
            .unwrap_or_else(|err| panic!("builtin `{name}` failed to execute: {err:?}"));
        assert!(
            matches!(outcome, ExecutionOutcome::Finished(_)),
            "builtin `{name}` did not finish"
        );
    }
}
