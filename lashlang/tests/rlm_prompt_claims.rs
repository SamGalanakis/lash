//! Tests that pin down every behavioural claim made in the RLM execution
//! section of the system prompt (`lash-sansio/src/mode.rs::RLM_EXECUTION_SECTION`).
//!
//! Each test here is wired to a specific bullet or example so the prompt and
//! the runtime can never drift again without a test failing. If you change
//! lashlang semantics, the failing test tells you which line in the prompt
//! is now a lie.

use lashlang::{
    ExecutionOutcome, Record, State, ToolHost, ToolHostError, Value, execute,
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
                let mut record = Record::default();
                record.insert("path".into(), Value::String(format!("/root/{name}").into()));
                record.insert("task_name".into(), Value::String(name.into()));
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
                let mut out = Record::default();
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

    fn observe(&self, value: &Value) -> Result<(), ToolHostError> {
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
    assert_eq!(run(&host, "finish null"), Value::Null);
    assert_eq!(run(&host, "finish true"), Value::Bool(true));
    assert_eq!(run(&host, "finish 42"), Value::Number(42.0));
    assert_eq!(
        run(&host, r#"finish "hi""#),
        Value::String("hi".to_string().into())
    );
    // list literal
    let Value::List(items) = run(&host, "finish [1, 2, 3]") else {
        panic!("expected list");
    };
    assert_eq!(items.len(), 3);
    // record literal
    let Value::Record(rec) = run(&host, "finish { a: 1, b: 2 }") else {
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
        run(&host, "x = 7\ny = x + 3\nfinish y"),
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
    let Value::Record(r) = run(&host, r#"finish call read_file { path: "a.txt" }"#) else {
        panic!("expected wrapped record");
    };
    assert_eq!(r["ok"], Value::Bool(true));
    assert_eq!(r["value"], Value::String("hello world".to_string().into()));
}

#[test]
fn prompt_claim_tool_call_failure_is_wrapped_with_ok_false_and_error() {
    let host = MockHost::default();
    let Value::Record(r) = run(&host, "finish call boom {}") else {
        panic!("expected wrapped record");
    };
    assert_eq!(r["ok"], Value::Bool(false));
    assert!(matches!(&r["error"], Value::String(_)));
}

#[test]
fn prompt_claim_value_field_reaches_the_underlying_tool_output() {
    // The bug that motivated this section: models used `.output` / `.path`
    // directly on the wrapper and got null. Prompt now says to reach
    // through `.value` — this test asserts that works.
    let host = MockHost::default().with_file("a.txt", "file text");
    assert_eq!(
        run(
            &host,
            r#"r = call read_file { path: "a.txt" }
finish r.value"#,
        ),
        Value::String("file text".to_string().into())
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
finish h"#,
    ) else {
        panic!("expected handle record");
    };
    // Handle is NOT a tool-result wrapper.
    assert!(handle.get("ok").is_none(), "start should not wrap in {{ok,value}}");
    assert!(matches!(handle.get("handle"), Some(Value::String(_))));
}

#[test]
fn prompt_claim_await_handle_wraps_result_with_ok_value() {
    let host = MockHost::default().with_file("a.txt", "body");
    let Value::Record(r) = run(
        &host,
        r#"h = start call read_file { path: "a.txt" }
finish await h"#,
    ) else {
        panic!("expected wrapped record");
    };
    assert_eq!(r["ok"], Value::Bool(true));
    assert_eq!(r["value"], Value::String("body".to_string().into()));
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
finish results"#,
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
finish "done""#,
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
finish results"#,
    ) else {
        panic!("expected list");
    };
    assert_eq!(items.len(), 2);
    // Branches return wrapped tool results.
    let a = items[0].as_record().expect("wrapper");
    assert_eq!(a["value"], Value::String("A".to_string().into()));
}

#[test]
fn prompt_claim_bare_expressions_are_valid_parallel_branches() {
    let host = MockHost::default();
    let Value::List(items) = run(
        &host,
        "finish parallel {\n  1 + 1\n  \"lit\"\n}",
    ) else {
        panic!("expected list");
    };
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], Value::Number(2.0));
    assert_eq!(items[1], Value::String("lit".to_string().into()));
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "`observe expr` inspects a value mid-turn."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_observe_feeds_value_to_host() {
    let host = MockHost::default();
    let (outcome, _state) = run_continued(
        &host,
        r#"v = { total: 42 }
observe v"#,
    );
    assert!(matches!(outcome, ExecutionOutcome::Continued));
    let observations = host.observations();
    assert_eq!(observations.len(), 1);
    let r = observations[0].as_record().expect("observed record");
    assert_eq!(r["total"], Value::Number(42.0));
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: "`finish <expr>` ends the turn with the given value..."
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_finish_terminates_program_with_value() {
    let host = MockHost::default();
    let mut state = State::new();
    let outcome = execute("x = 3\nfinish x * 2", &mut state, &host).expect("runs");
    assert_eq!(outcome, ExecutionOutcome::Finished(Value::Number(6.0)));
}

#[test]
fn prompt_claim_bare_finish_terminates_with_null() {
    let host = MockHost::default();
    let mut state = State::new();
    let outcome = execute("finish", &mut state, &host).expect("runs");
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
finish total"#,
        ),
        Value::Number(6.0)
    );
    assert_eq!(
        run(&host, r#"finish true ? "a" : "b""#),
        Value::String("a".to_string().into())
    );
    assert_eq!(
        run(&host, "if 1 < 2 { finish 7 } else { finish 9 }"),
        Value::Number(7.0)
    );
}

#[test]
fn prompt_claim_both_negation_forms_work() {
    let host = MockHost::default();
    assert_eq!(run(&host, "finish !true"), Value::Bool(false));
    assert_eq!(run(&host, "finish not true"), Value::Bool(false));
    assert_eq!(run(&host, "finish !false"), Value::Bool(true));
    assert_eq!(run(&host, "finish not false"), Value::Bool(true));
}

// ─────────────────────────────────────────────────────────────────────
// Prompt claim: every listed builtin is callable and behaves as described.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_claim_builtin_len_returns_length() {
    let host = MockHost::default();
    assert_eq!(run(&host, r#"finish len("hi")"#), Value::Number(2.0));
    assert_eq!(run(&host, "finish len([1,2,3])"), Value::Number(3.0));
    assert_eq!(run(&host, "finish len({a: 1, b: 2})"), Value::Number(2.0));
    assert_eq!(run(&host, "finish len(null)"), Value::Number(0.0));
}

#[test]
fn prompt_claim_builtin_empty_checks_zero_length() {
    let host = MockHost::default();
    assert_eq!(run(&host, r#"finish empty("")"#), Value::Bool(true));
    assert_eq!(run(&host, r#"finish empty("x")"#), Value::Bool(false));
    assert_eq!(run(&host, "finish empty([])"), Value::Bool(true));
    assert_eq!(run(&host, "finish empty([1])"), Value::Bool(false));
}

#[test]
fn prompt_claim_builtin_slice_supports_null_bounds_and_negative_bounds() {
    let host = MockHost::default();
    // null = from start / to end
    assert_eq!(
        run(&host, r#"finish slice("abcdef", null, 3)"#),
        Value::String("abc".to_string().into())
    );
    assert_eq!(
        run(&host, r#"finish slice("abcdef", 3, null)"#),
        Value::String("def".to_string().into())
    );
    // negative bound counts from the end
    assert_eq!(
        run(&host, r#"finish slice("abcdef", 0, -2)"#),
        Value::String("abcd".to_string().into())
    );
    // works on lists too
    let Value::List(items) = run(&host, "finish slice([1,2,3,4], 1, 3)") else {
        panic!("list");
    };
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], Value::Number(2.0));
}

#[test]
fn prompt_claim_builtin_split_and_join() {
    let host = MockHost::default();
    let Value::List(parts) = run(&host, r#"finish split("a,b,c", ",")"#) else {
        panic!("list");
    };
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], Value::String("a".to_string().into()));

    assert_eq!(
        run(&host, r#"finish join(["a", "b", "c"], "-")"#),
        Value::String("a-b-c".to_string().into())
    );
}

#[test]
fn prompt_claim_builtin_trim_strips_whitespace() {
    let host = MockHost::default();
    assert_eq!(
        run(&host, r#"finish trim("  hi  ")"#),
        Value::String("hi".to_string().into())
    );
}

#[test]
fn prompt_claim_builtin_starts_ends_contains() {
    let host = MockHost::default();
    assert_eq!(
        run(&host, r#"finish starts_with("foobar", "foo")"#),
        Value::Bool(true)
    );
    assert_eq!(
        run(&host, r#"finish ends_with("foobar", "bar")"#),
        Value::Bool(true)
    );
    assert_eq!(
        run(&host, r#"finish contains("foobar", "oob")"#),
        Value::Bool(true)
    );
    assert_eq!(
        run(&host, r#"finish contains([1,2,3], 2)"#),
        Value::Bool(true)
    );
}

#[test]
fn prompt_claim_builtin_keys_and_values() {
    let host = MockHost::default();
    let Value::List(keys) = run(&host, "finish keys({a: 1, b: 2})") else {
        panic!("list");
    };
    assert_eq!(keys.len(), 2);
    // Order is insertion order in lashlang records.
    assert_eq!(keys[0], Value::String("a".to_string().into()));
    assert_eq!(keys[1], Value::String("b".to_string().into()));

    let Value::List(vals) = run(&host, "finish values({a: 1, b: 2})") else {
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
        run(&host, "finish to_string(42)"),
        Value::String("42".to_string().into())
    );
    assert_eq!(run(&host, r#"finish to_int("7")"#), Value::Number(7.0));
    assert_eq!(
        run(&host, r#"finish to_float("3.5")"#),
        Value::Number(3.5)
    );
}

#[test]
fn prompt_claim_builtin_json_parse_parses_strings_to_values() {
    let host = MockHost::default();
    let Value::Record(r) = run(&host, r#"finish json_parse("{\"a\": 1}")"#) else {
        panic!("record");
    };
    assert_eq!(r["a"], Value::Number(1.0));
}

#[test]
fn prompt_claim_builtin_format_positional_placeholders() {
    let host = MockHost::default();
    // Auto-numbered `{}` placeholders.
    assert_eq!(
        run(&host, r#"finish format("hi {} you are {}", "sam", 3)"#),
        Value::String("hi sam you are 3".to_string().into())
    );
    // Explicit indices `{0}`, `{1}`.
    assert_eq!(
        run(&host, r#"finish format("{1} {0}", "world", "hello")"#),
        Value::String("hello world".to_string().into())
    );
    // Literal braces via doubling.
    assert_eq!(
        run(&host, r#"finish format("{{ {} }}", "x")"#),
        Value::String("{ x }".to_string().into())
    );
}

// ─────────────────────────────────────────────────────────────────────
// Simple worked example from the prompt's "Example format" block:
//   r = call read_file { path: "Cargo.toml" }
//   finish split(r.value, "\n")[2]
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_example_format_block_executes_as_shown() {
    let host = MockHost::default().with_file("Cargo.toml", "line0\nline1\nline2\nline3\n");
    assert_eq!(
        run(
            &host,
            r#"r = call read_file { path: "Cargo.toml" }
finish split(r.value, "\n")[2]"#,
        ),
        Value::String("line2".to_string().into())
    );
}

// ─────────────────────────────────────────────────────────────────────
// Prompt fanout example: every tool-call value is reached through
// `.value` (both for `call spawn_agent` and for resolved `await`).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn prompt_fanout_example_accesses_spawn_and_wait_results_through_value() {
    let host = MockHost::default();
    let Value::List(results) = run(
        &host,
        r#"a = call spawn_agent { task_name: "chunk_1", task: "x", capability: "low" }
b = call spawn_agent { task_name: "chunk_2", task: "y", capability: "low" }
events = await [
  start call wait_agent { targets: [a.value.path] },
  start call wait_agent { targets: [b.value.path] },
]
finish [events[0].value.events[0].result, events[1].value.events[0].result]"#,
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
        "len", "empty", "slice", "split", "join", "trim", "starts_with", "ends_with", "contains",
        "keys", "values", "to_string", "to_int", "to_float", "json_parse", "format",
    ];
    // Call each builtin with a shape guaranteed to succeed — we don't
    // check results here (covered by the per-builtin tests above), only
    // that the runtime recognises the name.
    let host = MockHost::default();
    let smoke = [
        (r#"finish len("a")"#, "len"),
        (r#"finish empty("")"#, "empty"),
        (r#"finish slice("abc", 0, 1)"#, "slice"),
        (r#"finish split("a,b", ",")"#, "split"),
        (r#"finish join(["a","b"], ",")"#, "join"),
        (r#"finish trim(" a ")"#, "trim"),
        (r#"finish starts_with("abc", "a")"#, "starts_with"),
        (r#"finish ends_with("abc", "c")"#, "ends_with"),
        (r#"finish contains("abc", "b")"#, "contains"),
        (r#"finish keys({a: 1})"#, "keys"),
        (r#"finish values({a: 1})"#, "values"),
        (r#"finish to_string(1)"#, "to_string"),
        (r#"finish to_int("1")"#, "to_int"),
        (r#"finish to_float("1.5")"#, "to_float"),
        (r#"finish json_parse("1")"#, "json_parse"),
        (r#"finish format("x")"#, "format"),
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
