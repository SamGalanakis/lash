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
    async fn call(&self, name: String, args: Record) -> Result<Value, ToolHostError> {
        match name.as_str() {
            "read_file" => {
                let path = expect_string(&args, "path")?;
                match self.files.get(path) {
                    Some(content) => Ok(Value::String(content.clone().into())),
                    None => Err(ToolHostError::new(format!("missing file: {path}"))),
                }
            }
            "glob" => {
                let pattern = expect_string(&args, "pattern")?;
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
                tokio::time::sleep(Duration::from_millis(50)).await;
                self.active.fetch_sub(1, Ordering::SeqCst);
                Ok(Value::String(
                    expect_string(&args, "value")?.to_string().into(),
                ))
            }
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }

    async fn start_call(&self, name: String, args: Record) -> Result<Value, ToolHostError> {
        self.call(name, args).await
    }

    async fn await_handle(&self, handle: Value) -> Result<Value, ToolHostError> {
        Ok(handle)
    }

    async fn print(&self, value: Value) -> Result<(), ToolHostError> {
        self.observations
            .lock()
            .expect("observation mutex")
            .push(value);
        Ok(())
    }
}

fn finished(outcome: ExecutionOutcome) -> Value {
    match outcome {
        ExecutionOutcome::Finished(value) => value,
        ExecutionOutcome::Continued => panic!("expected `submit`"),
    }
}

async fn runtime_error(source: &str) -> RuntimeError {
    let host = TestHost::default();
    let mut state = State::new();
    match execute(source, &mut state, &host)
        .await
        .expect_err("execution should fail")
    {
        ExecuteError::Runtime(error) => error,
        ExecuteError::Parse(error) => panic!("expected runtime error, got parse error: {error:?}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn parser_handles_precedence_and_parallel() {
    let program = parse(
        r#"
        total = 1 + 2 * 3
        parallel {
          left = call glob { pattern: "src/*.rs" }
          right = call read_file { path: "src/lib.rs" }
        }
        submit total
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 3);
}

#[tokio::test(flavor = "current_thread")]
async fn parser_accepts_double_slash_comments() {
    let program = parse(
        r#"
        // setup
        total = 1 + 2
        // submit
        submit total
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn parser_accepts_semicolons_as_statement_separators() {
    let program = parse(
        r#"
        x = 1; y = 2;
        submit x;
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 3);
}

#[tokio::test(flavor = "current_thread")]
async fn start_is_contextual_not_reserved() {
    let host = TestHost::default().with_file("a.txt", "async");
    let mut state = State::new();
    let value = finished(
        execute(
            r#"
            start = 1
            for start in range(3) {
              last = start
            }
            rec = { start: last }
            h = start call read_file { path: "a.txt" }
            submit { value: start, field: rec.start, awaited: (await h)? }
            "#,
            &mut state,
            &host,
        )
        .await
        .expect("contextual start program should run"),
    );

    let record = value.as_record().expect("record");
    assert_eq!(record["value"], Value::Number(1.0));
    assert_eq!(record["field"], Value::Number(2.0));
    assert_eq!(record["awaited"], Value::String("async".to_string().into()));

    let err = runtime_error("submit start()").await;
    assert!(matches!(err, RuntimeError::UnknownBuiltin { name } if name == "start"));
}

#[tokio::test(flavor = "current_thread")]
async fn range_supports_python_style_steps() {
    let host = TestHost::default();
    let mut state = State::new();
    let value = finished(
        execute(
            r#"
            stepped = []
            for i in range(5, 0, -2) {
              stepped = push(stepped, i)
            }
            submit {
              up: range(0, 5, 2),
              down: range(5, 0, -2),
              empty_up: range(5, 0, 2),
              empty_down: range(0, 5, -2),
              iterated: stepped
            }
            "#,
            &mut state,
            &host,
        )
        .await
        .expect("stepped ranges should run"),
    );

    let record = value.as_record().expect("record");
    assert_eq!(
        record["up"],
        Value::List(vec![Value::Number(0.0), Value::Number(2.0), Value::Number(4.0)].into())
    );
    assert_eq!(
        record["down"],
        Value::List(vec![Value::Number(5.0), Value::Number(3.0), Value::Number(1.0)].into())
    );
    assert_eq!(record["empty_up"], Value::List(Vec::new().into()));
    assert_eq!(record["empty_down"], Value::List(Vec::new().into()));
    assert_eq!(record["iterated"], record["down"]);
}

#[tokio::test(flavor = "current_thread")]
async fn integer_division_helpers_use_mathematical_rounding() {
    let host = TestHost::default();
    let mut state = State::new();
    let value = finished(
        execute(
            r#"
            items = range(10)
            stride = ceil_div(len(items), 3)
            starts = []
            for i in range(0, len(items), stride) {
              starts = push(starts, i)
            }
            submit {
              ceil_pos: ceil_div(10, 3),
              floor_pos: floor_div(10, 3),
              ceil_neg: ceil_div(-10, 3),
              floor_neg: floor_div(-10, 3),
              starts: starts
            }
            "#,
            &mut state,
            &host,
        )
        .await
        .expect("division helpers should run"),
    );

    let record = value.as_record().expect("record");
    assert_eq!(record["ceil_pos"], Value::Number(4.0));
    assert_eq!(record["floor_pos"], Value::Number(3.0));
    assert_eq!(record["ceil_neg"], Value::Number(-3.0));
    assert_eq!(record["floor_neg"], Value::Number(-4.0));
    assert_eq!(
        record["starts"],
        Value::List(vec![Value::Number(0.0), Value::Number(4.0), Value::Number(8.0)].into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn numeric_helper_errors_are_rejected() {
    for source in [
        "submit range(0, 5, 0)",
        "submit range(0, 5, 1.5)",
        "submit range(1000001, 0, -1)",
        "submit ceil_div(1.5, 1)",
        "submit floor_div(1, 0)",
        "submit ceil_div(\"1\", 1)",
    ] {
        let err = runtime_error(source).await;
        assert!(matches!(
            err,
            RuntimeError::TypeError { .. } | RuntimeError::ValueError { .. }
        ));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn parser_accepts_trailing_semicolon_after_raw_string() {
    let program = parse(
        r#"
        msg = r'''hello''';
        submit msg
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn parser_treats_semicolon_like_whitespace_between_idents() {
    let with_semi = parse("x = 1;y = 2").expect("semicolon-separated should parse");
    let with_newline = parse("x = 1\ny = 2").expect("newline-separated should parse");
    assert_eq!(with_semi.statements.len(), with_newline.statements.len());
    assert_eq!(with_semi.statements.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn multiline_strings_are_expression_values() {
    let host = TestHost::default();
    let mut state = State::new();
    let value = finished(
        execute(
            r####"
            submit """first\n"quoted"
second"""
            "####,
            &mut state,
            &host,
        )
        .await
        .expect("program should run"),
    );

    assert_eq!(value, Value::String("first\n\"quoted\"\nsecond".into()));
}

#[tokio::test(flavor = "current_thread")]
async fn raw_multiline_strings_preserve_patch_text() {
    let host = TestHost::default();
    let mut state = State::new();
    let value = finished(
        execute(
            r####"
            patch = r"""*** Begin Patch
*** Update File: src/lib.rs
@@
-old
+new
\n { braces stay raw }
*** End Patch"""
            submit patch
            "####,
            &mut state,
            &host,
        )
        .await
        .expect("program should run"),
    );

    assert_eq!(
        value,
        Value::String(
            "*** Begin Patch\n*** Update File: src/lib.rs\n@@\n-old\n+new\n\\n { braces stay raw }\n*** End Patch"
                .into()
        )
    );
}

#[tokio::test(flavor = "current_thread")]
async fn raw_triple_single_strings_preserve_script_text() {
    let host = TestHost::default();
    let mut state = State::new();
    let value = finished(
        execute(
            r####"
            script = r'''python3 - <<'PY'
print("""hello""")
\n { braces stay raw }
PY'''
            submit script
            "####,
            &mut state,
            &host,
        )
        .await
        .expect("program should run"),
    );

    assert_eq!(
        value,
        Value::String(
            "python3 - <<'PY'\nprint(\"\"\"hello\"\"\")\n\\n { braces stay raw }\nPY".into()
        )
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parser_accepts_comment_only_program() {
    let program = parse(
        r#"
        // comment one
        // comment two
        "#,
    )
    .expect("program should parse");

    assert!(program.statements.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn parser_accepts_inline_trailing_comments_in_blocks() {
    let program = parse(
        r#"
        if true { // enter block
          value = 1 // assign
        } else { // fallback
          value = 2
        }
        submit value // done
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn parser_accepts_else_if_chains() {
    let program = parse(
        r#"
        if false {
          answer = 1
        } else if true {
          answer = 2
        } else {
          answer = 3
        }
        submit answer
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn parser_allows_parallel_in_expression_position() {
    let program = parse(
        r#"
        results = parallel {
          left = call glob { pattern: "src/*.rs" }
          right = call read_file { path: "src/lib.rs" }
        }
        submit results
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn parser_allows_bare_expression_statements() {
    let program = parse(
        r#"
        "branch_a"
        submit "done"
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
    assert!(matches!(program.statements[0], lashlang::Stmt::Expr(_)));
}

#[tokio::test(flavor = "current_thread")]
async fn parser_allows_bare_finish_at_the_end_of_a_block_or_program() {
    let program = parse(
        r#"
        if true {
          submit
        }
        submit
        "#,
    )
    .expect("program should parse");

    assert!(matches!(
        program.statements.as_slice(),
        [
            lashlang::Stmt::If { then_block, .. },
            lashlang::Stmt::Submit(None)
        ] if matches!(then_block.as_slice(), [lashlang::Stmt::Submit(None)])
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn executes_programs_with_double_slash_comments() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        // Create some values first
        total = 6 / 2
        // Return the result
        submit total
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::Number(3.0));
}

#[tokio::test(flavor = "current_thread")]
async fn bare_finish_returns_null() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute("submit", &mut state, &host)
            .await
            .expect("execution should succeed"),
    );

    assert_eq!(value, Value::Null);
}

#[tokio::test(flavor = "current_thread")]
async fn executes_inline_trailing_comments_inside_blocks() {
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
        submit total // final answer
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::Number(3.0));
}

#[tokio::test(flavor = "current_thread")]
async fn double_slash_inside_strings_is_not_a_comment() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        url = "https://example.com/a//b"
        submit url
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::String("https://example.com/a//b".to_string().into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn parser_accepts_ternary_in_call_arguments() {
    let program = parse(
        r#"
        result = format("{}", true ? "yes" : "no")
        submit result
        "#,
    )
    .expect("program should parse");

    assert_eq!(program.statements.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn executes_arithmetic_strings_and_finish() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        total = 1 + 2 * 3
        msg = format("total={}", total)
        submit msg
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("total=7".to_string().into()));
    assert_eq!(state.globals()["total"], Value::Number(7.0));
}

#[tokio::test(flavor = "current_thread")]
async fn executes_if_for_and_list_concat() {
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
        submit result
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("n=1,n=2,n=3,n=4".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn break_exits_loop_and_restores_loop_binding() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        item = "outer"
        seen = []
        for item in [1, 2, 3] {
          if item == 2 {
            break
          }
          seen = seen + [item]
        }
        submit { seen: seen, item: item }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let record = value.as_record().expect("expected record");
    assert_eq!(record["seen"], Value::List(vec![Value::Number(1.0)].into()));
    assert_eq!(record["item"], Value::String("outer".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn continue_skips_to_next_iteration() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        seen = []
        for n in [1, 2, 3, 4] {
          if n == 2 {
            continue
          }
          seen = seen + [n]
        }
        submit seen
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::List(vec![Value::Number(1.0), Value::Number(3.0), Value::Number(4.0)].into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn nested_loop_control_targets_nearest_loop() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        seen = []
        for outer in [1, 2] {
          for inner in [1, 2, 3] {
            if inner == 2 {
              continue
            }
            if inner == 3 {
              break
            }
            seen = seen + [format("{}:{}", outer, inner)]
          }
          seen = seen + [format("outer={}", outer)]
        }
        submit seen
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::List(
            vec![
                Value::String("1:1".to_string().into()),
                Value::String("outer=1".to_string().into()),
                Value::String("2:1".to_string().into()),
                Value::String("outer=2".to_string().into()),
            ]
            .into()
        )
    );
}

#[tokio::test(flavor = "current_thread")]
async fn submit_inside_loop_still_terminates_program() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        for n in [1, 2, 3] {
          submit n
        }
        submit 99
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::Number(1.0));
}

#[tokio::test(flavor = "current_thread")]
async fn ternary_selects_the_correct_branch() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        truthy = true ? "left" : "right"
        falsy = false ? "left" : "right"
        submit format("{}:{}", truthy, falsy)
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("left:right".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn ternary_is_right_associative() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        result = false ? 1 : true ? 2 : 3
        submit result
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::Number(2.0));
}

#[tokio::test(flavor = "current_thread")]
async fn ternary_has_lower_precedence_than_boolean_ops() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        result = false or true ? "yes" : "no"
        submit result
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("yes".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn ternary_short_circuits_unselected_branch() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        yes = true ? "ok" : missing_name
        no = false ? missing_name : "ok"
        submit format("{}:{}", yes, no)
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("ok:ok".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn unary_bang_aliases_not() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        a = !false
        b = !true
        submit [a, b]
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::List(vec![Value::Bool(true), Value::Bool(false)].into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn symbolic_boolean_aliases_match_word_operators() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        a = true && false
        b = false || true
        c = !false && (false || true)
        submit [a, b, c]
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::List(vec![Value::Bool(false), Value::Bool(true), Value::Bool(true)].into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn conditions_and_ternary_use_bounded_truthiness() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        a = 1 ? "yes" : "no"
        b = "" ? "yes" : "no"
        c = !0
        d = ![]
        submit [a, b, c, d]
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn string_concatenation_stringifies_non_string_side() {
    let host = TestHost::default().with_file("src/lib.rs", "pub fn main() {}");
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        found = call read_file { path: "src/lib.rs" }
        submit "status=" + found.ok + " value=" + found.value
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::String("status=true value=pub fn main() {}".to_string().into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn arithmetic_and_string_builtins_coerce_scalars() {
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
        submit {
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
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn to_string_stringifies_records() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        submit to_string({ ok: true, count: 2 })
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(
        value,
        Value::String("{\"count\":2,\"ok\":true}".to_string().into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn observe_captures_intermediate_values_without_ending_execution() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        item = { ok: true, count: 2 }
        print item
        print "step done"
        submit "final"
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn execution_can_continue_without_finish() {
    let host = TestHost::default();
    let mut state = State::new();

    let outcome = execute(
        r#"
        counter = 1
        print counter
        "#,
        &mut state,
        &host,
    )
    .await
    .expect("execution should succeed");

    assert_eq!(outcome, ExecutionOutcome::Continued);
    assert_eq!(state.globals()["counter"], Value::Number(1.0));
    let observed = host.observations.lock().expect("observation mutex");
    assert_eq!(observed.as_slice(), &[Value::Number(1.0)]);
}

#[tokio::test(flavor = "current_thread")]
async fn ternary_fixes_tool_result_formatting_pattern() {
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
        submit summary
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::String(text) = value else {
        panic!("expected string");
    };
    assert!(text.contains("found=ok"));
    assert!(text.contains("missing=failed:"));
}

#[tokio::test(flavor = "current_thread")]
async fn format_supports_indexed_reordering() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        submit format("b={1} a={0}", "x", "y")
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("b=y a=x".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn format_without_placeholders_returns_literal_string() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        submit format("plain")
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("plain".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn format_supports_escaped_braces() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        submit format("{{{}}}", 1)
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("{1}".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn format_rejects_mixed_placeholder_styles_end_to_end() {
    let error = runtime_error(
        r#"
        submit format("{} {1}", "x", "y")
        "#,
    )
    .await;

    assert_eq!(
        error,
        RuntimeError::ValueError {
            message: "can't mix `{}` and indexed format placeholders".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn format_rejects_unused_args_end_to_end() {
    let error = runtime_error(
        r#"
        submit format("plain", 1)
        "#,
    )
    .await;

    assert_eq!(
        error,
        RuntimeError::ValueError {
            message: "format argument `0` is unused".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn format_rejects_unmatched_braces_end_to_end() {
    let open_error = runtime_error(
        r#"
        submit format("{")
        "#,
    )
    .await;
    assert_eq!(
        open_error,
        RuntimeError::ValueError {
            message: "unmatched `{` in format string".to_string()
        }
    );

    let close_error = runtime_error(
        r#"
        submit format("}")
        "#,
    )
    .await;
    assert_eq!(
        close_error,
        RuntimeError::ValueError {
            message: "unmatched `}` in format string".to_string()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn tool_calls_return_result_records() {
    let host = TestHost::default().with_file("src/lib.rs", "pub fn main() {}");
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        found = call read_file { path: "src/lib.rs" }
        missing = call read_file { path: "src/missing.rs" }
        submit { found: found, missing: missing }
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn parallel_executes_concurrently_and_merges_distinct_bindings() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        parallel {
          left = call sleep_echo { value: "a" }
          right = call sleep_echo { value: "b" }
        }
        submit { left: left, right: right }
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn parallel_expression_returns_branch_results_in_order() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        results = parallel {
          left = call sleep_echo { value: "a" }
          right = call sleep_echo { value: "b" }
        }
        submit { results: results, left: left, right: right }
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn named_parallel_expression_returns_record_results() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        results = parallel {
          first: call sleep_echo { value: "a" }
          second: call sleep_echo { value: "b" }
          computed: 40 + 2
        }
        submit {
          first: results.first?,
          second: results.second?,
          computed: results.computed
        }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["first"], Value::String("a".to_string().into()));
    assert_eq!(record["second"], Value::String("b".to_string().into()));
    assert_eq!(record["computed"], Value::Number(42.0));
    assert!(host.max_active.load(Ordering::SeqCst) >= 2);
}

#[tokio::test(flavor = "current_thread")]
async fn parallel_expression_accepts_bare_expression_branches() {
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
        submit results
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn slice_null_bounds_default_to_start_or_end() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        values = [10, 20, 30, 40, 50]
        submit {
          list_tail: slice(values, 3, null),
          list_head: slice(values, null, 2),
          string_tail: slice("abcdef", 4, null),
          string_head: slice("abcdef", null, 2)
        }
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn negative_indices_and_record_contains_are_supported() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        values = [10, 20, 30]
        text = "abc"
        submit {
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
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn dynamic_record_indexing_reads_fields() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        key = "foo"
        record = { foo: 42 }
        submit { found: record[key], missing: record["missing"] }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["found"], Value::Number(42.0));
    assert_eq!(record["missing"], Value::Null);
}

#[tokio::test(flavor = "current_thread")]
async fn indexed_and_field_assignment_update_collections() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        record = {}
        key = "count"
        record[key] = 1
        record.count = record.count + 1
        record.extra = "ok"
        items = [1, 2, 3]
        items[1] = 20
        items[-1] = 30
        submit { record: record, items: items }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    let counts = record["record"]
        .as_record()
        .expect("expected nested record");
    assert_eq!(counts["count"], Value::Number(2.0));
    assert_eq!(counts["extra"], Value::String("ok".into()));
    assert_eq!(
        record["items"],
        Value::List(vec![Value::Number(1.0), Value::Number(20.0), Value::Number(30.0)].into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn nested_path_assignment_and_histogram_loops_work() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        state = { groups: { a: { counts: [1, 2] }, b: { counts: [3] } } }
        g = "a"
        state.groups[g].counts[1] = 5
        counts = {}
        labels = ["a", "b", "a", "c", "b", "a"]
        for label in labels {
          counts[label] = counts[label] + 1
        }
        submit { state: state, counts: counts }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    let state = record["state"].as_record().expect("expected state record");
    let groups = state["groups"].as_record().expect("expected groups record");
    let group_a = groups["a"].as_record().expect("expected group record");
    assert_eq!(
        group_a["counts"],
        Value::List(vec![Value::Number(1.0), Value::Number(5.0)].into())
    );
    let counts = record["counts"]
        .as_record()
        .expect("expected counts record");
    assert_eq!(counts["a"], Value::Number(3.0));
    assert_eq!(counts["b"], Value::Number(2.0));
    assert_eq!(counts["c"], Value::Number(1.0));
}

#[tokio::test(flavor = "current_thread")]
async fn path_assignment_preserves_alias_isolation() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        record = { x: 1, nested: { y: 1 }, items: [1, 2] }
        alias = record
        record.x = 2
        record.nested.y = 3
        record.items[0] = 9
        submit { record: record, alias: alias }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    let updated = record["record"]
        .as_record()
        .expect("expected updated record");
    let alias = record["alias"].as_record().expect("expected alias record");
    assert_eq!(updated["x"], Value::Number(2.0));
    assert_eq!(
        updated["nested"].as_record().unwrap()["y"],
        Value::Number(3.0)
    );
    assert_eq!(
        updated["items"],
        Value::List(vec![Value::Number(9.0), Value::Number(2.0)].into())
    );
    assert_eq!(alias["x"], Value::Number(1.0));
    assert_eq!(
        alias["nested"].as_record().unwrap()["y"],
        Value::Number(1.0)
    );
    assert_eq!(
        alias["items"],
        Value::List(vec![Value::Number(1.0), Value::Number(2.0)].into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn path_assignment_reports_invalid_targets() {
    assert!(matches!(
        runtime_error("items = [1]\nitems[2] = 2").await,
        RuntimeError::ValueError { message } if message.contains("out of bounds")
    ));
    assert!(matches!(
        runtime_error("items = [1]\nitems[0.5] = 2").await,
        RuntimeError::TypeError { message } if message.contains("integer")
    ));
    assert!(matches!(
        runtime_error("items = [1]\nitems[\"0\"] = 2").await,
        RuntimeError::TypeError { message } if message.contains("integer")
    ));
    assert!(matches!(
        runtime_error("text = \"abc\"\ntext[0] = \"x\"").await,
        RuntimeError::TypeError { message } if message.contains("string")
    ));
    assert!(matches!(
        runtime_error("record = {}\nrecord.missing.value = 1").await,
        RuntimeError::ValueError { message } if message.contains("missing field")
    ));
    assert!(matches!(
        runtime_error("record = { item: 1 }\nrecord.item.value = 2").await,
        RuntimeError::TypeError { message } if message.contains("number")
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn parallel_path_assignment_conflicts_on_root_slot() {
    let host = TestHost::default();
    let mut state = State::new();

    let error = execute(
        r#"
        record = {}
        parallel {
          record.a = 1
          record.b = 2
        }
        submit record
        "#,
        &mut state,
        &host,
    )
    .await
    .expect_err("execution should fail");

    assert!(matches!(
        error,
        lashlang::ExecuteError::Runtime(RuntimeError::ParallelConflict { name }) if name == "record"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn else_if_chains_execute_without_extra_braces() {
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
        submit label
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("medium".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn slice_supports_negative_bounds() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        values = [10, 20, 30, 40, 50]
        submit {
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
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn range_and_push_cover_common_collection_building() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        indexes = range(0, 3)
        extended = push(indexes, 3)
        loop_total = 0
        for n in range(0, 4) {
          loop_total = loop_total + n
        }
        submit {
          indexes: indexes,
          extended: extended,
          from_zero: range(3),
          negative: range(-2, 1),
          empty: range(5, 2),
          loop_total: loop_total
        }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(
        record["indexes"],
        Value::List(vec![Value::Number(0.0), Value::Number(1.0), Value::Number(2.0)].into())
    );
    assert_eq!(
        record["extended"],
        Value::List(
            vec![
                Value::Number(0.0),
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0),
            ]
            .into()
        )
    );
    assert_eq!(
        record["from_zero"],
        Value::List(vec![Value::Number(0.0), Value::Number(1.0), Value::Number(2.0)].into())
    );
    assert_eq!(
        record["negative"],
        Value::List(vec![Value::Number(-2.0), Value::Number(-1.0), Value::Number(0.0)].into())
    );
    assert_eq!(record["empty"], Value::List(Vec::new().into()));
    assert_eq!(record["loop_total"], Value::Number(6.0));
}

#[tokio::test(flavor = "current_thread")]
async fn for_loop_assignments_carry_across_iterations() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        raw = split(" x , y , z ", ",")
        parts = []
        count = 0
        snapshots = []
        for part in raw {
          parts = push(parts, trim(part))
          count = count + 1
          snapshots = push(snapshots, { part: trim(part), parts: parts, count: count })
        }
        submit { parts: parts, count: count, snapshots: snapshots }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(
        record["parts"],
        Value::List(
            vec![
                Value::String("x".into()),
                Value::String("y".into()),
                Value::String("z".into()),
            ]
            .into()
        )
    );
    assert_eq!(record["count"], Value::Number(3.0));
    let Value::List(snapshots) = &record["snapshots"] else {
        panic!("expected snapshots list");
    };
    assert_eq!(snapshots.len(), 3);
}

#[tokio::test(flavor = "current_thread")]
async fn named_parallel_accepts_commas_and_keyword_record_keys_execute() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        result = parallel {
          parallel: { parallel: "ok" },
          quoted: { "with space": 2 },
        }
        submit {
          branch: result.parallel.parallel,
          quoted_key: keys(result.quoted)[0],
          quoted_value: values(result.quoted)[0]
        }
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    let Value::Record(record) = value else {
        panic!("expected record");
    };
    assert_eq!(record["branch"], Value::String("ok".into()));
    assert_eq!(record["quoted_key"], Value::String("with space".into()));
    assert_eq!(record["quoted_value"], Value::Number(2.0));
}

#[tokio::test(flavor = "current_thread")]
async fn string_comparisons_are_lexicographic() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        submit {
          lt: "abc" < "def",
          gt: "xyz" > "abc",
          le: "abc" <= "abc",
          ge: "xyz" >= "abc"
        }
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn stringification_preserves_integer_format_inside_containers() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        submit {
          list_text: to_string([1, 2]),
          record_text: to_string({ a: 1, b: 2.5 })
        }
        "#,
            &mut state,
            &host,
        )
        .await
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

#[tokio::test(flavor = "current_thread")]
async fn parallel_rejects_conflicting_assignments() {
    let host = TestHost::default();
    let mut state = State::new();

    let error = execute(
        r#"
        parallel {
          result = call sleep_echo { value: "a" }
          result = call sleep_echo { value: "b" }
        }
        submit result
        "#,
        &mut state,
        &host,
    )
    .await
    .expect_err("execution should fail");

    assert!(matches!(
        error,
        lashlang::ExecuteError::Runtime(RuntimeError::ParallelConflict { .. })
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn optimized_parallel_tool_calls_reject_conflicts_across_three_branches() {
    let host = TestHost::default();
    let mut state = State::new();

    let error = execute(
        r#"
        parallel {
          result = call sleep_echo { value: "a" }
          other = call sleep_echo { value: "b" }
          result = call sleep_echo { value: "c" }
        }
        submit result
        "#,
        &mut state,
        &host,
    )
    .await
    .expect_err("execution should fail");

    assert!(matches!(
        error,
        lashlang::ExecuteError::Runtime(RuntimeError::ParallelConflict { name }) if name == "result"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn optimized_parallel_expression_tool_calls_reject_conflicts_across_three_branches() {
    let host = TestHost::default();
    let mut state = State::new();

    let error = execute(
        r#"
        values = parallel {
          result = call sleep_echo { value: "a" }
          other = call sleep_echo { value: "b" }
          result = call sleep_echo { value: "c" }
        }
        submit values
        "#,
        &mut state,
        &host,
    )
    .await
    .expect_err("execution should fail");

    assert!(matches!(
        error,
        lashlang::ExecuteError::Runtime(RuntimeError::ParallelConflict { name }) if name == "result"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn snapshot_round_trip_preserves_repl_like_state() {
    let host = TestHost::default();
    let mut state = State::new();

    finished(
        execute(
            r#"
        counter = 1
        submit counter
        "#,
            &mut state,
            &host,
        )
        .await
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
        submit counter
        "#,
            &mut restored,
            &host,
        )
        .await
        .expect("restored execution should succeed"),
    );

    assert_eq!(value, Value::Number(2.0));
}

#[tokio::test(flavor = "current_thread")]
async fn json_and_record_helpers_work() {
    let host = TestHost::default();
    let mut state = State::new();

    let value = finished(
        execute(
            r#"
        obj = json_parse("{\"path\":\"src/lib.rs\",\"line\":7}")
        submit format("{}:{}", obj.path, obj.line)
        "#,
            &mut state,
            &host,
        )
        .await
        .expect("execution should succeed"),
    );

    assert_eq!(value, Value::String("src/lib.rs:7".to_string().into()));
}

#[tokio::test(flavor = "current_thread")]
async fn finish_inside_parallel_is_rejected() {
    let host = TestHost::default();
    let mut state = State::new();

    let error = execute(
        r#"
        parallel {
          submit "nope"
        }
        submit "done"
        "#,
        &mut state,
        &host,
    )
    .await
    .expect_err("execution should fail");

    assert!(matches!(
        error,
        lashlang::ExecuteError::Runtime(RuntimeError::FinishInsideParallel)
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn parse_errors_are_surface_level_and_precise() {
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

// ------------------------------------------------------------------
//  End-to-end Type literal integration tests
// ------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn end_to_end_type_value_is_json_schema_shaped() {
    let program = parse(
        r#"
        Books = Type {
          title: str,
          genre: enum["fiction", "non-fiction"],
          tags: list[str],
          meta: Type {
            pages: int,
            published: int
          },
          isbn: str?
        }
        submit Books
        "#,
    )
    .expect("should parse");
    let host = TestHost::default();
    let mut state = State::new();
    let outcome = lashlang::execute_program(&program, &mut state, &host)
        .await
        .expect("should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let schema = lashlang::unwrap_type_value(&value)
        .and_then(Value::as_record)
        .expect("wrapped type");
    assert_eq!(schema["type"], Value::String("object".into()));
    let required = match &schema["required"] {
        Value::List(items) => items,
        _ => panic!("required must be list"),
    };
    // isbn is optional → 4 required
    assert_eq!(required.len(), 4);
}

#[tokio::test(flavor = "current_thread")]
async fn type_is_usable_as_a_tool_call_argument() {
    #[derive(Default)]
    struct CaptureHost {
        captured: std::sync::Mutex<Option<Value>>,
    }
    impl ToolHost for CaptureHost {
        async fn call(&self, _: String, args: Record) -> Result<Value, ToolHostError> {
            *self.captured.lock().unwrap() = args.get("output").cloned();
            Ok(Value::Null)
        }
    }
    let host = CaptureHost::default();
    let program = parse(
        r#"
        Shape = Type { name: str, labels: list[enum["a","b"]] }
        call spawn_agent { task: "find X", output: Shape }
        submit null
        "#,
    )
    .expect("should parse");
    let mut state = State::new();
    lashlang::execute_program(&program, &mut state, &host)
        .await
        .expect("should run");
    let captured = host.captured.lock().unwrap().clone().expect("captured arg");
    let inner = lashlang::unwrap_type_value(&captured).expect("wrapped type");
    let schema = inner.as_record().expect("schema record");
    assert_eq!(schema["type"], Value::String("object".into()));
    let props = schema["properties"].as_record().unwrap();
    let labels = props["labels"].as_record().unwrap();
    assert_eq!(labels["type"], Value::String("array".into()));
    let items = labels["items"].as_record().unwrap();
    let enum_values = match &items["enum"] {
        Value::List(items) => items,
        _ => panic!("enum should be list"),
    };
    assert_eq!(enum_values.len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn validate_reuses_type_literals_for_intermediate_checks() {
    let host = TestHost::default();
    let mut state = State::new();
    let value = finished(
        execute(
            r#"
            raw = {
              name: "lashlang",
              version: "0.2.61",
              labels: ["agent", "runtime"]
            }
            package = validate(raw, Type {
              name: str,
              version: str,
              labels: list[str]
            })
            submit package
            "#,
            &mut state,
            &host,
        )
        .await
        .expect("validate should succeed"),
    );
    let package = value.as_record().expect("package record");
    assert_eq!(
        package["name"],
        Value::String("lashlang".to_string().into())
    );

    let mut state = State::new();
    let err = execute(
        r#"
        submit validate(
          { name: "lashlang", labels: ["agent", 42] },
          Type { name: str, labels: list[str] }
        )
        "#,
        &mut state,
        &host,
    )
    .await
    .expect_err("validate should fail");
    let ExecuteError::Runtime(RuntimeError::ValueError { message }) = err else {
        panic!("expected validation runtime error");
    };
    assert!(
        message.contains("$.labels[1]: expected string, got number"),
        "{message}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn undefined_ref_in_type_produces_runtime_error() {
    let program = parse("submit Type { inner: Missing }").expect("should parse");
    let host = TestHost::default();
    let mut state = State::new();
    let err = lashlang::execute_program(&program, &mut state, &host)
        .await
        .expect_err("Missing is undefined");
    assert!(matches!(err, RuntimeError::UndefinedVariable { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn snapshot_round_trip_preserves_type_values() {
    let program = parse(
        r#"
        Books = Type { title: str, count: int }
        submit Books
        "#,
    )
    .expect("should parse");
    let host = TestHost::default();
    let mut state = State::new();
    let outcome = lashlang::execute_program(&program, &mut state, &host)
        .await
        .expect("should run");
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("expected finish");
    };
    let snapshot = state.snapshot();
    let serialized = serde_json::to_string(&snapshot).expect("serialize");
    let restored: lashlang::Snapshot = serde_json::from_str(&serialized).expect("deserialize");
    let restored_state = State::from_snapshot(restored);
    // Re-execute a program that references Books — the ref should still resolve.
    let program2 = parse("submit Books").expect("parse");
    let mut state2 = restored_state;
    let outcome2 = lashlang::execute_program(&program2, &mut state2, &host)
        .await
        .expect("run");
    let ExecutionOutcome::Finished(v2) = outcome2 else {
        panic!("expected finish");
    };
    assert_eq!(value, v2);
}
