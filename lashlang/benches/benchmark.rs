use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lashlang::{
    ExecutionOutcome, Record, State, ToolHost, ToolHostError, Value, compile_program, execute,
    execute_compiled, parse,
};
use std::hint::black_box;
use std::time::Duration;

fn lashlang_benchmarks(c: &mut Criterion) {
    let source = benchmark_program();
    let host = BenchHost;
    let program = parse(source).expect("benchmark program should parse");
    let compiled = compile_program(&program);

    let mut group = c.benchmark_group("lashlang");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(60);

    group.bench_function(BenchmarkId::new("parse_execute", source.len()), |b| {
        b.iter(|| {
            let mut state = seeded_state();
            let outcome =
                execute(black_box(source), &mut state, &host).expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("execute_only", source.len()), |b| {
        b.iter(|| {
            let mut state = seeded_state();
            let outcome = execute_compiled(black_box(&compiled), &mut state, &host)
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("snapshot_execute", source.len()), |b| {
        b.iter(|| {
            let mut state = seeded_state();
            let snapshot = state.snapshot();
            let encoded = serde_json::to_vec(&snapshot).expect("snapshot encode");
            let decoded = serde_json::from_slice(&encoded).expect("snapshot decode");
            state = State::from_snapshot(decoded);
            let outcome = execute_compiled(black_box(&compiled), &mut state, &host)
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.finish();
}

fn expect_finished(outcome: ExecutionOutcome) -> Value {
    match outcome {
        ExecutionOutcome::Finished(value) => value,
        ExecutionOutcome::Continued => panic!("benchmark program must finish"),
    }
}

fn seeded_state() -> State {
    let mut globals = Record::default();
    globals.insert(
        "history".to_string(),
        Value::List(
            vec![
                Value::String("alpha".to_string().into()),
                Value::String("beta".to_string().into()),
                Value::String("gamma".to_string().into()),
            ]
            .into(),
        ),
    );
    globals.insert(
        "ctx".to_string(),
        Value::Record({
            let mut record = Record::default();
            record.insert("user".to_string(), Value::String("sam".to_string().into()));
            record.insert("attempt".to_string(), Value::Number(3.0));
            record.into()
        }),
    );
    State::from_snapshot(lashlang::Snapshot { globals })
}

fn benchmark_program() -> &'static str {
    r#"
items = [
  { label: "alpha", weight: 1, active: true },
  { label: "beta", weight: 2, active: false },
  { label: "gamma", weight: 3, active: true }
]
total = 0
labels = []
for item in items {
  total = total + item.weight
  if item.active {
    labels = labels + [format("{0}:{1}", item.label, item.weight)]
  }
}
parallel {
  lookup = call echo { value: join(labels, ",") }
  stats = call echo { value: { total: total, count: len(items), seen: len(history) } }
}
summary = format(
  "user={0};attempt={1};active={2};total={3};count={4};seen={5}",
  ctx.user,
  ctx.attempt,
  lookup.value,
  stats.value.total,
  stats.value.count,
  stats.value.seen
)
finish summary
"#
}

struct BenchHost;

impl ToolHost for BenchHost {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        match name {
            "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }
}

criterion_group!(benches, lashlang_benchmarks);
criterion_main!(benches);
