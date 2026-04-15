use lashlang::{
    ExecutionOutcome, ProfileReport, Record, State, ToolHost, ToolHostError, Value,
    compile_program, parse, profile_compiled,
};
use std::env;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Scenario {
    Baseline,
    AsyncAwait,
}

fn main() {
    let mut args = env::args().skip(1);
    let first = args.next();
    let second = args.next();

    let (scenario, iterations) = match (first.as_deref(), second.as_deref()) {
        (Some("baseline"), maybe_iterations) => (
            Scenario::Baseline,
            maybe_iterations
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000),
        ),
        (Some("async_await"), maybe_iterations) => (
            Scenario::AsyncAwait,
            maybe_iterations
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000),
        ),
        (Some(iterations), None) => (
            Scenario::Baseline,
            iterations.parse::<usize>().ok().unwrap_or(10_000),
        ),
        _ => (Scenario::Baseline, 10_000),
    };

    let source = benchmark_program(scenario);
    let host = BenchHost;
    let program = parse(source).expect("benchmark program should parse");
    let compiled = compile_program(&program);
    let mut profile = ProfileReport::default();

    for _ in 0..iterations {
        let mut state = seeded_state();
        let (outcome, run_profile) =
            profile_compiled(&compiled, &mut state, &host).expect("profiled execution");
        profile.merge(&run_profile);
        let ExecutionOutcome::Finished(_) = outcome else {
            panic!("benchmark program must finish");
        };
    }

    println!("lashlang profile");
    println!("scenario: {}", scenario_name(scenario));
    println!("iterations: {iterations}");
    println!("program_bytes: {}", source.len());
    println!();
    println!("instruction_hotspots:");
    for stat in profile.instruction_stats().iter().take(12) {
        println!(
            "{:<16} count={:<10} total_ms={:<10.3} avg_ns={}",
            stat.name,
            stat.count,
            stat.total_ns as f64 / 1_000_000.0,
            stat.avg_ns()
        );
    }
    println!();
    println!("builtin_hotspots:");
    for stat in profile.builtin_stats().iter().take(12) {
        println!(
            "{:<16} count={:<10} total_ms={:<10.3} avg_ns={}",
            stat.name,
            stat.count,
            stat.total_ns as f64 / 1_000_000.0,
            stat.avg_ns()
        );
    }
}

fn scenario_name(scenario: Scenario) -> &'static str {
    match scenario {
        Scenario::Baseline => "baseline",
        Scenario::AsyncAwait => "async_await",
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

fn benchmark_program(scenario: Scenario) -> &'static str {
    match scenario {
        Scenario::Baseline => {
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
        Scenario::AsyncAwait => {
            r#"
handles = [
  start call echo { value: "alpha" },
  start call echo { value: "beta" },
  start call echo { value: "gamma" }
]
results = await handles
formatted = []
for result in results {
  if result.ok {
    formatted = formatted + [result.value]
  } else {
    formatted = formatted + [result.error]
  }
}
finish join(formatted, ",")
"#
        }
    }
}

struct BenchHost;

impl ToolHost for BenchHost {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        match name {
            "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }

    fn start_call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        match name {
            "echo" => {
                let mut record = Record::default();
                record.insert("__handle__".to_string(), Value::String("task".into()));
                record.insert("tool".to_string(), Value::String(name.to_string().into()));
                record.insert(
                    "value".to_string(),
                    args.get("value").cloned().unwrap_or(Value::Null),
                );
                Ok(Value::Record(Arc::new(record)))
            }
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }

    fn await_handle(&self, handle: &Value) -> Result<Value, ToolHostError> {
        let record = handle
            .as_record()
            .ok_or_else(|| ToolHostError::new("expected handle record"))?;
        Ok(record.get("value").cloned().unwrap_or(Value::Null))
    }
}
