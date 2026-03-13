use lashlang::{
    ExecutionOutcome, ProfileReport, Record, State, ToolHost, ToolHostError, Value,
    compile_program, parse, profile_compiled,
};
use std::env;

fn main() {
    let iterations = env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10_000);

    let source = benchmark_program();
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
