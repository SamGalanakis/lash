use lashlang::{Record, State, ToolHost, ToolHostError, Value};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum Scenario {
    Baseline,
    AsyncAwait,
    DirectUnwrap,
    GeneralParallel,
}

impl Scenario {
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "baseline" => Self::Baseline,
            "async_await" => Self::AsyncAwait,
            "direct_unwrap" => Self::DirectUnwrap,
            "general_parallel" => Self::GeneralParallel,
            _ => return None,
        })
    }
}

impl fmt::Display for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Baseline => "baseline",
            Self::AsyncAwait => "async_await",
            Self::DirectUnwrap => "direct_unwrap",
            Self::GeneralParallel => "general_parallel",
        })
    }
}

pub fn seeded_state() -> State {
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
            record.insert("user".to_string(), Value::String("sam".into()));
            record.insert("attempt".to_string(), Value::Number(3.0));
            record.into()
        }),
    );
    State::from_snapshot(lashlang::Snapshot { globals })
}

pub fn benchmark_program(scenario: Scenario) -> &'static str {
    match scenario {
        Scenario::Baseline => {
            r#"
items = [
  { label: "alpha", weight: 1, active: true },
  { label: "beta", weight: 2, active: false },
  { label: "gamma", weight: 3, active: true }
]
indexes = range(0, len(items))
all_indexes = push(indexes, len(items))
total = 0
labels = []
for item in items {
  total = total + item.weight
  if item.active {
    labels = labels + [format("{0}:{1}", item.label, item.weight)]
  }
}
fanout = parallel {
  lookup: call echo { value: join(labels, ",") }
  stats: call echo { value: { total: total, count: len(items), seen: len(history), index_count: len(all_indexes) } }
}
lookup_value = fanout.lookup?
stats_value = validate(fanout.stats?, Type { total: int, count: int, seen: int, index_count: int })
summary = format(
  "user={0};attempt={1};active={2};total={3};count={4};seen={5};indexes={6}",
  ctx.user,
  ctx.attempt,
  lookup_value,
  stats_value.total,
  stats_value.count,
  stats_value.seen,
  stats_value.index_count
)
submit summary
"#
        }
        Scenario::AsyncAwait => {
            r#"
handles = {
  alpha: start call echo { value: "alpha" },
  beta: start call echo { value: "beta" },
  gamma: start call echo { value: "gamma" }
}
results = await handles
formatted = [results.alpha?, results.beta?, results.gamma?]
submit join(formatted, ",")
"#
        }
        Scenario::DirectUnwrap => {
            r#"
first = (call echo { value: "alpha" })?
second = (call echo { value: format("{0}:{1}", first, "beta") })?
third = (call echo { value: join([first, second], ",") })?
submit third
"#
        }
        Scenario::GeneralParallel => {
            r#"
seed = ["alpha", "beta", "gamma"]
results = parallel {
  left: format("{0}:{1}", seed[0], len(seed))
  right: format("{0}:{1}", seed[1], len(seed))
}
submit format("{0}|{1}", results.left, results.right)
"#
        }
    }
}

pub struct BenchHost;

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
