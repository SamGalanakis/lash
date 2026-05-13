use compact_str::ToCompactString;
use lashlang::{
    ProjectedBindings, ProjectedFuture, ProjectedHostValue, ProjectedRead, ProjectedValue, Record,
    State, ToolHost, ToolHostCall, ToolHostError, Value,
};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum Scenario {
    Baseline,
    LanguageSurface,
    AsyncAwait,
    DirectUnwrap,
    GeneralParallel,
    LoopControl,
    IndexedAssignment,
    ProjectedValues,
}

impl Scenario {
    pub const ALL: &'static [Self] = &[
        Self::Baseline,
        Self::LanguageSurface,
        Self::AsyncAwait,
        Self::DirectUnwrap,
        Self::GeneralParallel,
        Self::LoopControl,
        Self::IndexedAssignment,
        Self::ProjectedValues,
    ];

    #[allow(dead_code)]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "baseline" => Self::Baseline,
            "language_surface" => Self::LanguageSurface,
            "async_await" => Self::AsyncAwait,
            "direct_unwrap" => Self::DirectUnwrap,
            "general_parallel" => Self::GeneralParallel,
            "loop_control" => Self::LoopControl,
            "indexed_assignment" => Self::IndexedAssignment,
            "projected_values" => Self::ProjectedValues,
            _ => return None,
        })
    }

    #[allow(dead_code)]
    pub fn expected_values() -> &'static str {
        "baseline, language_surface, async_await, direct_unwrap, general_parallel, loop_control, indexed_assignment, projected_values, or all"
    }
}

impl fmt::Display for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Baseline => "baseline",
            Self::LanguageSurface => "language_surface",
            Self::AsyncAwait => "async_await",
            Self::DirectUnwrap => "direct_unwrap",
            Self::GeneralParallel => "general_parallel",
            Self::LoopControl => "loop_control",
            Self::IndexedAssignment => "indexed_assignment",
            Self::ProjectedValues => "projected_values",
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
        Scenario::LanguageSurface => {
            r#"
source = join(history, ",")
tokens = split(source, ",")
trimmed_user = trim(format(" {0} ", ctx.user))
beta_index = find(source, "beta")
line_matches = grep_text(join(tokens, "\n"), "a")
count = len(tokens)
empty_tail = empty(slice(tokens, count, count))
predicates = [
  contains(source, tokens[1]),
  starts_with(source, tokens[0]),
  ends_with(source, tokens[2])
]
numeric = {
  neg: -ctx.attempt,
  sum: ctx.attempt + count,
  diff: count - 1,
  product: count * 2,
  quotient: count / 2,
  modulo: count % 2,
  parsed_int: to_int(ctx.attempt),
  parsed_float: to_float(ctx.attempt)
}
logic = not false and (count > 2 or empty_tail)
comparisons = [
  count == 3,
  count != 4,
  count < 4,
  count <= 3,
  count > 2,
  count >= 3
]
choice = logic ? "yes" : "no"
json_text = "{\"attempt\":" + to_string(ctx.attempt) + ",\"ok\":true}"
parsed = json_parse(json_text)
positional = parallel {
  format("left:{0}", tokens[0]),
  call echo { value: tokens[1] },
  len(tokens)
}
named = parallel {
  lookup: call echo { value: source }
  summary: format("{0}:{1}", trimmed_user, count)
}
cancelled = start call echo { value: "cancelled" }
cancel cancelled
handle = start call echo { value: "awaited" }
awaited = (await handle)?
direct = (call echo { value: trimmed_user })?
print direct
state = {
  tags: push(tokens, "delta"),
  counts: {},
  kept: [],
  predicates: predicates,
  comparisons: comparisons,
  numeric: numeric,
  parsed: parsed,
  line_matches: line_matches
}
state.tags[1] = "beta"
for token in state.tags {
  state.counts[token] = state.counts[token] + 1
}
for token in keys(state.counts) {
  if token == "beta" {
    continue
  }
  if token == "delta" {
    break
  }
  state.kept = push(state.kept, token)
}
Payload = Type {
  user: str,
  choice: enum["yes", "no"],
  tags: list[str],
  counts: dict,
  kept: list[str],
  maybe: str | null,
  optional_note: str?
}
validated = validate(
  {
    user: direct,
    choice: choice,
    tags: state.tags,
    counts: state.counts,
    kept: state.kept,
    maybe: null
  },
  Payload
)
submit {
  direct: direct,
  awaited: awaited,
  positional: positional,
  lookup: named.lookup?,
  summary: named.summary,
  values: values(state.counts),
  beta_index: beta_index,
  line_matches: line_matches,
  first_two: slice(state.tags, null, 2),
  validated: validated,
  stringified: to_string(validated)
}
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
        Scenario::LoopControl => {
            r#"
items = range(0, 128)
outer = "restored"
kept = 0
skipped = 0
for outer in items {
  if outer < 32 {
    skipped = skipped + 1
    continue
  }
  if outer >= 96 {
    break
  }
  if outer % 3 == 0 {
    continue
  }
  kept = kept + 1
}
submit { kept: kept, skipped: skipped, outer: outer }
"#
        }
        Scenario::IndexedAssignment => {
            r#"
groups = ["alpha", "beta", "alpha", "gamma", "beta", "alpha", "delta", "gamma"]
counts = {}
for group in groups {
  counts[group] = counts[group] + 1
}
state = { groups: { alpha: { count: 0 }, beta: { count: 0 }, gamma: { count: 0 }, delta: { count: 0 } } }
for group in keys(counts) {
  state.groups[group].count = counts[group]
}
summary = []
for group in keys(counts) {
  summary = summary + [format("{0}:{1}", group, state.groups[group].count)]
}
submit { counts: counts, state: state, summary: join(summary, ",") }
"#
        }
        Scenario::ProjectedValues => {
            r#"
first = history[0]
second = history[1]
body_head = docs.body[0]
body_match_index = find(docs.body, "markdown")
body_matches = grep_text(docs.body, "markdown")
second_matches = grep_text(second.content, "response")
if docs.body {
  body_truthy = true
} else {
  body_truthy = false
}
print docs.body
summary = {
  history_len: len(history),
  first_role: first.role,
  first_content: first.content,
  second_content: second.content,
  doc_title: docs.title,
  doc_summary: docs.summary,
  body_head: body_head,
  body_match_index: body_match_index,
  body_matches: body_matches,
  second_matches: second_matches,
  body_truthy: body_truthy,
  body_text: docs.body
}
submit summary
"#
        }
    }
}

pub fn projected_bindings(scenario: Scenario) -> ProjectedBindings {
    let mut bindings = ProjectedBindings::new();
    if !matches!(scenario, Scenario::ProjectedValues) {
        return bindings;
    }
    bindings.insert(
        "history",
        ProjectedValue::custom("history", Arc::new(ProjectedList::history())),
    );
    bindings.insert(
        "docs",
        ProjectedValue::scalar("docs", projected_docs_record()),
    );
    bindings
}

fn projected_docs_record() -> Value {
    let mut record = Record::default();
    record.insert("title".to_string(), Value::String("Authoring Rules".into()));
    record.insert(
        "summary".to_string(),
        Value::String("Rules for editing workflow JSON drafts.".into()),
    );
    record.insert(
        "body".to_string(),
        Value::Projected(ProjectedValue::custom(
            "docs.body",
            Arc::new(ProjectedText::new("body", "lazy markdown body")),
        )),
    );
    Value::Record(Arc::new(record))
}

struct ProjectedList {
    name: &'static str,
    values: Arc<[Value]>,
}

impl ProjectedList {
    fn history() -> Self {
        Self {
            name: "history",
            values: vec![
                history_item("user", "alpha request"),
                history_item("assistant", "beta response"),
                history_item("tool", "gamma observation"),
            ]
            .into(),
        }
    }
}

impl ProjectedHostValue for ProjectedList {
    fn type_name(&self) -> &str {
        "list"
    }

    fn len(&self) -> ProjectedFuture<'_, Option<usize>> {
        Box::pin(async { Some(self.values.len()) })
    }

    fn get_index(&self, index: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async move {
            let Some(index) = resolve_index(&index, self.values.len()) else {
                return ProjectedRead::Missing;
            };
            self.values
                .get(index)
                .cloned()
                .map(ProjectedRead::Value)
                .unwrap_or(ProjectedRead::Missing)
        })
    }

    fn render(&self) -> ProjectedFuture<'_, String> {
        Box::pin(async move { format!("<{}:{}>", self.name, self.values.len()) })
    }

    fn materialize(&self) -> ProjectedFuture<'_, Value> {
        Box::pin(async move { Value::List(self.values.clone()) })
    }
}

fn history_item(role: &str, content: &str) -> Value {
    let mut record = Record::default();
    record.insert("role".to_string(), Value::String(role.to_string().into()));
    record.insert(
        "content".to_string(),
        Value::Projected(ProjectedValue::custom(
            format!("history.{role}.content"),
            Arc::new(ProjectedText::new("content", content)),
        )),
    );
    Value::Record(Arc::new(record))
}

struct ProjectedText {
    name: &'static str,
    text: Arc<str>,
}

impl ProjectedText {
    fn new(name: &'static str, text: impl Into<Arc<str>>) -> Self {
        Self {
            name,
            text: text.into(),
        }
    }
}

impl ProjectedHostValue for ProjectedText {
    fn type_name(&self) -> &str {
        "string"
    }

    fn len(&self) -> ProjectedFuture<'_, Option<usize>> {
        Box::pin(async { Some(self.text.chars().count()) })
    }

    fn get_index(&self, index: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async move {
            let Some(index) = resolve_index(&index, self.text.chars().count()) else {
                return ProjectedRead::Missing;
            };
            self.text
                .chars()
                .nth(index)
                .map(|ch| ProjectedRead::Value(Value::String(ch.to_compact_string())))
                .unwrap_or(ProjectedRead::Missing)
        })
    }

    fn find(&self, needle: Value, start: usize) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async move {
            let Value::String(needle) = needle else {
                return ProjectedRead::Missing;
            };
            ProjectedRead::Value(match find_text(&self.text, &needle, start) {
                Some(index) => Value::Number(index as f64),
                None => Value::Null,
            })
        })
    }

    fn grep_text(&self, needle: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async move {
            let Value::String(needle) = needle else {
                return ProjectedRead::Missing;
            };
            grep_text_records(&self.text, &needle)
                .map(ProjectedRead::Value)
                .unwrap_or(ProjectedRead::Missing)
        })
    }

    fn render(&self) -> ProjectedFuture<'_, String> {
        Box::pin(async move { format!("<{}:{} chars>", self.name, self.text.chars().count()) })
    }

    fn materialize(&self) -> ProjectedFuture<'_, Value> {
        Box::pin(async move { Value::String(self.text.as_ref().into()) })
    }
}

fn grep_text_records(text: &str, needle: &str) -> Option<Value> {
    if needle.is_empty() {
        return None;
    }

    let needle_len = needle.chars().count();
    let needle_value = Value::String(needle.into());
    let mut matches = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let Some(start) = find_text(line, needle, 0) else {
            continue;
        };
        let mut record = Record::with_capacity(5);
        record.insert_str("line", Value::Number((line_index + 1) as f64));
        record.insert_str("text", Value::String(line.into()));
        record.insert_str("match", needle_value.clone());
        record.insert_str("start", Value::Number(start as f64));
        record.insert_str("end", Value::Number((start + needle_len) as f64));
        matches.push(Value::Record(Arc::new(record)));
    }
    Some(Value::List(matches.into()))
}

fn find_text(text: &str, needle: &str, start: usize) -> Option<usize> {
    let start_byte = if start == 0 {
        0
    } else {
        byte_index_for_char(text, start)?
    };
    if needle.is_empty() {
        return Some(start);
    }
    let tail = &text[start_byte..];
    let match_byte = tail.find(needle)?;
    Some(start + tail[..match_byte].chars().count())
}

fn byte_index_for_char(text: &str, target: usize) -> Option<usize> {
    let mut char_count = 0;
    for (byte_index, _) in text.char_indices() {
        if char_count == target {
            return Some(byte_index);
        }
        char_count += 1;
    }
    (char_count == target).then_some(text.len())
}

fn resolve_index(index: &Value, len: usize) -> Option<usize> {
    let Value::Number(index) = index else {
        return None;
    };
    if !index.is_finite() || index.fract() != 0.0 {
        return None;
    }
    let len = len as isize;
    let index = *index as isize;
    let normalized = if index < 0 { len + index } else { index };
    (0..len)
        .contains(&normalized)
        .then_some(normalized as usize)
}

pub struct BenchHost;

impl ToolHost for BenchHost {
    async fn call(&self, name: String, args: Record) -> Result<Value, ToolHostError> {
        bench_call(&name, &args)
    }

    async fn call_batch(&self, calls: Vec<ToolHostCall>) -> Vec<Result<Value, ToolHostError>> {
        calls
            .into_iter()
            .map(|call| bench_call(&call.name, &call.args))
            .collect()
    }

    async fn start_call(&self, name: String, args: Record) -> Result<Value, ToolHostError> {
        Self::task_handle(&name, &args)
    }

    async fn await_handle(&self, handle: Value) -> Result<Value, ToolHostError> {
        let record = handle
            .as_record()
            .ok_or_else(|| ToolHostError::new("expected handle record"))?;
        Ok(record.get("value").cloned().unwrap_or(Value::Null))
    }

    async fn cancel_handle(&self, handle: Value) -> Result<Value, ToolHostError> {
        Ok(handle)
    }
}

fn bench_call(name: &str, args: &Record) -> Result<Value, ToolHostError> {
    match name {
        "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
        _ => Err(unknown_tool(name)),
    }
}

fn unknown_tool(name: &str) -> ToolHostError {
    ToolHostError::new(format!("unknown tool: {name}"))
}

impl BenchHost {
    fn task_handle(name: &str, args: &Record) -> Result<Value, ToolHostError> {
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
            _ => Err(unknown_tool(name)),
        }
    }
}
