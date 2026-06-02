use compact_str::ToCompactString;
use lashlang::{
    AbilityOp, AbilityResult, ExecutionHost, ExecutionHostError, ImageValue, LASH_HOST_VALUE_KEY,
    LASH_HOST_VALUE_TYPE_KEY, LASH_PROCESS_NAME_KEY, LashlangAbilities, LashlangSurface,
    LinkedModule, ListValue, ProjectedBindings, ProjectedFuture, ProjectedHostValue,
    ProjectedReadRequest, ProjectedReadResponse, ProjectedValue, Record, ResourceCatalog, State,
    TypeExpr, TypeField, Value, from_json,
};
use std::fmt;
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug)]
pub enum Scenario {
    Baseline,
    LanguageSurface,
    AsyncAwait,
    DirectUnwrap,
    GeneralFanout,
    LoopControl,
    IndexedAssignment,
    ProjectedValues,
    LargeData,
    CachePressure,
    ProjectedOperations,
    TypeSystemStress,
    WrappedErrorPaths,
    ToolControlSurface,
    SnapshotProjectedState,
    ContinueAsSeedSurface,
    TriggerRegistrySurface,
    SyntaxTextSurface,
    IntegerRangeSurface,
    FanoutExpressionSurface,
    ImageSurface,
}

impl Scenario {
    pub const ALL: &'static [Self] = &[
        Self::Baseline,
        Self::LanguageSurface,
        Self::AsyncAwait,
        Self::DirectUnwrap,
        Self::GeneralFanout,
        Self::LoopControl,
        Self::IndexedAssignment,
        Self::ProjectedValues,
        Self::LargeData,
        Self::CachePressure,
        Self::ProjectedOperations,
        Self::TypeSystemStress,
        Self::WrappedErrorPaths,
        Self::ToolControlSurface,
        Self::SnapshotProjectedState,
        Self::ContinueAsSeedSurface,
        Self::TriggerRegistrySurface,
        Self::SyntaxTextSurface,
        Self::IntegerRangeSurface,
        Self::FanoutExpressionSurface,
        Self::ImageSurface,
    ];

    #[allow(dead_code)]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "baseline" => Self::Baseline,
            "language_surface" => Self::LanguageSurface,
            "async_await" => Self::AsyncAwait,
            "direct_unwrap" => Self::DirectUnwrap,
            "general_fanout" => Self::GeneralFanout,
            "loop_control" => Self::LoopControl,
            "indexed_assignment" => Self::IndexedAssignment,
            "projected_values" => Self::ProjectedValues,
            "large_data" => Self::LargeData,
            "cache_pressure" => Self::CachePressure,
            "projected_operations" => Self::ProjectedOperations,
            "type_system_stress" => Self::TypeSystemStress,
            "wrapped_error_paths" => Self::WrappedErrorPaths,
            "tool_control_surface" => Self::ToolControlSurface,
            "snapshot_projected_state" => Self::SnapshotProjectedState,
            "continue_as_seed_surface" => Self::ContinueAsSeedSurface,
            "trigger_registry_surface" => Self::TriggerRegistrySurface,
            "syntax_text_surface" => Self::SyntaxTextSurface,
            "integer_range_surface" => Self::IntegerRangeSurface,
            "fanout_expression_surface" => Self::FanoutExpressionSurface,
            "image_surface" => Self::ImageSurface,
            _ => return None,
        })
    }

    #[allow(dead_code)]
    pub fn expected_values() -> &'static str {
        "baseline, language_surface, async_await, direct_unwrap, general_fanout, loop_control, indexed_assignment, projected_values, large_data, cache_pressure, projected_operations, type_system_stress, wrapped_error_paths, tool_control_surface, snapshot_projected_state, continue_as_seed_surface, trigger_registry_surface, syntax_text_surface, integer_range_surface, fanout_expression_surface, image_surface, or all"
    }
}

impl fmt::Display for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Baseline => "baseline",
            Self::LanguageSurface => "language_surface",
            Self::AsyncAwait => "async_await",
            Self::DirectUnwrap => "direct_unwrap",
            Self::GeneralFanout => "general_fanout",
            Self::LoopControl => "loop_control",
            Self::IndexedAssignment => "indexed_assignment",
            Self::ProjectedValues => "projected_values",
            Self::LargeData => "large_data",
            Self::CachePressure => "cache_pressure",
            Self::ProjectedOperations => "projected_operations",
            Self::TypeSystemStress => "type_system_stress",
            Self::WrappedErrorPaths => "wrapped_error_paths",
            Self::ToolControlSurface => "tool_control_surface",
            Self::SnapshotProjectedState => "snapshot_projected_state",
            Self::ContinueAsSeedSurface => "continue_as_seed_surface",
            Self::TriggerRegistrySurface => "trigger_registry_surface",
            Self::SyntaxTextSurface => "syntax_text_surface",
            Self::IntegerRangeSurface => "integer_range_surface",
            Self::FanoutExpressionSurface => "fanout_expression_surface",
            Self::ImageSurface => "image_surface",
        })
    }
}

#[allow(dead_code)]
pub fn seeded_state() -> State {
    seeded_state_for(Scenario::Baseline)
}

pub fn seeded_state_for(scenario: Scenario) -> State {
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
    if matches!(scenario, Scenario::SnapshotProjectedState) {
        globals.insert("snap".to_string(), snapshot_projected_record());
    }
    if matches!(scenario, Scenario::ImageSurface) {
        globals.insert(
            "img".to_string(),
            Value::Image(ImageValue::new(
                "img-1",
                "chart.png",
                1234,
                Some(640),
                Some(480),
            )),
        );
    }
    State::from_snapshot(lashlang::Snapshot { globals })
}

const BENCH_PROCESS_DECLS: &str = r#"
process echo(value: any) {
  finish value
}

process spawn_child(task: str, capability: str) {
  finish { claim: format("done:{0}", task) }
}

process query_llm(prompt: str, model: str) {
  finish { text: "benchmark summary", tokens: 42 }
}
"#;

pub fn benchmark_program(scenario: Scenario) -> String {
    let main = match scenario {
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
lookup_handle = start echo(value: join(labels, ","))
stats_handle = start echo(value: { total: total, count: len(items), seen: len(history), index_count: len(all_indexes) })
fanout = await { lookup: lookup_handle, stats: stats_handle }
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
positional_results = await [
  start echo(value: format("left:{0}", tokens[0])),
  start echo(value: tokens[1]),
  start echo(value: len(tokens))
]
positional = [positional_results[0]?, positional_results[1], positional_results[2]?]
named_results = await {
  lookup: start echo(value: source),
  summary: start echo(value: format("{0}:{1}", trimmed_user, count))
}
named = { lookup: named_results.lookup, summary: named_results.summary? }
cancelled = start echo(value: "cancelled")
cancel cancelled
handle = start echo(value: "awaited")
awaited = (await handle)?
direct = await tools.echo({ value: trimmed_user })?
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
  alpha: start echo(value: "alpha"),
  beta: start echo(value: "beta"),
  gamma: start echo(value: "gamma")
}
results = await handles
formatted = [results.alpha?, results.beta?, results.gamma?]
submit join(formatted, ",")
"#
        }
        Scenario::DirectUnwrap => {
            r#"
first = await tools.echo({ value: "alpha" })?
second = await tools.echo({ value: format("{0}:{1}", first, "beta") })?
third = await tools.echo({ value: join([first, second], ",") })?
submit third
"#
        }
        Scenario::GeneralFanout => {
            r#"
seed = ["alpha", "beta", "gamma"]
results = await {
  left: start echo(value: format("{0}:{1}", seed[0], len(seed))),
  right: start echo(value: format("{0}:{1}", seed[1], len(seed)))
}
submit format("{0}|{1}", results.left?, results.right?)
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
        Scenario::LargeData => {
            r#"
items = range(0, 512)
groups = {}
total = 0
evens = []
odds = []
for item in items {
  key = format("bucket_{0}", item % 16)
  groups[key] = groups[key] + 1
  total = total + item
  if item % 2 == 0 {
    evens = push(evens, item)
  } else {
    odds = push(odds, item)
  }
}
lines = []
for key in keys(groups) {
  lines = push(lines, format("{0}:{1}", key, groups[key]))
}
payload = {
  count: len(items),
  total: total,
  groups: groups,
  evens: len(evens),
  odds: len(odds),
  summary: join(lines, "|")
}
submit validate(payload, Type {
  count: int,
  total: int,
  groups: dict,
  evens: int,
  odds: int,
  summary: str
})
"#
        }
        Scenario::CachePressure => {
            r#"
seed = {
  user: ctx.user,
  attempt: ctx.attempt,
  history_len: len(history),
  labels: ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
}
a0 = format("{0}:{1}", seed.labels[0], seed.attempt)
a1 = format("{0}:{1}", seed.labels[1], seed.history_len)
a2 = format("{0}:{1}", seed.labels[2], len(a0))
a3 = format("{0}:{1}", seed.labels[3], len(a1))
a4 = format("{0}:{1}", seed.labels[4], len(a2))
a5 = format("{0}:{1}", seed.labels[5], len(a3))
r0 = { name: "r0", value: a0, next: a1 }
r1 = { name: "r1", value: a1, next: a2 }
r2 = { name: "r2", value: a2, next: a3 }
r3 = { name: "r3", value: a3, next: a4 }
r4 = { name: "r4", value: a4, next: a5 }
r5 = { name: "r5", value: a5, next: a0 }
TypeA = Type { name: str, value: str, next: str }
validated = [
  validate(r0, TypeA),
  validate(r1, TypeA),
  validate(r2, TypeA),
  validate(r3, TypeA),
  validate(r4, TypeA),
  validate(r5, TypeA)
]
submit join([
  validated[0].value,
  validated[1].value,
  validated[2].value,
  validated[3].value,
  validated[4].value,
  validated[5].value
], "|")
"#
        }
        Scenario::ProjectedOperations => {
            r#"
summary = {
  len: len(proj.items),
  empty: empty(proj.items),
  keys: keys(proj.record),
  values: values(proj.record),
  contains: contains(proj.items, "beta"),
  starts: starts_with(proj.text, "alpha"),
  ends: ends_with(proj.text, "delta"),
  split_count: len(split(proj.text, " ")),
  join: join(proj.items, ","),
  trim: trim(proj.padded),
  slice_text: slice(proj.text, 6, 10),
  slice_list: slice(proj.items, 1, 3),
  pushed: push(proj.items, "epsilon"),
  as_int: to_int(proj.number),
  as_float: to_float(proj.number),
  parsed: json_parse(proj.json),
  first: proj.items[0],
  field: proj.record.topic
}
submit summary
"#
        }
        Scenario::TypeSystemStress => {
            r#"
Meta = Type {
  source: str,
  attempt: int,
  tags: list[enum["alpha", "beta", "gamma", "delta"]],
  optional_note: str?
}
Item = Type {
  id: int,
  title: str,
  score: float,
  active: bool,
  meta: Meta,
  maybe: str | int | null
}
items = []
for i in range(0, 64) {
  raw = {
    id: i,
    title: format("item-{0}", i),
    score: i / 2,
    active: i % 2 == 0,
    meta: {
      source: ctx.user,
      attempt: ctx.attempt,
      tags: ["alpha", "beta", "gamma"]
    },
    maybe: i % 3 == 0 ? null : format("v{0}", i)
  }
  items = push(items, validate(raw, Item))
}
submit {
  count: len(items),
  first: items[0].title,
  last: items[63].title,
  tags: join(items[1].meta.tags, ",")
}
"#
        }
        Scenario::WrappedErrorPaths => {
            r#"
missing = await tools.missing_tool({ value: "x" })
boom = await tools.boom({ reason: "explicit" })
ok = await tools.echo({ value: "still-running" })
probe = await shell.exec({ cmd: "test -f Cargo.lock", allow_nonzero_exit: true })
submit {
  missing_ok: missing.ok,
  missing_error: contains(missing.error, "unknown tool"),
  boom_ok: boom.ok,
  boom_error: contains(boom.error, "explicit failure"),
  ok_value: ok.value,
  probe_exit: probe.value.exit_code,
  probe_done: probe.value.done
}
"#
        }
        Scenario::ToolControlSurface => {
            r#"
first = start spawn_child(task: "inspect auth", capability: "explore")
second = start spawn_child(task: "inspect api", capability: "explore")
llm = start query_llm(prompt: "summarize benchmark", model: "gpt-5.4-mini")
probe = start echo(value: "app log")
handles = await processes.list({})?
results = await {
  first: first,
  second: second,
  llm: llm,
  probe: probe
}
cancel second
submit {
  first: results.first.value.claim,
  second: results.second.value.claim,
  llm: results.llm.value.text,
  probe: results.probe.value,
  tools: len(handles)
}
"#
        }
        Scenario::SnapshotProjectedState => {
            r#"
head = slice(snap.projected.body, 0, 16)
materialized = to_string(snap.projected.body)
nested_head = snap.mixed.nested.projected_title
snap.mixed.count = snap.mixed.count + 1
submit {
  id: snap.id,
  normal_title: snap.normal.title,
  head: head,
  materialized_len: len(materialized),
  nested_head: nested_head,
  count: snap.mixed.count,
  tags: join(snap.normal.tags, ",")
}
"#
        }
        Scenario::ContinueAsSeedSurface => {
            r#"
agent = start spawn_child(task: "inspect carry-forward", capability: "explore")
handles = await processes.list({})?
frame = await control.continue_as({
  task: "continue from compact state",
    seed: {
      projected_problem: proj.text,
      nested_projected: { body: proj.json },
      computed_summary: format("{0}:{1}", ctx.user, len(history)),
      live_agent: handles[0],
      started_agent: agent
    }
  })?
submit {
  frame_id: frame.frame_id,
  task: frame.task,
  seed_keys: frame.seed_keys,
  projected_count: frame.projected_count,
  global_count: frame.global_count
}
"#
        }
        Scenario::TriggerRegistrySurface => {
            r#"
type ButtonPressed = { button: "Red" | "Blue", message: str, pressed_at: str }

process daily_digest(tick: cron.Tick) {
  finish { kind: "daily_digest", id: tick.id }
}

process on_button(event: ButtonPressed) {
  finish { kind: "button", button: event.button }
}

daily_handle = await triggers.register({
  source: cron.Schedule({ expr: "0 8 * * *", tz: "UTC" }),
  target: daily_digest,
  name: "daily_digest"
})?
button_handle = await triggers.register({
  source: ui.button.pressed({}),
  target: on_button,
  name: "button watcher"
})?
registrations = await triggers.list({ target: daily_digest })?
cancelled = await triggers.cancel({ handle: daily_handle })?
submit {
  daily_handle: daily_handle.id,
  button_handle: button_handle.id,
  registration_count: len(registrations),
  listed_target: registrations[0].target.process_name,
  listed_source: registrations[0].source_type,
  cancelled: cancelled
}
"#
        }
        Scenario::SyntaxTextSurface => {
            r####"
// Exercise parser-heavy string forms, comments, semicolon recovery, and text builtins.
patch = r"""*** Begin Patch
*** Update File: crates/lashlang/src/lib.rs
@@
-old
+new
\n { braces stay raw }
*** End Patch""";
script = r'''python3 - <<'PY'
print("""double quotes are preserved""")
\n { braces stay raw }
PY''';
plain = """first
"quoted"
second""";
"bare expression branch"
pieces = [
  len(patch),
  contains(patch, "*** Begin Patch"),
  starts_with(script, "python3"),
  ends_with(trim(script), "PY"),
  len(split(plain, "\n")),
  slice(plain, 0, 5)
]
submit {
  patch_head: slice(patch, 0, 15),
  script_head: slice(script, 0, 7),
  plain_lines: len(split(plain, "\n")),
  pieces: pieces
}
"####
        }
        Scenario::IntegerRangeSurface => {
            r#"
items = range(-8, 9)
forward = range(0, 10, 3)
backward = range(7, -3, -2)
stride = ceil_div(len(items), 4)
starts = []
windows = []
for i in range(0, len(items), stride) {
  starts = push(starts, i)
  windows = push(windows, slice(items, i, i + stride))
}
text = "alpha beta gamma beta delta"
first_beta = find(text, "beta")
second_beta = find(text, "beta", first_beta + 1)
submit {
  count: len(items),
  first: items[0],
  last: items[-1],
  forward: forward,
  backward: backward,
  stride: stride,
  starts: starts,
  windows: windows,
  mid: slice(items, 2, -2),
  head: slice(text, null, 5),
  tail: slice(text, -5, null),
  first_beta: first_beta,
  second_beta: second_beta,
  ceil_neg: ceil_div(-10, 3),
  floor_neg: floor_div(-10, 3)
}
"#
        }
        Scenario::FanoutExpressionSurface => {
            r#"
left = await tools.echo({ value: "left" })
right = await tools.echo({ value: "right" })
computed = len(history) + 39
discarded = ["branch_a", 40 + 2, len(history)]
batched_results = await {
  first: start echo(value: left.value),
  second: start echo(value: right.value)
}
batched = { first: batched_results.first, second: batched_results.second, computed: computed }
submit {
  left: left.value,
  right: right.value,
  computed: computed,
  discarded: discarded,
  first: batched.first?,
  second: batched.second?,
  batched_computed: batched.computed
}
"#
        }
        Scenario::ImageSurface => {
            r#"
descriptor = to_string(img)
metadata = {
  id: img.id,
  label: img.label,
  size: img.size,
  width: img.width,
  height: img.height,
  missing: img.missing
}
print img
submit {
  metadata: metadata,
  descriptor_has_type: contains(descriptor, "\"type\":\"image\""),
  descriptor_has_id: contains(descriptor, "\"id\":\"img-1\""),
  dims: format("{0}x{1}", img.width, img.height),
  size_bucket: floor_div(img.size, 100)
}
"#
        }
    };
    format!("{BENCH_PROCESS_DECLS}\n{main}")
}

pub fn benchmark_surface() -> &'static LashlangSurface {
    static SURFACE: OnceLock<LashlangSurface> = OnceLock::new();
    SURFACE.get_or_init(build_benchmark_surface)
}

fn build_benchmark_surface() -> LashlangSurface {
    let mut resources = ResourceCatalog::tool_default(["echo", "boom", "missing_tool"]);
    lashlang::add_trigger_resource_operations(&mut resources);
    resources.add_trigger_source_constructor(
        ["cron", "Schedule"],
        TypeExpr::Object(vec![
            TypeField {
                name: "expr".into(),
                ty: TypeExpr::Str,
                optional: false,
            },
            TypeField {
                name: "tz".into(),
                ty: TypeExpr::Str,
                optional: true,
            },
        ]),
        TypeExpr::Ref("cron.Tick".into()),
    );
    resources.add_trigger_source_constructor(
        ["ui", "button", "pressed"],
        TypeExpr::Object(Vec::new()),
        TypeExpr::Object(vec![
            TypeField {
                name: "button".into(),
                ty: TypeExpr::Union(vec![
                    TypeExpr::Enum(vec!["Red".into()]),
                    TypeExpr::Enum(vec!["Blue".into()]),
                ]),
                optional: false,
            },
            TypeField {
                name: "message".into(),
                ty: TypeExpr::Str,
                optional: false,
            },
            TypeField {
                name: "pressed_at".into(),
                ty: TypeExpr::Str,
                optional: false,
            },
        ]),
    );
    resources.add_module_instance(["shell"], "Shell");
    resources.add_operation(
        "Shell",
        "exec",
        "exec_command",
        TypeExpr::Any,
        TypeExpr::Any,
    );
    resources.add_module_instance(["llm"], "Llm");
    resources.add_operation("Llm", "query", "llm_query", TypeExpr::Any, TypeExpr::Any);
    resources.add_module_instance(["agents"], "Agents");
    resources.add_operation(
        "Agents",
        "spawn",
        "spawn_agent",
        TypeExpr::Any,
        TypeExpr::Any,
    );
    resources.add_module_instance(["processes"], "Processes");
    resources.add_operation(
        "Processes",
        "list",
        "list_process_handles",
        TypeExpr::Any,
        TypeExpr::Any,
    );
    resources.add_module_instance(["control"], "Control");
    resources.add_operation(
        "Control",
        "continue_as",
        "continue_as",
        TypeExpr::Any,
        TypeExpr::Any,
    );
    LashlangSurface::new(resources, LashlangAbilities::all())
}

pub fn linked_benchmark_program(source: &str) -> LinkedModule {
    LinkedModule::link(
        lashlang::parse(source).expect("benchmark program should parse"),
        benchmark_surface(),
    )
    .expect("benchmark program should link")
}

pub fn projected_bindings(scenario: Scenario) -> ProjectedBindings {
    let mut bindings = ProjectedBindings::new();
    if !matches!(
        scenario,
        Scenario::ProjectedValues | Scenario::ProjectedOperations | Scenario::ContinueAsSeedSurface
    ) {
        return bindings;
    }
    match scenario {
        Scenario::ProjectedValues => {
            bindings.insert(
                "history",
                ProjectedValue::custom("history", Arc::new(ProjectedList::history())),
            );
            bindings.insert(
                "docs",
                ProjectedValue::scalar("docs", projected_docs_record()),
            );
        }
        Scenario::ProjectedOperations | Scenario::ContinueAsSeedSurface => {
            bindings.insert(
                "proj",
                ProjectedValue::scalar("proj", projected_operations_record()),
            );
        }
        _ => {}
    }
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

fn snapshot_projected_record() -> Value {
    let mut projected = Record::default();
    projected.insert(
        "body".to_string(),
        Value::Projected(ProjectedValue::custom(
            "snap.projected.body",
            Arc::new(ProjectedText::new(
                "snapshot_body",
                "projected body stays lazy across snapshot markers",
            )),
        )),
    );

    let mut normal = Record::default();
    normal.insert("title".to_string(), Value::String("Snapshot Rules".into()));
    normal.insert(
        "tags".to_string(),
        Value::List(
            vec![
                Value::String("snapshot".into()),
                Value::String("projected".into()),
                Value::String("mixed".into()),
            ]
            .into(),
        ),
    );

    let mut nested = Record::default();
    nested.insert(
        "projected_title".to_string(),
        Value::Projected(ProjectedValue::custom(
            "snap.mixed.nested.projected_title",
            Arc::new(ProjectedText::new("nested_title", "Nested Projection")),
        )),
    );

    let mut mixed = Record::default();
    mixed.insert("count".to_string(), Value::Number(7.0));
    mixed.insert("nested".to_string(), Value::Record(Arc::new(nested)));

    let mut root = Record::default();
    root.insert("id".to_string(), Value::String("snapshot-mixed".into()));
    root.insert("normal".to_string(), Value::Record(Arc::new(normal)));
    root.insert("projected".to_string(), Value::Record(Arc::new(projected)));
    root.insert("mixed".to_string(), Value::Record(Arc::new(mixed)));
    Value::Record(Arc::new(root))
}

fn projected_operations_record() -> Value {
    let mut record = Record::default();
    record.insert(
        "items".to_string(),
        Value::Projected(ProjectedValue::custom(
            "proj.items",
            Arc::new(ProjectedList {
                name: "items",
                values: vec![
                    Value::String("alpha".into()),
                    Value::String("beta".into()),
                    Value::String("gamma".into()),
                    Value::String("delta".into()),
                ]
                .into(),
            }),
        )),
    );
    record.insert(
        "text".to_string(),
        Value::Projected(ProjectedValue::custom(
            "proj.text",
            Arc::new(ProjectedText::new("text", "alpha beta gamma delta")),
        )),
    );
    record.insert(
        "padded".to_string(),
        Value::Projected(ProjectedValue::custom(
            "proj.padded",
            Arc::new(ProjectedText::new("padded", "  alpha  ")),
        )),
    );
    record.insert(
        "json".to_string(),
        Value::Projected(ProjectedValue::custom(
            "proj.json",
            Arc::new(ProjectedText::new("json", "{\"ok\":true,\"count\":4}")),
        )),
    );
    record.insert(
        "number".to_string(),
        Value::Projected(ProjectedValue::scalar("proj.number", Value::Number(4.0))),
    );
    record.insert(
        "record".to_string(),
        Value::Projected(ProjectedValue::scalar("proj.record", {
            let mut inner = Record::default();
            inner.insert("topic".to_string(), Value::String("bench".into()));
            inner.insert("count".to_string(), Value::Number(4.0));
            Value::Record(Arc::new(inner))
        })),
    );
    Value::Record(Arc::new(record))
}

struct ProjectedList {
    name: &'static str,
    values: ListValue,
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

fn string_value(value: &Value) -> Result<compact_str::CompactString, ()> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string().into()),
        Value::Bool(value) => Ok(if *value { "true" } else { "false" }.into()),
        Value::Null => Ok("null".into()),
        _ => Err(()),
    }
}

fn clamp_slice_bounds(
    start: Option<isize>,
    end: Option<isize>,
    len: usize,
) -> Option<(usize, usize)> {
    let len = len as isize;
    let normalize = |bound: Option<isize>, default| {
        let bound = bound.unwrap_or(default);
        if bound < 0 { len + bound } else { bound }.clamp(0, len)
    };
    let start = normalize(start, 0);
    let end = normalize(end, len);
    (start < end).then_some((start as usize, end as usize))
}

fn slice_string(text: &str, start: Option<isize>, end: Option<isize>) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let Some((start, end)) = clamp_slice_bounds(start, end, chars.len()) else {
        return String::new();
    };
    chars[start..end].iter().collect()
}

impl ProjectedHostValue for ProjectedList {
    fn type_name(&self) -> &str {
        "list"
    }

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            let ProjectedReadRequest::Index(index) = request else {
                return match request {
                    ProjectedReadRequest::Len => ProjectedReadResponse::Len(self.values.len()),
                    ProjectedReadRequest::Empty => {
                        ProjectedReadResponse::Bool(self.values.is_empty())
                    }
                    ProjectedReadRequest::Truthy => ProjectedReadResponse::Bool(true),
                    ProjectedReadRequest::Contains(needle) => ProjectedReadResponse::Bool(
                        self.values.iter().any(|value| value == &needle),
                    ),
                    ProjectedReadRequest::Join(sep) => {
                        let Ok(sep) = string_value(&sep) else {
                            return ProjectedReadResponse::Missing;
                        };
                        let mut joined = String::new();
                        for (index, value) in self.values.iter().enumerate() {
                            let Ok(value) = string_value(value) else {
                                return ProjectedReadResponse::Missing;
                            };
                            if index > 0 {
                                joined.push_str(sep.as_ref());
                            }
                            joined.push_str(value.as_ref());
                        }
                        ProjectedReadResponse::Value(Value::String(joined.into()))
                    }
                    ProjectedReadRequest::Slice { start, end } => {
                        let Some((start, end)) = clamp_slice_bounds(start, end, self.values.len())
                        else {
                            return ProjectedReadResponse::Value(Value::List(Vec::new().into()));
                        };
                        ProjectedReadResponse::Value(Value::List(
                            self.values[start..end].to_vec().into(),
                        ))
                    }
                    ProjectedReadRequest::Push(item) => {
                        let mut values = self.values.to_vec();
                        values.push(item);
                        ProjectedReadResponse::Value(Value::List(values.into()))
                    }
                    ProjectedReadRequest::Render => ProjectedReadResponse::Text(format!(
                        "<{}:{}>",
                        self.name,
                        self.values.len()
                    )),
                    ProjectedReadRequest::Materialize => {
                        ProjectedReadResponse::Value(Value::List(self.values.clone()))
                    }
                    _ => ProjectedReadResponse::Missing,
                };
            };
            let Some(index) = resolve_index(&index, self.values.len()) else {
                return ProjectedReadResponse::Missing;
            };
            self.values
                .get(index)
                .cloned()
                .map(ProjectedReadResponse::Value)
                .unwrap_or(ProjectedReadResponse::Missing)
        })
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

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            match request {
                ProjectedReadRequest::Len => ProjectedReadResponse::Len(self.text.chars().count()),
                ProjectedReadRequest::Empty => ProjectedReadResponse::Bool(self.text.is_empty()),
                ProjectedReadRequest::Truthy => ProjectedReadResponse::Bool(!self.text.is_empty()),
                ProjectedReadRequest::Index(index) => {
                    let Some(index) = resolve_index(&index, self.text.chars().count()) else {
                        return ProjectedReadResponse::Missing;
                    };
                    self.text
                        .chars()
                        .nth(index)
                        .map(|ch| {
                            ProjectedReadResponse::Value(Value::String(ch.to_compact_string()))
                        })
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Find { needle, start } => {
                    let Value::String(needle) = needle else {
                        return ProjectedReadResponse::Missing;
                    };
                    ProjectedReadResponse::Value(match find_text(&self.text, &needle, start) {
                        Some(index) => Value::Number(index as f64),
                        None => Value::Null,
                    })
                }
                ProjectedReadRequest::GrepText(needle) => {
                    let Value::String(needle) = needle else {
                        return ProjectedReadResponse::Missing;
                    };
                    grep_text_records(&self.text, &needle)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::StartsWith(prefix) => {
                    let Value::String(prefix) = prefix else {
                        return ProjectedReadResponse::Missing;
                    };
                    ProjectedReadResponse::Bool(self.text.starts_with(&*prefix))
                }
                ProjectedReadRequest::EndsWith(suffix) => {
                    let Value::String(suffix) = suffix else {
                        return ProjectedReadResponse::Missing;
                    };
                    ProjectedReadResponse::Bool(self.text.ends_with(&*suffix))
                }
                ProjectedReadRequest::Split(needle) => {
                    let Value::String(needle) = needle else {
                        return ProjectedReadResponse::Missing;
                    };
                    ProjectedReadResponse::Value(Value::List(
                        self.text
                            .split(&*needle)
                            .map(|part| Value::String(part.into()))
                            .collect::<Vec<_>>()
                            .into(),
                    ))
                }
                ProjectedReadRequest::Trim => {
                    ProjectedReadResponse::Value(Value::String(self.text.trim().into()))
                }
                ProjectedReadRequest::Slice { start, end } => ProjectedReadResponse::Value(
                    Value::String(slice_string(&self.text, start, end).into()),
                ),
                ProjectedReadRequest::ToNumber => self
                    .text
                    .parse::<f64>()
                    .ok()
                    .map(Value::Number)
                    .map(ProjectedReadResponse::Value)
                    .unwrap_or(ProjectedReadResponse::Missing),
                ProjectedReadRequest::JsonParse => {
                    serde_json::from_str::<serde_json::Value>(&self.text)
                        .map(from_json)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Render => ProjectedReadResponse::Text(format!(
                    "<{}:{} chars>",
                    self.name,
                    self.text.chars().count()
                )),
                ProjectedReadRequest::Materialize => {
                    ProjectedReadResponse::Value(Value::String(self.text.as_ref().into()))
                }
                _ => ProjectedReadResponse::Missing,
            }
        })
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

impl ExecutionHost for BenchHost {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::ResourceOperation(operation) => {
                let empty = Record::new();
                let args = operation
                    .args
                    .first()
                    .and_then(Value::as_record)
                    .unwrap_or(&empty);
                bench_call(&operation.operation, args).map(AbilityResult::Value)
            }
            AbilityOp::StartProcess(start) => {
                Self::task_handle(&start.process_name, &start.args).map(AbilityResult::Value)
            }
            AbilityOp::Await(handle) => {
                let record = handle
                    .as_record()
                    .ok_or_else(|| ExecutionHostError::new("expected handle record"))?;
                Ok(AbilityResult::Value(
                    record.get("value").cloned().unwrap_or(Value::Null),
                ))
            }
            AbilityOp::Cancel(handle) => Ok(AbilityResult::Value(handle)),
            AbilityOp::Print(_) => Ok(AbilityResult::Unit),
            AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                Ok(AbilityResult::Value(value))
            }
            _ => Err(ExecutionHostError::new("unsupported host ability")),
        }
    }
}

fn bench_call(name: &str, args: &Record) -> Result<Value, ExecutionHostError> {
    match name {
        "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
        "boom" => Err(ExecutionHostError::new("explicit failure for benchmark")),
        "exec_command" => {
            let mut record = Record::default();
            record.insert("status".to_string(), Value::String("completed".into()));
            record.insert("done".to_string(), Value::Bool(true));
            record.insert("running".to_string(), Value::Bool(false));
            record.insert("exit_code".to_string(), Value::Number(1.0));
            record.insert(
                "output".to_string(),
                Value::String(
                    format!(
                        "ran: {}",
                        args.get("cmd")
                            .and_then(|value| match value {
                                Value::String(text) => Some(text.as_str()),
                                _ => None,
                            })
                            .unwrap_or("")
                    )
                    .into(),
                ),
            );
            Ok(Value::Record(Arc::new(record)))
        }
        "llm_query" | "query_llm" => {
            let mut record = Record::default();
            record.insert(
                "text".to_string(),
                Value::String("benchmark summary".into()),
            );
            record.insert("tokens".to_string(), Value::Number(42.0));
            Ok(Value::Record(Arc::new(record)))
        }
        "spawn_agent" | "spawn_child" => {
            let task = args
                .get("task")
                .and_then(|value| match value {
                    Value::String(text) => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or("agent");
            let mut record = Record::default();
            record.insert(
                "claim".to_string(),
                Value::String(format!("done:{task}").into()),
            );
            Ok(Value::Record(Arc::new(record)))
        }
        "list_process_handles" => Ok(process_handles_record()),
        "continue_as" => Ok(continue_as_record(args)),
        "triggers.register" => Ok(trigger_register_record(args)),
        "triggers.list" => Ok(trigger_list_value(args)),
        "triggers.cancel" => Ok(Value::Bool(true)),
        _ => Err(unknown_tool(name)),
    }
}

fn trigger_register_record(args: &Record) -> Value {
    let source_type = args
        .get("source")
        .and_then(Value::as_record)
        .and_then(|source| source.get(LASH_HOST_VALUE_TYPE_KEY))
        .and_then(string_ref)
        .unwrap_or("unknown.Source");
    let process_name = args
        .get("target")
        .and_then(Value::as_record)
        .and_then(|target| target.get(LASH_PROCESS_NAME_KEY))
        .and_then(string_ref)
        .unwrap_or("target");

    let mut record = Record::default();
    record.insert("type".to_string(), Value::String("trigger_handle".into()));
    record.insert(
        "id".to_string(),
        Value::String(format!("trigger:{source_type}:{process_name}").into()),
    );
    record.insert(
        "source_type".to_string(),
        Value::String(source_type.to_string().into()),
    );
    record.insert(
        "process_name".to_string(),
        Value::String(process_name.to_string().into()),
    );
    Value::Record(Arc::new(record))
}

fn trigger_list_value(args: &Record) -> Value {
    let process_name = args
        .get("target")
        .and_then(Value::as_record)
        .and_then(|target| target.get(LASH_PROCESS_NAME_KEY))
        .and_then(string_ref)
        .unwrap_or("target");
    let source_type = match process_name {
        "daily_digest" => "cron.Schedule",
        "on_button" => "ui.button.pressed",
        _ => "unknown.Source",
    };

    let mut target = Record::default();
    target.insert(
        "process_name".to_string(),
        Value::String(process_name.to_string().into()),
    );
    target.insert("input_name".to_string(), Value::String("event".into()));

    let mut route = Record::default();
    route.insert(
        "handle".to_string(),
        Value::String(format!("trigger:{source_type}:{process_name}").into()),
    );
    route.insert(
        "source_type".to_string(),
        Value::String(source_type.to_string().into()),
    );
    route.insert("source".to_string(), trigger_source_value(source_type));
    route.insert("target".to_string(), Value::Record(Arc::new(target)));
    route.insert("enabled".to_string(), Value::Bool(true));
    Value::List(vec![Value::Record(Arc::new(route))].into())
}

fn trigger_source_value(source_type: &str) -> Value {
    let mut source = Record::default();
    source.insert(
        LASH_HOST_VALUE_TYPE_KEY.to_string(),
        Value::String(source_type.to_string().into()),
    );
    source.insert(
        LASH_HOST_VALUE_KEY.to_string(),
        Value::Record(Arc::new(Record::default())),
    );
    Value::Record(Arc::new(source))
}

fn continue_as_record(args: &Record) -> Value {
    let seed = args.get("seed").and_then(Value::as_record);
    let mut seed_keys = Vec::new();
    let mut projected_count = 0usize;
    let mut global_count = 0usize;
    if let Some(seed) = seed {
        for (key, value) in seed.iter() {
            seed_keys.push(Value::String(key.into()));
            if matches!(value, Value::Projected(_)) {
                projected_count += 1;
            } else {
                global_count += 1;
            }
        }
    }

    let mut record = Record::default();
    record.insert("ok".to_string(), Value::Bool(true));
    record.insert("frame_id".to_string(), Value::String("frame:bench".into()));
    record.insert(
        "task".to_string(),
        args.get("task")
            .cloned()
            .unwrap_or_else(|| Value::String("continue".into())),
    );
    record.insert("seed_keys".to_string(), Value::List(seed_keys.into()));
    record.insert(
        "projected_count".to_string(),
        Value::Number(projected_count as f64),
    );
    record.insert(
        "global_count".to_string(),
        Value::Number(global_count as f64),
    );
    Value::Record(Arc::new(record))
}

fn string_ref(value: &Value) -> Option<&str> {
    match value {
        Value::String(value) => Some(value.as_str()),
        _ => None,
    }
}

fn process_handles_record() -> Value {
    let mut chunk_1 = Record::default();
    chunk_1.insert("__handle__".to_string(), Value::String("process".into()));
    chunk_1.insert("id".to_string(), Value::String("spawn-one".into()));
    chunk_1.insert("process_id".to_string(), Value::String("spawn-one".into()));
    chunk_1.insert("tool".to_string(), Value::String("spawn_child".into()));
    chunk_1.insert("value".to_string(), spawn_child_value("inspect auth"));

    let mut chunk_2 = Record::default();
    chunk_2.insert("__handle__".to_string(), Value::String("process".into()));
    chunk_2.insert("id".to_string(), Value::String("spawn-two".into()));
    chunk_2.insert("process_id".to_string(), Value::String("spawn-two".into()));
    chunk_2.insert("tool".to_string(), Value::String("spawn_child".into()));
    chunk_2.insert("value".to_string(), spawn_child_value("inspect api"));

    Value::List(
        vec![
            Value::Record(Arc::new(chunk_1)),
            Value::Record(Arc::new(chunk_2)),
        ]
        .into(),
    )
}

fn spawn_child_value(name: &str) -> Value {
    let mut record = Record::default();
    record.insert(
        "claim".to_string(),
        Value::String(format!("done:{name}").into()),
    );
    Value::Record(Arc::new(record))
}

fn unknown_tool(name: &str) -> ExecutionHostError {
    ExecutionHostError::new(format!("unknown tool: {name}"))
}

impl BenchHost {
    fn task_handle(name: &str, args: &Record) -> Result<Value, ExecutionHostError> {
        match name {
            "echo" | "query_llm" | "spawn_child" | "continue_as" => {
                let mut record = Record::default();
                record.insert("__handle__".to_string(), Value::String("process".into()));
                record.insert("tool".to_string(), Value::String(name.to_string().into()));
                record.insert("value".to_string(), bench_call(name, args)?);
                Ok(Value::Record(Arc::new(record)))
            }
            _ => Err(unknown_tool(name)),
        }
    }
}
