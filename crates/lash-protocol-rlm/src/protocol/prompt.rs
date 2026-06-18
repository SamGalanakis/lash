pub const LASHLANG_BUILTINS_SECTION: &str = r#"### Builtins

Call as functions (e.g. `len(x)`, `slice(s, 0, 200)`). For `slice`, `null` bounds mean start/end; negative bounds count from the end.

- `len(x)` — length of string/list/record (0 for null); use `image.size` for images
- `empty(x)` — true if length is 0
- `slice(s, start, end)` — substring or sublist
- `range(end)` / `range(start, end)` / `range(start, end, step)` — integer list, end-exclusive; positive or negative `step`, never `0`
- `ceil_div(a, b)` / `floor_div(a, b)` — integer division helpers for chunk/count math; divisor must not be `0`
- `push(list, item)` — new list with one item appended
- `split(s, sep)` / `join(list, sep)` — string split/join
- `find(s, needle, start?)` — zero-based character index of the first literal match, or `null`; `start` defaults to `0` and is a non-negative character index; an empty `needle` returns `start` when it is in bounds
- `grep_text(s, needle)` — literal in-memory line search; `needle` must be non-empty; returns one record per matching line: `{ line: int, text: str, match: str, start: int, end: int }`, where `line` is 1-based, `text` is the line without its line ending, and `start`/`end` are zero-based character offsets within that line's `text` with `end` exclusive
- `trim(s)` — strip whitespace
- `starts_with(s, prefix)` / `ends_with(s, suffix)` / `contains(haystack, needle)`
- `keys(record)` / `values(record)`
- `to_string(x)` / `to_int(x)` / `to_float(x)`
- `json_parse(s)` — parse a JSON string into a value
- `format(template, arg0, arg1, ...)` — positional interpolation: `{}` auto-numbers, `{0}` / `{1}` pick a specific arg, `{{` / `}}` escape literal braces. Do not wrap args in a list: use `format("It is {}.", trim(now.output))`, not `format("It is {}.", [trim(now.output)])`.
- `validate(value, Type { ... })` — check an intermediate value against a Type literal and return it unchanged, or abort with a validation error
"#;

pub const LASHLANG_BUILTINS_NO_IMAGES_SECTION: &str = r#"### Builtins

Call as functions (e.g. `len(x)`, `slice(s, 0, 200)`). For `slice`, `null` bounds mean start/end; negative bounds count from the end.

- `len(x)` — length of string/list/record (0 for null)
- `empty(x)` — true if length is 0
- `slice(s, start, end)` — substring or sublist
- `range(end)` / `range(start, end)` / `range(start, end, step)` — integer list, end-exclusive; positive or negative `step`, never `0`
- `ceil_div(a, b)` / `floor_div(a, b)` — integer division helpers for chunk/count math; divisor must not be `0`
- `push(list, item)` — new list with one item appended
- `split(s, sep)` / `join(list, sep)` — string split/join
- `find(s, needle, start?)` — zero-based character index of the first literal match, or `null`; `start` defaults to `0` and is a non-negative character index; an empty `needle` returns `start` when it is in bounds
- `grep_text(s, needle)` — literal in-memory line search; `needle` must be non-empty; returns one record per matching line: `{ line: int, text: str, match: str, start: int, end: int }`, where `line` is 1-based, `text` is the line without its line ending, and `start`/`end` are zero-based character offsets within that line's `text` with `end` exclusive
- `trim(s)` — strip whitespace
- `starts_with(s, prefix)` / `ends_with(s, suffix)` / `contains(haystack, needle)`
- `keys(record)` / `values(record)`
- `to_string(x)` / `to_int(x)` / `to_float(x)`
- `json_parse(s)` — parse a JSON string into a value
- `format(template, arg0, arg1, ...)` — positional interpolation: `{}` auto-numbers, `{0}` / `{1}` pick a specific arg, `{{` / `}}` escape literal braces. Do not wrap args in a list: use `format("It is {}.", trim(now.output))`, not `format("It is {}.", [trim(now.output)])`.
- `validate(value, Type { ... })` — check an intermediate value against a Type literal and return it unchanged, or abort with a validation error
"#;

pub const LASHLANG_COMMON_PATTERNS_SECTION: &str = r#"### Common patterns

Operation-level errors are different from successful results that contain domain errors. `?` aborts the block only when the module operation itself failed:

```lashlang
probe = await web.search({ query: "value" })?
submit format("Search returned {} characters.", len(to_string(probe)))
```

Build collections with explicit loops, not comprehensions:

```lashlang
items = []
for key in ["a", "b"] {
  items = push(items, { key: key, size: len(key) })
}
submit format("Built {} items.", len(items))
```

Print what you need to see. Pull in the parts relevant to your next step — a whole value when it is small and all useful, otherwise keys, lengths, selected fields, or slices of large ones:

```lashlang
result = await web.search({ query: "value" })?
text = to_string(result)
print { chars: len(text), head: slice(text, 0, 1200) }
```

For dependent multi-step work, inspect intermediate results before submitting success. Reaching `submit` ends the turn even mid-block:

```lashlang
first = await web.search({ query: "value" })?
second = await web.fetch({ url: first[0].url })?
if contains(to_string(second), "needs_more_work") {
  print { first: first, second: second }
} else {
  submit format("Fetched result: {}", slice(to_string(second), 0, 1200))
}
```
"#;

pub const LASHLANG_TYPE_LITERALS_SECTION: &str = r#"### Type literals

`Type { field: shape, ... }` describes a record shape. Field separators are commas (trailing comma OK).

- Scalars: `str`, `int`, `float`, `bool`, `dict`, `any`, `null`.
- Collections: `list[shape]`, `enum["a", "b"]`, nested `Type { ... }`.
- **Optional field** — put `?` after the type: `email: str?` means the field may be absent from the record. If the field IS present, its value must be a string; `null` is **not** allowed.
- **Nullable field** — use a union with `null`: `email: str | null` means the field is required and its value is either a string or null.
- **Unions** — `a | b | c`, e.g. `status: str | int`, `value: str | null`.
- Nested shapes require the `Type` keyword: `nested: Type { ok: bool }` (bare `{ ok: bool }` is rejected — that's a record value, not a type).

```lashlang
profile = validate(record, Type { name: str, email: str?, tags: list[str] })
```
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RlmPromptFeatures {
    pub images: bool,
    pub common_patterns: bool,
    pub type_literals: bool,
    pub decomposition: bool,
}

impl Default for RlmPromptFeatures {
    fn default() -> Self {
        Self {
            images: true,
            common_patterns: true,
            type_literals: true,
            decomposition: true,
        }
    }
}

pub fn rlm_execution_section_for_host_environment(
    features: RlmPromptFeatures,
    surface: &lashlang::LashlangHostEnvironment,
) -> String {
    let has_operations = surface.resources.has_operations();
    let mut sections = Vec::new();
    sections.push(render_execution_intro(has_operations));
    sections.push(render_language_section(
        features.images,
        has_operations,
        &surface.abilities,
        &surface.language_features,
    ));
    if let Some(section) = render_host_environment_section(surface) {
        sections.push(section);
    }
    sections.push(if features.images {
        LASHLANG_BUILTINS_SECTION.to_string()
    } else {
        LASHLANG_BUILTINS_NO_IMAGES_SECTION.to_string()
    });
    if features.common_patterns {
        sections.push(render_common_patterns(has_operations));
    }
    if features.type_literals {
        sections.push(LASHLANG_TYPE_LITERALS_SECTION.to_string());
    }
    if features.decomposition {
        sections.push(render_decomposition_section(
            has_operations,
            surface.abilities.processes,
        ));
    }
    sections.join("\n\n")
}

fn render_host_environment_section(surface: &lashlang::LashlangHostEnvironment) -> Option<String> {
    let mut operation_lines = Vec::new();
    for (_, module) in surface.resources.module_instances() {
        if let Some(resource_type) =
            surface
                .resources
                .resolve_alias(&lashlang::ResourceRefExpr::resolved(
                    module
                        .path
                        .iter()
                        .map(|segment| segment.as_str().into())
                        .collect(),
                    module.resource_type.clone(),
                    module.alias.clone(),
                ))
        {
            for (operation, binding) in &resource_type.operations {
                operation_lines.push(format!(
                    "- `await {}.{}({})? -> {}`",
                    module.alias,
                    operation,
                    lashlang::format_type_expr(&binding.input_ty),
                    lashlang::format_type_expr(&binding.output_ty)
                ));
            }
        }
    }
    let mut data_type_lines = Vec::new();
    for (_, data_type) in surface.resources.named_data_types() {
        data_type_lines.push(format!(
            "- `type {} = {}`",
            data_type.name(),
            lashlang::format_type_expr(data_type.ty())
        ));
    }
    let mut constructor_lines = Vec::new();
    for (_, constructor) in surface.resources.value_constructors() {
        let output_ty = match &constructor.output_ty {
            lashlang::TypeExpr::Ref(name) => surface
                .resources
                .resolve_trigger_source(name.as_str())
                .map(|binding| format!("TriggerSource<{}>", binding.event_type_name()))
                .unwrap_or_else(|| lashlang::format_type_expr(&constructor.output_ty)),
            _ => lashlang::format_type_expr(&constructor.output_ty),
        };
        constructor_lines.push(format!(
            "- `{}({}) -> {}`",
            constructor.path.join("."),
            lashlang::format_type_expr(&constructor.input_ty),
            output_ty
        ));
    }
    let mut protocol_lines = Vec::new();
    let trigger_register = lashlang::TriggerHostOperation::Register.host_operation();
    for (source_ty, binding) in surface.resources.trigger_sources() {
        protocol_lines.push(format!(
            "- `{}` can be passed to `{}` and emits `{}`",
            source_ty,
            trigger_register,
            binding.event_type_name()
        ));
    }
    if operation_lines.is_empty()
        && data_type_lines.is_empty()
        && constructor_lines.is_empty()
        && protocol_lines.is_empty()
    {
        return None;
    }
    let mut section = String::from("### Host Surface");
    if !operation_lines.is_empty() {
        section.push_str("\n\nAwaited runtime operations:\n\n");
        section.push_str(&operation_lines.join("\n"));
    }
    if !data_type_lines.is_empty() {
        section.push_str("\n\nNamed host data types:\n\n");
        section.push_str(&data_type_lines.join("\n"));
    }
    if !constructor_lines.is_empty() {
        section.push_str("\n\nPure value constructors. Do not `await` these; use them wherever expressions are allowed:\n\n");
        section.push_str(&constructor_lines.join("\n"));
    }
    if !protocol_lines.is_empty() {
        section.push_str("\n\nTrigger source protocol metadata:\n\n");
        section.push_str(&protocol_lines.join("\n"));
    }
    Some(section)
}

fn render_execution_intro(has_operations: bool) -> String {
    let mut section = String::from("**All actions go through `lashlang`.** ");
    if has_operations {
        section.push_str("Invoke documented operations with module syntax like `await agents.spawn({ ... })?` or `await web.search({ ... })?` from inside a fenced `lashlang` block. Start from operations listed under **Showcased Tools**; if a discovery tool is available, use it to find additional module call forms before calling them. Emit a block whenever you need to call an available operation or compute a value. Plain prose is for direct conversational replies that need no action.");
    } else {
        section.push_str("Use fenced `lashlang` blocks to compute values, inspect current variables, and submit final answers. No module operations are available in this turn, so do not invent tool calls. Plain prose is for direct conversational replies that need no computation.");
    }
    section.push_str(
        r#"

### `print` vs `submit`

- `print <expr>` — inspect a value and keep going; output appears on the next step. Print the part you need to decide the next step: a whole value when it is small and all of it is useful (e.g. state you consult each turn), otherwise selected fields, samples, or slices. Avoid dumping a large value when only part of it is relevant.
- `submit <expr>` — final answer; ends the turn. Strings pass through as the reply. Non-string values render as pretty JSON for machine consumers; for user-facing turns, follow the current final-answer format guidance.

Never `submit` a raw tool-result dump. If you need to look at something, `print` it, then `submit` a summary on a later step.

### Turn shape

**Exactly one ` ```lashlang ` fenced block per response.** Anything after the first block closes is dropped. Only ` ```lashlang ` is recognised — `rlm` and other labels are treated as plain prose.

- Write small blocks. Each should do one focused step.
- Keep prose around the block to one or two sentences of reasoning. Don't describe an action in prose instead of executing it.
- After each result, decide: another block (more work), or finish.
"#,
    );
    if has_operations {
        section.push_str(
            r#"
Example — inspect with an available operation, then submit:

````
Checking the available data.

```lashlang
result = await web.search({ query: "value" })?
print { summary: slice(to_string(result), 0, 200) }
```
````

…then on the next turn, once you've seen what you need:

````
```lashlang
submit "The bound version is 0.2.61."
```
````
"#,
        );
    } else {
        section.push_str(
            r#"
Example — compute and submit:

````
```lashlang
items = ["alpha", "beta"]
submit format("Found {} items; first item: {}.", len(items), items[0])
```
````
"#,
        );
    }
    section
}

fn render_language_section(
    images: bool,
    has_operations: bool,
    abilities: &lashlang::LashlangAbilities,
    language_features: &lashlang::LashlangLanguageFeatures,
) -> String {
    let mut bullets = Vec::new();
    if images {
        bullets.push("- Values: null, booleans, numbers, strings, lists, records, and immutable `Image` handles. Literals: `[a, b]`, `{ a: 1, b: 2 }`.".to_string());
        bullets.push("- Images: image-producing tools may return an `Image` value. Read metadata with `.id`, `.label`, `.size`, `.width`, `.height`; fields are read-only. `print(image)` or `print` on a list/record containing images sends both descriptor text and the actual image attachment to the next model call. `submit(image)`, `to_string(image)`, and JSON-like serialization emit only `{ \"type\": \"image\", \"id\": ..., \"label\": ..., \"size\": ..., \"width\": ..., \"height\": ... }`. `len(image)` is invalid; use `.size`.".to_string());
    } else {
        bullets.push("- Values: null, booleans, numbers, strings, lists, and records. Literals: `[a, b]`, `{ a: 1, b: 2 }`.".to_string());
    }
    bullets.push("- Strings: `\"...\"` supports `\\n`, `\\r`, `\\t`, `\\\"`, and `\\\\`; `\"\"\"...\"\"\"` is multiline with the same escapes; `r\"\"\"...\"\"\"` and `r'''...'''` are raw multiline strings and preserve content exactly. Use raw multiline strings for JSON, Markdown, and other payloads with braces, backslashes, quotes, heredocs, or `@@` hunk markers.".to_string());
    bullets.push("- Assign with `name = expr`. Variables persist across fenced blocks within the turn. You can also update mutable collection paths rooted at a variable: `record.field = value`, `record[key] = value`, `list[i] = value`, and nested forms like `state.groups[g].count = count + 1`. Record field/index assignment inserts or replaces fields; list assignment replaces an existing integer index only. Record indexing reads dynamic string-coerced keys and returns `null` when missing, so histogram code can use `counts[g] = counts[g] + 1`.".to_string());
    if has_operations {
        bullets.push("- Module operations: call host capabilities through documented lowercase module paths, e.g. `await agents.spawn({ task: task })?`, `await web.search({ query: q })?`, or `await gmail.work.send({ to: to, body: body })?`. Host operations are awaited effects; pure UpperCamel value constructors shown in the Host Surface are ordinary expressions and must not be awaited. Bare calls are builtins only, not tools. `?` aborts the block with sanitized operation metadata if the operation fails.".to_string());
    }
    if abilities.sleep {
        bullets.push("- Sleep: pause foreground code or process code with `sleep for \"5s\"` or `sleep until deadline`. Durations accept milliseconds, `ms`, `s`, `m`, or `h`; deadlines accept RFC3339 text or Unix epoch milliseconds.".to_string());
    }
    if abilities.processes {
        let mut forms = vec![
            "`yield value`",
            "`wake value`",
            "`finish value`",
            "`fail value`",
        ];
        if abilities.process_signals {
            forms.push("`payload = wait_signal(\"name\")`");
        }
        let trigger_process_note = if abilities.triggers {
            " matching trigger occurrences can also create process runs through registered triggers."
        } else {
            ""
        };
        let signal_declaration_note = if abilities.process_signals {
            " Add typed inbound signals with `signals { name: TYPE }` when the process receives external messages, e.g. `process worker() signals { approve: { ok: bool } } { payload = wait_signal(\"approve\") finish payload }`."
        } else {
            ""
        };
        bullets.push(format!(
            "- Background processes: `process name(param: TYPE) {{ ... }}` declares a reusable process definition.{signal_declaration_note} `handle = start name(param: value)` creates one process run from that definition and returns its run handle;{trigger_process_note} For account-parametric work, pass typed module authorities explicitly, e.g. `process notify(mail: Gmail) {{ await mail.send({{ body: body }})? finish true }}` and `start notify(mail: gmail.work)`. For one-off concrete automations, a process body may reference concrete host paths such as `agents`, `web`, or `gmail.work` directly; params and locals shadow those captures. Inside a process use {}. `wake value` emits a `process.wake` event that notifies the agent/session with `value`; use it when process progress, trigger output, or other background work should re-enter the model as context. `finish value` completes the run and stores `value` as the process success value. `fail value` completes it as failed; falling off the end is `finish null`. `submit` and `print` are foreground-only and invalid inside processes. Parallelism comes from starting all independent process handles before waiting for any of them; join a list or record of handles with `results = await handles`. `await handle` waits and returns a result wrapper like `{{ ok: true, value: ... }}`; when you need fields from the `finish` value, use `result = (await handle)?` and then read `result.field`. Cancel a live run with `cancel handle` (best-effort). If the Host Surface includes `processes.list`, use `await processes.list({{}})?` for running runs, `await processes.list({{ definition: name }})?` for runs of a definition, and `await processes.list({{ status: \"any\" }})?` for visible run history.",
            join_words(&forms),
        ));
        if abilities.process_signals {
            bullets.push("- Signalling processes: `signal_run(handle, \"name\", payload)` sends a typed `signal.name` event to a running process and may be used from the foreground turn as well as inside a process body, like `await handle` and `cancel handle`. The receiving side, `payload = wait_signal(\"name\")`, parks a process until that named signal arrives and is only valid inside a process body.".to_string());
        }
    }
    if language_features.label_annotations {
        bullets.push("- Execution labels: use `@label(title: \"Label\")` or `@label(title: \"Label\", description: \"Details\")` to name important Lashlang phases and graph steps. At top level, label meaningful setup, resource calls, submissions, branches, loops, or process declarations. Inside a `process` body, label durable steps such as awaited module calls, `start`, `sleep`, `wait_signal`, `signal_run`, `wake`, `yield`, `finish`, `fail`, `if`, loops, and setup statements that explain the process. Titles/descriptions must be string literals; do not use variables, interpolation, icons, colors, layout hints, or extra keys.".to_string());
    }
    if abilities.triggers && abilities.processes {
        let trigger_register = lashlang::TriggerHostOperation::Register.host_operation();
        let trigger_list = lashlang::TriggerHostOperation::List.host_operation();
        let trigger_cancel = lashlang::TriggerHostOperation::Cancel.host_operation();
        bullets.push(format!("- Trigger registry: a trigger registration connects a typed source value to a process definition plus explicit inputs. Register with `handle = await {trigger_register}({{ source: source, target: daily_digest, inputs: {{ tick: trigger.event }}, name: \"daily_digest\" }})?`. Constructors build source values; the host/plugin that owns the source lists stored subscriptions by source type/key and emits trigger occurrences when source-specific events happen. `target` is a process definition value. `inputs` is required and maps every process param exactly once. `trigger.event` is the direct whole-event value inside `inputs`; fixed inputs can pass concrete authorities like `gmail.work` or `agents` for account-parametric processes. Use `await {trigger_list}({{}})?` to discover visible registrations, or filter with `{{ target: daily_digest }}`, `{{ name: \"daily_digest\" }}`, `{{ source_type: \"cron.Schedule\" }}`, and `{{ enabled: true }}`. Use `await {trigger_cancel}({{ handle: handle }})?` to disable future occurrence delivery for that registration."));
    }
    if has_operations {
        let scheduling = if abilities.processes {
            "- Operation scheduling: consecutive `await module.op(...)` statements run one at a time. For independent module operations, use aggregate await: `results = await { first: module.a({ ... }), second: module.b({ ... }), label: \"kept\" }`; direct receiver-call leaves in nested lists/records fan out through the host scheduler, pure value leaves are preserved, and results keep the same shape. Put `?` on each operation leaf you want unwrapped, e.g. `first: module.a({ ... })?`; all siblings finish before the first source-order failure is reported. Process parallelism still comes from starting all independent process handles before joining them with `await handles` or `await { key: handle }`."
        } else {
            "- Operation scheduling: consecutive `await module.op(...)` statements run one at a time. For independent module operations, use aggregate await: `results = await { first: module.a({ ... }), second: module.b({ ... }), label: \"kept\" }`; direct receiver-call leaves in nested lists/records fan out through the host scheduler, pure value leaves are preserved, and results keep the same shape. Put `?` on each operation leaf you want unwrapped, e.g. `first: module.a({ ... })?`; all siblings finish before the first source-order failure is reported."
        };
        bullets.push(scheduling.to_string());
    }
    bullets.push("- Control flow: statement `if`/`for`/`while`; `break` exits the nearest loop; `continue` skips to the nearest loop's next iteration; expression ternary `cond ? yes : no` (there is no expression-form `if`); boolean negation via `!cond` or `not cond`. Prefer bounded `while` loops where possible and bounded `for` loops over ranges/lists for fill or retry logic. `submit` is different from `break`: it ends the whole program/turn.".to_string());
    bullets.push("- Bare expressions are valid statements in normal blocks.".to_string());
    bullets.push("- The **Bound Variables** section lists values already in scope, plus `history` — use them directly in lashlang, don't recreate them. Small values show inline; large values show only type and size. Other available read-only values may be listed separately without value previews. `print` a variable (or the part you need) to see contents it only summarizes.".to_string());
    format!("### Language\n\n{}", bullets.join("\n"))
}

fn render_common_patterns(has_operations: bool) -> String {
    if has_operations {
        return LASHLANG_COMMON_PATTERNS_SECTION.to_string();
    }
    r#"### Common patterns

Build collections with explicit loops, not comprehensions:

```lashlang
items = []
for key in ["a", "b"] {
  items = push(items, { key: key, size: len(key) })
}
submit format("Built {} items.", len(items))
```

Print what you need to see. Pull in the parts relevant to your next step — a whole value when it is small and all useful, otherwise keys, lengths, selected fields, or slices of large ones:

```lashlang
text = to_string(input)
print { chars: len(text), head: slice(text, 0, 1200) }
```

For multi-step work, inspect intermediate results before submitting success. Reaching `submit` ends the turn even mid-block:

```lashlang
first = trim(input.question)
if empty(first) {
  print { problem: "missing question" }
} else {
  submit format("Question: {}", first)
}
```"#
        .to_string()
}

fn render_decomposition_section(has_operations: bool, processes: bool) -> String {
    let mut section = String::from(
        "### Working with context\n\nYour turn's REPL trace is your working memory — keep it decision-sized and current. Two kinds of values need different handling: large transient artifacts (files, search results, long pages, raw tool dumps) stay in variables — `print` only the fields or slices you need, not the whole thing; small durable state you consult each turn (a map, plan, or checklist) should stay visible — small values show inline under **Bound Variables**, and you can `print` them whole when you need to reason over all of them.\n\nChoose the lightest mechanism that preserves progress:\n\n- Current variables already hold what you need → reason inline in lashlang.",
    );
    if has_operations {
        if processes {
            section.push_str("\n- Several independent slow operations are needed -> use aggregate await over a record/list of direct module calls plus any pure values you want preserved, putting `?` on each operation leaf that should unwrap.");
        } else {
            section.push_str("\n- Several independent operations are needed -> use aggregate await over a record/list of direct module calls plus any pure values you want preserved, putting `?` on each operation leaf that should unwrap.");
        }
    }
    section.push_str("\n- The trace is bloated, stale, or failed attempts dominate -> use an available continuation tool to switch to a fresh AgentFrame with concrete state.");
    if has_operations && processes {
        section.push_str("\n- Anything tool-specific (parameters, return shapes, lifecycle) lives under **Showcased Tools** — don't infer a tool exists from these generic examples.\n\nExample parallel fan-out around an available operation (aggregate await preserves the record shape; use `?` on each leaf to unwrap it):\n\n```lashlang\nresults = await {\n  one: web.search({ query: \"one\" })?,\n  two: web.search({ query: \"two\" })?\n}\nsubmit format(\"First result: {}\\n\\nSecond result: {}\", slice(to_string(results.one), 0, 800), slice(to_string(results.two), 0, 800))\n```");
    } else if has_operations {
        section.push_str("\n- Anything tool-specific (parameters, return shapes, lifecycle) lives under **Showcased Tools** — don't infer a tool exists from these generic examples.");
    } else {
        section.push_str("\n- No module operations are available in this turn — don't infer one exists from generic lashlang syntax.");
    }
    section
}

fn join_words(words: &[&str]) -> String {
    match words {
        [] => String::new(),
        [one] => (*one).to_string(),
        [one, two] => format!("{one} or {two}"),
        _ => {
            let mut out = words[..words.len() - 1].join(", ");
            out.push_str(", or ");
            out.push_str(words[words.len() - 1]);
            out
        }
    }
}
