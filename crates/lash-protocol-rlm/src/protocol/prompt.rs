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

Operation-level errors are different from successful results that contain domain errors. `?` aborts the block only when the resource operation itself failed:

```lashlang
probe = await TOOL.default.tool_name({ key: "value" })?
submit probe
```

Build collections with explicit loops, not comprehensions:

```lashlang
items = []
for key in ["a", "b"] {
  value = await TOOL.default.tool_name({ key: key })?
  items = push(items, { key: key, value: value })
}
submit items
```

Print narrow observations. Keep large values in variables and print only keys, lengths, selected fields, or slices:

```lashlang
result = await TOOL.default.tool_name({ key: "value" })?
text = to_string(result)
print { chars: len(text), head: slice(text, 0, 1200) }
```

For multi-step work, inspect intermediate results before submitting success. Reaching `submit` ends the turn even mid-block:

```lashlang
first = await TOOL.default.tool_a({ key: "value" })?
second = await TOOL.default.tool_b({ input: first })?
if contains(to_string(second), "needs_more_work") {
  print { first: first, second: second }
} else {
  submit second
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

pub fn rlm_execution_section_for_surface(
    features: RlmPromptFeatures,
    surface: &lashlang::LashlangSurface,
) -> String {
    let has_operations = surface.resources.has_operations();
    let mut sections = Vec::new();
    sections.push(render_execution_intro(has_operations));
    sections.push(render_language_section(
        features.images,
        has_operations,
        &surface.abilities,
    ));
    if let Some(section) = render_host_events_section(surface) {
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

fn render_host_events_section(surface: &lashlang::LashlangSurface) -> Option<String> {
    let mut lines = Vec::new();
    for (resource_type, catalog) in &surface.resources.resource_types {
        for alias in &catalog.aliases {
            for (event, binding) in &catalog.trigger_events {
                lines.push(format!(
                    "- `{}.{alias}.{event}` payload `{}`",
                    resource_type,
                    lashlang::format_type_expr(&binding.payload_ty)
                ));
            }
        }
    }
    if lines.is_empty() {
        return None;
    }
    Some(format!(
        "### Host Events\n\nHosts may emit these typed events. Register declaration-only `trigger` modules against them; host events are not tools and cannot be called directly.\n\n{}",
        lines.join("\n")
    ))
}

fn render_execution_intro(has_operations: bool) -> String {
    let mut section = String::from("**All actions go through `lashlang`.** ");
    if has_operations {
        section.push_str("Invoke documented operations with receiver syntax like `await TOOL.default.tool_name({ ... })?` from inside a fenced `lashlang` block. Start from resources listed under **Showcased Tools**; if a discovery tool is available, use it to find and load additional operations before calling them. Emit a block whenever you need to call an available operation or compute a value. Plain prose is for direct conversational replies that need no action.");
    } else {
        section.push_str("Use fenced `lashlang` blocks to compute values, inspect current variables, and submit structured answers. No receiver operations are available in this turn, so do not invent tool calls. Plain prose is for direct conversational replies that need no computation.");
    }
    section.push_str(
        r#"

### `print` vs `submit`

- `print <expr>` — inspect a value and keep going. Output appears on the next step and is capped: keep full tool results in variables and print only lengths, selected fields, samples, or slices. Don't print large objects just to hand-copy IDs back into code.
- `submit <expr>` — final answer; ends the turn. Strings pass through as the reply; other values render as pretty JSON. If a **Required output** schema is present, the value must match it.

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
result = await TOOL.default.tool_name({ key: "value" })?
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
submit { count: len(items), first: items[0] }
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
        bullets.push("- Resource operations: call host capabilities through an explicit receiver, e.g. `await TOOL.default.tool_name({ key: value })?` or `await RESOURCE.alias.operation({ query: q })?`. `?` aborts the block with sanitized operation metadata if the operation fails.".to_string());
    }
    if abilities.processes {
        let mut forms = vec![
            "`yield value`",
            "`wake value`",
            "`finish value`",
            "`fail value`",
        ];
        if abilities.process_sleep {
            forms.push("`sleep for \"5s\"`");
            forms.push("`sleep until deadline`");
        }
        if abilities.process_signals {
            forms.push("`payload = wait signal`");
            forms.push("`signal run handle with payload`");
        }
        let mut catalog_scopes = vec!["top level"];
        if abilities.triggers {
            catalog_scopes.push("trigger bindings");
        }
        if abilities.schedules.cron {
            catalog_scopes.push("schedules");
        }
        bullets.push(format!(
            "- Background processes: declare reusable work with `process name(param: TYPE) {{ ... }}` and start it with `handle = start name(param: value)`. Process bodies use only passed resource handles, input values, locals, and builtins; catalog handles such as `TOOL.default` belong at {}. Inside a process use {}; falling off the end is `finish null`. `submit` and `print` are foreground-only and invalid inside processes. Resolve a handle with `await handle` or `(await handle)?`; cancel with `cancel handle` (best-effort).",
            join_words(&catalog_scopes),
            join_words(&forms)
        ));
    }
    if abilities.triggers && abilities.processes {
        bullets.push("- Triggers: declaration-only resource-event activation. Use `trigger name on RESOURCE.alias.event as event -> process_name(param: event, tool: RESOURCE.alias)` or `trigger name on each RESOURCE.event as resource, event -> process_name(resource_param: resource, event_param: event)`. Bind every target process parameter explicitly. Trigger args may only be the whole event payload, a concrete catalog resource alias, or the resource handle from `on each`; triggers cannot contain code, `await`, `print`, `submit`, loops, assignments, resource operations, or computed records.".to_string());
    }
    if abilities.schedules.cron {
        bullets.push("- Schedules: declare cron activation with `schedule name every cron(\"0 * * * *\") as tick { ... }`.".to_string());
    }
    if has_operations {
        let tail = if abilities.processes {
            " Use a named process for multi-step background work."
        } else {
            ""
        };
        bullets.push(format!(
            "- Independent operation fanout: call independent operations directly and store their values, e.g. `a = await TOOL.default.tool_a({{ key: \"one\" }})?` and `b = await TOOL.default.tool_b({{ key: \"two\" }})?`.{tail}"
        ));
    }
    bullets.push("- Control flow: statement `if`/`for`; `break` exits the nearest `for`; `continue` skips to the nearest `for`'s next iteration; expression ternary `cond ? yes : no` (there is no expression-form `if`); boolean negation via `!cond` or `not cond`. There is no `while` loop; use bounded `for` loops over ranges/lists for fill or retry logic. `submit` is different from `break`: it ends the whole program/turn.".to_string());
    bullets.push("- Bare expressions are valid statements in normal blocks.".to_string());
    bullets.push("- When a **Bound Variables** section appears in the prompt, those names are already in scope inside lashlang blocks — read them directly instead of restating their values.".to_string());
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
submit items
```

Print narrow observations. Keep large values in variables and print only keys, lengths, selected fields, or slices:

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
  submit first
}
```"#
        .to_string()
}

fn render_decomposition_section(has_operations: bool, processes: bool) -> String {
    let mut section = String::from(
        "### Working with context\n\nYour turn's REPL trace is your working memory. Keep it small, decision-sized, and current. Big artifacts (files, search results, long pages, raw tool dumps) live outside — pull them in only when you need to compute over them yourself. Keep full results in variables; `print` only lengths, keys, selected fields, or slices, never large objects you intend to hand-copy IDs from.\n\nChoose the lightest mechanism that preserves progress:\n\n- Current variables already hold what you need → reason inline in lashlang.",
    );
    if has_operations {
        if processes {
            section.push_str("\n- Several independent operations are needed → call each receiver operation and keep the values in variables; use a named `process` for multi-step background work.");
        } else {
            section.push_str("\n- Several independent operations are needed → call each receiver operation and keep the values in variables.");
        }
    }
    section.push_str("\n- The trace is bloated, stale, or failed attempts dominate → use an available continuation tool to hand off concrete state to a fresh successor.");
    if has_operations {
        section.push_str("\n- Anything tool-specific (parameters, return shapes, lifecycle) lives under **Showcased Tools** — don't infer a tool exists from these generic examples.\n\nExample fanout to two available operations (use `?` for fail-fast unwrapping):\n\n```lashlang\na = await TOOL.default.tool_a({ key: \"one\" })?\nb = await TOOL.default.tool_b({ key: \"two\" })?\none = a\ntwo = b\nsubmit [one, two]\n```");
    } else {
        section.push_str("\n- No receiver operations are available in this turn — don't infer one exists from generic lashlang syntax.");
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
