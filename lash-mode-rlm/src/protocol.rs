//! RLM protocol driver + all its associated helpers:
//!
//! - The [`RlmDriver`] itself — extracts the first fenced `lashlang`
//!   block from the assistant text and dispatches `StartExec`.
//! - The [`rlm_execution_section`] prompt copy.
//! - Fence-extraction utilities (`extract_first_lashlang_fence`,
//!   [`contains_closed_lashlang_fence`]).
//! - Typed-RLM schema validation and the auxiliary messages used when
//!   the model fails to produce a schema-matching `submit`.

use lash_core::sansio::{
    CheckpointResumeAction, CompletedToolCall, ProtocolDriverHandle, WaitingExecState,
    WaitingLlmState, driver_state,
};
use lash_core::session_model::{
    ConversationRecord, Message, MessageRole, Part, PartKind, PruneState, SessionEvent,
    SessionEventRecord, fresh_message_id, make_error_event, shared_parts,
};
use lash_core::{
    AttachmentRef, CheckpointKind, DriverAction, DriverContextView, ExecResponse, LlmOutputPart,
    LlmResponse, ToolCallRecord, TurnFinish, TurnOutcome, TurnStop, append_assistant_text_part,
    normalized_response_parts,
};
use lash_rlm_types::{RlmDiagnosticEvent, RlmModeEvent, RlmTermination, RlmTrajectoryEntry};
use serde_json::Value;

pub const RLM_EXECUTION_SECTION: &str = r#"**All actions go through `lashlang`.** Invoke documented tools as `call tool_name { ... }` from inside a fenced `lashlang` block. Start from tools listed under **Showcased Tools**; if a discovery tool is available, use it to find and load additional tools before calling them. Emit a block whenever you need to call an available tool or compute a value. Plain prose is for direct conversational replies that need no action.

### `print` vs `submit`

- `print <expr>` — inspect a value and keep going. Output appears on the next step and is capped: keep full tool results in variables and print only lengths, selected fields, samples, or slices. Don't print large objects just to hand-copy IDs back into code.
- `submit <expr>` — final answer; ends the turn. Strings pass through as the reply; other values render as pretty JSON. If a **Required output** schema is present, the value must match it.

Never `submit` a raw tool-result dump. If you need to look at something, `print` it, then `submit` a summary on a later step.

### Turn shape

**Exactly one ` ```lashlang ` fenced block per response.** Anything after the first block closes is dropped. Only ` ```lashlang ` is recognised — `rlm` and other labels are treated as plain prose.

- Write small blocks. Each should do one focused step: call a tool or two, `print` what you need to see, then stop.
- Keep prose around the block to one or two sentences of reasoning. Don't describe an action in prose instead of executing it.
- After each result, decide: another block (more work), or finish.

Example — inspect with an available tool, then submit:

````
Checking the available data.

```lashlang
result = (call tool_name { key: "value" })?
print { summary: slice(to_string(result), 0, 200) }
```
````

…then on the next turn, once you've seen what you need:

````
```lashlang
submit "The bound version is 0.2.61."
```
````

"#;

fn render_terminal_output_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

pub const LASHLANG_LANGUAGE_VALUES_SECTION: &str = r#"### Language

- Values: null, booleans, numbers, strings, lists, records, and immutable `Image` handles. Literals: `[a, b]`, `{ a: 1, b: 2 }`.
- Images: image-producing tools may return an `Image` value. Read metadata with `.id`, `.label`, `.size`, `.width`, `.height`; fields are read-only. `print(image)` or `print` on a list/record containing images sends both descriptor text and the actual image attachment to the next model call. `submit(image)`, `to_string(image)`, and JSON-like serialization emit only `{ "type": "image", "id": ..., "label": ..., "size": ..., "width": ..., "height": ... }`. `len(image)` is invalid; use `.size`.
- Strings: `"..."` supports `\n`, `\r`, `\t`, `\"`, and `\\`; `"""..."""` is multiline with the same escapes; `r"""..."""` and `r'''...'''` are raw multiline strings and preserve content exactly. Use raw multiline strings for JSON, Markdown, and other payloads with braces, backslashes, quotes, heredocs, or `@@` hunk markers.
- Assign with `name = expr`. Variables persist across fenced blocks within the turn. You can also update mutable collection paths rooted at a variable: `record.field = value`, `record[key] = value`, `list[i] = value`, and nested forms like `state.groups[g].count = count + 1`. Record field/index assignment inserts or replaces fields; list assignment replaces an existing integer index only. Record indexing reads dynamic string-coerced keys and returns `null` when missing, so histogram code can use `counts[g] = counts[g] + 1`.
- Call a tool: `call tool { arg: expr }`. Every tool call returns a wrapper record: `{ ok: true, value: <tool output> }` on success, `{ ok: false, error: "..." }` on failure. For the common happy path, append `?` to unwrap it: `(call tool { arg: expr })?` returns `.value` or aborts this block with the tool error. Keep the raw wrapper only when you intentionally need `.ok`, `.value`, or `.error` for branching/retry/reporting.
- Background start: `start call tool { arg: expr }` returns a **handle** (not wrapped). Some documented tools may also return handles directly. Resolve a handle with `await handle` when you want to wait for completion, or use `(await handle)?` for fail-fast unwrapping. `await [h1, h2]` returns a list of wrappers in order. Cancel with `cancel handle` (best-effort). If a handle-listing tool is available, use its documented contract to rediscover live handles.
- Independent parallel tool calls: `parallel { ... }`. Prefer named branches (`parallel { a: call ... b: call ... }`) so results come back as a record (`results.a`, `results.b`). Positional branches still return a list in order. Do not use `parallel` when one branch needs another branch's output.
- Control flow: statement `if`/`for`; `break` exits the nearest `for`; `continue` skips to the nearest `for`'s next iteration; expression ternary `cond ? yes : no` (there is no expression-form `if`); boolean negation via `!cond` or `not cond`. There is no `while` loop; use bounded `for` loops over ranges/lists for fill or retry logic. `submit` is different from `break`: it ends the whole program/turn.
- Bare expressions are valid statements. Inside `parallel { ... }`, a bare expression contributes its value to the result list.
- When a **Bound Variables** section appears in the prompt, those names are already in scope inside lashlang blocks — read them directly instead of restating their values.
"#;

pub const LASHLANG_LANGUAGE_VALUES_NO_IMAGES_SECTION: &str = r#"### Language

- Values: null, booleans, numbers, strings, lists, and records. Literals: `[a, b]`, `{ a: 1, b: 2 }`.
- Strings: `"..."` supports `\n`, `\r`, `\t`, `\"`, and `\\`; `"""..."""` is multiline with the same escapes; `r"""..."""` and `r'''...'''` are raw multiline strings and preserve content exactly. Use raw multiline strings for JSON, Markdown, and other payloads with braces, backslashes, quotes, heredocs, or `@@` hunk markers.
- Assign with `name = expr`. Variables persist across fenced blocks within the turn. You can also update mutable collection paths rooted at a variable: `record.field = value`, `record[key] = value`, `list[i] = value`, and nested forms like `state.groups[g].count = count + 1`. Record field/index assignment inserts or replaces fields; list assignment replaces an existing integer index only. Record indexing reads dynamic string-coerced keys and returns `null` when missing, so histogram code can use `counts[g] = counts[g] + 1`.
- Call a tool: `call tool { arg: expr }`. Every tool call returns a wrapper record: `{ ok: true, value: <tool output> }` on success, `{ ok: false, error: "..." }` on failure. For the common happy path, append `?` to unwrap it: `(call tool { arg: expr })?` returns `.value` or aborts this block with the tool error. Keep the raw wrapper only when you intentionally need `.ok`, `.value`, or `.error` for branching/retry/reporting.
- Background start: `start call tool { arg: expr }` returns a **handle** (not wrapped). Some documented tools may also return handles directly. Resolve a handle with `await handle` when you want to wait for completion, or use `(await handle)?` for fail-fast unwrapping. `await [h1, h2]` returns a list of wrappers in order. Cancel with `cancel handle` (best-effort). If a handle-listing tool is available, use its documented contract to rediscover live handles.
- Independent parallel tool calls: `parallel { ... }`. Prefer named branches (`parallel { a: call ... b: call ... }`) so results come back as a record (`results.a`, `results.b`). Positional branches still return a list in order. Do not use `parallel` when one branch needs another branch's output.
- Control flow: statement `if`/`for`; `break` exits the nearest `for`; `continue` skips to the nearest `for`'s next iteration; expression ternary `cond ? yes : no` (there is no expression-form `if`); boolean negation via `!cond` or `not cond`. There is no `while` loop; use bounded `for` loops over ranges/lists for fill or retry logic. `submit` is different from `break`: it ends the whole program/turn.
- Bare expressions are valid statements. Inside `parallel { ... }`, a bare expression contributes its value to the result list.
- When a **Bound Variables** section appears in the prompt, those names are already in scope inside lashlang blocks — read them directly instead of restating their values.
"#;

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

Tool-level errors are different from successful tool results that contain domain errors. `?` unwraps the lash tool wrapper and aborts the block only when the tool call itself failed:

```lashlang
probe = (call tool_name { key: "value" })?
submit probe
```

Build collections with explicit loops, not comprehensions:

```lashlang
items = []
for key in ["a", "b"] {
  value = (call tool_name { key: key })?
  items = push(items, { key: key, value: value })
}
submit items
```

Print narrow observations. Keep large values in variables and print only keys, lengths, selected fields, or slices:

```lashlang
result = (call tool_name { key: "value" })?
text = to_string(result)
print { chars: len(text), head: slice(text, 0, 1200) }
```

For multi-step work, inspect intermediate results before submitting success. Reaching `submit` ends the turn even mid-block:

```lashlang
first = (call tool_a { key: "value" })?
second = (call tool_b { input: first })?
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

const RLM_DECOMPOSITION_SECTION: &str = r#"### Working with context

Your turn's REPL trace is your working memory. Keep it small, decision-sized, and current. Big artifacts (files, search results, long pages, raw tool dumps) live outside — pull them in only when you need to compute over them yourself. Keep full results in variables; `print` only lengths, keys, selected fields, or slices, never large objects you intend to hand-copy IDs from.

Choose the lightest mechanism that preserves progress:

- Current variables already hold what you need → reason inline in lashlang.
- Several independent tool calls can run at once → `parallel { ... }` or `start call ...` + `await`.
- The trace is bloated, stale, or failed attempts dominate → use an available continuation tool to hand off concrete state to a fresh successor.
- Anything tool-specific (parameters, return shapes, lifecycle) lives under **Showcased Tools** — don't infer a tool exists from these generic examples.

Example fanout to two available tools (use `?` for fail-fast unwrapping):

```lashlang
a = start call tool_a { key: "one" }
b = start call tool_b { key: "two" }
results = parallel { one: (await a)?, two: (await b)? }
submit [results.one, results.two]
```"#;

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

pub fn rlm_execution_section() -> String {
    rlm_execution_section_with_features(RlmPromptFeatures::default())
}

pub fn rlm_execution_section_with_features(features: RlmPromptFeatures) -> String {
    let mut sections = Vec::new();
    sections.push(RLM_EXECUTION_SECTION);
    sections.push(if features.images {
        LASHLANG_LANGUAGE_VALUES_SECTION
    } else {
        LASHLANG_LANGUAGE_VALUES_NO_IMAGES_SECTION
    });
    sections.push(if features.images {
        LASHLANG_BUILTINS_SECTION
    } else {
        LASHLANG_BUILTINS_NO_IMAGES_SECTION
    });
    if features.common_patterns {
        sections.push(LASHLANG_COMMON_PATTERNS_SECTION);
    }
    if features.type_literals {
        sections.push(LASHLANG_TYPE_LITERALS_SECTION);
    }
    if features.decomposition {
        sections.push(RLM_DECOMPOSITION_SECTION);
    }
    sections.join("\n\n")
}

pub struct RlmDriver;

fn rlm_termination(options: &lash_core::ModeTurnOptions) -> RlmTermination {
    options
        .decode(&lash_core::ExecutionMode::new("rlm"))
        .ok()
        .flatten()
        .unwrap_or_default()
}

#[derive(Default)]
struct RlmDriverState {
    reasoning: String,
    tool_calls: Vec<ToolCallRecord>,
    images: Vec<AttachmentRef>,
    /// One entry per `print` from the executed lashlang block (plus any
    /// raw stdout-style emission). Replaces the old split between a
    /// concatenated `combined_output: String` and a parallel
    /// `observations: Vec<String>` — the two carried the same content.
    output: Vec<String>,
    exec_error: Option<String>,
    executed_code: Option<String>,
    terminal_finish: Option<Value>,
}

struct FenceExtraction {
    code: String,
    had_extra_fences: bool,
}

impl ProtocolDriverHandle<lash_core::HostModeProtocol> for RlmDriver {
    fn prepare_mode_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.project_llm_request(false),
            driver_state: Some(driver_state(RlmDriverState::default())),
        }]
    }

    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_>,
        mut waiting: WaitingLlmState,
        llm_response: LlmResponse,
        _text_streamed: bool,
    ) -> Vec<DriverAction> {
        let mut actions = vec![DriverAction::Emit(SessionEvent::LlmResponse {
            mode_iteration: ctx.mode_iteration(),
            content: llm_response.full_text.clone(),
            duration_ms: 0,
        })];

        let mut assistant_text = String::new();
        let mut reasoning_text = String::new();
        for part in normalized_response_parts(&llm_response) {
            match part {
                LlmOutputPart::Text { text, .. } => {
                    append_assistant_text_part(&mut assistant_text, &text);
                }
                LlmOutputPart::Reasoning { text, replay } => {
                    let reasoning = if text.trim().is_empty() {
                        replay
                            .as_ref()
                            .map(|meta| meta.summary.join("\n\n"))
                            .unwrap_or_default()
                    } else {
                        text
                    };
                    append_assistant_text_part(&mut reasoning_text, &reasoning);
                }
                LlmOutputPart::ToolCall { .. } => {}
            }
        }

        if assistant_text.trim().is_empty() && reasoning_text.trim().is_empty() {
            actions.push(DriverAction::Emit(make_error_event(
                "llm_provider",
                Some("empty_response"),
                "Model returned no assistant text.",
                None,
            )));
            actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                TurnStop::ProviderError,
            )));
            return actions;
        }

        let extraction = extract_first_lashlang_fence(&assistant_text);
        let Some(fence) = extraction else {
            let termination = rlm_termination(ctx.termination());
            if matches!(termination, RlmTermination::ProseOrSubmit) {
                actions.push(DriverAction::AppendEvents(vec![
                    diagnostic_event(
                        "llm_extraction",
                        serde_json::json!({
                            "found_lashlang_fence": false,
                            "prose_only_ends_turn": true,
                            "assistant_text_chars": assistant_text.chars().count(),
                            "reasoning_chars": reasoning_text.chars().count(),
                            "finalization_reason": "prose_or_submit",
                        }),
                    ),
                    conversation_event(internal_assistant_prose_message(assistant_text.clone())),
                ]));
                if !assistant_text.trim().is_empty() {
                    actions.push(DriverAction::Emit(SessionEvent::Message {
                        text: assistant_text.clone(),
                        kind: "final".to_string(),
                    }));
                }
                actions.push(DriverAction::StartCheckpoint {
                    checkpoint: CheckpointKind::BeforeCompletion,
                    on_empty: CheckpointResumeAction::Finish(TurnOutcome::Finished(
                        TurnFinish::AssistantMessage {
                            text: assistant_text.clone(),
                        },
                    )),
                });
                return actions;
            }
            let RlmTermination::SubmitRequired { schema } = termination else {
                unreachable!("ProseOrSubmit returned above");
            };
            actions.push(DriverAction::AppendEvents(vec![diagnostic_event(
                "llm_extraction",
                serde_json::json!({
                    "found_lashlang_fence": false,
                    "prose_only_ends_turn": false,
                    "assistant_text_chars": assistant_text.chars().count(),
                    "reasoning_chars": reasoning_text.chars().count(),
                    "finalization_reason": "submit_required",
                }),
            )]));
            let mut events = Vec::new();
            if !assistant_text.trim().is_empty() {
                events.push(conversation_event(internal_assistant_prose_message(
                    assistant_text,
                )));
            }
            events.push(conversation_event(submit_required_reminder_message(
                schema.is_some(),
            )));
            actions.push(DriverAction::AppendEvents(events));
            actions.push(DriverAction::AdvanceModeIteration);
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::AfterWork,
                on_empty: CheckpointResumeAction::PrepareIteration,
            });
            return actions;
        };

        actions.push(DriverAction::AppendEvents(vec![diagnostic_event(
            "llm_extraction",
            serde_json::json!({
                "found_lashlang_fence": true,
                "had_extra_fences": fence.had_extra_fences,
                "code_chars": fence.code.chars().count(),
                "assistant_text_chars": assistant_text.chars().count(),
                "reasoning_chars": reasoning_text.chars().count(),
                "decision": "execute_lashlang",
            }),
        )]));

        let mut state = waiting
            .take_driver_state::<RlmDriverState>()
            .unwrap_or_default();
        state.executed_code = Some(fence.code.clone());
        state.reasoning = combine_reasoning_and_text(&reasoning_text, &assistant_text);

        // Emit the raw lashlang source as a `Message` with kind
        // `lashlang_code` so the CLI can reveal it in the full-expand
        // view (Alt+O) above the tool activities it produced.
        actions.push(DriverAction::Emit(SessionEvent::Message {
            text: fence.code.clone(),
            kind: "lashlang_code".to_string(),
        }));
        actions.push(DriverAction::StartExec {
            code: fence.code,
            driver_state: driver_state(state),
        });
        actions
    }

    fn handle_tool_results(
        &self,
        _ctx: DriverContextView<'_>,
        _completed: Vec<CompletedToolCall>,
    ) -> Vec<DriverAction> {
        Vec::new()
    }

    fn handle_exec_result(
        &self,
        ctx: DriverContextView<'_>,
        waiting: WaitingExecState,
        result: Result<ExecResponse, String>,
    ) -> Vec<DriverAction> {
        let mut state = waiting
            .into_driver_state::<RlmDriverState>()
            .unwrap_or_default();
        let mut actions = Vec::new();

        match result {
            Ok(response) => {
                let terminal_outcome = response
                    .tool_calls
                    .iter()
                    .find_map(terminal_outcome_from_tool_result);
                for tool_call in &response.tool_calls {
                    actions.push(DriverAction::Emit(SessionEvent::ToolCall {
                        call_id: tool_call.call_id.clone(),
                        name: tool_call.tool.clone(),
                        args: tool_call.args.clone(),
                        result: tool_call.result.clone(),
                        success: tool_call.success,
                        duration_ms: tool_call.duration_ms,
                    }));
                }
                state.tool_calls.extend(response.tool_calls);
                state.images.extend(response.printed_images);
                if !response.output.is_empty() {
                    // Raw lashlang-runtime stdout (rare); treat as an
                    // anonymous output entry alongside the typed prints.
                    state.output.push(response.output);
                }
                for observation in response.observations {
                    if !observation.is_empty() {
                        state.output.push(observation);
                    }
                }
                if let Some(raw_error) = response.error {
                    state.exec_error = Some(raw_error);
                }
                if let Some(finish_value) = response.terminal_finish {
                    state.terminal_finish = Some(finish_value);
                }
                if let Some(outcome) = terminal_outcome {
                    actions.push(DriverAction::AppendEvents(vec![trajectory_event(
                        trajectory_entry(ctx.mode_iteration(), &state, None, None),
                    )]));
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish(outcome),
                    });
                    return actions;
                }
            }
            Err(error) => {
                state.exec_error = Some(error);
            }
        }

        if let Some(finish_value) = &state.terminal_finish {
            // Typed-RLM: validate against the declared schema. If it fails,
            // surface the error to the model and loop; otherwise fall
            // through to the shared terminate-with-value path below.
            if let RlmTermination::SubmitRequired {
                schema: Some(schema),
            } = rlm_termination(ctx.termination())
                && let Err(error_text) = validate_finish_value(finish_value, &schema)
            {
                actions.push(DriverAction::AppendEvents(vec![
                    trajectory_event(trajectory_entry(
                        ctx.mode_iteration(),
                        &state,
                        Some(error_text.clone()),
                        None,
                    )),
                    conversation_event(submit_schema_mismatch_message(&error_text)),
                ]));
                actions.push(DriverAction::AdvanceModeIteration);
                actions.push(DriverAction::StartCheckpoint {
                    checkpoint: CheckpointKind::AfterWork,
                    on_empty: CheckpointResumeAction::PrepareIteration,
                });
                return actions;
            }

            let rendered = render_terminal_output_value(finish_value);
            actions.push(DriverAction::AppendEvents(vec![trajectory_event(
                trajectory_entry(
                    ctx.mode_iteration(),
                    &state,
                    None,
                    Some(finish_value.clone()),
                ),
            )]));
            if !rendered.trim().is_empty() {
                actions.push(DriverAction::Emit(SessionEvent::Message {
                    text: rendered.clone(),
                    kind: "final".to_string(),
                }));
            }
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish(TurnOutcome::Finished(
                    TurnFinish::SubmittedValue {
                        value: finish_value.clone(),
                    },
                )),
            });
            return actions;
        }

        actions.push(DriverAction::AppendEvents(vec![trajectory_event(
            trajectory_entry(ctx.mode_iteration(), &state, None, None),
        )]));
        actions.push(DriverAction::AdvanceModeIteration);
        if ctx.should_force_exit_after_grace_turn() {
            actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                TurnStop::MaxTurns,
            )));
            return actions;
        }
        actions.push(DriverAction::ScheduleTurnLimitFinal);
        actions.push(DriverAction::StartCheckpoint {
            checkpoint: CheckpointKind::AfterWork,
            on_empty: CheckpointResumeAction::PrepareIteration,
        });
        actions
    }
}

fn terminal_outcome_from_tool_result(record: &ToolCallRecord) -> Option<TurnOutcome> {
    if !record.success {
        return None;
    }
    match record.control.as_ref()? {
        lash_core::ToolControl::Handoff { session_id } if !session_id.trim().is_empty() => {
            Some(TurnOutcome::Handoff {
                session_id: session_id.clone(),
            })
        }
        lash_core::ToolControl::Finish { value } => {
            Some(TurnOutcome::Finished(TurnFinish::ToolValue {
                tool_name: record.tool.clone(),
                value: value.clone(),
            }))
        }
        lash_core::ToolControl::Fail { value } => Some(TurnOutcome::Stopped(TurnStop::ToolError {
            tool_name: record.tool.clone(),
            value: value.clone(),
        })),
        lash_core::ToolControl::Handoff { .. } => None,
    }
}

fn trajectory_entry(
    mode_iteration: usize,
    state: &RlmDriverState,
    validation_error: Option<String>,
    final_output: Option<Value>,
) -> RlmTrajectoryEntry {
    RlmTrajectoryEntry {
        id: format!("rlm_step_{mode_iteration}"),
        mode_iteration,
        reasoning: state.reasoning.clone(),
        code: state.executed_code.clone().unwrap_or_default(),
        output: state.output.clone(),
        tool_calls: state.tool_calls.clone(),
        images: state.images.clone(),
        error: validation_error.or_else(|| state.exec_error.clone()),
        final_output,
    }
}

fn conversation_event(message: Message) -> SessionEventRecord {
    SessionEventRecord::Conversation(ConversationRecord::from_message(message))
}

fn trajectory_event(entry: RlmTrajectoryEntry) -> SessionEventRecord {
    SessionEventRecord::Mode(crate::rlm_mode_event(RlmModeEvent::RlmTrajectoryEntry(
        entry,
    )))
}

fn diagnostic_event(phase: &str, payload: Value) -> SessionEventRecord {
    SessionEventRecord::Mode(crate::rlm_mode_event(RlmModeEvent::RlmDiagnostic(
        RlmDiagnosticEvent {
            phase: phase.to_string(),
            payload,
        },
    )))
}

// ─────────────────────────────────────────────────────────────────────
//  Shared helpers: fence extraction, typed-RLM messages, schema check
// ─────────────────────────────────────────────────────────────────────

/// Return `true` when `text` contains a complete ` ```lashlang ` fenced
/// block — opener of N backticks followed by a closing run of M ≥ N
/// backticks anywhere (no newline requirement on either side). Exposed
/// for the stream mask, which raises `abort_stream` as soon as this is
/// true.
pub fn contains_closed_lashlang_fence(text: &str) -> bool {
    first_lashlang_fence_span(text).is_some_and(|span| span.close_len > 0)
}

#[derive(Debug, Clone, Copy)]
struct FenceSpan {
    /// Byte offset of the first byte of the body (after the opener
    /// line's terminating `\n`).
    body_start: usize,
    /// Byte offset of the first backtick of the closer (or `text.len()`
    /// when the body extends to EOF without a closer).
    body_end: usize,
    /// Number of backticks consumed by the closer. `0` when unclosed.
    /// Always ≤ `opener_len` — extra backticks in the closer run stay
    /// in the residual text so adjacent fences glued together (the
    /// "``````lashlang" pattern) still parse the way they always did.
    close_len: usize,
}

fn first_lashlang_fence_span(text: &str) -> Option<FenceSpan> {
    let bytes = text.as_bytes();
    let mut search_from = 0usize;
    while search_from < bytes.len() {
        let rel = text[search_from..].find("```")?;
        let open = search_from + rel;
        // CommonMark-style variable-length fences: count consecutive
        // backticks at the opener. N must be ≥3 (find("```") guarantees
        // that). Anything past N backticks is part of the opener run
        // and the matching closer must be at least N backticks long.
        let opener_len = bytes[open..].iter().take_while(|&&b| b == b'`').count();
        let after_open = open + opener_len;
        // The opening `\`\`\`` doesn't have to be on its own line —
        // an inline opener after prose still parses. The `lashlang`
        // language tag is distinctive enough that collisions with
        // inline prose are essentially impossible.
        let rest = &text[after_open..];
        let lang_end = rest.find('\n').unwrap_or(rest.len());
        let lang = rest[..lang_end].trim();
        if lang != "lashlang" {
            // Skip past this opener run and keep looking — a non-
            // lashlang language tag belongs to a different code block.
            search_from = after_open;
            continue;
        }
        let body_start = after_open + lang_end + 1;
        if body_start > text.len() {
            return None;
        }

        // Closer: the first run of ≥`opener_len` consecutive backticks
        // after `body_start`. We consume exactly `opener_len` of them
        // — leftover backticks in the run stay in the text so the
        // "``````lashlang" double-fence pattern still resolves into
        // two adjacent blocks.
        let (close, close_len) = match find_closing_fence(&bytes[body_start..], opener_len) {
            Some((rel, _run_len)) => (body_start + rel, opener_len),
            None => (text.len(), 0),
        };
        return Some(FenceSpan {
            body_start,
            body_end: close,
            close_len,
        });
    }
    None
}

/// Find the first run of consecutive backticks of length ≥ `min_len` in
/// `bytes`. Returns `(start_byte_offset, run_length)`.
fn find_closing_fence(bytes: &[u8], min_len: usize) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i;
            while i < bytes.len() && bytes[i] == b'`' {
                i += 1;
            }
            let len = i - start;
            if len >= min_len {
                return Some((start, len));
            }
        } else {
            i += 1;
        }
    }
    None
}

fn extract_first_lashlang_fence(text: &str) -> Option<FenceExtraction> {
    let span = first_lashlang_fence_span(text)?;
    let code = text[span.body_start..span.body_end]
        .trim_end_matches('\n')
        .to_string();
    let after_close = (span.body_end + span.close_len).min(text.len());
    let had_extra_fences = first_lashlang_fence_span(&text[after_close..]).is_some();
    Some(FenceExtraction {
        code,
        had_extra_fences,
    })
}

fn combine_reasoning_and_text(reasoning: &str, text: &str) -> String {
    match (reasoning.trim().is_empty(), text.trim().is_empty()) {
        (true, true) => String::new(),
        (true, false) => text.to_string(),
        (false, true) => reasoning.to_string(),
        (false, false) => format!("{reasoning}\n\n{text}"),
    }
}

fn internal_assistant_prose_message(content: String) -> Message {
    prose_message(
        content,
        Some(lash_core::MessageOrigin::Plugin {
            plugin_id: "mode_rlm".to_string(),
            transient: false,
        }),
    )
}

fn prose_message(content: String, origin: Option<lash_core::MessageOrigin>) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::Assistant,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Prose,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin,
    }
}

fn submit_required_reminder_message(requires_schema: bool) -> Message {
    let id = fresh_message_id();
    let content = if requires_schema {
        "Deliver the final answer from a fenced ```lashlang block by calling `submit <value>` with a value matching the required output schema. Plain text outside a fence is not delivered."
    } else {
        "Deliver the final answer from a fenced ```lashlang block by calling `submit <value>`. Plain text outside a fence is not delivered."
    };
    Message {
        id: id.clone(),
        role: MessageRole::System,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: content.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: Some(lash_core::MessageOrigin::Plugin {
            plugin_id: "mode_rlm".to_string(),
            transient: false,
        }),
    }
}

fn submit_schema_mismatch_message(error_text: &str) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::System,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: format!(
                "The `submit` value didn't match the required output schema:\n{error_text}\n\nFix the value and call `submit <corrected>` from another fenced ```lashlang block."
            ),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: Some(lash_core::MessageOrigin::Plugin {
            plugin_id: "mode_rlm".to_string(),
            transient: false,
        }),
    }
}

fn validate_finish_value(value: &Value, schema: &Value) -> Result<(), String> {
    let compiled = jsonschema::JSONSchema::compile(schema)
        .map_err(|err| format!("required output schema is invalid: {err}"))?;
    if let Err(errors) = compiled.validate(value) {
        let message = errors
            .map(|err| err.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(message);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rlm_execution_section_default_prompt_is_golden() {
        insta::assert_snapshot!("rlm_execution_section_default", rlm_execution_section());
    }

    #[test]
    fn rlm_execution_section_no_images_prompt_is_golden() {
        insta::assert_snapshot!(
            "rlm_execution_section_no_images",
            rlm_execution_section_with_features(RlmPromptFeatures {
                images: false,
                ..RlmPromptFeatures::default()
            })
        );
    }

    #[test]
    fn execution_section_does_not_advertise_unregistered_peer_capability() {
        let section = rlm_execution_section();

        assert!(!section.contains("capability: \"peer\""));
        assert!(!section.contains("`peer`"));
    }

    #[test]
    fn execution_section_keeps_tool_specific_examples_out_of_core_prompt() {
        let section = rlm_execution_section();

        for tool_name in [
            "read_file",
            "exec_command",
            "apply_patch",
            "llm_query",
            "spawn_agent",
            "continue_as",
            "list_async_handles",
        ] {
            assert!(
                !section.contains(tool_name),
                "core RLM prompt should not mention tool-specific example `{tool_name}`"
            );
        }
    }

    #[test]
    fn execution_section_can_disable_image_guidance() {
        let section = rlm_execution_section_with_features(RlmPromptFeatures {
            images: false,
            ..RlmPromptFeatures::default()
        });

        assert!(!section.contains("Image"));
        assert!(!section.contains("image.size"));
        assert!(section.contains("### Language"));
        assert!(section.contains("### Builtins"));
        assert!(section.contains("### Common patterns"));
        assert!(section.contains("### Type literals"));
    }

    #[test]
    fn execution_section_states_no_while_loop() {
        let section = rlm_execution_section();

        assert!(section.contains("There is no `while` loop"));
        assert!(section.contains("use bounded `for` loops over ranges/lists"));
    }

    #[test]
    fn fence_detector_accepts_inline_opener_after_prose() {
        // Regression: reasoning models emit the opening fence mid-line:
        // `…required output shape.```lashlang\n…`. Requiring newline
        // before ``` caused the detector to miss the block entirely,
        // which made the RLM turn terminate after one mode_iteration
        // without executing anything.
        let text = "I'll inspect the prompt.```lashlang\nprint slice(input.prompt, 0, 10)\n```";
        let extraction = extract_first_lashlang_fence(text)
            .expect("inline opener with newline-terminated closer should parse");
        assert_eq!(extraction.code, "print slice(input.prompt, 0, 10)");
        assert!(contains_closed_lashlang_fence(text));
    }

    #[test]
    fn fence_detector_still_accepts_newline_preceded_opener() {
        let text = "prose\n\n```lashlang\nsubmit 1\n```";
        let extraction = extract_first_lashlang_fence(text).expect("should parse");
        assert_eq!(extraction.code, "submit 1");
    }

    #[test]
    fn fence_detector_closer_matches_anywhere() {
        // `\`\`\`` closes the block wherever it appears. Simpler mental
        // model: `\`\`\`lashlang` starts, `\`\`\`` stops. No newline
        // requirement on either side.
        let text = "```lashlang\nsubmit 1``` more prose";
        let extraction = extract_first_lashlang_fence(text).expect("should extract");
        assert_eq!(extraction.code, "submit 1");
    }

    #[test]
    fn fence_detector_recovers_from_double_triple_concatenation() {
        // Reasoning-mode output sometimes emits ``` ``` back-to-back
        // (closer of one block immediately followed by opener of the
        // next with no prose between). The detector should still find
        // the first valid block.
        let text = "lead-in.```lashlang\nprint 1\n``````lashlang\nprint 2\n```";
        let extraction = extract_first_lashlang_fence(text)
            .expect("should extract the first block even with glued-on second block");
        assert_eq!(extraction.code, "print 1");
        assert!(extraction.had_extra_fences);
    }

    #[test]
    fn fence_detector_ignores_unknown_lang_tag() {
        // `python` is not lashlang — the detector must look further.
        let text = "```python\nprint('x')\n```\n\n```lashlang\nsubmit 1\n```";
        let extraction = extract_first_lashlang_fence(text).expect("should skip python block");
        assert_eq!(extraction.code, "submit 1");
    }

    #[test]
    fn fence_detector_four_backticks_allows_embedded_triple() {
        // Variable-length fences: opener of 4 backticks lets the body
        // contain literal ``` (which would otherwise terminate a 3-bt
        // block). Closer must be ≥4 backticks.
        let text = "````lashlang\nprint \"```\"\nsubmit 1\n````";
        let extraction = extract_first_lashlang_fence(text)
            .expect("4-backtick fence should allow embedded triple-backticks");
        assert_eq!(extraction.code, "print \"```\"\nsubmit 1");
        assert!(contains_closed_lashlang_fence(text));
    }

    #[test]
    fn fence_detector_four_backtick_opener_accepts_longer_closer() {
        // CommonMark allows the closer to be longer than the opener.
        // 4-backtick opener closed by 5-backtick closer.
        let text = "````lashlang\nsubmit 1\n`````";
        let extraction =
            extract_first_lashlang_fence(text).expect("4-bt opener should accept 5-bt closer");
        assert_eq!(extraction.code, "submit 1");
    }

    #[test]
    fn fence_detector_four_backtick_opener_ignores_inner_triple() {
        // Inner ``` runs (length 3 < opener 4) are NOT closers — body
        // continues until a run of ≥4 backticks (or EOF).
        let text = "````lashlang\nbody with ``` inside\nmore body\n````\ntail";
        let extraction = extract_first_lashlang_fence(text)
            .expect("4-bt opener should not be closed by inner triple");
        assert_eq!(extraction.code, "body with ``` inside\nmore body");
    }
}
