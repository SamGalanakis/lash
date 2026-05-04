//! RLM protocol driver + all its associated helpers:
//!
//! - The [`RlmDriver`] itself — extracts the first fenced `lashlang`
//!   block from the assistant text and dispatches `StartExec`.
//! - The [`rlm_execution_section`] prompt copy.
//! - Fence-extraction utilities (`extract_first_lashlang_fence`,
//!   [`contains_closed_lashlang_fence`]).
//! - Typed-RLM schema validation and the auxiliary messages used when
//!   the model fails to produce a schema-matching `submit`.

use lash::sansio::{
    CheckpointResumeAction, CompletedToolCall, ProtocolDriverHandle, WaitingExecState,
    WaitingLlmState, driver_state,
};
use lash::session_model::{
    ConversationRecord, Message, MessageRole, ModeEvent, Part, PartKind, PruneState, SessionEvent,
    SessionEventRecord, fresh_message_id, make_error_event, shared_parts,
};
use lash::{
    AttachmentRef, CheckpointKind, DriverAction, DriverContextView, ExecResponse, LlmOutputPart,
    LlmResponse, ToolCallRecord, TurnFinish, TurnOutcome, TurnStop, append_assistant_text_part,
    normalized_response_parts,
};
use lash_rlm_types::{RlmDiagnosticEvent, RlmModeEvent, RlmTermination, RlmTrajectoryEntry};
use serde_json::Value;

pub const RLM_EXECUTION_SECTION: &str = r#"**All actions go through `lashlang`.** Tools listed under **Showcased Tools** and catalogued tools found with `search_tools` are invoked as `call tool_name { ... }` from inside a fenced `lashlang` block. Emit a block whenever you need to call a tool, read a file, run a command, search the repo, spawn a subagent, or compute a value. Plain prose is for direct conversational replies that need no action.

### `print` vs `submit`

- `print <expr>` — inspect a value and keep going. Output appears on the next step so you can refine. Use this to peek at tool results, check a slice of a file, or confirm a value before acting.
- `submit <expr>` — final answer; ends the turn. Strings pass through as the reply; other values render as pretty JSON. If a **Required output** schema is present, the value must match it.

Never `submit` a raw tool-result dump. If you need to look at something, `print` it, then `submit` a summary on a later step.

### Turn shape

**Exactly one ` ```lashlang ` fenced block per response.** Anything after the first block closes is dropped. Only ` ```lashlang ` is recognised — `rlm` and other labels are treated as plain prose.

- Write small blocks. Each should do one focused step: call a tool or two, `print` what you need to see, then stop.
- Keep prose around the block to one or two sentences of reasoning. Don't describe an action in prose instead of executing it.
- After each result, decide: another block (more work), or finish.

Example — inspect, then submit:

````
Checking the bound version.

```lashlang
text = (call read_file { path: "Cargo.toml" })?
print slice(text, 0, 200)
```
````

…then on the next turn, once you've seen what you need:

````
```lashlang
submit "The bound version is 0.2.61."
```
````

"#;

pub const LASHLANG_LANGUAGE_REFERENCE: &str = r#"### Language

- Values: null, booleans, numbers, strings, lists, records, and immutable `Image` handles. Literals: `[a, b]`, `{ a: 1, b: 2 }`.
- Images: image-producing tools such as `read_file` on a PNG/JPEG return an `Image` value. Read metadata with `.id`, `.label`, `.size`, `.width`, `.height`; fields are read-only. `print(image)` or `print` on a list/record containing images sends both descriptor text and the actual image attachment to the next model call. `submit(image)`, `to_string(image)`, and JSON-like serialization emit only `{ "type": "image", "id": ..., "label": ..., "size": ..., "width": ..., "height": ... }`. `len(image)` is invalid; use `.size`.
- Strings: `"..."` supports `\n`, `\r`, `\t`, `\"`, and `\\`; `"""..."""` is multiline with the same escapes; `r"""..."""` and `r'''...'''` are raw multiline strings and preserve content exactly. Use raw multiline strings for patches, scripts, JSON, Markdown, and other payloads with braces, backslashes, quotes, heredocs, or `@@` hunk markers.
- Assign with `name = expr`. Variables persist across fenced blocks within the turn. You can also update mutable collection paths rooted at a variable: `record.field = value`, `record[key] = value`, `list[i] = value`, and nested forms like `state.groups[g].count = count + 1`. Record field/index assignment inserts or replaces fields; list assignment replaces an existing integer index only. Record indexing reads dynamic string-coerced keys and returns `null` when missing, so histogram code can use `counts[g] = counts[g] + 1`.
- Call a tool: `call tool { arg: expr }`. Every tool call returns a wrapper record: `{ ok: true, value: <tool output> }` on success, `{ ok: false, error: "..." }` on failure. For the common happy path, append `?` to unwrap it: `(call tool { arg: expr })?` returns `.value` or aborts this block with the tool error. Keep the raw wrapper only when you intentionally need `.ok`, `.value`, or `.error` for branching/retry/reporting.
- Background start: `start call tool { arg: expr }` returns a **handle** (not wrapped). `call monitor { ... }` also returns a handle because monitors are long-lived background event sources. Resolve a handle with `await handle` when you want to wait for completion, or use `(await handle)?` for fail-fast unwrapping. `await [h1, h2]` returns a list of wrappers in order. Cancel with `cancel handle` (best-effort). Use `list_async_handles` to rediscover live monitor, subagent, and tool handles.
- Independent parallel tool calls: `parallel { ... }`. Prefer named branches (`parallel { a: call ... b: call ... }`) so results come back as a record (`results.a`, `results.b`). Positional branches still return a list in order. Do not use `parallel` when one branch needs another branch's output.
- Control flow: statement `if`/`for`; `break` exits the nearest `for`; `continue` skips to the nearest `for`'s next iteration; expression ternary `cond ? yes : no` (there is no expression-form `if`); boolean negation via `!cond` or `not cond`. `submit` is different from `break`: it ends the whole program/turn.
- Bare expressions are valid statements. Inside `parallel { ... }`, a bare expression contributes its value to the result list.
- If the prompt includes a **Bound Variables** section, those names are already in scope — use them, don't rebuild them from prose.

### Builtins

Call as functions (e.g. `len(x)`, `slice(s, 0, 200)`). For `slice`, `null` bounds mean start/end; negative bounds count from the end.

- `len(x)` — length of string/list/record (0 for null); use `image.size` for images
- `empty(x)` — true if length is 0
- `slice(s, start, end)` — substring or sublist
- `range(end)` / `range(start, end)` — integer list, end-exclusive
- `push(list, item)` — new list with one item appended
- `split(s, sep)` / `join(list, sep)` — string split/join
- `trim(s)` — strip whitespace
- `starts_with(s, prefix)` / `ends_with(s, suffix)` / `contains(haystack, needle)`
- `keys(record)` / `values(record)`
- `to_string(x)` / `to_int(x)` / `to_float(x)`
- `json_parse(s)` — parse a JSON string into a value
- `format(template, arg0, arg1, ...)` — positional interpolation: `{}` auto-numbers, `{0}` / `{1}` pick a specific arg, `{{` / `}}` escape literal braces. Do not wrap args in a list: use `format("It is {}.", trim(now.output))`, not `format("It is {}.", [trim(now.output)])`.
- `validate(value, Type { ... })` — check an intermediate value against a Type literal and return it unchanged, or abort with a validation error

### Common patterns

Expected shell failures are shell results, not lash tool failures. `allow_nonzero_exit` exists on shell tools only; with it, `?` unwraps the lash tool wrapper and you inspect `exit_code` yourself:

```lashlang
probe = (call exec_command { cmd: "test -f Cargo.lock", allow_nonzero_exit: true })?
submit probe.exit_code == 0 ? "Cargo.lock exists" : "Cargo.lock is missing"
```

Build collections with explicit loops, not comprehensions:

```lashlang
items = []
for path in ["Cargo.toml", "README.md"] {
  text = (call read_file { path: path })?
  items = push(items, { path: path, chars: len(text) })
}
submit items
```

Print narrow observations. Keep large values in variables and print only keys, lengths, selected fields, or slices:

```lashlang
text = (call read_file { path: "Cargo.toml" })?
print { chars: len(text), head: slice(text, 0, 1200) }
```

For non-trivial edits, patch first, then validate before submitting success:

```lashlang
patch = r"""*** Begin Patch
*** Update File: src/lib.rs
@@
-old
+new
*** End Patch"""
(call apply_patch { input: patch })?
check = (call exec_command { cmd: "cargo check --workspace --all-targets", allow_nonzero_exit: true })?
if check.exit_code != 0 {
  print slice(check.output, 0, 4000)
} else {
  submit "Edit applied and validation passed."
}
```

### Type literals

`Type { field: shape, ... }` describes a record shape. Field separators are commas (trailing comma OK).

- Scalars: `str`, `int`, `float`, `bool`, `dict`, `any`, `null`.
- Collections: `list[shape]`, `enum["a", "b"]`, nested `Type { ... }`.
- **Optional field** — put `?` after the type: `email: str?` means the field may be absent from the record. If the field IS present, its value must be a string; `null` is **not** allowed.
- **Nullable field** — use a union with `null`: `email: str | null` means the field is required and its value is either a string or null.
- **Unions** — `a | b | c`, e.g. `status: str | int`, `value: str | null`.
- Nested shapes require the `Type` keyword: `nested: Type { ok: bool }` (bare `{ ok: bool }` is rejected — that's a record value, not a type).
"#;

const RLM_DECOMPOSITION_SECTION: &str = r#"### Working with context

Your turn's REPL trace is your working memory. Keep it small, decision-sized, and current. Big artifacts (files, search results, long pages, raw tool dumps) live outside — pull them in only when you need to compute over them yourself. Keep full results in variables; `print` only lengths, keys, selected fields, or slices, never large objects you intend to hand-copy IDs from.

Choose the lightest mechanism that preserves progress:

| Situation | Use | Why |
|---|---|---|
| Small task, data already in current variables | Inline lashlang / direct reasoning | No extra model call or context needed |
| Semantic extraction, summarization, classification, judging, or transformation over data already in variables | `llm_query` | Cheap one-shot LLM call; no child session, no tools, no REPL |
| Need file/repo/web inspection, shell commands, validation, edits, or multi-step investigation in isolated context | `spawn_agent` | Child session can use tools and return focused facts |
| Several independent investigations can run in parallel | `start call spawn_agent` + `await` / `parallel` | Fanout while keeping the parent trace small |
| Current trace is bloated/stale, failed attempts dominate, or context is tight | `continue_as` | Tail-call to a clean successor with only packed state |
| Read-only investigation | `spawn_agent` with `capability: "explore"` | Safer restricted default |
| Needs edits or recursive spawning | `spawn_agent` with `capability: "peer"` | Broader authority |

Hard boundaries:

- Do not use `llm_query` if the answer requires reading files, running commands, searching, fetching URLs, inspecting repository state, or using any tool. First gather the needed data inline or with a subagent; then use `llm_query` only on the gathered data.
- `spawn_agent` branches work and returns a result to the current session. Use it when the parent should keep its current state and collect a focused result.
- `continue_as` transfers the whole task to a successor and ends the current session. Use it when the current session state is no longer worth preserving.

Anti-patterns: don't spawn a subagent for a trivial check you can do inline; don't use `llm_query` to avoid reading required source material; don't use `continue_as` to delegate one subtask; don't pass bulky raw logs/search results into `continue_as.seed`; don't use `peer` when `explore` is enough.

Example fanout to two subagents (use `?` for fail-fast unwrapping):

```lashlang
a = start call spawn_agent { agent_name: "read_chunk_1", task: "Read chunk 1 and extract the key claim", capability: "explore", output: { claim: "str" } }
b = start call spawn_agent { agent_name: "read_chunk_2", task: "Read chunk 2 and extract the key claim", capability: "explore", output: { claim: "str" } }
handles = (call list_async_handles {})?
results = parallel { one: (await handles.subagent.read_chunk_1)?, two: (await handles.subagent.read_chunk_2)? }
submit [results.one.claim, results.two.claim]
```"#;

pub fn rlm_execution_section() -> String {
    format!(
        "{RLM_EXECUTION_SECTION}\n\n{LASHLANG_LANGUAGE_REFERENCE}\n\n{RLM_DECOMPOSITION_SECTION}"
    )
}

pub struct RlmDriver;

fn rlm_termination(options: &lash::ModeTurnOptions) -> RlmTermination {
    options
        .decode(&lash::ExecutionMode::new("rlm"))
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

impl ProtocolDriverHandle<lash::HostModeProtocol> for RlmDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
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
            iteration: ctx.iteration(),
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
                LlmOutputPart::Reasoning { text, summary, .. } => {
                    let reasoning = if text.trim().is_empty() {
                        summary.join("\n\n")
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
            let RlmTermination::Finish {
                schema,
                include_submit_prompt,
            } = rlm_termination(ctx.termination());
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
                events.push(conversation_event(assistant_prose_message(assistant_text)));
            }
            events.push(conversation_event(submit_required_reminder_message(
                schema.is_some(),
                include_submit_prompt,
            )));
            actions.push(DriverAction::AppendEvents(events));
            actions.push(DriverAction::AdvanceIteration);
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
                let continue_as_successor = response
                    .tool_calls
                    .iter()
                    .find_map(continue_as_successor_from_tool_result);
                let submitted_error = response
                    .tool_calls
                    .iter()
                    .rev()
                    .find(|record| record.tool == "submit_error" && record.success)
                    .map(|record| record.args.clone());
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
                if let Some(successor_session_id) = continue_as_successor {
                    actions.push(DriverAction::AppendEvents(vec![trajectory_event(
                        trajectory_entry(ctx.iteration(), &state, None, None),
                    )]));
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish(TurnOutcome::Handoff {
                            session_id: successor_session_id,
                        }),
                    });
                    return actions;
                }
                if let Some(value) = submitted_error {
                    actions.push(DriverAction::AppendEvents(vec![trajectory_event(
                        trajectory_entry(ctx.iteration(), &state, None, None),
                    )]));
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish(TurnOutcome::Stopped(
                            TurnStop::SubmittedError {
                                channel_id: "subagent.submit_error".to_string(),
                                value,
                            },
                        )),
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
            if let RlmTermination::Finish {
                schema: Some(schema),
                ..
            } = rlm_termination(ctx.termination())
                && let Err(error_text) = validate_finish_value(finish_value, &schema)
            {
                actions.push(DriverAction::AppendEvents(vec![
                    trajectory_event(trajectory_entry(
                        ctx.iteration(),
                        &state,
                        Some(error_text.clone()),
                        None,
                    )),
                    conversation_event(submit_schema_mismatch_message(&error_text)),
                ]));
                actions.push(DriverAction::AdvanceIteration);
                actions.push(DriverAction::StartCheckpoint {
                    checkpoint: CheckpointKind::AfterWork,
                    on_empty: CheckpointResumeAction::PrepareIteration,
                });
                return actions;
            }

            let rendered = match finish_value {
                Value::Null => String::new(),
                Value::String(text) => text.clone(),
                other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
            };
            actions.push(DriverAction::AppendEvents(vec![trajectory_event(
                trajectory_entry(ctx.iteration(), &state, None, Some(finish_value.clone())),
            )]));
            if !rendered.trim().is_empty() {
                actions.push(DriverAction::Emit(SessionEvent::Message {
                    text: rendered.clone(),
                    kind: "final".to_string(),
                }));
                actions.push(DriverAction::AppendEvents(vec![conversation_event(
                    assistant_prose_message(rendered),
                )]));
            }
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish(TurnOutcome::Finished(
                    TurnFinish::Submission {
                        channel_id: "rlm.submit".to_string(),
                        value: finish_value.clone(),
                    },
                )),
            });
            return actions;
        }

        actions.push(DriverAction::AppendEvents(vec![trajectory_event(
            trajectory_entry(ctx.iteration(), &state, None, None),
        )]));
        actions.push(DriverAction::AdvanceIteration);
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

fn continue_as_successor_from_tool_result(record: &ToolCallRecord) -> Option<String> {
    if record.tool != "continue_as" || !record.success {
        return None;
    }
    record
        .result
        .get("_continue_as")
        .and_then(Value::as_str)
        .filter(|session_id| !session_id.trim().is_empty())
        .map(ToOwned::to_owned)
}

fn trajectory_entry(
    iteration: usize,
    state: &RlmDriverState,
    validation_error: Option<String>,
    final_output: Option<Value>,
) -> RlmTrajectoryEntry {
    RlmTrajectoryEntry {
        id: format!("rlm_step_{iteration}"),
        iteration,
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
    SessionEventRecord::Mode(ModeEvent::rlm(RlmModeEvent::RlmTrajectoryEntry(entry)))
}

fn diagnostic_event(phase: &str, payload: Value) -> SessionEventRecord {
    SessionEventRecord::Mode(ModeEvent::rlm(RlmModeEvent::RlmDiagnostic(
        RlmDiagnosticEvent {
            phase: phase.to_string(),
            payload,
        },
    )))
}

// ─────────────────────────────────────────────────────────────────────
//  Shared helpers: fence extraction, typed-RLM messages, schema check
// ─────────────────────────────────────────────────────────────────────

/// Return `true` when `text` contains a complete ` ```lashlang ` (or
/// `rlm`/`lash`) fenced block — opener of N backticks followed by a
/// closing run of M ≥ N backticks anywhere (no newline requirement on
/// either side). Exposed for the stream mask, which raises
/// `abort_stream` as soon as this is true.
pub fn contains_closed_lashlang_fence(text: &str) -> bool {
    first_lashlang_fence_span(text).is_some_and(|span| span.close_len > 0)
}

#[derive(Debug, Clone, Copy)]
struct FenceSpan {
    /// Byte offset of the first backtick of the opener.
    #[allow(dead_code)]
    open_pos: usize,
    /// Number of backticks in the opener (≥3).
    #[allow(dead_code)]
    opener_len: usize,
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
        // reasoning models commonly emit `…required shape.\`\`\`lashlang\n`
        // inline after prose. The language tag is distinctive enough
        // (`lashlang`/`rlm`/`lash`) that collisions with inline prose
        // are essentially impossible.
        let rest = &text[after_open..];
        let lang_end = rest.find('\n').unwrap_or(rest.len());
        let lang = rest[..lang_end].trim();
        if !matches!(lang, "lashlang" | "rlm" | "lash") {
            // Skip past this opener run and keep looking — a non-lash
            // language tag belongs to a different code block.
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
            open_pos: open,
            opener_len,
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

fn assistant_prose_message(content: String) -> Message {
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
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        user_input: None,
        origin: None,
    }
}

fn submit_required_reminder_message(requires_schema: bool, include_submit_prompt: bool) -> Message {
    let id = fresh_message_id();
    let content = if !include_submit_prompt {
        "[runtime] This session must continue from a fenced ```lashlang block until the task-specific completion path is satisfied. Plain text outside a fence is not delivered."
    } else if requires_schema {
        "[runtime] The final answer must be delivered from a fenced ```lashlang block by calling `submit <value>` with a value matching the required output schema. Plain text outside a fence is not delivered."
    } else {
        "[runtime] The final answer must be delivered from a fenced ```lashlang block by calling `submit <value>`. Plain text outside a fence is not delivered."
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
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        user_input: None,
        origin: Some(lash::MessageOrigin::Plugin {
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
                "[runtime] Your `submit` value didn't match the required output schema:\n{error_text}\n\nFix the value and call `submit <corrected>` from another fenced ```lashlang block."
            ),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        user_input: None,
        origin: Some(lash::MessageOrigin::Plugin {
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
    fn fence_detector_accepts_inline_opener_after_prose() {
        // Regression: reasoning models emit the opening fence mid-line:
        // `…required output shape.```lashlang\n…`. Requiring newline
        // before ``` caused the detector to miss the block entirely,
        // which made the RLM turn terminate after one iteration
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
