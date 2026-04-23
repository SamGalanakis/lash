//! RLM protocol driver + all its associated helpers:
//!
//! - The [`RlmDriver`] itself — extracts the first fenced `lashlang`
//!   block from the assistant text and dispatches `StartExec`.
//! - The [`RLM_EXECUTION_SECTION`] prompt copy.
//! - Fence-extraction utilities (`extract_first_lashlang_fence`,
//!   [`contains_closed_lashlang_fence`]).
//! - Typed-RLM schema validation and the auxiliary messages used when
//!   the model fails to produce a schema-matching `submit`.

use std::sync::Arc;

use lash::sansio::{
    CheckpointResumeAction, CompletedToolCall, DriverAction, DriverContextView,
    ProtocolDriverHandle, RlmTermination, WaitingExecState, WaitingLlmState, driver_state,
};
use lash::session_model::message::{PartAttachment, data_url_for_bytes};
use lash::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, SessionEvent, fresh_message_id,
    make_error_event,
};
use lash::{
    CheckpointKind, ExecResponse, LlmOutputPart, LlmResponse, ModeBuildInput, ModeConfig,
    ModePreamble, PromptContribution, ToolCallRecord, ToolImage, append_assistant_text_part,
    normalized_response_parts,
};
use serde_json::Value;

pub const RLM_EXECUTION_SECTION: &str = r#"In RLM mode, **all execution goes through `lashlang`**. The API request intentionally sends **no** native tool schema (the `tools` array is empty by design) — **this does not mean you have no tools**. Every tool listed under **Available Tools** below is callable via `call tool_name { ... }` from inside a fenced `lashlang` block. Never tell the user you cannot inspect files, run commands, or use tools; you can. Emit a lashlang block whenever you need to call a tool, read a file, run a command, search the repo, spawn a subagent, or compute a value from prior results. Plain prose is **only** for direct conversational replies that need no action.

### Two terminators — the distinction matters

- `print <expr>` — **inspect a value; keep going.** Output is added to this turn's observations and shows up on the next step so you can refine. Use this to peek at tool results, check a slice of a file, or confirm a value before acting. The program does **not** end.
- `submit <expr>` — **this is my final answer; end the turn.** The program ends and `<expr>` becomes the assistant's reply to the user (strings pass through; other values are pretty-JSON). If a **Required output** schema is present, the value must match it.

Never `submit` a raw tool-result dump to end the turn. If you need to look at something, `print` it and continue on the next step with a prose summary or a schema-shaped submit.

### Turn shape

**Exactly one ` ```lashlang ` fenced block per response.** The runtime aborts the LLM stream the instant your first block closes (` ``` ` on its own line) and runs it immediately. A second fenced block, or trailing prose after the first one, is silently dropped. Only ` ```lashlang ` is recognised — `rlm` and other labels are treated as plain prose. Do not speculate about what the block will return; that's what the next turn is for.

- **Write small blocks.** Each block should do one focused step: call a tool or two, `print` what you need to see, then stop. Do not paste a whole program that reads five files and submits them.
- Keep prose around the block to one or two sentences of reasoning. Do not describe an action in prose instead of executing it; if you say you will read a file, the block must contain the `call read_file`.
- After each result, decide: another fenced block (more work), or `submit <final>` / a prose-only reply (done).
- In interactive chat you may also end by replying with prose and no fenced block — useful for streamed text. Either way the turn ends. Root-session `submit` values are rendered as-is; formatting requirements (Markdown, schema) are covered below.

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

### Language

- Values: null, booleans, numbers, strings, lists, records. Literals: `[a, b]`, `{ a: 1, b: 2 }`.
- Assign with `name = expr`. Variables persist across fenced blocks within the turn.
- Call a tool: `call tool { arg: expr }`. Every tool call returns a wrapper record: `{ ok: true, value: <tool output> }` on success, `{ ok: false, error: "..." }` on failure. For the common happy path, append `?` to unwrap it: `(call tool { arg: expr })?` returns `.value` or aborts this block with the tool error. Keep the raw wrapper only when you intentionally need `.ok`, `.value`, or `.error` for branching/retry/reporting.
- Background start: `start call tool { arg: expr }` returns a **handle** (not wrapped). Resolve it with `await handle` — that returns the same `{ ok, value }` wrapper as a synchronous `call`. Use `(await handle)?` for the common happy path. `await [h1, h2]` returns a list of wrappers in order. Cancel with `cancel handle` (best-effort).
- Independent parallel tool calls: `parallel { ... }`. Prefer named branches (`parallel { a: call ... b: call ... }`) so results come back as a record (`results.a`, `results.b`). Positional branches still return a list in order. Do not use `parallel` when one branch needs another branch's output.
- Control flow: statement `if`/`for`; expression ternary `cond ? yes : no` (there is no expression-form `if`); boolean negation via `!cond` or `not cond`.
- Bare expressions are valid statements. Inside `parallel { ... }`, a bare expression contributes its value to the result list.
- If the prompt includes a **Bound Variables** section, those names are already in scope — use them, don't rebuild them from prose.

### Builtins

Call as functions (e.g. `len(x)`, `slice(s, 0, 200)`). For `slice`, `null` bounds mean start/end; negative bounds count from the end.

- `len(x)` — length of string/list/record (0 for null)
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
- `format(template, arg0, arg1, ...)` — positional interpolation: `{}` auto-numbers, `{0}` / `{1}` pick a specific arg, `{{` / `}}` escape literal braces
- `validate(value, Type { ... })` — check an intermediate value against a Type literal and return it unchanged, or abort with a validation error

### Type literals

`Type { field: shape, ... }` describes a record shape. Field separators are commas (trailing comma OK).

- Scalars: `str`, `int`, `float`, `bool`, `dict`, `any`, `null`.
- Collections: `list[shape]`, `enum["a", "b"]`, nested `Type { ... }`.
- **Optional field** — put `?` after the type: `email: str?` means the field may be absent from the record. If the field IS present, its value must be a string; `null` is **not** allowed.
- **Nullable field** — use a union with `null`: `email: str | null` means the field is required and its value is either a string or null.
- **Unions** — `a | b | c`, e.g. `status: str | int`, `value: str | null`.
- Nested shapes require the `Type` keyword: `nested: Type { ok: bool }` (bare `{ ok: bool }` is rejected — that's a record value, not a type).

### Decomposition

- Break big tasks into small steps. Prefer narrow `print`-then-continue checks over brute-force scans.
- Use `print` to verify a subquestion before acting on it. Use `submit` only when you are ready to deliver the final answer.
- Use `start`/`await` when a long-running tool can progress in the background while you do other work — especially `wait_agent`. Prefer record-shaped fanout (`await { a: h1, b: h2 }`) so resolved results are named.

Example fanout to two subagents (use `?` for fail-fast unwrapping):

```lashlang
a = (call spawn_agent { task_name: "read_chunk_1", task: "Read chunk 1 and extract the key claim", capability: "low", output: { claim: "str" } })?
b = (call spawn_agent { task_name: "read_chunk_2", task: "Read chunk 2 and extract the key claim", capability: "low", output: { claim: "str" } })?
events = await {
  a: start call wait_agent { targets: [a.target], timeout_ms: 30000 },
  b: start call wait_agent { targets: [b.target], timeout_ms: 30000 },
}
submit [events.a?.completion.result, events.b?.completion.result]
```"#;

/// Build the RLM-mode preamble. Called by the plugin factory from
/// `build_preamble`.
pub fn build_rlm_preamble(input: ModeBuildInput) -> ModePreamble {
    let omitted_tool_count = input.tool_surface.omitted_tool_count();
    let mut prompt_contributions = Vec::new();

    let tool_docs = input.tool_surface.prompt_tool_docs();
    if !tool_docs.trim().is_empty() {
        prompt_contributions.push(PromptContribution::execution("Available Tools", tool_docs));
    }
    if omitted_tool_count > 0 {
        prompt_contributions.push(PromptContribution::guidance(
            "Tool Discovery",
            "Use `discover_tools` to inspect additional discoverable tools that are omitted from Available Tools for brevity. Call `discover_tools()` with no query to browse the catalog, or `discover_tools(query: \"git\")` to filter by keyword. If a result is marked loadable but not callable, call `load_tools(names=[...])`; the runtime refreshes the surface for the next step in the same turn.",
        ));
    }
    prompt_contributions.extend(input.extra_prompt_contributions);

    ModePreamble {
        config: ModeConfig {
            protocol: Arc::new(RlmDriver),
            sync_execution_surface: true,
        },
        tool_specs: Arc::new(Vec::new()),
        tool_names: input.tool_surface.tool_names(),
        omitted_tool_count,
        execution_prompt: RLM_EXECUTION_SECTION.to_string(),
        prompt_contributions,
    }
}

pub struct RlmDriver;

#[derive(Default)]
struct RlmDriverState {
    tool_calls: Vec<ToolCallRecord>,
    images: Vec<ToolImage>,
    combined_output: String,
    exec_error: Option<String>,
    executed_code: Option<String>,
    terminal_finish: Option<Value>,
}

struct FenceExtraction {
    code: String,
    had_extra_fences: bool,
}

impl ProtocolDriverHandle for RlmDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.build_llm_request(false),
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
        for part in normalized_response_parts(&llm_response) {
            match part {
                LlmOutputPart::Text { text } => {
                    append_assistant_text_part(&mut assistant_text, &text);
                }
                // RLM mode never re-feeds reasoning items to the model and
                // doesn't surface them to the user — drop them silently.
                LlmOutputPart::Reasoning { .. } => {}
                LlmOutputPart::ToolCall { .. } => {}
            }
        }

        if assistant_text.trim().is_empty() {
            actions.push(DriverAction::Emit(make_error_event(
                "llm_provider",
                Some("empty_response"),
                "Model returned no assistant text.",
                None,
            )));
            actions.push(DriverAction::Finish);
            return actions;
        }

        let extraction = extract_first_lashlang_fence(&assistant_text);
        let Some(fence) = extraction else {
            match ctx.rlm_termination() {
                RlmTermination::ProseWithoutFence => {
                    actions.push(DriverAction::AppendMessages(vec![assistant_prose_message(
                        assistant_text,
                    )]));
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish,
                    });
                }
                RlmTermination::Finish { .. } => {
                    actions.push(DriverAction::AppendMessages(vec![
                        assistant_prose_message(assistant_text),
                        typed_rlm_finish_reminder_message(),
                    ]));
                    actions.push(DriverAction::AdvanceIteration);
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::AfterWork,
                        on_empty: CheckpointResumeAction::PrepareIteration,
                    });
                }
            }
            return actions;
        };

        let _ = fence.had_extra_fences;

        let mut state = waiting
            .take_driver_state::<RlmDriverState>()
            .unwrap_or_default();
        state.executed_code = Some(fence.code.clone());

        actions.push(DriverAction::AppendMessages(vec![assistant_prose_message(
            assistant_text,
        )]));
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
                for tool_call in &response.tool_calls {
                    actions.push(DriverAction::Emit(SessionEvent::ToolCall {
                        call_id: None,
                        name: tool_call.tool.clone(),
                        args: tool_call.args.clone(),
                        result: tool_call.result.clone(),
                        success: tool_call.success,
                        duration_ms: tool_call.duration_ms,
                    }));
                }
                state.tool_calls.extend(response.tool_calls);
                state.images.extend(response.images);
                if !response.output.is_empty() {
                    state.combined_output.push_str(&response.output);
                }
                for observation in response.observations {
                    if !observation.is_empty() {
                        if !state.combined_output.is_empty()
                            && !state.combined_output.ends_with('\n')
                        {
                            state.combined_output.push('\n');
                        }
                        state.combined_output.push_str(&observation);
                        if !state.combined_output.ends_with('\n') {
                            state.combined_output.push('\n');
                        }
                    }
                }
                if let Some(raw_error) = response.error {
                    state.exec_error = Some(raw_error);
                }
                if let Some(finish_value) = response.terminal_finish {
                    state.terminal_finish = Some(finish_value);
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
            } = ctx.rlm_termination()
                && let Err(error_text) = validate_finish_value(finish_value, schema)
            {
                actions.push(DriverAction::AppendMessages(vec![
                    typed_rlm_schema_error_message(&error_text),
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
            if !rendered.trim().is_empty() {
                actions.push(DriverAction::AppendMessages(vec![assistant_prose_message(
                    rendered,
                )]));
            }
            if matches!(ctx.rlm_termination(), RlmTermination::Finish { .. }) {
                actions.push(DriverAction::Emit(SessionEvent::TypedFinish {
                    value: finish_value.clone(),
                }));
            }
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish,
            });
            return actions;
        }

        let mut result_payload = serde_json::json!({
            "observations": state.combined_output,
            "tool_calls": state.tool_calls,
            "error": state.exec_error,
        });
        if !state.images.is_empty() {
            let images = state
                .images
                .iter()
                .map(|img| {
                    serde_json::json!({
                        "label": img.label,
                        "mime": img.mime,
                    })
                })
                .collect::<Vec<_>>();
            result_payload["images"] = Value::Array(images);
        }

        let success = result_payload
            .get("error")
            .is_none_or(|value| value.is_null());
        let result_call_id = format!("rlm_exec_{}", ctx.iteration());
        let execute_args = state
            .executed_code
            .as_ref()
            .map(|code| serde_json::json!({ "code": code }))
            .unwrap_or_else(|| serde_json::json!({}));
        actions.push(DriverAction::Emit(SessionEvent::ToolCall {
            call_id: Some(result_call_id.clone()),
            name: "execute_lashlang".to_string(),
            args: execute_args,
            result: result_payload.clone(),
            success,
            duration_ms: 0,
        }));
        actions.push(DriverAction::AppendMessages(vec![rlm_result_message(
            &result_call_id,
            success,
            &result_payload,
            &state.images,
        )]));
        actions.push(DriverAction::AdvanceIteration);
        if ctx.should_force_exit_after_grace_turn() {
            actions.push(DriverAction::Finish);
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

// ─────────────────────────────────────────────────────────────────────
//  Shared helpers: fence extraction, typed-RLM messages, schema check
// ─────────────────────────────────────────────────────────────────────

/// Return `true` when `text` contains a complete ` ```lashlang ` (or
/// `rlm`/`lash`) fenced block — opener followed by a closing ` ``` `
/// anywhere (no newline requirement on either side). Exposed for the
/// stream mask, which raises `abort_stream` as soon as this is true.
pub fn contains_closed_lashlang_fence(text: &str) -> bool {
    let Some((_, _body_start, body_end)) = first_lashlang_fence_span(text) else {
        return false;
    };
    body_end + 3 <= text.len() && &text[body_end..body_end + 3] == "```"
}

fn first_lashlang_fence_span(text: &str) -> Option<(usize, usize, usize)> {
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("```") {
        let open = search_from + rel;
        // The opening `\`\`\`` doesn't have to be on its own line —
        // reasoning models commonly emit `…required shape.\`\`\`lashlang\n`
        // inline after prose. The language tag is distinctive enough
        // (`lashlang`/`rlm`/`lash`) that collisions with inline prose
        // are essentially impossible. The *closer* still has to live
        // on its own line (newline-preceded) — that's the real
        // structural signal, enforced in the inner loop below.
        let after_open = open + 3;
        let rest = &text[after_open..];
        let lang_end = rest.find('\n').unwrap_or(rest.len());
        let lang = rest[..lang_end].trim();
        if !matches!(lang, "lashlang" | "rlm" | "lash") {
            search_from = after_open;
            continue;
        }
        let body_start = after_open + lang_end + 1;
        if body_start > text.len() {
            return None;
        }

        // The first `\`\`\`` after the opener closes the block, no
        // matter where it sits on the line. Lashlang strings use `"`,
        // so the risk of a backtick-triple inside otherwise-valid
        // code is essentially zero, and this matches the mental model
        // the prompt describes: `\`\`\`` starts, `\`\`\`` stops.
        let close = text[body_start..]
            .find("```")
            .map(|rel| body_start + rel)
            .unwrap_or(text.len());
        return Some((open, body_start, close));
    }
    None
}

fn extract_first_lashlang_fence(text: &str) -> Option<FenceExtraction> {
    let (_open, body_start, body_end) = first_lashlang_fence_span(text)?;
    let code = text[body_start..body_end]
        .trim_end_matches('\n')
        .to_string();
    let after_close = (body_end + 3).min(text.len());
    let had_extra_fences = first_lashlang_fence_span(&text[after_close..]).is_some();
    Some(FenceExtraction {
        code,
        had_extra_fences,
    })
}

fn assistant_prose_message(content: String) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::Assistant,
        parts: vec![Part {
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
        }],
        user_input: None,
        origin: None,
    }
}

fn typed_rlm_finish_reminder_message() -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::User,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: "[runtime] You're in a typed RLM session. End by emitting a fenced ```lashlang block that calls `submit <expr>` with a value matching the required output schema. Prose-only replies are not accepted as the final answer here.".to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
        }],
        user_input: None,
        origin: None,
    }
}

fn typed_rlm_schema_error_message(error_text: &str) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::User,
        parts: vec![Part {
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
        }],
        user_input: None,
        origin: None,
    }
}

fn rlm_result_message(
    result_call_id: &str,
    success: bool,
    result_payload: &Value,
    images: &[ToolImage],
) -> Message {
    // Emit the exec output as a plain `PartKind::Text` under a
    // user-role message instead of a synthetic `PartKind::ToolResult`
    // tied to a `tool_call_id` the model never produced. Providers
    // that strictly enforce tool_call ↔ tool_result pairing (e.g.
    // Azure's OpenAI endpoint) reject the orphan `role=tool` message
    // the adapter would otherwise emit. This matches dspy.RLM's
    // semantic: the exec output is an observation the runtime feeds
    // back as user content, not a response to a tool the model
    // called.
    //
    // We keep `tool_call_id` + `tool_name` on the Text part so the
    // interrupted-session projection (see `project_interrupted_blocks`)
    // can still identify these messages as RLM exec results and
    // render them as activities rather than raw user input. The
    // Part→LlmContentBlock mapping ignores both fields when
    // `kind == Text`, so the wire format is `role=user, content=…`
    // with no tool-call fakery.
    let id = fresh_message_id();
    let mut parts = vec![Part {
        id: format!("{id}.p0"),
        kind: PartKind::Text,
        content: format_repl_result_text(success, result_payload),
        attachment: None,
        tool_call_id: Some(result_call_id.to_string()),
        tool_name: Some("execute_lashlang".to_string()),
        tool_item_id: None,
        tool_signature: None,
        prune_state: PruneState::Intact,
        reasoning_meta: None,
    }];
    for (image_offset, image) in images.iter().enumerate() {
        parts.push(Part {
            id: format!("{id}.p{}", parts.len()),
            kind: PartKind::Text,
            content: format!("[Tool image: {}]", image.label),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
        });
        parts.push(Part {
            id: format!("{id}.p{}", parts.len()),
            kind: PartKind::Image,
            content: String::new(),
            attachment: Some(PartAttachment {
                mime: image.mime.clone(),
                url: data_url_for_bytes(&image.mime, &image.data),
                filename: Some(format!("tool-image-{image_offset}")),
            }),
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
        });
    }
    Message {
        id,
        role: MessageRole::User,
        parts,
        user_input: None,
        origin: None,
    }
}

fn format_repl_result_text(success: bool, result_payload: &Value) -> String {
    let mut sections = vec!["[Lashlang execution result]".to_string()];
    if !success {
        sections.push("status: error".to_string());
    }
    if let Some(observations) = result_payload
        .get("observations")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        sections.push(format!("observations:\n{observations}"));
    }
    if let Some(arr) = result_payload
        .get("tool_calls")
        .and_then(Value::as_array)
        .filter(|arr| !arr.is_empty())
    {
        sections.push(format!(
            "tool_calls: {}",
            serde_json::to_string(arr).unwrap_or_else(|_| "[]".into())
        ));
    }
    if let Some(error) = result_payload
        .get("error")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        sections.push(format!("error:\n{error}"));
    }
    if let Some(images) = result_payload
        .get("images")
        .and_then(Value::as_array)
        .filter(|arr| !arr.is_empty())
    {
        sections.push(format!(
            "images: {}",
            serde_json::to_string(images).unwrap_or_else(|_| "[]".into())
        ));
    }
    sections.join("\n\n")
}

fn validate_finish_value(value: &Value, schema: &Value) -> Result<(), String> {
    fn matches_type(value: &Value, expected: &str) -> bool {
        match expected {
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value
                .as_f64()
                .is_some_and(|n| n.is_finite() && n.fract() == 0.0),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            "null" => value.is_null(),
            _ => true,
        }
    }

    fn type_name(value: &Value) -> &'static str {
        match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    fn check(value: &Value, schema: &Value, path: &str) -> Result<(), String> {
        let Some(schema_obj) = schema.as_object() else {
            return Ok(());
        };

        if let Some(ty) = schema_obj.get("type").and_then(Value::as_str)
            && !matches_type(value, ty)
        {
            return Err(format!("{path}: expected {ty}, got {}", type_name(value)));
        }

        if let Some(properties) = schema_obj.get("properties").and_then(Value::as_object)
            && let Some(obj) = value.as_object()
        {
            if let Some(required) = schema_obj.get("required").and_then(Value::as_array) {
                for required_field in required {
                    if let Some(name) = required_field.as_str()
                        && !obj.contains_key(name)
                    {
                        return Err(format!("{path}: missing required field `{name}`"));
                    }
                }
            }
            for (name, sub_schema) in properties {
                if let Some(sub_value) = obj.get(name) {
                    let sub_path = if path.is_empty() {
                        name.clone()
                    } else {
                        format!("{path}.{name}")
                    };
                    check(sub_value, sub_schema, &sub_path)?;
                }
            }
        }

        if let Some(items_schema) = schema_obj.get("items")
            && let Some(arr) = value.as_array()
        {
            for (idx, item) in arr.iter().enumerate() {
                let sub_path = format!("{path}[{idx}]");
                check(item, items_schema, &sub_path)?;
            }
        }

        Ok(())
    }

    check(value, schema, "")
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
}
