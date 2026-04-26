//! RLM protocol driver + all its associated helpers:
//!
//! - The [`RlmDriver`] itself — extracts the first fenced `lashlang`
//!   block from the assistant text and dispatches `StartExec`.
//! - The [`rlm_execution_section`] prompt copy.
//! - Fence-extraction utilities (`extract_first_lashlang_fence`,
//!   [`contains_closed_lashlang_fence`]).
//! - Typed-RLM schema validation and the auxiliary messages used when
//!   the model fails to produce a schema-matching `submit`.

use std::sync::Arc;

use lash::sansio::{
    CheckpointResumeAction, CompletedToolCall, ContextProjector, ProtocolDriverHandle,
    WaitingExecState, WaitingLlmState, driver_state,
};
use lash::session_model::{
    ConversationRecord, Message, MessageRole, ModeEvent, Part, PartKind, PruneState, SessionEvent,
    SessionEventRecord, fresh_message_id, make_error_event,
};
use lash::{
    CheckpointKind, DriverAction, DriverContextView, ExecResponse, LlmOutputPart, LlmResponse,
    ModeBuildInput, ModeConfig, ModePreamble, ProjectorContext, PromptContribution, ToolCallRecord,
    append_assistant_text_part, head_tail_truncate, normalized_response_parts,
};
use lash_rlm_types::{RlmModeEvent, RlmTermination, RlmTrajectoryEntry};
use serde_json::Value;

pub const RLM_EXECUTION_SECTION: &str = r#"**All actions go through `lashlang`.** Every tool listed under **Available Tools** is callable as `call tool_name { ... }` from inside a fenced `lashlang` block. Emit a block whenever you need to call a tool, read a file, run a command, search the repo, spawn a subagent, or compute a value. Plain prose is for direct conversational replies that need no action.

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
- `format(template, arg0, arg1, ...)` — positional interpolation: `{}` auto-numbers, `{0}` / `{1}` pick a specific arg, `{{` / `}}` escape literal braces. Do not wrap args in a list: use `format("It is {}.", trim(now.output))`, not `format("It is {}.", [trim(now.output)])`.
- `validate(value, Type { ... })` — check an intermediate value against a Type literal and return it unchanged, or abort with a validation error

### Type literals

`Type { field: shape, ... }` describes a record shape. Field separators are commas (trailing comma OK).

- Scalars: `str`, `int`, `float`, `bool`, `dict`, `any`, `null`.
- Collections: `list[shape]`, `enum["a", "b"]`, nested `Type { ... }`.
- **Optional field** — put `?` after the type: `email: str?` means the field may be absent from the record. If the field IS present, its value must be a string; `null` is **not** allowed.
- **Nullable field** — use a union with `null`: `email: str | null` means the field is required and its value is either a string or null.
- **Unions** — `a | b | c`, e.g. `status: str | int`, `value: str | null`.
- Nested shapes require the `Type` keyword: `nested: Type { ok: bool }` (bare `{ ok: bool }` is rejected — that's a record value, not a type).
"#;

const RLM_DECOMPOSITION_SECTION: &str = r#"### Decomposition

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

pub fn rlm_execution_section() -> String {
    format!(
        "{RLM_EXECUTION_SECTION}\n\n{LASHLANG_LANGUAGE_REFERENCE}\n\n{RLM_DECOMPOSITION_SECTION}"
    )
}

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
            projector: Arc::new(RlmContextProjector),
            sync_execution_surface: true,
        },
        tool_specs: Arc::new(Vec::new()),
        tool_names: input.tool_surface.tool_names(),
        omitted_tool_count,
        execution_prompt: rlm_execution_section(),
        prompt_contributions,
    }
}

pub struct RlmDriver;

struct RlmContextProjector;

impl ContextProjector<lash::HostModeProtocol> for RlmContextProjector {
    fn project(&self, ctx: ProjectorContext<'_>) -> lash::LlmRequest {
        let task_context = build_task_context(ctx.messages.as_slice());

        let rlm_trajectory = rlm_trajectory_from_events(ctx.events);
        let repl_history = format_rlm_trajectory(&rlm_trajectory, 10_000);
        let user_prompt = format!(
            "Task\n{task_context}\n\nREPL history\n{repl_history}\n\nIteration\n{}\n\nFinalization\nCall `submit <value>` from lashlang when the task is complete. Do not answer in prose without a lashlang block.",
            ctx.iteration + 1
        );

        let mut messages = Vec::new();
        if !ctx.config.system_prompt.trim().is_empty() {
            messages.push(lash::llm::types::LlmMessage::text(
                lash::llm::types::LlmRole::System,
                ctx.config.system_prompt.as_str().to_owned(),
            ));
        }
        messages.push(lash::llm::types::LlmMessage::text(
            lash::llm::types::LlmRole::User,
            user_prompt,
        ));

        lash::LlmRequest {
            model: ctx.config.model.clone(),
            messages,
            attachments: Vec::new(),
            tools: Arc::new(Vec::new()),
            tool_choice: lash::llm::types::LlmToolChoice::None,
            model_variant: ctx.config.model_variant.clone(),
            session_id: ctx.config.run_session_id.clone(),
            output_spec: None,
            stream_events: None,
        }
    }
}

const MAX_TASK_CONTEXT_USER_MESSAGES: usize = 4;
const MAX_RLM_TRAJECTORY_STEPS: usize = 12;

fn build_task_context(messages: &[Message]) -> String {
    let mut user_chunks: Vec<&str> = Vec::new();
    for message in messages {
        if !matches!(message.role, MessageRole::User) {
            continue;
        }
        for part in &message.parts {
            if matches!(part.kind, PartKind::Text | PartKind::Prose) {
                let trimmed = part.content.trim();
                if !trimmed.is_empty() {
                    user_chunks.push(trimmed);
                }
            }
        }
    }
    if user_chunks.is_empty() {
        return "No user task context is available.".to_string();
    }

    let total_user_chunks = user_chunks.len();
    let mut selected: Vec<&str> = Vec::new();
    if total_user_chunks <= MAX_TASK_CONTEXT_USER_MESSAGES {
        selected = user_chunks;
    } else {
        selected.push(user_chunks[0]);
        let tail_count = MAX_TASK_CONTEXT_USER_MESSAGES.saturating_sub(1);
        selected.extend(
            user_chunks[user_chunks.len().saturating_sub(tail_count)..]
                .iter()
                .copied(),
        );
    }

    let mut out = String::with_capacity(selected.iter().map(|s| s.len() + 8).sum());
    for (idx, chunk) in selected.iter().enumerate() {
        if idx == 1 && total_user_chunks > MAX_TASK_CONTEXT_USER_MESSAGES {
            let hidden = total_user_chunks.saturating_sub(MAX_TASK_CONTEXT_USER_MESSAGES);
            use std::fmt::Write as _;
            let _ = write!(out, "[{hidden} earlier user messages omitted]\n\n");
        }
        out.push_str("User: ");
        out.push_str(chunk);
        out.push_str("\n\n");
    }
    out
}

fn format_rlm_trajectory(entries: &[RlmTrajectoryEntry], max_output_chars: usize) -> String {
    if entries.is_empty() {
        return "You have not interacted with the lashlang REPL yet.".to_string();
    }
    let start = entries.len().saturating_sub(MAX_RLM_TRAJECTORY_STEPS);
    let visible = &entries[start..];
    let mut out = String::new();
    if start > 0 {
        use std::fmt::Write as _;
        let _ = write!(out, "[{start} earlier REPL steps omitted]\n\n");
    }
    for (idx, entry) in visible.iter().enumerate() {
        let (preview, raw_len) = head_tail_truncate(&entry.output, max_output_chars);
        if idx > 0 {
            out.push_str("\n\n");
        }
        use std::fmt::Write as _;
        let _ = write!(
            out,
            "=== Step {} ===\nReasoning: {}\nCode:\n```lashlang\n{}\n```\nOutput ({raw_len} chars):\n{}",
            start + idx + 1,
            entry.reasoning.trim(),
            entry.code.trim(),
            preview
        );
        if let Some(error) = &entry.error {
            out.push_str("\nError:\n");
            out.push_str(error);
        }
        if let Some(final_output) = &entry.final_output {
            out.push_str("\nFinal output:\n");
            out.push_str(
                &serde_json::to_string_pretty(final_output)
                    .unwrap_or_else(|_| final_output.to_string()),
            );
        }
    }
    out
}

fn rlm_trajectory_from_events(events: &[SessionEventRecord]) -> Vec<RlmTrajectoryEntry> {
    lash_rlm_types::project_trajectory(events.iter().filter_map(|event| match event {
        SessionEventRecord::Mode(event) => event.rlm_event(),
        _ => None,
    }))
}

#[derive(Default)]
struct RlmDriverState {
    reasoning: String,
    tool_calls: Vec<ToolCallRecord>,
    observations: Vec<String>,
    combined_output: String,
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
                LlmOutputPart::Text { text } => {
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
            actions.push(DriverAction::Finish);
            return actions;
        }

        let extraction = extract_first_lashlang_fence(&assistant_text);
        let Some(fence) = extraction else {
            match ctx.termination().rlm_termination() {
                RlmTermination::ProseWithoutFence => {
                    if !assistant_text.trim().is_empty() {
                        actions.push(DriverAction::AppendEvents(vec![conversation_event(
                            assistant_prose_message(assistant_text),
                        )]));
                    }
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish,
                    });
                }
                RlmTermination::Finish { .. } => {
                    let mut events = Vec::new();
                    if !assistant_text.trim().is_empty() {
                        events.push(conversation_event(assistant_prose_message(assistant_text)));
                    }
                    events.push(conversation_event(submit_required_reminder_message()));
                    actions.push(DriverAction::AppendEvents(events));
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
                let baton_successor = response
                    .tool_calls
                    .iter()
                    .find_map(baton_successor_from_tool_result);
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
                let _ = response.images;
                if !response.output.is_empty() {
                    state.combined_output.push_str(&response.output);
                }
                for observation in response.observations {
                    if !observation.is_empty() {
                        state.observations.push(observation.clone());
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
                if let Some(successor_session_id) = baton_successor {
                    actions.push(DriverAction::AppendEvents(vec![trajectory_event(
                        trajectory_entry(ctx.iteration(), &state, None, None),
                    )]));
                    actions.push(DriverAction::Emit(SessionEvent::SessionHandoff {
                        session_id: successor_session_id,
                    }));
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish,
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
            } = ctx.termination().rlm_termination()
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
            if matches!(
                ctx.termination().rlm_termination(),
                RlmTermination::Finish { .. }
            ) {
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

        actions.push(DriverAction::AppendEvents(vec![trajectory_event(
            trajectory_entry(ctx.iteration(), &state, None, None),
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

fn baton_successor_from_tool_result(record: &ToolCallRecord) -> Option<String> {
    if record.tool != "pass_baton" || !record.success {
        return None;
    }
    record
        .result
        .get("_baton")
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
    let output = state.combined_output.clone();
    let output_raw_len = output.chars().count();
    RlmTrajectoryEntry {
        id: format!("rlm_step_{iteration}"),
        iteration,
        reasoning: state.reasoning.clone(),
        code: state.executed_code.clone().unwrap_or_default(),
        output,
        observations: state.observations.clone(),
        tool_calls: state.tool_calls.clone(),
        error: validation_error.or_else(|| state.exec_error.clone()),
        final_output,
        output_raw_len,
    }
}

fn conversation_event(message: Message) -> SessionEventRecord {
    SessionEventRecord::Conversation(ConversationRecord::from_message(message))
}

fn trajectory_event(entry: RlmTrajectoryEntry) -> SessionEventRecord {
    SessionEventRecord::Mode(ModeEvent::rlm(RlmModeEvent::RlmTrajectoryEntry(entry)))
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

fn submit_required_reminder_message() -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::User,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: "[runtime] An output schema is required for the final answer. Wrap your reply in a fenced ```lashlang block and call `submit <value>` with a value matching the schema. Plain text outside a fence is not delivered.".to_string(),
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

fn submit_schema_mismatch_message(error_text: &str) -> Message {
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
