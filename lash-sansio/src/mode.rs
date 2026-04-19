use std::sync::Arc;

use crate::llm::types::{LlmOutputPart, LlmResponse, LlmToolSpec};
use crate::sansio::{
    CheckpointResumeAction, CompletedToolCall, DriverAction, DriverContextView, PendingToolCall,
    ProtocolDriverHandle, RlmTermination, WaitingExecState, WaitingLlmState, driver_state,
};
use crate::session_model::message::{PartAttachment, ReasoningMeta, data_url_for_bytes};
use crate::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, SessionEvent, format_tool_result_content,
    fresh_message_id, make_error_event, reassign_part_ids,
};
use crate::{
    CheckpointKind, ExecutionMode, PromptContribution, ToolCallRecord, ToolImage, ToolSurface,
};
use serde_json::Value;

const STANDARD_EXECUTION_SECTION: &str = r#"Use direct tool calls when execution is needed.

- Use `batch` for two or more independent tool calls. Serialize calls when later arguments depend on earlier results.
- After applying a change, verify the end-state. Do not re-verify before acting.
- For direct conversational requests that need no tools, respond in prose only."#;

const RLM_EXECUTION_SECTION: &str = r#"In this mode you write `lashlang` code inside your prose response and the runtime executes it. There is no native tool-call envelope — you embed code directly.

Format every work step like this:

````
Brief reasoning here in plain prose (one or two sentences is fine).

```lashlang
result = call tool_name { arg: value }
observe result
```
````

- Wrap each work step in **exactly one** ` ```lashlang ` fenced block. Only the first block runs per turn — additional blocks are ignored.
- Plain prose alongside the block becomes your reasoning trace; keep it short.
- After each execution result, decide whether to write another fenced block (more work to do) or finish the turn in pure prose with no fenced block (task complete).
- When the task is complete, reply with prose only — no fenced lashlang block — and that ends the turn.
- Work iteratively: inspect, act, observe, continue. Most tasks take multiple lashlang steps, not one large step.
- Verify the concrete end state before finalizing in prose when possible.
- Do not describe what you would do instead of doing it.

### RLM Language

`lashlang` is a small workflow language for tool orchestration.

- Values are null, booleans, numbers, strings, lists, and records.
- List and record literals use comma-separated entries: `[a, b]`, `{ a: 1, b: 2 }`. Tool-argument records follow the same rule.
- Assign with `name = expr`. Variables persist across iterations — anything you bind in one fenced block is still in scope on the next.
- If the prompt includes a **Bound Variables** section, those names are already in scope. Access them directly in lashlang instead of rebuilding them from prose.
- Bare expressions are valid statements. In `parallel { ... }`, a bare-expression branch contributes that value to the result list.
- Call tools with `call tool_name { arg: expr }`.
- Start any tool call in the background with `start call tool_name { arg: expr }`. This returns a handle value.
- Resolve a background handle with `await handle`. If you already have a list of handles, `await handles` returns a list of results in order.
- Stop a background handle with `cancel handle`. Cancellation is best-effort: Lash always stops waiting locally, and cooperative tools are also asked to stop their underlying work.
- Use `parallel { ... }` only for independent tool calls. If one call needs another call's output, do not put them in the same `parallel { ... }`.
- `parallel { ... }` returns a list of branch results in order.
- Use ternary expressions for inline branching: `cond ? yes : no`. There is no expression-form `if`.
- Control flow is limited to statement `if` and `for`.
- Boolean negation supports both `!cond` and `not cond`.
- Use `observe expr` to inspect a value and continue execution.
- `observe` output and tool results are fed back into the next iteration (your context), so inspect first and refine on the next step if needed.
- You must explicitly use `observe` to inspect values and make progress based on them. Do not rely on implicit inspection through tool results or execution errors.

### Builtins

Call these as functions — e.g. `len(x)`, `slice(s, 0, 200)`. For `slice`, `null` bounds mean "from start" / "to end"; negative bounds count from the end.

- `len(x)` — length of a string/list/record (0 for null)
- `empty(x)` — true if length is 0
- `slice(s, start, end)` — substring or sublist
- `split(s, sep)` / `join(list, sep)` — string split/join
- `trim(s)` — strip whitespace
- `starts_with(s, prefix)` / `ends_with(s, suffix)` / `contains(haystack, needle)`
- `keys(record)` / `values(record)`
- `to_string(x)` / `to_int(x)` / `to_float(x)`
- `json_parse(s)` — parse a JSON string into a value
- `format(template, record)` — string interpolation

### Decomposition

- Break large tasks into smaller, self-contained steps.
- Prefer narrow checks over brute-force scanning when the input is large.
- Use focused intermediate observations to verify subquestions before finalizing.
- Keep each step concrete and bounded instead of attempting the whole task at once.
- Use `start`/`await` when a long-running tool can make progress in the background while you do other work. This is especially useful for `wait_agent`.

Example fanout pattern:

```lashlang
h1 = call spawn_agent { task_name: "read_chunk_1", task: "Read chunk 1 and extract the key claim", capability: "low", output: { claim: "str" } }
h2 = call spawn_agent { task_name: "read_chunk_2", task: "Read chunk 2 and extract the key claim", capability: "low", output: { claim: "str" } }
handles = [
  start call wait_agent { targets: [h1.path], timeout_ms: 30000 },
  start call wait_agent { targets: [h2.path], timeout_ms: 30000 }
]
events = await handles
results = [events[0].events[0].result, events[1].events[0].result]
finish results
```"#;

#[derive(Clone)]
pub struct ModeConfig {
    pub protocol: Arc<dyn ProtocolDriverHandle>,
    pub sync_execution_surface: bool,
}

#[derive(Clone)]
pub struct ModePreamble {
    pub config: ModeConfig,
    pub tool_specs: Arc<Vec<LlmToolSpec>>,
    pub tool_names: Vec<String>,
    pub omitted_tool_count: usize,
    pub execution_prompt: String,
    pub prompt_contributions: Vec<PromptContribution>,
}

#[derive(Clone, Debug)]
pub struct ModeBuildInput {
    pub mode: ExecutionMode,
    pub tool_surface: ToolSurface,
    pub extra_prompt_contributions: Vec<PromptContribution>,
}

pub fn build_mode_preamble(input: ModeBuildInput) -> ModePreamble {
    match input.mode {
        ExecutionMode::Standard => build_standard_mode_preamble(input),
        ExecutionMode::Rlm => build_rlm_mode_preamble(input),
    }
}

fn build_standard_mode_preamble(input: ModeBuildInput) -> ModePreamble {
    ModePreamble {
        config: ModeConfig {
            protocol: Arc::new(StandardDriver),
            sync_execution_surface: false,
        },
        tool_specs: Arc::new(input.tool_surface.model_tool_specs()),
        tool_names: input.tool_surface.tool_names(),
        omitted_tool_count: 0,
        execution_prompt: STANDARD_EXECUTION_SECTION.to_string(),
        prompt_contributions: input.extra_prompt_contributions,
    }
}

fn build_rlm_mode_preamble(input: ModeBuildInput) -> ModePreamble {
    let omitted_tool_count = input.tool_surface.omitted_tool_count();
    let mut prompt_contributions = Vec::new();

    let tool_docs = input.tool_surface.prompt_tool_docs();
    if !tool_docs.trim().is_empty() {
        prompt_contributions.push(PromptContribution::execution("Available Tools", tool_docs));
    }
    if omitted_tool_count > 0 {
        prompt_contributions.push(PromptContribution::guidance(
            "Tool Discovery",
            "Use `search_tools` to inspect the additional available tools that are omitted from Available Tools for brevity. With no query, it browses the full active tool catalog; use focused queries when you know the kind of tool you need.",
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

pub struct StandardDriver;
pub struct RlmDriver;

#[derive(Default)]
struct RlmDriverState {
    tool_calls: Vec<ToolCallRecord>,
    images: Vec<ToolImage>,
    combined_output: String,
    exec_error: Option<String>,
    executed_code: Option<String>,
    terminal_finish: Option<serde_json::Value>,
}

struct FenceExtraction {
    code: String,
    had_extra_fences: bool,
}

impl ProtocolDriverHandle for StandardDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.build_llm_request(true),
            driver_state: None,
        }]
    }

    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_>,
        _waiting: WaitingLlmState,
        llm_response: LlmResponse,
        text_streamed: bool,
    ) -> Vec<DriverAction> {
        let response_parts = normalized_response_parts(&llm_response);
        let mut assistant_text = String::new();
        let mut tool_calls: Vec<(String, String, String, Option<String>)> = Vec::new();
        // Reasoning items captured with their position in the original
        // response. The `usize` is the index in `tool_calls` that this
        // reasoning item originally preceded, so we can interleave
        // reasoning → tool_call in the emission order Codex produced.
        // `Option<ReasoningMeta>` carries the encrypted roundtrip payload
        // when present (fix 1.3b); when None, the item is display-only
        // (fix 1.3a) — still rendered in the UI but never re-fed.
        let mut reasoning_items: Vec<(usize, Option<ReasoningMeta>, String)> = Vec::new();
        let mut actions = Vec::new();

        for part in response_parts {
            match part {
                LlmOutputPart::Text { text } => {
                    if !text.is_empty() {
                        let previous_len = assistant_text.len();
                        append_assistant_text_part(&mut assistant_text, &text);
                        if !text_streamed {
                            actions.push(DriverAction::Emit(SessionEvent::TextDelta {
                                content: assistant_text[previous_len..].to_string(),
                            }));
                        }
                    }
                }
                LlmOutputPart::Reasoning {
                    text,
                    id,
                    summary,
                    encrypted_content,
                } => {
                    let trimmed = text.trim().to_string();
                    // Skip fully-empty reasoning items (no display text and
                    // no roundtrip payload).
                    if trimmed.is_empty() && encrypted_content.is_none() {
                        continue;
                    }
                    let meta = if encrypted_content.is_some() {
                        Some(ReasoningMeta {
                            id,
                            summary,
                            encrypted_content,
                        })
                    } else {
                        None
                    };
                    reasoning_items.push((tool_calls.len(), meta, trimmed));
                }
                LlmOutputPart::ToolCall {
                    call_id,
                    tool_name,
                    input_json,
                    id,
                } => {
                    tool_calls.push((call_id, tool_name, input_json, id));
                }
            }
        }

        actions.push(DriverAction::Emit(SessionEvent::LlmResponse {
            iteration: ctx.iteration(),
            content: assistant_text.clone(),
            duration_ms: 0,
        }));

        if tool_calls.is_empty() {
            if assistant_text.trim().is_empty() && reasoning_items.is_empty() {
                actions.push(DriverAction::Emit(make_error_event(
                    "llm_provider",
                    Some("empty_response"),
                    "Model returned no assistant text or tool calls.",
                    None,
                )));
                actions.push(DriverAction::Finish);
                return actions;
            }

            // Build assistant message: reasoning parts FIRST (so the UI
            // can draw the "thinking" block above the prose), then prose.
            let asst_id = fresh_message_id();
            let mut parts_out = Vec::new();
            for (_, meta, text) in reasoning_items {
                parts_out.push(reasoning_part(&asst_id, parts_out.len(), text, meta));
            }
            if !assistant_text.trim().is_empty() {
                parts_out.push(Part {
                    id: format!("{}.p{}", asst_id, parts_out.len()),
                    kind: PartKind::Prose,
                    content: assistant_text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                });
            }
            if parts_out.is_empty() {
                actions.push(DriverAction::Emit(make_error_event(
                    "llm_provider",
                    Some("empty_response"),
                    "Model returned no assistant text or tool calls.",
                    None,
                )));
                actions.push(DriverAction::Finish);
                return actions;
            }
            actions.push(DriverAction::AppendMessages(vec![Message {
                id: asst_id,
                role: MessageRole::Assistant,
                parts: parts_out,
                user_input: None,
                origin: None,
            }]));
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish,
            });
            return actions;
        }

        let asst_id = fresh_message_id();
        let mut assistant_parts = Vec::new();
        // Reasoning parts come FIRST so the renderer can draw the
        if !assistant_text.trim().is_empty() {
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::Prose,
                content: assistant_text,
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            });
        }

        let mut calls = Vec::new();
        // Interleave reasoning items with tool calls to preserve the
        // original emission order. Codex re-feeds expect the sequence
        // `reasoning → function_call` from the turn in which both were
        // produced — swapping them drops the chain-of-thought pairing.
        let mut reasoning_iter = reasoning_items.into_iter().peekable();
        for (tool_index, (call_id, tool_name, input_json, item_id)) in
            tool_calls.into_iter().enumerate()
        {
            while let Some((insert_index, _, _)) = reasoning_iter.peek() {
                if *insert_index > tool_index {
                    break;
                }
                let (_, meta, text) = reasoning_iter.next().expect("peek ok");
                assistant_parts.push(reasoning_part(
                    &asst_id,
                    assistant_parts.len(),
                    text,
                    meta,
                ));
            }
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::ToolCall,
                content: input_json.clone(),
                attachment: None,
                tool_call_id: Some(call_id.clone()),
                tool_name: Some(tool_name.clone()),
                tool_item_id: item_id.clone(),
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            });

            let args = serde_json::from_str::<Value>(&input_json)
                .unwrap_or_else(|_| serde_json::json!({}));
            calls.push(PendingToolCall {
                call_id,
                tool_name,
                args,
                item_id,
            });
        }
        // Drain any reasoning that lives after the last tool call.
        for (_, meta, text) in reasoning_iter {
            assistant_parts.push(reasoning_part(
                &asst_id,
                assistant_parts.len(),
                text,
                meta,
            ));
        }

        if !assistant_parts.is_empty() {
            actions.push(DriverAction::AppendMessages(vec![Message {
                id: asst_id,
                role: MessageRole::Assistant,
                parts: assistant_parts,
                user_input: None,
                origin: None,
            }]));
        }

        actions.push(DriverAction::StartTools { calls });
        actions
    }

    fn handle_tool_results(
        &self,
        ctx: DriverContextView<'_>,
        completed: Vec<CompletedToolCall>,
    ) -> Vec<DriverAction> {
        let mut actions = Vec::new();
        let mut result_parts = Vec::new();

        for outcome in completed {
            result_parts.push(Part {
                id: String::new(),
                kind: PartKind::ToolResult,
                content: format_tool_result_content(
                    outcome.model_result.success,
                    &outcome.model_result.result,
                ),
                attachment: None,
                tool_call_id: Some(outcome.call_id.clone()),
                tool_name: Some(outcome.tool_name.clone()),
                tool_item_id: None,
                prune_state: PruneState::Intact,
            reasoning_meta: None,
            });

            for (image_offset, image) in outcome.model_result.images.into_iter().enumerate() {
                result_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[Tool image: {}]", image.label),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
            reasoning_meta: None,
                });
                result_parts.push(Part {
                    id: String::new(),
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
                    prune_state: PruneState::Intact,
            reasoning_meta: None,
                });
            }
        }

        if !result_parts.is_empty() {
            let user_id = fresh_message_id();
            reassign_part_ids(&user_id, &mut result_parts);
            actions.push(DriverAction::AppendMessages(vec![Message {
                id: user_id,
                role: MessageRole::User,
                parts: result_parts,
                user_input: None,
                origin: None,
            }]));
        }

        actions.push(DriverAction::AdvanceIteration);
        let next_iteration = ctx.iteration() + 1;
        if let Some(max_turns) = ctx.max_turns()
            && next_iteration >= ctx.run_offset() + max_turns
        {
            actions.push(DriverAction::AppendMessages(vec![
                turn_limit_exhausted_message(max_turns),
            ]));
            actions.push(DriverAction::Finish);
            return actions;
        }

        actions.push(DriverAction::StartCheckpoint {
            checkpoint: CheckpointKind::AfterWork,
            on_empty: CheckpointResumeAction::PrepareIteration,
        });
        actions
    }

    fn handle_exec_result(
        &self,
        _ctx: DriverContextView<'_>,
        _waiting: WaitingExecState,
        _result: Result<crate::ExecResponse, String>,
    ) -> Vec<DriverAction> {
        Vec::new()
    }
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
        let mut reasoning_paragraphs: Vec<String> = Vec::new();
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
        result: Result<crate::ExecResponse, String>,
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

        if let Some(finish_value) = &state.terminal_finish
            && let RlmTermination::Finish { schema } = ctx.rlm_termination()
        {
            if let Some(schema) = schema
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
                serde_json::Value::String(text) => text.clone(),
                other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
            };
            actions.push(DriverAction::AppendMessages(vec![assistant_prose_message(
                rendered,
            )]));
            actions.push(DriverAction::Emit(SessionEvent::TypedFinish {
                value: finish_value.clone(),
            }));
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
            result_payload["images"] = serde_json::Value::Array(images);
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
            call_id: Some(result_call_id),
            name: "execute_lashlang".to_string(),
            args: execute_args,
            result: result_payload.clone(),
            success,
            duration_ms: 0,
        }));
        actions.push(DriverAction::AppendMessages(vec![rlm_result_message(
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

fn normalized_response_parts(llm_response: &LlmResponse) -> Vec<LlmOutputPart> {
    if llm_response.parts.is_empty() && !llm_response.full_text.is_empty() {
        vec![LlmOutputPart::Text {
            text: llm_response.full_text.clone(),
        }]
    } else {
        llm_response.parts.clone()
    }
}

/// Build a Reasoning `Part` from a reasoning item. `meta` is Some when the
/// item carries encrypted content for Codex re-feeding (fix 1.3b); None for
/// display-only summaries (fix 1.3a).
fn reasoning_part(
    asst_id: &str,
    index: usize,
    text: String,
    meta: Option<ReasoningMeta>,
) -> Part {
    Part {
        id: format!("{asst_id}.p{index}"),
        kind: PartKind::Reasoning,
        content: text,
        attachment: None,
        tool_call_id: None,
        tool_name: None,
        tool_item_id: None,
        prune_state: PruneState::Intact,
        reasoning_meta: meta,
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
            content: "[runtime] You're in a typed RLM session. End by emitting a fenced ```lashlang block that calls `finish <expr>` with a value matching the required output schema. Prose-only replies are not accepted as the final answer here.".to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
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
                "[runtime] Your `finish` value didn't match the required output schema:\n{error_text}\n\nFix the value and call `finish <corrected>` from another fenced ```lashlang block."
            ),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
        }],
        user_input: None,
        origin: None,
    }
}

fn turn_limit_exhausted_message(max_turns: usize) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::System,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Error,
            content: format!("Turn limit reached ({max_turns}) before a final assistant response."),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
        }],
        user_input: None,
        origin: None,
    }
}

fn rlm_result_message(success: bool, result_payload: &Value, images: &[ToolImage]) -> Message {
    let id = fresh_message_id();
    let mut parts = vec![Part {
        id: format!("{id}.p0"),
        kind: PartKind::Text,
        content: format_repl_result_text(success, result_payload),
        attachment: None,
        tool_call_id: None,
        tool_name: None,
        tool_item_id: None,
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

fn extract_first_lashlang_fence(text: &str) -> Option<FenceExtraction> {
    fn find_fence(text: &str) -> Option<(usize, usize, usize)> {
        let mut search_from = 0usize;
        while let Some(rel) = text[search_from..].find("```") {
            let open = search_from + rel;
            let preceded_by_newline =
                open == 0 || text.as_bytes().get(open - 1).copied() == Some(b'\n');
            if !preceded_by_newline {
                search_from = open + 3;
                continue;
            }
            let after_open = open + 3;
            let rest = &text[after_open..];
            let lang_end = rest.find('\n').unwrap_or(rest.len());
            let lang = rest[..lang_end].trim();
            // `lashlang` is the canonical tag; `rlm` is the legacy alias and
            // `lash` is accepted as a common abbreviation models frequently
            // emit. Anything else is prose inside another code fence.
            if !matches!(lang, "lashlang" | "rlm" | "lash") {
                search_from = after_open;
                continue;
            }
            let body_start = after_open + lang_end + 1;
            if body_start > text.len() {
                return None;
            }

            let mut cursor = body_start;
            loop {
                let Some(rel) = text[cursor..].find("```") else {
                    return Some((open, body_start, text.len()));
                };
                let close = cursor + rel;
                let preceded_by_newline =
                    close == 0 || text.as_bytes().get(close - 1).copied() == Some(b'\n');
                if preceded_by_newline {
                    return Some((open, body_start, close));
                }
                cursor = close + 3;
            }
        }
        None
    }

    let (_open, body_start, body_end) = find_fence(text)?;
    let code = text[body_start..body_end]
        .trim_end_matches('\n')
        .to_string();
    let after_close = (body_end + 3).min(text.len());
    let had_extra_fences = find_fence(&text[after_close..]).is_some();

    Some(FenceExtraction {
        code,
        had_extra_fences,
    })
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

fn append_assistant_text_part(out: &mut String, next: &str) {
    if out.is_empty() {
        out.push_str(next);
        return;
    }

    let prev_trailing_newlines = out.chars().rev().take_while(|ch| *ch == '\n').count();
    let next_leading_newlines = next.chars().take_while(|ch| *ch == '\n').count();
    let total_boundary_newlines = prev_trailing_newlines + next_leading_newlines;
    if total_boundary_newlines < 2 {
        out.push_str(&"\n".repeat(2 - total_boundary_newlines));
    }

    out.push_str(next);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolDefinition, ToolExecutionMode, ToolParam};

    fn tool(name: &str, injected: bool) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("Tool {name}"),
            params: vec![ToolParam::typed("query", "str")],
            returns: "str".to_string(),
            examples: Vec::new(),
            enabled: true,
            injected,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
        }
    }

    #[test]
    fn rlm_mode_includes_available_tools_and_tool_discovery_notes() {
        let surface = ToolSurface {
            tools: vec![
                tool("search_tools", true),
                tool("grep", true),
                tool("glob", false),
            ],
            tool_list_notes: vec!["extra note".to_string()],
        };
        let preamble = build_mode_preamble(ModeBuildInput {
            mode: ExecutionMode::Rlm,
            tool_surface: surface,
            extra_prompt_contributions: Vec::new(),
        });

        assert!(preamble.execution_prompt.contains("lashlang"));
        assert_eq!(preamble.omitted_tool_count, 1);
        assert!(
            preamble
                .prompt_contributions
                .iter()
                .any(|contribution| contribution.title.as_deref() == Some("Available Tools"))
        );
        assert!(
            preamble
                .prompt_contributions
                .iter()
                .any(|contribution| contribution.title.as_deref() == Some("Tool Discovery"))
        );
    }
}
