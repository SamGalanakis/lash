//! Standard execution mode: the model drives tools via the native
//! function-calling envelope of its LLM transport.
//!
//! This crate owns:
//!
//! - [`StandardDriver`] — the [`ProtocolDriverHandle`] that dispatches
//!   native tool calls and weaves reasoning parts into the assistant
//!   message timeline.
//! - The [`BuiltinStandardModePluginFactory`] plugin that claims the
//!   protocol-driver slot so the runtime can run Standard-mode
//!   sessions.
//! - The `batch` tool that composes parallel native tool calls (only
//!   exposed in Standard mode).

use std::sync::Arc;

use async_trait::async_trait;
use lash_core::llm::types::{ProviderReasoningReplay, ProviderReplayMeta, ResponseTextMeta};
use lash_core::plugin::{
    ModeNativeToolsPlugin, ModeProtocolDriverPlugin, ModeSessionContext, ModeSessionPlugin,
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash_core::sansio::{
    CheckpointResumeAction, CompletedToolCall, PendingToolCall, ProtocolDriverHandle,
    WaitingExecState, WaitingLlmState,
};
use lash_core::session_model::message::PartAttachment;
use lash_core::session_model::{
    ConversationRecord, Message, MessageRole, Part, PartKind, PruneState, SessionEvent,
    SessionEventRecord, fresh_message_id, make_error_event, reassign_part_ids, shared_parts,
};
use lash_core::tool_dispatch::{
    ParallelToolCallSpec, ToolDispatchContext, dispatch_parallel_tool_calls,
};

mod batch;
use batch::batch_tool_definition;
use lash_core::{
    CheckpointKind, DriverAction, DriverContextView, ExecutionMode, LlmOutputPart, LlmResponse,
    ModeBuildInput, ModeConfig, ModePreamble, ProgressSender, SessionError, ToolContract,
    ToolManifest, ToolResult, TurnFinish, TurnOutcome, TurnStop, append_assistant_text_part,
    normalized_response_parts, reasoning_part,
};
use serde_json::Value;

const STANDARD_EXECUTION_SECTION: &str = r#"Use direct tool calls.

- Use `batch` (up to 25 calls) for two or more independent tool calls. Serialize calls when later arguments depend on earlier results.
- For direct conversational requests that need no tools, respond in prose only.

Example — two independent reads in one `batch` call:

```json
{
  "tool_calls": [
    { "tool": "read_file", "parameters": { "path": "src/main.rs" } },
    { "tool": "grep", "parameters": { "query": "ToolProvider", "path": "crates/lash/src/" } }
  ]
}
```"#;

const BATCH_MAX_TOOL_CALLS: usize = 25;

/// Plugin factory that installs the Standard-mode protocol driver,
/// session plugin, and native tool surface.
#[derive(Default)]
pub struct BuiltinStandardModePluginFactory;

impl BuiltinStandardModePluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl PluginFactory for BuiltinStandardModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_standard"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(StandardModePlugin {
            active: ctx.execution_mode == ExecutionMode::standard(),
        }))
    }
}

struct StandardModePlugin {
    active: bool,
}

impl SessionPlugin for StandardModePlugin {
    fn id(&self) -> &'static str {
        "mode_standard"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if self.active {
            reg.mode().session(Arc::new(StandardModeSession))?;
            reg.mode()
                .protocol_driver(Arc::new(StandardProtocolDriver))?;
            reg.mode().native_tools(Arc::new(StandardModeNativeTools))?;
        }
        Ok(())
    }
}

struct StandardModeSession;

#[async_trait]
impl ModeSessionPlugin for StandardModeSession {
    async fn initialize_session(&self, _ctx: ModeSessionContext<'_>) -> Result<(), SessionError> {
        Ok(())
    }
}

struct StandardProtocolDriver;

impl ModeProtocolDriverPlugin for StandardProtocolDriver {
    fn mode_id(&self) -> &str {
        "standard"
    }

    fn build_preamble(&self, input: ModeBuildInput) -> ModePreamble {
        let tool_names = input.tool_surface.tool_names();
        let tool_names_fingerprint = input.tool_surface.tool_names_fingerprint();
        ModePreamble {
            config: ModeConfig::chat(
                Arc::new(StandardDriver),
                true,
                Arc::new(turn_limit_exhausted_message),
            ),
            tool_specs: input.tool_surface.model_tool_specs(),
            tool_names,
            tool_names_fingerprint,
            omitted_tool_count: 0,
            execution_prompt: Arc::from(STANDARD_EXECUTION_SECTION),
            prompt_contributions: input.extra_prompt_contributions,
        }
    }
}

fn turn_limit_exhausted_message(message_id: String, max_turns: usize) -> Message {
    Message {
        id: message_id.clone(),
        role: MessageRole::System,
        parts: shared_parts(vec![Part {
            id: format!("{message_id}.p0"),
            kind: PartKind::Error,
            content: format!("Turn limit reached ({max_turns}) before a final assistant response."),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}

struct StandardModeNativeTools;

#[async_trait]
impl ModeNativeToolsPlugin for StandardModeNativeTools {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![batch_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        (name == "batch").then(|| Arc::new(batch_tool_definition().contract()))
    }

    async fn execute(
        &self,
        context: &ToolDispatchContext<'_>,
        name: &str,
        args: &Value,
        progress: Option<&ProgressSender>,
    ) -> Option<ToolResult> {
        match name {
            "batch" => Some(execute_batch_tool_call(context, args, progress).await),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct BatchCallSpec {
    index: usize,
    tool: String,
    parameters: Value,
}

async fn execute_batch_tool_call(
    context: &ToolDispatchContext<'_>,
    args: &Value,
    progress: Option<&ProgressSender>,
) -> ToolResult {
    let specs = match parse_batch_specs(args) {
        Ok(specs) => specs,
        Err(err) => return err,
    };

    let mut immediate_outcomes = Vec::new();
    let mut parallel_specs = Vec::new();

    for spec in specs.into_iter().take(BATCH_MAX_TOOL_CALLS) {
        if spec.tool == "batch" {
            immediate_outcomes.push(serde_json::json!({
                "index": spec.index,
                "tool": spec.tool,
                "success": false,
                "duration_ms": 0,
                "error": "Tool 'batch' is not allowed inside batch",
            }));
            continue;
        }
        parallel_specs.push(ParallelToolCallSpec {
            index: spec.index,
            tool_name: spec.tool,
            args: spec.parameters,
        });
    }

    let mut parallel_outcomes =
        dispatch_parallel_tool_calls(Arc::new(context.clone()), parallel_specs, progress).await;
    for outcome in parallel_outcomes.drain(..) {
        let mut record = serde_json::Map::new();
        record.insert("index".to_string(), serde_json::json!(outcome.index));
        record.insert("tool".to_string(), serde_json::json!(outcome.record.tool));
        record.insert(
            "success".to_string(),
            serde_json::json!(outcome.record.output.is_success()),
        );
        record.insert(
            "duration_ms".to_string(),
            serde_json::json!(outcome.record.duration_ms),
        );
        record.insert(
            if outcome.record.output.is_success() {
                "result".to_string()
            } else {
                "error".to_string()
            },
            outcome.record.output.value_for_projection(),
        );
        immediate_outcomes.push(Value::Object(record));
    }

    for overflow_index in BATCH_MAX_TOOL_CALLS
        ..args
            .get("tool_calls")
            .and_then(|value| value.as_array())
            .map(|value| value.len())
            .unwrap_or_default()
    {
        immediate_outcomes.push(serde_json::json!({
            "index": overflow_index,
            "tool": args
                .get("tool_calls")
                .and_then(|value| value.as_array())
                .and_then(|items| items.get(overflow_index))
                .and_then(|item| item.get("tool"))
                .and_then(|value| value.as_str())
                .unwrap_or("unknown"),
            "success": false,
            "duration_ms": 0,
            "error": "Maximum of 25 tool calls allowed in batch",
        }));
    }

    immediate_outcomes.sort_by_key(|outcome| {
        outcome
            .get("index")
            .and_then(|value| value.as_u64())
            .unwrap_or(u64::MAX)
    });
    ToolResult::ok(serde_json::json!({
        "results": immediate_outcomes,
    }))
}

#[allow(clippy::result_large_err)]
fn parse_batch_specs(args: &Value) -> Result<Vec<BatchCallSpec>, ToolResult> {
    let Some(raw_calls) = args.get("tool_calls").and_then(|value| value.as_array()) else {
        return Err(ToolResult::err_fmt(
            "Missing required parameter: tool_calls",
        ));
    };
    if raw_calls.is_empty() {
        return Err(ToolResult::err_fmt(
            "Invalid tool_calls: expected at least one call",
        ));
    }

    let mut specs = Vec::with_capacity(raw_calls.len());
    for (index, item) in raw_calls.iter().enumerate() {
        let Some(object) = item.as_object() else {
            return Err(ToolResult::err_fmt(format_args!(
                "Invalid tool_calls[{index}]: expected object with tool and parameters"
            )));
        };
        let Some(tool) = object
            .get("tool")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|tool| !tool.is_empty())
        else {
            return Err(ToolResult::err_fmt(format_args!(
                "Invalid tool_calls[{index}].tool: expected non-empty string"
            )));
        };
        let parameters = object
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        specs.push(BatchCallSpec {
            index,
            tool: tool.to_string(),
            parameters,
        });
    }

    Ok(specs)
}

// ─────────────────────────────────────────────────────────────────────
// Standard protocol driver
// ─────────────────────────────────────────────────────────────────────

/// Protocol driver for Standard execution mode. Consumes native
/// tool-call envelopes from the LLM, dispatches them via
/// `DriverAction::StartTools`, and splices reasoning parts into the
/// assistant message so provider replay metadata preserves
/// chain-of-thought ordering.
pub struct StandardDriver;

struct StandardToolCall {
    call_id: String,
    tool_name: String,
    input_json: String,
    replay: Option<ProviderReplayMeta>,
}

fn last_message_has_tool_result(ctx: &DriverContextView<'_>) -> bool {
    ctx.messages().last().is_some_and(|message| {
        matches!(message.role, MessageRole::User)
            && message
                .parts
                .iter()
                .any(|part| matches!(part.kind, PartKind::ToolResult))
    })
}

impl ProtocolDriverHandle<lash_core::HostModeProtocol> for StandardDriver {
    fn prepare_mode_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.project_llm_request(true),
            driver_state: None,
        }]
    }

    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_>,
        _waiting: WaitingLlmState<lash_core::HostModeProtocol>,
        llm_response: LlmResponse,
        text_streamed: bool,
    ) -> Vec<DriverAction> {
        let response_parts = normalized_response_parts(&llm_response);
        let mut assistant_text = String::new();
        let mut assistant_text_parts: Vec<(String, Option<ResponseTextMeta>)> = Vec::new();
        let mut tool_calls: Vec<StandardToolCall> = Vec::new();
        // Reasoning items captured with their position in the original
        // response. The `usize` is the index in `tool_calls` that this
        // reasoning item originally preceded, so we can interleave
        // reasoning → tool_call in the provider's original emission order.
        // `Option<ProviderReasoningReplay>` carries roundtrip payload
        // when present (fix 1.3b); when None, the item is display-only
        // (fix 1.3a) — still rendered in the UI but never re-fed.
        let mut reasoning_items: Vec<(usize, Option<ProviderReasoningReplay>, String)> = Vec::new();
        let mut actions = Vec::new();

        for part in response_parts {
            match part {
                LlmOutputPart::Text {
                    text,
                    response_meta,
                } => {
                    if !text.is_empty() {
                        let previous_len = assistant_text.len();
                        append_assistant_text_part(&mut assistant_text, &text);
                        assistant_text_parts
                            .push((assistant_text[previous_len..].to_string(), response_meta));
                        if !text_streamed {
                            actions.push(DriverAction::Emit(SessionEvent::TextDelta {
                                content: assistant_text[previous_len..].to_string(),
                            }));
                        }
                    }
                }
                LlmOutputPart::Reasoning { text, replay } => {
                    let trimmed = text.trim().to_string();
                    // Skip fully-empty reasoning items (no display text and
                    // no roundtrip payload).
                    if trimmed.is_empty() && replay.as_ref().is_none_or(|meta| meta.is_empty()) {
                        continue;
                    }
                    reasoning_items.push((tool_calls.len(), replay, trimmed));
                }
                LlmOutputPart::ToolCall {
                    call_id,
                    tool_name,
                    input_json,
                    replay,
                } => {
                    tool_calls.push(StandardToolCall {
                        call_id,
                        tool_name,
                        input_json,
                        replay,
                    });
                }
            }
        }

        actions.push(DriverAction::Emit(SessionEvent::LlmResponse {
            mode_iteration: ctx.mode_iteration(),
            content: assistant_text.clone(),
            duration_ms: 0,
        }));

        if tool_calls.is_empty() {
            if assistant_text.trim().is_empty() && reasoning_items.is_empty() {
                if last_message_has_tool_result(&ctx) {
                    // A model can intentionally complete a tool-only request
                    // with an empty final answer, e.g. when the user says
                    // "do nothing else" after the tool action.
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish(TurnOutcome::Finished(
                            TurnFinish::AssistantMessage {
                                text: String::new(),
                            },
                        )),
                    });
                    return actions;
                }
                actions.push(DriverAction::Emit(make_error_event(
                    "llm_provider",
                    Some("empty_response"),
                    "Model returned no assistant text or tool calls.",
                    None,
                )));
                actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                    TurnStop::ProviderError,
                )));
                return actions;
            }

            let asst_id = fresh_message_id();
            let mut parts_out = Vec::new();
            for (_, meta, text) in reasoning_items {
                parts_out.push(reasoning_part(&asst_id, parts_out.len(), text, meta));
            }
            for (content, response_meta) in assistant_text_parts {
                if content.trim().is_empty() {
                    continue;
                }
                parts_out.push(Part {
                    id: format!("{}.p{}", asst_id, parts_out.len()),
                    kind: PartKind::Prose,
                    content,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta,
                });
            }
            if parts_out.is_empty() {
                actions.push(DriverAction::Emit(make_error_event(
                    "llm_provider",
                    Some("empty_response"),
                    "Model returned no assistant text or tool calls.",
                    None,
                )));
                actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                    TurnStop::ProviderError,
                )));
                return actions;
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

        let asst_id = fresh_message_id();
        let mut assistant_parts = Vec::new();
        for (content, response_meta) in assistant_text_parts {
            if content.trim().is_empty() {
                continue;
            }
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::Prose,
                content,
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta,
            });
        }

        let mut calls = Vec::new();
        // Interleave reasoning items with tool calls to preserve the
        // original emission order. Some provider replays expect the
        // sequence `reasoning → function_call` from the turn in which both
        // were produced; swapping them can drop the reasoning/tool pairing.
        let mut reasoning_iter = reasoning_items.into_iter().peekable();
        for (tool_index, tool_call) in tool_calls.into_iter().enumerate() {
            while let Some((insert_index, _, _)) = reasoning_iter.peek() {
                if *insert_index > tool_index {
                    break;
                }
                let (_, meta, text) = reasoning_iter.next().expect("peek ok");
                assistant_parts.push(reasoning_part(&asst_id, assistant_parts.len(), text, meta));
            }
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::ToolCall,
                content: tool_call.input_json.clone(),
                attachment: None,
                tool_call_id: Some(tool_call.call_id.clone()),
                tool_name: Some(tool_call.tool_name.clone()),
                tool_replay: tool_call.replay.clone(),
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            });

            let args = serde_json::from_str::<Value>(&tool_call.input_json)
                .unwrap_or_else(|_| serde_json::json!({}));
            calls.push(PendingToolCall {
                call_id: tool_call.call_id,
                tool_name: tool_call.tool_name,
                args,
                replay: tool_call.replay,
            });
        }
        for (_, meta, text) in reasoning_iter {
            assistant_parts.push(reasoning_part(&asst_id, assistant_parts.len(), text, meta));
        }

        if !assistant_parts.is_empty() {
            actions.push(DriverAction::AppendEvents(vec![conversation_event(
                Message {
                    id: asst_id,
                    role: MessageRole::Assistant,
                    parts: shared_parts(assistant_parts),
                    origin: None,
                },
            )]));
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
        let mut terminal_outcome = None;

        for outcome in completed {
            if terminal_outcome.is_none() && outcome.output.is_success() {
                terminal_outcome = match outcome.output.control.as_ref() {
                    Some(lash_core::ToolControl::Handoff { session_id })
                        if !session_id.trim().is_empty() =>
                    {
                        Some(TurnOutcome::Handoff {
                            session_id: session_id.clone(),
                        })
                    }
                    Some(lash_core::ToolControl::Finish { value }) => {
                        Some(TurnOutcome::Finished(TurnFinish::ToolValue {
                            tool_name: outcome.tool_name.clone(),
                            value: value.to_json_value(),
                        }))
                    }
                    Some(lash_core::ToolControl::Fail { failure }) => {
                        Some(TurnOutcome::Stopped(TurnStop::ToolError {
                            tool_name: outcome.tool_name.clone(),
                            value: failure.to_json_value(),
                        }))
                    }
                    _ => None,
                };
            }

            append_model_return_parts(&mut result_parts, outcome.model_return);
        }

        if !result_parts.is_empty() {
            let user_id = fresh_message_id();
            reassign_part_ids(&user_id, &mut result_parts);
            actions.push(DriverAction::AppendEvents(vec![conversation_event(
                Message {
                    id: user_id,
                    role: MessageRole::User,
                    parts: shared_parts(result_parts),
                    origin: None,
                },
            )]));
        }

        if let Some(outcome) = terminal_outcome {
            actions.push(DriverAction::Finish(outcome));
            return actions;
        }

        actions.push(DriverAction::AdvanceModeIteration);
        let next_mode_iteration = ctx.mode_iteration() + 1;
        if let Some(max_turns) = ctx.max_turns()
            && next_mode_iteration >= ctx.mode_run_offset() + max_turns
        {
            let message_id = fresh_message_id();
            actions.push(DriverAction::AppendEvents(vec![conversation_event(
                turn_limit_exhausted_message(message_id, max_turns),
            )]));
            actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                TurnStop::MaxTurns,
            )));
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
        _waiting: WaitingExecState<lash_core::HostModeProtocol>,
        _result: Result<lash_core::ExecResponse, String>,
    ) -> Vec<DriverAction> {
        Vec::new()
    }
}

fn append_model_return_parts(parts: &mut Vec<Part>, model_return: lash_core::ModelToolReturn) {
    for part in model_return.parts {
        match part {
            lash_core::ModelToolReturnPart::Text(content) => {
                if content.is_empty() {
                    continue;
                }
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::ToolResult,
                    content,
                    attachment: None,
                    tool_call_id: Some(model_return.call_id.clone()),
                    tool_name: Some(model_return.tool_name.clone()),
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                });
            }
            lash_core::ModelToolReturnPart::Attachment(reference) => {
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(PartAttachment { reference }),
                    tool_call_id: Some(model_return.call_id.clone()),
                    tool_name: Some(model_return.tool_name.clone()),
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                });
            }
        }
    }
}

fn conversation_event(message: Message) -> SessionEventRecord {
    SessionEventRecord::Conversation(ConversationRecord::from_message(message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{
        AttachmentId, AttachmentMeta, ImageMediaType, MediaType, ModelToolReturn, ToolCallOutput,
        ToolValue,
    };

    fn image_ref(id: &str) -> lash_core::AttachmentRef {
        AttachmentMeta::new(
            AttachmentId::new(id),
            MediaType::Image(ImageMediaType::Png),
            4,
            Some(1),
            Some(1),
            Some("tiny".to_string()),
        )
        .as_ref()
    }

    #[test]
    fn tool_attachment_round_trips_to_part_kind_image() {
        let attachment = image_ref("att-1");
        let output = ToolCallOutput::success(ToolValue::Attachment(attachment.clone()));
        let model_return =
            ModelToolReturn::from_output("call-9".to_string(), "screenshot".to_string(), &output);

        let mut parts: Vec<Part> = Vec::new();
        append_model_return_parts(&mut parts, model_return);

        assert_eq!(parts.len(), 1, "single attachment yields single part");
        let part = &parts[0];
        assert!(matches!(part.kind, PartKind::Image));
        assert_eq!(part.content, "");
        assert_eq!(part.tool_call_id.as_deref(), Some("call-9"));
        assert_eq!(part.tool_name.as_deref(), Some("screenshot"));
        let part_attachment = part.attachment.as_ref().expect("attachment present");
        assert_eq!(part_attachment.reference.id, attachment.id);
    }

    #[test]
    fn tool_text_and_attachment_round_trip_preserves_order() {
        let attachment = image_ref("att-2");
        let output = ToolCallOutput::success(ToolValue::Array(vec![
            ToolValue::String("before".into()),
            ToolValue::Attachment(attachment.clone()),
            ToolValue::String("after".into()),
        ]));
        let model_return =
            ModelToolReturn::from_output("call-10".to_string(), "snap".to_string(), &output);

        let mut parts: Vec<Part> = Vec::new();
        append_model_return_parts(&mut parts, model_return);

        // The array projection emits compact JSON text fragments around the
        // attachment, preserving in-order position.
        assert_eq!(parts.len(), 3, "text + image + text yields three parts");
        assert!(matches!(parts[0].kind, PartKind::ToolResult));
        assert!(parts[0].content.starts_with("[\"before\""));
        assert!(matches!(parts[1].kind, PartKind::Image));
        assert_eq!(
            parts[1]
                .attachment
                .as_ref()
                .expect("attachment")
                .reference
                .id,
            attachment.id
        );
        assert!(matches!(parts[2].kind, PartKind::ToolResult));
        assert!(parts[2].content.ends_with("\"after\"]"));
    }
}
