//! Provider-agnostic OpenAI Responses API machinery.
//!
//! The OpenAI Responses streaming protocol (`response.output_item.*`,
//! `response.output_text.delta`, `response.reasoning_summary_*`,
//! `response.function_call_arguments.*`, …) is spoken verbatim by both the
//! direct OpenAI provider and the Codex OAuth provider. This module owns the
//! single implementation of:
//!
//! * [`ResponsesStreamState`] — the incremental stream accumulator, including
//!   message-by-id reconciliation, reasoning-part assembly, tool-call
//!   buffering, and final-response merging.
//! * [`process_sse_event`] / [`parse_sse_payload`] — the SSE event state
//!   machine.
//! * Request-building primitives shared by both providers: schema projection
//!   ([`build_tools`], [`projected_schema`]), tool-choice mapping, role names,
//!   image parts, and final-response parsing ([`response_parts_from_value`]).
//!
//! Each provider keeps only its genuine specifics: the OpenAI provider owns
//! `build_responses_request_body` (surrogate sanitisation, OpenRouter/local
//! field gating, assistant-message id flushing); Codex owns its request body
//! (system→`instructions` hoisting, tool-result image folding,
//! `clamp_reasoning_effort`), its endpoint/headers, and its failure
//! classification.

use base64::Engine;
use serde_json::{Value, json};
use std::collections::HashMap;

use lash_core::SchemaProjectionOverride;
use lash_core::llm::transport::{LlmTransportError, ProviderFailureKind};
use lash_core::llm::types::{
    LlmAttachment, LlmContentBlock, LlmOutputPart, LlmRequest, LlmResponse, LlmRole,
    LlmTerminalReason, LlmToolChoice, LlmUsage, ProviderReasoningReplay, ProviderReplayMeta,
    ResponseTextMeta,
};
use lash_llm_transport::util::parse_i64;
use lash_openai_schema::{
    OpenAiSchemaProfile, SchemaProjectionError, project_schema, project_structured_output,
    project_tool_parameters, responses_error_is_retryable,
};

// ---------------------------------------------------------------------------
// Request-building primitives
// ---------------------------------------------------------------------------

pub fn role_name(role: &LlmRole) -> &'static str {
    match role {
        LlmRole::User => "user",
        LlmRole::Assistant => "assistant",
        LlmRole::System => "system",
    }
}

pub fn input_image_part(att: &LlmAttachment) -> Value {
    let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
    json!({
        "type": "input_image",
        "image_url": format!("data:{};base64,{}", att.mime, b64),
    })
}

pub fn tool_choice_value(choice: &LlmToolChoice) -> &'static str {
    match choice {
        LlmToolChoice::Auto => "auto",
        LlmToolChoice::None => "none",
        LlmToolChoice::Required => "required",
    }
}

pub fn projection_error(provider: &str, err: SchemaProjectionError) -> LlmTransportError {
    LlmTransportError::new(format!(
        "{provider} schema projection failed: {}",
        err.first_diagnostic()
    ))
    .with_kind(ProviderFailureKind::Validation)
    .with_raw(
        json!({
            "profile": format!("{:?}", err.profile),
            "diagnostics": err.diagnostics,
        })
        .to_string(),
    )
}

pub fn projected_schema(
    provider: &str,
    canonical: &Value,
    overrides: &[SchemaProjectionOverride],
    profile: OpenAiSchemaProfile,
) -> Result<Value, LlmTransportError> {
    if let Some(override_schema) = overrides
        .iter()
        .find(|projection| projection.profile == profile.projection_id())
        .map(|projection| projection.schema.clone())
    {
        return Ok(override_schema);
    }
    match profile {
        OpenAiSchemaProfile::ToolParameters => {
            project_tool_parameters(canonical).map(|projection| projection.schema)
        }
        OpenAiSchemaProfile::StructuredOutput => {
            project_structured_output(canonical).map(|projection| projection.schema)
        }
        OpenAiSchemaProfile::StrictToolParameters => {
            project_schema(canonical, profile).map(|projection| projection.schema)
        }
    }
    .map_err(|err| projection_error(provider, err))
}

pub fn build_tools(
    provider: &str,
    req: &lash_core::llm::types::LlmRequest,
) -> Result<Vec<Value>, LlmTransportError> {
    req.tools
        .iter()
        .map(|tool| {
            let parameters = projected_schema(
                provider,
                &tool.input_schema,
                &tool.input_schema_projections,
                OpenAiSchemaProfile::ToolParameters,
            )?;
            Ok(json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "strict": false,
            }))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Request input assembly
// ---------------------------------------------------------------------------

/// The handful of genuine deltas between the direct-OpenAI and Codex flavours
/// of the Responses `input` array. Everything else — the per-message block
/// loop, the reasoning-replay item shape, function_call/function_call_output
/// emission, the `system → instructions` hoist — is identical.
#[derive(Clone, Copy, Debug)]
pub struct ResponsesInputOptions {
    /// OpenAI assigns each assistant `message` item a stable id
    /// (`msg_lash_{message}_{part}` when the request carries none), tracks
    /// `status`/`phase` from [`ResponseTextMeta`], and tags `output_text`
    /// parts with an empty `annotations` array. Codex emits none of this.
    pub assistant_message_metadata: bool,
    /// Codex folds sibling user `input_image` parts that follow a
    /// `function_call_output` into that output's `output` array so the image
    /// reads as the tool's result. OpenAI keeps them as a standalone user turn.
    pub fold_tool_result_images: bool,
}

impl ResponsesInputOptions {
    /// Direct OpenAI Responses: synthetic assistant ids + phase/annotations,
    /// no tool-result image folding.
    pub const OPENAI: Self = Self {
        assistant_message_metadata: true,
        fold_tool_result_images: false,
    };

    /// Codex Responses: no synthetic ids/phase/annotations, fold tool-result
    /// images into the preceding `function_call_output`.
    pub const CODEX: Self = Self {
        assistant_message_metadata: false,
        fold_tool_result_images: true,
    };
}

#[allow(clippy::too_many_arguments)]
fn flush_pending_content(
    pending: &mut Vec<Value>,
    input: &mut Vec<Value>,
    role: &'static str,
    is_user: bool,
    opts: &ResponsesInputOptions,
    response_meta: Option<ResponseTextMeta>,
    message_index: usize,
    part_index: usize,
) {
    if pending.is_empty() {
        return;
    }
    let content = std::mem::take(pending);
    if opts.assistant_message_metadata && role == "assistant" {
        let meta = response_meta.unwrap_or(ResponseTextMeta {
            id: Some(format!("msg_lash_{message_index}_{part_index}")),
            status: Some("completed".to_string()),
            phase: None,
        });
        let mut item = json!({
            "type": "message",
            "role": "assistant",
            "id": meta.id.unwrap_or_else(|| format!("msg_lash_{message_index}_{part_index}")),
            "status": meta.status.unwrap_or_else(|| "completed".to_string()),
            "content": content,
        });
        if let Some(phase) = meta.phase.as_ref() {
            item["phase"] = json!(phase);
        }
        input.push(item);
        return;
    }
    if is_user
        && let Some(prev) = input.last_mut()
        && prev.get("role").and_then(|v| v.as_str()) == Some("user")
        && prev.get("content").is_some_and(|v| v.is_array())
    {
        prev["content"].as_array_mut().unwrap().extend(content);
    } else {
        input.push(json!({
            "role": role,
            "content": content,
        }));
    }
}

/// Walk backwards from the last input item: if the final entry is a user
/// `content` message whose parts are all `input_image`, and the entry before
/// it is a `function_call_output`, promote the image parts into the `output`
/// of that function_call_output so the server sees the image as the tool's
/// result rather than as a standalone user turn.
fn fold_tool_result_images(input: &mut Vec<Value>) {
    if input.len() < 2 {
        return;
    }
    let last_idx = input.len() - 1;
    let is_user_image_msg = input[last_idx].get("role").and_then(|v| v.as_str()) == Some("user")
        && input[last_idx]
            .get("content")
            .and_then(|c| c.as_array())
            .is_some_and(|parts| {
                parts
                    .iter()
                    .all(|p| p.get("type").and_then(|t| t.as_str()) == Some("input_image"))
            });
    if !is_user_image_msg {
        return;
    }
    let prev_is_call_output =
        input[last_idx - 1].get("type").and_then(|v| v.as_str()) == Some("function_call_output");
    if !prev_is_call_output {
        return;
    }
    let last = input.remove(last_idx);
    let image_parts = last
        .get("content")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();
    let prev = input.last_mut().expect("function_call_output present");
    if !prev["output"].is_array() {
        let existing_text = prev["output"]
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_default();
        let mut parts: Vec<Value> = Vec::new();
        if !existing_text.is_empty() {
            parts.push(json!({
                "type": "input_text",
                "text": existing_text,
            }));
        }
        prev["output"] = Value::Array(parts);
    }
    prev["output"].as_array_mut().unwrap().extend(image_parts);
}

fn reasoning_replay_item(text: &str, replay: Option<&ProviderReasoningReplay>) -> Option<Value> {
    // Only replay reasoning items that actually carry an encrypted blob.
    // Display-only summaries (no blob) must not be fed back — the server will
    // either ignore them or reject the turn.
    let blob = replay.and_then(|meta| meta.encrypted_content.as_deref())?;
    let summary = replay
        .map(|meta| meta.summary.as_slice())
        .unwrap_or_default();
    let summary_items: Vec<Value> = if summary.is_empty() {
        if text.is_empty() {
            Vec::new()
        } else {
            vec![json!({"type": "summary_text", "text": text})]
        }
    } else {
        summary
            .iter()
            .map(|entry| json!({"type": "summary_text", "text": entry}))
            .collect()
    };
    let mut item = json!({
        "type": "reasoning",
        "summary": summary_items,
        "encrypted_content": blob,
    });
    if let Some(id) = replay.and_then(|meta| meta.item_id.as_deref())
        && !id.is_empty()
    {
        item["id"] = json!(id);
    }
    Some(item)
}

/// Build the Responses `(instructions, input)` pair shared by both the direct
/// OpenAI provider and Codex. System-role text is hoisted into `instructions`;
/// the remaining blocks become the `input` array. `opts` selects the few real
/// provider deltas (synthetic assistant ids, tool-result image folding).
pub fn build_responses_input(
    req: &LlmRequest,
    opts: ResponsesInputOptions,
) -> (String, Vec<Value>) {
    let mut instructions: Vec<String> = Vec::new();
    let mut input: Vec<Value> = Vec::new();

    for (message_index, msg) in req.messages.iter().enumerate() {
        if matches!(msg.role, LlmRole::System) {
            for block in msg.blocks.iter() {
                if let LlmContentBlock::Text { text, .. } = block
                    && !text.is_empty()
                {
                    instructions.push(text.to_string());
                }
            }
            continue;
        }

        let role = role_name(&msg.role);
        let is_user = matches!(msg.role, LlmRole::User);
        let mut pending_content: Vec<Value> = Vec::new();
        let mut pending_meta: Option<ResponseTextMeta> = None;
        let mut pending_part_index = 0usize;

        // Codex folds the image/placeholder blocks that follow a ToolResult
        // into that tool's `output`. One scan yields both the folded image
        // parts (keyed by ToolResult block index) and the sibling indices to
        // skip in the main loop so they aren't double-emitted.
        let (tool_result_image_folds, consumed_after_tool_result) = if opts.fold_tool_result_images
        {
            collect_tool_result_image_folds(req, msg)
        } else {
            Default::default()
        };

        for (part_index, block) in msg.blocks.iter().enumerate() {
            if consumed_after_tool_result.contains(&part_index) {
                continue;
            }
            match block {
                LlmContentBlock::Text {
                    text,
                    response_meta,
                    ..
                } => {
                    if text.is_empty() {
                        continue;
                    }
                    if opts.assistant_message_metadata
                        && matches!(msg.role, LlmRole::Assistant)
                        && (!pending_content.is_empty() || response_meta.is_some())
                    {
                        flush_pending_content(
                            &mut pending_content,
                            &mut input,
                            role,
                            false,
                            &opts,
                            pending_meta.take(),
                            message_index,
                            pending_part_index,
                        );
                        pending_part_index = part_index;
                        pending_meta = response_meta.clone();
                    }
                    let part_type = if matches!(msg.role, LlmRole::Assistant) {
                        "output_text"
                    } else {
                        "input_text"
                    };
                    if opts.assistant_message_metadata && part_type == "output_text" {
                        pending_content.push(json!({
                            "type": part_type,
                            "text": text,
                            "annotations": [],
                        }));
                    } else {
                        pending_content.push(json!({
                            "type": part_type,
                            "text": text,
                        }));
                    }
                }
                LlmContentBlock::Image { attachment_idx } => {
                    if is_user && let Some(att) = req.attachments.get(*attachment_idx) {
                        pending_content.push(input_image_part(att));
                    }
                }
                LlmContentBlock::Reasoning { text, replay, .. } => {
                    flush_pending_content(
                        &mut pending_content,
                        &mut input,
                        role,
                        is_user,
                        &opts,
                        pending_meta.take(),
                        message_index,
                        pending_part_index,
                    );
                    if let Some(item) = reasoning_replay_item(text, replay.as_ref()) {
                        input.push(item);
                    }
                }
                LlmContentBlock::ToolCall {
                    call_id,
                    tool_name,
                    input_json,
                    replay,
                    ..
                } => {
                    flush_pending_content(
                        &mut pending_content,
                        &mut input,
                        role,
                        is_user,
                        &opts,
                        pending_meta.take(),
                        message_index,
                        pending_part_index,
                    );
                    let mut item = json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": input_json,
                    });
                    // `id` (e.g. `fc_...`) pairs a function_call with its
                    // sibling reasoning item across turns; omit when absent.
                    if let Some(id) = replay.as_ref().and_then(|meta| meta.item_id.as_deref()) {
                        item["id"] = json!(id);
                    }
                    input.push(item);
                }
                LlmContentBlock::ToolResult {
                    call_id, content, ..
                } => {
                    flush_pending_content(
                        &mut pending_content,
                        &mut input,
                        role,
                        is_user,
                        &opts,
                        pending_meta.take(),
                        message_index,
                        pending_part_index,
                    );
                    let image_parts = tool_result_image_folds
                        .get(&part_index)
                        .cloned()
                        .unwrap_or_default();
                    if image_parts.is_empty() {
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": content,
                        }));
                    } else {
                        let mut parts: Vec<Value> = Vec::new();
                        if !content.is_empty() {
                            parts.push(json!({
                                "type": "input_text",
                                "text": content,
                            }));
                        }
                        parts.extend(image_parts);
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": parts,
                        }));
                    }
                }
            }
        }
        flush_pending_content(
            &mut pending_content,
            &mut input,
            role,
            is_user,
            &opts,
            pending_meta.take(),
            message_index,
            pending_part_index,
        );

        if opts.fold_tool_result_images && is_user {
            fold_tool_result_images(&mut input);
        }
    }

    (instructions.join("\n\n"), input)
}

/// For each `ToolResult` block in `msg`, the Codex-folded image parts (its
/// trailing sibling `Image` / `[Tool image: …]` blocks) keyed by the
/// ToolResult's block index, plus the set of all sibling indices consumed this
/// way so the main loop skips them. One scan replaces the former skip-set
/// pre-pass plus a separate per-ToolResult re-scan.
fn collect_tool_result_image_folds(
    req: &LlmRequest,
    msg: &lash_core::llm::types::LlmMessage,
) -> (
    std::collections::HashMap<usize, Vec<Value>>,
    std::collections::HashSet<usize>,
) {
    let mut folds: std::collections::HashMap<usize, Vec<Value>> = std::collections::HashMap::new();
    let mut consumed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (idx, block) in msg.blocks.iter().enumerate() {
        if !matches!(block, LlmContentBlock::ToolResult { .. }) {
            continue;
        }
        let mut parts: Vec<Value> = Vec::new();
        for (j, sibling) in msg.blocks.iter().enumerate().skip(idx + 1) {
            match sibling {
                LlmContentBlock::Image { attachment_idx } => {
                    if let Some(att) = req.attachments.get(*attachment_idx) {
                        parts.push(input_image_part(att));
                    }
                    consumed.insert(j);
                }
                LlmContentBlock::Text { text: t, .. } if t.starts_with("[Tool image:") => {
                    consumed.insert(j);
                }
                _ => break,
            }
        }
        if !parts.is_empty() {
            folds.insert(idx, parts);
        }
    }
    (folds, consumed)
}

// ---------------------------------------------------------------------------
// Terminal reason + response assembly
// ---------------------------------------------------------------------------

/// Map a final Responses object to a terminal reason. Honours both
/// `incomplete_details` and the camelCase `incompleteDetails` some gateways
/// emit. Falls back to ToolUse/Stop based on the assembled parts.
pub fn terminal_reason_from_response_value(
    value: &Value,
    parts: &[LlmOutputPart],
) -> LlmTerminalReason {
    let incomplete_details = value
        .get("incomplete_details")
        .or_else(|| value.get("incompleteDetails"))
        .filter(|details| !details.is_null());
    let incomplete_reason =
        incomplete_details.and_then(|details| details.get("reason").and_then(Value::as_str));
    // Switch on the documented `incomplete_details.reason` rather than treating
    // every `incomplete` status as an output-token cap: only the token-limit
    // reasons are an OutputLimit, safety reasons are a ContentFilter, and any
    // other reason is a genuine provider failure.
    match incomplete_reason {
        Some("max_output_tokens" | "max_tokens") => return LlmTerminalReason::OutputLimit,
        Some("content_filter" | "safety") => return LlmTerminalReason::ContentFilter,
        Some(_) => return LlmTerminalReason::ProviderError,
        None => {}
    }
    if value.get("status").and_then(Value::as_str) == Some("cancelled") {
        return LlmTerminalReason::Cancelled;
    }
    if value.get("status").and_then(Value::as_str) == Some("failed") {
        return LlmTerminalReason::ProviderError;
    }
    // An `incomplete` status with no recognizable reason: prefer the assembled
    // parts (ToolUse/Stop) when present, otherwise surface a provider error.
    if value.get("status").and_then(Value::as_str) == Some("incomplete") && parts.is_empty() {
        return LlmTerminalReason::ProviderError;
    }
    terminal_reason_from_parts(parts)
}

/// Terminal reason from assembled parts alone: ToolUse when any tool call is
/// present, otherwise Stop.
pub fn terminal_reason_from_parts(parts: &[LlmOutputPart]) -> LlmTerminalReason {
    if parts
        .iter()
        .any(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
    {
        LlmTerminalReason::ToolUse
    } else {
        LlmTerminalReason::Stop
    }
}

/// Collapse a finished [`ResponsesStreamState`] into an [`LlmResponse`]. Used
/// by Codex; the direct OpenAI driver inlines an equivalent assembly with its
/// own provider-usage/streaming plumbing.
pub fn response_from_stream_state(
    state: ResponsesStreamState,
    request_body: Option<String>,
    http_summary: String,
) -> LlmResponse {
    let parts = state.response_parts();
    let terminal_reason = match &state.final_response {
        Some(final_response) => terminal_reason_from_response_value(final_response, &parts),
        None => terminal_reason_from_parts(&parts),
    };
    let full_text = if !state.full_text.is_empty() {
        state.full_text.clone()
    } else {
        parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>()
    };
    LlmResponse {
        full_text,
        parts,
        usage: state.usage,
        terminal_reason,
        terminal_diagnostic: None,
        provider_usage: None,
        request_body,
        http_summary: Some(http_summary),
    }
}

// ---------------------------------------------------------------------------
// Final-response parsing
// ---------------------------------------------------------------------------

pub fn response_text_meta_from_message_item(item: &Value) -> ResponseTextMeta {
    ResponseTextMeta {
        id: item.get("id").and_then(|v| v.as_str()).map(str::to_string),
        status: item
            .get("status")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| Some("completed".to_string())),
        phase: item
            .get("phase")
            .and_then(|v| v.as_str())
            .map(str::to_string),
    }
}

pub fn message_text_from_item(item: &Value) -> String {
    item.get("content")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|part| match part.get("type").and_then(|v| v.as_str()) {
            Some("output_text") => part.get("text").and_then(|v| v.as_str()),
            Some("refusal") => part
                .get("refusal")
                .and_then(|v| v.as_str())
                .or_else(|| part.get("text").and_then(|v| v.as_str())),
            _ => None,
        })
        .collect::<String>()
}

pub fn extract_text(value: &Value) -> String {
    if let Some(s) = value.get("output_text").and_then(|v| v.as_str()) {
        return s.to_string();
    }
    value
        .get("output")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(message_text_from_item)
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

pub fn has_structured_message_text(value: &Value) -> bool {
    value
        .get("output")
        .and_then(|v| v.as_array())
        .is_some_and(|output| {
            output.iter().any(|item| {
                item.get("type").and_then(|v| v.as_str()) == Some("message")
                    && !message_text_from_item(item).is_empty()
            })
        })
}

pub fn response_parts_from_value(value: &Value) -> Vec<LlmOutputPart> {
    let mut parts = Vec::new();
    if let Some(output) = value.get("output").and_then(|v| v.as_array()) {
        for item in output {
            match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "reasoning" => {
                    let summary = item
                        .get("summary")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|entry| {
                                    entry.get("text").and_then(|v| v.as_str()).map(String::from)
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let text = summary.join("\n\n");
                    parts.push(LlmOutputPart::Reasoning {
                        text,
                        replay: Some(ProviderReasoningReplay {
                            item_id: item.get("id").and_then(|v| v.as_str()).map(str::to_string),
                            encrypted_content: item
                                .get("encrypted_content")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            signature: None,
                            redacted: false,
                            summary,
                        }),
                    });
                }
                "message" => {
                    let text = message_text_from_item(item);
                    if !text.is_empty() {
                        parts.push(LlmOutputPart::Text {
                            text,
                            response_meta: Some(response_text_meta_from_message_item(item)),
                        });
                    }
                }
                "function_call" => {
                    let Some(name) = item.get("name").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let arguments = item
                        .get("arguments")
                        .map(|v| {
                            v.as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| v.to_string())
                        })
                        .unwrap_or_else(|| "{}".to_string());
                    parts.push(LlmOutputPart::ToolCall {
                        call_id: item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                        tool_name: name.to_string(),
                        input_json: arguments,
                        replay: item.get("id").and_then(|v| v.as_str()).map(|id| {
                            ProviderReplayMeta {
                                item_id: Some(id.to_string()),
                                opaque: None,
                            }
                        }),
                    });
                }
                _ => {}
            }
        }
    }
    if !parts
        .iter()
        .any(|part| matches!(part, LlmOutputPart::Text { text, .. } if !text.is_empty()))
        && let Some(text) = value.get("output_text").and_then(|v| v.as_str())
        && !text.is_empty()
    {
        parts.push(LlmOutputPart::Text {
            text: text.to_string(),
            response_meta: None,
        });
    }
    parts
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

pub fn usage_from_response_value(value: &Value) -> LlmUsage {
    usage_from_usage_value(value.get("usage").unwrap_or(&Value::Null))
}

pub fn usage_from_usage_value(usage: &Value) -> LlmUsage {
    let cached_tokens = parse_i64(
        usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .or_else(|| {
                usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
            })
            .or_else(|| usage.get("cached_input_tokens"))
            .or_else(|| usage.get("cached_tokens"))
            .or_else(|| usage.get("prompt_cache_hit_tokens")),
    );
    let cache_write_tokens = parse_i64(
        usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cache_write_tokens"))
            .or_else(|| {
                usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cache_write_tokens"))
            }),
    );
    LlmUsage {
        input_tokens: parse_i64(
            usage
                .get("input_tokens")
                .or_else(|| usage.get("prompt_tokens")),
        ),
        output_tokens: parse_i64(
            usage
                .get("output_tokens")
                .or_else(|| usage.get("completion_tokens")),
        ),
        cached_input_tokens: if cache_write_tokens > 0 {
            cached_tokens.saturating_sub(cache_write_tokens).max(0)
        } else {
            cached_tokens
        },
        reasoning_tokens: parse_i64(
            usage
                .get("reasoning_tokens")
                .or_else(|| {
                    usage
                        .get("output_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                })
                .or_else(|| {
                    usage
                        .get("completion_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                }),
        ),
    }
}

/// Overwrite `dst` with `next` field-by-field, keeping prior non-zero values
/// when the incoming field is zero. Streaming usage arrives incrementally and
/// later events may report only a subset of counters.
pub fn merge_usage(dst: &mut LlmUsage, next: &LlmUsage) {
    if next.input_tokens > 0 {
        dst.input_tokens = next.input_tokens;
    }
    if next.output_tokens > 0 {
        dst.output_tokens = next.output_tokens;
    }
    if next.cached_input_tokens > 0 {
        dst.cached_input_tokens = next.cached_input_tokens;
    }
    if next.reasoning_tokens > 0 {
        dst.reasoning_tokens = next.reasoning_tokens;
    }
}

// ---------------------------------------------------------------------------
// Stream state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct ResponsesStreamingToolCall {
    pub call_id: String,
    pub tool_name: String,
    pub input_json: String,
    /// Responses API item-id (e.g. `fc_...`). Preserved so we can re-emit it on
    /// the next request body alongside `call_id`; the server uses it to pair a
    /// function_call with its sibling reasoning item.
    pub item_id: String,
}

#[derive(Clone, Debug, Default)]
pub struct ResponsesStreamState {
    pub full_text: String,
    pub pending_text_deltas: Vec<String>,
    pub parts: Vec<LlmOutputPart>,
    pub usage: LlmUsage,
    pub provider_usage: Option<Value>,
    pub final_response: Option<Value>,
    pub current_text_part: Option<usize>,
    pub current_message_item_id: Option<String>,
    /// Maps a server message item-id to the index of its `Text` part so that
    /// repeated `output_item.added`/`.done` for the same id reconcile into one
    /// part instead of duplicating.
    pub message_parts: HashMap<String, usize>,
    /// Index of the reasoning-summary part currently receiving deltas. The
    /// server groups reasoning output into multiple "parts" (paragraphs); we
    /// keep one slot per part instead of merging into a single blob.
    pub current_reasoning_part: Option<usize>,
    pub reasoning_deltas: Vec<String>,
    pub tool_calls: HashMap<String, ResponsesStreamingToolCall>,
}

impl ResponsesStreamState {
    pub fn begin_message(&mut self, item: Option<&Value>) {
        let item_id = item
            .and_then(|item| item.get("id").and_then(|v| v.as_str()))
            .map(str::to_string);
        let meta = item.map(response_text_meta_from_message_item);
        let index = self.message_part_index(item_id.as_deref(), meta);
        self.current_text_part = Some(index);
        self.current_message_item_id = item_id;
    }

    pub fn finish_message(&mut self, item: Option<&Value>) {
        if let Some(item) = item {
            let text = message_text_from_item(item);
            let meta = response_text_meta_from_message_item(item);
            let item_id = meta.id.clone();
            let index = self.message_part_index(item_id.as_deref(), Some(meta));
            if !text.is_empty() {
                self.reconcile_text_part(index, &text);
            }
        }
        self.current_text_part = None;
        self.current_message_item_id = None;
    }

    pub fn push_text_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        let part_index = self.ensure_text_part_index();
        self.append_text_delta_to_part(part_index, piece);
    }

    fn reconcile_text_part(&mut self, part_index: usize, text: &str) {
        if text.is_empty() {
            return;
        }
        let existing = self
            .parts
            .get(part_index)
            .and_then(|part| match part {
                LlmOutputPart::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        if text == existing {
            return;
        }
        if let Some(suffix) = text.strip_prefix(existing.as_str()) {
            self.append_text_delta_to_part(part_index, suffix);
            return;
        }
        self.set_text_part(part_index, text.to_string());
    }

    pub fn merge_final_response(&mut self, response: &Value) {
        let structured_message_text = has_structured_message_text(response);
        for part in response_parts_from_value(response) {
            match part {
                LlmOutputPart::Text {
                    text,
                    response_meta,
                } => {
                    let item_id = response_meta.as_ref().and_then(|meta| meta.id.clone());
                    if item_id.is_none()
                        && !structured_message_text
                        && self.parts.iter().any(|part| {
                            matches!(part, LlmOutputPart::Text { text, .. } if !text.is_empty())
                        })
                    {
                        continue;
                    }
                    let index = self.message_part_index(item_id.as_deref(), response_meta);
                    self.reconcile_text_part(index, &text);
                }
                part @ LlmOutputPart::Reasoning { .. } => {
                    let part_item_id = match &part {
                        LlmOutputPart::Reasoning { replay, .. } => {
                            replay.as_ref().and_then(|meta| meta.item_id.as_deref())
                        }
                        _ => None,
                    };
                    if let Some(id) = part_item_id
                        && let Some(existing) = self.parts.iter_mut().find(|existing| {
                            matches!(existing, LlmOutputPart::Reasoning { replay, .. } if replay.as_ref().and_then(|meta| meta.item_id.as_deref()) == Some(id))
                        })
                    {
                        *existing = part;
                        continue;
                    }
                    if !self.parts.iter().any(|existing| existing == &part) {
                        self.parts.push(part);
                    }
                }
                part @ LlmOutputPart::ToolCall { .. } => {
                    let (part_item_id, part_call_id) = match &part {
                        LlmOutputPart::ToolCall {
                            replay, call_id, ..
                        } => (
                            replay.as_ref().and_then(|meta| meta.item_id.as_deref()),
                            call_id.as_str(),
                        ),
                        _ => (None, ""),
                    };
                    let duplicate = self.parts.iter().any(|existing| match existing {
                        LlmOutputPart::ToolCall {
                            replay: existing_replay,
                            call_id: existing_call_id,
                            ..
                        } => {
                            part_item_id
                                .zip(
                                    existing_replay
                                        .as_ref()
                                        .and_then(|meta| meta.item_id.as_deref()),
                                )
                                .is_some_and(|(a, b)| a == b)
                                || (!part_call_id.is_empty() && part_call_id == existing_call_id)
                        }
                        _ => false,
                    });
                    if !duplicate {
                        self.parts.push(part);
                    }
                }
            }
        }
        self.recompute_full_text();
    }

    pub fn ensure_text_part_index(&mut self) -> usize {
        if let Some(index) = self.current_text_part {
            return index;
        }
        if let Some(index) = self
            .parts
            .iter()
            .rposition(|part| matches!(part, LlmOutputPart::Text { .. }))
        {
            return index;
        }
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Text {
            text: String::new(),
            response_meta: None,
        });
        index
    }

    fn message_part_index(
        &mut self,
        item_id: Option<&str>,
        response_meta: Option<ResponseTextMeta>,
    ) -> usize {
        let index = if let Some(item_id) = item_id.filter(|id| !id.is_empty()) {
            if let Some(index) = self.message_parts.get(item_id).copied() {
                index
            } else {
                let index = self.parts.len();
                self.parts.push(LlmOutputPart::Text {
                    text: String::new(),
                    response_meta: response_meta.clone(),
                });
                self.message_parts.insert(item_id.to_string(), index);
                index
            }
        } else if let Some(index) = self.current_text_part {
            index
        } else {
            let index = self.parts.len();
            self.parts.push(LlmOutputPart::Text {
                text: String::new(),
                response_meta: response_meta.clone(),
            });
            index
        };

        if let Some(response_meta) = response_meta
            && let Some(LlmOutputPart::Text {
                response_meta: existing_meta,
                ..
            }) = self.parts.get_mut(index)
        {
            *existing_meta = Some(response_meta);
        }
        index
    }

    fn set_text_part(&mut self, part_index: usize, text: String) {
        if let Some(LlmOutputPart::Text { text: existing, .. }) = self.parts.get_mut(part_index) {
            *existing = text;
        }
        self.recompute_full_text();
    }

    fn append_text_delta_to_part(&mut self, part_index: usize, piece: &str) {
        if piece.is_empty() {
            return;
        }
        if let Some(LlmOutputPart::Text { text, .. }) = self.parts.get_mut(part_index) {
            text.push_str(piece);
        }
        self.pending_text_deltas.push(piece.to_string());
        self.recompute_full_text();
    }

    pub fn recompute_full_text(&mut self) {
        self.full_text.clear();
        for part in &self.parts {
            if let LlmOutputPart::Text { text, .. } = part {
                self.full_text.push_str(text);
            }
        }
    }

    pub fn begin_reasoning_part(&mut self) {
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Reasoning {
            text: String::new(),
            replay: None,
        });
        self.current_reasoning_part = Some(index);
    }

    pub fn push_reasoning_delta(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let index = match self.current_reasoning_part {
            Some(index) => index,
            None => {
                // Some providers send a delta before the `part.added` event.
                // Open an implicit part so we don't drop text.
                self.begin_reasoning_part();
                self.current_reasoning_part
                    .expect("reasoning part just pushed")
            }
        };
        if let Some(LlmOutputPart::Reasoning { text, .. }) = self.parts.get_mut(index) {
            text.push_str(delta);
        }
        self.reasoning_deltas.push(delta.to_string());
    }

    pub fn finish_reasoning_part(&mut self) {
        // Drop the cursor; the next `part.added` opens a fresh slot. Trim
        // trailing whitespace so concatenated paragraphs don't carry blanks.
        if let Some(index) = self.current_reasoning_part.take()
            && let Some(LlmOutputPart::Reasoning { text, .. }) = self.parts.get_mut(index)
        {
            let trimmed = text.trim_end();
            if trimmed.len() != text.len() {
                *text = trimmed.to_string();
            }
        }
    }

    /// Populate the most recent reasoning part with the authoritative payload
    /// from `response.output_item.done`: the `rs_...` id, the `summary[*].text`
    /// entries, and the `encrypted_content` blob replayed on the next turn.
    pub fn finalize_reasoning_item(&mut self, item: &Value) {
        let Some((_, part)) = self
            .parts
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(_, p)| matches!(p, LlmOutputPart::Reasoning { .. }))
        else {
            return;
        };
        let LlmOutputPart::Reasoning { replay, .. } = part else {
            return;
        };
        let meta = replay.get_or_insert_with(ProviderReasoningReplay::default);
        if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
            meta.item_id = Some(id.to_string());
        }
        if let Some(blob) = item.get("encrypted_content").and_then(|v| v.as_str()) {
            meta.encrypted_content = Some(blob.to_string());
        }
        if let Some(arr) = item.get("summary").and_then(|v| v.as_array()) {
            let texts: Vec<String> = arr
                .iter()
                .filter_map(|entry| entry.get("text").and_then(|v| v.as_str()).map(String::from))
                .collect();
            if !texts.is_empty() {
                meta.summary = texts;
            }
        }
    }

    pub fn take_reasoning_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reasoning_deltas)
    }

    pub fn take_text_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_text_deltas)
    }

    pub fn update_tool_call_from_item(&mut self, item: &Value) -> Option<String> {
        let item_id = item.get("id").and_then(|v| v.as_str())?.to_string();
        let tool_call = self.tool_calls.entry(item_id.clone()).or_default();
        if tool_call.item_id.is_empty() {
            tool_call.item_id = item_id.clone();
        }
        if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
            tool_call.call_id = call_id.to_string();
        }
        if let Some(tool_name) = item.get("name").and_then(|v| v.as_str()) {
            tool_call.tool_name = tool_name.to_string();
        }
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str())
            && !arguments.is_empty()
        {
            tool_call.input_json = arguments.to_string();
        }
        Some(item_id)
    }

    pub fn push_tool_call_delta(&mut self, item_id: &str, delta: &str) {
        if item_id.is_empty() || delta.is_empty() {
            return;
        }
        self.tool_calls
            .entry(item_id.to_string())
            .or_default()
            .input_json
            .push_str(delta);
    }

    pub fn set_tool_call_arguments(&mut self, item_id: &str, arguments: &str) {
        if item_id.is_empty() {
            return;
        }
        self.tool_calls
            .entry(item_id.to_string())
            .or_default()
            .input_json = arguments.to_string();
    }

    pub fn finish_tool_call(&mut self, item: &Value) -> Option<LlmOutputPart> {
        let item_id = self.update_tool_call_from_item(item)?;
        let mut tool_call = self.tool_calls.remove(&item_id).unwrap_or_default();
        if tool_call.call_id.is_empty() {
            tool_call.call_id = uuid::Uuid::new_v4().to_string();
        }
        if tool_call.tool_name.is_empty() {
            return None;
        }
        if tool_call.input_json.is_empty() {
            tool_call.input_json = "{}".to_string();
        }
        let part = LlmOutputPart::ToolCall {
            call_id: tool_call.call_id,
            tool_name: tool_call.tool_name,
            input_json: tool_call.input_json,
            replay: (!tool_call.item_id.is_empty()).then_some(ProviderReplayMeta {
                item_id: Some(tool_call.item_id),
                opaque: None,
            }),
        };
        if !self.parts.iter().any(|existing| existing == &part) {
            self.parts.push(part.clone());
            return Some(part);
        }
        None
    }

    /// Non-empty parts collected so far, falling back to the final response's
    /// parsed parts / text when nothing streamed.
    pub fn response_parts(&self) -> Vec<LlmOutputPart> {
        let parts = self
            .parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text, .. } if text.is_empty() => None,
                LlmOutputPart::Reasoning { text, .. } if text.trim().is_empty() => None,
                other => Some(other.clone()),
            })
            .collect::<Vec<_>>();
        if !parts.is_empty() {
            return parts;
        }
        if let Some(final_response) = &self.final_response {
            let parts = response_parts_from_value(final_response);
            if !parts.is_empty() {
                return parts;
            }
            let text = extract_text(final_response);
            if !text.is_empty() {
                return vec![LlmOutputPart::Text {
                    text,
                    response_meta: None,
                }];
            }
        }
        if !self.full_text.is_empty() {
            return vec![LlmOutputPart::Text {
                text: self.full_text.clone(),
                response_meta: None,
            }];
        }
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// SSE event state machine
// ---------------------------------------------------------------------------

fn error_message_from_response_failed(provider: &str, event: &Value) -> String {
    event
        .get("response")
        .and_then(|r| r.get("error"))
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .or_else(|| {
            event
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
        })
        .map(str::to_string)
        .unwrap_or_else(|| format!("{provider} response failed"))
}

/// Drive one SSE event into `state`. `emitted_parts`, when supplied, receives
/// each tool-call part as it finalizes so the caller can stream it. `provider`
/// names the backend for error messages.
pub fn process_sse_event(
    provider: &str,
    raw: &str,
    state: &mut ResponsesStreamState,
    emitted_parts: Option<&mut Vec<LlmOutputPart>>,
) -> Result<(), LlmTransportError> {
    let raw = raw.trim();
    if raw.is_empty() || raw == "[DONE]" {
        return Ok(());
    }
    let event: Value = serde_json::from_str(raw).map_err(|e| {
        LlmTransportError::new(format!("Invalid {provider} SSE payload: {e}")).with_raw(raw)
    })?;
    let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if event_type == "error" {
        let retryable = event
            .get("error")
            .map(responses_error_is_retryable)
            .unwrap_or(false);
        let message = event
            .get("message")
            .and_then(|v| v.as_str())
            .or_else(|| {
                event
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
            })
            .unwrap_or("OpenAI-compatible stream error");
        return Err(LlmTransportError::new(message)
            .retryable(retryable)
            .with_raw(event.to_string()));
    }

    if let Some(resp) = event.get("response") {
        state.final_response = Some(resp.clone());
        state.provider_usage = resp.get("usage").cloned();
        merge_usage(&mut state.usage, &usage_from_response_value(resp));
    } else {
        merge_usage(&mut state.usage, &usage_from_response_value(&event));
    }

    match event_type {
        "response.output_item.added" => {
            if let Some(item) = event.get("item") {
                match item.get("type").and_then(|v| v.as_str()) {
                    Some("message") => state.begin_message(Some(item)),
                    Some("function_call") => {
                        let _ = state.update_tool_call_from_item(item);
                    }
                    Some("reasoning") => state.begin_reasoning_part(),
                    _ => {}
                }
            }
        }
        "response.reasoning_summary_part.added" => state.begin_reasoning_part(),
        "response.reasoning_summary_text.delta" => {
            if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                state.push_reasoning_delta(delta);
            }
        }
        "response.reasoning_summary_text.done" => {
            // The `text` field is the full text for the current part; reconcile
            // by appending the missing suffix if our accumulator lags behind.
            if let Some(text) = event.get("text").and_then(|v| v.as_str())
                && let Some(index) = state.current_reasoning_part
                && let Some(LlmOutputPart::Reasoning { text: existing, .. }) =
                    state.parts.get(index)
            {
                let existing = existing.clone();
                if text != existing
                    && let Some(suffix) = text.strip_prefix(existing.as_str())
                {
                    state.push_reasoning_delta(suffix);
                }
            }
        }
        "response.reasoning_summary_part.done" => state.finish_reasoning_part(),
        "response.output_text.delta" => {
            if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                state.push_text_delta(delta);
            }
        }
        "response.output_text.done" => {}
        "response.function_call_arguments.delta" => {
            if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str())
                && let Some(delta) = event.get("delta").and_then(|v| v.as_str())
            {
                state.push_tool_call_delta(item_id, delta);
            }
        }
        "response.function_call_arguments.done" => {
            if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str())
                && let Some(arguments) = event.get("arguments").and_then(|v| v.as_str())
            {
                state.set_tool_call_arguments(item_id, arguments);
            }
        }
        "response.output_item.done" => {
            if let Some(item) = event.get("item") {
                match item.get("type").and_then(|v| v.as_str()) {
                    Some("message") => state.finish_message(Some(item)),
                    Some("reasoning") => {
                        state.finish_reasoning_part();
                        state.finalize_reasoning_item(item);
                    }
                    Some("function_call") => {
                        let part = state.finish_tool_call(item);
                        if let (Some(parts), Some(part)) = (emitted_parts, part) {
                            parts.push(part);
                        }
                    }
                    _ => {}
                }
            }
        }
        "response.completed" => {
            if let Some(resp_value) = event.get("response") {
                state.merge_final_response(resp_value);
            }
        }
        "response.failed" => {
            let error_value = event
                .get("response")
                .and_then(|r| r.get("error"))
                .or_else(|| event.get("error"))
                .cloned()
                .unwrap_or(Value::Null);
            return Err(LlmTransportError::new(error_message_from_response_failed(
                provider, &event,
            ))
            .retryable(responses_error_is_retryable(&error_value))
            .with_raw(event.to_string()));
        }
        _ => {}
    }
    Ok(())
}

/// Parse a buffered SSE payload (multiple `data:`-prefixed events separated by
/// blank lines) into `state`.
pub fn parse_sse_payload(
    provider: &str,
    payload: &str,
    state: &mut ResponsesStreamState,
) -> Result<(), LlmTransportError> {
    let mut event_lines: Vec<String> = Vec::new();
    for mut line in payload.lines().map(str::to_string) {
        if line.ends_with('\r') {
            line.pop();
        }
        if let Some(data) = line.strip_prefix("data:") {
            event_lines.push(data.trim().to_string());
            continue;
        }
        if line.starts_with("event:") {
            continue;
        }
        if line.trim().is_empty() && !event_lines.is_empty() {
            let raw = event_lines.join("\n");
            process_sse_event(provider, &raw, state, None)?;
            event_lines.clear();
        }
    }
    if !event_lines.is_empty() {
        let raw = event_lines.join("\n");
        process_sse_event(provider, &raw, state, None)?;
    }
    Ok(())
}
