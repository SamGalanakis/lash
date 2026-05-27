use std::{fmt::Write as _, sync::Arc};

use lash_core::llm::types::{LlmAttachment, LlmContentBlock, LlmMessage, LlmRole};
use lash_core::{BorrowedChronologicalEntry, BorrowedChronologicalPayload, head_tail_truncate};
#[cfg(test)]
use lash_core::{ChronologicalEntry, ChronologicalPayload};
#[cfg(test)]
use lash_rlm_types::RlmHistoryItem;
use lash_rlm_types::{RlmAttachmentRef, RlmHistoryRole, RlmImageRef};

use crate::projection::decode_rlm_protocol_event;

pub(super) struct RlmHistoryRenderInput<'a> {
    pub(super) events: &'a [lash_core::SessionEventRecord],
    pub(super) turn_messages: &'a lash_core::MessageSequence,
    pub(super) max_output_chars: usize,
    pub(super) protocol_iteration: usize,
    pub(super) finalization: &'a str,
    pub(super) required_output: Option<&'a str>,
    pub(super) budget_suffix: Option<&'a str>,
}

pub(super) fn build_rlm_history_messages_from_turn(
    input: RlmHistoryRenderInput<'_>,
    attachments: &mut Vec<LlmAttachment>,
) -> Vec<LlmMessage> {
    let mut messages = Vec::new();
    let mut saw_history = false;
    lash_core::visit_turn_view(input.events, input.turn_messages, &[], |entry| {
        saw_history = true;
        let mut blocks = vec![text_block(
            render_borrowed_history_entry(entry, input.max_output_chars),
            false,
        )];
        append_borrowed_entry_image_blocks(entry, attachments, &mut blocks);
        messages.push(LlmMessage::new(
            borrowed_history_entry_llm_role(entry),
            blocks,
        ));
    });
    append_current_iteration_message(
        &mut messages,
        saw_history,
        input.protocol_iteration,
        input.finalization,
        input.required_output,
        input.budget_suffix,
    );
    messages
}

#[cfg(test)]
pub(super) fn build_rlm_history_messages(
    projection: &lash_core::ChronologicalProjection,
    max_output_chars: usize,
    protocol_iteration: usize,
    finalization: &str,
    required_output: Option<&str>,
    budget_suffix: Option<&str>,
    attachments: &mut Vec<LlmAttachment>,
) -> Vec<LlmMessage> {
    let mut messages = Vec::new();

    if !projection.entries().is_empty() {
        for entry in projection.entries() {
            let mut blocks = vec![text_block(
                render_history_entry(entry, max_output_chars),
                false,
            )];
            append_entry_image_blocks(entry, attachments, &mut blocks);
            messages.push(LlmMessage::new(history_entry_llm_role(entry), blocks));
        }
    }
    append_current_iteration_message(
        &mut messages,
        !projection.entries().is_empty(),
        protocol_iteration,
        finalization,
        required_output,
        budget_suffix,
    );
    messages
}

fn append_current_iteration_message(
    messages: &mut Vec<LlmMessage>,
    saw_history: bool,
    protocol_iteration: usize,
    finalization: &str,
    required_output: Option<&str>,
    budget_suffix: Option<&str>,
) {
    if !saw_history {
        messages.push(LlmMessage::new(
            LlmRole::User,
            vec![text_block(
                "=== HISTORY ===\n\nNo chronological history is available.",
                false,
            )],
        ));
    } else {
        mark_last_history_text_cache_breakpoint(messages);
    }
    let mut current_prompt = format!(
        "\n\n\n=== CURRENT ITERATION: {protocol_iteration} ===\n\n\n=== FINALIZATION ===\n\n{finalization}",
    );
    if let Some(block) = required_output {
        current_prompt.push_str("\n\n=== REQUIRED OUTPUT ===\n\n");
        current_prompt.push_str(block);
    }
    if let Some(suffix) = budget_suffix {
        current_prompt.push_str("\n\n=== CONTEXT BUDGET ===\n\n");
        current_prompt.push_str(suffix);
    }
    messages.push(LlmMessage::new(
        LlmRole::User,
        vec![text_block(current_prompt, false)],
    ));
}

fn text_block(text: impl Into<Arc<str>>, cache_breakpoint: bool) -> LlmContentBlock {
    LlmContentBlock::Text {
        text: text.into(),
        response_meta: None,
        cache_breakpoint,
    }
}

#[cfg(test)]
fn history_entry_llm_role(entry: &ChronologicalEntry) -> LlmRole {
    match &entry.payload {
        ChronologicalPayload::Message(message) => match message.role {
            lash_core::MessageRole::User => LlmRole::User,
            lash_core::MessageRole::Assistant => LlmRole::Assistant,
            lash_core::MessageRole::System => LlmRole::System,
        },
        ChronologicalPayload::ToolCall(_) => LlmRole::User,
        ChronologicalPayload::ProtocolEvent(_) => LlmRole::Assistant,
    }
}

fn borrowed_history_entry_llm_role(entry: BorrowedChronologicalEntry<'_>) -> LlmRole {
    match entry.payload {
        BorrowedChronologicalPayload::Message(message) => match message.role {
            lash_core::MessageRole::User => LlmRole::User,
            lash_core::MessageRole::Assistant => LlmRole::Assistant,
            lash_core::MessageRole::System => LlmRole::System,
        },
        BorrowedChronologicalPayload::ToolCall(_) => LlmRole::User,
        BorrowedChronologicalPayload::ProtocolEvent(_) => LlmRole::Assistant,
    }
}

fn mark_last_history_text_cache_breakpoint(messages: &mut [LlmMessage]) {
    for message in messages.iter_mut().rev() {
        let Some(blocks) = Arc::get_mut(&mut message.blocks) else {
            continue;
        };
        for block in blocks.iter_mut().rev() {
            if let LlmContentBlock::Text {
                text,
                cache_breakpoint,
                ..
            } = block
                && !text.trim().is_empty()
            {
                *cache_breakpoint = true;
                return;
            }
        }
    }
}

#[cfg(test)]
pub(super) fn render_history_prompt(history: &[RlmHistoryItem], max_output_chars: usize) -> String {
    if history.is_empty() {
        return "No chronological history is available.".to_string();
    }
    let mut rendered = String::new();
    for (index, item) in history.iter().enumerate() {
        if !rendered.is_empty() {
            rendered.push_str("\n\n");
        }
        rendered.push_str(&render_history_item(index, item, max_output_chars));
    }
    rendered
}

#[cfg(test)]
fn render_history_item(index: usize, item: &RlmHistoryItem, max_output_chars: usize) -> String {
    let mut rendered = String::new();
    match item {
        RlmHistoryItem::Message {
            id: _,
            role,
            content,
            attachments,
        } => append_history_message(
            &mut rendered,
            index,
            role,
            content,
            attachments,
            max_output_chars,
        ),
        RlmHistoryItem::ToolCall {
            id: _,
            tool,
            args,
            output,
            duration_ms,
        } => append_history_tool_call(
            &mut rendered,
            HistoryToolCallRender {
                index,
                tool,
                args,
                result: output.value_for_projection(),
                status: match output.status() {
                    lash_core::ToolCallStatus::Success => "ok",
                    lash_core::ToolCallStatus::Failure => "error",
                    lash_core::ToolCallStatus::Cancelled => "cancelled",
                },
                duration_ms: *duration_ms,
                max_output_chars,
            },
        ),
        RlmHistoryItem::RlmStep {
            id: _,
            protocol_iteration,
            reasoning,
            code,
            output,
            tool_call_ids: _,
            images,
            error,
            final_output,
        } => append_repl_step(
            &mut rendered,
            ReplStepRender {
                index,
                protocol_iteration: *protocol_iteration,
                reasoning,
                code,
                output,
                images,
                error: error.as_deref(),
                final_output: final_output.as_ref(),
                max_output_chars,
            },
        ),
    }
    rendered
}

#[cfg(test)]
fn render_history_entry(entry: &ChronologicalEntry, max_output_chars: usize) -> String {
    let mut rendered = String::new();
    match &entry.payload {
        ChronologicalPayload::Message(message) => {
            let content = message_history_text(message);
            let attachments = message
                .parts
                .iter()
                .filter_map(|part| {
                    let attachment = part.attachment.as_ref()?;
                    Some(RlmAttachmentRef {
                        id: part.id.clone(),
                        media_type: attachment.reference.media_type,
                        label: attachment.reference.label.clone(),
                        reference: attachment.reference.id.to_string(),
                    })
                })
                .collect::<Vec<_>>();
            append_history_message(
                &mut rendered,
                entry.index,
                &history_role(message.role),
                &content,
                &attachments,
                max_output_chars,
            );
        }
        ChronologicalPayload::ToolCall(record) => append_history_tool_call(
            &mut rendered,
            HistoryToolCallRender {
                index: entry.index,
                tool: &record.tool,
                args: &record.args,
                result: record.output.value_for_projection(),
                status: match record.output.status() {
                    lash_core::ToolCallStatus::Success => "ok",
                    lash_core::ToolCallStatus::Failure => "error",
                    lash_core::ToolCallStatus::Cancelled => "cancelled",
                },
                duration_ms: record.duration_ms,
                max_output_chars,
            },
        ),
        ChronologicalPayload::ProtocolEvent(event) => {
            let Some(lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(step)) =
                decode_rlm_protocol_event(event)
            else {
                return rendered;
            };
            let images = step
                .images
                .iter()
                .map(|image| RlmImageRef {
                    id: image.id.to_string(),
                    media_type: image.media_type,
                    width: image.width,
                    height: image.height,
                    bytes: image.byte_len as usize,
                    label: image.label.clone(),
                })
                .collect::<Vec<_>>();
            append_repl_step(
                &mut rendered,
                ReplStepRender {
                    index: entry.index,
                    protocol_iteration: step.protocol_iteration,
                    reasoning: &step.reasoning,
                    code: &step.code,
                    output: &step.output,
                    images: &images,
                    error: step.error.as_deref(),
                    final_output: step.final_output.as_ref(),
                    max_output_chars,
                },
            );
        }
    }
    rendered
}

fn render_borrowed_history_entry(
    entry: BorrowedChronologicalEntry<'_>,
    max_output_chars: usize,
) -> String {
    let mut rendered = String::new();
    match entry.payload {
        BorrowedChronologicalPayload::Message(message) => {
            let content = message_history_text_parts(message.parts);
            let attachments = message
                .parts
                .iter()
                .filter_map(|part| {
                    let attachment = part.attachment.as_ref()?;
                    Some(RlmAttachmentRef {
                        id: part.id.clone(),
                        media_type: attachment.reference.media_type,
                        label: attachment.reference.label.clone(),
                        reference: attachment.reference.id.to_string(),
                    })
                })
                .collect::<Vec<_>>();
            append_history_message(
                &mut rendered,
                entry.index,
                &history_role(message.role),
                &content,
                &attachments,
                max_output_chars,
            );
        }
        BorrowedChronologicalPayload::ToolCall(record) => append_history_tool_call(
            &mut rendered,
            HistoryToolCallRender {
                index: entry.index,
                tool: &record.tool,
                args: &record.args,
                result: record.output.value_for_projection(),
                status: match record.output.status() {
                    lash_core::ToolCallStatus::Success => "ok",
                    lash_core::ToolCallStatus::Failure => "error",
                    lash_core::ToolCallStatus::Cancelled => "cancelled",
                },
                duration_ms: record.duration_ms,
                max_output_chars,
            },
        ),
        BorrowedChronologicalPayload::ProtocolEvent(event) => {
            let Some(lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(step)) =
                decode_rlm_protocol_event(event)
            else {
                return rendered;
            };
            let images = step
                .images
                .iter()
                .map(|image| RlmImageRef {
                    id: image.id.to_string(),
                    media_type: image.media_type,
                    width: image.width,
                    height: image.height,
                    bytes: image.byte_len as usize,
                    label: image.label.clone(),
                })
                .collect::<Vec<_>>();
            append_repl_step(
                &mut rendered,
                ReplStepRender {
                    index: entry.index,
                    protocol_iteration: step.protocol_iteration,
                    reasoning: &step.reasoning,
                    code: &step.code,
                    output: &step.output,
                    images: &images,
                    error: step.error.as_deref(),
                    final_output: step.final_output.as_ref(),
                    max_output_chars,
                },
            );
        }
    }
    rendered
}

struct HistoryToolCallRender<'a> {
    index: usize,
    tool: &'a str,
    args: &'a serde_json::Value,
    result: serde_json::Value,
    status: &'static str,
    duration_ms: u64,
    max_output_chars: usize,
}

fn append_history_tool_call(out: &mut String, call: HistoryToolCallRender<'_>) {
    let HistoryToolCallRender {
        index,
        tool,
        args,
        result,
        status,
        duration_ms,
        max_output_chars,
    } = call;
    let args = serde_json::to_string_pretty(args).unwrap_or_else(|_| args.to_string());
    let result = serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string());
    let (args_preview, args_raw_len) = head_tail_truncate(&args, max_output_chars);
    let (result_preview, result_raw_len) = head_tail_truncate(&result, max_output_chars);
    let args_ref = truncated_ref(
        args_raw_len,
        max_output_chars,
        &format!("history[{index}].args"),
    );
    let result_ref = truncated_ref(
        result_raw_len,
        max_output_chars,
        &format!("history[{index}].result"),
    );
    let _ = write!(
        out,
        "--- history[{index}] · tool_call · {tool} · {status} · {duration_ms} ms ---\n\nArguments ({args_raw_len} chars{args_ref}):\n{args_preview}\n\nResult ({result_raw_len} chars{result_ref}):\n{result_preview}"
    );
}

#[cfg(test)]
pub(super) fn append_entry_image_blocks(
    entry: &lash_core::ChronologicalEntry,
    attachments: &mut Vec<LlmAttachment>,
    blocks: &mut Vec<LlmContentBlock>,
) {
    match &entry.payload {
        lash_core::ChronologicalPayload::Message(message) => {
            for part in message.parts.iter() {
                let Some(attachment) = part.attachment.as_ref() else {
                    continue;
                };
                let attachment_idx = attachments.len();
                attachments.push(LlmAttachment::reference(attachment.reference.clone()));
                blocks.push(LlmContentBlock::Image { attachment_idx });
            }
        }
        lash_core::ChronologicalPayload::ProtocolEvent(event) => {
            if let Some(lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(entry)) =
                decode_rlm_protocol_event(event)
            {
                for image in &entry.images {
                    let attachment_idx = attachments.len();
                    attachments.push(LlmAttachment::reference(image.clone()));
                    blocks.push(LlmContentBlock::Image { attachment_idx });
                }
            }
        }
        lash_core::ChronologicalPayload::ToolCall(_) => {}
    }
}

fn append_borrowed_entry_image_blocks(
    entry: BorrowedChronologicalEntry<'_>,
    attachments: &mut Vec<LlmAttachment>,
    blocks: &mut Vec<LlmContentBlock>,
) {
    match entry.payload {
        BorrowedChronologicalPayload::Message(message) => {
            for part in message.parts {
                let Some(attachment) = part.attachment.as_ref() else {
                    continue;
                };
                let attachment_idx = attachments.len();
                attachments.push(LlmAttachment::reference(attachment.reference.clone()));
                blocks.push(LlmContentBlock::Image { attachment_idx });
            }
        }
        BorrowedChronologicalPayload::ProtocolEvent(event) => {
            if let Some(lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(entry)) =
                decode_rlm_protocol_event(event)
            {
                for image in &entry.images {
                    let attachment_idx = attachments.len();
                    attachments.push(LlmAttachment::reference(image.clone()));
                    blocks.push(LlmContentBlock::Image { attachment_idx });
                }
            }
        }
        BorrowedChronologicalPayload::ToolCall(_) => {}
    }
}

fn append_history_message(
    out: &mut String,
    index: usize,
    role: &RlmHistoryRole,
    content: &str,
    attachments: &[RlmAttachmentRef],
    max_output_chars: usize,
) {
    let (preview, raw_len) = head_tail_truncate(content, max_output_chars);
    let full_ref = truncated_ref(
        raw_len,
        max_output_chars,
        &format!("history[{index}].content"),
    );
    let _ = write!(
        out,
        "--- history[{index}] · {} message · {raw_len} chars{full_ref} ---\n\n{preview}",
        message_role_label(role).to_lowercase(),
    );
    if !attachments.is_empty() {
        out.push_str("\n\nAttachments:");
        for (attachment_index, attachment) in attachments.iter().enumerate() {
            let rendered = serde_json::to_string(attachment)
                .unwrap_or_else(|_| "{\"error\":\"unrenderable attachment\"}".to_string());
            let _ = write!(
                out,
                "\n- history[{index}].attachments[{attachment_index}]: {rendered}"
            );
        }
    }
}

#[cfg(test)]
fn message_history_text(message: &lash_core::Message) -> String {
    message_history_text_parts(message.parts.as_slice())
}

fn message_history_text_parts(parts: &[lash_core::Part]) -> String {
    let chunks = parts
        .iter()
        .filter(|part| {
            matches!(
                part.kind,
                lash_core::PartKind::Text | lash_core::PartKind::Prose
            )
        })
        .map(|part| part.content.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    chunks.join("\n\n")
}

fn history_role(role: lash_core::MessageRole) -> RlmHistoryRole {
    match role {
        lash_core::MessageRole::User => RlmHistoryRole::User,
        lash_core::MessageRole::System => RlmHistoryRole::System,
        lash_core::MessageRole::Assistant => RlmHistoryRole::Assistant,
    }
}

fn message_role_label(role: &RlmHistoryRole) -> &'static str {
    match role {
        RlmHistoryRole::User => "User",
        RlmHistoryRole::Assistant => "Assistant",
        RlmHistoryRole::System => "System",
    }
}

struct ReplStepRender<'a> {
    index: usize,
    protocol_iteration: usize,
    reasoning: &'a str,
    code: &'a str,
    output: &'a [String],
    images: &'a [RlmImageRef],
    error: Option<&'a str>,
    final_output: Option<&'a serde_json::Value>,
    max_output_chars: usize,
}

fn append_repl_step(out: &mut String, step: ReplStepRender<'_>) {
    let ReplStepRender {
        index,
        protocol_iteration,
        reasoning,
        code,
        output,
        images,
        error,
        final_output,
        max_output_chars,
    } = step;
    let reasoning = reasoning_without_first_fence(reasoning).trim().to_string();
    let (reasoning_preview, reasoning_raw_len) = head_tail_truncate(&reasoning, max_output_chars);
    let reasoning_ref = truncated_ref(
        reasoning_raw_len,
        max_output_chars,
        &format!("history[{index}].reasoning"),
    );
    let _ = write!(
        out,
        "--- history[{index}] · rlm step · protocol_iteration {protocol_iteration} ---\n\nReasoning ({reasoning_raw_len} chars{reasoning_ref}):\n{}\n\nCode:\n```lashlang\n{}\n```",
        if reasoning_preview.is_empty() {
            "(none)"
        } else {
            &reasoning_preview
        },
        code.trim(),
    );

    // One block per `print` (or raw stdout emission). Tool calls used to
    // get their own section here for retrieval-without-print, but that
    // duplicated content the model could fetch via `print result` and
    // bloated history; the `code` field above shows every receiver
    // operation the model wrote.
    for (output_index, item) in output.iter().enumerate() {
        let (preview, raw_len) = head_tail_truncate(item, max_output_chars);
        let full_ref = truncated_ref(
            raw_len,
            max_output_chars,
            &format!("history[{index}].output[{output_index}]"),
        );
        let _ = write!(
            out,
            "\n\nhistory[{index}].output[{output_index}] ({raw_len} chars{full_ref}):\n{preview}"
        );
    }

    if !images.is_empty() {
        out.push_str("\n\nImages:");
        for (image_index, image) in images.iter().enumerate() {
            let rendered = serde_json::to_string(image)
                .unwrap_or_else(|_| "{\"error\":\"unrenderable image\"}".to_string());
            let _ = write!(
                out,
                "\n- history[{index}].images[{image_index}]: {rendered}"
            );
        }
    }

    if let Some(error) = error {
        out.push_str("\n\nError:\n");
        out.push_str(error);
    }
    if let Some(final_output) = final_output {
        out.push_str("\n\nFinal output:\n");
        out.push_str(
            &serde_json::to_string_pretty(final_output)
                .unwrap_or_else(|_| final_output.to_string()),
        );
    }
}

fn truncated_ref(raw_len: usize, max_output_chars: usize, reference: &str) -> String {
    if raw_len > max_output_chars {
        format!(", full: {reference}")
    } else {
        String::new()
    }
}

fn reasoning_without_first_fence(text: &str) -> String {
    let Some(open_rel) = text.find("```") else {
        return text.to_string();
    };
    // CommonMark variable-length fences: count opener backticks; the
    // closer must be a run of ≥N backticks. Mirrors the runtime
    // extractor in `protocol.rs::first_lashlang_fence_span`.
    let opener_len = text.as_bytes()[open_rel..]
        .iter()
        .take_while(|&&b| b == b'`')
        .count();
    let after_open = open_rel + opener_len;
    let rest = &text[after_open..];
    let Some(lang_end_rel) = rest.find('\n') else {
        return text[..open_rel].to_string();
    };
    let lang = rest[..lang_end_rel].trim();
    if !matches!(lang, "lashlang" | "rlm" | "lash") {
        return text.to_string();
    }
    let body_start = after_open + lang_end_rel + 1;
    let body_bytes = &text.as_bytes()[body_start..];
    let mut close = text.len();
    let mut consumed = 0usize;
    let mut i = 0;
    while i < body_bytes.len() {
        if body_bytes[i] == b'`' {
            let start = i;
            while i < body_bytes.len() && body_bytes[i] == b'`' {
                i += 1;
            }
            if i - start >= opener_len {
                close = body_start + start;
                consumed = opener_len;
                break;
            }
        } else {
            i += 1;
        }
    }
    let after_close = (close + consumed).min(text.len());
    let mut out = String::new();
    out.push_str(text[..open_rel].trim_end());
    let tail = text[after_close..].trim_start();
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(tail);
    }
    out
}
