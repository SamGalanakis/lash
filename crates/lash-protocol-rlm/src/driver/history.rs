//! RLM history rendering: turns the chronological turn view into the LLM
//! message sequence the model sees each iteration.
//!
//! Contract:
//! - **Message roles.** Each prior step is a chat message, not one packed
//!   blob. User messages render as `User`; prior RLM steps and protocol events
//!   render as `Assistant` (see `borrowed_history_entry_llm_role`).
//! - **Cache fence.** The last history message is marked with a
//!   `cache_breakpoint` (`mark_last_history_text_cache_breakpoint`) so the
//!   provider can reuse the stable history prefix across iterations. Only the
//!   volatile `=== CURRENT ITERATION ===` tail — iteration number, turn
//!   events, finalization, required-output schema, context budget — is appended
//!   uncached by `append_current_iteration_message`.
//! - **Re-fetch handle.** Outputs longer than `max_output_chars` are
//!   head/tail truncated and tagged with `full: history[N].output[M]`
//!   (`truncated_ref`). That handle resolves: the `history` projected binding
//!   carries the full untruncated value, so the model can recover it by
//!   re-printing the reference. Proven by
//!   `projection::context::tests::history_step_output_resolves_full_untruncated_value`.
//! - **Variables.** The live variable namespace is surfaced separately as the
//!   "Bound Variables" prompt contribution (`crate::rlm_support::render_bound_variables`),
//!   not here — history carries the trajectory, not current globals.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::sync::Arc;

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
    pub(super) turn_causes: &'a [lash_core::TurnCause],
    pub(super) max_output_chars: usize,
    pub(super) protocol_iteration: usize,
    pub(super) finalization: &'a str,
    pub(super) required_output: Option<&'a str>,
    pub(super) final_answer_format: Option<&'a str>,
    pub(super) budget_suffix: Option<&'a str>,
}

#[derive(Clone, Copy)]
pub(super) struct CurrentIterationMessageInput<'a> {
    pub(super) saw_history: bool,
    pub(super) protocol_iteration: usize,
    pub(super) turn_causes: &'a [lash_core::TurnCause],
    pub(super) finalization: &'a str,
    pub(super) required_output: Option<&'a str>,
    pub(super) final_answer_format: Option<&'a str>,
    pub(super) budget_suffix: Option<&'a str>,
}

#[cfg(test)]
pub(super) struct RlmHistoryTestRenderInput<'a> {
    pub(super) projection: &'a lash_core::ChronologicalProjection,
    pub(super) max_output_chars: usize,
    pub(super) protocol_iteration: usize,
    pub(super) finalization: &'a str,
    pub(super) required_output: Option<&'a str>,
    pub(super) final_answer_format: Option<&'a str>,
    pub(super) budget_suffix: Option<&'a str>,
}

pub(super) fn build_rlm_history_messages_from_turn(
    input: RlmHistoryRenderInput<'_>,
    attachments: &mut Vec<LlmAttachment>,
) -> Vec<LlmMessage> {
    let mut messages = Vec::new();
    let mut saw_history = false;
    let active_cause_ids = input
        .turn_causes
        .iter()
        .map(|cause| cause.id.as_str())
        .collect::<HashSet<_>>();
    lash_core::visit_turn_view(input.events, input.turn_messages, |entry| {
        if borrowed_entry_is_active_cause(entry, &active_cause_ids) {
            return;
        }
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
        CurrentIterationMessageInput {
            saw_history,
            protocol_iteration: input.protocol_iteration,
            turn_causes: input.turn_causes,
            finalization: input.finalization,
            required_output: input.required_output,
            final_answer_format: input.final_answer_format,
            budget_suffix: input.budget_suffix,
        },
    );
    messages
}

#[cfg(test)]
pub(super) fn build_rlm_history_messages(
    input: RlmHistoryTestRenderInput<'_>,
    attachments: &mut Vec<LlmAttachment>,
) -> Vec<LlmMessage> {
    let mut messages = Vec::new();

    if !input.projection.entries().is_empty() {
        for entry in input.projection.entries() {
            let mut blocks = vec![text_block(
                render_history_entry(entry, input.max_output_chars),
                false,
            )];
            append_entry_image_blocks(entry, attachments, &mut blocks);
            messages.push(LlmMessage::new(history_entry_llm_role(entry), blocks));
        }
    }
    append_current_iteration_message(
        &mut messages,
        CurrentIterationMessageInput {
            saw_history: !input.projection.entries().is_empty(),
            protocol_iteration: input.protocol_iteration,
            turn_causes: &[],
            finalization: input.finalization,
            required_output: input.required_output,
            final_answer_format: input.final_answer_format,
            budget_suffix: input.budget_suffix,
        },
    );
    messages
}

fn append_current_iteration_message(
    messages: &mut Vec<LlmMessage>,
    input: CurrentIterationMessageInput<'_>,
) {
    if !input.saw_history {
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
        "\n\n\n=== CURRENT ITERATION: {} ===",
        input.protocol_iteration
    );
    if let Some(turn_events) = lash_core::render_turn_causes_prompt(input.turn_causes) {
        current_prompt.push_str("\n\n");
        current_prompt.push_str(&turn_events);
    }
    current_prompt.push_str("\n\n\n=== FINALIZATION ===\n\n");
    current_prompt.push_str(input.finalization);
    if let Some(block) = input.required_output {
        current_prompt.push_str("\n\n=== REQUIRED OUTPUT ===\n\n");
        current_prompt.push_str(block);
    }
    if let Some(guidance) = input.final_answer_format {
        current_prompt.push_str("\n\n=== FINAL ANSWER FORMAT ===\n\n");
        current_prompt.push_str(guidance);
    }
    if let Some(suffix) = input.budget_suffix {
        current_prompt.push_str("\n\n=== CONTEXT BUDGET ===\n\n");
        current_prompt.push_str(suffix);
    }
    messages.push(LlmMessage::new(
        LlmRole::User,
        vec![text_block(current_prompt, false)],
    ));
}

fn borrowed_entry_is_active_cause(
    entry: BorrowedChronologicalEntry<'_>,
    active_cause_ids: &HashSet<&str>,
) -> bool {
    matches!(
        entry.payload,
        BorrowedChronologicalPayload::Message(message)
            if matches!(message.role, lash_core::MessageRole::Event)
                && active_cause_ids.contains(message.id)
    )
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
            lash_core::MessageRole::Event => LlmRole::User,
        },
        ChronologicalPayload::ProtocolEvent(_) => LlmRole::Assistant,
    }
}

fn borrowed_history_entry_llm_role(entry: BorrowedChronologicalEntry<'_>) -> LlmRole {
    match entry.payload {
        BorrowedChronologicalPayload::Message(message) => match message.role {
            lash_core::MessageRole::User => LlmRole::User,
            lash_core::MessageRole::Assistant => LlmRole::Assistant,
            lash_core::MessageRole::System => LlmRole::System,
            lash_core::MessageRole::Event => LlmRole::User,
        },
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
        RlmHistoryItem::RlmStep {
            id: _,
            protocol_iteration,
            reasoning,
            code,
            output,
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
        lash_core::MessageRole::Event => RlmHistoryRole::Event,
    }
}

fn message_role_label(role: &RlmHistoryRole) -> &'static str {
    match role {
        RlmHistoryRole::User => "User",
        RlmHistoryRole::Assistant => "Assistant",
        RlmHistoryRole::System => "System",
        RlmHistoryRole::Event => "Event",
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

/// Strip the executed ` ```lashlang ` block out of the reasoning
/// preview — the code already renders verbatim in the dedicated
/// "Code:" section below, so leaving it in the reasoning duplicates it.
///
/// Routes through the canonical [`crate::fence_scan`] scanner so this
/// strips *exactly* the fence the driver extracts and executes. The
/// older bespoke copy here only inspected the first backtick run and
/// recognized a divergent alias set (`rlm`/`lash`); a reasoning string
/// that opened with some other fenced sample (e.g. a ` ```python `
/// illustration) before the real ` ```lashlang ` block slipped past it
/// entirely, leaking the executed code into the reasoning preview.
fn reasoning_without_first_fence(text: &str) -> String {
    let Some(span) = crate::fence_scan::first_lashlang_fence_span(text) else {
        return text.to_string();
    };
    let after_close = (span.body_end + span.close_len).min(text.len());
    let mut out = String::new();
    out.push_str(text[..span.open_start].trim_end());
    let tail = text[after_close..].trim_start();
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(tail);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::reasoning_without_first_fence;

    #[test]
    fn strips_the_canonical_lashlang_block() {
        let reasoning = "I'll compute it.\n\n```lashlang\nsubmit 1\n```";
        assert_eq!(
            reasoning_without_first_fence(reasoning),
            "I'll compute it.",
            "the executed block is rendered separately and must not be duplicated"
        );
    }

    #[test]
    fn strips_lashlang_block_after_a_non_lashlang_sample() {
        // Regression: the reasoning leads with an illustrative
        // ```python sample, then the real ```lashlang block. The old
        // stripper only inspected the first backtick run, saw a
        // non-lashlang tag, and returned the text unchanged — leaking
        // the executed code into the reasoning preview (where it then
        // duplicated the dedicated "Code:" section). The unified
        // scanner skips the sample and strips the real block.
        let reasoning = "For example:\n\n```python\nx = 1\n```\n\nNow the real one:\n\n```lashlang\nsubmit x\n```";
        let stripped = reasoning_without_first_fence(reasoning);
        assert!(
            !stripped.contains("submit x"),
            "the executed lashlang block must be stripped, got: {stripped:?}"
        );
        assert!(
            stripped.contains("```python"),
            "the non-lashlang sample is prose and stays in the reasoning"
        );
    }

    #[test]
    fn leaves_reasoning_without_a_lashlang_fence_untouched() {
        let reasoning = "Just thinking out loud, no code here.";
        assert_eq!(reasoning_without_first_fence(reasoning), reasoning);
    }
}
