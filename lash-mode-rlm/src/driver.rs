use std::sync::RwLock;
use std::{fmt::Write as _, sync::Arc};

use lash::llm::types::{LlmAttachment, LlmContentBlock, LlmMessage, LlmRole, LlmToolChoice};
use lash::sansio::ContextProjector;
use lash::{
    ChronologicalEntry, ChronologicalPayload, LlmRequest, ModeBuildInput, ModeConfig, ModePreamble,
    ProjectorContext, PromptContribution, PromptUsage, head_tail_truncate,
};
#[cfg(test)]
use lash_rlm_types::RlmHistoryItem;
use lash_rlm_types::{RlmAttachmentRef, RlmHistoryRole, RlmImageRef, RlmTermination};

/// Cell shared between the RLM mode plugin's turn-prepare hook (writer)
/// and the projector (reader). The plugin's hook captures `prompt_usage`
/// from `TurnTransformContext` each turn and stores it here so the
/// projector can render the budget suffix into the volatile turn-tail
/// message — keeping the cached system prefix byte-stable.
pub type SharedPromptUsage = Arc<RwLock<Option<PromptUsage>>>;

#[derive(Clone)]
pub struct RlmProjectorConfig {
    pub max_output_chars: usize,
    pub max_budget_tokens: Option<usize>,
    pub last_prompt_usage: SharedPromptUsage,
    pub prompt_features: crate::protocol::RlmPromptFeatures,
}

impl Default for RlmProjectorConfig {
    fn default() -> Self {
        Self {
            max_output_chars: 10_000,
            max_budget_tokens: None,
            last_prompt_usage: Arc::new(RwLock::new(None)),
            prompt_features: crate::protocol::RlmPromptFeatures::default(),
        }
    }
}

pub fn build_rlm_preamble(input: ModeBuildInput, config: RlmProjectorConfig) -> ModePreamble {
    let tool_surface = (*input.tool_surface).clone();
    let omitted_tool_count = tool_surface.omitted_tool_count();
    let mut prompt_contributions = Vec::new();

    let tool_docs = tool_surface.prompt_tool_docs();
    if !tool_docs.trim().is_empty() {
        prompt_contributions.push(PromptContribution::execution("Showcased Tools", tool_docs));
    }
    prompt_contributions.extend(input.extra_prompt_contributions);

    ModePreamble {
        config: ModeConfig {
            protocol: Arc::new(crate::protocol::RlmDriver),
            projector: Arc::new(RlmContextProjector {
                max_output_chars: config.max_output_chars,
                max_budget_tokens: config.max_budget_tokens,
                last_prompt_usage: config.last_prompt_usage,
            }),
            sync_execution_surface: true,
        },
        tool_specs: Arc::new(Vec::new()),
        tool_names: tool_surface.tool_names(),
        omitted_tool_count,
        execution_prompt: crate::protocol::rlm_execution_section_with_features(
            config.prompt_features,
        ),
        prompt_contributions,
    }
}

#[cfg(test)]
mod catalogue_tests {
    use super::*;
    use lash::{ExecutionMode, ToolActivation, ToolAvailabilityConfig, ToolExecutionMode};

    fn tool(name: &str) -> lash::ToolDefinition {
        lash::ToolDefinition::new(
            name,
            format!("Tool {name}"),
            serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
            serde_json::json!({ "type": "string" }),
        )
        .with_availability(ToolAvailabilityConfig::documented())
        .with_activation(ToolActivation::Always)
        .with_execution_mode(ToolExecutionMode::Parallel)
    }

    #[test]
    fn rlm_preamble_uses_resolved_tool_surface_without_search_tool_special_cases() {
        let surface = lash::ToolSurface::from_tools(
            vec![tool("search_tools"), tool("grep")],
            ExecutionMode::new("test"),
        );

        let preamble = build_rlm_preamble(
            lash::ModeBuildInput {
                mode: ExecutionMode::new("test"),
                tool_surface: Arc::new(surface),
                extra_prompt_contributions: Vec::new(),
            },
            RlmProjectorConfig::default(),
        );

        assert_eq!(preamble.omitted_tool_count, 0);
        assert_eq!(preamble.tool_names, vec!["search_tools", "grep"]);
        let prompt = preamble
            .prompt_contributions
            .iter()
            .map(|contribution| contribution.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(prompt.contains("search_tools"));
        assert!(prompt.contains("grep"));
    }

    #[test]
    fn finish_finalization_prompt_defaults_to_submit_guidance() {
        let prompt = rlm_finalization_prompt(&RlmTermination::default());

        assert!(prompt.contains("submit <value>"));
    }

    #[test]
    fn prose_or_submit_finalization_prompt_allows_direct_prose() {
        let prompt = rlm_finalization_prompt(&RlmTermination::ProseOrSubmit);

        assert!(prompt.contains("Either finish your turn with prose only"));
        assert!(prompt.contains("or use `submit` in lashlang"));
        assert!(prompt.contains("Do not duplicate"));
    }
}

struct RlmContextProjector {
    max_output_chars: usize,
    max_budget_tokens: Option<usize>,
    last_prompt_usage: SharedPromptUsage,
}

fn rlm_termination(options: &lash::ModeTurnOptions) -> RlmTermination {
    options
        .decode(&lash::ExecutionMode::new("rlm"))
        .ok()
        .flatten()
        .unwrap_or_default()
}

impl ContextProjector<lash::HostModeProtocol> for RlmContextProjector {
    fn project(&self, ctx: ProjectorContext<'_>) -> LlmRequest {
        let projection = projection_from_projector_context(&ctx);
        let termination = rlm_termination(&ctx.config.termination);
        let finalization = rlm_finalization_prompt(&termination);
        let required_output = required_output_block(&termination);
        let budget_suffix = self.last_prompt_usage.read().ok().and_then(|guard| {
            crate::rlm_support::format_budget_suffix(
                ctx.mode_iteration + 1,
                guard.as_ref(),
                self.max_budget_tokens,
            )
        });

        let mut messages = Vec::new();
        if !ctx.config.system_prompt.trim().is_empty() {
            messages.push(LlmMessage::text(
                LlmRole::System,
                std::sync::Arc::clone(&ctx.config.system_prompt),
            ));
        }
        let mut attachments = Vec::new();
        messages.extend(build_rlm_history_messages(
            &projection,
            self.max_output_chars,
            ctx.mode_iteration + 1,
            finalization,
            required_output.as_deref(),
            budget_suffix.as_deref(),
            &mut attachments,
        ));

        LlmRequest {
            model: ctx.config.model.clone(),
            messages,
            attachments,
            tools: Arc::new(Vec::new()),
            tool_choice: LlmToolChoice::None,
            model_variant: ctx.config.model_variant.clone(),
            session_id: ctx.config.run_session_id.clone(),
            output_spec: None,
            stream_events: None,
            provider_trace: None,
        }
    }
}

fn build_rlm_history_messages(
    projection: &lash::ChronologicalProjection,
    max_output_chars: usize,
    mode_iteration: usize,
    finalization: &str,
    required_output: Option<&str>,
    budget_suffix: Option<&str>,
    attachments: &mut Vec<LlmAttachment>,
) -> Vec<LlmMessage> {
    let mut messages = Vec::new();

    if projection.entries().is_empty() {
        messages.push(LlmMessage::new(
            LlmRole::User,
            vec![text_block(
                "=== HISTORY ===\n\nNo chronological history is available.",
                false,
            )],
        ));
    } else {
        for entry in projection.entries() {
            let mut blocks = vec![text_block(
                render_history_entry(entry, max_output_chars),
                false,
            )];
            append_entry_image_blocks(entry, attachments, &mut blocks);
            messages.push(LlmMessage::new(history_entry_llm_role(entry), blocks));
        }
        mark_last_history_text_cache_breakpoint(&mut messages);
    }

    let mut current_prompt = format!(
        "\n\n\n=== CURRENT ITERATION: {mode_iteration} ===\n\n\n=== FINALIZATION ===\n\n{finalization}",
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
    messages
}

fn required_output_block(termination: &RlmTermination) -> Option<String> {
    match termination {
        RlmTermination::SubmitRequired {
            schema: Some(schema),
        } => Some(render_value_schema_contract(schema)),
        _ => None,
    }
}

fn render_value_schema_contract(schema: &serde_json::Value) -> String {
    let input_contract =
        lash::ToolDefinition::new("submit", "", schema.clone(), serde_json::json!({}))
            .compact_contract();

    if input_contract.parameters.is_empty() {
        return lash::ToolDefinition::new(
            "submit",
            "",
            lash::ToolDefinition::default_input_schema(),
            schema.clone(),
        )
        .compact_contract()
        .returns;
    }

    let head = format!(
        "{{ {} }}",
        input_contract
            .parameters
            .iter()
            .filter_map(|value| value.get("signature").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let lines = input_contract
        .parameters
        .iter()
        .filter_map(compact_doc_line)
        .collect::<Vec<_>>();

    if lines.is_empty() {
        head
    } else {
        format!("{head}\nFields:\n{}", lines.join("\n"))
    }
}

fn compact_doc_line(value: &serde_json::Value) -> Option<String> {
    let signature = value.get("signature")?.as_str()?.trim();
    if signature.is_empty() {
        return None;
    }
    let description = value
        .get("description")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    Some(match description {
        Some(description) => format!("- `{signature}` — {description}"),
        None => format!("- `{signature}`"),
    })
}

fn text_block(text: impl Into<Arc<str>>, cache_breakpoint: bool) -> LlmContentBlock {
    LlmContentBlock::Text {
        text: text.into(),
        response_meta: None,
        cache_breakpoint,
    }
}

fn history_entry_llm_role(entry: &ChronologicalEntry) -> LlmRole {
    match &entry.payload {
        ChronologicalPayload::Message(message) => match message.role {
            lash::MessageRole::User => LlmRole::User,
            lash::MessageRole::Assistant => LlmRole::Assistant,
            lash::MessageRole::System => LlmRole::System,
        },
        ChronologicalPayload::ToolCall(_) => LlmRole::User,
        ChronologicalPayload::ModeEvent(_) => LlmRole::Assistant,
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

fn rlm_finalization_prompt(termination: &RlmTermination) -> &'static str {
    match termination {
        RlmTermination::SubmitRequired { .. } => {
            "The turn must finish through `submit <value>`. Prose alone does not end the turn."
        }
        RlmTermination::ProseOrSubmit => {
            "Either finish your turn with prose only, without a lashlang block, or use `submit` in lashlang. Do not duplicate the submitted answer in prose."
        }
    }
}

impl RlmContextProjector {
    #[cfg(test)]
    fn format_history(&self, projection: &lash::ChronologicalProjection) -> String {
        let history = crate::rlm_history_projection(projection);
        render_history_prompt(history.history(), self.max_output_chars)
    }
}

fn projection_from_projector_context(ctx: &ProjectorContext<'_>) -> lash::ChronologicalProjection {
    let read_view = lash::SessionReadView::from_derived_message_view(
        lash::SessionStateEnvelope::default(),
        Arc::new(ctx.events.to_vec()),
        Arc::new(ctx.messages.iter().cloned().collect()),
        Arc::new(Vec::new()),
    );
    read_view.chronological_projection()
}

#[cfg(test)]
fn projection_from_events(events: &[lash::SessionEventRecord]) -> lash::ChronologicalProjection {
    let read_view = lash::SessionReadView::from_derived_message_view(
        lash::SessionStateEnvelope::default(),
        Arc::new(events.to_vec()),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    read_view.chronological_projection()
}

#[cfg(test)]
fn render_history_prompt(history: &[RlmHistoryItem], max_output_chars: usize) -> String {
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
            result,
            success,
            duration_ms,
        } => append_history_tool_call(
            &mut rendered,
            HistoryToolCallRender {
                index,
                tool,
                args,
                result,
                success: *success,
                duration_ms: *duration_ms,
                max_output_chars,
            },
        ),
        RlmHistoryItem::RlmStep {
            id: _,
            mode_iteration,
            reasoning,
            code,
            output,
            tool_calls: _,
            images,
            error,
            final_output,
        } => append_repl_step(
            &mut rendered,
            ReplStepRender {
                index,
                mode_iteration: *mode_iteration,
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
                result: &record.result,
                success: record.success,
                duration_ms: record.duration_ms,
                max_output_chars,
            },
        ),
        ChronologicalPayload::ModeEvent(event) => {
            let Some(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(step)) =
                crate::decode_rlm_mode_event(event)
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
                    mode_iteration: step.mode_iteration,
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
    result: &'a serde_json::Value,
    success: bool,
    duration_ms: u64,
    max_output_chars: usize,
}

fn append_history_tool_call(out: &mut String, call: HistoryToolCallRender<'_>) {
    let HistoryToolCallRender {
        index,
        tool,
        args,
        result,
        success,
        duration_ms,
        max_output_chars,
    } = call;
    let args = serde_json::to_string_pretty(args).unwrap_or_else(|_| args.to_string());
    let result = serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string());
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
    let status = if success { "ok" } else { "error" };
    let _ = write!(
        out,
        "--- history[{index}] · tool_call · {tool} · {status} · {duration_ms} ms ---\n\nArguments ({args_raw_len} chars{args_ref}):\n{args_preview}\n\nResult ({result_raw_len} chars{result_ref}):\n{result_preview}"
    );
}

fn append_entry_image_blocks(
    entry: &lash::ChronologicalEntry,
    attachments: &mut Vec<LlmAttachment>,
    blocks: &mut Vec<LlmContentBlock>,
) {
    match &entry.payload {
        lash::ChronologicalPayload::Message(message) => {
            for part in message.parts.iter() {
                let Some(attachment) = part.attachment.as_ref() else {
                    continue;
                };
                let attachment_idx = attachments.len();
                attachments.push(LlmAttachment::reference(attachment.reference.clone()));
                blocks.push(LlmContentBlock::Image { attachment_idx });
            }
        }
        lash::ChronologicalPayload::ModeEvent(event) => {
            if let Some(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry)) =
                crate::decode_rlm_mode_event(event)
            {
                for image in &entry.images {
                    let attachment_idx = attachments.len();
                    attachments.push(LlmAttachment::reference(image.clone()));
                    blocks.push(LlmContentBlock::Image { attachment_idx });
                }
            }
        }
        lash::ChronologicalPayload::ToolCall(_) => {}
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

fn message_history_text(message: &lash::Message) -> String {
    let chunks = message
        .parts
        .iter()
        .filter(|part| matches!(part.kind, lash::PartKind::Text | lash::PartKind::Prose))
        .map(|part| part.content.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    chunks.join("\n\n")
}

fn history_role(role: lash::MessageRole) -> RlmHistoryRole {
    match role {
        lash::MessageRole::User => RlmHistoryRole::User,
        lash::MessageRole::System => RlmHistoryRole::System,
        lash::MessageRole::Assistant => RlmHistoryRole::Assistant,
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
    mode_iteration: usize,
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
        mode_iteration,
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
        "--- history[{index}] · rlm step · mode_iteration {mode_iteration} ---\n\nReasoning ({reasoning_raw_len} chars{reasoning_ref}):\n{}\n\nCode:\n```lashlang\n{}\n```",
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
    // bloated history; the `code` field above shows every `(call …)`
    // the model wrote.
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

#[cfg(test)]
mod tests {
    use super::*;
    use lash::session_model::{
        ConversationRecord, MessageRole, Part, PartKind, PruneState, SessionEventRecord,
    };
    use lash_rlm_types::{RlmModeEvent, RlmTrajectoryEntry};

    fn user_event(id: &str, text: &str) -> SessionEventRecord {
        SessionEventRecord::Conversation(ConversationRecord {
            id: id.to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: text.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: None,
        })
    }

    fn step_event(mode_iteration: usize, code: &str, output: &str) -> SessionEventRecord {
        SessionEventRecord::Mode(crate::rlm_mode_event(RlmModeEvent::RlmTrajectoryEntry(
            RlmTrajectoryEntry {
                id: format!("rlm_step_{mode_iteration}"),
                mode_iteration,
                reasoning: "thinking".to_string(),
                code: code.to_string(),
                output: if output.is_empty() {
                    Vec::new()
                } else {
                    vec![output.to_string()]
                },
                tool_calls: Vec::new(),
                images: Vec::new(),
                error: None,
                final_output: None,
            },
        )))
    }

    fn tool_event() -> SessionEventRecord {
        SessionEventRecord::Tool(lash::ToolEvent::Invocation {
            stable_key: "call_1".to_string(),
            record: lash::ToolCallRecord {
                call_id: Some("call_1".to_string()),
                tool: "lookup".to_string(),
                args: serde_json::json!({"q": "first"}),
                result: serde_json::json!({"answer": "done"}),
                success: true,
                duration_ms: 4,
                control: None,
            },
        })
    }

    fn projector(max_output_chars: usize) -> RlmContextProjector {
        RlmContextProjector {
            max_output_chars,
            max_budget_tokens: None,
            last_prompt_usage: Arc::new(RwLock::new(None)),
        }
    }

    #[test]
    fn chronological_history_renders_messages_and_steps_in_order() {
        let projector = projector(100);
        let events = [
            user_event("u1", "first"),
            step_event(0, "print 1", "1"),
            user_event("u2", "second"),
            step_event(1, "print 2", "2"),
        ];
        let history = projector.format_history(&projection_from_events(&events));

        assert!(history.contains("--- history[0] · user message · 5 chars ---\n\nfirst"));
        assert!(history.contains("--- history[1] · rlm step · mode_iteration 0 ---"));
        assert!(history.contains("Code:\n```lashlang\nprint 1\n```"));
        assert!(history.contains("history[1].output[0] (1 chars):\n1"));
        assert!(history.contains("--- history[2] · user message · 6 chars ---\n\nsecond"));
        assert!(history.contains("--- history[3] · rlm step · mode_iteration 1 ---"));
        assert!(history.contains("history[3].output[0] (1 chars):\n2"));
        // Old combined "Output" + "Tool calls" sections were removed —
        // each `print` is now its own block, and tool calls are visible
        // inline in the `code` block above.
        assert!(!history.contains("\n\nOutput ("));
        assert!(!history.contains("\n\nTool calls:"));
        assert!(!history.contains("Task"));
        assert!(!history.contains("user_input_"));
    }

    #[test]
    fn chronological_history_keeps_tool_call_indexes() {
        let projector = projector(1000);
        let events = [
            user_event("u1", "first"),
            tool_event(),
            step_event(0, "x = 1", "1"),
        ];
        let history = projector.format_history(&projection_from_events(&events));

        assert!(history.contains("--- history[0] · user message"));
        assert!(history.contains("--- history[1] · tool_call · lookup · ok · 4 ms ---"));
        assert!(history.contains("--- history[2] · rlm step · mode_iteration 0 ---"));
        assert!(!history.contains("full: history[1].result"));
    }

    #[test]
    fn long_user_message_gets_full_history_reference() {
        let projector = projector(10);
        let history = projector.format_history(&projection_from_events(&[user_event(
            "u1",
            "abcdefghijklmnopqrstuvwxyz",
        )]));

        assert!(history.contains("full: history[0].content"));
        assert!(history.contains("... (16 characters omitted) ..."));
        assert!(!history.contains("user_input_"));
    }

    #[test]
    fn plugin_origin_is_not_rendered_in_history() {
        let projector = projector(100);
        let event = SessionEventRecord::Conversation(ConversationRecord {
            id: "plugin".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "plugin.p0".to_string(),
                kind: PartKind::Text,
                content: "synthetic plugin message".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: Some(lash::MessageOrigin::Plugin {
                plugin_id: "test".to_string(),
                transient: false,
            }),
        });

        let history = projector.format_history(&projection_from_events(&[event]));
        assert!(history.contains("--- history[0] · user message"));
        assert!(history.contains("synthetic plugin message"));
        assert!(!history.contains("from plugin"));
        assert!(!history.contains("test"));
    }

    #[test]
    fn printed_images_render_as_llm_image_blocks() {
        let event = SessionEventRecord::Mode(crate::rlm_mode_event(
            RlmModeEvent::RlmTrajectoryEntry(RlmTrajectoryEntry {
                id: "rlm_step_1".to_string(),
                mode_iteration: 1,
                reasoning: String::new(),
                code: "print img".to_string(),
                output: vec![r#"{"type":"image","id":"img"}"#.to_string()],
                tool_calls: Vec::new(),
                images: vec![lash::AttachmentRef {
                    id: lash::AttachmentId::new("img-ref"),
                    media_type: lash::MediaType::Image(lash::ImageMediaType::Png),
                    byte_len: 3,
                    width: Some(1),
                    height: Some(1),
                    label: Some("img.png".to_string()),
                }],
                error: None,
                final_output: None,
            }),
        ));
        let mut attachments = Vec::new();
        let mut blocks = Vec::new();

        let projection = projection_from_events(&[event]);
        append_entry_image_blocks(
            projection.entries().first().expect("entry"),
            &mut attachments,
            &mut blocks,
        );

        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].mime, "image/png");
        assert!(attachments[0].data.is_empty());
        assert_eq!(
            attachments[0]
                .reference
                .as_ref()
                .map(|reference| reference.id.as_str()),
            Some("img-ref")
        );
        assert!(matches!(
            blocks.as_slice(),
            [LlmContentBlock::Image { attachment_idx: 0 }]
        ));
    }

    #[test]
    fn rlm_prompt_projects_history_as_chat_messages_with_rolling_cache_breakpoint() {
        let projection =
            projection_from_events(&[user_event("u1", "first"), step_event(0, "print 1", "1")]);
        let mut attachments = Vec::new();

        let messages = build_rlm_history_messages(
            &projection,
            1000,
            2,
            rlm_finalization_prompt(&RlmTermination::default()),
            None,
            None,
            &mut attachments,
        );

        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[0].role, LlmRole::User));
        assert!(matches!(messages[1].role, LlmRole::Assistant));
        assert!(matches!(messages[2].role, LlmRole::User));
        assert!(matches!(
            messages[0].blocks.first(),
            Some(LlmContentBlock::Text {
                text,
                cache_breakpoint: false,
                ..
            }) if text.starts_with("--- history[0] · user message")
        ));
        assert!(matches!(
            messages[1].blocks.first(),
            Some(LlmContentBlock::Text {
                text,
                cache_breakpoint: true,
                ..
            }) if text.starts_with("--- history[1] · rlm step")
        ));
        assert!(matches!(
            messages[2].blocks.first(),
            Some(LlmContentBlock::Text {
                text,
                cache_breakpoint: false,
                ..
            }) if text.contains("=== CURRENT ITERATION: 2 ===")
        ));
    }

    #[test]
    fn rlm_prompt_renders_required_output_block_when_schema_present() {
        let projection = projection_from_events(&[user_event("u1", "first")]);
        let mut attachments = Vec::new();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["call", "fold"] },
                "amount": { "type": "integer", "minimum": 0 }
            },
            "required": ["action"]
        });

        let messages = build_rlm_history_messages(
            &projection,
            1000,
            1,
            "Call submit",
            Some(&render_value_schema_contract(&schema)),
            None,
            &mut attachments,
        );

        let tail = messages
            .last()
            .and_then(|message| message.blocks.first())
            .and_then(|block| match block {
                LlmContentBlock::Text { text, .. } => Some(text.as_ref()),
                _ => None,
            })
            .expect("tail block");
        assert!(tail.contains("=== REQUIRED OUTPUT ==="));
        assert!(tail.contains("{ action: enum[\"call\", \"fold\"], amount?: int >= 0 }"));
        assert!(tail.contains("Fields:"));
    }

    #[test]
    fn render_value_schema_contract_renders_object_shape_with_field_table() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["call", "fold"] },
                "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
            },
            "required": ["action"]
        });

        let rendered = render_value_schema_contract(&schema);
        let head = rendered.lines().next().expect("at least one line");
        assert_eq!(
            head,
            "{ action: enum[\"call\", \"fold\"], confidence?: float >= 0 <= 1 }"
        );
        assert!(rendered.contains("Fields:"));
        assert!(rendered.contains("- `action: enum[\"call\", \"fold\"]`"));
        assert!(rendered.contains("- `confidence?: float >= 0 <= 1`"));
    }

    #[test]
    fn render_value_schema_contract_falls_back_to_compact_label_for_scalars() {
        let scalar = serde_json::json!({ "type": "string" });
        assert_eq!(render_value_schema_contract(&scalar), "str");

        let array = serde_json::json!({ "type": "array", "items": { "type": "integer" } });
        assert_eq!(render_value_schema_contract(&array), "list[int]");

        let nullable_string = serde_json::json!({ "type": ["string", "null"] });
        assert_eq!(render_value_schema_contract(&nullable_string), "str | null");
    }

    #[test]
    fn incremental_render_extends_cached_prefix_on_subsequent_calls() {
        let projector = projector(100);
        let initial = projector.format_history(&projection_from_events(&[
            user_event("u1", "first"),
            step_event(0, "print 1", "1"),
        ]));
        assert!(initial.contains("--- history[0] · user message"));
        assert!(initial.contains("--- history[1] · rlm step · mode_iteration 0 ---"));

        let extended = projector.format_history(&projection_from_events(&[
            user_event("u1", "first"),
            step_event(0, "print 1", "1"),
            user_event("u2", "second"),
            step_event(1, "print 2", "2"),
        ]));
        assert!(extended.starts_with(&initial));
        assert!(extended.contains("--- history[2] · user message"));
        assert!(extended.contains("--- history[3] · rlm step · mode_iteration 1 ---"));
    }
}
