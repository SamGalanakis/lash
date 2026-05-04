use std::{collections::BTreeMap, fmt::Write as _, sync::Arc};

use lash::llm::types::{LlmAttachment, LlmContentBlock, LlmMessage, LlmRole, LlmToolChoice};
use lash::sansio::ContextProjector;
use lash::{
    LlmRequest, ModeBuildInput, ModeConfig, ModePreamble, ProjectorContext, PromptContribution,
    head_tail_truncate, session_model::SessionEventRecord,
};
use lash_rlm_types::{
    RlmAttachmentRef, RlmHistoryItem, RlmHistoryRole, RlmImageRef, RlmModeEvent, RlmTermination,
};

#[derive(Clone, Debug)]
pub struct RlmProjectorConfig {
    pub max_output_chars: usize,
}

impl Default for RlmProjectorConfig {
    fn default() -> Self {
        Self {
            max_output_chars: 10_000,
        }
    }
}

pub fn build_rlm_preamble(input: ModeBuildInput, config: RlmProjectorConfig) -> ModePreamble {
    let tool_surface = rlm_prompt_tool_surface((*input.tool_surface).clone());
    let omitted_tool_count = tool_surface.omitted_tool_count();
    let mut prompt_contributions = Vec::new();

    let tool_docs = tool_surface.prompt_tool_docs();
    if !tool_docs.trim().is_empty() {
        prompt_contributions.push(PromptContribution::execution("Showcased Tools", tool_docs));
    }
    if let Some(catalogue_prompt) = tool_catalogue_prompt(&tool_surface) {
        prompt_contributions.push(PromptContribution::guidance(
            "Tool Catalogue",
            catalogue_prompt,
        ));
    }
    prompt_contributions.extend(input.extra_prompt_contributions);

    ModePreamble {
        config: ModeConfig {
            protocol: Arc::new(crate::protocol::RlmDriver),
            projector: Arc::new(RlmContextProjector {
                max_output_chars: config.max_output_chars,
            }),
            sync_execution_surface: true,
        },
        tool_specs: Arc::new(Vec::new()),
        tool_names: tool_surface.tool_names(),
        omitted_tool_count,
        execution_prompt: crate::protocol::rlm_execution_section(),
        prompt_contributions,
    }
}

fn rlm_prompt_tool_surface(mut surface: lash::ToolSurface) -> lash::ToolSurface {
    let has_omitted_tools = surface.omitted_tool_count() > 0;
    if !has_omitted_tools {
        for entry in &mut surface.tools {
            if entry.definition.name == "search_tools" {
                entry.availability = lash::ToolAvailability::Hidden;
            }
        }
    }
    for entry in &mut surface.tools {
        if entry.definition.name == "load_tools" {
            entry.availability = lash::ToolAvailability::Hidden;
        }
    }
    surface
}

const CATALOGUE_NAMESPACE_LIMIT: usize = 100;
const CATALOGUE_TOOL_NAME_LIMIT: usize = 50;

fn tool_catalogue_prompt(surface: &lash::ToolSurface) -> Option<String> {
    let omitted_tool_count = surface.omitted_tool_count();
    if omitted_tool_count == 0 {
        return None;
    }

    let mut by_namespace: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for tool in surface.omitted_tools_iter() {
        let namespace = tool
            .definition
            .discovery
            .namespace
            .as_deref()
            .filter(|namespace| !namespace.trim().is_empty())
            .unwrap_or("default");
        by_namespace
            .entry(namespace)
            .or_default()
            .push(tool.definition.name.as_str());
    }
    for names in by_namespace.values_mut() {
        names.sort_unstable();
    }

    let mut rendered = format!(
        "Catalogued tools: {omitted_tool_count} not showcased here; searchable with `search_tools`.\n\
         When a task needs a tool not showcased here, run `search_tools(query=...)` and call the relevant result by name. \
         Results use the same compact contract shape as showcased tools: signature, returns, description, and capped examples."
    );

    if by_namespace.len() <= CATALOGUE_NAMESPACE_LIMIT {
        rendered.push_str("\n\nNamespaces: ");
        for (index, (namespace, names)) in by_namespace.iter().enumerate() {
            if index > 0 {
                rendered.push_str(", ");
            }
            let _ = write!(rendered, "{namespace}({})", names.len());
        }
    } else {
        let _ = write!(
            rendered,
            "\n\nNamespaces: {} total; use `search_tools` to narrow them.",
            by_namespace.len()
        );
    }

    if omitted_tool_count <= CATALOGUE_TOOL_NAME_LIMIT {
        rendered.push_str("\n\nCatalogued names:");
        for (namespace, names) in by_namespace {
            rendered.push('\n');
            let _ = write!(rendered, "{namespace}: {}", names.join(", "));
        }
    }

    Some(rendered)
}

#[cfg(test)]
mod catalogue_tests {
    use super::*;
    use lash::{
        ExecutionMode, ToolActivation, ToolAvailability, ToolAvailabilityConfig,
        ToolDiscoveryMetadata, ToolExecutionMode, build_tool_surface,
    };

    fn tool(
        name: &str,
        availability: ToolAvailability,
        namespace: Option<&str>,
    ) -> lash::ToolDefinition {
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
        .with_availability(ToolAvailabilityConfig::same(availability))
        .with_activation(ToolActivation::Always)
        .with_discovery(ToolDiscoveryMetadata {
            namespace: namespace.map(str::to_string),
            aliases: Vec::new(),
        })
        .with_execution_mode(ToolExecutionMode::Parallel)
    }

    #[test]
    fn catalogue_prompt_lists_namespace_counts_and_small_catalogue_names() {
        let surface = build_tool_surface(lash::ToolSurfaceBuildInput {
            tools: vec![
                tool(
                    "search_tools",
                    ToolAvailability::Documented,
                    Some("runtime"),
                ),
                tool("grep", ToolAvailability::Callable, Some("filesystem")),
                tool("read_file", ToolAvailability::Callable, Some("filesystem")),
                tool(
                    "spotify_search_songs",
                    ToolAvailability::Callable,
                    Some("appworld"),
                ),
            ],
            mode: ExecutionMode::new("test"),
            contributions: Vec::new(),
        });

        let prompt = tool_catalogue_prompt(&surface).expect("catalogue prompt");
        assert!(prompt.contains("Catalogued tools: 3 not showcased here"));
        assert!(prompt.contains("appworld(1)"));
        assert!(prompt.contains("filesystem(2)"));
        assert!(prompt.contains("filesystem: grep, read_file"));
        assert!(prompt.contains("appworld: spotify_search_songs"));
    }

    #[test]
    fn catalogue_prompt_omits_names_for_large_catalogues() {
        let mut tools = vec![tool(
            "search_tools",
            ToolAvailability::Documented,
            Some("runtime"),
        )];
        for index in 0..=CATALOGUE_TOOL_NAME_LIMIT {
            tools.push(tool(
                &format!("tool_{index}"),
                ToolAvailability::Callable,
                Some("bulk"),
            ));
        }
        let surface = build_tool_surface(lash::ToolSurfaceBuildInput {
            tools,
            mode: ExecutionMode::new("test"),
            contributions: Vec::new(),
        });

        let prompt = tool_catalogue_prompt(&surface).expect("catalogue prompt");
        assert!(prompt.contains("bulk(51)"));
        assert!(!prompt.contains("Catalogued names:"));
    }

    #[test]
    fn rlm_preamble_hides_catalogue_tools_when_everything_is_showcased() {
        let surface = build_tool_surface(lash::ToolSurfaceBuildInput {
            tools: vec![
                tool(
                    "search_tools",
                    ToolAvailability::Documented,
                    Some("runtime"),
                ),
                tool("load_tools", ToolAvailability::Documented, Some("runtime")),
                tool("grep", ToolAvailability::Documented, Some("filesystem")),
            ],
            mode: ExecutionMode::new("test"),
            contributions: Vec::new(),
        });

        let preamble = build_rlm_preamble(
            lash::ModeBuildInput {
                mode: ExecutionMode::new("test"),
                tool_surface: Arc::new(surface),
                extra_prompt_contributions: Vec::new(),
            },
            RlmProjectorConfig::default(),
        );

        assert_eq!(preamble.omitted_tool_count, 0);
        assert_eq!(preamble.tool_names, vec!["grep".to_string()]);
        let prompt = preamble
            .prompt_contributions
            .iter()
            .map(|contribution| contribution.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!prompt.contains("search_tools"));
        assert!(!prompt.contains("load_tools"));
        assert!(!prompt.contains("Tool Catalogue"));
    }

    #[test]
    fn rlm_preamble_keeps_search_tools_but_hides_load_tools_when_catalogue_has_omitted_tools() {
        let surface = build_tool_surface(lash::ToolSurfaceBuildInput {
            tools: vec![
                tool(
                    "search_tools",
                    ToolAvailability::Documented,
                    Some("runtime"),
                ),
                tool("load_tools", ToolAvailability::Documented, Some("runtime")),
                tool("grep", ToolAvailability::Callable, Some("filesystem")),
            ],
            mode: ExecutionMode::new("test"),
            contributions: Vec::new(),
        });

        let preamble = build_rlm_preamble(
            lash::ModeBuildInput {
                mode: ExecutionMode::new("test"),
                tool_surface: Arc::new(surface),
                extra_prompt_contributions: Vec::new(),
            },
            RlmProjectorConfig::default(),
        );

        assert_eq!(preamble.omitted_tool_count, 1);
        assert!(preamble.tool_names.contains(&"search_tools".to_string()));
        assert!(!preamble.tool_names.contains(&"load_tools".to_string()));
        let prompt = preamble
            .prompt_contributions
            .iter()
            .map(|contribution| contribution.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(prompt.contains("search_tools"));
        assert!(!prompt.contains("load_tools"));
        assert!(prompt.contains("Catalogued tools: 1 not showcased here"));
    }

    #[test]
    fn finish_finalization_prompt_defaults_to_submit_guidance() {
        let prompt = rlm_finalization_prompt(&RlmTermination::Finish {
            schema: None,
            include_submit_prompt: true,
        });

        assert!(prompt.contains("submit <value>"));
    }

    #[test]
    fn finish_finalization_prompt_can_suppress_submit_guidance() {
        let prompt = rlm_finalization_prompt(&RlmTermination::Finish {
            schema: None,
            include_submit_prompt: false,
        });

        assert!(prompt.contains("task-specific completion path"));
        assert!(!prompt.contains("submit"));
    }
}

struct RlmContextProjector {
    max_output_chars: usize,
}

impl ContextProjector<lash::HostModeProtocol> for RlmContextProjector {
    fn project(&self, ctx: ProjectorContext<'_>) -> LlmRequest {
        let history = self.format_history(ctx.events);
        let termination = ctx.config.termination.rlm_termination();
        let finalization = rlm_finalization_prompt(&termination);
        let user_prompt = format!(
            "=== HISTORY ===\n\n{history}\n\n\n=== CURRENT ITERATION: {} ===\n\n\n=== FINALIZATION ===\n\n{finalization}",
            ctx.iteration + 1
        );

        let mut messages = Vec::new();
        if !ctx.config.system_prompt.trim().is_empty() {
            messages.push(LlmMessage::text(
                LlmRole::System,
                std::sync::Arc::clone(&ctx.config.system_prompt),
            ));
        }
        let mut attachments = Vec::new();
        let mut user_blocks = vec![LlmContentBlock::Text {
            text: user_prompt.into(),
            response_meta: None,
        }];
        append_history_image_blocks(ctx.events, &mut attachments, &mut user_blocks);
        messages.push(LlmMessage::new(LlmRole::User, user_blocks));

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

fn rlm_finalization_prompt(termination: &RlmTermination) -> &'static str {
    match termination {
        RlmTermination::Finish {
            include_submit_prompt: true,
            ..
        } => {
            "Call `submit <value>` from lashlang when the task is complete. Do not answer in prose without a lashlang block. Avoid submitting large raw variables; summarize results or submit a concise final answer instead."
        }
        RlmTermination::Finish {
            include_submit_prompt: false,
            ..
        } => {
            "Continue in lashlang blocks until the task-specific completion path is satisfied. Do not answer in prose without a lashlang block."
        }
    }
}

impl RlmContextProjector {
    fn format_history(&self, events: &[SessionEventRecord]) -> String {
        let projection = lash::ChronologicalProjection::from_events(events.iter());
        render_history_prompt(&projection.rlm_history(), self.max_output_chars)
    }
}

fn render_history_prompt(history: &[RlmHistoryItem], max_output_chars: usize) -> String {
    if history.is_empty() {
        return "No chronological history is available.".to_string();
    }
    let mut rendered = String::new();
    for (index, item) in history.iter().enumerate() {
        if !rendered.is_empty() {
            rendered.push_str("\n\n");
        }
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
                iteration,
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
                    iteration: *iteration,
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

fn append_history_image_blocks(
    events: &[SessionEventRecord],
    attachments: &mut Vec<LlmAttachment>,
    blocks: &mut Vec<LlmContentBlock>,
) {
    for event in events {
        match event {
            SessionEventRecord::Conversation(record) => {
                for part in record.parts.iter() {
                    let Some(attachment) = part.attachment.as_ref() else {
                        continue;
                    };
                    let attachment_idx = attachments.len();
                    attachments.push(LlmAttachment::reference(attachment.reference.clone()));
                    blocks.push(LlmContentBlock::Image { attachment_idx });
                }
            }
            SessionEventRecord::Mode(event) => {
                let Some(RlmModeEvent::RlmTrajectoryEntry(entry)) = event.rlm_event() else {
                    continue;
                };
                for image in &entry.images {
                    let attachment_idx = attachments.len();
                    attachments.push(LlmAttachment::reference(image.clone()));
                    blocks.push(LlmContentBlock::Image { attachment_idx });
                }
            }
            SessionEventRecord::Tool(_) | SessionEventRecord::StateSnapshot(_) => {}
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

fn message_role_label(role: &RlmHistoryRole) -> &'static str {
    match role {
        RlmHistoryRole::User => "User",
        RlmHistoryRole::Assistant => "Assistant",
        RlmHistoryRole::System => "System",
    }
}

struct ReplStepRender<'a> {
    index: usize,
    iteration: usize,
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
        iteration,
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
        "--- history[{index}] · rlm step · iteration {iteration} ---\n\nReasoning ({reasoning_raw_len} chars{reasoning_ref}):\n{}\n\nCode:\n```lashlang\n{}\n```",
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
    let after_open = open_rel + 3;
    let rest = &text[after_open..];
    let Some(lang_end_rel) = rest.find('\n') else {
        return text[..open_rel].to_string();
    };
    let lang = rest[..lang_end_rel].trim();
    if !matches!(lang, "lashlang" | "rlm" | "lash") {
        return text.to_string();
    }
    let body_start = after_open + lang_end_rel + 1;
    let close = text[body_start..]
        .find("```")
        .map(|rel| body_start + rel)
        .unwrap_or(text.len());
    let after_close = (close + 3).min(text.len());
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
    use lash::session_model::{ConversationRecord, MessageRole, Part, PartKind, PruneState};
    use lash_rlm_types::RlmTrajectoryEntry;

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
            user_input: None,
            origin: None,
        })
    }

    fn step_event(iteration: usize, code: &str, output: &str) -> SessionEventRecord {
        SessionEventRecord::Mode(lash::ModeEvent::rlm(RlmModeEvent::RlmTrajectoryEntry(
            RlmTrajectoryEntry {
                id: format!("rlm_step_{iteration}"),
                iteration,
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
            },
        })
    }

    fn projector(max_output_chars: usize) -> RlmContextProjector {
        RlmContextProjector { max_output_chars }
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
        let history = projector.format_history(&events);

        assert!(history.contains("--- history[0] · user message · 5 chars ---\n\nfirst"));
        assert!(history.contains("--- history[1] · rlm step · iteration 0 ---"));
        assert!(history.contains("Code:\n```lashlang\nprint 1\n```"));
        assert!(history.contains("history[1].output[0] (1 chars):\n1"));
        assert!(history.contains("--- history[2] · user message · 6 chars ---\n\nsecond"));
        assert!(history.contains("--- history[3] · rlm step · iteration 1 ---"));
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
        let history = projector.format_history(&events);

        assert!(history.contains("--- history[0] · user message"));
        assert!(history.contains("--- history[1] · tool_call · lookup · ok · 4 ms ---"));
        assert!(history.contains("--- history[2] · rlm step · iteration 0 ---"));
        assert!(!history.contains("full: history[1].result"));
    }

    #[test]
    fn long_user_message_gets_full_history_reference() {
        let projector = projector(10);
        let history = projector.format_history(&[user_event("u1", "abcdefghijklmnopqrstuvwxyz")]);

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
            user_input: None,
            origin: Some(lash::MessageOrigin::Plugin {
                plugin_id: "test".to_string(),
                transient: false,
            }),
        });

        let history = projector.format_history(&[event]);
        assert!(history.contains("--- history[0] · user message"));
        assert!(history.contains("synthetic plugin message"));
        assert!(!history.contains("from plugin"));
        assert!(!history.contains("test"));
    }

    #[test]
    fn printed_images_render_as_llm_image_blocks() {
        let event = SessionEventRecord::Mode(lash::ModeEvent::rlm(
            RlmModeEvent::RlmTrajectoryEntry(RlmTrajectoryEntry {
                id: "rlm_step_1".to_string(),
                iteration: 1,
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

        append_history_image_blocks(&[event], &mut attachments, &mut blocks);

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
    fn incremental_render_extends_cached_prefix_on_subsequent_calls() {
        let projector = projector(100);
        let initial =
            projector.format_history(&[user_event("u1", "first"), step_event(0, "print 1", "1")]);
        assert!(initial.contains("--- history[0] · user message"));
        assert!(initial.contains("--- history[1] · rlm step · iteration 0 ---"));

        let extended = projector.format_history(&[
            user_event("u1", "first"),
            step_event(0, "print 1", "1"),
            user_event("u2", "second"),
            step_event(1, "print 2", "2"),
        ]);
        assert!(extended.starts_with(&initial));
        assert!(extended.contains("--- history[2] · user message"));
        assert!(extended.contains("--- history[3] · rlm step · iteration 1 ---"));
    }
}
