mod history;

use std::sync::{Arc, RwLock};

#[cfg(test)]
use lash_core::llm::types::LlmContentBlock;
use lash_core::llm::types::{LlmMessage, LlmRole, LlmToolChoice};
use lash_core::sansio::ContextProjector;
use lash_core::{
    LlmRequest, ProjectorContext, PromptContribution, PromptUsage, ProtocolBuildInput,
    TurnDriverConfig, TurnDriverPreamble,
};
use lash_rlm_types::RlmTermination;

#[cfg(test)]
use crate::projection::{rlm_history_projection, rlm_protocol_event};
use crate::rlm_support::decode_rlm_termination_options;

use history::{RlmHistoryRenderInput, build_rlm_history_messages_from_turn};
#[cfg(test)]
use history::{append_entry_image_blocks, build_rlm_history_messages, render_history_prompt};

/// Cell shared between the RLM protocol plugin's turn-prepare hook (writer)
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

pub fn build_rlm_preamble(
    input: ProtocolBuildInput,
    config: RlmProjectorConfig,
) -> TurnDriverPreamble {
    let tool_surface = input.tool_surface.as_ref();
    let omitted_tool_count = tool_surface.omitted_tool_count();
    let tool_names = tool_surface.tool_names();
    let tool_names_fingerprint = tool_surface.tool_names_fingerprint();
    let mut prompt_contributions = Vec::new();

    let tool_docs = tool_surface.prompt_tool_docs();
    if !tool_docs.trim().is_empty() {
        prompt_contributions.push(PromptContribution::execution("Showcased Tools", tool_docs));
    }
    prompt_contributions.extend(input.extra_prompt_contributions);

    TurnDriverPreamble {
        config: TurnDriverConfig {
            protocol: Arc::new(crate::protocol::RlmDriver),
            projector: Arc::new(RlmContextProjector {
                max_output_chars: config.max_output_chars,
                max_budget_tokens: config.max_budget_tokens,
                last_prompt_usage: config.last_prompt_usage,
            }),
            sync_execution_surface: true,
            turn_limit_final_message: Arc::new(crate::protocol::turn_limit_final_message),
        },
        tool_specs: Arc::new(Vec::new()),
        tool_names,
        tool_names_fingerprint,
        omitted_tool_count,
        execution_prompt: Arc::from(crate::protocol::rlm_execution_section_for_surface(
            config.prompt_features,
            &input.lashlang_surface,
        )),
        prompt_contributions,
    }
}

#[cfg(test)]
mod catalogue_tests {
    use super::*;
    use lash_core::{ToolActivation, ToolAvailabilityConfig, ToolScheduling};

    fn tool(name: &str) -> lash_core::ToolDefinition {
        lash_core::ToolDefinition::raw(
            format!("tool:{name}"),
            name,
            format!("Tool {name}"),
            serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
            serde_json::json!({ "type": "string" }),
        )
        .with_availability(ToolAvailabilityConfig::showcased())
        .with_activation(ToolActivation::Always)
        .with_scheduling(ToolScheduling::Parallel)
    }

    #[test]
    fn rlm_preamble_uses_resolved_tool_surface_without_search_tool_special_cases() {
        let definitions = vec![tool("search_tools"), tool("grep")];
        let contracts = definitions
            .iter()
            .map(|tool| (tool.name().to_string(), Arc::new(tool.contract())))
            .collect();
        let surface = lash_core::ToolSurface::from_tools(
            definitions
                .into_iter()
                .map(|tool| tool.manifest())
                .collect(),
            contracts,
        );

        let preamble = build_rlm_preamble(
            lash_core::ProtocolBuildInput {
                tool_surface: Arc::new(surface),
                lashlang_surface: lashlang::LashlangSurface::new(
                    lashlang::ResourceCatalog::tool_default(["search_tools", "grep"]),
                    lashlang::LashlangAbilities::all(),
                ),
                extra_prompt_contributions: Vec::new(),
            },
            RlmProjectorConfig::default(),
        );

        assert_eq!(preamble.omitted_tool_count, 0);
        assert_eq!(preamble.tool_names.as_ref(), &vec!["search_tools", "grep"]);
        let prompt = preamble
            .prompt_contributions
            .iter()
            .map(|contribution| contribution.content.as_ref())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(prompt.contains("search_tools"));
        assert!(prompt.contains("grep"));
    }

    #[test]
    fn rlm_preamble_uses_lashlang_surface_abilities() {
        let definitions = vec![tool("grep")];
        let contracts = definitions
            .iter()
            .map(|tool| (tool.name().to_string(), Arc::new(tool.contract())))
            .collect();
        let surface = lash_core::ToolSurface::from_tools(
            definitions
                .into_iter()
                .map(|tool| tool.manifest())
                .collect(),
            contracts,
        );

        let preamble = build_rlm_preamble(
            lash_core::ProtocolBuildInput {
                tool_surface: Arc::new(surface),
                lashlang_surface: lashlang::LashlangSurface::new(
                    lashlang::ResourceCatalog::tool_default(["grep"]),
                    lashlang::LashlangAbilities::default(),
                ),
                extra_prompt_contributions: Vec::new(),
            },
            RlmProjectorConfig::default(),
        );

        assert!(!preamble.execution_prompt.contains("process name"));
        assert!(!preamble.execution_prompt.contains("sleep for"));
        assert!(preamble.execution_prompt.contains("Module operations"));
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

impl ContextProjector<lash_core::HostTurnProtocol> for RlmContextProjector {
    fn project(&self, ctx: ProjectorContext<'_>) -> Arc<LlmRequest> {
        let termination = decode_rlm_termination_options(&ctx.config.termination)
            .expect("RLM turn options are validated before prompt projection");
        let finalization = rlm_finalization_prompt(&termination);
        let required_output = required_output_block(&termination);
        let budget_suffix = self.last_prompt_usage.read().ok().and_then(|guard| {
            crate::rlm_support::format_budget_suffix(
                ctx.protocol_iteration + 1,
                guard.as_ref(),
                self.max_budget_tokens,
            )
        });

        let mut messages = Vec::new();
        if !ctx.config.system_prompt.trim().is_empty() {
            messages.push(LlmMessage::text(
                LlmRole::System,
                Arc::clone(&ctx.config.system_prompt),
            ));
        }
        let mut attachments = Vec::new();
        messages.extend(build_rlm_history_messages_from_turn(
            RlmHistoryRenderInput {
                events: ctx.events,
                turn_messages: ctx.messages,
                turn_causes: ctx.turn_causes,
                max_output_chars: self.max_output_chars,
                protocol_iteration: ctx.protocol_iteration + 1,
                finalization,
                required_output: required_output.as_deref(),
                budget_suffix: budget_suffix.as_deref(),
            },
            &mut attachments,
        ));

        Arc::new(LlmRequest {
            model: ctx.config.model.clone(),
            messages,
            attachments,
            tools: Arc::new(Vec::new()),
            tool_choice: LlmToolChoice::None,
            model_variant: ctx.config.model_variant.clone(),
            session_id: ctx.config.run_session_id.clone(),
            output_spec: None,
            stream_events: None,
            generation: ctx.config.generation.clone(),
            provider_trace: None,
        })
    }
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
    let input_contract = lash_core::ToolDefinition::raw(
        "tool:submit",
        "submit",
        "",
        schema.clone(),
        serde_json::json!({}),
    )
    .compact_contract();

    if input_contract.parameters.is_empty() {
        return lash_core::ToolDefinition::raw(
            "tool:submit",
            "submit",
            "",
            lash_core::ToolDefinition::default_input_schema(),
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
    fn format_history(&self, projection: &lash_core::ChronologicalProjection) -> String {
        let history = rlm_history_projection(projection);
        render_history_prompt(history.history(), self.max_output_chars)
    }
}

#[cfg(test)]
fn projection_from_events(
    events: &[lash_core::SessionEventRecord],
) -> lash_core::ChronologicalProjection {
    lash_core::ChronologicalProjection::from_turn_view(
        events,
        &lash_core::MessageSequence::default(),
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::session_model::{
        ConversationRecord, MessageRole, Part, PartKind, PruneState, SessionEventRecord,
    };
    use lash_rlm_types::{RlmProtocolEvent, RlmTrajectoryEntry};

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
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: None,
        })
    }

    fn step_event(protocol_iteration: usize, code: &str, output: &str) -> SessionEventRecord {
        SessionEventRecord::Protocol(rlm_protocol_event(RlmProtocolEvent::RlmTrajectoryEntry(
            RlmTrajectoryEntry {
                id: format!("rlm_step_{protocol_iteration}"),
                protocol_iteration,
                reasoning: "thinking".to_string(),
                code: code.to_string(),
                output: if output.is_empty() {
                    Vec::new()
                } else {
                    vec![output.to_string()]
                },
                tool_call_ids: Vec::new(),
                images: Vec::new(),
                error: None,
                final_output: None,
            },
        )))
    }

    fn tool_event() -> SessionEventRecord {
        SessionEventRecord::Tool(lash_core::ToolEvent::Invocation {
            stable_key: "call_1".to_string(),
            record: lash_core::ToolCallRecord {
                call_id: Some("call_1".to_string()),
                tool: "lookup".to_string(),
                args: serde_json::json!({"q": "first"}),
                output: lash_core::ToolCallOutput::success(serde_json::json!({"answer": "done"})),
                duration_ms: 4,
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
        assert!(history.contains("--- history[1] · rlm step · protocol_iteration 0 ---"));
        assert!(history.contains("Code:\n```lashlang\nprint 1\n```"));
        assert!(history.contains("history[1].output[0] (1 chars):\n1"));
        assert!(history.contains("--- history[2] · user message · 6 chars ---\n\nsecond"));
        assert!(history.contains("--- history[3] · rlm step · protocol_iteration 1 ---"));
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
        assert!(history.contains("--- history[2] · rlm step · protocol_iteration 0 ---"));
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
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: Some(lash_core::MessageOrigin::Plugin {
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
    fn process_wake_history_renders_as_chronological_event_context() {
        let projector = projector(1000);
        let event = SessionEventRecord::Conversation(ConversationRecord {
            id: "wake:abc".to_string(),
            role: MessageRole::Event,
            parts: vec![Part {
                id: "wake:abc.p0".to_string(),
                kind: PartKind::Text,
                content: "Background process wake\nProcess: process-1\nEvent: process.wake #7\nWake input:\nblue button pressed".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: Some(lash_core::MessageOrigin::Process {
                process_id: "process-1".to_string(),
                event_type: "process.wake".to_string(),
                sequence: 7,
                wake_id: Some("wake:abc".to_string()),
            }),
        });
        let projection = projection_from_events(&[event]);
        let mut attachments = Vec::new();

        let messages = build_rlm_history_messages(
            &projection,
            1000,
            1,
            rlm_finalization_prompt(&RlmTermination::default()),
            None,
            None,
            &mut attachments,
        );
        let history = projector.format_history(&projection);

        assert!(history.contains("--- history[0] · event message"));
        assert!(history.contains("Background process wake"));
        assert!(history.contains("blue button pressed"));
        assert!(!history.contains("system message"));
        assert!(matches!(messages[0].role, LlmRole::User));
    }

    #[test]
    fn active_turn_causes_render_in_current_turn_events_without_history_duplication() {
        let cause = lash_core::TurnCause {
            id: "wake:abc".to_string(),
            event_type: "process.wake".to_string(),
            origin: lash_core::MessageOrigin::Process {
                process_id: "process-1".to_string(),
                event_type: "process.wake".to_string(),
                sequence: 7,
                wake_id: Some("wake:abc".to_string()),
            },
            text: "Background process wake\nProcess: process-1\nEvent: process.wake #7\nWake input:\nblue button pressed".to_string(),
        };
        let event_message = cause.to_event_message();
        let messages = lash_core::MessageSequence::from(vec![event_message]);
        let mut attachments = Vec::new();

        let rendered = build_rlm_history_messages_from_turn(
            RlmHistoryRenderInput {
                events: &[],
                turn_messages: &messages,
                turn_causes: std::slice::from_ref(&cause),
                max_output_chars: 1000,
                protocol_iteration: 0,
                finalization: rlm_finalization_prompt(&RlmTermination::default()),
                required_output: None,
                budget_suffix: None,
            },
            &mut attachments,
        );

        let combined = rendered
            .iter()
            .flat_map(|message| message.blocks.iter())
            .filter_map(|block| match block {
                LlmContentBlock::Text { text, .. } => Some(text.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(combined.contains("=== TURN EVENTS ==="));
        assert!(combined.contains("blue button pressed"));
        assert!(!combined.contains("--- history[0] · event message"));
        assert!(rendered.iter().any(|message| {
            message.role == LlmRole::User
                && message.blocks.iter().any(|block| {
                    matches!(
                        block,
                        LlmContentBlock::Text { text, .. }
                            if text.contains("=== TURN EVENTS ===")
                    )
                })
        }));
    }

    #[test]
    fn printed_images_render_as_llm_image_blocks() {
        let event = SessionEventRecord::Protocol(rlm_protocol_event(
            RlmProtocolEvent::RlmTrajectoryEntry(RlmTrajectoryEntry {
                id: "rlm_step_1".to_string(),
                protocol_iteration: 1,
                reasoning: String::new(),
                code: "print img".to_string(),
                output: vec![r#"{"type":"image","id":"img"}"#.to_string()],
                tool_call_ids: Vec::new(),
                images: vec![lash_core::AttachmentRef {
                    id: lash_core::AttachmentId::new("img-ref"),
                    media_type: lash_core::MediaType::Image(lash_core::ImageMediaType::Png),
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
        assert!(initial.contains("--- history[1] · rlm step · protocol_iteration 0 ---"));

        let extended = projector.format_history(&projection_from_events(&[
            user_event("u1", "first"),
            step_event(0, "print 1", "1"),
            user_event("u2", "second"),
            step_event(1, "print 2", "2"),
        ]));
        assert!(extended.starts_with(&initial));
        assert!(extended.contains("--- history[2] · user message"));
        assert!(extended.contains("--- history[3] · rlm step · protocol_iteration 1 ---"));
    }
}
