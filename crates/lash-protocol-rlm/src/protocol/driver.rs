use lash_core::sansio::{
    CheckpointResumeAction, CompletedToolCall, ProtocolDriverHandle, WaitingExecState,
    WaitingLlmState,
};
use lash_core::session_model::{
    ConversationRecord, Message, MessageRole, Part, PartKind, PruneState, SessionEvent,
    SessionEventRecord, fresh_message_id, make_error_event, shared_parts,
};
use lash_core::{
    CheckpointKind, DriverAction, DriverContextView, ExecResponse, LlmOutputPart, LlmResponse,
    ToolCallRecord, TurnFinish, TurnOutcome, TurnStop, append_assistant_text_part,
    normalized_response_parts,
};
use lash_rlm_types::{RlmDiagnosticEvent, RlmProtocolEvent, RlmTermination, RlmTrajectoryEntry};
use serde_json::Value;

use crate::projection::rlm_protocol_event;
use crate::rlm_support::decode_rlm_termination_options;

use super::actions::{invalid_driver_state_actions, invalid_turn_options_actions};
use super::cell::{CellExtraction, extract_lashlang_cell};
use super::finish::{
    finish_required_reminder_message, finish_schema_mismatch_message,
    internal_assistant_prose_message, invalid_lashlang_cell_message, turn_limit_final_message,
    validate_finish_value,
};
use super::state::{RlmDriverState, decode_rlm_driver_state, rlm_driver_state};

pub struct RlmDriver;

impl ProtocolDriverHandle<lash_core::HostTurnProtocol> for RlmDriver {
    fn prepare_protocol_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        if let Err(err) = decode_rlm_termination_options(ctx.termination()) {
            return invalid_turn_options_actions(err);
        }
        vec![DriverAction::StartLlm {
            request: ctx.project_llm_request(false),
            driver_state: Some(rlm_driver_state(RlmDriverState::default())),
        }]
    }

    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_>,
        mut waiting: WaitingLlmState<lash_core::HostTurnProtocol>,
        llm_response: LlmResponse,
        _text_streamed: bool,
    ) -> Vec<DriverAction> {
        let mut actions = vec![DriverAction::Emit(SessionEvent::LlmResponse {
            protocol_iteration: ctx.protocol_iteration(),
            content: llm_response.full_text.clone(),
            duration_ms: 0,
        })];

        let mut assistant_text = String::new();
        let mut reasoning_text = String::new();
        for part in normalized_response_parts(&llm_response) {
            match part {
                LlmOutputPart::Text { text, .. } => {
                    append_assistant_text_part(&mut assistant_text, &text);
                }
                LlmOutputPart::Reasoning { text, replay } => {
                    let reasoning = if text.trim().is_empty() {
                        replay
                            .as_ref()
                            .map(|meta| meta.summary.join("\n\n"))
                            .unwrap_or_default()
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
            actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                TurnStop::ProviderError,
            )));
            return actions;
        }

        let termination = match decode_rlm_termination_options(ctx.termination()) {
            Ok(termination) => termination,
            Err(err) => return invalid_turn_options_actions(err),
        };

        let extraction = match extract_lashlang_cell(&assistant_text) {
            Ok(extraction) => extraction,
            Err(err) => {
                actions.push(DriverAction::AppendEvents(vec![diagnostic_event(
                    "llm_extraction",
                    llm_extraction_payload(
                        "invalid_lashlang_cell",
                        &termination,
                        LlmExtractionCounts::prose_only(&assistant_text, &reasoning_text),
                    ),
                )]));
                if let Err(err) = continue_or_stop_after_nonterminal(
                    &ctx,
                    &mut actions,
                    Vec::new(),
                    vec![conversation_event(invalid_lashlang_cell_message(
                        err.message(),
                    ))],
                ) {
                    return invalid_turn_options_actions(err);
                }
                return actions;
            }
        };
        let Some(cell) = extraction else {
            if matches!(termination, RlmTermination::Natural) {
                actions.push(DriverAction::AppendEvents(vec![diagnostic_event(
                    "llm_extraction",
                    llm_extraction_payload(
                        "finish_prose",
                        &termination,
                        LlmExtractionCounts::prose_only(&assistant_text, &reasoning_text),
                    ),
                )]));
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
            let RlmTermination::FinishRequired { ref schema } = termination else {
                unreachable!("Natural returned above");
            };
            actions.push(DriverAction::AppendEvents(vec![diagnostic_event(
                "llm_extraction",
                llm_extraction_payload(
                    "request_finish",
                    &termination,
                    LlmExtractionCounts::prose_only(&assistant_text, &reasoning_text),
                ),
            )]));
            let mut events = Vec::new();
            if !assistant_text.trim().is_empty() {
                events.push(conversation_event(internal_assistant_prose_message(
                    assistant_text,
                )));
            }
            events.push(conversation_event(finish_required_reminder_message(
                schema.is_some(),
            )));
            if let Err(err) =
                continue_or_stop_after_nonterminal(&ctx, &mut actions, Vec::new(), events)
            {
                return invalid_turn_options_actions(err);
            }
            return actions;
        };

        actions.push(DriverAction::AppendEvents(vec![diagnostic_event(
            "llm_extraction",
            llm_extraction_payload(
                "execute_lashlang",
                &termination,
                LlmExtractionCounts::cell(&assistant_text, &reasoning_text, &cell),
            ),
        )]));

        let Some(raw_state) = waiting.take_driver_state() else {
            return invalid_driver_state_actions("missing RLM driver state".to_string());
        };
        let mut state = match decode_rlm_driver_state(raw_state) {
            Ok(state) => state,
            Err(err) => return invalid_driver_state_actions(err),
        };
        state.executed_code = Some(cell.code.clone());
        state.reasoning = reasoning_text;
        state.prose = cell.prose.clone();

        // Emit the raw lashlang source as a `Message` with kind
        // `lashlang_code` so the CLI can reveal it in the full-expand
        // view (Alt+O) above the tool activities it produced.
        actions.push(DriverAction::Emit(SessionEvent::Message {
            text: cell.code.clone(),
            kind: "lashlang_code".to_string(),
        }));
        actions.push(DriverAction::StartExec {
            language: "lashlang".to_string(),
            code: cell.code,
            driver_state: rlm_driver_state(state),
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
        waiting: WaitingExecState<lash_core::HostTurnProtocol>,
        result: Result<ExecResponse, String>,
    ) -> Vec<DriverAction> {
        let mut state = match decode_rlm_driver_state(waiting.into_driver_state()) {
            Ok(state) => state,
            Err(err) => return invalid_driver_state_actions(err),
        };
        let mut actions = Vec::new();

        match result {
            Ok(response) => {
                let terminal_outcome = response
                    .tool_calls
                    .iter()
                    .find_map(terminal_outcome_from_tool_result);
                state.images.extend(response.printed_images);
                for observation in response.observations {
                    if !observation.is_empty() {
                        state.output.push(observation);
                    }
                }
                if let Some(raw_error) = response.error {
                    state.exec_error = Some(raw_error);
                }
                if let Some(finish_value) = response.terminal_finish {
                    state.terminal_finish = Some(finish_value);
                }
                if let Some(outcome) = terminal_outcome {
                    actions.push(DriverAction::AppendEvents(trajectory_events(
                        ctx.protocol_iteration(),
                        &state,
                        None,
                        None,
                    )));
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish(outcome),
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
            let termination = match decode_rlm_termination_options(ctx.termination()) {
                Ok(termination) => termination,
                Err(err) => return invalid_turn_options_actions(err),
            };
            if let RlmTermination::FinishRequired {
                schema: Some(schema),
            } = termination
                && let Err(error_text) = validate_finish_value(finish_value, &schema)
            {
                if let Err(err) = continue_or_stop_after_nonterminal(
                    &ctx,
                    &mut actions,
                    trajectory_events(
                        ctx.protocol_iteration(),
                        &state,
                        Some(error_text.clone()),
                        None,
                    ),
                    vec![conversation_event(finish_schema_mismatch_message(
                        &error_text,
                    ))],
                ) {
                    return invalid_turn_options_actions(err);
                }
                return actions;
            }

            actions.push(DriverAction::AppendEvents(trajectory_events(
                ctx.protocol_iteration(),
                &state,
                None,
                Some(finish_value.clone()),
            )));
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish(TurnOutcome::Finished(
                    TurnFinish::FinalValue {
                        value: finish_value.clone(),
                    },
                )),
            });
            return actions;
        }

        if let Err(err) = continue_or_stop_after_nonterminal(
            &ctx,
            &mut actions,
            trajectory_events(ctx.protocol_iteration(), &state, None, None),
            Vec::new(),
        ) {
            return invalid_turn_options_actions(err);
        }
        actions
    }
}

fn continue_or_stop_after_nonterminal(
    ctx: &DriverContextView<'_>,
    actions: &mut Vec<DriverAction>,
    durable_events: Vec<SessionEventRecord>,
    retry_events: Vec<SessionEventRecord>,
) -> Result<(), String> {
    if !durable_events.is_empty() {
        actions.push(DriverAction::AppendEvents(durable_events));
    }
    actions.push(DriverAction::AdvanceProtocolIteration);

    if ctx.should_force_exit_after_grace_turn() {
        actions.push(DriverAction::Finish(TurnOutcome::Stopped(
            TurnStop::MaxTurns,
        )));
        return Ok(());
    }

    let next_protocol_iteration = ctx.protocol_iteration() + 1;
    let reached_turn_limit = ctx
        .max_turns()
        .is_some_and(|max_turns| next_protocol_iteration >= ctx.protocol_run_offset() + max_turns);
    if reached_turn_limit {
        match decode_rlm_termination_options(ctx.termination())? {
            RlmTermination::FinishRequired { .. } => {
                actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                    TurnStop::MaxTurns,
                )));
                return Ok(());
            }
            RlmTermination::Natural => {
                if let Some(max_turns) = ctx.max_turns() {
                    actions.push(DriverAction::ScheduleTurnLimitFinal {
                        message: turn_limit_final_message(fresh_message_id(), max_turns),
                    });
                }
            }
        }
    } else if !retry_events.is_empty() {
        actions.push(DriverAction::AppendEvents(retry_events));
    }

    actions.push(DriverAction::StartCheckpoint {
        checkpoint: CheckpointKind::AfterWork,
        on_empty: CheckpointResumeAction::PrepareIteration,
    });
    Ok(())
}

fn terminal_outcome_from_tool_result(record: &ToolCallRecord) -> Option<TurnOutcome> {
    if !record.output.is_success() {
        return None;
    }
    match record.output.control.as_ref()? {
        lash_core::ToolControl::SwitchAgentFrame {
            frame_id,
            task: Some(task),
            ..
        } if !frame_id.trim().is_empty() && !task.trim().is_empty() => {
            Some(TurnOutcome::AgentFrameSwitch {
                frame_id: frame_id.clone(),
                task: task.clone(),
            })
        }
        lash_core::ToolControl::Finish { value } => {
            Some(TurnOutcome::Finished(TurnFinish::ToolValue {
                tool_name: record.tool.clone(),
                value: value.to_json_value(),
            }))
        }
        lash_core::ToolControl::Fail { failure } => {
            Some(TurnOutcome::Stopped(TurnStop::ToolError {
                tool_name: record.tool.clone(),
                value: failure.to_json_value(),
            }))
        }
        lash_core::ToolControl::SwitchAgentFrame { .. } => None,
    }
}

fn trajectory_entry(
    protocol_iteration: usize,
    state: &RlmDriverState,
    validation_error: Option<String>,
    final_output: Option<Value>,
) -> RlmTrajectoryEntry {
    RlmTrajectoryEntry {
        id: format!("lashlang_step_{protocol_iteration}"),
        protocol_iteration,
        code: state.executed_code.clone().unwrap_or_default(),
        output: state.output.clone(),
        images: state.images.clone(),
        error: validation_error.or_else(|| state.exec_error.clone()),
        final_output,
    }
}

fn trajectory_events(
    protocol_iteration: usize,
    state: &RlmDriverState,
    validation_error: Option<String>,
    final_output: Option<Value>,
) -> Vec<SessionEventRecord> {
    let mut events = Vec::new();
    if let Some(message) = assistant_content_message(&state.reasoning, &state.prose) {
        events.push(conversation_event(message));
    }
    events.push(trajectory_event(trajectory_entry(
        protocol_iteration,
        state,
        validation_error,
        final_output,
    )));
    events
}

fn assistant_content_message(reasoning: &str, prose: &str) -> Option<Message> {
    let mut parts = Vec::new();
    let id = fresh_message_id();
    let reasoning = reasoning.trim();
    if !reasoning.is_empty() {
        parts.push(Part {
            id: format!("{id}.r"),
            kind: PartKind::Reasoning,
            content: reasoning.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        });
    }
    let prose = prose.trim();
    if !prose.is_empty() {
        parts.push(Part {
            id: format!("{id}.t"),
            kind: PartKind::Text,
            content: prose.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        });
    }
    (!parts.is_empty()).then(|| Message {
        id,
        role: MessageRole::Assistant,
        parts: shared_parts(parts),
        origin: Some(lash_core::MessageOrigin::Plugin {
            plugin_id: "rlm_protocol".to_string(),
            transient: false,
        }),
    })
}

fn conversation_event(message: Message) -> SessionEventRecord {
    SessionEventRecord::Conversation(ConversationRecord::from_message(message))
}

fn trajectory_event(entry: RlmTrajectoryEntry) -> SessionEventRecord {
    SessionEventRecord::Protocol(rlm_protocol_event(RlmProtocolEvent::RlmTrajectoryEntry(
        entry,
    )))
}

fn diagnostic_event(phase: &str, payload: Value) -> SessionEventRecord {
    SessionEventRecord::Protocol(rlm_protocol_event(RlmProtocolEvent::RlmDiagnostic(
        RlmDiagnosticEvent {
            phase: phase.to_string(),
            payload,
        },
    )))
}

struct LlmExtractionCounts {
    full_text_chars: usize,
    prose_chars: usize,
    code_chars: usize,
    reasoning_chars: usize,
    lashlang_cell_count: usize,
}

impl LlmExtractionCounts {
    fn prose_only(assistant_text: &str, reasoning_text: &str) -> Self {
        Self {
            full_text_chars: assistant_text.chars().count(),
            prose_chars: assistant_text.chars().count(),
            code_chars: 0,
            reasoning_chars: reasoning_text.chars().count(),
            lashlang_cell_count: 0,
        }
    }

    fn cell(assistant_text: &str, reasoning_text: &str, cell: &CellExtraction) -> Self {
        Self {
            full_text_chars: assistant_text.chars().count(),
            prose_chars: cell.prose.chars().count(),
            code_chars: cell.code.chars().count(),
            reasoning_chars: reasoning_text.chars().count(),
            lashlang_cell_count: cell.lashlang_cell_count,
        }
    }
}

fn llm_extraction_payload(
    decision: &str,
    termination: &RlmTermination,
    counts: LlmExtractionCounts,
) -> Value {
    serde_json::json!({
        "decision": decision,
        "termination": termination_diagnostic_name(termination),
        "counts": {
            "full_text_chars": counts.full_text_chars,
            "prose_chars": counts.prose_chars,
            "code_chars": counts.code_chars,
            "reasoning_chars": counts.reasoning_chars,
            "lashlang_cell_count": counts.lashlang_cell_count,
        },
    })
}

fn termination_diagnostic_name(termination: &RlmTermination) -> &'static str {
    match termination {
        RlmTermination::FinishRequired { .. } => "finish_required",
        RlmTermination::Natural => "natural",
    }
}
