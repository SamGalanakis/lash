use lash_core::sansio::{
    CheckpointResumeAction, CompletedToolCall, ProtocolDriverHandle, WaitingExecState,
    WaitingLlmState,
};
use lash_core::session_model::{
    ConversationRecord, Message, SessionEvent, SessionEventRecord, fresh_message_id,
    make_error_event,
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
use super::fence::extract_first_lashlang_fence;
use super::finish::{
    internal_assistant_prose_message, submit_required_reminder_message,
    submit_schema_mismatch_message, turn_limit_final_message, validate_finish_value,
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

        let extraction = extract_first_lashlang_fence(&assistant_text);
        let Some(fence) = extraction else {
            let termination = match decode_rlm_termination_options(ctx.termination()) {
                Ok(termination) => termination,
                Err(err) => return invalid_turn_options_actions(err),
            };
            if matches!(termination, RlmTermination::ProseOrSubmit) {
                actions.push(DriverAction::AppendEvents(vec![diagnostic_event(
                    "llm_extraction",
                    serde_json::json!({
                        "found_lashlang_fence": false,
                        "prose_only_ends_turn": true,
                        "assistant_text_chars": assistant_text.chars().count(),
                        "reasoning_chars": reasoning_text.chars().count(),
                        "finalization_reason": "prose_or_submit",
                    }),
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
            let RlmTermination::SubmitRequired { schema } = termination else {
                unreachable!("ProseOrSubmit returned above");
            };
            actions.push(DriverAction::AppendEvents(vec![diagnostic_event(
                "llm_extraction",
                serde_json::json!({
                    "found_lashlang_fence": false,
                    "prose_only_ends_turn": false,
                    "assistant_text_chars": assistant_text.chars().count(),
                    "reasoning_chars": reasoning_text.chars().count(),
                    "finalization_reason": "submit_required",
                }),
            )]));
            let mut events = Vec::new();
            if !assistant_text.trim().is_empty() {
                events.push(conversation_event(internal_assistant_prose_message(
                    assistant_text,
                )));
            }
            events.push(conversation_event(submit_required_reminder_message(
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
            serde_json::json!({
                "found_lashlang_fence": true,
                "had_extra_fences": fence.had_extra_fences,
                "code_chars": fence.code.chars().count(),
                "assistant_text_chars": assistant_text.chars().count(),
                "reasoning_chars": reasoning_text.chars().count(),
                "decision": "execute_lashlang",
            }),
        )]));

        let Some(raw_state) = waiting.take_driver_state() else {
            return invalid_driver_state_actions("missing RLM driver state".to_string());
        };
        let mut state = match decode_rlm_driver_state(raw_state) {
            Ok(state) => state,
            Err(err) => return invalid_driver_state_actions(err),
        };
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
                    actions.push(DriverAction::AppendEvents(vec![trajectory_event(
                        trajectory_entry(ctx.protocol_iteration(), &state, None, None),
                    )]));
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
            if let RlmTermination::SubmitRequired {
                schema: Some(schema),
            } = termination
                && let Err(error_text) = validate_finish_value(finish_value, &schema)
            {
                if let Err(err) = continue_or_stop_after_nonterminal(
                    &ctx,
                    &mut actions,
                    vec![trajectory_event(trajectory_entry(
                        ctx.protocol_iteration(),
                        &state,
                        Some(error_text.clone()),
                        None,
                    ))],
                    vec![conversation_event(submit_schema_mismatch_message(
                        &error_text,
                    ))],
                ) {
                    return invalid_turn_options_actions(err);
                }
                return actions;
            }

            actions.push(DriverAction::AppendEvents(vec![trajectory_event(
                trajectory_entry(
                    ctx.protocol_iteration(),
                    &state,
                    None,
                    Some(finish_value.clone()),
                ),
            )]));
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish(TurnOutcome::Finished(
                    TurnFinish::SubmittedValue {
                        value: finish_value.clone(),
                    },
                )),
            });
            return actions;
        }

        if let Err(err) = continue_or_stop_after_nonterminal(
            &ctx,
            &mut actions,
            vec![trajectory_event(trajectory_entry(
                ctx.protocol_iteration(),
                &state,
                None,
                None,
            ))],
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
            RlmTermination::SubmitRequired { .. } => {
                actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                    TurnStop::MaxTurns,
                )));
                return Ok(());
            }
            RlmTermination::ProseOrSubmit => {
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
        id: format!("rlm_step_{protocol_iteration}"),
        protocol_iteration,
        reasoning: state.reasoning.clone(),
        code: state.executed_code.clone().unwrap_or_default(),
        output: state.output.clone(),
        images: state.images.clone(),
        error: validation_error.or_else(|| state.exec_error.clone()),
        final_output,
    }
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

fn combine_reasoning_and_text(reasoning: &str, text: &str) -> String {
    match (reasoning.trim().is_empty(), text.trim().is_empty()) {
        (true, true) => String::new(),
        (true, false) => text.to_string(),
        (false, true) => reasoning.to_string(),
        (false, false) => format!("{reasoning}\n\n{text}"),
    }
}
