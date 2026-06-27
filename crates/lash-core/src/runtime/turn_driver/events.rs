use super::*;

pub(in crate::runtime) async fn send_session_event(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    event: SessionEvent,
) {
    if !event_tx.is_closed() {
        match &event {
            SessionEvent::TokenUsage {
                protocol_iteration,
                usage,
                cumulative,
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::Usage {
                        protocol_iteration: *protocol_iteration,
                        usage: usage.clone(),
                        cumulative: cumulative.clone(),
                    },
                )
                .await;
            }
            // ChildTokenUsage is projected to TurnEvent::ChildUsage at its
            // origin in `session_services::usage::ChildUsageEventRelay::emit`,
            // not here. Child usage events bypass `send_session_event` because
            // they're produced by the session manager rather than the parent's
            // turn driver.
            SessionEvent::LlmRequest {
                protocol_iteration, ..
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::ModelRequestStarted {
                        protocol_iteration: *protocol_iteration,
                    },
                )
                .await;
            }
            SessionEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
                ..
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::RetryStatus {
                        wait_seconds: *wait_seconds,
                        attempt: *attempt,
                        max_attempts: *max_attempts,
                        reason: reason.clone(),
                    },
                )
                .await;
            }
            SessionEvent::PluginEvent { plugin_id, event } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::PluginRuntime {
                        plugin_id: plugin_id.clone(),
                        event: event.clone(),
                    },
                )
                .await;
            }
            SessionEvent::InjectedTurnInputAccepted { inputs, checkpoint } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::QueuedInputAccepted {
                        checkpoint: *checkpoint,
                        inputs: inputs.clone(),
                    },
                )
                .await;
            }
            SessionEvent::InjectedMessagesCommitted {
                messages,
                checkpoint,
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::QueuedMessagesCommitted {
                        messages: messages.clone(),
                        checkpoint: *checkpoint,
                    },
                )
                .await;
            }
            SessionEvent::Error { message, .. } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::Error {
                        message: message.clone(),
                    },
                )
                .await;
            }
            SessionEvent::TurnOutcome {
                outcome: TurnOutcome::Finished(TurnFinish::FinalValue { value }),
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::FinalValue {
                        value: value.clone(),
                    },
                )
                .await;
            }
            SessionEvent::TurnOutcome {
                outcome: TurnOutcome::Finished(TurnFinish::ToolValue { tool_name, value }),
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::ToolValue {
                        tool_name: tool_name.clone(),
                        value: value.clone(),
                    },
                )
                .await;
            }
            _ => {}
        }
        let _ = event_tx.send(RuntimeStreamEvent::Session(event)).await;
    }
}

pub(in crate::runtime) async fn send_turn_activity(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    correlation_id: TurnActivityId,
    event: TurnEvent,
) {
    if !event_tx.is_closed() {
        let activity = TurnActivity::new(correlation_id, event);
        let _ = event_tx.send(RuntimeStreamEvent::Turn(activity)).await;
    }
}

async fn send_independent_turn_event(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    event: TurnEvent,
) {
    send_turn_activity(event_tx, TurnActivityId::fresh(), event).await;
}

pub(in crate::runtime) async fn emit_semantic_response_parts(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    response: &LlmResponse,
    prose_projector: Option<&dyn crate::plugin::AssistantProseProjectorPlugin>,
) {
    let has_text_correlation_ids = response.parts.iter().any(|part| {
        matches!(
            part,
            LlmOutputPart::Text {
                response_meta: Some(meta),
                ..
            } if meta.id.is_some()
        )
    });
    let mut emitted_text = false;
    for part in &response.parts {
        match part {
            LlmOutputPart::Text {
                text,
                response_meta,
            } if has_text_correlation_ids && !text.is_empty() => {
                let text = project_assistant_prose(text, prose_projector);
                if text.is_empty() {
                    continue;
                }
                emitted_text = true;
                let correlation_id = response_meta
                    .as_ref()
                    .and_then(|meta| meta.id.clone())
                    .map(TurnActivityId::new)
                    .unwrap_or_else(TurnActivityId::fresh);
                send_turn_activity(
                    event_tx,
                    correlation_id,
                    TurnEvent::AssistantProseDelta { text },
                )
                .await;
            }
            LlmOutputPart::Reasoning { text, replay } if !text.is_empty() => {
                let correlation_id = replay
                    .as_ref()
                    .and_then(|meta| meta.item_id.clone())
                    .map(TurnActivityId::new)
                    .unwrap_or_else(TurnActivityId::fresh);
                send_turn_activity(
                    event_tx,
                    correlation_id,
                    TurnEvent::ReasoningDelta { text: text.clone() },
                )
                .await;
            }
            _ => {}
        }
    }
    let full_text = project_assistant_prose(&response.full_text, prose_projector);
    let parts_text;
    let full_text = if full_text.is_empty() && !has_text_correlation_ids {
        parts_text =
            project_assistant_prose(&response_text_from_parts(&response.parts), prose_projector);
        parts_text.as_str()
    } else {
        full_text.as_str()
    };
    if !emitted_text && !full_text.is_empty() {
        send_independent_turn_event(
            event_tx,
            TurnEvent::AssistantProseDelta {
                text: full_text.to_string(),
            },
        )
        .await;
    }
}

fn project_assistant_prose(
    text: &str,
    projector: Option<&dyn crate::plugin::AssistantProseProjectorPlugin>,
) -> String {
    projector
        .map(|projector| projector.project_assistant_prose(text))
        .unwrap_or_else(|| text.to_string())
}

fn response_text_from_parts(parts: &[LlmOutputPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            LlmOutputPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}
