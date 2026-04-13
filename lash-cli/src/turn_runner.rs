use lash::session_model::{MessageRole, Part, PartKind, PruneState, fresh_message_id};
use lash::*;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::app::{PreparedTurn, UiResumeState};
use crate::input_items::build_items_from_editor_input;
use crate::{persist_live_runtime_snapshot, persist_root_session_state};

/// Returned by the spawned runtime task so callers can reclaim ownership.
pub(crate) struct RuntimeRunResult {
    pub(crate) stream_id: u64,
    pub(crate) runtime: LashRuntime,
    pub(crate) result: AssembledTurn,
}

pub(crate) fn make_turn_input(turn: &PreparedTurn) -> TurnInput {
    let (items, image_blobs) =
        build_items_from_editor_input(&turn.effective_text, turn.images.clone());
    TurnInput {
        items,
        image_blobs,
        user_input: Some(turn.input_provenance.clone()),
        mode: Some(RunMode::Normal),
    }
}

fn append_turn_input_message(messages: &mut Vec<Message>, turn_input: &TurnInput) {
    let user_id = fresh_message_id();
    let mut image_ids = Vec::new();
    let mut user_parts = Vec::new();

    for item in &turn_input.items {
        match item {
            InputItem::Text { text } => {
                if text.is_empty() {
                    continue;
                }
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: text.clone(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::FileRef { path } => {
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: format!("[file: {path}]"),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::DirRef { path } => {
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: format!("[directory: {}]", path.trim_end_matches('/')),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::ImageRef { id } => {
                let Some(bytes) = turn_input.image_blobs.get(id) else {
                    continue;
                };
                if image_ids.iter().any(|candidate| candidate == id) {
                    continue;
                }
                image_ids.push(id.clone());
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(lash::session_model::message::PartAttachment {
                        mime: "image/png".to_string(),
                        url: lash::session_model::message::data_url_for_bytes("image/png", bytes),
                        filename: None,
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
        }
    }

    if user_parts.is_empty() {
        user_parts.push(Part {
            id: format!("{user_id}.p0"),
            kind: PartKind::Text,
            content: String::new(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        });
    }

    messages.push(Message {
        id: user_id,
        role: MessageRole::User,
        parts: user_parts,
        user_input: turn_input.user_input.clone(),
        origin: None,
    });
}

pub(crate) fn pending_turn_snapshot(
    state: &SessionStateEnvelope,
    turn_input: &TurnInput,
) -> DurableTurnSnapshot {
    let mut messages = state.messages.clone();
    append_turn_input_message(&mut messages, turn_input);
    DurableTurnSnapshot {
        messages,
        tool_calls: state.tool_calls.clone(),
        iteration: state.iteration,
    }
}

pub(crate) fn persist_pending_turn(
    store: &Store,
    runtime: &LashRuntime,
    turn_input: &TurnInput,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
) {
    let persisted_state = runtime.export_state();
    persist_live_runtime_snapshot(
        store,
        Some(persisted_state.session_graph.clone()),
        pending_turn_snapshot(&persisted_state, turn_input),
        ui_state,
        dynamic_state,
        &persisted_state.policy,
        persisted_state.token_usage.clone(),
        persisted_state.last_prompt_usage.clone(),
        &persisted_state.token_ledger,
    );
}

fn refresh_state_from_committed_graph(store: &Store, state: &mut SessionStateEnvelope) {
    let Some(graph) = store.load_session_graph() else {
        return;
    };
    state.session_graph = graph;
    state.refresh_from_session_graph();
}

async fn snapshot_execution_state_into(
    runtime: &mut LashRuntime,
    state: &mut SessionStateEnvelope,
) {
    state.execution_state_snapshot = match runtime.snapshot_execution_state().await {
        Ok(Some(blob)) => Some(blob),
        Ok(None) => None,
        Err(err) => {
            tracing::warn!("failed to snapshot execution state for persistence: {err}");
            None
        }
    };
}

pub(crate) async fn persist_runtime_turn_state(
    runtime: &mut LashRuntime,
    state: &mut SessionStateEnvelope,
    interrupted: bool,
    store: &Store,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
) {
    snapshot_execution_state_into(runtime, state).await;
    if interrupted {
        persist_live_runtime_snapshot(
            store,
            store.load_session_graph(),
            DurableTurnSnapshot {
                messages: state.messages.clone(),
                tool_calls: state.tool_calls.clone(),
                iteration: state.iteration,
            },
            ui_state,
            dynamic_state,
            &state.policy,
            state.token_usage.clone(),
            state.last_prompt_usage.clone(),
            &state.token_ledger,
        );
    } else {
        refresh_state_from_committed_graph(store, state);
        persist_root_session_state(store, state, ui_state, dynamic_state);
    }
    runtime.set_state(state.clone());
}

pub(crate) fn spawn_runtime_turn<S>(
    mut runtime: LashRuntime,
    turn_input: TurnInput,
    sink: S,
    stream_id: u64,
) -> (CancellationToken, oneshot::Receiver<RuntimeRunResult>)
where
    S: EventSink + Send + Sync + 'static,
{
    let (return_tx, return_rx) = oneshot::channel();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();

    tokio::spawn(async move {
        tracing::debug!(stream_id, "runtime turn task spawned");
        let result = match runtime.stream_turn(turn_input, &sink, task_cancel).await {
            Ok(turn) => turn,
            Err(err) => AssembledTurn {
                state: runtime.export_state(),
                status: TurnStatus::Failed,
                assistant_output: AssistantOutput {
                    safe_text: String::new(),
                    raw_text: String::new(),
                    state: OutputState::EmptyOutput,
                },
                has_plugin_visible_output: false,
                done_reason: DoneReason::RuntimeError,
                execution: ExecutionSummary {
                    mode: runtime.export_state().policy.execution_mode,
                    had_tool_calls: false,
                    had_code_execution: false,
                },
                token_usage: TokenUsage::default(),
                tool_calls: Vec::new(),
                errors: vec![TurnIssue {
                    kind: "runtime".to_string(),
                    code: Some(err.code),
                    message: err.message,
                }],
                typed_finish: None,
            },
        };
        tracing::debug!(stream_id, status = ?result.status, "runtime turn task returning runtime");
        let _ = return_tx.send(RuntimeRunResult {
            stream_id,
            runtime,
            result,
        });
    });

    (cancel, return_rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_state_from_committed_graph_recovers_latest_token_ledger() {
        let store = Store::memory().expect("store");
        let mut graph = SessionGraph::default();
        graph.append_message(Message {
            id: fresh_message_id(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "p0".into(),
                kind: PartKind::Text,
                content: "hello".into(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        });
        let ledger = vec![TokenLedgerEntry {
            source: "turn".into(),
            model: "gpt-5.4-mini".into(),
            usage: TokenUsage {
                input_tokens: 42,
                output_tokens: 11,
                cached_input_tokens: 6,
                reasoning_tokens: 8,
            },
        }];
        graph.record_runtime_state(
            &PersistedSessionConfig {
                provider_id: "openai-compatible".into(),
                configured_model: "gpt-5.4-mini".into(),
                context_window: 0,
                execution_mode: ExecutionMode::Rlm,
                context_approach: ContextApproach::RollingHistory(RollingHistoryConfig::default()),
                model_variant: None,
            },
            &PersistedTurnState {
                iteration: 2,
                token_usage: TokenUsage {
                    input_tokens: 30,
                    output_tokens: 7,
                    cached_input_tokens: 5,
                    reasoning_tokens: 6,
                },
                last_prompt_usage: None,
            },
            None,
            None,
            None,
            &ledger,
        );
        store.save_session_graph(graph.clone());

        let mut stale_state = SessionStateEnvelope {
            session_graph: SessionGraph::default(),
            token_ledger: Vec::new(),
            ..SessionStateEnvelope::default()
        };
        refresh_state_from_committed_graph(&store, &mut stale_state);

        assert_eq!(stale_state.iteration, 2);
        assert_eq!(stale_state.messages.len(), 1);
        assert_eq!(stale_state.token_ledger.len(), 1);
        assert_eq!(stale_state.token_ledger[0].source, "turn");
        assert_eq!(stale_state.token_ledger[0].model, "gpt-5.4-mini");
        assert_eq!(stale_state.token_ledger[0].usage.input_tokens, 42);
        assert_eq!(stale_state.token_ledger[0].usage.output_tokens, 11);
        assert_eq!(stale_state.token_ledger[0].usage.cached_input_tokens, 6);
        assert_eq!(stale_state.token_ledger[0].usage.reasoning_tokens, 8);
        assert_eq!(stale_state.token_usage.input_tokens, 30);
        assert_eq!(stale_state.token_usage.output_tokens, 7);
    }
}
