use lash::CancellationToken;
use lash::LashSession;
use lash::TurnInput;
use lash_core::runtime::RuntimeSessionState;
use lash_core::{
    AssistantOutput, ExecutionScope, ExecutionSummary, OutputState, TokenUsage, TurnIssue,
    TurnOutcome, TurnStop,
};
#[cfg(test)]
use lash_sqlite_store::Store;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::app::PreparedTurn;
use crate::input_items::build_items_from_editor_input;

/// Returned by the spawned session task after the app-owned session has been updated in place.
pub(crate) struct RuntimeRunResult {
    pub(crate) stream_id: u64,
    pub(crate) result: lash::TurnResult,
}

pub(crate) fn make_turn_input(turn: &PreparedTurn) -> TurnInput {
    let (items, image_blobs) =
        build_items_from_editor_input(&turn.effective_text, turn.images.clone());
    TurnInput::items(items).with_image_blobs(image_blobs)
}

pub(crate) fn spawn_session_turn(
    session: LashSession,
    turn_input: TurnInput,
    stream_id: u64,
) -> (CancellationToken, oneshot::Receiver<RuntimeRunResult>) {
    let (return_tx, return_rx) = oneshot::channel();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task_session = session.clone();
    let return_session = session;

    let task = tokio::spawn(async move {
        tracing::debug!(stream_id, "runtime turn task spawned");
        let result = match async {
            let turn_id = format!("cli-turn:{stream_id}");
            let effect_host = task_session.effect_host();
            let scoped = effect_host.scoped(ExecutionScope::turn(
                task_session.session_id(),
                turn_id.clone(),
            ))?;
            task_session
                .turn(turn_input)
                .turn_id(turn_id)
                .cancel(task_cancel)
                .advanced()
                .stream_to_with_scope(&lash::runtime::NoopTurnActivitySink, scoped)
                .await
        }
        .await
        {
            Ok(turn) => turn,
            Err(err) => runtime_error_turn_result(&task_session, err.to_string()).await,
        };
        tracing::debug!(stream_id, outcome = ?result.outcome, "runtime turn task completed");
        result
    });
    tokio::spawn(return_turn_result(
        "runtime turn",
        stream_id,
        return_session,
        task,
        return_tx,
    ));

    (cancel, return_rx)
}

pub(crate) fn spawn_session_queued_turn(
    session: LashSession,
    batch_ids: Vec<String>,
    stream_id: u64,
) -> (CancellationToken, oneshot::Receiver<RuntimeRunResult>) {
    let (return_tx, return_rx) = oneshot::channel();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task_session = session.clone();
    let return_session = session;

    let task = tokio::spawn(async move {
        tracing::debug!(stream_id, "queued runtime turn task spawned");
        let drain_id = batch_ids
            .first()
            .cloned()
            .unwrap_or_else(|| format!("cli-queue-drain:{stream_id}"));
        let result = match async {
            let effect_host = task_session.effect_host();
            let scoped = effect_host.scoped(ExecutionScope::queue_drain(
                task_session.session_id(),
                drain_id.clone(),
            ))?;
            task_session
                .queued_turn()
                .batch_ids(batch_ids)
                .drain_id(drain_id)
                .cancel(task_cancel)
                .advanced()
                .stream_to_with_scope(&lash::runtime::NoopTurnActivitySink, scoped)
                .await
        }
        .await
        {
            Ok(Some(turn)) => turn,
            Ok(None) => {
                runtime_error_turn_result(
                    &task_session,
                    "no durable queued work was ready".to_string(),
                )
                .await
            }
            Err(err) => runtime_error_turn_result(&task_session, err.to_string()).await,
        };
        tracing::debug!(stream_id, outcome = ?result.outcome, "queued runtime turn task completed");
        result
    });
    tokio::spawn(return_turn_result(
        "queued runtime turn",
        stream_id,
        return_session,
        task,
        return_tx,
    ));

    (cancel, return_rx)
}

async fn return_turn_result(
    task_label: &'static str,
    stream_id: u64,
    session: LashSession,
    task: JoinHandle<lash::TurnResult>,
    return_tx: oneshot::Sender<RuntimeRunResult>,
) {
    let result = match task.await {
        Ok(result) => result,
        Err(err) => {
            let failure = if err.is_panic() {
                "panicked"
            } else if err.is_cancelled() {
                "was cancelled"
            } else {
                "failed"
            };
            tracing::error!(
                stream_id,
                task = task_label,
                error = %err,
                "runtime task join failed"
            );
            runtime_error_turn_result(&session, format!("{task_label} {failure}: {err}")).await
        }
    };
    let _ = return_tx.send(RuntimeRunResult { stream_id, result });
}

async fn runtime_error_turn_result(session: &LashSession, message: String) -> lash::TurnResult {
    let state = session
        .admin()
        .state()
        .persist_current()
        .await
        .unwrap_or_else(|_| RuntimeSessionState::default());
    let state = state.to_snapshot();
    lash::TurnResult {
        execution: ExecutionSummary {
            had_tool_calls: false,
            had_code_execution: false,
        },
        state,
        outcome: TurnOutcome::Stopped(TurnStop::RuntimeError),
        assistant_output: AssistantOutput {
            safe_text: String::new(),
            raw_text: String::new(),
            state: OutputState::EmptyOutput,
        },
        usage: TokenUsage::default(),
        children_usage: Vec::new(),
        tool_calls: Vec::new(),
        errors: vec![TurnIssue {
            kind: "runtime".to_string(),
            code: Some(message.clone()),
            terminal_reason: None,
            message,
            raw: None,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::usage::TokenLedgerEntry;
    use lash_core::session_model::fresh_message_id;
    use lash_core::{
        HydratedSessionCheckpoint, Message, MessageRole, Part, PartKind, PersistedSessionConfig,
        PersistedTurnState, PruneState, SessionGraph, SessionHead, refresh_persisted_session_state,
    };

    #[tokio::test]
    async fn refresh_runtime_persistence_state_recovers_latest_token_ledger() {
        let store = Store::memory().await.expect("store");
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
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: None,
        });
        let ledger = vec![TokenLedgerEntry {
            source: "turn".into(),
            model: "gpt-5.4-mini".into(),
            usage: TokenUsage {
                input_tokens: 42,
                output_tokens: 11,
                cache_read_input_tokens: 6,
                cache_write_input_tokens: 0,
                reasoning_output_tokens: 8,
            },
        }];
        let checkpoint_ref = store
            .put_checkpoint(&HydratedSessionCheckpoint {
                turn_state: PersistedTurnState {
                    turn_index: 2,
                    token_usage: TokenUsage {
                        input_tokens: 30,
                        output_tokens: 7,
                        cache_read_input_tokens: 5,
                        cache_write_input_tokens: 0,
                        reasoning_output_tokens: 6,
                    },
                    last_prompt_usage: None,
                    protocol_turn_options: Default::default(),
                },
                tool_state_ref: None,
                tool_state: None,
                plugin_snapshot_ref: None,
                plugin_snapshot_revision: None,
                plugin_snapshot: None,
                execution_state_ref: None,
                execution_state: None,
            })
            .await
            .checkpoint_ref;
        store.append_usage_deltas(&ledger).await;
        store
            .save_session_head(SessionHead {
                session_id: "root".to_string(),
                head_revision: 0,
                agent_frames: Vec::new(),
                current_agent_frame_id: String::new(),
                graph: graph.clone(),
                config: PersistedSessionConfig {
                    provider_id: "openai-compatible".into(),
                    model: lash_core::ModelSpec::from_token_limits(
                        "gpt-5.4-mini",
                        None,
                        200_000,
                        None,
                    )
                    .expect("valid model spec"),
                },
                checkpoint_ref: Some(checkpoint_ref),
                token_ledger: ledger,
            })
            .await;

        let mut persistence_state = RuntimeSessionState {
            session_graph: SessionGraph::default(),
            token_ledger: Vec::new(),
            ..RuntimeSessionState::default()
        };
        refresh_persisted_session_state(&store, &mut persistence_state)
            .await
            .expect("refresh persisted session state");
        let stale_state = persistence_state;

        assert_eq!(stale_state.turn_index, 2);
        assert_eq!(stale_state.read_view().messages().len(), 1);
        assert_eq!(stale_state.token_ledger.len(), 1);
        assert_eq!(stale_state.token_ledger[0].source, "turn");
        assert_eq!(stale_state.token_ledger[0].model, "gpt-5.4-mini");
        assert_eq!(stale_state.token_ledger[0].usage.input_tokens, 42);
        assert_eq!(stale_state.token_ledger[0].usage.output_tokens, 11);
        assert_eq!(stale_state.token_ledger[0].usage.cache_read_input_tokens, 6);
        assert_eq!(stale_state.token_ledger[0].usage.reasoning_output_tokens, 8);
        assert_eq!(stale_state.token_usage.input_tokens, 30);
        assert_eq!(stale_state.token_usage.output_tokens, 7);
    }
}
