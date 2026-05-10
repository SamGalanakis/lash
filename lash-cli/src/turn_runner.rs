use lash::*;
use lash_embed::LashSession;
#[cfg(test)]
use lash_sqlite_store::Store;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::app::PreparedTurn;
use crate::input_items::build_items_from_editor_input;

/// Returned by the spawned session task after the app-owned session has been updated in place.
pub(crate) struct RuntimeRunResult {
    pub(crate) stream_id: u64,
    pub(crate) result: lash_embed::TurnResult,
}

pub(crate) fn make_turn_input(turn: &PreparedTurn) -> TurnInput {
    let (items, image_blobs) =
        build_items_from_editor_input(&turn.effective_text, turn.images.clone());
    TurnInput {
        items,
        image_blobs,
        mode: Some(RunMode::Normal),
        mode_turn_options: None,
        trace_turn_id: None,
        mode_extension: None,
        turn_context: lash::TurnContext::default(),
    }
}

pub(crate) fn spawn_session_turn<S>(
    session: LashSession,
    turn_input: TurnInput,
    sink: S,
    stream_id: u64,
) -> (CancellationToken, oneshot::Receiver<RuntimeRunResult>)
where
    S: TurnActivitySink + Send + Sync + 'static,
{
    let (return_tx, return_rx) = oneshot::channel();
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();

    tokio::spawn(async move {
        tracing::debug!(stream_id, "runtime turn task spawned");
        let result = match session
            .turn(turn_input)
            .cancel(task_cancel)
            .stream(&sink)
            .await
        {
            Ok(turn) => turn,
            Err(err) => {
                let state = session
                    .control()
                    .state()
                    .persist_current()
                    .await
                    .unwrap_or_else(|_| PersistedSessionState::default());
                let state = SessionStateEnvelope {
                    session_id: state.session_id,
                    policy: state.policy,
                    session_graph: state.session_graph,
                    turn_index: state.turn_index,
                    token_usage: state.token_usage,
                    last_prompt_usage: state.last_prompt_usage,
                    mode_turn_options: state.mode_turn_options,
                };
                lash_embed::TurnResult {
                    execution: ExecutionSummary {
                        mode: state.policy.execution_mode.clone(),
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
                        code: Some(err.to_string()),
                        message: err.to_string(),
                        raw: None,
                    }],
                }
            }
        };
        tracing::debug!(stream_id, outcome = ?result.outcome, "runtime turn task completed");
        let _ = return_tx.send(RuntimeRunResult { stream_id, result });
    });

    (cancel, return_rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::session_model::fresh_message_id;

    #[tokio::test]
    async fn refresh_runtime_persistence_state_recovers_latest_token_ledger() {
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
                tool_item_id: None,
                tool_signature: None,
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
                cached_input_tokens: 6,
                reasoning_tokens: 8,
            },
        }];
        let checkpoint_ref = store
            .put_checkpoint(&lash::HydratedSessionCheckpoint {
                turn_state: PersistedTurnState {
                    turn_index: 2,
                    token_usage: TokenUsage {
                        input_tokens: 30,
                        output_tokens: 7,
                        cached_input_tokens: 5,
                        reasoning_tokens: 6,
                    },
                    last_prompt_usage: None,
                    mode_turn_options: Default::default(),
                },
                tool_state_ref: None,
                tool_state: None,
                plugin_snapshot_ref: None,
                plugin_snapshot_revision: None,
                plugin_snapshot: None,
                execution_state_ref: None,
                execution_state: None,
            })
            .checkpoint_ref;
        store.append_usage_deltas(&ledger);
        store.save_session_head(lash::SessionHead {
            session_id: "root".to_string(),
            head_revision: 0,
            graph: graph.clone(),
            config: PersistedSessionConfig {
                provider_id: "openai-compatible".into(),
                configured_model: "gpt-5.4-mini".into(),
                context_window: 0,
                execution_mode: ExecutionMode::new("rlm"),
                standard_context_approach: Some(StandardContextApproach::RollingHistory(
                    RollingHistoryConfig,
                )),
                model_variant: None,
            },
            checkpoint_ref: Some(checkpoint_ref),
            token_ledger: ledger,
        });

        let mut persistence_state = lash::PersistedSessionState {
            session_graph: SessionGraph::default(),
            token_ledger: Vec::new(),
            ..lash::PersistedSessionState::default()
        };
        lash::refresh_persisted_session_state(&store, &mut persistence_state)
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
        assert_eq!(stale_state.token_ledger[0].usage.cached_input_tokens, 6);
        assert_eq!(stale_state.token_ledger[0].usage.reasoning_tokens, 8);
        assert_eq!(stale_state.token_usage.input_tokens, 30);
        assert_eq!(stale_state.token_usage.output_tokens, 7);
    }
}
