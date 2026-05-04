use lash::*;
#[cfg(test)]
use lash_sqlite_store::Store;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::app::PreparedTurn;
use crate::input_items::build_items_from_editor_input;

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
        mode_turn_options: None,
    }
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
                outcome: TurnOutcome::Stopped(TurnStop::RuntimeError),
                assistant_output: AssistantOutput {
                    safe_text: String::new(),
                    raw_text: String::new(),
                    state: OutputState::EmptyOutput,
                },
                has_plugin_visible_output: false,
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
                    raw: None,
                }],
            },
        };
        tracing::debug!(stream_id, outcome = ?result.outcome, "runtime turn task returning runtime");
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
        let checkpoint_ref = store
            .put_checkpoint(&lash::HydratedSessionCheckpoint {
                turn_state: PersistedTurnState {
                    iteration: 2,
                    token_usage: TokenUsage {
                        input_tokens: 30,
                        output_tokens: 7,
                        cached_input_tokens: 5,
                        reasoning_tokens: 6,
                    },
                    last_prompt_usage: None,
                    mode_turn_options: Default::default(),
                },
                dynamic_state_ref: None,
                dynamic_state: None,
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

        assert_eq!(stale_state.iteration, 2);
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
