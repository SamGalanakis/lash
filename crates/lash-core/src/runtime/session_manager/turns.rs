use super::*;

impl ManagedSessionCapability {
    pub(in crate::runtime::session_manager) async fn start_turn(
        &self,
        current: &CurrentSessionCapability,
        usage: &UsageCapability,
        request: crate::SessionTurnRequest<'_>,
    ) -> Result<AssembledTurn, crate::PluginError> {
        let (
            crate::SessionTurnInput {
                session_id,
                turn_id,
                input,
            },
            scoped_effect_controller,
        ) = request.into_parts();
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(&session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let policy = {
            let runtime = runtime.runtime.lock().await;
            runtime.session_policy()
        };
        let cancel = CancellationToken::new();
        let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(100);
        let usage_source = self.child_usage_source(usage, &session_id);
        let sink = ChannelEventSink {
            tx: event_tx,
            live_usage: Some(LiveChildUsageForwarder {
                turn_id: turn_id.to_string(),
                session_id: session_id.to_string(),
                source: usage_source,
                model: policy.model.id.clone(),
                token_ledger: Arc::clone(&usage.token_ledger),
                child_turn_live_usage: Arc::clone(&usage.child_turn_live_usage),
                relay: usage.child_usage_event_relay.clone(),
            }),
        };
        let event_drain = tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
        {
            let mut turns = self.turns.lock().await;
            if turns
                .values()
                .any(|turn| turn.session_id == session_id.as_str())
            {
                return Err(crate::PluginError::Session(format!(
                    "session `{session_id}` already has a running turn"
                )));
            }
            turns.insert(
                turn_id.to_string(),
                ManagedSessionTurn {
                    session_id: session_id.to_string(),
                },
            );
        }
        let turn = {
            let mut runtime_guard = runtime.runtime.lock().await;
            let result = Box::pin(async {
                let run = runtime_guard
                    .stream_turn_with_agent_frames(
                        input,
                        crate::runtime::TurnOptions::new(cancel, scoped_effect_controller)
                            .with_events(&sink),
                    )
                    .await
                    .map_err(|err| crate::PluginError::Session(err.to_string()))?;
                let turn = run.into_final_turn().ok_or_else(|| {
                    crate::PluginError::Session(
                        "agent frame run completed without a turn".to_string(),
                    )
                })?;
                Ok(turn)
            })
            .await;
            runtime.publish_from(&runtime_guard);
            result
        };
        self.turns.lock().await.remove(&turn_id);
        drop(sink);
        let _ = event_drain.await;
        let live_reported = self.turn_live_usage(usage, &turn_id);
        if let Ok(turn) = &turn {
            let source = self.child_usage_source(usage, &session_id);
            if let Some(remainder) = subtract_usage(&live_reported, &turn.token_usage) {
                usage.record_token_usage(&source, &turn.state.policy.model.id, &remainder);
            }
        }
        usage.persist_current_usage_ledger(current).await?;
        turn
    }

    fn child_usage_source(&self, usage: &UsageCapability, session_id: &str) -> String {
        usage
            .child_sources
            .lock()
            .expect("child usage sources lock")
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| "child".to_string())
    }

    fn turn_live_usage(&self, usage: &UsageCapability, turn_id: &str) -> TokenUsage {
        usage
            .child_turn_live_usage
            .lock()
            .expect("child turn live usage lock")
            .remove(turn_id)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn session_turn_request_requires_matching_scope_and_sets_trace_turn_id() {
        let controller = crate::InlineRuntimeEffectController;
        let scoped_effect_controller = crate::ScopedEffectController::borrowed(
            &controller,
            crate::ExecutionScope::turn("child", "child-turn"),
        )
        .expect("turn scope");
        let request = crate::SessionTurnRequest::new(
            "child",
            "child-turn",
            crate::TurnInput::text("run child"),
            scoped_effect_controller,
        )
        .expect("valid child turn request");

        assert_eq!(request.session_id(), "child");
        assert_eq!(request.turn_id(), "child-turn");
        assert_eq!(request.input().trace_turn_id.as_deref(), Some("child-turn"));
    }

    #[test]
    fn session_turn_request_rejects_mismatched_execution_scope() {
        let controller = crate::InlineRuntimeEffectController;
        let scoped_effect_controller = crate::ScopedEffectController::borrowed(
            &controller,
            crate::ExecutionScope::turn("child", "other-turn"),
        )
        .expect("turn scope");
        let err = match crate::SessionTurnRequest::new(
            "child",
            "child-turn",
            crate::TurnInput::text("run child"),
            scoped_effect_controller,
        ) {
            Ok(_) => panic!("mismatched turn scope should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("same id"));
    }
}
