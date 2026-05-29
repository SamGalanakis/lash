use super::*;

impl ManagedSessionCapability {
    pub(in crate::runtime::session_manager) async fn start_turn(
        &self,
        current: &CurrentSessionCapability,
        usage: &UsageCapability,
        session_id: &str,
        input: TurnInput,
    ) -> Result<AssembledTurn, crate::PluginError> {
        if self
            .turns
            .lock()
            .await
            .values()
            .any(|turn| turn.session_id == session_id)
        {
            return Err(crate::PluginError::Session(format!(
                "session `{session_id}` already has a running turn"
            )));
        }
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let policy = {
            let runtime = runtime.runtime.lock().await;
            runtime.session_policy()
        };
        let turn_id = uuid::Uuid::new_v4().to_string();
        let cancel = CancellationToken::new();
        let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(100);
        let usage_source = self.child_usage_source(usage, session_id);
        let runtime_clone = runtime.clone();
        let cancel_clone = cancel.clone();
        let sink = ChannelEventSink {
            tx: event_tx,
            live_usage: Some(LiveChildUsageForwarder {
                turn_id: turn_id.clone(),
                session_id: session_id.to_string(),
                source: usage_source,
                model: policy.model.id.clone(),
                token_ledger: Arc::clone(&usage.token_ledger),
                child_turn_live_usage: Arc::clone(&usage.child_turn_live_usage),
                relay: usage.child_usage_event_relay.clone(),
            }),
        };
        let event_drain = tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
        let task = tokio::spawn(async move {
            let mut runtime = runtime_clone.runtime.lock().await;
            runtime
                .refresh_session_tool_surface()
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
            let run = runtime
                .stream_turn_with_agent_frames(
                    input,
                    crate::runtime::TurnOptions::new(cancel_clone).with_events(&sink),
                )
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
            let turn = run.into_final_turn().ok_or_else(|| {
                crate::PluginError::Session("agent frame run completed without a turn".to_string())
            })?;
            runtime_clone.publish_from(&runtime);
            Ok(turn)
        });
        self.turns.lock().await.insert(
            turn_id.clone(),
            ManagedSessionTurn {
                session_id: session_id.to_string(),
                task,
            },
        );
        let managed = self
            .turns
            .lock()
            .await
            .remove(&turn_id)
            .ok_or_else(|| crate::PluginError::Session(format!("unknown turn `{turn_id}`")))?;
        let session_id = managed.session_id.clone();
        let turn = managed
            .task
            .await
            .map_err(|err| crate::PluginError::Session(format!("turn task failed: {err}")))?;
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
