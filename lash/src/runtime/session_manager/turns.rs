use super::*;

impl RuntimeSessionManager {
    pub(in crate::runtime::session_manager) async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, crate::PluginError> {
        if self
            .managed
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
            let registry = self.managed.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let policy = {
            let runtime = runtime.lock().await;
            runtime.session_policy()
        };
        let turn_id = uuid::Uuid::new_v4().to_string();
        let cancel = CancellationToken::new();
        let (event_tx, event_rx) = mpsc::channel::<SessionEvent>(100);
        let usage_source = self
            .usage
            .child_sources
            .lock()
            .expect("child usage sources lock")
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| "child".to_string());
        let runtime_clone = Arc::clone(&runtime);
        let cancel_clone = cancel.clone();
        let sink = ChannelEventSink {
            tx: event_tx,
            live_usage: Some(LiveChildUsageForwarder {
                turn_id: turn_id.clone(),
                session_id: session_id.to_string(),
                source: usage_source,
                model: policy.model.clone(),
                token_ledger: Arc::clone(&self.usage.token_ledger),
                child_turn_live_usage: Arc::clone(&self.usage.child_turn_live_usage),
                relay: self.usage.child_usage_event_relay.clone(),
            }),
        };
        let task = tokio::spawn(async move {
            let mut runtime = runtime_clone.lock().await;
            runtime
                .refresh_session_tool_surface()
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))?;
            runtime
                .stream_turn(input, &sink, cancel_clone)
                .await
                .map_err(|err| crate::PluginError::Session(err.to_string()))
        });
        self.managed.turns.lock().await.insert(
            turn_id.clone(),
            ManagedSessionTurn {
                session_id: session_id.to_string(),
                cancel,
                task,
            },
        );
        Ok(crate::plugin::SessionTurnHandle {
            turn_id,
            session_id: session_id.to_string(),
            policy,
            events: event_rx,
        })
    }

    pub(in crate::runtime::session_manager) async fn await_turn(
        &self,
        turn_id: &str,
    ) -> Result<AssembledTurn, crate::PluginError> {
        let managed = self
            .managed
            .turns
            .lock()
            .await
            .remove(turn_id)
            .ok_or_else(|| crate::PluginError::Session(format!("unknown turn `{turn_id}`")))?;
        let session_id = managed.session_id.clone();
        let turn = managed
            .task
            .await
            .map_err(|err| crate::PluginError::Session(format!("turn task failed: {err}")))?;
        let live_reported = self
            .usage
            .child_turn_live_usage
            .lock()
            .expect("child turn live usage lock")
            .remove(turn_id)
            .unwrap_or_default();
        if let Ok(turn) = &turn {
            let source = self
                .usage
                .child_sources
                .lock()
                .expect("child usage sources lock")
                .get(&session_id)
                .cloned()
                .unwrap_or_else(|| "child".to_string());
            if let Some(remainder) = subtract_usage(&live_reported, &turn.token_usage) {
                self.record_token_usage(&source, &turn.state.policy.model, &remainder);
            }
        }
        self.persist_current_usage_ledger().await?;
        turn
    }

    pub(in crate::runtime::session_manager) async fn cancel_turn(
        &self,
        turn_id: &str,
    ) -> Result<(), crate::PluginError> {
        let turns = self.managed.turns.lock().await;
        let managed = turns
            .get(turn_id)
            .ok_or_else(|| crate::PluginError::Session(format!("unknown turn `{turn_id}`")))?;
        managed.cancel.cancel();
        Ok(())
    }
}
