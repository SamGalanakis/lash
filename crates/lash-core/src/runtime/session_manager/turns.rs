use super::*;

impl ManagedSessionCapability {
    pub(in crate::runtime::session_manager) async fn start_turn(
        &self,
        current: &CurrentSessionCapability,
        usage: &UsageCapability,
        session_id: &str,
        turn_id: &str,
        mut input: TurnInput,
        scoped_effect_controller: crate::ScopedEffectController<'_>,
    ) -> Result<AssembledTurn, crate::PluginError> {
        if turn_id.trim().is_empty() {
            return Err(crate::PluginError::Session(
                "session turns require a non-empty stable turn id".to_string(),
            ));
        }
        if scoped_effect_controller.turn_id() != Some(turn_id) {
            return Err(crate::PluginError::Session(format!(
                "session turn `{turn_id}` requires an effect turn scope with the same id"
            )));
        }
        if scoped_effect_controller.effect_scope().session_id() != Some(session_id) {
            return Err(crate::PluginError::Session(format!(
                "session turn `{turn_id}` requires an effect scope for session `{session_id}`"
            )));
        }
        if let Some(input_turn_id) = input.trace_turn_id.as_deref() {
            if input_turn_id != turn_id {
                return Err(crate::PluginError::Session(format!(
                    "input trace_turn_id `{input_turn_id}` does not match turn id `{turn_id}`"
                )));
            }
        }
        input.trace_turn_id = Some(turn_id.to_string());
        let runtime = {
            let registry = self.registry.lock().await;
            registry.get(session_id).cloned()
        }
        .ok_or_else(|| crate::PluginError::Session(format!("unknown session `{session_id}`")))?;
        let policy = {
            let runtime = runtime.runtime.lock().await;
            runtime.session_policy()
        };
        let cancel = CancellationToken::new();
        let (event_tx, mut event_rx) = mpsc::channel::<SessionEvent>(100);
        let usage_source = self.child_usage_source(usage, session_id);
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
            if turns.values().any(|turn| turn.session_id == session_id) {
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
            let result = async {
                runtime_guard
                    .refresh_session_tool_surface()
                    .await
                    .map_err(|err| crate::PluginError::Session(err.to_string()))?;
                let run = runtime_guard
                    .stream_turn_with_agent_frames(
                        input,
                        crate::runtime::TurnOptions::new(cancel)
                            .with_events(&sink)
                            .with_scoped_effect_controller(scoped_effect_controller),
                    )
                    .await
                    .map_err(|err| crate::PluginError::Session(err.to_string()))?;
                let turn = run.into_final_turn().ok_or_else(|| {
                    crate::PluginError::Session(
                        "agent frame run completed without a turn".to_string(),
                    )
                })?;
                Ok(turn)
            }
            .await;
            runtime.publish_from(&runtime_guard);
            result
        };
        self.turns.lock().await.remove(turn_id);
        drop(sink);
        let _ = event_drain.await;
        let live_reported = self.turn_live_usage(usage, turn_id);
        if let Ok(turn) = &turn {
            let source = self.child_usage_source(usage, session_id);
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
    fn managed_child_turns_use_caller_supplied_scope_and_turn_id() {
        let source = include_str!("turns.rs");
        assert!(source.contains(".with_scoped_effect_controller(scoped_effect_controller)"));
        let generated_turn_id_call = concat!("uuid::Uuid::", "new_v4");
        assert!(!source.contains(generated_turn_id_call));
    }
}
