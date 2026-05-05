use super::*;

impl RuntimeTurnDriver {
    pub(super) async fn run_checkpoint(
        &mut self,
        machine: &mut TurnMachine,
        checkpoint: CheckpointKind,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    ) -> Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeError> {
        let mut committed = self
            .session
            .turn_injection_bridge()
            .drain()
            .map_err(|err| RuntimeError {
                code: "turn_injection_bridge".to_string(),
                message: err,
            })?;
        let injected = self
            .session
            .turn_input_injection_bridge()
            .drain()
            .map_err(|err| RuntimeError {
                code: "turn_input_injection_bridge".to_string(),
                message: err,
            })?
            .into_iter()
            .map(|item| item.message)
            .collect::<Vec<_>>();
        let plugins = Arc::clone(self.session.plugins());
        let applied = plugins
            .apply_checkpoint(CheckpointHookContext {
                session_id: self.session_id.clone(),
                checkpoint,
                state: self.checkpoint_state_view(machine.message_sequence(), machine.iteration()),
                host: self.session_manager.clone(),
            })
            .await
            .map_err(|err| RuntimeError {
                code: "plugin_checkpoint".to_string(),
                message: err.to_string(),
            })?;
        if !injected.is_empty() {
            send_session_event(
                event_tx,
                SessionEvent::InjectedTurnInputAccepted {
                    messages: injected.clone(),
                    checkpoint,
                },
            )
            .await;
        }
        committed.extend(applied.messages);
        emit_session_events(event_tx, applied.events).await;
        if let Some(abort) = applied.abort {
            return Err(RuntimeError {
                code: abort.code,
                message: abort.message,
            });
        }

        if !committed.is_empty() {
            send_session_event(
                event_tx,
                SessionEvent::InjectedMessagesCommitted {
                    messages: committed.clone(),
                    checkpoint,
                },
            )
            .await;
        }

        Ok((committed, injected))
    }

    pub(super) async fn prepare_provider(
        &mut self,
        policy: &mut SessionPolicy,
    ) -> Result<String, SessionEvent> {
        match policy.provider.ensure_fresh().await {
            Ok(true) => {
                if let Some(path) = self.host.core.credential_store_path.as_ref() {
                    let _ = crate::provider::save_provider(path, &policy.provider);
                }
            }
            Err(e) => {
                return Err(make_error_event(
                    "token_refresh",
                    Some("refresh_failed"),
                    format!(
                        "Token refresh failed: {}. Re-authenticate with /provider and retry.",
                        e
                    ),
                    Some(e.to_string()),
                ));
            }
            _ => {}
        }

        let model = policy.provider.resolve_model(&policy.model);
        match policy.provider.ensure_ready().await {
            Ok(changed) => {
                if changed && let Some(path) = self.host.core.credential_store_path.as_ref() {
                    let _ = crate::provider::save_provider(path, &policy.provider);
                }
            }
            Err(e) => {
                return Err(make_error_event(
                    "llm_provider",
                    e.code.as_deref(),
                    format!(
                        "LLM provider initialization failed: {}. Run /provider to reconfigure credentials, then retry.",
                        e.message
                    ),
                    e.raw,
                ));
            }
        }

        Ok(model)
    }

    pub(super) async fn run_exec_code(
        &mut self,
        code: &str,
        messages: crate::MessageSequence,
        iteration: usize,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    ) -> Result<crate::ExecResponse, String> {
        let (session_event_tx, mut session_event_rx) = mpsc::channel::<SessionEvent>(100);
        let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel::<SandboxMessage>();
        self.session.set_message_sender(msg_tx);
        let relay_tx = event_tx.clone();
        let relay_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    maybe_sandbox = msg_rx.recv() => {
                        let Some(sandbox_msg) = maybe_sandbox else {
                            // Sandbox channel closed; drain remaining session events.
                            while let Some(event) = session_event_rx.recv().await {
                                send_session_event(&relay_tx, event).await;
                            }
                            break;
                        };
                        if sandbox_msg.kind != "final" && !relay_tx.is_closed() {
                            let _ = relay_tx
                                .send(RuntimeStreamEvent::Session(SessionEvent::Message {
                                    text: sandbox_msg.text,
                                    kind: sandbox_msg.kind,
                                }))
                                .await;
                        }
                    }
                    maybe_event = session_event_rx.recv() => {
                        let Some(event) = maybe_event else {
                            // Session channel closed; drain remaining sandbox messages.
                            while let Some(sandbox_msg) = msg_rx.recv().await {
                                if sandbox_msg.kind != "final" && !relay_tx.is_closed() {
                                    let _ = relay_tx
                                        .send(RuntimeStreamEvent::Session(SessionEvent::Message {
                                            text: sandbox_msg.text,
                                            kind: sandbox_msg.kind,
                                        }))
                                        .await;
                                }
                            }
                            break;
                        };
                        send_session_event(&relay_tx, event).await;
                    }
                }
            }
        });
        let manager = self.session_manager.clone();
        let mode_session = Arc::clone(self.session.plugins().mode_session());
        let read_view = self.checkpoint_state_view(messages, iteration);
        let rlm_globals = read_view.shared_rlm_globals();
        let rlm_chronological_projection = Arc::new(read_view.chronological_projection());
        let context = self.session.mode_execution_context(
            &self.session_id,
            manager,
            session_event_tx.clone(),
            rlm_globals,
            rlm_chronological_projection,
        );
        let result = mode_session
            .execute_code(
                context,
                crate::ExecRequest {
                    code: code.to_string(),
                    accept_finish: true,
                },
            )
            .await
            .map_err(|e| e.to_string());
        drop(session_event_tx);
        self.session.clear_message_sender();
        let _ = relay_handle.await;
        result
    }
}
