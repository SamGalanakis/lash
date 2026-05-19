use super::*;

impl RuntimeTurnDriver<'_> {
    pub(in crate::runtime) async fn run_checkpoint(
        &mut self,
        machine: &mut TurnMachine,
        checkpoint: CheckpointKind,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    ) -> Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeError> {
        let mut committed = self
            .session
            .turn_injection_bridge()
            .drain()
            .map_err(|err| RuntimeError::new(RuntimeErrorCode::TurnInjectionBridge, err))?;
        let injected = self
            .session
            .turn_input_injection_bridge()
            .drain()
            .map_err(|err| RuntimeError::new(RuntimeErrorCode::TurnInputInjectionBridge, err))?;
        let injected_messages = injected
            .iter()
            .map(|item| item.message.clone())
            .collect::<Vec<_>>();
        let plugins = Arc::clone(self.session.plugins());
        let applied = plugins
            .apply_checkpoint(CheckpointHookContext {
                session_id: self.session_id.clone(),
                checkpoint,
                state: self
                    .checkpoint_state_view(machine.message_sequence(), machine.mode_iteration()),
                host: self.session_manager.clone(),
            })
            .await
            .map_err(|err| {
                RuntimeError::new(RuntimeErrorCode::PluginCheckpoint, err.to_string())
            })?;
        if !injected.is_empty() {
            send_session_event(
                event_tx,
                SessionEvent::InjectedTurnInputAccepted {
                    inputs: injected
                        .iter()
                        .cloned()
                        .map(|item| crate::AcceptedInjectedTurnInput {
                            id: item.id,
                            message: item.message,
                        })
                        .collect(),
                    checkpoint,
                },
            )
            .await;
        }
        committed.extend(applied.messages);
        emit_session_events(event_tx, applied.events).await;
        if let Some(abort) = applied.abort {
            return Err(RuntimeError::new(abort.code, abort.message));
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

        Ok((committed, injected_messages))
    }

    pub(super) async fn prepare_provider(
        &mut self,
        policy: &mut SessionPolicy,
    ) -> Result<String, SessionEvent> {
        let model = policy.provider.resolve_model(&policy.model);
        if let Some(variant) = policy.model_variant.as_deref()
            && let Err(message) = policy.provider.validate_variant(&model, variant)
        {
            return Err(make_error_event(
                "llm_provider",
                Some("invalid_model_variant"),
                message.clone(),
                Some(message),
            ));
        }
        Ok(model)
    }

    pub(in crate::runtime) async fn run_exec_code(
        &mut self,
        code: &str,
        messages: crate::MessageSequence,
        mode_iteration: usize,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    ) -> Result<crate::ExecResponse, String> {
        let (session_event_tx, mut session_event_rx) = mpsc::channel::<SessionEvent>(100);
        let (turn_event_tx, mut turn_event_rx) = mpsc::channel::<TurnActivity>(100);
        let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel::<SandboxMessage>();
        self.session.set_message_sender(msg_tx);
        let relay_tx = event_tx.clone();
        let relay_handle = tokio::spawn(async move {
            let mut sandbox_closed = false;
            let mut session_closed = false;
            let mut turn_closed = false;
            while !(sandbox_closed && session_closed && turn_closed) {
                tokio::select! {
                    biased;
                    maybe_sandbox = msg_rx.recv(), if !sandbox_closed => {
                        let Some(sandbox_msg) = maybe_sandbox else {
                            sandbox_closed = true;
                            continue;
                        };
                        if sandbox_msg.kind != "lashlang_code" && !relay_tx.is_closed() {
                            let _ = relay_tx
                                .send(RuntimeStreamEvent::Session(SessionEvent::Message {
                                    text: sandbox_msg.text,
                                    kind: sandbox_msg.kind,
                                }))
                                .await;
                        }
                    }
                    maybe_event = session_event_rx.recv(), if !session_closed => {
                        let Some(event) = maybe_event else {
                            session_closed = true;
                            continue;
                        };
                        send_session_event(&relay_tx, event).await;
                    }
                    maybe_turn_event = turn_event_rx.recv(), if !turn_closed => {
                        let Some(event) = maybe_turn_event else {
                            turn_closed = true;
                            continue;
                        };
                        let _ = relay_tx.send(RuntimeStreamEvent::Turn(event)).await;
                    }
                }
            }
        });
        let manager = self.session_manager.clone();
        let mode_session = Arc::clone(self.session.plugins().mode_session());
        let read_view = self.checkpoint_state_view(messages, mode_iteration);
        let chronological_projection = read_view.shared_chronological_projection();
        let effect_controller =
            crate::runtime::RuntimeEffectControllerHandle::borrowed(self.effect_scope.controller());
        let direct_completions = manager.direct_completion_client(
            effect_controller.clone_scoped(),
            Some(self.turn_id.clone()),
            self.turn_lease.clone(),
        );
        let context = self
            .session
            .mode_execution_context(
                &self.session_id,
                manager.clone() as Arc<dyn crate::plugin::RuntimeSessionHost>,
                effect_controller,
                Arc::clone(&self.host.core.effect_controller),
                direct_completions,
                session_event_tx.clone(),
                chronological_projection,
                self.mode_extension.clone(),
                self.turn_context.clone(),
            )
            .map_err(|err| err.to_string())?
            .with_turn_event_sender(turn_event_tx.clone());
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
        drop(turn_event_tx);
        self.session.clear_message_sender();
        let _ = relay_handle.await;
        result
    }
}
