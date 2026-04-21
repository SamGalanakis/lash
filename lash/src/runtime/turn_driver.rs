use super::*;

mod effects;
mod streaming;
mod tools;

pub(in crate::runtime) use streaming::llm_response_has_content;

async fn send_session_event(event_tx: &mpsc::Sender<RuntimeStreamEvent>, event: SessionEvent) {
    if !event_tx.is_closed() {
        let _ = event_tx.send(RuntimeStreamEvent::Session(event)).await;
    }
}

pub(super) struct RuntimeTurnDriver {
    pub(super) session: Session,
    pub(super) policy: SessionPolicy,
    pub(super) host: RuntimeHost,
    pub(super) session_id: String,
    pub(super) base_graph: crate::SessionGraph,
    pub(super) tool_calls: Arc<Vec<ToolCallRecord>>,
    pub(super) llm_stream_summaries: HashMap<usize, LlmStreamSummary>,
    pub(super) session_manager: Arc<dyn SessionManager>,
    pub(super) prompt_bridge: HostPromptBridge,
    pub(super) rlm_termination: crate::RlmTermination,
    pub(super) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}

impl RuntimeTurnDriver {
    fn mark_phase_begin(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.begin(phase);
        }
    }

    fn mark_phase_end(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.end(phase);
        }
    }

    pub(super) async fn run(
        &mut self,
        messages: crate::MessageSequence,
        event_tx: mpsc::Sender<RuntimeStreamEvent>,
        cancel: CancellationToken,
        run_offset: usize,
    ) -> (crate::MessageSequence, usize) {
        macro_rules! emit {
            ($event:expr) => {
                send_session_event(&event_tx, $event).await
            };
        }
        let result = async {
            let mut machine = match self
                .prepare_turn_machine(messages, &event_tx, run_offset)
                .await
            {
                Ok(machine) => machine,
                Err(result) => return result,
            };
            loop {
                let Some(effect) = machine.poll_effect() else {
                    break;
                };
                match effect {
                    Effect::Emit(event) => emit!(event),
                    Effect::Done {
                        messages,
                        iteration,
                    } => return (messages, iteration),
                    Effect::LlmCall { id, request } => {
                        if cancel.is_cancelled() {
                            emit!(SessionEvent::Done);
                            return (crate::MessageSequence::default(), run_offset);
                        }
                        let iteration = machine.iteration();
                        let (result, text_streamed) = self
                            .run_llm_call(&mut machine, id, request, iteration, &event_tx, &cancel)
                            .await;
                        machine.handle_response(Response::LlmComplete {
                            id,
                            result,
                            text_streamed,
                        });
                    }
                    Effect::Checkpoint { id, checkpoint } => {
                        match self
                            .run_checkpoint(&mut machine, checkpoint, &event_tx)
                            .await
                        {
                            Ok((messages, transient_messages)) => {
                                machine.handle_response(Response::Checkpoint {
                                    id,
                                    messages,
                                    transient_messages,
                                });
                            }
                            Err(err) => {
                                machine.fail_turn(make_error_event(
                                    "plugin",
                                    Some(&err.code),
                                    err.message,
                                    None,
                                ));
                            }
                        }
                    }
                    Effect::SyncExecutionSurface { id } => {
                        let result = self
                            .session
                            .refresh_tool_surface()
                            .await
                            .map_err(|err| err.to_string());
                        machine.handle_response(Response::ExecutionSurfaceSynced { id, result });
                    }
                    Effect::ToolCalls { id, calls } => {
                        let results = self.run_tool_calls(calls, &event_tx, &cancel).await;
                        Arc::make_mut(&mut self.tool_calls).extend(results.iter().map(|outcome| {
                            ToolCallRecord {
                                call_id: Some(outcome.call_id.clone()),
                                tool: outcome.tool_name.clone(),
                                args: outcome.args.clone(),
                                result: outcome.state_result.result.clone(),
                                success: outcome.state_result.success,
                                duration_ms: outcome.duration_ms,
                            }
                        }));
                        machine.handle_response(Response::ToolResults { id, results });
                    }
                    Effect::Sleep { id, duration } => {
                        tokio::time::sleep(duration).await;
                        machine.handle_response(Response::Timeout { id });
                    }
                    Effect::Log { event } => self.handle_log_event(event),
                    Effect::CancelLlm { .. } => {}
                    Effect::ExecCode { id, code } => {
                        let result = self.run_exec_code(&code, &event_tx).await;
                        let response = match result {
                            Ok(output) => Response::ExecResult {
                                id,
                                result: Ok(output),
                            },
                            Err(error) => Response::ExecResult {
                                id,
                                result: Err(error),
                            },
                        };
                        machine.handle_response(response);
                    }
                }
            }

            (crate::MessageSequence::default(), run_offset)
        }
        .await;
        self.prompt_bridge.clear_sender();
        result
    }

    async fn prepare_turn_machine(
        &mut self,
        messages: crate::MessageSequence,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        run_offset: usize,
    ) -> Result<TurnMachine, (crate::MessageSequence, usize)> {
        macro_rules! emit {
            ($event:expr) => {
                send_session_event(event_tx, $event).await
            };
        }

        let execution_mode = self.policy.execution_mode;
        let mut session_policy = self.runtime_session_policy();
        let model = match self.prepare_provider(&mut session_policy).await {
            Ok(model) => model,
            Err(event) => {
                emit!(event);
                emit!(SessionEvent::Done);
                return Err((messages.clone(), run_offset));
            }
        };
        self.mark_phase_begin(RuntimeTurnPhase::PromptBuild);
        let tool_surface = self.session.tool_surface(&self.session_id, execution_mode);
        let mode_preamble = self.session.mode_preamble(&self.session_id, execution_mode);
        let prompt_state = SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: session_policy.clone(),
            iteration: run_offset,
            ..Default::default()
        };
        let plugin_prompt_contributions = match self
            .session
            .plugins()
            .collect_prompt_contributions(PromptHookContext {
                session_id: self.session_id.clone(),
                host: Arc::clone(&self.session_manager),
                state: crate::SessionReadView::from_graph_projection(
                    &prompt_state,
                    self.base_graph.clone(),
                    messages.shared(),
                    Arc::clone(&self.tool_calls),
                ),
                rlm_termination: self.rlm_termination.clone(),
            })
            .await
        {
            Ok(contributions) => contributions,
            Err(err) => {
                emit!(make_error_event(
                    "plugin_prompt",
                    None,
                    err.to_string(),
                    Some(err.to_string()),
                ));
                emit!(SessionEvent::Done);
                return Err((messages, run_offset));
            }
        };
        let mut all_prompt_contributions = self.session.context_prompt_contributions().to_vec();
        all_prompt_contributions.extend(plugin_prompt_contributions);
        let prepared = crate::build_turn(crate::SansIoTurnInput {
            session_id: self.session_id.clone(),
            run_session_id: session_policy.session_id.clone(),
            model,
            mode: execution_mode,
            messages,
            run_offset,
            mode_preamble,
            tool_surface,
            prompt_template: self.host.core.prompt_template.clone(),
            prompt_contributions: all_prompt_contributions,
            max_turns: session_policy.max_turns,
            model_variant: session_policy.model_variant.clone(),
            emit_llm_debug_log: self.host.core.llm_logger.is_some(),
            rlm_termination: self.rlm_termination.clone(),
            retry_policy: self.host.core.retry_policy.clone(),
        });
        self.policy = session_policy;
        self.mark_phase_end(RuntimeTurnPhase::PromptBuild);
        Ok(prepared.machine)
    }

    async fn run_llm_call(
        &mut self,
        machine: &mut TurnMachine,
        effect_id: crate::sansio::EffectId,
        request: LlmRequest,
        iteration: usize,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        match self.policy.execution_mode {
            ExecutionMode::Standard => {
                self.run_standard_llm_call(request, iteration, event_tx, cancel)
                    .await
            }
            ExecutionMode::Rlm => {
                let _ = machine;
                let _ = effect_id;
                self.run_standard_llm_call(request, iteration, event_tx, cancel)
                    .await
            }
        }
    }

    fn runtime_session_policy(&self) -> SessionPolicy {
        self.policy.clone()
    }

    fn checkpoint_state_view(
        &self,
        messages: Arc<Vec<Message>>,
        iteration: usize,
    ) -> crate::SessionReadView {
        let state = SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: self.policy.clone(),
            session_graph: crate::SessionGraph::default(),
            iteration,
            token_usage: TokenUsage::default(),
            last_prompt_usage: None,
        };
        crate::SessionReadView::from_graph_projection(
            &state,
            self.base_graph.clone(),
            messages,
            Arc::clone(&self.tool_calls),
        )
    }
}
