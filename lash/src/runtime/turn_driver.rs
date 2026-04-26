use super::*;
use crate::{ModePreamble, PluginError, ToolSurface};
use lash_sansio::{PreparedPrompt, PromptCache, PromptContribution, PromptTemplate};

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
    pub(super) progress_graph: TurnGraphOverlay,
    pub(super) progress_state: PersistedSessionState,
    pub(super) llm_stream_summaries: HashMap<usize, LlmStreamSummary>,
    pub(super) session_manager: Arc<dyn SessionManager>,
    pub(super) prompt_bridge: HostPromptBridge,
    pub(super) mode_turn_options: crate::ModeTurnOptions,
    pub(super) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}

struct PreparedExecutionSurface {
    execution_mode: ExecutionMode,
    tool_surface: ToolSurface,
    mode_preamble: Arc<ModePreamble>,
    prompt_contributions: Vec<PromptContribution>,
}

impl PreparedExecutionSurface {
    fn build_prompt(
        &self,
        template: PromptTemplate,
        prompt_cache: Option<Arc<PromptCache>>,
    ) -> PreparedPrompt {
        let mut prompt_contributions = self.mode_preamble.prompt_contributions.clone();
        prompt_contributions.extend(self.prompt_contributions.iter().cloned());
        let prompt_contributions = self
            .tool_surface
            .filter_prompt_contributions(prompt_contributions);
        lash_sansio::build_prompt_cached(
            crate::PromptBuildInput {
                mode: self.execution_mode.clone(),
                template,
                execution_prompt: self.mode_preamble.execution_prompt.clone(),
                tool_names: self.mode_preamble.tool_names.clone(),
                omitted_tool_count: self.mode_preamble.omitted_tool_count,
                contributions: prompt_contributions,
            },
            prompt_cache.as_deref(),
        )
    }
}

impl RuntimeTurnDriver {
    async fn persist_progress_boundary(
        &mut self,
        messages: crate::MessageSequence,
        events: Arc<Vec<crate::SessionEventRecord>>,
        iteration: usize,
    ) {
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return;
        }
        let Some(store) = self.session.history_store() else {
            return;
        };

        self.progress_state.policy = self.policy.clone();
        self.progress_state.iteration = iteration;
        let existing_events = self.progress_graph.active_events_arc().len();
        if events.len() > existing_events {
            self.progress_graph
                .append_events(events[existing_events..].iter().cloned());
        }
        if let Some(appended_messages) = self
            .progress_graph
            .message_delta_if_current_preserved(messages.iter())
        {
            self.progress_graph
                .append_projected_conversation_messages(&appended_messages);
        } else {
            let projected_messages = messages.shared();
            let tool_calls = self.progress_graph.tool_calls_arc();
            self.progress_graph
                .replace_projection(projected_messages.as_slice(), tool_calls.as_slice());
        }

        if let Ok(snapshot) = self.session.snapshot_execution_state().await {
            self.progress_state.set_execution_state_snapshot(snapshot);
        }
        let plugins = self.session.plugins();
        self.progress_state
            .refresh_plugin_snapshots(plugins.as_ref());

        let graph = self.progress_graph.graph_commit(
            self.progress_state.persisted_graph_node_count,
            self.progress_state.graph_replace_required,
        );
        let commit = crate::store::PersistedStateCommit::persisted_state_with_graph_commit(
            &self.progress_state,
            graph,
            &[],
        );
        let result = match store.apply_runtime_commit(commit).await {
            Ok(result) => result,
            Err(err) => {
                tracing::warn!("failed to persist runtime progress boundary: {err}");
                return;
            }
        };
        self.progress_state.apply_persisted_commit_result(result);
    }

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
                    Effect::Progress {
                        messages,
                        events,
                        iteration,
                    } => {
                        self.persist_progress_boundary(messages, events, iteration)
                            .await
                    }
                    Effect::Done {
                        messages,
                        events: _,
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
                    Effect::SyncExecutionSurface {
                        id,
                        update_machine_config,
                    } => {
                        let result = self
                            .refresh_execution_surface(&machine, update_machine_config)
                            .await
                            .map_err(|err| err.to_string());
                        machine.handle_response(Response::ExecutionSurfaceSynced { id, result });
                    }
                    Effect::ToolCalls { id, calls } => {
                        let results = self.run_tool_calls(calls, &event_tx, &cancel).await;
                        self.progress_graph
                            .record_tool_calls(results.iter().map(|outcome| ToolCallRecord {
                                call_id: Some(outcome.call_id.clone()),
                                tool: outcome.tool_name.clone(),
                                args: outcome.args.clone(),
                                result: outcome.state_result.result.clone(),
                                success: outcome.state_result.success,
                                duration_ms: outcome.duration_ms,
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
                        if let Ok(output) = &result {
                            self.progress_graph
                                .record_tool_calls(output.tool_calls.iter().cloned());
                        }
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

        let execution_mode = self.policy.execution_mode.clone();
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
        let execution_surface = match self
            .prepare_execution_surface(
                execution_mode,
                &session_policy,
                run_offset,
                messages.clone(),
            )
            .await
        {
            Ok(surface) => surface,
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
        let prepared = crate::build_turn(crate::SansIoTurnInput {
            session_id: self.session_id.clone(),
            run_session_id: session_policy.session_id.clone(),
            autonomous: session_policy.autonomous,
            model,
            mode: execution_surface.execution_mode,
            messages,
            events: self.progress_state.shared_active_events(),
            run_offset,
            tool_surface: execution_surface.tool_surface,
            mode_preamble: execution_surface.mode_preamble,
            prompt_template: self.host.core.prompt_template.clone(),
            prompt_contributions: execution_surface.prompt_contributions,
            max_turns: session_policy.max_turns,
            model_variant: session_policy.model_variant.clone(),
            emit_llm_debug_log: self.host.core.llm_logger.is_some(),
            termination: self.mode_turn_options.clone(),
            retry_policy: self.host.core.retry_policy.clone(),
            prompt_cache: Some(self.session.prompt_cache()),
        });
        self.policy = session_policy;
        self.mark_phase_end(RuntimeTurnPhase::PromptBuild);
        Ok(prepared.machine)
    }

    async fn refresh_execution_surface(
        &mut self,
        machine: &crate::TurnMachine,
        update_machine_config: bool,
    ) -> Result<Option<crate::sansio::ExecutionSurfaceSync>, crate::SessionError> {
        self.session.refresh_tool_surface().await?;
        if !update_machine_config {
            return Ok(None);
        }

        let policy = self.policy.clone();
        let execution_surface = self
            .prepare_execution_surface(
                policy.execution_mode.clone(),
                &policy,
                machine.iteration(),
                machine.message_sequence(),
            )
            .await
            .map_err(|err| crate::SessionError::Protocol(err.to_string()))?;
        let prepared_prompt = execution_surface.build_prompt(
            self.host.core.prompt_template.clone(),
            Some(self.session.prompt_cache()),
        );

        Ok(Some(crate::sansio::ExecutionSurfaceSync {
            system_prompt: prepared_prompt.system_prompt,
            tool_specs: execution_surface.mode_preamble.tool_specs.clone(),
        }))
    }

    async fn prepare_execution_surface(
        &mut self,
        execution_mode: ExecutionMode,
        session_policy: &SessionPolicy,
        iteration: usize,
        messages: crate::MessageSequence,
    ) -> Result<PreparedExecutionSurface, PluginError> {
        let tool_surface = self
            .session
            .tool_surface(&self.session_id, execution_mode.clone());
        let mode_preamble = self
            .session
            .mode_preamble(&self.session_id, execution_mode.clone());
        let prompt_state = SessionStateEnvelope {
            session_id: self.session_id.clone(),
            policy: session_policy.clone(),
            iteration,
            ..Default::default()
        };
        let plugin_prompt_contributions = self
            .session
            .plugins()
            .collect_prompt_contributions(PromptHookContext {
                session_id: self.session_id.clone(),
                host: Arc::clone(&self.session_manager),
                state: crate::SessionReadView::from_graph_message_sequence(
                    &prompt_state,
                    self.progress_graph.base_graph(),
                    messages,
                    self.progress_graph.tool_calls_arc(),
                ),
                mode_turn_options: self.mode_turn_options.clone(),
            })
            .await?;
        let mut prompt_contributions = self.session.context_prompt_contributions().to_vec();
        prompt_contributions.extend(plugin_prompt_contributions);
        Ok(PreparedExecutionSurface {
            execution_mode,
            tool_surface,
            mode_preamble,
            prompt_contributions,
        })
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
        let _ = machine;
        let _ = effect_id;
        self.run_standard_llm_call(request, iteration, event_tx, cancel)
            .await
    }

    fn runtime_session_policy(&self) -> SessionPolicy {
        self.policy.clone()
    }

    fn checkpoint_state_view(
        &self,
        messages: crate::MessageSequence,
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
        crate::SessionReadView::from_graph_message_sequence(
            &state,
            self.progress_graph.base_graph(),
            messages,
            self.progress_graph.tool_calls_arc(),
        )
    }
}
