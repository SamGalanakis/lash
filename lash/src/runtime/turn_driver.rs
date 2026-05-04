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
    pub(super) turn_pipeline: TurnCommitPipeline,
    pub(super) llm_stream_summaries: HashMap<usize, LlmStreamSummary>,
    pub(super) session_manager: Arc<dyn RuntimeSessionHost>,
    pub(super) prompt_bridge: HostPromptBridge,
    pub(super) mode_turn_options: crate::ModeTurnOptions,
    pub(super) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}

struct PreparedExecutionSurface {
    execution_mode: ExecutionMode,
    tool_surface: Arc<ToolSurface>,
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
    fn emit_trace(&self, iteration: usize, event: lash_trace::TraceEvent) {
        crate::trace::emit_trace(
            &self.host.core.trace_sink,
            &self.host.core.trace_context,
            lash_trace::TraceContext::default()
                .for_session(self.session_id.clone())
                .for_iteration(iteration),
            event,
        );
    }

    fn emit_trace_at(
        &self,
        iteration: usize,
        event: lash_trace::TraceEvent,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        crate::trace::emit_trace_at(
            &self.host.core.trace_sink,
            &self.host.core.trace_context,
            lash_trace::TraceContext::default()
                .for_session(self.session_id.clone())
                .for_iteration(iteration),
            event,
            timestamp,
        );
    }

    fn emit_tool_call_trace_at(
        &self,
        iteration: usize,
        record: &crate::ToolCallRecord,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        self.emit_trace_at(
            iteration,
            lash_trace::TraceEvent::ToolCallCompleted {
                call_id: record.call_id.clone(),
                name: record.tool.clone(),
                args: record.args.clone(),
                result: record.result.clone(),
                success: record.success,
                duration_ms: record.duration_ms,
            },
            timestamp,
        );
    }

    fn emit_tool_call_started_trace_at(
        &self,
        iteration: usize,
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        self.emit_trace_at(
            iteration,
            lash_trace::TraceEvent::ToolCallStarted {
                call_id,
                name,
                args,
            },
            timestamp,
        );
    }

    fn emit_tool_call_trace(&self, iteration: usize, record: &crate::ToolCallRecord) {
        self.emit_trace(
            iteration,
            lash_trace::TraceEvent::ToolCallCompleted {
                call_id: record.call_id.clone(),
                name: record.tool.clone(),
                args: record.args.clone(),
                result: record.result.clone(),
                success: record.success,
                duration_ms: record.duration_ms,
            },
        );
    }

    fn emit_tool_call_started_trace(
        &self,
        iteration: usize,
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
    ) {
        self.emit_trace(
            iteration,
            lash_trace::TraceEvent::ToolCallStarted {
                call_id,
                name,
                args,
            },
        );
    }

    fn emit_rlm_exec_step(&self, iteration: usize, phase: &str, payload: serde_json::Value) {
        let mut payload = payload;
        if let Some(object) = payload.as_object_mut() {
            object.insert("phase".to_string(), serde_json::json!(phase));
        }
        self.emit_trace(
            iteration,
            lash_trace::TraceEvent::ModeStep {
                mode: "rlm".to_string(),
                payload,
            },
        );
    }

    async fn persist_progress_boundary(
        &mut self,
        messages: crate::MessageSequence,
        events: Arc<Vec<crate::SessionEventRecord>>,
        iteration: usize,
    ) {
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return;
        }
        let boundary = self
            .turn_pipeline
            .progress_boundary(
                &mut self.session,
                self.policy.clone(),
                iteration,
                messages,
                events,
            )
            .await;
        if boundary.persisted {
            for event in &boundary.mirrored_events {
                self.emit_mode_event_trace(iteration, event);
            }
        }
    }

    fn mark_phase_begin(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.begin(phase);
        }
    }

    fn emit_mode_event_trace(&self, iteration: usize, event: &crate::SessionEventRecord) {
        if !self.host.core.trace_level.is_extended() {
            return;
        }
        let crate::SessionEventRecord::Mode(mode_event) = event else {
            return;
        };
        if let Some(lash_rlm_types::RlmModeEvent::RlmDiagnostic(diagnostic)) =
            mode_event.rlm_event()
        {
            self.emit_rlm_exec_step(iteration, &diagnostic.phase, diagnostic.payload);
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
                    Effect::Emit(event) => {
                        if let SessionEvent::TokenUsage {
                            usage, cumulative, ..
                        } = &event
                        {
                            self.turn_pipeline.state_mut().token_usage = cumulative.clone();
                            self.turn_pipeline.state_mut().last_prompt_usage =
                                normalize_prompt_usage(&self.policy.provider, usage);
                        }
                        emit!(event)
                    }
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
                        if let Ok(response) = &result {
                            let usage = TokenUsage {
                                input_tokens: response.usage.input_tokens,
                                output_tokens: response.usage.output_tokens,
                                cached_input_tokens: response.usage.cached_input_tokens,
                                reasoning_tokens: response.usage.reasoning_tokens,
                            };
                            self.turn_pipeline.state_mut().last_prompt_usage =
                                normalize_prompt_usage(&self.policy.provider, &usage);
                        }
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
                        if self.host.core.trace_sink.is_some() {
                            for pending in &calls {
                                self.emit_tool_call_started_trace(
                                    machine.iteration(),
                                    Some(pending.call_id.clone()),
                                    pending.tool_name.clone(),
                                    pending.args.clone(),
                                );
                            }
                        }
                        let results = self.run_tool_calls(calls, &event_tx, &cancel).await;
                        if self.host.core.trace_sink.is_some() {
                            for outcome in &results {
                                let record = ToolCallRecord {
                                    call_id: Some(outcome.call_id.clone()),
                                    tool: outcome.tool_name.clone(),
                                    args: outcome.args.clone(),
                                    result: outcome.state_result.result.clone(),
                                    success: outcome.state_result.success,
                                    duration_ms: outcome.duration_ms,
                                };
                                self.emit_tool_call_trace(machine.iteration(), &record);
                            }
                        }
                        self.turn_pipeline
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
                        let iteration = machine.iteration();
                        if self.host.core.trace_sink.is_some() {
                            self.emit_rlm_exec_step(
                                iteration,
                                "exec_code_started",
                                serde_json::json!({
                                    "code": code,
                                    "code_chars": code.chars().count(),
                                }),
                            );
                        }
                        let exec_started_at = chrono::Utc::now();
                        let result = self.run_exec_code(&code, &event_tx).await;
                        if let Ok(output) = &result {
                            if self.host.core.trace_sink.is_some() {
                                self.emit_rlm_exec_step(
                                    iteration,
                                    "exec_code_completed",
                                    serde_json::json!({
                                        "duration_ms": output.duration_ms,
                                        "output": output.output,
                                        "output_chars": output.output.chars().count(),
                                        "observation_count": output.observations.len(),
                                        "observation_truncation": output.observation_truncation,
                                        "error": output.error,
                                        "terminal_finish": output.terminal_finish,
                                        "terminal_finish_present": output.terminal_finish.is_some(),
                                        "tool_call_count": output.tool_calls.len(),
                                    }),
                                );
                                // Reconstruct synthetic per-tool timestamps
                                // by walking the cumulative durations forward
                                // from when exec started. Tools fire
                                // sequentially within a lashlang block, so
                                // this is a close approximation to actual
                                // wall-clock timing — far better than
                                // stamping every event with the post-exec
                                // emission time.
                                let mut cursor = exec_started_at;
                                for record in &output.tool_calls {
                                    let started_at = cursor;
                                    let completed_at = started_at
                                        + chrono::Duration::milliseconds(record.duration_ms as i64);
                                    self.emit_tool_call_started_trace_at(
                                        iteration,
                                        record.call_id.clone(),
                                        record.tool.clone(),
                                        record.args.clone(),
                                        started_at,
                                    );
                                    self.emit_tool_call_trace_at(iteration, record, completed_at);
                                    cursor = completed_at;
                                }
                                if !output.observation_truncation.is_empty() {
                                    self.emit_rlm_exec_step(
                                        iteration,
                                        "observation_projection",
                                        serde_json::json!({
                                            "projections": output.observation_truncation,
                                        }),
                                    );
                                }
                            }
                            self.turn_pipeline
                                .record_tool_calls(output.tool_calls.iter().cloned());
                        } else if let Err(error) = &result
                            && self.host.core.trace_sink.is_some()
                        {
                            self.emit_rlm_exec_step(
                                iteration,
                                "exec_code_failed",
                                serde_json::json!({ "error": error }),
                            );
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
            events: self.turn_pipeline.active_events(),
            run_offset,
            tool_surface: execution_surface.tool_surface,
            mode_preamble: execution_surface.mode_preamble,
            prompt_template: self.host.core.prompt_template.clone(),
            prompt_contributions: execution_surface.prompt_contributions,
            max_turns: session_policy.max_turns,
            model_variant: session_policy.model_variant.clone(),
            emit_llm_trace: false,
            termination: self.mode_turn_options.clone(),
            retry_policy: self.host.core.retry_policy.clone(),
            prompt_cache: Some(self.session.prompt_cache()),
        });
        if self.host.core.trace_sink.is_some() {
            let prompt_hash =
                lash_trace::sha256_hex(prepared.prepared_prompt.system_prompt.as_bytes());
            let prompt_chars = prepared.prepared_prompt.system_prompt.chars().count();
            crate::trace::emit_trace(
                &self.host.core.trace_sink,
                &self.host.core.trace_context,
                lash_trace::TraceContext::default()
                    .for_session(self.session_id.clone())
                    .for_iteration(run_offset),
                lash_trace::TraceEvent::PromptBuilt {
                    prompt_hash: prompt_hash.clone(),
                    prompt_chars,
                    components: vec![lash_trace::TracePromptComponent {
                        id: "system_prompt".to_string(),
                        kind: "rendered_prompt".to_string(),
                        hash: prompt_hash,
                        chars: Some(prompt_chars),
                    }],
                },
            );
        }
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
        let plugin_prompt_contributions = self
            .session
            .plugins()
            .collect_prompt_contributions(PromptHookContext {
                session_id: self.session_id.clone(),
                host: self.session_manager.clone(),
                state: self.turn_pipeline.read_view(
                    session_policy.clone(),
                    iteration,
                    self.mode_turn_options.clone(),
                    messages,
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
        self.turn_pipeline.read_view(
            self.policy.clone(),
            iteration,
            self.mode_turn_options.clone(),
            messages,
        )
    }
}
