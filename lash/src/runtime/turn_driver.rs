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
    pub(super) turn_id: String,
    pub(super) turn_index: usize,
    pub(super) turn_pipeline: TurnCommitPipeline,
    pub(super) llm_stream_summaries: HashMap<usize, LlmStreamSummary>,
    pub(super) next_llm_ordinal: usize,
    pub(super) session_manager: Arc<dyn RuntimeSessionHost>,
    pub(super) prompt_bridge: HostPromptBridge,
    pub(super) mode_turn_options: crate::ModeTurnOptions,
    pub(super) mode_extension: Option<crate::ModeTurnExtensionHandle>,
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
    fn trace_context(&self, mode_iteration: usize) -> lash_trace::TraceContext {
        lash_trace::TraceContext::default()
            .for_session(self.session_id.clone())
            .for_turn_index(self.turn_index)
            .for_mode_iteration(mode_iteration)
            .for_turn(self.turn_id.clone())
    }

    fn llm_call_id(&mut self, mode_iteration: usize) -> String {
        let ordinal = self.next_llm_ordinal;
        self.next_llm_ordinal += 1;
        format!(
            "{}:{}:{}:{}",
            self.session_id, self.turn_index, mode_iteration, ordinal
        )
    }

    fn emit_trace(&self, mode_iteration: usize, event: lash_trace::TraceEvent) {
        crate::trace::emit_trace(
            &self.host.core.trace_sink,
            &self.host.core.trace_context,
            self.trace_context(mode_iteration),
            event,
        );
    }

    fn emit_trace_at(
        &self,
        mode_iteration: usize,
        event: lash_trace::TraceEvent,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        crate::trace::emit_trace_at(
            &self.host.core.trace_sink,
            &self.host.core.trace_context,
            self.trace_context(mode_iteration),
            event,
            timestamp,
        );
    }

    fn emit_tool_call_trace_at(
        &self,
        mode_iteration: usize,
        record: &crate::ToolCallRecord,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        self.emit_trace_at(
            mode_iteration,
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
        mode_iteration: usize,
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        self.emit_trace_at(
            mode_iteration,
            lash_trace::TraceEvent::ToolCallStarted {
                call_id,
                name,
                args,
            },
            timestamp,
        );
    }

    fn emit_tool_call_trace(&self, mode_iteration: usize, record: &crate::ToolCallRecord) {
        self.emit_trace(
            mode_iteration,
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
        mode_iteration: usize,
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
    ) {
        self.emit_trace(
            mode_iteration,
            lash_trace::TraceEvent::ToolCallStarted {
                call_id,
                name,
                args,
            },
        );
    }

    async fn persist_progress_boundary(
        &mut self,
        messages: crate::MessageSequence,
        events: Arc<Vec<crate::SessionEventRecord>>,
        mode_iteration: usize,
    ) {
        if !crate::messages_are_prompt_resume_safe(messages.iter()) {
            return;
        }
        let boundary = self
            .turn_pipeline
            .progress_boundary(
                &mut self.session,
                self.policy.clone(),
                self.turn_index,
                messages,
                events,
            )
            .await;
        if boundary.persisted {
            for event in &boundary.mirrored_events {
                self.emit_mode_event_trace(mode_iteration, event);
            }
        }
    }

    fn mark_phase_begin(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.begin(phase);
        }
    }

    fn emit_mode_event_trace(&self, mode_iteration: usize, event: &crate::SessionEventRecord) {
        let crate::SessionEventRecord::Mode(mode_event) = event else {
            return;
        };
        self.emit_trace(mode_iteration, mode_step_trace_event(mode_event));
    }

    fn emit_rlm_diagnostic_trace(
        &self,
        mode_iteration: usize,
        phase: &str,
        payload: serde_json::Value,
    ) {
        let mode_event = crate::ModeEvent::rlm(lash_rlm_types::RlmModeEvent::RlmDiagnostic(
            lash_rlm_types::RlmDiagnosticEvent {
                phase: phase.to_string(),
                payload,
            },
        ));
        self.emit_trace(mode_iteration, mode_step_trace_event(&mode_event));
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
                        mode_iteration,
                    } => {
                        self.persist_progress_boundary(messages, events, mode_iteration)
                            .await
                    }
                    Effect::Done {
                        messages,
                        events: _,
                        mode_iteration,
                    } => return (messages, mode_iteration),
                    Effect::LlmCall { id, request } => {
                        if cancel.is_cancelled() {
                            emit!(SessionEvent::Done);
                            return (crate::MessageSequence::default(), run_offset);
                        }
                        let iteration = machine.mode_iteration();
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
                                    machine.mode_iteration(),
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
                                    control: outcome.state_result.control.clone(),
                                };
                                self.emit_tool_call_trace(machine.mode_iteration(), &record);
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
                                control: outcome.state_result.control.clone(),
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
                        let iteration = machine.mode_iteration();
                        if self.host.core.trace_sink.is_some() {
                            self.emit_rlm_diagnostic_trace(
                                iteration,
                                "exec_code_started",
                                serde_json::json!({
                                    "code": code,
                                    "code_chars": code.chars().count(),
                                }),
                            );
                        }
                        let exec_started_at = chrono::Utc::now();
                        let result = self
                            .run_exec_code(&code, machine.message_sequence(), iteration, &event_tx)
                            .await;
                        if let Ok(output) = &result {
                            if self.host.core.trace_sink.is_some() {
                                self.emit_rlm_diagnostic_trace(
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
                                    self.emit_rlm_diagnostic_trace(
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
                            self.emit_rlm_diagnostic_trace(
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
                self.turn_index,
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
            mode_run_offset: run_offset,
            tool_surface: execution_surface.tool_surface,
            mode_preamble: execution_surface.mode_preamble,
            prompt_template: self.host.core.prompt_template.clone(),
            prompt_contributions: {
                let mut contributions = self.host.core.prompt_contributions.clone();
                contributions.extend(execution_surface.prompt_contributions);
                contributions
            },
            max_turns: session_policy.max_turns,
            model_variant: session_policy.model_variant.clone(),
            emit_llm_trace: false,
            termination: self.mode_turn_options.clone(),
            prompt_cache: Some(self.session.prompt_cache()),
        });
        if self.host.core.trace_sink.is_some() {
            let prompt_hash =
                lash_trace::sha256_hex(prepared.prepared_prompt.system_prompt.as_bytes());
            let prompt_chars = prepared.prepared_prompt.system_prompt.chars().count();
            crate::trace::emit_trace(
                &self.host.core.trace_sink,
                &self.host.core.trace_context,
                self.trace_context(run_offset),
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
                self.turn_index,
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
        turn_index: usize,
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
                    turn_index,
                    self.mode_turn_options.clone(),
                    messages,
                ),
                mode_turn_options: self.mode_turn_options.clone(),
            })
            .await?;
        let mut prompt_contributions = self.session.context_prompt_contributions().to_vec();
        prompt_contributions.extend(plugin_prompt_contributions);
        if let Some(extension) = &self.mode_extension {
            prompt_contributions.extend(extension.prompt_contributions());
        }
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
        mode_iteration: usize,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        let _ = machine;
        let _ = effect_id;
        self.run_standard_llm_call(request, mode_iteration, event_tx, cancel)
            .await
    }

    fn runtime_session_policy(&self) -> SessionPolicy {
        self.policy.clone()
    }

    fn checkpoint_state_view(
        &self,
        messages: crate::MessageSequence,
        _mode_iteration: usize,
    ) -> crate::SessionReadView {
        self.turn_pipeline.read_view(
            self.policy.clone(),
            self.turn_index,
            self.mode_turn_options.clone(),
            messages,
        )
    }
}

fn mode_step_trace_event(mode_event: &crate::ModeEvent) -> lash_trace::TraceEvent {
    lash_trace::TraceEvent::ModeStep {
        mode: mode_event.mode_id.plugin_id().to_string(),
        payload: mode_event.payload.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_rlm_types::{RlmModeEvent, RlmTrajectoryEntry};

    #[test]
    fn mode_step_trace_event_preserves_rlm_trajectory_code() {
        let mode_event =
            crate::ModeEvent::rlm(RlmModeEvent::RlmTrajectoryEntry(RlmTrajectoryEntry {
                id: "rlm_step_7".to_string(),
                mode_iteration: 7,
                reasoning: "inspect".to_string(),
                code: "print \"hi\"".to_string(),
                output: vec!["hi".to_string()],
                tool_calls: Vec::new(),
                images: Vec::new(),
                error: None,
                final_output: Some(serde_json::json!("done")),
            }));

        let lash_trace::TraceEvent::ModeStep { mode, payload } = mode_step_trace_event(&mode_event)
        else {
            panic!("expected mode step trace event");
        };

        assert_eq!(mode, "rlm");
        assert_eq!(
            payload
                .get("RlmTrajectoryEntry")
                .and_then(|entry| entry.get("code"))
                .and_then(serde_json::Value::as_str),
            Some("print \"hi\"")
        );
        assert_eq!(
            payload
                .get("RlmTrajectoryEntry")
                .and_then(|entry| entry.get("final_output")),
            Some(&serde_json::json!("done"))
        );
    }
}
