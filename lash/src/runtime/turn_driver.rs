use super::*;
use crate::{ModePreamble, PluginError, ToolSurface};
use lash_sansio::{PreparedPrompt, PromptCache, PromptLayer};

mod effects;
mod streaming;
mod tools;

async fn send_session_event(event_tx: &mpsc::Sender<RuntimeStreamEvent>, event: SessionEvent) {
    if !event_tx.is_closed() {
        match &event {
            SessionEvent::TokenUsage {
                mode_iteration,
                usage,
                cumulative,
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::Usage {
                        mode_iteration: *mode_iteration,
                        usage: usage.clone(),
                        cumulative: cumulative.clone(),
                    },
                )
                .await;
            }
            // ChildTokenUsage is projected to TurnEvent::ChildUsage at its
            // origin in `session_manager::usage::ChildUsageEventRelay::emit`,
            // not here. Child usage events bypass `send_session_event` because
            // they're produced by the session manager rather than the parent's
            // turn driver.
            SessionEvent::LlmRequest { mode_iteration, .. } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::ModelRequestStarted {
                        mode_iteration: *mode_iteration,
                    },
                )
                .await;
            }
            SessionEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
                ..
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::RetryStatus {
                        wait_seconds: *wait_seconds,
                        attempt: *attempt,
                        max_attempts: *max_attempts,
                        reason: reason.clone(),
                    },
                )
                .await;
            }
            SessionEvent::PluginEvent { plugin_id, event } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::PluginSurface {
                        plugin_id: plugin_id.clone(),
                        event: event.clone(),
                    },
                )
                .await;
            }
            SessionEvent::InjectedTurnInputAccepted { inputs, checkpoint } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::QueuedInputAccepted {
                        checkpoint: *checkpoint,
                        inputs: inputs.clone(),
                    },
                )
                .await;
            }
            SessionEvent::InjectedMessagesCommitted {
                messages,
                checkpoint,
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::QueuedMessagesCommitted {
                        messages: messages.clone(),
                        checkpoint: *checkpoint,
                    },
                )
                .await;
            }
            SessionEvent::Error { message, .. } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::Error {
                        message: message.clone(),
                    },
                )
                .await;
            }
            SessionEvent::TurnOutcome {
                outcome: TurnOutcome::Finished(TurnFinish::SubmittedValue { value }),
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::SubmittedValue {
                        value: value.clone(),
                    },
                )
                .await;
            }
            SessionEvent::TurnOutcome {
                outcome: TurnOutcome::Finished(TurnFinish::ToolValue { tool_name, value }),
            } => {
                send_independent_turn_event(
                    event_tx,
                    TurnEvent::ToolValue {
                        tool_name: tool_name.clone(),
                        value: value.clone(),
                    },
                )
                .await;
            }
            _ => {}
        }
        let _ = event_tx.send(RuntimeStreamEvent::Session(event)).await;
    }
}

async fn send_turn_activity(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    correlation_id: TurnActivityId,
    event: TurnEvent,
) {
    if !event_tx.is_closed() {
        let activity = TurnActivity::new(correlation_id, event);
        let _ = event_tx.send(RuntimeStreamEvent::Turn(activity)).await;
    }
}

async fn send_independent_turn_event(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    event: TurnEvent,
) {
    send_turn_activity(event_tx, TurnActivityId::fresh(), event).await;
}

async fn emit_semantic_response_parts(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    response: &LlmResponse,
    strip_lashlang_fences: bool,
) {
    let has_text_correlation_ids = response.parts.iter().any(|part| {
        matches!(
            part,
            LlmOutputPart::Text {
                response_meta: Some(meta),
                ..
            } if meta.id.is_some()
        )
    });
    let mut emitted_text = false;
    for part in &response.parts {
        match part {
            LlmOutputPart::Text {
                text,
                response_meta,
            } if has_text_correlation_ids && !text.is_empty() => {
                let text = semantic_text_part(text, strip_lashlang_fences);
                if text.is_empty() {
                    continue;
                }
                emitted_text = true;
                let correlation_id = response_meta
                    .as_ref()
                    .and_then(|meta| meta.id.clone())
                    .map(TurnActivityId::new)
                    .unwrap_or_else(TurnActivityId::fresh);
                send_turn_activity(
                    event_tx,
                    correlation_id,
                    TurnEvent::AssistantProseDelta {
                        text: text.to_string(),
                    },
                )
                .await;
            }
            LlmOutputPart::Reasoning { text, item_id, .. } if !text.is_empty() => {
                let correlation_id = item_id
                    .clone()
                    .map(TurnActivityId::new)
                    .unwrap_or_else(TurnActivityId::fresh);
                send_turn_activity(
                    event_tx,
                    correlation_id,
                    TurnEvent::ReasoningDelta { text: text.clone() },
                )
                .await;
            }
            _ => {}
        }
    }
    let full_text = semantic_text_part(&response.full_text, strip_lashlang_fences);
    let fallback_text;
    let full_text = if full_text.is_empty() && !has_text_correlation_ids {
        fallback_text = semantic_text_part(
            &response_text_from_parts(&response.parts),
            strip_lashlang_fences,
        )
        .to_string();
        fallback_text.as_str()
    } else {
        full_text
    };
    if !emitted_text && !full_text.is_empty() {
        send_independent_turn_event(
            event_tx,
            TurnEvent::AssistantProseDelta {
                text: full_text.to_string(),
            },
        )
        .await;
    }
}

fn semantic_text_part(text: &str, strip_lashlang_fences: bool) -> &str {
    if strip_lashlang_fences {
        prose_before_lashlang_fence(text)
    } else {
        text
    }
}

fn response_text_from_parts(parts: &[LlmOutputPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            LlmOutputPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn prose_before_lashlang_fence(text: &str) -> &str {
    let Some(open) = text.find("```") else {
        return text;
    };
    let after_ticks = open + 3;
    let rest = &text[after_ticks..];
    let Some(newline) = rest.find('\n') else {
        return text;
    };
    if rest[..newline].trim() == "lashlang" {
        return text[..open].trim_end();
    }
    text
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
    pub(super) mode_turn_options: crate::ModeTurnOptions,
    pub(super) mode_extension: Option<crate::ModeTurnExtensionHandle>,
    pub(super) turn_context: crate::TurnContext,
    pub(super) turn_phase_probe: Option<Arc<dyn RuntimeTurnPhaseProbe>>,
}

struct PreparedExecutionSurface {
    execution_mode: ExecutionMode,
    tool_surface: Arc<ToolSurface>,
    mode_preamble: Arc<ModePreamble>,
    prompt: PromptLayer,
}

impl PreparedExecutionSurface {
    fn build_prompt(
        &self,
        core_prompt: &PromptLayer,
        session_prompt: &PromptLayer,
        turn_prompt: &PromptLayer,
        prompt_cache: Option<Arc<PromptCache>>,
    ) -> PreparedPrompt {
        let mut capability_prompt = PromptLayer::new();
        for contribution in self.mode_preamble.prompt_contributions.iter().cloned() {
            capability_prompt.add_contribution(contribution);
        }
        let resolved = crate::resolve_prompt_layers([
            &capability_prompt,
            &self.prompt,
            core_prompt,
            session_prompt,
            turn_prompt,
        ]);
        let prompt_contributions = self
            .tool_surface
            .filter_prompt_contributions(resolved.contributions);
        lash_sansio::build_prompt_cached(
            crate::PromptBuildInput {
                mode: self.execution_mode.clone(),
                template: resolved.template,
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

    fn emit_mode_diagnostic_trace(
        &self,
        mode_iteration: usize,
        phase: &str,
        payload: serde_json::Value,
    ) {
        let mode_event = crate::ModeEvent::typed(
            self.policy.execution_mode.clone(),
            serde_json::json!({
                "diagnostic": {
                    "phase": phase,
                    "payload": payload,
                }
            }),
        )
        .expect("mode diagnostic event serializes");
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
        async {
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
                        match self.before_llm_call(&machine, &request).await {
                            Ok(Some(crate::ModeLlmCallAction::Handoff { session_id })) => {
                                machine.finish_with_outcome(crate::TurnOutcome::Handoff {
                                    session_id,
                                });
                                continue;
                            }
                            Ok(None) => {}
                            Err(err) => {
                                machine.fail_turn(make_error_event(
                                    "rlm_forced_context_fallback",
                                    Some("forced_fallback_failed"),
                                    err.to_string(),
                                    Some(err.to_string()),
                                ));
                                continue;
                            }
                        }
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
                            if !text_streamed {
                                emit_semantic_response_parts(
                                    &event_tx,
                                    response,
                                    self.policy.execution_mode.plugin_id() == "rlm",
                                )
                                .await;
                            }
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
                                    result: outcome.result.result.clone(),
                                    success: outcome.result.success,
                                    duration_ms: outcome.duration_ms,
                                    control: outcome.result.control.clone(),
                                };
                                self.emit_tool_call_trace(machine.mode_iteration(), &record);
                            }
                        }
                        self.turn_pipeline
                            .record_tool_calls(results.iter().map(|outcome| ToolCallRecord {
                                call_id: Some(outcome.call_id.clone()),
                                tool: outcome.tool_name.clone(),
                                args: outcome.args.clone(),
                                result: outcome.result.result.clone(),
                                success: outcome.result.success,
                                duration_ms: outcome.duration_ms,
                                control: outcome.result.control.clone(),
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
                        let code_correlation_id = TurnActivityId::new(format!("code:{id:?}"));
                        let iteration = machine.mode_iteration();
                        if self.host.core.trace_sink.is_some() {
                            self.emit_mode_diagnostic_trace(
                                iteration,
                                "exec_code_started",
                                serde_json::json!({
                                    "code": code,
                                    "code_chars": code.chars().count(),
                                }),
                            );
                        }
                        send_turn_activity(
                            &event_tx,
                            code_correlation_id.clone(),
                            TurnEvent::CodeBlockStarted {
                                language: "lashlang".to_string(),
                                code: code.clone(),
                            },
                        )
                        .await;
                        let exec_started_at = std::time::Instant::now();
                        let result = self
                            .run_exec_code(&code, machine.message_sequence(), iteration, &event_tx)
                            .await;
                        match &result {
                            Ok(output) => {
                                send_turn_activity(
                                    &event_tx,
                                    code_correlation_id.clone(),
                                    TurnEvent::CodeBlockCompleted {
                                        language: "lashlang".to_string(),
                                        output: output.output.clone(),
                                        error: output.error.clone(),
                                        success: output.error.is_none(),
                                        duration_ms: output.duration_ms,
                                    },
                                )
                                .await;
                            }
                            Err(error) => {
                                send_turn_activity(
                                    &event_tx,
                                    code_correlation_id.clone(),
                                    TurnEvent::CodeBlockCompleted {
                                        language: "lashlang".to_string(),
                                        output: String::new(),
                                        error: Some(error.clone()),
                                        success: false,
                                        duration_ms: exec_started_at.elapsed().as_millis() as u64,
                                    },
                                )
                                .await;
                            }
                        }
                        if let Ok(output) = &result {
                            if self.host.core.trace_sink.is_some() {
                                self.emit_mode_diagnostic_trace(
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
                                if !output.observation_truncation.is_empty() {
                                    self.emit_mode_diagnostic_trace(
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
                            self.emit_mode_diagnostic_trace(
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
        .await
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
        let mut mode_prompt = PromptLayer::new();
        for contribution in execution_surface
            .mode_preamble
            .prompt_contributions
            .iter()
            .cloned()
        {
            mode_prompt.add_contribution(contribution);
        }
        let resolved_prompt = crate::resolve_prompt_layers([
            &mode_prompt,
            &execution_surface.prompt,
            &self.host.core.prompt,
            &session_policy.prompt,
            self.turn_context.prompt_layer(),
        ]);
        let mut mode_preamble = (*execution_surface.mode_preamble).clone();
        mode_preamble.prompt_contributions.clear();
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
            mode_preamble: Arc::new(mode_preamble),
            prompt_template: resolved_prompt.template,
            prompt_contributions: resolved_prompt.contributions,
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
            &self.host.core.prompt,
            &policy.prompt,
            self.turn_context.prompt_layer(),
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
                turn_context: self.turn_context.clone(),
            })
            .await?;
        let mut prompt = PromptLayer::new();
        for contribution in self.session.context_prompt_contributions().iter().cloned() {
            prompt.add_contribution(contribution);
        }
        for contribution in plugin_prompt_contributions {
            prompt.add_contribution(contribution);
        }
        if let Some(extension) = &self.mode_extension {
            for contribution in extension.prompt_contributions() {
                prompt.add_contribution(contribution);
            }
        }
        Ok(PreparedExecutionSurface {
            execution_mode,
            tool_surface,
            mode_preamble,
            prompt,
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

    async fn before_llm_call(
        &mut self,
        machine: &TurnMachine,
        request: &LlmRequest,
    ) -> Result<Option<crate::ModeLlmCallAction>, PluginError> {
        let latest_prompt_usage = self.turn_pipeline.state_mut().last_prompt_usage.clone();
        self.session
            .plugins()
            .mode_session()
            .before_llm_call(
                crate::ModeBeforeLlmCallContext {
                    session_id: self.session_id.clone(),
                    host: self.session_manager.clone(),
                    state: self.checkpoint_state_view(
                        machine.message_sequence(),
                        machine.mode_iteration(),
                    ),
                    latest_prompt_usage,
                },
                request,
            )
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

    #[test]
    fn mode_step_trace_event_preserves_mode_payload() {
        let mode_event = crate::ModeEvent::typed(
            crate::ExecutionMode::new("custom"),
            serde_json::json!({
                "code": "print \"hi\"",
                "final_output": "done"
            }),
        )
        .expect("mode event");

        let lash_trace::TraceEvent::ModeStep { mode, payload } = mode_step_trace_event(&mode_event)
        else {
            panic!("expected mode step trace event");
        };

        assert_eq!(mode, "custom");
        assert_eq!(
            payload.get("code").and_then(serde_json::Value::as_str),
            Some("print \"hi\"")
        );
        assert_eq!(
            payload.get("final_output"),
            Some(&serde_json::json!("done"))
        );
    }
}
