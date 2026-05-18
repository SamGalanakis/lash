use std::sync::Arc;

use lash_trace::{TraceError, TraceEvent, TraceProviderStreamEvent, TraceRuntimeStreamEvent};

use super::*;

/// Result of running stream hooks over a visible chunk. Carries both
/// the (possibly rewritten) text and an `abort_requested` flag that the
/// LLM runner uses to break the stream early when a plugin has decided
/// the response is complete (e.g. the RLM mask detecting a closed
/// lashlang fence).
pub(super) struct StreamChunkOutcome {
    pub(super) chunk: String,
    pub(super) reasoning_deltas: Vec<String>,
    pub(super) abort_requested: bool,
}

async fn emit_plugin_runtime_events_runtime(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    plugin_id: &str,
    events: Vec<crate::PluginRuntimeEvent>,
) {
    for event in crate::plugin::plugin_runtime_session_events(plugin_id, events) {
        send_session_event(event_tx, event).await;
    }
}

fn refine_terminal_reason_for_context_window(
    response: &mut LlmResponse,
    max_context_tokens: Option<usize>,
) {
    if response.terminal_reason != crate::LlmTerminalReason::OutputLimit {
        return;
    }
    if response.usage.output_tokens != 0 {
        return;
    }
    let Some(max_context_tokens) = max_context_tokens.filter(|value| *value > 0) else {
        return;
    };
    let prompt_tokens = response
        .usage
        .input_tokens
        .saturating_add(response.usage.cached_input_tokens)
        .max(0) as usize;
    if prompt_tokens >= max_context_tokens.saturating_mul(95) / 100 {
        response.terminal_reason = crate::LlmTerminalReason::ContextOverflow;
        response.terminal_diagnostic = Some(
            "Model produced no output because the prompt reached the configured context window."
                .to_string(),
        );
    }
}

impl RuntimeTurnDriver<'_> {
    async fn transform_assistant_stream_chunk(
        &mut self,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        chunk: String,
    ) -> Result<StreamChunkOutcome, LlmCallError> {
        if !self.session.plugins().has_assistant_stream_hooks() {
            return Ok(StreamChunkOutcome {
                chunk,
                reasoning_deltas: Vec::new(),
                abort_requested: false,
            });
        }

        let original = chunk.clone();
        let transforms = self
            .session
            .plugins()
            .transform_assistant_stream(&self.session_id, chunk)
            .await
            .map_err(|err| LlmCallError {
                message: err.to_string(),
                retryable: false,
                raw: None,
                code: Some("plugin_assistant_stream".to_string()),
                terminal_reason: crate::LlmTerminalReason::ProviderError,
                request_body: None,
            })?;
        let mut current = String::new();
        let mut first = true;
        let mut abort_requested = false;
        let mut reasoning_deltas = Vec::new();
        for emitted in transforms {
            if first {
                first = false;
            }
            current = emitted.value.chunk.clone();
            reasoning_deltas.extend(emitted.value.reasoning_deltas.clone());
            if emitted.value.abort_stream {
                abort_requested = true;
            }
            emit_plugin_runtime_events_runtime(event_tx, &emitted.plugin_id, emitted.value.events)
                .await;
        }
        let chunk = if first { original } else { current };
        Ok(StreamChunkOutcome {
            chunk,
            reasoning_deltas,
            abort_requested,
        })
    }

    async fn transform_assistant_response(
        &mut self,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        response: LlmResponse,
    ) -> Result<LlmResponse, LlmCallError> {
        let original = response.clone();
        let transforms = self
            .session
            .plugins()
            .transform_assistant_response(&self.session_id, response)
            .await
            .map_err(|err| LlmCallError {
                message: err.to_string(),
                retryable: false,
                raw: None,
                code: Some("plugin_assistant_response".to_string()),
                terminal_reason: crate::LlmTerminalReason::ProviderError,
                request_body: None,
            })?;
        let mut current: Option<LlmResponse> = None;
        for emitted in transforms {
            emit_plugin_runtime_events_runtime(event_tx, &emitted.plugin_id, emitted.value.events)
                .await;
            current = Some(emitted.value.response);
        }
        Ok(current.unwrap_or(original))
    }

    pub(in crate::runtime) async fn run_standard_llm_call(
        &mut self,
        request: Arc<LlmRequest>,
        mode_iteration: usize,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        let request = (*request).clone();
        let request = match crate::attachments::resolve_llm_request_attachments(
            request,
            self.host.core.attachment_store.as_ref(),
        ) {
            Ok(request) => request,
            Err(err) => {
                return (
                    Err(LlmCallError {
                        message: err.to_string(),
                        retryable: false,
                        raw: None,
                        code: Some("attachment_resolution_failed".to_string()),
                        terminal_reason: crate::LlmTerminalReason::ProviderError,
                        request_body: None,
                    }),
                    false,
                );
            }
        };
        let trace_enabled = self.host.core.trace_sink.is_some();
        let llm_call_id = trace_enabled.then(|| self.llm_call_id(mode_iteration));
        if let Some(llm_call_id) = llm_call_id.as_ref() {
            crate::runtime::effect_host::emit_llm_trace_started(
                &self.host.core.trace_sink,
                &self.host.core.trace_context,
                self.trace_context(mode_iteration)
                    .for_llm_call(llm_call_id.clone()),
                &request,
            );
        }
        let (llm_stream_tx, mut llm_stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
        let mut debug = LlmStreamDebugState::new();
        let provider_trace =
            self.provider_trace_sender(mode_iteration, llm_call_id.clone(), &debug);
        let llm_request = LlmRequest {
            stream_events: transport_stream_events(&self.policy.provider, Some(llm_stream_tx)),
            provider_trace,
            ..request
        };

        let mut call_provider = self.policy.provider.clone();
        let mut llm_task = tokio::spawn(async move {
            let result = call_provider.complete(llm_request).await;
            (result, call_provider)
        });

        let mut text_streamed = false;
        let mut streamed_usage = LlmUsage::default();
        let mut stream_accumulator = StandardStreamAccumulator::default();
        let mut abort_requested = false;
        let mut assistant_prose_correlation = None;
        let mut reasoning_correlation = None;
        let mut stream_state = StandardStreamState {
            text_streamed: &mut text_streamed,
            streamed_usage: &mut streamed_usage,
            stream_accumulator: &mut stream_accumulator,
            debug: &mut debug,
            mode_iteration,
            assistant_prose_correlation: &mut assistant_prose_correlation,
            reasoning_correlation: &mut reasoning_correlation,
            abort_requested: &mut abort_requested,
        };
        let result = loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    llm_task.abort();
                    break Err(LlmCallError {
                        message: "cancelled".to_string(),
                        retryable: false,
                        raw: None,
                        code: Some("cancelled".to_string()),
                        terminal_reason: crate::LlmTerminalReason::Cancelled,
                        request_body: None,
                    });
                }
                Some(stream_event) = llm_stream_rx.recv() => {
                    if let Err(err) = self
                        .forward_standard_stream_event(event_tx, stream_event, &mut stream_state)
                        .await
                    {
                        break Err(err);
                    }
                    if *stream_state.abort_requested {
                        // A plugin stream hook asked us to end the LLM
                        // call now (e.g. RLM mask saw a closed fence).
                        // Drain events already sitting in the buffer, then
                        // wait briefly for the provider's final completion
                        // event. Some providers attach usage there, and the
                        // next prompt's budget contribution depends on the
                        // driver seeing that accounting before the next
                        // mode_iteration starts.
                        if let Err(err) = self
                            .drain_standard_stream_queue(event_tx, &mut llm_stream_rx, &mut stream_state)
                            .await
                        {
                            break Err(err);
                        }
                        streamed_usage = collect_trailing_usage_before_abort(
                            &mut llm_task,
                            &mut llm_stream_rx,
                            streamed_usage.clone(),
                        )
                        .await;

                        let mut resp = LlmResponse {
                            deltas: Vec::new(),
                            full_text: stream_accumulator.full_text(),
                            parts: Vec::new(),
                            usage: streamed_usage.clone(),
                            terminal_reason: crate::LlmTerminalReason::Stop,
                            terminal_diagnostic: None,
                            provider_usage: None,
                            request_body: None,
                            http_summary: None,
                        };
                        stream_accumulator.apply_to_response(&mut resp);
                        let resp = match self.transform_assistant_response(event_tx, resp).await {
                            Ok(resp) => resp,
                            Err(err) => break Err(err),
                        };

                        break Ok(resp);
                    }
                }
                join = &mut llm_task => {
                    let (result, provider_after) = match join {
                        Ok(v) => v,
                        Err(e) => break Err(LlmCallError {
                            message: format!("internal task failed: {e}"),
                            retryable: false,
                            raw: None,
                            code: Some("task_join_failed".to_string()),
                            terminal_reason: crate::LlmTerminalReason::ProviderError,
                            request_body: None,
                        }),
                    };
                    self.policy.provider = provider_after;
                    if let Err(err) = self
                        .drain_standard_stream_queue(event_tx, &mut llm_stream_rx, &mut stream_state)
                        .await
                    {
                        break Err(err);
                    }
                    match result {
                        Ok(mut resp) => {
                            if response_usage_is_empty(&resp.usage) {
                                resp.usage = streamed_usage.clone();
                            }
                            stream_accumulator.apply_to_response(&mut resp);
                            let resp = match self.transform_assistant_response(event_tx, resp).await {
                                Ok(resp) => resp,
                                Err(err) => break Err(err),
                            };
                            break Ok(resp)
                        }
                        Err(e) => break Err(LlmCallError {
                            message: e.message,
                            retryable: e.retryable,
                            raw: e.raw,
                            code: e.code,
                            terminal_reason: e.terminal_reason,
                            request_body: e.request_body,
                        }),
                    }
                }
            }
        };

        let result = result.map(|mut response| {
            refine_terminal_reason_for_context_window(
                &mut response,
                self.policy.max_context_tokens,
            );
            response
        });

        if let Err(err) = &result {
            tracing::error!(
                session_id = %self.session_id,
                turn = mode_iteration,
                retryable = err.retryable,
                code = ?err.code,
                raw_present = err.raw.is_some(),
                request_body_present = err.request_body.is_some(),
                message = %err.message,
                "llm call failed"
            );
        }
        if let Some(llm_call_id) = llm_call_id {
            let stream_summary = debug.summary.to_json();
            match &result {
                Ok(response) => {
                    crate::runtime::effect_host::emit_llm_trace_completed(
                        &self.host.core.trace_sink,
                        &self.host.core.trace_context,
                        self.trace_context(mode_iteration).for_llm_call(llm_call_id),
                        response,
                        debug.elapsed_ms(),
                        Some(stream_summary.clone()),
                    );
                }
                Err(error) => {
                    crate::runtime::effect_host::emit_llm_trace_failed(
                        &self.host.core.trace_sink,
                        &self.host.core.trace_context,
                        self.trace_context(mode_iteration).for_llm_call(llm_call_id),
                        crate::runtime::effect_host::LlmTraceFailure::from(error),
                        Some(stream_summary.clone()),
                    );
                }
            }
        }
        if trace_enabled {
            self.llm_stream_summaries
                .insert(mode_iteration, debug.summary);
        }
        (result, text_streamed)
    }

    pub(super) fn handle_log_event(&mut self, event: crate::sansio::LogEvent) {
        if self.host.core.trace_sink.is_none() {
            return;
        }

        match event {
            crate::sansio::LogEvent::LlmDebug {
                session_id,
                mode_iteration,
                usage,
                provider_usage,
                response_text,
                response_parts,
                ..
            } => {
                let stream_summary = self.llm_stream_summaries.remove(&mode_iteration);
                crate::trace::emit_trace(
                    &self.host.core.trace_sink,
                    &self.host.core.trace_context,
                    self.trace_context(mode_iteration)
                        .for_session(session_id)
                        .for_llm_call(format!(
                            "{}:{}:{}:log",
                            self.session_id, self.turn_index, mode_iteration
                        )),
                    TraceEvent::LlmCallCompleted {
                        response: crate::trace::trace_llm_response(
                            response_text,
                            0,
                            None,
                            response_parts,
                        ),
                        usage: Some(crate::trace::trace_usage_from_session(&usage)),
                        provider_usage,
                        stream_summary: stream_summary.map(|summary| summary.to_json()),
                    },
                );
            }
            crate::sansio::LogEvent::LlmError {
                session_id,
                mode_iteration,
                message,
                retryable,
                raw,
                code,
                terminal_reason,
                ..
            } => {
                let stream_summary = self.llm_stream_summaries.remove(&mode_iteration);
                crate::trace::emit_trace(
                    &self.host.core.trace_sink,
                    &self.host.core.trace_context,
                    self.trace_context(mode_iteration)
                        .for_session(session_id)
                        .for_llm_call(format!(
                            "{}:{}:{}:log",
                            self.session_id, self.turn_index, mode_iteration
                        )),
                    TraceEvent::LlmCallFailed {
                        error: TraceError {
                            message,
                            retryable,
                            terminal_reason: Some(terminal_reason.code().to_string()),
                            code,
                            raw,
                        },
                        stream_summary: stream_summary.map(|summary| summary.to_json()),
                    },
                );
            }
        }
    }

    fn log_llm_stream_event(&self, debug: &mut LlmStreamDebugState, log: LlmStreamEventLog<'_>) {
        if self.host.core.trace_sink.is_none() {
            return;
        }

        let elapsed_ms = debug.elapsed_ms();
        if matches!(log.event_type, "delta" | "text_part") {
            debug
                .summary
                .record_text_chunk(log.text.visible, elapsed_ms);
        }

        if !self.host.core.trace_level.is_extended() {
            return;
        }

        let mut event = TraceRuntimeStreamEvent {
            sequence: debug.next_sequence(),
            elapsed_ms,
            event_name: log.event_type.to_string(),
            raw_text: log.text.raw.map(str::to_string),
            visible_text: log.text.visible.map(str::to_string),
            item_id: log.item_id.map(str::to_string),
            output_index: None,
            call_id: None,
            tool_name: None,
            input_json: None,
            usage: log.usage.map(crate::trace::trace_usage_from_llm),
        };

        if let Some(tool_call) = log.tool_call {
            event.call_id = Some(tool_call.call_id.to_string());
            event.tool_name = Some(tool_call.tool_name.to_string());
            event.input_json = Some(
                serde_json::from_str(tool_call.input_json).unwrap_or_else(|_| {
                    serde_json::Value::String(tool_call.input_json.to_string())
                }),
            );
        }

        crate::trace::emit_trace(
            &self.host.core.trace_sink,
            &self.host.core.trace_context,
            self.trace_context(log.mode_iteration),
            TraceEvent::RuntimeStreamEvent { event },
        );
    }

    fn provider_trace_sender(
        &self,
        mode_iteration: usize,
        llm_call_id: Option<String>,
        debug: &LlmStreamDebugState,
    ) -> Option<LlmProviderTraceSender> {
        if !self.host.core.trace_level.is_extended() || self.host.core.trace_sink.is_none() {
            return None;
        }

        let llm_call_id = llm_call_id?;
        let sink = self.host.core.trace_sink.clone();
        let base_context = self.host.core.trace_context.clone();
        let context = self.trace_context(mode_iteration);
        let created_at = debug.created_at;
        let sequence = Arc::new(std::sync::atomic::AtomicU64::new(0));

        Some(LlmProviderTraceSender::new(
            move |provider_event: LlmProviderTraceEvent| {
                let sequence = sequence.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let raw_json = serde_json::from_str::<serde_json::Value>(&provider_event.raw).ok();
                let item_id = raw_json.as_ref().and_then(provider_item_id);
                let output_index = raw_json.as_ref().and_then(provider_output_index);
                let event = TraceProviderStreamEvent {
                    provider: provider_event.provider.to_string(),
                    sequence,
                    elapsed_ms: created_at.elapsed().as_millis() as u64,
                    event_name: provider_event.event_name,
                    item_id,
                    output_index,
                    raw_len: provider_event.raw.len(),
                    raw_sha256: lash_trace::sha256_hex(provider_event.raw.as_bytes()),
                    raw_json,
                };
                crate::trace::emit_trace(
                    &sink,
                    &base_context,
                    context.clone().for_llm_call(llm_call_id.clone()),
                    TraceEvent::ProviderStreamEvent { event },
                );
            },
        ))
    }

    async fn forward_standard_stream_event(
        &mut self,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        stream_event: LlmStreamEvent,
        state: &mut StandardStreamState<'_>,
    ) -> Result<(), LlmCallError> {
        match stream_event {
            LlmStreamEvent::Delta(delta) => {
                if !delta.is_empty() {
                    *state.text_streamed = true;
                    let raw_delta = self.host.core.trace_sink.as_ref().map(|_| delta.clone());
                    let outcome = self
                        .transform_assistant_stream_chunk(event_tx, delta)
                        .await?;
                    if outcome.abort_requested {
                        *state.abort_requested = true;
                    }
                    for reasoning_delta in outcome.reasoning_deltas {
                        state.stream_accumulator.push_reasoning(
                            reasoning_delta.clone(),
                            None,
                            Vec::new(),
                            None,
                        );
                        send_session_event(
                            event_tx,
                            SessionEvent::ReasoningDelta {
                                content: reasoning_delta.clone(),
                            },
                        )
                        .await;
                        let correlation_id =
                            stream_correlation_id(state.reasoning_correlation, None);
                        send_turn_activity(
                            event_tx,
                            correlation_id,
                            TurnEvent::ReasoningDelta {
                                text: reasoning_delta,
                            },
                        )
                        .await;
                    }
                    let delta = outcome.chunk;
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            mode_iteration: state.mode_iteration,
                            event_type: "delta",
                            text: LlmDebugText {
                                raw: raw_delta.as_deref(),
                                visible: Some(&delta),
                            },
                            item_id: None,
                            usage: None,
                            tool_call: None,
                        },
                    );
                    if !delta.is_empty() {
                        state.stream_accumulator.push_text(&delta);
                        send_session_event(
                            event_tx,
                            SessionEvent::TextDelta {
                                content: delta.clone(),
                            },
                        )
                        .await;
                        let correlation_id =
                            stream_correlation_id(state.assistant_prose_correlation, None);
                        send_turn_activity(
                            event_tx,
                            correlation_id,
                            TurnEvent::AssistantProseDelta { text: delta },
                        )
                        .await;
                    }
                }
            }
            LlmStreamEvent::ReasoningDelta(delta) => {
                if !delta.is_empty() {
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            mode_iteration: state.mode_iteration,
                            event_type: "reasoning_delta",
                            text: LlmDebugText {
                                raw: None,
                                visible: Some(&delta),
                            },
                            item_id: None,
                            usage: None,
                            tool_call: None,
                        },
                    );
                    // Delta-only streaming path (fix 1.3a display). No
                    // encrypted content yet — that arrives with the full
                    // item on `output_item.done` (fix 1.3b).
                    state
                        .stream_accumulator
                        .push_reasoning(delta.clone(), None, Vec::new(), None);
                    send_session_event(
                        event_tx,
                        SessionEvent::ReasoningDelta {
                            content: delta.clone(),
                        },
                    )
                    .await;
                    let correlation_id = stream_correlation_id(state.reasoning_correlation, None);
                    send_turn_activity(
                        event_tx,
                        correlation_id,
                        TurnEvent::ReasoningDelta { text: delta },
                    )
                    .await;
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text,
                response_meta,
            }) => {
                if !text.is_empty() {
                    *state.text_streamed = true;
                    let item_id = response_meta.as_ref().and_then(|meta| meta.id.as_deref());
                    let raw_text = self.host.core.trace_sink.as_ref().map(|_| text.clone());
                    let outcome = self
                        .transform_assistant_stream_chunk(event_tx, text)
                        .await?;
                    if outcome.abort_requested {
                        *state.abort_requested = true;
                    }
                    for reasoning_delta in outcome.reasoning_deltas {
                        state.stream_accumulator.push_reasoning(
                            reasoning_delta.clone(),
                            None,
                            Vec::new(),
                            None,
                        );
                        send_session_event(
                            event_tx,
                            SessionEvent::ReasoningDelta {
                                content: reasoning_delta.clone(),
                            },
                        )
                        .await;
                        let correlation_id =
                            stream_correlation_id(state.reasoning_correlation, None);
                        send_turn_activity(
                            event_tx,
                            correlation_id,
                            TurnEvent::ReasoningDelta {
                                text: reasoning_delta,
                            },
                        )
                        .await;
                    }
                    let text = outcome.chunk;
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            mode_iteration: state.mode_iteration,
                            event_type: "text_part",
                            text: LlmDebugText {
                                raw: raw_text.as_deref(),
                                visible: Some(&text),
                            },
                            item_id,
                            usage: None,
                            tool_call: None,
                        },
                    );
                    if !text.is_empty() {
                        state.stream_accumulator.push_text(&text);
                        send_session_event(
                            event_tx,
                            SessionEvent::TextDelta {
                                content: text.clone(),
                            },
                        )
                        .await;
                        let correlation_id =
                            stream_correlation_id(state.assistant_prose_correlation, item_id);
                        send_turn_activity(
                            event_tx,
                            correlation_id,
                            TurnEvent::AssistantProseDelta { text },
                        )
                        .await;
                    }
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                replay,
            }) => {
                let item_id = replay.as_ref().and_then(|meta| meta.item_id.as_deref());
                self.log_llm_stream_event(
                    state.debug,
                    LlmStreamEventLog {
                        mode_iteration: state.mode_iteration,
                        event_type: "tool_call_part",
                        text: LlmDebugText {
                            raw: None,
                            visible: None,
                        },
                        item_id,
                        usage: None,
                        tool_call: Some(LlmDebugToolCall {
                            call_id: &call_id,
                            tool_name: &tool_name,
                            input_json: &input_json,
                        }),
                    },
                );
                state
                    .stream_accumulator
                    .push_tool_call(call_id, tool_name, input_json, replay);
            }
            LlmStreamEvent::Part(LlmOutputPart::Reasoning { text, replay }) => {
                let item_id = replay.as_ref().and_then(|meta| meta.item_id.as_deref());
                if !text.is_empty() {
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            mode_iteration: state.mode_iteration,
                            event_type: "reasoning_part",
                            text: LlmDebugText {
                                raw: None,
                                visible: Some(&text),
                            },
                            item_id,
                            usage: None,
                            tool_call: None,
                        },
                    );
                    send_session_event(
                        event_tx,
                        SessionEvent::ReasoningDelta {
                            content: text.clone(),
                        },
                    )
                    .await;
                    let correlation_id =
                        stream_correlation_id(state.reasoning_correlation, item_id);
                    send_turn_activity(
                        event_tx,
                        correlation_id,
                        TurnEvent::ReasoningDelta { text: text.clone() },
                    )
                    .await;
                }
                state.stream_accumulator.push_reasoning(
                    text,
                    replay.as_ref().and_then(|meta| meta.item_id.clone()),
                    replay
                        .as_ref()
                        .map(|meta| meta.summary.clone())
                        .unwrap_or_default(),
                    replay
                        .as_ref()
                        .and_then(|meta| meta.encrypted_content.clone()),
                );
            }
            LlmStreamEvent::Usage(usage) => {
                self.log_llm_stream_event(
                    state.debug,
                    LlmStreamEventLog {
                        mode_iteration: state.mode_iteration,
                        event_type: "usage",
                        text: LlmDebugText {
                            raw: None,
                            visible: None,
                        },
                        item_id: None,
                        usage: Some(&usage),
                        tool_call: None,
                    },
                );
                *state.streamed_usage = usage;
            }
            LlmStreamEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
            } => {
                send_session_event(
                    event_tx,
                    SessionEvent::RetryStatus {
                        wait_seconds,
                        attempt,
                        max_attempts,
                        reason,
                        envelope: None,
                    },
                )
                .await;
            }
        }
        Ok(())
    }

    async fn drain_standard_stream_queue(
        &mut self,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        llm_stream_rx: &mut tokio::sync::mpsc::UnboundedReceiver<LlmStreamEvent>,
        state: &mut StandardStreamState<'_>,
    ) -> Result<(), LlmCallError> {
        while let Ok(stream_event) = llm_stream_rx.try_recv() {
            self.forward_standard_stream_event(event_tx, stream_event, state)
                .await?;
        }
        Ok(())
    }
}

fn response_usage_is_empty(usage: &LlmUsage) -> bool {
    usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_input_tokens == 0
        && usage.reasoning_tokens == 0
}

fn provider_item_id(value: &serde_json::Value) -> Option<String> {
    value
        .get("item_id")
        .or_else(|| value.get("item").and_then(|item| item.get("id")))
        .or_else(|| {
            value
                .get("response")
                .and_then(|response| response.get("id"))
        })
        .or_else(|| value.get("id"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn provider_output_index(value: &serde_json::Value) -> Option<i64> {
    value
        .get("output_index")
        .or_else(|| value.get("index"))
        .and_then(|value| value.as_i64())
}

fn stream_correlation_id(
    fallback_slot: &mut Option<TurnActivityId>,
    provider_item_id: Option<&str>,
) -> TurnActivityId {
    if let Some(provider_item_id) = provider_item_id {
        return TurnActivityId::new(provider_item_id.to_string());
    }
    fallback_slot
        .get_or_insert_with(TurnActivityId::fresh)
        .clone()
}

/// Wait up to 2s for a late `Usage` event from the provider after an
/// RLM stream-mask abort. The usage is returned on the response itself so
/// sansio records it synchronously, which makes prompt-budget guidance
/// available to the next mode iteration.
async fn collect_trailing_usage_before_abort<T>(
    llm_task: &mut tokio::task::JoinHandle<T>,
    llm_stream_rx: &mut tokio::sync::mpsc::UnboundedReceiver<LlmStreamEvent>,
    initial_usage: LlmUsage,
) -> LlmUsage {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(2_000);
    let mut latest = initial_usage;
    loop {
        match tokio::time::timeout_at(deadline, llm_stream_rx.recv()).await {
            Err(_) | Ok(None) => break,
            Ok(Some(LlmStreamEvent::Usage(usage))) => {
                latest = usage;
                break;
            }
            Ok(Some(_)) => continue,
        }
    }
    llm_task.abort();
    latest
}
