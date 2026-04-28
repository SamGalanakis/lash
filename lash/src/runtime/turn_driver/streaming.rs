use std::sync::Arc;

use lash_trace::{TraceContext, TraceError, TraceEvent};

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

async fn emit_plugin_surface_events_runtime(
    event_tx: &mpsc::Sender<RuntimeStreamEvent>,
    plugin_id: &str,
    events: Vec<crate::PluginSurfaceEvent>,
) {
    for event in crate::plugin::plugin_surface_session_events(plugin_id, events) {
        send_session_event(event_tx, event).await;
    }
}

impl RuntimeTurnDriver {
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
            .transform_assistant_stream(&self.session_id, chunk, Arc::clone(&self.session_manager))
            .await
            .map_err(|err| LlmCallError {
                message: err.to_string(),
                retryable: false,
                raw: None,
                code: Some("plugin_assistant_stream".to_string()),
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
            emit_plugin_surface_events_runtime(event_tx, &emitted.plugin_id, emitted.value.events)
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
            .transform_assistant_response(
                &self.session_id,
                response,
                Arc::clone(&self.session_manager),
            )
            .await
            .map_err(|err| LlmCallError {
                message: err.to_string(),
                retryable: false,
                raw: None,
                code: Some("plugin_assistant_response".to_string()),
                request_body: None,
            })?;
        let mut current: Option<LlmResponse> = None;
        for emitted in transforms {
            emit_plugin_surface_events_runtime(event_tx, &emitted.plugin_id, emitted.value.events)
                .await;
            current = Some(emitted.value.response);
        }
        Ok(current.unwrap_or(original))
    }

    pub(super) async fn run_standard_llm_call(
        &mut self,
        request: LlmRequest,
        iteration: usize,
        event_tx: &mpsc::Sender<RuntimeStreamEvent>,
        cancel: &CancellationToken,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        let trace_enabled = self.host.core.trace_sink.is_some();
        let llm_call_id = if trace_enabled {
            Some(format!("{}:{iteration}", self.session_id))
        } else {
            None
        };
        if let Some(llm_call_id) = llm_call_id.as_ref() {
            crate::trace::emit_trace(
                &self.host.core.trace_sink,
                &self.host.core.trace_context,
                TraceContext::default()
                    .for_session(self.session_id.clone())
                    .for_iteration(iteration)
                    .for_llm_call(llm_call_id.clone()),
                TraceEvent::LlmCallStarted {
                    request: crate::trace::trace_llm_request(&request),
                },
            );
        }
        let (llm_stream_tx, mut llm_stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
        let llm_request = LlmRequest {
            stream_events: transport_stream_events(&self.policy.provider, Some(llm_stream_tx)),
            ..request
        };

        let mut call_provider = self.policy.provider.clone();
        let mut llm_task = tokio::spawn(async move {
            let result = call_provider.complete(llm_request).await;
            (result, call_provider)
        });

        let mut text_streamed = false;
        let mut streamed_usage = LlmUsage::default();
        let mut streamed_output = StandardStreamFallback::default();
        let mut debug = LlmStreamDebugState::new();
        let mut abort_requested = false;
        let buffer_stream_fallback = self.policy.provider.requires_streaming()
            || self.session.plugins().has_assistant_stream_hooks();
        let mut stream_state = StandardStreamState {
            text_streamed: &mut text_streamed,
            streamed_usage: &mut streamed_usage,
            streamed_output: &mut streamed_output,
            buffer_stream_fallback,
            debug: &mut debug,
            iteration,
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
                        // wait briefly for the provider's final
                        // `response.completed` SSE event. Codex attaches
                        // usage there, and the next prompt's budget
                        // contribution depends on the driver seeing that
                        // accounting before the next iteration starts.
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
                            full_text: streamed_output.full_text(),
                            parts: Vec::new(),
                            usage: streamed_usage.clone(),
                            provider_usage: None,
                            request_body: None,
                            http_summary: None,
                        };
                        streamed_output.apply_to_response(&mut resp);
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
                            streamed_output.apply_to_response(&mut resp);
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
                            request_body: e.request_body,
                        }),
                    }
                }
            }
        };

        if let Err(err) = &result {
            tracing::error!(
                session_id = %self.session_id,
                turn = iteration,
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
                    crate::trace::emit_trace(
                        &self.host.core.trace_sink,
                        &self.host.core.trace_context,
                        TraceContext::default()
                            .for_session(self.session_id.clone())
                            .for_iteration(iteration)
                            .for_llm_call(llm_call_id),
                        TraceEvent::LlmCallCompleted {
                            response: crate::trace::trace_llm_response(
                                response.full_text.clone(),
                                debug.elapsed_ms(),
                                crate::trace::trace_output_parts(&response.parts),
                            ),
                            usage: Some(crate::trace::trace_usage_from_llm(&response.usage)),
                            provider_usage: response.provider_usage.clone(),
                            stream_summary: Some(stream_summary.clone()),
                        },
                    );
                }
                Err(error) => {
                    crate::trace::emit_trace(
                        &self.host.core.trace_sink,
                        &self.host.core.trace_context,
                        TraceContext::default()
                            .for_session(self.session_id.clone())
                            .for_iteration(iteration)
                            .for_llm_call(llm_call_id),
                        TraceEvent::LlmCallFailed {
                            error: TraceError {
                                message: error.message.clone(),
                                retryable: error.retryable,
                                code: error.code.clone(),
                                raw: error.raw.clone(),
                            },
                            stream_summary: Some(stream_summary.clone()),
                        },
                    );
                }
            }
        }
        if trace_enabled {
            self.llm_stream_summaries.insert(iteration, debug.summary);
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
                iteration,
                usage,
                provider_usage,
                response_text,
                response_parts,
                ..
            } => {
                let stream_summary = self.llm_stream_summaries.remove(&iteration);
                crate::trace::emit_trace(
                    &self.host.core.trace_sink,
                    &self.host.core.trace_context,
                    TraceContext::default()
                        .for_session(session_id)
                        .for_iteration(iteration)
                        .for_llm_call(format!("{}:{iteration}", self.session_id)),
                    TraceEvent::LlmCallCompleted {
                        response: crate::trace::trace_llm_response(
                            response_text,
                            0,
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
                iteration,
                message,
                retryable,
                raw,
                code,
                ..
            } => {
                let stream_summary = self.llm_stream_summaries.remove(&iteration);
                crate::trace::emit_trace(
                    &self.host.core.trace_sink,
                    &self.host.core.trace_context,
                    TraceContext::default()
                        .for_session(session_id)
                        .for_iteration(iteration)
                        .for_llm_call(format!("{}:{iteration}", self.session_id)),
                    TraceEvent::LlmCallFailed {
                        error: TraceError {
                            message,
                            retryable,
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

        let mut entry = serde_json::json!({
            "kind": "stream_event",
            "turn": log.iteration,
            "ts": chrono::Utc::now().to_rfc3339(),
            "session_id": log.session_id,
            "sequence": debug.next_sequence(),
            "elapsed_ms": elapsed_ms,
            "event": log.event_type,
        });

        if let Some(object) = entry.as_object_mut() {
            if let Some(text) = log.text.raw {
                object.insert(
                    "raw_text".to_string(),
                    serde_json::Value::String(text.to_string()),
                );
                object.insert(
                    "raw_chars".to_string(),
                    serde_json::Value::from(text.chars().count() as u64),
                );
            }
            if let Some(text) = log.text.visible {
                object.insert(
                    "visible_text".to_string(),
                    serde_json::Value::String(text.to_string()),
                );
                object.insert(
                    "visible_chars".to_string(),
                    serde_json::Value::from(text.chars().count() as u64),
                );
            }
            if let Some(usage) = log.usage {
                object.insert(
                    "usage".to_string(),
                    serde_json::json!({
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cached_input_tokens": usage.cached_input_tokens,
                        "reasoning_tokens": usage.reasoning_tokens,
                    }),
                );
            }
            if let Some(tool_call) = log.tool_call {
                object.insert(
                    "call_id".to_string(),
                    serde_json::Value::String(tool_call.call_id.to_string()),
                );
                object.insert(
                    "tool_name".to_string(),
                    serde_json::Value::String(tool_call.tool_name.to_string()),
                );
                object.insert(
                    "input_json".to_string(),
                    serde_json::Value::String(tool_call.input_json.to_string()),
                );
            }
        }

        crate::trace::emit_trace(
            &self.host.core.trace_sink,
            &self.host.core.trace_context,
            TraceContext::default()
                .for_session(self.session_id.clone())
                .for_iteration(log.iteration)
                .for_llm_call(format!("{}:{}", self.session_id, log.iteration)),
            TraceEvent::Custom {
                name: "runtime.stream_event".to_string(),
                payload: entry,
            },
        );
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
                        if state.buffer_stream_fallback {
                            state.streamed_output.push_reasoning(
                                reasoning_delta.clone(),
                                None,
                                Vec::new(),
                                None,
                            );
                        }
                        send_session_event(
                            event_tx,
                            SessionEvent::ReasoningDelta {
                                content: reasoning_delta,
                            },
                        )
                        .await;
                    }
                    let delta = outcome.chunk;
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            session_id: &self.session_id,
                            iteration: state.iteration,
                            event_type: "delta",
                            text: LlmDebugText {
                                raw: raw_delta.as_deref(),
                                visible: Some(&delta),
                            },
                            usage: None,
                            tool_call: None,
                        },
                    );
                    if !delta.is_empty() {
                        if state.buffer_stream_fallback {
                            state.streamed_output.push_text(&delta);
                        }
                        send_session_event(event_tx, SessionEvent::TextDelta { content: delta })
                            .await;
                    }
                }
            }
            LlmStreamEvent::ReasoningDelta(delta) => {
                if !delta.is_empty() {
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            session_id: &self.session_id,
                            iteration: state.iteration,
                            event_type: "reasoning_delta",
                            text: LlmDebugText {
                                raw: None,
                                visible: Some(&delta),
                            },
                            usage: None,
                            tool_call: None,
                        },
                    );
                    // Delta-only streaming path (fix 1.3a display). No
                    // encrypted content yet — that arrives with the full
                    // item on `output_item.done` (fix 1.3b).
                    if state.buffer_stream_fallback {
                        state
                            .streamed_output
                            .push_reasoning(delta.clone(), None, Vec::new(), None);
                    }
                    send_session_event(event_tx, SessionEvent::ReasoningDelta { content: delta })
                        .await;
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::Text { text }) => {
                if !text.is_empty() {
                    *state.text_streamed = true;
                    let raw_text = self.host.core.trace_sink.as_ref().map(|_| text.clone());
                    let outcome = self
                        .transform_assistant_stream_chunk(event_tx, text)
                        .await?;
                    if outcome.abort_requested {
                        *state.abort_requested = true;
                    }
                    for reasoning_delta in outcome.reasoning_deltas {
                        if state.buffer_stream_fallback {
                            state.streamed_output.push_reasoning(
                                reasoning_delta.clone(),
                                None,
                                Vec::new(),
                                None,
                            );
                        }
                        send_session_event(
                            event_tx,
                            SessionEvent::ReasoningDelta {
                                content: reasoning_delta,
                            },
                        )
                        .await;
                    }
                    let text = outcome.chunk;
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            session_id: &self.session_id,
                            iteration: state.iteration,
                            event_type: "text_part",
                            text: LlmDebugText {
                                raw: raw_text.as_deref(),
                                visible: Some(&text),
                            },
                            usage: None,
                            tool_call: None,
                        },
                    );
                    if !text.is_empty() {
                        if state.buffer_stream_fallback {
                            state.streamed_output.push_text(&text);
                        }
                        send_session_event(event_tx, SessionEvent::TextDelta { content: text })
                            .await;
                    }
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                item_id,
                signature,
            }) => {
                self.log_llm_stream_event(
                    state.debug,
                    LlmStreamEventLog {
                        session_id: &self.session_id,
                        iteration: state.iteration,
                        event_type: "tool_call_part",
                        text: LlmDebugText {
                            raw: None,
                            visible: None,
                        },
                        usage: None,
                        tool_call: Some(LlmDebugToolCall {
                            call_id: &call_id,
                            tool_name: &tool_name,
                            input_json: &input_json,
                        }),
                    },
                );
                if state.buffer_stream_fallback {
                    state
                        .streamed_output
                        .push_tool_call(call_id, tool_name, input_json, item_id, signature);
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::Reasoning {
                text,
                signature: _,
                redacted: _,
                item_id,
                encrypted_content,
                summary,
            }) => {
                if !text.is_empty() {
                    self.log_llm_stream_event(
                        state.debug,
                        LlmStreamEventLog {
                            session_id: &self.session_id,
                            iteration: state.iteration,
                            event_type: "reasoning_part",
                            text: LlmDebugText {
                                raw: None,
                                visible: Some(&text),
                            },
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
                }
                if state.buffer_stream_fallback {
                    state
                        .streamed_output
                        .push_reasoning(text, item_id, summary, encrypted_content);
                }
            }
            LlmStreamEvent::Usage(usage) => {
                self.log_llm_stream_event(
                    state.debug,
                    LlmStreamEventLog {
                        session_id: &self.session_id,
                        iteration: state.iteration,
                        event_type: "usage",
                        text: LlmDebugText {
                            raw: None,
                            visible: None,
                        },
                        usage: Some(&usage),
                        tool_call: None,
                    },
                );
                *state.streamed_usage = usage;
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

pub(in crate::runtime) fn llm_response_has_content(response: &LlmResponse) -> bool {
    if !response.full_text.is_empty() {
        return true;
    }
    response.parts.iter().any(|part| match part {
        LlmOutputPart::Text { text } => !text.is_empty(),
        // Reasoning-only responses still count as "has content" so the
        // adapter's stream-fallback buffer is preserved for replay.
        LlmOutputPart::Reasoning { .. } => true,
        LlmOutputPart::ToolCall { .. } => true,
    })
}

/// Wait up to 2s for a late `Usage` event from the provider after an
/// RLM stream-mask abort. The usage is returned on the response itself so
/// sansio records it synchronously, which makes prompt-budget guidance
/// available to the next iteration.
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
