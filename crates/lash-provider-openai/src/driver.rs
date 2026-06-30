use crate::support::*;

#[derive(Clone, Copy, Debug)]
pub(crate) enum CompletionEndpoint {
    Responses,
    ChatCompletions,
}

impl CompletionEndpoint {
    fn request_trace_name(self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::ChatCompletions => "chat/completions",
        }
    }

    fn path(self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::ChatCompletions => "chat/completions",
        }
    }

    fn serialize_error(self) -> &'static str {
        match self {
            Self::Responses => "Failed to serialize Responses body",
            Self::ChatCompletions => "Failed to serialize Chat Completions body",
        }
    }

    fn response_start_timeout_error(self) -> &'static str {
        match self {
            Self::Responses => "OpenAI-compatible response start timed out",
            Self::ChatCompletions => "OpenAI-compatible chat response start timed out",
        }
    }

    fn response_body_timeout_error(self) -> &'static str {
        match self {
            Self::Responses => "OpenAI-compatible response body timed out",
            Self::ChatCompletions => "OpenAI-compatible chat response body timed out",
        }
    }

    fn stream_chunk_timeout_error(self) -> &'static str {
        match self {
            Self::Responses => "OpenAI-compatible stream chunk timed out",
            Self::ChatCompletions => "OpenAI-compatible chat stream chunk timed out",
        }
    }

    fn request_failed_prefix(self) -> &'static str {
        match self {
            Self::Responses => "OpenAI-compatible request failed",
            Self::ChatCompletions => "OpenAI-compatible chat request failed",
        }
    }

    fn http_summary(self, url: &str, stream: bool) -> String {
        if stream {
            format!("HTTP POST {url} (stream)")
        } else {
            format!("HTTP POST {url}")
        }
    }
}

pub(crate) async fn complete(
    provider: &mut OpenAiCompatibleProvider,
    req: LlmRequest,
    endpoint: CompletionEndpoint,
) -> Result<LlmResponse, LlmTransportError> {
    let stream_events = req.stream_events.clone();
    let provider_trace = req.provider_trace.clone();
    let timeouts = provider.options.llm_timeouts();
    let stream = stream_events.is_some();
    let body = match endpoint {
        CompletionEndpoint::Responses => provider.build_responses_request_body(&req, stream)?,
        CompletionEndpoint::ChatCompletions => provider.build_chat_request_body(&req, stream)?,
    };
    emit_provider_request_trace(
        provider_trace.as_ref(),
        endpoint.request_trace_name(),
        &body,
    );
    let body_bytes = serde_json::to_vec(&body)
        .map_err(|e| LlmTransportError::new(format!("{}: {e}", endpoint.serialize_error())))?;
    let request_body = bytes::Bytes::from(body_bytes);
    let request_body_for_error = String::from_utf8_lossy(&request_body).into_owned();
    let url = format!(
        "{}/{}",
        provider.base_url.trim_end_matches('/'),
        endpoint.path()
    );
    let compat = provider.resolved_compat(endpoint);
    let mut headers = vec![
        (
            "Authorization".to_string(),
            format!("Bearer {}", provider.api_key),
        ),
        ("Content-Type".to_string(), "application/json".to_string()),
        ("Accept".to_string(), "text/event-stream".to_string()),
    ];
    if compat.cache_session_affinity
        && let Some(scope) = req.scope.as_ref()
    {
        headers.push(("session_id".to_string(), scope.session_id.clone()));
        headers.push(("x-client-request-id".to_string(), scope.request_id.clone()));
    }
    let http_request = LlmHttpRequest {
        method: LlmHttpMethod::Post,
        url: url.clone(),
        headers,
        body: request_body.clone(),
        body_for_error: Some(request_body_for_error.clone()),
        response_start_timeout_message: Some(endpoint.response_start_timeout_error().to_string()),
    };
    let resp = provider
        .transport
        .send(
            http_request,
            response_start_timeout(timeouts.request_timeout, timeouts.chunk_timeout, stream),
        )
        .await?;

    let status = resp.status;
    if !resp.is_success() {
        let headers = resp.headers;
        let text = read_http_body_text(
            resp.body,
            timeouts.request_timeout,
            endpoint.response_body_timeout_error(),
        )
        .await
        .unwrap_or_default();
        let detail = extract_error_detail(&text);
        let message = detail
            .map(|detail| {
                format!(
                    "{} with {}: {}",
                    endpoint.request_failed_prefix(),
                    status,
                    detail
                )
            })
            .unwrap_or_else(|| format!("{} with {}", endpoint.request_failed_prefix(), status));
        // Retryability is decided centrally by the provider failure
        // classifier from the attached HTTP status; no inline override here.
        return Err(LlmTransportError::new(message)
            .with_status(status)
            .with_headers(headers)
            .with_raw(text)
            .with_request_body(request_body_for_error));
    }

    let is_sse = header_contains(&resp.headers, "content-type", "text/event-stream");

    if is_sse {
        drive_streaming_response(
            provider,
            endpoint,
            resp.body,
            stream_events,
            provider_trace,
            url,
            timeouts.chunk_timeout,
        )
        .await
    } else {
        complete_buffered_response(
            provider,
            endpoint,
            resp.body,
            stream_events,
            provider_trace,
            url,
            timeouts.request_timeout,
        )
        .await
    }
}

async fn complete_buffered_response(
    provider: &OpenAiCompatibleProvider,
    endpoint: CompletionEndpoint,
    body: LlmHttpBody,
    stream_events: Option<LlmEventSender>,
    provider_trace: Option<LlmProviderTraceSender>,
    url: String,
    timeout: Option<std::time::Duration>,
) -> Result<LlmResponse, LlmTransportError> {
    let text = read_http_body_text(body, timeout, endpoint.response_body_timeout_error()).await?;
    emit_provider_trace(provider_trace.as_ref(), "openai_compatible", &text);
    match endpoint {
        CompletionEndpoint::Responses => {
            complete_buffered_responses(provider, text, stream_events, url)
        }
        CompletionEndpoint::ChatCompletions => {
            complete_buffered_chat(provider, text, stream_events, url)
        }
    }
}

fn complete_buffered_responses(
    provider: &OpenAiCompatibleProvider,
    text: String,
    stream_events: Option<LlmEventSender>,
    url: String,
) -> Result<LlmResponse, LlmTransportError> {
    let mut state = ResponsesStreamState::default();
    if text.trim_start().starts_with("data:") || text.contains("\ndata:") {
        OpenAiCompatibleProvider::parse_sse_payload(&text, &mut state)?;
    } else {
        let value: Value = serde_json::from_str(&text).map_err(|e| {
            LlmTransportError::new(format!("Invalid Responses JSON: {e}")).with_raw(text.clone())
        })?;
        state.provider_usage = value.get("usage").cloned();
        state.usage = usage_from_response_value(&value);
        state.parts = OpenAiCompatibleProvider::response_parts_from_value(&value);
        state.recompute_full_text();
        state.final_response = Some(value);
    }
    let parts = state.response_parts();
    if !has_response_content(&parts) {
        return Err(empty_response_error(text));
    }
    if let Some(tx) = &stream_events {
        if state.usage != LlmUsage::default() {
            tx.send(LlmStreamEvent::Usage(state.usage.clone()));
        }
        if provider.options.expose_thinking {
            for part in &parts {
                if let LlmOutputPart::Reasoning { text, .. } = part
                    && !text.is_empty()
                {
                    tx.send(LlmStreamEvent::ReasoningDelta(text.clone()));
                }
            }
        }
        if !state.full_text.is_empty() {
            tx.send(LlmStreamEvent::Delta(state.full_text.clone()));
        }
    }
    let terminal_reason = state
        .final_response
        .as_ref()
        .map(|value| terminal_reason_from_responses_value(value, &parts))
        .unwrap_or_else(|| terminal_reason_from_parts(&parts));
    Ok(LlmResponse {
        full_text: state.full_text,
        parts,
        usage: state.usage,
        terminal_reason,
        terminal_diagnostic: None,
        provider_usage: state.provider_usage,
        request_body: None,
        http_summary: Some(CompletionEndpoint::Responses.http_summary(&url, false)),
    })
}

fn complete_buffered_chat(
    provider: &OpenAiCompatibleProvider,
    text: String,
    stream_events: Option<LlmEventSender>,
    url: String,
) -> Result<LlmResponse, LlmTransportError> {
    let mut state = ChatStreamState::default();
    let mut parsed_parts = None;
    if text.trim_start().starts_with("data:") || text.contains("\ndata:") {
        OpenAiCompatibleProvider::parse_chat_sse_payload(&text, &mut state)?;
    } else {
        let value: Value = serde_json::from_str(&text).map_err(|e| {
            LlmTransportError::new(format!("Invalid Chat Completions JSON: {e}"))
                .with_raw(text.clone())
        })?;
        state.provider_usage = value.get("usage").cloned();
        state.usage = usage_from_response_value(&value);
        let parts = OpenAiCompatibleProvider::chat_response_parts_from_value(&value);
        let terminal_reason = terminal_reason_from_chat_value(&value, &parts);
        state.full_text = parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        parsed_parts = Some(parts);
        state.final_response_raw = Some(text.clone());
        state.terminal_reason = terminal_reason;
    }
    let parts = parsed_parts.unwrap_or_else(|| state.parts());
    if !has_response_content(&parts) {
        return Err(empty_response_error(text));
    }
    if let Some(tx) = &stream_events {
        if state.usage != LlmUsage::default() {
            tx.send(LlmStreamEvent::Usage(state.usage.clone()));
        }
        if !state.full_text.is_empty() {
            tx.send(LlmStreamEvent::Delta(state.full_text.clone()));
        }
        if provider.options.expose_thinking {
            for part in parts
                .iter()
                .filter(|part| matches!(part, LlmOutputPart::Reasoning { .. }))
            {
                tx.send(LlmStreamEvent::Part(part.clone()));
            }
        }
        for part in parts
            .iter()
            .filter(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
        {
            tx.send(LlmStreamEvent::Part(part.clone()));
        }
    }
    let terminal_reason = if state.terminal_reason == LlmTerminalReason::Unknown {
        terminal_reason_from_parts(&parts)
    } else {
        state.terminal_reason
    };
    Ok(LlmResponse {
        full_text: state.full_text,
        parts,
        usage: state.usage,
        terminal_reason,
        terminal_diagnostic: None,
        provider_usage: state.provider_usage,
        request_body: None,
        http_summary: Some(CompletionEndpoint::ChatCompletions.http_summary(&url, false)),
    })
}

async fn drive_streaming_response(
    provider: &OpenAiCompatibleProvider,
    endpoint: CompletionEndpoint,
    body: LlmHttpBody,
    stream_events: Option<LlmEventSender>,
    provider_trace: Option<LlmProviderTraceSender>,
    url: String,
    chunk_timeout: std::time::Duration,
) -> Result<LlmResponse, LlmTransportError> {
    match endpoint {
        CompletionEndpoint::Responses => {
            drive_streaming_responses(
                provider,
                body,
                stream_events,
                provider_trace,
                url,
                chunk_timeout,
            )
            .await
        }
        CompletionEndpoint::ChatCompletions => {
            drive_streaming_chat(
                provider,
                body,
                stream_events,
                provider_trace,
                url,
                chunk_timeout,
            )
            .await
        }
    }
}

async fn drive_streaming_responses(
    provider: &OpenAiCompatibleProvider,
    body: LlmHttpBody,
    stream_events: Option<LlmEventSender>,
    provider_trace: Option<LlmProviderTraceSender>,
    url: String,
    chunk_timeout: std::time::Duration,
) -> Result<LlmResponse, LlmTransportError> {
    let mut state = ResponsesStreamState::default();
    let mut emitted_parts = Vec::new();
    let expose_thinking = provider.options.expose_thinking;
    drive_sse_response(
        body,
        chunk_timeout,
        CompletionEndpoint::Responses.stream_chunk_timeout_error(),
        |raw| {
            emit_provider_trace(provider_trace.as_ref(), "openai_compatible", raw);
            let prev_usage = state.usage.clone();
            OpenAiCompatibleProvider::process_sse_event(raw, &mut state, Some(&mut emitted_parts))?;
            emit_stream_progress(
                stream_events.as_ref(),
                state.take_text_deltas(),
                &state.usage,
                &prev_usage,
            );
            if let Some(tx) = &stream_events {
                for delta in state.take_reasoning_deltas() {
                    if expose_thinking {
                        tx.send(LlmStreamEvent::ReasoningDelta(delta));
                    }
                }
                for part in emitted_parts.drain(..) {
                    if matches!(part, LlmOutputPart::Reasoning { .. }) && !expose_thinking {
                        continue;
                    }
                    tx.send(LlmStreamEvent::Part(part));
                }
            } else {
                emitted_parts.clear();
                state.take_reasoning_deltas();
            }
            Ok(())
        },
    )
    .await?;

    let parts = state.response_parts();
    if !has_response_content(&parts) {
        return Err(empty_response_error(
            state
                .final_response
                .as_ref()
                .map(Value::to_string)
                .unwrap_or_default(),
        ));
    }
    let terminal_reason = state
        .final_response
        .as_ref()
        .map(|value| terminal_reason_from_responses_value(value, &parts))
        .unwrap_or_else(|| terminal_reason_from_parts(&parts));
    Ok(LlmResponse {
        full_text: state.full_text,
        parts,
        usage: state.usage,
        terminal_reason,
        terminal_diagnostic: None,
        provider_usage: state.provider_usage,
        request_body: None,
        http_summary: Some(CompletionEndpoint::Responses.http_summary(&url, true)),
    })
}

async fn drive_streaming_chat(
    provider: &OpenAiCompatibleProvider,
    body: LlmHttpBody,
    stream_events: Option<LlmEventSender>,
    provider_trace: Option<LlmProviderTraceSender>,
    url: String,
    chunk_timeout: std::time::Duration,
) -> Result<LlmResponse, LlmTransportError> {
    let mut state = ChatStreamState::default();
    let expose_thinking = provider.options.expose_thinking;
    drive_sse_response(
        body,
        chunk_timeout,
        CompletionEndpoint::ChatCompletions.stream_chunk_timeout_error(),
        |raw| {
            emit_provider_trace(provider_trace.as_ref(), "openai_compatible", raw);
            let prev_usage = state.usage.clone();
            OpenAiCompatibleProvider::process_chat_sse_event(raw, &mut state)?;
            emit_stream_progress(
                stream_events.as_ref(),
                state.take_text_deltas(),
                &state.usage,
                &prev_usage,
            );
            if let Some(tx) = &stream_events {
                for delta in state.take_reasoning_deltas() {
                    if expose_thinking {
                        tx.send(LlmStreamEvent::ReasoningDelta(delta));
                    }
                }
            } else {
                state.take_reasoning_deltas();
            }
            Ok(())
        },
    )
    .await?;

    let parts = state.parts();
    if !has_response_content(&parts) {
        return Err(empty_response_error(
            state.final_response_raw.clone().unwrap_or_default(),
        ));
    }
    if let Some(tx) = &stream_events {
        for part in parts
            .iter()
            .filter(|part| matches!(part, LlmOutputPart::ToolCall { .. }))
        {
            tx.send(LlmStreamEvent::Part(part.clone()));
        }
    }
    let terminal_reason = if state.terminal_reason == LlmTerminalReason::Unknown {
        terminal_reason_from_parts(&parts)
    } else {
        state.terminal_reason
    };
    Ok(LlmResponse {
        full_text: state.full_text,
        parts,
        usage: state.usage,
        terminal_reason,
        terminal_diagnostic: None,
        provider_usage: state.provider_usage,
        request_body: None,
        http_summary: Some(CompletionEndpoint::ChatCompletions.http_summary(&url, true)),
    })
}
