use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use opentelemetry::trace::{
    Span, SpanContext, SpanKind, Status, TraceContextExt, Tracer, TracerProvider,
};
use opentelemetry::{Context, InstrumentationScope, KeyValue, Value as OtelValue, global};
use serde_json::Value;

use crate::{TraceContext, TraceEvent, TraceRecord, TraceSink, TraceTokenUsage};

const INSTRUMENTATION_NAME: &str = "lash-trace";

/// Controls which structured Lash trace data is attached to OpenTelemetry
/// spans.
#[derive(Clone, Debug)]
pub struct OtelTraceOptions {
    /// Attach the full Lash trace event as a JSON attribute. This is useful for
    /// local collectors and debugging, but it can exceed backend attribute
    /// limits in production.
    pub include_event_json: bool,
    /// Attach `TraceContext.metadata` as `lash.metadata.*` attributes.
    pub include_context_metadata: bool,
    /// Attach event payload fields that can be large, such as tool args/results
    /// and custom payloads, as compact JSON attributes.
    pub include_payload_json: bool,
}

impl Default for OtelTraceOptions {
    fn default() -> Self {
        Self {
            include_event_json: false,
            include_context_metadata: true,
            include_payload_json: false,
        }
    }
}

pub struct OtelTraceSink<T = global::BoxedTracer>
where
    T: Tracer + Send + Sync,
    T::Span: Send + Sync + 'static,
{
    tracer: T,
    options: OtelTraceOptions,
    active: Mutex<HashMap<String, ActiveSpan<T::Span>>>,
}

struct ActiveSpan<S: Span> {
    span: S,
    context: SpanContext,
}

impl OtelTraceSink<global::BoxedTracer> {
    /// Build a sink from the process-global OpenTelemetry tracer provider.
    ///
    /// This keeps exporter/provider setup with the embedding host while giving
    /// Lash a ready-to-install `TraceSink`.
    pub fn from_global_provider() -> Self {
        let scope = InstrumentationScope::builder(INSTRUMENTATION_NAME)
            .with_version(env!("CARGO_PKG_VERSION"))
            .build();
        Self::new(global::tracer_provider().tracer_with_scope(scope))
    }
}

impl<T> OtelTraceSink<T>
where
    T: Tracer + Send + Sync,
    T::Span: Send + Sync + 'static,
{
    pub fn new(tracer: T) -> Self {
        Self::with_options(tracer, OtelTraceOptions::default())
    }

    pub fn with_options(tracer: T, options: OtelTraceOptions) -> Self {
        Self {
            tracer,
            options,
            active: Mutex::new(HashMap::new()),
        }
    }

    pub fn options(&self) -> &OtelTraceOptions {
        &self.options
    }

    fn start_active(&self, key: String, record: &TraceRecord, name: &'static str) {
        let parent = if matches!(&record.event, TraceEvent::TurnStarted { .. }) {
            None
        } else {
            parent_for(record, &self.active)
        };
        let mut span = self.build_span(record, name, parent, record_time(record), None);
        span.set_attributes(self.attributes_for(record));
        let context = span.span_context().clone();
        if let Ok(mut active) = self.active.lock() {
            if let Some(mut existing) = active.remove(&key) {
                existing.span.end_with_timestamp(record_time(record));
            }
            active.insert(key, ActiveSpan { span, context });
        } else {
            span.end_with_timestamp(record_time(record));
        }
    }

    fn end_active(&self, key: &str, record: &TraceRecord, success: bool) -> bool {
        let Ok(mut active) = self.active.lock() else {
            return false;
        };
        let Some(mut active_span) = active.remove(key) else {
            return false;
        };
        let mut attrs = lifecycle_end_attributes(record, &self.options);
        attrs.extend(self.attributes_for(record));
        active_span.span.set_attributes(attrs);
        if !success {
            active_span.span.set_status(error_status(record));
        }
        active_span.span.end_with_timestamp(record_time(record));
        true
    }

    fn emit_instant(&self, record: &TraceRecord, name: &'static str, duration_ms: Option<u64>) {
        let end = record_time(record);
        let start = duration_ms
            .and_then(|ms| end.checked_sub(Duration::from_millis(ms)))
            .unwrap_or(end);
        let mut span = self.build_span(
            record,
            name,
            parent_for(record, &self.active),
            start,
            Some(end),
        );
        span.set_attributes(self.attributes_for(record));
        if is_error_event(&record.event) {
            span.set_status(error_status(record));
        }
        span.end_with_timestamp(end);
    }

    fn build_span(
        &self,
        record: &TraceRecord,
        name: &'static str,
        parent: Option<SpanContext>,
        start: SystemTime,
        end: Option<SystemTime>,
    ) -> T::Span {
        let mut builder = self
            .tracer
            .span_builder(name)
            .with_kind(SpanKind::Internal)
            .with_start_time(start)
            .with_attributes(common_attributes(record, &self.options));
        if let Some(end) = end {
            builder = builder.with_end_time(end);
        }
        match parent {
            Some(parent) => {
                let parent_cx = Context::new().with_remote_span_context(parent);
                builder.start_with_context(&self.tracer, &parent_cx)
            }
            None => builder.start(&self.tracer),
        }
    }

    fn attributes_for(&self, record: &TraceRecord) -> Vec<KeyValue> {
        event_attributes(record, &self.options)
    }
}

impl<T> TraceSink for OtelTraceSink<T>
where
    T: Tracer + Send + Sync,
    T::Span: Send + Sync + 'static,
{
    fn append(&self, record: &TraceRecord) {
        match &record.event {
            TraceEvent::TurnStarted { .. } => {
                if let Some(key) = turn_key(&record.context) {
                    self.start_active(key, record, "lash.turn");
                } else {
                    self.emit_instant(record, "lash.turn.started", None);
                }
            }
            TraceEvent::TurnCompleted { status, .. } => {
                let ended = turn_key(&record.context)
                    .as_deref()
                    .is_some_and(|key| self.end_active(key, record, status != "failed"));
                if !ended {
                    self.emit_instant(record, "lash.turn.completed", None);
                }
            }
            TraceEvent::LlmCallStarted { .. } => {
                if let Some(key) = llm_key(&record.context) {
                    self.start_active(key, record, "lash.llm");
                } else {
                    self.emit_instant(record, "lash.llm.started", None);
                }
            }
            TraceEvent::LlmCallCompleted { response, .. } => {
                let ended = llm_key(&record.context)
                    .as_deref()
                    .is_some_and(|key| self.end_active(key, record, true));
                if !ended {
                    self.emit_instant(record, "lash.llm", Some(response.duration_ms));
                }
            }
            TraceEvent::LlmCallFailed { .. } => {
                let ended = llm_key(&record.context)
                    .as_deref()
                    .is_some_and(|key| self.end_active(key, record, false));
                if !ended {
                    self.emit_instant(record, "lash.llm", None);
                }
            }
            TraceEvent::ToolCallCompleted { duration_ms, .. } => {
                self.emit_instant(record, "lash.tool", Some(*duration_ms));
            }
            TraceEvent::SessionStarted { .. } => self.emit_instant(record, "lash.session", None),
            TraceEvent::PromptBuilt { .. } => self.emit_instant(record, "lash.prompt", None),
            TraceEvent::ModeStep { .. } => self.emit_instant(record, "lash.mode_step", None),
            TraceEvent::TokenUsage { .. } => self.emit_instant(record, "lash.token_usage", None),
            TraceEvent::Custom { .. } => self.emit_instant(record, "lash.custom", None),
        }
    }
}

fn common_attributes(record: &TraceRecord, options: &OtelTraceOptions) -> Vec<KeyValue> {
    let mut attrs = vec![
        KeyValue::new("lash.trace.schema_version", record.schema_version as i64),
        KeyValue::new("lash.trace.record_id", record.id.clone()),
        KeyValue::new("lash.trace.event_type", event_type(&record.event)),
    ];
    context_attributes(&mut attrs, &record.context, options);
    if options.include_event_json
        && let Ok(json) = serde_json::to_string(record)
    {
        attrs.push(KeyValue::new("lash.trace.record_json", json));
    }
    attrs
}

fn lifecycle_end_attributes(record: &TraceRecord, options: &OtelTraceOptions) -> Vec<KeyValue> {
    let mut attrs = vec![
        KeyValue::new(
            "lash.trace.end_schema_version",
            record.schema_version as i64,
        ),
        KeyValue::new("lash.trace.end_record_id", record.id.clone()),
        KeyValue::new("lash.trace.end_event_type", event_type(&record.event)),
    ];
    if options.include_event_json
        && let Ok(json) = serde_json::to_string(record)
    {
        attrs.push(KeyValue::new("lash.trace.end_record_json", json));
    }
    attrs
}

fn context_attributes(
    attrs: &mut Vec<KeyValue>,
    context: &TraceContext,
    options: &OtelTraceOptions,
) {
    push_opt(attrs, "lash.context.run_id", &context.run_id);
    push_opt(attrs, "lash.context.experiment_id", &context.experiment_id);
    push_opt(attrs, "lash.context.candidate_id", &context.candidate_id);
    push_opt(
        attrs,
        "lash.context.candidate_parent_id",
        &context.candidate_parent_id,
    );
    push_opt(attrs, "lash.context.example_id", &context.example_id);
    push_opt(attrs, "lash.context.split", &context.split);
    push_opt(attrs, "lash.context.session_id", &context.session_id);
    push_opt(attrs, "lash.context.turn_id", &context.turn_id);
    push_opt(attrs, "lash.context.graph_node_id", &context.graph_node_id);
    push_opt(
        attrs,
        "lash.context.parent_graph_node_id",
        &context.parent_graph_node_id,
    );
    if let Some(iteration) = context.iteration {
        attrs.push(KeyValue::new("lash.context.iteration", iteration as i64));
    }
    push_opt(attrs, "lash.context.effect_id", &context.effect_id);
    push_opt(attrs, "lash.context.llm_call_id", &context.llm_call_id);

    if options.include_context_metadata {
        for (key, value) in &context.metadata {
            attrs.push(KeyValue::new(
                format!("lash.metadata.{key}"),
                otel_value(value),
            ));
        }
    }
}

fn event_attributes(record: &TraceRecord, options: &OtelTraceOptions) -> Vec<KeyValue> {
    let mut attrs = Vec::new();
    match &record.event {
        TraceEvent::SessionStarted { metadata } | TraceEvent::TurnStarted { metadata } => {
            attrs.push(KeyValue::new("lash.metadata.count", metadata.len() as i64));
            push_payload_json(&mut attrs, options, "lash.metadata.json", metadata);
        }
        TraceEvent::PromptBuilt {
            prompt_hash,
            prompt_chars,
            components,
        } => {
            attrs.push(KeyValue::new("lash.prompt.hash", prompt_hash.clone()));
            attrs.push(KeyValue::new("lash.prompt.chars", *prompt_chars as i64));
            attrs.push(KeyValue::new(
                "lash.prompt.component_count",
                components.len() as i64,
            ));
            push_payload_json(
                &mut attrs,
                options,
                "lash.prompt.components_json",
                components,
            );
        }
        TraceEvent::LlmCallStarted { request } => {
            attrs.push(KeyValue::new("gen_ai.request.model", request.model.clone()));
            push_opt(
                &mut attrs,
                "gen_ai.request.model_variant",
                &request.model_variant,
            );
            attrs.push(KeyValue::new("lash.llm.stream", request.stream));
            attrs.push(KeyValue::new(
                "lash.llm.tool_choice",
                request.tool_choice.clone(),
            ));
            attrs.push(KeyValue::new(
                "lash.llm.message_count",
                request.messages.len() as i64,
            ));
            attrs.push(KeyValue::new(
                "lash.llm.tool_count",
                request.tools.len() as i64,
            ));
            attrs.push(KeyValue::new(
                "lash.llm.attachment_count",
                request.attachments.len() as i64,
            ));
            push_payload_json(&mut attrs, options, "lash.llm.request_json", request);
        }
        TraceEvent::LlmCallCompleted {
            response,
            usage,
            provider_usage,
            stream_summary,
        } => {
            attrs.push(KeyValue::new(
                "lash.llm.duration_ms",
                response.duration_ms as i64,
            ));
            attrs.push(KeyValue::new(
                "gen_ai.response.text_chars",
                response.text.len() as i64,
            ));
            if let Some(usage) = usage {
                usage_attributes(&mut attrs, "gen_ai.usage", usage);
            }
            push_payload_json(
                &mut attrs,
                options,
                "lash.llm.provider_usage_json",
                provider_usage,
            );
            push_payload_json(
                &mut attrs,
                options,
                "lash.llm.stream_summary_json",
                stream_summary,
            );
            push_payload_json(&mut attrs, options, "lash.llm.response_json", response);
        }
        TraceEvent::LlmCallFailed {
            error,
            stream_summary,
        } => {
            attrs.push(KeyValue::new(
                "error.type",
                error.code.clone().unwrap_or_default(),
            ));
            attrs.push(KeyValue::new("error.message", error.message.clone()));
            attrs.push(KeyValue::new("lash.error.retryable", error.retryable));
            push_payload_json(
                &mut attrs,
                options,
                "lash.llm.stream_summary_json",
                stream_summary,
            );
            push_payload_json(&mut attrs, options, "lash.error.raw", &error.raw);
        }
        TraceEvent::ToolCallCompleted {
            call_id,
            name,
            args,
            result,
            success,
            duration_ms,
        } => {
            push_opt(&mut attrs, "lash.tool.call_id", call_id);
            attrs.push(KeyValue::new("lash.tool.name", name.clone()));
            attrs.push(KeyValue::new("lash.tool.success", *success));
            attrs.push(KeyValue::new("lash.tool.duration_ms", *duration_ms as i64));
            push_payload_json(&mut attrs, options, "lash.tool.args_json", args);
            push_payload_json(&mut attrs, options, "lash.tool.result_json", result);
        }
        TraceEvent::ModeStep { mode, payload } => {
            attrs.push(KeyValue::new("lash.mode", mode.clone()));
            push_payload_json(&mut attrs, options, "lash.mode.payload_json", payload);
        }
        TraceEvent::TokenUsage { usage, cumulative } => {
            usage_attributes(&mut attrs, "lash.usage", usage);
            if let Some(cumulative) = cumulative {
                usage_attributes(&mut attrs, "lash.usage.cumulative", cumulative);
            }
        }
        TraceEvent::TurnCompleted {
            status,
            done_reason,
        } => {
            attrs.push(KeyValue::new("lash.turn.status", status.clone()));
            attrs.push(KeyValue::new("lash.turn.done_reason", done_reason.clone()));
        }
        TraceEvent::Custom { name, payload } => {
            attrs.push(KeyValue::new("lash.custom.name", name.clone()));
            push_payload_json(&mut attrs, options, "lash.custom.payload_json", payload);
        }
    }
    attrs
}

fn usage_attributes(attrs: &mut Vec<KeyValue>, prefix: &str, usage: &TraceTokenUsage) {
    attrs.push(KeyValue::new(
        format!("{prefix}.input_tokens"),
        usage.input_tokens,
    ));
    attrs.push(KeyValue::new(
        format!("{prefix}.output_tokens"),
        usage.output_tokens,
    ));
    attrs.push(KeyValue::new(
        format!("{prefix}.cached_input_tokens"),
        usage.cached_input_tokens,
    ));
    attrs.push(KeyValue::new(
        format!("{prefix}.reasoning_tokens"),
        usage.reasoning_tokens,
    ));
}

fn push_opt(attrs: &mut Vec<KeyValue>, key: &'static str, value: &Option<String>) {
    if let Some(value) = value {
        attrs.push(KeyValue::new(key, value.clone()));
    }
}

fn push_payload_json<T: serde::Serialize>(
    attrs: &mut Vec<KeyValue>,
    options: &OtelTraceOptions,
    key: &'static str,
    value: &T,
) {
    if options.include_payload_json
        && let Ok(json) = serde_json::to_string(value)
    {
        attrs.push(KeyValue::new(key, json));
    }
}

fn otel_value(value: &Value) -> OtelValue {
    match value {
        Value::Bool(value) => OtelValue::Bool(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                OtelValue::I64(value)
            } else if let Some(value) = value.as_u64() {
                OtelValue::I64(value.min(i64::MAX as u64) as i64)
            } else if let Some(value) = value.as_f64() {
                OtelValue::F64(value)
            } else {
                OtelValue::String(value.to_string().into())
            }
        }
        Value::String(value) => OtelValue::String(value.clone().into()),
        Value::Null => OtelValue::String("null".into()),
        Value::Array(_) | Value::Object(_) => {
            OtelValue::String(serde_json::to_string(value).unwrap_or_default().into())
        }
    }
}

fn parent_for<T>(
    record: &TraceRecord,
    active: &Mutex<HashMap<String, ActiveSpan<T>>>,
) -> Option<SpanContext>
where
    T: Span,
{
    let key = turn_key(&record.context)?;
    let active = active.lock().ok()?;
    active.get(&key).map(|span| span.context.clone())
}

fn turn_key(context: &TraceContext) -> Option<String> {
    let session_id = context.session_id.as_deref()?;
    let turn_id = context
        .turn_id
        .as_deref()
        .or(context.graph_node_id.as_deref())?;
    Some(format!("turn:{session_id}:{turn_id}"))
}

fn llm_key(context: &TraceContext) -> Option<String> {
    context
        .llm_call_id
        .as_deref()
        .map(|llm_call_id| format!("llm:{llm_call_id}"))
}

fn record_time(record: &TraceRecord) -> SystemTime {
    DateTime::parse_from_rfc3339(&record.timestamp)
        .map(|time| time.with_timezone(&Utc).into())
        .unwrap_or_else(|_| SystemTime::now())
}

fn event_type(event: &TraceEvent) -> &'static str {
    match event {
        TraceEvent::SessionStarted { .. } => "session_started",
        TraceEvent::TurnStarted { .. } => "turn_started",
        TraceEvent::PromptBuilt { .. } => "prompt_built",
        TraceEvent::LlmCallStarted { .. } => "llm_call_started",
        TraceEvent::LlmCallCompleted { .. } => "llm_call_completed",
        TraceEvent::LlmCallFailed { .. } => "llm_call_failed",
        TraceEvent::ToolCallCompleted { .. } => "tool_call_completed",
        TraceEvent::ModeStep { .. } => "mode_step",
        TraceEvent::TokenUsage { .. } => "token_usage",
        TraceEvent::TurnCompleted { .. } => "turn_completed",
        TraceEvent::Custom { .. } => "custom",
    }
}

fn is_error_event(event: &TraceEvent) -> bool {
    match event {
        TraceEvent::LlmCallFailed { .. } => true,
        TraceEvent::ToolCallCompleted { success, .. } => !success,
        TraceEvent::TurnCompleted { status, .. } => status == "failed",
        _ => false,
    }
}

fn error_status(record: &TraceRecord) -> Status {
    match &record.event {
        TraceEvent::LlmCallFailed { error, .. } => Status::error(error.message.clone()),
        TraceEvent::ToolCallCompleted { name, .. } => {
            Status::error(format!("tool call failed: {name}"))
        }
        TraceEvent::TurnCompleted { status, .. } => Status::error(status.clone()),
        _ => Status::error("lash trace event failed"),
    }
}

#[cfg(test)]
mod tests {
    use opentelemetry::trace::noop::NoopTracerProvider;

    use super::*;
    use crate::{TraceLlmRequest, TraceRecord};

    #[test]
    fn otel_sink_accepts_turn_and_llm_lifecycle() {
        let tracer = NoopTracerProvider::new().tracer("test");
        let sink = OtelTraceSink::new(tracer);
        let context = TraceContext::default()
            .for_session("session-1")
            .for_llm_call("llm-1");
        let turn_context = TraceContext {
            turn_id: Some("turn-1".to_string()),
            ..context.clone()
        };

        sink.append(&TraceRecord::new(
            turn_context.clone(),
            TraceEvent::TurnStarted {
                metadata: Default::default(),
            },
        ));
        sink.append(&TraceRecord::new(
            turn_context.clone(),
            TraceEvent::LlmCallStarted {
                request: TraceLlmRequest {
                    model: "gpt-test".to_string(),
                    model_variant: None,
                    messages: Vec::new(),
                    attachments: Vec::new(),
                    tools: Vec::new(),
                    tool_choice: "auto".to_string(),
                    output_spec: None,
                    stream: true,
                },
            },
        ));
        sink.append(&TraceRecord::new(
            turn_context.clone(),
            TraceEvent::LlmCallFailed {
                error: crate::TraceError {
                    message: "boom".to_string(),
                    retryable: false,
                    code: Some("test".to_string()),
                    raw: None,
                },
                stream_summary: None,
            },
        ));
        sink.append(&TraceRecord::new(
            turn_context,
            TraceEvent::TurnCompleted {
                status: "failed".to_string(),
                done_reason: "error".to_string(),
            },
        ));

        assert!(sink.active.lock().unwrap().is_empty());
    }
}
