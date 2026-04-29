use std::sync::Arc;

use lash_trace::{
    TraceAttachment, TraceContentBlock, TraceContext, TraceEvent, TraceLlmMessage, TraceLlmRequest,
    TraceLlmResponse, TraceRecord, TraceSink, TraceTokenUsage, TraceToolSpec, sha256_hex,
};

use crate::llm::types::{
    LlmAttachment, LlmContentBlock, LlmMessage, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmRole,
    LlmToolChoice, LlmUsage,
};
use crate::session_model::TokenUsage;

pub(crate) fn emit_trace(
    sink: &Option<Arc<dyn TraceSink>>,
    base_context: &TraceContext,
    context: TraceContext,
    event: TraceEvent,
) {
    let Some(sink) = sink else {
        return;
    };
    let mut merged = base_context.clone();
    merge_context(&mut merged, context);
    sink.append(&TraceRecord::new(merged, event));
}

fn merge_context(base: &mut TraceContext, overlay: TraceContext) {
    if overlay.run_id.is_some() {
        base.run_id = overlay.run_id;
    }
    if overlay.experiment_id.is_some() {
        base.experiment_id = overlay.experiment_id;
    }
    if overlay.candidate_id.is_some() {
        base.candidate_id = overlay.candidate_id;
    }
    if overlay.candidate_parent_id.is_some() {
        base.candidate_parent_id = overlay.candidate_parent_id;
    }
    if overlay.example_id.is_some() {
        base.example_id = overlay.example_id;
    }
    if overlay.split.is_some() {
        base.split = overlay.split;
    }
    if overlay.session_id.is_some() {
        base.session_id = overlay.session_id;
    }
    if overlay.turn_id.is_some() {
        base.turn_id = overlay.turn_id;
    }
    if overlay.graph_node_id.is_some() {
        base.graph_node_id = overlay.graph_node_id;
    }
    if overlay.parent_graph_node_id.is_some() {
        base.parent_graph_node_id = overlay.parent_graph_node_id;
    }
    if overlay.iteration.is_some() {
        base.iteration = overlay.iteration;
    }
    if overlay.effect_id.is_some() {
        base.effect_id = overlay.effect_id;
    }
    if overlay.llm_call_id.is_some() {
        base.llm_call_id = overlay.llm_call_id;
    }
    base.metadata.extend(overlay.metadata);
}

pub(crate) fn trace_llm_request(req: &LlmRequest) -> TraceLlmRequest {
    TraceLlmRequest {
        model: req.model.clone(),
        model_variant: req.model_variant.clone(),
        messages: req.messages.iter().map(trace_llm_message).collect(),
        attachments: req.attachments.iter().map(trace_attachment).collect(),
        tools: req
            .tools
            .iter()
            .map(|tool| TraceToolSpec {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.input_schema.clone(),
                output_schema: tool.output_schema.clone(),
            })
            .collect(),
        tool_choice: match req.tool_choice {
            LlmToolChoice::Auto => "auto",
            LlmToolChoice::None => "none",
            LlmToolChoice::Required => "required",
        }
        .to_string(),
        output_spec: req.output_spec.as_ref().map(trace_output_spec),
        stream: req.stream_events.is_some(),
    }
}

fn trace_llm_message(message: &LlmMessage) -> TraceLlmMessage {
    TraceLlmMessage {
        role: match message.role {
            LlmRole::User => "user",
            LlmRole::Assistant => "assistant",
            LlmRole::System => "system",
        }
        .to_string(),
        blocks: message.blocks.iter().map(trace_content_block).collect(),
    }
}

fn trace_content_block(block: &LlmContentBlock) -> TraceContentBlock {
    match block {
        LlmContentBlock::Text { text, .. } => TraceContentBlock::Text {
            text: text.to_string(),
        },
        LlmContentBlock::Image { attachment_idx } => TraceContentBlock::Image {
            attachment_idx: *attachment_idx,
        },
        LlmContentBlock::ToolCall {
            call_id,
            tool_name,
            input_json,
            item_id,
            signature,
        } => TraceContentBlock::ToolCall {
            call_id: Some(call_id.clone()),
            tool_name: tool_name.clone(),
            input_json: serde_json::from_str(input_json)
                .unwrap_or_else(|_| serde_json::Value::String(input_json.clone())),
            item_id: item_id.clone(),
            has_signature: signature.is_some(),
        },
        LlmContentBlock::ToolResult {
            call_id,
            content,
            tool_name,
        } => TraceContentBlock::ToolResult {
            call_id: Some(call_id.clone()),
            tool_name: tool_name.clone(),
            content: content.clone(),
        },
        LlmContentBlock::Reasoning {
            text,
            signature,
            redacted,
            item_id,
            encrypted_content,
            summary,
        } => TraceContentBlock::Reasoning {
            text: text.clone(),
            item_id: item_id.clone(),
            summary: summary.clone(),
            has_encrypted: encrypted_content.is_some() || signature.is_some(),
            redacted: *redacted,
        },
    }
}

fn trace_attachment(attachment: &LlmAttachment) -> TraceAttachment {
    TraceAttachment {
        mime: attachment.mime.clone(),
        filename: None,
        bytes_sha256: Some(sha256_hex(&attachment.data)),
        bytes_len: Some(attachment.data.len()),
    }
}

fn trace_output_spec(spec: &LlmOutputSpec) -> serde_json::Value {
    match spec {
        LlmOutputSpec::JsonObject => serde_json::json!({ "type": "json_object" }),
        LlmOutputSpec::JsonSchema(schema) => serde_json::json!({
            "type": "json_schema",
            "name": schema.name,
            "schema": schema.schema,
            "strict": schema.strict,
        }),
    }
}

pub(crate) fn trace_llm_response(
    text: String,
    duration_ms: u64,
    parts: Option<serde_json::Value>,
) -> TraceLlmResponse {
    TraceLlmResponse {
        text,
        duration_ms,
        parts,
    }
}

pub(crate) fn trace_usage_from_llm(usage: &LlmUsage) -> TraceTokenUsage {
    TraceTokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    }
}

pub(crate) fn trace_usage_from_session(usage: &TokenUsage) -> TraceTokenUsage {
    TraceTokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    }
}

pub(crate) fn trace_output_parts(parts: &[LlmOutputPart]) -> Option<serde_json::Value> {
    let parts = parts
        .iter()
        .map(|part| match part {
            LlmOutputPart::Text { text, .. } => serde_json::json!({
                "type": "text",
                "text": text,
            }),
            LlmOutputPart::Reasoning {
                text,
                item_id,
                summary,
                encrypted_content,
                signature,
                redacted,
            } => serde_json::json!({
                "type": "reasoning",
                "id": item_id,
                "summary": summary,
                "text": text,
                "has_encrypted": encrypted_content.is_some() || signature.is_some(),
                "redacted": redacted,
            }),
            LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                item_id,
                signature,
            } => serde_json::json!({
                "type": "tool_call",
                "call_id": call_id,
                "tool_name": tool_name,
                "input_json": input_json,
                "id": item_id,
                "has_signature": signature.is_some(),
            }),
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then_some(serde_json::Value::Array(parts))
}
