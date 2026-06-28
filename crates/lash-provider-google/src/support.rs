//! Crate-internal prelude. Submodules `use crate::support::*` to share the
//! common imports without repeating the list, mirroring the OpenAI crate's
//! layout.

pub(crate) use async_trait::async_trait;
pub(crate) use base64::Engine;
pub(crate) use serde::Deserialize;
pub(crate) use serde_json::{Value, json};

pub(crate) use lash_core::llm::transport::{LlmTransportError, validate_image_attachments};
pub(crate) use lash_core::llm::types::{
    LlmAttachment, LlmContentBlock, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse, LlmRole,
    LlmTerminalReason, LlmToolChoice, LlmUsage, ProviderReasoningReplay, ProviderReplayMeta,
    ResponseTextMeta,
};
pub(crate) use lash_core::provider::{
    Provider, ProviderComponents, ProviderFactory, ProviderModelPolicy, ProviderOptions,
    resolve_generation_policy,
};
pub(crate) use lash_llm_transport::normalize::{
    serialize_options_tail, terminal_reason_from_parts,
};
pub(crate) use lash_llm_transport::streaming::{drive_sse_response, emit_stream_progress};
pub(crate) use lash_llm_transport::timeouts::response_start_timeout;
pub(crate) use lash_llm_transport::util::{emit_provider_trace, parse_i64};
pub(crate) use lash_llm_transport::{
    LlmHttpRequest, LlmHttpTransport, ReqwestLlmHttpTransport, first_header_value,
    read_http_body_text,
};

pub(crate) use crate::config::*;

pub(crate) fn http_error_envelope_from_pairs(
    message: impl Into<String>,
    status: u16,
    headers: Vec<(String, String)>,
    raw_body: impl Into<String>,
    request_body: Option<String>,
) -> LlmTransportError {
    let mut err = LlmTransportError::new(message)
        .with_status(status)
        .with_headers(headers)
        .with_raw(raw_body);
    if let Some(request_body) = request_body {
        err = err.with_request_body(request_body);
    }
    err
}
