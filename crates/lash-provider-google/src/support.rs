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
    http_error_envelope, serialize_options_tail, terminal_reason_from_parts,
};
pub(crate) use lash_llm_transport::streaming::{drive_sse_response, emit_stream_progress};
pub(crate) use lash_llm_transport::timeouts::{
    build_http_client, read_response_text, request_body_snapshot, response_start_timeout,
    send_request,
};
pub(crate) use lash_llm_transport::util::{emit_provider_trace, parse_i64};

pub(crate) use crate::config::*;
