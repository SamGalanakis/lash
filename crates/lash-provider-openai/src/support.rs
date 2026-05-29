pub(crate) use async_trait::async_trait;
pub(crate) use base64::Engine;
pub(crate) use serde::Deserialize;
pub(crate) use serde_json::{Value, json};
pub(crate) use std::collections::HashMap;

pub(crate) use lash_core::SchemaProjectionOverride;
pub(crate) use lash_core::llm::transport::{LlmTransportError, validate_image_attachments};
pub(crate) use lash_core::llm::types::{
    LlmAttachment, LlmContentBlock, LlmEventSender, LlmOutputPart, LlmOutputSpec,
    LlmProviderTraceSender, LlmRequest, LlmResponse, LlmRole, LlmStreamEvent, LlmTerminalReason,
    LlmUsage, ProviderReplayMeta,
};
// `ResponseTextMeta` is only referenced by the crate's `#[cfg(test)]`
// assertions (the request/response shapes that exercise the shared Responses
// input builder), so gate the re-export to test builds to keep the non-test
// lib free of unused-import warnings.
#[cfg(test)]
pub(crate) use lash_core::llm::types::ResponseTextMeta;
pub(crate) use lash_core::provider::{
    CacheRetention, Provider, ProviderComponents, ProviderFactory, ProviderModelPolicy,
    ProviderOptions, resolve_generation_policy,
};
pub(crate) use lash_llm_transport::streaming::{drive_sse_response, emit_stream_progress};
pub(crate) use lash_llm_transport::timeouts::{
    build_http_client, header_pairs, read_response_text, request_body_snapshot_bytes,
    response_start_timeout, send_request,
};
pub(crate) use lash_llm_transport::util::{
    OPENAI_IMAGE_MIMES, emit_provider_trace, extract_error_detail,
};
pub(crate) use lash_openai_schema::{OpenAiSchemaProfile, model_id, responses_error_is_retryable};

pub(crate) use crate::chat::*;
pub(crate) use crate::common::*;
pub(crate) use crate::config::*;
pub(crate) use crate::driver::*;
pub(crate) use crate::policy::*;
pub(crate) use crate::responses_shared::{ResponsesStreamState, role_name, tool_choice_value};
