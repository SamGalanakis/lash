//! Crate-internal prelude. Submodules `use crate::support::*` to share the
//! common imports without repeating the list, mirroring the OpenAI crate's
//! layout.

pub(crate) use async_trait::async_trait;
pub(crate) use base64::Engine;
pub(crate) use serde::Deserialize;
pub(crate) use serde_json::{Value, json};

pub(crate) use lash_core::llm::transport::{
    LlmTransportError, ProviderFailureKind, validate_image_attachments,
};
pub(crate) use lash_core::llm::types::{
    LlmContentBlock, LlmEventSender, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse,
    LlmRole, LlmStreamEvent, LlmTerminalReason, LlmToolChoice, LlmUsage, ProviderReasoningReplay,
};
pub(crate) use lash_core::provider::{
    CacheRetention, Provider, ProviderComponents, ProviderFactory, ProviderModelPolicy,
    ProviderOptions, resolve_generation_policy,
};
pub(crate) use lash_core::{
    ProviderSchemaCapabilities, SchemaPurpose, SchemaResolutionError, SchemaResolutionRequest,
    resolve_schema,
};
pub(crate) use lash_llm_transport::normalize::{
    merge_usage, serialize_options_tail, terminal_reason_from_parts,
};
pub(crate) use lash_llm_transport::streaming::drive_sse_response;
pub(crate) use lash_llm_transport::timeouts::response_start_timeout;
pub(crate) use lash_llm_transport::util::{OPENAI_IMAGE_MIMES, emit_provider_trace};
pub(crate) use lash_llm_transport::{
    LlmHttpRequest, LlmHttpTransport, ReqwestLlmHttpTransport, read_http_body_text,
};

pub(crate) use crate::config::*;
pub(crate) use crate::policy::*;
