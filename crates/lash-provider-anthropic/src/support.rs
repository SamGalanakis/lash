//! Crate-internal prelude. Submodules `use crate::support::*` to share the
//! common imports without repeating the list, mirroring the OpenAI crate's
//! layout.

pub(crate) use async_trait::async_trait;
pub(crate) use base64::Engine;
pub(crate) use serde::Deserialize;
pub(crate) use serde_json::{Value, json};

pub(crate) use lash_core::llm::transport::{
    LlmTransportError, ProviderFailureKind, unsupported_attachment_capability,
};
pub(crate) use lash_core::llm::types::{
    AttachmentSource, LlmContentBlock, LlmEventSender, LlmOutputPart, LlmOutputSpec, LlmRequest,
    LlmResponse, LlmRole, LlmStreamEvent, LlmTerminalReason, LlmToolChoice, LlmUsage,
    ProviderReasoningReplay,
};
pub(crate) use lash_core::provider::{
    CacheRetention, Provider, ProviderComponents, ProviderFactory, ProviderOptions,
    ReasoningDisableEncoding, ReasoningEncoding, ReasoningSelection, StreamTermination,
    resolve_generation_policy,
};
pub(crate) use lash_core::{
    ProviderSchemaCapabilities, SchemaPurpose, SchemaResolutionError, SchemaResolutionRequest,
    resolve_schema,
};
pub(crate) use lash_llm_transport::normalize::{
    http_error_envelope, merge_usage, serialize_options_tail, terminal_reason_from_parts,
};
pub(crate) use lash_llm_transport::streaming::drive_sse_response;
pub(crate) use lash_llm_transport::timeouts::response_start_timeout;
pub(crate) use lash_llm_transport::util::{emit_provider_request_trace, emit_provider_trace};
pub(crate) use lash_llm_transport::{
    LlmHttpRequest, LlmHttpTransport, ReqwestLlmHttpTransport, read_http_body_text,
};

pub(crate) use crate::config::*;
pub(crate) use crate::policy::*;

pub(crate) const ANTHROPIC_IMAGE_MIMES: &[&str] =
    &["image/jpeg", "image/png", "image/gif", "image/webp"];

pub(crate) fn known_attachment_acceptors(source: &AttachmentSource) -> Vec<&'static str> {
    if let AttachmentSource::ProviderFile { provider_scope, .. } = source {
        return match provider_scope.provider.to_ascii_lowercase().as_str() {
            "openai" => vec!["OpenAI Responses"],
            "anthropic" => vec!["Anthropic Messages"],
            "google" | "google_oauth" | "gemini" => vec!["Google Gemini"],
            _ => Vec::new(),
        };
    }
    const OPENAI_FILES: &[&str] = &[
        "application/pdf",
        "application/json",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/csv",
        "text/html",
        "text/markdown",
        "text/plain",
    ];
    let mime = source.media_type().expect("MIME-bearing source").as_str();
    let borrowed_url = matches!(source, AttachmentSource::ExternalUrl { .. });
    let mut providers = Vec::new();
    if ANTHROPIC_IMAGE_MIMES.contains(&mime) || OPENAI_FILES.contains(&mime) {
        providers.push("OpenAI Responses");
    }
    if ANTHROPIC_IMAGE_MIMES.contains(&mime) {
        providers.push("OpenAI Chat Completions");
        providers.push("Anthropic Messages");
    } else if mime == "application/pdf" {
        providers.push("Anthropic Messages");
    }
    if !borrowed_url
        && (matches!(
            mime,
            "image/jpeg" | "image/png" | "image/webp" | "image/heic" | "image/heif"
        ) || matches!(mime.split_once('/'), Some(("audio" | "text" | "video", _)))
            || mime == "application/pdf")
    {
        providers.push("Google Gemini");
    }
    providers
}
