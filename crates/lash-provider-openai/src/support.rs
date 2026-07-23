pub(crate) use async_trait::async_trait;
pub(crate) use base64::Engine;
pub(crate) use serde::Deserialize;
pub(crate) use serde_json::{Value, json};
pub(crate) use std::collections::HashMap;

pub(crate) use lash_core::llm::transport::{
    LlmTransportError, ProviderFailureKind, unsupported_attachment_capability,
};
pub(crate) use lash_core::llm::types::{
    AttachmentSource, ExecutionEvidence, LlmContentBlock, LlmEventSender, LlmOutputPart,
    LlmOutputSpec, LlmProviderTraceSender, LlmRequest, LlmResponse, LlmRole, LlmStreamEvent,
    LlmTerminalReason, LlmUsage, ProviderReplayMeta,
};
pub(crate) use lash_core::{ProviderSchemaCapabilities, SchemaPurpose};
// `ResponseTextMeta` is only referenced by the crate's `#[cfg(test)]`
// assertions (the request/response shapes that exercise the shared Responses
// input builder), so gate the re-export to test builds to keep the non-test
// lib free of unused-import warnings.
pub(crate) use crate::schema::responses_error_is_retryable;
#[cfg(test)]
pub(crate) use lash_core::llm::types::{LlmRequestScope, ResponseTextMeta};
pub(crate) use lash_core::provider::{
    CacheControlDialect, CacheRetention, Provider, ProviderComponents, ProviderFactory,
    ProviderOptions, StreamTermination, resolve_generation_policy,
};
pub(crate) use lash_llm_transport::streaming::{drive_sse_response, emit_stream_progress};
pub(crate) use lash_llm_transport::timeouts::response_start_timeout;
pub(crate) use lash_llm_transport::util::{
    emit_provider_request_trace, emit_provider_trace, extract_error_detail,
};
pub(crate) use lash_llm_transport::{
    LlmHttpBody, LlmHttpMethod, LlmHttpRequest, LlmHttpTransport, first_header_value,
    header_contains, http_error_envelope, read_http_body_text,
};

pub(crate) use crate::chat::*;
pub(crate) use crate::common::*;
pub(crate) use crate::config::*;
pub(crate) use crate::driver::*;

pub(crate) const OPENAI_IMAGE_MIMES: &[&str] =
    &["image/jpeg", "image/png", "image/gif", "image/webp"];
pub(crate) const OPENAI_FILE_MIMES: &[&str] = &[
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

pub(crate) fn known_attachment_acceptors(source: &AttachmentSource) -> Vec<&'static str> {
    if let AttachmentSource::ProviderFile { provider_scope, .. } = source {
        return match provider_scope.provider.to_ascii_lowercase().as_str() {
            "openai" => vec!["OpenAI Responses"],
            "anthropic" => vec!["Anthropic Messages"],
            "google" | "google_oauth" | "gemini" => vec!["Google Gemini"],
            _ => Vec::new(),
        };
    }
    let mime = source.media_type().expect("MIME-bearing source").as_str();
    let borrowed_url = matches!(source, AttachmentSource::ExternalUrl { .. });
    let mut providers = Vec::new();
    if OPENAI_IMAGE_MIMES.contains(&mime) || OPENAI_FILE_MIMES.contains(&mime) {
        providers.push("OpenAI Responses");
    }
    if OPENAI_IMAGE_MIMES.contains(&mime) {
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
pub(crate) use crate::reasoning::*;
pub(crate) use crate::response_metadata::*;
pub(crate) use crate::responses_shared::{ResponsesStreamState, role_name, tool_choice_value};
