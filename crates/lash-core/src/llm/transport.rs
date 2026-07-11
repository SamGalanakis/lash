//! Transport-level failure type shared by provider transport components.

pub use lash_http_transport::retry_after_from_headers;
/// Canonical provider-failure classification.
pub use lash_sansio::llm::types::ProviderFailureKind;
pub type ProviderFailure = lash_http_transport::HttpTransportError;
pub type LlmTransportError = lash_http_transport::HttpTransportError;

/// Validate that every image attachment in `req` carries a MIME type accepted
/// by the provider. Returns a `Validation`-kind `LlmTransportError` with code
/// `unsupported_image_format` and a descriptive message on the first
/// unsupported attachment, naming the provider and the offending MIME.
///
/// Provider adapters call this at the top of their request-building pipeline
/// to fail fast with a clear runtime-side error rather than relying on the
/// upstream API to reject the request with a less actionable message.
#[expect(
    clippy::result_large_err,
    reason = "provider transport errors are a public typed API and carry request/response diagnostics"
)]
pub fn validate_image_attachments(
    req: &lash_sansio::llm::types::LlmRequest,
    accepted_mimes: &[&str],
    provider_name: &str,
) -> Result<(), LlmTransportError> {
    for (idx, att) in req.attachments.iter().enumerate() {
        let mime = att.mime.trim().to_ascii_lowercase();
        let normalized = if mime == "image/jpg" {
            "image/jpeg"
        } else {
            mime.as_str()
        };
        if !accepted_mimes.contains(&normalized) {
            return Err(ProviderFailure::new(format!(
                "{provider_name} does not accept image attachments of type `{}` (attachment index {idx}); accepted: {}",
                att.mime,
                accepted_mimes.join(", "),
            ))
            .with_kind(ProviderFailureKind::Validation)
            .with_code("unsupported_image_format"));
        }
    }
    Ok(())
}
