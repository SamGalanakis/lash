//! Gemini Files resumable upload: hash-keyed caching of uploaded image
//! attachments and the two-step (start / upload+finalize) upload protocol.

use sha2::{Digest, Sha256};

use crate::config::{UploadedAttachmentCacheKey, UploadedAttachmentRef};
use crate::support::*;

const GEMINI_FILES_UPLOAD_URL: &str =
    "https://generativelanguage.googleapis.com/upload/v1beta/files";

impl GoogleOAuthProvider {
    fn upload_cache_key(
        project_id: Option<&str>,
        att: &LlmAttachment,
    ) -> UploadedAttachmentCacheKey {
        UploadedAttachmentCacheKey {
            project_id: project_id.unwrap_or_default().to_string(),
            mime: att.mime.clone(),
            hash: format!("{:x}", Sha256::digest(&att.data)),
        }
    }

    fn uploaded_attachment_filename(key: &UploadedAttachmentCacheKey) -> String {
        let ext = match key.mime.as_str() {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/jpg" => "jpg",
            "image/webp" => "webp",
            "image/gif" => "gif",
            "image/heic" => "heic",
            "image/heif" => "heif",
            "image/bmp" => "bmp",
            "image/tiff" => "tiff",
            _ => "bin",
        };
        format!("lash-{}.{}", &key.hash[..12], ext)
    }

    async fn upload_attachment_cached(
        &self,
        access_token: &str,
        project_id: Option<&str>,
        att: &LlmAttachment,
    ) -> Result<UploadedAttachmentRef, LlmTransportError> {
        let key = Self::upload_cache_key(project_id, att);
        if let Some(existing) = Self::uploaded_attachment_cache()
            .lock()
            .await
            .get(&key)
            .cloned()
        {
            return Ok(existing);
        }

        let uploaded = self
            .upload_attachment(
                access_token,
                project_id,
                att,
                &Self::uploaded_attachment_filename(&key),
            )
            .await?;
        Self::uploaded_attachment_cache()
            .lock()
            .await
            .insert(key, uploaded.clone());
        Ok(uploaded)
    }

    async fn upload_attachment(
        &self,
        access_token: &str,
        project_id: Option<&str>,
        att: &LlmAttachment,
        filename: &str,
    ) -> Result<UploadedAttachmentRef, LlmTransportError> {
        let mut start = self
            .client
            .post(GEMINI_FILES_UPLOAD_URL)
            .bearer_auth(access_token)
            .header("Content-Type", "application/json")
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header(
                "X-Goog-Upload-Header-Content-Length",
                att.data.len().to_string(),
            )
            .header("X-Goog-Upload-Header-Content-Type", att.mime.as_str())
            .header("X-Goog-Upload-File-Name", filename)
            .json(&json!({
                "file": {
                    "displayName": filename,
                    "mimeType": att.mime,
                    "sizeBytes": att.data.len().to_string(),
                }
            }));
        if let Some(project_id) = project_id.filter(|project_id| !project_id.trim().is_empty()) {
            start = start.header("x-goog-user-project", project_id);
        }

        let start_resp = send_request(
            start,
            None,
            self.options.llm_timeouts().request_timeout,
            "Gemini Files upload start timed out",
        )
        .await?;
        if !start_resp.status().is_success() {
            let status = start_resp.status().as_u16();
            let headers = start_resp.headers().clone();
            let body = read_response_text(
                start_resp,
                self.options.llm_timeouts().request_timeout,
                "Gemini Files upload start body timed out",
            )
            .await
            .unwrap_or_default();
            return Err(http_error_envelope(
                format!("Gemini Files upload start failed with {}", status),
                status,
                &headers,
                body,
                None,
            ));
        }

        let upload_url = start_resp
            .headers()
            .get("x-goog-upload-url")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                LlmTransportError::new(
                    "Gemini Files upload start response missing x-goog-upload-url header",
                )
            })?
            .to_string();

        let mut finalize = self
            .client
            .post(upload_url)
            .bearer_auth(access_token)
            .header("X-Goog-Upload-Command", "upload, finalize")
            .header("X-Goog-Upload-Offset", "0")
            .header("Content-Length", att.data.len().to_string())
            .body(att.data.clone());
        if let Some(project_id) = project_id.filter(|project_id| !project_id.trim().is_empty()) {
            finalize = finalize.header("x-goog-user-project", project_id);
        }

        let finalize_resp = send_request(
            finalize,
            None,
            self.options.llm_timeouts().request_timeout,
            "Gemini Files upload finalize timed out",
        )
        .await?;
        if !finalize_resp.status().is_success() {
            let status = finalize_resp.status().as_u16();
            let headers = finalize_resp.headers().clone();
            let body = read_response_text(
                finalize_resp,
                self.options.llm_timeouts().request_timeout,
                "Gemini Files upload finalize body timed out",
            )
            .await
            .unwrap_or_default();
            return Err(http_error_envelope(
                format!("Gemini Files upload finalize failed with {}", status),
                status,
                &headers,
                body,
                None,
            ));
        }

        let upload_status = finalize_resp
            .headers()
            .get("x-goog-upload-status")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = read_response_text(
            finalize_resp,
            self.options.llm_timeouts().request_timeout,
            "Gemini Files upload finalize body timed out",
        )
        .await?;
        if upload_status
            .as_deref()
            .is_some_and(|status| status != "final")
        {
            return Err(LlmTransportError::new(format!(
                "Gemini Files upload finalize returned unexpected status `{}`",
                upload_status.unwrap_or_default()
            ))
            .with_raw(body));
        }

        let value: Value = serde_json::from_str(&body).map_err(|err| {
            LlmTransportError::new(format!("Invalid Gemini Files upload JSON: {err}"))
                .with_raw(body.clone())
        })?;
        let file = value.get("file").unwrap_or(&value);
        let uri = if let Some(uri) = file.get("uri").and_then(|value| value.as_str()) {
            uri.to_string()
        } else if let Some(name) = file.get("name").and_then(|value| value.as_str()) {
            format!("https://generativelanguage.googleapis.com/v1beta/{name}")
        } else {
            return Err(
                LlmTransportError::new("Gemini Files upload response missing file uri")
                    .with_raw(body.clone()),
            );
        };

        Ok(UploadedAttachmentRef { uri })
    }

    pub(crate) async fn prepare_attachment_parts(
        &self,
        access_token: &str,
        project_id: Option<&str>,
        attachments: &[LlmAttachment],
    ) -> (Vec<Value>, bool) {
        let mut parts = Vec::with_capacity(attachments.len());
        let mut used_uploaded_files = false;

        for att in attachments {
            if !att.mime.starts_with("image/") {
                parts.push(Self::inline_attachment_part(att));
                continue;
            }

            match self
                .upload_attachment_cached(access_token, project_id, att)
                .await
            {
                Ok(uploaded) => {
                    used_uploaded_files = true;
                    parts.push(json!({
                        "fileData": {
                            "mimeType": att.mime,
                            "fileUri": uploaded.uri,
                        }
                    }));
                }
                Err(_) => parts.push(Self::inline_attachment_part(att)),
            }
        }

        (parts, used_uploaded_files)
    }
}
