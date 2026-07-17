use std::collections::BTreeMap;

use crate::config::OpenAiResolvedCompat;
use lash_core::llm::transport::LlmTransportError;

/// Accumulates allowlisted wire observations for one request attempt.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResponseMetadataCapture {
    headers: Vec<String>,
    body_paths: Vec<String>,
    captured: BTreeMap<String, serde_json::Value>,
}

impl ResponseMetadataCapture {
    pub(crate) fn from_compat(compat: &OpenAiResolvedCompat) -> Self {
        Self {
            headers: compat
                .response_metadata_headers
                .iter()
                .map(|name| name.to_ascii_lowercase())
                .collect(),
            body_paths: compat.response_metadata_body_paths.clone(),
            captured: BTreeMap::new(),
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        !self.headers.is_empty() || !self.body_paths.is_empty()
    }

    pub(crate) fn capture_headers(&mut self, headers: &[(String, String)]) {
        for allowed_name in &self.headers {
            if self
                .captured
                .contains_key(&format!("header:{allowed_name}"))
            {
                continue;
            }
            if let Some((_, value)) = headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(allowed_name))
            {
                self.captured.insert(
                    format!("header:{allowed_name}"),
                    serde_json::Value::String(value.clone()),
                );
            }
        }
    }

    pub(crate) fn capture_body(&mut self, value: &serde_json::Value) {
        for pointer in &self.body_paths {
            if let Some(value) = value.pointer(pointer) {
                self.captured
                    .insert(format!("body:{pointer}"), value.clone());
            }
        }
    }

    pub(crate) fn capture_sse_body(&mut self, payload: &str) -> Result<(), LlmTransportError> {
        if !self.is_active() {
            return Ok(());
        }
        lash_llm_transport::frame_sse_payload(payload, |raw| {
            if let Ok(value) = serde_json::from_str(raw) {
                self.capture_body(&value);
            }
            Ok(())
        })
    }

    pub(crate) fn into_map(self) -> BTreeMap<String, serde_json::Value> {
        self.captured
    }
}
