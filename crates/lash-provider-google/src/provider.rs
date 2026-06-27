//! The [`Provider`] trait implementation: config serialization, the `complete`
//! orchestration (attachment prep, request build, inline-fallback retry), the
//! request/stream executor, and project-id resolution.

use crate::support::*;

impl GoogleOAuthProvider {
    fn should_retry_inline(err: &LlmTransportError) -> bool {
        matches!(err.code.as_deref(), Some("400" | "404"))
            || err.raw.as_deref().is_some_and(|raw| {
                raw.contains("fileData") || raw.contains("fileUri") || raw.contains("file_uri")
            })
    }

    async fn execute_request(
        &self,
        access_token: &str,
        request: Value,
        stream_events: Option<lash_core::llm::types::LlmEventSender>,
        provider_trace: Option<lash_core::llm::types::LlmProviderTraceSender>,
    ) -> Result<LlmResponse, LlmTransportError> {
        let request_body = serde_json::to_string(&request).ok();
        let method = if stream_events.is_some() {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let url = Self::method_url(method);
        let mut http = self
            .client
            .post(&url)
            .bearer_auth(access_token)
            .json(&request);
        if stream_events.is_some() {
            http = http.query(&[("alt", "sse")]);
        }
        let resp = send_request(
            http,
            request_body.clone().map(request_body_snapshot),
            response_start_timeout(
                self.options.llm_timeouts().request_timeout,
                self.options.llm_timeouts().chunk_timeout,
                stream_events.is_some(),
            ),
            "Cloud Code response start timed out",
        )
        .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let body = read_response_text(
                resp,
                self.options.llm_timeouts().request_timeout,
                "Cloud Code response body timed out",
            )
            .await
            .unwrap_or_default();
            return Err(http_error_envelope(
                format!("Cloud Code request failed with {}", status),
                status,
                &headers,
                body,
                request_body,
            ));
        }

        if stream_events.is_none() {
            let text = read_response_text(
                resp,
                self.options.llm_timeouts().request_timeout,
                "Cloud Code response body timed out",
            )
            .await?;
            emit_provider_trace(provider_trace.as_ref(), "google", &text);
            let value: Value = serde_json::from_str(&text).map_err(|e| {
                LlmTransportError::new(format!("Invalid Cloud Code response JSON: {e}"))
                    .with_raw(text.clone())
            })?;
            let origin_model = request.get("model").and_then(Value::as_str);
            let parts = Self::response_parts_from_value(&value, origin_model);
            let full_text = lash_core::visible_response_text_from_parts(&parts);
            let usage = value
                .get("usageMetadata")
                .map(|meta| {
                    Self::usage_from_event(&json!({
                        "response": {
                            "usageMetadata": meta
                        }
                    }))
                })
                .unwrap_or_default();
            let terminal_reason = Self::terminal_reason_from_value(&value, &parts);
            return Ok(LlmResponse {
                full_text,
                parts,
                usage,
                terminal_reason,
                terminal_diagnostic: None,
                provider_usage: None,
                request_body,
                http_summary: Some(format!("HTTP POST {}", url)),
            });
        }

        let mut full = String::new();
        let mut usage = LlmUsage::default();
        let mut text_parts: Vec<LlmOutputPart> = Vec::new();
        let mut tool_call_parts: Vec<LlmOutputPart> = Vec::new();
        let mut finish_event: Option<Value> = None;
        let origin_model = request
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string);
        drive_sse_response(
            LlmHttpBody::from_reqwest_response(resp),
            self.options.llm_timeouts().chunk_timeout,
            "Cloud Code stream chunk timed out",
            |raw| {
                emit_provider_trace(provider_trace.as_ref(), "google", raw);
                let mut text_deltas = Vec::new();
                let prev_usage = usage.clone();
                Self::process_sse_event_with_text_parts(
                    raw,
                    &mut full,
                    &mut text_deltas,
                    &mut usage,
                    Some(&mut tool_call_parts),
                    Some(&mut text_parts),
                    origin_model.as_deref(),
                    &mut finish_event,
                )?;
                emit_stream_progress(stream_events.as_ref(), text_deltas, &usage, &prev_usage);
                Ok(())
            },
        )
        .await?;

        let mut parts = text_parts;
        if parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>()
            .is_empty()
            && !full.is_empty()
        {
            parts.push(LlmOutputPart::Text {
                text: full.clone(),
                response_meta: None,
            });
        }
        parts.extend(tool_call_parts);

        // Mirror the non-streaming path: derive the terminal reason from the
        // last `finishReason` observed across the SSE events. When no event
        // carried one, `terminal_reason_from_value` on a value without a
        // finishReason falls back to ToolUse/Stop from the assembled parts.
        let terminal_reason =
            Self::terminal_reason_from_value(finish_event.as_ref().unwrap_or(&Value::Null), &parts);

        Ok(LlmResponse {
            full_text: full,
            parts,
            usage,
            terminal_reason,
            terminal_diagnostic: None,
            provider_usage: None,
            request_body,
            http_summary: Some(format!("HTTP POST {}?alt=sse", url)),
        })
    }

    async fn resolve_project_id(
        &self,
        access_token: &str,
        project_hint: Option<&str>,
    ) -> Result<Option<String>, LlmTransportError> {
        let mut metadata = json!({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        });
        if let Some(project) = project_hint.filter(|p| !p.trim().is_empty()) {
            metadata["duetProject"] = json!(project);
        }

        let req = json!({
            "cloudaicompanionProject": project_hint,
            "metadata": metadata,
        });
        let request_body = serde_json::to_string(&req).ok();

        let resp = self
            .client
            .post(Self::method_url("loadCodeAssist"))
            .bearer_auth(access_token)
            .json(&req)
            .send()
            .await
            .map_err(|e| {
                let error = LlmTransportError::new(format!("HTTP request failed: {e}"));
                if let Some(request_body) = request_body.clone() {
                    error.with_request_body(request_body)
                } else {
                    error
                }
            })?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let body = resp.text().await.unwrap_or_default();
            return Err(http_error_envelope(
                format!("Cloud Code loadCodeAssist failed with {}", status),
                status,
                &headers,
                body,
                request_body,
            ));
        }
        let body: Value = resp.json().await.map_err(|e| {
            LlmTransportError::new(format!("Invalid Cloud Code loadCodeAssist JSON: {e}"))
        })?;
        Ok(body
            .get("cloudaicompanionProject")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| project_hint.map(|s| s.to_string())))
    }
}

#[async_trait]
impl Provider for GoogleOAuthProvider {
    fn kind(&self) -> &'static str {
        "google_oauth"
    }

    fn options(&self) -> ProviderOptions {
        self.options.clone()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.options = options;
    }

    fn serialize_config(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "access_token".to_string(),
            serde_json::Value::String(self.access_token.clone()),
        );
        map.insert(
            "refresh_token".to_string(),
            serde_json::Value::String(self.refresh_token.clone()),
        );
        map.insert(
            "expires_at".to_string(),
            serde_json::Value::Number(self.expires_at.into()),
        );
        if let Some(project_id) = &self.project_id {
            map.insert(
                "project_id".to_string(),
                serde_json::Value::String(project_id.clone()),
            );
        }
        serialize_options_tail(&mut map, &self.options);
        serde_json::Value::Object(map)
    }

    async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        validate_image_attachments(
            &req,
            &[
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/heic",
                "image/heif",
            ],
            "Google Gemini",
        )?;
        let stream_events = req.stream_events.clone();
        let provider_trace = req.provider_trace.clone();
        let access_token = self.access_token.clone();
        if self.project_id.is_none() {
            let hint = std::env::var("GOOGLE_CLOUD_PROJECT")
                .ok()
                .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT_ID").ok());
            self.project_id = self
                .resolve_project_id(&access_token, hint.as_deref())
                .await?;
        }
        let project_id = self.project_id.clone();

        let inline_attachment_parts = req
            .attachments
            .iter()
            .map(Self::inline_attachment_part)
            .collect::<Vec<_>>();
        let inline_contents =
            Self::build_contents_with_attachment_parts(&req, &inline_attachment_parts);

        let (attachment_parts, used_uploaded_files) = self
            .prepare_attachment_parts(&access_token, project_id.as_deref(), &req.attachments)
            .await;
        let contents = if used_uploaded_files {
            Self::build_contents_with_attachment_parts(&req, &attachment_parts)
        } else {
            inline_contents.clone()
        };

        let request = Self::build_request(self, &req, contents, project_id.as_deref());

        match self
            .execute_request(
                &access_token,
                request,
                stream_events.clone(),
                provider_trace.clone(),
            )
            .await
        {
            Ok(response) => Ok(response),
            Err(err) if used_uploaded_files && Self::should_retry_inline(&err) => {
                let inline_request =
                    Self::build_request(self, &req, inline_contents, project_id.as_deref());
                self.execute_request(&access_token, inline_request, stream_events, provider_trace)
                    .await
            }
            Err(err) => Err(err),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}
