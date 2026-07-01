//! Request-body construction: translating an [`LlmRequest`] into the Cloud
//! Code (Gemini) wire shape (contents, systemInstruction, tools, generation
//! and thinking config), plus the inline-attachment-part helpers.

use crate::policy::{GoogleModelPolicy, GoogleThinkingConfig};
use crate::support::*;

/// Pi-mono sentinel: Gemini 3 refuses to run when a function_call is
/// replayed without a thoughtSignature. The server recognises this magic
/// string and skips signature validation for the item, so lash can round-
/// trip tool calls captured from non-Gemini models without crashing the
/// turn. Matches `google-shared.ts:51`.
const SKIP_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

impl GoogleOAuthProvider {
    pub(crate) const PROVIDER_KIND: &'static str = "google_oauth";

    pub(crate) fn inline_attachment_part(att: &LlmAttachment) -> Value {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&att.data);
        json!({
            "inlineData": {
                "mimeType": att.mime,
                "data": b64,
            }
        })
    }

    fn attachment_part_for_index(attachment_parts: &[Value], idx: usize) -> Value {
        attachment_parts
            .get(idx)
            .cloned()
            .unwrap_or_else(|| json!({ "text": "[Image attached]" }))
    }

    fn valid_same_origin_text_signature(
        req: &LlmRequest,
        meta: &ResponseTextMeta,
    ) -> Option<String> {
        if meta.origin_provider.as_deref() != Some(Self::PROVIDER_KIND)
            || meta.origin_model.as_deref() != Some(req.model.as_str())
        {
            return None;
        }
        let signature = meta.provider_payload.as_deref()?.trim();
        if signature.is_empty() {
            return None;
        }
        base64::engine::general_purpose::STANDARD
            .decode(signature)
            .ok()
            .filter(|bytes| !bytes.is_empty())?;
        Some(signature.to_string())
    }

    pub(crate) fn build_contents_with_attachment_parts(
        req: &LlmRequest,
        attachment_parts: &[Value],
    ) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        let is_gemini_3 = req.model.to_ascii_lowercase().contains("gemini-3");

        for msg in &req.messages {
            if matches!(msg.role, LlmRole::System) {
                // System content is hoisted into `systemInstruction` on the
                // Gemini request, not the `contents` list.
                continue;
            }
            let role = match msg.role {
                LlmRole::Assistant => "model",
                LlmRole::User | LlmRole::System => "user",
            };

            let mut parts: Vec<Value> = Vec::new();
            for block in msg.blocks.iter() {
                match block {
                    LlmContentBlock::Text {
                        text,
                        response_meta,
                        ..
                    } => {
                        if text.is_empty() {
                            continue;
                        }
                        let mut part = json!({ "text": text });
                        if matches!(msg.role, LlmRole::Assistant)
                            && let Some(signature) = response_meta
                                .as_ref()
                                .and_then(|meta| Self::valid_same_origin_text_signature(req, meta))
                        {
                            part["thoughtSignature"] = Value::String(signature);
                        }
                        parts.push(part);
                    }
                    LlmContentBlock::Image { attachment_idx } => {
                        if matches!(msg.role, LlmRole::User) {
                            parts.push(Self::attachment_part_for_index(
                                attachment_parts,
                                *attachment_idx,
                            ));
                        }
                    }
                    LlmContentBlock::ToolCall {
                        call_id,
                        tool_name,
                        input_json,
                        replay,
                        ..
                    } => {
                        let mut part = json!({
                            "functionCall": {
                                "id": call_id,
                                "name": tool_name,
                                "args": serde_json::from_str::<Value>(input_json)
                                    .unwrap_or_else(|_| json!({"_raw": input_json})),
                            }
                        });
                        // Gemini 3 rejects turns where a function_call
                        // from a thinking-enabled run is replayed without
                        // its original thoughtSignature. When we don't
                        // have the real signature (cross-model hop, older
                        // session), drop in the pi sentinel.
                        let effective = replay
                            .as_ref()
                            .and_then(|meta| meta.opaque.clone())
                            .or_else(|| is_gemini_3.then(|| SKIP_THOUGHT_SIGNATURE.to_string()));
                        if let Some(sig) = effective {
                            part["thoughtSignature"] = Value::String(sig);
                        }
                        parts.push(part);
                    }
                    LlmContentBlock::ToolResult {
                        call_id,
                        content,
                        tool_name,
                    } => {
                        parts.push(json!({
                            "functionResponse": {
                                "id": call_id,
                                "name": tool_name.clone().unwrap_or_else(|| "tool".to_string()),
                                "response": { "output": content },
                            }
                        }));
                    }
                    LlmContentBlock::Reasoning { text, replay, .. } => {
                        // Gemini replays reasoning as a `thought:true`
                        // text part carrying the thoughtSignature.
                        let sig = replay.as_ref().and_then(|meta| meta.signature.clone());
                        if sig.is_none() && text.trim().is_empty() {
                            continue;
                        }
                        let mut part = json!({
                            "text": if text.is_empty() { String::from(" ") } else { text.clone() },
                            "thought": true,
                        });
                        if let Some(s) = sig {
                            part["thoughtSignature"] = Value::String(s);
                        }
                        parts.push(part);
                    }
                }
            }

            if parts.is_empty() {
                continue;
            }

            // Merge with previous same-role turn so text + images + tool
            // calls land as a single `contents` entry (matches the old
            // behavior expected by Gemini clients).
            if let Some(prev) = out.last_mut()
                && prev.get("role").and_then(|r| r.as_str()) == Some(role)
                && prev.get("parts").is_some_and(|p| p.is_array())
            {
                prev["parts"].as_array_mut().unwrap().extend(parts);
            } else {
                out.push(json!({
                    "role": role,
                    "parts": parts,
                }));
            }
        }
        out
    }

    /// Claude models served through the Google/Vertex gateway take their tool
    /// schema under the `parameters` field (with JSON-Schema meta keys stripped)
    /// instead of the Gemini-native `parametersJsonSchema`. This is a live,
    /// always-on special case for `claude-*` on Vertex, not deprecated behavior.
    fn uses_claude_on_vertex_tool_parameters(model: &str) -> bool {
        model.starts_with("claude-")
    }

    /// Strip the JSON-Schema meta keys the Vertex `parameters` field rejects for
    /// `claude-*` models (`$schema`, `$defs`, `$id`, `definitions`), recursing
    /// through nested objects and arrays.
    fn sanitized_claude_on_vertex_schema(schema: &Value) -> Value {
        match schema {
            Value::Object(map) => {
                let mut out = serde_json::Map::new();
                for (key, value) in map {
                    if matches!(key.as_str(), "$schema" | "$defs" | "$id" | "definitions") {
                        continue;
                    }
                    out.insert(key.clone(), Self::sanitized_claude_on_vertex_schema(value));
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .map(Self::sanitized_claude_on_vertex_schema)
                    .collect::<Vec<_>>(),
            ),
            other => other.clone(),
        }
    }

    fn google_tool_choice(choice: &LlmToolChoice) -> &'static str {
        match choice {
            LlmToolChoice::Auto => "AUTO",
            LlmToolChoice::None => "NONE",
            LlmToolChoice::Required => "ANY",
        }
    }

    fn system_instruction(req: &LlmRequest) -> Option<Value> {
        let mut parts: Vec<String> = Vec::new();
        for msg in &req.messages {
            if !matches!(msg.role, LlmRole::System) {
                continue;
            }
            for block in msg.blocks.iter() {
                if let LlmContentBlock::Text { text, .. } = block
                    && !text.is_empty()
                {
                    parts.push(text.to_string());
                }
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(json!({
                "parts": [{ "text": parts.join("\n\n") }],
            }))
        }
    }

    pub(crate) fn build_request(
        provider: &GoogleOAuthProvider,
        req: &LlmRequest,
        contents: Vec<Value>,
        project_id: Option<&str>,
    ) -> Value {
        let thinking_config = req
            .model_variant
            .as_deref()
            .and_then(|variant| GoogleModelPolicy.thinking_config(&req.model, variant));
        let policy =
            resolve_generation_policy(&req.generation, &provider.options, 32_768, thinking_config);
        let mut request = json!({
            "model": req.model,
            "user_prompt_id": uuid::Uuid::new_v4().to_string(),
            "request": {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": policy.max_output_tokens,
                }
            }
        });
        if let Some(system_instruction) = Self::system_instruction(req) {
            request["request"]["systemInstruction"] = system_instruction;
        }
        request["request"]["sessionId"] = json!(req.session_id());
        if let Some(config) = policy.thinking {
            match config {
                GoogleThinkingConfig::Level { level } => {
                    let mut thinking_config = json!({
                        "thinkingLevel": level,
                    });
                    if policy.expose_thinking {
                        thinking_config["includeThoughts"] = json!(true);
                    }
                    request["request"]["generationConfig"]["thinkingConfig"] = thinking_config;
                }
                GoogleThinkingConfig::Budget { budget_tokens } => {
                    let mut thinking_config = json!({
                        "thinkingBudget": budget_tokens,
                    });
                    if policy.expose_thinking {
                        thinking_config["includeThoughts"] = json!(true);
                    }
                    request["request"]["generationConfig"]["thinkingConfig"] = thinking_config;
                }
            }
        }
        if !req.tools.is_empty() {
            let use_claude_on_vertex_parameters =
                Self::uses_claude_on_vertex_tool_parameters(&req.model);
            request["request"]["tools"] = json!([{
                "functionDeclarations": req
                    .tools
                    .iter()
                    .map(|tool| {
                        let mut declaration = json!({
                            "name": tool.name.clone(),
                            "description": tool.description.clone(),
                        });
                        if use_claude_on_vertex_parameters {
                            declaration["parameters"] =
                                Self::sanitized_claude_on_vertex_schema(tool.input_schema.canonical());
                        } else {
                            declaration["parametersJsonSchema"] =
                                tool.input_schema.canonical().clone();
                        }
                        declaration
                    })
                    .collect::<Vec<_>>()
            }]);
            request["request"]["toolConfig"] = json!({
                "functionCallingConfig": {
                    "mode": Self::google_tool_choice(&req.tool_choice),
                }
            });
        }
        if let Some(output_spec) = &req.output_spec {
            request["request"]["generationConfig"]["responseMimeType"] = json!("application/json");
            if let LlmOutputSpec::JsonSchema(schema) = output_spec {
                request["request"]["generationConfig"]["responseSchema"] =
                    schema.schema.canonical().clone();
            }
        }
        if let Some(project) = project_id.filter(|p| !p.trim().is_empty()) {
            request["project"] = json!(project);
        }
        request
    }
}
