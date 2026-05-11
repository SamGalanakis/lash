use crate::support::*;

#[derive(Clone, Debug, Default)]
pub(crate) struct ChatStreamingToolCall {
    pub(crate) call_id: String,
    pub(crate) tool_name: String,
    pub(crate) input_json: String,
    pub(crate) signature: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ChatStreamState {
    pub(crate) full_text: String,
    pub(crate) deltas: Vec<String>,
    pub(crate) reasoning_text: String,
    pub(crate) reasoning_deltas: Vec<String>,
    pub(crate) usage: LlmUsage,
    pub(crate) provider_usage: Option<Value>,
    pub(crate) tool_calls: HashMap<usize, ChatStreamingToolCall>,
    pub(crate) final_response: Option<Value>,
}

impl ChatStreamState {
    pub(crate) fn push_text_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        self.full_text.push_str(piece);
        self.deltas.push(piece.to_string());
    }

    pub(crate) fn push_reasoning_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        self.reasoning_text.push_str(piece);
        self.reasoning_deltas.push(piece.to_string());
    }

    pub(crate) fn update_tool_call_delta(&mut self, value: &Value) {
        let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        let tool_call = self.tool_calls.entry(index).or_default();
        if let Some(id) = value.get("id").and_then(Value::as_str)
            && !id.is_empty()
        {
            tool_call.call_id = id.to_string();
        }
        if let Some(function) = value.get("function") {
            if let Some(name) = function.get("name").and_then(Value::as_str)
                && !name.is_empty()
            {
                tool_call.tool_name = name.to_string();
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str)
                && !arguments.is_empty()
            {
                tool_call.input_json.push_str(arguments);
            }
        }
    }

    pub(crate) fn apply_reasoning_details(&mut self, details: &Value) {
        let Some(details) = details.as_array() else {
            return;
        };
        for detail in details {
            if detail.get("type").and_then(Value::as_str) != Some("reasoning.encrypted") {
                continue;
            }
            let Some(id) = detail.get("id").and_then(Value::as_str) else {
                continue;
            };
            if detail.get("data").and_then(Value::as_str).is_none() {
                continue;
            }
            for tool_call in self.tool_calls.values_mut() {
                if tool_call.call_id == id {
                    tool_call.signature = Some(detail.to_string());
                    break;
                }
            }
        }
    }

    pub(crate) fn take_reasoning_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reasoning_deltas)
    }

    pub(crate) fn parts(&self) -> Vec<LlmOutputPart> {
        let mut parts = Vec::new();
        if !self.reasoning_text.trim().is_empty() {
            parts.push(LlmOutputPart::Reasoning {
                text: self.reasoning_text.clone(),
                signature: None,
                redacted: false,
                item_id: None,
                encrypted_content: None,
                summary: Vec::new(),
            });
        }
        if !self.full_text.is_empty() {
            parts.push(LlmOutputPart::Text {
                text: self.full_text.clone(),
                response_meta: None,
            });
        }
        let mut tool_calls = self.tool_calls.iter().collect::<Vec<_>>();
        tool_calls.sort_by_key(|(index, _)| **index);
        for (_, tool_call) in tool_calls {
            if tool_call.tool_name.is_empty() {
                continue;
            }
            parts.push(LlmOutputPart::ToolCall {
                call_id: if tool_call.call_id.is_empty() {
                    uuid::Uuid::new_v4().to_string()
                } else {
                    tool_call.call_id.clone()
                },
                tool_name: tool_call.tool_name.clone(),
                input_json: if tool_call.input_json.is_empty() {
                    "{}".to_string()
                } else {
                    tool_call.input_json.clone()
                },
                item_id: None,
                signature: tool_call.signature.clone(),
            });
        }
        if parts.is_empty()
            && let Some(final_response) = &self.final_response
        {
            parts = OpenAiGenericProvider::chat_response_parts_from_value(final_response);
        }
        parts
    }
}
