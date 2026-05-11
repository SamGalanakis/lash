use crate::support::*;

#[derive(Clone, Debug, Default)]
pub(crate) struct ResponsesStreamingToolCall {
    pub(crate) call_id: String,
    pub(crate) tool_name: String,
    pub(crate) input_json: String,
    pub(crate) item_id: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ResponsesStreamState {
    pub(crate) full_text: String,
    pub(crate) deltas: Vec<String>,
    pub(crate) parts: Vec<LlmOutputPart>,
    pub(crate) usage: LlmUsage,
    pub(crate) provider_usage: Option<Value>,
    pub(crate) current_text_part: Option<usize>,
    pub(crate) current_reasoning_part: Option<usize>,
    pub(crate) reasoning_deltas: Vec<String>,
    pub(crate) tool_calls: HashMap<String, ResponsesStreamingToolCall>,
    pub(crate) final_response: Option<Value>,
}

impl ResponsesStreamState {
    pub(crate) fn begin_message(&mut self, item: Option<&Value>) {
        let meta = item.map(OpenAiCompatibleProvider::response_text_meta_from_message_item);
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Text {
            text: String::new(),
            response_meta: meta,
        });
        self.current_text_part = Some(index);
    }

    pub(crate) fn finish_message(&mut self, item: Option<&Value>) {
        if let Some(item) = item {
            let text = OpenAiCompatibleProvider::message_text_from_item(item);
            let meta = OpenAiCompatibleProvider::response_text_meta_from_message_item(item);
            let index = self.ensure_text_part_index();
            if let Some(LlmOutputPart::Text {
                text: existing,
                response_meta,
            }) = self.parts.get_mut(index)
            {
                if !text.is_empty() {
                    *existing = text;
                }
                *response_meta = Some(meta);
            }
            self.recompute_full_text();
        }
        self.current_text_part = None;
    }

    pub(crate) fn push_text_delta(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        let index = self.ensure_text_part_index();
        if let Some(LlmOutputPart::Text { text, .. }) = self.parts.get_mut(index) {
            text.push_str(piece);
        }
        self.deltas.push(piece.to_string());
        self.recompute_full_text();
    }

    pub(crate) fn ensure_text_part_index(&mut self) -> usize {
        if let Some(index) = self.current_text_part {
            return index;
        }
        if let Some(index) = self
            .parts
            .iter()
            .rposition(|part| matches!(part, LlmOutputPart::Text { .. }))
        {
            return index;
        }
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Text {
            text: String::new(),
            response_meta: None,
        });
        index
    }

    pub(crate) fn recompute_full_text(&mut self) {
        self.full_text.clear();
        for part in &self.parts {
            if let LlmOutputPart::Text { text, .. } = part {
                self.full_text.push_str(text);
            }
        }
    }

    pub(crate) fn begin_reasoning_part(&mut self) {
        let index = self.parts.len();
        self.parts.push(LlmOutputPart::Reasoning {
            text: String::new(),
            signature: None,
            redacted: false,
            item_id: None,
            encrypted_content: None,
            summary: Vec::new(),
        });
        self.current_reasoning_part = Some(index);
    }

    pub(crate) fn push_reasoning_delta(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let index = match self.current_reasoning_part {
            Some(index) => index,
            None => {
                self.begin_reasoning_part();
                self.current_reasoning_part
                    .expect("reasoning part just pushed")
            }
        };
        if let Some(LlmOutputPart::Reasoning { text, .. }) = self.parts.get_mut(index) {
            text.push_str(delta);
        }
        self.reasoning_deltas.push(delta.to_string());
    }

    pub(crate) fn finish_reasoning_part(&mut self) {
        if let Some(index) = self.current_reasoning_part.take()
            && let Some(LlmOutputPart::Reasoning { text, .. }) = self.parts.get_mut(index)
        {
            let trimmed = text.trim_end();
            if trimmed.len() != text.len() {
                *text = trimmed.to_string();
            }
        }
    }

    pub(crate) fn finalize_reasoning_item(&mut self, item: &Value) {
        let Some((_, part)) = self
            .parts
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(_, part)| matches!(part, LlmOutputPart::Reasoning { .. }))
        else {
            return;
        };
        let LlmOutputPart::Reasoning {
            item_id,
            encrypted_content,
            summary,
            ..
        } = part
        else {
            return;
        };
        if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
            *item_id = Some(id.to_string());
        }
        if let Some(blob) = item.get("encrypted_content").and_then(|v| v.as_str()) {
            *encrypted_content = Some(blob.to_string());
        }
        if let Some(arr) = item.get("summary").and_then(|v| v.as_array()) {
            let texts = arr
                .iter()
                .filter_map(|entry| entry.get("text").and_then(|v| v.as_str()).map(String::from))
                .collect::<Vec<_>>();
            if !texts.is_empty() {
                *summary = texts;
            }
        }
    }

    pub(crate) fn update_tool_call_from_item(&mut self, item: &Value) -> Option<String> {
        let item_id = item.get("id").and_then(|v| v.as_str())?.to_string();
        let tool_call = self.tool_calls.entry(item_id.clone()).or_default();
        tool_call.item_id.clone_from(&item_id);
        if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
            tool_call.call_id = call_id.to_string();
        }
        if let Some(name) = item.get("name").and_then(|v| v.as_str()) {
            tool_call.tool_name = name.to_string();
        }
        if let Some(args) = item.get("arguments").and_then(|v| v.as_str())
            && !args.is_empty()
        {
            tool_call.input_json = args.to_string();
        }
        Some(item_id)
    }

    pub(crate) fn push_tool_call_delta(&mut self, item_id: &str, delta: &str) {
        if item_id.is_empty() || delta.is_empty() {
            return;
        }
        self.tool_calls
            .entry(item_id.to_string())
            .or_default()
            .input_json
            .push_str(delta);
    }

    pub(crate) fn set_tool_call_arguments(&mut self, item_id: &str, arguments: &str) {
        if item_id.is_empty() {
            return;
        }
        self.tool_calls
            .entry(item_id.to_string())
            .or_default()
            .input_json = arguments.to_string();
    }

    pub(crate) fn finish_tool_call(&mut self, item: &Value) -> Option<LlmOutputPart> {
        let item_id = self.update_tool_call_from_item(item)?;
        let mut tool_call = self.tool_calls.remove(&item_id).unwrap_or_default();
        if tool_call.call_id.is_empty() {
            tool_call.call_id = uuid::Uuid::new_v4().to_string();
        }
        if tool_call.tool_name.is_empty() {
            return None;
        }
        if tool_call.input_json.is_empty() {
            tool_call.input_json = "{}".to_string();
        }
        let part = LlmOutputPart::ToolCall {
            call_id: tool_call.call_id,
            tool_name: tool_call.tool_name,
            input_json: tool_call.input_json,
            item_id: (!tool_call.item_id.is_empty()).then_some(tool_call.item_id),
            signature: None,
        };
        self.parts.push(part.clone());
        Some(part)
    }

    pub(crate) fn take_reasoning_deltas(&mut self) -> Vec<String> {
        std::mem::take(&mut self.reasoning_deltas)
    }

    pub(crate) fn response_parts(&self) -> Vec<LlmOutputPart> {
        let mut parts = self
            .parts
            .iter()
            .filter_map(|part| match part {
                LlmOutputPart::Text { text, .. } if text.is_empty() => None,
                LlmOutputPart::Reasoning { text, .. } if text.trim().is_empty() => None,
                other => Some(other.clone()),
            })
            .collect::<Vec<_>>();
        if parts.is_empty()
            && let Some(final_response) = &self.final_response
        {
            parts = OpenAiCompatibleProvider::response_parts_from_value(final_response);
        }
        parts
    }
}
