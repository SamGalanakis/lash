use crate::llm::types::{LlmAttachment, LlmMessage, LlmRole};
use crate::plugin::UserInputProvenance;
use base64::Engine;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

// ─── Structured message types for context-aware pruning ───

/// A structured message with typed parts for context management.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub parts: Vec<Part>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_input: Option<UserInputProvenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<MessageOrigin>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessageOrigin {
    Plugin {
        plugin_id: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        transient: bool,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Part {
    /// e.g. "m3.p0"
    pub id: String,
    pub kind: PartKind,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachment: Option<PartAttachment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Provider-specific item-id for a `ToolCall` part (e.g. Codex
    /// Responses API `fc_...`). Preserved across turns so the Codex
    /// adapter can re-emit it on the next request body. `None` for
    /// providers that don't surface a distinct item-id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_item_id: Option<String>,
    pub prune_state: PruneState,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PartKind {
    Text,
    Image,
    Code,
    Output,
    Error,
    Prose,
    ToolCall,
    ToolResult,
    /// Model "thinking" / reasoning summary. Stored on the session for
    /// display and snapshot fidelity, but deliberately excluded from the
    /// rendered prompt so it is never re-sent to the model as assistant
    /// input. Re-feeding uses the separate encrypted-reasoning path.
    Reasoning,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PartAttachment {
    pub mime: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PruneState {
    Intact,
    Cleared,
    Deleted {
        breadcrumb: String,
        archive_hash: String,
    },
    Summarized {
        summary: String,
        archive_hash: String,
    },
}

impl Part {
    pub fn prompt_char_count(&self) -> usize {
        if matches!(self.kind, PartKind::Image) {
            return self
                .attachment
                .as_ref()
                .map(|attachment| attachment.url.len())
                .unwrap_or_else(|| self.render().len());
        }
        self.render().len()
    }

    pub(crate) fn render(&self) -> String {
        if matches!(self.kind, PartKind::Image) {
            return if self.attachment.is_some() || self.content.trim().is_empty() {
                "[Image attached]".to_string()
            } else {
                self.content.clone()
            };
        }
        match &self.prune_state {
            PruneState::Intact => self.content.clone(),
            PruneState::Cleared => "[Old tool result content cleared]".to_string(),
            PruneState::Deleted {
                breadcrumb,
                archive_hash,
            } => format!("[pruned:{} — {}]", archive_hash, breadcrumb),
            PruneState::Summarized {
                summary,
                archive_hash,
            } => format!("[SUMMARY of original {}]\n{}", archive_hash, summary),
        }
    }
}

impl Message {
    /// Total character count of all parts (rendered).
    pub fn char_count(&self) -> usize {
        self.parts.iter().map(Part::prompt_char_count).sum()
    }

    pub fn user_input_provenance(&self) -> Option<&UserInputProvenance> {
        self.user_input.as_ref()
    }

    pub fn display_user_text(&self) -> Option<&str> {
        self.user_input_provenance()
            .map(|user_input| user_input.display_text.as_str())
    }

    pub fn effective_user_text(&self) -> Option<&str> {
        self.user_input_provenance()
            .map(|user_input| user_input.effective_text.as_str())
    }

    pub fn is_transient(&self) -> bool {
        matches!(
            self.origin,
            Some(MessageOrigin::Plugin {
                transient: true,
                ..
            })
        )
    }
}

pub fn data_url_for_bytes(mime: &str, bytes: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{b64}")
}

fn render_part_for_chat(role: MessageRole, part: &Part) -> String {
    let rendered = part.render();
    match role {
        MessageRole::System => match part.kind {
            PartKind::Code => rendered,
            PartKind::Output => format!("<output>\n{}\n</output>", rendered),
            PartKind::Error => format!("<error>\n{}\n</error>", rendered),
            PartKind::Text
            | PartKind::Image
            | PartKind::Prose
            | PartKind::ToolCall
            | PartKind::ToolResult
            | PartKind::Reasoning => rendered,
        },
        MessageRole::Assistant => match part.kind {
            PartKind::Code => rendered,
            PartKind::ToolCall => render_assistant_tool_call(part, &rendered),
            PartKind::Prose | PartKind::Text | PartKind::Image | PartKind::ToolResult => rendered,
            PartKind::Reasoning => rendered,
            _ => rendered,
        },
        MessageRole::User => rendered,
    }
}

fn render_assistant_tool_call(part: &Part, rendered: &str) -> String {
    let tool_name = part.tool_name.as_deref().unwrap_or("tool");
    let trimmed = rendered.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        format!("{tool_name}()")
    } else {
        format!("{tool_name}({trimmed})")
    }
}

fn attachment_from_part(part: &Part) -> Option<LlmAttachment> {
    if !matches!(part.kind, PartKind::Image) {
        return None;
    }
    let attachment = part.attachment.as_ref()?;
    let encoded = attachment
        .url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,"))
        .map(|(_, encoded)| encoded)?;
    let data = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    Some(LlmAttachment {
        mime: attachment.mime.clone(),
        data,
    })
}

fn render_message_for_transcript(msg: &Message, attachments: &mut Vec<LlmAttachment>) -> String {
    if let Some(display_text) = msg.display_user_text()
        && matches!(msg.role, MessageRole::User)
    {
        return display_text.to_string();
    }
    let mut out = Vec::new();
    for part in &msg.parts {
        // Reasoning summaries are not part of the replayable transcript —
        // they are local display annotations only.
        if matches!(part.kind, PartKind::Reasoning) {
            continue;
        }
        if let Some(attachment) = attachment_from_part(part) {
            attachments.push(attachment);
            out.push("[Image attached]".to_string());
            continue;
        }
        let rendered = render_part_for_chat(msg.role, part);
        if !rendered.trim().is_empty() {
            out.push(rendered);
        }
    }
    out.join("\n\n")
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub messages: Vec<LlmMessage>,
    pub attachments: Vec<LlmAttachment>,
}

#[derive(Debug)]
pub struct MessageSequence {
    base: Arc<Vec<Message>>,
    delta: Vec<Message>,
    owned: Option<Vec<Message>>,
    materialized: OnceLock<Arc<Vec<Message>>>,
    base_rendered_prompt: Option<Arc<RenderedPrompt>>,
}

impl Clone for MessageSequence {
    fn clone(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            delta: self.delta.clone(),
            owned: self.owned.clone(),
            materialized: OnceLock::new(),
            base_rendered_prompt: self.base_rendered_prompt.clone(),
        }
    }
}

impl Default for MessageSequence {
    fn default() -> Self {
        Self::from_owned(Vec::new())
    }
}

impl From<Vec<Message>> for MessageSequence {
    fn from(messages: Vec<Message>) -> Self {
        Self::from_owned(messages)
    }
}

impl std::ops::Deref for MessageSequence {
    type Target = [Message];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl MessageSequence {
    pub fn from_owned(messages: Vec<Message>) -> Self {
        Self {
            base: Arc::new(Vec::new()),
            delta: Vec::new(),
            owned: Some(messages),
            materialized: OnceLock::new(),
            base_rendered_prompt: None,
        }
    }

    pub fn from_base(base: Arc<Vec<Message>>) -> Self {
        Self {
            base,
            delta: Vec::new(),
            owned: None,
            materialized: OnceLock::new(),
            base_rendered_prompt: None,
        }
    }

    pub fn from_base_and_delta(base: Arc<Vec<Message>>, delta: Vec<Message>) -> Self {
        Self {
            base,
            delta,
            owned: None,
            materialized: OnceLock::new(),
            base_rendered_prompt: None,
        }
    }

    pub fn with_base_rendered_prompt(mut self, rendered: Option<Arc<RenderedPrompt>>) -> Self {
        self.base_rendered_prompt = rendered;
        self
    }

    pub fn len(&self) -> usize {
        match &self.owned {
            Some(owned) => owned.len(),
            None => self.base.len() + self.delta.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[Message] {
        if let Some(owned) = &self.owned {
            return owned.as_slice();
        }
        if self.delta.is_empty() {
            return self.base.as_slice();
        }
        self.materialized
            .get_or_init(|| {
                let mut combined = Vec::with_capacity(self.base.len() + self.delta.len());
                combined.extend(self.base.iter().cloned());
                combined.extend(self.delta.iter().cloned());
                Arc::new(combined)
            })
            .as_slice()
    }

    pub fn shared(&self) -> Arc<Vec<Message>> {
        if let Some(owned) = &self.owned {
            return Arc::clone(self.materialized.get_or_init(|| Arc::new(owned.clone())));
        }
        if self.delta.is_empty() {
            return Arc::clone(&self.base);
        }
        Arc::clone(self.materialized.get_or_init(|| {
            let mut combined = Vec::with_capacity(self.base.len() + self.delta.len());
            combined.extend(self.base.iter().cloned());
            combined.extend(self.delta.iter().cloned());
            Arc::new(combined)
        }))
    }

    pub fn make_mut(&mut self) -> &mut Vec<Message> {
        if self.owned.is_none() {
            let owned = if self.delta.is_empty() {
                Arc::unwrap_or_clone(Arc::clone(&self.base))
            } else if let Some(materialized) = self.materialized.get() {
                Arc::unwrap_or_clone(Arc::clone(materialized))
            } else {
                let mut combined = Vec::with_capacity(self.base.len() + self.delta.len());
                combined.extend(self.base.iter().cloned());
                combined.extend(self.delta.iter().cloned());
                combined
            };
            self.owned = Some(owned);
            self.base = Arc::new(Vec::new());
            self.delta.clear();
        }
        self.materialized = OnceLock::new();
        self.base_rendered_prompt = None;
        self.owned.as_mut().expect("message sequence owned state")
    }

    pub fn push(&mut self, message: Message) {
        self.make_mut().push(message);
    }

    pub fn extend(&mut self, messages: Vec<Message>) {
        self.make_mut().extend(messages);
    }

    pub fn replace(&mut self, messages: Vec<Message>) {
        self.base = Arc::new(Vec::new());
        self.delta.clear();
        self.owned = Some(messages);
        self.materialized = OnceLock::new();
        self.base_rendered_prompt = None;
    }

    pub fn into_vec(self) -> Vec<Message> {
        if let Some(owned) = self.owned {
            return owned;
        }
        if self.delta.is_empty() {
            return Arc::unwrap_or_clone(self.base);
        }
        if let Some(materialized) = self.materialized.into_inner() {
            return Arc::unwrap_or_clone(materialized);
        }
        let mut combined = Vec::with_capacity(self.base.len() + self.delta.len());
        combined.extend(self.base.iter().cloned());
        combined.extend(self.delta);
        combined
    }

    pub fn render_prompt(&self) -> RenderedPrompt {
        if let Some(owned) = &self.owned {
            return render_prompt(owned.as_slice());
        }
        if self.base.is_empty() {
            return render_prompt(self.delta.as_slice());
        }
        let mut rendered = self
            .base_rendered_prompt
            .as_ref()
            .map(|prompt| prompt.as_ref().clone())
            .unwrap_or_else(|| render_prompt(self.base.as_slice()));
        if !self.delta.is_empty() {
            append_rendered_prompt(&mut rendered, self.delta.as_slice());
        }
        rendered
    }
}

#[derive(Clone, Debug, Default)]
struct TranscriptTurn {
    user: Vec<String>,
    assistant: Vec<String>,
}

pub fn render_prompt(msgs: &[Message]) -> RenderedPrompt {
    let mut rendered = RenderedPrompt::default();
    append_rendered_prompt(&mut rendered, msgs);
    rendered
}

pub fn messages_are_live_resume_safe(messages: &[Message]) -> bool {
    let mut seen_tool_calls = HashSet::new();
    let mut completed_tool_calls = HashSet::new();

    for message in messages {
        for part in &message.parts {
            match part.kind {
                PartKind::ToolCall => {
                    if !matches!(message.role, MessageRole::Assistant) {
                        return false;
                    }
                    let Some(call_id) = part
                        .tool_call_id
                        .as_deref()
                        .map(str::trim)
                        .filter(|call_id| !call_id.is_empty())
                    else {
                        return false;
                    };
                    if !seen_tool_calls.insert(call_id.to_string()) {
                        return false;
                    }
                }
                PartKind::ToolResult => {
                    if !matches!(message.role, MessageRole::User) {
                        return false;
                    }
                    let Some(call_id) = part
                        .tool_call_id
                        .as_deref()
                        .map(str::trim)
                        .filter(|call_id| !call_id.is_empty())
                    else {
                        return false;
                    };
                    if !seen_tool_calls.contains(call_id) {
                        return false;
                    }
                    if !completed_tool_calls.insert(call_id.to_string()) {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }

    seen_tool_calls.len() == completed_tool_calls.len()
}

pub fn render_transcript_prompt(msgs: &[Message]) -> RenderedPrompt {
    let mut attachments = Vec::new();
    let mut turns = Vec::new();
    let mut current = TranscriptTurn::default();
    let mut has_current = false;

    for msg in msgs {
        let text = render_message_for_transcript(msg, &mut attachments);
        let has_text = !text.trim().is_empty();
        match msg.role {
            MessageRole::User => {
                if has_current && (!current.user.is_empty() || !current.assistant.is_empty()) {
                    turns.push(current);
                    current = TranscriptTurn::default();
                }
                if has_text {
                    current.user.push(text);
                }
                has_current = true;
            }
            MessageRole::Assistant | MessageRole::System => {
                if !has_current {
                    has_current = true;
                }
                if has_text {
                    current.assistant.push(text);
                }
            }
        }
    }

    if has_current && (!current.user.is_empty() || !current.assistant.is_empty()) {
        turns.push(current);
    }

    let mut text = String::new();
    text.push_str(
        "History:\nThis is a chronological transcript. `Assistant` refers to Lash, and you are continuing the same session.\n\n",
    );
    for (idx, turn) in turns.iter().enumerate() {
        text.push_str(&format!("=== Turn {} ===\n", idx + 1));
        text.push_str("User:\n");
        if turn.user.is_empty() {
            text.push_str("[No user content recorded]\n");
        } else {
            text.push_str(&turn.user.join("\n\n"));
            text.push('\n');
        }
        text.push('\n');
        text.push_str("Assistant (Lash, continuing this transcript):\n");
        let is_current_pending_turn = idx + 1 == turns.len() && turn.assistant.is_empty();
        if turn.assistant.is_empty() && !is_current_pending_turn {
            text.push_str("[No assistant content recorded]\n");
        } else if !turn.assistant.is_empty() {
            text.push_str(&turn.assistant.join("\n\n"));
            text.push('\n');
        }
        text.push('\n');
    }
    text.push_str(
        "Continue from the latest turn as Lash.\nIf the task is complete, provide the final answer.\nOtherwise produce the next valid step for this runtime.",
    );

    RenderedPrompt {
        messages: vec![LlmMessage {
            role: LlmRole::User,
            content: text,
            kind: "text".to_string(),
            image_idx: -1,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
        }],
        attachments,
    }
}

pub fn append_rendered_prompt(rendered: &mut RenderedPrompt, msgs: &[Message]) {
    append_structured_prompt(rendered, msgs)
}

#[cfg(test)]
fn render_structured_prompt(msgs: &[Message]) -> RenderedPrompt {
    let mut rendered = RenderedPrompt::default();
    append_structured_prompt(&mut rendered, msgs);
    rendered
}

fn append_structured_prompt(rendered: &mut RenderedPrompt, msgs: &[Message]) {
    let mut attachment_count = 0usize;
    let mut message_count = 0usize;
    for msg in msgs {
        for part in &msg.parts {
            message_count += 1;
            if matches!(msg.role, MessageRole::User)
                && matches!(part.kind, PartKind::Image)
                && part.attachment.is_some()
            {
                attachment_count += 1;
            }
        }
    }
    rendered.attachments.reserve(attachment_count);
    rendered.messages.reserve(message_count);

    for msg in msgs {
        for part in &msg.parts {
            match part.kind {
                // Reasoning summaries are display-only. Never feed them
                // back to the model as assistant input — the only legal
                // re-feeding path is the separate encrypted-reasoning
                // channel, which does not flow through this prompt
                // builder.
                PartKind::Reasoning => continue,
                PartKind::ToolCall => {
                    rendered.messages.push(LlmMessage {
                        role: LlmRole::Assistant,
                        content: part.content.clone(),
                        kind: "tool_call".to_string(),
                        image_idx: -1,
                        tool_call_id: part.tool_call_id.clone(),
                        tool_name: part.tool_name.clone(),
                        tool_item_id: part.tool_item_id.clone(),
                    });
                }
                PartKind::ToolResult => {
                    let text = part.render();
                    rendered.messages.push(LlmMessage {
                        role: llm_role_for_message(msg.role),
                        content: text,
                        kind: "tool_result".to_string(),
                        image_idx: -1,
                        tool_call_id: part.tool_call_id.clone(),
                        tool_name: part.tool_name.clone(),
                        tool_item_id: None,
                    });
                }
                _ => {
                    if let Some(attachment) = attachment_from_part(part)
                        && matches!(msg.role, MessageRole::User)
                    {
                        let image_idx = rendered.attachments.len();
                        rendered.attachments.push(attachment);
                        rendered.messages.push(LlmMessage {
                            role: LlmRole::User,
                            content: String::new(),
                            kind: "image".to_string(),
                            image_idx: image_idx as i64,
                            tool_call_id: None,
                            tool_name: None,
                            tool_item_id: None,
                        });
                        continue;
                    }

                    let mut text = render_part_for_chat(msg.role, part);
                    if text.trim().is_empty() {
                        continue;
                    }

                    if matches!(msg.role, MessageRole::System) {
                        text = format!("Runtime note:\n{text}");
                    }

                    rendered.messages.push(LlmMessage {
                        role: llm_role_for_message(msg.role),
                        content: text,
                        kind: "text".to_string(),
                        image_idx: -1,
                        tool_call_id: None,
                        tool_name: None,
                        tool_item_id: None,
                    });
                }
            }
        }
    }
}

fn llm_role_for_message(role: MessageRole) -> LlmRole {
    match role {
        MessageRole::User => LlmRole::User,
        MessageRole::Assistant => LlmRole::Assistant,
        MessageRole::System => LlmRole::System,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(kind: PartKind, content: &str) -> Part {
        Part {
            id: "p0".to_string(),
            kind,
            content: content.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            prune_state: PruneState::Intact,
        }
    }

    fn image_part(bytes: &[u8]) -> Part {
        Part {
            id: "p0".to_string(),
            kind: PartKind::Image,
            content: String::new(),
            attachment: Some(PartAttachment {
                mime: "image/png".to_string(),
                url: data_url_for_bytes("image/png", bytes),
                filename: None,
            }),
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            prune_state: PruneState::Intact,
        }
    }

    #[test]
    fn render_transcript_prompt_orders_turns_oldest_first() {
        let msgs = vec![
            Message {
                id: "m0".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "first")],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::Assistant,
                parts: vec![part(PartKind::Prose, "reply one")],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m2".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "second")],
                user_input: None,
                origin: None,
            },
        ];

        let rendered = render_transcript_prompt(&msgs);
        let text = &rendered.messages[0].content;

        assert!(text.contains("=== Turn 1 ===\nUser:\nfirst"));
        assert!(text.contains("Assistant (Lash, continuing this transcript):\nreply one"));
        assert!(text.contains("=== Turn 2 ===\nUser:\nsecond"));
    }

    #[test]
    fn display_user_text_prefers_user_input_provenance() {
        let message = Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![part(
                PartKind::Text,
                "Use /wholehog\n\n<skill>\n<name>wholehog</name>\nbody\n</skill>",
            )],
            user_input: Some(UserInputProvenance {
                display_text: "Use /wholehog".to_string(),
                effective_text: "Use /wholehog\n\n<skill>\n<name>wholehog</name>\nbody\n</skill>"
                    .to_string(),
                transforms: vec![crate::plugin::UserInputTransform::SkillBlockAppend {
                    skill_name: "wholehog".to_string(),
                    skill_path: "/tmp/wholehog/SKILL.md".to_string(),
                }],
            }),
            origin: None,
        };

        assert_eq!(message.display_user_text(), Some("Use /wholehog"));
        assert!(
            message
                .effective_user_text()
                .is_some_and(|text| text.contains("<skill>"))
        );
        assert!(message.user_input_provenance().is_some());
    }

    #[test]
    fn render_prompt_repl_preserves_message_boundaries() {
        let msgs = vec![
            Message {
                id: "m1".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "first")],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m2".to_string(),
                role: MessageRole::Assistant,
                parts: vec![
                    part(PartKind::Prose, "reply one"),
                    part(PartKind::Code, "x = 1"),
                ],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m3".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "second")],
                user_input: None,
                origin: None,
            },
        ];

        let rendered = render_prompt(&msgs);
        assert_eq!(rendered.messages.len(), 4);
        assert_eq!(rendered.messages[0].content, "first");
        assert!(rendered.messages[1].content.contains("reply one"));
        assert_eq!(rendered.messages[2].content, "x = 1");
        assert_eq!(rendered.messages[2].kind, "text");
        assert_eq!(rendered.messages[3].content, "second");
    }

    #[test]
    fn render_structured_prompt_preserves_tool_protocol_and_user_images() {
        let msgs = vec![
            Message {
                id: "m0".to_string(),
                role: MessageRole::System,
                parts: vec![part(PartKind::Text, "note")],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "show this"), image_part(&[1, 2, 3])],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m2".to_string(),
                role: MessageRole::Assistant,
                parts: vec![Part {
                    id: "m2.p0".to_string(),
                    kind: PartKind::ToolCall,
                    content: r#"{"path":"README.md"}"#.to_string(),
                    attachment: None,
                    tool_call_id: Some("tc1".to_string()),
                    tool_name: Some("read_file".to_string()),
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m3".to_string(),
                role: MessageRole::User,
                parts: vec![Part {
                    id: "m3.p0".to_string(),
                    kind: PartKind::ToolResult,
                    content: "ok".to_string(),
                    attachment: None,
                    tool_call_id: Some("tc1".to_string()),
                    tool_name: Some("read_file".to_string()),
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
        ];

        let rendered = render_structured_prompt(&msgs);
        assert_eq!(rendered.messages.len(), 5);
        assert_eq!(rendered.messages[0].role, LlmRole::System);
        assert_eq!(rendered.messages[0].content, "Runtime note:\nnote");
        assert_eq!(rendered.messages[1].kind, "text");
        assert_eq!(rendered.messages[2].kind, "image");
        assert_eq!(rendered.messages[2].image_idx, 0);
        assert_eq!(rendered.attachments.len(), 1);
        assert_eq!(rendered.messages[3].kind, "tool_call");
        assert_eq!(rendered.messages[4].kind, "tool_result");
    }

    #[test]
    fn render_structured_prompt_preserves_empty_tool_results() {
        let msgs = vec![
            Message {
                id: "m0".to_string(),
                role: MessageRole::Assistant,
                parts: vec![Part {
                    id: "m0.p0".to_string(),
                    kind: PartKind::ToolCall,
                    content: r#"{"question":"Pick one"}"#.to_string(),
                    attachment: None,
                    tool_call_id: Some("ask_1".to_string()),
                    tool_name: Some("ask".to_string()),
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::User,
                parts: vec![Part {
                    id: "m1.p0".to_string(),
                    kind: PartKind::ToolResult,
                    content: String::new(),
                    attachment: None,
                    tool_call_id: Some("ask_1".to_string()),
                    tool_name: Some("ask".to_string()),
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
        ];

        let rendered = render_structured_prompt(&msgs);
        assert_eq!(rendered.messages.len(), 2);
        assert_eq!(rendered.messages[0].kind, "tool_call");
        assert_eq!(rendered.messages[1].kind, "tool_result");
        assert_eq!(rendered.messages[1].tool_call_id.as_deref(), Some("ask_1"));
        assert!(rendered.messages[1].content.is_empty());
    }

    #[test]
    fn render_transcript_prompt_collects_images() {
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![image_part(&[9, 8, 7])],
            user_input: None,
            origin: None,
        }];

        let rendered = render_transcript_prompt(&msgs);
        let text = &rendered.messages[0].content;
        assert!(text.contains("[Image attached]"));
        assert_eq!(rendered.attachments.len(), 1);
    }

    #[test]
    fn render_transcript_prompt_omits_missing_assistant_placeholder_for_current_turn() {
        let msgs = vec![
            Message {
                id: "m0".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "first")],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::Assistant,
                parts: vec![part(PartKind::Prose, "reply one")],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m2".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "second")],
                user_input: None,
                origin: None,
            },
        ];

        let rendered = render_transcript_prompt(&msgs);
        let text = &rendered.messages[0].content;

        assert!(text.contains("=== Turn 2 ===\nUser:\nsecond"));
        assert!(!text.contains("=== Turn 2 ===\nUser:\nsecond\n\nAssistant (Lash, continuing this transcript):\n[No assistant content recorded]"));
    }

    #[test]
    fn render_transcript_prompt_preserves_tool_name_for_assistant_tool_calls() {
        let msgs = vec![
            Message {
                id: "m0".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "what time is it")],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::Assistant,
                parts: vec![Part {
                    id: "m1.p0".to_string(),
                    kind: PartKind::ToolCall,
                    content: r#"{"cmd":"date"}"#.to_string(),
                    attachment: None,
                    tool_call_id: Some("tc1".to_string()),
                    tool_name: Some("exec_command".to_string()),
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
        ];

        let rendered = render_transcript_prompt(&msgs);
        let text = &rendered.messages[0].content;

        assert!(text.contains(r#"exec_command({"cmd":"date"})"#));
    }

    #[test]
    fn render_transcript_prompt_omits_runtime_notes_section() {
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![part(PartKind::Text, "hi")],
            user_input: None,
            origin: None,
        }];

        let rendered = render_transcript_prompt(&msgs);
        let text = &rendered.messages[0].content;
        assert!(!text.contains("Runtime Notes:"));
    }

    #[test]
    fn live_resume_safety_accepts_completed_tool_history() {
        let msgs = vec![
            Message {
                id: "m0".to_string(),
                role: MessageRole::Assistant,
                parts: vec![Part {
                    id: "m0.p0".to_string(),
                    kind: PartKind::ToolCall,
                    content: r#"{"path":"README.md"}"#.to_string(),
                    attachment: None,
                    tool_call_id: Some("tc1".to_string()),
                    tool_name: Some("read_file".to_string()),
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::User,
                parts: vec![Part {
                    id: "m1.p0".to_string(),
                    kind: PartKind::ToolResult,
                    content: "ok".to_string(),
                    attachment: None,
                    tool_call_id: Some("tc1".to_string()),
                    tool_name: Some("read_file".to_string()),
                    tool_item_id: None,
                    prune_state: PruneState::Intact,
                }],
                user_input: None,
                origin: None,
            },
        ];

        assert!(messages_are_live_resume_safe(&msgs));
    }

    #[test]
    fn reasoning_parts_survive_snapshot_but_never_reach_the_model() {
        let reasoning_part = Part {
            id: "m1.p0".to_string(),
            kind: PartKind::Reasoning,
            content: "Thinking about how to answer.".to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        };

        let msgs = vec![Message {
            id: "m1".to_string(),
            role: MessageRole::Assistant,
            parts: vec![
                reasoning_part.clone(),
                part(PartKind::Prose, "Here is the answer."),
            ],
            user_input: None,
            origin: None,
        }];

        // JSON round-trip preserves the reasoning part — the snapshot
        // layer must not silently drop it, otherwise replays would lose
        // the trace.
        let serialized = serde_json::to_string(&msgs).expect("serialize messages");
        let deserialized: Vec<Message> =
            serde_json::from_str(&serialized).expect("deserialize messages");
        assert_eq!(deserialized[0].parts.len(), 2);
        assert!(matches!(
            deserialized[0].parts[0].kind,
            PartKind::Reasoning
        ));
        assert_eq!(
            deserialized[0].parts[0].content,
            "Thinking about how to answer."
        );

        // But the rendered LLM prompt must NOT include the reasoning
        // content in any assistant message — that's the safety property
        // for the next-turn re-feed path.
        let rendered = render_structured_prompt(&msgs);
        assert_eq!(rendered.messages.len(), 1);
        assert_eq!(rendered.messages[0].role, LlmRole::Assistant);
        assert_eq!(rendered.messages[0].content, "Here is the answer.");
        assert!(
            !rendered.messages[0]
                .content
                .contains("Thinking about how to answer.")
        );

        // Even when the assistant message consists solely of reasoning,
        // no assistant turn should be sent to the model.
        let reasoning_only = vec![Message {
            id: "m2".to_string(),
            role: MessageRole::Assistant,
            parts: vec![reasoning_part],
            user_input: None,
            origin: None,
        }];
        let rendered_only = render_structured_prompt(&reasoning_only);
        assert!(rendered_only.messages.is_empty());
    }

    #[test]
    fn live_resume_safety_rejects_unmatched_tool_calls() {
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::ToolCall,
                content: r#"{"path":"README.md"}"#.to_string(),
                attachment: None,
                tool_call_id: Some("tc1".to_string()),
                tool_name: Some("read_file".to_string()),
                tool_item_id: None,
                prune_state: PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }];

        assert!(!messages_are_live_resume_safe(&msgs));
    }
}
