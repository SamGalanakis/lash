use crate::ExecutionMode;
use crate::llm::types::{LlmAttachment, LlmMessage, LlmRole};
use base64::Engine;

// ─── Structured message types for context-aware pruning ───

/// A structured message with typed parts for context management.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub parts: Vec<Part>,
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
    Plugin { plugin_id: String },
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
}

pub fn data_url_for_bytes(mime: &str, bytes: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{b64}")
}

fn render_part_for_chat(role: MessageRole, part: &Part) -> String {
    let rendered = part.render();
    match role {
        MessageRole::System => match part.kind {
            PartKind::Code => format!("<repl>\n{}\n</repl>", rendered),
            PartKind::Output => format!("<output>\n{}\n</output>", rendered),
            PartKind::Error => format!("<error>\n{}\n</error>", rendered),
            PartKind::Text
            | PartKind::Image
            | PartKind::Prose
            | PartKind::ToolCall
            | PartKind::ToolResult => rendered,
        },
        MessageRole::Assistant => match part.kind {
            PartKind::Code => format!("<repl>\n{}\n</repl>", rendered),
            PartKind::ToolCall => render_assistant_tool_call(part, &rendered),
            PartKind::Prose | PartKind::Text | PartKind::Image | PartKind::ToolResult => rendered,
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
    let mut out = Vec::new();
    for part in &msg.parts {
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

#[derive(Clone, Debug, Default)]
struct TranscriptTurn {
    user: Vec<String>,
    assistant: Vec<String>,
}

pub fn render_prompt(msgs: &[Message], mode: ExecutionMode) -> RenderedPrompt {
    match mode {
        ExecutionMode::Repl => render_repl_chat_prompt(msgs),
        ExecutionMode::Standard => render_structured_prompt(msgs),
    }
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
        }],
        attachments,
    }
}

fn render_repl_chat_prompt(msgs: &[Message]) -> RenderedPrompt {
    let mut attachments = Vec::new();
    let mut messages = Vec::new();

    for msg in msgs {
        let mut current_chunks = Vec::new();

        for part in &msg.parts {
            if let Some(attachment) = attachment_from_part(part)
                && matches!(msg.role, MessageRole::User)
            {
                if !current_chunks.is_empty() {
                    messages.push(LlmMessage {
                        role: llm_role_for_message(msg.role),
                        content: current_chunks.join("\n\n"),
                        kind: "text".to_string(),
                        image_idx: -1,
                        tool_call_id: None,
                        tool_name: None,
                    });
                    current_chunks.clear();
                }

                let image_idx = attachments.len();
                attachments.push(attachment);
                messages.push(LlmMessage {
                    role: LlmRole::User,
                    content: String::new(),
                    kind: "image".to_string(),
                    image_idx: image_idx as i64,
                    tool_call_id: None,
                    tool_name: None,
                });
                continue;
            }

            let mut rendered = render_part_for_chat(msg.role, part);
            if rendered.trim().is_empty() {
                continue;
            }
            if matches!(msg.role, MessageRole::System) {
                rendered = format!("Runtime note:\n{rendered}");
            }
            current_chunks.push(rendered);
        }

        if !current_chunks.is_empty() {
            messages.push(LlmMessage {
                role: llm_role_for_message(msg.role),
                content: current_chunks.join("\n\n"),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            });
        }
    }

    RenderedPrompt {
        messages,
        attachments,
    }
}

fn render_structured_prompt(msgs: &[Message]) -> RenderedPrompt {
    let mut attachments = Vec::new();
    let mut messages = Vec::new();

    for msg in msgs {
        for part in &msg.parts {
            match part.kind {
                PartKind::ToolCall => {
                    messages.push(LlmMessage {
                        role: LlmRole::Assistant,
                        content: part.content.clone(),
                        kind: "tool_call".to_string(),
                        image_idx: -1,
                        tool_call_id: part.tool_call_id.clone(),
                        tool_name: part.tool_name.clone(),
                    });
                }
                PartKind::ToolResult => {
                    let rendered = part.render();
                    messages.push(LlmMessage {
                        role: llm_role_for_message(msg.role),
                        content: rendered,
                        kind: "tool_result".to_string(),
                        image_idx: -1,
                        tool_call_id: part.tool_call_id.clone(),
                        tool_name: part.tool_name.clone(),
                    });
                }
                _ => {
                    if let Some(attachment) = attachment_from_part(part)
                        && matches!(msg.role, MessageRole::User)
                    {
                        let image_idx = attachments.len();
                        attachments.push(attachment);
                        messages.push(LlmMessage {
                            role: LlmRole::User,
                            content: String::new(),
                            kind: "image".to_string(),
                            image_idx: image_idx as i64,
                            tool_call_id: None,
                            tool_name: None,
                        });
                        continue;
                    }

                    let mut rendered = render_part_for_chat(msg.role, part);
                    if rendered.trim().is_empty() {
                        continue;
                    }

                    if matches!(msg.role, MessageRole::System) {
                        rendered = format!("Runtime note:\n{rendered}");
                    }

                    messages.push(LlmMessage {
                        role: llm_role_for_message(msg.role),
                        content: rendered,
                        kind: "text".to_string(),
                        image_idx: -1,
                        tool_call_id: None,
                        tool_name: None,
                    });
                }
            }
        }
    }

    RenderedPrompt {
        messages,
        attachments,
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
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::Assistant,
                parts: vec![part(PartKind::Prose, "reply one")],
                origin: None,
            },
            Message {
                id: "m2".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "second")],
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
    fn render_prompt_repl_preserves_message_boundaries() {
        let msgs = vec![
            Message {
                id: "m1".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "first")],
                origin: None,
            },
            Message {
                id: "m2".to_string(),
                role: MessageRole::Assistant,
                parts: vec![
                    part(PartKind::Prose, "reply one"),
                    part(PartKind::Code, "x = 1"),
                ],
                origin: None,
            },
            Message {
                id: "m3".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "second")],
                origin: None,
            },
        ];

        let rendered = render_prompt(&msgs, ExecutionMode::Repl);
        assert_eq!(rendered.messages.len(), 3);
        assert_eq!(rendered.messages[0].content, "first");
        assert!(rendered.messages[1].content.contains("reply one"));
        assert!(
            rendered.messages[1]
                .content
                .contains("<repl>\nx = 1\n</repl>")
        );
        assert_eq!(rendered.messages[2].content, "second");
    }

    #[test]
    fn render_structured_prompt_preserves_tool_protocol_and_user_images() {
        let msgs = vec![
            Message {
                id: "m0".to_string(),
                role: MessageRole::System,
                parts: vec![part(PartKind::Text, "note")],
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "show this"), image_part(&[1, 2, 3])],
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
                    prune_state: PruneState::Intact,
                }],
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
                    prune_state: PruneState::Intact,
                }],
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
                    prune_state: PruneState::Intact,
                }],
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
                    prune_state: PruneState::Intact,
                }],
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
                origin: None,
            },
            Message {
                id: "m1".to_string(),
                role: MessageRole::Assistant,
                parts: vec![part(PartKind::Prose, "reply one")],
                origin: None,
            },
            Message {
                id: "m2".to_string(),
                role: MessageRole::User,
                parts: vec![part(PartKind::Text, "second")],
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
                    prune_state: PruneState::Intact,
                }],
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
            origin: None,
        }];

        let rendered = render_transcript_prompt(&msgs);
        let text = &rendered.messages[0].content;
        assert!(!text.contains("Runtime Notes:"));
    }
}
