pub use crate::llm::types::{LlmMessage as ChatMsg, LlmRole};

pub const IMAGE_REF_PREFIX: &str = "__LASH_IMAGE_IDX:";

// ─── Structured message types for context-aware pruning ───

/// A structured message with typed parts for context management.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub parts: Vec<Part>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Part {
    /// e.g. "m3.p0"
    pub id: String,
    pub kind: PartKind,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    pub prune_state: PruneState,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PartKind {
    Text,
    Code,
    Output,
    Error,
    Prose,
    ToolCall,
    ToolResult,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PruneState {
    Intact,
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
    pub(crate) fn render(&self) -> String {
        match &self.prune_state {
            PruneState::Intact => self.content.clone(),
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
        self.parts.iter().map(|p| p.render().len()).sum()
    }
}

fn render_part_for_chat(role: MessageRole, part: &Part) -> String {
    let rendered = part.render();
    match role {
        MessageRole::System => match part.kind {
            PartKind::Code => format!("<repl>\n{}\n</repl>", rendered),
            PartKind::Output => format!("<output>\n{}\n</output>", rendered),
            PartKind::Error => format!("<error>\n{}\n</error>", rendered),
            PartKind::Text | PartKind::Prose | PartKind::ToolCall | PartKind::ToolResult => {
                rendered
            }
        },
        MessageRole::Assistant => match part.kind {
            PartKind::Code => format!("<repl>\n{}\n</repl>", rendered),
            PartKind::Prose | PartKind::Text | PartKind::ToolCall | PartKind::ToolResult => {
                rendered
            }
            _ => rendered,
        },
        MessageRole::User => rendered,
    }
}

/// Convert Vec<Message> to Vec<ChatMsg> (for LLM calls).
pub fn messages_to_chat(msgs: &[Message]) -> Vec<ChatMsg> {
    let mut out: Vec<ChatMsg> = Vec::new();
    for msg in msgs {
        if msg.parts.is_empty() {
            out.push(ChatMsg {
                role: match msg.role {
                    MessageRole::User => LlmRole::User,
                    MessageRole::Assistant => LlmRole::Assistant,
                    MessageRole::System => LlmRole::System,
                },
                content: String::new(),
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            });
            continue;
        }

        for part in &msg.parts {
            let rendered = render_part_for_chat(msg.role, part);
            if matches!(part.kind, PartKind::ToolCall) {
                out.push(ChatMsg {
                    role: LlmRole::Assistant,
                    content: rendered,
                    kind: "tool_call".to_string(),
                    image_idx: -1,
                    tool_call_id: part.tool_call_id.clone(),
                    tool_name: part.tool_name.clone(),
                });
                continue;
            }
            if matches!(part.kind, PartKind::ToolResult) {
                out.push(ChatMsg {
                    role: LlmRole::User,
                    content: rendered,
                    kind: "tool_result".to_string(),
                    image_idx: -1,
                    tool_call_id: part.tool_call_id.clone(),
                    tool_name: part.tool_name.clone(),
                });
                continue;
            }
            if let Some(idx_str) = rendered.strip_prefix(IMAGE_REF_PREFIX)
                && let Ok(idx) = idx_str.parse::<i64>()
            {
                // Route image payloads as user messages for broad provider compatibility.
                out.push(ChatMsg {
                    role: LlmRole::User,
                    content: String::new(),
                    kind: "image".to_string(),
                    image_idx: idx,
                    tool_call_id: None,
                    tool_name: None,
                });
                continue;
            }

            out.push(ChatMsg {
                role: match msg.role {
                    MessageRole::User => LlmRole::User,
                    MessageRole::Assistant => LlmRole::Assistant,
                    MessageRole::System => LlmRole::System,
                },
                content: rendered,
                kind: "text".to_string(),
                image_idx: -1,
                tool_call_id: None,
                tool_name: None,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(kind: PartKind, content: &str) -> Part {
        Part {
            id: "p0".to_string(),
            kind,
            content: content.to_string(),
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }
    }

    #[test]
    fn messages_to_chat_preserves_system_output_wrapping() {
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::System,
            parts: vec![part(PartKind::Output, "hello")],
        }];

        let out = messages_to_chat(&msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, LlmRole::System);
        assert_eq!(out[0].kind, "text");
        assert_eq!(out[0].content, "<output>\nhello\n</output>");
    }

    #[test]
    fn messages_to_chat_turns_system_image_ref_into_image_message() {
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::System,
            parts: vec![part(PartKind::Text, "__LASH_IMAGE_IDX:3")],
        }];

        let out = messages_to_chat(&msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, LlmRole::User);
        assert_eq!(out[0].kind, "image");
        assert_eq!(out[0].image_idx, 3);
    }

    #[test]
    fn messages_to_chat_expands_tool_parts_into_llm_tool_messages() {
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![
                Part {
                    id: "m0.p0".to_string(),
                    kind: PartKind::ToolCall,
                    content: r#"{"path":"src/lib.rs"}"#.to_string(),
                    tool_call_id: Some("call_1".to_string()),
                    tool_name: Some("read_file".to_string()),
                    prune_state: PruneState::Intact,
                },
                Part {
                    id: "m0.p1".to_string(),
                    kind: PartKind::ToolResult,
                    content: "file contents".to_string(),
                    tool_call_id: Some("call_1".to_string()),
                    tool_name: Some("read_file".to_string()),
                    prune_state: PruneState::Intact,
                },
            ],
        }];

        let out = messages_to_chat(&msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, LlmRole::Assistant);
        assert_eq!(out[0].kind, "tool_call");
        assert_eq!(out[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(out[0].tool_name.as_deref(), Some("read_file"));
        assert_eq!(out[0].content, r#"{"path":"src/lib.rs"}"#);
        assert_eq!(out[1].role, LlmRole::User);
        assert_eq!(out[1].kind, "tool_result");
        assert_eq!(out[1].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(out[1].tool_name.as_deref(), Some("read_file"));
        assert_eq!(out[1].content, "file contents");
    }
}
