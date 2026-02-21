pub use crate::baml_client::types::ChatMsg;

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
    pub prune_state: PruneState,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PartKind {
    Text,
    Code,
    Output,
    Error,
    Prose,
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
    /// Convert structured message to ChatMsg for LLM prompt.
    pub fn to_chat_msg(&self) -> ChatMsg {
        let content = if self.role == MessageRole::System {
            self.parts
                .iter()
                .map(|p| {
                    let rendered = p.render();
                    match p.kind {
                        PartKind::Code => format!("<code>\n{}\n</code>", rendered),
                        PartKind::Output => format!("<output>\n{}\n</output>", rendered),
                        PartKind::Error => format!("<error>\n{}\n</error>", rendered),
                        PartKind::Text | PartKind::Prose => rendered,
                    }
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        } else if self.role == MessageRole::Assistant {
            // Assistant messages: prose renders as-is, code wrapped in <code> tags
            self.parts
                .iter()
                .map(|p| {
                    let rendered = p.render();
                    match p.kind {
                        PartKind::Code => format!("<code>\n{}\n</code>", rendered),
                        PartKind::Prose | PartKind::Text => rendered,
                        _ => rendered,
                    }
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        } else {
            self.parts
                .iter()
                .map(|p| p.render())
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        ChatMsg {
            role: match self.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
            }
            .to_string(),
            content,
            kind: "text".to_string(),
            image_idx: -1,
        }
    }

    /// Total character count of all parts (rendered).
    pub fn char_count(&self) -> usize {
        self.parts.iter().map(|p| p.render().len()).sum()
    }
}

/// Convert Vec<Message> to Vec<ChatMsg> (for LLM calls).
pub fn messages_to_chat(msgs: &[Message]) -> Vec<ChatMsg> {
    let mut out: Vec<ChatMsg> = Vec::new();
    for msg in msgs {
        if msg.role != MessageRole::User {
            out.push(msg.to_chat_msg());
            continue;
        }

        if msg.parts.is_empty() {
            out.push(msg.to_chat_msg());
            continue;
        }

        for part in &msg.parts {
            let rendered = part.render();
            if let Some(idx_str) = rendered.strip_prefix(IMAGE_REF_PREFIX)
                && let Ok(idx) = idx_str.parse::<i64>()
            {
                out.push(ChatMsg {
                    role: "user".to_string(),
                    content: String::new(),
                    kind: "image".to_string(),
                    image_idx: idx,
                });
                continue;
            }

            out.push(ChatMsg {
                role: "user".to_string(),
                content: rendered,
                kind: "text".to_string(),
                image_idx: -1,
            });
        }
    }
    out
}
