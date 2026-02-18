pub use crate::baml_client::types::ChatMsg;

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
        }
    }

    /// Build a Message from a ChatMsg (for session loading).
    /// Parses the markdown fenced code block format for system messages.
    pub fn from_chat_msg(msg: &ChatMsg, id: String) -> Self {
        let role = match msg.role.as_str() {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            _ => MessageRole::System,
        };

        if role == MessageRole::System {
            let mut parts = Vec::new();
            let sections = parse_sections(&msg.content);
            for (idx, (kind, text)) in sections.into_iter().enumerate() {
                parts.push(Part {
                    id: format!("{}.p{}", id, idx),
                    kind,
                    content: text,
                    prune_state: PruneState::Intact,
                });
            }

            if parts.is_empty() {
                parts.push(Part {
                    id: format!("{}.p0", id),
                    kind: PartKind::Text,
                    content: msg.content.clone(),
                    prune_state: PruneState::Intact,
                });
            }

            return Message { id, role, parts };
        }

        if role == MessageRole::Assistant {
            // Parse assistant messages: prose + ```python code blocks
            let mut parts = Vec::new();
            let sections = parse_assistant_sections(&msg.content);
            for (idx, (kind, text)) in sections.into_iter().enumerate() {
                parts.push(Part {
                    id: format!("{}.p{}", id, idx),
                    kind,
                    content: text,
                    prune_state: PruneState::Intact,
                });
            }
            if parts.is_empty() {
                parts.push(Part {
                    id: format!("{}.p0", id),
                    kind: PartKind::Prose,
                    content: msg.content.clone(),
                    prune_state: PruneState::Intact,
                });
            }
            return Message { id, role, parts };
        }

        // User messages: single Text part
        Message {
            id: id.clone(),
            role,
            parts: vec![Part {
                id: format!("{}.p0", id),
                kind: PartKind::Text,
                content: msg.content.clone(),
                prune_state: PruneState::Intact,
            }],
        }
    }

    /// Total character count of all parts (rendered).
    pub fn char_count(&self) -> usize {
        self.parts.iter().map(|p| p.render().len()).sum()
    }
}

/// Parse XML-tagged sections in system/feedback messages:
///   <code>...</code> -> Code
///   <output>...</output> -> Output
///   <error>...</error> -> Error
///   bare text -> Text
fn parse_sections(content: &str) -> Vec<(PartKind, String)> {
    let mut sections = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        if trimmed == "<code>" {
            i += 1;
            let mut code = String::new();
            while i < lines.len() && lines[i].trim() != "</code>" {
                if !code.is_empty() {
                    code.push('\n');
                }
                code.push_str(lines[i]);
                i += 1;
            }
            i += 1; // skip </code>
            if !code.is_empty() {
                sections.push((PartKind::Code, code));
            }
        } else if trimmed == "<output>" {
            i += 1;
            let mut output = String::new();
            while i < lines.len() && lines[i].trim() != "</output>" {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(lines[i]);
                i += 1;
            }
            i += 1; // skip </output>
            if !output.is_empty() {
                sections.push((PartKind::Output, output));
            }
        } else if trimmed == "<error>" {
            i += 1;
            let mut error = String::new();
            while i < lines.len() && lines[i].trim() != "</error>" {
                if !error.is_empty() {
                    error.push('\n');
                }
                error.push_str(lines[i]);
                i += 1;
            }
            i += 1; // skip </error>
            if !error.is_empty() {
                sections.push((PartKind::Error, error));
            }
        } else if !trimmed.is_empty() {
            // Plain text
            let mut text = String::from(lines[i]);
            i += 1;
            while i < lines.len() {
                let next = lines[i].trim();
                if next == "<code>" || next == "<output>" || next == "<error>" {
                    break;
                }
                text.push('\n');
                text.push_str(lines[i]);
                i += 1;
            }
            let trimmed_text = text.trim().to_string();
            if !trimmed_text.is_empty() {
                sections.push((PartKind::Text, trimmed_text));
            }
        } else {
            i += 1;
        }
    }

    sections
}

/// Parse assistant message content into alternating Prose/Code sections.
fn parse_assistant_sections(content: &str) -> Vec<(PartKind, String)> {
    let mut sections = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        if trimmed == "<code>" {
            i += 1;
            let mut code = String::new();
            while i < lines.len() && lines[i].trim() != "</code>" {
                if !code.is_empty() {
                    code.push('\n');
                }
                code.push_str(lines[i]);
                i += 1;
            }
            i += 1; // skip </code>
            if !code.is_empty() {
                sections.push((PartKind::Code, code));
            }
        } else if !trimmed.is_empty() {
            // Prose text
            let mut text = String::from(lines[i]);
            i += 1;
            while i < lines.len() {
                if lines[i].trim() == "<code>" {
                    break;
                }
                text.push('\n');
                text.push_str(lines[i]);
                i += 1;
            }
            let trimmed_text = text.trim().to_string();
            if !trimmed_text.is_empty() {
                sections.push((PartKind::Prose, trimmed_text));
            }
        } else {
            i += 1;
        }
    }

    sections
}

/// Convert Vec<ChatMsg> to Vec<Message> (for loading legacy sessions).
pub fn messages_from_chat(msgs: &[ChatMsg]) -> Vec<Message> {
    msgs.iter()
        .enumerate()
        .map(|(i, m)| Message::from_chat_msg(m, format!("m{}", i)))
        .collect()
}

/// Convert Vec<Message> to Vec<ChatMsg> (for LLM calls).
pub fn messages_to_chat(msgs: &[Message]) -> Vec<ChatMsg> {
    msgs.iter().map(|m| m.to_chat_msg()).collect()
}
