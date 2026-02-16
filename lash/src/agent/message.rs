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
                        PartKind::Code => format!("```python\n{}\n```", rendered),
                        PartKind::Output => format!("Output:\n```\n{}\n```", rendered),
                        PartKind::Error => format!("Error:\n```\n{}\n```", rendered),
                        PartKind::Text => rendered,
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
            let sections = parse_markdown_sections(&msg.content);
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

        // User and assistant messages: single Text part
        Message {
            id: id.clone(),
            role,
            parts: vec![Part {
                id: format!("{}.p0", id),
                kind: if role == MessageRole::Assistant {
                    PartKind::Code
                } else {
                    PartKind::Text
                },
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

/// Parse markdown fenced code block format:
///   ```python ... ``` → Code
///   Output:\n``` ... ``` → Output
///   Error:\n``` ... ``` → Error
///   bare text → Text
fn parse_markdown_sections(content: &str) -> Vec<(PartKind, String)> {
    let mut sections = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        if line == "```python" {
            // Code block
            i += 1;
            let mut code = String::new();
            while i < lines.len() && lines[i] != "```" {
                if !code.is_empty() {
                    code.push('\n');
                }
                code.push_str(lines[i]);
                i += 1;
            }
            i += 1; // skip closing ```
            if !code.is_empty() {
                sections.push((PartKind::Code, code));
            }
        } else if line == "Output:" && i + 1 < lines.len() && lines[i + 1] == "```" {
            // Output block
            i += 2; // skip "Output:" and opening ```
            let mut output = String::new();
            while i < lines.len() && lines[i] != "```" {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(lines[i]);
                i += 1;
            }
            i += 1; // skip closing ```
            if !output.is_empty() {
                sections.push((PartKind::Output, output));
            }
        } else if line == "Error:" && i + 1 < lines.len() && lines[i + 1] == "```" {
            // Error block
            i += 2;
            let mut error = String::new();
            while i < lines.len() && lines[i] != "```" {
                if !error.is_empty() {
                    error.push('\n');
                }
                error.push_str(lines[i]);
                i += 1;
            }
            i += 1; // skip closing ```
            if !error.is_empty() {
                sections.push((PartKind::Error, error));
            }
        } else if !line.trim().is_empty() {
            // Plain text
            let mut text = String::from(line);
            i += 1;
            while i < lines.len()
                && !lines[i].starts_with("```")
                && lines[i] != "Output:"
                && lines[i] != "Error:"
            {
                text.push('\n');
                text.push_str(lines[i]);
                i += 1;
            }
            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                sections.push((PartKind::Text, trimmed));
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
