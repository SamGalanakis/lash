pub use crate::llm::types::LlmPromptPart;

pub const IMAGE_REF_PREFIX: &str = "__LASH_IMAGE_IDX:";

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

fn render_message_for_transcript(msg: &Message, image_indices: &mut Vec<usize>) -> String {
    let mut out = Vec::new();
    for part in &msg.parts {
        let rendered = render_part_for_chat(msg.role, part);
        if let Some(idx_str) = rendered.strip_prefix(IMAGE_REF_PREFIX)
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            image_indices.push(idx);
            out.push("[Image attached]".to_string());
            continue;
        }
        if !rendered.trim().is_empty() {
            out.push(rendered);
        }
    }
    out.join("\n\n")
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub user_prompt: Vec<LlmPromptPart>,
    pub image_indices: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
struct TranscriptTurn {
    user: Vec<String>,
    assistant: Vec<String>,
}

pub fn render_transcript_prompt(
    msgs: &[Message],
    base_context: &str,
    project_instructions: &str,
) -> RenderedPrompt {
    let mut image_indices = Vec::new();
    let mut turns = Vec::new();
    let mut current = TranscriptTurn::default();
    let mut has_current = false;

    for msg in msgs {
        let text = render_message_for_transcript(msg, &mut image_indices);
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
    if !base_context.trim().is_empty() {
        text.push_str("Project:\n");
        text.push_str(base_context.trim());
        text.push_str("\n\n");
    }
    if !project_instructions.trim().is_empty() {
        text.push_str("Project Instructions:\n");
        text.push_str(project_instructions.trim());
        text.push_str("\n\n");
    }
    text.push_str(
        "History:\nThis is a chronological transcript. `Assistant` refers to Lash, and you are continuing as the same agent.\n\n",
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
        user_prompt: vec![LlmPromptPart::Text(text)],
        image_indices,
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

        let rendered = render_transcript_prompt(&msgs, "ctx", "rules");
        let text = match &rendered.user_prompt[0] {
            LlmPromptPart::Text(text) => text,
            LlmPromptPart::Image(_) => panic!("expected text prompt"),
        };

        assert!(text.contains("Project:\nctx"));
        assert!(text.contains("Project Instructions:\nrules"));
        assert!(text.contains("=== Turn 1 ===\nUser:\nfirst"));
        assert!(text.contains("Assistant (Lash, continuing this transcript):\nreply one"));
        assert!(text.contains("=== Turn 2 ===\nUser:\nsecond"));
    }

    #[test]
    fn render_transcript_prompt_collects_images() {
        let msgs = vec![Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![part(PartKind::Text, "__LASH_IMAGE_IDX:3")],
            origin: None,
        }];

        let rendered = render_transcript_prompt(&msgs, "", "");
        assert_eq!(rendered.image_indices, vec![3]);
        let text = match &rendered.user_prompt[0] {
            LlmPromptPart::Text(text) => text,
            LlmPromptPart::Image(_) => panic!("expected text prompt"),
        };
        assert!(text.contains("[Image attached]"));
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

        let rendered = render_transcript_prompt(&msgs, "", "");
        let text = match &rendered.user_prompt[0] {
            LlmPromptPart::Text(text) => text,
            LlmPromptPart::Image(_) => panic!("expected text prompt"),
        };

        assert!(text.contains("=== Turn 2 ===\nUser:\nsecond"));
        assert!(!text.contains("=== Turn 2 ===\nUser:\nsecond\n\nAssistant (Lash, continuing this transcript):\n[No assistant content recorded]"));
    }
}
