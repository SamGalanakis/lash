use std::path::PathBuf;
use std::time::SystemTime;

use lash::TokenUsage;
use lash::agent::{ChatMsg, Message};

use crate::app::DisplayBlock;

pub struct SessionInfo {
    pub filename: String,
    pub model: String,
    pub message_count: usize,
    pub first_message: String,
    pub modified: SystemTime,
}

impl SessionInfo {
    /// Format the file modification time as a human-readable relative string.
    pub fn relative_time(&self) -> String {
        let elapsed = self.modified.elapsed().unwrap_or_default();
        let secs = elapsed.as_secs();
        if secs < 60 {
            "just now".to_string()
        } else if secs < 3600 {
            format!("{}m ago", secs / 60)
        } else if secs < 86400 {
            format!("{}h ago", secs / 3600)
        } else if secs < 604800 {
            format!("{}d ago", secs / 86400)
        } else {
            format!("{}w ago", secs / 604800)
        }
    }
}

pub fn sessions_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".lash").join("sessions")
}

/// List available session files, most recently modified first.
pub fn list_sessions() -> Vec<SessionInfo> {
    let dir = sessions_dir();
    let mut sessions = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return sessions,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let filename = path.file_name().unwrap().to_string_lossy().to_string();

        let modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Parse first line for session metadata
        let first_line = match content.lines().next() {
            Some(l) => l,
            None => continue,
        };
        let val: serde_json::Value = match serde_json::from_str(first_line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // Skip child sessions (spawned by delegate_task)
        if val.get("parent_session_id").is_some_and(|v| !v.is_null()) {
            continue;
        }
        let model = val
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Scan for first user_input and count messages
        let mut message_count = 0;
        let mut first_message = String::new();
        for line in content.lines() {
            let v: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let is_user = v
                .get("type")
                .and_then(|t| t.as_str())
                .is_some_and(|t| t == "user_input");
            if is_user {
                message_count += 1;
                if first_message.is_empty() {
                    first_message = v
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                }
            }
        }

        sessions.push(SessionInfo {
            filename,
            model,
            message_count,
            first_message,
            modified,
        });
    }
    // Sort by modification time, most recent first
    sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
    sessions
}

/// Load a session JSONL file, reconstructing display blocks and structured messages.
/// Converts legacy ChatMsg-based events into the new Message format.
pub fn load_session(filename: &str) -> Option<(Vec<Message>, Vec<DisplayBlock>)> {
    let path = sessions_dir().join(filename);
    let content = std::fs::read_to_string(&path).ok()?;

    let mut chat_msgs = Vec::new();
    let mut blocks = Vec::new();
    let mut pending_text = String::new();
    let mut sub_agent_count: usize = 0;

    for line in content.lines() {
        let val: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = match val.get("type").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => continue,
        };
        match ty {
            "user_input" => {
                let text = val
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                chat_msgs.push(ChatMsg {
                    role: "user".to_string(),
                    content: text.clone(),
                });
                blocks.push(DisplayBlock::UserInput(text));
            }
            "text_delta" => {
                if let Some(c) = val.get("content").and_then(|v| v.as_str()) {
                    pending_text.push_str(c);
                }
            }
            "llm_request" => {
                // New iteration — discard buffered text from previous iteration
                pending_text.clear();
            }
            "code_block" => {
                // Discard buffered text (intermediate thinking with code fences)
                pending_text.clear();
                if let Some(code) = val.get("code").and_then(|v| v.as_str()) {
                    blocks.push(DisplayBlock::CodeBlock {
                        code: code.to_string(),
                        expanded: false,
                        continuation: false,
                    });
                }
            }
            "tool_call" => {
                if let (Some(name), Some(success), Some(duration_ms)) = (
                    val.get("name").and_then(|v| v.as_str()),
                    val.get("success").and_then(|v| v.as_bool()),
                    val.get("duration_ms").and_then(|v| v.as_u64()),
                ) {
                    blocks.push(DisplayBlock::ToolCall {
                        name: name.to_string(),
                        success,
                        duration_ms,
                    });
                }
            }
            "code_output" => {
                let output = val
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let error = val
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                if !output.is_empty() || error.is_some() {
                    blocks.push(DisplayBlock::CodeOutput { output, error });
                }
            }
            "llm_response" => {
                // Reconstruct assistant message for chat history
                if let Some(c) = val.get("content").and_then(|v| v.as_str()) {
                    chat_msgs.push(ChatMsg {
                        role: "assistant".to_string(),
                        content: c.to_string(),
                    });
                }
            }
            "done" => {
                // Discard buffered text (code/comments only, no user-visible content)
                pending_text.clear();
            }
            "sub_agent_done" => {
                sub_agent_count += 1;
                let task = val
                    .get("task")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let usage: TokenUsage = val
                    .get("usage")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                let tool_calls =
                    val.get("tool_calls").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let iterations =
                    val.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let success = val.get("success").and_then(|v| v.as_bool()).unwrap_or(true);
                blocks.push(DisplayBlock::SubAgentResult {
                    task,
                    usage,
                    tool_calls,
                    iterations,
                    success,
                    is_last: true, // will be fixed below
                });
            }
            "message" => {
                let text = val
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let kind = val.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                if (kind == "final" || kind == "say") && !text.is_empty() {
                    blocks.push(DisplayBlock::AssistantText(text));
                }
            }
            "error" => {
                if let Some(msg) = val.get("message").and_then(|v| v.as_str()) {
                    blocks.push(DisplayBlock::Error(msg.to_string()));
                }
            }
            _ => {}
        }
    }

    // Fix is_last flags for consecutive SubAgentResult blocks
    if sub_agent_count > 0 {
        // Walk backward through blocks; within each consecutive run of SubAgentResults,
        // only the last one should have is_last = true.
        let mut in_run = false;
        for i in (0..blocks.len()).rev() {
            if let DisplayBlock::SubAgentResult {
                ref mut is_last, ..
            } = blocks[i]
            {
                if !in_run {
                    *is_last = true;
                    in_run = true;
                } else {
                    *is_last = false;
                }
            } else {
                in_run = false;
            }
        }
    }

    // Convert legacy ChatMsgs to structured Messages
    let messages = lash::agent::messages_from_chat(&chat_msgs);

    Some((messages, blocks))
}
