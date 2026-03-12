use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::Result;
use lash_core::AgentEvent;
use lash_core::TokenUsage;
use lash_core::agent::{Message, MessageRole, Part, PartKind, PruneState};

use crate::activity::{ActivityKind, ActivityState, ActivityStatus, merge_exploration_activity};
use crate::app::DisplayBlock;
use crate::replay::{AssistantReplay, push_assistant_text_block};
use crate::util::{is_manual_interrupt_error, manual_interrupt_message};

pub struct SessionInfo {
    pub filename: String,
    pub session_id: String,
    pub message_count: usize,
    pub first_message: String,
    pub modified: SystemTime,
}

pub struct SessionStart {
    pub session_id: String,
    pub session_name: String,
}

pub struct LoadedSession {
    pub messages: Vec<Message>,
    pub blocks: Vec<DisplayBlock>,
    pub last_token_usage: TokenUsage,
}

pub struct SessionLogger {
    file: std::io::BufWriter<std::fs::File>,
    pub session_id: String,
    filename: String,
    pending_turn: Vec<serde_json::Value>,
}

impl SessionLogger {
    pub fn new(model: &str, session_id: Option<String>, session_name: String) -> Result<Self> {
        let dir = lash_core::lash_home().join("sessions");
        std::fs::create_dir_all(&dir)?;

        let now = chrono::Local::now();
        let filename = format!("{}.jsonl", now.format("%Y%m%d_%H%M%S"));
        let path = dir.join(&filename);
        let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let session_id = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let mut logger = Self {
            file,
            session_id: session_id.clone(),
            filename,
            pending_turn: Vec::new(),
        };
        logger.write_json(&serde_json::json!({
            "type": "session_start",
            "session_id": session_id,
            "session_name": session_name,
            "ts": now.to_rfc3339(),
            "model": model,
            "cwd": std::env::current_dir().ok().map(|p| p.to_string_lossy().to_string()),
        }))?;
        logger.flush_file()?;

        Ok(logger)
    }

    pub fn resume(filename: &str) -> Result<Self> {
        let start = load_session_start(filename)
            .ok_or_else(|| anyhow::anyhow!("Could not load session metadata for {}", filename))?;
        let path = sessions_dir().join(filename);
        let file = std::io::BufWriter::new(
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)?,
        );
        Ok(Self {
            file,
            session_id: start.session_id,
            filename: filename.to_string(),
            pending_turn: Vec::new(),
        })
    }

    fn write_json(&mut self, value: &serde_json::Value) -> Result<()> {
        use std::io::Write;
        serde_json::to_writer(&mut self.file, value)?;
        self.file.write_all(b"\n")?;
        Ok(())
    }

    fn flush_file(&mut self) -> Result<()> {
        use std::io::Write;
        self.file.flush()?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.flush_pending_turn();
        self.flush_file()
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn clone_history_from(&mut self, source_filename: &str) -> Result<()> {
        let path = sessions_dir().join(source_filename);
        use std::io::Write;
        let reader = BufReader::new(File::open(&path)?);
        for line in reader.lines().skip(1) {
            let line = line?;
            self.file.write_all(line.as_bytes())?;
            self.file.write_all(b"\n")?;
        }
        self.flush_file()
    }

    fn flush_pending_turn(&mut self) {
        if self.pending_turn.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending_turn);
        for value in pending {
            let _ = self.write_json(&value);
        }
        let _ = self.flush_file();
    }

    pub fn log_user_input(&mut self, input: &str) {
        if !self.pending_turn.is_empty() {
            self.flush_pending_turn();
        }
        self.pending_turn.push(serde_json::json!({
            "type": "user_input",
            "ts": chrono::Local::now().to_rfc3339(),
            "content": input,
        }));
        self.flush_pending_turn();
    }

    pub fn log_event(&mut self, event: &AgentEvent) {
        let mut value = serde_json::to_value(event).unwrap_or_default();
        let mut event_type = String::new();
        if let serde_json::Value::Object(ref mut map) = value {
            map.insert(
                "ts".into(),
                serde_json::Value::String(chrono::Local::now().to_rfc3339()),
            );
            event_type = map
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
        }
        self.pending_turn.push(value);
        if should_flush_event(&event_type) {
            self.flush_pending_turn();
        }
    }
}

impl Drop for SessionLogger {
    fn drop(&mut self) {
        self.flush_pending_turn();
    }
}

fn should_flush_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "tool_call"
            | "code_output"
            | "message"
            | "llm_request"
            | "llm_response"
            | "retry_status"
            | "sub_agent_done"
            | "done"
            | "error"
            | "prompt"
    )
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
    lash_core::lash_home().join("sessions")
}

pub fn load_session_start(filename: &str) -> Option<SessionStart> {
    let path = sessions_dir().join(filename);
    let file = File::open(&path).ok()?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    reader.read_line(&mut first_line).ok()?;
    if first_line.is_empty() {
        return None;
    }
    let val: serde_json::Value = serde_json::from_str(&first_line).ok()?;
    Some(SessionStart {
        session_id: val.get("session_id")?.as_str()?.to_string(),
        session_name: val
            .get("session_name")
            .and_then(|v| v.as_str())
            .unwrap_or(filename.trim_end_matches(".jsonl"))
            .to_string(),
    })
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

        let file = match File::open(&path) {
            Ok(file) => file,
            Err(_) => continue,
        };
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Parse first line for session metadata
        let first_line = match lines.next() {
            Some(Ok(line)) => line,
            _ => continue,
        };
        let val: serde_json::Value = match serde_json::from_str(&first_line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // Skip child sessions (spawned by delegate_task)
        if val.get("parent_session_id").is_some_and(|v| !v.is_null()) {
            continue;
        }
        let session_id = val
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        // Scan for first user_input and count messages
        let mut message_count = 0;
        let mut first_message = String::new();
        for line in lines {
            let Ok(line) = line else {
                continue;
            };
            let v: serde_json::Value = match serde_json::from_str(&line) {
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
            session_id,
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
pub fn load_session(filename: &str) -> Option<LoadedSession> {
    let path = sessions_dir().join(filename);
    let file = File::open(&path).ok()?;
    let reader = BufReader::new(file);

    let mut messages = Vec::new();
    let mut blocks = Vec::new();
    let mut sub_agent_count: usize = 0;
    let mut assistant_replay = AssistantReplay::default();
    let mut last_token_usage = TokenUsage::default();
    let mut activity_state = ActivityState::default();

    for line in reader.lines() {
        let Ok(line) = line else {
            continue;
        };
        let val: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = match val.get("type").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => continue,
        };
        match ty {
            "user_input" => {
                assistant_replay.flush(&mut blocks);
                let text = val
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let msg_id = format!("m{}", messages.len());
                messages.push(Message {
                    id: msg_id.clone(),
                    role: MessageRole::User,
                    parts: vec![Part {
                        id: format!("{msg_id}.p0"),
                        kind: PartKind::Text,
                        content: text.clone(),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: PruneState::Intact,
                    }],
                    origin: None,
                });
                blocks.push(DisplayBlock::UserInput(text));
            }
            "text_delta" => {
                if let Some(text) = val.get("content").and_then(|v| v.as_str()) {
                    assistant_replay.push_text_delta(text);
                }
            }
            "llm_request" => {
                assistant_replay.flush(&mut blocks);
            }
            "done" => {
                assistant_replay.flush(&mut blocks);
            }
            "code_block" => {
                assistant_replay.flush(&mut blocks);
                if let Some(code) = val.get("code").and_then(|v| v.as_str()) {
                    blocks.push(DisplayBlock::CodeBlock {
                        code: code.to_string(),
                        continuation: false,
                    });
                }
            }
            "tool_call" => {
                assistant_replay.flush(&mut blocks);
                if let (Some(name), Some(success), Some(duration_ms)) = (
                    val.get("name").and_then(|v| v.as_str()),
                    val.get("success").and_then(|v| v.as_bool()),
                    val.get("duration_ms").and_then(|v| v.as_u64()),
                ) {
                    let args = val
                        .get("args")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({}));
                    let result = val
                        .get("result")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    for activity in activity_state.blocks_for_tool_call(
                        name,
                        args,
                        result,
                        success,
                        duration_ms,
                    ) {
                        if let Some(DisplayBlock::Activity(existing)) = blocks.last_mut()
                            && existing.kind == ActivityKind::Exploration
                            && activity.kind == ActivityKind::Exploration
                            && existing.status == ActivityStatus::Completed
                            && activity.status == ActivityStatus::Completed
                            && merge_exploration_activity(existing, activity.clone())
                        {
                            continue;
                        }
                        blocks.push(DisplayBlock::Activity(activity));
                    }
                }
            }
            "code_output" => {
                assistant_replay.flush(&mut blocks);
                let output = val
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let error = val
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .filter(|s| !s.trim().is_empty());
                if !output.is_empty() || error.is_some() {
                    blocks.push(DisplayBlock::CodeOutput { output, error });
                }
            }
            "llm_response" => {
                if let Some(c) = val.get("content").and_then(|v| v.as_str()) {
                    assistant_replay.remember_llm_response(c);
                    let msg_id = format!("m{}", messages.len());
                    messages.push(Message {
                        id: msg_id.clone(),
                        role: MessageRole::Assistant,
                        parts: vec![Part {
                            id: format!("{msg_id}.p0"),
                            kind: PartKind::Prose,
                            content: c.to_string(),
                            tool_call_id: None,
                            tool_name: None,
                            prune_state: PruneState::Intact,
                        }],
                        origin: None,
                    });
                }
            }
            "sub_agent_done" => {
                assistant_replay.flush(&mut blocks);
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
            "token_usage" => {
                if let Some(usage) = val
                    .get("usage")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                {
                    last_token_usage = usage;
                }
            }
            "message" => {
                assistant_replay.flush(&mut blocks);
                let text = val
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let kind = val.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                if kind == "final" && !text.is_empty() {
                    push_assistant_text_block(&mut blocks, &text);
                }
            }
            "error" => {
                assistant_replay.flush(&mut blocks);
                if let Some(msg) = val.get("message").and_then(|v| v.as_str()) {
                    let code = val
                        .get("envelope")
                        .and_then(|v| v.get("code"))
                        .and_then(|v| v.as_str());
                    if is_manual_interrupt_error(msg, code) {
                        blocks.push(DisplayBlock::SystemMessage(
                            manual_interrupt_message().to_string(),
                        ));
                    } else {
                        blocks.push(DisplayBlock::Error(msg.to_string()));
                    }
                }
            }
            _ => {}
        }
    }

    assistant_replay.flush(&mut blocks);

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

    Some(LoadedSession {
        messages,
        blocks,
        last_token_usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::AgentEvent;
    use serde_json::json;

    #[test]
    fn load_session_reconstructs_streamed_assistant_blocks() {
        let filename = format!("test-{}.jsonl", uuid::Uuid::new_v4());
        let path = sessions_dir().join(&filename);
        std::fs::create_dir_all(sessions_dir()).unwrap();
        std::fs::write(
            &path,
            concat!(
                "{\"type\":\"session_start\",\"session_id\":\"s1\",\"session_name\":\"demo\"}\n",
                "{\"type\":\"user_input\",\"content\":\"Hi\"}\n",
                "{\"type\":\"text_delta\",\"content\":\"Hello\"}\n",
                "{\"type\":\"text_delta\",\"content\":\" world\"}\n",
                "{\"type\":\"llm_response\",\"content\":\"Hello world\"}\n",
                "{\"type\":\"done\"}\n"
            ),
        )
        .unwrap();

        let loaded = load_session(&filename).unwrap();
        let blocks = loaded.blocks;
        assert!(matches!(blocks.first(), Some(DisplayBlock::UserInput(text)) if text == "Hi"));
        assert!(
            matches!(blocks.get(1), Some(DisplayBlock::AssistantText(text)) if text == "Hello world")
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_session_ignores_empty_code_output_errors() {
        let filename = format!("test-{}.jsonl", uuid::Uuid::new_v4());
        let path = sessions_dir().join(&filename);
        std::fs::create_dir_all(sessions_dir()).unwrap();
        std::fs::write(
            &path,
            concat!(
                "{\"type\":\"session_start\",\"session_id\":\"s1\",\"session_name\":\"demo\"}\n",
                "{\"type\":\"user_input\",\"content\":\"Hi\"}\n",
                "{\"type\":\"code_output\",\"output\":\"\",\"error\":\"\"}\n",
                "{\"type\":\"done\"}\n"
            ),
        )
        .unwrap();

        let loaded = load_session(&filename).unwrap();
        let blocks = loaded.blocks;
        assert!(matches!(blocks.first(), Some(DisplayBlock::UserInput(text)) if text == "Hi"));
        assert_eq!(blocks.len(), 1);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_session_maps_cancelled_error_to_system_message() {
        let filename = format!("test-{}.jsonl", uuid::Uuid::new_v4());
        let path = sessions_dir().join(&filename);
        std::fs::create_dir_all(sessions_dir()).unwrap();
        std::fs::write(
            &path,
            concat!(
                "{\"type\":\"session_start\",\"session_id\":\"s1\",\"session_name\":\"demo\"}\n",
                "{\"type\":\"user_input\",\"content\":\"Hi\"}\n",
                "{\"type\":\"error\",\"message\":\"LLM error: cancelled\",\"envelope\":{\"kind\":\"llm_provider\",\"code\":\"cancelled\",\"user_message\":\"LLM error: cancelled\"}}\n",
                "{\"type\":\"done\"}\n"
            ),
        )
        .unwrap();

        let loaded = load_session(&filename).unwrap();
        assert!(matches!(
            loaded.blocks.get(1),
            Some(DisplayBlock::SystemMessage(msg)) if msg == "Manually interrupted."
        ));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn logger_flushes_tool_calls_before_done() {
        let mut logger =
            SessionLogger::new("gpt-test", Some("s1".to_string()), "demo".to_string()).unwrap();
        let path = sessions_dir().join(logger.filename());

        logger.log_user_input("What time is it?");
        logger.log_event(&AgentEvent::ToolCall {
            name: "exec_command".to_string(),
            args: json!({"cmd": "date"}),
            result: json!({"wall_time_seconds": 0.01, "exit_code": 0, "output": "Thu\n"}),
            success: true,
            duration_ms: 0,
        });

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("\"type\":\"user_input\""));
        assert!(contents.contains("\"type\":\"tool_call\""));
        assert!(contents.contains("\"exec_command\""));

        let _ = std::fs::remove_file(path);
    }
}
