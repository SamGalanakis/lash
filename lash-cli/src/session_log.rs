use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result};
use lash::agent::{Message, MessageRole, PartKind};
use lash::{Store, TokenUsage};

use crate::app::{DisplayBlock, PersistedUiState};

pub struct SessionInfo {
    pub filename: String,
    pub session_id: String,
    pub message_count: usize,
    pub first_message: String,
    pub modified: SystemTime,
    pub cwd: Option<PathBuf>,
}

pub struct SessionStart {
    pub session_id: String,
    pub session_name: String,
}

pub struct LoadedSession {
    pub messages: Vec<Message>,
    pub blocks: Vec<DisplayBlock>,
    pub last_token_usage: TokenUsage,
    pub plugin_mode_indicators: BTreeMap<String, String>,
}

pub struct SessionLogger {
    store: Arc<Store>,
    pub session_id: String,
    filename: String,
}

impl SessionLogger {
    pub fn new(
        store: Arc<Store>,
        filename: String,
        model: &str,
        session_id: Option<String>,
        session_name: String,
    ) -> Result<Self> {
        std::fs::create_dir_all(sessions_dir())?;
        let session_id = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        store.save_session_meta(lash::SessionMeta {
            session_id: session_id.clone(),
            session_name: session_name.clone(),
            created_at: chrono::Local::now().to_rfc3339(),
            model: model.to_string(),
            cwd: std::env::current_dir()
                .ok()
                .and_then(|path| path.to_str().map(str::to_string)),
            parent_session_id: None,
        });
        Ok(Self {
            store,
            session_id,
            filename,
        })
    }

    pub fn resume(store: Arc<Store>, filename: &str) -> Result<Self> {
        let start = store
            .load_session_meta()
            .map(|meta| SessionStart {
                session_id: meta.session_id,
                session_name: meta.session_name,
            })
            .ok_or_else(|| anyhow::anyhow!("Could not load session metadata for {}", filename))?;
        Ok(Self {
            store,
            session_id: start.session_id,
            filename: filename.to_string(),
        })
    }

    pub fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    pub fn clone_history_from(&mut self, source_filename: &str) -> Result<()> {
        let source = Store::open(&sessions_dir().join(source_filename))?;
        self.store
            .history_copy_from_store(&source, crate::ROOT_SESSION_ID, crate::ROOT_SESSION_ID);
        Ok(())
    }

    pub fn mark_as_child_of(&self, parent_session_id: &str) -> Result<()> {
        let meta = self
            .store
            .load_session_meta()
            .ok_or_else(|| anyhow::anyhow!("Missing session metadata for {}", self.filename))?;
        self.store.save_session_meta(lash::SessionMeta {
            session_id: meta.session_id,
            session_name: meta.session_name,
            created_at: meta.created_at,
            model: meta.model,
            cwd: meta.cwd,
            parent_session_id: Some(parent_session_id.to_string()),
        });
        Ok(())
    }
}

impl SessionInfo {
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

    pub fn cwd_label(&self) -> Option<String> {
        let cwd = self.cwd.as_deref()?;
        let name = cwd
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .map(|name| format!("/{name}"));
        if name.is_some() {
            name
        } else if cwd == Path::new("/") {
            Some("/".to_string())
        } else {
            None
        }
    }

    fn project_match_rank(&self, current_cwd: Option<&Path>) -> u8 {
        let Some(current_cwd) = current_cwd else {
            return 0;
        };
        let Some(session_cwd) = self.cwd.as_deref() else {
            return 0;
        };

        if session_cwd == current_cwd {
            3
        } else if current_cwd.starts_with(session_cwd) || session_cwd.starts_with(current_cwd) {
            2
        } else {
            0
        }
    }
}

pub fn sessions_dir() -> PathBuf {
    lash::lash_home().join("sessions")
}

pub fn new_session_filename() -> String {
    format!("{}.db", chrono::Local::now().format("%Y%m%d_%H%M%S"))
}

fn is_resumable_session_store(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("db")
}

fn parse_session_info(path: &Path, filename: String, modified: SystemTime) -> Option<SessionInfo> {
    let store = Store::open(path).ok()?;
    let meta = store.load_session_meta()?;
    if meta.parent_session_id.is_some() {
        return None;
    }

    let messages = load_messages(&store).ok()?;
    let mut message_count = 0usize;
    let mut first_message = String::new();
    for message in messages {
        if message.role != MessageRole::User {
            continue;
        }
        message_count += 1;
        if first_message.is_empty() {
            first_message = preview_message_text(&message);
        }
    }

    Some(SessionInfo {
        filename,
        session_id: meta.session_id,
        message_count,
        first_message,
        modified,
        cwd: meta.cwd.map(PathBuf::from),
    })
}

fn collect_session_candidates() -> Vec<(PathBuf, String, SystemTime)> {
    let mut candidates = Vec::new();
    let entries = match std::fs::read_dir(sessions_dir()) {
        Ok(entries) => entries,
        Err(_) => return candidates,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !is_resumable_session_store(&path) {
            continue;
        }
        let modified = std::fs::metadata(&path)
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let Some(filename) = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
        else {
            continue;
        };
        candidates.push((path, filename, modified));
    }
    candidates.sort_by(|a, b| b.2.cmp(&a.2));
    candidates
}

pub fn load_session_start(filename: &str) -> Result<SessionStart> {
    let store = Store::open(&sessions_dir().join(filename))?;
    let meta = store
        .load_session_meta()
        .ok_or_else(|| anyhow::anyhow!("Could not load session metadata for {}", filename))?;
    Ok(SessionStart {
        session_id: meta.session_id,
        session_name: meta.session_name,
    })
}

pub fn list_recent_sessions(limit: usize) -> Vec<SessionInfo> {
    if limit == 0 {
        return Vec::new();
    }

    let current_cwd = std::env::current_dir().ok();
    let mut sessions: Vec<_> = collect_session_candidates()
        .into_iter()
        .filter_map(|(path, filename, modified)| parse_session_info(&path, filename, modified))
        .collect();
    sessions.sort_by(|a, b| {
        b.project_match_rank(current_cwd.as_deref())
            .cmp(&a.project_match_rank(current_cwd.as_deref()))
            .then_with(|| b.modified.cmp(&a.modified))
            .then_with(|| a.filename.cmp(&b.filename))
    });
    sessions.truncate(limit);
    sessions
}

pub fn load_session(filename: &str) -> Result<LoadedSession> {
    let store = Store::open(&sessions_dir().join(filename))?;
    let messages = load_messages(&store)?;
    let ui_state = load_ui_state(&store)?;
    tracing::debug!(
        session_file = filename,
        messages = messages.len(),
        blocks = ui_state.blocks.len(),
        plugin_mode_indicators = ui_state.plugin_mode_indicators.len(),
        "loaded persisted session snapshot"
    );

    Ok(LoadedSession {
        messages,
        blocks: ui_state.blocks,
        last_token_usage: ui_state.last_response_usage,
        plugin_mode_indicators: ui_state.plugin_mode_indicators,
    })
}

fn load_messages(store: &Store) -> Result<Vec<Message>> {
    let state = store
        .load_agent_state(crate::ROOT_SESSION_ID)
        .ok_or_else(|| anyhow::anyhow!("Missing root session snapshot"))?;
    serde_json::from_str(&state.messages_json).context("Invalid persisted message snapshot")
}

fn load_ui_state(store: &Store) -> Result<PersistedUiState> {
    let state = store
        .load_agent_state(crate::ROOT_SESSION_ID)
        .ok_or_else(|| anyhow::anyhow!("Missing root session snapshot"))?;
    serde_json::from_str(&state.ui_json).context("Invalid persisted UI snapshot")
}

fn preview_message_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| preview_part_text(&part.kind, &part.content))
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn preview_part_text(kind: &PartKind, content: &str) -> Option<String> {
    if matches!(kind, PartKind::ToolCall | PartKind::ToolResult) {
        return None;
    }
    if matches!(kind, PartKind::Image) {
        return Some("[Image attached]".to_string());
    }
    (!content.trim().is_empty()).then(|| content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, TempDirGuard, env_lock};

    fn with_temp_lash_home(test_name: &str, f: impl FnOnce()) {
        let _env_guard = env_lock().blocking_lock();
        let temp = TempDirGuard::new(test_name);
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        std::fs::create_dir_all(sessions_dir()).expect("sessions dir");
        f();
    }

    fn persist_root_snapshot(store: &Store, messages: Vec<Message>, ui_state: PersistedUiState) {
        let messages_json = serde_json::to_string(&messages).expect("messages");
        let ui_json = serde_json::to_string(&ui_state).expect("ui state");
        store.save_agent_state(lash::AgentState {
            agent_id: crate::ROOT_SESSION_ID.to_string(),
            messages_json,
            tool_calls_json: "[]".to_string(),
            ui_json,
            iteration: 1,
            config_json: "{}".to_string(),
            repl_snapshot: None,
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        });
    }

    fn text_message(role: MessageRole, id: &str, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: vec![lash::Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: lash::PruneState::Intact,
            }],
            origin: None,
        }
    }

    #[test]
    fn load_session_reads_persisted_root_snapshot() {
        with_temp_lash_home("lash-session-load-root-snapshot", || {
            let filename = new_session_filename();
            let path = sessions_dir().join(&filename);
            let store = Arc::new(Store::open(&path).unwrap());
            SessionLogger::new(
                Arc::clone(&store),
                filename.clone(),
                "gpt-test",
                Some("s1".into()),
                "demo".into(),
            )
            .unwrap();
            let messages = vec![
                text_message(MessageRole::User, "m0", "Hi"),
                text_message(MessageRole::Assistant, "m1", "Hello world"),
            ];
            persist_root_snapshot(
                &store,
                messages.clone(),
                PersistedUiState {
                    blocks: vec![
                        DisplayBlock::UserInput("Hi".into()),
                        DisplayBlock::AssistantText("Hello world".into()),
                    ],
                    last_response_usage: TokenUsage {
                        input_tokens: 12,
                        output_tokens: 7,
                        cached_input_tokens: 2,
                        reasoning_tokens: 1,
                    },
                    plugin_mode_indicators: BTreeMap::from([(
                        "plan_mode".to_string(),
                        "plan".to_string(),
                    )]),
                },
            );

            let loaded = load_session(&filename).unwrap();
            assert_eq!(loaded.messages.len(), 2);
            assert!(matches!(
                loaded.blocks.first(),
                Some(DisplayBlock::UserInput(text)) if text == "Hi"
            ));
            assert!(matches!(
                loaded.blocks.get(1),
                Some(DisplayBlock::AssistantText(text)) if text == "Hello world"
            ));
            assert_eq!(loaded.last_token_usage.input_tokens, 12);
            assert_eq!(
                loaded.plugin_mode_indicators.get("plan_mode"),
                Some(&"plan".to_string())
            );
        });
    }

    #[test]
    fn list_recent_sessions_ignores_child_sessions() {
        with_temp_lash_home("lash-session-log-list", || {
            let parent = sessions_dir().join("parent.db");
            let child = sessions_dir().join("child.db");
            let parent_store = Arc::new(Store::open(&parent).unwrap());
            let child_store = Store::open(&child).unwrap();
            SessionLogger::new(
                Arc::clone(&parent_store),
                "parent.db".into(),
                "gpt-test",
                Some("parent".into()),
                "demo".into(),
            )
            .unwrap();
            persist_root_snapshot(
                &parent_store,
                vec![text_message(MessageRole::User, "m0", "hello there")],
                PersistedUiState {
                    blocks: vec![DisplayBlock::UserInput("hello there".into())],
                    ..PersistedUiState::default()
                },
            );
            child_store.save_session_meta(lash::SessionMeta {
                session_id: "child".to_string(),
                session_name: "demo".to_string(),
                created_at: "2026-03-25T10:00:00Z".to_string(),
                model: "gpt-test".to_string(),
                cwd: std::env::current_dir()
                    .ok()
                    .and_then(|path| path.to_str().map(str::to_string)),
                parent_session_id: Some("parent".to_string()),
            });
            persist_root_snapshot(
                &child_store,
                vec![text_message(MessageRole::User, "m0", "child prompt")],
                PersistedUiState {
                    blocks: vec![DisplayBlock::UserInput("child prompt".into())],
                    ..PersistedUiState::default()
                },
            );

            let sessions = list_recent_sessions(10);
            assert_eq!(sessions.len(), 1);
            assert_eq!(sessions[0].filename, "parent.db");
            assert_eq!(sessions[0].message_count, 1);
            assert_eq!(sessions[0].first_message, "hello there");
        });
    }

    #[test]
    fn clone_history_from_copies_history_turns() {
        with_temp_lash_home("lash-session-log-clone", || {
            let source_filename = "source.db".to_string();
            let target_filename = "target.db".to_string();
            let source_store =
                Arc::new(Store::open(&sessions_dir().join(&source_filename)).unwrap());
            let target_store =
                Arc::new(Store::open(&sessions_dir().join(&target_filename)).unwrap());
            let mut source = SessionLogger::new(
                Arc::clone(&source_store),
                source_filename.clone(),
                "gpt-test",
                Some("source".into()),
                "source".into(),
            )
            .unwrap();
            source_store.history_upsert_turn(
                crate::ROOT_SESSION_ID,
                lash::store::HistoryTurnRecord {
                    index: 0,
                    user_message: "hello".into(),
                    prose: "world".into(),
                    code: String::new(),
                    output: String::new(),
                    error: None,
                    tool_calls: Vec::new(),
                    files_read: Vec::new(),
                    files_written: Vec::new(),
                },
            );

            let mut target = SessionLogger::new(
                Arc::clone(&target_store),
                target_filename.clone(),
                "gpt-test",
                Some("target".into()),
                "target".into(),
            )
            .unwrap();
            target.clone_history_from(&source_filename).unwrap();

            let turns = target_store.history_export(crate::ROOT_SESSION_ID);
            assert_eq!(turns.len(), 1);
            assert_eq!(turns[0].user_message, "hello");
            assert_eq!(turns[0].prose, "world");
            let _ = (&mut source, &mut target);
        });
    }
}
