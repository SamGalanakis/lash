use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
#[cfg(test)]
use lash::ToolCallRecord;
use lash::session_model::Message;
#[cfg(test)]
use lash::session_model::{MessageRole, PartKind};
use lash::{Store, TokenUsage};

#[cfg(test)]
use crate::app::UiResumeState;
use crate::app::{DisplayBlock, projected_blocks_from_state};
use crate::resume_snapshot;
use crate::ui_resume;

#[derive(Clone, Debug)]
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
    pub streaming_output: Vec<String>,
    pub streaming_output_hidden: usize,
    pub streaming_output_partial: String,
}

pub struct SessionLogger {
    store: Arc<Store>,
    pub session_id: String,
    filename: String,
}

#[derive(Clone)]
pub struct DbSessionStoreFactory {
    sessions_dir: PathBuf,
}

impl DbSessionStoreFactory {
    pub fn new(sessions_dir: PathBuf) -> Self {
        Self { sessions_dir }
    }

    fn next_path(&self) -> Result<PathBuf, String> {
        std::fs::create_dir_all(&self.sessions_dir).map_err(|err| err.to_string())?;
        let filename = format!(
            "{}-{}.db",
            chrono::Local::now().format("%Y%m%d_%H%M%S"),
            &uuid::Uuid::new_v4().to_string()[..8]
        );
        Ok(self.sessions_dir.join(filename))
    }
}

impl lash::SessionStoreFactory for DbSessionStoreFactory {
    fn create_store(
        &self,
        request: &lash::SessionStoreCreateRequest,
    ) -> Result<Arc<dyn lash::RuntimeStore>, String> {
        let path = self.next_path()?;
        let store = Arc::new(Store::open(&path).map_err(|err| err.to_string())?);
        store.save_session_meta(lash::SessionMeta {
            session_id: request.session_id.clone(),
            session_name: request.session_id.clone(),
            created_at: chrono::Local::now().to_rfc3339(),
            model: request.policy.model.clone(),
            cwd: std::env::current_dir()
                .ok()
                .and_then(|path| path.to_str().map(str::to_string)),
            parent_session_id: request.parent_session_id.clone(),
        });
        Ok(store as Arc<dyn lash::RuntimeStore>)
    }
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

    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    pub fn db_path(&self) -> PathBuf {
        sessions_dir().join(&self.filename)
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
    let store = Store::open_readonly(path).ok()?;
    let info = store.load_picker_info()?;
    if info.parent_session_id.is_some() {
        return None;
    }

    Some(SessionInfo {
        filename,
        session_id: info.session_id,
        message_count: info.user_message_count,
        first_message: info.first_user_message,
        modified,
        cwd: info.cwd.map(PathBuf::from),
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

pub fn filename_for_session_id(session_id: &str) -> Option<String> {
    collect_session_candidates()
        .into_iter()
        .find_map(|(path, filename, _)| {
            let store = Store::open_readonly(&path).ok()?;
            let meta = store.load_session_meta()?;
            (meta.session_id == session_id).then_some(filename)
        })
}

pub fn list_recent_sessions(limit: usize) -> Vec<SessionInfo> {
    if limit == 0 {
        return Vec::new();
    }

    // Candidates are pre-sorted by modified time (latest first).
    let mut sessions: Vec<_> = collect_session_candidates()
        .into_iter()
        .filter_map(|(path, filename, modified)| parse_session_info(&path, filename, modified))
        .take(limit)
        .collect();
    sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
    sessions
}

pub fn load_session(filename: &str) -> Result<LoadedSession> {
    let store = Store::open(&sessions_dir().join(filename))?;
    if let Some(live) = resume_snapshot::load_live_resume_snapshot(&store) {
        let messages = live.graph.project_messages();
        let tool_calls = live.graph.project_tool_calls();
        let checkpoint = live
            .snapshot
            .checkpoint_ref
            .as_ref()
            .and_then(|blob_ref| store.get_checkpoint(blob_ref));
        let live_turn_state = live.delta.as_ref().map(|delta| &delta.turn_state);
        let blocks = projected_blocks_from_state(&messages, &tool_calls, &live.ui_state);
        return Ok(LoadedSession {
            messages,
            blocks,
            last_token_usage: live_turn_state
                .map(|state| state.token_usage.clone())
                .or_else(|| checkpoint.map(|checkpoint| checkpoint.turn_state.token_usage))
                .unwrap_or_default(),
            plugin_mode_indicators: live.ui_state.plugin_mode_indicators.clone(),
            streaming_output: live.ui_state.streaming_output.clone(),
            streaming_output_hidden: live.ui_state.streaming_output_hidden,
            streaming_output_partial: live.ui_state.streaming_output_partial.clone(),
        });
    }
    let head = store.load_session_head().unwrap_or_default();
    let graph = head.graph;
    let messages = graph.project_messages();
    let tool_calls = graph.project_tool_calls();
    let ui_state = ui_resume::load_ui_resume_state(&store);
    let checkpoint = head
        .checkpoint_ref
        .as_ref()
        .and_then(|blob_ref| store.get_checkpoint(blob_ref));
    let last_response_usage = ui_state.last_response_usage.clone();
    let plugin_mode_indicators = ui_state.plugin_mode_indicators.clone();
    let streaming_output = ui_state.streaming_output.clone();
    let streaming_output_hidden = ui_state.streaming_output_hidden;
    let streaming_output_partial = ui_state.streaming_output_partial.clone();
    let blocks = projected_blocks_from_state(&messages, &tool_calls, &ui_state);
    tracing::debug!(
        session_file = filename,
        messages = messages.len(),
        tool_calls = tool_calls.len(),
        blocks = blocks.len(),
        plugin_mode_indicators = plugin_mode_indicators.len(),
        graph_nodes = graph.nodes.len(),
        "loaded persisted session snapshot"
    );

    Ok(LoadedSession {
        messages,
        blocks,
        last_token_usage: checkpoint
            .map(|checkpoint| checkpoint.turn_state.token_usage)
            .unwrap_or(last_response_usage),
        plugin_mode_indicators,
        streaming_output,
        streaming_output_hidden,
        streaming_output_partial,
    })
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

    fn persist_root_snapshot(
        store: &Store,
        messages: Vec<Message>,
        tool_calls: Vec<ToolCallRecord>,
        ui_state: UiResumeState,
    ) {
        ui_resume::save_ui_resume_state(store, &ui_state);
        let graph = lash::SessionGraph::from_projection(&messages, &tool_calls);
        let checkpoint_ref = store
            .put_checkpoint(&lash::HydratedSessionCheckpoint {
                turn_state: lash::PersistedTurnState {
                    iteration: 1,
                    token_usage: ui_state.last_response_usage.clone(),
                    last_prompt_usage: None,
                },
                dynamic_state_ref: None,
                dynamic_state: Some(lash::DynamicStateSnapshot {
                    base_generation: 0,
                    tools: BTreeMap::new(),
                    enabled_tools: std::collections::BTreeSet::new(),
                }),
                plugin_snapshot_ref: None,
                plugin_snapshot_revision: None,
                plugin_snapshot: None,
            })
            .checkpoint_ref;
        store.save_session_head(lash::SessionHead {
            session_id: "root".to_string(),
            graph,
            config: lash::PersistedSessionConfig {
                provider_id: "openai_generic".to_string(),
                configured_model: "gpt-test".to_string(),
                context_window: 200_000,
                execution_mode: lash::ExecutionMode::Standard,
                context_approach: lash::ContextApproach::default(),
                model_variant: None,
            },
            checkpoint_ref: Some(checkpoint_ref),
            token_ledger: Vec::new(),
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
                tool_item_id: None,
                tool_signature: None,
                prune_state: lash::PruneState::Intact,
                reasoning_meta: None,
            }],
            user_input: None,
            origin: None,
        }
    }

    fn tool_result_message(id: &str, call_id: &str, tool_name: &str) -> Message {
        Message {
            id: id.to_string(),
            role: MessageRole::User,
            parts: vec![lash::Part {
                id: format!("{id}.p0"),
                kind: PartKind::ToolResult,
                content: String::new(),
                attachment: None,
                tool_call_id: Some(call_id.to_string()),
                tool_name: Some(tool_name.to_string()),
                tool_item_id: None,
                tool_signature: None,
                prune_state: lash::PruneState::Intact,
                reasoning_meta: None,
            }],
            user_input: None,
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
                Vec::new(),
                UiResumeState {
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
                    streaming_output: vec!["started git status --short".to_string()],
                    streaming_output_hidden: 2,
                    streaming_output_partial: "partial tool line".to_string(),
                    ..UiResumeState::default()
                },
            );

            let loaded = load_session(&filename).unwrap();
            assert_eq!(loaded.messages.len(), 2);
            // blocks[0] is the TurnStart marker the projection emits above
            // the first user input.
            assert!(matches!(
                loaded.blocks.first(),
                Some(DisplayBlock::TurnStart(_))
            ));
            assert!(matches!(
                loaded.blocks.get(1),
                Some(DisplayBlock::UserInput(text)) if text == "Hi"
            ));
            assert!(matches!(
                loaded.blocks.get(2),
                Some(DisplayBlock::AssistantText(text)) if text == "Hello world"
            ));
            assert_eq!(loaded.last_token_usage.input_tokens, 12);
            assert_eq!(
                loaded.plugin_mode_indicators.get("plan_mode"),
                Some(&"plan".to_string())
            );
            assert_eq!(
                loaded.streaming_output,
                vec!["started git status --short".to_string()]
            );
            assert_eq!(loaded.streaming_output_hidden, 2);
            assert_eq!(loaded.streaming_output_partial, "partial tool line");
        });
    }

    #[test]
    fn load_session_projects_activity_blocks_from_session_graph() {
        with_temp_lash_home("lash-session-load-activity-blocks", || {
            let filename = new_session_filename();
            let path = sessions_dir().join(&filename);
            let store = Arc::new(Store::open(&path).unwrap());
            SessionLogger::new(
                Arc::clone(&store),
                filename.clone(),
                "gpt-test",
                Some("s-activity".into()),
                "demo".into(),
            )
            .unwrap();
            let messages = vec![
                text_message(MessageRole::User, "m0", "inspect repo"),
                tool_result_message("m1", "call-shell", "exec_command"),
                text_message(MessageRole::Assistant, "m2", "Done"),
            ];
            let tool_calls = vec![ToolCallRecord {
                call_id: Some("call-shell".to_string()),
                tool: "exec_command".to_string(),
                args: serde_json::json!({"cmd":"git status --short"}),
                result: serde_json::json!({
                    "stdout":"",
                    "stderr":"",
                    "exit_code":0
                }),
                success: true,
                duration_ms: 42,
            }];
            persist_root_snapshot(&store, messages, tool_calls, UiResumeState::default());

            let loaded = load_session(&filename).unwrap();
            // blocks[0] = TurnStart, [1] = UserInput, [2] = Activity, [3] = AssistantText
            assert!(matches!(
                loaded.blocks.first(),
                Some(DisplayBlock::TurnStart(_))
            ));
            assert!(matches!(
                loaded.blocks.get(2),
                Some(DisplayBlock::Activity(activity))
                    if activity.call.summary == "git status --short"
            ));
            assert!(matches!(
                loaded.blocks.get(3),
                Some(DisplayBlock::AssistantText(text)) if text == "Done"
            ));
        });
    }

    #[test]
    fn load_session_rebuilds_blocks_from_messages_when_ui_snapshot_is_stale() {
        with_temp_lash_home("lash-session-load-rebuilds-blocks", || {
            let filename = new_session_filename();
            let path = sessions_dir().join(&filename);
            let store = Arc::new(Store::open(&path).unwrap());
            SessionLogger::new(
                Arc::clone(&store),
                filename.clone(),
                "gpt-test",
                Some("s2".into()),
                "demo".into(),
            )
            .unwrap();
            let assistant = "Line one\n\n1. first\n2. second\n";
            let messages = vec![
                text_message(MessageRole::User, "m0", "Hi"),
                text_message(MessageRole::Assistant, "m1", assistant),
            ];
            persist_root_snapshot(&store, messages, Vec::new(), UiResumeState::default());

            let loaded = load_session(&filename).unwrap();
            // blocks[0] = TurnStart, [1] = UserInput, [2] = AssistantText
            assert!(matches!(
                loaded.blocks.get(2),
                Some(DisplayBlock::AssistantText(text)) if text == assistant.trim()
            ));
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
                Vec::new(),
                UiResumeState::default(),
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
                Vec::new(),
                UiResumeState::default(),
            );

            let sessions = list_recent_sessions(10);
            assert_eq!(sessions.len(), 1);
            assert_eq!(sessions[0].filename, "parent.db");
            assert_eq!(sessions[0].message_count, 1);
            assert_eq!(sessions[0].first_message, "hello there");
        });
    }
}
