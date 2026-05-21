use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
#[cfg(test)]
use lash_core::ToolCallRecord;
use lash_core::session_model::Message;
#[cfg(test)]
use lash_core::session_model::{MessageRole, PartKind};
use lash_core::{SessionStateEnvelope, TokenUsage};
use lash_sqlite_store::Store;

use crate::app::{
    LiveToolOutput, PreparedTurn, UiTimeline, UiTimelineItem, timeline_from_read_view,
};

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
    pub session_id: String,
    pub session_name: String,
    pub filename: String,
    pub messages: Vec<Message>,
    pub blocks: UiTimeline,
    pub last_token_usage: TokenUsage,
    pub plugin_mode_indicators: BTreeMap<String, String>,
    pub live_tool_output: LiveToolOutput,
}

pub struct SessionLogger {
    store: Arc<Store>,
    pub session_id: String,
    filename: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct HostInputSidecar {
    inputs: Vec<HostInputRecord>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HostInputRecord {
    display_text: String,
    effective_text: String,
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
        store.save_session_meta(lash_core::SessionMeta {
            session_id: session_id.clone(),
            session_name: session_name.clone(),
            created_at: chrono::Local::now().to_rfc3339(),
            model: model.to_string(),
            cwd: std::env::current_dir()
                .ok()
                .and_then(|path| path.to_str().map(str::to_string)),
            relation: lash_core::SessionRelation::Root,
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

    fn host_input_path_for(filename: &str) -> PathBuf {
        sessions_dir().join(format!("{filename}.ui.json"))
    }

    fn host_input_path(&self) -> PathBuf {
        Self::host_input_path_for(&self.filename)
    }

    pub fn record_host_input(&self, turn: &PreparedTurn) -> Result<()> {
        if turn.display_text.trim().is_empty() || turn.display_text == turn.effective_text {
            return Ok(());
        }
        let path = self.host_input_path();
        let mut sidecar = load_host_input_sidecar_path(&path).unwrap_or_default();
        sidecar.inputs.push(HostInputRecord {
            display_text: turn.display_text.clone(),
            effective_text: turn.effective_text.clone(),
        });
        std::fs::write(path, serde_json::to_vec_pretty(&sidecar)?)?;
        Ok(())
    }

    pub fn mark_as_child_of(&self, parent_session_id: &str) -> Result<()> {
        let meta = self
            .store
            .load_session_meta()
            .ok_or_else(|| anyhow::anyhow!("Missing session metadata for {}", self.filename))?;
        self.store.save_session_meta(lash_core::SessionMeta {
            session_id: meta.session_id,
            session_name: meta.session_name,
            created_at: meta.created_at,
            model: meta.model,
            cwd: meta.cwd,
            relation: lash_core::SessionRelation::Child {
                parent_session_id: parent_session_id.to_string(),
                originating_tool_call_id: None,
            },
        });
        Ok(())
    }
}

fn load_host_input_sidecar_path(path: &Path) -> Result<HostInputSidecar> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(HostInputSidecar::default()),
        Err(err) => Err(err.into()),
    }
}

fn apply_host_input_sidecar(blocks: &mut UiTimeline, sidecar: &HostInputSidecar) {
    for item in blocks.iter_mut() {
        let UiTimelineItem::UserInput(text) = item else {
            continue;
        };
        if let Some(record) = sidecar
            .inputs
            .iter()
            .find(|record| record.effective_text.trim() == text.trim())
        {
            *text = record.display_text.trim().to_string();
        }
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
    crate::paths::lash_home().join("sessions")
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
    if info.parent_session_id().is_some() {
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
    candidates.sort_by_key(|candidate| Reverse(candidate.2));
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

fn filename_for_session_meta(
    candidates: Vec<(PathBuf, String, SystemTime)>,
    mut matches: impl FnMut(&lash_core::SessionMeta) -> bool,
) -> Option<String> {
    candidates.into_iter().find_map(|(path, filename, _)| {
        let store = Store::open_readonly(&path).ok()?;
        let meta = store.load_session_meta()?;
        matches(&meta).then_some(filename)
    })
}

pub(crate) fn filename_for_session_identifier(identifier: &str) -> Option<String> {
    let identifier = identifier.trim();
    if identifier.is_empty() {
        return None;
    }

    let candidates = collect_session_candidates();
    if candidates
        .iter()
        .any(|(_, filename, _)| filename == identifier)
    {
        return Some(identifier.to_string());
    }

    filename_for_session_meta(candidates, |meta| {
        meta.session_id == identifier || meta.session_name == identifier
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
    sessions.sort_by_key(|session| Reverse(session.modified));
    sessions
}

pub fn load_session(filename: &str) -> Result<LoadedSession> {
    let store = Store::open(&sessions_dir().join(filename))?;
    let meta = store
        .load_session_meta()
        .ok_or_else(|| anyhow::anyhow!("Could not load session metadata for {}", filename))?;
    let head = store.load_session_head().unwrap_or_default();
    let checkpoint_ref = head.checkpoint_ref.clone();
    let state = SessionStateEnvelope {
        session_id: head.session_id,
        session_graph: head.graph,
        ..SessionStateEnvelope::default()
    };
    let read_view = state.read_view();
    let messages = read_view.messages().to_vec();
    let ui_state = crate::app::UiProjectionState::default();
    let checkpoint = checkpoint_ref
        .as_ref()
        .and_then(|blob_ref| store.get_checkpoint(blob_ref));
    let plugin_mode_indicators = ui_state.plugin_mode_indicators.clone();
    let live_tool_output = ui_state.live_tool_output.clone();
    let mut blocks = timeline_from_read_view(&read_view, &ui_state);
    if let Ok(sidecar) = load_host_input_sidecar_path(&SessionLogger::host_input_path_for(filename))
    {
        apply_host_input_sidecar(&mut blocks, &sidecar);
    }
    tracing::debug!(
        session_file = filename,
        messages = read_view.messages().len(),
        tool_calls = read_view.tool_calls().len(),
        blocks = blocks.len(),
        plugin_mode_indicators = plugin_mode_indicators.len(),
        graph_nodes = read_view.materialized_session_graph().nodes.len(),
        "loaded persisted session snapshot"
    );

    Ok(LoadedSession {
        session_id: meta.session_id,
        session_name: meta.session_name,
        filename: filename.to_string(),
        messages,
        blocks,
        last_token_usage: checkpoint
            .map(|checkpoint| checkpoint.turn_state.token_usage)
            .unwrap_or_default(),
        plugin_mode_indicators,
        live_tool_output,
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
        token_usage: TokenUsage,
    ) {
        let graph = lash_core::SessionGraph::from_active_read_state(&messages, &tool_calls);
        let checkpoint_ref = store
            .put_checkpoint(&lash_core::HydratedSessionCheckpoint {
                turn_state: lash_core::PersistedTurnState {
                    turn_index: 1,
                    token_usage,
                    last_prompt_usage: None,
                    mode_turn_options: Default::default(),
                },
                tool_state_ref: None,
                tool_state: Some(lash_core::ToolState::default()),
                plugin_snapshot_ref: None,
                plugin_snapshot_revision: None,
                plugin_snapshot: None,
                execution_state_ref: None,
                execution_state: None,
            })
            .checkpoint_ref;
        store.save_session_head(lash_core::SessionHead {
            session_id: "root".to_string(),
            head_revision: 0,
            graph,
            config: lash_core::PersistedSessionConfig {
                provider_id: "openai_generic".to_string(),
                configured_model: "gpt-test".to_string(),
                context_window: 200_000,
                execution_mode: lash_core::ExecutionMode::standard(),
                standard_context_approach: Some(lash_core::StandardContextApproach::default()),
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
            parts: vec![lash_core::Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: lash_core::PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: None,
        }
    }

    fn tool_result_message(id: &str, call_id: &str, tool_name: &str) -> Message {
        Message {
            id: id.to_string(),
            role: MessageRole::User,
            parts: vec![lash_core::Part {
                id: format!("{id}.p0"),
                kind: PartKind::ToolResult,
                content: String::new(),
                attachment: None,
                tool_call_id: Some(call_id.to_string()),
                tool_name: Some(tool_name.to_string()),
                tool_replay: None,
                prune_state: lash_core::PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
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
                TokenUsage {
                    input_tokens: 12,
                    output_tokens: 7,
                    cached_input_tokens: 2,
                    reasoning_tokens: 1,
                },
            );

            let loaded = load_session(&filename).unwrap();
            assert_eq!(loaded.messages.len(), 2);
            // blocks[0] is the TurnStart marker the projection emits above
            // the first user input.
            assert!(matches!(
                loaded.blocks.first(),
                Some(UiTimelineItem::TurnStart(_))
            ));
            assert!(matches!(
                loaded.blocks.get(1),
                Some(UiTimelineItem::UserInput(text)) if text == "Hi"
            ));
            assert!(matches!(
                loaded.blocks.get(2),
                Some(UiTimelineItem::AssistantText(text)) if text == "Hello world"
            ));
            assert_eq!(loaded.last_token_usage.input_tokens, 12);
            assert!(loaded.plugin_mode_indicators.is_empty());
            assert!(loaded.live_tool_output.lines.is_empty());
            assert_eq!(loaded.live_tool_output.hidden, 0);
            assert!(loaded.live_tool_output.partial.is_empty());
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
                output: lash_core::ToolCallOutput::success(serde_json::json!({
                    "stdout":"",
                    "stderr":"",
                    "exit_code":0
                })),
                duration_ms: 42,
            }];
            persist_root_snapshot(&store, messages, tool_calls, TokenUsage::default());

            let loaded = load_session(&filename).unwrap();
            // blocks[0] = TurnStart, [1] = UserInput, [2] = Activity, [3] = AssistantText
            assert!(matches!(
                loaded.blocks.first(),
                Some(UiTimelineItem::TurnStart(_))
            ));
            assert!(matches!(
                loaded.blocks.get(2),
                Some(UiTimelineItem::Activity(activity))
                    if activity.call.summary == "git status --short"
            ));
            assert!(matches!(
                loaded.blocks.get(3),
                Some(UiTimelineItem::AssistantText(text)) if text == "Done"
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
            persist_root_snapshot(&store, messages, Vec::new(), TokenUsage::default());

            let loaded = load_session(&filename).unwrap();
            // blocks[0] = TurnStart, [1] = UserInput, [2] = AssistantText
            assert!(matches!(
                loaded.blocks.get(2),
                Some(UiTimelineItem::AssistantText(text)) if text == assistant.trim()
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
                TokenUsage::default(),
            );
            child_store.save_session_meta(lash_core::SessionMeta {
                session_id: "child".to_string(),
                session_name: "demo".to_string(),
                created_at: "2026-03-25T10:00:00Z".to_string(),
                model: "gpt-test".to_string(),
                cwd: std::env::current_dir()
                    .ok()
                    .and_then(|path| path.to_str().map(str::to_string)),
                relation: lash_core::SessionRelation::Child {
                    parent_session_id: "parent".to_string(),
                    originating_tool_call_id: None,
                },
            });
            persist_root_snapshot(
                &child_store,
                vec![text_message(MessageRole::User, "m0", "child prompt")],
                Vec::new(),
                TokenUsage::default(),
            );

            let sessions = list_recent_sessions(10);
            assert_eq!(sessions.len(), 1);
            assert_eq!(sessions[0].filename, "parent.db");
            assert_eq!(sessions[0].message_count, 1);
            assert_eq!(sessions[0].first_message, "hello there");
        });
    }

    #[test]
    fn session_identifier_resolves_filename_id_and_name_including_children() {
        with_temp_lash_home("lash-session-identifier", || {
            let parent = sessions_dir().join("parent.db");
            let child = sessions_dir().join("child.db");
            let parent_store = Arc::new(Store::open(&parent).unwrap());
            let child_store = Store::open(&child).unwrap();
            SessionLogger::new(
                Arc::clone(&parent_store),
                "parent.db".into(),
                "gpt-test",
                Some("parent-id".into()),
                "parent-name".into(),
            )
            .unwrap();
            child_store.save_session_meta(lash_core::SessionMeta {
                session_id: "child-id".to_string(),
                session_name: "child-name".to_string(),
                created_at: "2026-03-25T10:00:00Z".to_string(),
                model: "gpt-test".to_string(),
                cwd: None,
                relation: lash_core::SessionRelation::Child {
                    parent_session_id: "parent-id".to_string(),
                    originating_tool_call_id: None,
                },
            });

            assert_eq!(
                filename_for_session_identifier("parent.db").as_deref(),
                Some("parent.db")
            );
            assert_eq!(
                filename_for_session_identifier("parent-id").as_deref(),
                Some("parent.db")
            );
            assert_eq!(
                filename_for_session_identifier("parent-name").as_deref(),
                Some("parent.db")
            );
            assert_eq!(
                filename_for_session_identifier("child-id").as_deref(),
                Some("child.db")
            );
            assert_eq!(
                filename_for_session_identifier("child-name").as_deref(),
                Some("child.db")
            );
        });
    }
}
