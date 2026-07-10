use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use lash_core::session_model::Message;
#[cfg(test)]
use lash_core::session_model::{MessageRole, PartKind};
use lash_core::{SessionSnapshot, TokenUsage};
use lash_sqlite_store::Store;

use crate::app::{
    LiveToolOutput, PreparedTurn, UiActivityJournal, UiActivityRecord, UiTimeline, UiTimelineItem,
    timeline_from_read_view,
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
    pub(crate) ui_activity_journal: UiActivityJournal,
}

pub struct SessionLogger {
    store: Arc<Store>,
    pub session_id: String,
    filename: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct HostUiSidecar {
    #[serde(default)]
    inputs: Vec<HostInputRecord>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HostInputRecord {
    display_text: String,
    effective_text: String,
}

impl SessionLogger {
    pub async fn new(
        store: Arc<Store>,
        filename: String,
        model: &str,
        session_id: Option<String>,
        session_name: String,
    ) -> Result<Self> {
        std::fs::create_dir_all(sessions_dir())?;
        let session_id = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        store
            .save_session_meta(lash_core::SessionMeta {
                session_id: session_id.clone(),
                session_name: session_name.clone(),
                created_at: chrono::Local::now().to_rfc3339(),
                model: model.to_string(),
                cwd: std::env::current_dir()
                    .ok()
                    .and_then(|path| path.to_str().map(str::to_string)),
                relation: lash_core::SessionRelation::Root,
            })
            .await;
        Ok(Self {
            store,
            session_id,
            filename,
        })
    }

    pub async fn resume(store: Arc<Store>, filename: &str) -> Result<Self> {
        let start = store
            .load_session_meta()
            .await
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

    fn ui_sidecar_path_for(filename: &str) -> PathBuf {
        sessions_dir().join(format!("{filename}.ui.json"))
    }

    fn ui_activity_log_path_for(filename: &str) -> PathBuf {
        sessions_dir().join(format!("{filename}.ui-activity.jsonl"))
    }

    fn ui_sidecar_path(&self) -> PathBuf {
        Self::ui_sidecar_path_for(&self.filename)
    }

    fn ui_activity_log_path(&self) -> PathBuf {
        Self::ui_activity_log_path_for(&self.filename)
    }

    pub fn record_host_input(&self, turn: &PreparedTurn) -> Result<()> {
        if turn.display_text.trim().is_empty() || turn.display_text == turn.effective_text {
            return Ok(());
        }
        let path = self.ui_sidecar_path();
        let mut sidecar = load_host_ui_sidecar_path(&path).unwrap_or_default();
        sidecar.inputs.push(HostInputRecord {
            display_text: turn.display_text.clone(),
            effective_text: turn.effective_text.clone(),
        });
        std::fs::write(path, serde_json::to_vec_pretty(&sidecar)?)?;
        Ok(())
    }

    pub(crate) fn append_ui_activity_records(&self, records: &[UiActivityRecord]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let path = self.ui_activity_log_path();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        for record in records {
            serde_json::to_writer(&mut file, record)?;
            file.write_all(b"\n")?;
        }
        Ok(())
    }

    pub async fn mark_as_child_of(&self, parent_session_id: &str) -> Result<()> {
        let meta = self
            .store
            .load_session_meta()
            .await
            .ok_or_else(|| anyhow::anyhow!("Missing session metadata for {}", self.filename))?;
        self.store
            .save_session_meta(lash_core::SessionMeta {
                session_id: meta.session_id,
                session_name: meta.session_name,
                created_at: meta.created_at,
                model: meta.model,
                cwd: meta.cwd,
                relation: lash_core::SessionRelation::Child {
                    parent_session_id: parent_session_id.to_string(),
                    caused_by: None,
                },
            })
            .await;
        Ok(())
    }
}

fn load_host_ui_sidecar_path(path: &Path) -> Result<HostUiSidecar> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(HostUiSidecar::default()),
        Err(err) => Err(err.into()),
    }
}

fn load_ui_activity_journal_path(path: &Path) -> Result<UiActivityJournal> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(UiActivityJournal::default());
        }
        Err(err) => return Err(err.into()),
    };
    let mut journal = UiActivityJournal::default();
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        journal.apply_record(serde_json::from_str::<UiActivityRecord>(&line)?);
    }
    Ok(journal)
}

fn apply_host_input_sidecar(blocks: &mut UiTimeline, sidecar: &HostUiSidecar) {
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
    if path.extension().and_then(|ext| ext.to_str()) != Some("db") {
        return false;
    }
    // Component sidecars live next to the session store as
    // `<session>.db.<component>.db` (effects, processes, triggers,
    // artifacts); anything with an interior `.db.` segment is a sidecar,
    // not a resumable session store.
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| !name.contains(".db."))
}

async fn parse_session_info(
    path: &Path,
    filename: String,
    modified: SystemTime,
) -> Option<SessionInfo> {
    let store = Store::open_readonly(path).await.ok()?;
    let info = store.load_picker_info().await?;
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

pub async fn load_session_start(filename: &str) -> Result<SessionStart> {
    let store = Store::open(&sessions_dir().join(filename)).await?;
    let meta = store
        .load_session_meta()
        .await
        .ok_or_else(|| anyhow::anyhow!("Could not load session metadata for {}", filename))?;
    Ok(SessionStart {
        session_id: meta.session_id,
        session_name: meta.session_name,
    })
}

async fn filename_for_session_meta(
    candidates: Vec<(PathBuf, String, SystemTime)>,
    mut matches: impl FnMut(&lash_core::SessionMeta) -> bool,
) -> Option<String> {
    for (path, filename, _) in candidates {
        // A candidate that fails to open or carries no session meta (an
        // unreadable file, or a store that has never committed) must not
        // abort the whole search — skip it and keep looking.
        let Ok(store) = Store::open_readonly(&path).await else {
            continue;
        };
        let Some(meta) = store.load_session_meta().await else {
            continue;
        };
        if matches(&meta) {
            return Some(filename);
        }
    }
    None
}

pub(crate) async fn filename_for_session_identifier(identifier: &str) -> Option<String> {
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
    .await
}

pub async fn list_recent_sessions(limit: usize) -> Vec<SessionInfo> {
    if limit == 0 {
        return Vec::new();
    }

    // Candidates are pre-sorted by modified time (latest first).
    let mut sessions = Vec::new();
    for (path, filename, modified) in collect_session_candidates() {
        if let Some(info) = parse_session_info(&path, filename, modified).await {
            sessions.push(info);
            if sessions.len() >= limit {
                break;
            }
        }
    }
    sessions.sort_by_key(|session| Reverse(session.modified));
    sessions
}

pub async fn load_session(filename: &str) -> Result<LoadedSession> {
    let store = Store::open(&sessions_dir().join(filename)).await?;
    let meta = store
        .load_session_meta()
        .await
        .ok_or_else(|| anyhow::anyhow!("Could not load session metadata for {}", filename))?;
    let head = store.load_session_head().await.unwrap_or_default();
    let checkpoint_ref = head.checkpoint_ref.clone();
    let state = SessionSnapshot {
        session_id: head.session_id,
        session_graph: head.graph,
        ..SessionSnapshot::default()
    };
    let read_view = state.read_view();
    let messages = read_view.messages().to_vec();
    let sidecar = load_host_ui_sidecar_path(&SessionLogger::ui_sidecar_path_for(filename))
        .unwrap_or_default();
    let activity_journal =
        load_ui_activity_journal_path(&SessionLogger::ui_activity_log_path_for(filename))
            .unwrap_or_default();
    let ui_state = crate::app::UiProjectionState {
        activity_journal,
        ..crate::app::UiProjectionState::default()
    };
    let checkpoint = checkpoint_ref
        .as_ref()
        .map(|blob_ref| async { store.get_checkpoint(blob_ref).await });
    let checkpoint = match checkpoint {
        Some(checkpoint) => checkpoint.await,
        None => None,
    };
    let plugin_mode_indicators = ui_state.plugin_mode_indicators.clone();
    let live_tool_output = ui_state.live_tool_output.clone();
    let mut blocks = timeline_from_read_view(&read_view, &ui_state);
    apply_host_input_sidecar(&mut blocks, &sidecar);
    tracing::debug!(
        session_file = filename,
        messages = read_view.messages().len(),
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
        ui_activity_journal: ui_state.activity_journal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activity::{ActivityBlock, ActivityKind, ActivityStatus};
    use crate::test_support::{EnvVarGuard, TempDirGuard, env_lock};

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(future)
    }

    fn with_temp_lash_home(test_name: &str, f: impl FnOnce()) {
        let _env_guard = env_lock().blocking_lock();
        let temp = TempDirGuard::new(test_name);
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        std::fs::create_dir_all(sessions_dir()).expect("sessions dir");
        f();
    }

    async fn persist_root_snapshot(store: &Store, messages: Vec<Message>, token_usage: TokenUsage) {
        let graph = lash_core::SessionGraph::from_active_read_state(&messages);
        let checkpoint_ref = store
            .put_checkpoint(&lash_core::store::HydratedSessionCheckpoint {
                turn_state: lash_core::PersistedTurnState {
                    turn_index: 1,
                    token_usage,
                    last_prompt_usage: None,
                    protocol_turn_options: Default::default(),
                },
                tool_state_ref: None,
                tool_state: Some(lash_core::ToolState::default()),
                plugin_snapshot_ref: None,
                plugin_snapshot_revision: None,
                plugin_snapshot: None,
                execution_state_ref: None,
                execution_state: None,
            })
            .await
            .checkpoint_ref;
        store
            .save_session_head(lash_core::store::SessionHead {
                session_id: "root".to_string(),
                head_revision: 0,
                agent_frames: Vec::new(),
                current_agent_frame_id: String::new(),
                graph,
                config: lash_core::PersistedSessionConfig {
                    provider_id: "openai_generic".to_string(),
                    model: lash_core::ModelSpec::from_token_limits(
                        "gpt-test",
                        Default::default(),
                        200_000,
                        None,
                    )
                    .expect("valid model spec"),
                },
                checkpoint_ref: Some(checkpoint_ref),
                token_ledger: Vec::new(),
            })
            .await;
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
            block_on(async {
                let filename = new_session_filename();
                let path = sessions_dir().join(&filename);
                let store = Arc::new(Store::open(&path).await.unwrap());
                SessionLogger::new(
                    Arc::clone(&store),
                    filename.clone(),
                    "gpt-test",
                    Some("s1".into()),
                    "demo".into(),
                )
                .await
                .unwrap();
                let messages = vec![
                    text_message(MessageRole::User, "m0", "Hi"),
                    text_message(MessageRole::Assistant, "m1", "Hello world"),
                ];
                persist_root_snapshot(
                    &store,
                    messages.clone(),
                    TokenUsage {
                        input_tokens: 12,
                        output_tokens: 7,
                        cache_read_input_tokens: 2,
                        cache_write_input_tokens: 0,
                        reasoning_output_tokens: 1,
                    },
                )
                .await;

                let loaded = load_session(&filename).await.unwrap();
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
        });
    }

    #[test]
    fn load_session_does_not_hydrate_activity_blocks_from_detached_tool_history() {
        with_temp_lash_home("lash-session-load-activity-blocks", || {
            block_on(async {
                let filename = new_session_filename();
                let path = sessions_dir().join(&filename);
                let store = Arc::new(Store::open(&path).await.unwrap());
                SessionLogger::new(
                    Arc::clone(&store),
                    filename.clone(),
                    "gpt-test",
                    Some("s-activity".into()),
                    "demo".into(),
                )
                .await
                .unwrap();
                let messages = vec![
                    text_message(MessageRole::User, "m0", "inspect repo"),
                    tool_result_message("m1", "call-shell", "exec_command"),
                    text_message(MessageRole::Assistant, "m2", "Done"),
                ];
                persist_root_snapshot(&store, messages, TokenUsage::default()).await;

                let loaded = load_session(&filename).await.unwrap();
                // blocks[0] = TurnStart, [1] = UserInput, [2] = AssistantText.
                // Detached tool invocation records are no longer persisted,
                // so a bare tool-result protocol message is not enough to
                // synthesize an activity block on resume.
                assert!(matches!(
                    loaded.blocks.first(),
                    Some(UiTimelineItem::TurnStart(_))
                ));
                assert!(matches!(
                    loaded.blocks.get(2),
                    Some(UiTimelineItem::AssistantText(text)) if text == "Done"
                ));
            });
        });
    }

    #[test]
    fn load_session_projects_cli_owned_ui_activity_journal() {
        with_temp_lash_home("lash-session-load-ui-activity-journal", || {
            block_on(async {
                let filename = new_session_filename();
                let path = sessions_dir().join(&filename);
                let store = Arc::new(Store::open(&path).await.unwrap());
                SessionLogger::new(
                    Arc::clone(&store),
                    filename.clone(),
                    "gpt-test",
                    Some("s-ui-activity".into()),
                    "demo".into(),
                )
                .await
                .unwrap();
                let messages = vec![
                    text_message(MessageRole::User, "m0", "inspect repo"),
                    text_message(MessageRole::Assistant, "m1", "Done"),
                ];
                persist_root_snapshot(&store, messages, TokenUsage::default()).await;

                let mut activity_journal = UiActivityJournal::default();
                let record = UiActivityRecord::new(
                    0,
                    0,
                    ActivityBlock::new(
                        ActivityKind::GenericTool,
                        "exec_command",
                        serde_json::json!({"cmd": "pwd"}),
                        "Run pwd",
                        ActivityStatus::Completed,
                        serde_json::json!({"exit_code": 0, "stdout": "/workspace/code/lash\n"}),
                        4,
                    )
                    .with_call_id(Some("lashlang:tool-0".to_string())),
                );
                activity_journal.apply_record(record.clone());
                let logger = SessionLogger::resume(Arc::clone(&store), &filename)
                    .await
                    .unwrap();
                logger.append_ui_activity_records(&[record]).unwrap();

                let loaded = load_session(&filename).await.unwrap();
                assert_eq!(loaded.ui_activity_journal, activity_journal);
                assert!(loaded.blocks.iter().any(|block| matches!(
                    block,
                    UiTimelineItem::Activity(activity)
                        if activity.call.call_id.as_deref() == Some("lashlang:tool-0")
                            && activity.result.status == ActivityStatus::Completed
                )));
            });
        });
    }

    #[test]
    fn ui_activity_log_does_not_persist_large_raw_tool_outputs() {
        with_temp_lash_home("lash-session-ui-activity-omits-raw", || {
            block_on(async {
                let filename = new_session_filename();
                let path = sessions_dir().join(&filename);
                let store = Arc::new(Store::open(&path).await.unwrap());
                let logger = SessionLogger::new(
                    Arc::clone(&store),
                    filename.clone(),
                    "gpt-test",
                    Some("s-ui-raw".into()),
                    "demo".into(),
                )
                .await
                .unwrap();
                let large_raw = "x".repeat(128 * 1024);
                let record = UiActivityRecord::new(
                    0,
                    0,
                    ActivityBlock::new(
                        ActivityKind::GenericTool,
                        "exec_command",
                        serde_json::json!({"cmd": large_raw}),
                        "Run command",
                        ActivityStatus::Completed,
                        serde_json::json!({"stdout": large_raw, "exit_code": 0}),
                        12,
                    )
                    .with_call_id(Some("lashlang:tool-raw".to_string())),
                );

                logger.append_ui_activity_records(&[record]).unwrap();

                let sidecar =
                    std::fs::read_to_string(SessionLogger::ui_activity_log_path_for(&filename))
                        .unwrap();
                assert!(!sidecar.contains("\"raw\""));
                assert!(!sidecar.contains("\"args\""));
                assert!(sidecar.len() < 4096, "sidecar was {} bytes", sidecar.len());
            });
        });
    }

    #[test]
    fn ui_activity_log_replays_started_and_completed_as_one_completed_row() {
        with_temp_lash_home("lash-session-ui-activity-replace-running", || {
            block_on(async {
                let filename = new_session_filename();
                let path = sessions_dir().join(&filename);
                let store = Arc::new(Store::open(&path).await.unwrap());
                SessionLogger::new(
                    Arc::clone(&store),
                    filename.clone(),
                    "gpt-test",
                    Some("s-ui-replay".into()),
                    "demo".into(),
                )
                .await
                .unwrap();
                let messages = vec![
                    text_message(MessageRole::User, "m0", "inspect repo"),
                    text_message(MessageRole::Assistant, "m1", "Done"),
                ];
                persist_root_snapshot(&store, messages, TokenUsage::default()).await;
                let logger = SessionLogger::resume(Arc::clone(&store), &filename)
                    .await
                    .unwrap();
                let started = UiActivityRecord::new(
                    0,
                    0,
                    ActivityBlock::new(
                        ActivityKind::ShellCommand,
                        "exec_command",
                        serde_json::json!({"cmd": "pwd"}),
                        "Run pwd",
                        ActivityStatus::Running,
                        serde_json::Value::Null,
                        0,
                    )
                    .with_call_id(Some("lashlang:tool-0".to_string())),
                );
                let completed = UiActivityRecord::new(
                    0,
                    0,
                    ActivityBlock::new(
                        ActivityKind::ShellCommand,
                        "exec_command",
                        serde_json::json!({"cmd": "pwd"}),
                        "Run pwd",
                        ActivityStatus::Completed,
                        serde_json::json!({"stdout": "/workspace/code/lash\n", "exit_code": 0}),
                        4,
                    )
                    .with_call_id(Some("lashlang:tool-0".to_string())),
                );
                logger
                    .append_ui_activity_records(&[started, completed])
                    .unwrap();

                let loaded = load_session(&filename).await.unwrap();
                let activities = loaded
                    .ui_activity_journal
                    .entries()
                    .iter()
                    .flat_map(|entry| entry.items.iter())
                    .filter_map(|item| match item {
                        UiTimelineItem::Activity(activity) => Some(activity),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(activities.len(), 1);
                assert_eq!(
                    activities[0].call.call_id.as_deref(),
                    Some("lashlang:tool-0")
                );
                assert_eq!(activities[0].result.status, ActivityStatus::Completed);
            });
        });
    }

    #[test]
    fn ui_activity_log_appends_one_record_per_tool_event() {
        with_temp_lash_home("lash-session-ui-activity-appends", || {
            block_on(async {
                let filename = new_session_filename();
                let path = sessions_dir().join(&filename);
                let store = Arc::new(Store::open(&path).await.unwrap());
                let logger = SessionLogger::new(
                    Arc::clone(&store),
                    filename.clone(),
                    "gpt-test",
                    Some("s-ui-append".into()),
                    "demo".into(),
                )
                .await
                .unwrap();
                let first = UiActivityRecord::new(
                    0,
                    0,
                    ActivityBlock::new(
                        ActivityKind::GenericTool,
                        "exec_command",
                        serde_json::json!({"cmd": "pwd"}),
                        "Run pwd",
                        ActivityStatus::Running,
                        serde_json::Value::Null,
                        0,
                    )
                    .with_call_id(Some("lashlang:tool-0".to_string())),
                );
                logger.append_ui_activity_records(&[first]).unwrap();
                let first_write =
                    std::fs::read_to_string(SessionLogger::ui_activity_log_path_for(&filename))
                        .unwrap();
                assert_eq!(first_write.lines().count(), 1);

                let second = UiActivityRecord::new(
                    0,
                    0,
                    ActivityBlock::new(
                        ActivityKind::GenericTool,
                        "exec_command",
                        serde_json::json!({"cmd": "pwd"}),
                        "Run pwd",
                        ActivityStatus::Completed,
                        serde_json::json!({"stdout": "/workspace/code/lash\n", "exit_code": 0}),
                        3,
                    )
                    .with_call_id(Some("lashlang:tool-0".to_string())),
                );
                logger.append_ui_activity_records(&[second]).unwrap();
                let second_write =
                    std::fs::read_to_string(SessionLogger::ui_activity_log_path_for(&filename))
                        .unwrap();
                assert_eq!(second_write.lines().count(), 2);
                assert!(second_write.starts_with(&first_write));
            });
        });
    }

    #[test]
    fn load_session_rebuilds_blocks_from_messages_when_ui_snapshot_is_stale() {
        with_temp_lash_home("lash-session-load-rebuilds-blocks", || {
            block_on(async {
                let filename = new_session_filename();
                let path = sessions_dir().join(&filename);
                let store = Arc::new(Store::open(&path).await.unwrap());
                SessionLogger::new(
                    Arc::clone(&store),
                    filename.clone(),
                    "gpt-test",
                    Some("s2".into()),
                    "demo".into(),
                )
                .await
                .unwrap();
                let assistant = "Line one\n\n1. first\n2. second\n";
                let messages = vec![
                    text_message(MessageRole::User, "m0", "Hi"),
                    text_message(MessageRole::Assistant, "m1", assistant),
                ];
                persist_root_snapshot(&store, messages, TokenUsage::default()).await;

                let loaded = load_session(&filename).await.unwrap();
                // blocks[0] = TurnStart, [1] = UserInput, [2] = AssistantText
                assert!(matches!(
                    loaded.blocks.get(2),
                    Some(UiTimelineItem::AssistantText(text)) if text == assistant.trim()
                ));
            });
        });
    }

    #[test]
    fn list_recent_sessions_ignores_child_sessions() {
        with_temp_lash_home("lash-session-log-list", || {
            block_on(async {
                let parent = sessions_dir().join("parent.db");
                let child = sessions_dir().join("child.db");
                let parent_store = Arc::new(Store::open(&parent).await.unwrap());
                let child_store = Store::open(&child).await.unwrap();
                SessionLogger::new(
                    Arc::clone(&parent_store),
                    "parent.db".into(),
                    "gpt-test",
                    Some("parent".into()),
                    "demo".into(),
                )
                .await
                .unwrap();
                persist_root_snapshot(
                    &parent_store,
                    vec![text_message(MessageRole::User, "m0", "hello there")],
                    TokenUsage::default(),
                )
                .await;
                child_store
                    .save_session_meta(lash_core::SessionMeta {
                        session_id: "child".to_string(),
                        session_name: "demo".to_string(),
                        created_at: "2026-03-25T10:00:00Z".to_string(),
                        model: "gpt-test".to_string(),
                        cwd: std::env::current_dir()
                            .ok()
                            .and_then(|path| path.to_str().map(str::to_string)),
                        relation: lash_core::SessionRelation::Child {
                            parent_session_id: "parent".to_string(),
                            caused_by: None,
                        },
                    })
                    .await;
                persist_root_snapshot(
                    &child_store,
                    vec![text_message(MessageRole::User, "m0", "child prompt")],
                    TokenUsage::default(),
                )
                .await;

                let sessions = list_recent_sessions(10).await;
                assert_eq!(sessions.len(), 1);
                assert_eq!(sessions[0].filename, "parent.db");
                assert_eq!(sessions[0].message_count, 1);
                assert_eq!(sessions[0].first_message, "hello there");
            });
        });
    }

    #[test]
    fn session_identifier_resolves_filename_id_and_name_including_children() {
        with_temp_lash_home("lash-session-identifier", || {
            block_on(async {
                let parent = sessions_dir().join("parent.db");
                let child = sessions_dir().join("child.db");
                let parent_store = Arc::new(Store::open(&parent).await.unwrap());
                let child_store = Store::open(&child).await.unwrap();
                SessionLogger::new(
                    Arc::clone(&parent_store),
                    "parent.db".into(),
                    "gpt-test",
                    Some("parent-id".into()),
                    "parent-name".into(),
                )
                .await
                .unwrap();
                child_store
                    .save_session_meta(lash_core::SessionMeta {
                        session_id: "child-id".to_string(),
                        session_name: "child-name".to_string(),
                        created_at: "2026-03-25T10:00:00Z".to_string(),
                        model: "gpt-test".to_string(),
                        cwd: None,
                        relation: lash_core::SessionRelation::Child {
                            parent_session_id: "parent-id".to_string(),
                            caused_by: None,
                        },
                    })
                    .await;

                assert_eq!(
                    filename_for_session_identifier("parent.db")
                        .await
                        .as_deref(),
                    Some("parent.db")
                );
                assert_eq!(
                    filename_for_session_identifier("parent-id")
                        .await
                        .as_deref(),
                    Some("parent.db")
                );
                assert_eq!(
                    filename_for_session_identifier("parent-name")
                        .await
                        .as_deref(),
                    Some("parent.db")
                );
                assert_eq!(
                    filename_for_session_identifier("child-id").await.as_deref(),
                    Some("child.db")
                );
                assert_eq!(
                    filename_for_session_identifier("child-name")
                        .await
                        .as_deref(),
                    Some("child.db")
                );
            });
        });
    }

    #[test]
    fn session_identifier_resolution_survives_sidecar_pollution() {
        with_temp_lash_home("lash-session-sidecars", || {
            block_on(async {
                let parent = sessions_dir().join("parent.db");
                let parent_store = Arc::new(Store::open(&parent).await.unwrap());
                SessionLogger::new(
                    Arc::clone(&parent_store),
                    "parent.db".into(),
                    "gpt-test",
                    Some("parent-id".into()),
                    "parent-name".into(),
                )
                .await
                .unwrap();

                // Component sidecars share the `.db` extension and, being
                // written after the session store, sort first in the
                // modified-time candidate order. An unreadable sidecar must
                // not abort name/id resolution, and no sidecar may appear as
                // a resumable session.
                std::fs::write(sessions_dir().join("parent.db.effects.db"), b"not sqlite").unwrap();
                std::fs::write(sessions_dir().join("parent.db.processes.db"), b"junk").unwrap();

                assert_eq!(
                    filename_for_session_identifier("parent-name")
                        .await
                        .as_deref(),
                    Some("parent.db")
                );
                assert_eq!(
                    filename_for_session_identifier("parent-id")
                        .await
                        .as_deref(),
                    Some("parent.db")
                );
                let sessions = list_recent_sessions(10).await;
                assert_eq!(sessions.len(), 1);
                assert_eq!(sessions[0].filename, "parent.db");
            });
        });
    }
}
