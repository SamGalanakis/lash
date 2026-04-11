use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use lash::{PluginSurfaceEvent, PromptRequest, SessionEvent};
use lash_tui::ScreenSnapshot;
use serde::{Deserialize, Serialize};

use crate::app::{App, PreparedTurn};
use crate::repo_status::RepoStatus;
use crate::{render, scratch_tui};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UiTraceFixture {
    pub width: u16,
    pub height: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<UiTraceContext>,
    pub ops: Vec<UiTraceOp>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UiTraceContext {
    pub model: String,
    pub session_name: String,
    pub cwd: String,
    pub repo_status: Option<TraceRepoStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TraceRepoStatus {
    pub repo_name: String,
    pub branch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worktree: Option<String>,
}

impl TraceRepoStatus {
    pub(crate) fn from_repo_status(status: &RepoStatus) -> Self {
        Self {
            repo_name: status.repo_name.clone(),
            branch: status.branch.clone(),
            worktree: status.worktree.clone(),
        }
    }

    #[cfg(test)]
    pub(crate) fn into_repo_status(self, cwd: &str) -> RepoStatus {
        RepoStatus {
            repo_root: PathBuf::from(cwd),
            repo_name: self.repo_name,
            branch: self.branch,
            worktree: self.worktree,
        }
    }
}

impl UiTraceContext {
    pub(crate) fn from_app(app: &App) -> Self {
        Self {
            model: app.model.clone(),
            session_name: app.session_name.clone(),
            cwd: app.cwd.clone(),
            repo_status: app
                .repo_status
                .as_ref()
                .map(TraceRepoStatus::from_repo_status),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum UiTraceOp {
    StartTurn,
    UserTurn { text: String },
    QueueTurn { text: String },
    QueuePendingSteer { text: String },
    SlashCommand { text: String },
    SystemMessage { text: String },
    InputInsertText { text: String },
    InputBackspace,
    InputDelete,
    MoveCursorLeft,
    MoveCursorRight,
    MoveCursorHome,
    MoveCursorEnd,
    HistoryUp,
    HistoryDown,
    SuggestionUp,
    SuggestionDown,
    SuggestionComplete,
    EmitPrompt { request: TracePromptRequest },
    PromptUp,
    PromptDown,
    PromptToggleCurrentOption,
    PromptToggleNoteFocus,
    PromptInsertText { text: String },
    PromptBackspace,
    PromptDismiss,
    SubmitPrompt,
    ScrollUp { amount: usize },
    ScrollDown { amount: usize },
    Event { event: TraceSessionEvent },
    Render { snapshot: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TracePromptPanel {
    pub title: String,
    pub markdown: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum TracePromptRequest {
    Freeform {
        question: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        wait_seconds: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        panel: Option<TracePromptPanel>,
    },
    Single {
        question: String,
        options: Vec<String>,
        #[serde(default)]
        allow_note: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        wait_seconds: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        panel: Option<TracePromptPanel>,
    },
    Multi {
        question: String,
        options: Vec<String>,
        #[serde(default)]
        allow_note: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        wait_seconds: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        panel: Option<TracePromptPanel>,
    },
}

impl TracePromptRequest {
    pub(crate) fn from_request(request: &PromptRequest) -> Self {
        if request.is_freeform() {
            Self::Freeform {
                question: request.question.clone(),
                wait_seconds: request.wait.as_ref().map(|wait| wait.seconds),
                panel: request.panel.as_ref().map(|panel| TracePromptPanel {
                    title: panel.title.clone(),
                    markdown: panel.markdown.clone(),
                }),
            }
        } else {
            match request.selection_mode {
                lash::PromptSelectionMode::Single => Self::Single {
                    question: request.question.clone(),
                    options: request.options.clone(),
                    allow_note: request.allows_note(),
                    wait_seconds: request.wait.as_ref().map(|wait| wait.seconds),
                    panel: request.panel.as_ref().map(|panel| TracePromptPanel {
                        title: panel.title.clone(),
                        markdown: panel.markdown.clone(),
                    }),
                },
                lash::PromptSelectionMode::Multi => Self::Multi {
                    question: request.question.clone(),
                    options: request.options.clone(),
                    allow_note: request.allows_note(),
                    wait_seconds: request.wait.as_ref().map(|wait| wait.seconds),
                    panel: request.panel.as_ref().map(|panel| TracePromptPanel {
                        title: panel.title.clone(),
                        markdown: panel.markdown.clone(),
                    }),
                },
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TraceSessionEvent {
    TextDelta {
        content: String,
    },
    ToolCall {
        #[serde(default)]
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
        result: serde_json::Value,
        success: bool,
        duration_ms: u64,
    },
    Message {
        text: String,
        kind: String,
    },
    LlmRequest {
        iteration: usize,
        message_count: usize,
        tool_list: String,
    },
    Error {
        message: String,
    },
    Done,
    PluginEvent {
        plugin_id: String,
        event: TracePluginSurfaceEvent,
    },
}

impl TraceSessionEvent {
    pub(crate) fn from_session_event(event: &SessionEvent) -> Option<Self> {
        match event {
            SessionEvent::TextDelta { content } => Some(Self::TextDelta {
                content: content.clone(),
            }),
            SessionEvent::ToolCall {
                call_id,
                name,
                args,
                result,
                success,
                duration_ms,
            } => Some(Self::ToolCall {
                call_id: call_id.clone(),
                name: name.clone(),
                args: args.clone(),
                result: result.clone(),
                success: *success,
                duration_ms: *duration_ms,
            }),
            SessionEvent::Message { text, kind } => Some(Self::Message {
                text: text.clone(),
                kind: kind.clone(),
            }),
            SessionEvent::LlmRequest {
                iteration,
                message_count,
                tool_list,
            } => Some(Self::LlmRequest {
                iteration: *iteration,
                message_count: *message_count,
                tool_list: tool_list.clone(),
            }),
            SessionEvent::Error { message, .. } => Some(Self::Error {
                message: message.clone(),
            }),
            SessionEvent::Done => Some(Self::Done),
            SessionEvent::PluginEvent { plugin_id, event } => Some(Self::PluginEvent {
                plugin_id: plugin_id.clone(),
                event: TracePluginSurfaceEvent::from_event(event),
            }),
            SessionEvent::Prompt { .. }
            | SessionEvent::DurableSnapshot { .. }
            | SessionEvent::LlmResponse { .. }
            | SessionEvent::TokenUsage { .. }
            | SessionEvent::RetryStatus { .. }
            | SessionEvent::InjectedMessagesCommitted { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TracePluginSurfaceEvent {
    ModeIndicatorUpsert {
        key: String,
        label: String,
    },
    ModeIndicatorClear {
        key: String,
    },
    PanelUpsert {
        key: String,
        title: String,
        content: String,
    },
    PanelAppend {
        key: String,
        content: String,
    },
    PanelClear {
        key: String,
    },
}

impl TracePluginSurfaceEvent {
    pub(crate) fn from_event(event: &PluginSurfaceEvent) -> Self {
        match event {
            PluginSurfaceEvent::ModeIndicatorUpsert { key, label } => Self::ModeIndicatorUpsert {
                key: key.clone(),
                label: label.clone(),
            },
            PluginSurfaceEvent::ModeIndicatorClear { key } => {
                Self::ModeIndicatorClear { key: key.clone() }
            }
            PluginSurfaceEvent::PanelUpsert {
                key,
                title,
                content,
            } => Self::PanelUpsert {
                key: key.clone(),
                title: title.clone(),
                content: content.clone(),
            },
            PluginSurfaceEvent::PanelAppend { key, content } => Self::PanelAppend {
                key: key.clone(),
                content: content.clone(),
            },
            PluginSurfaceEvent::PanelClear { key } => Self::PanelClear { key: key.clone() },
            PluginSurfaceEvent::Custom { name, payload } => Self::PanelAppend {
                key: format!("custom:{name}"),
                content: payload.to_string(),
            },
        }
    }
}

pub(crate) fn render_screen_snapshot(app: &mut App, width: u16, height: u16) -> ScreenSnapshot {
    let viewport_height = render::history_viewport_height(app, width, height);
    let viewport_width = width as usize;
    app.ensure_height_cache_pub(viewport_width, viewport_height);
    app.refresh_follow_output_anchor(viewport_width, viewport_height);
    let total = app.total_content_height(viewport_width, viewport_height);
    let max_scroll = total.saturating_sub(viewport_height);
    app.scroll_offset = app.scroll_offset.min(max_scroll);
    app.history_area = render::history_area(app, width, height);
    lash_tui::render_snapshot(width, height, |frame| {
        scratch_tui::draw(frame, app);
    })
}

pub(crate) fn render_screen_text(app: &mut App, width: u16, height: u16) -> String {
    render_screen_snapshot(app, width, height)
        .visible_lines_trimmed()
        .join("\n")
}

pub(crate) struct UiTraceRecorder {
    fixture: UiTraceFixture,
    trace_path: PathBuf,
    snapshot_name: String,
    checkpoint_interval: Option<Duration>,
    last_checkpoint_at: Instant,
    checkpoint_index: usize,
    snapshots: Vec<(String, String)>,
}

impl UiTraceRecorder {
    pub(crate) fn new(
        path: impl AsRef<Path>,
        width: u16,
        height: u16,
        checkpoint_interval: Option<Duration>,
    ) -> Self {
        let trace_path = normalize_trace_path(path.as_ref());
        let snapshot_name = trace_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(|stem| format!("{stem}.snap"))
            .unwrap_or_else(|| "ui_trace.snap".to_string());
        Self {
            fixture: UiTraceFixture {
                width,
                height,
                context: None,
                ops: Vec::new(),
            },
            trace_path,
            snapshot_name,
            checkpoint_interval,
            last_checkpoint_at: Instant::now(),
            checkpoint_index: 0,
            snapshots: Vec::new(),
        }
    }

    pub(crate) fn set_size(&mut self, width: u16, height: u16) {
        self.fixture.width = width;
        self.fixture.height = height;
    }

    pub(crate) fn capture_app_context(&mut self, app: &App) {
        self.fixture.context = Some(UiTraceContext::from_app(app));
    }

    pub(crate) fn record_start_turn(&mut self) {
        self.fixture.ops.push(UiTraceOp::StartTurn);
    }

    pub(crate) fn record_user_turn(&mut self, turn: &PreparedTurn) {
        self.fixture.ops.push(UiTraceOp::UserTurn {
            text: turn.display_text.clone(),
        });
    }

    pub(crate) fn record_queue_turn(&mut self, turn: &PreparedTurn) {
        self.fixture.ops.push(UiTraceOp::QueueTurn {
            text: turn.display_text.clone(),
        });
    }

    pub(crate) fn record_queue_pending_steer(&mut self, turn: &PreparedTurn) {
        self.fixture.ops.push(UiTraceOp::QueuePendingSteer {
            text: turn.display_text.clone(),
        });
    }

    pub(crate) fn record_slash_command(&mut self, text: impl Into<String>) {
        self.fixture
            .ops
            .push(UiTraceOp::SlashCommand { text: text.into() });
    }

    pub(crate) fn record_input_insert_text(&mut self, text: impl Into<String>) {
        self.fixture
            .ops
            .push(UiTraceOp::InputInsertText { text: text.into() });
    }

    pub(crate) fn record_input_backspace(&mut self) {
        self.fixture.ops.push(UiTraceOp::InputBackspace);
    }

    pub(crate) fn record_input_delete(&mut self) {
        self.fixture.ops.push(UiTraceOp::InputDelete);
    }

    pub(crate) fn record_move_cursor_left(&mut self) {
        self.fixture.ops.push(UiTraceOp::MoveCursorLeft);
    }

    pub(crate) fn record_move_cursor_right(&mut self) {
        self.fixture.ops.push(UiTraceOp::MoveCursorRight);
    }

    pub(crate) fn record_move_cursor_home(&mut self) {
        self.fixture.ops.push(UiTraceOp::MoveCursorHome);
    }

    pub(crate) fn record_move_cursor_end(&mut self) {
        self.fixture.ops.push(UiTraceOp::MoveCursorEnd);
    }

    pub(crate) fn record_history_up(&mut self) {
        self.fixture.ops.push(UiTraceOp::HistoryUp);
    }

    pub(crate) fn record_history_down(&mut self) {
        self.fixture.ops.push(UiTraceOp::HistoryDown);
    }

    pub(crate) fn record_suggestion_up(&mut self) {
        self.fixture.ops.push(UiTraceOp::SuggestionUp);
    }

    pub(crate) fn record_suggestion_down(&mut self) {
        self.fixture.ops.push(UiTraceOp::SuggestionDown);
    }

    pub(crate) fn record_suggestion_complete(&mut self) {
        self.fixture.ops.push(UiTraceOp::SuggestionComplete);
    }

    pub(crate) fn record_emit_prompt(&mut self, request: &PromptRequest) {
        self.fixture.ops.push(UiTraceOp::EmitPrompt {
            request: TracePromptRequest::from_request(request),
        });
    }

    pub(crate) fn record_prompt_up(&mut self) {
        self.fixture.ops.push(UiTraceOp::PromptUp);
    }

    pub(crate) fn record_prompt_down(&mut self) {
        self.fixture.ops.push(UiTraceOp::PromptDown);
    }

    pub(crate) fn record_prompt_toggle_current_option(&mut self) {
        self.fixture.ops.push(UiTraceOp::PromptToggleCurrentOption);
    }

    pub(crate) fn record_prompt_toggle_note_focus(&mut self) {
        self.fixture.ops.push(UiTraceOp::PromptToggleNoteFocus);
    }

    pub(crate) fn record_prompt_insert_text(&mut self, text: impl Into<String>) {
        self.fixture
            .ops
            .push(UiTraceOp::PromptInsertText { text: text.into() });
    }

    pub(crate) fn record_submit_prompt(&mut self) {
        self.fixture.ops.push(UiTraceOp::SubmitPrompt);
    }

    pub(crate) fn record_prompt_backspace(&mut self) {
        self.fixture.ops.push(UiTraceOp::PromptBackspace);
    }

    pub(crate) fn record_prompt_dismiss(&mut self) {
        self.fixture.ops.push(UiTraceOp::PromptDismiss);
    }

    pub(crate) fn record_scroll_up(&mut self, amount: usize) {
        self.fixture.ops.push(UiTraceOp::ScrollUp { amount });
    }

    pub(crate) fn record_scroll_down(&mut self, amount: usize) {
        self.fixture.ops.push(UiTraceOp::ScrollDown { amount });
    }

    pub(crate) fn record_session_event(&mut self, event: &SessionEvent) {
        if let Some(event) = TraceSessionEvent::from_session_event(event) {
            self.fixture.ops.push(UiTraceOp::Event { event });
        }
    }

    pub(crate) fn maybe_record_render_checkpoint(&mut self, screen: &str) {
        let Some(interval) = self.checkpoint_interval else {
            return;
        };
        let now = Instant::now();
        if now.duration_since(self.last_checkpoint_at) < interval {
            return;
        }
        self.last_checkpoint_at = now;
        self.checkpoint_index += 1;
        let name = format!(
            "{}.{:03}.snap",
            self.trace_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("ui_trace"),
            self.checkpoint_index
        );
        self.fixture.ops.push(UiTraceOp::Render {
            snapshot: name.clone(),
        });
        self.snapshots.push((name, normalize_snapshot_text(screen)));
    }

    pub(crate) fn finish(mut self, final_screen: &str) -> anyhow::Result<()> {
        self.fixture.ops.push(UiTraceOp::Render {
            snapshot: self.snapshot_name.clone(),
        });

        if let Some(parent) = self.trace_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let snapshot_path = self.trace_path.with_file_name(&self.snapshot_name);
        for (name, contents) in &self.snapshots {
            fs::write(self.trace_path.with_file_name(name), contents)?;
        }
        fs::write(&snapshot_path, normalize_snapshot_text(final_screen))?;
        fs::write(
            &self.trace_path,
            serde_json::to_string_pretty(&self.fixture)?,
        )?;
        Ok(())
    }
}

fn normalize_snapshot_text(screen: &str) -> String {
    if screen.ends_with('\n') {
        screen.to_string()
    } else {
        format!("{screen}\n")
    }
}

fn normalize_trace_path(path: &Path) -> PathBuf {
    if path.extension().is_some() {
        path.to_path_buf()
    } else {
        path.with_extension("json")
    }
}

fn aux_ops() -> &'static Mutex<Option<Vec<UiTraceOp>>> {
    static AUX_OPS: OnceLock<Mutex<Option<Vec<UiTraceOp>>>> = OnceLock::new();
    AUX_OPS.get_or_init(|| Mutex::new(None))
}

pub(crate) fn enable_aux_op_recording() {
    *aux_ops().lock().expect("aux op lock") = Some(Vec::new());
}

pub(crate) fn disable_aux_op_recording() {
    *aux_ops().lock().expect("aux op lock") = None;
}

pub(crate) fn drain_aux_ops_into(recorder: &mut UiTraceRecorder) {
    let mut guard = aux_ops().lock().expect("aux op lock");
    let Some(ops) = guard.as_mut() else {
        return;
    };
    recorder.fixture.ops.append(ops);
}

pub(crate) fn record_system_message_aux(text: &str) {
    let mut guard = aux_ops().lock().expect("aux op lock");
    let Some(ops) = guard.as_mut() else {
        return;
    };
    ops.push(UiTraceOp::SystemMessage {
        text: text.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_writes_trace_and_snapshot_files() {
        let dir = std::env::temp_dir().join(format!(
            "lash-ui-trace-{}",
            uuid::Uuid::new_v4().as_simple()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        let trace_path = dir.join("capture");

        let mut recorder = UiTraceRecorder::new(&trace_path, 80, 24, None);
        recorder.record_user_turn(&PreparedTurn::new("hello".into(), Vec::new()));
        recorder.record_start_turn();
        recorder.record_session_event(&SessionEvent::TextDelta {
            content: "world".into(),
        });
        recorder.finish("hello\nworld").expect("finish trace");

        let trace_json =
            fs::read_to_string(dir.join("capture.json")).expect("read generated trace file");
        let trace: UiTraceFixture = serde_json::from_str(&trace_json).expect("parse trace");
        assert_eq!(trace.width, 80);
        assert_eq!(trace.height, 24);
        assert!(matches!(trace.ops[0], UiTraceOp::UserTurn { .. }));
        assert!(matches!(
            trace.ops.last(),
            Some(UiTraceOp::Render { snapshot }) if snapshot == "capture.snap"
        ));

        let snapshot =
            fs::read_to_string(dir.join("capture.snap")).expect("read generated snapshot file");
        assert_eq!(snapshot, "hello\nworld\n");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn recorder_writes_periodic_checkpoint_snapshots() {
        let dir = std::env::temp_dir().join(format!(
            "lash-ui-trace-checkpoints-{}",
            uuid::Uuid::new_v4().as_simple()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        let trace_path = dir.join("capture");

        let mut recorder = UiTraceRecorder::new(&trace_path, 80, 24, Some(Duration::ZERO));
        recorder.record_start_turn();
        recorder.maybe_record_render_checkpoint("first frame");
        recorder.maybe_record_render_checkpoint("second frame");
        recorder.finish("final frame").expect("finish trace");

        let trace_json =
            fs::read_to_string(dir.join("capture.json")).expect("read generated trace file");
        let trace: UiTraceFixture = serde_json::from_str(&trace_json).expect("parse trace");
        let renders = trace
            .ops
            .iter()
            .filter_map(|op| match op {
                UiTraceOp::Render { snapshot } => Some(snapshot.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            renders,
            vec!["capture.001.snap", "capture.002.snap", "capture.snap"]
        );

        let first =
            fs::read_to_string(dir.join("capture.001.snap")).expect("read first checkpoint");
        let second =
            fs::read_to_string(dir.join("capture.002.snap")).expect("read second checkpoint");
        let final_snapshot =
            fs::read_to_string(dir.join("capture.snap")).expect("read final snapshot");
        assert_eq!(first, "first frame\n");
        assert_eq!(second, "second frame\n");
        assert_eq!(final_snapshot, "final frame\n");

        let _ = fs::remove_dir_all(dir);
    }
}
