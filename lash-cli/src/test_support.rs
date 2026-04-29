use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;

use lash::{PromptRequest, PromptResponse, SessionEvent};
use lash_tui::ScreenSnapshot;
use lash_ui::UiExtensions;
use tokio::sync::Mutex;

use crate::app::{App, PreparedTurn, PromptState};
use crate::overlay::PromptFocus;
use crate::ui_trace::{TraceRepoStatus, render_screen_snapshot};
use crate::{apply_ui_host_effects, render};

pub(crate) fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

pub(crate) struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    pub(crate) fn set(key: &'static str, value: &Path) -> Self {
        let previous = std::env::var_os(key);
        // Tests serialize access with env_lock() so mutating process env is safe here.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.as_ref() {
            // Tests serialize access with env_lock() so mutating process env is safe here.
            unsafe { std::env::set_var(self.key, previous) };
        } else {
            // Tests serialize access with env_lock() so mutating process env is safe here.
            unsafe { std::env::remove_var(self.key) };
        }
    }
}

pub(crate) struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    pub(crate) fn new(prefix: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("{prefix}-{}", uuid::Uuid::new_v4().as_simple()));
        std::fs::create_dir_all(&path).expect("temp dir");
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub(crate) struct UiHarness {
    pub app: App,
    pub width: u16,
    pub height: u16,
    ui_extensions: Arc<UiExtensions>,
}

impl UiHarness {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        let ui_extensions = Arc::new(UiExtensions::builtin().expect("builtin ui extensions"));
        let mut app = App::new(
            "gpt-5.4".to_string(),
            "test-session".to_string(),
            "test-session-id".into(),
        );
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("lash-cli crate should live under repo root")
            .to_path_buf();
        app.cwd = repo_root.display().to_string();
        app.repo_status = Some(
            TraceRepoStatus {
                repo_name: "lash".to_string(),
                branch: "staging".to_string(),
                worktree: None,
            }
            .into_repo_status(&app.cwd),
        );
        app.set_ui_extensions(Arc::clone(&ui_extensions));
        Self {
            app,
            width,
            height,
            ui_extensions,
        }
    }

    pub(crate) fn start_turn(&mut self) -> &mut Self {
        self.app.start_turn();
        self
    }

    pub(crate) fn user_turn(&mut self, text: impl Into<String>) -> PreparedTurn {
        let turn = PreparedTurn::new(text.into(), Vec::new());
        self.app.push_prepared_user_input(&turn);
        turn
    }

    pub(crate) fn queue_turn(&mut self, text: impl Into<String>) -> PreparedTurn {
        let turn = PreparedTurn::new(text.into(), Vec::new());
        self.app.queue_turn(turn.clone());
        turn
    }

    pub(crate) fn dispatch_event(&mut self, event: SessionEvent) -> &mut Self {
        match event {
            SessionEvent::Prompt {
                request,
                response_tx,
            } => {
                let effects = self
                    .ui_extensions
                    .effects_for_session_event(&SessionEvent::Prompt {
                        request: request.clone(),
                        response_tx: response_tx.clone(),
                    });
                apply_ui_host_effects(&mut self.app, effects);
                let focus = if request.is_freeform() {
                    PromptFocus::Text
                } else {
                    PromptFocus::Options
                };
                self.app.show_prompt(PromptState {
                    request,
                    focus,
                    cursor: 0,
                    scroll_offset: 0,
                    selected: Default::default(),
                    reply_text: String::new(),
                    reply_cursor: 0,
                    response_tx,
                });
            }
            other => {
                let effects = self.ui_extensions.effects_for_session_event(&other);
                self.app.handle_session_event(other);
                apply_ui_host_effects(&mut self.app, effects);
            }
        }
        self
    }

    pub(crate) fn emit_prompt(
        &mut self,
        request: PromptRequest,
    ) -> std::sync::mpsc::Receiver<PromptResponse> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.dispatch_event(SessionEvent::Prompt {
            request,
            response_tx: tx,
        });
        rx
    }

    pub(crate) fn take_prompt_response(&mut self) -> Option<String> {
        self.app.take_prompt_response()
    }

    pub(crate) fn render(&mut self) -> ScreenSnapshot {
        render_screen_snapshot(&mut self.app, self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::{PluginSurfaceEvent, SessionEvent};
    use serde_json::json;

    fn line_index_containing(lines: &[String], needle: &str) -> Option<usize> {
        lines.iter().position(|line| line.contains(needle))
    }

    #[test]
    fn ui_harness_renders_streamed_turn_end_to_end() {
        let mut harness = UiHarness::new(50, 14);
        harness.user_turn("write a poem");
        harness.start_turn();
        harness.dispatch_event(SessionEvent::LlmRequest {
            iteration: 0,
            message_count: 1,
            tool_list: "read_file, shell".to_string(),
        });
        harness.dispatch_event(SessionEvent::TextDelta {
            content: "A short line.\nAnother line.".to_string(),
        });

        let screen = harness.render();
        let lines = screen.non_empty_visible_lines();
        assert!(lines.iter().any(|line| line.contains("write a poem")));
        assert!(lines.iter().any(|line| line.contains("A short line.")));
        assert!(lines.iter().any(|line| line.contains("Another line.")));
    }

    #[test]
    fn ui_harness_intercepts_prompt_and_renders_overlay() {
        let mut harness = UiHarness::new(60, 16);
        let rx = harness.emit_prompt(
            PromptRequest::single("Pick one", vec!["Alpha".into(), "Beta".into()])
                .with_optional_note(),
        );
        harness.app.prompt_down();
        harness.app.prompt_toggle_note_focus();
        harness.app.prompt_insert_text("because");

        let screen = harness.render();
        let lines = screen.non_empty_visible_lines();
        assert!(harness.app.has_prompt());
        assert!(lines.iter().any(|line| line.contains("because")));

        let display = harness.take_prompt_response().expect("prompt display");
        assert!(display.contains("2. Beta"));
        let response = rx.recv().expect("prompt response");
        assert_eq!(
            response,
            PromptResponse::Single {
                selection: "Beta".to_string(),
                note: Some("because".to_string()),
            }
        );
    }

    #[test]
    fn ui_harness_renders_inline_ask_panel_end_to_end() {
        let mut harness = UiHarness::new(68, 20);
        harness.user_turn("ask me");
        harness.dispatch_event(SessionEvent::ToolCall {
            call_id: Some("call-1".into()),
            name: "ask".into(),
            args: json!({
                "question": "Pick one",
                "options": ["Alpha", "Beta", "Gamma"],
            }),
            result: json!({
                "kind": "single",
                "selection": "Beta",
                "note": "because",
            }),
            success: true,
            duration_ms: 12,
        });

        let screen = harness.render();
        let lines = screen.non_empty_visible_lines();
        assert!(lines.iter().any(|line| line.contains("QUESTION")));
        assert!(lines.iter().any(|line| line.contains("Pick one")));
        assert!(lines.iter().any(|line| line.contains("◉ 2. Beta")));
        assert!(lines.iter().any(|line| line.contains("Note · because")));
    }

    #[test]
    fn ui_harness_renders_queue_preview_sections_end_to_end() {
        let mut harness = UiHarness::new(72, 18);
        harness
            .app
            .queue_pending_steer(PreparedTurn::new("after tool do this".into(), Vec::new()));
        harness.queue_turn("queued follow-up one");
        harness.queue_turn("queued follow-up two");
        harness.queue_turn("queued follow-up three");

        let screen = harness.render();
        let lines = screen.non_empty_visible_lines();

        assert!(
            lines
                .iter()
                .any(|line| line.contains("after next tool/result"))
        );
        assert!(lines.iter().any(|line| line.contains("after tool do this")));
        assert!(lines.iter().any(|line| line.contains("next full turn · 3")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("queued follow-up one"))
        );
        assert!(lines.iter().any(|line| line.contains("+1 more")));
    }

    #[test]
    fn ui_harness_renders_accepted_injected_turn_input_during_active_turn() {
        let mut harness = UiHarness::new(72, 18);
        harness.user_turn("initial question");
        harness.start_turn();
        harness
            .app
            .queue_pending_steer(PreparedTurn::new("follow up now".into(), Vec::new()));
        harness.dispatch_event(SessionEvent::InjectedTurnInputAccepted {
            messages: vec![lash::PluginMessage::text(
                lash::MessageRole::User,
                "follow up now",
            )],
            checkpoint: lash::CheckpointKind::AfterWork,
        });
        harness.dispatch_event(SessionEvent::TextDelta {
            content: "I saw the follow up.".into(),
        });

        let screen = harness.render();
        let lines = screen.non_empty_visible_lines();

        assert!(lines.iter().any(|line| line.contains("initial question")));
        assert!(lines.iter().any(|line| line.contains("follow up now")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("I saw the follow up."))
        );
    }

    #[test]
    fn ui_harness_renders_plugin_panel_and_mode_indicator_end_to_end() {
        let mut harness = UiHarness::new(72, 20);
        harness.dispatch_event(SessionEvent::PluginEvent {
            plugin_id: "plan_mode".into(),
            event: PluginSurfaceEvent::ModeIndicatorUpsert {
                key: "status".into(),
                label: "PLAN".into(),
            },
        });
        harness.dispatch_event(SessionEvent::PluginEvent {
            plugin_id: "plan_mode".into(),
            event: PluginSurfaceEvent::PanelUpsert {
                key: "board".into(),
                title: "TASK BOARD".into(),
                content: "- First\n- Second".into(),
            },
        });

        let screen = harness.render();
        let lines = screen.visible_lines_trimmed();

        assert!(lines.iter().any(|line| line.contains("TASK BOARD")));
        assert!(lines.iter().any(|line| line.contains("First")));
        assert!(lines.iter().any(|line| line.contains("Second")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("PLAN · lash · staging"))
        );
        assert!(!lines.iter().any(|line| line.contains("test-session")));
    }

    #[test]
    fn ui_harness_scroll_and_follow_behaves_end_to_end() {
        let mut harness = UiHarness::new(36, 12);
        harness.user_turn("write lines");
        harness.start_turn();
        harness.dispatch_event(SessionEvent::LlmRequest {
            iteration: 0,
            message_count: 1,
            tool_list: "read_file".into(),
        });
        harness.dispatch_event(SessionEvent::TextDelta {
            content: (0..10)
                .map(|idx| format!("line {idx}"))
                .collect::<Vec<_>>()
                .join("\n"),
        });
        harness.render();
        harness.dispatch_event(SessionEvent::TextDelta {
            content: (10..18)
                .map(|idx| format!("\nline {idx}"))
                .collect::<String>(),
        });

        let followed = harness.render().visible_lines_trimmed();
        assert!(followed.iter().any(|line| line.contains("line 17")));
        assert!(
            !followed.iter().any(|line| line.contains("line 0")),
            "tail-follow should keep the newest streamed text visible"
        );

        harness.app.scroll_up(4);
        let scrolled = harness.render().visible_lines_trimmed();
        assert!(
            !scrolled.iter().any(|line| line.contains("line 17")),
            "manual scroll should pause tail-follow"
        );
        let anchored_line = scrolled
            .iter()
            .find(|line| line.contains("line "))
            .cloned()
            .expect("older visible line after scrolling");

        harness.dispatch_event(SessionEvent::TextDelta {
            content: "\nline 18\nline 19".into(),
        });
        let paused = harness.render().visible_lines_trimmed();
        assert!(paused.iter().any(|line| line == &anchored_line));
        assert!(!paused.iter().any(|line| line.contains("line 19")));

        let viewport_height =
            render::history_viewport_height(&harness.app, harness.width, harness.height);
        harness
            .app
            .scroll_down(usize::MAX / 2, viewport_height, harness.width as usize);
        let resumed = harness.render().visible_lines_trimmed();
        assert!(resumed.iter().any(|line| line.contains("line 19")));
    }

    #[test]
    fn ui_harness_first_visible_output_keeps_context_before_switching_to_tail_follow() {
        let mut harness = UiHarness::new(44, 12);
        harness.user_turn("write a very long answer please");
        harness.start_turn();
        harness.dispatch_event(SessionEvent::LlmRequest {
            iteration: 0,
            message_count: 1,
            tool_list: String::new(),
        });

        let before = harness.render().visible_lines_trimmed();
        assert!(
            before
                .iter()
                .any(|line| line.contains("write a very long answer please"))
        );

        harness.dispatch_event(SessionEvent::TextDelta {
            content: "first line\nsecond line\nthird line\nfourth line\nfifth line\nsixth line"
                .into(),
        });
        let first_output = harness.render().visible_lines_trimmed();
        let prompt_idx = line_index_containing(&first_output, "write a very long answer please")
            .expect("prompt line");
        let first_idx =
            line_index_containing(&first_output, "first line").expect("first output line");
        assert!(
            prompt_idx < first_idx,
            "prompt context should stay visible above first output"
        );

        harness.dispatch_event(SessionEvent::TextDelta {
            content: "\nseventh line\neighth line\nninth line\ntenth line".into(),
        });
        let tailed = harness.render().visible_lines_trimmed();
        assert!(tailed.iter().any(|line| line.contains("tenth line")));
    }
}
