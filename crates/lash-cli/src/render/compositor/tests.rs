#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_model::PromptRequest;
    use lash_core::PromptUsage;
    use lash_tui_extensions::{
        TuiExtension, TuiExtensions, TuiHostEffect, TuiRenderContext, TuiSurfaceSize,
        TuiSurfaceSlot, TuiSurfaceSpec,
    };
    use std::sync::Arc;
    use std::sync::mpsc;

    use crate::config::ThemeName;
    use crate::overlay::{CommandPaletteAction, CommandPaletteItem, PromptFocus, PromptState};

    #[test]
    fn animated_lash_word_cycles_slash_through_wordmark() {
        let frames = [0_u64, 200, 400, 600, 800]
            .into_iter()
            .map(std::time::Duration::from_millis)
            .map(animated_lash_word)
            .map(|spans| {
                spans
                    .into_iter()
                    .map(|span| span.content.into_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert_eq!(frames, vec!["/LASH", "L/ASH", "LA/SH", "LAS/H", "LASH/"]);
    }

    #[test]
    fn turn_status_label_uses_only_public_run_states() {
        assert_eq!(
            turn_status_label_for_state(CliRunState::Working),
            TurnStatusLabel::Working
        );
        assert_eq!(
            turn_status_label_for_state(CliRunState::RunningTool),
            TurnStatusLabel::RunningTool
        );
        assert_eq!(
            turn_status_label_for_state(CliRunState::Thinking),
            TurnStatusLabel::Thinking
        );
        assert_eq!(
            turn_status_label_for_state(CliRunState::Responding),
            TurnStatusLabel::Responding
        );
    }

    #[test]
    fn status_bar_shows_context_window_usage() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        // Isolate the meter from the test runner's environment: `App::new`
        // probes the live cwd/git branch, and a long branch name would crowd
        // the higher-priority repo/cwd slots into the right half and drop the
        // context meter. This test asserts only the meter renders.
        app.repo_status = None;
        app.cwd = String::new();
        app.model_variant = Some("high".into());
        app.usage.context_window = Some(1_100_000);
        app.usage.last_prompt_usage = Some(PromptUsage {
            prompt_context_tokens: 0,
            input_tokens: 0,
            cache_read_input_tokens: 0,
            cache_write_input_tokens: 0,
            context_budget_tokens: 7_000,
        });

        let snapshot = lash_tui::render_snapshot(80, 4, |frame| draw(frame, &mut app));
        let bottom = snapshot.visible_line_trimmed(3);
        assert!(bottom.contains("gpt-5.4 high · standard"));
        assert!(!bottom.contains("lash · gpt-5.4"));
        // Context meter: tokens used + integer percent, separated by a
        // middle dot. The old representation was `7.0k / 1.1M (0.6%)`,
        // which gave three renderings of the same number.
        assert!(bottom.contains("ctx 7.0k · 1%"));
    }

    #[test]
    fn status_bar_hides_meter_during_first_turn_before_input_accounting_lands() {
        // Regression: while the very first response is streaming, only
        // `live_output_tokens_estimate` is nonzero; using it as the displayed
        // total reads as if those streamed bytes were the entire context, so
        // the bar shows e.g. `36 · 0%` against a 1.1M-token window. The
        // meter should stay off until real input accounting lands.
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.repo_status = None;
        app.cwd = String::new();
        app.usage.context_window = Some(1_100_000);
        app.start_turn();
        app.usage.live_output_tokens_estimate = 36;
        // No `last_prompt_usage`, no `last_response_usage` — first turn.

        let snapshot = lash_tui::render_snapshot(80, 4, |frame| draw(frame, &mut app));
        let bottom = snapshot.visible_line_trimmed(3);
        // What we don't want is a raw token count or a percent sourced from
        // the streaming output estimate.
        assert!(
            !bottom.contains('%'),
            "unexpected percent on bottom line: {bottom}"
        );
        assert!(
            !bottom.contains("36"),
            "unexpected token count on bottom line: {bottom}"
        );
    }

    #[test]
    fn toast_renders_as_top_right_overlay() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.show_toast("Copied to clipboard", ToastKind::Info);

        let snapshot = lash_tui::render_snapshot(80, 12, |frame| draw(frame, &mut app));
        let line = snapshot.visible_line_trimmed(3);

        assert!(line.contains("Copied to clipboard"), "line: {line}");
        assert!(line.ends_with("│"), "line: {line}");
    }

    #[test]
    fn stale_working_status_does_not_make_idle_cli_working() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let (chrome_ext, chrome_state) = crate::chrome_ui::ChromeTuiExtension::new();
        let ui_extensions =
            Arc::new(TuiExtensions::new(vec![chrome_ext]).expect("chrome extension"));
        app.set_ui_extensions(ui_extensions);
        app.set_chrome_state(chrome_state);

        app.handle_turn_activity(lash_core::TurnActivity::independent(
            lash_core::TurnEvent::QueuedWorkStarted {
                boundary: lash_core::runtime::QueuedWorkClaimBoundary::Idle,
                batch_ids: Vec::new(),
                causes: Vec::new(),
            },
        ));
        assert!(!app.turn_active());
        assert_eq!(app.run_state, CliRunState::Idle);

        sync_chrome_turn_status(&app);
        let snapshot = lash_tui::render_snapshot(80, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("Idle"));
        assert!(!visible.contains("Working"));
    }

    #[test]
    fn session_picker_remains_visible_across_repeated_renders() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "current-session-id".into());
        app.show_session_picker(vec![crate::session_log::SessionInfo {
            filename: "previous.db".into(),
            session_id: "previous-session-id".into(),
            message_count: 3,
            first_message: "previous task".into(),
            modified: std::time::SystemTime::now(),
            cwd: Some(std::path::PathBuf::from("/workspace/code/lash")),
        }]);

        let first = lash_tui::render_snapshot(96, 24, |frame| draw(frame, &mut app));
        assert!(app.has_session_picker());
        let first_lines = first.visible_lines_trimmed();
        assert!(
            first_lines
                .iter()
                .any(|line| line.contains("Resume Session (1/1)"))
        );
        assert!(
            first_lines
                .iter()
                .any(|line| line.contains("previous task"))
        );

        app.dirty = false;
        app.on_tick();
        let second = lash_tui::render_snapshot(96, 24, |frame| draw(frame, &mut app));
        assert!(app.has_session_picker());
        let second_lines = second.visible_lines_trimmed();
        assert!(
            second_lines
                .iter()
                .any(|line| line.contains("Resume Session (1/1)"))
        );
        assert!(
            second_lines
                .iter()
                .any(|line| line.contains("previous task"))
        );

        let compact = lash_tui::render_snapshot(96, 10, |frame| draw(frame, &mut app));
        let compact_lines = compact.visible_lines_trimmed();
        assert!(
            compact_lines
                .iter()
                .any(|line| line.contains("Resume Session (1/1)")),
            "session picker should shrink instead of disappearing in compact layouts"
        );
    }

    #[test]
    fn session_picker_filters_sessions_by_typed_query() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "current-session-id".into());
        app.show_session_picker(vec![
            crate::session_log::SessionInfo {
                filename: "alpha.db".into(),
                session_id: "alpha-session".into(),
                message_count: 2,
                first_message: "debug the resume picker".into(),
                modified: std::time::SystemTime::now(),
                cwd: Some(std::path::PathBuf::from("/workspace/code/lash")),
            },
            crate::session_log::SessionInfo {
                filename: "beta.db".into(),
                session_id: "beta-session".into(),
                message_count: 4,
                first_message: "write release notes".into(),
                modified: std::time::SystemTime::now(),
                cwd: Some(std::path::PathBuf::from("/tmp/elsewhere")),
            },
        ]);
        for ch in "release".chars() {
            app.session_picker_insert_query_char(ch);
        }

        let snapshot = lash_tui::render_snapshot(96, 24, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("Resume Session (1/2)"));
        assert!(visible.contains("write release notes"));
        assert!(!visible.contains("debug the resume picker"));
    }

    #[test]
    fn command_palette_renders_grouped_settings_with_current_marker() {
        theme::set_active_theme(ThemeName::Lash);
        let mut app = App::new("gpt-5.4".into(), "test".into(), "current-session-id".into());
        app.show_command_palette(vec![
            CommandPaletteItem::new(
                "Settings",
                "Theme: Lash",
                "Use Lash's high-contrast dark palette.",
                CommandPaletteAction::Theme(ThemeName::Lash),
            )
            .footer("lash")
            .current(true),
            CommandPaletteItem::new(
                "Settings",
                "Theme: System",
                "Use terminal defaults and ANSI palette colors.",
                CommandPaletteAction::Theme(ThemeName::System),
            )
            .footer("system"),
            CommandPaletteItem::new(
                "Session",
                "Resume Session",
                "Search previous sessions.",
                CommandPaletteAction::Builtin(crate::command::Command::Resume(None)),
            )
            .footer("/resume"),
        ]);

        let snapshot = lash_tui::render_snapshot(100, 24, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("Commands (1/3)"), "{visible}");
        assert!(visible.contains("Settings"), "{visible}");
        assert!(visible.contains("● Theme: Lash"), "{visible}");
        assert!(visible.contains("Theme: System"), "{visible}");
        assert!(visible.contains("Session"), "{visible}");
        assert!(visible.contains("esc"), "{visible}");
        assert!(
            (0..snapshot.height).any(|y| {
                (0..snapshot.width).any(|x| {
                    snapshot
                        .cell(x, y)
                        .is_some_and(|cell| cell.style.bg == Some(theme::selection_bg()))
                })
            }),
            "selected command row should use semantic selection background"
        );
    }

    #[test]
    fn system_command_palette_uses_readable_list_selection_styles() {
        theme::set_active_theme(ThemeName::System);
        let mut app = App::new("gpt-5.4".into(), "test".into(), "current-session-id".into());
        app.show_command_palette(vec![
            CommandPaletteItem::new(
                "Conversation",
                "/clear",
                "Reset conversation",
                CommandPaletteAction::Builtin(crate::command::Command::Clear),
            ),
            CommandPaletteItem::new(
                "Conversation",
                "/compact",
                "Open a compaction frame seeded by a summary",
                CommandPaletteAction::Builtin(crate::command::Command::Compact(None)),
            ),
        ]);

        let snapshot = lash_tui::render_snapshot(100, 24, |frame| draw(frame, &mut app));
        let clear_row = (0..snapshot.height)
            .find(|y| snapshot.visible_line(*y).contains("/clear"))
            .expect("selected command row");
        let compact_row = (0..snapshot.height)
            .find(|y| snapshot.visible_line(*y).contains("/compact"))
            .expect("unselected command row");

        assert!(
            (0..snapshot.width).any(|x| {
                snapshot
                    .cell(x, clear_row)
                    .is_some_and(|cell| cell.style.bg == Some(theme::selected_row_bg()))
            }),
            "selected command row should use selected-row background"
        );
        assert!(
            (0..snapshot.width).all(|x| {
                snapshot
                    .cell(x, clear_row)
                    .is_none_or(|cell| cell.style.bg != Some(theme::selection_bg()))
            }),
            "menu selection should not reuse text selection background in System theme"
        );
        assert!(
            (0..snapshot.width).any(|x| {
                snapshot
                    .cell(x, compact_row)
                    .is_some_and(|cell| cell.style.fg == Some(theme::text_muted()))
            }),
            "unselected command rows should use readable list text"
        );
        theme::set_active_theme(ThemeName::Lash);
    }

    #[test]
    fn system_theme_idle_chrome_does_not_paint_background_cells() {
        theme::set_active_theme(ThemeName::System);
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());

        let snapshot = lash_tui::render_snapshot(100, 28, |frame| draw(frame, &mut app));

        for y in 0..snapshot.height {
            for x in 0..snapshot.width {
                let cell = snapshot.cell(x, y).expect("cell exists");
                assert_eq!(
                    cell.style.bg, None,
                    "system idle chrome painted a background at ({x}, {y})"
                );
            }
        }
        theme::set_active_theme(ThemeName::Lash);
    }

    #[test]
    fn lash_theme_idle_chrome_keeps_background_under_text_and_rules() {
        theme::set_active_theme(ThemeName::Lash);
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());

        let snapshot = lash_tui::render_snapshot(100, 28, |frame| draw(frame, &mut app));

        for y in 0..snapshot.height {
            for x in 0..snapshot.width {
                let cell = snapshot.cell(x, y).expect("cell exists");
                assert!(
                    cell.style.bg.is_some(),
                    "lash idle chrome left terminal background visible at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn idle_footer_stays_idle_when_background_processes_exist() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let (chrome_ext, chrome_state) = crate::chrome_ui::ChromeTuiExtension::new();
        let ui_extensions =
            Arc::new(TuiExtensions::new(vec![chrome_ext]).expect("chrome extension"));
        app.set_ui_extensions(ui_extensions);
        app.set_chrome_state(chrome_state);
        app.update_processes(vec![lash_core::ProcessHandleSummary::new(
            "process-1",
            lash_core::ProcessHandleDescriptor::new(Some("lashlang"), Some("responder")),
            lash_core::ProcessLifecycleStatus::Running,
        )]);
        sync_chrome_turn_status(&app);

        let snapshot = lash_tui::render_snapshot(80, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("Idle"));
        assert!(!visible.contains("Working"));
        assert!(visible.contains("1 process"));
    }

    #[test]
    fn bottom_metadata_omits_session_name_and_repo_name() {
        let mut app = App::new(
            "gpt-5.4".into(),
            "autumn-falls".into(),
            "test-session-id".into(),
        );
        app.repo_status = Some(crate::repo_status::RepoStatus {
            repo_root: std::path::PathBuf::from("/tmp/lash"),
            repo_name: "lash".into(),
            branch: "staging".into(),
            worktree: None,
        });

        let snapshot = lash_tui::render_snapshot(84, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("staging"));
        assert!(!visible.contains("lash · staging"));
        assert!(!visible.contains("autumn-falls"));
    }

    #[test]
    fn process_dock_renders_below_input_and_overview_as_overlay() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.update_processes(vec![
            lash_core::ProcessHandleSummary::new(
                "process-1",
                lash_core::ProcessHandleDescriptor::new(Some("lashlang"), Some("responder")),
                lash_core::ProcessLifecycleStatus::Running,
            )
            .with_definition(Some(
                serde_json::to_value(lashlang::ProcessDefinitionIdentity::new(
                    lashlang::ModuleRef::new(&lashlang::ContentHash::new("module")),
                    lashlang::HostRequirementsRef::new(&lashlang::ContentHash::new("host")),
                    lashlang::ProcessRef::new(lashlang::ContentHash::new("process"), 1),
                    "responder",
                ))
                .expect("process definition serializes"),
            )),
        ]);
        app.select_next_process();

        let areas = render::chrome_areas(&app, 80, 16);
        let snapshot = lash_tui::render_snapshot(80, 16, |frame| draw(frame, &mut app));
        assert!(areas.process.y > areas.input.y);
        assert!(
            snapshot
                .visible_line_trimmed(areas.process.y)
                .contains("Background")
        );

        let overview = app
            .selected_process_overview_state()
            .expect("process overview");
        app.show_process_overview(overview);
        let snapshot = lash_tui::render_snapshot(80, 16, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");
        assert!(visible.contains("Process responder"));
        assert!(visible.contains("definition"));
    }

    #[test]
    fn option_prompt_starts_at_top_of_question() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let (response_tx, _response_rx) = mpsc::channel();
        app.show_prompt(PromptState {
            request: PromptRequest::single(
                "Plan .lash/plans/15d5a2bd-841d-4729-8968-ae7874385e16.md is ready. Exit plan mode now?",
                vec!["Exit plan mode".into(), "Keep planning".into()],
            )
            .with_optional_note(),
            focus: PromptFocus::Options,
            cursor: 0,
            scroll_offset: 0,
            selected: Default::default(),
            reply_text: String::new(),
            reply_cursor: 0,
            response_tx,
        });

        let snapshot = lash_tui::render_snapshot(84, 14, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");
        assert!(visible.contains("Plan .lash/plans/15d5a2bd-841d-4729-8968-ae7874385e16.md"));
        assert!(visible.contains("Choices"));
        assert!(!visible.contains("Question"));
        assert!(!visible.contains("┌"));
    }

    #[test]
    fn prompt_panel_can_scroll_when_content_exceeds_viewport() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let (response_tx, _response_rx) = mpsc::channel();
        app.show_prompt(PromptState {
            request: PromptRequest::single("Exit plan mode?", vec!["Exit".into()])
                .with_markdown_panel(
                    "PLAN",
                    "# Plan\n\nline 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\nline 11\nline 12",
                ),
            focus: PromptFocus::Options,
            cursor: 0,
            scroll_offset: 7,
            selected: Default::default(),
            reply_text: String::new(),
            reply_cursor: 0,
            response_tx,
        });

        let snapshot = lash_tui::render_snapshot(60, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");
        assert!(!visible.contains("line 1"));
        assert!(visible.contains("line 6") || visible.contains("line 7"));
        assert!(visible.contains("Choices"));
        assert!(visible.contains("Exit"));
    }

    #[test]
    fn history_selection_highlights_visible_cells() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.timeline = vec![crate::app::UiTimelineItem::UserInput(
            "alpha\nbeta\ngamma".into(),
        )]
        .into();
        app.selection.anchor = (2, 1);
        app.selection.end = (5, 1);
        app.selection.visible = true;

        let history = render::history_area(&app, 40, 9);
        let snapshot = lash_tui::render_snapshot(40, 9, |frame| draw(frame, &mut app));
        // "alpha/beta/gamma" is 3 lines; bottom-anchored in the history
        // viewport it gains `height - 3` rows of headroom above it, so the
        // selected "beta" row lands at `top_pad + 1`.
        let top_pad = history.height as usize - 3;
        let row = history.y + (top_pad + 1) as u16;
        assert_eq!(
            snapshot.cell(2, row).and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
        assert_eq!(
            snapshot.cell(4, row).and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
    }

    #[test]
    fn history_selection_tracks_content_rows_while_scrolled() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        // Eight lines overflow the ~5-row history viewport, so the transcript
        // is genuinely scrolled (top-anchor pad is zero) and `scroll_offset`
        // is meaningful — the regime this test is about.
        app.timeline = vec![crate::app::UiTimelineItem::UserInput(
            "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\neta\ntheta".into(),
        )]
        .into();
        app.scroll_offset = 1;
        app.selection.anchor = (2, 2);
        app.selection.end = (4, 2);
        app.selection.visible = true;

        let history = render::history_area(&app, 40, 9);
        let snapshot = lash_tui::render_snapshot(40, 9, |frame| draw(frame, &mut app));
        // Content row 2 ("gamma") with scroll_offset 1 sits at viewport row 1.
        assert_eq!(
            snapshot
                .cell(2, history.y + 1)
                .and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
    }

    #[test]
    fn input_selection_highlights_visible_cells() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.set_input("alpha beta".into());
        app.start_input_selection(2);
        app.update_input_selection(7);
        app.finish_input_selection();

        let input = render::input_content_area(&app, 40, 9);
        let snapshot = lash_tui::render_snapshot(40, 9, |frame| draw(frame, &mut app));
        assert_eq!(
            snapshot
                .cell(input.x + 4, input.y)
                .and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
        assert_eq!(
            snapshot
                .cell(input.x + 8, input.y)
                .and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
    }

    struct SurfaceTestTuiExtension;

    #[async_trait::async_trait]
    impl TuiExtension for SurfaceTestTuiExtension {
        fn id(&self) -> &'static str {
            "surface_test"
        }

        async fn invoke_action(
            &self,
            _action: &str,
            _arg: Option<&str>,
            _ctx: lash_tui_extensions::TuiExtensionContext,
        ) -> Result<Vec<TuiHostEffect>, String> {
            Ok(Vec::new())
        }

        fn render_surface(
            &self,
            surface_key: &str,
            _ctx: TuiRenderContext<'_>,
            frame: &mut Frame<'_>,
        ) {
            let label = match surface_key {
                "workspace" => "WORKSPACE",
                "footer" => "FOOTER",
                "overlay" => "OVERLAY",
                other => other,
            };
            frame.write_text(0, 0, label, Style::default(), frame.area().width);
        }

        fn handle_turn_event(&self, event: &lash_core::TurnEvent) -> Vec<TuiHostEffect> {
            match event {
                lash_core::TurnEvent::AssistantProseDelta { text } if text == "mount" => vec![
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "workspace".to_string(),
                            slot: TuiSurfaceSlot::Workspace,
                            size: TuiSurfaceSize::Auto,
                            order: 0,
                            focusable: true,
                            visible: true,
                            modal: false,
                        },
                    },
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "footer".to_string(),
                            slot: TuiSurfaceSlot::Footer,
                            size: TuiSurfaceSize::Lines(1),
                            order: 0,
                            focusable: false,
                            visible: true,
                            modal: false,
                        },
                    },
                ],
                lash_core::TurnEvent::AssistantProseDelta { text } if text == "overlay" => vec![
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "overlay".to_string(),
                            slot: TuiSurfaceSlot::Overlay,
                            size: TuiSurfaceSize::Fixed {
                                width: 16,
                                height: 3,
                            },
                            order: 10,
                            focusable: true,
                            visible: true,
                            modal: true,
                        },
                    },
                    TuiHostEffect::FocusSurface {
                        key: "overlay".to_string(),
                    },
                ],
                _ => Vec::new(),
            }
        }
    }

    #[test]
    fn workspace_surface_replaces_history_and_footer_renders_above_input() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.timeline = vec![crate::app::UiTimelineItem::UserInput("history line".into())].into();
        let ui_extensions = Arc::new(
            TuiExtensions::new(vec![Arc::new(SurfaceTestTuiExtension)])
                .expect("surface extensions"),
        );
        ui_extensions.effects_for_turn_event(&lash_core::TurnEvent::AssistantProseDelta {
            text: "mount".to_string(),
        });
        app.set_ui_extensions(Arc::clone(&ui_extensions));

        let snapshot = lash_tui::render_snapshot(40, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("WORKSPACE"));
        assert!(visible.contains("FOOTER"));
        assert!(!visible.contains("history line"));
    }

    #[test]
    fn overlay_surface_renders_last_on_centered_scrim() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let ui_extensions = Arc::new(
            TuiExtensions::new(vec![Arc::new(SurfaceTestTuiExtension)])
                .expect("surface extensions"),
        );
        ui_extensions.effects_for_turn_event(&lash_core::TurnEvent::AssistantProseDelta {
            text: "mount".to_string(),
        });
        ui_extensions.effects_for_turn_event(&lash_core::TurnEvent::AssistantProseDelta {
            text: "overlay".to_string(),
        });
        app.set_ui_extensions(Arc::clone(&ui_extensions));

        let snapshot = lash_tui::render_snapshot(40, 12, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("OVERLAY"));
    }

    #[test]
    fn short_transcript_bottom_anchors_against_input_with_plan_gutter() {
        use crate::app::{PlanDockItem, PlanDockItemStatus, PlanDockState, UiTimelineItem};
        let mut app = App::new("gpt-5.5".into(), "test".into(), "test-session-id".into());
        app.model_variant = Some("xhigh".into());
        app.execution_mode_label = "rlm".into();
        app.repo_status = None;
        app.cwd = String::new();
        app.timeline = vec![UiTimelineItem::UserInput(
            "ok make the required changes and ship it".into(),
        )]
        .into();
        app.plan_dock = Some(PlanDockState {
            title: "Plan".into(),
            meta: None,
            items: vec![
                PlanDockItem {
                    text: "Check git state vs main".into(),
                    status: PlanDockItemStatus::Done,
                },
                PlanDockItem {
                    text: "Monitor GitHub Actions and fix failures".into(),
                    status: PlanDockItemStatus::Active,
                },
            ],
        });

        let lines = lash_tui::render_snapshot(80, 16, |frame| draw(frame, &mut app))
            .visible_lines_trimmed();
        let row = |needle: &str| {
            lines
                .iter()
                .position(|line| line.contains(needle))
                .unwrap_or_else(|| panic!("missing {needle:?} in {lines:?}"))
        };

        // Headroom is at the top: a short transcript drops to the floor of
        // the viewport rather than stranding the input below an empty gap.
        assert!(
            lines[0].trim().is_empty(),
            "expected top headroom, got {:?}",
            lines[0]
        );

        // The user message and the Plan header are separated by a blank gutter.
        let message = row("ok make the required changes");
        let plan = row("Plan");
        assert_eq!(plan, message + 2, "expected one gutter row before Plan");
        assert!(lines[message + 1].trim().is_empty(), "gutter row not blank");

        // The last checklist item hugs the input: the row directly below it is
        // the input's top rule (full-width ─), not empty space.
        let last_item = row("Monitor GitHub Actions");
        assert!(
            lines[last_item + 1].starts_with("───"),
            "checklist should sit directly above the input rule, got {:?}",
            lines[last_item + 1]
        );
    }

    #[test]
    fn slash_suggestions_use_clean_band_and_name_description_hierarchy() {
        use crate::editor::{Suggestion, SuggestionKind};
        let mut app = App::new("gpt-5.5".into(), "test".into(), "test-session-id".into());
        app.repo_status = None;
        app.cwd = String::new();
        app.set_input("/".into());
        app.editor.suggestion_kind = SuggestionKind::Command;
        app.editor.suggestion_idx = 0;
        app.editor.suggestions = vec![
            Suggestion {
                name: "/clear".into(),
                description: "Reset conversation".into(),
                match_indices: vec![0],
            },
            Suggestion {
                name: "/compact".into(),
                description: "Open a compaction frame".into(),
                match_indices: vec![0],
            },
        ];

        let snapshot = lash_tui::render_snapshot(80, 18, |frame| draw(frame, &mut app));
        let lines = snapshot.visible_lines_trimmed();
        let sel_y = lines.iter().position(|l| l.contains("clear")).expect("clear row") as u16;
        let idle_y = lines
            .iter()
            .position(|l| l.contains("compact"))
            .expect("compact row") as u16;

        // The two name/description tiers are actually distinct colors, so the
        // hierarchy this test asserts isn't a no-op.
        assert_ne!(theme::text_primary(), theme::text_muted());
        assert_ne!(theme::text_muted(), theme::text_subtle());

        // Selected row: a chalk rail (the old gold ▶ marker is gone — no brand
        // on the gutter) over a full-width amber band.
        let rail = snapshot.cell(1, sel_y).expect("rail cell");
        assert_eq!(rail.ch, '▌');
        assert_eq!(rail.style.fg, Some(theme::text_primary()));
        assert_ne!(rail.style.fg, Some(theme::brand()));
        assert_eq!(rail.style.bg, Some(theme::selected_row_bg()));
        // Band runs edge-to-edge: it reaches just inside the right border and
        // fills most of the row, rather than stopping where the text ends.
        let right_border = (0..80u16)
            .rev()
            .find(|&x| snapshot.cell(x, sel_y).map(|c| c.ch) == Some('│'))
            .expect("right border");
        assert_eq!(
            snapshot
                .cell(right_border - 1, sel_y)
                .and_then(|c| c.style.bg),
            Some(theme::selected_row_bg()),
            "selection band should reach the right edge",
        );
        let band_cells = (1..right_border)
            .filter(|&x| {
                snapshot.cell(x, sel_y).and_then(|c| c.style.bg) == Some(theme::selected_row_bg())
            })
            .count();
        assert!(band_cells >= 25, "selection band should fill the row, got {band_cells}");

        // Selected name reads at full chalk; its description one tier down.
        assert_eq!(snapshot.cell(3, sel_y).and_then(|c| c.style.fg), Some(theme::text_primary()));
        let sel_desc_x = lines[sel_y as usize].find("Reset").expect("desc") as u16;
        assert_eq!(
            snapshot.cell(sel_desc_x, sel_y).and_then(|c| c.style.fg),
            Some(theme::text_muted())
        );

        // Idle row: the matched '/' carries the brand, the rest of the name sits
        // at muted chalk, and the description recedes another tier — name and
        // description are no longer the same flat gray.
        assert_eq!(snapshot.cell(3, idle_y).and_then(|c| c.style.fg), Some(theme::brand()));
        assert_eq!(snapshot.cell(4, idle_y).and_then(|c| c.style.fg), Some(theme::text_muted()));
        let idle_desc_x = lines[idle_y as usize].find("Open").expect("desc") as u16;
        assert_eq!(
            snapshot.cell(idle_desc_x, idle_y).and_then(|c| c.style.fg),
            Some(theme::text_subtle())
        );
    }
}
