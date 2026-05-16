use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use lash_core::TokenUsage;
use lash_tui::{Color, Column, ColumnWidth, Frame, Line, Rect, Style, Table, TableRow, TableState};
use lash_tui_extensions::{
    TuiExtension, TuiExtensions, TuiHostEffect, TuiRenderContext, TuiSurfaceSize, TuiSurfaceSlot,
    TuiSurfaceSpec,
};

use crate::SkillCatalog;
use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ExplorationOp, ExplorationOpKind,
};
use crate::app::{App, FollowOutputMode, PreparedTurn, TextSelection, UiTimelineItem};
use crate::cli_support::apply_ui_host_effects;
use crate::render;

use super::scenarios::{BENCH_HEIGHT, BENCH_WIDTH, UiPerfScenario, UiPerfWorkload};

pub(crate) struct UiPerfHarness {
    pub(crate) app: App,
    surface_state: Option<Arc<SurfaceBenchmarkState>>,
    workload: UiPerfWorkload,
}

impl UiPerfHarness {
    pub(crate) fn total_content_rows(&mut self, history_height: usize) -> usize {
        match &self.surface_state {
            Some(_) => self.workload.surface_row_count,
            None => self.app.total_content_height(
                render::history_area(&self.app, BENCH_WIDTH, BENCH_HEIGHT).width as usize,
                history_height,
            ),
        }
    }

    pub(crate) fn reset_scroll(&mut self) {
        self.app.follow_mode = FollowOutputMode::Paused;
        self.app.scroll_offset = 0;
        if let Some(state) = &self.surface_state {
            state.set_scroll(0);
            state.set_selected(0);
        }
    }

    pub(crate) fn can_scroll_forward(
        &self,
        history_height: usize,
        total_content_rows: usize,
    ) -> bool {
        match &self.surface_state {
            Some(state) => {
                state.scroll() + surface_body_height(history_height) < total_content_rows
            }
            None => self.app.scroll_offset + history_height < total_content_rows,
        }
    }

    pub(crate) fn can_scroll_backward(&self) -> bool {
        match &self.surface_state {
            Some(state) => state.scroll() > 0,
            None => self.app.scroll_offset > 0,
        }
    }

    pub(crate) fn scroll_forward(
        &mut self,
        delta: usize,
        history_height: usize,
        history_width: usize,
        total_content_rows: usize,
    ) {
        if let Some(state) = &self.surface_state {
            let body_height = surface_body_height(history_height);
            let max_scroll = total_content_rows.saturating_sub(body_height);
            let next = (state.scroll() + delta).min(max_scroll);
            state.set_scroll(next);
            state.set_selected((next + body_height / 3).min(total_content_rows.saturating_sub(1)));
            return;
        }
        self.app.scroll_down(delta, history_height, history_width);
    }

    pub(crate) fn scroll_backward(
        &mut self,
        delta: usize,
        history_height: usize,
        _history_width: usize,
        total_content_rows: usize,
    ) {
        if let Some(state) = &self.surface_state {
            let body_height = surface_body_height(history_height);
            let next = state.scroll().saturating_sub(delta);
            state.set_scroll(next);
            state.set_selected((next + body_height / 3).min(total_content_rows.saturating_sub(1)));
            return;
        }
        self.app.scroll_up(delta);
    }

    pub(crate) fn prepare_selection(&mut self, history_height: usize, total_content_rows: usize) {
        self.reset_scroll();
        if let Some(state) = &self.surface_state {
            state.set_selected(
                (surface_body_height(history_height) / 2).min(total_content_rows.saturating_sub(1)),
            );
            return;
        }
        let selection_end_row = total_content_rows.saturating_sub(2);
        self.app.selection = TextSelection {
            anchor: (0, history_height / 2),
            end: (BENCH_WIDTH.saturating_sub(2), selection_end_row),
            active: false,
            visible: true,
        };
    }

    pub(crate) fn advance_selection(
        &mut self,
        delta: usize,
        history_height: usize,
        history_width: usize,
        total_content_rows: usize,
    ) {
        if let Some(state) = &self.surface_state {
            let body_height = surface_body_height(history_height);
            let max_scroll = total_content_rows.saturating_sub(body_height);
            let next = if state.scroll() + body_height >= total_content_rows {
                0
            } else {
                (state.scroll() + delta).min(max_scroll)
            };
            state.set_scroll(next);
            state.set_selected((next + body_height / 2).min(total_content_rows.saturating_sub(1)));
            return;
        }
        self.app.scroll_down(delta, history_height, history_width);
        if self.app.scroll_offset + history_height >= total_content_rows {
            self.app.scroll_offset = 0;
        }
    }
}

#[derive(Default)]
struct SurfaceBenchmarkState {
    scroll: AtomicUsize,
    selected: AtomicUsize,
}

impl SurfaceBenchmarkState {
    fn scroll(&self) -> usize {
        self.scroll.load(Ordering::Relaxed)
    }

    fn set_scroll(&self, value: usize) {
        self.scroll.store(value, Ordering::Relaxed);
    }

    fn selected(&self) -> usize {
        self.selected.load(Ordering::Relaxed)
    }

    fn set_selected(&self, value: usize) {
        self.selected.store(value, Ordering::Relaxed);
    }
}

struct SurfaceBenchmarkTuiExtension {
    table: Table<'static>,
    state: Arc<SurfaceBenchmarkState>,
}

#[async_trait::async_trait]
impl TuiExtension for SurfaceBenchmarkTuiExtension {
    fn id(&self) -> &'static str {
        "ui_perf_surface"
    }

    async fn invoke_action(
        &self,
        _action: &str,
        _arg: Option<&str>,
        _ctx: lash_tui_extensions::TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String> {
        Ok(Vec::new())
    }

    fn render_surface(&self, surface_key: &str, ctx: TuiRenderContext<'_>, frame: &mut Frame<'_>) {
        match surface_key {
            "workspace" => {
                let mut state = TableState::default();
                state.scroll.offset = self.state.scroll();
                state.selection.selected = Some(self.state.selected());
                state.focused = ctx.focused;
                self.table.render(frame, &mut state);
            }
            "footer" => {
                let label = format!(
                    " j/k scroll  enter inspect  rows {}  sel {} ",
                    self.state.scroll(),
                    self.state.selected()
                );
                frame.fill(
                    frame.area(),
                    ' ',
                    Style::default().bg(Color::rgb(20, 23, 30)),
                );
                frame.write_text(
                    0,
                    0,
                    &label,
                    Style::default().fg(Color::rgb(200, 208, 220)),
                    frame.area().width,
                );
            }
            "overlay" => {
                let area = frame.area();
                frame.draw_box(
                    Rect::new(0, 0, area.width, area.height),
                    Style::default().fg(Color::rgb(180, 184, 195)),
                    Some(Style::default().bg(Color::rgb(17, 19, 25))),
                );
                frame.write_text(
                    2,
                    1.min(area.height.saturating_sub(1)),
                    "Surface profiler overlay",
                    Style::default().fg(Color::rgb(230, 232, 239)),
                    area.width.saturating_sub(4),
                );
            }
            _ => {}
        }
    }

    fn handle_turn_event(&self, event: &lash_core::TurnEvent) -> Vec<TuiHostEffect> {
        match event {
            lash_core::TurnEvent::AssistantProseDelta { text } if text == "ui_perf_mount" => vec![
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
                TuiHostEffect::FocusSurface {
                    key: "workspace".to_string(),
                },
            ],
            lash_core::TurnEvent::AssistantProseDelta { text } if text == "ui_perf_overlay" => {
                vec![
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "overlay".to_string(),
                            slot: TuiSurfaceSlot::Overlay,
                            size: TuiSurfaceSize::Fixed {
                                width: 36,
                                height: 4,
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
                ]
            }
            _ => Vec::new(),
        }
    }
}

pub(crate) fn build_benchmark_harness(
    scenario: UiPerfScenario,
    workload: UiPerfWorkload,
) -> UiPerfHarness {
    let mut app = build_benchmark_app(workload.turn_count);
    let surface_state = match scenario {
        UiPerfScenario::HistoryRender => None,
        UiPerfScenario::WorkspaceSurface | UiPerfScenario::WorkspaceOverlay => {
            let state = Arc::new(SurfaceBenchmarkState::default());
            let extension = Arc::new(SurfaceBenchmarkTuiExtension {
                table: build_surface_table(workload.surface_row_count),
                state: Arc::clone(&state),
            });
            let ui_extensions = Arc::new(
                TuiExtensions::new(vec![extension]).expect("surface benchmark extensions"),
            );
            apply_ui_host_effects(
                &mut app,
                ui_extensions.effects_for_turn_event(&lash_core::TurnEvent::AssistantProseDelta {
                    text: "ui_perf_mount".to_string(),
                }),
            );
            if scenario == UiPerfScenario::WorkspaceOverlay {
                apply_ui_host_effects(
                    &mut app,
                    ui_extensions.effects_for_turn_event(
                        &lash_core::TurnEvent::AssistantProseDelta {
                            text: "ui_perf_overlay".to_string(),
                        },
                    ),
                );
            }
            app.set_ui_extensions(ui_extensions);
            Some(state)
        }
        UiPerfScenario::StreamingReactor
        | UiPerfScenario::SlowSnapshot
        | UiPerfScenario::FileIndexStorm => None,
    };

    UiPerfHarness {
        app,
        surface_state,
        workload,
    }
}

pub(crate) fn build_benchmark_app(turn_count: usize) -> App {
    let mut app = App::new(
        "gpt-5.4".to_string(),
        "ui-perf".to_string(),
        "test-session-id".into(),
    );
    app.timeline.clear();
    app.token_usage = TokenUsage {
        input_tokens: 208_000,
        output_tokens: 11_500,
        cached_input_tokens: 0,
        reasoning_tokens: 0,
    };
    app.context_window = Some(1_100_000);
    app.model_variant = Some("high".to_string());

    let skills = SkillCatalog::default();
    for turn in 0..turn_count {
        let turn_label = format!("turn-{turn}");
        let turn = PreparedTurn::prepare_with_large_pastes(
            format!(
                "Investigate render-path regression batch {turn} and summarize the visible state."
            ),
            Vec::new(),
            &skills,
            Vec::new(),
        );
        app.push_prepared_user_input(&turn);
        app.timeline
            .push(UiTimelineItem::AssistantText(long_assistant_text(
                turn.preview().as_str(),
            )));
        if turn.draft_id.is_empty() {
            unreachable!("prepared turns should always have a draft id");
        }
        app.timeline
            .push(UiTimelineItem::Activity(Box::new(exploration_activity(
                turn.preview().as_str(),
                turn.display_text.as_str(),
            ))));
        if turn.display_text.len().is_multiple_of(3) {
            app.timeline
                .push(UiTimelineItem::Activity(Box::new(snippet_activity(
                    &turn_label,
                    false,
                ))));
        }
        if turn.display_text.len().is_multiple_of(5) {
            app.timeline
                .push(UiTimelineItem::Activity(Box::new(snippet_activity(
                    &turn_label,
                    true,
                ))));
        }
    }
    app.invalidate_height_cache();
    app
}

fn build_surface_table(row_count: usize) -> Table<'static> {
    let rows = (0..row_count)
        .map(|index| {
            TableRow::new(vec![
                Line::from(format!("job-{index:04}")).into(),
                if index % 11 == 0 {
                    "blocked".into()
                } else if index % 5 == 0 {
                    "running".into()
                } else {
                    "ready".into()
                },
                Line::from(format!("batch {}", index / 8)).into(),
                Line::from(format!("render path audit row {index}")).into(),
            ])
        })
        .collect();
    Table {
        columns: vec![
            Column {
                header: "job".into(),
                width: ColumnWidth::Length(10),
            },
            Column {
                header: "status".into(),
                width: ColumnWidth::Length(9),
            },
            Column {
                header: "batch".into(),
                width: ColumnWidth::Length(10),
            },
            Column {
                header: "summary".into(),
                width: ColumnWidth::Fill(1),
            },
        ],
        rows,
        header_style: Style::default()
            .bg(Color::rgb(32, 36, 44))
            .fg(Color::rgb(230, 232, 239)),
        row_style: Style::default().fg(Color::rgb(204, 208, 218)),
        selected_row_style: Style::default().bg(Color::rgb(44, 50, 62)),
        focused_selected_row_style: Style::default().bg(Color::rgb(58, 74, 96)),
    }
}

fn surface_body_height(history_height: usize) -> usize {
    history_height.saturating_sub(1).max(1)
}

fn long_assistant_text(subject: &str) -> String {
    format!(
        "I traced the live render path for {subject} and narrowed the current cost centers.\n\n\
- The history viewport still projects and wraps visible rows on demand.\n\
- Scroll-heavy sessions amplify any repeated line shaping work.\n\
- Selection highlighting compounds that when wide spans are repainted cell by cell.\n\n\
Next I’m using the synthetic UI benchmark workload to lock those costs down and verify each simplification against the same scroll/render path.\n\n\
### Current assessment\n\
- The compact history feed is stable but still layout-heavy.\n\
- Snippet previews and markdown sections are the best stress case for repeated shaping.\n\
- The next pass should reuse block layouts instead of regenerating them on scroll."
    )
}

pub(crate) fn streaming_delta_text(units: usize) -> String {
    format!(
        "stream batch with {units} coalesced text and reasoning deltas; foreground input remains priority"
    )
}

fn exploration_activity(subject: &str, detail_seed: &str) -> ActivityBlock {
    ActivityBlock::new(
        ActivityKind::Exploration,
        "grep",
        serde_json::json!({}),
        "Explored",
        ActivityStatus::Completed,
        serde_json::json!({}),
        13,
    )
    .with_detail_lines(vec![
        format!("Search \"render cache|height cache|selection\" in {detail_seed}"),
        format!("Read src/render/mod.rs for {subject}"),
        "Read src/app/view.rs for cumulative height math".to_string(),
        "Read src/scratch_tui.rs for selection painting".to_string(),
    ])
    .with_extra(Some(crate::activity::ActivityExtra::Exploration(vec![
        ExplorationOp {
            kind: ExplorationOpKind::Search,
            subject: subject.to_string(),
        },
        ExplorationOp {
            kind: ExplorationOpKind::Read,
            subject: "src/render/mod.rs".to_string(),
        },
        ExplorationOp {
            kind: ExplorationOpKind::Read,
            subject: "src/app/view.rs".to_string(),
        },
        ExplorationOp {
            kind: ExplorationOpKind::Read,
            subject: "src/scratch_tui.rs".to_string(),
        },
    ])))
}

fn snippet_activity(subject: &str, markdown: bool) -> ActivityBlock {
    let content = if markdown {
        "## Render cache checklist\n\n- Cache rendered block lines by width and expand level.\n- Make height cache read lengths from that shared render cache.\n- Stop rebuilding wrapped markdown lines on every scroll tick.\n".to_string()
    } else {
        "pub(crate) fn draw_history(frame: &mut Frame<'_>, app: &mut App, area: Rect) {\n    // synthetic preview payload for UI perf benchmark\n    // the real benchmark exercises the renderer and wrapping path\n    // with large code-like panels visible in the viewport\n    let viewport_height = area.height as usize;\n    let viewport_width = area.width as usize;\n    let _ = (viewport_height, viewport_width);\n}\n".to_string()
    };
    ActivityBlock::new(
        ActivityKind::GenericTool,
        "preview_text",
        serde_json::json!({}),
        format!("preview render/mod.rs:120-164 for {subject}"),
        ActivityStatus::Completed,
        serde_json::json!({}),
        7,
    )
    .with_detail_lines(vec![
        "preview crates/lash-cli/src/render/mod.rs:120-164".to_string(),
    ])
    .with_artifact(Some(ActivityArtifact::TextPreview {
        title: Some("Render cache candidate".to_string()),
        text: content,
    }))
}
