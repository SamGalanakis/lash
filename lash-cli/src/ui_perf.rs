use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Context;
use chrono::Utc;
use lash::{SkillCatalog, TokenUsage};
use lash_tui::{
    Color, Column, ColumnWidth, Frame, Line, PerfCounters, PerfPhase, Rect, Style, Table, TableRow,
    TableState,
};
use lash_ui::{
    UiExtension, UiExtensions, UiHostEffect, UiRenderContext, UiSurfaceSize, UiSurfaceSlot,
    UiSurfaceSpec,
};
use serde::Serialize;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ExplorationOp, ExplorationOpKind,
};
use crate::app::{App, FollowOutputMode, PreparedTurn, TextSelection, UiTimelineItem};
use crate::cli_support::apply_ui_host_effects;
use crate::render;
use crate::ui_trace::render_screen_snapshot_with_perf;

const BENCH_WIDTH: u16 = 220;
const BENCH_HEIGHT: u16 = 72;
const TURN_COUNT: usize = 480;
const SCROLL_DELTA: usize = 3;
const SCROLL_PASSES: usize = 2;
const SELECTION_SCROLL_DELTA: usize = 2;
const SELECTION_FRAMES: usize = 320;
const SURFACE_ROW_COUNT: usize = 1_600;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UiPerfScenario {
    History,
    Workspace,
    WorkspaceOverlay,
}

impl UiPerfScenario {
    const ALL: [Self; 3] = [Self::History, Self::Workspace, Self::WorkspaceOverlay];
}

#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct UiPerfFramePerf {
    render_build_ms: f64,
    diff_scan_ms: f64,
    changed_rows: u64,
    changed_cells: u64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct UiPerfFramePerfAggregate {
    render_build_avg_ms: f64,
    render_build_max_ms: f64,
    diff_scan_avg_ms: f64,
    diff_scan_max_ms: f64,
    changed_rows_avg: f64,
    changed_cells_avg: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfRunResult {
    build_case_ms: f64,
    initial_render_ms: f64,
    initial_perf: UiPerfFramePerf,
    height_cache_rebuild_ms: f64,
    scroll_render_total_ms: f64,
    scroll_render_avg_ms: f64,
    scroll_render_max_ms: f64,
    scroll_frames: usize,
    scroll_perf: UiPerfFramePerfAggregate,
    selection_render_total_ms: f64,
    selection_render_avg_ms: f64,
    selection_render_max_ms: f64,
    selection_frames: usize,
    selection_perf: UiPerfFramePerfAggregate,
    total_blocks: usize,
    total_content_rows: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfMetricSummary {
    min: f64,
    median: f64,
    max: f64,
    mean: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfSummary {
    runs: usize,
    build_case_ms: UiPerfMetricSummary,
    initial_render_ms: UiPerfMetricSummary,
    initial_render_build_ms: UiPerfMetricSummary,
    initial_diff_scan_ms: UiPerfMetricSummary,
    height_cache_rebuild_ms: UiPerfMetricSummary,
    scroll_render_avg_ms: UiPerfMetricSummary,
    scroll_render_max_ms: UiPerfMetricSummary,
    scroll_render_build_avg_ms: UiPerfMetricSummary,
    scroll_diff_scan_avg_ms: UiPerfMetricSummary,
    selection_render_avg_ms: UiPerfMetricSummary,
    selection_render_max_ms: UiPerfMetricSummary,
    selection_render_build_avg_ms: UiPerfMetricSummary,
    selection_diff_scan_avg_ms: UiPerfMetricSummary,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfScenarioReport {
    scenario: UiPerfScenario,
    results: Vec<UiPerfRunResult>,
    summary: UiPerfSummary,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfReport {
    created_at: String,
    version: String,
    width: u16,
    height: u16,
    turn_count: usize,
    scroll_delta: usize,
    scroll_passes: usize,
    selection_frames: usize,
    warmups: usize,
    runs: usize,
    scenarios: Vec<UiPerfScenarioReport>,
}

pub(crate) fn default_output_path() -> PathBuf {
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("lash-cli crate should live under repo root")
        .join(".benchmarks")
        .join("ui-perf")
        .join(format!("{stamp}.json"))
}

pub(crate) fn run_cli(
    out: Option<PathBuf>,
    runs: usize,
    warmups: usize,
    version: &str,
) -> anyhow::Result<()> {
    let runs = runs.max(1);
    let scenarios = UiPerfScenario::ALL
        .into_iter()
        .map(|scenario| {
            for _ in 0..warmups {
                let _ = run_once(scenario);
            }

            let mut results = Vec::with_capacity(runs);
            for _ in 0..runs {
                results.push(run_once(scenario));
            }

            UiPerfScenarioReport {
                scenario,
                summary: summarize(&results),
                results,
            }
        })
        .collect::<Vec<_>>();

    let report = UiPerfReport {
        created_at: Utc::now().to_rfc3339(),
        version: version.to_string(),
        width: BENCH_WIDTH,
        height: BENCH_HEIGHT,
        turn_count: TURN_COUNT,
        scroll_delta: SCROLL_DELTA,
        scroll_passes: SCROLL_PASSES,
        selection_frames: SELECTION_FRAMES,
        warmups,
        runs,
        scenarios,
    };

    let out_path = out.unwrap_or_else(default_output_path);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create benchmark output dir {}", parent.display()))?;
    }
    fs::write(&out_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write benchmark report {}", out_path.display()))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "out": out_path,
            "scenarios": report
                .scenarios
                .iter()
                .map(|scenario| serde_json::json!({
                    "scenario": scenario.scenario,
                    "summary": scenario.summary,
                }))
                .collect::<Vec<_>>(),
        }))?
    );
    Ok(())
}

fn run_once(scenario: UiPerfScenario) -> UiPerfRunResult {
    let started = Instant::now();
    let mut harness = build_benchmark_harness(scenario);
    let build_case_ms = elapsed_ms(started);

    let initial_render_started = Instant::now();
    let (mut snapshot, initial_perf_counters) =
        render_screen_snapshot_with_perf(&mut harness.app, BENCH_WIDTH, BENCH_HEIGHT, None);
    let initial_render_ms = elapsed_ms(initial_render_started);
    let initial_perf = frame_perf_sample(&initial_perf_counters);

    let history_height = render::history_viewport_height(&harness.app, BENCH_WIDTH, BENCH_HEIGHT);
    let history_width =
        render::history_area(&harness.app, BENCH_WIDTH, BENCH_HEIGHT).width as usize;

    harness.app.invalidate_height_cache();
    let height_cache_started = Instant::now();
    let total_content_rows = harness.total_content_rows(history_height);
    let height_cache_rebuild_ms = elapsed_ms(height_cache_started);

    let mut scroll_frame_durations = Vec::new();
    let mut scroll_perf_samples = Vec::new();
    harness.reset_scroll();
    for _ in 0..SCROLL_PASSES {
        while harness.can_scroll_forward(history_height, total_content_rows) {
            let frame_started = Instant::now();
            harness.scroll_forward(
                SCROLL_DELTA,
                history_height,
                history_width,
                total_content_rows,
            );
            let (next_snapshot, perf) = render_screen_snapshot_with_perf(
                &mut harness.app,
                BENCH_WIDTH,
                BENCH_HEIGHT,
                Some(&snapshot),
            );
            snapshot = next_snapshot;
            scroll_frame_durations.push(elapsed_ms(frame_started));
            scroll_perf_samples.push(frame_perf_sample(&perf));
        }
        while harness.can_scroll_backward() {
            let frame_started = Instant::now();
            harness.scroll_backward(
                SCROLL_DELTA,
                history_height,
                history_width,
                total_content_rows,
            );
            let (next_snapshot, perf) = render_screen_snapshot_with_perf(
                &mut harness.app,
                BENCH_WIDTH,
                BENCH_HEIGHT,
                Some(&snapshot),
            );
            snapshot = next_snapshot;
            scroll_frame_durations.push(elapsed_ms(frame_started));
            scroll_perf_samples.push(frame_perf_sample(&perf));
        }
    }

    harness.prepare_selection(history_height, total_content_rows);
    snapshot =
        render_screen_snapshot_with_perf(&mut harness.app, BENCH_WIDTH, BENCH_HEIGHT, None).0;
    let mut selection_frame_durations = Vec::new();
    let mut selection_perf_samples = Vec::new();
    for _ in 0..SELECTION_FRAMES {
        let frame_started = Instant::now();
        harness.advance_selection(
            SELECTION_SCROLL_DELTA,
            history_height,
            history_width,
            total_content_rows,
        );
        let (next_snapshot, perf) = render_screen_snapshot_with_perf(
            &mut harness.app,
            BENCH_WIDTH,
            BENCH_HEIGHT,
            Some(&snapshot),
        );
        snapshot = next_snapshot;
        selection_frame_durations.push(elapsed_ms(frame_started));
        selection_perf_samples.push(frame_perf_sample(&perf));
    }

    UiPerfRunResult {
        build_case_ms,
        initial_render_ms,
        initial_perf,
        height_cache_rebuild_ms,
        scroll_render_total_ms: scroll_frame_durations.iter().sum(),
        scroll_render_avg_ms: average(&scroll_frame_durations),
        scroll_render_max_ms: max_value(&scroll_frame_durations),
        scroll_frames: scroll_frame_durations.len(),
        scroll_perf: aggregate_frame_perf(&scroll_perf_samples),
        selection_render_total_ms: selection_frame_durations.iter().sum(),
        selection_render_avg_ms: average(&selection_frame_durations),
        selection_render_max_ms: max_value(&selection_frame_durations),
        selection_frames: selection_frame_durations.len(),
        selection_perf: aggregate_frame_perf(&selection_perf_samples),
        total_blocks: harness.app.blocks.len(),
        total_content_rows,
    }
}

struct UiPerfHarness {
    app: App,
    surface_state: Option<Arc<SurfaceBenchmarkState>>,
}

impl UiPerfHarness {
    fn total_content_rows(&mut self, history_height: usize) -> usize {
        match &self.surface_state {
            Some(_) => SURFACE_ROW_COUNT,
            None => self.app.total_content_height(
                render::history_area(&self.app, BENCH_WIDTH, BENCH_HEIGHT).width as usize,
                history_height,
            ),
        }
    }

    fn reset_scroll(&mut self) {
        self.app.follow_mode = FollowOutputMode::Paused;
        self.app.scroll_offset = 0;
        if let Some(state) = &self.surface_state {
            state.set_scroll(0);
            state.set_selected(0);
        }
    }

    fn can_scroll_forward(&self, history_height: usize, total_content_rows: usize) -> bool {
        match &self.surface_state {
            Some(state) => {
                state.scroll() + surface_body_height(history_height) < total_content_rows
            }
            None => self.app.scroll_offset + history_height < total_content_rows,
        }
    }

    fn can_scroll_backward(&self) -> bool {
        match &self.surface_state {
            Some(state) => state.scroll() > 0,
            None => self.app.scroll_offset > 0,
        }
    }

    fn scroll_forward(
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

    fn scroll_backward(
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

    fn prepare_selection(&mut self, history_height: usize, total_content_rows: usize) {
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

    fn advance_selection(
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

struct SurfaceBenchmarkExtension {
    table: Table<'static>,
    state: Arc<SurfaceBenchmarkState>,
}

#[async_trait::async_trait]
impl UiExtension for SurfaceBenchmarkExtension {
    fn id(&self) -> &'static str {
        "ui_perf_surface"
    }

    async fn invoke_action(
        &self,
        _action: &str,
        _arg: Option<&str>,
        _ctx: lash_ui::UiContext<'_>,
    ) -> Result<Vec<UiHostEffect>, String> {
        Ok(Vec::new())
    }

    fn render_surface(&self, surface_key: &str, ctx: UiRenderContext<'_>, frame: &mut Frame<'_>) {
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

    fn handle_session_event(&self, event: &lash::SessionEvent) -> Vec<UiHostEffect> {
        match event {
            lash::SessionEvent::TextDelta { content } if content == "ui_perf_mount" => vec![
                UiHostEffect::MountSurface {
                    spec: UiSurfaceSpec {
                        key: "workspace".to_string(),
                        slot: UiSurfaceSlot::Workspace,
                        size: UiSurfaceSize::Auto,
                        order: 0,
                        focusable: true,
                        visible: true,
                        modal: false,
                    },
                },
                UiHostEffect::MountSurface {
                    spec: UiSurfaceSpec {
                        key: "footer".to_string(),
                        slot: UiSurfaceSlot::Footer,
                        size: UiSurfaceSize::Lines(1),
                        order: 0,
                        focusable: false,
                        visible: true,
                        modal: false,
                    },
                },
                UiHostEffect::FocusSurface {
                    key: "workspace".to_string(),
                },
            ],
            lash::SessionEvent::TextDelta { content } if content == "ui_perf_overlay" => vec![
                UiHostEffect::MountSurface {
                    spec: UiSurfaceSpec {
                        key: "overlay".to_string(),
                        slot: UiSurfaceSlot::Overlay,
                        size: UiSurfaceSize::Fixed {
                            width: 36,
                            height: 4,
                        },
                        order: 10,
                        focusable: true,
                        visible: true,
                        modal: true,
                    },
                },
                UiHostEffect::FocusSurface {
                    key: "overlay".to_string(),
                },
            ],
            _ => Vec::new(),
        }
    }
}

fn build_benchmark_harness(scenario: UiPerfScenario) -> UiPerfHarness {
    let mut app = build_benchmark_app();
    let surface_state = match scenario {
        UiPerfScenario::History => None,
        UiPerfScenario::Workspace | UiPerfScenario::WorkspaceOverlay => {
            let state = Arc::new(SurfaceBenchmarkState::default());
            let extension = Arc::new(SurfaceBenchmarkExtension {
                table: build_surface_table(),
                state: Arc::clone(&state),
            });
            let ui_extensions =
                Arc::new(UiExtensions::new(vec![extension]).expect("surface benchmark extensions"));
            apply_ui_host_effects(
                &mut app,
                ui_extensions.effects_for_session_event(&lash::SessionEvent::TextDelta {
                    content: "ui_perf_mount".to_string(),
                }),
            );
            if scenario == UiPerfScenario::WorkspaceOverlay {
                apply_ui_host_effects(
                    &mut app,
                    ui_extensions.effects_for_session_event(&lash::SessionEvent::TextDelta {
                        content: "ui_perf_overlay".to_string(),
                    }),
                );
            }
            app.set_ui_extensions(ui_extensions);
            Some(state)
        }
    };

    UiPerfHarness { app, surface_state }
}

fn build_benchmark_app() -> App {
    let mut app = App::new(
        "gpt-5.4".to_string(),
        "ui-perf".to_string(),
        "test-session-id".into(),
    );
    app.blocks.clear();
    app.token_usage = TokenUsage {
        input_tokens: 208_000,
        output_tokens: 11_500,
        cached_input_tokens: 0,
        reasoning_tokens: 0,
    };
    app.context_window = Some(1_100_000);
    app.model_variant = Some("high".to_string());

    let skills = SkillCatalog::default();
    for turn in 0..TURN_COUNT {
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
        app.blocks
            .push(UiTimelineItem::AssistantText(long_assistant_text(
                turn.preview().as_str(),
            )));
        if turn.draft_id.is_empty() {
            unreachable!("prepared turns should always have a draft id");
        }
        app.blocks
            .push(UiTimelineItem::Activity(Box::new(exploration_activity(
                turn.preview().as_str(),
                turn.display_text.as_str(),
            ))));
        if turn.display_text.len().is_multiple_of(3) {
            app.blocks
                .push(UiTimelineItem::Activity(Box::new(snippet_activity(
                    &turn_label,
                    false,
                ))));
        }
        if turn.display_text.len().is_multiple_of(5) {
            app.blocks
                .push(UiTimelineItem::Activity(Box::new(snippet_activity(
                    &turn_label,
                    true,
                ))));
        }
    }
    app.invalidate_height_cache();
    app
}

fn build_surface_table() -> Table<'static> {
    let rows = (0..SURFACE_ROW_COUNT)
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
        "preview lash-cli/src/render/mod.rs:120-164".to_string(),
    ])
    .with_artifact(Some(ActivityArtifact::TextPreview {
        title: Some("Render cache candidate".to_string()),
        text: content,
    }))
}

fn summarize(results: &[UiPerfRunResult]) -> UiPerfSummary {
    UiPerfSummary {
        runs: results.len(),
        build_case_ms: metric_summary(results.iter().map(|run| run.build_case_ms).collect()),
        initial_render_ms: metric_summary(
            results.iter().map(|run| run.initial_render_ms).collect(),
        ),
        initial_render_build_ms: metric_summary(
            results
                .iter()
                .map(|run| run.initial_perf.render_build_ms)
                .collect(),
        ),
        initial_diff_scan_ms: metric_summary(
            results
                .iter()
                .map(|run| run.initial_perf.diff_scan_ms)
                .collect(),
        ),
        height_cache_rebuild_ms: metric_summary(
            results
                .iter()
                .map(|run| run.height_cache_rebuild_ms)
                .collect(),
        ),
        scroll_render_avg_ms: metric_summary(
            results.iter().map(|run| run.scroll_render_avg_ms).collect(),
        ),
        scroll_render_max_ms: metric_summary(
            results.iter().map(|run| run.scroll_render_max_ms).collect(),
        ),
        scroll_render_build_avg_ms: metric_summary(
            results
                .iter()
                .map(|run| run.scroll_perf.render_build_avg_ms)
                .collect(),
        ),
        scroll_diff_scan_avg_ms: metric_summary(
            results
                .iter()
                .map(|run| run.scroll_perf.diff_scan_avg_ms)
                .collect(),
        ),
        selection_render_avg_ms: metric_summary(
            results
                .iter()
                .map(|run| run.selection_render_avg_ms)
                .collect(),
        ),
        selection_render_max_ms: metric_summary(
            results
                .iter()
                .map(|run| run.selection_render_max_ms)
                .collect(),
        ),
        selection_render_build_avg_ms: metric_summary(
            results
                .iter()
                .map(|run| run.selection_perf.render_build_avg_ms)
                .collect(),
        ),
        selection_diff_scan_avg_ms: metric_summary(
            results
                .iter()
                .map(|run| run.selection_perf.diff_scan_avg_ms)
                .collect(),
        ),
    }
}

fn frame_perf_sample(perf: &PerfCounters) -> UiPerfFramePerf {
    UiPerfFramePerf {
        render_build_ms: nanos_to_ms(perf.phase(PerfPhase::RenderBuild).total_nanos),
        diff_scan_ms: nanos_to_ms(perf.phase(PerfPhase::DiffScan).total_nanos),
        changed_rows: perf.frame.changed_rows,
        changed_cells: perf.frame.changed_cells,
    }
}

fn aggregate_frame_perf(samples: &[UiPerfFramePerf]) -> UiPerfFramePerfAggregate {
    UiPerfFramePerfAggregate {
        render_build_avg_ms: average(
            &samples
                .iter()
                .map(|sample| sample.render_build_ms)
                .collect::<Vec<_>>(),
        ),
        render_build_max_ms: max_value(
            &samples
                .iter()
                .map(|sample| sample.render_build_ms)
                .collect::<Vec<_>>(),
        ),
        diff_scan_avg_ms: average(
            &samples
                .iter()
                .map(|sample| sample.diff_scan_ms)
                .collect::<Vec<_>>(),
        ),
        diff_scan_max_ms: max_value(
            &samples
                .iter()
                .map(|sample| sample.diff_scan_ms)
                .collect::<Vec<_>>(),
        ),
        changed_rows_avg: round3(
            samples
                .iter()
                .map(|sample| sample.changed_rows as f64)
                .sum::<f64>()
                / samples.len().max(1) as f64,
        ),
        changed_cells_avg: round3(
            samples
                .iter()
                .map(|sample| sample.changed_cells as f64)
                .sum::<f64>()
                / samples.len().max(1) as f64,
        ),
    }
}

fn metric_summary(mut values: Vec<f64>) -> UiPerfMetricSummary {
    values.sort_by(f64::total_cmp);
    let median = if values.len().is_multiple_of(2) {
        let upper = values.len() / 2;
        (values[upper - 1] + values[upper]) / 2.0
    } else {
        values[values.len() / 2]
    };
    UiPerfMetricSummary {
        min: round3(*values.first().unwrap_or(&0.0)),
        median: round3(median),
        max: round3(*values.last().unwrap_or(&0.0)),
        mean: round3(values.iter().sum::<f64>() / values.len().max(1) as f64),
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    round3(started.elapsed().as_secs_f64() * 1000.0)
}

fn nanos_to_ms(nanos: u64) -> f64 {
    round3(nanos as f64 / 1_000_000.0)
}

fn average(values: &[f64]) -> f64 {
    round3(values.iter().sum::<f64>() / values.len().max(1) as f64)
}

fn max_value(values: &[f64]) -> f64 {
    round3(values.iter().copied().fold(0.0, f64::max))
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_ui_perf_benchmark_produces_consistent_shape() {
        for scenario in UiPerfScenario::ALL {
            let run = run_once(scenario);
            assert!(run.total_blocks >= TURN_COUNT * 2);
            assert!(run.total_content_rows > BENCH_HEIGHT as usize);
            assert!(run.scroll_frames > 0);
            assert_eq!(run.selection_frames, SELECTION_FRAMES);
            assert!(run.initial_perf.render_build_ms >= 0.0);
        }
    }
}
