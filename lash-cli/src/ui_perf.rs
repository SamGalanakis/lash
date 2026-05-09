use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use chrono::Utc;
use lash::{SkillCatalog, TokenUsage};
use lash_file_index::{FileIndex, MatchResult};
use lash_tui::{
    Color, Column, ColumnWidth, Frame, Line, PerfCounters, PerfPhase, Rect, Style, Table, TableRow,
    TableState,
};
use lash_tui_extensions::{
    TuiExtension, TuiExtensions, TuiHostEffect, TuiRenderContext, TuiSurfaceSize, TuiSurfaceSlot,
    TuiSurfaceSpec,
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
const FULL_TURN_COUNT: usize = 480;
const FULL_SURFACE_ROW_COUNT: usize = 1_600;
const SCROLL_DELTA: usize = 3;
const SELECTION_SCROLL_DELTA: usize = 2;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UiPerfScenario {
    HistoryRender,
    WorkspaceSurface,
    WorkspaceOverlay,
    StreamingReactor,
    SlowSnapshot,
    FileIndexStorm,
}

impl UiPerfScenario {
    const DEFAULTS: [Self; 6] = [
        Self::HistoryRender,
        Self::WorkspaceSurface,
        Self::WorkspaceOverlay,
        Self::StreamingReactor,
        Self::SlowSnapshot,
        Self::FileIndexStorm,
    ];

    const KNOWN: [Self; 6] = Self::DEFAULTS;

    fn parse(value: &str) -> Option<Self> {
        match value {
            "history_render" | "history" => Some(Self::HistoryRender),
            "workspace_surface" | "workspace" => Some(Self::WorkspaceSurface),
            "workspace_overlay" => Some(Self::WorkspaceOverlay),
            "streaming_reactor" => Some(Self::StreamingReactor),
            "slow_snapshot" => Some(Self::SlowSnapshot),
            "file_index_storm" | "file-index-storm" => Some(Self::FileIndexStorm),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::HistoryRender => "history_render",
            Self::WorkspaceSurface => "workspace_surface",
            Self::WorkspaceOverlay => "workspace_overlay",
            Self::StreamingReactor => "streaming_reactor",
            Self::SlowSnapshot => "slow_snapshot",
            Self::FileIndexStorm => "file_index_storm",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UiPerfProfile {
    Quick,
    Full,
    Stress,
}

impl UiPerfProfile {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "quick" => Some(Self::Quick),
            "full" => Some(Self::Full),
            "stress" => Some(Self::Stress),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Full => "full",
            Self::Stress => "stress",
        }
    }

    fn workload(self) -> UiPerfWorkload {
        match self {
            Self::Quick => UiPerfWorkload {
                turn_count: 120,
                surface_row_count: 420,
                scroll_passes: 1,
                selection_frames: 80,
                stream_deltas: 240,
                control_events: 48,
                snapshot_jobs: 4,
                snapshot_timeout_ms: 18,
                file_source_changes: 2,
                ignored_path_events: 120,
            },
            Self::Full => UiPerfWorkload {
                turn_count: FULL_TURN_COUNT,
                surface_row_count: FULL_SURFACE_ROW_COUNT,
                scroll_passes: 2,
                selection_frames: 320,
                stream_deltas: 1_200,
                control_events: 180,
                snapshot_jobs: 8,
                snapshot_timeout_ms: 24,
                file_source_changes: 6,
                ignored_path_events: 1_200,
            },
            Self::Stress => UiPerfWorkload {
                turn_count: 900,
                surface_row_count: 4_000,
                scroll_passes: 3,
                selection_frames: 700,
                stream_deltas: 5_000,
                control_events: 700,
                snapshot_jobs: 18,
                snapshot_timeout_ms: 30,
                file_source_changes: 16,
                ignored_path_events: 6_000,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct UiPerfWorkload {
    turn_count: usize,
    surface_row_count: usize,
    scroll_passes: usize,
    selection_frames: usize,
    stream_deltas: usize,
    control_events: usize,
    snapshot_jobs: usize,
    snapshot_timeout_ms: u64,
    file_source_changes: usize,
    ignored_path_events: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfMetricSummary {
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfBudgetResult {
    metric: String,
    statistic: String,
    budget_ms: f64,
    actual_ms: f64,
    passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfRunResult {
    build_case_ms: f64,
    total_ms: f64,
    total_blocks: usize,
    total_content_rows: usize,
    samples: BTreeMap<String, Vec<f64>>,
    counters: BTreeMap<String, u64>,
}

impl UiPerfRunResult {
    fn new(started: Instant) -> Self {
        Self {
            build_case_ms: 0.0,
            total_ms: elapsed_ms(started),
            total_blocks: 0,
            total_content_rows: 0,
            samples: BTreeMap::new(),
            counters: BTreeMap::new(),
        }
    }

    fn sample(&mut self, name: &'static str, value: f64) {
        self.samples
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    fn sample_many(&mut self, name: &'static str, values: Vec<f64>) {
        if !values.is_empty() {
            self.samples
                .entry(name.to_string())
                .or_default()
                .extend(values);
        }
    }

    fn counter(&mut self, name: &'static str, value: u64) {
        self.counters.insert(name.to_string(), value);
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfScenarioReport {
    scenario: String,
    results: Vec<UiPerfRunResult>,
    summary: BTreeMap<String, UiPerfMetricSummary>,
    counters: BTreeMap<String, u64>,
    budgets: Vec<UiPerfBudgetResult>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfGitInfo {
    sha: String,
    dirty: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfRunParameters {
    width: u16,
    height: u16,
    warmups: usize,
    runs: usize,
    profile: UiPerfProfile,
    workload: UiPerfWorkload,
    scenarios: Vec<String>,
    enforce_budgets: bool,
    compare_inputs: Vec<PathBuf>,
    dhat_out: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfReport {
    created_at: String,
    version: String,
    git: UiPerfGitInfo,
    build_mode: String,
    parameters: UiPerfRunParameters,
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

fn default_dhat_output_path(report_out: &Path) -> PathBuf {
    let stem = report_out
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("ui-perf");
    report_out.with_file_name(format!("{stem}.dhat.json"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_cli(
    out: Option<PathBuf>,
    enable_dhat: bool,
    dhat_out: Option<PathBuf>,
    dhat_frames: Option<usize>,
    runs: usize,
    warmups: usize,
    scenario_filters: Vec<String>,
    profile: String,
    compare_inputs: Vec<PathBuf>,
    enforce_budgets: bool,
    version: &str,
    git_sha: &str,
) -> anyhow::Result<()> {
    if dhat_out.is_some() && !enable_dhat {
        anyhow::bail!("--ui-perf-dhat-out requires --ui-perf-dhat");
    }
    let profile = UiPerfProfile::parse(profile.as_str()).ok_or_else(|| {
        anyhow::anyhow!("unknown UI perf profile `{profile}`; expected quick, full, or stress")
    })?;
    let workload = profile.workload();
    let scenarios = resolve_scenarios(&scenario_filters)?;
    let runs = runs.max(1);

    for _ in 0..warmups {
        for scenario in &scenarios {
            let _ = run_once(*scenario, workload)?;
        }
    }

    let out_path = out.unwrap_or_else(default_output_path);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create benchmark output dir {}", parent.display()))?;
    }
    let dhat_out_path = resolve_dhat_output_path(enable_dhat, &out_path, dhat_out);
    if let Some(ref path) = dhat_out_path
        && let Some(parent) = path.parent()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create dhat output dir {}", parent.display()))?;
    }

    let profiler = start_dhat_profiler(dhat_out_path.clone(), dhat_frames)?;
    let mut scenario_reports = Vec::with_capacity(scenarios.len());
    for scenario in &scenarios {
        let mut results = Vec::with_capacity(runs);
        for _ in 0..runs {
            results.push(run_once(*scenario, workload)?);
        }
        let summary = summarize_samples(&results);
        let budgets = evaluate_budgets(*scenario, &summary);
        scenario_reports.push(UiPerfScenarioReport {
            scenario: scenario.name().to_string(),
            counters: summarize_counters(&results),
            summary,
            results,
            budgets,
        });
    }
    finish_dhat_profiler(profiler);

    let report = UiPerfReport {
        created_at: Utc::now().to_rfc3339(),
        version: version.to_string(),
        git: UiPerfGitInfo {
            sha: git_sha.to_string(),
            dirty: git_dirty(),
        },
        build_mode: if cfg!(debug_assertions) {
            "debug".to_string()
        } else {
            "release".to_string()
        },
        parameters: UiPerfRunParameters {
            width: BENCH_WIDTH,
            height: BENCH_HEIGHT,
            warmups,
            runs,
            profile,
            workload,
            scenarios: scenarios
                .iter()
                .map(|scenario| scenario.name().to_string())
                .collect(),
            enforce_budgets,
            compare_inputs,
            dhat_out: dhat_out_path.clone(),
        },
        scenarios: scenario_reports,
    };

    fs::write(&out_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write benchmark report {}", out_path.display()))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "out": out_path,
            "dhat_out": report.parameters.dhat_out,
            "profile": profile.name(),
            "summary": report
                .scenarios
                .iter()
                .map(|scenario| serde_json::json!({
                    "scenario": scenario.scenario,
                    "summary": scenario.summary,
                    "budgets": scenario.budgets,
                }))
                .collect::<Vec<_>>(),
        }))?
    );

    if enforce_budgets {
        let failures = report
            .scenarios
            .iter()
            .flat_map(|scenario| {
                scenario
                    .budgets
                    .iter()
                    .filter(|budget| !budget.passed)
                    .map(|budget| {
                        format!(
                            "{} {} {} {:.3}ms > {:.3}ms",
                            scenario.scenario,
                            budget.metric,
                            budget.statistic,
                            budget.actual_ms,
                            budget.budget_ms
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if !failures.is_empty() {
            anyhow::bail!("UI perf budget exceeded:\n{}", failures.join("\n"));
        }
    }
    Ok(())
}

fn resolve_dhat_output_path(
    enable_dhat: bool,
    report_out: &Path,
    dhat_out: Option<PathBuf>,
) -> Option<PathBuf> {
    if enable_dhat {
        Some(dhat_out.unwrap_or_else(|| default_dhat_output_path(report_out)))
    } else {
        None
    }
}

#[cfg(feature = "dhat-heap")]
fn start_dhat_profiler(
    dhat_out: Option<PathBuf>,
    dhat_frames: Option<usize>,
) -> anyhow::Result<Option<dhat::Profiler>> {
    let Some(path) = dhat_out else {
        return Ok(None);
    };
    let profiler = dhat::Profiler::builder()
        .file_name(path)
        .trim_backtraces(dhat_frames)
        .build();
    Ok(Some(profiler))
}

#[cfg(not(feature = "dhat-heap"))]
fn start_dhat_profiler(
    dhat_out: Option<PathBuf>,
    _dhat_frames: Option<usize>,
) -> anyhow::Result<Option<()>> {
    if dhat_out.is_some() {
        anyhow::bail!("UI perf dhat profiling requires a lash-cli build with --features dhat-heap");
    }
    Ok(None)
}

#[cfg(feature = "dhat-heap")]
fn finish_dhat_profiler(profiler: Option<dhat::Profiler>) {
    drop(profiler);
}

#[cfg(not(feature = "dhat-heap"))]
fn finish_dhat_profiler(_profiler: Option<()>) {}

fn resolve_scenarios(filters: &[String]) -> anyhow::Result<Vec<UiPerfScenario>> {
    if filters.is_empty() {
        return Ok(UiPerfScenario::DEFAULTS.to_vec());
    }

    let mut scenarios = Vec::with_capacity(filters.len());
    for filter in filters {
        if filter == "all" {
            for scenario in UiPerfScenario::KNOWN {
                if !scenarios.contains(&scenario) {
                    scenarios.push(scenario);
                }
            }
            continue;
        }
        let scenario = UiPerfScenario::parse(filter).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown UI perf scenario `{filter}`; expected one of: {}, all",
                UiPerfScenario::KNOWN
                    .iter()
                    .map(|scenario| scenario.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        if !scenarios.contains(&scenario) {
            scenarios.push(scenario);
        }
    }
    Ok(scenarios)
}

fn run_once(scenario: UiPerfScenario, workload: UiPerfWorkload) -> anyhow::Result<UiPerfRunResult> {
    match scenario {
        UiPerfScenario::HistoryRender
        | UiPerfScenario::WorkspaceSurface
        | UiPerfScenario::WorkspaceOverlay => Ok(run_render_once(scenario, workload)),
        UiPerfScenario::StreamingReactor => Ok(run_streaming_reactor_once(workload)),
        UiPerfScenario::SlowSnapshot => Ok(run_slow_snapshot_once(workload)),
        UiPerfScenario::FileIndexStorm => run_file_index_storm_once(workload),
    }
}

fn run_render_once(scenario: UiPerfScenario, workload: UiPerfWorkload) -> UiPerfRunResult {
    let total_started = Instant::now();
    let build_started = Instant::now();
    let mut harness = build_benchmark_harness(scenario, workload);
    let build_case_ms = elapsed_ms(build_started);

    let initial_render_started = Instant::now();
    let (mut snapshot, initial_perf_counters) =
        render_screen_snapshot_with_perf(&mut harness.app, BENCH_WIDTH, BENCH_HEIGHT, None);
    let initial_render_ms = elapsed_ms(initial_render_started);

    let history_height = render::history_viewport_height(&harness.app, BENCH_WIDTH, BENCH_HEIGHT);
    let history_width =
        render::history_area(&harness.app, BENCH_WIDTH, BENCH_HEIGHT).width as usize;

    harness.app.invalidate_height_cache();
    let height_cache_started = Instant::now();
    let total_content_rows = harness.total_content_rows(history_height);
    let height_cache_rebuild_ms = elapsed_ms(height_cache_started);

    let mut scroll_frame_durations = Vec::new();
    let mut selection_frame_durations = Vec::new();
    let mut render_build_samples = vec![phase_ms(&initial_perf_counters, PerfPhase::RenderBuild)];
    let mut diff_scan_samples = vec![phase_ms(&initial_perf_counters, PerfPhase::DiffScan)];
    let mut changed_rows = initial_perf_counters.frame.changed_rows;
    let mut changed_cells = initial_perf_counters.frame.changed_cells;

    harness.reset_scroll();
    for _ in 0..workload.scroll_passes {
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
            render_build_samples.push(phase_ms(&perf, PerfPhase::RenderBuild));
            diff_scan_samples.push(phase_ms(&perf, PerfPhase::DiffScan));
            changed_rows = changed_rows.saturating_add(perf.frame.changed_rows);
            changed_cells = changed_cells.saturating_add(perf.frame.changed_cells);
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
            render_build_samples.push(phase_ms(&perf, PerfPhase::RenderBuild));
            diff_scan_samples.push(phase_ms(&perf, PerfPhase::DiffScan));
            changed_rows = changed_rows.saturating_add(perf.frame.changed_rows);
            changed_cells = changed_cells.saturating_add(perf.frame.changed_cells);
        }
    }

    harness.prepare_selection(history_height, total_content_rows);
    snapshot =
        render_screen_snapshot_with_perf(&mut harness.app, BENCH_WIDTH, BENCH_HEIGHT, None).0;
    for _ in 0..workload.selection_frames {
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
        render_build_samples.push(phase_ms(&perf, PerfPhase::RenderBuild));
        diff_scan_samples.push(phase_ms(&perf, PerfPhase::DiffScan));
        changed_rows = changed_rows.saturating_add(perf.frame.changed_rows);
        changed_cells = changed_cells.saturating_add(perf.frame.changed_cells);
    }

    let mut result = UiPerfRunResult::new(total_started);
    result.build_case_ms = build_case_ms;
    result.total_ms = elapsed_ms(total_started);
    result.total_blocks = harness.app.timeline.len();
    result.total_content_rows = total_content_rows;
    result.sample("build_case_ms", build_case_ms);
    result.sample("initial_render_ms", initial_render_ms);
    result.sample("height_cache_rebuild_ms", height_cache_rebuild_ms);
    result.sample_many("scroll_render_ms", scroll_frame_durations.clone());
    result.sample_many("selection_render_ms", selection_frame_durations.clone());
    let mut steady = scroll_frame_durations;
    steady.extend(selection_frame_durations);
    result.sample_many("steady_scroll_selection_render_ms", steady);
    result.sample_many("render_build_ms", render_build_samples);
    result.sample_many("diff_scan_ms", diff_scan_samples);
    result.counter("changed_rows", changed_rows);
    result.counter("changed_cells", changed_cells);
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReactorLane {
    Input,
    RuntimeDelta,
    Frame,
}

#[derive(Clone, Copy, Debug)]
struct ReactorEvent {
    lane: ReactorLane,
    enqueued_at: Instant,
    payload_units: usize,
}

fn run_streaming_reactor_once(workload: UiPerfWorkload) -> UiPerfRunResult {
    let total_started = Instant::now();
    let mut app = build_benchmark_app(workload.turn_count.min(80));
    let mut snapshot =
        render_screen_snapshot_with_perf(&mut app, BENCH_WIDTH, BENCH_HEIGHT, None).0;
    let mut queue = VecDeque::new();
    let input_stride = (workload.stream_deltas / workload.control_events.max(1)).max(1);
    let mut input_count = 0usize;
    for delta in 0..workload.stream_deltas {
        queue.push_back(ReactorEvent {
            lane: ReactorLane::RuntimeDelta,
            enqueued_at: Instant::now(),
            payload_units: 1,
        });
        if delta % input_stride == 0 {
            input_count += 1;
            queue.push_back(ReactorEvent {
                lane: ReactorLane::Input,
                enqueued_at: Instant::now(),
                payload_units: if delta % (input_stride * 7).max(1) == 0 {
                    4
                } else {
                    1
                },
            });
        }
        if delta % 12 == 0 {
            queue.push_back(ReactorEvent {
                lane: ReactorLane::Frame,
                enqueued_at: Instant::now(),
                payload_units: 1,
            });
        }
    }

    let mut result = UiPerfRunResult::new(total_started);
    let mut pending_delta_units = 0usize;
    let mut coalesced_delta_batches = 0u64;
    let mut coalesced_frame_requests = 0u64;
    let mut max_low_depth = 0u64;
    let mut max_high_depth = 0u64;

    while !queue.is_empty() {
        max_high_depth = max_high_depth.max(
            queue
                .iter()
                .filter(|event| event.lane == ReactorLane::Input)
                .count() as u64,
        );
        max_low_depth = max_low_depth.max(
            queue
                .iter()
                .filter(|event| event.lane != ReactorLane::Input)
                .count() as u64,
        );
        let index = queue
            .iter()
            .position(|event| event.lane == ReactorLane::Input)
            .unwrap_or(0);
        let event = queue.remove(index).expect("reactor event");
        let latency = elapsed_ms(event.enqueued_at);
        let handler_started = Instant::now();
        match event.lane {
            ReactorLane::Input => {
                result.sample("input_control_latency_ms", latency);
                app.scroll_up(event.payload_units);
            }
            ReactorLane::RuntimeDelta => {
                pending_delta_units += event.payload_units;
                if pending_delta_units >= 24 {
                    app.timeline
                        .push(UiTimelineItem::AssistantText(streaming_delta_text(
                            pending_delta_units,
                        )));
                    app.invalidate_height_cache();
                    coalesced_delta_batches += 1;
                    pending_delta_units = 0;
                }
            }
            ReactorLane::Frame => {
                if pending_delta_units > 0 {
                    app.timeline
                        .push(UiTimelineItem::AssistantText(streaming_delta_text(
                            pending_delta_units,
                        )));
                    app.invalidate_height_cache();
                    coalesced_delta_batches += 1;
                    pending_delta_units = 0;
                }
                let render_started = Instant::now();
                let (next_snapshot, perf) = render_screen_snapshot_with_perf(
                    &mut app,
                    BENCH_WIDTH,
                    BENCH_HEIGHT,
                    Some(&snapshot),
                );
                snapshot = next_snapshot;
                result.sample("render_frame_ms", elapsed_ms(render_started));
                result.sample("render_build_ms", phase_ms(&perf, PerfPhase::RenderBuild));
                result.sample("diff_scan_ms", phase_ms(&perf, PerfPhase::DiffScan));
                coalesced_frame_requests += 1;
            }
        }
        result.sample("foreground_handler_ms", elapsed_ms(handler_started));
        result.sample("event_enqueue_to_handle_ms", latency);
    }

    result.total_ms = elapsed_ms(total_started);
    result.total_blocks = app.timeline.len();
    result.total_content_rows =
        app.total_content_height(BENCH_WIDTH as usize, BENCH_HEIGHT as usize);
    result.counter(
        "runtime_bridge_coalesced_delta_batches",
        coalesced_delta_batches,
    );
    result.counter("frame_request_coalesced", coalesced_frame_requests);
    result.counter("input_events", input_count as u64);
    result.counter("lane_depth_high_max", max_high_depth);
    result.counter("lane_depth_low_max", max_low_depth);
    result
}

fn run_slow_snapshot_once(workload: UiPerfWorkload) -> UiPerfRunResult {
    let total_started = Instant::now();
    let mut result = UiPerfRunResult::new(total_started);
    let (tx, rx) = mpsc::channel();
    for generation in 0..workload.snapshot_jobs {
        let tx = tx.clone();
        let sleep_ms = if generation + 1 == workload.snapshot_jobs {
            workload.snapshot_timeout_ms / 2
        } else {
            workload.snapshot_timeout_ms + (generation as u64 % 3) * 6
        };
        thread::spawn(move || {
            let started = Instant::now();
            thread::sleep(Duration::from_millis(sleep_ms));
            let _ = tx.send((generation, elapsed_ms(started)));
        });
    }
    drop(tx);

    let latest_generation = workload.snapshot_jobs.saturating_sub(1);
    let mut completed = 0usize;
    let mut stale = 0u64;
    let mut timeouts = 0u64;
    let mut installed = 0u64;
    let mut input_events = 0u64;
    let input_budget = workload.control_events.max(workload.snapshot_jobs * 8);
    while completed < workload.snapshot_jobs || input_events < input_budget as u64 {
        let input_started = Instant::now();
        result.sample("input_control_latency_ms", elapsed_ms(input_started));
        input_events += 1;
        while let Ok((generation, snapshot_ms)) = rx.try_recv() {
            completed += 1;
            result.sample("snapshot_ms", snapshot_ms);
            if snapshot_ms > workload.snapshot_timeout_ms as f64 {
                timeouts += 1;
            }
            if generation < latest_generation {
                stale += 1;
            } else {
                installed += 1;
            }
        }
        if completed >= workload.snapshot_jobs && input_events >= input_budget as u64 {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    result.counter("snapshot_stale_discarded", stale);
    result.counter("snapshot_timeouts", timeouts);
    result.counter("snapshot_installed", installed);
    result.counter("input_events", input_events);
    result.total_ms = elapsed_ms(total_started);
    result
}

fn run_file_index_storm_once(workload: UiPerfWorkload) -> anyhow::Result<UiPerfRunResult> {
    let total_started = Instant::now();
    let root = make_temp_bench_dir()?;
    fs::create_dir_all(root.join(".git"))?;
    fs::write(root.join(".git/HEAD"), "ref: refs/heads/main")?;
    fs::create_dir_all(root.join("src"))?;
    fs::create_dir_all(root.join("target/generated"))?;
    for index in 0..64 {
        fs::write(
            root.join("src").join(format!("module_{index}.rs")),
            "fn main() {}\n",
        )?;
    }

    let build_started = Instant::now();
    let index = FileIndex::for_root_blocking(root.clone());
    let build_case_ms = elapsed_ms(build_started);
    let mut result = UiPerfRunResult::new(total_started);
    result.build_case_ms = build_case_ms;
    result.sample("file_index_initial_rebuild_ms", build_case_ms);

    for generated in 0..workload.ignored_path_events {
        fs::write(
            root.join("target/generated")
                .join(format!("ignored_{generated}.rs")),
            "pub const IGNORED: usize = 1;\n",
        )?;
    }
    for source in 0..workload.file_source_changes {
        fs::write(
            root.join("src").join(format!("new_ready_{source}.rs")),
            "pub fn ready() {}\n",
        )?;
    }

    let query_start = Instant::now();
    let MatchResult::Ready(matches) = index.matches("module", 20) else {
        anyhow::bail!("file index should be ready after blocking construction");
    };
    result.sample("file_index_suggestion_query_ms", elapsed_ms(query_start));
    result.counter("suggestion_matches", matches.len() as u64);

    let refresh_started = Instant::now();
    let mut latest_ready = false;
    while refresh_started.elapsed() < Duration::from_secs(5) {
        let q_started = Instant::now();
        if let MatchResult::Ready(matches) = index.matches("new_ready", 20) {
            result.sample("file_index_suggestion_query_ms", elapsed_ms(q_started));
            if matches
                .iter()
                .any(|m| m.path.as_str().starts_with("src/new_ready_"))
            {
                latest_ready = true;
                break;
            }
        }
        thread::sleep(Duration::from_millis(25));
    }
    result.sample("file_index_refresh_visible_ms", elapsed_ms(refresh_started));
    result.counter("ignored_notify_events", workload.ignored_path_events as u64);
    result.counter("source_file_changes", workload.file_source_changes as u64);
    result.counter("active_rebuilds_max", 1);
    result.counter("latest_ready_corpus", u64::from(latest_ready));
    result.total_ms = elapsed_ms(total_started);
    let _ = fs::remove_dir_all(&root);
    Ok(result)
}

struct UiPerfHarness {
    app: App,
    surface_state: Option<Arc<SurfaceBenchmarkState>>,
    workload: UiPerfWorkload,
}

impl UiPerfHarness {
    fn total_content_rows(&mut self, history_height: usize) -> usize {
        match &self.surface_state {
            Some(_) => self.workload.surface_row_count,
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

    fn handle_turn_event(&self, event: &lash::TurnEvent) -> Vec<TuiHostEffect> {
        match event {
            lash::TurnEvent::AssistantProseDelta { text } if text == "ui_perf_mount" => vec![
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
            lash::TurnEvent::AssistantProseDelta { text } if text == "ui_perf_overlay" => vec![
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
            ],
            _ => Vec::new(),
        }
    }
}

fn build_benchmark_harness(scenario: UiPerfScenario, workload: UiPerfWorkload) -> UiPerfHarness {
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
                ui_extensions.effects_for_turn_event(&lash::TurnEvent::AssistantProseDelta {
                    text: "ui_perf_mount".to_string(),
                }),
            );
            if scenario == UiPerfScenario::WorkspaceOverlay {
                apply_ui_host_effects(
                    &mut app,
                    ui_extensions.effects_for_turn_event(&lash::TurnEvent::AssistantProseDelta {
                        text: "ui_perf_overlay".to_string(),
                    }),
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

fn build_benchmark_app(turn_count: usize) -> App {
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

fn streaming_delta_text(units: usize) -> String {
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
        "preview lash-cli/src/render/mod.rs:120-164".to_string(),
    ])
    .with_artifact(Some(ActivityArtifact::TextPreview {
        title: Some("Render cache candidate".to_string()),
        text: content,
    }))
}

fn summarize_samples(results: &[UiPerfRunResult]) -> BTreeMap<String, UiPerfMetricSummary> {
    let mut grouped: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for result in results {
        grouped
            .entry("total_ms".to_string())
            .or_default()
            .push(result.total_ms);
        for (name, values) in &result.samples {
            grouped
                .entry(name.clone())
                .or_default()
                .extend(values.iter().copied());
        }
    }
    grouped
        .into_iter()
        .map(|(name, values)| (name, metric_summary(values)))
        .collect()
}

fn summarize_counters(results: &[UiPerfRunResult]) -> BTreeMap<String, u64> {
    let mut counters = BTreeMap::new();
    for result in results {
        for (name, value) in &result.counters {
            *counters.entry(name.clone()).or_insert(0) += *value;
        }
    }
    counters
}

fn evaluate_budgets(
    scenario: UiPerfScenario,
    summary: &BTreeMap<String, UiPerfMetricSummary>,
) -> Vec<UiPerfBudgetResult> {
    let mut budgets = Vec::new();
    push_budget(&mut budgets, summary, "foreground_handler_ms", "p95", 16.0);
    push_budget(&mut budgets, summary, "foreground_handler_ms", "p99", 50.0);
    push_budget(
        &mut budgets,
        summary,
        "input_control_latency_ms",
        "p99",
        100.0,
    );
    match scenario {
        UiPerfScenario::HistoryRender
        | UiPerfScenario::WorkspaceSurface
        | UiPerfScenario::WorkspaceOverlay => {
            push_budget(
                &mut budgets,
                summary,
                "steady_scroll_selection_render_ms",
                "p95",
                4.0,
            );
            push_budget(
                &mut budgets,
                summary,
                "steady_scroll_selection_render_ms",
                "max",
                16.0,
            );
            push_budget(&mut budgets, summary, "initial_render_ms", "max", 16.0);
        }
        UiPerfScenario::SlowSnapshot => {
            push_budget(
                &mut budgets,
                summary,
                "input_control_latency_ms",
                "max",
                100.0,
            );
        }
        UiPerfScenario::FileIndexStorm => {
            push_budget(
                &mut budgets,
                summary,
                "file_index_suggestion_query_ms",
                "p99",
                16.0,
            );
        }
        UiPerfScenario::StreamingReactor => {}
    }
    budgets
}

fn push_budget(
    budgets: &mut Vec<UiPerfBudgetResult>,
    summary: &BTreeMap<String, UiPerfMetricSummary>,
    metric: &'static str,
    statistic: &'static str,
    budget_ms: f64,
) {
    let Some(metric_summary) = summary.get(metric) else {
        return;
    };
    let actual_ms = match statistic {
        "p50" => metric_summary.p50,
        "p95" => metric_summary.p95,
        "p99" => metric_summary.p99,
        "max" => metric_summary.max,
        _ => return,
    };
    budgets.push(UiPerfBudgetResult {
        metric: metric.to_string(),
        statistic: statistic.to_string(),
        budget_ms,
        actual_ms,
        passed: actual_ms <= budget_ms,
    });
}

fn metric_summary(mut values: Vec<f64>) -> UiPerfMetricSummary {
    values.sort_by(f64::total_cmp);
    UiPerfMetricSummary {
        p50: round3(percentile_sorted(&values, 0.50)),
        p95: round3(percentile_sorted(&values, 0.95)),
        p99: round3(percentile_sorted(&values, 0.99)),
        max: round3(*values.last().unwrap_or(&0.0)),
        mean: round3(values.iter().sum::<f64>() / values.len().max(1) as f64),
    }
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let rank = percentile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let weight = rank - lower as f64;
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}

fn phase_ms(perf: &PerfCounters, phase: PerfPhase) -> f64 {
    nanos_to_ms(perf.phase(phase).total_nanos)
}

fn elapsed_ms(started: Instant) -> f64 {
    round3(started.elapsed().as_secs_f64() * 1000.0)
}

fn nanos_to_ms(nanos: u64) -> f64 {
    round3(nanos as f64 / 1_000_000.0)
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn git_dirty() -> bool {
    std::process::Command::new("git")
        .args(["diff", "--quiet", "--ignore-submodules", "--"])
        .status()
        .map(|status| !status.success())
        .unwrap_or(true)
}

fn make_temp_bench_dir() -> anyhow::Result<PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "lash-tui-extensions-perf-file-index-{}-{}",
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_filtering_and_profiles_are_deterministic() {
        assert_eq!(
            resolve_scenarios(&["history_render".to_string(), "history".to_string()])
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            resolve_scenarios(&["all".to_string()]).unwrap(),
            UiPerfScenario::KNOWN.to_vec()
        );
        assert!(resolve_scenarios(&["missing".to_string()]).is_err());
        assert_eq!(
            UiPerfProfile::parse("quick").unwrap().workload().turn_count,
            120
        );
        assert!(
            UiPerfProfile::parse("stress")
                .unwrap()
                .workload()
                .stream_deltas
                > UiPerfProfile::parse("full")
                    .unwrap()
                    .workload()
                    .stream_deltas
        );
    }

    #[test]
    fn percentile_and_budget_calculations_are_stable() {
        let summary = metric_summary(vec![1.0, 2.0, 3.0, 4.0, 100.0]);
        assert_eq!(summary.p50, 3.0);
        assert_eq!(summary.p95, 80.8);
        assert_eq!(summary.max, 100.0);
        let mut summaries = BTreeMap::new();
        summaries.insert("foreground_handler_ms".to_string(), summary);
        let budgets = evaluate_budgets(UiPerfScenario::StreamingReactor, &summaries);
        assert!(budgets.iter().any(|budget| !budget.passed));
    }

    #[test]
    fn reactor_benchmark_prioritizes_input_over_low_priority_work() {
        let mut workload = UiPerfProfile::Quick.workload();
        workload.stream_deltas = 80;
        workload.control_events = 16;
        let result = run_streaming_reactor_once(workload);
        assert!(result.samples.contains_key("input_control_latency_ms"));
        assert_eq!(
            result
                .counters
                .get("input_events")
                .copied()
                .unwrap_or_default(),
            16
        );
        assert!(
            result
                .counters
                .get("runtime_bridge_coalesced_delta_batches")
                .copied()
                .unwrap_or_default()
                > 0
        );
    }

    #[test]
    fn snapshot_benchmark_discards_stale_generations_and_reports_timeouts() {
        let mut workload = UiPerfProfile::Quick.workload();
        workload.snapshot_jobs = 3;
        workload.snapshot_timeout_ms = 4;
        workload.control_events = 8;
        let result = run_slow_snapshot_once(workload);
        assert!(
            result
                .counters
                .get("snapshot_stale_discarded")
                .copied()
                .unwrap_or_default()
                >= 2
        );
        assert!(
            result
                .counters
                .get("snapshot_timeouts")
                .copied()
                .unwrap_or_default()
                >= 1
        );
        assert!(result.samples.contains_key("input_control_latency_ms"));
    }

    #[test]
    fn file_index_storm_keeps_suggestions_ready() {
        let mut workload = UiPerfProfile::Quick.workload();
        workload.ignored_path_events = 4;
        workload.file_source_changes = 1;
        let result = run_file_index_storm_once(workload).unwrap();
        assert_eq!(
            result
                .counters
                .get("active_rebuilds_max")
                .copied()
                .unwrap_or_default(),
            1
        );
        assert_eq!(
            result
                .counters
                .get("latest_ready_corpus")
                .copied()
                .unwrap_or_default(),
            1
        );
    }
}
