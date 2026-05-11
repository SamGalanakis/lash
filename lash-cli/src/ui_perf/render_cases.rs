use std::time::Instant;

use lash_tui::{PerfCounters, PerfPhase};

use crate::perf_support::time::{elapsed_ms, nanos_to_ms};
use crate::render;
use crate::ui_trace::render_screen_snapshot_with_perf;

use super::measurement::UiPerfRunResult;
use super::scenarios::{
    BENCH_HEIGHT, BENCH_WIDTH, SCROLL_DELTA, SELECTION_SCROLL_DELTA, UiPerfScenario, UiPerfWorkload,
};
use super::surface_fixture::build_benchmark_harness;

pub(crate) fn run_render_once(
    scenario: UiPerfScenario,
    workload: UiPerfWorkload,
) -> UiPerfRunResult {
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

fn phase_ms(perf: &PerfCounters, phase: PerfPhase) -> f64 {
    nanos_to_ms(perf.phase(phase).total_nanos)
}
