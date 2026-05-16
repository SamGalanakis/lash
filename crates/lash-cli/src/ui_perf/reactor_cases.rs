use std::collections::VecDeque;
use std::time::Instant;

use lash_tui::{PerfCounters, PerfPhase};

use crate::app::UiTimelineItem;
use crate::perf_support::time::{elapsed_ms, nanos_to_ms};
use crate::ui_trace::render_screen_snapshot_with_perf;

use super::measurement::UiPerfRunResult;
use super::scenarios::{BENCH_HEIGHT, BENCH_WIDTH, UiPerfWorkload};
use super::surface_fixture::{build_benchmark_app, streaming_delta_text};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReactorLane {
    Input,
    RuntimeDelta,
    Frame,
}

fn phase_ms(perf: &PerfCounters, phase: PerfPhase) -> f64 {
    nanos_to_ms(perf.phase(phase).total_nanos)
}

#[derive(Clone, Copy, Debug)]
struct ReactorEvent {
    lane: ReactorLane,
    enqueued_at: Instant,
    payload_units: usize,
}

pub(crate) fn run_streaming_reactor_once(workload: UiPerfWorkload) -> UiPerfRunResult {
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
