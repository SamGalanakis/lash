use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crate::perf_support::time::elapsed_ms;

use super::measurement::UiPerfRunResult;
use super::scenarios::UiPerfWorkload;

pub(crate) fn run_slow_snapshot_once(workload: UiPerfWorkload) -> UiPerfRunResult {
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
