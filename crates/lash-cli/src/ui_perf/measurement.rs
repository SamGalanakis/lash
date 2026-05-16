use std::collections::BTreeMap;
use std::time::Instant;

use serde::Serialize;

use crate::perf_support::time::elapsed_ms;

use super::file_index_cases::run_file_index_storm_once;
use super::reactor_cases::run_streaming_reactor_once;
use super::render_cases::run_render_once;
use super::scenarios::{UiPerfScenario, UiPerfWorkload};
use super::workloads::run_slow_snapshot_once;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfRunResult {
    pub(crate) build_case_ms: f64,
    pub(crate) total_ms: f64,
    pub(crate) total_blocks: usize,
    pub(crate) total_content_rows: usize,
    pub(crate) samples: BTreeMap<String, Vec<f64>>,
    pub(crate) counters: BTreeMap<String, u64>,
}

impl UiPerfRunResult {
    pub(crate) fn new(started: Instant) -> Self {
        Self {
            build_case_ms: 0.0,
            total_ms: elapsed_ms(started),
            total_blocks: 0,
            total_content_rows: 0,
            samples: BTreeMap::new(),
            counters: BTreeMap::new(),
        }
    }

    pub(crate) fn sample(&mut self, name: &'static str, value: f64) {
        self.samples
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    pub(crate) fn sample_many(&mut self, name: &'static str, values: Vec<f64>) {
        if !values.is_empty() {
            self.samples
                .entry(name.to_string())
                .or_default()
                .extend(values);
        }
    }

    pub(crate) fn counter(&mut self, name: &'static str, value: u64) {
        self.counters.insert(name.to_string(), value);
    }
}

pub(crate) fn run_once(
    scenario: UiPerfScenario,
    workload: UiPerfWorkload,
) -> anyhow::Result<UiPerfRunResult> {
    match scenario {
        UiPerfScenario::HistoryRender
        | UiPerfScenario::WorkspaceSurface
        | UiPerfScenario::WorkspaceOverlay => Ok(run_render_once(scenario, workload)),
        UiPerfScenario::StreamingReactor => Ok(run_streaming_reactor_once(workload)),
        UiPerfScenario::SlowSnapshot => Ok(run_slow_snapshot_once(workload)),
        UiPerfScenario::FileIndexStorm => run_file_index_storm_once(workload),
    }
}
