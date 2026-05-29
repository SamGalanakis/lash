use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::Serialize;

use crate::perf_support::dhat;
use crate::perf_support::git::git_dirty;
use crate::perf_support::metrics::{
    PercentileMetricSummary as UiPerfMetricSummary, percentile_summary,
};
use crate::perf_support::paths;
use crate::perf_support::report as report_support;

use super::measurement::{UiPerfRunResult, run_once};
use super::scenarios::{BENCH_HEIGHT, BENCH_WIDTH, UiPerfProfile, UiPerfScenario, UiPerfWorkload};

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfScenarioReport {
    scenario: String,
    results: Vec<UiPerfRunResult>,
    summary: BTreeMap<String, UiPerfMetricSummary>,
    counters: BTreeMap<String, u64>,
    budgets: Vec<UiPerfBudgetResult>,
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
    paths::default_report_path("ui-perf")
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
    report_support::ensure_parent_dir(&out_path, "benchmark output")?;
    let dhat_out_path = resolve_dhat_output_path(enable_dhat, &out_path, dhat_out);
    dhat::ensure_dhat_parent(dhat_out_path.as_ref())?;

    let profiler = dhat::start_dhat_profiler(
        dhat_out_path.clone(),
        dhat_frames,
        "UI perf dhat profiling requires a lash-cli build with --features dhat-heap",
    )?;
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
    dhat::finish_dhat_profiler(profiler);

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

    report_support::write_json_report(&out_path, &report)?;

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
    dhat::resolve_dhat_output_path(enable_dhat, report_out, dhat_out, "ui-perf")
}

fn resolve_scenarios(filters: &[String]) -> anyhow::Result<Vec<UiPerfScenario>> {
    report_support::resolve_named_scenarios(
        filters,
        &UiPerfScenario::DEFAULTS,
        &UiPerfScenario::KNOWN,
        UiPerfScenario::parse,
        UiPerfScenario::name,
        "UI perf",
    )
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
        UiPerfScenario::TimelineProjection => {
            push_budget(
                &mut budgets,
                summary,
                "timeline_from_read_view_ms",
                "p95",
                50.0,
            );
        }
        UiPerfScenario::ActivityProjection => {
            push_budget(
                &mut budgets,
                summary,
                "turn_activity_handle_ms",
                "p95",
                16.0,
            );
        }
        UiPerfScenario::HtmlExport => {
            push_budget(&mut budgets, summary, "html_export_render_ms", "p95", 250.0);
        }
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

fn metric_summary(values: Vec<f64>) -> UiPerfMetricSummary {
    percentile_summary(values)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui_perf::file_index_cases::run_file_index_storm_once;
    use crate::ui_perf::reactor_cases::run_streaming_reactor_once;
    use crate::ui_perf::workloads::run_slow_snapshot_once;

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
