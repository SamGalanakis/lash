use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::Utc;
use lash::usage::SessionUsageReport;
use serde::Serialize;

use crate::perf_support::dhat;
use crate::perf_support::metrics::{
    BasicMetricSummary as RuntimePerfMetricSummary, basic_summary, optional_basic_summary,
};
use crate::perf_support::paths;
use crate::perf_support::report as report_support;
use crate::perf_support::time::round3;

use super::measurement::*;
use super::scenarios::RuntimePerfScenario;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfReport {
    created_at: String,
    version: String,
    warmups: usize,
    runs: usize,
    chat_turns: usize,
    scenarios: Vec<String>,
    dhat_out: Option<PathBuf>,
    results: Vec<RuntimePerfRunResult>,
    summary: Vec<RuntimePerfScenarioSummary>,
    budget_results: Vec<RuntimePerfBudgetResult>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfBudgetResult {
    scenario: String,
    metric: String,
    statistic: String,
    actual: Option<f64>,
    budget: Option<f64>,
    passed: bool,
    reason: Option<String>,
}

pub(crate) fn default_output_path() -> PathBuf {
    paths::default_report_path("runtime-perf")
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_cli(
    out: Option<PathBuf>,
    enable_dhat: bool,
    dhat_out: Option<PathBuf>,
    dhat_frames: Option<usize>,
    runs: usize,
    warmups: usize,
    scenario_filters: Vec<String>,
    chat_turns: usize,
    enforce_budgets: bool,
    version: &str,
) -> anyhow::Result<()> {
    if dhat_out.is_some() && !enable_dhat {
        anyhow::bail!("--runtime-perf-dhat-out requires --runtime-perf-dhat");
    }
    let scenarios = resolve_scenarios(&scenario_filters)?;
    let runs = runs.max(1);
    let chat_turns = chat_turns.max(1);

    for _ in 0..warmups {
        for scenario in &scenarios {
            let _ = run_once(*scenario, chat_turns).await?;
        }
    }

    let out_path = out.unwrap_or_else(default_output_path);
    report_support::ensure_parent_dir(&out_path, "benchmark output")?;
    let dhat_out_path = resolve_dhat_output_path(enable_dhat, &out_path, dhat_out);
    dhat::ensure_dhat_parent(dhat_out_path.as_ref())?;

    let profiler = dhat::start_dhat_profiler(
        dhat_out_path.clone(),
        dhat_frames,
        "runtime perf dhat profiling requires a lash-cli build with --features dhat-heap",
    )?;
    let mut results = Vec::with_capacity(runs * scenarios.len());
    for _ in 0..runs {
        for scenario in &scenarios {
            results.push(run_once(*scenario, chat_turns).await?);
        }
    }
    dhat::finish_dhat_profiler(profiler);

    let summary = summarize(&results, &scenarios, chat_turns);
    let budget_results = evaluate_budgets(&summary, &scenarios);
    let report = RuntimePerfReport {
        created_at: Utc::now().to_rfc3339(),
        version: version.to_string(),
        warmups,
        runs,
        chat_turns,
        scenarios: scenarios
            .iter()
            .map(|scenario| scenario.name().to_string())
            .collect(),
        dhat_out: dhat_out_path.clone(),
        summary,
        budget_results,
        results,
    };

    report_support::write_json_report(&out_path, &report)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "out": out_path,
            "dhat_out": report.dhat_out,
            "summary": report.summary,
            "budget_results": report.budget_results,
        }))?
    );
    if enforce_budgets {
        let failures = report
            .budget_results
            .iter()
            .filter(|budget| !budget.passed)
            .map(|budget| {
                let actual = budget
                    .actual
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "missing".to_string());
                let budget_value = budget
                    .budget
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "required".to_string());
                let reason = budget.reason.as_deref().unwrap_or("budget exceeded");
                format!(
                    "{} {} {} actual={} budget={} ({reason})",
                    budget.scenario, budget.metric, budget.statistic, actual, budget_value
                )
            })
            .collect::<Vec<_>>();
        if !failures.is_empty() {
            anyhow::bail!("Runtime perf budget exceeded:\n{}", failures.join("\n"));
        }
    }
    Ok(())
}

fn resolve_dhat_output_path(
    enable_dhat: bool,
    report_out: &Path,
    dhat_out: Option<PathBuf>,
) -> Option<PathBuf> {
    dhat::resolve_dhat_output_path(enable_dhat, report_out, dhat_out, "runtime-perf")
}

fn resolve_scenarios(filters: &[String]) -> anyhow::Result<Vec<RuntimePerfScenario>> {
    report_support::resolve_named_scenarios(
        filters,
        &RuntimePerfScenario::DEFAULTS,
        &RuntimePerfScenario::KNOWN,
        RuntimePerfScenario::parse,
        RuntimePerfScenario::name,
        "runtime perf",
    )
}
fn summarize(
    results: &[RuntimePerfRunResult],
    scenarios: &[RuntimePerfScenario],
    chat_turns: usize,
) -> Vec<RuntimePerfScenarioSummary> {
    scenarios
        .iter()
        .filter_map(|scenario| {
            let matching = results
                .iter()
                .filter(|result| result.scenario == scenario.name())
                .collect::<Vec<_>>();
            if matching.is_empty() {
                return None;
            }
            Some(RuntimePerfScenarioSummary {
                scenario: scenario.name().to_string(),
                runs: matching.len(),
                chat_turns,
                build_runtime_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.build_runtime_ms)
                        .collect::<Vec<_>>(),
                ),
                seed_state_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.seed_state_ms)
                        .collect::<Vec<_>>(),
                ),
                run_turn_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.run_turn_ms)
                        .collect::<Vec<_>>(),
                ),
                await_background_work_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.await_background_work_ms)
                        .collect::<Vec<_>>(),
                ),
                export_state_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.export_state_ms)
                        .collect::<Vec<_>>(),
                ),
                total_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.total_ms)
                        .collect::<Vec<_>>(),
                ),
                rss_after_export_kb: summarize_optional_metric(
                    matching
                        .iter()
                        .filter_map(|result| {
                            result.memory.rss_after_export_kb.map(|value| value as f64)
                        })
                        .collect::<Vec<_>>(),
                ),
                rss_growth_kb: summarize_optional_metric(
                    matching
                        .iter()
                        .filter_map(|result| result.memory.rss_growth_kb.map(|value| value as f64))
                        .collect::<Vec<_>>(),
                ),
                hwm_growth_kb: summarize_optional_metric(
                    matching
                        .iter()
                        .filter_map(|result| result.memory.hwm_growth_kb.map(|value| value as f64))
                        .collect::<Vec<_>>(),
                ),
                build_runtime_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.build_runtime.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                build_runtime_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.build_runtime.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                seed_state_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.seed_state.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                seed_state_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.seed_state.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                run_turn_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.run_turn.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                run_turn_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.run_turn.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                await_background_work_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| {
                            result.allocations.await_background_work.bytes_allocated as f64
                        })
                        .collect::<Vec<_>>(),
                ),
                await_background_work_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| {
                            result.allocations.await_background_work.net_live_bytes as f64
                        })
                        .collect::<Vec<_>>(),
                ),
                export_state_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.export_state.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                export_state_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.export_state.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                total_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.total.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                total_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.total.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                phase_summary: summarize_phase_profiles(
                    &matching
                        .iter()
                        .map(|result| result.phase_profile.clone())
                        .collect::<Vec<_>>(),
                ),
                first_turn: summarize_turn_group(
                    &matching
                        .iter()
                        .filter_map(|result| result.turns.first().cloned())
                        .collect::<Vec<_>>(),
                ),
                steady_state_turn: summarize_optional_turn_group(
                    &matching
                        .iter()
                        .filter_map(|result| mean_turn_result(&result.turns[1..]))
                        .collect::<Vec<_>>(),
                ),
                last_turn: summarize_turn_group(
                    &matching
                        .iter()
                        .filter_map(|result| result.turns.last().cloned())
                        .collect::<Vec<_>>(),
                ),
                sample_session_nodes: matching[0].session_nodes,
                sample_active_path_messages: matching[0].active_path_messages,
                sample_extra_counters: matching[0].extra_counters.clone(),
            })
        })
        .collect()
}

fn evaluate_budgets(
    summaries: &[RuntimePerfScenarioSummary],
    scenarios: &[RuntimePerfScenario],
) -> Vec<RuntimePerfBudgetResult> {
    let mut budgets = Vec::new();
    for scenario in scenarios {
        let Some(summary) = summaries
            .iter()
            .find(|summary| summary.scenario == scenario.name())
        else {
            budgets.push(RuntimePerfBudgetResult {
                scenario: scenario.name().to_string(),
                metric: "scenario_output".to_string(),
                statistic: "present".to_string(),
                actual: None,
                budget: Some(1.0),
                passed: false,
                reason: Some("scenario did not produce a summary".to_string()),
            });
            continue;
        };

        for phase in required_phases(*scenario) {
            let passed = summary.phase_summary.contains_key(*phase);
            budgets.push(RuntimePerfBudgetResult {
                scenario: scenario.name().to_string(),
                metric: format!("phase:{phase}"),
                statistic: "present".to_string(),
                actual: if passed { Some(1.0) } else { None },
                budget: Some(1.0),
                passed,
                reason: (!passed).then(|| "required phase metrics were missing".to_string()),
            });
        }

        push_max_budget(
            &mut budgets,
            summary,
            "total_alloc_bytes",
            "median",
            summary.total_alloc_bytes.median,
            allocation_budget_bytes(*scenario),
        );
        push_max_budget(
            &mut budgets,
            summary,
            "run_turn_alloc_bytes",
            "median",
            summary.run_turn_alloc_bytes.median,
            run_turn_allocation_budget_bytes(*scenario),
        );
        push_max_budget(
            &mut budgets,
            summary,
            "total_ms",
            "median",
            summary.total_ms.median,
            wall_clock_budget_ms(*scenario),
        );
    }
    budgets
}

fn push_max_budget(
    budgets: &mut Vec<RuntimePerfBudgetResult>,
    summary: &RuntimePerfScenarioSummary,
    metric: &str,
    statistic: &str,
    actual: f64,
    budget: f64,
) {
    budgets.push(RuntimePerfBudgetResult {
        scenario: summary.scenario.clone(),
        metric: metric.to_string(),
        statistic: statistic.to_string(),
        actual: Some(actual),
        budget: Some(budget),
        passed: actual <= budget,
        reason: (actual > budget).then(|| "metric exceeded checked-in guard budget".to_string()),
    });
}

fn required_phases(scenario: RuntimePerfScenario) -> &'static [&'static str] {
    match scenario {
        RuntimePerfScenario::OpenAiResponsesSseParse => &[
            "openai_responses_sse_parse.parse_payload",
            "openai_responses_sse_parse.project_parts",
        ],
        RuntimePerfScenario::DirectLlmClient => &["direct_llm_client.complete"],
        RuntimePerfScenario::ProcessListStress => &[
            "process_list_stress.list_live",
            "process_list_stress.list_all",
            "process_list_stress.render_live_json",
            "process_list_stress.render_all_json",
        ],
        RuntimePerfScenario::TurnCheckpoint => &[
            "standard_llm_checkpoint",
            "standard_parallel_tools_checkpoint",
            "rlm_exec_checkpoint",
        ],
        RuntimePerfScenario::LiveReplayPressure => &[
            "live_replay.append",
            "live_replay.current_cursor_parse",
            "live_replay.replay_after_cursor",
            "live_replay.subscribe_buffered",
            "live_replay.trim_by_capacity",
            "live_replay.gap_handling",
        ],
        RuntimePerfScenario::EmbedStandard | RuntimePerfScenario::EmbedRlm => &[],
        _ => &[
            "context_transform",
            "before_turn_hooks",
            "prompt_build",
            "effect_loop",
            "finalize_turn",
            "persist_turn",
            "final_commit",
            "post_persist_hooks",
        ],
    }
}

fn allocation_budget_bytes(scenario: RuntimePerfScenario) -> f64 {
    match scenario {
        RuntimePerfScenario::ToolDiscoverySearch => 1_500_000_000.0,
        RuntimePerfScenario::RlmLargeToolSurface => 1_000_000_000.0,
        RuntimePerfScenario::LiveReplayPressure => 128_000_000.0,
        RuntimePerfScenario::OpenAiResponsesSseParse
        | RuntimePerfScenario::DirectLlmClient
        | RuntimePerfScenario::TurnCheckpoint => 256_000_000.0,
        _ => 1_000_000_000.0,
    }
}

fn run_turn_allocation_budget_bytes(scenario: RuntimePerfScenario) -> f64 {
    match scenario {
        RuntimePerfScenario::ToolDiscoverySearch => 1_000_000_000.0,
        RuntimePerfScenario::LiveReplayPressure => 96_000_000.0,
        RuntimePerfScenario::OpenAiResponsesSseParse
        | RuntimePerfScenario::DirectLlmClient
        | RuntimePerfScenario::TurnCheckpoint => 192_000_000.0,
        _ => 750_000_000.0,
    }
}

fn wall_clock_budget_ms(scenario: RuntimePerfScenario) -> f64 {
    match scenario {
        RuntimePerfScenario::ToolDiscoverySearch | RuntimePerfScenario::RlmLargeToolSurface => {
            20_000.0
        }
        RuntimePerfScenario::LiveReplayPressure => 5_000.0,
        _ => 10_000.0,
    }
}

fn summarize_phase_profiles(
    profiles: &[BTreeMap<String, RuntimePerfPhaseRunResult>],
) -> BTreeMap<String, RuntimePerfPhaseSummary> {
    let mut by_phase: BTreeMap<String, Vec<&RuntimePerfPhaseRunResult>> = BTreeMap::new();
    for profile in profiles {
        for (phase, metrics) in profile {
            by_phase.entry(phase.clone()).or_default().push(metrics);
        }
    }

    by_phase
        .into_iter()
        .map(|(phase, metrics)| {
            let summary = RuntimePerfPhaseSummary {
                duration_ms: summarize_metric(
                    metrics.iter().map(|metric| metric.duration_ms).collect(),
                ),
                alloc_bytes: summarize_metric(
                    metrics
                        .iter()
                        .map(|metric| metric.allocations.bytes_allocated as f64)
                        .collect(),
                ),
                live_bytes: summarize_metric(
                    metrics
                        .iter()
                        .map(|metric| metric.allocations.net_live_bytes as f64)
                        .collect(),
                ),
                rss_growth_kb: summarize_optional_metric(
                    metrics
                        .iter()
                        .filter_map(|metric| metric.rss_growth_kb.map(|value| value as f64))
                        .collect(),
                ),
            };
            (phase, summary)
        })
        .collect()
}

fn summarize_turn_group(turns: &[RuntimePerfTurnResult]) -> RuntimePerfTurnSummary {
    RuntimePerfTurnSummary {
        total_ms: summarize_metric(turns.iter().map(|turn| turn.total_ms).collect()),
        run_turn_ms: summarize_metric(turns.iter().map(|turn| turn.run_turn_ms).collect()),
        await_background_work_ms: summarize_metric(
            turns
                .iter()
                .map(|turn| turn.await_background_work_ms)
                .collect(),
        ),
        rss_growth_kb: summarize_optional_metric(
            turns
                .iter()
                .filter_map(|turn| turn.memory.rss_growth_kb.map(|value| value as f64))
                .collect(),
        ),
        total_alloc_bytes: summarize_metric(
            turns
                .iter()
                .map(|turn| turn.allocations.total.bytes_allocated as f64)
                .collect(),
        ),
        total_live_bytes: summarize_metric(
            turns
                .iter()
                .map(|turn| turn.allocations.total.net_live_bytes as f64)
                .collect(),
        ),
        phase_summary: summarize_phase_profiles(
            &turns
                .iter()
                .map(|turn| turn.phase_profile.clone())
                .collect::<Vec<_>>(),
        ),
    }
}

fn summarize_optional_turn_group(
    turns: &[RuntimePerfTurnResult],
) -> Option<RuntimePerfTurnSummary> {
    if turns.is_empty() {
        None
    } else {
        Some(summarize_turn_group(turns))
    }
}

fn mean_turn_result(turns: &[RuntimePerfTurnResult]) -> Option<RuntimePerfTurnResult> {
    if turns.is_empty() {
        return None;
    }

    let count = turns.len() as f64;
    Some(RuntimePerfTurnResult {
        turn_index: turns[0].turn_index,
        run_turn_ms: round3(turns.iter().map(|turn| turn.run_turn_ms).sum::<f64>() / count),
        await_background_work_ms: round3(
            turns
                .iter()
                .map(|turn| turn.await_background_work_ms)
                .sum::<f64>()
                / count,
        ),
        total_ms: round3(turns.iter().map(|turn| turn.total_ms).sum::<f64>() / count),
        memory: RuntimePerfTurnMemoryRunResult {
            rss_before_kb: None,
            rss_after_turn_kb: None,
            rss_after_await_kb: None,
            peak_hwm_before_kb: None,
            peak_hwm_after_await_kb: None,
            rss_growth_kb: mean_option_i64(turns.iter().map(|turn| turn.memory.rss_growth_kb)),
            hwm_growth_kb: mean_option_i64(turns.iter().map(|turn| turn.memory.hwm_growth_kb)),
        },
        allocations: RuntimePerfTurnAllocationRunResult {
            run_turn: mean_allocation_delta(turns.iter().map(|turn| &turn.allocations.run_turn)),
            await_background_work: mean_allocation_delta(
                turns
                    .iter()
                    .map(|turn| &turn.allocations.await_background_work),
            ),
            total: mean_allocation_delta(turns.iter().map(|turn| &turn.allocations.total)),
        },
        phase_profile: mean_phase_profiles(turns.iter().map(|turn| &turn.phase_profile)),
        turn_usage: mean_token_usage(turns.iter().map(|turn| &turn.turn_usage)),
        usage_delta: SessionUsageReport::default(),
        cumulative_usage: SessionUsageReport::default(),
    })
}
fn summarize_metric(values: Vec<f64>) -> RuntimePerfMetricSummary {
    basic_summary(values)
}

fn summarize_optional_metric(values: Vec<f64>) -> Option<RuntimePerfMetricSummary> {
    optional_basic_summary(values)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_perf::openai_compat::openai_compat_sse_body;
    use crate::runtime_perf::providers::benchmark_stream_profile;

    #[test]
    fn default_scenarios_cover_all_synthetic_runtime_paths() {
        assert_eq!(
            resolve_scenarios(&[]).unwrap(),
            RuntimePerfScenario::KNOWN.to_vec()
        );
        assert_eq!(
            resolve_scenarios(&["all".to_string()]).unwrap(),
            RuntimePerfScenario::KNOWN.to_vec()
        );
        assert_eq!(
            resolve_scenarios(&["standard".to_string(), "all".to_string()])
                .unwrap()
                .len(),
            RuntimePerfScenario::KNOWN.len()
        );
    }

    #[test]
    fn openai_compat_stream_fixture_uses_chat_completions_sse_shape() {
        let profile = benchmark_stream_profile(RuntimePerfScenario::OpenAiCompatStream);
        let body = String::from_utf8(openai_compat_sse_body(&profile)).unwrap();
        assert!(body.contains(r#""object":"chat.completion.chunk""#));
        assert!(body.contains(r#""choices""#));
        assert!(body.contains(r#""delta":{"content":"#));
        assert!(body.contains(r#""usage":{"#));
        assert!(!body.contains(r#""type":"response.output_text.delta""#));
    }
}
