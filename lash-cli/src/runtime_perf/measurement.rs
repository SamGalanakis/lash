use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Context;
use lash::usage::SessionUsageReport;
use lash_core::runtime::{RuntimeTurnPhase, RuntimeTurnPhaseProbe};
use lash_core::{InputItem, TokenUsage, TurnInput};
use lash_mode_rlm::RlmTurnInputExt;
use serde::Serialize;
use stats_alloc::Stats;
use tokio_util::sync::CancellationToken;

use crate::perf_support::memory::{ProcessMemorySample, diff_opt_i64, process_memory_sample};
use crate::perf_support::metrics::BasicMetricSummary as RuntimePerfMetricSummary;
use crate::perf_support::time::{elapsed_ms, round3};

use super::harness::{
    benchmark_prompt, build_embed_core, build_runtime, prepare_turn, rlm_perf_projected_bindings,
    seed_runtime_state, validate_runtime_perf_turn,
};
use super::scenarios::RuntimePerfScenario;
use super::store::RuntimePerfStore;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfRunResult {
    pub(crate) scenario: String,
    pub(crate) chat_turns: usize,
    pub(crate) build_runtime_ms: f64,
    pub(crate) seed_state_ms: f64,
    pub(crate) run_turn_ms: f64,
    pub(crate) await_background_work_ms: f64,
    pub(crate) export_state_ms: f64,
    pub(crate) total_ms: f64,
    pub(crate) session_nodes: usize,
    pub(crate) active_path_messages: usize,
    pub(crate) memory: RuntimePerfMemoryRunResult,
    pub(crate) allocations: RuntimePerfAllocationRunResult,
    pub(crate) phase_profile: BTreeMap<String, RuntimePerfPhaseRunResult>,
    pub(crate) turns: Vec<RuntimePerfTurnResult>,
    pub(crate) cumulative_usage: SessionUsageReport,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfTurnResult {
    pub(crate) turn_index: usize,
    pub(crate) run_turn_ms: f64,
    pub(crate) await_background_work_ms: f64,
    pub(crate) total_ms: f64,
    pub(crate) memory: RuntimePerfTurnMemoryRunResult,
    pub(crate) allocations: RuntimePerfTurnAllocationRunResult,
    pub(crate) phase_profile: BTreeMap<String, RuntimePerfPhaseRunResult>,
    pub(crate) turn_usage: TokenUsage,
    pub(crate) usage_delta: SessionUsageReport,
    pub(crate) cumulative_usage: SessionUsageReport,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfTurnMemoryRunResult {
    pub(crate) rss_before_kb: Option<u64>,
    pub(crate) rss_after_turn_kb: Option<u64>,
    pub(crate) rss_after_await_kb: Option<u64>,
    pub(crate) peak_hwm_before_kb: Option<u64>,
    pub(crate) peak_hwm_after_await_kb: Option<u64>,
    pub(crate) rss_growth_kb: Option<i64>,
    pub(crate) hwm_growth_kb: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfTurnAllocationRunResult {
    pub(crate) run_turn: RuntimePerfAllocationDelta,
    pub(crate) await_background_work: RuntimePerfAllocationDelta,
    pub(crate) total: RuntimePerfAllocationDelta,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfMemoryRunResult {
    pub(crate) rss_before_kb: Option<u64>,
    pub(crate) rss_after_build_kb: Option<u64>,
    pub(crate) rss_after_seed_kb: Option<u64>,
    pub(crate) rss_after_turn_kb: Option<u64>,
    pub(crate) rss_after_await_kb: Option<u64>,
    pub(crate) rss_after_export_kb: Option<u64>,
    pub(crate) peak_hwm_before_kb: Option<u64>,
    pub(crate) peak_hwm_after_export_kb: Option<u64>,
    pub(crate) rss_growth_kb: Option<i64>,
    pub(crate) hwm_growth_kb: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfAllocationDelta {
    pub(crate) allocations: usize,
    pub(crate) deallocations: usize,
    pub(crate) reallocations: usize,
    pub(crate) bytes_allocated: usize,
    pub(crate) bytes_deallocated: usize,
    pub(crate) bytes_reallocated: isize,
    pub(crate) net_live_bytes: i64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfAllocationRunResult {
    pub(crate) build_runtime: RuntimePerfAllocationDelta,
    pub(crate) seed_state: RuntimePerfAllocationDelta,
    pub(crate) run_turn: RuntimePerfAllocationDelta,
    pub(crate) await_background_work: RuntimePerfAllocationDelta,
    pub(crate) export_state: RuntimePerfAllocationDelta,
    pub(crate) total: RuntimePerfAllocationDelta,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfPhaseRunResult {
    pub(crate) duration_ms: f64,
    pub(crate) allocations: RuntimePerfAllocationDelta,
    pub(crate) rss_growth_kb: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfPhaseSummary {
    pub(crate) duration_ms: RuntimePerfMetricSummary,
    pub(crate) alloc_bytes: RuntimePerfMetricSummary,
    pub(crate) live_bytes: RuntimePerfMetricSummary,
    pub(crate) rss_growth_kb: Option<RuntimePerfMetricSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfScenarioSummary {
    pub(crate) scenario: String,
    pub(crate) runs: usize,
    pub(crate) chat_turns: usize,
    pub(crate) build_runtime_ms: RuntimePerfMetricSummary,
    pub(crate) seed_state_ms: RuntimePerfMetricSummary,
    pub(crate) run_turn_ms: RuntimePerfMetricSummary,
    pub(crate) await_background_work_ms: RuntimePerfMetricSummary,
    pub(crate) export_state_ms: RuntimePerfMetricSummary,
    pub(crate) total_ms: RuntimePerfMetricSummary,
    pub(crate) rss_after_export_kb: Option<RuntimePerfMetricSummary>,
    pub(crate) rss_growth_kb: Option<RuntimePerfMetricSummary>,
    pub(crate) hwm_growth_kb: Option<RuntimePerfMetricSummary>,
    pub(crate) build_runtime_alloc_bytes: RuntimePerfMetricSummary,
    pub(crate) build_runtime_live_bytes: RuntimePerfMetricSummary,
    pub(crate) seed_state_alloc_bytes: RuntimePerfMetricSummary,
    pub(crate) seed_state_live_bytes: RuntimePerfMetricSummary,
    pub(crate) run_turn_alloc_bytes: RuntimePerfMetricSummary,
    pub(crate) run_turn_live_bytes: RuntimePerfMetricSummary,
    pub(crate) await_background_work_alloc_bytes: RuntimePerfMetricSummary,
    pub(crate) await_background_work_live_bytes: RuntimePerfMetricSummary,
    pub(crate) export_state_alloc_bytes: RuntimePerfMetricSummary,
    pub(crate) export_state_live_bytes: RuntimePerfMetricSummary,
    pub(crate) total_alloc_bytes: RuntimePerfMetricSummary,
    pub(crate) total_live_bytes: RuntimePerfMetricSummary,
    pub(crate) phase_summary: BTreeMap<String, RuntimePerfPhaseSummary>,
    pub(crate) first_turn: RuntimePerfTurnSummary,
    pub(crate) steady_state_turn: Option<RuntimePerfTurnSummary>,
    pub(crate) last_turn: RuntimePerfTurnSummary,
    pub(crate) sample_session_nodes: usize,
    pub(crate) sample_active_path_messages: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfTurnSummary {
    pub(crate) total_ms: RuntimePerfMetricSummary,
    pub(crate) run_turn_ms: RuntimePerfMetricSummary,
    pub(crate) await_background_work_ms: RuntimePerfMetricSummary,
    pub(crate) rss_growth_kb: Option<RuntimePerfMetricSummary>,
    pub(crate) total_alloc_bytes: RuntimePerfMetricSummary,
    pub(crate) total_live_bytes: RuntimePerfMetricSummary,
    pub(crate) phase_summary: BTreeMap<String, RuntimePerfPhaseSummary>,
}

#[derive(Clone, Copy)]
struct PhaseStart {
    started_at: Instant,
    alloc_before: Stats,
    memory_before: ProcessMemorySample,
}

#[derive(Default)]
struct RuntimePerfPhaseProbeState {
    started: HashMap<RuntimeTurnPhase, PhaseStart>,
    completed: BTreeMap<String, RuntimePerfPhaseRunResult>,
}

#[derive(Default)]
struct RuntimePerfPhaseProbe {
    state: Mutex<RuntimePerfPhaseProbeState>,
}

impl RuntimePerfPhaseProbe {
    fn take_completed(&self) -> BTreeMap<String, RuntimePerfPhaseRunResult> {
        let mut state = self.state.lock().expect("phase probe lock");
        std::mem::take(&mut state.completed)
    }
}

impl RuntimeTurnPhaseProbe for RuntimePerfPhaseProbe {
    fn begin(&self, phase: RuntimeTurnPhase) {
        let mut state = self.state.lock().expect("phase probe lock");
        state.started.insert(
            phase,
            PhaseStart {
                started_at: Instant::now(),
                alloc_before: allocator_stats(),
                memory_before: process_memory_sample(),
            },
        );
    }

    fn end(&self, phase: RuntimeTurnPhase) {
        let mut state = self.state.lock().expect("phase probe lock");
        let Some(start) = state.started.remove(&phase) else {
            return;
        };
        let alloc_after = allocator_stats();
        let memory_after = process_memory_sample();
        state.completed.insert(
            phase_name(phase).to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(start.started_at),
                allocations: alloc_delta(start.alloc_before, alloc_after),
                rss_growth_kb: diff_opt_i64(start.memory_before.rss_kb, memory_after.rss_kb),
            },
        );
    }
}
pub(crate) async fn run_once(
    scenario: RuntimePerfScenario,
    chat_turns: usize,
) -> anyhow::Result<RuntimePerfRunResult> {
    if matches!(
        scenario,
        RuntimePerfScenario::EmbedStandard | RuntimePerfScenario::EmbedRlm
    ) {
        return run_once_embed(scenario, chat_turns).await;
    }

    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let mut runtime = build_runtime(scenario).await?;
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    seed_runtime_state(&mut runtime, scenario).await?;
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    for turn_index in 0..chat_turns {
        prepare_turn(&mut runtime, scenario, turn_index).await?;

        let phase_probe = Arc::new(RuntimePerfPhaseProbe::default());
        runtime.set_turn_phase_probe(phase_probe.clone());

        let before_turn_usage = runtime.usage_report();
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let mut turn_input = TurnInput {
            items: vec![InputItem::Text {
                text: benchmark_prompt(scenario, turn_index),
            }],
            image_blobs: Default::default(),
            mode_turn_options: None,
            trace_turn_id: None,
            mode_extension: None,
            turn_context: lash_core::TurnContext::default(),
        };
        if matches!(scenario, RuntimePerfScenario::RlmGlobals) {
            turn_input =
                turn_input.rlm_project(rlm_perf_projected_bindings(scenario, turn_index)?)?;
        }
        let turn = runtime
            .run_turn_assembled(turn_input, CancellationToken::new())
            .await
            .with_context(|| {
                format!(
                    "run runtime perf scenario {} turn {}",
                    scenario.name(),
                    turn_index + 1
                )
            })?;
        validate_runtime_perf_turn(scenario, turn_index, &turn.outcome, &turn.assistant_output)?;
        let run_turn_ms = elapsed_ms(turn_started);
        let run_turn_alloc = alloc_delta(turn_before_alloc, allocator_stats());
        let after_turn_memory = process_memory_sample();

        let await_before_alloc = allocator_stats();
        let background_started = Instant::now();
        runtime.await_background_work().await.with_context(|| {
            format!(
                "await background work for {} turn {}",
                scenario.name(),
                turn_index + 1
            )
        })?;
        let await_background_work_ms = elapsed_ms(background_started);
        let await_background_work_alloc = alloc_delta(await_before_alloc, allocator_stats());
        let after_await_memory = process_memory_sample();
        let turn_total_alloc =
            sum_allocation_deltas([&run_turn_alloc, &await_background_work_alloc]);

        let cumulative_usage = runtime.usage_report();
        let usage_delta_entries =
            lash_core::diff_usage_reports(&before_turn_usage, &cumulative_usage)
                .map_err(anyhow::Error::msg)?;
        turns.push(RuntimePerfTurnResult {
            turn_index,
            run_turn_ms,
            await_background_work_ms,
            total_ms: round3(run_turn_ms + await_background_work_ms),
            memory: RuntimePerfTurnMemoryRunResult {
                rss_before_kb: turn_before_memory.rss_kb,
                rss_after_turn_kb: after_turn_memory.rss_kb,
                rss_after_await_kb: after_await_memory.rss_kb,
                peak_hwm_before_kb: turn_before_memory.hwm_kb,
                peak_hwm_after_await_kb: after_await_memory.hwm_kb,
                rss_growth_kb: diff_opt_i64(turn_before_memory.rss_kb, after_await_memory.rss_kb),
                hwm_growth_kb: diff_opt_i64(turn_before_memory.hwm_kb, after_await_memory.hwm_kb),
            },
            allocations: RuntimePerfTurnAllocationRunResult {
                run_turn: run_turn_alloc,
                await_background_work: await_background_work_alloc,
                total: turn_total_alloc,
            },
            phase_profile: phase_probe.take_completed(),
            turn_usage: turn.token_usage,
            usage_delta: SessionUsageReport::from_entries(&usage_delta_entries),
            cumulative_usage,
        });
    }

    let export_before_alloc = allocator_stats();
    let export_started = Instant::now();
    let state = runtime.export_state();
    let cumulative_usage = runtime.usage_report();
    let export_state_ms = elapsed_ms(export_started);
    let export_state_alloc = alloc_delta(export_before_alloc, allocator_stats());
    let after_export_memory = process_memory_sample();
    let total_alloc = alloc_delta(total_before_alloc, allocator_stats());
    let last_turn_memory = turns.last().map(|turn| &turn.memory);

    Ok(RuntimePerfRunResult {
        scenario: scenario.name().to_string(),
        chat_turns,
        build_runtime_ms,
        seed_state_ms,
        run_turn_ms: round3(turns.iter().map(|turn| turn.run_turn_ms).sum()),
        await_background_work_ms: round3(
            turns.iter().map(|turn| turn.await_background_work_ms).sum(),
        ),
        export_state_ms,
        total_ms: elapsed_ms(total_started),
        session_nodes: state.session_graph.nodes.len(),
        active_path_messages: state.read_view().messages().len(),
        memory: RuntimePerfMemoryRunResult {
            rss_before_kb: before_memory.rss_kb,
            rss_after_build_kb: after_build_memory.rss_kb,
            rss_after_seed_kb: after_seed_memory.rss_kb,
            rss_after_turn_kb: last_turn_memory.and_then(|memory| memory.rss_after_turn_kb),
            rss_after_await_kb: last_turn_memory.and_then(|memory| memory.rss_after_await_kb),
            rss_after_export_kb: after_export_memory.rss_kb,
            peak_hwm_before_kb: before_memory.hwm_kb,
            peak_hwm_after_export_kb: after_export_memory.hwm_kb,
            rss_growth_kb: diff_opt_i64(before_memory.rss_kb, after_export_memory.rss_kb),
            hwm_growth_kb: diff_opt_i64(before_memory.hwm_kb, after_export_memory.hwm_kb),
        },
        allocations: RuntimePerfAllocationRunResult {
            build_runtime: build_runtime_alloc,
            seed_state: seed_state_alloc,
            run_turn: sum_allocation_deltas(turns.iter().map(|turn| &turn.allocations.run_turn)),
            await_background_work: sum_allocation_deltas(
                turns
                    .iter()
                    .map(|turn| &turn.allocations.await_background_work),
            ),
            export_state: export_state_alloc,
            total: total_alloc,
        },
        phase_profile: sum_phase_profiles(turns.iter().map(|turn| &turn.phase_profile)),
        turns,
        cumulative_usage,
    })
}

pub(crate) async fn run_once_embed(
    scenario: RuntimePerfScenario,
    chat_turns: usize,
) -> anyhow::Result<RuntimePerfRunResult> {
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let store = Arc::new(RuntimePerfStore::default());
    let core = build_embed_core(scenario, Arc::clone(&store))?;
    let session = core
        .session(format!("runtime-perf-{}", scenario.name()))
        .mode(lash::ModeId::new(scenario.execution_mode().plugin_id()))
        .open()
        .await
        .with_context(|| format!("open embed session for {}", scenario.name()))?;
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    for turn_index in 0..chat_turns {
        let before_turn_usage = SessionUsageReport::default();
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let turn = session
            .run(lash_core::TurnInput::text(benchmark_prompt(
                scenario, turn_index,
            )))
            .await
            .with_context(|| {
                format!(
                    "run embed runtime perf scenario {} turn {}",
                    scenario.name(),
                    turn_index + 1
                )
            })?;
        validate_runtime_perf_turn(
            scenario,
            turn_index,
            &turn.result.outcome,
            &turn.result.assistant_output,
        )?;
        let run_turn_ms = elapsed_ms(turn_started);
        let run_turn_alloc = alloc_delta(turn_before_alloc, allocator_stats());
        let after_turn_memory = process_memory_sample();

        let await_before_alloc = allocator_stats();
        let background_started = Instant::now();
        let await_background_work_ms = elapsed_ms(background_started);
        let await_background_work_alloc = alloc_delta(await_before_alloc, allocator_stats());
        let after_await_memory = process_memory_sample();
        let turn_total_alloc =
            sum_allocation_deltas([&run_turn_alloc, &await_background_work_alloc]);

        turns.push(RuntimePerfTurnResult {
            turn_index,
            run_turn_ms,
            await_background_work_ms,
            total_ms: round3(run_turn_ms + await_background_work_ms),
            memory: RuntimePerfTurnMemoryRunResult {
                rss_before_kb: turn_before_memory.rss_kb,
                rss_after_turn_kb: after_turn_memory.rss_kb,
                rss_after_await_kb: after_await_memory.rss_kb,
                peak_hwm_before_kb: turn_before_memory.hwm_kb,
                peak_hwm_after_await_kb: after_await_memory.hwm_kb,
                rss_growth_kb: diff_opt_i64(turn_before_memory.rss_kb, after_await_memory.rss_kb),
                hwm_growth_kb: diff_opt_i64(turn_before_memory.hwm_kb, after_await_memory.hwm_kb),
            },
            allocations: RuntimePerfTurnAllocationRunResult {
                run_turn: run_turn_alloc,
                await_background_work: await_background_work_alloc,
                total: turn_total_alloc,
            },
            phase_profile: BTreeMap::new(),
            turn_usage: turn.result.usage,
            usage_delta: before_turn_usage,
            cumulative_usage: SessionUsageReport::default(),
        });
    }

    let export_before_alloc = allocator_stats();
    let export_started = Instant::now();
    let read_view = session.read_view();
    let export_state_ms = elapsed_ms(export_started);
    let export_state_alloc = alloc_delta(export_before_alloc, allocator_stats());
    let after_export_memory = process_memory_sample();
    let total_alloc = alloc_delta(total_before_alloc, allocator_stats());
    let last_turn_memory = turns.last().map(|turn| &turn.memory);

    Ok(RuntimePerfRunResult {
        scenario: scenario.name().to_string(),
        chat_turns,
        build_runtime_ms,
        seed_state_ms,
        run_turn_ms: round3(turns.iter().map(|turn| turn.run_turn_ms).sum()),
        await_background_work_ms: round3(
            turns.iter().map(|turn| turn.await_background_work_ms).sum(),
        ),
        export_state_ms,
        total_ms: elapsed_ms(total_started),
        session_nodes: store.graph_node_count(),
        active_path_messages: read_view.messages().len(),
        memory: RuntimePerfMemoryRunResult {
            rss_before_kb: before_memory.rss_kb,
            rss_after_build_kb: after_build_memory.rss_kb,
            rss_after_seed_kb: after_seed_memory.rss_kb,
            rss_after_turn_kb: last_turn_memory.and_then(|memory| memory.rss_after_turn_kb),
            rss_after_await_kb: last_turn_memory.and_then(|memory| memory.rss_after_await_kb),
            rss_after_export_kb: after_export_memory.rss_kb,
            peak_hwm_before_kb: before_memory.hwm_kb,
            peak_hwm_after_export_kb: after_export_memory.hwm_kb,
            rss_growth_kb: diff_opt_i64(before_memory.rss_kb, after_export_memory.rss_kb),
            hwm_growth_kb: diff_opt_i64(before_memory.hwm_kb, after_export_memory.hwm_kb),
        },
        allocations: RuntimePerfAllocationRunResult {
            build_runtime: build_runtime_alloc,
            seed_state: seed_state_alloc,
            run_turn: sum_allocation_deltas(turns.iter().map(|turn| &turn.allocations.run_turn)),
            await_background_work: sum_allocation_deltas(
                turns
                    .iter()
                    .map(|turn| &turn.allocations.await_background_work),
            ),
            export_state: export_state_alloc,
            total: total_alloc,
        },
        phase_profile: BTreeMap::new(),
        turns,
        cumulative_usage: SessionUsageReport::default(),
    })
}
pub(crate) fn sum_phase_profiles<'a>(
    profiles: impl IntoIterator<Item = &'a BTreeMap<String, RuntimePerfPhaseRunResult>>,
) -> BTreeMap<String, RuntimePerfPhaseRunResult> {
    let mut totals: BTreeMap<String, RuntimePerfPhaseRunResult> = BTreeMap::new();
    for profile in profiles {
        for (phase, metrics) in profile {
            let entry = totals
                .entry(phase.clone())
                .or_insert_with(|| RuntimePerfPhaseRunResult {
                    duration_ms: 0.0,
                    allocations: zero_allocation_delta(),
                    rss_growth_kb: Some(0),
                });
            entry.duration_ms = round3(entry.duration_ms + metrics.duration_ms);
            entry.allocations = sum_allocation_deltas([&entry.allocations, &metrics.allocations]);
            entry.rss_growth_kb = sum_optional_i64(entry.rss_growth_kb, metrics.rss_growth_kb);
        }
    }
    totals
}

pub(crate) fn mean_phase_profiles<'a>(
    profiles: impl IntoIterator<Item = &'a BTreeMap<String, RuntimePerfPhaseRunResult>>,
) -> BTreeMap<String, RuntimePerfPhaseRunResult> {
    let profiles = profiles.into_iter().collect::<Vec<_>>();
    if profiles.is_empty() {
        return BTreeMap::new();
    }
    let count = profiles.len() as f64;
    let sums = sum_phase_profiles(profiles);
    sums.into_iter()
        .map(|(phase, metrics)| {
            (
                phase,
                RuntimePerfPhaseRunResult {
                    duration_ms: round3(metrics.duration_ms / count),
                    allocations: scale_allocation_delta(&metrics.allocations, count),
                    rss_growth_kb: metrics
                        .rss_growth_kb
                        .map(|value| ((value as f64) / count).round() as i64),
                },
            )
        })
        .collect()
}

pub(crate) fn sum_allocation_deltas<'a>(
    deltas: impl IntoIterator<Item = &'a RuntimePerfAllocationDelta>,
) -> RuntimePerfAllocationDelta {
    let mut total = zero_allocation_delta();
    for delta in deltas {
        total.allocations += delta.allocations;
        total.deallocations += delta.deallocations;
        total.reallocations += delta.reallocations;
        total.bytes_allocated += delta.bytes_allocated;
        total.bytes_deallocated += delta.bytes_deallocated;
        total.bytes_reallocated += delta.bytes_reallocated;
        total.net_live_bytes += delta.net_live_bytes;
    }
    total
}

pub(crate) fn mean_allocation_delta<'a>(
    deltas: impl IntoIterator<Item = &'a RuntimePerfAllocationDelta>,
) -> RuntimePerfAllocationDelta {
    let deltas = deltas.into_iter().collect::<Vec<_>>();
    if deltas.is_empty() {
        return zero_allocation_delta();
    }
    let count = deltas.len() as f64;
    scale_allocation_delta(&sum_allocation_deltas(deltas), count)
}

pub(crate) fn scale_allocation_delta(
    delta: &RuntimePerfAllocationDelta,
    divisor: f64,
) -> RuntimePerfAllocationDelta {
    RuntimePerfAllocationDelta {
        allocations: ((delta.allocations as f64) / divisor).round() as usize,
        deallocations: ((delta.deallocations as f64) / divisor).round() as usize,
        reallocations: ((delta.reallocations as f64) / divisor).round() as usize,
        bytes_allocated: ((delta.bytes_allocated as f64) / divisor).round() as usize,
        bytes_deallocated: ((delta.bytes_deallocated as f64) / divisor).round() as usize,
        bytes_reallocated: ((delta.bytes_reallocated as f64) / divisor).round() as isize,
        net_live_bytes: ((delta.net_live_bytes as f64) / divisor).round() as i64,
    }
}

pub(crate) fn zero_allocation_delta() -> RuntimePerfAllocationDelta {
    RuntimePerfAllocationDelta {
        allocations: 0,
        deallocations: 0,
        reallocations: 0,
        bytes_allocated: 0,
        bytes_deallocated: 0,
        bytes_reallocated: 0,
        net_live_bytes: 0,
    }
}

pub(crate) fn mean_token_usage<'a>(usages: impl IntoIterator<Item = &'a TokenUsage>) -> TokenUsage {
    let usages = usages.into_iter().collect::<Vec<_>>();
    if usages.is_empty() {
        return TokenUsage::default();
    }
    let count = usages.len() as i64;
    TokenUsage {
        input_tokens: usages.iter().map(|usage| usage.input_tokens).sum::<i64>() / count,
        output_tokens: usages.iter().map(|usage| usage.output_tokens).sum::<i64>() / count,
        cached_input_tokens: usages
            .iter()
            .map(|usage| usage.cached_input_tokens)
            .sum::<i64>()
            / count,
        reasoning_tokens: usages
            .iter()
            .map(|usage| usage.reasoning_tokens)
            .sum::<i64>()
            / count,
    }
}

pub(crate) fn mean_option_i64(values: impl IntoIterator<Item = Option<i64>>) -> Option<i64> {
    let values = values.into_iter().flatten().collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some((values.iter().sum::<i64>() as f64 / values.len() as f64).round() as i64)
    }
}

pub(crate) fn sum_optional_i64(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}
pub(crate) fn phase_name(phase: RuntimeTurnPhase) -> &'static str {
    match phase {
        RuntimeTurnPhase::ContextTransform => "context_transform",
        RuntimeTurnPhase::BeforeTurnHooks => "before_turn_hooks",
        RuntimeTurnPhase::PromptBuild => "prompt_build",
        RuntimeTurnPhase::EffectLoop => "effect_loop",
        RuntimeTurnPhase::FinalizeTurn => "finalize_turn",
        RuntimeTurnPhase::PersistTurn => "persist_turn",
    }
}

#[cfg(not(feature = "dhat-heap"))]
pub(crate) fn allocator_stats() -> Stats {
    crate::GLOBAL_ALLOCATOR.stats()
}

#[cfg(feature = "dhat-heap")]
pub(crate) fn allocator_stats() -> Stats {
    Stats::default()
}

pub(crate) fn alloc_delta(before: Stats, after: Stats) -> RuntimePerfAllocationDelta {
    let diff = after - before;
    RuntimePerfAllocationDelta {
        allocations: diff.allocations,
        deallocations: diff.deallocations,
        reallocations: diff.reallocations,
        bytes_allocated: diff.bytes_allocated,
        bytes_deallocated: diff.bytes_deallocated,
        bytes_reallocated: diff.bytes_reallocated,
        net_live_bytes: diff.bytes_allocated as i64 - diff.bytes_deallocated as i64,
    }
}
