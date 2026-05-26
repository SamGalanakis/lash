use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use lash::usage::SessionUsageReport;
use lash_core::llm::types::LlmResponse;
use lash_core::runtime::{RuntimeTurnPhase, RuntimeTurnPhaseProbe};
use lash_core::sansio::{
    ChatContextProjector, CompletedToolCall, PendingToolCall, ProtocolDriverHandle,
    WaitingExecState, WaitingLlmState,
};
use lash_core::{
    DriverAction, DriverContextView, Effect, ExecResponse, InputItem, Message, MessageRole,
    ModeTurnOptions, ModelToolReturn, Part, PartKind, PruneState, Response, TokenUsage,
    ToolCallOutput, ToolCancellation, ToolFailure, ToolFailureClass, TurnFinish, TurnInput,
    TurnMachine, TurnMachineConfig, TurnOutcome, shared_parts,
};
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

const RUNTIME_PERF_TURN_TIMEOUT_ENV: &str = "LASH_RUNTIME_PERF_TURN_TIMEOUT_MS";
const DEFAULT_RUNTIME_PERF_TURN_TIMEOUT: Duration = Duration::from_secs(10);

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

async fn runtime_perf_timed<T, F>(
    scenario: RuntimePerfScenario,
    turn_index: usize,
    phase: &str,
    cancel: Option<CancellationToken>,
    future: F,
) -> anyhow::Result<T>
where
    F: Future<Output = anyhow::Result<T>>,
{
    let timeout = runtime_perf_turn_timeout();
    match tokio::time::timeout(timeout, future).await {
        Ok(result) => result,
        Err(_) => {
            if let Some(cancel) = cancel {
                cancel.cancel();
            }
            anyhow::bail!(
                "runtime perf scenario {} turn {} {phase} timed out after {} ms; profiling aborts instead of looping. Override with {RUNTIME_PERF_TURN_TIMEOUT_ENV}.",
                scenario.name(),
                turn_index + 1,
                timeout.as_millis()
            );
        }
    }
}

fn runtime_perf_turn_timeout() -> Duration {
    std::env::var(RUNTIME_PERF_TURN_TIMEOUT_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_RUNTIME_PERF_TURN_TIMEOUT)
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

struct ScopedPerfEffectController;

#[async_trait::async_trait]
impl lash::advanced::RuntimeEffectController for ScopedPerfEffectController {
    async fn execute_effect(
        &self,
        envelope: lash::advanced::RuntimeEffectEnvelope,
        local_executor: lash::advanced::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<lash::advanced::RuntimeEffectOutcome, lash::advanced::RuntimeEffectControllerError>
    {
        local_executor.execute(envelope).await
    }
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
    if matches!(scenario, RuntimePerfScenario::TurnCheckpoint) {
        return run_once_turn_checkpoint(chat_turns).await;
    }

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
        let mut extra_phase_profile = BTreeMap::new();
        if matches!(scenario, RuntimePerfScenario::StoreReopen) && turn_index > 0 {
            let store = runtime.store();
            let store_factory_before_alloc = allocator_stats();
            let store_factory_before_memory = process_memory_sample();
            let store_factory_started = Instant::now();
            let _core = runtime.core();
            extra_phase_profile.insert(
                "store_reopen.store_factory_create".to_string(),
                RuntimePerfPhaseRunResult {
                    duration_ms: elapsed_ms(store_factory_started),
                    allocations: alloc_delta(store_factory_before_alloc, allocator_stats()),
                    rss_growth_kb: diff_opt_i64(
                        store_factory_before_memory.rss_kb,
                        process_memory_sample().rss_kb,
                    ),
                },
            );

            let load_before_alloc = allocator_stats();
            let load_before_memory = process_memory_sample();
            let load_started = Instant::now();
            let state =
                lash::persistence::load_persisted_session_state_active_path(store.as_ref(), None)
                    .await?
                    .ok_or_else(|| {
                        anyhow::anyhow!("store_reopen expected persisted session state")
                    })?;
            extra_phase_profile.insert(
                "store_reopen.persisted_load".to_string(),
                RuntimePerfPhaseRunResult {
                    duration_ms: elapsed_ms(load_started),
                    allocations: alloc_delta(load_before_alloc, allocator_stats()),
                    rss_growth_kb: diff_opt_i64(
                        load_before_memory.rss_kb,
                        process_memory_sample().rss_kb,
                    ),
                },
            );

            let hydrate_before_alloc = allocator_stats();
            let hydrate_before_memory = process_memory_sample();
            let hydrate_started = Instant::now();
            runtime.reopen_with_state(scenario, state).await?;
            extra_phase_profile.insert(
                "store_reopen.runtime_hydration".to_string(),
                RuntimePerfPhaseRunResult {
                    duration_ms: elapsed_ms(hydrate_started),
                    allocations: alloc_delta(hydrate_before_alloc, allocator_stats()),
                    rss_growth_kb: diff_opt_i64(
                        hydrate_before_memory.rss_kb,
                        process_memory_sample().rss_kb,
                    ),
                },
            );
        }
        prepare_turn(&mut runtime, scenario, turn_index).await?;

        let phase_probe = Arc::new(RuntimePerfPhaseProbe::default());
        runtime.set_turn_phase_probe(phase_probe.clone()).await;

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
        let cancel = CancellationToken::new();
        let turn = if matches!(scenario, RuntimePerfScenario::ScopedEffectController) {
            let effect_controller = ScopedPerfEffectController;
            let turn_id = format!("runtime-perf-scoped-{}", turn_index + 1);
            let effect_scope =
                lash::advanced::RuntimeEffectControllerScope::new(&effect_controller, &turn_id)
                    .map_err(anyhow::Error::from)?;
            runtime_perf_timed(
                scenario,
                turn_index,
                "run_turn",
                Some(cancel.clone()),
                runtime.run_turn_with_effect_scope(turn_input, cancel, effect_scope),
            )
            .await
        } else {
            runtime_perf_timed(
                scenario,
                turn_index,
                "run_turn",
                Some(cancel.clone()),
                runtime.run_turn(turn_input, cancel),
            )
            .await
        }
        .with_context(|| {
            format!(
                "run runtime perf scenario {} turn {}",
                scenario.name(),
                turn_index + 1
            )
        })?;
        validate_runtime_perf_turn(scenario, turn_index, &turn)?;
        let run_turn_ms = elapsed_ms(turn_started);
        let run_turn_alloc = alloc_delta(turn_before_alloc, allocator_stats());
        let after_turn_memory = process_memory_sample();

        let await_before_alloc = allocator_stats();
        let background_started = Instant::now();
        runtime_perf_timed(
            scenario,
            turn_index,
            "await_background_work",
            None,
            runtime.await_background_work(),
        )
        .await
        .with_context(|| {
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
        let mut phase_profile = phase_probe.take_completed();
        phase_profile.extend(extra_phase_profile);
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
            phase_profile,
            turn_usage: turn.usage,
            usage_delta: SessionUsageReport::from_entries(&usage_delta_entries),
            cumulative_usage,
        });
    }

    let export_before_alloc = allocator_stats();
    let export_started = Instant::now();
    let state = runtime.export_state().await;
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

async fn run_once_turn_checkpoint(chat_turns: usize) -> anyhow::Result<RuntimePerfRunResult> {
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let configs = CheckpointConfigs::new();
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    let seed_messages = checkpoint_messages();
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    for turn_index in 0..chat_turns {
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let mut phase_profile = BTreeMap::new();

        let llm_phase = measure_checkpoint_phase("standard_llm_checkpoint", || {
            checkpoint_pending_llm(&configs, &seed_messages, turn_index)
        })?;
        phase_profile.insert(llm_phase.0, llm_phase.1);

        let tools_phase = measure_checkpoint_phase("standard_parallel_tools_checkpoint", || {
            checkpoint_pending_parallel_tools(&configs, &seed_messages, turn_index)
        })?;
        phase_profile.insert(tools_phase.0, tools_phase.1);

        let exec_phase = measure_checkpoint_phase("rlm_exec_checkpoint", || {
            checkpoint_pending_exec(&configs, &seed_messages, turn_index)
        })?;
        phase_profile.insert(exec_phase.0, exec_phase.1);

        let run_turn_ms = elapsed_ms(turn_started);
        let run_turn_alloc = alloc_delta(turn_before_alloc, allocator_stats());
        let after_turn_memory = process_memory_sample();

        let await_before_alloc = allocator_stats();
        let background_started = Instant::now();
        tokio::task::yield_now().await;
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
            phase_profile,
            turn_usage: TokenUsage::default(),
            usage_delta: SessionUsageReport::default(),
            cumulative_usage: SessionUsageReport::default(),
        });
    }

    let export_before_alloc = allocator_stats();
    let export_started = Instant::now();
    serde_json::to_vec(&seed_messages)?;
    let export_state_ms = elapsed_ms(export_started);
    let export_state_alloc = alloc_delta(export_before_alloc, allocator_stats());
    let after_export_memory = process_memory_sample();
    let total_alloc = alloc_delta(total_before_alloc, allocator_stats());
    let last_turn_memory = turns.last().map(|turn| &turn.memory);

    Ok(RuntimePerfRunResult {
        scenario: RuntimePerfScenario::TurnCheckpoint.name().to_string(),
        chat_turns,
        build_runtime_ms,
        seed_state_ms,
        run_turn_ms: round3(turns.iter().map(|turn| turn.run_turn_ms).sum()),
        await_background_work_ms: round3(
            turns.iter().map(|turn| turn.await_background_work_ms).sum(),
        ),
        export_state_ms,
        total_ms: elapsed_ms(total_started),
        session_nodes: seed_messages.len(),
        active_path_messages: seed_messages.len(),
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
        cumulative_usage: SessionUsageReport::default(),
    })
}

struct CheckpointConfigs {
    llm: Arc<dyn ProtocolDriverHandle<lash_core::HostModeProtocol>>,
    tools: Arc<dyn ProtocolDriverHandle<lash_core::HostModeProtocol>>,
    exec: Arc<dyn ProtocolDriverHandle<lash_core::HostModeProtocol>>,
}

impl CheckpointConfigs {
    fn new() -> Self {
        Self {
            llm: Arc::new(CheckpointDriver::Llm),
            tools: Arc::new(CheckpointDriver::Tools),
            exec: Arc::new(CheckpointDriver::Exec),
        }
    }

    fn llm_config(&self) -> TurnMachineConfig {
        checkpoint_config(Arc::clone(&self.llm))
    }

    fn tools_config(&self) -> TurnMachineConfig {
        checkpoint_config(Arc::clone(&self.tools))
    }

    fn exec_config(&self) -> TurnMachineConfig {
        checkpoint_config(Arc::clone(&self.exec))
    }
}

#[derive(Clone, Copy)]
enum CheckpointDriver {
    Llm,
    Tools,
    Exec,
}

impl ProtocolDriverHandle<lash_core::HostModeProtocol> for CheckpointDriver {
    fn prepare_mode_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        match self {
            Self::Llm => vec![DriverAction::StartLlm {
                request: ctx.project_llm_request(false),
                driver_state: None,
            }],
            Self::Tools => vec![DriverAction::StartTools {
                calls: checkpoint_tool_calls(ctx.mode_iteration()),
            }],
            Self::Exec => vec![DriverAction::StartExec {
                code: checkpoint_exec_code(ctx.mode_iteration()),
                driver_state: serde_json::json!({
                    "phase": "exec_code",
                    "ip": ctx.mode_iteration(),
                    "stack": (0..64).map(|index| serde_json::json!({
                        "slot": index,
                        "value": format!("checkpoint-stack-value-{index}")
                    })).collect::<Vec<_>>(),
                }),
            }],
        }
    }

    fn handle_llm_success(
        &self,
        _ctx: DriverContextView<'_>,
        _waiting: WaitingLlmState<lash_core::HostModeProtocol>,
        _llm_response: LlmResponse,
        _text_streamed: bool,
    ) -> Vec<DriverAction> {
        vec![DriverAction::Finish(TurnOutcome::Finished(
            TurnFinish::AssistantMessage {
                text: "runtime perf benchmark ok".to_string(),
            },
        ))]
    }

    fn handle_tool_results(
        &self,
        _ctx: DriverContextView<'_>,
        _completed: Vec<CompletedToolCall>,
    ) -> Vec<DriverAction> {
        vec![DriverAction::Finish(TurnOutcome::Finished(
            TurnFinish::AssistantMessage {
                text: "runtime perf benchmark ok".to_string(),
            },
        ))]
    }

    fn handle_exec_result(
        &self,
        _ctx: DriverContextView<'_>,
        _waiting: WaitingExecState<lash_core::HostModeProtocol>,
        _result: Result<ExecResponse, String>,
    ) -> Vec<DriverAction> {
        vec![DriverAction::Finish(TurnOutcome::Finished(
            TurnFinish::SubmittedValue {
                value: serde_json::json!("runtime perf benchmark ok"),
            },
        ))]
    }
}

fn checkpoint_config(
    protocol_driver: Arc<dyn ProtocolDriverHandle<lash_core::HostModeProtocol>>,
) -> TurnMachineConfig {
    TurnMachineConfig {
        protocol_driver,
        projector: Arc::new(ChatContextProjector),
        sync_execution_surface: false,
        model: "mock-model".to_string(),
        max_turns: Some(8),
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        run_session_id: Some("runtime-perf-turn-checkpoint".to_string()),
        autonomous: false,
        tool_specs: Arc::new(Vec::new()),
        system_prompt: Arc::from(
            "Synthetic durability profiler prompt. Preserve pending effects across checkpoint restore.",
        ),
        session_id: "runtime-perf-turn-checkpoint".to_string(),
        emit_llm_trace: false,
        termination: ModeTurnOptions::default(),
        turn_limit_final_message: Arc::new(runtime_perf_turn_limit_final_message),
    }
}

fn runtime_perf_turn_limit_final_message(message_id: String, max_turns: usize) -> Message {
    Message {
        id: message_id.clone(),
        role: MessageRole::System,
        parts: shared_parts(vec![Part {
            id: format!("{message_id}.p0"),
            kind: PartKind::Error,
            content: format!("Turn limit reached ({max_turns}) before runtime perf completion."),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}

fn checkpoint_messages() -> Vec<Message> {
    (0usize..36)
        .map(|index| {
            let role = if index.is_multiple_of(2) {
                MessageRole::User
            } else {
                MessageRole::Assistant
            };
            checkpoint_message(
                format!("checkpoint-msg-{index}"),
                role,
                format!(
                    "Historical durability profiler message {index}. This payload is intentionally long enough to make TurnCheckpoint serialization include realistic prompt and transcript bytes. The current topic is standard and RLM turn-effect replay across LLM, tool, checkpoint, sleep, and ExecCode boundaries."
                ),
            )
        })
        .collect()
}

fn checkpoint_message(id: String, role: MessageRole, content: String) -> Message {
    Message {
        id: id.clone(),
        role,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}

fn measure_checkpoint_phase(
    name: &'static str,
    f: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<(String, RuntimePerfPhaseRunResult)> {
    let before_alloc = allocator_stats();
    let before_memory = process_memory_sample();
    let started = Instant::now();
    f()?;
    let after_alloc = allocator_stats();
    let after_memory = process_memory_sample();
    Ok((
        name.to_string(),
        RuntimePerfPhaseRunResult {
            duration_ms: elapsed_ms(started),
            allocations: alloc_delta(before_alloc, after_alloc),
            rss_growth_kb: diff_opt_i64(before_memory.rss_kb, after_memory.rss_kb),
        },
    ))
}

fn checkpoint_pending_llm(
    configs: &CheckpointConfigs,
    seed_messages: &[Message],
    turn_index: usize,
) -> anyhow::Result<()> {
    let config = configs.llm_config();
    let mut machine = checkpoint_machine(config, seed_messages, turn_index);
    let effect = next_checkpoint_effect(&mut machine)
        .ok_or_else(|| anyhow::anyhow!("checkpoint llm scenario produced no effect"))?;
    let Effect::LlmCall { id, .. } = effect else {
        anyhow::bail!("checkpoint llm scenario expected LlmCall effect");
    };
    let checkpoint = machine.checkpoint();
    let bytes = serde_json::to_vec(&checkpoint)?;
    let checkpoint = serde_json::from_slice(&bytes)?;
    let mut restored = TurnMachine::restore_from_checkpoint(configs.llm_config(), checkpoint);
    assert_restored_llm(&mut restored, id)?;
    restored.handle_response(Response::LlmComplete {
        id,
        result: Ok(LlmResponse {
            full_text: "runtime perf benchmark ok".to_string(),
            ..LlmResponse::default()
        }),
        text_streamed: false,
    });
    drain_checkpoint_machine(&mut restored);
    Ok(())
}

fn checkpoint_pending_parallel_tools(
    configs: &CheckpointConfigs,
    seed_messages: &[Message],
    turn_index: usize,
) -> anyhow::Result<()> {
    let config = configs.tools_config();
    let mut machine = checkpoint_machine(config, seed_messages, turn_index);
    let effect = next_checkpoint_effect(&mut machine)
        .ok_or_else(|| anyhow::anyhow!("checkpoint tools scenario produced no effect"))?;
    let Effect::ToolCalls { id, calls } = effect else {
        anyhow::bail!("checkpoint tools scenario expected ToolCalls effect");
    };
    let checkpoint = machine.checkpoint();
    let bytes = serde_json::to_vec(&checkpoint)?;
    let checkpoint = serde_json::from_slice(&bytes)?;
    let mut restored = TurnMachine::restore_from_checkpoint(configs.tools_config(), checkpoint);
    assert_restored_tool_batch(&mut restored, id, calls.len())?;
    restored.handle_response(Response::ToolResults {
        id,
        results: calls
            .into_iter()
            .enumerate()
            .map(|(index, call)| completed_checkpoint_tool(index, call))
            .collect(),
    });
    drain_checkpoint_machine(&mut restored);
    Ok(())
}

fn checkpoint_pending_exec(
    configs: &CheckpointConfigs,
    seed_messages: &[Message],
    turn_index: usize,
) -> anyhow::Result<()> {
    let config = configs.exec_config();
    let mut machine = checkpoint_machine(config, seed_messages, turn_index);
    let effect = next_checkpoint_effect(&mut machine)
        .ok_or_else(|| anyhow::anyhow!("checkpoint exec scenario produced no effect"))?;
    let Effect::ExecCode { id, code } = effect else {
        anyhow::bail!("checkpoint exec scenario expected ExecCode effect");
    };
    let checkpoint = machine.checkpoint();
    let bytes = serde_json::to_vec(&checkpoint)?;
    let checkpoint = serde_json::from_slice(&bytes)?;
    let mut restored = TurnMachine::restore_from_checkpoint(configs.exec_config(), checkpoint);
    assert_restored_exec(&mut restored, id, &code)?;
    restored.handle_response(Response::ExecResult {
        id,
        result: Ok(ExecResponse {
            output: "runtime perf benchmark ok".to_string(),
            observations: vec![
                "checkpoint observation: resumed after ExecCode effect boundary".to_string(),
            ],
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: Some(serde_json::json!("runtime perf benchmark ok")),
        }),
    });
    drain_checkpoint_machine(&mut restored);
    Ok(())
}

fn checkpoint_machine(
    config: TurnMachineConfig,
    seed_messages: &[Message],
    turn_index: usize,
) -> TurnMachine {
    let mut messages = seed_messages.to_vec();
    messages.push(checkpoint_message(
        format!("checkpoint-live-turn-{turn_index}"),
        MessageRole::User,
        format!(
            "Durability checkpoint profiler live turn {}",
            turn_index + 1
        ),
    ));
    TurnMachine::new(config, messages, Arc::new(Vec::new()), turn_index)
}

fn checkpoint_tool_calls(mode_iteration: usize) -> Vec<PendingToolCall> {
    (0..24)
        .map(|index| PendingToolCall {
            call_id: format!("checkpoint-call-{mode_iteration}-{index}"),
            tool_name: format!("checkpoint_parallel_tool_{}", index % 6),
            args: serde_json::json!({
                "index": index,
                "mode_iteration": mode_iteration,
                "payload": format!("synthetic parallel durability payload {index}")
            }),
            replay: None,
        })
        .collect()
}

fn completed_checkpoint_tool(index: usize, call: PendingToolCall) -> CompletedToolCall {
    let output = match index % 4 {
        0 => ToolCallOutput::success(serde_json::json!({
            "ok": true,
            "index": index,
            "payload": call.args,
        })),
        1 => ToolCallOutput::failure(ToolFailure::tool(
            ToolFailureClass::Execution,
            "checkpoint_tool_failed",
            format!("synthetic failure for {}", call.call_id),
        )),
        2 => ToolCallOutput::cancelled(ToolCancellation::runtime(format!(
            "synthetic cancellation for {}",
            call.call_id
        ))),
        _ => ToolCallOutput::success(serde_json::json!({
            "ok": true,
            "index": index,
            "large": "x".repeat(128),
        })),
    };
    CompletedToolCall {
        call_id: call.call_id.clone(),
        tool_name: call.tool_name.clone(),
        args: call.args,
        model_return: ModelToolReturn::from_output(
            call.call_id.clone(),
            call.tool_name.clone(),
            &output,
        ),
        output,
        duration_ms: 1,
        replay: call.replay,
    }
}

fn checkpoint_exec_code(mode_iteration: usize) -> String {
    format!(
        r#"process benchmark_echo_process(tool: TOOL, value: str, ordinal: int) {{
  result = await tool.benchmark_echo({{ value: value, ordinal: ordinal }})?
  finish result
}}

print("checkpoint turn {mode_iteration}")
first = start benchmark_echo_process(tool: TOOL.default, value: "runtime perf benchmark ok", ordinal: 1)
second = start benchmark_echo_process(tool: TOOL.default, value: "runtime perf benchmark ok", ordinal: 2)
third = start benchmark_echo_process(tool: TOOL.default, value: "runtime perf benchmark ok", ordinal: 3)
fanout = await {{
  a: first,
  b: second,
  c: third
}}
submit fanout.a?.value"#
    )
}

fn assert_restored_llm(
    machine: &mut TurnMachine,
    expected_id: lash_core::EffectId,
) -> anyhow::Result<()> {
    match next_checkpoint_effect(machine) {
        Some(Effect::LlmCall { id, .. }) if id == expected_id => Ok(()),
        Some(_) => anyhow::bail!("restored checkpoint did not replay LlmCall"),
        None => anyhow::bail!("restored checkpoint had no LlmCall"),
    }
}

fn assert_restored_tool_batch(
    machine: &mut TurnMachine,
    expected_id: lash_core::EffectId,
    expected_calls: usize,
) -> anyhow::Result<()> {
    match next_checkpoint_effect(machine) {
        Some(Effect::ToolCalls { id, calls })
            if id == expected_id && calls.len() == expected_calls =>
        {
            Ok(())
        }
        Some(_) => anyhow::bail!("restored checkpoint did not replay matching ToolCalls"),
        None => anyhow::bail!("restored checkpoint had no ToolCalls"),
    }
}

fn assert_restored_exec(
    machine: &mut TurnMachine,
    expected_id: lash_core::EffectId,
    expected_code: &str,
) -> anyhow::Result<()> {
    match next_checkpoint_effect(machine) {
        Some(Effect::ExecCode { id, code }) if id == expected_id && code == expected_code => Ok(()),
        Some(_) => anyhow::bail!("restored checkpoint did not replay matching ExecCode"),
        None => anyhow::bail!("restored checkpoint had no ExecCode"),
    }
}

fn drain_checkpoint_machine(machine: &mut TurnMachine) {
    while machine.poll_effect().is_some() {}
}

fn next_checkpoint_effect(machine: &mut TurnMachine) -> Option<Effect> {
    loop {
        match machine.poll_effect()? {
            Effect::Emit(_)
            | Effect::Log { .. }
            | Effect::Progress { .. }
            | Effect::Done { .. } => continue,
            effect => return Some(effect),
        }
    }
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
        let cancel = CancellationToken::new();
        let turn = runtime_perf_timed(
            scenario,
            turn_index,
            "run_turn",
            Some(cancel.clone()),
            async {
                session
                    .turn(lash_core::TurnInput::text(benchmark_prompt(
                        scenario, turn_index,
                    )))
                    .cancel(cancel)
                    .collect_session_events_with(&lash::advanced::NoopEventSink)
                    .await
                    .map_err(anyhow::Error::from)
            },
        )
        .await
        .with_context(|| {
            format!(
                "run embed runtime perf scenario {} turn {}",
                scenario.name(),
                turn_index + 1
            )
        })?;
        validate_runtime_perf_turn(scenario, turn_index, &turn)?;
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
            turn_usage: turn.usage,
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
