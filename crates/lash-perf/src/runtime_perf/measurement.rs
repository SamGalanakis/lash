use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use lash::usage::SessionUsageReport;
use lash_core::llm::types::{LlmResponse, LlmUsage};
use lash_core::runtime::{RuntimeTurnPhase, RuntimeTurnPhaseProbe};
use lash_core::sansio::{
    ChatContextProjector, CompletedToolCall, PendingToolCall, ProtocolDriverHandle,
    WaitingExecState, WaitingLlmState,
};
use lash_core::{
    DriverAction, DriverContextView, Effect, ExecResponse, InputItem, LiveReplayResult,
    LiveReplayStore, LiveReplaySubscribeResult, Message, MessageRole, ModelToolReturn, Part,
    PartKind, ProtocolTurnOptions, PruneState, Response, SessionObservationEventPayload,
    SessionRevision, TokenUsage, ToolCallOutput, ToolCancellation, ToolFailure, ToolFailureClass,
    TurnFinish, TurnInput, TurnMachine, TurnMachineConfig, TurnOutcome, shared_parts,
};
use lash_protocol_rlm::RlmTurnInputExt;
use serde::Serialize;
use stats_alloc::Stats;
use tokio_util::sync::CancellationToken;

use crate::perf_support::memory::{ProcessMemorySample, diff_opt_i64, process_memory_sample};
use crate::perf_support::metrics::BasicMetricSummary as RuntimePerfMetricSummary;
use crate::perf_support::tempdir::make_temp_bench_dir;
use crate::perf_support::time::{elapsed_ms, round3};

use super::harness::{
    RuntimePerfTraceConfig, benchmark_prompt, build_embed_core, build_runtime,
    build_runtime_with_sqlite_store, build_runtime_with_store, prepare_turn,
    rlm_perf_projected_bindings, seed_runtime_state, validate_runtime_perf_turn,
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
    pub(crate) extra_counters: BTreeMap<String, u64>,
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
    pub(crate) sample_extra_counters: BTreeMap<String, u64>,
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
    named_started: HashMap<String, Vec<PhaseStart>>,
    completed: BTreeMap<String, RuntimePerfPhaseRunResult>,
}

#[derive(Default)]
struct RuntimePerfPhaseProbe {
    state: Mutex<RuntimePerfPhaseProbeState>,
}

struct ScopedPerfEffectController;

#[async_trait::async_trait]
impl lash::runtime::RuntimeEffectController for ScopedPerfEffectController {
    async fn execute_effect(
        &self,
        envelope: lash::runtime::RuntimeEffectEnvelope,
        local_executor: lash::runtime::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<lash::runtime::RuntimeEffectOutcome, lash::runtime::RuntimeEffectControllerError>
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
        record_completed_phase(&mut state.completed, phase_name(phase).to_string(), start);
    }

    fn begin_named(&self, phase: &str) {
        let mut state = self.state.lock().expect("phase probe lock");
        state
            .named_started
            .entry(phase.to_string())
            .or_default()
            .push(PhaseStart {
                started_at: Instant::now(),
                alloc_before: allocator_stats(),
                memory_before: process_memory_sample(),
            });
    }

    fn end_named(&self, phase: &str) {
        let mut state = self.state.lock().expect("phase probe lock");
        let Some(starts) = state.named_started.get_mut(phase) else {
            return;
        };
        let Some(start) = starts.pop() else {
            return;
        };
        if starts.is_empty() {
            state.named_started.remove(phase);
        }
        record_completed_phase(&mut state.completed, phase.to_string(), start);
    }
}

fn record_completed_phase(
    completed: &mut BTreeMap<String, RuntimePerfPhaseRunResult>,
    name: String,
    start: PhaseStart,
) {
    let alloc_after = allocator_stats();
    let memory_after = process_memory_sample();
    let metrics = RuntimePerfPhaseRunResult {
        duration_ms: elapsed_ms(start.started_at),
        allocations: alloc_delta(start.alloc_before, alloc_after),
        rss_growth_kb: diff_opt_i64(start.memory_before.rss_kb, memory_after.rss_kb),
    };
    let entry = completed
        .entry(name)
        .or_insert_with(|| RuntimePerfPhaseRunResult {
            duration_ms: 0.0,
            allocations: zero_allocation_delta(),
            rss_growth_kb: Some(0),
        });
    entry.duration_ms = round3(entry.duration_ms + metrics.duration_ms);
    entry.allocations = sum_allocation_deltas([&entry.allocations, &metrics.allocations]);
    entry.rss_growth_kb = sum_optional_i64(entry.rss_growth_kb, metrics.rss_growth_kb);
}
pub(crate) async fn run_once(
    scenario: RuntimePerfScenario,
    chat_turns: usize,
) -> anyhow::Result<RuntimePerfRunResult> {
    if matches!(scenario, RuntimePerfScenario::TurnCheckpoint) {
        return run_once_turn_checkpoint(chat_turns).await;
    }

    if matches!(scenario, RuntimePerfScenario::LiveReplayPressure) {
        return run_once_live_replay_pressure(chat_turns).await;
    }

    if matches!(
        scenario,
        RuntimePerfScenario::TraceJsonlStandard | RuntimePerfScenario::TraceJsonlExtended
    ) {
        return run_once_trace_jsonl(scenario, chat_turns).await;
    }

    if matches!(scenario, RuntimePerfScenario::OpenAiResponsesSseParse) {
        return run_once_openai_responses_sse_parse(chat_turns).await;
    }

    if matches!(scenario, RuntimePerfScenario::DirectLlmClient) {
        return run_once_direct_llm_client(chat_turns).await;
    }

    if matches!(scenario, RuntimePerfScenario::ProcessListStress) {
        return run_once_process_list_stress(chat_turns).await;
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
    let sqlite_root = if matches!(scenario, RuntimePerfScenario::SqliteStoreReopen) {
        Some(make_temp_bench_dir("lash-runtime-perf-sqlite-store")?)
    } else {
        None
    };
    let mut runtime = if let Some(root) = sqlite_root.as_ref() {
        build_runtime_with_sqlite_store(scenario, root.clone()).await?
    } else {
        build_runtime(scenario).await?
    };
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
        if matches!(scenario, RuntimePerfScenario::SqliteStoreReopen) && turn_index > 0 {
            let reopen_before_alloc = allocator_stats();
            let reopen_before_memory = process_memory_sample();
            let reopen_started = Instant::now();
            runtime.reopen_session(scenario).await?;
            extra_phase_profile.insert(
                "sqlite_store_reopen.runtime_reopen".to_string(),
                RuntimePerfPhaseRunResult {
                    duration_ms: elapsed_ms(reopen_started),
                    allocations: alloc_delta(reopen_before_alloc, allocator_stats()),
                    rss_growth_kb: diff_opt_i64(
                        reopen_before_memory.rss_kb,
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
            protocol_turn_options: None,
            trace_turn_id: None,
            protocol_extension: None,
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
            let scoped_effect_controller = lash::runtime::ScopedEffectController::borrowed(
                &effect_controller,
                lash::runtime::EffectScope::turn(
                    format!("runtime-perf-{}", scenario.name()),
                    &turn_id,
                ),
            )
            .map_err(anyhow::Error::from)?;
            runtime_perf_timed(
                scenario,
                turn_index,
                "run_turn",
                Some(cancel.clone()),
                runtime.run_turn_with_effect_scope(turn_input, cancel, scoped_effect_controller),
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
    if let Some(root) = sqlite_root {
        runtime.close().await?;
        let _ = std::fs::remove_dir_all(root);
    }

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
        extra_counters: BTreeMap::new(),
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

const LIVE_REPLAY_EVENTS_PER_TURN: usize = 96;
const LIVE_REPLAY_MAIN_CAPACITY: usize = 256;
const LIVE_REPLAY_TRIM_CAPACITY: usize = 8;

async fn run_once_live_replay_pressure(chat_turns: usize) -> anyhow::Result<RuntimePerfRunResult> {
    let scenario = RuntimePerfScenario::LiveReplayPressure;
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let store = lash_core::InMemoryLiveReplayStore::with_bounds(
        LIVE_REPLAY_MAIN_CAPACITY,
        Duration::from_secs(120),
    );
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    let mut appended_events = 0usize;
    let mut replayed_events = 0usize;
    let mut subscribed_buffered_events = 0usize;
    let mut subscribed_live_events = 0usize;
    let mut trim_gaps = 0usize;
    let mut unavailable_gaps = 0usize;

    for turn_index in 0..chat_turns {
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let mut phase_profile = BTreeMap::new();
        let session_id = format!("runtime-perf-live-replay-{turn_index}");
        let revision = SessionRevision::new(turn_index as u64 + 1);
        let start_cursor = store.current_cursor(&session_id, revision);

        let (first_cursor, append_phase) =
            measure_runtime_perf_phase("live_replay.append", || {
                let mut first_cursor = None;
                for event_index in 0..LIVE_REPLAY_EVENTS_PER_TURN {
                    let event = store.append(
                        &session_id,
                        revision,
                        live_replay_text_payload(format!("turn-{turn_index}-event-{event_index}")),
                    )?;
                    if first_cursor.is_none() {
                        first_cursor = Some(event.cursor.clone());
                    }
                }
                first_cursor.ok_or_else(|| anyhow::anyhow!("live replay append produced no cursor"))
            })?;
        appended_events += LIVE_REPLAY_EVENTS_PER_TURN;
        phase_profile.insert(append_phase.0, append_phase.1);

        let (current_cursor, current_phase) =
            measure_runtime_perf_phase("live_replay.current_cursor_parse", || {
                let cursor = store.current_cursor(&session_id, revision);
                match store.replay_after_cursor(&cursor)? {
                    LiveReplayResult::Replayed(events) if events.is_empty() => Ok(cursor),
                    LiveReplayResult::Replayed(events) => anyhow::bail!(
                        "current cursor replay unexpectedly returned {} events",
                        events.len()
                    ),
                    LiveReplayResult::Gap(reason) => {
                        anyhow::bail!("current cursor replay returned gap {reason:?}")
                    }
                }
            })?;
        phase_profile.insert(current_phase.0, current_phase.1);

        let (replay_count, replay_phase) =
            measure_runtime_perf_phase("live_replay.replay_after_cursor", || {
                match store.replay_after_cursor(&start_cursor)? {
                    LiveReplayResult::Replayed(events) => Ok(events.len()),
                    LiveReplayResult::Gap(reason) => {
                        anyhow::bail!("start cursor replay returned gap {reason:?}")
                    }
                }
            })?;
        if replay_count != LIVE_REPLAY_EVENTS_PER_TURN {
            anyhow::bail!(
                "live replay expected {} replayed events, got {replay_count}",
                LIVE_REPLAY_EVENTS_PER_TURN
            );
        }
        replayed_events += replay_count;
        phase_profile.insert(replay_phase.0, replay_phase.1);

        let ((buffered_count, live_count), subscribe_phase) =
            measure_runtime_perf_async_phase("live_replay.subscribe_buffered", async {
                let mut subscription = match store.subscribe_after_cursor(&first_cursor)? {
                    LiveReplaySubscribeResult::Subscribed(subscription) => subscription,
                    LiveReplaySubscribeResult::Gap(reason) => {
                        anyhow::bail!("subscribe after first cursor returned gap {reason:?}")
                    }
                };
                let mut buffered_count = 0usize;
                for _ in 1..LIVE_REPLAY_EVENTS_PER_TURN {
                    tokio::time::timeout(Duration::from_secs(1), subscription.next_event())
                        .await
                        .context("timed out reading buffered live replay event")??;
                    buffered_count += 1;
                }
                store.append(
                    &session_id,
                    revision,
                    live_replay_text_payload(format!("turn-{turn_index}-live-event")),
                )?;
                tokio::time::timeout(Duration::from_secs(1), subscription.next_event())
                    .await
                    .context("timed out reading live replay event")??;
                Ok((buffered_count, 1usize))
            })
            .await?;
        subscribed_buffered_events += buffered_count;
        subscribed_live_events += live_count;
        phase_profile.insert(subscribe_phase.0, subscribe_phase.1);

        let (trim_gap_count, trim_phase) =
            measure_runtime_perf_phase("live_replay.trim_by_capacity", || {
                let trim_store = lash_core::InMemoryLiveReplayStore::with_bounds(
                    LIVE_REPLAY_TRIM_CAPACITY,
                    Duration::from_secs(120),
                );
                let trim_session_id = format!("runtime-perf-live-replay-trim-{turn_index}");
                let trim_start = trim_store.current_cursor(&trim_session_id, revision);
                for event_index in 0..(LIVE_REPLAY_TRIM_CAPACITY * 3) {
                    trim_store.append(
                        &trim_session_id,
                        revision,
                        live_replay_text_payload(format!("trim-{turn_index}-{event_index}")),
                    )?;
                }
                trim_store.trim_session(&trim_session_id)?;
                match trim_store.replay_after_cursor(&trim_start)? {
                    LiveReplayResult::Gap(lash_core::LiveReplayGapReason::Trimmed) => Ok(1usize),
                    LiveReplayResult::Gap(reason) => {
                        anyhow::bail!("capacity trim returned wrong gap {reason:?}")
                    }
                    LiveReplayResult::Replayed(events) => anyhow::bail!(
                        "capacity trim expected gap, got {} replayed events",
                        events.len()
                    ),
                }
            })?;
        trim_gaps += trim_gap_count;
        phase_profile.insert(trim_phase.0, trim_phase.1);

        let (unavailable_gap_count, gap_phase) =
            measure_runtime_perf_phase("live_replay.gap_handling", || {
                let ahead_cursor: lash_core::SessionCursor =
                    serde_json::from_value(serde_json::json!(format!(
                        "lashsc1:{}:999999:{}",
                        revision.as_u64(),
                        session_id
                    )))?;
                let mut gaps = 0usize;
                match store.replay_after_cursor(&ahead_cursor)? {
                    LiveReplayResult::Gap(lash_core::LiveReplayGapReason::Unavailable) => gaps += 1,
                    LiveReplayResult::Gap(reason) => {
                        anyhow::bail!("ahead replay returned wrong gap {reason:?}")
                    }
                    LiveReplayResult::Replayed(events) => anyhow::bail!(
                        "ahead replay expected gap, got {} replayed events",
                        events.len()
                    ),
                }
                match store.subscribe_after_cursor(&ahead_cursor)? {
                    LiveReplaySubscribeResult::Gap(lash_core::LiveReplayGapReason::Unavailable) => {
                        gaps += 1
                    }
                    LiveReplaySubscribeResult::Gap(reason) => {
                        anyhow::bail!("ahead subscribe returned wrong gap {reason:?}")
                    }
                    LiveReplaySubscribeResult::Subscribed(_) => {
                        anyhow::bail!("ahead subscribe expected gap")
                    }
                }
                Ok(gaps)
            })?;
        unavailable_gaps += unavailable_gap_count;
        phase_profile.insert(gap_phase.0, gap_phase.1);

        match store.replay_after_cursor(&current_cursor)? {
            LiveReplayResult::Replayed(events) if events.len() == 1 => {}
            LiveReplayResult::Replayed(events) => anyhow::bail!(
                "current cursor should see only the live event after subscribe, got {}",
                events.len()
            ),
            LiveReplayResult::Gap(reason) => {
                anyhow::bail!("current cursor after live append returned gap {reason:?}")
            }
        }

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
    let _export_shape = serde_json::json!({
        "appended_events": appended_events,
        "replayed_events": replayed_events,
        "subscribed_buffered_events": subscribed_buffered_events,
        "subscribed_live_events": subscribed_live_events,
        "trim_gaps": trim_gaps,
        "unavailable_gaps": unavailable_gaps,
    })
    .to_string();
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
        session_nodes: appended_events,
        active_path_messages: replayed_events,
        extra_counters: BTreeMap::from([
            ("appended_events".to_string(), appended_events as u64),
            ("replayed_events".to_string(), replayed_events as u64),
            (
                "subscribed_buffered_events".to_string(),
                subscribed_buffered_events as u64,
            ),
            (
                "subscribed_live_events".to_string(),
                subscribed_live_events as u64,
            ),
            ("trim_gaps".to_string(), trim_gaps as u64),
            ("unavailable_gaps".to_string(), unavailable_gaps as u64),
        ]),
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

async fn run_once_trace_jsonl(
    scenario: RuntimePerfScenario,
    chat_turns: usize,
) -> anyhow::Result<RuntimePerfRunResult> {
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let trace_root = make_temp_bench_dir("lash-runtime-perf-trace-jsonl")?;
    let trace_path = trace_root.join("runtime-trace.jsonl");
    let lashlang_trace_path = matches!(scenario, RuntimePerfScenario::TraceJsonlExtended)
        .then(|| trace_root.join("lashlang-execution.jsonl"));
    let trace_config = RuntimePerfTraceConfig {
        trace_jsonl_path: Some(trace_path.clone()),
        lashlang_execution_jsonl_path: lashlang_trace_path.clone(),
        trace_level: if matches!(scenario, RuntimePerfScenario::TraceJsonlExtended) {
            lash::tracing::TraceLevel::Extended
        } else {
            lash::tracing::TraceLevel::Standard
        },
    };
    let mut runtime = build_runtime_with_store(scenario, None, Some(trace_config)).await?;
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
        runtime.set_turn_phase_probe(phase_probe.clone()).await;

        let before_turn_usage = runtime.usage_report();
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let turn_input = TurnInput {
            items: vec![InputItem::Text {
                text: benchmark_prompt(scenario, turn_index),
            }],
            image_blobs: Default::default(),
            protocol_turn_options: None,
            trace_turn_id: None,
            protocol_extension: None,
            turn_context: lash_core::TurnContext::default(),
        };
        let cancel = CancellationToken::new();
        let turn = runtime_perf_timed(
            scenario,
            turn_index,
            "run_turn",
            Some(cancel.clone()),
            runtime.run_turn(turn_input, cancel),
        )
        .await
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
    let (trace_counters, inspect_phase) =
        measure_runtime_perf_phase("trace_jsonl.inspect_files", || {
            inspect_trace_jsonl_files(&trace_path, lashlang_trace_path.as_deref())
        })?;
    let total_alloc = alloc_delta(total_before_alloc, allocator_stats());
    let last_turn_memory = turns.last().map(|turn| &turn.memory);
    let mut phase_profile = sum_phase_profiles(turns.iter().map(|turn| &turn.phase_profile));
    phase_profile.insert(inspect_phase.0, inspect_phase.1);
    runtime.close().await?;
    let _ = std::fs::remove_dir_all(trace_root);

    if trace_counters
        .get("trace_records")
        .copied()
        .unwrap_or_default()
        == 0
    {
        anyhow::bail!("trace_jsonl scenario produced no runtime trace records");
    }
    if matches!(scenario, RuntimePerfScenario::TraceJsonlExtended)
        && trace_counters
            .get("lashlang_execution_trace_records")
            .copied()
            .unwrap_or_default()
            == 0
    {
        anyhow::bail!("extended trace_jsonl scenario produced no Lashlang execution records");
    }

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
        extra_counters: trace_counters,
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
        phase_profile,
        turns,
        cumulative_usage,
    })
}

fn live_replay_text_payload(text: impl Into<String>) -> SessionObservationEventPayload {
    SessionObservationEventPayload::TurnActivity(lash_core::TurnActivity::independent(
        lash_core::TurnEvent::AssistantProseDelta { text: text.into() },
    ))
}

fn inspect_trace_jsonl_files(
    trace_path: &std::path::Path,
    lashlang_trace_path: Option<&std::path::Path>,
) -> anyhow::Result<BTreeMap<String, u64>> {
    let mut counters = BTreeMap::new();
    let (trace_bytes, trace_records) = jsonl_file_stats(trace_path)?;
    counters.insert("trace_bytes".to_string(), trace_bytes);
    counters.insert("trace_records".to_string(), trace_records);
    if let Some(path) = lashlang_trace_path {
        let (lashlang_bytes, lashlang_records) = jsonl_file_stats(path)?;
        counters.insert("lashlang_execution_trace_bytes".to_string(), lashlang_bytes);
        counters.insert(
            "lashlang_execution_trace_records".to_string(),
            lashlang_records,
        );
    }
    Ok(counters)
}

fn jsonl_file_stats(path: &std::path::Path) -> anyhow::Result<(u64, u64)> {
    let bytes = std::fs::metadata(path)
        .with_context(|| format!("stat trace file {}", path.display()))?
        .len();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read trace file {}", path.display()))?;
    let records = text.lines().filter(|line| !line.trim().is_empty()).count() as u64;
    Ok((bytes, records))
}

async fn run_once_openai_responses_sse_parse(
    chat_turns: usize,
) -> anyhow::Result<RuntimePerfRunResult> {
    let scenario = RuntimePerfScenario::OpenAiResponsesSseParse;
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let payloads = (0..chat_turns)
        .map(openai_responses_sse_payload)
        .collect::<Vec<_>>();
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    let payload_bytes = payloads.iter().map(String::len).sum::<usize>();
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    let mut parsed_parts = 0usize;
    for (turn_index, payload) in payloads.iter().enumerate() {
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let mut phase_profile = BTreeMap::new();

        let (state, parse_phase) =
            measure_runtime_perf_phase("openai_responses_sse_parse.parse_payload", || {
                let mut state =
                    lash_provider_openai::responses_shared::ResponsesStreamState::default();
                lash_provider_openai::responses_shared::parse_sse_payload(
                    "OpenAI", payload, &mut state,
                )?;
                Ok(state)
            })?;
        phase_profile.insert(parse_phase.0, parse_phase.1);

        if !state.full_text.contains("runtime perf benchmark ok") {
            anyhow::bail!(
                "runtime perf scenario {} turn {} failed to parse benchmark marker",
                scenario.name(),
                turn_index + 1
            );
        }

        let (parts_len, parts_phase) =
            measure_runtime_perf_phase("openai_responses_sse_parse.project_parts", || {
                Ok(state.response_parts().len())
            })?;
        parsed_parts += parts_len;
        phase_profile.insert(parts_phase.0, parts_phase.1);

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
            turn_usage: token_usage_from_llm_usage(&state.usage),
            usage_delta: SessionUsageReport::default(),
            cumulative_usage: SessionUsageReport::default(),
        });
    }

    let export_before_alloc = allocator_stats();
    let export_started = Instant::now();
    let _export_shape = serde_json::json!({
        "payload_bytes": payload_bytes,
        "parsed_parts": parsed_parts,
    })
    .to_string();
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
        session_nodes: parsed_parts,
        active_path_messages: chat_turns,
        extra_counters: BTreeMap::from([
            ("payload_bytes".to_string(), payload_bytes as u64),
            ("parsed_parts".to_string(), parsed_parts as u64),
        ]),
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

async fn run_once_direct_llm_client(chat_turns: usize) -> anyhow::Result<RuntimePerfRunResult> {
    let scenario = RuntimePerfScenario::DirectLlmClient;
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let provider = super::providers::benchmark_provider(scenario).into_handle();
    let mut client = lash::direct::DirectLlmClient::new(provider);
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    let mut response_bytes = 0usize;
    for turn_index in 0..chat_turns {
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let response = runtime_perf_timed(
            scenario,
            turn_index,
            "direct_llm_client.complete",
            None,
            async {
                client
                    .complete(direct_llm_client_request(turn_index))
                    .await
                    .map_err(anyhow::Error::from)
            },
        )
        .await
        .with_context(|| {
            format!(
                "run runtime perf scenario {} turn {}",
                scenario.name(),
                turn_index + 1
            )
        })?;
        validate_direct_llm_response(turn_index, &response)?;
        response_bytes += response.full_text.len();
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

        let mut phase_profile = BTreeMap::new();
        phase_profile.insert(
            "direct_llm_client.complete".to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: run_turn_ms,
                allocations: run_turn_alloc.clone(),
                rss_growth_kb: diff_opt_i64(turn_before_memory.rss_kb, after_turn_memory.rss_kb),
            },
        );

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
            turn_usage: token_usage_from_llm_usage(&response.usage),
            usage_delta: SessionUsageReport::default(),
            cumulative_usage: SessionUsageReport::default(),
        });
    }

    let export_before_alloc = allocator_stats();
    let export_started = Instant::now();
    let _export_shape = serde_json::json!({
        "response_bytes": response_bytes,
        "responses": turns.len(),
    })
    .to_string();
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
        session_nodes: 0,
        active_path_messages: chat_turns,
        extra_counters: BTreeMap::from([
            ("response_bytes".to_string(), response_bytes as u64),
            ("responses".to_string(), turns.len() as u64),
        ]),
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

const PROCESS_LIST_STRESS_BATCH: usize = 128;

async fn run_once_process_list_stress(chat_turns: usize) -> anyhow::Result<RuntimePerfRunResult> {
    let scenario = RuntimePerfScenario::ProcessListStress;
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let registry: Arc<dyn lash_core::ProcessRegistry> =
        Arc::new(lash_core::TestLocalProcessRegistry::default());
    let session_scope = lash_core::SessionScope::new("runtime-perf-process-list");
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    let process_count = chat_turns.max(1) * PROCESS_LIST_STRESS_BATCH;
    for index in 0..process_count {
        let process_id = format!("process-list-stress-{index:05}");
        registry
            .register_process(process_list_stress_registration(
                process_id.clone(),
                session_scope.clone(),
                index,
            ))
            .await?;
        registry
            .grant_handle(
                &session_scope,
                &process_id,
                lash_core::ProcessHandleDescriptor::new(
                    Some("stress"),
                    Some(format!("process-list-stress-{index:05}")),
                ),
            )
            .await?;
        if index % 2 == 1 {
            registry
                .complete_process(
                    &process_id,
                    lash_core::ProcessAwaitOutput::Success {
                        value: serde_json::json!({ "index": index }),
                        control: None,
                    },
                )
                .await?;
        }
    }
    // Dedicated long-lived process for the signal/wait phases: its event log
    // grows across turns, so the phases also expose append-cost growth with
    // log length (the durable-suspension hot path).
    let signal_process_id = "process-signal-stress";
    let signal_event_type = lash_core::process_signal_event_type("stress")?;
    registry
        .register_process(
            lash_core::ProcessRegistration::new(
                signal_process_id,
                lash_core::ProcessInput::External {
                    metadata: serde_json::json!({ "label": "signal stress" }),
                },
                lash_core::ProcessProvenance::host("runtime-perf"),
            )
            .with_event_types(vec![lash_core::ProcessEventType {
                name: signal_event_type.clone(),
                payload_schema: lash_core::LashSchema::any(),
                semantics: lash_core::ProcessEventSemanticsSpec::default(),
            }]),
        )
        .await?;
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    let mut rendered_payload_bytes = 0usize;
    for turn_index in 0..chat_turns {
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let mut phase_profile = BTreeMap::new();

        let phase_started = Instant::now();
        let phase_before_alloc = allocator_stats();
        let phase_before_memory = process_memory_sample();
        let live_entries = registry.list_live_handle_grants(&session_scope).await?;
        phase_profile.insert(
            "process_list_stress.list_live".to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(phase_started),
                allocations: alloc_delta(phase_before_alloc, allocator_stats()),
                rss_growth_kb: diff_opt_i64(
                    phase_before_memory.rss_kb,
                    process_memory_sample().rss_kb,
                ),
            },
        );
        if live_entries.iter().any(|(_, record)| record.is_terminal()) {
            anyhow::bail!("process_list_stress live listing included a terminal process");
        }

        let phase_started = Instant::now();
        let phase_before_alloc = allocator_stats();
        let phase_before_memory = process_memory_sample();
        let all_entries = registry.list_handle_grants(&session_scope).await?;
        phase_profile.insert(
            "process_list_stress.list_all".to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(phase_started),
                allocations: alloc_delta(phase_before_alloc, allocator_stats()),
                rss_growth_kb: diff_opt_i64(
                    phase_before_memory.rss_kb,
                    process_memory_sample().rss_kb,
                ),
            },
        );
        if all_entries.len() != process_count {
            anyhow::bail!(
                "process_list_stress all listing expected {process_count} entries, got {}",
                all_entries.len()
            );
        }

        let phase_started = Instant::now();
        let phase_before_alloc = allocator_stats();
        let phase_before_memory = process_memory_sample();
        let global_records = registry
            .list_processes(&lash_core::ProcessListFilter {
                status: lash_core::ProcessStatusFilter::Any,
                ..lash_core::ProcessListFilter::default()
            })
            .await?;
        phase_profile.insert(
            "process_list_stress.list_global".to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(phase_started),
                allocations: alloc_delta(phase_before_alloc, allocator_stats()),
                rss_growth_kb: diff_opt_i64(
                    phase_before_memory.rss_kb,
                    process_memory_sample().rss_kb,
                ),
            },
        );
        if global_records.len() != process_count + 1 {
            anyhow::bail!(
                "process_list_stress global listing expected {} records, got {}",
                process_count + 1,
                global_records.len()
            );
        }

        const SIGNALS_PER_TURN: usize = 32;
        let phase_started = Instant::now();
        let phase_before_alloc = allocator_stats();
        let phase_before_memory = process_memory_sample();
        for signal_index in 0..SIGNALS_PER_TURN {
            registry
                .append_event(
                    signal_process_id,
                    lash_core::ProcessEventAppendRequest::new(
                        signal_event_type.clone(),
                        serde_json::json!({ "turn": turn_index, "n": signal_index }),
                    )
                    .with_replay_key(format!(
                        "process:{signal_process_id}:signal.stress:{turn_index}:{signal_index}"
                    )),
                )
                .await?;
        }
        phase_profile.insert(
            "process_list_stress.signal_append".to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(phase_started),
                allocations: alloc_delta(phase_before_alloc, allocator_stats()),
                rss_growth_kb: diff_opt_i64(
                    phase_before_memory.rss_kb,
                    process_memory_sample().rss_kb,
                ),
            },
        );

        let phase_started = Instant::now();
        let phase_before_alloc = allocator_stats();
        let phase_before_memory = process_memory_sample();
        let waiting = registry
            .set_process_wait(
                signal_process_id,
                lash_core::WaitState {
                    since_ms: turn_index as u64 + 1,
                    kind: lash_core::WaitKind::Signal {
                        name: "stress".to_string(),
                        event_type: signal_event_type.clone(),
                        key: format!(
                            "process:{signal_process_id}:signal.stress:{}",
                            turn_index + 1
                        ),
                        ordinal: turn_index as u64 + 1,
                    },
                },
            )
            .await?;
        if waiting.wait.is_none() {
            anyhow::bail!("process_list_stress wait facet did not round-trip");
        }
        registry.clear_process_wait(signal_process_id).await?;
        phase_profile.insert(
            "process_list_stress.wait_roundtrip".to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(phase_started),
                allocations: alloc_delta(phase_before_alloc, allocator_stats()),
                rss_growth_kb: diff_opt_i64(
                    phase_before_memory.rss_kb,
                    process_memory_sample().rss_kb,
                ),
            },
        );

        // Env-spec hashing is the new per-start cost (content-addressed
        // capture); measure it standalone so regressions in stable_hash or
        // spec encoding surface here rather than inside start latency.
        const ENV_HASHES_PER_TURN: usize = 64;
        let phase_started = Instant::now();
        let phase_before_alloc = allocator_stats();
        let phase_before_memory = process_memory_sample();
        for hash_index in 0..ENV_HASHES_PER_TURN {
            let mut options = lash_core::PluginOptions::default();
            options.plugins.insert(
                "stress".to_string(),
                serde_json::json!({ "turn": turn_index, "n": hash_index }),
            );
            let spec = lash_core::ProcessExecutionEnvSpec::new(
                options,
                lash_core::SessionPolicy::default(),
            );
            let env_ref = spec
                .stable_ref()
                .map_err(|err| anyhow::anyhow!("env spec hashing failed: {err}"))?;
            if env_ref.as_str().is_empty() {
                anyhow::bail!("process_list_stress env hash produced an empty ref");
            }
        }
        phase_profile.insert(
            "process_list_stress.env_spec_hash".to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(phase_started),
                allocations: alloc_delta(phase_before_alloc, allocator_stats()),
                rss_growth_kb: diff_opt_i64(
                    phase_before_memory.rss_kb,
                    process_memory_sample().rss_kb,
                ),
            },
        );

        let (live_payload_len, live_render_phase) =
            measure_runtime_perf_phase("process_list_stress.render_live_json", || {
                serde_json::to_string(&process_list_tool_payload(&live_entries))
                    .map(|payload| payload.len())
                    .map_err(anyhow::Error::from)
            })?;
        phase_profile.insert(live_render_phase.0, live_render_phase.1);
        let (all_payload_len, all_render_phase) =
            measure_runtime_perf_phase("process_list_stress.render_all_json", || {
                serde_json::to_string(&process_list_tool_payload(&all_entries))
                    .map(|payload| payload.len())
                    .map_err(anyhow::Error::from)
            })?;
        phase_profile.insert(all_render_phase.0, all_render_phase.1);
        rendered_payload_bytes += live_payload_len + all_payload_len;

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
    let _export_shape = serde_json::json!({
        "process_count": process_count,
        "rendered_payload_bytes": rendered_payload_bytes,
    })
    .to_string();
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
        session_nodes: process_count,
        active_path_messages: process_count / 2,
        extra_counters: BTreeMap::from([
            ("process_count".to_string(), process_count as u64),
            (
                "rendered_payload_bytes".to_string(),
                rendered_payload_bytes as u64,
            ),
        ]),
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

fn process_list_stress_registration(
    process_id: String,
    session_scope: lash_core::SessionScope,
    index: usize,
) -> lash_core::ProcessRegistration {
    lash_core::ProcessRegistration::new(
        process_id,
        lash_core::ProcessInput::External {
            metadata: serde_json::json!({ "index": index }),
        },
        lash_core::ProcessProvenance::session(session_scope, "runtime-perf-process-list"),
    )
}

fn process_list_tool_payload(
    entries: &[lash_core::runtime::ProcessHandleGrantEntry],
) -> serde_json::Value {
    serde_json::json!(
        entries
            .iter()
            .cloned()
            .map(lash_core::ProcessHandleSummary::from)
            .collect::<Vec<_>>()
    )
}

const OPENAI_RESPONSES_SSE_CHUNK_COUNT: usize = 256;
const OPENAI_RESPONSES_SSE_CHUNK_BYTES: usize = 96;

fn openai_responses_sse_payload(turn_index: usize) -> String {
    let alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
    let mut full_text = String::new();
    let mut body = String::new();
    let message_id = format!("msg-runtime-perf-{turn_index}");
    let reasoning_id = format!("rs-runtime-perf-{turn_index}");
    let function_item_id = format!("fc-runtime-perf-{turn_index}");
    let function_call_id = format!("call-runtime-perf-{turn_index}");

    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "id": message_id.as_str(),
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": []
            }
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "id": reasoning_id.as_str(),
                "type": "reasoning",
                "summary": []
            }
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.reasoning_summary_part.added",
            "item_id": reasoning_id.as_str(),
            "summary_index": 0,
            "part": { "type": "summary_text", "text": "" }
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": reasoning_id.as_str(),
            "summary_index": 0,
            "delta": "parser benchmark reasoning "
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.reasoning_summary_text.done",
            "item_id": reasoning_id.as_str(),
            "summary_index": 0,
            "text": "parser benchmark reasoning "
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.reasoning_summary_part.done",
            "item_id": reasoning_id.as_str(),
            "summary_index": 0,
            "part": { "type": "summary_text", "text": "parser benchmark reasoning " }
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "id": reasoning_id.as_str(),
                "type": "reasoning",
                "summary": [
                    { "type": "summary_text", "text": "parser benchmark reasoning " }
                ]
            }
        }),
    );

    for index in 0..OPENAI_RESPONSES_SSE_CHUNK_COUNT {
        let prefix = format!("responses-chunk-{index:03}: ");
        let fill_len = OPENAI_RESPONSES_SSE_CHUNK_BYTES.saturating_sub(prefix.len() + 1);
        let fill = alphabet
            .chars()
            .cycle()
            .skip(index % alphabet.len())
            .take(fill_len)
            .collect::<String>();
        let delta = format!("{prefix}{fill}\n");
        full_text.push_str(&delta);
        push_sse_event(
            &mut body,
            serde_json::json!({
                "type": "response.output_text.delta",
                "item_id": message_id.as_str(),
                "content_index": 0,
                "delta": delta
            }),
        );
    }
    full_text.push_str("runtime perf benchmark ok");

    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.output_text.delta",
            "item_id": message_id.as_str(),
            "content_index": 0,
            "delta": "runtime perf benchmark ok"
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.output_text.done",
            "item_id": message_id.as_str(),
            "content_index": 0,
            "text": full_text.as_str()
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "id": message_id.as_str(),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_text.as_str()
                    }
                ]
            }
        }),
    );

    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "id": function_item_id.as_str(),
                "type": "function_call",
                "call_id": function_call_id.as_str(),
                "name": "benchmark_echo",
                "arguments": ""
            }
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "item_id": function_item_id.as_str(),
            "delta": "{\"value\":"
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "item_id": function_item_id.as_str(),
            "delta": "\"runtime perf benchmark ok\"}"
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.function_call_arguments.done",
            "item_id": function_item_id.as_str(),
            "arguments": "{\"value\":\"runtime perf benchmark ok\"}"
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "id": function_item_id.as_str(),
                "type": "function_call",
                "call_id": function_call_id.as_str(),
                "name": "benchmark_echo",
                "arguments": "{\"value\":\"runtime perf benchmark ok\"}"
            }
        }),
    );
    push_sse_event(
        &mut body,
        serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": format!("resp-runtime-perf-{turn_index}"),
                "type": "response",
                "status": "completed",
                "output": [
                    {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": full_text.as_str()
                            }
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 1024,
                    "output_tokens": 64,
                    "input_tokens_details": {
                        "cached_tokens": 512
                    },
                    "output_tokens_details": {
                        "reasoning_tokens": 48
                    }
                }
            }
        }),
    );
    body.push_str("data: [DONE]\n\n");
    body
}

fn push_sse_event(body: &mut String, event: serde_json::Value) {
    body.push_str("data: ");
    body.push_str(&event.to_string());
    body.push_str("\n\n");
}

fn direct_llm_client_request(turn_index: usize) -> lash::direct::DirectRequest {
    lash::direct::DirectRequest::json_schema(
        "mock-model",
        format!(
            "Direct LLM client runtime perf turn {}. Return the benchmark marker.",
            turn_index + 1
        ),
        lash::direct::DirectJsonSchema {
            name: "runtime_perf_direct_completion".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "value", "error"],
                "properties": {
                    "kind": { "type": "string", "enum": ["value", "error"] },
                    "value": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "null" }
                        ]
                    },
                    "error": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "null" }
                        ]
                    }
                }
            }),
            strict: true,
        },
    )
}

fn validate_direct_llm_response(turn_index: usize, response: &LlmResponse) -> anyhow::Result<()> {
    let value: serde_json::Value = serde_json::from_str(&response.full_text)
        .with_context(|| format!("parse direct_llm_client turn {} JSON", turn_index + 1))?;
    if value.get("value").and_then(serde_json::Value::as_str) == Some("runtime perf benchmark ok") {
        return Ok(());
    }
    anyhow::bail!(
        "runtime perf scenario direct_llm_client turn {} produced unexpected response: {}",
        turn_index + 1,
        response.full_text
    );
}

fn measure_runtime_perf_phase<T>(
    name: &'static str,
    f: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<(T, (String, RuntimePerfPhaseRunResult))> {
    let before_alloc = allocator_stats();
    let before_memory = process_memory_sample();
    let started = Instant::now();
    let value = f()?;
    let after_alloc = allocator_stats();
    let after_memory = process_memory_sample();
    Ok((
        value,
        (
            name.to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(started),
                allocations: alloc_delta(before_alloc, after_alloc),
                rss_growth_kb: diff_opt_i64(before_memory.rss_kb, after_memory.rss_kb),
            },
        ),
    ))
}

async fn measure_runtime_perf_async_phase<T, F>(
    name: &'static str,
    future: F,
) -> anyhow::Result<(T, (String, RuntimePerfPhaseRunResult))>
where
    F: Future<Output = anyhow::Result<T>>,
{
    let before_alloc = allocator_stats();
    let before_memory = process_memory_sample();
    let started = Instant::now();
    let value = future.await?;
    let after_alloc = allocator_stats();
    let after_memory = process_memory_sample();
    Ok((
        value,
        (
            name.to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(started),
                allocations: alloc_delta(before_alloc, after_alloc),
                rss_growth_kb: diff_opt_i64(before_memory.rss_kb, after_memory.rss_kb),
            },
        ),
    ))
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
        extra_counters: BTreeMap::new(),
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
    llm: Arc<dyn ProtocolDriverHandle<lash_core::HostTurnProtocol>>,
    tools: Arc<dyn ProtocolDriverHandle<lash_core::HostTurnProtocol>>,
    exec: Arc<dyn ProtocolDriverHandle<lash_core::HostTurnProtocol>>,
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

impl ProtocolDriverHandle<lash_core::HostTurnProtocol> for CheckpointDriver {
    fn prepare_protocol_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        match self {
            Self::Llm => vec![DriverAction::StartLlm {
                request: ctx.project_llm_request(false),
                driver_state: None,
            }],
            Self::Tools => vec![DriverAction::StartTools {
                calls: checkpoint_tool_calls(ctx.protocol_iteration()),
            }],
            Self::Exec => vec![DriverAction::StartExec {
                code: checkpoint_exec_code(ctx.protocol_iteration()),
                driver_state: lash_core::ProtocolDriverState::new(
                    "runtime_perf_checkpoint",
                    serde_json::json!({
                        "phase": "exec_code",
                        "ip": ctx.protocol_iteration(),
                        "stack": (0..64).map(|index| serde_json::json!({
                            "slot": index,
                            "value": format!("checkpoint-stack-value-{index}")
                        })).collect::<Vec<_>>(),
                    }),
                ),
            }],
        }
    }

    fn handle_llm_success(
        &self,
        _ctx: DriverContextView<'_>,
        _waiting: WaitingLlmState<lash_core::HostTurnProtocol>,
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
        _waiting: WaitingExecState<lash_core::HostTurnProtocol>,
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
    protocol_driver: Arc<dyn ProtocolDriverHandle<lash_core::HostTurnProtocol>>,
) -> TurnMachineConfig {
    TurnMachineConfig {
        protocol_driver,
        projector: Arc::new(ChatContextProjector),
        sync_execution_environment: false,
        model: "mock-model".to_string(),
        max_context_tokens: None,
        max_turns: Some(8),
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        run_session_id: Some("runtime-perf-turn-checkpoint".to_string()),
        autonomous: false,
        tool_specs: Arc::new(Vec::new()),
        system_prompt: Arc::from(
            "Synthetic sans-IO checkpoint profiler prompt. Preserve pending effects across checkpoint restore.",
        ),
        session_id: "runtime-perf-turn-checkpoint".to_string(),
        emit_llm_trace: false,
        termination: ProtocolTurnOptions::default(),
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
                    "Historical checkpoint profiler message {index}. This payload is intentionally long enough to make TurnCheckpoint serialization include realistic prompt and transcript bytes. The current topic is standard and RLM turn-effect replay across LLM, tool, checkpoint, sleep, and ExecCode boundaries."
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

fn checkpoint_tool_calls(protocol_iteration: usize) -> Vec<PendingToolCall> {
    (0..24)
        .map(|index| PendingToolCall {
            call_id: format!("checkpoint-call-{protocol_iteration}-{index}"),
            tool_name: format!("checkpoint_parallel_tool_{}", index % 6),
            args: serde_json::json!({
                "index": index,
                "protocol_iteration": protocol_iteration,
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

fn checkpoint_exec_code(protocol_iteration: usize) -> String {
    format!(
        r#"process benchmark_echo_process(tool: Tools, value: str, ordinal: int) {{
  result = await tool.benchmark_echo({{ value: value, ordinal: ordinal }})?
  finish result
}}

print("checkpoint turn {protocol_iteration}")
first = start benchmark_echo_process(tool: tools, value: "runtime perf benchmark ok", ordinal: 1)
second = start benchmark_echo_process(tool: tools, value: "runtime perf benchmark ok", ordinal: 2)
third = start benchmark_echo_process(tool: tools, value: "runtime perf benchmark ok", ordinal: 3)
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
        .mode(scenario.execution_mode())
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
                let effect_host = session.effect_host();
                let scoped_effect_controller = effect_host
                    .scoped(lash::runtime::EffectScope::turn(
                        session.session_id(),
                        format!("runtime-perf-embed-{}", turn_index + 1),
                    ))
                    .map_err(anyhow::Error::from)?;
                session
                    .turn(lash_core::TurnInput::text(benchmark_prompt(
                        scenario, turn_index,
                    )))
                    .cancel(cancel)
                    .advanced()
                    .collect_session_events_with_scope(
                        &lash::runtime::NoopEventSink,
                        scoped_effect_controller,
                    )
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
        extra_counters: BTreeMap::new(),
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

fn token_usage_from_llm_usage(usage: &LlmUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
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
        RuntimeTurnPhase::FinalCommit => "final_commit",
        RuntimeTurnPhase::PostPersistHooks => "post_persist_hooks",
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
