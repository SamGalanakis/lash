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
    match scenario {
        RuntimePerfScenario::TurnCheckpoint => return run_once_turn_checkpoint(chat_turns).await,
        RuntimePerfScenario::LiveReplayPressure => {
            return run_once_live_replay_pressure(chat_turns).await;
        }
        RuntimePerfScenario::TraceJsonlStandard | RuntimePerfScenario::TraceJsonlExtended => {
            return run_once_trace_jsonl(scenario, chat_turns).await;
        }
        RuntimePerfScenario::OpenAiResponsesSseParse => {
            return run_once_openai_responses_sse_parse(chat_turns).await;
        }
        RuntimePerfScenario::DirectLlmClient => {
            return run_once_direct_llm_client(chat_turns).await;
        }
        RuntimePerfScenario::ProcessListStress => {
            return run_once_process_list_stress(chat_turns).await;
        }
        RuntimePerfScenario::EmbedStandard | RuntimePerfScenario::EmbedRlm => {
            return run_once_embed(scenario, chat_turns).await;
        }
        RuntimePerfScenario::Standard
        | RuntimePerfScenario::Rlm
        | RuntimePerfScenario::StandardToolCalls
        | RuntimePerfScenario::StandardAsyncToolCompletion
        | RuntimePerfScenario::RlmToolCalls
        | RuntimePerfScenario::RlmAsyncToolCompletion
        | RuntimePerfScenario::RlmProcessHandles
        | RuntimePerfScenario::RlmProcessAsyncToolCompletion
        | RuntimePerfScenario::RlmLlmQuery
        | RuntimePerfScenario::RlmGlobals
        | RuntimePerfScenario::RlmLargeToolCatalog
        | RuntimePerfScenario::ObservationalMemory
        | RuntimePerfScenario::ObservationalMemoryMaintenance
        | RuntimePerfScenario::OpenAiCompatStream
        | RuntimePerfScenario::StandardShellOutput
        | RuntimePerfScenario::ToolDiscoverySearch
        | RuntimePerfScenario::ScopedEffectController
        | RuntimePerfScenario::StoreReopen
        | RuntimePerfScenario::SqliteStoreReopen => {}
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
                lash::runtime::ExecutionScope::turn(
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
                runtime.run_turn_with_execution_scope(turn_input, cancel, scoped_effect_controller),
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
