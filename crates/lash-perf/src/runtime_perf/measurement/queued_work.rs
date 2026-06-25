const QUEUED_WORK_JOIN_BATCHES_PER_TURN: usize = 32;
const QUEUED_WORK_SEED_OTHER_SESSION_BATCHES: usize = 64;
const QUEUED_WORK_CLAIM_TTL_MS: u64 = 30_000;

async fn run_once_queued_work_claim_stress(
    chat_turns: usize,
) -> anyhow::Result<RuntimePerfRunResult> {
    let scenario = RuntimePerfScenario::QueuedWorkClaimStress;
    let session_id = "runtime-perf-queued-work";
    let owner = lash_core::LeaseOwnerIdentity::opaque("runtime-perf", "queued-work-stress");
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let store = Arc::new(RuntimePerfStore::default());
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    for index in 0..QUEUED_WORK_SEED_OTHER_SESSION_BATCHES {
        store
            .enqueue_queued_work(
                QueuedWorkBatchDraft::new(
                    "runtime-perf-queued-work-other",
                    DeliveryPolicy::EarliestSafeBoundary,
                    SlotPolicy::Join,
                    vec![QueuedWorkPayload::turn_input(TurnInput::text(format!(
                        "other queued work {index}"
                    )))],
                )
                .with_merge_key(MergeKey::Group("other-session-noise".to_string()))
                .with_source_key(format!("other:{index}")),
            )
            .await?;
    }
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    let mut enqueued_batches = QUEUED_WORK_SEED_OTHER_SESSION_BATCHES;
    let mut command_claims = 0usize;
    let mut join_claims = 0usize;
    let mut join_batches_claimed = 0usize;
    let mut exclusive_claims = 0usize;
    let mut completed_batches = 0usize;

    for turn_index in 0..chat_turns {
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let mut phase_profile = BTreeMap::new();

        let (lease, phase) =
            measure_runtime_perf_async_phase("queued_work.claim_session_lease", async {
                store
                    .try_claim_session_execution_lease(session_id, &owner, QUEUED_WORK_CLAIM_TTL_MS)
                    .await?
                    .acquired()
                    .ok_or_else(|| anyhow::anyhow!("queued-work stress lease was busy"))
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);

        let (_, phase) =
            measure_runtime_perf_async_phase("queued_work.enqueue_mixed_batch", async {
                enqueue_queued_work_stress_turn(store.as_ref(), session_id, turn_index).await
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        enqueued_batches += QUEUED_WORK_JOIN_BATCHES_PER_TURN + 2;

        let (command_claim, phase) =
            measure_runtime_perf_async_phase("queued_work.claim_session_command", async {
                store
                    .claim_leading_ready_session_command(
                        session_id,
                        &lease.fence(),
                        &owner,
                        QUEUED_WORK_CLAIM_TTL_MS,
                    )
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("queued-work stress expected command claim"))
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        command_claims += 1;
        if command_claim.batches.len() != 1 || command_claim.exclusive_session_command().is_none() {
            anyhow::bail!("queued-work stress command claim did not contain one session command");
        }

        let (_, phase) =
            measure_runtime_perf_async_phase("queued_work.complete_session_command", async {
                store
                    .commit_runtime_state(queued_work_stress_commit(
                        session_id,
                        &lease,
                        vec![command_claim.completion()],
                        false,
                    ))
                    .await
                    .map(|_| ())
                    .map_err(anyhow::Error::from)
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        completed_batches += 1;

        let (join_claim, phase) =
            measure_runtime_perf_async_phase("queued_work.claim_join_turn_work", async {
                store
                    .claim_ready_queued_work(
                        session_id,
                        &lease.fence(),
                        &owner,
                        QueuedWorkClaimBoundary::Idle,
                        QUEUED_WORK_CLAIM_TTL_MS,
                        QUEUED_WORK_JOIN_BATCHES_PER_TURN,
                    )
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("queued-work stress expected join claim"))
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        if join_claim.batches.len() != QUEUED_WORK_JOIN_BATCHES_PER_TURN {
            anyhow::bail!(
                "queued-work stress expected {} joined batches, got {}",
                QUEUED_WORK_JOIN_BATCHES_PER_TURN,
                join_claim.batches.len()
            );
        }
        join_claims += 1;
        join_batches_claimed += join_claim.batches.len();
        let join_batch_ids = join_claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.clone())
            .collect::<Vec<_>>();

        let (join_claim, phase) =
            measure_runtime_perf_async_phase("queued_work.renew_join_claim", async {
                store
                    .renew_queued_work_claim(&join_claim, QUEUED_WORK_CLAIM_TTL_MS)
                    .await
                    .map_err(anyhow::Error::from)
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);

        let (_, phase) =
            measure_runtime_perf_async_phase("queued_work.abandon_join_claim", async {
                store
                    .abandon_queued_work_claim(&join_claim)
                    .await
                    .map_err(anyhow::Error::from)
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);

        let (join_claim, phase) =
            measure_runtime_perf_async_phase("queued_work.reclaim_by_batch_ids", async {
                store
                    .claim_ready_queued_work_by_batch_ids(
                        session_id,
                        &lease.fence(),
                        &owner,
                        QueuedWorkClaimBoundary::Idle,
                        QUEUED_WORK_CLAIM_TTL_MS,
                        &join_batch_ids,
                    )
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("queued-work stress expected exact reclaim"))
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        if join_claim
            .batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .ne(join_batch_ids.iter().map(String::as_str))
        {
            anyhow::bail!("queued-work stress exact reclaim returned different batches");
        }

        let (_, phase) =
            measure_runtime_perf_async_phase("queued_work.complete_join_turn_work", async {
                store
                    .commit_runtime_state(queued_work_stress_commit(
                        session_id,
                        &lease,
                        vec![join_claim.completion()],
                        false,
                    ))
                    .await
                    .map(|_| ())
                    .map_err(anyhow::Error::from)
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        completed_batches += join_claim.batches.len();

        let (exclusive_claim, phase) =
            measure_runtime_perf_async_phase("queued_work.claim_exclusive_turn_work", async {
                store
                    .claim_ready_queued_work(
                        session_id,
                        &lease.fence(),
                        &owner,
                        QueuedWorkClaimBoundary::Idle,
                        QUEUED_WORK_CLAIM_TTL_MS,
                        QUEUED_WORK_JOIN_BATCHES_PER_TURN,
                    )
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("queued-work stress expected exclusive claim"))
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        if exclusive_claim.batches.len() != 1 {
            anyhow::bail!(
                "queued-work stress expected one exclusive batch, got {}",
                exclusive_claim.batches.len()
            );
        }
        exclusive_claims += 1;

        let (_, phase) =
            measure_runtime_perf_async_phase("queued_work.complete_exclusive_turn_work", async {
                store
                    .commit_runtime_state(queued_work_stress_commit(
                        session_id,
                        &lease,
                        vec![exclusive_claim.completion()],
                        true,
                    ))
                    .await
                    .map(|_| ())
                    .map_err(anyhow::Error::from)
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        completed_batches += 1;

        let (pending, phase) =
            measure_runtime_perf_async_phase("queued_work.list_pending", async {
                store
                    .list_pending_queued_work(session_id)
                    .await
                    .map_err(anyhow::Error::from)
            })
            .await?;
        phase_profile.insert(phase.0, phase.1);
        if !pending.is_empty() {
            anyhow::bail!(
                "queued-work stress left {} pending batches for measured session",
                pending.len()
            );
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
    let remaining_measured = store.list_queued_work(session_id).await?.len();
    let remaining_other = store
        .list_queued_work("runtime-perf-queued-work-other")
        .await?
        .len();
    let _export_shape = serde_json::json!({
        "enqueued_batches": enqueued_batches,
        "completed_batches": completed_batches,
        "remaining_measured_batches": remaining_measured,
        "remaining_other_batches": remaining_other,
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
        stack_profile: None,
        build_runtime_ms,
        seed_state_ms,
        run_turn_ms: round3(turns.iter().map(|turn| turn.run_turn_ms).sum()),
        await_background_work_ms: round3(
            turns.iter().map(|turn| turn.await_background_work_ms).sum(),
        ),
        export_state_ms,
        total_ms: elapsed_ms(total_started),
        session_nodes: enqueued_batches,
        active_path_messages: completed_batches,
        extra_counters: BTreeMap::from([
            ("enqueued_batches".to_string(), enqueued_batches as u64),
            ("command_claims".to_string(), command_claims as u64),
            ("join_claims".to_string(), join_claims as u64),
            (
                "join_batches_claimed".to_string(),
                join_batches_claimed as u64,
            ),
            ("exclusive_claims".to_string(), exclusive_claims as u64),
            ("completed_batches".to_string(), completed_batches as u64),
            (
                "remaining_measured_batches".to_string(),
                remaining_measured as u64,
            ),
            (
                "remaining_other_batches".to_string(),
                remaining_other as u64,
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

async fn enqueue_queued_work_stress_turn(
    store: &RuntimePerfStore,
    session_id: &str,
    turn_index: usize,
) -> anyhow::Result<()> {
    store
        .enqueue_queued_work(
            QueuedWorkBatchDraft::new(
                session_id,
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Exclusive,
                vec![QueuedWorkPayload::session_command(
                    SessionCommand::RefreshToolCatalog {
                        reason: format!("queued-work-stress-{turn_index}"),
                    },
                )],
            )
            .with_source_key(format!("command:{turn_index}")),
        )
        .await?;

    for batch_index in 0..QUEUED_WORK_JOIN_BATCHES_PER_TURN {
        store
            .enqueue_queued_work(
                QueuedWorkBatchDraft::new(
                    session_id,
                    DeliveryPolicy::EarliestSafeBoundary,
                    SlotPolicy::Join,
                    vec![QueuedWorkPayload::turn_input(TurnInput::text(format!(
                        "queued work stress turn {turn_index} batch {batch_index}"
                    )))],
                )
                .with_merge_key(MergeKey::Group(format!("turn:{turn_index}")))
                .with_source_key(format!("turn:{turn_index}:join:{batch_index}")),
            )
            .await?;
    }

    store
        .enqueue_queued_work(
            QueuedWorkBatchDraft::new(
                session_id,
                DeliveryPolicy::EarliestSafeBoundary,
                SlotPolicy::Exclusive,
                vec![QueuedWorkPayload::turn_input(TurnInput::text(format!(
                    "queued work stress exclusive {turn_index}"
                )))],
            )
            .with_source_key(format!("turn:{turn_index}:exclusive")),
        )
        .await?;
    Ok(())
}

fn queued_work_stress_commit(
    session_id: &str,
    lease: &SessionExecutionLease,
    completed_queue_claims: Vec<QueuedWorkCompletion>,
    release_lease: bool,
) -> RuntimeCommit {
    RuntimeCommit {
        session_id: session_id.to_string(),
        expected_head_revision: None,
        session_execution_lease: Some(lease.fence()),
        release_session_execution_lease: release_lease.then(|| lease.completion()),
        config: PersistedSessionConfig::default(),
        agent_frames: Vec::new(),
        current_agent_frame_id: String::new(),
        graph: GraphCommitDelta::Unchanged { leaf_node_id: None },
        checkpoint: HydratedSessionCheckpoint::default(),
        usage_deltas: Vec::new(),
        turn_commit: None,
        completed_queue_claims,
        committed_attachment_ids: Vec::new(),
    }
}
