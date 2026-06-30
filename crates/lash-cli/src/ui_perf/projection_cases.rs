use std::time::Instant;

use lash_core::{
    Message, MessageRole, Part, PartKind, PruneState, SessionReadView, SessionSnapshot,
    ToolCallOutput, ToolCancellation, ToolFailure, ToolFailureClass, ToolResult, TurnActivity,
    TurnEvent,
};
use serde_json::json;

use crate::app::{
    App, PreparedTurn, UiProjectionState, UiTimeline, UiTimelineItem, timeline_from_read_view,
};
use crate::turn_runner::make_turn_input;
use crate::ui_trace::render_screen_snapshot_with_perf;
use lash_perf::perf_support::time::elapsed_ms;

use super::measurement::UiPerfRunResult;
use super::scenarios::{BENCH_HEIGHT, BENCH_WIDTH, UiPerfWorkload};

pub(crate) fn run_timeline_projection_once(workload: UiPerfWorkload) -> UiPerfRunResult {
    let total_started = Instant::now();
    let build_started = Instant::now();
    let read_view = build_projection_read_view(workload.turn_count);
    let ui_state = UiProjectionState::default();
    let build_case_ms = elapsed_ms(build_started);

    let mut result = UiPerfRunResult::new(total_started);
    result.build_case_ms = build_case_ms;
    result.sample("build_case_ms", build_case_ms);

    let passes = workload.scroll_passes.max(1);
    let mut latest_timeline = UiTimeline::default();
    let mut chronological_entries = 0usize;
    for _ in 0..passes {
        let projection_started = Instant::now();
        let projection = read_view.chronological_projection();
        result.sample(
            "chronological_projection_ms",
            elapsed_ms(projection_started),
        );
        chronological_entries = projection.entries().len();

        let timeline_started = Instant::now();
        latest_timeline = timeline_from_read_view(&read_view, &ui_state);
        result.sample("timeline_from_read_view_ms", elapsed_ms(timeline_started));
    }

    let mut app = App::new(
        "gpt-5.4".to_string(),
        "ui-projection-perf".to_string(),
        "ui-projection-session".to_string(),
    );
    app.start_turn();
    let finish_started = Instant::now();
    app.finish_turn_from_read_view(&read_view);
    result.sample("finish_turn_from_read_view_ms", elapsed_ms(finish_started));

    result.total_ms = elapsed_ms(total_started);
    result.total_blocks = latest_timeline.items().len();
    result.total_content_rows = app.total_content_height(220, 72);
    result.counter("chronological_entries", chronological_entries as u64);
    count_timeline_items(&mut result, latest_timeline.items());
    result
}

pub(crate) fn run_activity_projection_once(workload: UiPerfWorkload) -> UiPerfRunResult {
    let total_started = Instant::now();
    let mut result = UiPerfRunResult::new(total_started);
    let mut app = App::new(
        "gpt-5.4".to_string(),
        "ui-activity-perf".to_string(),
        "ui-activity-session".to_string(),
    );
    app.start_turn();

    let events = workload.control_events.max(1);
    for index in 0..events {
        let event = live_activity_event(index);
        let started = Instant::now();
        app.handle_turn_activity(TurnActivity::independent(event));
        result.sample("turn_activity_handle_ms", elapsed_ms(started));
    }

    let height_started = Instant::now();
    let total_content_rows = app.total_content_height(220, 72);
    result.sample("post_activity_height_ms", elapsed_ms(height_started));

    result.total_ms = elapsed_ms(total_started);
    result.total_blocks = app.timeline.len();
    result.total_content_rows = total_content_rows;
    count_timeline_items(&mut result, app.timeline.items());
    result
}

pub(crate) fn run_turn_interrupt_steer_reconciliation_once(
    workload: UiPerfWorkload,
) -> UiPerfRunResult {
    let total_started = Instant::now();
    let build_started = Instant::now();
    let read_view = build_projection_read_view(workload.turn_count);
    let mut app = App::new(
        "gpt-5.4".to_string(),
        "ui-interrupt-perf".to_string(),
        "ui-interrupt-session".to_string(),
    );
    app.finish_turn_from_read_view(&read_view);
    app.start_turn();
    app.handle_turn_activity(TurnActivity::independent(TurnEvent::AssistantProseDelta {
        text: "I am midway through the current answer while a steer arrives.".to_string(),
    }));

    let active = PreparedTurn::prepare_with_effective_text(
        "Also inspect the pending interrupt reconciliation path.".to_string(),
        "Also inspect the pending interrupt reconciliation path.".to_string(),
        Vec::new(),
    );
    let deferred = PreparedTurn::prepare_with_effective_text(
        "Carry this steer into the next turn after the interrupt.".to_string(),
        "Carry this steer into the next turn after the interrupt.".to_string(),
        Vec::new(),
    );
    let queued = PreparedTurn::prepare_with_effective_text(
        "Then run the queued follow-up exactly once.".to_string(),
        "Then run the queued follow-up exactly once.".to_string(),
        Vec::new(),
    );
    app.cache_draft_presentation(active.clone());
    app.cache_draft_presentation(deferred.clone());
    app.cache_draft_presentation(queued.clone());

    let active_pending = pending_turn_input_for_ui_perf(
        1,
        "ui-ti-active-accepted",
        &active,
        lash_core::TurnInputIngress::active_turn(
            "ui-active-turn",
            lash_core::TurnInputCheckpointBoundary::AfterWork,
        ),
    );
    let mut deferred_pending = pending_turn_input_for_ui_perf(
        2,
        "ui-ti-active-deferred",
        &deferred,
        lash_core::TurnInputIngress::active_turn(
            "ui-active-turn",
            lash_core::TurnInputCheckpointBoundary::BeforeCompletion,
        ),
    );
    let queued_pending = pending_turn_input_for_ui_perf(
        3,
        "ui-ti-next",
        &queued,
        lash_core::TurnInputIngress::NextTurn,
    );
    let active_input_id = active_pending.input_id.clone();
    app.set_pending_turn_input_snapshot(vec![
        active_pending,
        deferred_pending.clone(),
        queued_pending.clone(),
    ]);
    let build_case_ms = elapsed_ms(build_started);

    let mut result = UiPerfRunResult::new(total_started);
    result.build_case_ms = build_case_ms;
    result.sample("build_case_ms", build_case_ms);

    let initial_render_started = Instant::now();
    let (mut snapshot, _) =
        render_screen_snapshot_with_perf(&mut app, BENCH_WIDTH, BENCH_HEIGHT, None);
    result.sample(
        "queue_preview_render_ms",
        elapsed_ms(initial_render_started),
    );

    let accept_started = Instant::now();
    app.push_prepared_user_input(&active);
    app.remove_pending_turn_inputs(std::slice::from_ref(&active_input_id));
    result.sample(
        "active_accept_reconciliation_ms",
        elapsed_ms(accept_started),
    );

    let accepted_render_started = Instant::now();
    let (next_snapshot, _) =
        render_screen_snapshot_with_perf(&mut app, BENCH_WIDTH, BENCH_HEIGHT, Some(&snapshot));
    snapshot = next_snapshot;
    result.sample(
        "queue_preview_render_ms",
        elapsed_ms(accepted_render_started),
    );

    app.note_manual_interrupt_requested();
    let ui_state = UiProjectionState::from_app(&app);
    let interrupt_started = Instant::now();
    app.finish_interrupted_turn_from_read_view(
        &read_view,
        &ui_state,
        crate::util::manual_interrupt_message(),
    );
    deferred_pending.ingress = lash_core::TurnInputIngress::NextTurn;
    deferred_pending.state = lash_core::TurnInputState::DeferredNextTurn;
    app.set_pending_turn_input_snapshot(vec![deferred_pending.clone(), queued_pending.clone()]);
    result.sample("interrupt_reconciliation_ms", elapsed_ms(interrupt_started));

    let interrupted_render_started = Instant::now();
    let (next_snapshot, _) =
        render_screen_snapshot_with_perf(&mut app, BENCH_WIDTH, BENCH_HEIGHT, Some(&snapshot));
    snapshot = next_snapshot;
    result.sample(
        "queue_preview_render_ms",
        elapsed_ms(interrupted_render_started),
    );

    let idle_dispatch_started = Instant::now();
    let ready_inputs = app.pending_turn_input_snapshot().to_vec();
    for pending in &ready_inputs {
        if let Some(turn) = app.take_prepared_turn_for_pending_input(pending) {
            app.push_prepared_user_input(&turn);
        }
    }
    let ready_input_ids = ready_inputs
        .iter()
        .map(|input| input.input_id.clone())
        .collect::<Vec<_>>();
    app.remove_pending_turn_inputs(&ready_input_ids);
    result.sample(
        "idle_dispatch_reconciliation_ms",
        elapsed_ms(idle_dispatch_started),
    );

    let final_render_started = Instant::now();
    let (_snapshot, _) =
        render_screen_snapshot_with_perf(&mut app, BENCH_WIDTH, BENCH_HEIGHT, Some(&snapshot));
    result.sample("queue_preview_render_ms", elapsed_ms(final_render_started));

    result.total_ms = elapsed_ms(total_started);
    result.total_blocks = app.timeline.len();
    result.total_content_rows =
        app.total_content_height(BENCH_WIDTH as usize, BENCH_HEIGHT as usize);
    result.counter("pending_inputs_seeded", 3);
    result.counter("pending_inputs_dispatched", ready_input_ids.len() as u64);
    result.counter("timeline_blocks_after_reconcile", app.timeline.len() as u64);
    count_timeline_items(&mut result, app.timeline.items());
    result
}

fn pending_turn_input_for_ui_perf(
    enqueue_seq: u64,
    input_id: &str,
    turn: &PreparedTurn,
    ingress: lash_core::TurnInputIngress,
) -> lash_core::PendingTurnInput {
    let state = match ingress {
        lash_core::TurnInputIngress::ActiveTurn { .. } => lash_core::TurnInputState::PendingActive,
        lash_core::TurnInputIngress::NextTurn => lash_core::TurnInputState::DeferredNextTurn,
    };
    lash_core::PendingTurnInput {
        input_id: input_id.to_string(),
        session_id: "ui-interrupt-session".to_string(),
        enqueue_seq,
        source_key: Some(format!("host:{}", turn.draft_id)),
        ingress,
        state,
        enqueued_at_ms: enqueue_seq,
        input: make_turn_input(turn),
    }
}

pub(crate) fn build_projection_read_view(turn_count: usize) -> SessionReadView {
    let mut graph = lash_core::SessionGraph::default();

    for turn in 0..turn_count {
        let user = text_message(
            &format!("u{turn}"),
            MessageRole::User,
            &format!("Inspect projection turn {turn}"),
        );
        graph.append_message(user.clone());

        let assistant = assistant_message(
            &format!("a{turn}"),
            &format!("Reasoning for projection turn {turn}"),
            &format!("Projection result {turn} with a concise rendered answer."),
        );
        graph.append_message(assistant);

        if turn % 5 == 0 {
            graph.append_protocol_event(lash_protocol_rlm::rlm_protocol_event(
                lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(
                    lash_rlm_types::RlmTrajectoryEntry {
                        id: format!("lashlang_step_{turn}"),
                        protocol_iteration: turn,
                        code: "now = await shell.exec({ cmd: \"date -u\" })?\nprint now"
                            .to_string(),
                        output: vec!["time".to_string()],
                        images: Vec::new(),
                        error: None,
                        final_output: (turn % 10 == 0).then(|| json!("RLM final output")),
                    },
                ),
            ));
        }
    }

    let state = SessionSnapshot {
        session_graph: graph,
        ..SessionSnapshot::default()
    };
    SessionReadView::from_snapshot(&state)
}

fn live_activity_event(index: usize) -> TurnEvent {
    match index % 14 {
        0 => TurnEvent::ModelRequestStarted {
            protocol_iteration: index / 14,
        },
        1 => TurnEvent::ReasoningDelta {
            text: format!("reasoning chunk {index}\n"),
        },
        2 => TurnEvent::AssistantProseDelta {
            text: format!("assistant prose chunk {index}\n"),
        },
        3 => TurnEvent::ToolCallStarted {
            call_id: Some(format!("shell-{index}")),
            name: "exec_command".to_string(),
            args: json!({ "cmd": "cargo test -p lash-cli" }),
        },
        4 => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("shell-{index}")),
            name: "exec_command".to_string(),
            args: json!({ "cmd": "cargo test -p lash-cli", "workdir": "/home/sam/code/lash" }),
            output: ToolCallOutput::success(json!({ "output": "ok\n", "exit_code": 0 })),
            duration_ms: 12,
        },
        5 => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("shell-fail-{index}")),
            name: "exec_command".to_string(),
            args: json!({ "cmd": "false", "workdir": "/home/sam/code/lash" }),
            output: ToolCallOutput::failure(ToolFailure::tool(
                ToolFailureClass::Execution,
                "exit_1",
                "command exited with 1",
            )),
            duration_ms: 6,
        },
        6 => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("cancel-{index}")),
            name: "read_file".to_string(),
            args: json!({ "path": "README.md" }),
            output: ToolCallOutput::cancelled(ToolCancellation::runtime("tool call cancelled")),
            duration_ms: 1,
        },
        7 => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("agent-{index}")),
            name: "spawn_agent".to_string(),
            args: json!({ "task": "inspect projection path", "capability": "explore" }),
            output: ToolCallOutput::success(json!({ "claim": "done" })),
            duration_ms: 22,
        },
        8 => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("process-{index}")),
            name: "list_process_handles".to_string(),
            args: json!({}),
            output: ToolCallOutput::success(json!([
                {
                    "__handle__": "process",
                    "id": "tool-call-app-log",
                    "process_id": "tool-call-app-log",
                    "descriptor": { "kind": "tool", "label": "app_log" },
                    "status": "running"
                }
            ])),
            duration_ms: 2,
        },
        9 => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("batch-{index}")),
            name: "batch".to_string(),
            args: json!({
                "tool_calls": [
                    {"tool": "read_file", "parameters": {"path": "crates/lash-cli/src/app/projection.rs"}},
                    {"tool": "grep", "parameters": {"query": "timeline_from_read_view"}}
                ]
            }),
            output: ToolCallOutput::success(json!({
                "results": [
                    {"tool": "read_file", "success": true, "result": "projection source"},
                    {"tool": "grep", "success": false, "error": "no matches"}
                ]
            })),
            duration_ms: 10,
        },
        10 => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("plan-{index}")),
            name: "update_plan".to_string(),
            args: json!({
                "plan": [
                    {"step": "Inspect projection", "status": "completed"},
                    {"step": "Patch profiler", "status": "in_progress"}
                ]
            }),
            output: ToolCallOutput::success(json!({ "ok": true })),
            duration_ms: 1,
        },
        11 => TurnEvent::RetryStatus {
            wait_seconds: 1,
            attempt: 2,
            max_attempts: 3,
            reason: "provider backpressure while profiling".to_string(),
        },
        12 => TurnEvent::FinalValue {
            value: json!({ "status": "done", "index": index }),
        },
        _ => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("generic-{index}")),
            name: "search_tools".to_string(),
            args: json!({ "query": "projection tools" }),
            output: ToolResult::err(json!("tool search failed"))
                .into_done_output()
                .expect("static failure output"),
            duration_ms: 4,
        },
    }
}

fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
    Message {
        id: id.to_string(),
        role,
        parts: vec![part(&format!("{id}.p0"), PartKind::Text, content)].into(),
        origin: None,
    }
}

fn assistant_message(id: &str, reasoning: &str, text: &str) -> Message {
    Message {
        id: id.to_string(),
        role: MessageRole::Assistant,
        parts: vec![
            part(&format!("{id}.r0"), PartKind::Reasoning, reasoning),
            part(&format!("{id}.p0"), PartKind::Text, text),
        ]
        .into(),
        origin: None,
    }
}

fn part(id: &str, kind: PartKind, content: &str) -> Part {
    Part {
        id: id.to_string(),
        kind,
        content: content.to_string(),
        attachment: None,
        tool_call_id: None,
        tool_name: None,
        tool_replay: None,
        prune_state: PruneState::Intact,
        reasoning_meta: None,
        response_meta: None,
    }
}

fn count_timeline_items(result: &mut UiPerfRunResult, items: &[UiTimelineItem]) {
    let mut user_inputs = 0u64;
    let mut assistant_texts = 0u64;
    let mut reasoning = 0u64;
    let mut activities = 0u64;
    let mut lashlang_code = 0u64;
    for item in items {
        match item {
            UiTimelineItem::UserInput(_) => user_inputs += 1,
            UiTimelineItem::AssistantText(_) => assistant_texts += 1,
            UiTimelineItem::AssistantReasoning(_) => reasoning += 1,
            UiTimelineItem::Activity(_) => activities += 1,
            UiTimelineItem::LashlangCode(_) => lashlang_code += 1,
            UiTimelineItem::TurnStart(_)
            | UiTimelineItem::ShellOutput { .. }
            | UiTimelineItem::Error(_)
            | UiTimelineItem::SystemMessage(_)
            | UiTimelineItem::PluginPanel(_)
            | UiTimelineItem::Splash => {}
        }
    }
    result.counter("timeline_user_inputs", user_inputs);
    result.counter("timeline_assistant_texts", assistant_texts);
    result.counter("timeline_reasoning_blocks", reasoning);
    result.counter("timeline_activity_blocks", activities);
    result.counter("timeline_lashlang_code_blocks", lashlang_code);
}
