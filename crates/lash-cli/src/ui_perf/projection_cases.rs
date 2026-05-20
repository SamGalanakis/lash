use std::time::Instant;

use lash_core::{
    Message, MessageRole, Part, PartKind, PruneState, SessionReadView, SessionStateEnvelope,
    ToolCallOutput, ToolCallRecord, ToolCancellation, ToolFailure, ToolFailureClass, ToolResult,
    TurnActivity, TurnEvent,
};
use serde_json::json;

use crate::app::{App, UiProjectionState, UiTimeline, UiTimelineItem, timeline_from_read_view};
use crate::perf_support::time::elapsed_ms;

use super::measurement::UiPerfRunResult;
use super::scenarios::UiPerfWorkload;

pub(crate) fn run_timeline_projection_once(workload: UiPerfWorkload) -> UiPerfRunResult {
    let total_started = Instant::now();
    let build_started = Instant::now();
    let read_view = build_projection_read_view(workload.turn_count);
    let ui_state = UiProjectionState {
        live_assistant_text: Some("live assistant tail that should be reconciled".to_string()),
        live_reasoning_text: Some("live reasoning tail".to_string()),
        ..UiProjectionState::default()
    };
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

fn build_projection_read_view(turn_count: usize) -> SessionReadView {
    let mut graph = lash_core::SessionGraph::default();
    let mut tool_calls = Vec::new();

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

        if turn % 3 == 0 {
            tool_calls.push(tool_record(
                format!("call-read-{turn}"),
                "read_file",
                json!({ "path": format!("crates/lash-cli/src/app/projection-{turn}.rs") }),
                ToolCallOutput::success(
                    json!({ "path": "projection.rs", "content": "fn main() {}" }),
                ),
                3,
            ));
        }

        if turn % 5 == 0 {
            let call_id = format!("call-rlm-{turn}");
            tool_calls.push(tool_record(
                call_id.clone(),
                "exec_command",
                json!({ "cmd": "date -u" }),
                ToolCallOutput::success(json!({ "output": "time\n", "exit_code": 0 })),
                5,
            ));
            graph.append_mode_event(lash_mode_rlm::rlm_mode_event(
                lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(
                    lash_rlm_types::RlmTrajectoryEntry {
                        id: format!("rlm_step_{turn}"),
                        mode_iteration: turn,
                        reasoning: format!(
                            "Check runtime state {turn}.\n\n```lashlang\nsubmit \"ok\"\n```"
                        ),
                        code: "now = (call exec_command { cmd: \"date -u\" })?\nprint now"
                            .to_string(),
                        output: vec!["time".to_string()],
                        tool_call_ids: vec![call_id],
                        images: Vec::new(),
                        error: None,
                        final_output: (turn % 10 == 0).then(|| json!("RLM final output")),
                    },
                ),
            ));
        }
    }

    graph.append_active_read_delta(&[], &tool_calls);
    let state = SessionStateEnvelope {
        session_graph: graph,
        ..SessionStateEnvelope::default()
    };
    SessionReadView::from_exported_state(&state)
}

fn live_activity_event(index: usize) -> TurnEvent {
    match index % 14 {
        0 => TurnEvent::ModelRequestStarted {
            mode_iteration: index / 14,
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
            call_id: Some(format!("monitor-{index}")),
            name: "monitor".to_string(),
            args: json!({ "command": "tail -f app.log", "description": "app log" }),
            output: ToolCallOutput::success(json!({
                "process_id": "monitor:app-log",
                "producer": "monitor",
                "state": "running",
                "description": "app log"
            })),
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
        12 => TurnEvent::SubmittedValue {
            value: json!({ "status": "done", "index": index }),
        },
        _ => TurnEvent::ToolCallCompleted {
            call_id: Some(format!("generic-{index}")),
            name: "search_tools".to_string(),
            args: json!({ "query": "projection tools" }),
            output: ToolResult::err(json!("tool search failed")).into_output(),
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

fn tool_record(
    call_id: String,
    tool: &'static str,
    args: serde_json::Value,
    output: ToolCallOutput,
    duration_ms: u64,
) -> ToolCallRecord {
    ToolCallRecord {
        call_id: Some(call_id),
        tool: tool.to_string(),
        args,
        output,
        duration_ms,
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
