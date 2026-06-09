use std::path::PathBuf;
use std::time::Instant;

use lash_export::LoadedSession;
use lash_export::trace::{LlmCallUsage, LlmPromptSnapshot, RequestMessage};

use lash_perf::perf_support::time::elapsed_ms;

use super::measurement::UiPerfRunResult;
use super::projection_cases::build_projection_read_view;
use super::scenarios::UiPerfWorkload;

pub(crate) fn run_html_export_once(workload: UiPerfWorkload) -> UiPerfRunResult {
    let total_started = Instant::now();
    let build_started = Instant::now();
    let read_view = build_projection_read_view(workload.turn_count);
    let chronological = read_view.chronological_projection().into_entries();
    let prompts = export_prompt_snapshots(workload.turn_count);
    let session = LoadedSession {
        meta: None,
        chronological,
        trace_path: PathBuf::from("ui-perf.trace.jsonl"),
        context_window_tokens: Some(200_000),
        llm_prompts: prompts,
    };
    let build_case_ms = elapsed_ms(build_started);

    let mut result = UiPerfRunResult::new(total_started);
    result.build_case_ms = build_case_ms;
    result.sample("build_case_ms", build_case_ms);

    let passes = workload.scroll_passes.max(1);
    let mut rendered = String::new();
    for _ in 0..passes {
        let render_started = Instant::now();
        rendered = lash_export::html::render(&session);
        result.sample("html_export_render_ms", elapsed_ms(render_started));
    }

    result.total_ms = elapsed_ms(total_started);
    result.total_blocks = session.chronological.len();
    result.total_content_rows = rendered.lines().count();
    result.counter("html_export_bytes", rendered.len() as u64);
    result.counter("html_export_prompts", session.llm_prompts.len() as u64);
    result
}

fn export_prompt_snapshots(turn_count: usize) -> Vec<LlmPromptSnapshot> {
    let count = turn_count.div_ceil(5).max(1);
    (0..count)
        .map(|index| {
            let system_text = if index % 4 == 0 {
                "You are lash. Use tools carefully and report concise progress.".to_string()
            } else {
                format!(
                    "You are lash. Export profiling prompt family {}.",
                    index % 4
                )
            };
            let request_text = format!(
                "Synthetic export request {index}. Inspect chronological entries, tool calls, prompt usage, and RLM steps."
            );
            LlmPromptSnapshot {
                session_id: Some("ui-export-perf".to_string()),
                turn_index: Some(index as u64),
                protocol_iteration: Some((index % 3) as u64),
                llm_call_id: Some(format!("ui-export-perf:{index}")),
                caused_by: None,
                timestamp: None,
                model: Some("gpt-5.4".to_string()),
                model_variant: None,
                system_chars: system_text.chars().count(),
                system_hash: format!("system-hash-{}", index % 4),
                system_text,
                message_count: 3,
                total_chars: 180 + request_text.chars().count(),
                request_messages: vec![RequestMessage {
                    role: "user".to_string(),
                    chars: request_text.chars().count(),
                    text: request_text,
                }],
                request_chars: 120,
                request_hash: format!("request-hash-{index}"),
                usage: Some(LlmCallUsage {
                    input_tokens: 3_000 + index as i64 * 25,
                    output_tokens: 150,
                    cached_input_tokens: if index % 2 == 0 { 1_200 } else { 0 },
                    reasoning_tokens: 48,
                    duration_ms: Some(250 + index as u64),
                }),
            }
        })
        .collect()
}
