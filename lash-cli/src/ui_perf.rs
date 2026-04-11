use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use chrono::Utc;
use lash::{SkillCatalog, TokenUsage};
use serde::Serialize;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ExplorationOp,
    ExplorationOpKind, SnippetPreviewArtifact, SnippetRenderMode,
};
use crate::app::{App, DisplayBlock, FollowOutputMode, PreparedTurn, TextSelection};
use crate::render;
use crate::ui_trace::render_screen_snapshot;

const BENCH_WIDTH: u16 = 220;
const BENCH_HEIGHT: u16 = 72;
const TURN_COUNT: usize = 480;
const SCROLL_DELTA: usize = 3;
const SCROLL_PASSES: usize = 2;
const SELECTION_SCROLL_DELTA: usize = 2;
const SELECTION_FRAMES: usize = 320;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfRunResult {
    build_case_ms: f64,
    initial_render_ms: f64,
    height_cache_rebuild_ms: f64,
    scroll_render_total_ms: f64,
    scroll_render_avg_ms: f64,
    scroll_render_max_ms: f64,
    scroll_frames: usize,
    selection_render_total_ms: f64,
    selection_render_avg_ms: f64,
    selection_render_max_ms: f64,
    selection_frames: usize,
    total_blocks: usize,
    total_content_rows: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfMetricSummary {
    min: f64,
    median: f64,
    max: f64,
    mean: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfSummary {
    runs: usize,
    build_case_ms: UiPerfMetricSummary,
    initial_render_ms: UiPerfMetricSummary,
    height_cache_rebuild_ms: UiPerfMetricSummary,
    scroll_render_avg_ms: UiPerfMetricSummary,
    scroll_render_max_ms: UiPerfMetricSummary,
    selection_render_avg_ms: UiPerfMetricSummary,
    selection_render_max_ms: UiPerfMetricSummary,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UiPerfReport {
    created_at: String,
    version: String,
    width: u16,
    height: u16,
    turn_count: usize,
    scroll_delta: usize,
    scroll_passes: usize,
    selection_frames: usize,
    warmups: usize,
    runs: usize,
    results: Vec<UiPerfRunResult>,
    summary: UiPerfSummary,
}

pub(crate) fn default_output_path() -> PathBuf {
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("lash-cli crate should live under repo root")
        .join(".benchmarks")
        .join("ui-perf")
        .join(format!("{stamp}.json"))
}

pub(crate) fn run_cli(
    out: Option<PathBuf>,
    runs: usize,
    warmups: usize,
    version: &str,
) -> anyhow::Result<()> {
    let runs = runs.max(1);

    for _ in 0..warmups {
        let _ = run_once();
    }

    let mut results = Vec::with_capacity(runs);
    for _ in 0..runs {
        results.push(run_once());
    }

    let report = UiPerfReport {
        created_at: Utc::now().to_rfc3339(),
        version: version.to_string(),
        width: BENCH_WIDTH,
        height: BENCH_HEIGHT,
        turn_count: TURN_COUNT,
        scroll_delta: SCROLL_DELTA,
        scroll_passes: SCROLL_PASSES,
        selection_frames: SELECTION_FRAMES,
        warmups,
        runs,
        summary: summarize(&results),
        results,
    };

    let out_path = out.unwrap_or_else(default_output_path);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create benchmark output dir {}", parent.display()))?;
    }
    fs::write(&out_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write benchmark report {}", out_path.display()))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "out": out_path,
            "summary": report.summary,
        }))?
    );
    Ok(())
}

fn run_once() -> UiPerfRunResult {
    let started = Instant::now();
    let mut app = build_benchmark_app();
    let build_case_ms = elapsed_ms(started);

    let initial_render_started = Instant::now();
    let _ = render_screen_snapshot(&mut app, BENCH_WIDTH, BENCH_HEIGHT);
    let initial_render_ms = elapsed_ms(initial_render_started);

    let history_height = render::history_viewport_height(&app, BENCH_WIDTH, BENCH_HEIGHT);
    let history_width = render::history_area(&app, BENCH_WIDTH, BENCH_HEIGHT).width as usize;

    app.invalidate_height_cache();
    let height_cache_started = Instant::now();
    let total_content_rows = app.total_content_height(history_width, history_height);
    let height_cache_rebuild_ms = elapsed_ms(height_cache_started);

    let mut scroll_frame_durations = Vec::new();
    app.scroll_offset = 0;
    app.follow_mode = FollowOutputMode::Paused;
    for _ in 0..SCROLL_PASSES {
        while app.scroll_offset + history_height < total_content_rows {
            let frame_started = Instant::now();
            app.scroll_down(SCROLL_DELTA, history_height, history_width);
            let _ = render_screen_snapshot(&mut app, BENCH_WIDTH, BENCH_HEIGHT);
            scroll_frame_durations.push(elapsed_ms(frame_started));
        }
        while app.scroll_offset > 0 {
            let frame_started = Instant::now();
            app.scroll_up(SCROLL_DELTA);
            let _ = render_screen_snapshot(&mut app, BENCH_WIDTH, BENCH_HEIGHT);
            scroll_frame_durations.push(elapsed_ms(frame_started));
        }
    }

    let selection_end_row = total_content_rows.saturating_sub(2);
    app.selection = TextSelection {
        anchor: (0, history_height / 2),
        end: (BENCH_WIDTH.saturating_sub(2), selection_end_row),
        active: false,
        visible: true,
    };
    app.scroll_offset = 0;
    let mut selection_frame_durations = Vec::new();
    for _ in 0..SELECTION_FRAMES {
        let frame_started = Instant::now();
        app.scroll_down(SELECTION_SCROLL_DELTA, history_height, history_width);
        let _ = render_screen_snapshot(&mut app, BENCH_WIDTH, BENCH_HEIGHT);
        selection_frame_durations.push(elapsed_ms(frame_started));
        if app.scroll_offset + history_height >= total_content_rows {
            app.scroll_offset = 0;
        }
    }

    UiPerfRunResult {
        build_case_ms,
        initial_render_ms,
        height_cache_rebuild_ms,
        scroll_render_total_ms: scroll_frame_durations.iter().sum(),
        scroll_render_avg_ms: average(&scroll_frame_durations),
        scroll_render_max_ms: max_value(&scroll_frame_durations),
        scroll_frames: scroll_frame_durations.len(),
        selection_render_total_ms: selection_frame_durations.iter().sum(),
        selection_render_avg_ms: average(&selection_frame_durations),
        selection_render_max_ms: max_value(&selection_frame_durations),
        selection_frames: selection_frame_durations.len(),
        total_blocks: app.blocks.len(),
        total_content_rows,
    }
}

fn build_benchmark_app() -> App {
    let mut app = App::new("gpt-5.4".to_string(), "ui-perf".to_string());
    app.blocks.clear();
    app.token_usage = TokenUsage {
        input_tokens: 208_000,
        output_tokens: 11_500,
        cached_input_tokens: 0,
        reasoning_tokens: 0,
    };
    app.context_window = Some(1_100_000);
    app.model_variant = Some("high".to_string());

    let skills = SkillCatalog::default();
    for turn in 0..TURN_COUNT {
        let turn_label = format!("turn-{turn}");
        let turn = PreparedTurn::prepare_with_large_pastes(
            format!(
                "Investigate render-path regression batch {turn} and summarize the visible state."
            ),
            Vec::new(),
            &skills,
            Vec::new(),
        );
        app.push_prepared_user_input(&turn);
        app.blocks
            .push(DisplayBlock::AssistantText(long_assistant_text(
                turn.preview().as_str(),
            )));
        if turn.draft_id.is_empty() {
            unreachable!("prepared turns should always have a draft id");
        }
        app.blocks
            .push(DisplayBlock::Activity(Box::new(exploration_activity(
                turn.preview().as_str(),
                turn.display_text.as_str(),
            ))));
        if turn.display_text.len().is_multiple_of(3) {
            app.blocks
                .push(DisplayBlock::Activity(Box::new(snippet_activity(
                    &turn_label,
                    false,
                ))));
        }
        if turn.display_text.len().is_multiple_of(5) {
            app.blocks
                .push(DisplayBlock::Activity(Box::new(snippet_activity(
                    &turn_label,
                    true,
                ))));
        }
    }
    app.invalidate_height_cache();
    app
}

fn long_assistant_text(subject: &str) -> String {
    format!(
        "I traced the live render path for {subject} and narrowed the current cost centers.\n\n\
- The history viewport still projects and wraps visible rows on demand.\n\
- Scroll-heavy sessions amplify any repeated line shaping work.\n\
- Selection highlighting compounds that when wide spans are repainted cell by cell.\n\n\
Next I’m using the synthetic UI benchmark workload to lock those costs down and verify each simplification against the same scroll/render path.\n\n\
### Current assessment\n\
- The compact history feed is stable but still layout-heavy.\n\
- Snippet previews and markdown sections are the best stress case for repeated shaping.\n\
- The next pass should reuse block layouts instead of regenerating them on scroll."
    )
}

fn exploration_activity(subject: &str, detail_seed: &str) -> ActivityBlock {
    ActivityBlock::new(
        ActivityKind::Exploration,
        "grep",
        serde_json::json!({}),
        "Explored",
        ActivityStatus::Completed,
        serde_json::json!({}),
        13,
    )
    .with_detail_lines(vec![
        format!("Search \"render cache|height cache|selection\" in {detail_seed}"),
        format!("Read src/render/mod.rs for {subject}"),
        "Read src/app/view.rs for cumulative height math".to_string(),
        "Read src/scratch_tui.rs for selection painting".to_string(),
    ])
    .with_extra(Some(crate::activity::ActivityExtra::Exploration(vec![
        ExplorationOp {
            kind: ExplorationOpKind::Search,
            subject: subject.to_string(),
        },
        ExplorationOp {
            kind: ExplorationOpKind::Read,
            subject: "src/render/mod.rs".to_string(),
        },
        ExplorationOp {
            kind: ExplorationOpKind::Read,
            subject: "src/app/view.rs".to_string(),
        },
        ExplorationOp {
            kind: ExplorationOpKind::Read,
            subject: "src/scratch_tui.rs".to_string(),
        },
    ])))
}

fn snippet_activity(subject: &str, markdown: bool) -> ActivityBlock {
    let (content, render_mode, language) = if markdown {
        (
            "## Render cache checklist\n\n- Cache rendered block lines by width and expand level.\n- Make height cache read lengths from that shared render cache.\n- Stop rebuilding wrapped markdown lines on every scroll tick.\n".to_string(),
            SnippetRenderMode::Markdown,
            None,
        )
    } else {
        (
            "pub(crate) fn draw_history(frame: &mut Frame<'_>, app: &mut App, area: Rect) {\n    // synthetic snippet payload for UI perf benchmark\n    // the real benchmark exercises the actual renderer and wrapping path\n    // with large code and markdown panels visible in the viewport\n    let viewport_height = area.height as usize;\n    let viewport_width = area.width as usize;\n    let _ = (viewport_height, viewport_width);\n}\n".to_string(),
            SnippetRenderMode::Code,
            Some("rust".to_string()),
        )
    };
    ActivityBlock::new(
        ActivityKind::GenericTool,
        "show_snippet_to_user",
        serde_json::json!({}),
        format!("show render/mod.rs:120-164 to user for {subject}"),
        ActivityStatus::Completed,
        serde_json::json!({}),
        7,
    )
    .with_detail_lines(vec![
        "show lash-cli/src/render/mod.rs:120-164 to user".to_string(),
    ])
    .with_artifact(Some(ActivityArtifact::SnippetPreview(
        SnippetPreviewArtifact {
            title: Some("Render cache candidate".to_string()),
            path: "lash-cli/src/render/mod.rs".to_string(),
            start_line: 120,
            end_line: 164,
            content,
            render_mode,
            language,
        },
    )))
}

fn summarize(results: &[UiPerfRunResult]) -> UiPerfSummary {
    UiPerfSummary {
        runs: results.len(),
        build_case_ms: metric_summary(results.iter().map(|run| run.build_case_ms).collect()),
        initial_render_ms: metric_summary(
            results.iter().map(|run| run.initial_render_ms).collect(),
        ),
        height_cache_rebuild_ms: metric_summary(
            results
                .iter()
                .map(|run| run.height_cache_rebuild_ms)
                .collect(),
        ),
        scroll_render_avg_ms: metric_summary(
            results.iter().map(|run| run.scroll_render_avg_ms).collect(),
        ),
        scroll_render_max_ms: metric_summary(
            results.iter().map(|run| run.scroll_render_max_ms).collect(),
        ),
        selection_render_avg_ms: metric_summary(
            results
                .iter()
                .map(|run| run.selection_render_avg_ms)
                .collect(),
        ),
        selection_render_max_ms: metric_summary(
            results
                .iter()
                .map(|run| run.selection_render_max_ms)
                .collect(),
        ),
    }
}

fn metric_summary(mut values: Vec<f64>) -> UiPerfMetricSummary {
    values.sort_by(f64::total_cmp);
    let median = if values.len().is_multiple_of(2) {
        let upper = values.len() / 2;
        (values[upper - 1] + values[upper]) / 2.0
    } else {
        values[values.len() / 2]
    };
    UiPerfMetricSummary {
        min: round3(*values.first().unwrap_or(&0.0)),
        median: round3(median),
        max: round3(*values.last().unwrap_or(&0.0)),
        mean: round3(values.iter().sum::<f64>() / values.len().max(1) as f64),
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    round3(started.elapsed().as_secs_f64() * 1000.0)
}

fn average(values: &[f64]) -> f64 {
    round3(values.iter().sum::<f64>() / values.len().max(1) as f64)
}

fn max_value(values: &[f64]) -> f64 {
    round3(values.iter().copied().fold(0.0, f64::max))
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_ui_perf_benchmark_produces_consistent_shape() {
        let run = run_once();
        assert!(run.total_blocks >= TURN_COUNT * 2);
        assert!(run.total_content_rows > BENCH_HEIGHT as usize);
        assert!(run.scroll_frames > 0);
        assert!(run.selection_frames == SELECTION_FRAMES);
    }
}
