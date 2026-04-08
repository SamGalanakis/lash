mod activity;
mod artifact;
mod prompt;
mod queue;
#[cfg(test)]
mod tests;

use lash_tui::{Line, Modifier, Rect, Span, Style};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, PatchFilePreview,
    QuestionPanelArtifact, QuestionPanelSelectionMode, SnippetPreviewArtifact, SnippetRenderMode,
    patch_file_subject, patch_status_title,
};
use crate::app::{
    App, DisplayBlock, PreparedTurn, PromptState, SPLASH_CONTENT_HEIGHT, SPLASH_SCROLLBACK_HEIGHT,
    preview_text_lines,
};
use crate::assistant_text;
use crate::diff::render_inline_diff;
use crate::input_items::image_marker_ranges;
use crate::markdown;
use crate::text_display;
use crate::text_layout;
use crate::theme;

use self::activity::render_activity_block;
use self::artifact::{
    render_plan_block, render_question_panel_artifact, render_section_panel_block,
};
use self::prompt::prompt_height;

pub(crate) use self::prompt::prompt_content_lines_snapshot;
pub(crate) use self::queue::queue_preview_lines_snapshot;

const INPUT_HORIZONTAL_PADDING: u16 = 1;
const PROMPT_HORIZONTAL_PADDING: u16 = 1;
const MIN_HISTORY_HEIGHT: u16 = 3;
const MAX_INPUT_HEIGHT: u16 = 20;
const COMPACT_ACTIVITY_FEED_MAX_ITEMS: usize = 5;
const COMPACT_ACTIVITY_FEED_MAX_ROWS_PER_ITEM: usize = 2;
const COMPACT_PATCH_PREVIEW_MAX_FILES: usize = 5;
const STREAMING_OUTPUT_INLINE_MAX_ROWS: usize = 4;
const SCROLL_INDICATOR_HIDE_TAIL_ROWS: usize = 2;
const SCROLL_INDICATOR_MIN_HEIGHT: usize = 2;
const QUEUE_SECTION_ITEM_LIMIT: usize = 2;
const QUEUE_SECTION_WRAP_LIMIT: usize = 2;

pub(crate) struct InputRenderSnapshot {
    pub lines: Vec<Line<'static>>,
    pub cursor: (u16, u16),
    pub scroll_offset: usize,
    pub badge: Option<Line<'static>>,
}

fn padded_content_width(frame_width: u16, horizontal_padding: u16) -> usize {
    frame_width.saturating_sub(horizontal_padding.saturating_mul(2)) as usize
}

fn prompt_inner_width(frame_width: u16) -> usize {
    frame_width.saturating_sub(PROMPT_HORIZONTAL_PADDING.saturating_mul(2)) as usize
}

fn desired_input_height(app: &App, frame_width: u16) -> u16 {
    if app.has_prompt() {
        prompt_height(app, frame_width)
    } else {
        let inner_w = padded_content_width(frame_width, INPUT_HORIZONTAL_PADDING);
        let visual_lines = input_visual_lines(app.input(), inner_w);
        (visual_lines as u16 + 2).min(MAX_INPUT_HEIGHT)
    }
}

fn input_height(app: &App, frame_width: u16, frame_height: u16, reserved_height: u16) -> u16 {
    let desired = desired_input_height(app, frame_width);
    let max_allowed = frame_height
        .saturating_sub(reserved_height)
        .saturating_sub(MIN_HISTORY_HEIGHT);
    desired.min(max_allowed)
}

fn queue_preview_height(app: &App, frame_width: u16) -> u16 {
    queue_preview_lines_snapshot(app, frame_width).len() as u16
}

fn turn_status_height(app: &App) -> u16 {
    if app.live_turn.is_some() { 1 } else { 0 }
}

pub fn history_viewport_height(app: &App, frame_width: u16, frame_height: u16) -> usize {
    let status_h = turn_status_height(app);
    let queued_h = queue_preview_height(app, frame_width);
    let reserved_height = 1 + status_h + queued_h;
    let input_h = input_height(app, frame_width, frame_height, reserved_height);
    let overhead = 1 + status_h + queued_h + input_h;
    frame_height.saturating_sub(overhead) as usize
}

pub fn history_area(app: &App, frame_width: u16, frame_height: u16) -> Rect {
    let status_h = turn_status_height(app);
    let queued_h = queue_preview_height(app, frame_width);
    let reserved_height = 1 + status_h + queued_h;
    let input_h = input_height(app, frame_width, frame_height, reserved_height);
    let history_height = frame_height.saturating_sub(1 + status_h + queued_h + input_h);
    Rect::new(0, 1, frame_width, history_height)
}

#[cfg(test)]
pub fn rendered_block_height(
    blocks: &[DisplayBlock],
    idx: usize,
    expand_level: u8,
    viewport_width: usize,
    viewport_height: usize,
) -> usize {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks = blocks.to_vec();
    app.expand_level = expand_level;
    render_block_lines(&app, idx, viewport_width, viewport_height).len()
}

pub(crate) fn extract_history_selection_text(
    app: &App,
    frame_width: u16,
    frame_height: u16,
) -> Option<String> {
    if !(app.selection.active || app.selection.visible) {
        return None;
    }

    let history = history_area(app, frame_width, frame_height);
    if history.width == 0 || history.height == 0 {
        return None;
    }

    let lines =
        history_content_lines_snapshot(app, history.width as usize, history.height as usize);
    let ((start_col, start_row), (end_col, end_row)) = selection_ordered(&app.selection);
    let mut out = String::new();

    for row in start_row..=end_row {
        if !out.is_empty() {
            out.push('\n');
        }
        let local_start = if row == start_row {
            start_col.saturating_sub(history.x) as usize
        } else {
            0
        };
        let local_end = if row == end_row {
            end_col.saturating_sub(history.x) as usize
        } else {
            history.width as usize
        };
        if local_end <= local_start {
            continue;
        }
        if let Some(line) = lines.get(row) {
            out.push_str(line_slice_by_display_columns(line, local_start, local_end).trim_end());
        }
    }

    if out.is_empty() { None } else { Some(out) }
}

pub(crate) fn input_render_snapshot(app: &App, area: Rect) -> InputRenderSnapshot {
    let mut lines = Vec::new();
    let image_markers = image_marker_ranges(app.input());
    let total_width = padded_content_width(area.width, INPUT_HORIZONTAL_PADDING);
    let prefix_w = 2;

    for (i, logical_line) in app.input().split('\n').enumerate() {
        let segments = wrap_line(logical_line, prefix_w, prefix_w, total_width);
        for (j, &(seg_start, seg_end)) in segments.iter().enumerate() {
            let seg_spans = styled_input_segment(logical_line, seg_start, seg_end, &image_markers);
            let prefix = if i == 0 && j == 0 {
                Span::styled(format!("{} ", theme::PROMPT_CHAR), theme::prompt())
            } else {
                Span::styled("  ", Style::default().fg(theme::ASH))
            };
            let mut spans = vec![prefix];
            spans.extend(seg_spans);
            lines.push(Line::from(spans));
        }
    }

    let clamped_cursor = app.cursor_pos().min(app.input().len());
    let (cursor_row, cursor_col) = input_cursor_position(app.input(), clamped_cursor, total_width);
    let content_h = area.height.saturating_sub(2) as usize;
    let scroll_offset = if content_h > 0 && cursor_row >= content_h {
        cursor_row - content_h + 1
    } else {
        0
    };

    InputRenderSnapshot {
        lines,
        cursor: (cursor_col as u16, (cursor_row - scroll_offset) as u16),
        scroll_offset,
        badge: build_input_badge(app),
    }
}

pub(crate) fn find_visible_block(app: &App, scroll_offset: usize) -> (usize, usize) {
    if app.height_cache_snapshot().is_empty() {
        return (0, 0);
    }
    let cache = app.height_cache_snapshot();
    let idx = cache.partition_point(|&cumulative| cumulative <= scroll_offset);
    if idx >= app.blocks.len() {
        return (app.blocks.len(), 0);
    }
    let block_start = if idx == 0 { 0 } else { cache[idx - 1] };
    (idx, scroll_offset - block_start)
}

fn history_content_lines_snapshot(
    app: &App,
    viewport_width: usize,
    viewport_height: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for idx in 0..app.blocks.len() {
        lines.extend(render_block_lines(
            app,
            idx,
            viewport_width,
            viewport_height,
        ));
    }
    if let Some(live_lines) = app.live_assistant_lines_snapshot() {
        if app.live_assistant_leading_padding() > 0 {
            lines.push(Line::from(""));
        }
        lines.extend(live_lines.iter().cloned());
    }
    lines
}

fn line_slice_by_display_columns(line: &Line<'_>, start: usize, end: usize) -> String {
    let mut out = String::new();
    let mut col = 0usize;
    for span in &line.spans {
        for ch in span.content.chars() {
            let width = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
            let next = col + width;
            if next <= start {
                col = next;
                continue;
            }
            if col >= end {
                return out;
            }
            if next > start && col < end {
                out.push(ch);
            }
            col = next;
        }
    }
    out
}

fn selection_ordered(sel: &crate::app::TextSelection) -> ((u16, usize), (u16, usize)) {
    let (ax, ay) = sel.anchor;
    let (ex, ey) = sel.end;
    if ay < ey || (ay == ey && ax <= ex) {
        ((ax, ay), (ex, ey))
    } else {
        ((ex, ey), (ax, ay))
    }
}

fn truncate_to_display_width(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if text_display::visible_width(text) <= max_width {
        return text.to_string();
    }
    let target = max_width.saturating_sub(1);
    let mut out = String::new();
    let mut width = 0usize;
    for ch in text.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + w > target {
            break;
        }
        out.push(ch);
        width += w;
    }
    out.push('…');
    out
}

fn wrap_line(
    text: &str,
    first_prefix_width: usize,
    continuation_prefix_width: usize,
    total_width: usize,
) -> Vec<(usize, usize)> {
    if total_width == 0 {
        return vec![(0, text.len())];
    }
    let first_cap = total_width.saturating_sub(first_prefix_width).max(1);
    let continuation_cap = total_width.saturating_sub(continuation_prefix_width).max(1);
    let mut result = Vec::new();
    let mut line_start = 0usize;
    let mut col = 0usize;
    let mut capacity = first_cap;

    for (byte_idx, ch) in text.char_indices() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if col + w > capacity && col > 0 {
            result.push((line_start, byte_idx));
            line_start = byte_idx;
            col = w;
            capacity = continuation_cap;
        } else {
            col += w;
        }
    }
    result.push((line_start, text.len()));
    result
}

fn input_visual_lines(input: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let prefix_w = 2;
    input
        .split('\n')
        .map(|line| wrap_line(line, prefix_w, prefix_w, width).len())
        .sum::<usize>()
        .max(1)
}

fn input_cursor_position(input: &str, cursor_pos: usize, full_width: usize) -> (usize, usize) {
    let prefix_w = 2usize;
    let mut vis_row = 0usize;
    let mut byte_offset = 0usize;

    for logical_line in input.split('\n') {
        let line_end = byte_offset + logical_line.len();
        let segments = wrap_line(logical_line, prefix_w, prefix_w, full_width);

        if cursor_pos <= line_end {
            let cursor_in_line = cursor_pos - byte_offset;
            for (i, &(seg_start, seg_end)) in segments.iter().enumerate() {
                let is_last = i == segments.len() - 1;
                if cursor_in_line >= seg_start && (cursor_in_line < seg_end || is_last) {
                    let text_before = &logical_line[seg_start..cursor_in_line];
                    return (vis_row, UnicodeWidthStr::width(text_before) + prefix_w);
                }
                vis_row += 1;
            }
            return (vis_row, prefix_w);
        }

        vis_row += segments.len();
        byte_offset = line_end + 1;
    }

    (vis_row.saturating_sub(1), prefix_w)
}

fn build_input_badge(app: &App) -> Option<Line<'static>> {
    let badge_labels: Vec<&str> = app
        .plugin_mode_indicators
        .values()
        .map(|label| label.as_str())
        .collect();
    let mut spans = Vec::new();
    if !badge_labels.is_empty() {
        for (idx, label) in badge_labels.iter().enumerate() {
            if idx > 0 {
                spans.push(Span::styled(" · ", Style::default().fg(theme::ASH)));
            }
            spans.push(Span::styled(
                (*label).to_string(),
                Style::default()
                    .fg(theme::SODIUM)
                    .add_modifier(Modifier::Bold),
            ));
        }
        spans.push(Span::styled(" · ", Style::default().fg(theme::ASH)));
    }

    let location_label = app
        .repo_status
        .as_ref()
        .map(|repo| format!("{} · {}", repo.repo_name, repo.display_ref()))
        .unwrap_or_else(|| app.cwd.clone());
    spans.push(text_display::sanitize_span(
        location_label,
        Style::default().fg(theme::ASH_TEXT),
    ));
    Some(Line::from(spans))
}

fn styled_input_segment(
    logical_line: &str,
    seg_start: usize,
    seg_end: usize,
    image_markers: &[(std::ops::Range<usize>, usize)],
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut cursor = seg_start;

    for (range, _) in image_markers {
        if range.end <= seg_start || range.start >= seg_end {
            continue;
        }
        let clamped_start = range.start.max(seg_start);
        let clamped_end = range.end.min(seg_end);
        if cursor < clamped_start {
            spans.push(Span::styled(
                logical_line[cursor..clamped_start].to_string(),
                theme::user_input(),
            ));
        }
        spans.push(Span::styled(
            logical_line[clamped_start..clamped_end].to_string(),
            theme::image_marker(),
        ));
        cursor = clamped_end;
    }

    if cursor < seg_end {
        spans.push(Span::styled(
            logical_line[cursor..seg_end].to_string(),
            theme::user_input(),
        ));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), theme::user_input()));
    }
    spans
}

fn styled_user_input_segment(text: &str, seg_start: usize, seg_end: usize) -> Vec<Span<'static>> {
    vec![Span::styled(
        text[seg_start..seg_end].to_string(),
        theme::user_input(),
    )]
}

fn build_code_fold_summary(blocks: &[DisplayBlock], idx: usize) -> String {
    let mut block_count = 0usize;
    let mut line_count = 0usize;

    for block in &blocks[idx..] {
        match block {
            DisplayBlock::CodeBlock { code, .. } => {
                block_count += 1;
                line_count += code.lines().count().max(1);
            }
            DisplayBlock::Activity(_) | DisplayBlock::CodeOutput { .. } => continue,
            _ => break,
        }
    }

    format!(
        "▶ {} code block{} · {} line{}",
        block_count,
        if block_count == 1 { "" } else { "s" },
        line_count,
        if line_count == 1 { "" } else { "s" }
    )
}

pub(crate) fn render_block_lines(
    app: &App,
    idx: usize,
    viewport_width: usize,
    viewport_height: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    render_block_into(app, idx, &mut lines, viewport_width, viewport_height);
    lines
}

#[cfg(test)]
fn render_block(
    blocks: &[DisplayBlock],
    idx: usize,
    expand_level: u8,
    viewport_width: usize,
    viewport_height: usize,
) -> Vec<Line<'static>> {
    let mut app = App::new("test-model".into(), "test".into());
    app.blocks = blocks.to_vec();
    app.expand_level = expand_level;
    render_block_lines(&app, idx, viewport_width, viewport_height)
}

fn render_block_into(
    app: &App,
    idx: usize,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    viewport_height: usize,
) {
    let blocks = &app.blocks;
    let expand_level = app.expand_level;
    match &blocks[idx] {
        DisplayBlock::UserInput(text) => {
            if idx > 0 && !matches!(blocks[idx - 1], DisplayBlock::Splash) {
                let rule_width = (viewport_width * 2 / 5).max(8).min(viewport_width);
                let pad_left = (viewport_width.saturating_sub(rule_width)) / 2;
                lines.push(Line::from(""));
                let mut spans = Vec::new();
                if pad_left > 0 {
                    spans.push(Span::raw(" ".repeat(pad_left)));
                }
                spans.push(Span::styled(
                    "─".repeat(rule_width),
                    theme::turn_separator(),
                ));
                lines.push(Line::from(spans));
            }

            let marker_style = Style::default().fg(theme::SODIUM);
            let prefix_w = 2;
            let cap = viewport_width.saturating_sub(prefix_w);
            let mut first = true;
            for line in text.lines() {
                let wrapped = if cap == 0 || line.is_empty() {
                    vec![(0, line.len())]
                } else {
                    text_layout::wrap_text_ranges_wordwise(line, cap)
                };
                for (seg_start, seg_end) in wrapped {
                    let prefix = if first {
                        Span::styled("● ", marker_style)
                    } else {
                        Span::raw("  ")
                    };
                    first = false;
                    let mut spans = vec![prefix];
                    spans.extend(styled_user_input_segment(line, seg_start, seg_end));
                    lines.push(Line::from(spans));
                }
            }
        }
        DisplayBlock::AssistantText(text) => {
            let add_spacing_before = idx > 0
                && !matches!(
                    blocks[idx - 1],
                    DisplayBlock::AssistantText(_) | DisplayBlock::Splash
                );
            lines.extend(assistant_text::render_assistant_text_block(
                text,
                viewport_width,
                add_spacing_before,
            ));
        }
        DisplayBlock::CodeBlock { code, continuation } => match expand_level {
            0 => {
                if !*continuation {
                    lines.push(Line::from(Span::styled(
                        truncate_to_display_width(
                            &build_code_fold_summary(blocks, idx),
                            viewport_width,
                        ),
                        theme::code_header(),
                    )));
                }
            }
            1 => {}
            _ => {
                for line in code.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("│ ", theme::code_scribe()),
                        Span::styled(line.to_string(), theme::code_content()),
                    ]));
                }
            }
        },
        DisplayBlock::Activity(activity) => {
            render_activity_block(activity, expand_level, lines, viewport_width);
            if app.live_tool_output_anchor_block_index() == Some(idx) {
                render_live_tool_output_inline(lines, app, &activity.kind, viewport_width);
            }
        }
        DisplayBlock::CodeOutput { output, error } => {
            if expand_level >= 2 && !output.is_empty() {
                for line in output.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("│ ", theme::code_chrome()),
                        Span::styled(line.to_string(), theme::system_output()),
                    ]));
                }
            }
            if let Some(err) = error {
                let summary = err
                    .lines()
                    .find(|line| {
                        !line.trim().is_empty()
                            && !line.trim_start().starts_with("File ")
                            && !line.trim_start().starts_with("Traceback")
                    })
                    .unwrap_or_else(|| err.lines().next().unwrap_or("error"));
                lines.push(Line::from(Span::styled(
                    format!("✖ Execution failed: {}", summary.trim()),
                    theme::error(),
                )));
                if expand_level >= 2 {
                    for line in err.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("│ ", theme::code_chrome()),
                            Span::styled(line.to_string(), theme::error()),
                        ]));
                    }
                }
            }
        }
        DisplayBlock::ShellOutput {
            command,
            output,
            error,
        } => {
            lines.push(Line::from(Span::styled(
                format!("$ {command}"),
                theme::code_chrome(),
            )));
            if expand_level >= 2 {
                for line in output.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("│ ", theme::code_chrome()),
                        Span::styled(line.to_string(), theme::system_output()),
                    ]));
                }
                if let Some(err) = error {
                    for line in err.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("│ ", theme::code_chrome()),
                            Span::styled(line.to_string(), theme::error()),
                        ]));
                    }
                }
            }
        }
        DisplayBlock::Error(message) => render_bordered_text_block(
            "ERROR",
            message.lines().map(str::to_string).collect(),
            lines,
            viewport_width,
            theme::error_border(),
            theme::error_title(),
            theme::error(),
        ),
        DisplayBlock::SystemMessage(text) => {
            for line in text.lines() {
                lines.push(Line::from(Span::styled(
                    line.to_string(),
                    theme::system_message(),
                )));
            }
        }
        DisplayBlock::PlanContent(content) => render_plan_block(content, lines, viewport_width),
        DisplayBlock::PluginPanel(panel) => {
            render_section_panel_block(&panel.title, &panel.content, lines, viewport_width);
        }
        DisplayBlock::Splash => {
            render_splash(lines, viewport_width, viewport_height, blocks.len() == 1)
        }
    }
}

fn render_bordered_text_block(
    title_text: &str,
    content_lines: Vec<String>,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    border_style: Style,
    title_style: Style,
    text_style: Style,
) {
    render_bordered_styled_block(
        title_text,
        &content_lines,
        lines,
        viewport_width,
        border_style,
        title_style,
        |chunk| vec![Span::styled(chunk.to_string(), text_style)],
    );
}

fn render_bordered_styled_block<F>(
    title_text: &str,
    content_lines: &[String],
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    border_style: Style,
    title_style: Style,
    mut style_chunk: F,
) where
    F: FnMut(&str) -> Vec<Span<'static>>,
{
    let title = format!(" {title_text} ");
    let title_w = UnicodeWidthStr::width(title.as_str());
    let fill_w = viewport_width.saturating_sub(3 + title_w);
    lines.push(Line::from(vec![
        Span::styled("┌─", border_style),
        Span::styled(title, title_style),
        Span::styled("─".repeat(fill_w), border_style),
        Span::styled("┐", border_style),
    ]));

    let inner_w = viewport_width.saturating_sub(4).max(1);
    for raw_line in content_lines {
        let segments = if raw_line.is_empty() {
            vec![(0usize, 0usize)]
        } else {
            wrap_line(raw_line, 0, 0, inner_w)
        };
        for (start, end) in segments {
            let chunk = if raw_line.is_empty() {
                String::new()
            } else {
                truncate_to_display_width(&raw_line[start..end], inner_w)
            };
            let styled = style_chunk(&chunk);
            let visible_width: usize = styled.iter().map(Span::width).sum();
            let pad = inner_w.saturating_sub(visible_width);
            let mut row = Vec::new();
            row.push(Span::styled("│ ", border_style));
            row.extend(styled);
            row.push(Span::raw(" ".repeat(pad)));
            row.push(Span::styled(" │", border_style));
            lines.push(Line::from(row));
        }
    }

    let bottom_fill = viewport_width.saturating_sub(2);
    lines.push(Line::from(Span::styled(
        format!("└{}┘", "─".repeat(bottom_fill)),
        border_style,
    )));
}

fn render_splash(
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    viewport_height: usize,
    fullscreen: bool,
) {
    let chalk = theme::assistant_text();
    let sodium = Style::default().fg(theme::SODIUM);
    let content_width = 30;
    let content_height = SPLASH_CONTENT_HEIGHT;
    let cx = viewport_width.saturating_sub(content_width) / 2;
    let cy = if fullscreen {
        viewport_height.saturating_sub(content_height) / 2
    } else {
        1
    };
    let pad = " ".repeat(cx);

    for _ in 0..cy {
        lines.push(Line::from(""));
    }

    let logo: &[(&str, &str)] = &[
        ("██       ████   ██████  ", "  ██"),
        ("██      ██  ██  ██     ", "█  ██"),
        ("██      ██████  ██████", "██████"),
        ("██      ██  ██      █", " ██  ██"),
        ("██████  ██  ██  ████", "  ██  ██"),
    ];
    for &(before, after) in logo {
        lines.push(Line::from(vec![
            Span::styled(format!("{pad}{before}"), chalk),
            Span::styled("██", sodium),
            Span::styled(after, chalk),
        ]));
    }
    lines.push(Line::from(Span::styled(
        format!("{pad}                   ██"),
        sodium,
    )));
    lines.push(Line::from(vec![
        Span::styled(format!("{pad}──────────"), sodium),
        Span::styled("──────────", Style::default().fg(theme::ASH_MID)),
        Span::styled("──────────", Style::default().fg(theme::ASH)),
    ]));
    lines.push(Line::from(""));

    let target_height = if fullscreen {
        viewport_height
    } else {
        SPLASH_SCROLLBACK_HEIGHT
    };
    for _ in (cy + content_height)..target_height {
        lines.push(Line::from(""));
    }
}

fn append_streaming_output_lines(
    lines: &mut Vec<Line<'static>>,
    app: &App,
    viewport_width: usize,
    prefix: &str,
    prefix_style: Style,
    content_style: Style,
    row_limit: usize,
) {
    if app.streaming_output_height() == 0 {
        return;
    }

    let mut hidden_rows = app.streaming_output_hidden;
    let mut logical = Vec::with_capacity(app.streaming_output_height());
    logical.extend(app.streaming_output.iter().cloned());
    if !app.streaming_output_partial.is_empty() {
        logical.push(app.streaming_output_partial.clone());
    }

    if row_limit > 0 {
        let visible_tail = row_limit.saturating_sub(1);
        if logical.len() > visible_tail {
            let trimmed = logical.len().saturating_sub(visible_tail);
            hidden_rows += trimmed;
            logical = logical.split_off(trimmed);
        }
    }

    if hidden_rows > 0 {
        logical.insert(0, format!("… {hidden_rows} earlier live rows hidden …"));
    }

    let continuation = " ".repeat(prefix.chars().count());
    for line in logical {
        push_wrapped_prefixed(
            lines,
            prefix.to_string(),
            continuation.clone(),
            &line,
            content_style,
            viewport_width,
        );
        let rendered_rows = text_layout::wrap_text_ranges_wordwise(
            &line,
            viewport_width.saturating_sub(UnicodeWidthStr::width(prefix)),
        )
        .len()
        .max(1)
        .min(lines.len());
        for row in lines.iter_mut().rev().take(rendered_rows) {
            if let Some(first) = row.spans.first_mut() {
                first.style = prefix_style;
            }
        }
    }
}

fn render_live_tool_output_inline(
    lines: &mut Vec<Line<'static>>,
    app: &App,
    activity_kind: &ActivityKind,
    viewport_width: usize,
) {
    if app.streaming_output_height() == 0 || viewport_width == 0 {
        return;
    }

    let (prefix, prefix_style, content_style) = if *activity_kind == ActivityKind::Delegate {
        (
            "    │ ",
            Style::default()
                .fg(theme::SODIUM)
                .add_modifier(Modifier::Bold),
            Style::default().fg(theme::CHALK_MID),
        )
    } else {
        ("    │ ", theme::code_chrome(), theme::system_output())
    };

    append_streaming_output_lines(
        lines,
        app,
        viewport_width,
        prefix,
        prefix_style,
        content_style,
        STREAMING_OUTPUT_INLINE_MAX_ROWS,
    );
}

pub(crate) fn history_scroll_indicator(app: &App, area: Rect) -> Option<(u16, u16, u16)> {
    if area.width == 0 || area.height == 0 {
        return None;
    }
    let viewport_height = area.height as usize;
    let total_content_height = app.height_cache_snapshot().last().copied().unwrap_or(0);
    let max_scroll = total_content_height.saturating_sub(viewport_height);
    if max_scroll == 0 {
        return None;
    }
    let scroll_offset = app.scroll_offset.min(max_scroll);
    if max_scroll.saturating_sub(scroll_offset) <= SCROLL_INDICATOR_HIDE_TAIL_ROWS {
        return None;
    }

    let min_height = if viewport_height >= 4 {
        SCROLL_INDICATOR_MIN_HEIGHT
    } else {
        1
    };
    let height = ((viewport_height * viewport_height).div_ceil(total_content_height))
        .clamp(min_height, viewport_height);
    let travel = viewport_height.saturating_sub(height);
    let y = area.y
        + if travel == 0 {
            0
        } else {
            ((scroll_offset * travel) / max_scroll) as u16
        };
    Some((area.right().saturating_sub(1), y, height as u16))
}

fn push_wrapped_prefixed(
    lines: &mut Vec<Line<'static>>,
    prefix: String,
    continuation: String,
    text: &str,
    style: Style,
    width: usize,
) {
    let available = width.saturating_sub(UnicodeWidthStr::width(prefix.as_str()));
    if available == 0 {
        lines.push(Line::from(Span::styled(prefix, style)));
        return;
    }
    let segments = if text.is_empty() {
        vec![(0usize, 0usize)]
    } else {
        text_layout::wrap_text_ranges_wordwise(text, available)
    };
    for (segment_idx, &(start, end)) in segments.iter().enumerate() {
        let shown_prefix = if segment_idx == 0 {
            prefix.clone()
        } else {
            continuation.clone()
        };
        lines.push(Line::from(vec![
            Span::styled(shown_prefix, style),
            text_display::sanitize_span(text[start..end].to_string(), style),
        ]));
    }
}

fn truncate_with_forced_ellipsis(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if max_width == 1 {
        return "…".to_string();
    }
    let target = max_width.saturating_sub(1);
    let mut out = String::new();
    let mut width = 0usize;
    for ch in text.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + ch_width > target {
            break;
        }
        out.push(ch);
        width += ch_width;
    }
    out.push('…');
    out
}
