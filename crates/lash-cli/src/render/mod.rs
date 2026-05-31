mod activity;
mod artifact;
mod prompt;
mod queue;
#[cfg(test)]
mod tests;

use crate::SkillCatalog;
use crate::cli_support::selection_ordered;
use crate::skill_prompt::collect_skill_mentions_with_ranges;
use lash_tui::{Line, Modifier, Rect, Span, Style};
use lash_tui_extensions::TuiSurfaceSlot;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, PatchFilePreview,
    QuestionPanelArtifact, QuestionPanelSelectionMode, SnippetPreviewArtifact, SnippetRenderMode,
    patch_file_subject, patch_status_title,
};
use crate::app::{
    App, PreparedTurn, PromptState, SPLASH_CONTENT_HEIGHT, SPLASH_SCROLLBACK_HEIGHT,
    UiTimelineItem, preview_text_lines,
};
use crate::assistant_text;
use crate::diff::render_inline_diff;
use crate::editor::EditorState;
use crate::input_items::image_marker_ranges;
use crate::markdown;
use crate::text_display;
use crate::text_layout;
use crate::theme;

use self::activity::render_activity_block;
use self::artifact::{render_question_panel_artifact, render_section_panel_block};
use self::prompt::prompt_height;

pub(crate) use self::prompt::prompt_max_scroll;
pub(crate) use self::prompt::prompt_render_snapshot;
pub(crate) use self::queue::queue_preview_lines_snapshot;

const INPUT_HORIZONTAL_PADDING: u16 = 1;
const PROMPT_HORIZONTAL_PADDING: u16 = 1;
const MIN_HISTORY_HEIGHT: u16 = 3;
const MAX_INPUT_HEIGHT: u16 = 10;
const COMPACT_ACTIVITY_FEED_MAX_ITEMS: usize = 10;
const COMPACT_ACTIVITY_FEED_MAX_ROWS_PER_ITEM: usize = 2;
const COMPACT_PATCH_PREVIEW_MAX_FILES: usize = 5;
const STREAMING_OUTPUT_INLINE_MAX_ROWS: usize = 4;
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
        prompt_height(app, frame_width, u16::MAX)
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

/// Rows the plan checklist contributes to the trailing end of the
/// transcript: 1 blank gutter + N items. Drops the `PLAN` header and
/// the scribe rule that were dock-chrome when the panel was pinned —
/// inline in scrollable history neither earns its row.
pub fn plan_dock_trailing_height(app: &App) -> usize {
    match app.plan_dock.as_ref() {
        Some(plan) if !plan.is_empty() => 1 + plan.items.len(),
        _ => 0,
    }
}

pub fn process_trailing_height(app: &App) -> usize {
    usize::from(!app.processes.is_empty()) + app.processes.len()
}

#[derive(Clone, Copy, Debug)]
struct ChromeLayout {
    history_height: u16,
    dock_height: u16,
    queue_height: u16,
    footer_height: u16,
    input_height: u16,
}

fn chrome_layout(app: &App, frame_width: u16, frame_height: u16) -> ChromeLayout {
    let surfaces = app.ui_extensions().surface_scene();
    let queue_height = queue_preview_height(app, frame_width);
    let footer_available = frame_height
        .saturating_sub(1 + queue_height)
        .saturating_sub(MIN_HISTORY_HEIGHT);
    let footer_height = surfaces.stack_height(TuiSurfaceSlot::Footer, footer_available);
    let dock_available = frame_height
        .saturating_sub(1 + queue_height + footer_height)
        .saturating_sub(MIN_HISTORY_HEIGHT);
    let dock_height = surfaces.stack_height(TuiSurfaceSlot::Dock, dock_available);
    let reserved_height = 1 + dock_height + queue_height + footer_height;
    let input_height = input_height(app, frame_width, frame_height, reserved_height);
    let history_height =
        frame_height.saturating_sub(1 + dock_height + queue_height + footer_height + input_height);
    ChromeLayout {
        history_height,
        dock_height,
        queue_height,
        footer_height,
        input_height,
    }
}

/// Every chrome region for one frame, derived from a single `chrome_layout`
/// pass. The frame draw uses this instead of calling `history_area`,
/// `dock_area`, … separately, each of which recomputes the whole layout.
pub struct ChromeAreas {
    pub status: Rect,
    pub history: Rect,
    pub dock: Rect,
    pub queue: Rect,
    pub footer: Rect,
    pub input: Rect,
    pub body: Rect,
}

/// Compute every chrome region in one `chrome_layout` pass. The individual
/// `*_area` accessors below stay for callers that need a single region; the
/// draw loop uses this to avoid recomputing the layout once per region.
pub fn chrome_areas(app: &App, frame_width: u16, frame_height: u16) -> ChromeAreas {
    let layout = chrome_layout(app, frame_width, frame_height);
    let history_y = 1;
    let dock_y = history_y + layout.history_height;
    let queue_y = dock_y + layout.dock_height;
    let footer_y = queue_y + layout.queue_height;
    let input_y = footer_y + layout.footer_height;
    ChromeAreas {
        status: Rect::new(0, 0, frame_width, 1),
        history: Rect::new(0, history_y, frame_width, layout.history_height),
        dock: Rect::new(0, dock_y, frame_width, layout.dock_height),
        queue: Rect::new(0, queue_y, frame_width, layout.queue_height),
        footer: Rect::new(0, footer_y, frame_width, layout.footer_height),
        input: Rect::new(0, input_y, frame_width, layout.input_height),
        body: Rect::new(0, 1, frame_width, frame_height.saturating_sub(1)),
    }
}

pub fn history_viewport_height(app: &App, frame_width: u16, frame_height: u16) -> usize {
    chrome_layout(app, frame_width, frame_height).history_height as usize
}

pub fn history_area(app: &App, frame_width: u16, frame_height: u16) -> Rect {
    let layout = chrome_layout(app, frame_width, frame_height);
    Rect::new(0, 1, frame_width, layout.history_height)
}

pub fn input_area(app: &App, frame_width: u16, frame_height: u16) -> Rect {
    let layout = chrome_layout(app, frame_width, frame_height);
    let y =
        1 + layout.history_height + layout.dock_height + layout.queue_height + layout.footer_height;
    Rect::new(0, y, frame_width, layout.input_height)
}

pub fn input_content_area(app: &App, frame_width: u16, frame_height: u16) -> Rect {
    input_content_area_from_frame(input_area(app, frame_width, frame_height))
}

fn input_content_area_from_frame(area: Rect) -> Rect {
    Rect::new(
        area.x + INPUT_HORIZONTAL_PADDING,
        area.y + 1,
        area.width.saturating_sub(INPUT_HORIZONTAL_PADDING * 2),
        area.height.saturating_sub(2),
    )
}

/// Render the plan checklist as trailing transcript content: one blank
/// gutter row for visual separation from the prior block, then one row
/// per step. No `PLAN` header (the glyphs are self-explanatory) and no
/// scribe rule (that was dock-chrome — inline the checklist just reads
/// like a regular block). Returns `None` when no plan is active.
pub fn plan_dock_lines_snapshot(app: &App, _frame_width: u16) -> Option<Vec<Line<'static>>> {
    use crate::app::PlanDockItemStatus;

    let plan = app.plan_dock.as_ref()?;
    if plan.is_empty() {
        return None;
    }

    let mut lines: Vec<Line<'static>> = Vec::new();

    lines.push(Line::from(vec![Span::styled(
        "  Plan",
        theme::text_faint_style().add_modifier(Modifier::Dim),
    )]));

    // ── items
    for item in &plan.items {
        let (glyph, glyph_style, text_style) = match item.status {
            PlanDockItemStatus::Done => (
                "✓ ",
                Style::default()
                    .fg(theme::state_ok())
                    .add_modifier(Modifier::Bold),
                Style::default()
                    .fg(theme::text_faint())
                    .add_modifier(Modifier::Dim),
            ),
            PlanDockItemStatus::Active => (
                // `▶` — the `■` glyph is reserved for the live
                // assistant-text marker. Using a distinct arrow here
                // keeps "current plan step" visually orthogonal from
                // "model is speaking right now."
                "▶ ",
                Style::default()
                    .fg(theme::brand())
                    .add_modifier(Modifier::Bold),
                Style::default()
                    .fg(theme::text_primary())
                    .add_modifier(Modifier::Bold),
            ),
            PlanDockItemStatus::Pending => (
                "□ ",
                Style::default().fg(theme::text_subtle()),
                Style::default().fg(theme::text_muted()),
            ),
        };
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(glyph, glyph_style),
            Span::styled(item.text.clone(), text_style),
        ]));
    }

    Some(lines)
}

pub fn process_lines_snapshot(app: &App, _frame_width: u16) -> Option<Vec<Line<'static>>> {
    if app.processes.is_empty() {
        return None;
    }

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(vec![Span::styled(
        "  Background",
        theme::text_faint_style().add_modifier(Modifier::Dim),
    )]));
    for task in &app.processes {
        let state = match task.status.terminal_state() {
            None => "running",
            Some(lash_core::ProcessTerminalState::Completed) => "success",
            Some(lash_core::ProcessTerminalState::Failed) => "error",
            Some(lash_core::ProcessTerminalState::Cancelled) => "cancelled",
        };
        let producer = task.kind.as_str();
        let elapsed_duration = task
            .status_duration
            .unwrap_or_else(|| task.first_seen.elapsed());
        let elapsed =
            crate::util::format_duration_ms_if_visible(elapsed_duration.as_millis() as u64)
                .unwrap_or_else(|| "0:00".to_string());
        let state_style = match task.status.terminal_state() {
            None => theme::turn_status_state(),
            Some(lash_core::ProcessTerminalState::Completed) => theme::tool_success(),
            Some(lash_core::ProcessTerminalState::Failed)
            | Some(lash_core::ProcessTerminalState::Cancelled) => theme::tool_failure(),
        };
        lines.push(Line::from(vec![
            Span::styled("  ◆ ", theme::text_faint_style()),
            Span::styled(state.to_string(), state_style),
            Span::styled(" · ", theme::text_faint_style()),
            Span::styled(producer.to_string(), theme::text_subtle_style()),
            Span::styled(" · ", theme::text_faint_style()),
            Span::styled(task.label.clone(), Style::default().fg(theme::text_muted())),
            Span::styled(" · ", theme::text_faint_style()),
            Span::styled(elapsed, theme::text_faint_style()),
        ]));
    }

    Some(lines)
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
    let total_width = padded_content_width(area.width, INPUT_HORIZONTAL_PADDING);
    let prefix_w = 2;

    // Idle placeholder: when the input is empty and there's no pending
    // paste/image payload, render a faint hint that teaches the two
    // primary actions (type to chat, type `/` for commands). Disappears
    // on first keystroke because the render walks `app.input()` directly.
    if app.input().is_empty() && !app.has_pending_input_payload() {
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", theme::PROMPT_CHAR), theme::prompt()),
            Span::styled("Message · / for commands", theme::text_faint_style()),
        ]));
    } else {
        let inline_hint = inline_command_argument_hint(app);
        for (i, logical_line) in app.input().split('\n').enumerate() {
            let hinted_line = if i == 0 {
                inline_hint
                    .as_ref()
                    .map(|hint| format!("{logical_line}{hint}"))
            } else {
                None
            };
            let display_line = hinted_line.as_deref().unwrap_or(logical_line);
            // Compute image markers per-line so ranges are relative
            // to the logical line, not the full multi-line input.
            let line_image_markers = image_marker_ranges(logical_line);
            let segments = wrap_line(display_line, prefix_w, prefix_w, total_width);
            for (j, &(seg_start, seg_end)) in segments.iter().enumerate() {
                let seg_spans = if i == 0 {
                    if let Some(hint) = inline_hint.as_deref() {
                        styled_input_segment_with_hint(
                            logical_line,
                            hint,
                            seg_start,
                            seg_end,
                            &line_image_markers,
                            &app.skills,
                        )
                    } else {
                        styled_input_segment(
                            logical_line,
                            seg_start,
                            seg_end,
                            &line_image_markers,
                            &app.skills,
                        )
                    }
                } else {
                    styled_input_segment(
                        logical_line,
                        seg_start,
                        seg_end,
                        &line_image_markers,
                        &app.skills,
                    )
                };
                let prefix = if i == 0 && j == 0 {
                    Span::styled(format!("{} ", theme::PROMPT_CHAR), theme::prompt())
                } else {
                    Span::styled("  ", Style::default().fg(theme::border_faint()))
                };
                let mut spans = vec![prefix];
                spans.extend(seg_spans);
                lines.push(Line::from(spans));
            }
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

fn inline_command_argument_hint(app: &App) -> Option<String> {
    if app.cursor_pos() != app.input().len() || app.input().contains('\n') {
        return None;
    }
    let trimmed = app.input().trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let rest = &trimmed[1..];
    let (command, arg) = match rest.split_once(' ') {
        Some((command, arg)) => (command, Some(arg.trim())),
        None => (rest, None),
    };

    if arg.is_some_and(|value| !value.is_empty()) {
        return None;
    }

    let hint =
        crate::command::argument_hint(&format!("/{command}"), &app.skills, app.ui_extensions())?;

    let needs_leading_space = !app.input().chars().last().is_some_and(char::is_whitespace);
    Some(if needs_leading_space {
        format!(" {hint}")
    } else {
        hint
    })
}

pub(crate) fn input_byte_offset_for_screen_position(
    app: &App,
    frame_width: u16,
    frame_height: u16,
    column: u16,
    row: u16,
) -> Option<usize> {
    let area = input_area(app, frame_width, frame_height);
    let content_area = input_content_area_from_frame(area);
    if content_area.width == 0 || content_area.height == 0 {
        return None;
    }
    if row < content_area.y
        || row >= content_area.y + content_area.height
        || column < content_area.x
        || column >= content_area.x + content_area.width
    {
        return None;
    }

    let snapshot = input_render_snapshot(app, area);
    let visual_row = snapshot.scroll_offset + (row - content_area.y) as usize;
    let visual_col = (column - content_area.x) as usize;
    Some(input_byte_offset_at_visual_position(
        app.input(),
        visual_row,
        visual_col,
        content_area.width as usize,
    ))
}

pub(crate) fn input_selection_rects(app: &App, area: Rect) -> Vec<(u16, u16, u16)> {
    let Some(range) = app.input_selection_range() else {
        return Vec::new();
    };
    let content_area = input_content_area_from_frame(area);
    if content_area.width == 0 || content_area.height == 0 {
        return Vec::new();
    }

    let snapshot = input_render_snapshot(app, area);
    let visible_top = snapshot.scroll_offset;
    let visible_bottom = snapshot.scroll_offset + content_area.height as usize;
    let total_width = content_area.width as usize;
    let prefix_w = 2usize;
    let mut rects = Vec::new();
    let mut visual_row = 0usize;
    let mut byte_offset = 0usize;

    for logical_line in app.input().split('\n') {
        let line_start = byte_offset;
        let line_end = line_start + logical_line.len();
        let segments = wrap_line(logical_line, prefix_w, prefix_w, total_width);
        for &(seg_start, seg_end) in &segments {
            let segment_range = (line_start + seg_start)..(line_start + seg_end);
            if visual_row >= visible_top
                && visual_row < visible_bottom
                && range.start < segment_range.end
                && segment_range.start < range.end
            {
                let overlap_start = range.start.max(segment_range.start) - line_start;
                let overlap_end = range.end.min(segment_range.end) - line_start;
                let col_start =
                    prefix_w + UnicodeWidthStr::width(&logical_line[seg_start..overlap_start]);
                let col_end =
                    prefix_w + UnicodeWidthStr::width(&logical_line[seg_start..overlap_end]);
                if col_end > col_start {
                    rects.push((
                        content_area.x + col_start as u16,
                        content_area.y + (visual_row - visible_top) as u16,
                        (col_end - col_start) as u16,
                    ));
                }
            }
            visual_row += 1;
        }
        byte_offset = line_end + 1;
    }

    rects
}

pub(crate) fn find_visible_block(app: &App, scroll_offset: usize) -> (usize, usize) {
    if app.height_cache_snapshot().is_empty() {
        return (0, 0);
    }
    let cache = app.height_cache_snapshot();
    let idx = cache.partition_point(|&cumulative| cumulative <= scroll_offset);
    if idx >= app.timeline.len() {
        return (app.timeline.len(), 0);
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
    for idx in 0..app.timeline.len() {
        lines.extend(render_block_lines(
            app,
            idx,
            viewport_width,
            viewport_height,
        ));
    }
    lines.extend(live_tool_output_standalone_lines(app, viewport_width));
    if let Some(live_lines) = app.live_reasoning_lines_snapshot() {
        if app.live_reasoning_leading_padding() > 0 {
            lines.push(Line::from(""));
        }
        lines.extend(live_lines.iter().cloned());
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

fn input_byte_offset_at_visual_position(
    input: &str,
    target_row: usize,
    target_col: usize,
    full_width: usize,
) -> usize {
    let prefix_w = 2usize;
    let mut visual_row = 0usize;
    let mut byte_offset = 0usize;

    for logical_line in input.split('\n') {
        let segments = wrap_line(logical_line, prefix_w, prefix_w, full_width);
        for &(seg_start, seg_end) in &segments {
            if visual_row == target_row {
                let display_col = target_col.saturating_sub(prefix_w);
                let seg_text = &logical_line[seg_start..seg_end];
                let local = EditorState::byte_pos_at_display_col(seg_text, display_col);
                return byte_offset + seg_start + local;
            }
            visual_row += 1;
        }
        byte_offset += logical_line.len() + 1;
    }

    input.len()
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
                spans.push(Span::styled(
                    " · ",
                    Style::default().fg(theme::border_faint()),
                ));
            }
            spans.push(Span::styled(
                (*label).to_string(),
                Style::default()
                    .fg(theme::brand())
                    .add_modifier(Modifier::Bold),
            ));
        }
        spans.push(Span::styled(
            " · ",
            Style::default().fg(theme::border_faint()),
        ));
    }

    let location_label = app
        .repo_status
        .as_ref()
        .map(|repo| format!("{} · {}", repo.repo_name, repo.display_ref()))
        .unwrap_or_else(|| app.cwd.clone());
    spans.push(text_display::sanitize_span(
        location_label,
        Style::default().fg(theme::text_faint()),
    ));
    Some(Line::from(spans))
}

fn styled_input_segment(
    logical_line: &str,
    seg_start: usize,
    seg_end: usize,
    image_markers: &[(std::ops::Range<usize>, usize)],
    skills: &SkillCatalog,
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
            spans.extend(styled_text_with_slash_command(
                logical_line,
                cursor,
                clamped_start,
                skills,
                theme::user_input(),
                theme::slash_command_slash(),
            ));
        }
        spans.push(Span::styled(
            logical_line[clamped_start..clamped_end].to_string(),
            theme::image_marker(),
        ));
        cursor = clamped_end;
    }

    if cursor < seg_end {
        spans.extend(styled_text_with_slash_command(
            logical_line,
            cursor,
            seg_end,
            skills,
            theme::user_input(),
            theme::slash_command_slash(),
        ));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), theme::user_input()));
    }
    spans
}

fn styled_input_segment_with_hint(
    logical_line: &str,
    hint: &str,
    seg_start: usize,
    seg_end: usize,
    image_markers: &[(std::ops::Range<usize>, usize)],
    skills: &SkillCatalog,
) -> Vec<Span<'static>> {
    let input_end = logical_line.len();
    let mut spans = Vec::new();

    if seg_start < input_end {
        let actual_end = seg_end.min(input_end);
        spans.extend(styled_input_segment(
            logical_line,
            seg_start,
            actual_end,
            image_markers,
            skills,
        ));
    }

    if seg_end > input_end {
        let hint_start = seg_start.saturating_sub(input_end);
        let hint_end = (seg_end - input_end).min(hint.len());
        if hint_start < hint_end {
            spans.push(Span::styled(
                hint[hint_start..hint_end].to_string(),
                theme::text_faint_style(),
            ));
        }
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), theme::user_input()));
    }
    spans
}

fn styled_user_input_segment(
    text: &str,
    seg_start: usize,
    seg_end: usize,
    skills: &SkillCatalog,
) -> Vec<Span<'static>> {
    styled_text_with_slash_command(
        text,
        seg_start,
        seg_end,
        skills,
        theme::user_input(),
        theme::slash_command_slash(),
    )
}

fn styled_text_with_slash_command(
    text: &str,
    seg_start: usize,
    seg_end: usize,
    skills: &SkillCatalog,
    base_style: Style,
    slash_style: Style,
) -> Vec<Span<'static>> {
    if seg_start >= seg_end {
        return vec![text_display::sanitize_span(String::new(), base_style)];
    }
    let mut spans = Vec::new();
    let mut cursor = seg_start;
    for (slash_start, slash_end) in
        slash_command_ranges_in_segment(text, seg_start, seg_end, skills)
    {
        if slash_start > cursor {
            spans.push(text_display::sanitize_span(
                text[cursor..slash_start].to_string(),
                base_style,
            ));
        }
        spans.push(text_display::sanitize_span(
            text[slash_start..slash_end].to_string(),
            slash_style,
        ));
        cursor = slash_end;
    }
    if cursor < seg_end || spans.is_empty() {
        spans.push(text_display::sanitize_span(
            text[cursor..seg_end].to_string(),
            base_style,
        ));
    }
    if spans.is_empty() {
        spans.push(text_display::sanitize_span(String::new(), base_style));
    }
    spans
}

fn slash_command_ranges_in_segment(
    text: &str,
    seg_start: usize,
    seg_end: usize,
    skills: &SkillCatalog,
) -> Vec<(usize, usize)> {
    slash_command_slash_ranges(text, skills)
        .into_iter()
        .filter_map(|(slash_start, slash_end)| {
            if slash_end <= seg_start || slash_start >= seg_end {
                return None;
            }
            let clamped_start = slash_start.max(seg_start);
            let clamped_end = slash_end.min(seg_end);
            (clamped_start < clamped_end).then_some((clamped_start, clamped_end))
        })
        .collect()
}

fn slash_command_slash_ranges(text: &str, skills: &SkillCatalog) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let trimmed = text.trim_start();
    if trimmed.starts_with('/') {
        let slash_start = text.len() - trimmed.len();
        // Rendering only inspects the slash-character glyph — plugin commands are
        // treated the same way as builtins regardless, so passing an empty catalog
        // is sufficient here and avoids threading the app-level catalog through
        // the rendering layer.
        if crate::command::parse(trimmed, skills).is_some()
            || crate::command::slash_skill_prompt(trimmed, skills).is_some()
        {
            ranges.push((slash_start, slash_start + 1));
        }
    }

    for (range, name) in collect_skill_mentions_with_ranges(text) {
        if skills.get(&name).is_none() {
            continue;
        }
        let slash_range = (range.start, range.start + 1);
        if !ranges.contains(&slash_range) {
            ranges.push(slash_range);
        }
    }

    ranges.sort_unstable_by_key(|(start, _)| *start);
    ranges
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
    blocks: &[UiTimelineItem],
    idx: usize,
    expand_level: u8,
    viewport_width: usize,
    viewport_height: usize,
) -> Vec<Line<'static>> {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline = blocks.to_vec().into();
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
    let blocks = &app.timeline;
    let expand_level = app.expand_level;
    match &blocks[idx] {
        UiTimelineItem::TurnStart(turn) => {
            if turn.show_separator {
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
            // Turns with `show_separator: false` contribute no visible
            // lines — the marker is purely structural, letting later
            // features (turn folding, turn addressing) scan between
            // `TurnStart` markers without changing the renderer.
        }
        UiTimelineItem::UserInput(text) => {
            let marker_style = Style::default().fg(theme::brand());
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
                    spans.extend(styled_user_input_segment(
                        line,
                        seg_start,
                        seg_end,
                        &app.skills,
                    ));
                    lines.push(Line::from(spans));
                }
            }
        }
        UiTimelineItem::AssistantText(text) => {
            // Insert a blank row before the assistant's spoken text
            // unless it directly follows other spoken text (where the
            // gap would be noise). Reasoning used to be in this list,
            // but gluing a compact `┊ thinking …` line straight into
            // `■ I'll do X` reads as one block — the eye can't find
            // the seam.
            let add_spacing_before = idx > 0
                && !matches!(
                    blocks[idx - 1],
                    UiTimelineItem::AssistantText(_)
                        | UiTimelineItem::Splash
                        | UiTimelineItem::TurnStart(_)
                );
            lines.extend(assistant_text::render_assistant_text_block(
                text,
                viewport_width,
                add_spacing_before,
            ));
        }
        UiTimelineItem::AssistantReasoning(text) => {
            let add_spacing_before = idx > 0
                && !matches!(
                    blocks[idx - 1],
                    UiTimelineItem::AssistantReasoning(_)
                        | UiTimelineItem::Splash
                        | UiTimelineItem::TurnStart(_)
                );
            // Show full reasoning only when either (a) the user has
            // opted into full expansion (Alt+O -> level 2), or (b) this
            // is the live tail of a running turn so the stream stays
            // visible as it arrives. Reasoning is the heaviest block in
            // the transcript and lives at L2 alongside full artifacts
            // and shell output.
            let is_live_tail =
                idx + 1 == blocks.len() && app.running && !app.has_live_markdown_output();
            let should_expand = expand_level >= 2 || is_live_tail;
            if should_expand {
                lines.extend(assistant_text::render_assistant_reasoning_block(
                    text,
                    viewport_width,
                    add_spacing_before,
                ));
            } else {
                lines.extend(assistant_text::render_assistant_reasoning_block_compact(
                    text,
                    viewport_width,
                    add_spacing_before,
                ));
            }
        }
        UiTimelineItem::Activity(activity) => {
            render_activity_block(activity, expand_level, lines, viewport_width);
            if app.live_tool_output_anchor_block_index() == Some(idx) {
                render_live_tool_output_inline(lines, app, &activity.call.kind, viewport_width);
            }
        }
        UiTimelineItem::ShellOutput {
            command,
            output,
            error,
        } => {
            lines.push(Line::from(Span::styled(
                format!("$ {command}"),
                theme::code_chrome(),
            )));
            if expand_level >= 1 {
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
        UiTimelineItem::Error(message) => render_bordered_text_block(
            "ERROR",
            message.lines().map(str::to_string).collect(),
            lines,
            viewport_width,
            theme::error_border(),
            theme::error_title(),
            theme::error(),
        ),
        UiTimelineItem::SystemMessage(text) => {
            for line in text.lines() {
                lines.push(Line::from(Span::styled(
                    line.to_string(),
                    theme::system_message(),
                )));
            }
        }
        UiTimelineItem::PluginPanel(panel) => {
            render_section_panel_block(&panel.title, &panel.content, lines, viewport_width);
        }
        UiTimelineItem::LashlangCode(code) => {
            // Only shown at full expansion (Alt+O). At lower levels the
            // block contributes zero lines — its tool activities carry
            // the visible story already.
            if expand_level >= 2 {
                render_lashlang_code_block(code, lines, viewport_width);
            }
        }
        UiTimelineItem::Splash => {
            render_splash(lines, viewport_width, viewport_height, blocks.len() == 1)
        }
    }
}

/// Render the captured `lashlang` source for an RLM turn, with a dim `╎`
/// gutter to mark it as "what the model ran" (distinct from the `│`
/// shell gutter and `┊` reasoning gutter).
fn render_lashlang_code_block(code: &str, lines: &mut Vec<Line<'static>>, _viewport_width: usize) {
    let header_style = theme::code_chrome();
    let gutter_style = theme::code_chrome();
    let body_style = theme::system_output();

    lines.push(Line::from(Span::styled("lashlang", header_style)));
    for line in code.lines() {
        lines.push(Line::from(vec![
            Span::styled("╎ ", gutter_style),
            Span::styled(line.to_string(), body_style),
        ]));
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
    let sodium = Style::default().fg(theme::brand());
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
        Span::styled("──────────", Style::default().fg(theme::border_dim())),
        Span::styled("──────────", Style::default().fg(theme::border_faint())),
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
    if app.live.tool_output.height() == 0 {
        return;
    }

    let mut hidden_rows = app.live.tool_output.hidden;
    let mut logical = Vec::with_capacity(app.live.tool_output.height());
    logical.extend(app.live.tool_output.lines.iter().cloned());
    if !app.live.tool_output.partial.is_empty() {
        logical.push(app.live.tool_output.partial.clone());
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
    if app.live.tool_output.height() == 0 || viewport_width == 0 {
        return;
    }

    let (prefix, prefix_style, content_style) = if *activity_kind == ActivityKind::Subagent {
        (
            "    │ ",
            Style::default()
                .fg(theme::brand())
                .add_modifier(Modifier::Bold),
            Style::default().fg(theme::text_muted()),
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

pub(crate) fn live_tool_output_standalone_height(app: &App, viewport_width: usize) -> usize {
    live_tool_output_standalone_lines(app, viewport_width).len()
}

pub(crate) fn live_tool_output_standalone_lines(
    app: &App,
    viewport_width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if app.live.tool_output.height() == 0
        || viewport_width == 0
        || app.live.tool_output.title.is_none()
        || app.live_tool_output_anchor_block_index().is_some()
    {
        return lines;
    }

    if !matches!(
        app.timeline.last(),
        Some(UiTimelineItem::TurnStart(_) | UiTimelineItem::Splash)
    ) {
        lines.push(Line::from(""));
    }
    let title = app.live.tool_output.title.as_deref().unwrap_or_default();
    lines.push(Line::from(vec![
        Span::styled("• ", Style::default().fg(theme::brand())),
        Span::styled(title.to_string(), theme::code_chrome()),
    ]));
    append_streaming_output_lines(
        &mut lines,
        app,
        viewport_width,
        "  │ ",
        theme::code_chrome(),
        theme::system_output(),
        0,
    );
    lines
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
    // Show the bar whenever there is scrollable content above or below the
    // viewport. The old behavior hid it within 2 rows of the bottom — which
    // is exactly where the "there is more history above you" signal is
    // needed most, because the app starts in follow-mode pinned to the end
    // of a long session. Without the bar at the bottom, a user opening
    // their 300-turn conversation sees no hint that the scrollback exists.

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
    // Wrap against the widest prefix so no rendered line — first
    // segment or continuation — can overflow `width`. Budget based on
    // just the leading prefix would let a wider continuation push the
    // second row past the viewport boundary.
    let prefix_width = UnicodeWidthStr::width(prefix.as_str());
    let continuation_width = UnicodeWidthStr::width(continuation.as_str());
    let available = width.saturating_sub(prefix_width.max(continuation_width));
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
