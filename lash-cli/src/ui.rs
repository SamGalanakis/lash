use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Padding, Paragraph, Wrap},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, PatchFilePreview,
    patch_file_subject, patch_status_title,
};
use crate::app::{
    App, DisplayBlock, PreparedTurn, PromptState, SPLASH_CONTENT_HEIGHT, SPLASH_SCROLLBACK_HEIGHT,
    TextSelection, preview_text_lines,
};
use crate::assistant_text;
use crate::diff::render_inline_diff;
use crate::editor::SuggestionKind;
use crate::input_items::image_marker_ranges;
use crate::markdown;
use crate::text_display;
use crate::theme;
use lash::collect_skill_mentions_with_ranges;

const INPUT_HORIZONTAL_PADDING: u16 = 1;
const PROMPT_HORIZONTAL_PADDING: u16 = 1;
const MIN_HISTORY_HEIGHT: u16 = 3;
const MAX_INPUT_HEIGHT: u16 = 20;
const EXPLORATION_DETAIL_MAX_ROWS: usize = 3;
const SCROLL_INDICATOR_HIDE_TAIL_ROWS: usize = 2;
const SCROLL_INDICATOR_MIN_HEIGHT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScrollIndicator {
    x: u16,
    y: u16,
    height: u16,
}

fn padded_content_width(frame_width: u16, horizontal_padding: u16) -> usize {
    frame_width.saturating_sub(horizontal_padding.saturating_mul(2)) as usize
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

fn prompt_inner_width(frame_width: u16) -> usize {
    frame_width.saturating_sub(4 + PROMPT_HORIZONTAL_PADDING.saturating_mul(2)) as usize
}

fn queue_preview_height(app: &App, frame_width: u16) -> u16 {
    queue_preview_lines(app, frame_width).len() as u16
}

fn turn_status_height(app: &App) -> u16 {
    if app.live_turn.is_some() { 1 } else { 0 }
}

/// Exact history viewport height based on the same layout math used in draw().
pub fn history_viewport_height(app: &App, frame_width: u16, frame_height: u16) -> usize {
    let status_h = turn_status_height(app);
    let queued_h = queue_preview_height(app, frame_width);
    let reserved_height = 1 + status_h + queued_h;
    let input_h = input_height(app, frame_width, frame_height, reserved_height);
    let overhead = 1 + status_h + queued_h + input_h;
    frame_height.saturating_sub(overhead) as usize
}

/// Compute the history viewport Rect using the same layout math as draw().
pub fn history_area(app: &App, frame_width: u16, frame_height: u16) -> Rect {
    let status_h = turn_status_height(app);
    let queued_h = queue_preview_height(app, frame_width);
    let reserved_height = 1 + status_h + queued_h;
    let input_h = input_height(app, frame_width, frame_height, reserved_height);
    let chunks = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(status_h),
        Constraint::Length(queued_h),
        Constraint::Length(input_h),
    ])
    .split(Rect::new(0, 0, frame_width, frame_height));
    chunks[1]
}

pub fn draw(frame: &mut Frame, app: &App) {
    // Paint entire frame with FORM bg so no terminal background bleeds through
    frame.render_widget(Block::default().style(theme::history_bg()), frame.area());

    let status_h = turn_status_height(app);
    let queued_h = queue_preview_height(app, frame.area().width);
    let reserved_height = 1 + status_h + queued_h;
    let input_h = input_height(
        app,
        frame.area().width,
        frame.area().height,
        reserved_height,
    );

    let chunks = Layout::vertical([
        Constraint::Length(1),        // [0] status bar
        Constraint::Min(3),           // [1] history
        Constraint::Length(status_h), // [2] turn status (only during an active turn)
        Constraint::Length(queued_h), // [3] pending input preview
        Constraint::Length(input_h),  // [4] input (dynamic height)
    ])
    .split(frame.area());

    draw_status_bar(frame, app, chunks[0]);
    draw_history(frame, app, chunks[1]);
    if app.live_turn.is_some() {
        draw_turn_status(frame, app, chunks[2]);
    }
    draw_queue_preview(frame, app, chunks[3]);
    if app.has_prompt() {
        draw_prompt(frame, app, chunks[4]);
    } else {
        draw_input(frame, app, chunks[4]);
        draw_suggestions(frame, app, chunks[4]);
    }
    draw_session_picker(frame, app, chunks[1]); // overlay on history area
    draw_skill_picker(frame, app, chunks[1]); // overlay on history area

    // Apply text selection highlight over the rendered buffer
    if app.selection.active || app.selection.visible {
        apply_selection_highlight(frame, &app.selection, chunks[1], app.scroll_offset);
    }
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let width = area.width as usize;
    if width == 0 {
        return;
    }

    // ── Left-side segment widths (progressive: name → model → variant) ──
    let name_w = " lash".chars().count();
    let sep_w = theme::STATUS_SEP.chars().count();
    let model_w = app.model.chars().count();
    let variant_w = app
        .model_variant
        .as_ref()
        .map_or(0, |variant| sep_w + variant.chars().count());
    let left_1 = name_w;
    let left_2 = name_w + sep_w + model_w;
    let left_3 = left_2 + variant_w;

    // ── Right-side: token components ──
    let display_input_tokens = app.token_usage.input_tokens;
    let display_output_tokens = app.token_usage.output_tokens + app.live_output_tokens_estimate;
    let display_total_tokens = display_input_tokens + display_output_tokens;
    let has_usage = display_total_tokens > 0;
    let live_context_budget = current_context_budget_tokens(app);
    let ctx_pct = app.context_window.and_then(|ctx_win| {
        live_context_budget
            .and_then(|used| context_usage_pct_from_total(used, ctx_win))
            .or_else(|| {
                app.last_prompt_usage
                    .as_ref()
                    .and_then(|usage| context_usage_pct(usage, ctx_win))
            })
    });
    let ctx_display = app.context_window.and_then(|ctx_win| {
        live_context_budget
            .and_then(|used| format_context_usage_from_total(used, ctx_win))
            .or_else(|| {
                app.last_prompt_usage
                    .as_ref()
                    .and_then(|usage| format_context_usage(usage, ctx_win))
            })
    });
    let ctx_display_pct = ctx_pct.map(|pct| format!("{pct:.1}%"));

    let in_out = if has_usage {
        format!(
            "{} in \u{b7} {} out",
            crate::app::format_tokens(display_input_tokens),
            crate::app::format_tokens(display_output_tokens),
        )
    } else {
        String::new()
    };
    let total = if has_usage {
        format!("{} total", crate::app::format_tokens(display_total_tokens),)
    } else {
        String::new()
    };
    // Prefer current-turn context occupancy. Fall back to cumulative session tokens
    // only when we do not have a context-window-backed percentage.
    let right_variants = if let Some(ctx) = ctx_display {
        [
            ctx,
            ctx_display_pct.unwrap_or_default(),
            String::new(),
            String::new(),
        ]
    } else {
        [in_out.clone(), total.clone(), String::new(), String::new()]
    };

    let right_w = |idx: usize| -> usize {
        let s = &right_variants[idx];
        if s.is_empty() {
            0
        } else {
            s.chars().count() + 1 // +1 trailing space
        }
    };

    // ── Progressive fit (drop lowest priority first) ──
    let left_w = |level: usize| match level {
        3 => left_3,
        2 => left_2,
        _ => left_1,
    };
    let check = |ll: usize, ri: usize| -> bool {
        let lw = left_w(ll);
        let rw = right_w(ri);
        if rw > 0 {
            lw + 1 + rw <= width
        } else {
            lw <= width
        }
    };

    let mut left_level = 3usize;
    let mut right_idx = 0usize;

    if !check(left_level, right_idx) {
        right_idx = 1;
    }
    if !check(left_level, right_idx) {
        right_idx = 2;
    }
    if !check(left_level, right_idx) {
        right_idx = 3;
    }
    if !check(left_level, right_idx) {
        left_level = 3;
    }
    if !check(left_level, right_idx) {
        left_level = 2;
    }
    if !check(left_level, right_idx) {
        left_level = 1;
    }

    // ── Build spans ──
    let mut spans: Vec<Span> = vec![Span::styled(" lash", theme::app_title())];
    if left_level >= 2 {
        spans.push(Span::styled(theme::STATUS_SEP, theme::status_separator()));
        spans.push(Span::styled(&app.model, theme::model_name()));
        if left_level >= 3
            && let Some(variant) = &app.model_variant
        {
            spans.push(Span::styled(theme::STATUS_SEP, theme::status_separator()));
            spans.push(Span::styled(
                variant.clone(),
                Style::default()
                    .fg(theme::SODIUM)
                    .add_modifier(Modifier::BOLD),
            ));
        }
    }
    // Padding between left and right
    let actual_left: usize = spans.iter().map(|s| s.width()).sum();
    let total_right = right_w(right_idx);
    let pad = width.saturating_sub(actual_left + total_right);
    if pad > 0 {
        spans.push(Span::styled(" ".repeat(pad), theme::bar_bg()));
    }

    // Token spans
    if right_idx < 3 && !right_variants[right_idx].is_empty() {
        let token_text = &right_variants[right_idx];
        if let Some(pct) = ctx_pct {
            let ctx_color = if pct >= 80.0 {
                theme::SODIUM
            } else {
                theme::CHALK_DIM
            };
            spans.push(Span::styled(
                token_text.clone(),
                Style::default().fg(ctx_color),
            ));
        } else {
            spans.push(Span::styled(
                token_text.clone(),
                Style::default().fg(theme::CHALK_DIM),
            ));
        }
        spans.push(Span::styled(" ", theme::bar_bg()));
    }

    let bar = Paragraph::new(Line::from(spans)).style(theme::bar_bg());
    frame.render_widget(bar, area);
}

fn context_usage_total(usage: &lash::PromptUsage) -> i64 {
    usage.context_budget_tokens as i64
}

fn current_context_budget_tokens(app: &App) -> Option<i64> {
    if !app.running {
        return None;
    }
    let input = app.last_response_usage.input_tokens.max(0);
    let cached = app.last_response_usage.cached_input_tokens.max(0);
    let output = (app.last_response_usage.output_tokens + app.live_output_tokens_estimate).max(0);
    if input == 0 && cached == 0 && output == 0 {
        return None;
    }
    Some(if app.context_usage_excludes_cached_input {
        input + output + cached
    } else {
        (input - cached).max(0) + output + cached
    })
}

fn context_usage_pct_from_total(used: i64, context_window: u64) -> Option<f64> {
    if used <= 0 || context_window == 0 {
        return None;
    }
    Some(used as f64 / context_window as f64 * 100.0)
}

fn context_usage_pct(usage: &lash::PromptUsage, context_window: u64) -> Option<f64> {
    context_usage_pct_from_total(context_usage_total(usage), context_window)
}

fn format_context_usage_from_total(used: i64, context_window: u64) -> Option<String> {
    let pct = context_usage_pct_from_total(used, context_window)?;
    Some(format!(
        "{} / {} ({:.1}%)",
        crate::app::format_tokens(used),
        crate::app::format_tokens(context_window as i64),
        pct
    ))
}

fn format_context_usage(usage: &lash::PromptUsage, context_window: u64) -> Option<String> {
    format_context_usage_from_total(context_usage_total(usage), context_window)
}

fn draw_history(frame: &mut Frame, app: &App, area: Rect) {
    let viewport_height = area.height as usize;
    let viewport_width = area.width as usize;

    // Expansion changes can reduce the number or width of rendered rows. Clear the
    // viewport first so stale glyphs from the previous frame do not survive.
    frame.render_widget(Block::default().style(theme::history_bg()), area);

    // scroll_offset is already clamped by the main loop before draw()
    let scroll = app.scroll_offset;

    // Find the first visible block via binary search on the height cache
    let (first_idx, skip_lines) = find_visible_block_readonly(app, scroll, viewport_width);

    let mut lines: Vec<Line> = Vec::with_capacity(viewport_height + skip_lines + 20);

    // Render only blocks from first_idx until we have enough lines
    for idx in first_idx..app.blocks.len() {
        render_block(
            &app.blocks,
            idx,
            app.expand_level,
            &mut lines,
            viewport_width,
            viewport_height,
        );
        if lines.len() >= viewport_height + skip_lines {
            break;
        }
    }

    // Render live streaming tool output.
    // Delegate output gets its own branch lane so it doesn't blend into normal shell output.
    if app.streaming_output_height() > 0 && lines.len() < viewport_height + skip_lines {
        let delegate_stream = app.active_delegate.is_some();
        let mut streaming_lines = Vec::with_capacity(app.streaming_output_height());
        if app.streaming_output_hidden > 0 {
            streaming_lines.push(format!(
                "… {} earlier shell lines hidden …",
                app.streaming_output_hidden
            ));
        }
        streaming_lines.extend(app.streaming_output.iter().cloned());
        if !app.streaming_output_partial.is_empty() {
            streaming_lines.push(app.streaming_output_partial.clone());
        }
        for (idx, line) in streaming_lines.iter().enumerate() {
            let (prefix, prefix_style, content_style) = if delegate_stream {
                let prefix = if idx == 0 {
                    "\u{251c}\u{2500} "
                } else {
                    "\u{2502}  "
                };
                (
                    prefix,
                    Style::default()
                        .fg(theme::SODIUM)
                        .add_modifier(Modifier::BOLD),
                    Style::default().fg(theme::CHALK_MID),
                )
            } else {
                ("\u{2502} ", theme::code_chrome(), theme::code_content())
            };
            lines.push(Line::from(vec![
                Span::styled(prefix, prefix_style),
                Span::styled(line.clone(), content_style),
            ]));
            if lines.len() >= viewport_height + skip_lines {
                break;
            }
        }
    }

    // Use Paragraph's built-in scroll to skip visual rows correctly.
    // Manual .skip()/.take() on logical Lines is wrong because skip_lines
    // is in visual-row space (from the height cache) but Lines can wrap.
    let paragraph = Paragraph::new(wrap_rendered_lines(&lines, viewport_width))
        .style(theme::history_bg())
        .block(Block::default().borders(Borders::NONE))
        .scroll((skip_lines as u16, 0));

    frame.render_widget(paragraph, area);

    if let Some(indicator) = history_scroll_indicator(app, area) {
        render_history_scroll_indicator(frame, indicator);
    }
}

fn history_scroll_indicator(app: &App, area: Rect) -> Option<ScrollIndicator> {
    if area.width == 0 || area.height == 0 {
        return None;
    }

    let viewport_height = area.height as usize;
    let block_height = app.height_cache_snapshot().last().copied().unwrap_or(0);
    let total_content_height = block_height + app.streaming_output_height();
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
    let thumb_height = (viewport_height * viewport_height).div_ceil(total_content_height);
    let thumb_height = thumb_height.clamp(min_height, viewport_height);
    let max_top = viewport_height.saturating_sub(thumb_height);
    let top = if max_top == 0 {
        0
    } else {
        scroll_offset * max_top / max_scroll
    };

    Some(ScrollIndicator {
        x: area.right().saturating_sub(1),
        y: area.y + top as u16,
        height: thumb_height as u16,
    })
}

fn render_history_scroll_indicator(frame: &mut Frame, indicator: ScrollIndicator) {
    let style = theme::scroll_indicator();
    match indicator.height {
        0 => {}
        1 => {
            if let Some(cell) = frame.buffer_mut().cell_mut((indicator.x, indicator.y)) {
                cell.set_symbol("█");
                cell.set_style(style);
            }
        }
        2 => {
            if let Some(cell) = frame.buffer_mut().cell_mut((indicator.x, indicator.y)) {
                cell.set_symbol("▄");
                cell.set_style(style);
            }
            if let Some(cell) = frame
                .buffer_mut()
                .cell_mut((indicator.x, indicator.y.saturating_add(1)))
            {
                cell.set_symbol("▀");
                cell.set_style(style);
            }
        }
        _ => {
            if let Some(cell) = frame.buffer_mut().cell_mut((indicator.x, indicator.y)) {
                cell.set_symbol("▄");
                cell.set_style(style);
            }
            for row in 1..indicator.height.saturating_sub(1) {
                if let Some(cell) = frame
                    .buffer_mut()
                    .cell_mut((indicator.x, indicator.y.saturating_add(row)))
                {
                    cell.set_symbol("█");
                    cell.set_style(style);
                }
            }
            if let Some(cell) = frame.buffer_mut().cell_mut((
                indicator.x,
                indicator
                    .y
                    .saturating_add(indicator.height.saturating_sub(1)),
            )) {
                cell.set_symbol("▀");
                cell.set_style(style);
            }
        }
    }
}

/// Read-only version of find_visible_block that uses the pre-computed height cache.
/// The cache MUST be pre-warmed via `ensure_height_cache_pub` before calling draw().
fn find_visible_block_readonly(app: &App, scroll_offset: usize, _width: usize) -> (usize, usize) {
    if app.height_cache_snapshot().is_empty() {
        return (0, 0);
    }
    let cache = app.height_cache_snapshot();
    let idx = cache.partition_point(|&cumulative| cumulative <= scroll_offset);
    if idx >= app.blocks.len() {
        return (app.blocks.len(), 0);
    }
    let block_start = if idx == 0 { 0 } else { cache[idx - 1] };
    let skip = scroll_offset - block_start;
    (idx, skip)
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
    out.push('\u{2026}');
    out
}

fn truncate_with_suffix(text: &str, suffix: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    let suffix_width = text_display::visible_width(suffix);
    if suffix_width >= max_width {
        return truncate_to_display_width(suffix, max_width);
    }

    let text_width = text_display::visible_width(text);
    if text_width + suffix_width <= max_width {
        return format!("{text}{suffix}");
    }

    let prefix_width = max_width.saturating_sub(suffix_width);
    format!(
        "{}{}",
        truncate_to_display_width(text, prefix_width),
        suffix
    )
}

/// Build a ghost fold summary for a group of code blocks starting at `idx`.
/// `idx` should point to the first (non-continuation) CodeBlock in a group.
/// Scans forward through contiguous CodeBlock/Activity/CodeOutput blocks.
fn build_code_fold_summary(blocks: &[DisplayBlock], idx: usize) -> String {
    let mut block_count = 0usize;
    let mut line_count = 0usize;

    for b in &blocks[idx..] {
        match b {
            DisplayBlock::CodeBlock { code, .. } => {
                block_count += 1;
                line_count += code.lines().count().max(1);
            }
            DisplayBlock::Activity(_) | DisplayBlock::CodeOutput { .. } => continue,
            _ => break,
        }
    }

    format!(
        "\u{25b6} {} code block{} \u{b7} {} line{}",
        block_count,
        if block_count != 1 { "s" } else { "" },
        line_count,
        if line_count != 1 { "s" } else { "" }
    )
}

fn activity_style(status: ActivityStatus) -> Style {
    match status {
        ActivityStatus::Completed => theme::tool_success(),
        ActivityStatus::Failed => theme::tool_failure(),
    }
}

fn activity_icon(status: ActivityStatus) -> &'static str {
    match status {
        ActivityStatus::Completed => "+",
        ActivityStatus::Failed => "\u{00d7}",
    }
}

fn push_wrapped_prefixed<'a>(
    lines: &mut Vec<Line<'a>>,
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
    let mut first = true;
    let mut start = 0usize;
    let mut col = 0usize;
    for (idx, ch) in text.char_indices() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if col + w > available && col > 0 {
            let shown_prefix = if first {
                prefix.clone()
            } else {
                continuation.clone()
            };
            lines.push(Line::from(vec![
                Span::styled(shown_prefix, style),
                text_display::sanitize_span(text[start..idx].to_string(), style),
            ]));
            first = false;
            start = idx;
            col = w;
        } else {
            col += w;
        }
    }
    let shown_prefix = if first { prefix } else { continuation };
    lines.push(Line::from(vec![
        Span::styled(shown_prefix, style),
        text_display::sanitize_span(text[start..].to_string(), style),
    ]));
}

fn truncate_with_forced_ellipsis(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if max_width == 1 {
        return "\u{2026}".to_string();
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
    out.push('\u{2026}');
    out
}

fn render_exploration_detail_lines<'a>(
    lines: &mut Vec<Line<'a>>,
    detail_lines: &'a [String],
    viewport_width: usize,
    prefix: &str,
    prefix_style: Style,
) {
    let prefix_width = UnicodeWidthStr::width(prefix);
    let available = viewport_width.saturating_sub(prefix_width);
    let detail_style = Style::default().fg(theme::ASH_TEXT);

    if available == 0 {
        let limit = detail_lines.len().min(EXPLORATION_DETAIL_MAX_ROWS);
        for _ in 0..limit {
            lines.push(Line::from(Span::styled(prefix.to_string(), prefix_style)));
        }
        return;
    }

    let mut rendered_rows = 0usize;
    for (line_idx, detail) in detail_lines.iter().enumerate() {
        let segments = if detail.is_empty() {
            vec![(0usize, 0usize)]
        } else {
            wrap_line(detail, prefix_width, prefix_width, viewport_width)
        };

        for (segment_idx, &(start, end)) in segments.iter().enumerate() {
            if rendered_rows == EXPLORATION_DETAIL_MAX_ROWS {
                return;
            }
            rendered_rows += 1;

            let has_more_content =
                segment_idx + 1 < segments.len() || line_idx + 1 < detail_lines.len();
            let is_last_visible = rendered_rows == EXPLORATION_DETAIL_MAX_ROWS && has_more_content;
            let chunk = if detail.is_empty() {
                if is_last_visible {
                    "\u{2026}".to_string()
                } else {
                    String::new()
                }
            } else if is_last_visible {
                truncate_with_forced_ellipsis(&detail[start..end], available)
            } else {
                truncate_to_display_width(&detail[start..end], available)
            };

            let mut spans = vec![Span::styled(prefix.to_string(), prefix_style)];
            if !chunk.is_empty() {
                spans.push(Span::styled(chunk, detail_style));
            }
            lines.push(Line::from(spans));

            if is_last_visible {
                return;
            }
        }
    }
}

fn render_activity_artifact<'a>(
    lines: &mut Vec<Line<'a>>,
    artifact: &ActivityArtifact,
    viewport_width: usize,
    indent: &str,
) {
    match artifact {
        ActivityArtifact::DiffPreview { title, diff } => {
            lines.push(Line::from(vec![
                Span::styled(indent.to_string(), theme::code_chrome()),
                Span::styled(format!("{}:", title), theme::code_header()),
            ]));
            render_inline_diff(lines, diff, viewport_width, &format!("{}  ", indent));
        }
        ActivityArtifact::PatchPreview { files, .. } => {
            render_patch_preview(lines, files, viewport_width, indent, true);
        }
        ActivityArtifact::TextPreview { title, text } => {
            if let Some(title) = title {
                lines.push(Line::from(vec![
                    Span::styled(indent.to_string(), theme::code_chrome()),
                    Span::styled(format!("{}:", title), theme::code_header()),
                ]));
            }
            let prefix = format!("{}  ", indent);
            for line in preview_text_lines(text) {
                push_wrapped_prefixed(
                    lines,
                    prefix.clone(),
                    prefix.clone(),
                    &line,
                    theme::system_output(),
                    viewport_width,
                );
            }
        }
        ActivityArtifact::SourceList { title, items } => {
            lines.push(Line::from(vec![
                Span::styled(indent.to_string(), theme::code_chrome()),
                Span::styled(format!("{}:", title), theme::code_header()),
            ]));
            for item in items {
                let prefix = format!("{}  ", indent);
                push_wrapped_prefixed(
                    lines,
                    prefix.clone(),
                    prefix,
                    item,
                    theme::system_output(),
                    viewport_width,
                );
            }
        }
    }
}

fn render_patch_summary_line<'a>(
    lines: &mut Vec<Line<'a>>,
    prefix: &str,
    file: &PatchFilePreview,
    viewport_width: usize,
) {
    let subject = patch_file_subject(file);
    let counts = format!(" (+{} -{})", file.added, file.removed);
    let label = patch_status_title(&file.status);
    let label_width = UnicodeWidthStr::width(label).saturating_add(1);
    let available = viewport_width
        .saturating_sub(UnicodeWidthStr::width(prefix))
        .saturating_sub(label_width)
        .saturating_sub(UnicodeWidthStr::width(counts.as_str()));
    let mut spans = vec![Span::styled(prefix.to_string(), theme::patch_frame())];
    spans.push(Span::styled(label.to_string(), theme::patch_label()));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(
        truncate_to_display_width(&subject, available),
        theme::assistant_text(),
    ));
    spans.push(Span::styled(" (".to_string(), theme::code_chrome()));
    spans.push(Span::styled(format!("+{}", file.added), theme::patch_add()));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(
        format!("-{}", file.removed),
        theme::patch_remove(),
    ));
    spans.push(Span::styled(")".to_string(), theme::code_chrome()));
    lines.push(Line::from(spans));
}

fn render_patch_diff_lines<'a>(
    lines: &mut Vec<Line<'a>>,
    prefix: &str,
    diff: &str,
    viewport_width: usize,
) {
    render_inline_diff(lines, diff, viewport_width, prefix);
}

fn render_patch_preview<'a>(
    lines: &mut Vec<Line<'a>>,
    files: &[PatchFilePreview],
    viewport_width: usize,
    indent: &str,
    include_diffs: bool,
) {
    for file in files.iter() {
        let shows_diff = include_diffs && !file.diff.trim().is_empty();
        let row_prefix = format!("{indent}  ");
        render_patch_summary_line(lines, &row_prefix, file, viewport_width);
        if shows_diff {
            render_patch_diff_lines(lines, &format!("{indent}│ "), &file.diff, viewport_width);
        }
    }
}

fn render_patch_artifact<'a>(
    lines: &mut Vec<Line<'a>>,
    files: &[PatchFilePreview],
    viewport_width: usize,
    indent: &str,
    include_diffs: bool,
) {
    if files.len() == 1 {
        if include_diffs && !files[0].diff.trim().is_empty() {
            render_patch_diff_lines(
                lines,
                &format!("{indent}│ "),
                &files[0].diff,
                viewport_width,
            );
        }
        return;
    }

    render_patch_preview(lines, files, viewport_width, indent, include_diffs);
}

#[derive(Clone)]
struct ActivityLane {
    summary_prefix: String,
    summary_prefix_style: Style,
    detail_prefix: String,
    detail_prefix_style: Style,
    artifact_indent: String,
    parallel_child_prefix: String,
}

#[derive(Clone, Copy)]
struct CodeWorkflowLane;

fn is_code_workflow_activity(kind: ActivityKind) -> bool {
    matches!(
        kind,
        ActivityKind::Exploration
            | ActivityKind::ShellCommand
            | ActivityKind::ShellInteraction
            | ActivityKind::Edit
            | ActivityKind::Parallel
            | ActivityKind::GenericTool
    )
}

fn prompt_section_label(text: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(" ", Style::default()),
        Span::styled(
            text.to_string(),
            Style::default()
                .fg(theme::SODIUM)
                .add_modifier(Modifier::BOLD),
        ),
    ])
}

fn styled_question_chunk(chunk: &str) -> Vec<Span<'static>> {
    if let Some(rest) = chunk.strip_prefix("Question · ") {
        return vec![
            Span::styled(
                "Question",
                Style::default()
                    .fg(theme::SODIUM)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" · ", Style::default().fg(theme::ASH_MID)),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }

    if let Some(rest) = chunk.strip_prefix("Answer · ") {
        return vec![
            Span::styled(
                "Answer",
                Style::default()
                    .fg(theme::LICHEN)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" · ", Style::default().fg(theme::ASH_MID)),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }

    if let Some((num, rest)) = chunk.split_once(". ")
        && num.chars().all(|ch| ch.is_ascii_digit())
    {
        return vec![
            Span::styled(
                format!("{num}."),
                Style::default()
                    .fg(theme::SODIUM)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }

    vec![Span::styled(chunk.to_string(), theme::assistant_text())]
}

fn render_question_panel<'a>(
    detail_lines: &[String],
    lines: &mut Vec<Line<'a>>,
    viewport_width: usize,
) {
    let title = " QUESTION ".to_string();
    let title_w = UnicodeWidthStr::width(title.as_str());
    let fill_w = viewport_width.saturating_sub(3 + title_w);

    lines.push(Line::from(vec![
        Span::styled("┌─", Style::default().fg(theme::ASH)),
        Span::styled(
            title,
            Style::default()
                .fg(theme::SODIUM)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("─".repeat(fill_w), Style::default().fg(theme::ASH)),
        Span::styled("┐", Style::default().fg(theme::ASH)),
    ]));

    let inner_w = viewport_width.saturating_sub(4).max(1);
    for raw_line in detail_lines {
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
            let styled = styled_question_chunk(&chunk);
            let visible_width: usize = styled
                .iter()
                .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
                .sum();
            let pad = inner_w.saturating_sub(visible_width);
            let mut row = Vec::with_capacity(styled.len() + 3);
            row.push(Span::styled("│ ", Style::default().fg(theme::ASH)));
            row.extend(styled);
            row.push(Span::raw(" ".repeat(pad)));
            row.push(Span::styled(" │", Style::default().fg(theme::ASH)));
            lines.push(Line::from(row));
        }
    }

    let bottom_fill = viewport_width.saturating_sub(2);
    lines.push(Line::from(Span::styled(
        format!("└{}┘", "─".repeat(bottom_fill)),
        Style::default().fg(theme::ASH),
    )));
}

fn prompt_title(prompt: &PromptState) -> &'static str {
    if prompt.options.is_empty() {
        " Question "
    } else if prompt.is_editing_reply() {
        " Question · Write Reply "
    } else {
        " Question · Choose Reply "
    }
}

fn is_code_workflow_block(block: &DisplayBlock) -> bool {
    match block {
        DisplayBlock::CodeBlock { .. } | DisplayBlock::CodeOutput { .. } => true,
        DisplayBlock::Activity(activity) => is_code_workflow_activity(activity.kind.clone()),
        _ => false,
    }
}

fn is_code_workflow_bridge_block(block: &DisplayBlock) -> bool {
    matches!(
        block,
        DisplayBlock::PlanContent(_) | DisplayBlock::PluginPanel(_)
    )
}

fn has_code_workflow_neighbor(blocks: &[DisplayBlock], idx: usize, step: isize) -> bool {
    let mut cursor = idx as isize + step;
    while cursor >= 0 && cursor < blocks.len() as isize {
        let block = &blocks[cursor as usize];
        if is_code_workflow_block(block) {
            return true;
        }
        if is_code_workflow_bridge_block(block) {
            cursor += step;
            continue;
        }
        return false;
    }
    false
}

fn code_workflow_lane(blocks: &[DisplayBlock], idx: usize) -> Option<CodeWorkflowLane> {
    if !is_code_workflow_block(&blocks[idx]) {
        return None;
    }

    Some(CodeWorkflowLane)
}

fn code_workflow_bridge_gutter(
    blocks: &[DisplayBlock],
    idx: usize,
) -> Option<(&'static str, Style)> {
    if !is_code_workflow_bridge_block(&blocks[idx]) {
        return None;
    }

    let prev = has_code_workflow_neighbor(blocks, idx, -1);
    let next = has_code_workflow_neighbor(blocks, idx, 1);
    if prev && next {
        Some(("  ", theme::code_scribe()))
    } else {
        None
    }
}

fn code_workflow_summary_prefix(_lane: CodeWorkflowLane) -> &'static str {
    "· "
}

fn code_workflow_detail_prefix(_lane: CodeWorkflowLane) -> &'static str {
    "  "
}

fn code_workflow_artifact_indent(_lane: CodeWorkflowLane) -> &'static str {
    "  "
}

fn activity_uses_connected_lane(activity: &ActivityBlock, expand_level: u8) -> bool {
    let hide_success_shell_details = matches!(
        activity.kind,
        ActivityKind::ShellCommand | ActivityKind::ShellInteraction
    ) && activity.status != ActivityStatus::Failed;

    let has_detail_rows =
        expand_level >= 1 && !hide_success_shell_details && !activity.detail_lines.is_empty();
    let has_parallel_children = expand_level >= 1
        && activity.kind == ActivityKind::Parallel
        && !activity.children.is_empty();
    let has_delegate_children = expand_level >= 1
        && activity.kind == ActivityKind::Delegate
        && !activity.children.is_empty();
    let has_patch_preview = expand_level == 1
        && matches!(
            activity.artifact,
            Some(ActivityArtifact::PatchPreview { .. })
        );
    let has_expanded_artifact = expand_level >= 2 && activity.artifact.is_some();

    has_detail_rows
        || has_parallel_children
        || has_delegate_children
        || has_patch_preview
        || has_expanded_artifact
}

fn default_activity_lane(activity: &ActivityBlock, nested: bool, expand_level: u8) -> ActivityLane {
    let prefix = if nested { "  " } else { "" };
    let connected = activity_uses_connected_lane(activity, expand_level);
    let (summary_prefix, summary_prefix_style, detail_prefix, detail_prefix_style) =
        match activity.kind {
            ActivityKind::Exploration => (
                format!("{prefix}  "),
                Style::default()
                    .fg(theme::SODIUM)
                    .add_modifier(Modifier::BOLD),
                format!("{prefix}  "),
                Style::default().fg(theme::SODIUM),
            ),
            ActivityKind::Delegate => (
                if connected {
                    format!("{prefix}  ")
                } else {
                    format!("{prefix}◆ ")
                },
                theme::delegate_marker(),
                format!("{prefix}  "),
                theme::delegate_chrome(),
            ),
            ActivityKind::Edit => (
                format!("{prefix}  "),
                if activity.status == ActivityStatus::Failed {
                    theme::error().add_modifier(Modifier::BOLD)
                } else {
                    theme::edit_lane_bold()
                },
                format!("{prefix}  "),
                if activity.status == ActivityStatus::Failed {
                    theme::error()
                } else {
                    theme::edit_lane()
                },
            ),
            ActivityKind::ShellCommand | ActivityKind::ShellInteraction => (
                format!("{}{} ", prefix, activity_icon(activity.status)),
                if activity.status == ActivityStatus::Failed {
                    activity_style(activity.status)
                } else {
                    theme::shell_lane()
                },
                format!("{}  ", prefix),
                theme::shell_lane(),
            ),
            _ => (
                format!("{}{} ", prefix, activity_icon(activity.status)),
                activity_style(activity.status),
                format!("{}  ", prefix),
                theme::code_chrome(),
            ),
        };

    ActivityLane {
        artifact_indent: format!("{prefix}  "),
        parallel_child_prefix: format!("{}    ", prefix),
        summary_prefix,
        summary_prefix_style,
        detail_prefix,
        detail_prefix_style,
    }
}

fn code_workflow_activity_lane(lane: CodeWorkflowLane, kind: &ActivityKind) -> ActivityLane {
    let (summary_style, detail_style) = match kind {
        ActivityKind::Edit => (theme::edit_lane_bold(), theme::edit_lane()),
        ActivityKind::ShellCommand | ActivityKind::ShellInteraction => {
            (theme::shell_lane(), theme::shell_lane())
        }
        _ => (
            theme::code_scribe().add_modifier(Modifier::BOLD),
            theme::code_scribe(),
        ),
    };
    ActivityLane {
        summary_prefix: code_workflow_summary_prefix(lane).to_string(),
        summary_prefix_style: summary_style,
        detail_prefix: code_workflow_detail_prefix(lane).to_string(),
        detail_prefix_style: detail_style,
        artifact_indent: code_workflow_artifact_indent(lane).to_string(),
        parallel_child_prefix: format!("{}  ", code_workflow_artifact_indent(lane)),
    }
}

fn render_activity_block_with_lane<'a>(
    activity: &'a ActivityBlock,
    expand_level: u8,
    lines: &mut Vec<Line<'a>>,
    viewport_width: usize,
    lane: ActivityLane,
) {
    if activity.kind == ActivityKind::Ask && expand_level >= 1 {
        render_question_panel(&activity.detail_lines, lines, viewport_width);
        return;
    }

    let summary_style = if activity.kind == ActivityKind::Exploration {
        if activity.status == ActivityStatus::Failed {
            theme::error().add_modifier(Modifier::BOLD)
        } else {
            theme::explore_label()
        }
    } else if activity.kind == ActivityKind::Edit {
        if activity.status == ActivityStatus::Failed {
            theme::error().add_modifier(Modifier::BOLD)
        } else {
            theme::assistant_text().add_modifier(Modifier::BOLD)
        }
    } else if activity.kind == ActivityKind::Ask {
        Style::default()
            .fg(theme::SODIUM)
            .add_modifier(Modifier::BOLD)
    } else if activity.kind == ActivityKind::Delegate {
        Style::default().fg(theme::CHALK_MID)
    } else {
        activity_style(activity.status)
    };
    let summary_prefix_style = if activity.kind == ActivityKind::Exploration {
        if activity.status == ActivityStatus::Failed {
            theme::error().add_modifier(Modifier::BOLD)
        } else {
            theme::explore_marker()
        }
    } else {
        lane.summary_prefix_style
    };
    let detail_prefix = if activity.kind == ActivityKind::Exploration {
        format!("{} ", lane.detail_prefix)
    } else {
        lane.detail_prefix.clone()
    };
    let detail_prefix_style = if activity.kind == ActivityKind::Exploration {
        if activity.status == ActivityStatus::Failed {
            theme::error()
        } else {
            theme::explore_marker()
        }
    } else {
        lane.detail_prefix_style
    };
    let summary = if let Some(duration_text) =
        crate::util::format_duration_ms_if_visible(activity.duration_ms)
    {
        format!("{} · {}", activity.summary, duration_text)
    } else {
        activity.summary.clone()
    };
    let summary_text = truncate_to_display_width(
        &summary,
        viewport_width.saturating_sub(UnicodeWidthStr::width(lane.summary_prefix.as_str())),
    );
    lines.push(Line::from(vec![
        Span::styled(lane.summary_prefix.clone(), summary_prefix_style),
        Span::styled(summary_text, summary_style),
    ]));

    let hide_success_shell_details = matches!(
        activity.kind,
        ActivityKind::ShellCommand | ActivityKind::ShellInteraction
    ) && activity.status != ActivityStatus::Failed;

    if expand_level >= 1 && !hide_success_shell_details {
        if activity.kind == ActivityKind::Exploration {
            render_exploration_detail_lines(
                lines,
                &activity.detail_lines,
                viewport_width,
                detail_prefix.as_str(),
                detail_prefix_style,
            );
        } else {
            for detail in &activity.detail_lines {
                lines.push(Line::from(vec![
                    Span::styled(detail_prefix.clone(), detail_prefix_style),
                    Span::styled(
                        truncate_to_display_width(
                            detail,
                            viewport_width
                                .saturating_sub(UnicodeWidthStr::width(detail_prefix.as_str())),
                        ),
                        Style::default().fg(theme::ASH_TEXT),
                    ),
                ]));
            }
        }
        if expand_level == 1
            && let Some(ActivityArtifact::PatchPreview { files, .. }) = &activity.artifact
        {
            render_patch_artifact(
                lines,
                files,
                viewport_width,
                lane.artifact_indent.as_str(),
                true,
            );
        }
        if activity.kind == ActivityKind::Parallel {
            for child in &activity.children {
                lines.push(Line::from(vec![
                    Span::styled(lane.parallel_child_prefix.clone(), theme::code_chrome()),
                    Span::styled(child.summary.clone(), activity_style(child.status)),
                ]));
            }
        }
        if activity.kind == ActivityKind::Delegate && !activity.children.is_empty() {
            let children = &activity.children;
            for child in children {
                let connector = "  ";
                let connector_prefix = format!("{}{}", lane.artifact_indent, connector);
                let child_summary = if let Some(duration_text) =
                    crate::util::format_duration_ms_if_visible(child.duration_ms)
                {
                    format!("{} · {}", child.summary, duration_text)
                } else {
                    child.summary.clone()
                };
                let child_text = truncate_to_display_width(
                    &child_summary,
                    viewport_width
                        .saturating_sub(UnicodeWidthStr::width(connector_prefix.as_str())),
                );
                let child_style = if child.status == ActivityStatus::Failed {
                    theme::error()
                } else {
                    theme::delegate_child()
                };
                lines.push(Line::from(vec![
                    Span::styled(connector_prefix, theme::delegate_chrome()),
                    Span::styled(child_text, child_style),
                ]));
            }
        }
    }

    if expand_level >= 2 {
        if let Some(artifact) = &activity.artifact {
            match artifact {
                ActivityArtifact::PatchPreview { files, .. } => {
                    render_patch_artifact(
                        lines,
                        files,
                        viewport_width,
                        lane.artifact_indent.as_str(),
                        true,
                    );
                }
                _ => {
                    render_activity_artifact(
                        lines,
                        artifact,
                        viewport_width,
                        lane.artifact_indent.as_str(),
                    );
                }
            }
        }
        if activity.kind == ActivityKind::Parallel {
            for child in &activity.children {
                if let Some(artifact) = &child.artifact {
                    render_activity_artifact(
                        lines,
                        artifact,
                        viewport_width,
                        lane.artifact_indent.as_str(),
                    );
                }
            }
        }
    }
}

fn render_activity_block<'a>(
    activity: &'a ActivityBlock,
    expand_level: u8,
    lines: &mut Vec<Line<'a>>,
    viewport_width: usize,
    nested: bool,
) {
    render_activity_block_with_lane(
        activity,
        expand_level,
        lines,
        viewport_width,
        default_activity_lane(activity, nested, expand_level),
    );
}

pub fn rendered_block_height(
    blocks: &[DisplayBlock],
    idx: usize,
    expand_level: u8,
    viewport_width: usize,
    viewport_height: usize,
) -> usize {
    let mut lines = Vec::new();
    render_block_into(
        blocks,
        idx,
        expand_level,
        &mut lines,
        viewport_width,
        viewport_height,
    );
    wrap_rendered_lines(&lines, viewport_width).len()
}

fn clone_line_owned(line: &Line<'_>) -> Line<'static> {
    if line.spans.is_empty() {
        return Line::from("");
    }
    Line::from(
        line.spans
            .iter()
            .map(|span| text_display::sanitize_span(span.content.to_string(), span.style))
            .collect::<Vec<_>>(),
    )
}

fn push_wrapped_line(
    wrapped: &mut Vec<Line<'static>>,
    spans: &mut Vec<Span<'static>>,
    segment: &mut String,
    style: &mut Option<Style>,
) {
    if !segment.is_empty() {
        spans.push(Span::styled(
            std::mem::take(segment),
            style.take().unwrap_or_default(),
        ));
    }

    if spans.is_empty() {
        wrapped.push(Line::from(""));
    } else {
        wrapped.push(Line::from(std::mem::take(spans)));
    }
}

fn skip_leading_whitespace(text: &str, mut idx: usize) -> usize {
    while idx < text.len() {
        let Some(ch) = text[idx..].chars().next() else {
            break;
        };
        if !ch.is_whitespace() {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn trim_trailing_whitespace(text: &str, start: usize, end: usize) -> usize {
    let mut trimmed = end;
    while trimmed > start {
        let Some(ch) = text[..trimmed].chars().next_back() else {
            break;
        };
        if !ch.is_whitespace() {
            break;
        }
        trimmed -= ch.len_utf8();
    }
    trimmed
}

fn wrap_text_ranges_wordwise(text: &str, width: usize) -> Vec<(usize, usize)> {
    if width == 0 || text.is_empty() {
        return vec![(0, text.len())];
    }

    let mut wrapped = Vec::new();
    let mut start = 0usize;
    let mut continuation = false;

    'line: while start < text.len() {
        let line_start = if continuation {
            skip_leading_whitespace(text, start)
        } else {
            start
        };
        if line_start >= text.len() {
            break;
        }

        let mut idx = line_start;
        let mut row_width = 0usize;
        let mut last_break = None;
        let mut prev_was_whitespace = false;

        while idx < text.len() {
            let ch = text[idx..]
                .chars()
                .next()
                .expect("slice should start on a char boundary");
            let next_idx = idx + ch.len_utf8();
            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            let is_whitespace = ch.is_whitespace();

            if is_whitespace && !prev_was_whitespace && row_width > 0 {
                last_break = Some(idx);
            }

            if row_width + ch_width > width && row_width > 0 {
                if is_whitespace {
                    let line_end = trim_trailing_whitespace(text, line_start, idx);
                    if line_end > line_start {
                        wrapped.push((line_start, line_end));
                        start = skip_leading_whitespace(text, idx);
                        continuation = true;
                        continue 'line;
                    }
                }
                break;
            }
            row_width += ch_width;
            prev_was_whitespace = is_whitespace;
            idx = next_idx;
        }

        if idx >= text.len() {
            wrapped.push((line_start, text.len()));
            break;
        }

        if let Some(break_idx) = last_break {
            let line_end = trim_trailing_whitespace(text, line_start, break_idx);
            if line_end > line_start {
                wrapped.push((line_start, line_end));
                start = skip_leading_whitespace(text, break_idx);
                continuation = true;
                continue;
            }
        }

        wrapped.push((line_start, idx));
        start = idx;
        continuation = true;
    }

    if wrapped.is_empty() {
        wrapped.push((0, 0));
    }

    wrapped
}

fn wrap_rendered_lines(lines: &[Line<'_>], width: usize) -> Vec<Line<'static>> {
    if width == 0 {
        return lines.iter().map(clone_line_owned).collect();
    }

    let mut wrapped = Vec::with_capacity(lines.len());
    for line in lines {
        if text_display::line_visible_width(line) <= width {
            wrapped.push(clone_line_owned(line));
            continue;
        }

        let mut row = Vec::new();
        let mut segment = String::new();
        let mut segment_style: Option<Style> = None;
        let mut row_width = 0usize;

        for span in &line.spans {
            for ch in span.content.chars() {
                let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
                if row_width + ch_width > width && row_width > 0 {
                    push_wrapped_line(&mut wrapped, &mut row, &mut segment, &mut segment_style);
                    row_width = 0;
                }

                if segment_style != Some(span.style) && !segment.is_empty() {
                    row.push(Span::styled(
                        std::mem::take(&mut segment),
                        segment_style.take().unwrap_or_default(),
                    ));
                }
                segment_style = Some(span.style);
                segment.push(ch);
                row_width += ch_width;
            }
        }

        push_wrapped_line(&mut wrapped, &mut row, &mut segment, &mut segment_style);
    }

    wrapped
}

fn prefix_rendered_lines(lines: &mut [Line<'_>], prefix: &str, style: Style) {
    for line in lines.iter_mut() {
        line.spans
            .insert(0, Span::styled(prefix.to_string(), style));
    }
}

fn styled_user_input_segment<'a>(text: &'a str, seg_start: usize, seg_end: usize) -> Vec<Span<'a>> {
    let mut spans = Vec::new();
    let mut cursor = seg_start;

    for (range, _) in collect_skill_mentions_with_ranges(text) {
        if range.end <= seg_start || range.start >= seg_end {
            continue;
        }
        let start = range.start.max(seg_start);
        let end = range.end.min(seg_end);

        if cursor < start {
            spans.push(Span::styled(
                text[cursor..start].to_string(),
                theme::user_input(),
            ));
        }

        let mention = &text[start..end];
        if start == range.start {
            let mut chars = mention.chars();
            if let Some(sigil) = chars.next() {
                spans.push(Span::styled(
                    sigil.to_string(),
                    theme::resolved_token_sigil(),
                ));
                let rest = chars.as_str();
                if !rest.is_empty() {
                    spans.push(Span::styled(rest.to_string(), theme::resolved_token()));
                }
            }
        } else if !mention.is_empty() {
            spans.push(Span::styled(mention.to_string(), theme::resolved_token()));
        }

        cursor = end;
    }

    if cursor < seg_end {
        spans.push(Span::styled(
            text[cursor..seg_end].to_string(),
            theme::user_input(),
        ));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), theme::user_input()));
    }

    spans
}

fn render_block_into<'a>(
    blocks: &'a [DisplayBlock],
    idx: usize,
    expand_level: u8,
    lines: &mut Vec<Line<'a>>,
    viewport_width: usize,
    viewport_height: usize,
) {
    let block = &blocks[idx];
    match block {
        DisplayBlock::UserInput(text) => {
            // Horizontal rule before user input to separate turns (skip for first block / after Splash)
            if idx > 0 && !matches!(blocks[idx - 1], DisplayBlock::Splash) {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "\u{2500}".repeat(viewport_width),
                    theme::turn_separator(),
                )));
            }
            // Circle marker on first line only, continuation lines get 2-space indent.
            let marker_style = Style::default().fg(theme::SODIUM);
            let prefix_w = 2; // "● " or "  " is 2 columns
            let cap = viewport_width.saturating_sub(prefix_w);
            let mut is_first = true;

            for line in text.lines() {
                let wrapped = if cap == 0 || line.is_empty() {
                    vec![(0, line.len())]
                } else {
                    wrap_text_ranges_wordwise(line, cap)
                };

                for (seg_start, seg_end) in wrapped {
                    let prefix = if is_first {
                        Span::styled("\u{25CF} ", marker_style)
                    } else {
                        Span::raw("  ")
                    };
                    is_first = false;
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
        DisplayBlock::CodeBlock { code, continuation } => {
            let code_lane = code_workflow_lane(blocks, idx);
            match expand_level {
                0 => {
                    if !*continuation {
                        let summary = truncate_to_display_width(
                            &build_code_fold_summary(blocks, idx),
                            viewport_width.saturating_sub(code_lane.map_or(0, |lane| {
                                UnicodeWidthStr::width(code_workflow_summary_prefix(lane))
                            })),
                        );
                        if let Some(lane) = code_lane {
                            lines.push(Line::from(vec![
                                Span::styled(
                                    code_workflow_summary_prefix(lane),
                                    theme::code_scribe().add_modifier(Modifier::BOLD),
                                ),
                                Span::styled(summary, theme::code_header()),
                            ]));
                        } else {
                            lines.push(Line::from(Span::styled(summary, theme::code_header())));
                        }
                    }
                }
                1 => {}
                _ => {
                    // Full: render code with scribe border
                    for line in code.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("\u{2502} ", theme::code_scribe()),
                            Span::styled(line, theme::code_content()),
                        ]));
                    }
                }
            }
        }
        DisplayBlock::Activity(activity) => {
            if let Some(lane) = code_workflow_lane(blocks, idx)
                && is_code_workflow_activity(activity.kind.clone())
            {
                render_activity_block_with_lane(
                    activity,
                    expand_level,
                    lines,
                    viewport_width,
                    code_workflow_activity_lane(lane, &activity.kind),
                );
            } else if let Some((gutter, gutter_style)) = code_workflow_bridge_gutter(blocks, idx) {
                let start = lines.len();
                let gutter_width = UnicodeWidthStr::width(gutter);
                render_activity_block(
                    activity,
                    expand_level,
                    lines,
                    viewport_width.saturating_sub(gutter_width),
                    false,
                );
                prefix_rendered_lines(&mut lines[start..], gutter, gutter_style);
            } else {
                render_activity_block(activity, expand_level, lines, viewport_width, false);
            }
        }
        DisplayBlock::CodeOutput { output, error } => {
            if expand_level >= 2 && !output.is_empty() {
                for line in output.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::code_chrome()),
                        Span::styled(line, theme::system_output()),
                    ]));
                }
            }
            if let Some(err) = error {
                if expand_level >= 2 {
                    lines.push(Line::from(Span::styled(
                        "\u{2716} Execution failed",
                        theme::error(),
                    )));
                    for line in err.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("\u{2502} ", theme::code_chrome()),
                            Span::styled(line, theme::error()),
                        ]));
                    }
                } else {
                    // Level 0 and 1: show first meaningful error line only
                    let summary = err
                        .lines()
                        .find(|l| {
                            !l.trim().is_empty()
                                && !l.trim_start().starts_with("File ")
                                && !l.trim_start().starts_with("Traceback")
                        })
                        .unwrap_or(err.lines().next().unwrap_or("error"));
                    lines.push(Line::from(Span::styled(
                        format!("\u{2716} Execution failed: {}", summary.trim()),
                        theme::error(),
                    )));
                }
            }
        }
        DisplayBlock::ShellOutput {
            command,
            output,
            error,
        } => {
            lines.push(Line::from(Span::styled(
                format!("$ {}", command),
                theme::code_chrome(),
            )));
            if !output.is_empty() {
                for line in output.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::code_chrome()),
                        Span::styled(line, theme::system_output()),
                    ]));
                }
            }
            if let Some(err) = error {
                for line in err.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::code_chrome()),
                        Span::styled(line, theme::error()),
                    ]));
                }
            }
        }
        DisplayBlock::Error(msg) => {
            let title = " ERROR ".to_string();
            let title_w = UnicodeWidthStr::width(title.as_str());
            let fill_w = viewport_width.saturating_sub(3 + title_w);

            lines.push(Line::from(vec![
                Span::styled("\u{250c}\u{2500}", theme::error_border()),
                Span::styled(title, theme::error_title()),
                Span::styled("\u{2500}".repeat(fill_w), theme::error_border()),
                Span::styled("\u{2510}", theme::error_border()),
            ]));

            let inner_w = viewport_width.saturating_sub(4).max(1);
            for raw_line in msg.lines() {
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
                    let pad = inner_w.saturating_sub(UnicodeWidthStr::width(chunk.as_str()));
                    lines.push(Line::from(vec![
                        Span::styled("\u{2502} ", theme::error_border()),
                        Span::styled(chunk, theme::error()),
                        Span::raw(" ".repeat(pad)),
                        Span::styled(" \u{2502}", theme::error_border()),
                    ]));
                }
            }

            let bottom_fill = viewport_width.saturating_sub(2);
            lines.push(Line::from(Span::styled(
                format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(bottom_fill)),
                theme::error_border(),
            )));
        }
        DisplayBlock::SystemMessage(text) => {
            for line in text.lines() {
                lines.push(Line::from(Span::styled(line, theme::system_message())));
            }
        }
        DisplayBlock::PlanContent(content) => {
            render_plan_block(
                content,
                lines,
                viewport_width,
                code_workflow_bridge_gutter(blocks, idx),
            );
        }
        DisplayBlock::PluginPanel(panel) => {
            render_panel_block(
                &panel.title,
                &panel.content,
                lines,
                viewport_width,
                code_workflow_bridge_gutter(blocks, idx),
            );
        }
        DisplayBlock::Splash => {
            let chalk = theme::assistant_text();
            let sodium = Style::default().fg(theme::SODIUM);

            let content_width = 30;
            let content_height = SPLASH_CONTENT_HEIGHT;
            let cx = viewport_width.saturating_sub(content_width) / 2;
            let fullscreen = blocks.len() == 1;
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
                    Span::styled(format!("{}{}", pad, before), chalk),
                    Span::styled("██", sodium),
                    Span::styled(after, chalk),
                ]));
            }
            lines.push(Line::from(Span::styled(
                format!("{}                   ██", pad),
                sodium,
            )));
            lines.push(Line::from(vec![
                Span::styled(format!("{}──────────", pad), sodium),
                Span::styled("──────────", Style::default().fg(theme::ASH_MID)),
                Span::styled("──────────", Style::default().fg(theme::ASH)),
            ]));
            lines.push(Line::from(""));

            let target_height = if fullscreen {
                viewport_height
            } else {
                SPLASH_SCROLLBACK_HEIGHT
            };
            let rendered = cy + content_height;
            for _ in rendered..target_height {
                lines.push(Line::from(""));
            }
        }
    }
}

fn render_block<'a>(
    blocks: &'a [DisplayBlock],
    idx: usize,
    expand_level: u8,
    lines: &mut Vec<Line<'a>>,
    viewport_width: usize,
    viewport_height: usize,
) {
    render_block_into(
        blocks,
        idx,
        expand_level,
        lines,
        viewport_width,
        viewport_height,
    );
}

/// Render plan content as a bordered block with markdown inside.
fn render_plan_block<'a>(
    content: &'a str,
    lines: &mut Vec<Line<'a>>,
    viewport_width: usize,
    gutter: Option<(&'static str, Style)>,
) {
    let start = lines.len();
    let gutter_width = gutter.map_or(0, |(prefix, _)| UnicodeWidthStr::width(prefix));
    let viewport_width = viewport_width.saturating_sub(gutter_width);
    let title = " PLAN ".to_string();
    let title_w = UnicodeWidthStr::width(title.as_str());
    let fill_w = viewport_width.saturating_sub(3 + title_w);

    lines.push(Line::from(vec![
        Span::styled("\u{250c}\u{2500}", Style::default().fg(theme::ASH)),
        Span::styled(
            title,
            Style::default()
                .fg(theme::SODIUM)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("\u{2500}".repeat(fill_w), Style::default().fg(theme::ASH)),
        Span::styled("\u{2510}", Style::default().fg(theme::ASH)),
    ]));

    let inner_w = viewport_width.saturating_sub(4).max(1);

    // Breathing room after top border
    let pad_line = Line::from(vec![
        Span::styled("\u{2502} ", Style::default().fg(theme::ASH)),
        Span::raw(" ".repeat(inner_w)),
        Span::styled(" \u{2502}", Style::default().fg(theme::ASH)),
    ]);
    lines.push(pad_line.clone());

    for raw_line in content.lines() {
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
            let styled = styled_plan_chunk(&chunk);
            let visible_width: usize = styled
                .iter()
                .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
                .sum();
            let pad = inner_w.saturating_sub(visible_width);
            let mut row = Vec::with_capacity(styled.len() + 3);
            row.push(Span::styled("\u{2502} ", Style::default().fg(theme::ASH)));
            row.extend(styled);
            row.push(Span::raw(" ".repeat(pad)));
            row.push(Span::styled(" \u{2502}", Style::default().fg(theme::ASH)));
            lines.push(Line::from(row));
        }
    }

    // Breathing room before bottom border
    lines.push(Line::from(vec![
        Span::styled("\u{2502} ", Style::default().fg(theme::ASH)),
        Span::raw(" ".repeat(inner_w)),
        Span::styled(" \u{2502}", Style::default().fg(theme::ASH)),
    ]));

    let bottom_fill = viewport_width.saturating_sub(2);
    lines.push(Line::from(Span::styled(
        format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(bottom_fill)),
        Style::default().fg(theme::ASH),
    )));

    if let Some((prefix, style)) = gutter {
        prefix_rendered_lines(&mut lines[start..], prefix, style);
    }
}

fn styled_plan_chunk(chunk: &str) -> Vec<Span<'static>> {
    let Some(rest) = chunk.strip_prefix("\u{2713} ") else {
        if let Some(rest) = chunk.strip_prefix("\u{25b8} ") {
            return vec![
                Span::styled("\u{25b8}", theme::plan_active_marker()),
                Span::raw(" "),
                Span::styled(rest.to_string(), theme::assistant_text()),
            ];
        };
        if let Some(rest) = chunk.strip_prefix("\u{25cb} ") {
            return vec![
                Span::styled("\u{25cb}", theme::plan_pending_marker()),
                Span::raw(" "),
                Span::styled(rest.to_string(), theme::assistant_text()),
            ];
        };
        return vec![Span::styled(chunk.to_string(), theme::assistant_text())];
    };

    vec![
        Span::styled("\u{2713}", theme::plan_done_marker()),
        Span::raw(" "),
        Span::styled(rest.to_string(), theme::assistant_text()),
    ]
}

fn render_panel_block<'a>(
    title_text: &'a str,
    content: &'a str,
    lines: &mut Vec<Line<'a>>,
    viewport_width: usize,
    gutter: Option<(&'static str, Style)>,
) {
    let start = lines.len();
    let gutter_width = gutter.map_or(0, |(prefix, _)| UnicodeWidthStr::width(prefix));
    let viewport_width = viewport_width.saturating_sub(gutter_width);
    let title = format!(" {} ", title_text);
    let title_w = UnicodeWidthStr::width(title.as_str());
    let fill_w = viewport_width.saturating_sub(3 + title_w); // ┌─ + title + ┐

    // Top border: ┌─ PLAN ─────────────────────────┐
    lines.push(Line::from(vec![
        Span::styled("\u{250c}\u{2500}", Style::default().fg(theme::ASH)),
        Span::styled(
            title,
            Style::default()
                .fg(theme::SODIUM)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("\u{2500}".repeat(fill_w), Style::default().fg(theme::ASH)),
        Span::styled("\u{2510}", Style::default().fg(theme::ASH)),
    ]));

    // Content: render markdown, hard-wrap each rendered line to fit inside the box.
    // We draw both left and right borders per visual row.
    let inner_w = viewport_width.saturating_sub(4); // "│ " + content + " │"
    let md_lines = markdown::render_markdown(content, inner_w.max(1));
    for line in md_lines {
        let raw = line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>();

        let segments = if raw.is_empty() {
            vec![(0usize, 0usize)]
        } else {
            wrap_line(&raw, 0, 0, inner_w.max(1))
        };

        for (start, end) in segments {
            let chunk = if raw.is_empty() {
                String::new()
            } else {
                truncate_to_display_width(&raw[start..end], inner_w)
            };
            let pad = inner_w.saturating_sub(UnicodeWidthStr::width(chunk.as_str()));
            lines.push(Line::from(vec![
                Span::styled("\u{2502} ", Style::default().fg(theme::ASH)),
                Span::styled(chunk, theme::assistant_text()),
                Span::raw(" ".repeat(pad)),
                Span::styled(" \u{2502}", Style::default().fg(theme::ASH)),
            ]));
        }
    }

    // Bottom border: └──────────────────────────────┘
    let bottom_fill = viewport_width.saturating_sub(2); // minus └ and ┘
    lines.push(Line::from(Span::styled(
        format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(bottom_fill)),
        Style::default().fg(theme::ASH),
    )));

    if let Some((prefix, style)) = gutter {
        prefix_rendered_lines(&mut lines[start..], prefix, style);
    }
}

fn draw_turn_status(frame: &mut Frame, app: &App, area: Rect) {
    let Some(turn) = app.live_turn.as_ref() else {
        return;
    };
    if area.height == 0 {
        return;
    };
    frame.render_widget(Block::default().style(theme::turn_status_bar()), area);

    let brand = animated_lash_word(turn.turn_started_at.elapsed());
    let (label, label_style) = if turn.status_text == "error" {
        ("Error", theme::error().add_modifier(Modifier::BOLD))
    } else {
        ("Working", theme::turn_status_state())
    };
    let mut spans = Vec::new();
    spans.extend(brand);
    spans.push(Span::raw("  "));
    spans.push(Span::styled(label, label_style));
    if let Some(elapsed_text) = crate::util::format_duration_ms_if_visible(
        turn.turn_started_at.elapsed().as_millis() as u64,
    ) {
        spans.push(Span::raw("    "));
        spans.push(Span::styled(elapsed_text, theme::turn_status_elapsed()));
    }

    let status_line = Paragraph::new(Line::from(spans))
        .style(theme::turn_status_bar())
        .alignment(ratatui::layout::Alignment::Center);
    frame.render_widget(status_line, area);
}

fn animated_lash_word(elapsed: std::time::Duration) -> Vec<Span<'static>> {
    let frame = ((elapsed.as_millis() / 180) % 5) as usize;
    let glyphs = match frame {
        0 => vec!['/', 'L', 'A', 'S', 'H'],
        1 => vec!['L', '/', 'A', 'S', 'H'],
        2 => vec!['L', 'A', '/', 'S', 'H'],
        3 => vec!['L', 'A', 'S', '/', 'H'],
        _ => vec!['L', 'A', 'S', 'H', '/'],
    };

    glyphs
        .into_iter()
        .map(|glyph| {
            if glyph == '/' {
                Span::styled(glyph.to_string(), theme::turn_status_slash())
            } else {
                Span::styled(glyph.to_string(), theme::turn_status_brand())
            }
        })
        .collect()
}

const QUEUE_SECTION_ITEM_LIMIT: usize = 2;
const QUEUE_SECTION_WRAP_LIMIT: usize = 2;

fn queue_preview_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    if !app.has_queued_messages() || width < 12 {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let inner_width = width as usize;
    let pending_previews: Vec<String> = app
        .pending_steers
        .iter()
        .map(PreparedTurn::preview)
        .collect();
    let queued_previews: Vec<String> = app.queued_turns.iter().map(PreparedTurn::preview).collect();

    if !pending_previews.is_empty() {
        push_queue_section(
            &mut lines,
            inner_width,
            "◆ after next tool/result",
            &pending_previews,
            Style::default().fg(theme::SODIUM),
            Style::default().fg(theme::CHALK_MID),
        );
    }

    if !queued_previews.is_empty() {
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }
        push_queue_section(
            &mut lines,
            inner_width,
            "◇ next full turn",
            &queued_previews,
            Style::default().fg(theme::LICHEN),
            Style::default().fg(theme::CHALK_DIM),
        );
    }

    lines
}

fn push_queue_section(
    lines: &mut Vec<Line<'static>>,
    width: usize,
    title: &str,
    items: &[String],
    header_style: Style,
    item_style: Style,
) {
    let header = format!(
        "{}{}",
        title,
        if items.len() > 1 {
            format!(" · {}", items.len())
        } else {
            String::new()
        }
    );
    for (start, end) in wrap_line(&header, 0, 0, width.max(1)) {
        lines.push(Line::from(Span::styled(
            header[start..end].to_string(),
            header_style,
        )));
    }

    for item in items.iter().take(QUEUE_SECTION_ITEM_LIMIT) {
        push_wrapped_queue_item(
            lines,
            width,
            item,
            item_style,
            "  \u{21b3} ",
            "    ",
            QUEUE_SECTION_WRAP_LIMIT,
        );
    }

    if items.len() > QUEUE_SECTION_ITEM_LIMIT {
        lines.push(Line::from(vec![
            Span::styled("    +", Style::default().fg(theme::ASH)),
            Span::styled(
                format!("{}", items.len() - QUEUE_SECTION_ITEM_LIMIT),
                item_style,
            ),
            Span::styled(" more", Style::default().fg(theme::ASH_TEXT)),
        ]));
    }
}

fn push_wrapped_queue_item(
    lines: &mut Vec<Line<'static>>,
    width: usize,
    text: &str,
    style: Style,
    first_prefix: &str,
    continuation_prefix: &str,
    max_lines: usize,
) {
    let collapsed = text.replace('\n', " ");
    let segments = wrap_line(
        &collapsed,
        first_prefix.width(),
        continuation_prefix.width(),
        width.max(1),
    );
    for (idx, (start, end)) in segments.into_iter().take(max_lines).enumerate() {
        let prefix = if idx == 0 {
            first_prefix
        } else {
            continuation_prefix
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(theme::ASH)),
            text_display::sanitize_span(collapsed[start..end].to_string(), style),
        ]));
    }
    if wrap_line(
        &collapsed,
        first_prefix.width(),
        continuation_prefix.width(),
        width.max(1),
    )
    .len()
        > max_lines
    {
        lines.push(Line::from(vec![Span::styled(
            format!("{continuation_prefix}\u{2026}"),
            Style::default().fg(theme::ASH_TEXT),
        )]));
    }
}

/// Draw the pending-input preview block above the input.
fn draw_queue_preview(frame: &mut Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }
    let lines = queue_preview_lines(app, area.width);
    if lines.is_empty() {
        return;
    }

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(theme::FORM_RAISED)),
        area,
    );
}

/// Build the session/cwd badge as a Line suitable for `Block::title_bottom`.
fn build_input_badge(app: &App) -> Option<Line<'static>> {
    let badge_labels: Vec<&str> = app
        .plugin_mode_indicators
        .values()
        .map(|label| label.as_str())
        .collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    if !badge_labels.is_empty() {
        for (idx, label) in badge_labels.iter().enumerate() {
            if idx > 0 {
                spans.push(Span::styled(" \u{b7} ", Style::default().fg(theme::ASH)));
            }
            spans.push(Span::styled(
                (*label).to_string(),
                Style::default()
                    .fg(theme::SODIUM)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        spans.push(Span::styled(" \u{b7} ", Style::default().fg(theme::ASH)));
    }
    let location_label = app
        .repo_status
        .as_ref()
        .map(|repo| format!("{} · {}", repo.repo_name, repo.display_ref()))
        .unwrap_or_else(|| app.cwd.clone());
    spans.extend([
        Span::styled(
            app.session_name.clone(),
            Style::default().fg(theme::ASH_MID),
        ),
        Span::styled(" \u{b7} ", Style::default().fg(theme::ASH)),
        text_display::sanitize_span(location_label, Style::default().fg(theme::ASH_TEXT)),
    ]);
    Some(Line::from(spans))
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let mut lines = Vec::new();
    let image_markers = image_marker_ranges(app.input());

    // Multi-line input rendering with manual character-level wrapping.
    // We pre-wrap each logical line using wrap_line() so that rendering and
    // cursor positioning use identical wrapping logic (no Paragraph::wrap).
    let total_width = padded_content_width(area.width, INPUT_HORIZONTAL_PADDING);
    let prefix_w = 2; // "❯ " or "  "
    let input_lines: Vec<&str> = app.input().split('\n').collect();
    for (i, logical_line) in input_lines.iter().enumerate() {
        let segments = wrap_line(logical_line, prefix_w, prefix_w, total_width);
        for (j, &(seg_start, seg_end)) in segments.iter().enumerate() {
            let seg_spans = styled_input_segment(logical_line, seg_start, seg_end, &image_markers);
            if j == 0 {
                // First visual line of this logical line — gets prefix
                if i == 0 {
                    let mut spans = vec![Span::styled(
                        format!("{} ", theme::PROMPT_CHAR),
                        theme::prompt(),
                    )];
                    spans.extend(seg_spans);
                    lines.push(Line::from(spans));
                } else {
                    let mut spans = vec![Span::styled("  ", Style::default().fg(theme::ASH))];
                    spans.extend(seg_spans);
                    lines.push(Line::from(spans));
                }
            } else {
                // Wrapped continuation line — align with the text column.
                let mut spans = vec![Span::styled("  ", Style::default().fg(theme::ASH))];
                spans.extend(seg_spans);
                lines.push(Line::from(spans));
            }
        }
    }

    // Position cursor accounting for visual wrapping
    let clamped_cursor = app.cursor_pos().min(app.input().len());
    let (vis_row, vis_col) = input_cursor_position(app.input(), clamped_cursor, total_width);
    let cursor_abs_row = vis_row;
    let content_h = area.height.saturating_sub(2) as usize; // inside borders

    // Compute scroll offset to keep the cursor visible
    let scroll_offset = if content_h > 0 && cursor_abs_row >= content_h {
        cursor_abs_row - content_h + 1
    } else {
        0
    };

    // Build the bottom badge as a proper title on the input border
    let badge_line = build_input_badge(app);
    let mut block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(theme::input_border())
        .padding(Padding::horizontal(INPUT_HORIZONTAL_PADDING));
    if let Some(badge) = badge_line {
        block = block.title_bottom(badge.right_aligned());
    }
    let input = Paragraph::new(lines)
        .block(block)
        .scroll((scroll_offset as u16, 0));
    frame.render_widget(input, area);

    let cursor_x = area.x + INPUT_HORIZONTAL_PADDING + vis_col as u16;
    let cursor_y = area.y + 1 + (cursor_abs_row - scroll_offset) as u16;
    frame.set_cursor_position((cursor_x, cursor_y));
}

fn styled_input_segment<'a>(
    logical_line: &'a str,
    seg_start: usize,
    seg_end: usize,
    image_markers: &[(std::ops::Range<usize>, usize)],
) -> Vec<Span<'a>> {
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

/// Draw the autocomplete popup above the input area (slash commands or @path).
fn draw_suggestions(frame: &mut Frame, app: &App, input_area: Rect) {
    if app.suggestions().is_empty() {
        return;
    }

    let is_path = app.suggestion_kind() == SuggestionKind::Path;
    let count = app.suggestions().len() as u16;
    let desired_height = count + 2; // +2 for top/bottom border

    // Compute name column width from actual suggestions (+2 for padding)
    let name_col: usize = app
        .suggestions()
        .iter()
        .map(|(cmd, _)| UnicodeWidthStr::width(cmd.as_str()))
        .max()
        .unwrap_or(8)
        + 2;

    // Size popup to fit content: prefix(1) + name_col + gap(1) + desc(~20) + border(2)
    let min_desc_w: u16 = 16;
    let base_width: u16 = (name_col as u16 + min_desc_w + 3).max(if is_path { 50 } else { 40 });
    let width = base_width.min(input_area.width);

    // Suggestions should live in the gap directly above the input, never in the
    // top status-bar row or outside the frame.
    let top_bound = input_area
        .y
        .min(frame.area().bottom())
        .max(frame.area().y + 1);
    let available_height = top_bound.saturating_sub(frame.area().y + 1);
    let height = desired_height.min(available_height);
    if height == 0 || width == 0 {
        return;
    }
    let y = top_bound.saturating_sub(height);

    let popup_area = Rect::new(input_area.x, y, width, height);

    // Clear the area behind the popup
    frame.render_widget(Clear, popup_area);

    let items: Vec<Line> = app
        .suggestions()
        .iter()
        .enumerate()
        .map(|(i, (cmd, desc))| {
            if i == app.suggestion_idx() {
                Line::from(vec![
                    Span::styled(
                        format!(" {:<w$}", cmd, w = name_col),
                        Style::default()
                            .fg(theme::SODIUM)
                            .bg(theme::FORM_RAISED)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" {}", desc),
                        Style::default().fg(theme::CHALK_MID).bg(theme::FORM_RAISED),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled(
                        format!(" {:<w$}", cmd, w = name_col),
                        Style::default().fg(theme::CHALK_DIM),
                    ),
                    text_display::sanitize_span(
                        format!(" {}", desc),
                        Style::default().fg(theme::ASH_MID),
                    ),
                ])
            }
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ASH));
    let paragraph = Paragraph::new(items)
        .block(block)
        .style(Style::default().bg(theme::FORM_DEEP));
    frame.render_widget(paragraph, popup_area);
}

/// Draw the session picker popup centered in the history area.
fn draw_session_picker(frame: &mut Frame, app: &App, history_area: Rect) {
    let Some(session_picker) = app.session_picker_state() else {
        return;
    };

    let max_visible = 15u16;
    let count = session_picker.items.len() as u16;
    let visible = count.min(max_visible);
    let desired_height = visible + 2; // +2 for borders
    let width = 80u16.min(history_area.width.saturating_sub(4));
    if width == 0 || history_area.height == 0 {
        return;
    }

    // Center in the history pane and keep the popup inside that pane.
    let x = history_area.x + (history_area.width.saturating_sub(width)) / 2;
    let y = history_area.y + (history_area.height.saturating_sub(desired_height)) / 2;
    let height = desired_height.min(history_area.bottom().saturating_sub(y));
    if height == 0 {
        return;
    }
    let popup_area = Rect::new(x, y, width, height);

    frame.render_widget(Clear, popup_area);

    // If we have more items than visible, scroll the view to keep selection visible
    let scroll_offset = if session_picker.selected >= visible as usize {
        session_picker.selected - visible as usize + 1
    } else {
        0
    };

    // Layout: " > 8h ago  First user message truncated...  5 msgs "
    let time_col = 10; // fixed width for relative time
    let count_col = 8; // fixed width for "99 msgs"
    // inner_w = width - 2 (borders) - 3 (prefix " > " or "   ") - 2 (gaps around message)
    let inner_w = (width as usize).saturating_sub(2 + 3 + time_col + count_col + 2);

    let items: Vec<Line> = session_picker
        .items
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(visible as usize)
        .map(|(i, session)| {
            let selected = i == session_picker.selected;
            let prefix = if selected { " > " } else { "   " };

            let time_str = session.relative_time();
            let msg_count = format!("{} msgs", session.message_count);

            // Truncate the first message to fit, collapsing newlines.
            // Keep the session directory label visible at the end when present.
            let preview_base: String = session
                .first_message
                .chars()
                .map(|c| if c == '\n' { ' ' } else { c })
                .collect();
            let preview = if let Some(cwd_label) = session.cwd_label() {
                truncate_with_suffix(&preview_base, &format!(" {cwd_label}"), inner_w)
            } else {
                truncate_to_display_width(&preview_base, inner_w)
            };

            let (time_style, msg_style, count_style, prefix_style) = if selected {
                let bg = theme::FORM_RAISED;
                (
                    Style::default().fg(theme::SODIUM).bg(bg),
                    Style::default().fg(theme::CHALK).bg(bg),
                    Style::default().fg(theme::ASH_MID).bg(bg),
                    Style::default()
                        .fg(theme::SODIUM)
                        .bg(bg)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    Style::default().fg(theme::ASH_MID),
                    Style::default().fg(theme::CHALK_DIM),
                    Style::default().fg(theme::ASH),
                    Style::default().fg(theme::CHALK_DIM),
                )
            };

            Line::from(vec![
                Span::styled(prefix, prefix_style),
                Span::styled(format!("{:<tw$}", time_str, tw = time_col), time_style),
                text_display::sanitize_span(format!("{:<mw$}", preview, mw = inner_w), msg_style),
                Span::styled(format!("{:>cw$} ", msg_count, cw = count_col), count_style),
            ])
        })
        .collect();

    let title = format!(" Sessions ({}) ", session_picker.items.len());
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(theme::SODIUM)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ASH));
    let paragraph = Paragraph::new(items)
        .block(block)
        .style(Style::default().bg(theme::FORM_DEEP));
    frame.render_widget(paragraph, popup_area);
}

/// Draw the skill picker popup centered in the history area.
fn draw_skill_picker(frame: &mut Frame, app: &App, history_area: Rect) {
    let Some(skill_picker) = app.skill_picker_state() else {
        return;
    };

    let max_visible = 15u16;
    let count = skill_picker.items.len() as u16;
    let visible = count.min(max_visible);
    let desired_height = visible + 2; // +2 for borders
    let width = 60u16.min(history_area.width.saturating_sub(4));
    if width == 0 || history_area.height == 0 {
        return;
    }

    // Center in the history pane and keep the popup inside that pane.
    let x = history_area.x + (history_area.width.saturating_sub(width)) / 2;
    let y = history_area.y + (history_area.height.saturating_sub(desired_height)) / 2;
    let height = desired_height.min(history_area.bottom().saturating_sub(y));
    if height == 0 {
        return;
    }
    let popup_area = Rect::new(x, y, width, height);

    frame.render_widget(Clear, popup_area);

    // Scroll to keep selection visible
    let scroll_offset = if skill_picker.selected >= visible as usize {
        skill_picker.selected - visible as usize + 1
    } else {
        0
    };

    let name_col = 20usize;
    let desc_w = (width as usize).saturating_sub(2 + 3 + name_col + 1); // borders + prefix + name + gap

    let items: Vec<Line> = skill_picker
        .items
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(visible as usize)
        .map(|(i, (name, desc))| {
            let selected = i == skill_picker.selected;
            let prefix = if selected { " > " } else { "   " };

            let desc_truncated: String = desc.chars().take(desc_w).collect();

            let (name_style, desc_style, prefix_style) = if selected {
                let bg = theme::FORM_RAISED;
                (
                    Style::default()
                        .fg(theme::SODIUM)
                        .bg(bg)
                        .add_modifier(Modifier::BOLD),
                    Style::default().fg(theme::CHALK).bg(bg),
                    Style::default()
                        .fg(theme::SODIUM)
                        .bg(bg)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    Style::default().fg(theme::CHALK_DIM),
                    Style::default().fg(theme::ASH_MID),
                    Style::default().fg(theme::CHALK_DIM),
                )
            };

            Line::from(vec![
                Span::styled(prefix, prefix_style),
                Span::styled(format!("{:<w$}", name, w = name_col), name_style),
                Span::styled(format!(" {}", desc_truncated), desc_style),
            ])
        })
        .collect();

    let title = format!(" Skills ({}) ", skill_picker.items.len());
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(theme::SODIUM)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ASH));
    let paragraph = Paragraph::new(items)
        .block(block)
        .style(Style::default().bg(theme::FORM_DEEP));
    frame.render_widget(paragraph, popup_area);
}

/// Calculate the height needed for the inline prompt area.
fn prompt_height(app: &App, frame_width: u16) -> u16 {
    let prompt = match app.prompt_state() {
        Some(p) => p,
        None => return 3, // fallback
    };

    let inner_w = prompt_inner_width(frame_width);
    let content_h = prompt_content_lines(prompt, inner_w).len() as u16;
    (content_h + 2).min(MAX_INPUT_HEIGHT) // +2 for borders
}

fn push_wrapped_prefixed_lines(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    total_width: usize,
    first_prefix: Span<'static>,
    cont_prefix: Span<'static>,
    text_style: Style,
) {
    let first_prefix_width = first_prefix.width();
    let cont_prefix_width = cont_prefix.width();
    let mut first_visual_line = true;

    for logical_line in text.split('\n') {
        let prefix_width = if first_visual_line {
            first_prefix_width
        } else {
            cont_prefix_width
        };
        let segments = wrap_line(logical_line, prefix_width, cont_prefix_width, total_width);

        for (segment_idx, (start, end)) in segments.into_iter().enumerate() {
            let prefix = if first_visual_line && segment_idx == 0 {
                first_prefix.clone()
            } else {
                cont_prefix.clone()
            };
            lines.push(Line::from(vec![
                prefix,
                text_display::sanitize_span(logical_line[start..end].to_string(), text_style),
            ]));
            first_visual_line = false;
        }
    }
}

fn prompt_input_text(prompt: &PromptState) -> String {
    let mut display = prompt.reply_text.clone();
    let cursor = prompt.reply_cursor.min(display.len());
    display.insert(cursor, '\u{2588}');
    display
}

fn prompt_content_lines(prompt: &PromptState, inner_w: usize) -> Vec<Line<'static>> {
    let has_options = prompt.has_options();
    let show_text_input = prompt.is_editing_reply();
    let mut lines: Vec<Line<'static>> = Vec::new();

    if !prompt.question.is_empty() {
        let md_lines = markdown::render_markdown(&prompt.question, inner_w);
        for md_line in md_lines {
            lines.push(md_line);
        }
    }

    if !lines.is_empty() && (has_options || show_text_input) {
        lines.push(Line::from(""));
    }

    if has_options {
        lines.push(prompt_section_label("Choices"));
        for (idx, opt) in prompt.options.iter().enumerate() {
            let selected = prompt.selected_option_idx() == Some(idx);
            let text_style = if selected {
                Style::default()
                    .fg(theme::SODIUM)
                    .bg(theme::FORM_RAISED)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::CHALK_DIM)
            };
            let prefix_style = if selected {
                Style::default().fg(theme::SODIUM).bg(theme::FORM_RAISED)
            } else {
                text_style
            };
            push_wrapped_prefixed_lines(
                &mut lines,
                &format!("{}. {}", idx + 1, opt),
                inner_w,
                Span::styled(if selected { " \u{203a} " } else { "   " }, prefix_style),
                Span::styled("   ", prefix_style),
                text_style,
            );
        }

        let other_selected = prompt.selects_custom_reply();
        let other_style = if other_selected {
            Style::default()
                .fg(theme::SODIUM)
                .bg(theme::FORM_RAISED)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::CHALK_DIM)
        };
        let other_prefix_style = if other_selected {
            Style::default().fg(theme::SODIUM).bg(theme::FORM_RAISED)
        } else {
            other_style
        };
        push_wrapped_prefixed_lines(
            &mut lines,
            "Write my own answer",
            inner_w,
            Span::styled(
                if other_selected { " \u{203a} " } else { "   " },
                other_prefix_style,
            ),
            Span::styled("   ", other_prefix_style),
            other_style,
        );
    }

    if has_options && show_text_input {
        lines.push(Line::from(""));
    }

    if show_text_input {
        if has_options {
            lines.push(prompt_section_label("Reply"));
        }
        push_wrapped_prefixed_lines(
            &mut lines,
            &prompt_input_text(prompt),
            inner_w,
            Span::styled(format!(" {} ", theme::PROMPT_CHAR), theme::prompt()),
            Span::styled("   ", Style::default().fg(theme::ASH)),
            Style::default().fg(theme::CHALK),
        );
    }

    if !lines.is_empty() {
        lines.push(Line::from(""));
    }

    let help_text = if has_options {
        if show_text_input {
            "Enter submit  Shift+Tab newline  Tab choices  Esc cancel"
        } else {
            "↑↓ choose  Enter submit  Tab write reply  Esc cancel"
        }
    } else {
        "Enter submit  Shift+Tab newline  Esc cancel"
    };
    push_wrapped_prefixed_lines(
        &mut lines,
        help_text,
        inner_w,
        Span::styled(" ", Style::default().fg(theme::ASH)),
        Span::styled(" ", Style::default().fg(theme::ASH)),
        Style::default().fg(theme::ASH),
    );

    lines
}

/// Draw the agent prompt inline in the input area (replaces normal input).
fn draw_prompt(frame: &mut Frame, app: &App, area: Rect) {
    let prompt = match app.prompt_state() {
        Some(p) => p,
        None => return,
    };

    let inner_w = prompt_inner_width(area.width);
    let lines = prompt_content_lines(prompt, inner_w);

    let content_h = area.height.saturating_sub(2) as usize;
    let scroll_offset = if lines.len() > content_h {
        lines.len() - content_h
    } else {
        0
    };

    let title = prompt_title(prompt);
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(theme::SODIUM)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ASH_LIGHT))
        .padding(Padding::horizontal(PROMPT_HORIZONTAL_PADDING));
    let paragraph = Paragraph::new(lines)
        .block(block)
        .style(Style::default().bg(theme::FORM_DEEP))
        .wrap(Wrap { trim: false })
        .scroll((scroll_offset as u16, 0));
    frame.render_widget(paragraph, area);
}

/// Wrap a single logical line into visual line segments, returning byte-offset ranges.
/// `prefix_width` columns are reserved on the first visual line (for the prompt prefix).
/// Continuation (wrapped) visual lines use the full `total_width`.
/// Uses character-level wrapping with proper Unicode width calculation.
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
    let mut line_start = 0;
    let mut col = 0;
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

/// Count total visual lines the input text occupies, accounting for wrapping.
/// `width` is the total rendering width (prefix is accounted for internally).
fn input_visual_lines(input: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let prefix_w = 2; // "❯ " or "  "
    let mut total = 0;
    for line in input.split('\n') {
        total += wrap_line(line, prefix_w, prefix_w, width).len();
    }
    total.max(1)
}

/// Compute the visual (row, col) of the cursor in the input, accounting for wrapping.
/// `full_width` is the total rendering width; the 2-column prefix is handled internally.
fn input_cursor_position(input: &str, cursor_pos: usize, full_width: usize) -> (usize, usize) {
    let prefix_w = 2usize; // "❯ " or "  "
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
                    let col = UnicodeWidthStr::width(text_before);
                    return (vis_row, col + prefix_w);
                }
                vis_row += 1;
            }
            return (vis_row, prefix_w);
        }

        vis_row += segments.len();
        byte_offset = line_end + 1; // +1 for '\n'
    }

    (vis_row.saturating_sub(1), prefix_w)
}

/// Selection highlight background — muted steel blue that works on the dark theme.
const SELECTION_BG: ratatui::style::Color = ratatui::style::Color::Rgb(50, 60, 80);

/// Apply a subtle highlight over the selected region in the frame buffer.
/// Converts content-space selection coordinates to screen-space for rendering.
fn apply_selection_highlight(
    frame: &mut Frame,
    sel: &TextSelection,
    history_area: Rect,
    scroll_offset: usize,
) {
    let (start, end) = selection_ordered(sel);
    let view_top = scroll_offset;
    let view_bottom = scroll_offset + history_area.height as usize;

    // Only render the visible portion of the selection
    let vis_start_vrow = start.1.max(view_top);
    let vis_end_vrow = end.1.min(view_bottom.saturating_sub(1));
    if vis_start_vrow > vis_end_vrow {
        return;
    }

    let buf = frame.buffer_mut();
    for vrow in vis_start_vrow..=vis_end_vrow {
        let screen_y = history_area.y + (vrow - scroll_offset) as u16;
        let col_start = if vrow == start.1 {
            start.0
        } else {
            history_area.x
        };
        let col_end = if vrow == end.1 {
            end.0
        } else {
            history_area.x + history_area.width
        };
        for col in col_start..col_end {
            if let Some(cell) = buf.cell_mut((col, screen_y)) {
                cell.set_bg(SELECTION_BG);
            }
        }
    }
}

/// Extract the text content under the current selection from the frame buffer.
/// Converts content-space coordinates to screen-space for buffer reads.
pub fn extract_selected_text(
    frame_buf: &ratatui::buffer::Buffer,
    sel: &TextSelection,
    history_area: Rect,
    scroll_offset: usize,
) -> String {
    let (start, end) = selection_ordered(sel);
    let view_top = scroll_offset;
    let view_bottom = scroll_offset + history_area.height as usize;

    let mut result = String::new();
    for vrow in start.1..=end.1 {
        // Only extract text from visible rows
        if vrow < view_top || vrow >= view_bottom {
            if !result.is_empty() || vrow < start.1 {
                // Represent off-screen rows as blank lines to preserve structure
                if !result.is_empty() {
                    result.push('\n');
                }
            }
            continue;
        }
        let screen_y = history_area.y + (vrow - scroll_offset) as u16;
        let col_start = if vrow == start.1 {
            start.0
        } else {
            history_area.x
        };
        let col_end = if vrow == end.1 {
            end.0
        } else {
            history_area.x + history_area.width
        };
        let mut line = String::new();
        let mut col = col_start;
        while col < col_end {
            if let Some(cell) = frame_buf.cell((col, screen_y)) {
                line.push_str(cell.symbol());
                let w = unicode_width::UnicodeWidthStr::width(cell.symbol());
                col += w.max(1) as u16;
            } else {
                col += 1;
            }
        }
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(line.trim_end());
    }
    result
}

/// Return ordered (top-left, bottom-right) content-space coordinates.
fn selection_ordered(sel: &TextSelection) -> ((u16, usize), (u16, usize)) {
    let (ax, ay) = sel.anchor;
    let (ex, ey) = sel.end;
    if ay < ey || (ay == ey && ax <= ex) {
        ((ax, ay), (ex, ey))
    } else {
        ((ex, ey), (ax, ay))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{App, PromptState};
    use crate::overlay::{PromptFocus, PromptSelection};
    use ratatui::{Terminal, backend::TestBackend};

    fn workflow_activity(summary: &str) -> DisplayBlock {
        DisplayBlock::Activity(ActivityBlock {
            kind: ActivityKind::ShellCommand,
            status: ActivityStatus::Completed,
            tool_name: "exec_command".into(),
            summary: summary.into(),
            detail_lines: Vec::new(),
            duration_ms: 1,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        })
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    fn buffer_row_text(backend: &TestBackend, y: u16, width: u16) -> String {
        (0..width)
            .map(|x| backend.buffer().cell((x, y)).expect("buffer cell").symbol())
            .collect::<String>()
    }

    fn prompt_state(
        question: &str,
        options: Vec<&str>,
        reply_text: &str,
        editing_reply: bool,
    ) -> PromptState {
        let has_options = !options.is_empty();
        PromptState {
            question: question.into(),
            options: options.into_iter().map(str::to_string).collect(),
            selection: PromptSelection::Option(0),
            focus: if has_options && !editing_reply {
                PromptFocus::Selection
            } else {
                PromptFocus::ReplyEditor
            },
            reply_text: reply_text.into(),
            reply_cursor: reply_text.len(),
            response_tx: std::sync::mpsc::channel().0,
        }
    }

    #[test]
    fn plan_block_lines_stay_within_box_width() {
        let mut lines = Vec::new();
        let content = "A long paragraph that should wrap inside the plan box without escaping the right border.";
        let width = 32usize;
        render_plan_block(content, &mut lines, width, None);

        assert!(lines.len() >= 3, "expected top/content/bottom lines");

        // Top and bottom borders.
        assert!(line_text(&lines[0]).starts_with("\u{250c}"));
        assert!(line_text(lines.last().expect("bottom line")).starts_with("\u{2514}"));

        // Every rendered line should fit the viewport width exactly.
        for line in &lines {
            assert_eq!(line.width(), width, "line exceeded/fell short of box width");
        }

        // Middle content rows must have both left and right borders.
        for line in &lines[1..lines.len() - 1] {
            let text = line_text(line);
            assert!(text.starts_with("\u{2502} "), "missing left plan border");
            assert!(text.ends_with(" \u{2502}"), "missing right plan border");
        }
    }

    #[test]
    fn plan_block_renders_plain_text_without_markdown_list_bullets() {
        let mut lines = Vec::new();
        let content =
            "Testing all available tools one by one\n\n▸ Test filesystem tools\n○ Report results";
        render_plan_block(content, &mut lines, 64, None);

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("▸ Test filesystem tools"))
        );
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("○ Report results"))
        );
        assert!(!rendered.iter().any(|line| line.contains("•")));
        assert!(!rendered.iter().any(|line| line.contains("1.")));
    }

    #[test]
    fn plan_block_bridges_code_workflow_lane() {
        let blocks = [
            workflow_activity("first command"),
            DisplayBlock::PlanContent("Do the next thing".into()),
            workflow_activity("second command"),
        ];

        let mut lines = Vec::new();
        for idx in 0..blocks.len() {
            render_block(&blocks, idx, 1, &mut lines, 48, 20);
        }

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(rendered[0], "· first command");
        assert!(rendered.iter().any(|line| line.starts_with("  ┌─ PLAN ")));
        assert_eq!(
            rendered.last().map(String::as_str),
            Some("· second command")
        );
    }

    #[test]
    fn history_viewport_height_respects_dynamic_input_and_trays() {
        let mut app = App::new("model".into(), "session".into());
        app.blocks.clear(); // avoid splash-specific influence on expectations
        app.set_input("line1\nline2\nline3".into());
        app.queue_turn(crate::app::PreparedTurn::new("queued".into(), Vec::new()));
        app.live_turn = Some(crate::app::LiveTurnState {
            status_text: "thinking".into(),
            status_detail: None,
            phase_started_at: std::time::Instant::now(),
            turn_started_at: std::time::Instant::now(),
            assistant_block_idx: None,
            has_visible_output: false,
            transient_until: None,
        });

        let fw = 100u16;
        let fh = 40u16;
        let reserved_height = 1u16 + turn_status_height(&app) + queue_preview_height(&app, fw);
        let expected_overhead = 1u16 // status bar
            + turn_status_height(&app)
            + queue_preview_height(&app, fw)
            + input_height(&app, fw, fh, reserved_height);

        let got = history_viewport_height(&app, fw, fh);
        assert_eq!(got, fh.saturating_sub(expected_overhead) as usize);
    }

    #[test]
    fn pending_prompt_keeps_prompt_start_visible_above_footer() {
        let backend = TestBackend::new(20, 9);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.blocks.clear();
        app.blocks
            .push(DisplayBlock::AssistantText("history".into()));
        app.blocks.push(DisplayBlock::UserInput(
            ["line1", "line2", "line3"].join("\n"),
        ));
        app.start_turn();
        app.follow_mode = crate::app::FollowOutputMode::Contextual;

        terminal
            .draw(|frame| {
                let area = frame.area();
                let vh = history_viewport_height(&app, area.width, area.height);
                app.ensure_height_cache_pub(area.width as usize, vh);
                app.refresh_follow_output_anchor(area.width as usize, vh);
                let total = app.total_content_height(area.width as usize, vh);
                app.scroll_offset = app.scroll_offset.min(total.saturating_sub(vh));
                draw(frame, &app);
            })
            .expect("draw app");

        let rows: Vec<String> = (0..9)
            .map(|y| buffer_row_text(terminal.backend(), y, 20))
            .collect();

        assert!(
            rows.iter().take(5).any(|row| row.contains("history")),
            "history above the pending prompt should remain visible: {rows:?}"
        );
        assert!(
            rows.iter().take(5).any(|row| row.contains("line1")),
            "prompt start should remain visible above footer: {rows:?}"
        );
        assert!(
            rows[5].contains("Working"),
            "footer should still occupy the turn-status row: {rows:?}"
        );
    }

    #[test]
    fn wrapped_assistant_start_stays_visible_above_footer() {
        let backend = TestBackend::new(20, 8);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.blocks.clear();
        app.blocks.push(DisplayBlock::AssistantText(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda omega".into(),
        ));
        app.start_turn();
        app.follow_mode = crate::app::FollowOutputMode::Contextual;
        if let Some(turn) = app.live_turn.as_mut() {
            turn.has_visible_output = true;
        }

        terminal
            .draw(|frame| {
                let area = frame.area();
                let vh = history_viewport_height(&app, area.width, area.height);
                app.ensure_height_cache_pub(area.width as usize, vh);
                app.refresh_follow_output_anchor(area.width as usize, vh);
                let total = app.total_content_height(area.width as usize, vh);
                app.scroll_offset = app.scroll_offset.min(total.saturating_sub(vh));
                draw(frame, &app);
            })
            .expect("draw app");

        let rows: Vec<String> = (0..8)
            .map(|y| buffer_row_text(terminal.backend(), y, 20))
            .collect();

        assert!(
            rows.iter().take(4).any(|row| row.contains("alpha")),
            "assistant start should stay visible above footer: {rows:?}"
        );
        assert!(
            rows[4].contains("Working"),
            "footer should still occupy the turn-status row: {rows:?}"
        );
    }

    #[test]
    fn history_scroll_indicator_hides_when_near_bottom() {
        let mut app = App::new("model".into(), "session".into());
        app.blocks.clear();
        for idx in 0..18 {
            app.blocks
                .push(DisplayBlock::AssistantText(format!("line {idx}")));
        }

        let width = 24usize;
        let viewport_height = 6usize;
        app.ensure_height_cache_pub(width, viewport_height);
        let max_scroll = app
            .total_content_height(width, viewport_height)
            .saturating_sub(viewport_height);

        app.scroll_offset = max_scroll.saturating_sub(SCROLL_INDICATOR_HIDE_TAIL_ROWS);
        assert!(
            history_scroll_indicator(&app, Rect::new(0, 0, width as u16, viewport_height as u16))
                .is_none()
        );

        app.scroll_offset = max_scroll.saturating_sub(SCROLL_INDICATOR_HIDE_TAIL_ROWS + 1);
        let indicator =
            history_scroll_indicator(&app, Rect::new(0, 0, width as u16, viewport_height as u16))
                .expect("indicator");
        assert_eq!(indicator.x, width as u16 - 1);
        assert!(indicator.height >= 2);
    }

    #[test]
    fn draw_history_renders_scroll_indicator_pill_on_right_edge() {
        let backend = TestBackend::new(24, 6);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.blocks.clear();
        for idx in 0..18 {
            app.blocks
                .push(DisplayBlock::AssistantText(format!("line {idx}")));
        }

        let width = 24usize;
        let viewport_height = 6usize;
        app.ensure_height_cache_pub(width, viewport_height);
        let max_scroll = app
            .total_content_height(width, viewport_height)
            .saturating_sub(viewport_height);
        app.scroll_offset = max_scroll / 2;

        terminal
            .draw(|frame| draw_history(frame, &app, Rect::new(0, 0, 24, 6)))
            .expect("draw history");

        let symbols: Vec<(u16, String, ratatui::style::Color)> = (0..6u16)
            .map(|y| {
                let cell = terminal
                    .backend()
                    .buffer()
                    .cell((23, y))
                    .expect("indicator cell");
                (y, cell.symbol().to_string(), cell.fg)
            })
            .filter(|(_, symbol, _)| matches!(symbol.as_str(), "▄" | "█" | "▀"))
            .collect();

        assert!(!symbols.is_empty(), "expected scroll indicator glyphs");
        assert_eq!(
            symbols.first().map(|(_, symbol, _)| symbol.as_str()),
            Some("▄")
        );
        assert_eq!(
            symbols.last().map(|(_, symbol, _)| symbol.as_str()),
            Some("▀")
        );
        assert!(
            symbols.iter().all(|(_, _, fg)| *fg == theme::CHALK_DIM),
            "scroll indicator should use the pill color: {symbols:?}"
        );
    }

    #[test]
    fn prompt_content_lines_wrap_long_freeform_input() {
        let prompt = prompt_state(
            "Answer the question.",
            Vec::new(),
            "alpha beta gamma delta epsilon zeta eta theta iota omega",
            true,
        );

        let rendered = prompt_content_lines(&prompt, 20)
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join("\n");
        let normalized = rendered.replace('\n', " ");

        assert!(normalized.contains("alpha"));
        assert!(normalized.contains("theta"));
        assert!(normalized.contains("omega"));
    }

    #[test]
    fn prompt_height_accounts_for_wrapped_prompt_input() {
        let mut app = App::new("model".into(), "session".into());
        app.show_prompt(prompt_state(
            "Answer the question.",
            Vec::new(),
            "this answer keeps going until it wraps across multiple visible rows",
            true,
        ));

        assert!(
            prompt_height(&app, 20) > 7,
            "wrapped prompt input should grow the prompt area"
        );
    }

    #[test]
    fn wrap_line_reserves_continuation_prefix_width() {
        let text = "12345678901234567890";
        let segments = wrap_line(text, 2, 4, 10);
        let chunks = segments
            .iter()
            .map(|(start, end)| &text[*start..*end])
            .collect::<Vec<_>>();

        assert_eq!(chunks, vec!["12345678", "901234", "567890"]);
    }

    #[test]
    fn user_input_wrap_prefers_word_boundaries() {
        let blocks = [DisplayBlock::UserInput("alpha beta gamma delta".into())];
        let mut lines = Vec::new();

        render_block_into(&blocks, 0, 1, &mut lines, 12, 20);

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(rendered, vec!["● alpha beta", "  gamma", "  delta"]);
    }

    #[test]
    fn assistant_wrap_prefers_word_boundaries() {
        let blocks = [DisplayBlock::AssistantText("alpha beta gamma delta".into())];
        let mut lines = Vec::new();

        render_block_into(&blocks, 0, 1, &mut lines, 12, 20);

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(rendered, vec!["■ alpha beta", "  gamma", "  delta"]);
    }

    #[test]
    fn assistant_markdown_renders_expected_visible_lines() {
        let blocks = [DisplayBlock::AssistantText(
            "Use this minimal set:\n\n- `code`\n- `feature`\n- `issue`\n- `decision`\n\nThat’s probably the sweet spot.".into(),
        )];
        let mut lines = Vec::new();

        render_block_into(&blocks, 0, 1, &mut lines, 40, 20);

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(
            rendered,
            vec![
                "■ Use this minimal set:",
                "",
                "  code",
                "  feature",
                "  issue",
                "  decision",
                "",
                "  That’s probably the sweet spot.",
            ]
        );
    }

    #[test]
    fn assistant_text_connects_to_following_trace_lane() {
        let blocks = [
            DisplayBlock::AssistantText("Checking repo state".into()),
            DisplayBlock::Activity(ActivityBlock {
                kind: ActivityKind::Delegate,
                status: ActivityStatus::Completed,
                tool_name: "agent_result".into(),
                summary: "delegate done · inspect queue".into(),
                detail_lines: vec!["claude-sonnet · 3 iterations · 5 tool calls".into()],
                duration_ms: 4200,
                args: serde_json::Value::Null,
                result: serde_json::Value::Null,
                artifact: None,
                children: Vec::new(),
                extra: None,
            }),
        ];
        let mut lines = Vec::new();

        for idx in 0..blocks.len() {
            render_block_into(&blocks, idx, 1, &mut lines, 80, 20);
        }

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(rendered[0], "■ Checking repo state");
        assert_eq!(rendered[1], "  delegate done · inspect queue · 4.2s");
        assert_eq!(rendered[2], "  claude-sonnet · 3 iterations · 5 tool calls");
    }

    #[test]
    fn assistant_markdown_lists_preserve_line_boundaries_when_wrapped() {
        let blocks = [DisplayBlock::AssistantText(
            "3. **Reusable TUI primitives with explicit overlay/focus model**\n   - pi’s `packages/tui` has a clean model for `Component`, `Focusable`, overlays, focus handoff, cursor markers, and differential rendering.\n   - Lash’s TUI works, but some UX features look more hand-built/special-cased.\n   - Worth adapting as internal architecture, especially if you plan more pickers, popups, menus, or layered tools.\n\n4. **A stronger input editor abstraction**\n   - pi’s editor has explicit undo stack, kill ring, autocomplete plumbing, paste-marker handling, bracketed paste buffering, wrapped layout tracking.\n   - Lash already supports history/paste/images, but if the input box is going to keep growing features, a more editor-like core would pay off.\n   - High leverage if you want fewer ad hoc input-path bugs."
                .into(),
        )];
        let mut lines = Vec::new();

        render_block_into(&blocks, 0, 1, &mut lines, 86, 40);

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("4. A stronger input editor abstraction")),
            "expected fourth item to begin on its own rendered line: {rendered:#?}"
        );
        assert!(
            rendered.iter().all(|line| !line.contains("tools.4.")),
            "ordered list boundary collapsed during UI wrapping: {rendered:#?}"
        );
        assert!(
            rendered.iter().all(|line| !line.contains("bugs.5.")),
            "ordered list boundary collapsed during UI wrapping: {rendered:#?}"
        );
    }

    #[test]
    fn wordwise_wrap_falls_back_to_hard_breaks_for_long_tokens() {
        let segments = wrap_text_ranges_wordwise("supercalifragilistic", 6);
        let chunks = segments
            .iter()
            .map(|(start, end)| &"supercalifragilistic"[*start..*end])
            .collect::<Vec<_>>();

        assert_eq!(chunks, vec!["superc", "alifra", "gilist", "ic"]);
    }

    #[test]
    fn truncate_width_counts_ignore_ligature_break_markers() {
        let text = crate::text_display::neutralize_ligatures("a -> b != c && d");
        let truncated = truncate_to_display_width(&text, 8);
        assert_eq!(crate::text_display::visible_width(&truncated), 8);
    }

    #[test]
    fn queue_preview_sanitizes_ligature_sequences() {
        let mut app = App::new("model".into(), "session".into());
        app.queued_turns.push_back(PreparedTurn::new(
            "rg -n \"a -> b != c && d\"".into(),
            Vec::new(),
        ));
        let lines = queue_preview_lines(&app, 80);
        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert!(rendered.iter().any(|line| line.contains("rg -n")));
        let sanitized = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .any(|span| span.content.contains('\u{200C}'));
        assert!(
            sanitized,
            "expected ligature-breaking marker in rendered spans"
        );
    }

    #[test]
    fn ask_activity_renders_as_question_panel() {
        let activity = ActivityBlock {
            kind: ActivityKind::Ask,
            status: ActivityStatus::Completed,
            tool_name: "ask".into(),
            summary: "Question".into(),
            detail_lines: vec![
                "Question · Which environment should I use?".into(),
                "1. staging".into(),
                "2. prod".into(),
                "Answer · staging".into(),
            ],
            duration_ms: 0,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 48, false);

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(
            rendered[0],
            "┌─ QUESTION ───────────────────────────────────┐"
        );
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("Which environment should I use?"))
        );
        assert!(rendered.iter().any(|line| line.contains("1. staging")));
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("Answer · staging"))
        );
        assert_eq!(
            rendered.last().map(String::as_str),
            Some("└──────────────────────────────────────────────┘")
        );
    }

    #[test]
    fn input_height_grows_until_cap_and_preserves_history() {
        let mut app = App::new("model".into(), "session".into());
        app.set_input(
            (0..40)
                .map(|idx| format!("line {idx} with enough text to wrap around the viewport"))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        let height = input_height(&app, 40, 18, 1);
        assert_eq!(height, 14);
    }

    #[test]
    fn draw_input_keeps_cursor_visible_at_bottom_of_tall_message() {
        let backend = TestBackend::new(28, 5);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.set_input(
            (0..8)
                .map(|idx| format!("line {idx} wraps heavily"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        app.editor.cursor_pos = app.input().len();

        terminal
            .draw(|frame| draw_input(frame, &app, Rect::new(0, 0, 28, 5)))
            .expect("draw");

        let cursor = terminal
            .backend()
            .buffer()
            .cell((3, 3))
            .expect("cursor row cell");
        assert_eq!(cursor.symbol(), "l");

        let visible_rows: Vec<String> = (1..4)
            .map(|y| buffer_row_text(terminal.backend(), y, 28))
            .collect();
        assert!(
            visible_rows.iter().any(|row| row.contains("line 7")),
            "latest line should remain visible: {visible_rows:?}"
        );
    }

    #[test]
    fn turn_status_footer_only_renders_working_line() {
        let backend = TestBackend::new(48, 1);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.live_turn = Some(crate::app::LiveTurnState {
            status_text: "thinking".into(),
            status_detail: Some("waiting".into()),
            phase_started_at: std::time::Instant::now(),
            turn_started_at: std::time::Instant::now(),
            assistant_block_idx: None,
            has_visible_output: false,
            transient_until: None,
        });

        terminal
            .draw(|frame| draw_turn_status(frame, &app, frame.area()))
            .expect("draw footer");

        let backend = terminal.backend();
        let row = buffer_row_text(backend, 0, 48);

        assert!(row.contains("Working"));
        assert!(!row.contains("thinking"));
        assert!(!row.contains("waiting"));
    }

    #[test]
    fn turn_status_footer_replaces_working_with_error() {
        let backend = TestBackend::new(48, 1);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.live_turn = Some(crate::app::LiveTurnState {
            status_text: "error".into(),
            status_detail: Some("provider timeout".into()),
            phase_started_at: std::time::Instant::now(),
            turn_started_at: std::time::Instant::now(),
            assistant_block_idx: None,
            has_visible_output: false,
            transient_until: None,
        });

        terminal
            .draw(|frame| draw_turn_status(frame, &app, frame.area()))
            .expect("draw footer");

        let row = buffer_row_text(terminal.backend(), 0, 48);
        assert!(row.contains("Error"));
        assert!(!row.contains("Working"));
        assert!(!row.contains("provider timeout"));
    }

    #[test]
    fn turn_status_footer_shows_elapsed_after_threshold() {
        let backend = TestBackend::new(48, 1);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.live_turn = Some(crate::app::LiveTurnState {
            status_text: "thinking".into(),
            status_detail: None,
            phase_started_at: std::time::Instant::now() - std::time::Duration::from_secs(2),
            turn_started_at: std::time::Instant::now() - std::time::Duration::from_secs(2),
            assistant_block_idx: None,
            has_visible_output: false,
            transient_until: None,
        });

        terminal
            .draw(|frame| draw_turn_status(frame, &app, frame.area()))
            .expect("draw footer");

        let row = buffer_row_text(terminal.backend(), 0, 48);
        assert!(row.contains("Working"));
        assert!(row.contains("2.0s"));
    }

    #[test]
    fn styled_user_input_segment_highlights_skill_mentions_inline() {
        let spans = styled_user_input_segment("ok. /wholehog that.", 0, 19);

        assert_eq!(spans.len(), 4);
        assert_eq!(spans[0].content.as_ref(), "ok. ");
        assert_eq!(spans[1].content.as_ref(), "/");
        assert_eq!(spans[2].content.as_ref(), "wholehog");
        assert_eq!(spans[3].content.as_ref(), " that.");
    }

    #[test]
    fn suggestions_do_not_render_over_status_bar() {
        let backend = TestBackend::new(40, 6);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.set_input("hello".into());
        app.editor.cursor_pos = app.input().len();
        app.editor.suggestion_kind = SuggestionKind::Command;
        app.editor.suggestions = vec![
            ("/one".into(), "first suggestion".into()),
            ("/two".into(), "second suggestion".into()),
            ("/three".into(), "third suggestion".into()),
            ("/four".into(), "fourth suggestion".into()),
            ("/five".into(), "fifth suggestion".into()),
        ];

        terminal.draw(|frame| draw(frame, &app)).expect("draw app");

        let top = buffer_row_text(terminal.backend(), 0, 40);
        assert!(top.contains("lash"), "status bar disappeared: {top:?}");
        assert!(
            !top.contains("/one")
                && !top.contains("/two")
                && !top.contains("/three")
                && !top.contains("/four")
                && !top.contains("/five"),
            "suggestions overlapped the status bar: {top:?}"
        );
    }

    #[test]
    fn session_picker_stays_inside_history_pane() {
        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.set_input("hello".into());
        app.editor.cursor_pos = app.input().len();
        app.show_session_picker(
            (0..5)
                .map(|idx| crate::session_log::SessionInfo {
                    filename: format!("session-{idx}.db"),
                    session_id: format!("session-{idx}"),
                    message_count: idx + 1,
                    first_message: format!("first message {idx}"),
                    modified: std::time::SystemTime::now(),
                    cwd: None,
                })
                .collect(),
        );

        terminal.draw(|frame| draw(frame, &app)).expect("draw app");

        let input_row = buffer_row_text(terminal.backend(), 6, 40);
        assert!(
            input_row.contains(theme::PROMPT_CHAR),
            "session picker overflow erased input row: {input_row:?}"
        );
        assert!(
            !input_row.contains("Sessions") && !input_row.contains("first message"),
            "session picker overflowed into input area: {input_row:?}"
        );
    }

    #[test]
    fn session_picker_preview_keeps_cwd_suffix_visible() {
        let backend = TestBackend::new(80, 8);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.show_session_picker(vec![crate::session_log::SessionInfo {
            filename: "session-0.db".into(),
            session_id: "session-0".into(),
            message_count: 1,
            first_message:
                "a very long first message that should get truncated before the directory suffix"
                    .into(),
            modified: std::time::SystemTime::now(),
            cwd: Some(std::path::PathBuf::from("/home/sam/code/frontend-design")),
        }]);

        terminal.draw(|frame| draw(frame, &app)).expect("draw app");

        let rows: Vec<String> = (0..8)
            .map(|y| buffer_row_text(terminal.backend(), y, 80))
            .collect();
        assert!(
            rows.iter().any(|row| row.contains("/frontend-design")),
            "session picker row missing cwd suffix: {rows:?}"
        );
    }

    #[test]
    fn skill_picker_stays_inside_history_pane() {
        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.set_input("hello".into());
        app.editor.cursor_pos = app.input().len();
        app.show_skill_picker(
            (0..5)
                .map(|idx| (format!("skill-{idx}"), format!("skill description {idx}")))
                .collect(),
        );

        terminal.draw(|frame| draw(frame, &app)).expect("draw app");

        let input_row = buffer_row_text(terminal.backend(), 6, 40);
        assert!(
            input_row.contains(theme::PROMPT_CHAR),
            "skill picker overflow erased input row: {input_row:?}"
        );
        assert!(
            !input_row.contains("Skills") && !input_row.contains("skill-"),
            "skill picker overflowed into input area: {input_row:?}"
        );
    }

    #[test]
    fn history_redraw_clears_stale_cells_after_full_expand_toggle() {
        let backend = TestBackend::new(28, 4);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.blocks = vec![DisplayBlock::CodeBlock {
            code: "VERY_LONG_STALE_MARKER".into(),
            continuation: false,
        }];
        app.expand_level = 2;

        terminal
            .draw(|frame| {
                app.ensure_height_cache_pub(
                    frame.area().width as usize,
                    frame.area().height as usize,
                );
                draw_history(frame, &app, frame.area());
            })
            .expect("first draw");

        app.expand_level = 1;
        app.invalidate_height_cache();
        terminal
            .draw(|frame| {
                app.ensure_height_cache_pub(
                    frame.area().width as usize,
                    frame.area().height as usize,
                );
                draw_history(frame, &app, frame.area());
            })
            .expect("second draw");

        let buffer = terminal.backend();
        for y in 0..4 {
            let row = buffer_row_text(buffer, y, 28);
            assert!(
                !row.contains("VERY_LONG_STALE_MARKER"),
                "history redraw left stale text in row {y}: {row:?}"
            );
        }
    }

    #[test]
    fn full_draw_clears_stale_cell_modifiers_and_text() {
        let backend = TestBackend::new(32, 8);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| {
                let cell = frame
                    .buffer_mut()
                    .cell_mut((20, 3))
                    .expect("stale buffer cell");
                cell.set_symbol("R");
                cell.set_style(
                    Style::default()
                        .fg(theme::SODIUM)
                        .add_modifier(Modifier::UNDERLINED),
                );
            })
            .expect("seed stale frame");

        let mut app = App::new("model".into(), "session".into());
        app.blocks.clear();

        terminal.draw(|frame| draw(frame, &app)).expect("draw app");

        let cell = terminal
            .backend()
            .buffer()
            .cell((20, 3))
            .expect("redrawn cell");
        assert_eq!(cell.symbol(), " ");
        assert!(!cell.modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn queue_preview_lines_distinguish_checkpoint_and_next_turn() {
        let mut app = App::new("model".into(), "session".into());
        app.queue_pending_steer(PreparedTurn::new(
            "tighten the current assertion".into(),
            Vec::new(),
        ));
        app.queue_turn(PreparedTurn::new(
            "run follow-up validation".into(),
            Vec::new(),
        ));

        let rendered = queue_preview_lines(&app, 48)
            .iter()
            .map(line_text)
            .collect::<Vec<_>>();

        assert!(
            rendered
                .iter()
                .any(|line| line.contains("after next tool/result"))
        );
        assert!(rendered.iter().any(|line| line.contains("next full turn")));
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("tighten the current assertion"))
        );
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("run follow-up validation"))
        );
    }

    #[test]
    fn image_marker_uses_distinct_style_in_input() {
        let backend = TestBackend::new(40, 4);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut app = App::new("model".into(), "session".into());
        app.set_input("[Image #1] rest".into());
        app.editor.cursor_pos = app.input().len();

        terminal
            .draw(|frame| draw_input(frame, &app, Rect::new(0, 0, 40, 4)))
            .expect("draw");

        let image_cell = terminal
            .backend()
            .buffer()
            .cell((3, 1))
            .expect("image cell");
        let text_cell = terminal
            .backend()
            .buffer()
            .cell((14, 1))
            .expect("text cell");
        assert_eq!(image_cell.symbol(), "[");
        assert_eq!(image_cell.fg, theme::SODIUM);
        assert_eq!(text_cell.symbol(), "r");
        assert_eq!(text_cell.fg, theme::CHALK_MID);
    }

    #[test]
    fn queue_preview_height_grows_for_two_queue_sections() {
        let mut app = App::new("model".into(), "session".into());
        app.queue_pending_steer(PreparedTurn::new("checkpoint follow-up".into(), Vec::new()));
        app.queue_turn(PreparedTurn::new("next turn".into(), Vec::new()));

        assert!(queue_preview_height(&app, 64) > 1);
    }

    #[test]
    fn successful_shell_activity_hides_redundant_detail_lines() {
        let activity = ActivityBlock {
            kind: ActivityKind::ShellCommand,
            status: ActivityStatus::Completed,
            tool_name: "exec_command".into(),
            summary: "date '+%Y-%m-%d %H:%M:%S %Z'".into(),
            detail_lines: vec![
                "Command: date '+%Y-%m-%d %H:%M:%S %Z'".into(),
                "Workdir: /home/sam/code/lash".into(),
                "Exit code: 0".into(),
            ],
            duration_ms: 13,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 80, false);

        assert_eq!(line_text(&lines[0]), "+ date '+%Y-%m-%d %H:%M:%S %Z'");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn delegate_activity_renders_lichen_branch_lane() {
        let activity = ActivityBlock {
            kind: ActivityKind::Delegate,
            status: ActivityStatus::Completed,
            tool_name: "agent_call".into(),
            summary: "delegate · inspect queue rendering".into(),
            detail_lines: Vec::new(),
            duration_ms: 7,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 80, false);

        assert_eq!(line_text(&lines[0]), "◆ delegate · inspect queue rendering");
    }

    #[test]
    fn delegate_result_renders_summary_without_child_tool_tree() {
        let activity = ActivityBlock {
            kind: ActivityKind::Delegate,
            status: ActivityStatus::Completed,
            tool_name: "agent_result".into(),
            summary: "delegate done · inspect queue".into(),
            detail_lines: vec!["claude-sonnet · 3 iterations · 5 tool calls".into()],
            duration_ms: 4200,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 80, false);

        // Detail rows keep the connected delegate lane.
        assert_eq!(
            line_text(&lines[0]),
            "  delegate done · inspect queue · 4.2s"
        );
        assert_eq!(
            line_text(&lines[1]),
            "  claude-sonnet · 3 iterations · 5 tool calls"
        );
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn user_input_with_inline_paste_annotation_renders_normally() {
        let blocks = [DisplayBlock::UserInput(
            "before [pasted 1234 chars] after".into(),
        )];

        let mut lines = Vec::new();
        render_block(&blocks, 0, 1, &mut lines, 18, 10);

        let rendered: Vec<String> = lines.iter().map(line_text).collect();
        assert!(
            rendered
                .first()
                .is_some_and(|line| line.starts_with("● before [pasted"))
        );
        assert!(rendered.iter().skip(1).all(|line| line.starts_with("  ")));
        let normalized = rendered.join("").replace(['●', ' '], "");
        assert_eq!(normalized, "before[pasted1234chars]after");
    }

    #[test]
    fn code_workflow_lane_stays_connected_across_code_and_edit_blocks() {
        let blocks = [
            DisplayBlock::CodeBlock {
                code: "let answer = 42;".into(),
                continuation: false,
            },
            DisplayBlock::Activity(ActivityBlock {
                kind: ActivityKind::Edit,
                status: ActivityStatus::Completed,
                tool_name: "apply_patch".into(),
                summary: "Edited hello.txt (+2 -1)".into(),
                detail_lines: Vec::new(),
                duration_ms: 5,
                args: serde_json::Value::Null,
                result: serde_json::Value::Null,
                artifact: Some(ActivityArtifact::PatchPreview {
                    files: vec![PatchFilePreview {
                        path: "hello.txt".into(),
                        from_path: None,
                        status: "modified".into(),
                        added: 2,
                        removed: 1,
                        diff: String::new(),
                    }],
                    total_added: 2,
                    total_removed: 1,
                }),
                children: Vec::new(),
                extra: None,
            }),
            DisplayBlock::CodeBlock {
                code: "println!(\"done\");".into(),
                continuation: true,
            },
        ];

        let mut lines = Vec::new();
        for idx in 0..blocks.len() {
            render_block(&blocks, idx, 1, &mut lines, 80, 20);
        }

        assert_eq!(line_text(&lines[0]), "· Edited hello.txt (+2 -1)");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn compact_view_hides_code_blocks_but_full_view_keeps_them() {
        let blocks = [DisplayBlock::CodeBlock {
            code: "let answer = 42;\nprintln!(\"done\");".into(),
            continuation: false,
        }];

        let mut compact_lines = Vec::new();
        render_block(&blocks, 0, 1, &mut compact_lines, 80, 20);
        assert!(compact_lines.is_empty());

        let mut full_lines = Vec::new();
        render_block(&blocks, 0, 2, &mut full_lines, 80, 20);
        assert_eq!(line_text(&full_lines[0]), "│ let answer = 42;");
        assert_eq!(line_text(&full_lines[1]), "│ println!(\"done\");");
    }

    #[test]
    fn patch_activity_normal_view_renders_inline_diff_lines() {
        let activity = ActivityBlock {
            kind: ActivityKind::Edit,
            status: ActivityStatus::Completed,
            tool_name: "apply_patch".into(),
            summary: "Edited lash-cli/src/ui.rs (+3 -1)".into(),
            detail_lines: Vec::new(),
            duration_ms: 15,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: Some(ActivityArtifact::PatchPreview {
                files: vec![PatchFilePreview {
                    path: "lash-cli/src/ui.rs".into(),
                    from_path: None,
                    status: "modified".into(),
                    added: 3,
                    removed: 1,
                    diff: "--- a/lash-cli/src/ui.rs\n+++ b/lash-cli/src/ui.rs\n@@\n-old\n+new"
                        .into(),
                }],
                total_added: 3,
                total_removed: 1,
            }),
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 80, false);

        assert_eq!(line_text(&lines[0]), "  Edited lash-cli/src/ui.rs (+3 -1)");
        assert!(line_text(&lines[1]).contains("@@"));
        assert!(line_text(&lines[2]).contains("- old"));
        assert!(line_text(&lines[3]).contains("+ new"));
    }

    #[test]
    fn multi_file_patch_activity_keeps_compact_file_slate() {
        let activity = ActivityBlock {
            kind: ActivityKind::Edit,
            status: ActivityStatus::Completed,
            tool_name: "apply_patch".into(),
            summary: "Edited 2 files (+4 -1)".into(),
            detail_lines: Vec::new(),
            duration_ms: 15,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: Some(ActivityArtifact::PatchPreview {
                files: vec![
                    PatchFilePreview {
                        path: "a.rs".into(),
                        from_path: None,
                        status: "modified".into(),
                        added: 3,
                        removed: 1,
                        diff: String::new(),
                    },
                    PatchFilePreview {
                        path: "b.rs".into(),
                        from_path: None,
                        status: "added".into(),
                        added: 1,
                        removed: 0,
                        diff: String::new(),
                    },
                ],
                total_added: 4,
                total_removed: 1,
            }),
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 80, false);

        assert_eq!(line_text(&lines[0]), "  Edited 2 files (+4 -1)");
        assert_eq!(line_text(&lines[1]), "    Edited a.rs (+3 -1)");
        assert_eq!(line_text(&lines[2]), "    Added b.rs (+1 -0)");
    }

    #[test]
    fn patch_activity_full_view_renders_numbered_inline_diff_lines() {
        let activity = ActivityBlock {
            kind: ActivityKind::Edit,
            status: ActivityStatus::Completed,
            tool_name: "apply_patch".into(),
            summary: "Edited lash-cli/src/ui.rs (+1 -1)".into(),
            detail_lines: Vec::new(),
            duration_ms: 8,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: Some(ActivityArtifact::PatchPreview {
                files: vec![PatchFilePreview {
                    path: "lash-cli/src/ui.rs".into(),
                    from_path: Some("lash-cli/src/app.rs".into()),
                    status: "moved".into(),
                    added: 1,
                    removed: 1,
                    diff: "--- a/lash-cli/src/app.rs\n+++ b/lash-cli/src/ui.rs\n@@ -1,1 +1,1 @@\n-old\n+new"
                        .into(),
                }],
                total_added: 1,
                total_removed: 1,
            }),
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 2, &mut lines, 80, false);

        assert!(line_text(&lines[1]).contains("@@"));
        assert!(line_text(&lines[2]).contains("1 - old"));
        assert!(line_text(&lines[3]).contains("1 + new"));
        assert_eq!(
            lines[2].spans.last().expect("diff body").style,
            theme::patch_diff_remove_line()
        );
    }

    #[test]
    fn context_usage_pct_returns_fractional_percent() {
        let usage = lash::PromptUsage {
            prompt_context_tokens: 81_182,
            input_tokens: 78_182,
            cached_input_tokens: 3_000,
            context_budget_tokens: 82_382,
        };
        let pct = context_usage_pct(&usage, 1_050_000).expect("context pct");
        assert!(pct > 7.8 && pct < 7.9, "unexpected pct: {pct}");
    }

    #[test]
    fn format_context_usage_shows_provider_adjusted_values() {
        let usage = lash::PromptUsage {
            prompt_context_tokens: 84_000,
            input_tokens: 82_800,
            cached_input_tokens: 1_200,
            context_budget_tokens: 82_800,
        };
        assert_eq!(
            format_context_usage(&usage, 1_050_000).as_deref(),
            Some("82.8k / 1.1M (7.9%)")
        );
        assert_eq!(
            format_context_usage(
                &lash::PromptUsage {
                    prompt_context_tokens: 1_000,
                    input_tokens: 1_000,
                    cached_input_tokens: 0,
                    context_budget_tokens: 1_000,
                },
                1_050_000,
            )
            .as_deref(),
            Some("1.0k / 1.1M (0.1%)")
        );
        assert_eq!(
            format_context_usage(&lash::PromptUsage::default(), 1_050_000),
            None
        );
    }

    #[test]
    fn current_context_budget_tokens_uses_live_stream_usage() {
        let mut app = App::new("claude-opus-4-6".into(), "test".into());
        app.running = true;
        app.context_usage_excludes_cached_input = true;
        app.last_response_usage = lash::TokenUsage {
            input_tokens: 10_000,
            output_tokens: 2_000,
            cached_input_tokens: 1_500,
            reasoning_tokens: 0,
        };
        app.live_output_tokens_estimate = 300;

        assert_eq!(current_context_budget_tokens(&app), Some(13_800));
    }

    #[test]
    fn current_context_budget_tokens_adjusts_when_input_includes_cached() {
        let mut app = App::new("claude-opus-4-6".into(), "test".into());
        app.running = true;
        app.context_usage_excludes_cached_input = false;
        app.last_response_usage = lash::TokenUsage {
            input_tokens: 10_000,
            output_tokens: 2_000,
            cached_input_tokens: 1_500,
            reasoning_tokens: 0,
        };
        app.live_output_tokens_estimate = 300;

        assert_eq!(current_context_budget_tokens(&app), Some(12_300));
    }

    #[test]
    fn format_context_usage_uses_normalized_context_budget() {
        let usage = lash::PromptUsage {
            prompt_context_tokens: 1_000,
            input_tokens: 1_000,
            cached_input_tokens: 200,
            context_budget_tokens: 1_050,
        };
        assert_eq!(
            format_context_usage(&usage, 10_000).as_deref(),
            Some("1.1k / 10.0k (10.5%)")
        );
    }

    #[test]
    fn exploration_activity_renders_branch_lane() {
        let activity = ActivityBlock {
            kind: ActivityKind::Exploration,
            status: ActivityStatus::Completed,
            tool_name: "read_file".into(),
            summary: "EXPLORE".into(),
            detail_lines: vec!["list .".into(), "read src/main.rs".into()],
            duration_ms: 3,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 80, false);

        assert_eq!(line_text(&lines[0]), "  EXPLORE");
        assert_eq!(line_text(&lines[1]), "   list .");
        assert_eq!(line_text(&lines[2]), "   read src/main.rs");
    }

    #[test]
    fn exploration_activity_wraps_detail_lines() {
        let activity = ActivityBlock {
            kind: ActivityKind::Exploration,
            status: ActivityStatus::Completed,
            tool_name: "read_file".into(),
            summary: "EXPLORE".into(),
            detail_lines: vec!["abcdefghijklmnopqrstuvwx".into()],
            duration_ms: 3,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 20, false);

        assert_eq!(line_text(&lines[0]), "  EXPLORE");
        assert_eq!(line_text(&lines[1]), "   abcdefghijklmnopq");
        assert_eq!(line_text(&lines[2]), "   rstuvwx");
    }

    #[test]
    fn exploration_activity_caps_wrapped_details_with_ellipsis() {
        let activity = ActivityBlock {
            kind: ActivityKind::Exploration,
            status: ActivityStatus::Completed,
            tool_name: "read_file".into(),
            summary: "EXPLORE".into(),
            detail_lines: vec![
                "abcdefghijklmnopqrstuvwxyz0123456789".into(),
                "second line".into(),
            ],
            duration_ms: 3,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 20, false);

        assert_eq!(line_text(&lines[0]), "  EXPLORE");
        assert_eq!(lines.len(), 4);
        assert_eq!(line_text(&lines[1]), "   abcdefghijklmnopq");
        assert_eq!(line_text(&lines[2]), "   rstuvwxyz01234567");
        assert!(line_text(&lines[3]).ends_with('…'));
    }

    #[test]
    fn exploration_activity_without_details_uses_single_line_marker() {
        let activity = ActivityBlock {
            kind: ActivityKind::Exploration,
            status: ActivityStatus::Completed,
            tool_name: "read_file".into(),
            summary: "EXPLORE".into(),
            detail_lines: Vec::new(),
            duration_ms: 3,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 80, false);

        assert_eq!(line_text(&lines[0]), "  EXPLORE");
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn exploration_activity_height_matches_rendered_rows_when_truncated() {
        let activity = ActivityBlock {
            kind: ActivityKind::Exploration,
            status: ActivityStatus::Completed,
            tool_name: "read_file".into(),
            summary: "EXPLORE".into(),
            detail_lines: vec![
                "abcdefghijklmnopqrstuvwxyz0123456789".into(),
                "second line".into(),
            ],
            duration_ms: 3,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        };

        let expected_height = DisplayBlock::Activity(activity.clone()).height(1, 20, 0);
        let mut lines = Vec::new();
        render_activity_block(&activity, 1, &mut lines, 20, false);

        assert_eq!(expected_height, lines.len());
    }

    #[test]
    fn exploration_activity_artifact_height_matches_rendered_rows() {
        let activity = ActivityBlock {
            kind: ActivityKind::Exploration,
            status: ActivityStatus::Completed,
            tool_name: "read_file".into(),
            summary: "EXPLORE".into(),
            detail_lines: vec!["read src/app.rs".into()],
            duration_ms: 3,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: Some(ActivityArtifact::TextPreview {
                title: Some("preview".into()),
                text: "this is a long preview line that should wrap".into(),
            }),
            children: Vec::new(),
            extra: None,
        };

        let expected_height = DisplayBlock::Activity(activity.clone()).height(2, 20, 0);
        let mut lines = Vec::new();
        render_activity_block(&activity, 2, &mut lines, 20, false);

        assert_eq!(expected_height, lines.len());
    }

    #[test]
    fn text_preview_artifact_shows_hidden_line_marker_and_tail() {
        let text = (0..16)
            .map(|idx| format!("line-{idx}"))
            .collect::<Vec<_>>()
            .join("\n");
        let activity = ActivityBlock {
            kind: ActivityKind::Exploration,
            status: ActivityStatus::Completed,
            tool_name: "read_file".into(),
            summary: "EXPLORE".into(),
            detail_lines: vec!["read src/app.rs".into()],
            duration_ms: 3,
            args: serde_json::Value::Null,
            result: serde_json::Value::Null,
            artifact: Some(ActivityArtifact::TextPreview {
                title: Some("preview".into()),
                text,
            }),
            children: Vec::new(),
            extra: None,
        };

        let mut lines = Vec::new();
        render_activity_block(&activity, 2, &mut lines, 80, false);
        let rendered: Vec<String> = lines.iter().map(line_text).collect();

        assert!(
            rendered
                .iter()
                .any(|line| line.contains("… 5 lines hidden …"))
        );
        assert!(rendered.iter().any(|line| line.contains("line-0")));
        assert!(rendered.iter().any(|line| line.contains("line-15")));
    }

    #[test]
    fn animated_lash_word_cycles_slash_through_wordmark() {
        let frames = [0_u64, 200, 400, 600, 800]
            .into_iter()
            .map(std::time::Duration::from_millis)
            .map(animated_lash_word)
            .map(|spans| {
                spans
                    .into_iter()
                    .map(|span| span.content.to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert_eq!(frames, vec!["/LASH", "L/ASH", "LA/SH", "LAS/H", "LASH/"]);
    }
}
