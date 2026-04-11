use lash_tui::{Frame, Line, Modifier, Rect, Span, Style};
use unicode_width::UnicodeWidthStr;

use crate::app::{App, format_tokens};
use crate::editor::SuggestionKind;
use crate::{render, theme};

const INPUT_HORIZONTAL_PADDING: u16 = 1;
const PROMPT_HORIZONTAL_PADDING: u16 = 1;

pub fn draw(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }

    frame.clear(bg(theme::surface_base()));

    let history = render::history_area(app, area.width, area.height);
    let turn_height = if app.live_turn.is_some() { 1 } else { 0 };
    let queue_lines = render::queue_preview_lines_snapshot(app, area.width);
    let queue_height = queue_lines.len() as u16;
    let turn_area = Rect::new(0, history.bottom(), area.width, turn_height);
    let queue_area = Rect::new(0, turn_area.bottom(), area.width, queue_height);
    let input_area = Rect::new(
        0,
        queue_area.bottom(),
        area.width,
        area.height.saturating_sub(queue_area.bottom()),
    );

    draw_status_bar(frame, app, Rect::new(0, 0, area.width, 1));
    draw_history(frame, app, history);
    apply_selection_highlight(frame, app, history);
    if turn_height > 0 {
        draw_turn_status(frame, app, turn_area);
    }
    if queue_height > 0 {
        draw_lines_region(frame, queue_area, &queue_lines, bg(theme::surface_raised()));
    }
    if app.has_wait() {
        // History reads as inactive while we're waiting on an auto-resume.
        // Dimming it distinguishes "input is blocked, please wait" from
        // "the app is idle and you can type" — the second-pass critique's
        // "user might try to type during a scheduled wait" concern.
        draw_overlay_scrim(frame, history);
        draw_wait(frame, app, input_area);
    } else if app.has_prompt() {
        draw_overlay_scrim(frame, history);
        draw_prompt(frame, app, input_area);
    } else {
        draw_input(frame, app, input_area);
        draw_suggestions(frame, app, input_area);
    }
    draw_session_picker(frame, app, history);
    draw_skill_picker(frame, app, history);
}

// ─── Status bar slot grammar ─────────────────────────────────────────────────
//
// The status bar is built from labeled slots, each with a `priority`. When
// the window is too narrow to fit every slot, the lowest-priority slots are
// dropped first until the remainder fits. There is no character-level
// truncation: a slot either renders in full or not at all. This keeps the
// bar's shape legible at every width and makes it obvious which information
// is load-bearing (brand, model) versus decorative (variant, context meter).

#[derive(Clone)]
struct StatusSlot {
    spans: Vec<Span<'static>>,
    /// Higher = more important. Dropped last under width pressure.
    priority: u8,
}

impl StatusSlot {
    fn width(&self) -> usize {
        self.spans
            .iter()
            .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
            .sum()
    }
}

/// Greedy knapsack by descending priority. Returns the slots that fit,
/// preserving their original order.
fn fit_slots(slots: &[StatusSlot], budget: usize) -> Vec<Span<'static>> {
    if budget == 0 || slots.is_empty() {
        return Vec::new();
    }
    let mut indices: Vec<usize> = (0..slots.len()).collect();
    indices.sort_by(|a, b| slots[*b].priority.cmp(&slots[*a].priority));

    let mut included = vec![false; slots.len()];
    let mut used = 0usize;
    for idx in indices {
        let width = slots[idx].width();
        if width == 0 {
            continue;
        }
        if used + width <= budget {
            included[idx] = true;
            used += width;
        }
    }

    let mut spans = Vec::new();
    for (idx, slot) in slots.iter().enumerate() {
        if included[idx] {
            spans.extend(slot.spans.iter().cloned());
        }
    }
    spans
}

fn draw_status_bar(frame: &mut Frame<'_>, app: &App, area: Rect) {
    frame.fill(area, ' ', bg(theme::surface_raised()));
    if area.width == 0 {
        return;
    }

    let (left, right) = build_status_slots(app);
    let total = area.width as usize;

    // Right gets first refusal with up to half the bar. Whatever it leaves
    // (minus one column for visual breathing room) becomes the left budget.
    let right_spans = fit_slots(&right, total / 2);
    let right_width: usize = right_spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum();
    let gap = if right_width > 0 { 1 } else { 0 };
    let left_budget = total.saturating_sub(right_width + gap);
    let left_spans = fit_slots(&left, left_budget);

    if !left_spans.is_empty() {
        let line = Line::from(left_spans);
        let width = line.width() as u16;
        frame.write_line(0, 0, &line, width);
    }
    if !right_spans.is_empty() {
        let line = Line::from(right_spans);
        let width = line.width() as u16;
        let x = area.width.saturating_sub(width);
        frame.write_line(x, 0, &line, width);
    }
}

fn build_status_slots(app: &App) -> (Vec<StatusSlot>, Vec<StatusSlot>) {
    let sep_style = theme::text_faint_style();

    // LEFT side: brand · model · variant
    let mut left = Vec::new();
    left.push(StatusSlot {
        spans: vec![
            Span::raw(" "),
            Span::styled(
                "lash",
                Style::default()
                    .fg(theme::brand())
                    .add_modifier(Modifier::Bold),
            ),
        ],
        priority: 100, // always keep — identity anchor
    });
    if !app.model.is_empty() {
        left.push(StatusSlot {
            spans: vec![
                Span::styled(" · ", sep_style),
                Span::styled(app.model.clone(), theme::text_subtle_style()),
            ],
            priority: 80,
        });
    }
    if let Some(variant) = app
        .model_variant
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        left.push(StatusSlot {
            spans: vec![
                Span::styled(" · ", sep_style),
                Span::styled(
                    variant.to_string(),
                    Style::default()
                        .fg(theme::brand())
                        .add_modifier(Modifier::Bold),
                ),
            ],
            priority: 40, // drop before model under pressure
        });
    }

    // RIGHT side: context-window meter. The expand keybind is discoverable
    // via `/controls` and the `▸` marker on tool activities — no teaching
    // slot needed in the status bar.
    let mut right = Vec::new();
    if let Some(ctx) = status_bar_context_spans(app) {
        right.push(ctx);
    }

    (left, right)
}

fn status_bar_context_spans(app: &App) -> Option<StatusSlot> {
    let Some(context_window) = app.context_window else {
        let total = app.token_usage.input_tokens
            + app.token_usage.output_tokens
            + app.live_output_tokens_estimate;
        if total <= 0 {
            return None;
        }
        return Some(StatusSlot {
            spans: vec![
                Span::styled(format_tokens(total), theme::text_subtle_style()),
                Span::raw(" "),
            ],
            priority: 70,
        });
    };

    let used = current_context_budget_tokens(app)
        .or_else(|| {
            app.last_prompt_usage
                .as_ref()
                .map(|usage| usage.context_budget_tokens as i64)
                .filter(|used| *used > 0)
        })
        .unwrap_or_else(|| {
            (app.token_usage.input_tokens
                + app.token_usage.output_tokens
                + app.live_output_tokens_estimate)
                .max(0)
        });

    let pct = if context_window == 0 {
        0.0
    } else {
        used as f64 / context_window as f64 * 100.0
    };

    // The old layout was `3.4k / 1.1M (9.3%)` — three representations of the
    // same number: tokens used, total window, and the derived percentage.
    // The percentage is the only thing you can act on (it tells you how close
    // you are to the wall); the raw token count is useful for debugging.
    // Keep just those two, separated by the faint middle-dot used elsewhere
    // in the bar. Integer percentage — `9.3%` vs `9%` is false precision at
    // this scale.
    Some(StatusSlot {
        spans: vec![
            Span::styled(format_tokens(used), theme::text_subtle_style()),
            Span::styled(" · ", theme::text_faint_style()),
            Span::styled(format!("{pct:.0}%"), theme::text_subtle_style()),
            Span::raw(" "),
        ],
        priority: 70,
    })
}

fn draw_history(frame: &mut Frame<'_>, app: &mut App, area: Rect) {
    frame.fill(area, ' ', bg(theme::surface_base()));
    let viewport_height = area.height as usize;
    let viewport_width = area.width as usize;
    let scroll = app.scroll_offset;
    let block_height = app.height_cache_snapshot().last().copied().unwrap_or(0);
    let (first_idx, mut skip_lines) = if scroll >= block_height {
        (app.blocks.len(), scroll - block_height)
    } else {
        render::find_visible_block(app, scroll)
    };

    let mut written_rows = 0usize;
    'blocks: for idx in first_idx..app.blocks.len() {
        let block_lines = app.rendered_block_lines_cached(idx, viewport_width, viewport_height);
        if skip_lines >= block_lines.len() {
            skip_lines -= block_lines.len();
            continue;
        }
        for line in block_lines.iter().skip(skip_lines) {
            if written_rows >= viewport_height {
                break 'blocks;
            }
            frame.write_line(area.x, area.y + written_rows as u16, line, area.width);
            written_rows += 1;
        }
        skip_lines = 0;
    }

    if written_rows < viewport_height
        && let Some(live_lines) = app.live_assistant_lines_snapshot()
    {
        if app.live_assistant_leading_padding() > 0 {
            if skip_lines > 0 {
                skip_lines -= 1;
            } else if written_rows < viewport_height {
                written_rows += 1;
            }
        }
        for line in live_lines.iter().skip(skip_lines) {
            if written_rows >= viewport_height {
                break;
            }
            frame.write_line(area.x, area.y + written_rows as u16, line, area.width);
            written_rows += 1;
        }
    }

    if let Some((x, y, height)) = render::history_scroll_indicator(app, area) {
        for offset in 0..height {
            frame.write_text(x, y + offset, "│", fg(theme::text_subtle()), 1);
        }
    }
}

fn apply_selection_highlight(frame: &mut Frame<'_>, app: &App, history_area: Rect) {
    if !(app.selection.active || app.selection.visible) || history_area.height == 0 {
        return;
    }

    let ((start_col, start_row), (end_col, end_row)) = selection_ordered(&app.selection);
    let view_top = app.scroll_offset;
    let view_bottom = app.scroll_offset + history_area.height as usize;
    let visible_start = start_row.max(view_top);
    let visible_end = end_row.min(view_bottom.saturating_sub(1));

    if visible_start > visible_end {
        return;
    }

    for row in visible_start..=visible_end {
        let screen_y = history_area.y + (row - app.scroll_offset) as u16;
        let col_start = if row == start_row {
            start_col
        } else {
            history_area.x
        };
        let col_end = if row == end_row {
            end_col
        } else {
            history_area.x + history_area.width
        };
        let span_width = col_end.saturating_sub(col_start);
        if span_width > 0 {
            frame.patch_row_style_range(col_start, screen_y, span_width, |style| {
                style.bg(theme::SELECTION_BG)
            });
        }
    }
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

fn draw_turn_status(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(turn) = app.live_turn.as_ref() else {
        return;
    };
    frame.fill(area, ' ', bg(theme::surface_raised()));
    let label = if turn.status_text == "error" {
        "Error"
    } else if app.has_wait() {
        "Waiting"
    } else if app.has_prompt() {
        "Paused"
    } else if turn.status_text == "thinking" {
        "Thinking"
    } else if turn.status_text == "responding" {
        "Responding"
    } else {
        "Working"
    };
    let elapsed = if app.has_wait() {
        match app.wait_remaining_seconds() {
            Some(0) => "resuming".to_string(),
            Some(1) => "1s left".to_string(),
            Some(seconds) => format!("{seconds}s left"),
            None => String::new(),
        }
    } else {
        crate::util::format_duration_ms_if_visible(turn.turn_started_at.elapsed().as_millis() as u64)
            .unwrap_or_default()
    };
    let mut spans = animated_lash_word(turn.turn_started_at.elapsed());
    spans.push(Span::raw("  "));
    spans.push(Span::styled(label.to_string(), theme::turn_status_state()));
    if !elapsed.is_empty() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(elapsed, theme::turn_status_elapsed()));
    }
    let line = Line::from(spans);
    let x = area.width.saturating_sub(line.width() as u16) / 2;
    frame.write_line(x, area.y, &line, area.width.saturating_sub(x));
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

fn draw_input(frame: &mut Frame<'_>, app: &App, area: Rect) {
    if area.width < 2 || area.height < 2 {
        return;
    }
    let snapshot = render::input_render_snapshot(app, area);
    let content_area = Rect::new(
        area.x + INPUT_HORIZONTAL_PADDING,
        area.y + 1,
        area.width.saturating_sub(INPUT_HORIZONTAL_PADDING * 2),
        area.height.saturating_sub(2),
    );

    draw_top_bottom_rule(frame, area, fg(theme::border_faint()));
    for (idx, line) in snapshot
        .lines
        .iter()
        .skip(snapshot.scroll_offset)
        .take(content_area.height as usize)
        .enumerate()
    {
        frame.write_line(
            content_area.x,
            content_area.y + idx as u16,
            line,
            content_area.width,
        );
    }
    for (x, y, width) in render::input_selection_rects(app, area) {
        frame.patch_row_style_range(x, y, width, |style| style.bg(theme::SELECTION_BG));
    }
    if let Some(badge) = snapshot.badge {
        let width = badge.width() as u16;
        let x = area.width.saturating_sub(width + 1);
        frame.write_line(area.x + x, area.y + area.height - 1, &badge, width);
    }
    frame.set_cursor_position((
        area.x + INPUT_HORIZONTAL_PADDING + snapshot.cursor.0,
        area.y + 1 + snapshot.cursor.1,
    ));
}

fn draw_prompt(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(prompt) = app.prompt_state() else {
        return;
    };
    if area.width < 2 || area.height < 1 {
        return;
    }
    frame.fill(area, ' ', bg(theme::surface_deep()));
    let inner_width = area
        .width
        .saturating_sub(PROMPT_HORIZONTAL_PADDING.saturating_mul(2)) as usize;
    let lines = render::prompt_content_lines_snapshot(prompt, inner_width.max(1));
    let visible = area.height as usize;
    let max_scroll = lines.len().saturating_sub(visible);
    let scroll = if prompt.is_text_entry() {
        max_scroll
    } else {
        prompt.scroll_offset.min(max_scroll)
    };
    for (idx, line) in lines.iter().skip(scroll).take(visible).enumerate() {
        frame.write_line(
            area.x + PROMPT_HORIZONTAL_PADDING,
            area.y + idx as u16,
            line,
            area.width
                .saturating_sub(PROMPT_HORIZONTAL_PADDING.saturating_mul(2)),
        );
    }
}

fn draw_wait(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let Some(wait) = app.wait_state() else {
        return;
    };
    if area.width < 2 || area.height < 1 {
        return;
    }
    frame.fill(area, ' ', bg(theme::surface_deep()));
    let inner_width = area
        .width
        .saturating_sub(PROMPT_HORIZONTAL_PADDING.saturating_mul(2)) as usize;
    let lines = render::wait_content_lines_for_app(wait, inner_width.max(1));
    let visible = area.height as usize;
    for (idx, line) in lines.iter().take(visible).enumerate() {
        frame.write_line(
            area.x + PROMPT_HORIZONTAL_PADDING,
            area.y + idx as u16,
            line,
            area.width
                .saturating_sub(PROMPT_HORIZONTAL_PADDING.saturating_mul(2)),
        );
    }
}

fn draw_suggestions(frame: &mut Frame<'_>, app: &App, input_area: Rect) {
    if app.suggestions().is_empty() || app.suggestion_kind() == SuggestionKind::None {
        return;
    }
    let max_visible = app.suggestions().len().min(8);
    let name_col = app
        .suggestions()
        .iter()
        .take(max_visible)
        .map(|(name, _)| display_width(name))
        .max()
        .unwrap_or(8)
        .max(8);
    let width = input_area.width.min(72);
    let height = max_visible as u16 + 2;
    if width < 4 || input_area.y < height {
        return;
    }
    let popup = Rect::new(
        input_area.x,
        input_area.y.saturating_sub(height),
        width,
        height,
    );
    frame.draw_box(
        popup,
        fg(theme::border_faint()),
        Some(bg(theme::surface_deep())),
    );
    for (idx, (name, desc)) in app.suggestions().iter().take(max_visible).enumerate() {
        let selected = idx == app.suggestion_idx();
        let style = if selected {
            fg(theme::text_primary()).bg(theme::surface_raised())
        } else {
            fg(theme::text_subtle())
        };
        let row = format!(" {:<width$} {}", name, desc, width = name_col);
        frame.write_text(
            popup.x + 1,
            popup.y + 1 + idx as u16,
            &row,
            style,
            popup.width.saturating_sub(2),
        );
    }
}

/// Dim every cell in `area` so a popup drawn afterwards reads as a modal
/// overlay rather than a box floating on top of unchanged history. The
/// popup's own `draw_box` call replaces the style of the cells it covers,
/// so only the surrounding scrim remains dimmed.
fn draw_overlay_scrim(frame: &mut Frame<'_>, area: Rect) {
    for y in 0..area.height {
        frame.patch_row_style_range(area.x, area.y + y, area.width, |style| {
            style.add_modifier(Modifier::Dim)
        });
    }
}

fn draw_session_picker(frame: &mut Frame<'_>, app: &App, history_area: Rect) {
    let Some(picker) = app.session_picker_state() else {
        return;
    };
    let width = 80u16.min(history_area.width.saturating_sub(4));
    // Empty state still gets a visible row so the popup isn't a hollow box,
    // plus a footer row with the dismissal hint.
    let list_height = picker.items.len().min(15).max(1) as u16;
    let height = list_height + 3; // title + list + footer
    if width < 4 || history_area.height < height {
        return;
    }
    draw_overlay_scrim(frame, history_area);
    let popup = centered_rect(history_area, width, height);
    frame.draw_box(
        popup,
        fg(theme::border_faint()),
        Some(bg(theme::surface_deep())),
    );
    frame.write_text(
        popup.x + 2,
        popup.y,
        &format!("Sessions ({})", picker.items.len()),
        fg(theme::brand()).add_modifier(Modifier::Bold),
        popup.width.saturating_sub(4),
    );

    if picker.items.is_empty() {
        frame.write_text(
            popup.x + 2,
            popup.y + 1,
            "No sessions yet — type a message to begin",
            theme::text_faint_style(),
            popup.width.saturating_sub(4),
        );
    } else {
        let scroll = picker.selected.saturating_sub(list_height as usize - 1);
        let visible_items: Vec<_> = picker
            .items
            .iter()
            .skip(scroll)
            .take(list_height as usize)
            .collect();
        let time_col = visible_items
            .iter()
            .map(|s| display_width(&s.relative_time()))
            .max()
            .unwrap_or(6)
            .max(6);
        let count_col = visible_items
            .iter()
            .map(|s| display_width(&s.message_count.to_string()))
            .max()
            .unwrap_or(2)
            .max(2);
        for (row, session) in visible_items.iter().enumerate() {
            let selected = scroll + row == picker.selected;
            let prefix = if selected { "> " } else { "  " };
            let preview = session.first_message.replace('\n', " ");
            let cwd = session.cwd_label().unwrap_or_default();
            let line = format!(
                "{prefix}{:<time_col$} {:>count_col$} {}{}",
                session.relative_time(),
                session.message_count,
                preview,
                if cwd.is_empty() {
                    String::new()
                } else {
                    format!(" {cwd}")
                },
            );
            let style = if selected {
                fg(theme::text_primary()).bg(theme::surface_raised())
            } else {
                fg(theme::text_subtle())
            };
            frame.write_text(
                popup.x + 1,
                popup.y + 1 + row as u16,
                &line,
                style,
                popup.width.saturating_sub(2),
            );
        }
    }

    // Dismissal hint in the bottom border row. The overlay is a centered
    // box drawn on top of history with no scrim; the user needs at least
    // one explicit signal that it's modal and how to close it.
    let hint = "esc close · ↑↓ choose · enter open";
    let hint_width = display_width(hint) as u16;
    if popup.width > hint_width + 4 {
        frame.write_text(
            popup.x + popup.width - hint_width - 2,
            popup.y + popup.height - 1,
            hint,
            theme::text_faint_style(),
            hint_width,
        );
    }
}

fn draw_skill_picker(frame: &mut Frame<'_>, app: &App, history_area: Rect) {
    let Some(picker) = app.skill_picker_state() else {
        return;
    };
    let width = 60u16.min(history_area.width.saturating_sub(4));
    let list_height = picker.items.len().min(15).max(1) as u16;
    let height = list_height + 3; // title + list + footer
    if width < 4 || history_area.height < height {
        return;
    }
    draw_overlay_scrim(frame, history_area);
    let popup = centered_rect(history_area, width, height);
    frame.draw_box(
        popup,
        fg(theme::border_faint()),
        Some(bg(theme::surface_deep())),
    );
    frame.write_text(
        popup.x + 2,
        popup.y,
        &format!("Skills ({})", picker.items.len()),
        fg(theme::brand()).add_modifier(Modifier::Bold),
        popup.width.saturating_sub(4),
    );

    if picker.items.is_empty() {
        frame.write_text(
            popup.x + 2,
            popup.y + 1,
            "No skills installed",
            theme::text_faint_style(),
            popup.width.saturating_sub(4),
        );
    } else {
        let scroll = picker.selected.saturating_sub(list_height as usize - 1);
        let visible_items: Vec<_> = picker
            .items
            .iter()
            .skip(scroll)
            .take(list_height as usize)
            .collect();
        let name_col = visible_items
            .iter()
            .map(|(name, _)| display_width(name))
            .max()
            .unwrap_or(8)
            .max(8);
        for (row, (name, desc)) in visible_items.iter().enumerate() {
            let selected = scroll + row == picker.selected;
            let prefix = if selected { "> " } else { "  " };
            let line = format!("{prefix}{:<width$} {}", name, desc, width = name_col);
            let style = if selected {
                fg(theme::text_primary()).bg(theme::surface_raised())
            } else {
                fg(theme::text_subtle())
            };
            frame.write_text(
                popup.x + 1,
                popup.y + 1 + row as u16,
                &line,
                style,
                popup.width.saturating_sub(2),
            );
        }
    }

    let hint = "esc close · ↑↓ choose · enter insert";
    let hint_width = display_width(hint) as u16;
    if popup.width > hint_width + 4 {
        frame.write_text(
            popup.x + popup.width - hint_width - 2,
            popup.y + popup.height - 1,
            hint,
            theme::text_faint_style(),
            hint_width,
        );
    }
}

fn draw_lines_region(frame: &mut Frame<'_>, area: Rect, lines: &[Line<'static>], style: Style) {
    frame.fill(area, ' ', style);
    for (idx, line) in lines.iter().enumerate().take(area.height as usize) {
        frame.write_line(area.x, area.y + idx as u16, line, area.width);
    }
}

fn draw_top_bottom_rule(frame: &mut Frame<'_>, area: Rect, style: Style) {
    for x in 0..area.width {
        frame.write_text(area.x + x, area.y, "─", style, 1);
        frame.write_text(
            area.x + x,
            area.y + area.height.saturating_sub(1),
            "─",
            style,
            1,
        );
    }
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn fg(color: lash_tui::Color) -> Style {
    Style::default().fg(color)
}

fn bg(color: lash_tui::Color) -> Style {
    Style::default().bg(color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::{PromptRequest, PromptUsage};
    use std::sync::mpsc;

    use crate::overlay::{PromptFocus, PromptState, WaitState};

    #[test]
    fn animated_lash_word_cycles_slash_through_wordmark() {
        let frames = [0_u64, 200, 400, 600, 800]
            .into_iter()
            .map(std::time::Duration::from_millis)
            .map(animated_lash_word)
            .map(|spans| {
                spans
                    .into_iter()
                    .map(|span| span.content.into_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert_eq!(frames, vec!["/LASH", "L/ASH", "LA/SH", "LAS/H", "LASH/"]);
    }

    #[test]
    fn status_bar_shows_context_window_usage() {
        let mut app = App::new("gpt-5.4".into(), "test".into());
        app.model_variant = Some("high".into());
        app.context_window = Some(1_100_000);
        app.last_prompt_usage = Some(PromptUsage {
            prompt_context_tokens: 0,
            input_tokens: 0,
            cached_input_tokens: 0,
            context_budget_tokens: 7_000,
        });

        let snapshot = lash_tui::render_snapshot(80, 4, |frame| draw(frame, &mut app));
        let top = snapshot.visible_line_trimmed(0);
        assert!(top.contains("lash · gpt-5.4 · high"));
        // Context meter: tokens used + integer percent, separated by a
        // middle dot. The old representation was `7.0k / 1.1M (0.6%)`,
        // which gave three renderings of the same number.
        assert!(top.contains("7.0k · 1%"));
    }

    #[test]
    fn input_badge_omits_session_name() {
        let mut app = App::new("gpt-5.4".into(), "autumn-falls".into());
        app.repo_status = Some(crate::repo_status::RepoStatus {
            repo_root: std::path::PathBuf::from("/tmp/lash"),
            repo_name: "lash".into(),
            branch: "staging".into(),
            worktree: None,
        });

        let snapshot = lash_tui::render_snapshot(84, 8, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("lash · staging"));
        assert!(!visible.contains("autumn-falls"));
    }

    #[test]
    fn option_prompt_starts_at_top_of_question() {
        let mut app = App::new("gpt-5.4".into(), "test".into());
        let (response_tx, _response_rx) = mpsc::channel();
        app.show_prompt(PromptState {
            request: PromptRequest::single(
                "Plan .lash/plans/15d5a2bd-841d-4729-8968-ae7874385e16.md is ready. Exit plan mode now?",
                vec!["Exit plan mode".into(), "Keep planning".into()],
            )
            .with_optional_note(),
            focus: PromptFocus::Options,
            cursor: 0,
            scroll_offset: 0,
            selected: Default::default(),
            reply_text: String::new(),
            reply_cursor: 0,
            response_tx,
        });

        let snapshot = lash_tui::render_snapshot(84, 14, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");
        assert!(visible.contains("Plan .lash/plans/15d5a2bd-841d-4729-8968-ae7874385e16.md"));
        assert!(visible.contains("Choices"));
        assert!(!visible.contains("Question"));
        assert!(!visible.contains("┌"));
    }

    #[test]
    fn prompt_panel_can_scroll_when_content_exceeds_viewport() {
        let mut app = App::new("gpt-5.4".into(), "test".into());
        let (response_tx, _response_rx) = mpsc::channel();
        app.show_prompt(PromptState {
            request: PromptRequest::single("Exit plan mode?", vec!["Exit".into()])
                .with_markdown_panel(
                    "PLAN",
                    "# Plan\n\nline 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\nline 11\nline 12",
                ),
            focus: PromptFocus::Options,
            cursor: 0,
            scroll_offset: 5,
            selected: Default::default(),
            reply_text: String::new(),
            reply_cursor: 0,
            response_tx,
        });

        let snapshot = lash_tui::render_snapshot(60, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");
        assert!(!visible.contains("line 1"));
        assert!(visible.contains("line 6") || visible.contains("line 7"));
    }

    #[test]
    fn wait_status_strip_shows_waiting_and_remaining_time() {
        let mut app = App::new("gpt-5.4".into(), "test".into());
        app.start_turn();
        let (response_tx, _response_rx) = mpsc::channel();
        app.show_wait(WaitState::from_request(
            PromptRequest::freeform("Pausing briefly before continuing.").with_wait(5),
            response_tx,
        ));

        let snapshot = lash_tui::render_snapshot(72, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("/LASH  Waiting"));
        assert!(visible.contains("5s left"));
        assert!(visible.contains("Auto-resume in 5s"));
    }

    #[test]
    fn history_selection_highlights_visible_cells() {
        let mut app = App::new("gpt-5.4".into(), "test".into());
        app.blocks = vec![crate::app::DisplayBlock::UserInput(
            "alpha\nbeta\ngamma".into(),
        )];
        app.selection.anchor = (2, 1);
        app.selection.end = (5, 1);
        app.selection.visible = true;

        let history = render::history_area(&app, 40, 9);
        let snapshot = lash_tui::render_snapshot(40, 9, |frame| draw(frame, &mut app));
        assert_eq!(
            snapshot
                .cell(2, history.y + 1)
                .and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
        assert_eq!(
            snapshot
                .cell(4, history.y + 1)
                .and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
    }

    #[test]
    fn history_selection_tracks_content_rows_while_scrolled() {
        let mut app = App::new("gpt-5.4".into(), "test".into());
        app.blocks = vec![crate::app::DisplayBlock::UserInput(
            "alpha\nbeta\ngamma\ndelta".into(),
        )];
        app.scroll_offset = 1;
        app.selection.anchor = (2, 2);
        app.selection.end = (4, 2);
        app.selection.visible = true;

        let history = render::history_area(&app, 40, 9);
        let snapshot = lash_tui::render_snapshot(40, 9, |frame| draw(frame, &mut app));
        assert_eq!(
            snapshot
                .cell(2, history.y + 1)
                .and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
    }

    #[test]
    fn input_selection_highlights_visible_cells() {
        let mut app = App::new("gpt-5.4".into(), "test".into());
        app.set_input("alpha beta".into());
        app.start_input_selection(2);
        app.update_input_selection(7);
        app.finish_input_selection();

        let input = render::input_content_area(&app, 40, 9);
        let snapshot = lash_tui::render_snapshot(40, 9, |frame| draw(frame, &mut app));
        assert_eq!(
            snapshot
                .cell(input.x + 4, input.y)
                .and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
        assert_eq!(
            snapshot
                .cell(input.x + 8, input.y)
                .and_then(|cell| cell.style.bg),
            Some(theme::SELECTION_BG)
        );
    }
}
