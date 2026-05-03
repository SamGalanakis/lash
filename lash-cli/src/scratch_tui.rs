use lash_tui::{Frame, Line, Modifier, Rect, Span, Style, TermCapabilities};
use lash_ui::{UiRenderContext, UiSurfaceScene, UiSurfaceSlot};
use unicode_width::UnicodeWidthStr;

use crate::app::{App, format_tokens};
#[cfg(test)]
use crate::chrome_ui::animated_lash_word;
use crate::editor::SuggestionKind;
use crate::{render, theme};

const INPUT_HORIZONTAL_PADDING: u16 = 1;
const PROMPT_HORIZONTAL_PADDING: u16 = 1;

pub fn draw(frame: &mut Frame<'_>, app: &mut App) {
    draw_with_capabilities(frame, app, TermCapabilities::default());
}

pub fn draw_with_capabilities(
    frame: &mut Frame<'_>,
    app: &mut App,
    capabilities: TermCapabilities,
) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }

    frame.clear(bg(theme::surface_base()));

    sync_chrome_turn_status(app);

    let history = render::history_area(app, area.width, area.height);
    let dock_area = render::dock_area(app, area.width, area.height);
    let queue_lines = render::queue_preview_lines_snapshot(app, area.width);
    let queue_area = render::queue_area(app, area.width, area.height);
    let footer_area = render::footer_area(app, area.width, area.height);
    let input_area = render::input_area(app, area.width, area.height);
    let body_area = render::body_area(app, area.width, area.height);

    let surfaces = app.ui_extensions().surface_scene();
    draw_status_bar(frame, app, Rect::new(0, 0, area.width, 1));
    let surfaces = sync_surface_areas(app, surfaces, history, dock_area, footer_area, body_area);
    if surfaces.has_slot(UiSurfaceSlot::Workspace) {
        draw_workspace_surface(frame, app, &surfaces, history, capabilities);
    } else {
        draw_history(frame, app, history);
        apply_selection_highlight(frame, app, history);
    }
    if dock_area.height > 0 {
        draw_surface_stack(
            frame,
            app,
            &surfaces,
            UiSurfaceSlot::Dock,
            dock_area,
            capabilities,
        );
    }
    if queue_area.height > 0 {
        draw_lines_region(frame, queue_area, &queue_lines, bg(theme::surface_raised()));
    }
    if footer_area.height > 0 {
        draw_surface_stack(
            frame,
            app,
            &surfaces,
            UiSurfaceSlot::Footer,
            footer_area,
            capabilities,
        );
    }
    if app.has_prompt() {
        draw_overlay_scrim(frame, history);
        draw_prompt(frame, app, input_area);
    } else {
        draw_input(frame, app, input_area);
        draw_suggestions(frame, app, input_area);
    }
    draw_session_picker(frame, app, history);
    draw_tree(frame, app, history);
    draw_skill_picker(frame, app, history);
    draw_overlay_surface(frame, app, &surfaces, body_area, capabilities);
}

fn sync_surface_areas(
    app: &App,
    mut surfaces: UiSurfaceScene,
    history_area: Rect,
    dock_area: Rect,
    footer_area: Rect,
    body_area: Rect,
) -> UiSurfaceScene {
    let mut assignments = Vec::new();
    if let Some(surface) = surfaces.workspace.last_mut() {
        surface.area = Some(history_area);
        assignments.push((surface.id.clone(), history_area));
    }
    let mut dock_y = dock_area.y;
    let mut dock_remaining = dock_area.height;
    for surface in &mut surfaces.dock {
        if dock_remaining == 0 {
            break;
        }
        let height = surface.size.height().min(dock_remaining);
        if height == 0 {
            continue;
        }
        let area = Rect::new(dock_area.x, dock_y, dock_area.width, height);
        surface.area = Some(area);
        assignments.push((surface.id.clone(), area));
        dock_y = dock_y.saturating_add(height);
        dock_remaining = dock_remaining.saturating_sub(height);
    }
    let mut footer_y = footer_area.y;
    let mut footer_remaining = footer_area.height;
    for surface in &mut surfaces.footer {
        if footer_remaining == 0 {
            break;
        }
        let height = surface.size.height().min(footer_remaining);
        if height == 0 {
            continue;
        }
        let area = Rect::new(footer_area.x, footer_y, footer_area.width, height);
        surface.area = Some(area);
        assignments.push((surface.id.clone(), area));
        footer_y = footer_y.saturating_add(height);
        footer_remaining = footer_remaining.saturating_sub(height);
    }
    if let Some(surface) = surfaces.overlay.last_mut()
        && body_area.width > 0
        && body_area.height > 0
    {
        let width = surface
            .size
            .width()
            .unwrap_or_else(|| body_area.width.saturating_sub(4).max(1))
            .min(body_area.width);
        let height = surface.size.height().min(body_area.height).max(1);
        let x = body_area.x + body_area.width.saturating_sub(width) / 2;
        let y = body_area.y + body_area.height.saturating_sub(height) / 2;
        let area = Rect::new(x, y, width, height);
        surface.area = Some(area);
        assignments.push((surface.id.clone(), area));
    }
    app.ui_extensions().sync_surface_areas(assignments);
    surfaces
}

fn draw_workspace_surface(
    frame: &mut Frame<'_>,
    app: &App,
    surfaces: &UiSurfaceScene,
    area: Rect,
    capabilities: TermCapabilities,
) {
    let Some(surface) = surfaces.workspace.last() else {
        return;
    };
    let mut viewport = frame.viewport(area);
    app.ui_extensions().render_mounted_surface(
        surface,
        UiRenderContext {
            session_id: app.session_id.as_str(),
            capabilities,
            surface_id: &surface.id,
            focused: surfaces.focused.as_deref() == Some(surface.id.as_str()),
        },
        &mut viewport,
    );
}

fn draw_surface_stack(
    frame: &mut Frame<'_>,
    app: &App,
    surfaces: &UiSurfaceScene,
    slot: UiSurfaceSlot,
    area: Rect,
    capabilities: TermCapabilities,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    for surface in surfaces.surfaces(slot) {
        let Some(surface_area) = surface.area else {
            continue;
        };
        let mut viewport = frame.viewport(surface_area);
        viewport.clear(bg(theme::surface_raised()));
        app.ui_extensions().render_mounted_surface(
            surface,
            UiRenderContext {
                session_id: app.session_id.as_str(),
                capabilities,
                surface_id: &surface.id,
                focused: surfaces.focused.as_deref() == Some(surface.id.as_str()),
            },
            &mut viewport,
        );
    }
}

fn draw_overlay_surface(
    frame: &mut Frame<'_>,
    app: &App,
    surfaces: &UiSurfaceScene,
    area: Rect,
    capabilities: TermCapabilities,
) {
    let Some(surface) = surfaces.overlay.last() else {
        return;
    };
    let Some(surface_area) = surface.area else {
        return;
    };
    draw_overlay_scrim(frame, area);
    let mut viewport = frame.viewport(surface_area);
    viewport.clear(bg(theme::surface_base()));
    app.ui_extensions().render_mounted_surface(
        surface,
        UiRenderContext {
            session_id: app.session_id.as_str(),
            capabilities,
            surface_id: &surface.id,
            focused: surfaces.focused.as_deref() == Some(surface.id.as_str()),
        },
        &mut viewport,
    );
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
    if slots.len() > u64::BITS as usize {
        return fit_slots_fallback(slots, budget);
    }

    let mut included = 0u64;
    let mut visited = 0u64;
    let mut used = 0usize;
    while (visited.count_ones() as usize) < slots.len() {
        let mut best_idx = None;
        let mut best_priority = 0u8;
        for (idx, slot) in slots.iter().enumerate() {
            let bit = 1u64 << idx;
            if visited & bit != 0 {
                continue;
            }
            if best_idx.is_none() || slot.priority > best_priority {
                best_idx = Some(idx);
                best_priority = slot.priority;
            }
        }
        let Some(idx) = best_idx else {
            break;
        };
        let bit = 1u64 << idx;
        visited |= bit;
        let width = slots[idx].width();
        if width > 0 && used + width <= budget {
            included |= bit;
            used += width;
        }
    }

    let span_count = slots
        .iter()
        .enumerate()
        .filter(|(idx, _)| included & (1u64 << idx) != 0)
        .map(|(_, slot)| slot.spans.len())
        .sum();
    let mut spans = Vec::with_capacity(span_count);
    for (idx, slot) in slots.iter().enumerate() {
        if included & (1u64 << idx) != 0 {
            spans.extend(slot.spans.iter().cloned());
        }
    }
    spans
}

fn fit_slots_fallback(slots: &[StatusSlot], budget: usize) -> Vec<Span<'static>> {
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

    let span_count = slots
        .iter()
        .enumerate()
        .filter(|(idx, _)| included[*idx])
        .map(|(_, slot)| slot.spans.len())
        .sum();
    let mut spans = Vec::with_capacity(span_count);
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
    // Don't add `live_output_tokens_estimate` to either branch's total: on the
    // very first turn, before `last_response_usage` lands, that's the only
    // nonzero number — and using it as the displayed total reads as if the
    // streamed output bytes were the entire context, producing nonsense like
    // `36 · 0%` against a 1.1M-token window. Wait for real input accounting.
    let Some(context_window) = app.context_window else {
        let total = app.token_usage.input_tokens + app.token_usage.output_tokens;
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
        .or_else(|| {
            let total = app.token_usage.input_tokens + app.token_usage.output_tokens;
            (total > 0).then_some(total)
        })?;

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
        (app.timeline.len(), scroll - block_height)
    } else {
        render::find_visible_block(app, scroll)
    };

    let mut written_rows = 0usize;
    'blocks: for idx in first_idx..app.timeline.len() {
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
        && let Some(live_lines) = app.live_reasoning_lines_snapshot()
    {
        if app.live_reasoning_leading_padding() > 0 {
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
        skip_lines = skip_lines.saturating_sub(live_lines.len());
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
        // Advance skip_lines past the live markdown content rows so a
        // trailing plan-dock render picks up at the right offset.
        skip_lines = skip_lines.saturating_sub(live_lines.len());
    }

    // Plan checklist renders at the logical tail of history — part of
    // the scroll, not a pinned dock. Appears just after the live-
    // assistant trace so new turns push it down the transcript.
    if written_rows < viewport_height
        && let Some(plan_lines) = render::plan_dock_lines_snapshot(app, area.width)
    {
        for line in plan_lines.iter().skip(skip_lines) {
            if written_rows >= viewport_height {
                break;
            }
            frame.write_line(area.x, area.y + written_rows as u16, line, area.width);
            written_rows += 1;
        }
        skip_lines = skip_lines.saturating_sub(plan_lines.len());
    }

    if written_rows < viewport_height
        && let Some(task_lines) = render::background_task_lines_snapshot(app, area.width)
    {
        for line in task_lines.iter().skip(skip_lines) {
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

/// Push the live-turn snapshot into the `chrome_ui` extension and
/// mount/unmount its `turn_status` footer surface accordingly. The
/// indicator is hidden whenever a prompt is open — the prompt panel
/// itself communicates the paused state.
///
/// Callable from layout-query paths (e.g. tests) so the surface
/// registry is in sync before `chrome_layout` reads footer heights.
pub(crate) fn sync_chrome_turn_status(app: &App) {
    use crate::chrome_ui::{
        CHROME_UI_ID, TURN_STATUS_KEY, TurnStatusLabel, TurnStatusSnapshot, set_turn_status,
        turn_status_surface_spec,
    };

    let snapshot = if app.has_prompt() {
        None
    } else {
        let background = background_task_summary(app);
        Some(match app.live_turn.as_ref() {
            Some(turn) => TurnStatusSnapshot {
                label: match turn.status_text.as_str() {
                    "error" => TurnStatusLabel::Error,
                    "thinking" => TurnStatusLabel::Thinking,
                    "responding" => TurnStatusLabel::Responding,
                    status if status.contains("wait") => TurnStatusLabel::Waiting,
                    _ => TurnStatusLabel::RunningTool,
                },
                turn_started_at: Some(turn.turn_started_at),
                detail: combine_status_detail(turn.status_detail.as_deref(), background),
            },
            None => TurnStatusSnapshot {
                label: TurnStatusLabel::Idle,
                turn_started_at: None,
                detail: background,
            },
        })
    };

    set_turn_status(&app.chrome_state, snapshot.clone());

    let extensions = app.ui_extensions();
    let mounted = extensions.surface_is_mounted(CHROME_UI_ID, TURN_STATUS_KEY);
    match (snapshot.is_some(), mounted) {
        (true, false) => extensions.mount_surface(CHROME_UI_ID, turn_status_surface_spec()),
        (false, true) => extensions.unmount_surface(CHROME_UI_ID, TURN_STATUS_KEY),
        _ => {}
    }
}

fn combine_status_detail(primary: Option<&str>, background: Option<String>) -> Option<String> {
    match (primary.filter(|value| !value.is_empty()), background) {
        (Some(primary), Some(background)) => Some(format!("{primary} · {background}")),
        (Some(primary), None) => Some(primary.to_string()),
        (None, Some(background)) => Some(background),
        (None, None) => None,
    }
}

fn background_task_summary(app: &App) -> Option<String> {
    let running = app
        .background_tasks
        .iter()
        .filter(|task| task.run_state == lash::ManagedRunState::Running)
        .count();
    let idle = app
        .background_tasks
        .iter()
        .filter(|task| task.run_state == lash::ManagedRunState::Idle)
        .count();
    match (running, idle) {
        (0, 0) => None,
        (running, 0) => Some(format!(
            "{} background task{} running",
            running,
            if running == 1 { "" } else { "s" }
        )),
        (0, idle) => Some(format!(
            "{} background task{} idle",
            idle,
            if idle == 1 { "" } else { "s" }
        )),
        (running, idle) => Some(format!("{running} running · {idle} idle")),
    }
}

fn current_context_budget_tokens(app: &App) -> Option<i64> {
    if !app.running {
        return None;
    }
    let input = app.last_response_usage.input_tokens.max(0);
    let cached = app.last_response_usage.cached_input_tokens.max(0);
    // Suppress until input accounting from a completed response has landed.
    // Showing only the streaming output token estimate against the full
    // context window otherwise reads as `36 · 0%` on the first turn.
    if input == 0 && cached == 0 {
        return None;
    }
    let output = (app.last_response_usage.output_tokens + app.live_output_tokens_estimate).max(0);
    Some(if app.context_usage_excludes_cached_input {
        input + output + cached
    } else {
        (input - cached).max(0) + output + cached
    })
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
    let visible = area.height as usize;
    let snapshot = render::prompt_render_snapshot(prompt, inner_width.max(1), visible);
    let max_scroll = render::prompt_max_scroll(prompt, inner_width.max(1), visible);
    let scroll = if prompt.is_text_entry() && !snapshot.split_layout {
        max_scroll
    } else {
        prompt.scroll_offset.min(max_scroll)
    };
    let content_width = area
        .width
        .saturating_sub(PROMPT_HORIZONTAL_PADDING.saturating_mul(2));
    if snapshot.split_layout {
        for (idx, line) in snapshot
            .review_lines
            .iter()
            .skip(scroll)
            .take(snapshot.review_viewport_height)
            .enumerate()
        {
            frame.write_line(
                area.x + PROMPT_HORIZONTAL_PADDING,
                area.y + idx as u16,
                line,
                content_width,
            );
        }
        let interaction_y = area.y + snapshot.review_viewport_height as u16;
        for (idx, line) in snapshot.interaction_lines.iter().enumerate() {
            frame.write_line(
                area.x + PROMPT_HORIZONTAL_PADDING,
                interaction_y + idx as u16,
                line,
                content_width,
            );
        }
    } else {
        for (idx, line) in snapshot
            .combined_lines
            .iter()
            .skip(scroll)
            .take(visible)
            .enumerate()
        {
            frame.write_line(
                area.x + PROMPT_HORIZONTAL_PADDING,
                area.y + idx as u16,
                line,
                content_width,
            );
        }
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
        .map(|s| display_width(&s.name))
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
    let is_indexing = app.suggestion_kind() == SuggestionKind::Indexing;
    for (idx, suggestion) in app.suggestions().iter().take(max_visible).enumerate() {
        let selected = !is_indexing && idx == app.suggestion_idx();
        let base_style = if selected {
            fg(theme::text_primary()).bg(theme::surface_raised())
        } else if is_indexing {
            fg(theme::text_subtle())
                .add_modifier(Modifier::Italic)
                .add_modifier(Modifier::Dim)
        } else {
            fg(theme::text_subtle())
        };
        let line = build_suggestion_line(suggestion, name_col, base_style, selected);
        frame.write_line_styled(
            popup.x + 1,
            popup.y + 1 + idx as u16,
            &line,
            base_style,
            popup.width.saturating_sub(2),
        );
    }
}

fn build_suggestion_line<'a>(
    suggestion: &'a crate::editor::Suggestion,
    name_col: usize,
    base_style: Style,
    selected: bool,
) -> Line<'a> {
    let mut spans: Vec<Span<'a>> = Vec::new();
    spans.push(Span::raw(" "));

    // Bold the matched chars on top of `base_style`. Selected rows already
    // read at full strength, so we only add bold; on unselected rows we also
    // bump the foreground to text_primary so the matched chars actually pop
    // against the dim base style.
    let mut highlight = base_style.add_modifier(Modifier::Bold);
    if !selected {
        highlight = highlight.fg(theme::text_primary());
    }

    if suggestion.match_indices.is_empty() {
        spans.push(Span::raw(suggestion.name.as_str()));
    } else {
        let indices: std::collections::HashSet<u32> =
            suggestion.match_indices.iter().copied().collect();
        let mut current = String::new();
        let mut current_is_match: Option<bool> = None;
        for (char_idx, ch) in suggestion.name.chars().enumerate() {
            let is_match = indices.contains(&(char_idx as u32));
            match current_is_match {
                Some(prev) if prev == is_match => current.push(ch),
                Some(prev) => {
                    let style = if prev { highlight } else { base_style };
                    spans.push(Span::styled(std::mem::take(&mut current), style));
                    current.push(ch);
                    current_is_match = Some(is_match);
                }
                None => {
                    current.push(ch);
                    current_is_match = Some(is_match);
                }
            }
        }
        if let Some(prev) = current_is_match {
            let style = if prev { highlight } else { base_style };
            spans.push(Span::styled(current, style));
        }
    }

    let name_width = display_width(&suggestion.name);
    if name_width < name_col {
        spans.push(Span::raw(" ".repeat(name_col - name_width)));
    }
    if !suggestion.description.is_empty() {
        spans.push(Span::raw(" "));
        spans.push(Span::raw(suggestion.description.as_str()));
    }
    Line::from(spans)
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
    let list_height = picker.items.len().clamp(1, 15) as u16;
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

fn draw_tree(frame: &mut Frame<'_>, app: &App, history_area: Rect) {
    let Some(tree) = app.tree_state() else {
        return;
    };
    let rows = tree.rows();
    let width = 96u16.min(history_area.width.saturating_sub(4));
    let list_height = rows.len().clamp(1, 18) as u16;
    let height = list_height + 3;
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
        &format!("Tree ({})", rows.len()),
        fg(theme::brand()).add_modifier(Modifier::Bold),
        popup.width.saturating_sub(4),
    );

    if rows.is_empty() {
        frame.write_text(
            popup.x + 2,
            popup.y + 1,
            "No messages yet",
            theme::text_faint_style(),
            popup.width.saturating_sub(4),
        );
    } else {
        let selected_idx = tree
            .selected_node_id
            .as_deref()
            .and_then(|selected| rows.iter().position(|row| row.node_id == selected))
            .unwrap_or(0);
        let scroll = selected_idx.saturating_sub(list_height as usize - 1);
        for (row_idx, row) in rows
            .iter()
            .skip(scroll)
            .take(list_height as usize)
            .enumerate()
        {
            let selected = scroll + row_idx == selected_idx;
            let depth_indent = "  ".repeat(row.depth);
            let branch = if row.has_children {
                if row.collapsed { "▸" } else { "▾" }
            } else {
                "·"
            };
            let role = match row.message.role {
                lash::MessageRole::User => "user",
                lash::MessageRole::Assistant => "assistant",
                lash::MessageRole::System => "system",
            };
            let preview = crate::overlay::tree_message_preview(&row.message);
            let active_marker = if row.active { " *" } else { "" };
            let line = format!(
                "{}{} {} [{}] {}{}",
                if selected { "> " } else { "  " },
                depth_indent,
                branch,
                role,
                preview,
                active_marker
            );
            let style = if selected {
                fg(theme::text_primary()).bg(theme::surface_raised())
            } else if row.active {
                fg(theme::brand())
            } else {
                fg(theme::text_subtle())
            };
            frame.write_text(
                popup.x + 1,
                popup.y + 1 + row_idx as u16,
                &line,
                style,
                popup.width.saturating_sub(2),
            );
        }
    }

    let hint = "esc close · ↑↓ move · enter switch · ctrl/alt ←→ branch";
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
    let list_height = picker.items.len().clamp(1, 15) as u16;
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
    use lash_ui::{
        UiExtension, UiExtensions, UiHostEffect, UiRenderContext, UiSurfaceSize, UiSurfaceSlot,
        UiSurfaceSpec,
    };
    use std::sync::Arc;
    use std::sync::mpsc;

    use crate::overlay::{PromptFocus, PromptState};

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
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
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
    fn status_bar_hides_meter_during_first_turn_before_input_accounting_lands() {
        // Regression: while the very first response is streaming, only
        // `live_output_tokens_estimate` is nonzero; using it as the displayed
        // total reads as if those streamed bytes were the entire context, so
        // the bar shows e.g. `36 · 0%` against a 1.1M-token window. The
        // meter should stay hidden until real input accounting lands.
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.context_window = Some(1_100_000);
        app.running = true;
        app.live_output_tokens_estimate = 36;
        // No `last_prompt_usage`, no `last_response_usage` — first turn.

        let snapshot = lash_tui::render_snapshot(80, 4, |frame| draw(frame, &mut app));
        let top = snapshot.visible_line_trimmed(0);
        // The `·` separators between identity fields (`lash · gpt-5.4`)
        // are fine; what we don't want is a raw token count or a percent
        // sourced from the streaming output estimate.
        assert!(!top.contains('%'), "unexpected percent on top line: {top}");
        assert!(
            !top.contains("36"),
            "unexpected token count on top line: {top}"
        );
    }

    #[test]
    fn input_badge_omits_session_name() {
        let mut app = App::new(
            "gpt-5.4".into(),
            "autumn-falls".into(),
            "test-session-id".into(),
        );
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
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
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
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let (response_tx, _response_rx) = mpsc::channel();
        app.show_prompt(PromptState {
            request: PromptRequest::single("Exit plan mode?", vec!["Exit".into()])
                .with_markdown_panel(
                    "PLAN",
                    "# Plan\n\nline 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\nline 11\nline 12",
                ),
            focus: PromptFocus::Options,
            cursor: 0,
            scroll_offset: 7,
            selected: Default::default(),
            reply_text: String::new(),
            reply_cursor: 0,
            response_tx,
        });

        let snapshot = lash_tui::render_snapshot(60, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");
        assert!(!visible.contains("line 1"));
        assert!(visible.contains("line 6") || visible.contains("line 7"));
        assert!(visible.contains("Choices"));
        assert!(visible.contains("Exit"));
    }

    #[test]
    fn history_selection_highlights_visible_cells() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.timeline = vec![crate::app::UiTimelineItem::UserInput(
            "alpha\nbeta\ngamma".into(),
        )]
        .into();
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
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.timeline = vec![crate::app::UiTimelineItem::UserInput(
            "alpha\nbeta\ngamma\ndelta".into(),
        )]
        .into();
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
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
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

    struct SurfaceTestExtension;

    #[async_trait::async_trait]
    impl UiExtension for SurfaceTestExtension {
        fn id(&self) -> &'static str {
            "surface_test"
        }

        async fn invoke_action(
            &self,
            _action: &str,
            _arg: Option<&str>,
            _ctx: lash_ui::UiContext<'_>,
        ) -> Result<Vec<UiHostEffect>, String> {
            Ok(Vec::new())
        }

        fn render_surface(
            &self,
            surface_key: &str,
            _ctx: UiRenderContext<'_>,
            frame: &mut Frame<'_>,
        ) {
            let label = match surface_key {
                "workspace" => "WORKSPACE",
                "footer" => "FOOTER",
                "overlay" => "OVERLAY",
                other => other,
            };
            frame.write_text(0, 0, label, Style::default(), frame.area().width);
        }

        fn handle_session_event(&self, event: &lash::SessionEvent) -> Vec<UiHostEffect> {
            match event {
                lash::SessionEvent::TextDelta { content } if content == "mount" => vec![
                    UiHostEffect::MountSurface {
                        spec: UiSurfaceSpec {
                            key: "workspace".to_string(),
                            slot: UiSurfaceSlot::Workspace,
                            size: UiSurfaceSize::Auto,
                            order: 0,
                            focusable: true,
                            visible: true,
                            modal: false,
                        },
                    },
                    UiHostEffect::MountSurface {
                        spec: UiSurfaceSpec {
                            key: "footer".to_string(),
                            slot: UiSurfaceSlot::Footer,
                            size: UiSurfaceSize::Lines(1),
                            order: 0,
                            focusable: false,
                            visible: true,
                            modal: false,
                        },
                    },
                ],
                lash::SessionEvent::TextDelta { content } if content == "overlay" => vec![
                    UiHostEffect::MountSurface {
                        spec: UiSurfaceSpec {
                            key: "overlay".to_string(),
                            slot: UiSurfaceSlot::Overlay,
                            size: UiSurfaceSize::Fixed {
                                width: 16,
                                height: 3,
                            },
                            order: 10,
                            focusable: true,
                            visible: true,
                            modal: true,
                        },
                    },
                    UiHostEffect::FocusSurface {
                        key: "overlay".to_string(),
                    },
                ],
                _ => Vec::new(),
            }
        }
    }

    #[test]
    fn workspace_surface_replaces_history_and_footer_renders_above_input() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.timeline = vec![crate::app::UiTimelineItem::UserInput("history line".into())].into();
        let ui_extensions = Arc::new(
            UiExtensions::new(vec![Arc::new(SurfaceTestExtension)]).expect("surface extensions"),
        );
        ui_extensions.effects_for_session_event(&lash::SessionEvent::TextDelta {
            content: "mount".to_string(),
        });
        app.set_ui_extensions(Arc::clone(&ui_extensions));

        let snapshot = lash_tui::render_snapshot(40, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("WORKSPACE"));
        assert!(visible.contains("FOOTER"));
        assert!(!visible.contains("history line"));
    }

    #[test]
    fn overlay_surface_renders_last_on_centered_scrim() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let ui_extensions = Arc::new(
            UiExtensions::new(vec![Arc::new(SurfaceTestExtension)]).expect("surface extensions"),
        );
        ui_extensions.effects_for_session_event(&lash::SessionEvent::TextDelta {
            content: "mount".to_string(),
        });
        ui_extensions.effects_for_session_event(&lash::SessionEvent::TextDelta {
            content: "overlay".to_string(),
        });
        app.set_ui_extensions(Arc::clone(&ui_extensions));

        let snapshot = lash_tui::render_snapshot(40, 12, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("OVERLAY"));
    }
}
