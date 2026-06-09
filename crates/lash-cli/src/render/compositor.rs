use lash_tui::{Frame, Line, Modifier, Rect, Span, Style, TermCapabilities};
use lash_tui_extensions::{TuiRenderContext, TuiSurfaceScene, TuiSurfaceSlot};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::{App, CliRunState, format_tokens};
use crate::chrome_ui::TurnStatusLabel;
#[cfg(test)]
use crate::chrome_ui::animated_lash_word;
use crate::editor::SuggestionKind;
use crate::text_layout::{centered_rect, display_width, selection_ordered};
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

    // One layout pass for the whole frame; the `render::*_area` accessors each
    // recompute `chrome_layout`, so calling them per region would repeat that
    // work five times per draw.
    let render::ChromeAreas {
        status: status_area,
        history,
        dock: dock_area,
        queue: queue_area,
        footer: footer_area,
        input: input_area,
        process: process_area,
        body: body_area,
    } = render::chrome_areas(app, area.width, area.height);
    let queue_lines = render::queue_preview_lines_snapshot(app, area.width);

    let surfaces = app.ui_extensions().surface_scene();
    draw_status_bar(frame, app, status_area);
    let surfaces = sync_surface_areas(app, surfaces, history, dock_area, footer_area, body_area);
    if surfaces.has_slot(TuiSurfaceSlot::Workspace) {
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
            TuiSurfaceSlot::Dock,
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
            TuiSurfaceSlot::Footer,
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
    draw_process_dock(frame, app, process_area);
    draw_session_picker(frame, app, history);
    draw_tree(frame, app, history);
    draw_skill_picker(frame, app, history);
    draw_process_overview(frame, app, body_area);
    draw_document_overlay(frame, app, body_area);
    draw_overlay_surface(frame, app, &surfaces, body_area, capabilities);
}

fn sync_surface_areas(
    app: &App,
    mut surfaces: TuiSurfaceScene,
    history_area: Rect,
    dock_area: Rect,
    footer_area: Rect,
    body_area: Rect,
) -> TuiSurfaceScene {
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
    surfaces: &TuiSurfaceScene,
    area: Rect,
    capabilities: TermCapabilities,
) {
    let Some(surface) = surfaces.workspace.last() else {
        return;
    };
    let mut viewport = frame.viewport(area);
    app.ui_extensions().render_mounted_surface(
        surface,
        TuiRenderContext {
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
    surfaces: &TuiSurfaceScene,
    slot: TuiSurfaceSlot,
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
            TuiRenderContext {
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
    surfaces: &TuiSurfaceScene,
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
        TuiRenderContext {
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

    // LEFT side: brand · model · execution mode · variant
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
    if !app.execution_mode_label.is_empty() {
        left.push(StatusSlot {
            spans: vec![
                Span::styled(" · ", sep_style),
                Span::styled(app.execution_mode_label.clone(), theme::text_subtle_style()),
            ],
            priority: 70,
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

    // RIGHT side: context-window meter. The expand keybind is searchable
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
    let Some(context_window) = app.usage.context_window else {
        let total = app.usage.token_usage.input_tokens + app.usage.token_usage.output_tokens;
        if total <= 0 {
            return None;
        }
        return Some(StatusSlot {
            spans: vec![
                Span::styled("ctx ", theme::text_faint_style()),
                Span::styled(format_tokens(total), theme::text_subtle_style()),
                Span::raw(" "),
            ],
            priority: 70,
        });
    };

    let used = current_context_budget_tokens(app)
        .or_else(|| {
            app.usage
                .last_prompt_usage
                .as_ref()
                .map(|usage| usage.context_budget_tokens as i64)
                .filter(|used| *used > 0)
        })
        .or_else(|| {
            let total = app.usage.token_usage.input_tokens + app.usage.token_usage.output_tokens;
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
            Span::styled("ctx ", theme::text_faint_style()),
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
    }

    if let Some((x, y, height)) = render::history_scroll_indicator(app, area) {
        for offset in 0..height {
            frame.write_text(x, y + offset, "│", fg(theme::text_subtle()), 1);
        }
    }

    if written_rows == 0
        && app.live_tool_output_anchor_block_index().is_none()
        && !app.live.reasoning.has_renderable_output()
        && !app.live.assistant.has_renderable_output()
        && app.plan_dock.as_ref().is_none_or(|plan| plan.is_empty())
    {
        draw_empty_history_state(frame, app, area);
    }
}

fn draw_empty_history_state(frame: &mut Frame<'_>, app: &App, area: Rect) {
    if area.width == 0 || area.height == 0 || app.overlay.is_some() || app.has_suggestions() {
        return;
    }
    if area.width < 54 || area.height < 12 {
        let text = "Type a message or / for commands";
        let width = display_width(text) as u16;
        if area.width > width {
            frame.write_text(
                area.x + area.width.saturating_sub(width) / 2,
                area.y + area.height / 2,
                text,
                theme::text_faint_style(),
                width,
            );
        }
        return;
    }

    let logo: &[(&str, &str)] = &[
        ("██       ████   ██████  ", "  ██"),
        ("██      ██  ██  ██     ", "█  ██"),
        ("██      ██████  ██████", "██████"),
        ("██      ██  ██      █", " ██  ██"),
        ("██████  ██  ██  ████", "  ██  ██"),
    ];
    let content_width = 30u16;
    let start_x = area.x + area.width.saturating_sub(content_width) / 2;
    let start_y = area.y + area.height.saturating_sub(7) / 2;
    for (row, &(before, after)) in logo.iter().enumerate() {
        frame.write_line(
            start_x,
            start_y + row as u16,
            &Line::from(vec![
                Span::styled(before.to_string(), theme::assistant_text()),
                Span::styled("██", Style::default().fg(theme::brand())),
                Span::styled(after.to_string(), theme::assistant_text()),
            ]),
            content_width,
        );
    }
    frame.write_line(
        start_x,
        start_y + 5,
        &Line::from(vec![
            Span::styled("──────────", Style::default().fg(theme::brand())),
            Span::styled("──────────", Style::default().fg(theme::border_dim())),
            Span::styled("──────────", Style::default().fg(theme::border_faint())),
        ]),
        content_width,
    );
}

fn draw_process_dock(frame: &mut Frame<'_>, app: &App, area: Rect) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let Some(lines) = render::process_lines_snapshot(app, area.width) else {
        return;
    };
    draw_lines_region(frame, area, &lines, bg(theme::surface_raised()));
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

/// Push the live-turn snapshot into the `chrome_ui` extension and
/// mount/unmount its `turn_status` footer surface accordingly. The
/// indicator is off whenever a prompt is open — the prompt panel
/// itself communicates the paused state.
///
/// Publish the latest footer status snapshot to the already-mounted chrome
/// surface.
pub(crate) fn sync_chrome_turn_status(app: &App) {
    use crate::chrome_ui::{TurnStatusLabel, TurnStatusSnapshot, set_turn_status};

    let snapshot = if app.has_prompt() {
        None
    } else {
        let background = process_summary(app);
        Some(match app.live.turn.as_ref() {
            Some(turn) if app.turn_active() => TurnStatusSnapshot {
                label: turn_status_label_for_state(turn.run_state),
                turn_started_at: Some(turn.turn_started_at),
                detail: combine_status_detail(turn.status_detail.as_deref(), background),
            },
            Some(turn)
                if turn.run_state == CliRunState::Error
                    && turn
                        .transient_until
                        .is_some_and(|until| until > std::time::Instant::now()) =>
            {
                TurnStatusSnapshot {
                    label: TurnStatusLabel::Error,
                    turn_started_at: None,
                    detail: combine_status_detail(turn.status_detail.as_deref(), background),
                }
            }
            None => TurnStatusSnapshot {
                label: TurnStatusLabel::Idle,
                turn_started_at: None,
                detail: background,
            },
            Some(_) => TurnStatusSnapshot {
                label: TurnStatusLabel::Idle,
                turn_started_at: None,
                detail: background,
            },
        })
    };

    set_turn_status(&app.chrome_state, snapshot.clone());
}

fn turn_status_label_for_state(state: CliRunState) -> TurnStatusLabel {
    match state {
        CliRunState::Idle => TurnStatusLabel::Idle,
        CliRunState::Working => TurnStatusLabel::Working,
        CliRunState::Thinking => TurnStatusLabel::Thinking,
        CliRunState::Responding => TurnStatusLabel::Responding,
        CliRunState::RunningTool => TurnStatusLabel::RunningTool,
        CliRunState::Waiting => TurnStatusLabel::Waiting,
        CliRunState::Error => TurnStatusLabel::Error,
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

fn process_summary(app: &App) -> Option<String> {
    let running = app
        .processes
        .iter()
        .filter(|task| !task.status.is_terminal())
        .count();
    match running {
        0 => None,
        running => Some(format!(
            "{} process{}",
            running,
            if running == 1 { "" } else { "es" }
        )),
    }
}

fn current_context_budget_tokens(app: &App) -> Option<i64> {
    if !app.turn_active() {
        return None;
    }
    let input = app.usage.last_response_usage.input_tokens.max(0);
    let cached = app.usage.last_response_usage.cached_input_tokens.max(0);
    // Suppress until input accounting from a completed response has landed.
    // Showing only the streaming output token estimate against the full
    // context window otherwise reads as `36 · 0%` on the first turn.
    if input == 0 && cached == 0 {
        return None;
    }
    let output = (app.usage.last_response_usage.output_tokens
        + app.usage.live_output_tokens_estimate)
        .max(0);
    Some(if app.usage.context_usage_excludes_cached_input {
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
    let content_width = app
        .suggestions()
        .iter()
        .take(max_visible)
        .map(|s| {
            3 + name_col + usize::from(!s.description.is_empty()) + display_width(&s.description)
        })
        .max()
        .unwrap_or(20)
        .max(20);
    let width = (content_width as u16 + 2).min(input_area.width).min(72);
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
    spans.push(Span::styled(
        if selected { "▶ " } else { "  " },
        if selected {
            Style::default()
                .fg(theme::brand())
                .bg(theme::surface_raised())
                .add_modifier(Modifier::Bold)
        } else {
            base_style
        },
    ));

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
    if width < 4 || history_area.height < 5 {
        return;
    }
    let filtered_indices = picker.filtered_indices();
    let match_count = filtered_indices.len();
    let max_list_height = history_area.height.saturating_sub(4).max(1) as usize;
    let list_height = match_count.clamp(1, 15).min(max_list_height) as u16;
    let height = list_height + 4; // title + search + list + footer
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
        &format!("Resume Session ({match_count}/{})", picker.items.len()),
        fg(theme::brand()).add_modifier(Modifier::Bold),
        popup.width.saturating_sub(4),
    );
    let query_text = if picker.query.is_empty() {
        "Search: type to filter".to_string()
    } else {
        format!("Search: {}█", picker.query)
    };
    let query_style = if picker.query.is_empty() {
        theme::text_faint_style()
    } else {
        fg(theme::text_primary())
    };
    frame.write_text(
        popup.x + 2,
        popup.y + 1,
        &query_text,
        query_style,
        popup.width.saturating_sub(4),
    );

    if picker.items.is_empty() {
        frame.write_text(
            popup.x + 2,
            popup.y + 2,
            "No sessions yet",
            theme::text_faint_style(),
            popup.width.saturating_sub(4),
        );
    } else if match_count == 0 {
        frame.write_text(
            popup.x + 2,
            popup.y + 2,
            "No matching sessions",
            theme::text_faint_style(),
            popup.width.saturating_sub(4),
        );
    } else {
        let selected = picker.selected.min(match_count - 1);
        let scroll = selected.saturating_sub(list_height as usize - 1);
        let visible_items = filtered_indices
            .iter()
            .skip(scroll)
            .take(list_height as usize)
            .map(|idx| &picker.items[*idx])
            .collect::<Vec<_>>();
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
            let row_selected = scroll + row == selected;
            let prefix = if row_selected { "▶ " } else { "  " };
            let preview = if session.message_count == 0 {
                "No messages yet".to_string()
            } else {
                session.first_message.replace('\n', " ")
            };
            let cwd = session.cwd_label().unwrap_or_default();
            let reserved = 2 + time_col + 1 + count_col + 1 + cwd.len() + 1;
            let preview_width = popup
                .width
                .saturating_sub(2)
                .saturating_sub(reserved as u16) as usize;
            let preview = truncate_display_forced(&preview, preview_width.max(8));
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
            let style = if row_selected {
                fg(theme::text_primary())
                    .bg(theme::surface_raised())
                    .add_modifier(Modifier::Bold)
            } else if session.message_count == 0 && picker.showing_empty_sessions {
                theme::text_faint_style()
            } else {
                fg(theme::text_subtle())
            };
            frame.write_text(
                popup.x + 1,
                popup.y + 2 + row as u16,
                &line,
                style,
                popup.width.saturating_sub(2),
            );
        }
    }

    // Dismissal hint in the bottom border row. The overlay is a centered
    // box drawn on top of history with no scrim; the user needs at least
    // one explicit signal that it's modal and how to close it.
    let hint = "type search · ↑↓ choose · enter open · esc close";
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

fn truncate_display_forced(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 1 {
        return "…".to_string();
    }
    let target = max_width - 1;
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
                lash_core::MessageRole::User => "user",
                lash_core::MessageRole::Assistant => "assistant",
                lash_core::MessageRole::System => "system",
                lash_core::MessageRole::Event => "event",
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

fn draw_process_overview(frame: &mut Frame<'_>, app: &App, body_area: Rect) {
    let Some(overview) = app.process_overview_state() else {
        return;
    };
    let width = 72u16.min(body_area.width.saturating_sub(4));
    let row_height = overview.rows.len().max(1) as u16;
    let height = row_height + 3;
    if width < 24 || body_area.height < height {
        return;
    }

    draw_overlay_scrim(frame, body_area);
    let popup = centered_rect(body_area, width, height);
    frame.draw_box(
        popup,
        fg(theme::border_faint()),
        Some(bg(theme::surface_deep())),
    );
    frame.write_text(
        popup.x + 2,
        popup.y,
        &overview.title,
        fg(theme::brand()).add_modifier(Modifier::Bold),
        popup.width.saturating_sub(4),
    );

    let label_width = overview
        .rows
        .iter()
        .map(|(label, _)| display_width(label))
        .max()
        .unwrap_or(0)
        .min(18) as u16;
    for (row, (label, value)) in overview.rows.iter().enumerate() {
        let y = popup.y + 1 + row as u16;
        frame.write_text(
            popup.x + 2,
            y,
            label,
            theme::text_faint_style(),
            label_width,
        );
        frame.write_text(
            popup.x + 2 + label_width + 2,
            y,
            value,
            fg(theme::text_primary()),
            popup.width.saturating_sub(label_width + 6),
        );
    }

    let hint = "esc close · delete cancel";
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

fn draw_document_overlay(frame: &mut Frame<'_>, app: &App, body_area: Rect) {
    let Some(document) = app.document_state() else {
        return;
    };
    let width = 92u16.min(body_area.width.saturating_sub(4));
    let height = body_area.height.saturating_sub(2).min(24);
    if width < 32 || height < 8 {
        return;
    }

    draw_overlay_scrim(frame, body_area);
    let popup = centered_rect(body_area, width, height);
    frame.draw_box(
        popup,
        fg(theme::border_faint()),
        Some(bg(theme::surface_deep())),
    );
    frame.write_text(
        popup.x + 2,
        popup.y,
        &document.title,
        fg(theme::brand()).add_modifier(Modifier::Bold),
        popup.width.saturating_sub(4),
    );

    let content_area = Rect::new(
        popup.x + 2,
        popup.y + 1,
        popup.width.saturating_sub(4),
        popup.height.saturating_sub(3),
    );
    let lines = render::document_lines_snapshot(document, content_area.width as usize);
    let max_scroll = lines.len().saturating_sub(content_area.height as usize);
    let scroll = document.scroll_offset.min(max_scroll);
    for (row, line) in lines
        .iter()
        .skip(scroll)
        .take(content_area.height as usize)
        .enumerate()
    {
        frame.write_line(
            content_area.x,
            content_area.y + row as u16,
            line,
            content_area.width,
        );
    }

    if max_scroll > 0 {
        let track_height = content_area.height as usize;
        let thumb_height =
            ((track_height * track_height).div_ceil(lines.len())).clamp(1, track_height) as u16;
        let travel = content_area.height.saturating_sub(thumb_height);
        let y = content_area.y
            + if travel == 0 {
                0
            } else {
                scroll
                    .saturating_mul(travel as usize)
                    .checked_div(max_scroll)
                    .unwrap_or(0) as u16
            };
        for offset in 0..thumb_height {
            frame.write_text(
                popup.x + popup.width.saturating_sub(2),
                y + offset,
                "│",
                fg(theme::text_subtle()),
                1,
            );
        }
    }

    let hint = document
        .footer_hints
        .iter()
        .map(|hint| format!("{} {}", hint.key, hint.description))
        .collect::<Vec<_>>()
        .join(" · ");
    let hint_width = display_width(&hint) as u16;
    if popup.width > hint_width + 4 {
        frame.write_text(
            popup.x + popup.width - hint_width - 2,
            popup.y + popup.height - 1,
            &hint,
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

fn fg(color: lash_tui::Color) -> Style {
    Style::default().fg(color)
}

fn bg(color: lash_tui::Color) -> Style {
    Style::default().bg(color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_model::PromptRequest;
    use lash_core::PromptUsage;
    use lash_tui_extensions::{
        TuiExtension, TuiExtensions, TuiHostEffect, TuiRenderContext, TuiSurfaceSize,
        TuiSurfaceSlot, TuiSurfaceSpec,
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
    fn turn_status_label_uses_only_public_run_states() {
        assert_eq!(
            turn_status_label_for_state(CliRunState::Working),
            TurnStatusLabel::Working
        );
        assert_eq!(
            turn_status_label_for_state(CliRunState::RunningTool),
            TurnStatusLabel::RunningTool
        );
        assert_eq!(
            turn_status_label_for_state(CliRunState::Thinking),
            TurnStatusLabel::Thinking
        );
        assert_eq!(
            turn_status_label_for_state(CliRunState::Responding),
            TurnStatusLabel::Responding
        );
    }

    #[test]
    fn status_bar_shows_context_window_usage() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.model_variant = Some("high".into());
        app.usage.context_window = Some(1_100_000);
        app.usage.last_prompt_usage = Some(PromptUsage {
            prompt_context_tokens: 0,
            input_tokens: 0,
            cached_input_tokens: 0,
            context_budget_tokens: 7_000,
        });

        let snapshot = lash_tui::render_snapshot(80, 4, |frame| draw(frame, &mut app));
        let top = snapshot.visible_line_trimmed(0);
        assert!(top.contains("lash · gpt-5.4 · standard · high"));
        // Context meter: tokens used + integer percent, separated by a
        // middle dot. The old representation was `7.0k / 1.1M (0.6%)`,
        // which gave three renderings of the same number.
        assert!(top.contains("ctx 7.0k · 1%"));
    }

    #[test]
    fn status_bar_hides_meter_during_first_turn_before_input_accounting_lands() {
        // Regression: while the very first response is streaming, only
        // `live_output_tokens_estimate` is nonzero; using it as the displayed
        // total reads as if those streamed bytes were the entire context, so
        // the bar shows e.g. `36 · 0%` against a 1.1M-token window. The
        // meter should stay off until real input accounting lands.
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.usage.context_window = Some(1_100_000);
        app.start_turn();
        app.usage.live_output_tokens_estimate = 36;
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
    fn stale_working_status_does_not_make_idle_cli_working() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let (chrome_ext, chrome_state) = crate::chrome_ui::ChromeTuiExtension::new();
        let ui_extensions =
            Arc::new(TuiExtensions::new(vec![chrome_ext]).expect("chrome extension"));
        app.set_ui_extensions(ui_extensions);
        app.set_chrome_state(chrome_state);

        app.handle_turn_activity(lash_core::TurnActivity::independent(
            lash_core::TurnEvent::QueuedWorkStarted {
                boundary: lash_core::runtime::QueuedWorkClaimBoundary::Idle,
                batch_ids: Vec::new(),
                causes: Vec::new(),
            },
        ));
        assert!(!app.turn_active());
        assert_eq!(app.run_state, CliRunState::Idle);

        sync_chrome_turn_status(&app);
        let snapshot = lash_tui::render_snapshot(80, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("Idle"));
        assert!(!visible.contains("Working"));
    }

    #[test]
    fn session_picker_remains_visible_across_repeated_renders() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "current-session-id".into());
        app.show_session_picker(vec![crate::session_log::SessionInfo {
            filename: "previous.db".into(),
            session_id: "previous-session-id".into(),
            message_count: 3,
            first_message: "previous task".into(),
            modified: std::time::SystemTime::now(),
            cwd: Some(std::path::PathBuf::from("/workspace/code/lash")),
        }]);

        let first = lash_tui::render_snapshot(96, 24, |frame| draw(frame, &mut app));
        assert!(app.has_session_picker());
        let first_lines = first.visible_lines_trimmed();
        assert!(
            first_lines
                .iter()
                .any(|line| line.contains("Resume Session (1/1)"))
        );
        assert!(
            first_lines
                .iter()
                .any(|line| line.contains("previous task"))
        );

        app.dirty = false;
        app.on_tick();
        let second = lash_tui::render_snapshot(96, 24, |frame| draw(frame, &mut app));
        assert!(app.has_session_picker());
        let second_lines = second.visible_lines_trimmed();
        assert!(
            second_lines
                .iter()
                .any(|line| line.contains("Resume Session (1/1)"))
        );
        assert!(
            second_lines
                .iter()
                .any(|line| line.contains("previous task"))
        );

        let compact = lash_tui::render_snapshot(96, 10, |frame| draw(frame, &mut app));
        let compact_lines = compact.visible_lines_trimmed();
        assert!(
            compact_lines
                .iter()
                .any(|line| line.contains("Resume Session (1/1)")),
            "session picker should shrink instead of disappearing in compact layouts"
        );
    }

    #[test]
    fn session_picker_filters_sessions_by_typed_query() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "current-session-id".into());
        app.show_session_picker(vec![
            crate::session_log::SessionInfo {
                filename: "alpha.db".into(),
                session_id: "alpha-session".into(),
                message_count: 2,
                first_message: "debug the resume picker".into(),
                modified: std::time::SystemTime::now(),
                cwd: Some(std::path::PathBuf::from("/workspace/code/lash")),
            },
            crate::session_log::SessionInfo {
                filename: "beta.db".into(),
                session_id: "beta-session".into(),
                message_count: 4,
                first_message: "write release notes".into(),
                modified: std::time::SystemTime::now(),
                cwd: Some(std::path::PathBuf::from("/tmp/elsewhere")),
            },
        ]);
        for ch in "release".chars() {
            app.session_picker_insert_query_char(ch);
        }

        let snapshot = lash_tui::render_snapshot(96, 24, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("Resume Session (1/2)"));
        assert!(visible.contains("write release notes"));
        assert!(!visible.contains("debug the resume picker"));
    }

    #[test]
    fn idle_footer_stays_idle_when_background_processes_exist() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        let (chrome_ext, chrome_state) = crate::chrome_ui::ChromeTuiExtension::new();
        let ui_extensions =
            Arc::new(TuiExtensions::new(vec![chrome_ext]).expect("chrome extension"));
        app.set_ui_extensions(ui_extensions);
        app.set_chrome_state(chrome_state);
        app.update_processes(vec![lash_core::ProcessHandleSummary::new(
            "process-1",
            lash_core::ProcessHandleDescriptor::new(Some("lashlang"), Some("responder")),
            lash_core::ProcessLifecycleStatus::Running,
        )]);
        sync_chrome_turn_status(&app);

        let snapshot = lash_tui::render_snapshot(80, 10, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("Idle"));
        assert!(!visible.contains("Working"));
        assert!(visible.contains("1 process"));
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
    fn process_dock_renders_below_input_and_overview_as_overlay() {
        let mut app = App::new("gpt-5.4".into(), "test".into(), "test-session-id".into());
        app.update_processes(vec![
            lash_core::ProcessHandleSummary::new(
                "process-1",
                lash_core::ProcessHandleDescriptor::new(Some("lashlang"), Some("responder")),
                lash_core::ProcessLifecycleStatus::Running,
            )
            .with_definition(Some(lash_core::ProcessDefinitionSummary {
                name: "responder".into(),
            })),
        ]);
        app.select_next_process();

        let areas = render::chrome_areas(&app, 80, 16);
        let snapshot = lash_tui::render_snapshot(80, 16, |frame| draw(frame, &mut app));
        assert!(areas.process.y > areas.input.y);
        assert!(
            snapshot
                .visible_line_trimmed(areas.process.y)
                .contains("Background")
        );

        let overview = app
            .selected_process_overview_state()
            .expect("process overview");
        app.show_process_overview(overview);
        let snapshot = lash_tui::render_snapshot(80, 16, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");
        assert!(visible.contains("Process responder"));
        assert!(visible.contains("definition"));
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

    struct SurfaceTestTuiExtension;

    #[async_trait::async_trait]
    impl TuiExtension for SurfaceTestTuiExtension {
        fn id(&self) -> &'static str {
            "surface_test"
        }

        async fn invoke_action(
            &self,
            _action: &str,
            _arg: Option<&str>,
            _ctx: lash_tui_extensions::TuiExtensionContext<'_>,
        ) -> Result<Vec<TuiHostEffect>, String> {
            Ok(Vec::new())
        }

        fn render_surface(
            &self,
            surface_key: &str,
            _ctx: TuiRenderContext<'_>,
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

        fn handle_turn_event(&self, event: &lash_core::TurnEvent) -> Vec<TuiHostEffect> {
            match event {
                lash_core::TurnEvent::AssistantProseDelta { text } if text == "mount" => vec![
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "workspace".to_string(),
                            slot: TuiSurfaceSlot::Workspace,
                            size: TuiSurfaceSize::Auto,
                            order: 0,
                            focusable: true,
                            visible: true,
                            modal: false,
                        },
                    },
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "footer".to_string(),
                            slot: TuiSurfaceSlot::Footer,
                            size: TuiSurfaceSize::Lines(1),
                            order: 0,
                            focusable: false,
                            visible: true,
                            modal: false,
                        },
                    },
                ],
                lash_core::TurnEvent::AssistantProseDelta { text } if text == "overlay" => vec![
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "overlay".to_string(),
                            slot: TuiSurfaceSlot::Overlay,
                            size: TuiSurfaceSize::Fixed {
                                width: 16,
                                height: 3,
                            },
                            order: 10,
                            focusable: true,
                            visible: true,
                            modal: true,
                        },
                    },
                    TuiHostEffect::FocusSurface {
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
            TuiExtensions::new(vec![Arc::new(SurfaceTestTuiExtension)])
                .expect("surface extensions"),
        );
        ui_extensions.effects_for_turn_event(&lash_core::TurnEvent::AssistantProseDelta {
            text: "mount".to_string(),
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
            TuiExtensions::new(vec![Arc::new(SurfaceTestTuiExtension)])
                .expect("surface extensions"),
        );
        ui_extensions.effects_for_turn_event(&lash_core::TurnEvent::AssistantProseDelta {
            text: "mount".to_string(),
        });
        ui_extensions.effects_for_turn_event(&lash_core::TurnEvent::AssistantProseDelta {
            text: "overlay".to_string(),
        });
        app.set_ui_extensions(Arc::clone(&ui_extensions));

        let snapshot = lash_tui::render_snapshot(40, 12, |frame| draw(frame, &mut app));
        let visible = snapshot.visible_lines_trimmed().join("\n");

        assert!(visible.contains("OVERLAY"));
    }
}
