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
    if area.width < 54 || area.height < 12 || !theme::empty_state_logo_enabled() {
        draw_empty_history_hint(frame, area);
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

fn draw_empty_history_hint(frame: &mut Frame<'_>, area: Rect) {
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
                style.bg(theme::selection_bg())
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
