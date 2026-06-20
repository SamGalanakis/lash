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
        let selected = app.selected_process_id.as_deref() == Some(task.process_id.as_str());
        let state = match task.status.terminal_state() {
            None => "running",
            Some(lash_core::ProcessTerminalState::Completed) => "success",
            Some(lash_core::ProcessTerminalState::Failed) => "error",
            Some(lash_core::ProcessTerminalState::Cancelled) => "cancelled",
        };
        let producer = task.definition.as_deref().unwrap_or(task.kind.as_str());
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
        let selected_chrome = theme::process_selected_chrome();
        let row_style = |style: Style| {
            if selected {
                style.merge(selected_chrome)
            } else {
                style
            }
        };
        let glyph = if selected { "  ▶ " } else { "  ◆ " };
        let label_style = if selected {
            theme::process_selected_label()
        } else {
            Style::default().fg(theme::text_muted())
        };
        let mut spans = Vec::new();
        spans.push(Span::styled(
            glyph,
            if selected {
                theme::process_selected_indicator()
            } else {
                theme::turn_status_slash()
            },
        ));
        if selected {
            spans.push(Span::styled("SELECTED ", theme::process_selected_badge()));
        }
        spans.extend([
            Span::styled(state.to_string(), row_style(state_style)),
            Span::styled(" · ", row_style(theme::text_faint_style())),
            Span::styled(producer.to_string(), row_style(theme::text_subtle_style())),
            Span::styled(" · ", row_style(theme::text_faint_style())),
            Span::styled(task.label.clone(), label_style),
            Span::styled(" · ", row_style(theme::text_faint_style())),
            Span::styled(elapsed, row_style(theme::text_faint_style())),
        ]);
        lines.push(Line::from(spans));
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
        badge: None,
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
