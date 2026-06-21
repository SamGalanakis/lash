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
        frame.patch_row_style_range(x, y, width, |style| style.bg(theme::selection_bg()));
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
