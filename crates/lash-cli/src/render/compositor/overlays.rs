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

enum CommandPaletteRenderRow {
    Category(String),
    Item(usize),
}

fn command_palette_render_rows(
    palette: &crate::overlay::CommandPaletteState,
    filtered_indices: &[usize],
) -> Vec<CommandPaletteRenderRow> {
    let mut rows = Vec::new();
    let mut last_category: Option<&str> = None;
    for idx in filtered_indices {
        let item = &palette.items[*idx];
        if last_category != Some(item.category.as_str()) {
            rows.push(CommandPaletteRenderRow::Category(item.category.clone()));
            last_category = Some(item.category.as_str());
        }
        rows.push(CommandPaletteRenderRow::Item(*idx));
    }
    rows
}

fn draw_command_palette(frame: &mut Frame<'_>, app: &App, history_area: Rect) {
    let Some(palette) = app.command_palette_state() else {
        return;
    };
    let width = 92u16.min(history_area.width.saturating_sub(4));
    if width < 36 || history_area.height < 8 {
        return;
    }

    let filtered_indices = palette.filtered_indices();
    let match_count = filtered_indices.len();
    let max_list_height = history_area.height.saturating_sub(5).max(1) as usize;
    let render_rows = command_palette_render_rows(palette, &filtered_indices);
    let list_height = render_rows
        .len()
        .clamp(1, 18)
        .min(max_list_height) as u16;
    let height = list_height + 4;

    draw_overlay_scrim(frame, history_area);
    let popup = centered_rect(history_area, width, height);
    frame.draw_box(
        popup,
        fg(theme::border_faint()),
        Some(theme::surface_deep().fill()),
    );

    let title = match palette.selected_position() {
        Some((pos, total)) => format!("{} ({pos}/{total})", palette.title),
        None => format!("{} (0/{})", palette.title, palette.items.len()),
    };
    frame.write_text(
        popup.x + 2,
        popup.y,
        &title,
        fg(theme::brand()).add_modifier(Modifier::Bold),
        popup.width.saturating_sub(10),
    );
    frame.write_text(
        popup.x + popup.width.saturating_sub(7),
        popup.y,
        "esc",
        theme::text_faint_style(),
        3,
    );

    let query_text = if palette.query.is_empty() {
        "Search".to_string()
    } else {
        format!("Search  {}█", palette.query)
    };
    let query_style = if palette.query.is_empty() {
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

    if match_count == 0 {
        frame.write_text(
            popup.x + 2,
            popup.y + 2,
            "No matching commands",
            theme::text_faint_style(),
            popup.width.saturating_sub(4),
        );
    } else {
        let selected_filtered = palette.selected.min(match_count - 1);
        let selected_item_idx = filtered_indices[selected_filtered];
        let selected_row = render_rows
            .iter()
            .position(|row| matches!(row, CommandPaletteRenderRow::Item(idx) if *idx == selected_item_idx))
            .unwrap_or(0);
        let scroll = selected_row.saturating_sub(list_height as usize - 1);
        for (visible_row, row) in render_rows
            .iter()
            .skip(scroll)
            .take(list_height as usize)
            .enumerate()
        {
            let y = popup.y + 2 + visible_row as u16;
            match row {
                CommandPaletteRenderRow::Category(category) => {
                    frame.write_text(
                        popup.x + 2,
                        y,
                        category,
                        fg(theme::brand()).add_modifier(Modifier::Bold),
                        popup.width.saturating_sub(4),
                    );
                }
                CommandPaletteRenderRow::Item(idx) => {
                    let item = &palette.items[*idx];
                    let selected = *idx == selected_item_idx;
                    let row_style = if selected {
                        fg(theme::text_primary())
                            .bg(theme::selection_bg())
                            .add_modifier(Modifier::Bold)
                    } else {
                        theme::surface_deep().apply(fg(theme::text_subtle()))
                    };
                    frame.fill(
                        Rect::new(
                            popup.x + 1,
                            y,
                            popup.width.saturating_sub(2),
                            1,
                        ),
                        ' ',
                        row_style,
                    );
                    let marker = if item.current { "●" } else { " " };
                    let footer = item.footer.as_deref().unwrap_or_default();
                    let footer_width = display_width(footer);
                    let inner_width = popup.width.saturating_sub(4) as usize;
                    let left_width = inner_width.saturating_sub(footer_width.saturating_add(2));
                    let left = format!("{marker} {}", item.title);
                    let line = if left_width > display_width(&left) + 3 {
                        let desc_width = left_width.saturating_sub(display_width(&left) + 2);
                        format!(
                            "{}  {}",
                            left,
                            truncate_display_forced(&item.description, desc_width)
                        )
                    } else {
                        truncate_display_forced(&left, left_width)
                    };
                    frame.write_text(
                        popup.x + 2,
                        y,
                        &line,
                        row_style,
                        left_width as u16,
                    );
                    if !footer.is_empty() && inner_width > footer_width {
                        let footer_style = if selected {
                            row_style
                        } else {
                            theme::surface_deep().apply(theme::text_faint_style())
                        };
                        frame.write_text(
                            popup.x + popup.width.saturating_sub(2 + footer_width as u16),
                            y,
                            footer,
                            footer_style,
                            footer_width as u16,
                        );
                    }
                }
            }
        }
    }

    let hint = "type search · ↑↓ choose · enter run · esc close";
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
        Some(theme::surface_deep().fill()),
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
                theme::selected_row()
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
        Some(theme::surface_deep().fill()),
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
                theme::selected_row()
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
        Some(theme::surface_deep().fill()),
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
                theme::selected_row()
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
        Some(theme::surface_deep().fill()),
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
        Some(theme::surface_deep().fill()),
    );
    frame.write_text(
        popup.x + 2,
        popup.y,
        &document.title,
        fg(theme::brand()).add_modifier(Modifier::Bold),
        popup.width.saturating_sub(4),
    );

    let Some(content_area) = render::document_overlay_content_area(body_area) else {
        return;
    };
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

    apply_document_selection_highlight(frame, app, content_area, scroll);

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

fn apply_document_selection_highlight(
    frame: &mut Frame<'_>,
    app: &App,
    content_area: Rect,
    scroll: usize,
) {
    if !(app.selection.active || app.selection.visible) || content_area.height == 0 {
        return;
    }

    let ((start_col, start_row), (end_col, end_row)) = selection_ordered(&app.selection);
    let view_top = scroll;
    let view_bottom = scroll + content_area.height as usize;
    let visible_start = start_row.max(view_top);
    let visible_end = end_row.min(view_bottom.saturating_sub(1));
    if visible_start > visible_end {
        return;
    }

    for row in visible_start..=visible_end {
        let screen_y = content_area.y + (row - scroll) as u16;
        let col_start = if row == start_row {
            start_col
        } else {
            content_area.x
        };
        let col_end = if row == end_row {
            end_col
        } else {
            content_area.x + content_area.width
        };
        let span_width = col_end.saturating_sub(col_start);
        if span_width > 0 {
            frame.patch_row_style_range(col_start, screen_y, span_width, |style| {
                style.bg(theme::selection_bg())
            });
        }
    }
}
