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

pub(crate) fn document_lines_snapshot(
    document: &DocumentState,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let width = width.max(1);
    for (section_idx, section) in document.sections.iter().enumerate() {
        if section_idx > 0 {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            section.title.clone(),
            Style::default()
                .fg(theme::brand())
                .add_modifier(Modifier::Bold),
        )));
        for row in &section.rows {
            match row {
                DocumentRow::Text(text) => {
                    for (start, end) in wrap_line(text, 0, 0, width) {
                        lines.push(Line::from(Span::styled(
                            text[start..end].to_string(),
                            Style::default().fg(theme::text_muted()),
                        )));
                    }
                }
                DocumentRow::KeyValue { label, value } => {
                    push_document_key_value(&mut lines, label, value, width);
                }
                DocumentRow::Shortcut { keys, description } => {
                    push_document_shortcut(&mut lines, keys, description, width);
                }
            }
        }
    }
    lines
}

pub(crate) fn document_max_scroll(
    document: &DocumentState,
    width: usize,
    visible_height: usize,
) -> usize {
    document_lines_snapshot(document, width)
        .len()
        .saturating_sub(visible_height)
}

fn push_document_key_value(lines: &mut Vec<Line<'static>>, label: &str, value: &str, width: usize) {
    let label_width = 16usize.min(width.saturating_sub(1));
    let visible_label = truncate_to_display_width(label, label_width);
    let segments = wrap_line(value, label_width + 1, label_width + 1, width);
    for (idx, (start, end)) in segments.into_iter().enumerate() {
        let label = if idx == 0 {
            format!("{visible_label:<label_width$}")
        } else {
            " ".repeat(label_width)
        };
        lines.push(Line::from(vec![
            Span::styled(label, theme::text_faint_style()),
            Span::styled(" ", theme::text_faint_style()),
            Span::styled(
                value[start..end].to_string(),
                Style::default().fg(theme::text_muted()),
            ),
        ]));
    }
}

fn push_document_shortcut(
    lines: &mut Vec<Line<'static>>,
    keys: &str,
    description: &str,
    width: usize,
) {
    let key_width = 20usize.min(width.saturating_sub(1));
    let visible_keys = truncate_to_display_width(keys, key_width);
    let desc_width = width.saturating_sub(key_width + 1).max(1);
    let segments = wrap_line(description, 0, 0, desc_width);
    for (idx, (start, end)) in segments.into_iter().enumerate() {
        let key = if idx == 0 {
            format!("{visible_keys:<key_width$}")
        } else {
            " ".repeat(key_width)
        };
        lines.push(Line::from(vec![
            Span::styled(key, theme::help_key()),
            Span::styled(" ", theme::text_faint_style()),
            Span::styled(
                description[start..end].to_string(),
                Style::default().fg(theme::text_muted()),
            ),
        ]));
    }
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
