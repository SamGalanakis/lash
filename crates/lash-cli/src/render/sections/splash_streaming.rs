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
