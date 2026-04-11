use super::*;

pub(crate) fn wait_height(app: &App, frame_width: u16, frame_height: u16) -> u16 {
    let Some(wait) = app.wait_state() else {
        return 3;
    };
    let inner_w = prompt_inner_width(frame_width);
    let max_height = frame_height.saturating_sub(1).max(3);
    let content_h = wait_content_lines_for_app(wait, inner_w).len() as u16;
    content_h.max(1).min(max_height)
}

#[cfg(test)]
pub(crate) fn wait_content_lines_snapshot(
    wait: &crate::overlay::WaitState,
    inner_w: usize,
) -> Vec<Line<'static>> {
    wait_content_lines(wait, inner_w, wait.seconds)
}

pub(crate) fn wait_content_lines_for_app(
    wait: &crate::overlay::WaitState,
    inner_w: usize,
) -> Vec<Line<'static>> {
    wait_content_lines(wait, inner_w, wait.remaining_seconds().min(wait.seconds))
}

fn wait_section_label(text: &str, inner_w: usize) -> Line<'static> {
    let label_width = UnicodeWidthStr::width(text);
    let fill_width = inner_w.saturating_sub(label_width + 2);
    let mut spans = vec![Span::styled(
        text.to_string(),
        Style::default()
            .fg(theme::brand())
            .add_modifier(Modifier::Bold),
    )];
    if fill_width > 0 {
        spans.push(Span::styled(
            " ",
            Style::default().fg(theme::border_faint()),
        ));
        spans.push(Span::styled(
            "─".repeat(fill_width),
            Style::default().fg(theme::border_faint()),
        ));
    }
    Line::from(spans)
}

fn push_wrapped_plain_lines(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    total_width: usize,
    style: Style,
) {
    for logical_line in text.split('\n') {
        let segments = wrap_line(logical_line, 0, 0, total_width);
        for &(seg_start, seg_end) in &segments {
            lines.push(Line::from(vec![text_display::sanitize_span(
                logical_line[seg_start..seg_end].to_string(),
                style,
            )]));
        }
    }
}

fn normalize_panel_markdown(title: &str, markdown: &str) -> String {
    let mut lines = markdown.lines();
    let Some(first_line) = lines.next() else {
        return String::new();
    };
    let Some(heading) = first_line.trim().strip_prefix("# ") else {
        return markdown.to_string();
    };
    let normalized_heading = |text: &str| -> String {
        text.chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(|ch| ch.to_lowercase())
            .collect()
    };
    if normalized_heading(heading) != normalized_heading(title) {
        return markdown.to_string();
    }

    let mut remaining = lines.collect::<Vec<_>>();
    if remaining.first().is_some_and(|line| line.trim().is_empty()) {
        remaining.remove(0);
    }
    remaining.join("\n")
}

fn render_wait_panel_lines(title: &str, markdown_text: &str, inner_w: usize) -> Vec<Line<'static>> {
    let mut lines = vec![wait_section_label(title, inner_w)];
    let normalized = normalize_panel_markdown(title, markdown_text);
    if !normalized.trim().is_empty() {
        lines.push(Line::from(""));
        lines.extend(markdown::render_markdown(&normalized, inner_w));
    }
    lines
}

fn wait_help_line() -> Line<'static> {
    let items = [("ctrl+j", "resume now"), ("esc", "skip wait")];
    let mut spans = Vec::new();
    for (idx, (key, desc)) in items.into_iter().enumerate() {
        if idx > 0 {
            spans.push(Span::styled(
                " · ",
                Style::default().fg(theme::border_faint()),
            ));
        }
        spans.push(Span::styled(key.to_string(), theme::help_key()));
        spans.push(Span::styled(format!(" {desc}"), theme::help_desc()));
    }
    Line::from(spans)
}

fn wait_content_lines(
    wait: &crate::overlay::WaitState,
    inner_w: usize,
    remaining_seconds: u64,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    if let Some(panel) = wait.panel.as_ref() {
        lines.extend(render_wait_panel_lines(
            &panel.title,
            &panel.markdown,
            inner_w,
        ));
        lines.push(Line::from(""));
    } else {
        lines.push(wait_section_label("PAUSED", inner_w));
        lines.push(Line::from(""));
    }

    if !wait.question.trim().is_empty() {
        push_wrapped_plain_lines(
            &mut lines,
            &wait.question,
            inner_w,
            Style::default().fg(theme::text_muted()),
        );
        lines.push(Line::from(""));
    }

    let status = if remaining_seconds == 0 {
        "Resuming…".to_string()
    } else if remaining_seconds == 1 {
        "Auto-resume in 1s".to_string()
    } else {
        format!("Auto-resume in {remaining_seconds}s")
    };
    lines.push(Line::from(vec![
        Span::styled(
            "● ".to_string(),
            Style::default()
                .fg(theme::brand())
                .add_modifier(Modifier::Bold),
        ),
        Span::styled(
            status,
            Style::default()
                .fg(theme::text_primary())
                .add_modifier(Modifier::Bold),
        ),
    ]));

    if remaining_seconds > 0 {
        push_wrapped_plain_lines(
            &mut lines,
            "The run continues automatically when the timer ends.",
            inner_w,
            Style::default().fg(theme::text_faint()),
        );
    }

    lines.push(Line::from(""));
    lines.push(wait_help_line());
    lines
}
