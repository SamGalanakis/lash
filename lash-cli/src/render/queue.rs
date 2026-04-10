use super::*;

pub(crate) fn queue_preview_lines_snapshot(app: &App, width: u16) -> Vec<Line<'static>> {
    queue_preview_lines(app, width)
}

fn queue_preview_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    if !app.has_queued_messages() || width < 12 {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let inner_width = width as usize;
    let pending_previews: Vec<String> = app
        .pending_steers
        .iter()
        .map(PreparedTurn::preview)
        .collect();
    let queued_previews: Vec<String> = app.queued_turns.iter().map(PreparedTurn::preview).collect();

    if !pending_previews.is_empty() {
        push_queue_section(
            &mut lines,
            inner_width,
            "◆ after next tool/result",
            &pending_previews,
            &app.skills,
            Style::default().fg(theme::SODIUM),
            Style::default().fg(theme::CHALK_MID),
        );
    }

    if !queued_previews.is_empty() {
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }
        push_queue_section(
            &mut lines,
            inner_width,
            "◇ next full turn",
            &queued_previews,
            &app.skills,
            Style::default().fg(theme::LICHEN),
            Style::default().fg(theme::CHALK_DIM),
        );
    }

    lines
}

fn push_queue_section(
    lines: &mut Vec<Line<'static>>,
    width: usize,
    title: &str,
    items: &[String],
    skills: &lash::SkillCatalog,
    header_style: Style,
    item_style: Style,
) {
    let header = format!(
        "{title}{}",
        if items.len() > 1 {
            format!(" · {}", items.len())
        } else {
            String::new()
        }
    );
    for (start, end) in wrap_line(&header, 0, 0, width.max(1)) {
        lines.push(Line::from(Span::styled(
            header[start..end].to_string(),
            header_style,
        )));
    }
    for item in items.iter().take(QUEUE_SECTION_ITEM_LIMIT) {
        push_wrapped_queue_item(
            lines,
            width,
            item,
            skills,
            item_style,
            "  ↳ ",
            "    ",
            QUEUE_SECTION_WRAP_LIMIT,
        );
    }
    if items.len() > QUEUE_SECTION_ITEM_LIMIT {
        lines.push(Line::from(vec![
            Span::styled("    +", Style::default().fg(theme::ASH)),
            Span::styled(
                format!("{}", items.len() - QUEUE_SECTION_ITEM_LIMIT),
                item_style,
            ),
            Span::styled(" more", Style::default().fg(theme::ASH_TEXT)),
        ]));
    }
}

fn push_wrapped_queue_item(
    lines: &mut Vec<Line<'static>>,
    width: usize,
    text: &str,
    skills: &lash::SkillCatalog,
    style: Style,
    first_prefix: &str,
    continuation_prefix: &str,
    max_lines: usize,
) {
    let collapsed = text.replace('\n', " ");
    let segments = wrap_line(
        &collapsed,
        UnicodeWidthStr::width(first_prefix),
        UnicodeWidthStr::width(continuation_prefix),
        width.max(1),
    );
    for (idx, (start, end)) in segments.into_iter().take(max_lines).enumerate() {
        let prefix = if idx == 0 {
            first_prefix
        } else {
            continuation_prefix
        };
        let mut spans = vec![Span::styled(
            prefix.to_string(),
            Style::default().fg(theme::ASH),
        )];
        spans.extend(styled_text_with_slash_command(
            &collapsed,
            start,
            end,
            skills,
            style,
            theme::slash_command_slash(),
        ));
        lines.push(Line::from(spans));
    }
    if segments_len_exceeds(
        &collapsed,
        first_prefix,
        continuation_prefix,
        width.max(1),
        max_lines,
    ) {
        lines.push(Line::from(vec![Span::styled(
            format!("{continuation_prefix}…"),
            Style::default().fg(theme::ASH_TEXT),
        )]));
    }
}

fn segments_len_exceeds(
    text: &str,
    first_prefix: &str,
    continuation_prefix: &str,
    width: usize,
    max_lines: usize,
) -> bool {
    wrap_line(
        text,
        UnicodeWidthStr::width(first_prefix),
        UnicodeWidthStr::width(continuation_prefix),
        width,
    )
    .len()
        > max_lines
}
