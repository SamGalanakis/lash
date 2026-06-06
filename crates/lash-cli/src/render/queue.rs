use super::*;

pub(crate) fn queue_preview_lines_snapshot(app: &App, width: u16) -> Vec<Line<'static>> {
    queue_preview_lines(app, width)
}

fn queue_preview_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    if !app.has_queued_messages() || width < 12 {
        return Vec::new();
    }

    let mut after_current_step_previews = Vec::new();
    let mut next_turn_previews = Vec::new();
    for batch in app
        .queued_work_snapshot()
        .iter()
        .filter(|batch| !app.queued_batch_preview_suppressed(batch))
    {
        let target = if app.turn_active()
            && batch.delivery_policy == lash_core::DeliveryPolicy::EarliestSafeBoundary
        {
            &mut after_current_step_previews
        } else {
            &mut next_turn_previews
        };
        if let Some(turn) = app.prepared_turn_for_queued_batch(batch) {
            target.push(turn.preview());
        }
    }

    let mut lines = Vec::new();
    let inner_width = width as usize;
    if !after_current_step_previews.is_empty() {
        push_queue_section(
            &mut lines,
            inner_width,
            "◆ Will send in this turn",
            &after_current_step_previews,
            &app.skills,
            Style::default()
                .fg(theme::brand())
                .add_modifier(Modifier::Bold),
            Style::default().fg(theme::text_muted()),
        );
    }

    if !next_turn_previews.is_empty() {
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }
        push_queue_section(
            &mut lines,
            inner_width,
            "◇ Queued for next turn",
            &next_turn_previews,
            &app.skills,
            Style::default()
                .fg(theme::state_ok())
                .add_modifier(Modifier::Bold),
            Style::default().fg(theme::text_subtle()),
        );
    }

    lines
}

fn push_queue_section(
    lines: &mut Vec<Line<'static>>,
    width: usize,
    title: &str,
    items: &[String],
    skills: &crate::SkillCatalog,
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
            Span::styled("    +", Style::default().fg(theme::border_faint())),
            Span::styled(
                format!("{}", items.len() - QUEUE_SECTION_ITEM_LIMIT),
                item_style,
            ),
            Span::styled(" more", Style::default().fg(theme::text_faint())),
        ]));
    }
}

#[allow(clippy::too_many_arguments)]
fn push_wrapped_queue_item(
    lines: &mut Vec<Line<'static>>,
    width: usize,
    text: &str,
    skills: &crate::SkillCatalog,
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
            Style::default().fg(theme::border_faint()),
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
            Style::default().fg(theme::text_faint()),
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
