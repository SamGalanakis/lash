fn build_input_badge(app: &App) -> Option<Line<'static>> {
    let badge_labels: Vec<&str> = app
        .plugin_mode_indicators
        .values()
        .map(|label| label.as_str())
        .collect();
    let mut spans = Vec::new();
    if !badge_labels.is_empty() {
        for (idx, label) in badge_labels.iter().enumerate() {
            if idx > 0 {
                spans.push(Span::styled(
                    " · ",
                    Style::default().fg(theme::border_faint()),
                ));
            }
            spans.push(Span::styled(
                (*label).to_string(),
                Style::default()
                    .fg(theme::brand())
                    .add_modifier(Modifier::Bold),
            ));
        }
        spans.push(Span::styled(
            " · ",
            Style::default().fg(theme::border_faint()),
        ));
    }

    let location_label = app
        .repo_status
        .as_ref()
        .map(|repo| format!("{} · {}", repo.repo_name, repo.display_ref()))
        .unwrap_or_else(|| app.cwd.clone());
    spans.push(text_display::sanitize_span(
        location_label,
        Style::default().fg(theme::text_faint()),
    ));
    Some(Line::from(spans))
}

fn styled_input_segment(
    logical_line: &str,
    seg_start: usize,
    seg_end: usize,
    image_markers: &[(std::ops::Range<usize>, usize)],
    skills: &SkillCatalog,
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut cursor = seg_start;

    for (range, _) in image_markers {
        if range.end <= seg_start || range.start >= seg_end {
            continue;
        }
        let clamped_start = range.start.max(seg_start);
        let clamped_end = range.end.min(seg_end);
        if cursor < clamped_start {
            spans.extend(styled_text_with_slash_command(
                logical_line,
                cursor,
                clamped_start,
                skills,
                theme::user_input(),
                theme::slash_command_slash(),
            ));
        }
        spans.push(Span::styled(
            logical_line[clamped_start..clamped_end].to_string(),
            theme::image_marker(),
        ));
        cursor = clamped_end;
    }

    if cursor < seg_end {
        spans.extend(styled_text_with_slash_command(
            logical_line,
            cursor,
            seg_end,
            skills,
            theme::user_input(),
            theme::slash_command_slash(),
        ));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), theme::user_input()));
    }
    spans
}

fn styled_input_segment_with_hint(
    logical_line: &str,
    hint: &str,
    seg_start: usize,
    seg_end: usize,
    image_markers: &[(std::ops::Range<usize>, usize)],
    skills: &SkillCatalog,
) -> Vec<Span<'static>> {
    let input_end = logical_line.len();
    let mut spans = Vec::new();

    if seg_start < input_end {
        let actual_end = seg_end.min(input_end);
        spans.extend(styled_input_segment(
            logical_line,
            seg_start,
            actual_end,
            image_markers,
            skills,
        ));
    }

    if seg_end > input_end {
        let hint_start = seg_start.saturating_sub(input_end);
        let hint_end = (seg_end - input_end).min(hint.len());
        if hint_start < hint_end {
            spans.push(Span::styled(
                hint[hint_start..hint_end].to_string(),
                theme::text_faint_style(),
            ));
        }
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), theme::user_input()));
    }
    spans
}

fn styled_user_input_segment(
    text: &str,
    seg_start: usize,
    seg_end: usize,
    skills: &SkillCatalog,
) -> Vec<Span<'static>> {
    styled_text_with_slash_command(
        text,
        seg_start,
        seg_end,
        skills,
        theme::user_input(),
        theme::slash_command_slash(),
    )
}

fn styled_text_with_slash_command(
    text: &str,
    seg_start: usize,
    seg_end: usize,
    skills: &SkillCatalog,
    base_style: Style,
    slash_style: Style,
) -> Vec<Span<'static>> {
    if seg_start >= seg_end {
        return vec![text_display::sanitize_span(String::new(), base_style)];
    }
    let mut spans = Vec::new();
    let mut cursor = seg_start;
    for (slash_start, slash_end) in
        slash_command_ranges_in_segment(text, seg_start, seg_end, skills)
    {
        if slash_start > cursor {
            spans.push(text_display::sanitize_span(
                text[cursor..slash_start].to_string(),
                base_style,
            ));
        }
        spans.push(text_display::sanitize_span(
            text[slash_start..slash_end].to_string(),
            slash_style,
        ));
        cursor = slash_end;
    }
    if cursor < seg_end || spans.is_empty() {
        spans.push(text_display::sanitize_span(
            text[cursor..seg_end].to_string(),
            base_style,
        ));
    }
    if spans.is_empty() {
        spans.push(text_display::sanitize_span(String::new(), base_style));
    }
    spans
}

fn slash_command_ranges_in_segment(
    text: &str,
    seg_start: usize,
    seg_end: usize,
    skills: &SkillCatalog,
) -> Vec<(usize, usize)> {
    slash_command_slash_ranges(text, skills)
        .into_iter()
        .filter_map(|(slash_start, slash_end)| {
            if slash_end <= seg_start || slash_start >= seg_end {
                return None;
            }
            let clamped_start = slash_start.max(seg_start);
            let clamped_end = slash_end.min(seg_end);
            (clamped_start < clamped_end).then_some((clamped_start, clamped_end))
        })
        .collect()
}

fn slash_command_slash_ranges(text: &str, skills: &SkillCatalog) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let trimmed = text.trim_start();
    if trimmed.starts_with('/') {
        let slash_start = text.len() - trimmed.len();
        // Rendering only inspects the slash-character glyph — plugin commands are
        // treated the same way as builtins regardless, so passing an empty catalog
        // is sufficient here and avoids threading the app-level catalog through
        // the rendering layer.
        if crate::command::parse(trimmed, skills).is_some()
            || crate::command::slash_skill_prompt(trimmed, skills).is_some()
        {
            ranges.push((slash_start, slash_start + 1));
        }
    }

    for (range, name) in collect_skill_mentions_with_ranges(text) {
        if skills.get(&name).is_none() {
            continue;
        }
        let slash_range = (range.start, range.start + 1);
        if !ranges.contains(&slash_range) {
            ranges.push(slash_range);
        }
    }

    ranges.sort_unstable_by_key(|(start, _)| *start);
    ranges
}

pub(crate) fn render_block_lines(
    app: &App,
    idx: usize,
    viewport_width: usize,
    viewport_height: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    render_block_into(app, idx, &mut lines, viewport_width, viewport_height);
    lines
}
