#[cfg(test)]
fn render_block(
    blocks: &[UiTimelineItem],
    idx: usize,
    expand_level: u8,
    viewport_width: usize,
    viewport_height: usize,
) -> Vec<Line<'static>> {
    let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
    app.timeline = blocks.to_vec().into();
    app.expand_level = expand_level;
    render_block_lines(&app, idx, viewport_width, viewport_height)
}

fn render_block_into(
    app: &App,
    idx: usize,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    viewport_height: usize,
) {
    let blocks = &app.timeline;
    let expand_level = app.expand_level;
    match &blocks[idx] {
        UiTimelineItem::TurnStart(turn) => {
            if turn.show_separator {
                let rule_width = (viewport_width * 2 / 5).max(8).min(viewport_width);
                let pad_left = (viewport_width.saturating_sub(rule_width)) / 2;
                lines.push(Line::from(""));
                let mut spans = Vec::new();
                if pad_left > 0 {
                    spans.push(Span::raw(" ".repeat(pad_left)));
                }
                spans.push(Span::styled(
                    "─".repeat(rule_width),
                    theme::turn_separator(),
                ));
                lines.push(Line::from(spans));
            }
            // Turns with `show_separator: false` contribute no visible
            // lines — the marker is purely structural, letting later
            // features (turn folding, turn addressing) scan between
            // `TurnStart` markers without changing the renderer.
        }
        UiTimelineItem::UserInput(text) => {
            let marker_style = Style::default().fg(theme::brand());
            let prefix_w = 2;
            let cap = viewport_width.saturating_sub(prefix_w);
            let mut first = true;
            for line in text.lines() {
                let wrapped = if cap == 0 || line.is_empty() {
                    vec![(0, line.len())]
                } else {
                    text_layout::wrap_text_ranges_wordwise(line, cap)
                };
                for (seg_start, seg_end) in wrapped {
                    let prefix = if first {
                        Span::styled("● ", marker_style)
                    } else {
                        Span::raw("  ")
                    };
                    first = false;
                    let mut spans = vec![prefix];
                    spans.extend(styled_user_input_segment(
                        line,
                        seg_start,
                        seg_end,
                        &app.skills,
                    ));
                    lines.push(Line::from(spans));
                }
            }
        }
        UiTimelineItem::AssistantText(text) => {
            // Insert a blank row before the assistant's spoken text
            // unless it directly follows other spoken text (where the
            // gap would be noise). Reasoning used to be in this list,
            // but gluing a compact `┊ thinking …` line straight into
            // `■ I'll do X` reads as one block — the eye can't find
            // the seam.
            let add_spacing_before = idx > 0
                && !matches!(
                    blocks[idx - 1],
                    UiTimelineItem::AssistantText(_)
                        | UiTimelineItem::Splash
                        | UiTimelineItem::TurnStart(_)
                );
            lines.extend(assistant_text::render_assistant_text_block(
                text,
                viewport_width,
                add_spacing_before,
            ));
        }
        UiTimelineItem::AssistantReasoning(text) => {
            let add_spacing_before = idx > 0
                && !matches!(
                    blocks[idx - 1],
                    UiTimelineItem::AssistantReasoning(_)
                        | UiTimelineItem::Splash
                        | UiTimelineItem::TurnStart(_)
                );
            // Show full reasoning only when either (a) the user has
            // opted into full expansion (Alt+O -> level 2), or (b) this
            // is the live tail of a running turn so the stream stays
            // visible as it arrives. Reasoning is the heaviest block in
            // the transcript and lives at L2 alongside full artifacts
            // and shell output.
            let is_live_tail =
                idx + 1 == blocks.len() && app.turn_active() && !app.has_live_markdown_output();
            let should_expand = expand_level >= 2 || is_live_tail;
            if should_expand {
                lines.extend(assistant_text::render_assistant_reasoning_block(
                    text,
                    viewport_width,
                    add_spacing_before,
                ));
            } else {
                lines.extend(assistant_text::render_assistant_reasoning_block_compact(
                    text,
                    viewport_width,
                    add_spacing_before,
                ));
            }
        }
        UiTimelineItem::Activity(activity) => {
            render_activity_block(activity, expand_level, lines, viewport_width);
            if app.live_tool_output_anchor_block_index() == Some(idx) {
                render_live_tool_output_inline(lines, app, &activity.call.kind, viewport_width);
            }
        }
        UiTimelineItem::ShellOutput {
            command,
            output,
            error,
        } => {
            lines.push(Line::from(Span::styled(
                format!("$ {command}"),
                theme::code_chrome(),
            )));
            if expand_level >= 1 {
                for line in output.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("│ ", theme::code_chrome()),
                        Span::styled(line.to_string(), theme::system_output()),
                    ]));
                }
                if let Some(err) = error {
                    for line in err.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("│ ", theme::code_chrome()),
                            Span::styled(line.to_string(), theme::error()),
                        ]));
                    }
                }
            }
        }
        UiTimelineItem::Error(message) => render_bordered_text_block(
            "ERROR",
            message.lines().map(str::to_string).collect(),
            lines,
            viewport_width,
            theme::error_border(),
            theme::error_title(),
            theme::error(),
        ),
        UiTimelineItem::SystemMessage(text) => {
            for line in text.lines() {
                lines.push(Line::from(Span::styled(
                    line.to_string(),
                    theme::system_message(),
                )));
            }
        }
        UiTimelineItem::PluginPanel(panel) => {
            render_section_panel_block(&panel.title, &panel.content, lines, viewport_width);
        }
        UiTimelineItem::LashlangCode(code) => {
            // Only shown at full expansion (Alt+O). At lower levels the
            // block contributes zero lines — its tool activities carry
            // the visible story already.
            if expand_level >= 2 {
                render_lashlang_code_block(code, lines, viewport_width);
            }
        }
        UiTimelineItem::Splash => {
            render_splash(lines, viewport_width, viewport_height, blocks.len() == 1)
        }
    }
}

/// Render the captured `lashlang` source for an RLM turn, with a dim `╎`
/// gutter to mark it as "what the model ran" (distinct from the `│`
/// shell gutter and `┊` reasoning gutter).
fn render_lashlang_code_block(code: &str, lines: &mut Vec<Line<'static>>, _viewport_width: usize) {
    let header_style = theme::code_chrome();
    let gutter_style = theme::code_chrome();
    let body_style = theme::system_output();

    lines.push(Line::from(Span::styled("lashlang", header_style)));
    for line in code.lines() {
        lines.push(Line::from(vec![
            Span::styled("╎ ", gutter_style),
            Span::styled(line.to_string(), body_style),
        ]));
    }
}

fn render_bordered_text_block(
    title_text: &str,
    content_lines: Vec<String>,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    border_style: Style,
    title_style: Style,
    text_style: Style,
) {
    render_bordered_styled_block(
        title_text,
        &content_lines,
        lines,
        viewport_width,
        border_style,
        title_style,
        |chunk| vec![Span::styled(chunk.to_string(), text_style)],
    );
}

fn render_bordered_styled_block<F>(
    title_text: &str,
    content_lines: &[String],
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    border_style: Style,
    title_style: Style,
    mut style_chunk: F,
) where
    F: FnMut(&str) -> Vec<Span<'static>>,
{
    let title = format!(" {title_text} ");
    let title_w = UnicodeWidthStr::width(title.as_str());
    let fill_w = viewport_width.saturating_sub(3 + title_w);
    lines.push(Line::from(vec![
        Span::styled("┌─", border_style),
        Span::styled(title, title_style),
        Span::styled("─".repeat(fill_w), border_style),
        Span::styled("┐", border_style),
    ]));

    let inner_w = viewport_width.saturating_sub(4).max(1);
    for raw_line in content_lines {
        let segments = if raw_line.is_empty() {
            vec![(0usize, 0usize)]
        } else {
            wrap_line(raw_line, 0, 0, inner_w)
        };
        for (start, end) in segments {
            let chunk = if raw_line.is_empty() {
                String::new()
            } else {
                truncate_to_display_width(&raw_line[start..end], inner_w)
            };
            let styled = style_chunk(&chunk);
            let visible_width: usize = styled.iter().map(Span::width).sum();
            let pad = inner_w.saturating_sub(visible_width);
            let mut row = Vec::new();
            row.push(Span::styled("│ ", border_style));
            row.extend(styled);
            row.push(Span::raw(" ".repeat(pad)));
            row.push(Span::styled(" │", border_style));
            lines.push(Line::from(row));
        }
    }

    let bottom_fill = viewport_width.saturating_sub(2);
    lines.push(Line::from(Span::styled(
        format!("└{}┘", "─".repeat(bottom_fill)),
        border_style,
    )));
}
