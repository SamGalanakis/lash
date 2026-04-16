use super::*;

pub(crate) struct PromptRenderSnapshot {
    pub combined_lines: Vec<Line<'static>>,
    pub review_lines: Vec<Line<'static>>,
    pub interaction_lines: Vec<Line<'static>>,
    pub split_layout: bool,
    pub review_viewport_height: usize,
}

pub(crate) fn prompt_height(app: &App, frame_width: u16, frame_height: u16) -> u16 {
    let Some(prompt) = app.prompt_state() else {
        return 3;
    };
    let inner_w = prompt_inner_width(frame_width);
    let max_height = frame_height.saturating_sub(1).max(3);
    let content_h = prompt_content_lines(prompt, inner_w).len() as u16;
    content_h.max(1).min(max_height)
}

pub(crate) fn prompt_content_lines_snapshot(
    prompt: &PromptState,
    inner_w: usize,
) -> Vec<Line<'static>> {
    prompt_content_lines(prompt, inner_w)
}

pub(crate) fn prompt_max_scroll(
    prompt: &PromptState,
    inner_w: usize,
    visible_height: usize,
) -> usize {
    let snapshot = prompt_render_snapshot(prompt, inner_w, visible_height);
    if snapshot.split_layout {
        snapshot
            .review_lines
            .len()
            .saturating_sub(snapshot.review_viewport_height)
    } else {
        snapshot.combined_lines.len().saturating_sub(visible_height)
    }
}

pub(crate) fn prompt_render_snapshot(
    prompt: &PromptState,
    inner_w: usize,
    visible_height: usize,
) -> PromptRenderSnapshot {
    let review_lines = prompt_review_lines(prompt, inner_w);
    let interaction_lines = prompt_interaction_lines(prompt, inner_w);
    let combined_lines = join_prompt_sections(&review_lines, &interaction_lines);
    let split_layout = prompt.uses_split_layout()
        && !review_lines.is_empty()
        && !interaction_lines.is_empty()
        && visible_height > interaction_lines.len();
    let review_viewport_height = if split_layout {
        visible_height.saturating_sub(interaction_lines.len())
    } else {
        visible_height
    };
    PromptRenderSnapshot {
        combined_lines,
        review_lines,
        interaction_lines,
        split_layout,
        review_viewport_height,
    }
}

pub(crate) fn prompt_section_label(text: &str, inner_w: usize) -> Line<'static> {
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

fn prompt_input_text(prompt: &PromptState) -> (String, bool) {
    if prompt.supports_note() && !prompt.is_text_entry() && prompt.reply_text.is_empty() {
        return ("Add an optional note".to_string(), true);
    }
    let mut display = prompt.reply_text.clone();
    if prompt.is_text_entry() {
        let cursor = prompt.reply_cursor.min(display.len());
        display.insert(cursor, '█');
    }
    (display, false)
}

fn prompt_option_has_embedded_index(text: &str) -> bool {
    let trimmed = text.trim_start();
    if let Some(rest) = trimmed.strip_prefix('[') {
        let digits = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
        return digits > 0 && rest[digits..].starts_with(']');
    }
    let digits = trimmed.chars().take_while(|ch| ch.is_ascii_digit()).count();
    digits > 0 && matches!(trimmed[digits..].chars().next(), Some('.') | Some(')'))
}

fn prompt_option_text(idx: usize, option: &str) -> String {
    if prompt_option_has_embedded_index(option) {
        option.to_string()
    } else {
        format!("{}. {option}", idx + 1)
    }
}

fn prompt_help_items(prompt: &PromptState) -> Vec<(&'static str, &'static str)> {
    if prompt.has_options() {
        if prompt.supports_note() {
            if prompt.is_text_entry() {
                vec![
                    ("tab", "choices"),
                    ("enter", "submit"),
                    ("shift+tab", "newline"),
                    ("esc", "cancel"),
                ]
            } else if prompt.is_multi() {
                vec![
                    ("↑↓", "move"),
                    ("space", "toggle"),
                    ("tab", "note"),
                    ("enter", "submit"),
                    ("esc", "cancel"),
                ]
            } else {
                vec![
                    ("↑↓", "choose"),
                    ("tab", "note"),
                    ("enter", "submit"),
                    ("esc", "cancel"),
                ]
            }
        } else if prompt.is_multi() {
            vec![
                ("↑↓", "move"),
                ("space", "toggle"),
                ("enter", "submit"),
                ("esc", "cancel"),
            ]
        } else {
            vec![("↑↓", "choose"), ("enter", "submit"), ("esc", "cancel")]
        }
    } else {
        vec![
            ("enter", "submit"),
            ("shift+tab", "newline"),
            ("esc", "cancel"),
        ]
    }
}

fn prompt_help_line(prompt: &PromptState) -> Line<'static> {
    let mut spans = Vec::new();
    for (idx, (key, desc)) in prompt_help_items(prompt).into_iter().enumerate() {
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

fn push_wrapped_prefixed_lines(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    total_width: usize,
    first_prefix: Span<'static>,
    cont_prefix: Span<'static>,
    text_style: Style,
) {
    let first_prefix_width = first_prefix.width();
    let cont_prefix_width = cont_prefix.width();
    let mut first_visual_line = true;

    for logical_line in text.split('\n') {
        let prefix_width = if first_visual_line {
            first_prefix_width
        } else {
            cont_prefix_width
        };
        let segments = wrap_line(logical_line, prefix_width, cont_prefix_width, total_width);
        for (segment_idx, &(seg_start, seg_end)) in segments.iter().enumerate() {
            let prefix = if first_visual_line && segment_idx == 0 {
                first_prefix.clone()
            } else {
                cont_prefix.clone()
            };
            lines.push(Line::from(vec![
                prefix,
                text_display::sanitize_span(
                    logical_line[seg_start..seg_end].to_string(),
                    text_style,
                ),
            ]));
            first_visual_line = false;
        }
    }
}

fn push_wrapped_plain_lines(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    total_width: usize,
    style: Style,
) {
    push_wrapped_prefixed_lines(
        lines,
        text,
        total_width,
        Span::styled(String::new(), style),
        Span::styled(String::new(), style),
        style,
    );
}

fn normalize_prompt_panel_markdown(title: &str, markdown: &str) -> String {
    let mut lines = markdown.lines();
    let Some(first_line) = lines.next() else {
        return String::new();
    };
    let Some(heading) = first_line.trim().strip_prefix("# ") else {
        return markdown.to_string();
    };
    if normalized_panel_heading(heading) != normalized_panel_heading(title) {
        return markdown.to_string();
    }

    let mut remaining = lines.collect::<Vec<_>>();
    if remaining.first().is_some_and(|line| line.trim().is_empty()) {
        remaining.remove(0);
    }
    remaining.join("\n")
}

fn normalized_panel_heading(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn render_prompt_panel_lines(
    title: &str,
    markdown_text: &str,
    inner_w: usize,
) -> Vec<Line<'static>> {
    let mut lines = vec![prompt_section_label(title, inner_w)];
    let normalized = normalize_prompt_panel_markdown(title, markdown_text);
    if !normalized.trim().is_empty() {
        lines.push(Line::from(""));
        lines.extend(markdown::render_markdown(&normalized, inner_w));
    }
    lines
}

fn prompt_review_lines(prompt: &PromptState, inner_w: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(panel) = prompt.request.panel.as_ref() {
        lines.extend(render_prompt_panel_lines(
            &panel.title,
            &panel.markdown,
            inner_w,
        ));
        if !prompt.request.question.is_empty() {
            lines.push(Line::from(""));
        }
    }

    if !prompt.request.question.is_empty() {
        push_wrapped_plain_lines(
            &mut lines,
            &prompt.request.question,
            inner_w,
            Style::default().fg(theme::text_primary()),
        );
    }
    lines
}

fn prompt_interaction_lines(prompt: &PromptState, inner_w: usize) -> Vec<Line<'static>> {
    let has_options = prompt.has_options();
    let show_text_input = prompt.shows_text_input();
    let mut lines = Vec::new();

    if has_options {
        lines.push(prompt_section_label(
            if prompt.is_multi() {
                "Selections"
            } else {
                "Choices"
            },
            inner_w,
        ));
        for (idx, opt) in prompt.request.options.iter().enumerate() {
            let active = prompt.selected_option_idx() == Some(idx);
            let marked = prompt.option_marked(idx);
            let text_style = if active {
                Style::default()
                    .fg(theme::text_primary())
                    .bg(theme::surface_raised())
            } else if marked {
                Style::default().fg(theme::text_muted())
            } else {
                Style::default().fg(theme::text_subtle())
            };
            let prefix_style = if active {
                Style::default()
                    .fg(theme::brand())
                    .bg(theme::surface_raised())
                    .add_modifier(Modifier::Bold)
            } else if marked {
                Style::default()
                    .fg(theme::state_ok())
                    .add_modifier(Modifier::Bold)
            } else {
                text_style
            };
            let option_text = prompt_option_text(idx, opt);
            push_wrapped_prefixed_lines(
                &mut lines,
                &option_text,
                inner_w,
                Span::styled(
                    if prompt.is_multi() {
                        if active {
                            if marked { "› [x]" } else { "› [ ]" }
                        } else if marked {
                            "  [x]"
                        } else {
                            "  [ ]"
                        }
                    } else if active {
                        "› "
                    } else {
                        "  "
                    },
                    prefix_style,
                ),
                Span::styled(if prompt.is_multi() { "     " } else { "  " }, prefix_style),
                text_style,
            );
        }
    }

    if show_text_input {
        if has_options {
            lines.push(Line::from(""));
        }
        if prompt.supports_note() {
            lines.push(prompt_section_label("Note", inner_w));
        }
        let (input_text, is_placeholder) = prompt_input_text(prompt);
        push_wrapped_prefixed_lines(
            &mut lines,
            &input_text,
            inner_w,
            Span::styled(
                format!(" {} ", theme::PROMPT_CHAR),
                if prompt.is_text_entry() {
                    theme::prompt()
                } else {
                    Style::default().fg(theme::border_faint())
                },
            ),
            Span::styled("   ", Style::default().fg(theme::border_faint())),
            if is_placeholder {
                Style::default().fg(theme::border_faint())
            } else {
                Style::default().fg(theme::text_primary())
            },
        );
    }

    if !lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines.push(prompt_help_line(prompt));
    lines
}

fn join_prompt_sections(
    review_lines: &[Line<'static>],
    interaction_lines: &[Line<'static>],
) -> Vec<Line<'static>> {
    let mut lines = review_lines.to_vec();
    if !lines.is_empty() && !interaction_lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines.extend(interaction_lines.iter().cloned());
    lines
}

fn prompt_content_lines(prompt: &PromptState, inner_w: usize) -> Vec<Line<'static>> {
    let review_lines = prompt_review_lines(prompt, inner_w);
    let interaction_lines = prompt_interaction_lines(prompt, inner_w);
    join_prompt_sections(&review_lines, &interaction_lines)
}
