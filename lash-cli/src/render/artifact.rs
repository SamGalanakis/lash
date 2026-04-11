use super::*;

pub(super) fn render_question_panel_artifact(
    panel: &QuestionPanelArtifact,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
) {
    render_bordered_styled_block(
        "QUESTION",
        &question_panel_lines(panel),
        lines,
        viewport_width,
        Style::default().fg(theme::border_faint()),
        Style::default()
            .fg(theme::brand())
            .add_modifier(Modifier::Bold),
        styled_question_chunk,
    );
}

#[cfg(test)]
pub(super) fn render_snippet_preview(
    preview: &SnippetPreviewArtifact,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
) {
    render_snippet_preview_with_indent(preview, lines, viewport_width, "");
}

pub(super) fn render_snippet_preview_with_indent(
    preview: &SnippetPreviewArtifact,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    indent: &str,
) {
    let indent_width = UnicodeWidthStr::width(indent);
    let has_custom_title = preview.title.is_some();
    let title = preview.title.as_deref().unwrap_or("SNIPPET");
    lines.push(indented_line(
        prompt::prompt_section_label(title, viewport_width.saturating_sub(indent_width)),
        indent,
    ));
    if !has_custom_title {
        push_wrapped_styled_chunks_with_indent(
            lines,
            &snippet_meta_line(preview),
            viewport_width,
            indent,
            |chunk| {
                styled_snippet_chunk(chunk, theme::system_output(), preview.language.as_deref())
            },
        );
    }
    lines.push(Line::from(indent.to_string()));
    match preview.render_mode {
        SnippetRenderMode::Markdown => {
            if !preview.content.trim().is_empty() {
                let rendered = markdown::render_markdown(
                    &preview.content,
                    viewport_width.saturating_sub(indent_width),
                );
                lines.extend(rendered.into_iter().map(|line| indented_line(line, indent)));
            }
        }
        SnippetRenderMode::Code => {
            render_line_numbered_snippet_block(
                preview,
                lines,
                viewport_width,
                indent,
                theme::code_content(),
            );
        }
        SnippetRenderMode::Text => {
            render_line_numbered_snippet_block(
                preview,
                lines,
                viewport_width,
                indent,
                theme::system_output(),
            );
        }
    }
}

pub(super) fn render_plan_block(
    content: &str,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
) {
    let content_lines = content.lines().map(str::to_string).collect::<Vec<_>>();
    render_bordered_styled_block(
        "PLAN",
        &content_lines,
        lines,
        viewport_width,
        Style::default().fg(theme::border_faint()),
        Style::default()
            .fg(theme::brand())
            .add_modifier(Modifier::Bold),
        styled_plan_chunk,
    );
}

pub(super) fn render_section_panel_block(
    title_text: &str,
    content: &str,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
) {
    lines.push(prompt::prompt_section_label(title_text, viewport_width));
    if !content.trim().is_empty() {
        lines.push(Line::from(""));
        lines.extend(markdown::render_markdown(content, viewport_width));
    }
}

#[cfg(test)]
pub(super) fn styled_snippet_chunk_for_test(
    chunk: &str,
    content_style: Style,
    language: Option<&str>,
) -> Vec<Span<'static>> {
    styled_snippet_chunk(chunk, content_style, language)
}

fn question_panel_lines(panel: &QuestionPanelArtifact) -> Vec<String> {
    let mut lines = panel.prompt_lines.clone();
    for (idx, option) in panel.options.iter().enumerate() {
        let marker = match panel.selection_mode {
            Some(QuestionPanelSelectionMode::Multi) => {
                if option.selected {
                    "☑"
                } else {
                    "☐"
                }
            }
            _ => {
                if option.selected {
                    "◉"
                } else {
                    "○"
                }
            }
        };
        lines.push(format!("{marker} {}. {}", idx + 1, option.label));
    }
    if let Some(answer) = panel.answer.as_deref().filter(|value| !value.is_empty()) {
        lines.push(format!("Answer · {answer}"));
    }
    if let Some(note) = panel.note.as_deref().filter(|value| !value.is_empty()) {
        lines.push(format!("Note · {note}"));
    }
    lines
}

fn styled_question_chunk(chunk: &str) -> Vec<Span<'static>> {
    if let Some(rest) = chunk.strip_prefix("Answer · ") {
        return vec![
            Span::styled(
                "Answer",
                Style::default()
                    .fg(theme::state_ok())
                    .add_modifier(Modifier::Bold),
            ),
            Span::styled(" · ", Style::default().fg(theme::border_dim())),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }
    if let Some(rest) = chunk.strip_prefix("Note · ") {
        return vec![
            Span::styled(
                "Note",
                Style::default()
                    .fg(theme::state_ok())
                    .add_modifier(Modifier::Bold),
            ),
            Span::styled(" · ", Style::default().fg(theme::border_dim())),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }
    for (marker, selected) in [("◉ ", true), ("○ ", false), ("☑ ", true), ("☐ ", false)] {
        if let Some(rest) = chunk.strip_prefix(marker) {
            let marker_style = if selected {
                Style::default()
                    .fg(theme::state_ok())
                    .add_modifier(Modifier::Bold)
            } else {
                Style::default().fg(theme::border_dim())
            };
            let text_style = if selected {
                theme::assistant_text().add_modifier(Modifier::Bold)
            } else {
                theme::assistant_text()
            };
            return vec![
                Span::styled(marker.to_string(), marker_style),
                Span::styled(rest.to_string(), text_style),
            ];
        }
    }
    if let Some((num, rest)) = chunk.split_once(". ")
        && num.chars().all(|ch| ch.is_ascii_digit())
    {
        return vec![
            Span::styled(
                format!("{num}."),
                Style::default()
                    .fg(theme::brand())
                    .add_modifier(Modifier::Bold),
            ),
            Span::raw(" "),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }
    vec![Span::styled(chunk.to_string(), theme::assistant_text())]
}

fn styled_plan_chunk(chunk: &str) -> Vec<Span<'static>> {
    if let Some(rest) = chunk.strip_prefix("✓ ") {
        return vec![
            Span::styled("✓", theme::plan_done_marker()),
            Span::raw(" "),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }
    if let Some(rest) = chunk.strip_prefix("▸ ") {
        return vec![
            Span::styled("▸", theme::plan_active_marker()),
            Span::raw(" "),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }
    if let Some(rest) = chunk.strip_prefix("○ ") {
        return vec![
            Span::styled("○", theme::plan_pending_marker()),
            Span::raw(" "),
            Span::styled(rest.to_string(), theme::assistant_text()),
        ];
    }
    vec![Span::styled(chunk.to_string(), theme::assistant_text())]
}

fn render_line_numbered_snippet_block(
    preview: &SnippetPreviewArtifact,
    lines: &mut Vec<Line<'static>>,
    viewport_width: usize,
    indent: &str,
    text_style: Style,
) {
    let line_number_width = preview.end_line.to_string().len().max(2);
    let continuation_prefix = snippet_line_prefix(None, line_number_width, indent);
    if preview.content.is_empty() {
        push_wrapped_styled_line_with_continuation(
            lines,
            &Line::from(snippet_line_prefix(
                Some(preview.start_line),
                line_number_width,
                indent,
            )),
            viewport_width,
            continuation_prefix.clone(),
        );
    } else {
        for (offset, line) in preview.content.lines().enumerate() {
            let mut spans =
                snippet_line_prefix(Some(preview.start_line + offset), line_number_width, indent);
            if preview.language.is_some() && text_style == theme::code_content() {
                spans.extend(highlight_code_snippet(line, preview.language.as_deref()));
            } else {
                spans.push(Span::styled(line.to_string(), text_style));
            }
            push_wrapped_styled_line_with_continuation(
                lines,
                &Line::from(spans),
                viewport_width,
                continuation_prefix.clone(),
            );
        }
    }
}

fn snippet_line_prefix(
    line_number: Option<usize>,
    line_number_width: usize,
    indent: &str,
) -> Vec<Span<'static>> {
    let number = line_number
        .map(|value| format!("{value:>width$}", width = line_number_width))
        .unwrap_or_else(|| " ".repeat(line_number_width));
    let mut spans = Vec::new();
    if !indent.is_empty() {
        spans.push(Span::raw(indent.to_string()));
    }
    spans.push(Span::styled(number, theme::code_chrome()));
    spans.push(Span::styled(" │ ".to_string(), theme::code_chrome()));
    spans
}

fn push_wrapped_styled_line_with_continuation(
    lines: &mut Vec<Line<'static>>,
    line: &Line<'static>,
    viewport_width: usize,
    continuation_prefix: Vec<Span<'static>>,
) {
    let text = text_layout::line_text(line);
    let continuation_prefix_width = text_layout::spans_display_width(&continuation_prefix);
    let segments = if text.is_empty() {
        vec![(0usize, 0usize)]
    } else {
        wrap_line(&text, 0, continuation_prefix_width, viewport_width.max(1))
    };

    for (segment_idx, (start, end)) in segments.into_iter().enumerate() {
        let mut spans = Vec::new();
        if segment_idx > 0 {
            spans.extend(continuation_prefix.iter().cloned());
        }
        spans.extend(text_layout::slice_line_spans(line, start, end));
        lines.push(Line::from(spans));
    }
}

fn push_wrapped_styled_chunks_with_indent<F>(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    viewport_width: usize,
    indent: &str,
    mut style_chunk: F,
) where
    F: FnMut(&str) -> Vec<Span<'static>>,
{
    let available = viewport_width
        .saturating_sub(UnicodeWidthStr::width(indent))
        .max(1);
    let segments = if text.is_empty() {
        vec![(0usize, 0usize)]
    } else {
        text_layout::wrap_text_ranges_wordwise(text, available)
    };
    for (start, end) in segments {
        let chunk = if text.is_empty() {
            String::new()
        } else {
            text[start..end].to_string()
        };
        let mut spans = Vec::new();
        if !indent.is_empty() {
            spans.push(Span::raw(indent.to_string()));
        }
        spans.extend(style_chunk(&chunk));
        lines.push(Line::from(spans));
    }
}

fn indented_line(line: Line<'static>, indent: &str) -> Line<'static> {
    if indent.is_empty() {
        return line;
    }
    let mut spans = vec![Span::raw(indent.to_string())];
    spans.extend(line.spans);
    Line::from(spans)
}

fn snippet_meta_line(preview: &SnippetPreviewArtifact) -> String {
    let mut line = format!(
        "File · {}:{}-{}",
        preview.path, preview.start_line, preview.end_line
    );
    if let Some(language) = preview.language.as_deref() {
        line.push_str(" · ");
        line.push_str(language);
    }
    line
}

fn styled_snippet_chunk(
    chunk: &str,
    content_style: Style,
    language: Option<&str>,
) -> Vec<Span<'static>> {
    if let Some(rest) = chunk.strip_prefix("File · ") {
        return vec![
            Span::styled("File", theme::code_header()),
            Span::styled(" · ", Style::default().fg(theme::border_dim())),
            Span::styled(rest.to_string(), theme::system_output()),
        ];
    }
    if let Some((prefix, rest)) = chunk.split_once("│")
        && prefix.trim().chars().all(|ch| ch.is_ascii_digit())
    {
        let mut spans = vec![
            Span::styled(prefix.to_string(), theme::code_chrome()),
            Span::styled("│", theme::code_chrome()),
        ];
        if language.is_some() && content_style == theme::code_content() {
            spans.extend(highlight_code_snippet(rest, language));
        } else {
            spans.push(Span::styled(rest.to_string(), content_style));
        }
        return spans;
    }
    vec![Span::styled(chunk.to_string(), content_style)]
}

fn highlight_code_snippet(text: &str, language: Option<&str>) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut cursor = 0usize;

    while cursor < text.len() {
        if comment_marker_at(text, cursor, language).is_some() {
            push_styled_chunk(&mut spans, &text[cursor..], theme::code_comment());
            break;
        }

        let ch = text[cursor..]
            .chars()
            .next()
            .expect("slice should start on char boundary");
        let next = cursor + ch.len_utf8();

        if matches!(ch, '"' | '\'' | '`') {
            let end = string_end(text, cursor, ch);
            push_styled_chunk(&mut spans, &text[cursor..end], theme::code_string());
            cursor = end;
            continue;
        }

        if ch.is_ascii_alphabetic() || ch == '_' {
            let end = identifier_end(text, cursor);
            let word = &text[cursor..end];
            let style = if is_code_keyword(word, language) {
                theme::code_keyword()
            } else {
                theme::code_content()
            };
            push_styled_chunk(&mut spans, word, style);
            cursor = end;
            continue;
        }

        push_styled_chunk(&mut spans, &text[cursor..next], theme::code_content());
        cursor = next;
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), theme::code_content()));
    }
    spans
}

fn push_styled_chunk(spans: &mut Vec<Span<'static>>, text: &str, style: Style) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = spans.last_mut()
        && last.style == style
    {
        last.content.to_mut().push_str(text);
        return;
    }
    spans.push(Span::styled(text.to_string(), style));
}

fn comment_marker_at(text: &str, idx: usize, language: Option<&str>) -> Option<&'static str> {
    let markers = match language.unwrap_or_default() {
        "py" | "sh" | "bash" | "zsh" | "fish" | "yaml" | "toml" | "rb" | "nix" => &["#"][..],
        "sql" => &["--"][..],
        "html" => &["<!--"][..],
        _ => &["//", "#"][..],
    };
    markers
        .iter()
        .copied()
        .find(|marker| text[idx..].starts_with(marker))
}

fn string_end(text: &str, start: usize, quote: char) -> usize {
    let mut escaped = false;
    let mut idx = start + quote.len_utf8();
    while idx < text.len() {
        let ch = text[idx..]
            .chars()
            .next()
            .expect("slice should start on char boundary");
        idx += ch.len_utf8();
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && quote != '`' {
            escaped = true;
            continue;
        }
        if ch == quote {
            break;
        }
    }
    idx
}

fn identifier_end(text: &str, start: usize) -> usize {
    let mut idx = start;
    while idx < text.len() {
        let ch = text[idx..]
            .chars()
            .next()
            .expect("slice should start on char boundary");
        if !(ch.is_ascii_alphanumeric() || ch == '_') {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn is_code_keyword(word: &str, language: Option<&str>) -> bool {
    let word = word.trim();
    if word.is_empty() {
        return false;
    }
    match language.unwrap_or_default() {
        "py" => matches!(
            word,
            "def"
                | "class"
                | "import"
                | "from"
                | "return"
                | "if"
                | "elif"
                | "else"
                | "for"
                | "while"
                | "try"
                | "except"
                | "with"
                | "as"
                | "async"
                | "await"
                | "pass"
                | "yield"
                | "True"
                | "False"
                | "None"
                | "in"
                | "is"
        ),
        "sh" | "bash" | "zsh" | "fish" => matches!(
            word,
            "if" | "then"
                | "else"
                | "fi"
                | "for"
                | "do"
                | "done"
                | "case"
                | "esac"
                | "function"
                | "in"
                | "while"
        ),
        "sql" => matches!(
            word.to_ascii_uppercase().as_str(),
            "SELECT"
                | "FROM"
                | "WHERE"
                | "JOIN"
                | "LEFT"
                | "RIGHT"
                | "INNER"
                | "OUTER"
                | "INSERT"
                | "UPDATE"
                | "DELETE"
                | "CREATE"
                | "ALTER"
                | "DROP"
                | "GROUP"
                | "ORDER"
                | "BY"
                | "LIMIT"
                | "AND"
                | "OR"
                | "NOT"
                | "AS"
        ),
        _ => matches!(
            word,
            "fn" | "let"
                | "mut"
                | "pub"
                | "impl"
                | "struct"
                | "enum"
                | "trait"
                | "async"
                | "await"
                | "if"
                | "else"
                | "match"
                | "for"
                | "while"
                | "loop"
                | "return"
                | "use"
                | "mod"
                | "const"
                | "static"
                | "class"
                | "interface"
                | "function"
                | "def"
                | "import"
                | "export"
                | "package"
                | "new"
                | "try"
                | "catch"
                | "throw"
                | "throws"
                | "null"
                | "true"
                | "false"
                | "self"
                | "super"
                | "this"
                | "where"
                | "type"
                | "extends"
                | "implements"
                | "yield"
        ),
    }
}
