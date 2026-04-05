use lash::strip_repl_fragments;
use ratatui::{
    style::Style,
    text::{Line, Span},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{app::DisplayBlock, markdown, text_display, theme};

pub fn normalize_assistant_text(text: &str) -> String {
    let sanitized = strip_repl_fragments(text);
    let mut out = String::new();
    let mut started = false;
    let mut prev_blank = false;

    for line in sanitized.split('\n') {
        let is_blank = line.trim().is_empty();
        if !started {
            if is_blank {
                continue;
            }
            out.push_str(line);
            started = true;
            prev_blank = false;
            continue;
        }

        if is_blank {
            if !prev_blank {
                out.push('\n');
                prev_blank = true;
            }
        } else {
            out.push('\n');
            out.push_str(line);
            prev_blank = false;
        }
    }

    while out.ends_with('\n') {
        out.pop();
    }

    out
}

pub fn push_assistant_text_block(blocks: &mut Vec<DisplayBlock>, text: &str) -> bool {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return false;
    }
    blocks.push(DisplayBlock::AssistantText(cleaned));
    true
}

#[derive(Clone, Copy)]
struct StyledGlyph {
    ch: char,
    style: Style,
}

pub fn render_assistant_text_block(
    text: &str,
    viewport_width: usize,
    add_spacing_before: bool,
) -> Vec<Line<'static>> {
    let first_prefix = "■ ";
    let continuation_prefix = "  ";
    let prefix_w = UnicodeWidthStr::width(first_prefix);
    let content_width = viewport_width.saturating_sub(prefix_w);
    let rendered = markdown::render_markdown_compact(text, content_width);
    if rendered.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    if add_spacing_before {
        lines.push(Line::from(""));
    }

    let mut marker_placed = false;
    for line in rendered {
        let is_empty = line.spans.iter().all(|span| span.content.trim().is_empty());
        if is_empty {
            lines.push(Line::from(""));
            continue;
        }

        for subline in wrap_rendered_lines_wordwise(std::slice::from_ref(&line), content_width) {
            let prefix = if marker_placed {
                continuation_prefix
            } else {
                marker_placed = true;
                first_prefix
            };
            let mut spans = vec![Span::styled(prefix, theme::assistant_bar())];
            spans.extend(subline.spans);
            lines.push(Line::from(spans));
        }
    }

    lines
}

fn clone_line_owned(line: &Line<'_>) -> Line<'static> {
    let spans = line
        .spans
        .iter()
        .map(|span| Span::styled(span.content.to_string(), span.style))
        .collect::<Vec<_>>();
    Line::from(spans)
}

fn styled_line_to_glyphs(line: &Line<'_>) -> Vec<StyledGlyph> {
    let mut glyphs = Vec::new();
    for span in &line.spans {
        glyphs.extend(span.content.chars().map(|ch| StyledGlyph {
            ch,
            style: span.style,
        }));
    }
    glyphs
}

fn glyphs_to_line(glyphs: &[StyledGlyph]) -> Line<'static> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let mut style = None;

    for glyph in glyphs {
        if style != Some(glyph.style) && !current.is_empty() {
            spans.push(Span::styled(
                std::mem::take(&mut current),
                style.take().unwrap_or_default(),
            ));
        }
        style = Some(glyph.style);
        current.push(glyph.ch);
    }

    if !current.is_empty() {
        spans.push(Span::styled(current, style.take().unwrap_or_default()));
    }

    if spans.is_empty() {
        Line::from("")
    } else {
        Line::from(spans)
    }
}

fn trim_trailing_whitespace_glyphs(glyphs: &[StyledGlyph], start: usize, end: usize) -> usize {
    let mut trimmed = end;
    while trimmed > start && glyphs[trimmed - 1].ch.is_whitespace() {
        trimmed -= 1;
    }
    trimmed
}

fn skip_leading_whitespace_glyphs(glyphs: &[StyledGlyph], mut idx: usize) -> usize {
    while idx < glyphs.len() && glyphs[idx].ch.is_whitespace() {
        idx += 1;
    }
    idx
}

fn wrap_rendered_lines_wordwise(lines: &[Line<'_>], width: usize) -> Vec<Line<'static>> {
    if width == 0 {
        return lines.iter().map(clone_line_owned).collect();
    }

    let mut wrapped = Vec::with_capacity(lines.len());
    for line in lines {
        if text_display::line_visible_width(line) <= width {
            wrapped.push(clone_line_owned(line));
            continue;
        }

        let glyphs = styled_line_to_glyphs(line);
        if glyphs.is_empty() {
            wrapped.push(Line::from(""));
            continue;
        }

        let mut start = 0usize;
        let mut continuation = false;
        'line: while start < glyphs.len() {
            let line_start = if continuation {
                skip_leading_whitespace_glyphs(&glyphs, start)
            } else {
                start
            };
            if line_start >= glyphs.len() {
                break;
            }

            let mut idx = line_start;
            let mut row_width = 0usize;
            let mut last_break = None;
            let mut prev_was_whitespace = false;

            while idx < glyphs.len() {
                let is_whitespace = glyphs[idx].ch.is_whitespace();
                let ch_width = UnicodeWidthChar::width(glyphs[idx].ch).unwrap_or(0);

                if is_whitespace && !prev_was_whitespace && row_width > 0 {
                    last_break = Some(idx);
                }

                if row_width + ch_width > width && row_width > 0 {
                    if is_whitespace {
                        let line_end = trim_trailing_whitespace_glyphs(&glyphs, line_start, idx);
                        if line_end > line_start {
                            wrapped.push(glyphs_to_line(&glyphs[line_start..line_end]));
                            start = skip_leading_whitespace_glyphs(&glyphs, idx);
                            continuation = true;
                            continue 'line;
                        }
                    }
                    break;
                }
                row_width += ch_width;
                prev_was_whitespace = is_whitespace;
                idx += 1;
            }

            if idx >= glyphs.len() {
                wrapped.push(glyphs_to_line(&glyphs[line_start..]));
                break;
            }

            if let Some(break_idx) = last_break {
                let line_end = trim_trailing_whitespace_glyphs(&glyphs, line_start, break_idx);
                if line_end > line_start {
                    wrapped.push(glyphs_to_line(&glyphs[line_start..line_end]));
                    start = skip_leading_whitespace_glyphs(&glyphs, break_idx);
                    continuation = true;
                    continue;
                }
            }

            wrapped.push(glyphs_to_line(&glyphs[line_start..idx]));
            start = idx;
            continuation = true;
        }
    }

    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_assistant_text_collapses_blank_runs() {
        let raw = "\n\nline one\n\n\nline two\n\n";
        assert_eq!(normalize_assistant_text(raw), "line one\n\nline two");
    }

    #[test]
    fn normalize_assistant_text_handles_whitespace_only_lines() {
        let raw = " \n\t\nhello\n   \n\t \nworld\n";
        assert_eq!(normalize_assistant_text(raw), "hello\n\nworld");
    }

    #[test]
    fn normalize_assistant_text_strips_repl_fragments() {
        let raw = "<repl>\nproc\n</repl>\n\nok";
        assert_eq!(normalize_assistant_text(raw), "proc\n\nok");
    }

    #[test]
    fn normalize_assistant_text_strips_dangling_repl_fragments() {
        let raw = "<repl\nhello\n</repl";
        assert_eq!(normalize_assistant_text(raw), "hello");
    }

    #[test]
    fn render_assistant_text_block_keeps_markdown_structure() {
        let text = "Intro line.\n\n## Heading\n\n- item one\n- item two";
        let rendered = render_assistant_text_block(text, 48, false);
        let lines: Vec<String> = rendered
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect();

        assert_eq!(lines[0], "■ Intro line.");
        assert!(lines.iter().any(|line| line == "  Heading"));
        assert!(lines.iter().any(|line| line == ""));
        assert!(lines.iter().any(|line| line.contains("• item one")));
    }
}
