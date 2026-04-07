use lash::strip_repl_fragments;
use lash_tui::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::{app::DisplayBlock, markdown, text_layout, theme};

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

        for subline in text_layout::wrap_styled_line_with_prefix(
            &line,
            content_width,
            assistant_continuation_prefix,
        ) {
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

pub fn render_live_assistant_text_block(text: &str, viewport_width: usize) -> Vec<Line<'static>> {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return Vec::new();
    }

    let first_prefix = "■ ";
    let continuation_prefix = "  ";
    let prefix_w = UnicodeWidthStr::width(first_prefix);
    let content_width = viewport_width.saturating_sub(prefix_w);

    let mut lines = Vec::new();
    let mut marker_placed = false;

    for line in cleaned.split('\n') {
        if line.is_empty() {
            lines.push(Line::from(""));
            continue;
        }

        let wrapped = text_layout::wrap_text_ranges_wordwise(line, content_width);
        for (seg_start, seg_end) in wrapped {
            let prefix = if marker_placed {
                continuation_prefix
            } else {
                marker_placed = true;
                first_prefix
            };
            lines.push(Line::from(vec![
                Span::styled(prefix, theme::assistant_bar()),
                Span::styled(
                    line[seg_start..seg_end].to_string(),
                    theme::assistant_text(),
                ),
            ]));
        }
    }

    lines
}

fn assistant_continuation_prefix(line: &Line<'static>) -> Vec<Span<'static>> {
    let text = text_layout::line_text(line);
    let default_style = text_layout::continuation_prefix_style_from_line(line);

    if text.starts_with("│ ") {
        let style = line
            .spans
            .first()
            .map(|span| span.style)
            .unwrap_or(default_style);
        return vec![Span::styled("│ ".to_string(), style)];
    }

    let leading_ws_chars = text.chars().take_while(|ch| ch.is_whitespace()).count();
    let leading_ws = " ".repeat(leading_ws_chars);
    let trimmed = text.trim_start();

    if trimmed.starts_with("• ") {
        return vec![Span::styled(format!("{leading_ws}  "), default_style)];
    }

    let digits = trimmed.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digits > 0 && trimmed[digits..].starts_with(". ") {
        return vec![Span::styled(
            format!("{leading_ws}{}", " ".repeat(digits + 2)),
            default_style,
        )];
    }

    Vec::new()
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
        assert!(lines.iter().any(String::is_empty));
        assert!(lines.iter().any(|line| line.contains("• item one")));
    }

    #[test]
    fn render_assistant_text_block_preserves_bullet_continuation_indent() {
        let text = "- Complete a full spring-cleaning pass in two phases: first remove dead or stale things, then simplify what remains.";
        let rendered = render_assistant_text_block(text, 44, false);
        let lines: Vec<String> = rendered
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect();

        assert!(lines.len() >= 2);
        assert!(lines[0].starts_with("■ • "));
        assert!(lines[1].starts_with("    "));
    }
}
