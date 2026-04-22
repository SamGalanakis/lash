use lash_tui::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::{app::DisplayBlock, markdown, text_layout, theme};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkdownLane {
    Assistant,
    Reasoning,
}

pub fn normalize_assistant_text(text: &str) -> String {
    let mut out = String::new();
    let mut started = false;
    let mut prev_blank = false;

    for line in text.split('\n') {
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
    render_markdown_lane_documents(
        MarkdownLane::Assistant,
        [text],
        viewport_width,
        add_spacing_before,
    )
}

pub fn push_assistant_reasoning_block(blocks: &mut Vec<DisplayBlock>, text: &str) -> bool {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return false;
    }
    if let Some(DisplayBlock::AssistantReasoning(existing)) = blocks.last_mut() {
        // Consecutive reasoning blocks get merged into a single,
        // paragraph-separated block so the renderer draws one continuous
        // "thinking" span instead of a stack of near-identical
        // separators.
        return merge_assistant_reasoning_text(existing, &cleaned);
    }
    blocks.push(DisplayBlock::AssistantReasoning(cleaned));
    true
}

pub fn merge_assistant_reasoning_text(existing: &mut String, text: &str) -> bool {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return false;
    }

    let existing_cleaned = normalize_assistant_text(existing);
    if existing_cleaned.is_empty() {
        *existing = cleaned;
        return true;
    }
    if existing_cleaned == cleaned || existing_cleaned.ends_with(cleaned.as_str()) {
        if *existing != existing_cleaned {
            *existing = existing_cleaned;
            return true;
        }
        return false;
    }

    let merged = if cleaned.starts_with(existing_cleaned.as_str()) {
        cleaned
    } else {
        format!("{existing_cleaned}\n\n{cleaned}")
    };
    if *existing == merged {
        return false;
    }
    *existing = merged;
    true
}

pub fn render_assistant_reasoning_block(
    text: &str,
    viewport_width: usize,
    add_spacing_before: bool,
) -> Vec<Line<'static>> {
    render_markdown_lane_documents(
        MarkdownLane::Reasoning,
        [text],
        viewport_width,
        add_spacing_before,
    )
}

/// One-line collapsed form of a reasoning block used in history when
/// the turn has moved on. Shows the first meaningful line (stripped of
/// bold markdown wrappers) truncated to fit.
pub fn render_assistant_reasoning_block_compact(
    text: &str,
    viewport_width: usize,
    add_spacing_before: bool,
) -> Vec<Line<'static>> {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return Vec::new();
    }

    let prefix = "┊ ";
    let prefix_w = UnicodeWidthStr::width(prefix);
    let preview_budget = viewport_width.saturating_sub(prefix_w);
    let preview = compact_reasoning_preview(&cleaned, preview_budget);

    let mut lines = Vec::new();
    if add_spacing_before {
        lines.push(Line::from(""));
    }
    lines.push(Line::from(vec![
        Span::styled(prefix.to_string(), theme::assistant_reasoning_bar()),
        Span::styled(preview, theme::assistant_reasoning()),
    ]));
    lines
}

/// Extract a compact one-line preview from reasoning text. Strips
/// leading `**bold**` wrappers, takes the first non-empty line, and
/// truncates to `max_chars` with an ellipsis.
fn compact_reasoning_preview(text: &str, max_chars: usize) -> String {
    let first_line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("thinking");

    let stripped = first_line
        .strip_prefix("**")
        .and_then(|rest| rest.strip_suffix("**").or_else(|| rest.split("**").next()))
        .unwrap_or(first_line);

    if max_chars == 0 {
        return String::new();
    }
    let char_count = stripped.chars().count();
    if char_count <= max_chars {
        return stripped.to_string();
    }
    let budget = max_chars.saturating_sub(1);
    let mut out: String = stripped.chars().take(budget).collect();
    out.push('…');
    out
}

pub(crate) fn render_live_markdown_documents<'a, I>(
    lane: MarkdownLane,
    docs: I,
    viewport_width: usize,
) -> Vec<Line<'static>>
where
    I: IntoIterator<Item = &'a str>,
{
    render_markdown_lane_documents(lane, docs, viewport_width, false)
}

fn render_markdown_lane_documents<'a, I>(
    lane: MarkdownLane,
    docs: I,
    viewport_width: usize,
    add_spacing_before: bool,
) -> Vec<Line<'static>>
where
    I: IntoIterator<Item = &'a str>,
{
    let prefix = match lane {
        MarkdownLane::Assistant => "■ ",
        MarkdownLane::Reasoning => "┊ ",
    };
    let prefix_width = UnicodeWidthStr::width(prefix);
    let content_width = viewport_width.saturating_sub(prefix_width);

    let mut logical_lines = Vec::new();
    for doc in docs {
        let cleaned = normalize_assistant_text(doc);
        if cleaned.is_empty() {
            continue;
        }

        let mut rendered_doc = Vec::new();
        for line in markdown::render_markdown_compact(&cleaned, content_width) {
            if is_blank_line(&line) {
                rendered_doc.push(Line::from(""));
                continue;
            }

            let wrapped = text_layout::wrap_styled_line_with_prefix(
                &line,
                content_width,
                assistant_continuation_prefix,
            );
            match lane {
                MarkdownLane::Assistant => rendered_doc.extend(wrapped),
                MarkdownLane::Reasoning => {
                    rendered_doc.extend(wrapped.into_iter().map(reasoning_overlay_line))
                }
            }
        }

        if rendered_doc.is_empty() {
            continue;
        }
        if !logical_lines.is_empty() {
            logical_lines.push(Line::from(""));
        }
        logical_lines.extend(rendered_doc);
    }

    if logical_lines.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    if add_spacing_before {
        lines.push(Line::from(""));
    }

    match lane {
        MarkdownLane::Assistant => {
            let mut marker_placed = false;
            for line in logical_lines {
                if is_blank_line(&line) {
                    lines.push(Line::from(""));
                    continue;
                }
                let prefix = if marker_placed {
                    "  "
                } else {
                    marker_placed = true;
                    "■ "
                };
                let mut spans = vec![Span::styled(prefix, theme::assistant_bar())];
                spans.extend(line.spans);
                lines.push(Line::from(spans));
            }
        }
        MarkdownLane::Reasoning => {
            for line in logical_lines {
                if is_blank_line(&line) {
                    lines.push(Line::from(vec![Span::styled(
                        "┊ ".to_string(),
                        theme::assistant_reasoning_bar(),
                    )]));
                    continue;
                }
                let mut spans = vec![Span::styled(
                    "┊ ".to_string(),
                    theme::assistant_reasoning_bar(),
                )];
                spans.extend(line.spans);
                lines.push(Line::from(spans));
            }
        }
    }

    lines
}

fn is_blank_line(line: &Line<'_>) -> bool {
    line.spans.iter().all(|span| span.content.trim().is_empty())
}

fn reasoning_overlay_line(line: Line<'static>) -> Line<'static> {
    let overlay = theme::assistant_reasoning();
    Line::from(
        line.spans
            .into_iter()
            .map(|span| Span::styled(span.content.to_string(), span.style.merge(overlay)))
            .collect::<Vec<_>>(),
    )
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

    if trimmed.starts_with("• ") || trimmed.starts_with("· ") {
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
    fn compact_reasoning_preview_strips_bold_wrappers() {
        assert_eq!(
            compact_reasoning_preview("**Evaluating Git commands**\n\nbody text", 80),
            "Evaluating Git commands"
        );
    }

    #[test]
    fn compact_reasoning_preview_truncates_long_first_line() {
        let text =
            "Planning the push after inspecting a dozen git commits before deciding what to do";
        let preview = compact_reasoning_preview(text, 24);
        assert!(preview.ends_with('…'));
        assert!(preview.chars().count() <= 24);
    }

    #[test]
    fn render_assistant_reasoning_block_compact_emits_single_line() {
        let text = "**Evaluating Git commands**\n\nLong rambling paragraph that goes on and on.";
        let rendered = render_assistant_reasoning_block_compact(text, 60, false);
        assert_eq!(rendered.len(), 1);
        let content: String = rendered[0]
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(content, "┊ Evaluating Git commands");
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
