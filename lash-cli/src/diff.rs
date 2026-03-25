use ratatui::{
    style::Style,
    text::{Line, Span},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::theme;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InlineDiffKind {
    Add,
    Remove,
    Context,
    Hunk,
    Meta,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InlineDiffRow {
    line_number: Option<usize>,
    sign: Option<char>,
    text: String,
    kind: InlineDiffKind,
}

pub fn inline_diff_height(diff: &str, width: usize, prefix_width: usize) -> usize {
    let rows = parse_inline_diff_rows(diff);
    if rows.is_empty() {
        return 0;
    }

    let number_width = diff_line_number_width(&rows);
    let content_width = width.saturating_sub(prefix_width + number_width + 3);
    rows.iter()
        .map(|row| wrapped_line_height(&row.text, content_width))
        .sum()
}

pub fn render_inline_diff<'a>(
    lines: &mut Vec<Line<'a>>,
    diff: &str,
    viewport_width: usize,
    prefix: &str,
) {
    let rows = parse_inline_diff_rows(diff);
    if rows.is_empty() {
        return;
    }

    let number_width = diff_line_number_width(&rows);
    let prefix_width = UnicodeWidthStr::width(prefix);
    let content_width = viewport_width.saturating_sub(prefix_width + number_width + 3);

    for row in rows {
        render_inline_diff_row(lines, &row, number_width, content_width, prefix);
    }
}

fn render_inline_diff_row<'a>(
    lines: &mut Vec<Line<'a>>,
    row: &InlineDiffRow,
    number_width: usize,
    content_width: usize,
    prefix: &str,
) {
    let segments = wrap_segments(&row.text, content_width);
    let empty_number = " ".repeat(number_width);
    let (gutter_style, body_style) = inline_diff_styles(row.kind);

    for (idx, segment) in segments.iter().enumerate() {
        let (number, sign) = if idx == 0 {
            (
                row.line_number
                    .map(|value| format!("{value:>number_width$}"))
                    .unwrap_or_else(|| empty_number.clone()),
                row.sign
                    .map(|value| format!("{value} "))
                    .unwrap_or_else(|| "  ".to_string()),
            )
        } else {
            (empty_number.clone(), "  ".to_string())
        };

        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), theme::patch_frame()),
            Span::styled(number, gutter_style),
            Span::styled(" ".to_string(), gutter_style),
            Span::styled(sign, gutter_style),
            Span::styled(segment.clone(), body_style),
        ]));
    }
}

fn diff_line_number_width(rows: &[InlineDiffRow]) -> usize {
    rows.iter()
        .filter_map(|row| row.line_number)
        .max()
        .map(|value| value.to_string().len())
        .unwrap_or(1)
}

fn inline_diff_styles(kind: InlineDiffKind) -> (Style, Style) {
    match kind {
        InlineDiffKind::Add => (theme::patch_diff_add_gutter(), theme::patch_diff_add_line()),
        InlineDiffKind::Remove => (
            theme::patch_diff_remove_gutter(),
            theme::patch_diff_remove_line(),
        ),
        InlineDiffKind::Context => (
            theme::patch_diff_context_gutter(),
            theme::patch_diff_context_line(),
        ),
        InlineDiffKind::Hunk => (
            theme::patch_diff_hunk_gutter(),
            theme::patch_diff_hunk_line(),
        ),
        InlineDiffKind::Meta => (
            theme::patch_diff_context_gutter(),
            theme::patch_diff_meta_line(),
        ),
    }
}

fn wrap_segments(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut col = 0usize;
    for (idx, ch) in text.char_indices() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if col + ch_width > width && col > 0 {
            segments.push(text[start..idx].to_string());
            start = idx;
            col = ch_width;
        } else {
            col += ch_width;
        }
    }
    segments.push(text[start..].to_string());
    segments
}

fn wrapped_line_height(text: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let display_width = UnicodeWidthStr::width(text);
    if display_width == 0 {
        1
    } else {
        display_width.div_ceil(width)
    }
}

fn parse_inline_diff_rows(diff: &str) -> Vec<InlineDiffRow> {
    let mut rows = Vec::new();
    let mut old_line = 0usize;
    let mut new_line = 0usize;
    let mut in_hunk = false;

    for line in diff.lines() {
        if line.starts_with("--- ") || line.starts_with("+++ ") {
            continue;
        }
        if let Some((old_start, new_start)) = parse_hunk_header(line) {
            old_line = old_start;
            new_line = new_start;
            in_hunk = true;
            rows.push(InlineDiffRow {
                line_number: None,
                sign: None,
                text: line.to_string(),
                kind: InlineDiffKind::Hunk,
            });
            continue;
        }

        if in_hunk {
            if let Some(text) = line.strip_prefix('+') {
                rows.push(InlineDiffRow {
                    line_number: Some(new_line),
                    sign: Some('+'),
                    text: text.to_string(),
                    kind: InlineDiffKind::Add,
                });
                new_line += 1;
                continue;
            }
            if let Some(text) = line.strip_prefix('-') {
                rows.push(InlineDiffRow {
                    line_number: Some(old_line),
                    sign: Some('-'),
                    text: text.to_string(),
                    kind: InlineDiffKind::Remove,
                });
                old_line += 1;
                continue;
            }
            if let Some(text) = line.strip_prefix(' ') {
                rows.push(InlineDiffRow {
                    line_number: Some(new_line),
                    sign: Some(' '),
                    text: text.to_string(),
                    kind: InlineDiffKind::Context,
                });
                old_line += 1;
                new_line += 1;
                continue;
            }
        }

        rows.push(InlineDiffRow {
            line_number: None,
            sign: None,
            text: line.to_string(),
            kind: InlineDiffKind::Meta,
        });
    }

    rows
}

fn parse_hunk_header(line: &str) -> Option<(usize, usize)> {
    let ranges = line.strip_prefix("@@ -")?;
    let (old_range, new_part) = ranges.split_once(" +")?;
    let (new_range, _) = new_part.split_once(" @@")?;

    Some((
        parse_hunk_range_start(old_range)?,
        parse_hunk_range_start(new_range)?,
    ))
}

fn parse_hunk_range_start(range: &str) -> Option<usize> {
    range.split(',').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn render_inline_diff_adds_guttered_line_numbers() {
        let diff = "--- a/foo.rs\n+++ b/foo.rs\n@@ -10,2 +10,2 @@\n-old\n context\n+new\n... (4 more lines)";
        let mut lines = Vec::new();
        render_inline_diff(&mut lines, diff, 80, "  │ ");

        let rendered = rendered(&lines);
        assert!(rendered[0].contains("@@ -10,2 +10,2 @@"));
        assert!(rendered[1].contains("10 - old"));
        assert!(rendered[2].contains("10   context"));
        assert!(rendered[3].contains("11 + new"));
        assert!(rendered[4].contains("... (4 more lines)"));
    }

    #[test]
    fn inline_diff_height_counts_wrapped_rows() {
        let diff = "@@ -1,1 +1,1 @@\n+0123456789";
        assert_eq!(inline_diff_height(diff, 24, 4), 2);
    }
}
