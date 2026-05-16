use lash_tui::{Line, Span, Style};
use unicode_width::UnicodeWidthChar;

pub(crate) fn spans_display_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(Span::width).sum()
}

pub(crate) fn line_text(line: &Line<'_>) -> String {
    let mut text = String::new();
    for span in &line.spans {
        text.push_str(&span.content);
    }
    text
}

pub(crate) fn skip_leading_whitespace(text: &str, mut idx: usize) -> usize {
    while idx < text.len() {
        let Some(ch) = text[idx..].chars().next() else {
            break;
        };
        if !ch.is_whitespace() {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

pub(crate) fn trim_trailing_whitespace(text: &str, start: usize, end: usize) -> usize {
    let mut trimmed = end;
    while trimmed > start {
        let Some(ch) = text[..trimmed].chars().next_back() else {
            break;
        };
        if !ch.is_whitespace() {
            break;
        }
        trimmed -= ch.len_utf8();
    }
    trimmed
}

pub(crate) fn wrap_text_ranges_wordwise(text: &str, width: usize) -> Vec<(usize, usize)> {
    wrap_text_ranges_wordwise_prefixed(text, 0, 0, width)
}

pub(crate) fn wrap_text_ranges_wordwise_prefixed(
    text: &str,
    first_prefix_width: usize,
    continuation_prefix_width: usize,
    total_width: usize,
) -> Vec<(usize, usize)> {
    if total_width == 0 || text.is_empty() {
        return vec![(0, text.len())];
    }

    let mut wrapped = Vec::new();
    let mut start = 0usize;
    let mut continuation = false;

    'line: while start < text.len() {
        let line_start = if continuation {
            skip_leading_whitespace(text, start)
        } else {
            start
        };
        if line_start >= text.len() {
            break;
        }

        let cap = if continuation {
            total_width.saturating_sub(continuation_prefix_width).max(1)
        } else {
            total_width.saturating_sub(first_prefix_width).max(1)
        };

        let mut idx = line_start;
        let mut row_width = 0usize;
        let mut last_break = None;
        let mut prev_was_whitespace = false;

        while idx < text.len() {
            let ch = text[idx..]
                .chars()
                .next()
                .expect("slice should start on a char boundary");
            let ch_len = ch.len_utf8();

            if ch == '\n' {
                let line_end = trim_trailing_whitespace(text, line_start, idx);
                wrapped.push((line_start, line_end));
                start = idx + ch_len;
                continuation = false;
                continue 'line;
            }

            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);

            if row_width + ch_width > cap {
                let break_idx = last_break.unwrap_or(idx);
                let line_end = trim_trailing_whitespace(text, line_start, break_idx);
                wrapped.push((line_start, line_end));
                start = if last_break.is_some() {
                    skip_leading_whitespace(text, break_idx)
                } else {
                    break_idx
                };
                continuation = true;
                continue 'line;
            }

            row_width += ch_width;

            let is_whitespace = ch.is_whitespace();
            if is_whitespace || prev_was_whitespace {
                last_break = Some(idx);
            }
            prev_was_whitespace = is_whitespace;
            idx += ch_len;
        }

        let line_end = trim_trailing_whitespace(text, line_start, text.len());
        wrapped.push((line_start, line_end));
        break;
    }

    if wrapped.is_empty() {
        vec![(0, text.len())]
    } else {
        wrapped
    }
}

pub(crate) fn slice_line_spans(
    line: &Line<'static>,
    start: usize,
    end: usize,
) -> Vec<Span<'static>> {
    if start >= end {
        return Vec::new();
    }

    let mut spans = Vec::new();
    let mut idx = 0usize;
    for span in &line.spans {
        let span_end = idx + span.content.len();
        if span_end <= start {
            idx = span_end;
            continue;
        }
        if idx >= end {
            break;
        }
        let slice_start = start.saturating_sub(idx);
        let slice_end = (end - idx).min(span.content.len());
        if slice_start < slice_end {
            spans.push(Span::styled(
                span.content[slice_start..slice_end].to_string(),
                span.style,
            ));
        }
        idx = span_end;
    }
    spans
}

pub(crate) fn wrap_styled_line_with_prefix<F>(
    line: &Line<'static>,
    max_width: usize,
    continuation_prefix: F,
) -> Vec<Line<'static>>
where
    F: Fn(&Line<'static>) -> Vec<Span<'static>>,
{
    if max_width == 0 {
        return vec![Line::from("")];
    }

    let text = line_text(line);
    let prefix = continuation_prefix(line);
    let prefix_width = spans_display_width(&prefix);
    let ranges = wrap_text_ranges_wordwise_prefixed(&text, 0, prefix_width, max_width.max(1));
    let mut wrapped = Vec::with_capacity(ranges.len().max(1));

    for (idx, (start, end)) in ranges.into_iter().enumerate() {
        let mut spans = Vec::new();
        if idx > 0 && !prefix.is_empty() {
            spans.extend(prefix.iter().cloned());
        }
        spans.extend(slice_line_spans(line, start, end));
        wrapped.push(Line::from(spans));
    }

    if wrapped.is_empty() {
        vec![Line::from("")]
    } else {
        wrapped
    }
}

pub(crate) fn continuation_prefix_style_from_line(line: &Line<'static>) -> Style {
    line.spans
        .first()
        .map(|span| span.style)
        .unwrap_or_default()
}
