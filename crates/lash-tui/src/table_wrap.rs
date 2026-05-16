use unicode_width::UnicodeWidthChar;

use crate::table::TableCell;
use crate::{Line, Span, Style};

pub(crate) fn wrap_cell_lines(cell: &TableCell<'_>, width: u16) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }
    if !cell.wrap {
        return vec![cell.content.clone().into_owned()];
    }
    wrap_line(&cell.content, width)
}

fn wrap_line(line: &Line<'_>, width: u16) -> Vec<Line<'static>> {
    let max_width = width as usize;
    if max_width == 0 {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut current_spans: Vec<Span<'static>> = Vec::new();
    let mut current_text = String::new();
    let mut current_style = Style::default();
    let mut current_style_set = false;
    let mut current_width = 0usize;

    let flush_span = |spans: &mut Vec<Span<'static>>,
                      text: &mut String,
                      style: &mut Style,
                      style_set: &mut bool| {
        if text.is_empty() {
            return;
        }
        spans.push(Span::styled(std::mem::take(text), *style));
        *style_set = false;
    };

    let push_line = |lines: &mut Vec<Line<'static>>, spans: &mut Vec<Span<'static>>| {
        lines.push(Line::from(std::mem::take(spans)));
    };

    for span in &line.spans {
        for ch in span.content.chars() {
            if ch == '\n' {
                flush_span(
                    &mut current_spans,
                    &mut current_text,
                    &mut current_style,
                    &mut current_style_set,
                );
                push_line(&mut lines, &mut current_spans);
                current_width = 0;
                continue;
            }

            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if current_width > 0 && ch_width > 0 && current_width + ch_width > max_width {
                flush_span(
                    &mut current_spans,
                    &mut current_text,
                    &mut current_style,
                    &mut current_style_set,
                );
                push_line(&mut lines, &mut current_spans);
                current_width = 0;
            }

            if current_style_set && current_style != span.style {
                flush_span(
                    &mut current_spans,
                    &mut current_text,
                    &mut current_style,
                    &mut current_style_set,
                );
            }

            if !current_style_set {
                current_style = span.style;
                current_style_set = true;
            }

            current_text.push(ch);
            current_width += ch_width;
        }
    }

    flush_span(
        &mut current_spans,
        &mut current_text,
        &mut current_style,
        &mut current_style_set,
    );
    if !current_spans.is_empty() || lines.is_empty() {
        push_line(&mut lines, &mut current_spans);
    }
    lines
}
