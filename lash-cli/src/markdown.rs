use lash_tui::{Line, Modifier, Span, Style};
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::{text_layout, theme};

fn table_column_gap(max_width: usize, col_count: usize) -> usize {
    if col_count <= 1 || max_width < 48 {
        2
    } else {
        3
    }
}

fn spans_text(spans: &[Span<'_>]) -> String {
    let mut text = String::new();
    for span in spans {
        text.push_str(&span.content);
    }
    text
}

fn line_plain_text(line: &Line<'_>) -> String {
    spans_text(&line.spans)
}

fn cell_intrinsic_width(lines: &[Line<'_>]) -> usize {
    let width = lines.iter().map(Line::width).max().unwrap_or(0);
    width.max(1)
}

fn pad_styled_line(line: &Line<'static>, target: usize, fallback_style: Style) -> Line<'static> {
    let mut spans = line.spans.clone();
    let padding = target.saturating_sub(line.width());
    if padding > 0 {
        spans.push(Span::styled(" ".repeat(padding), fallback_style));
    }
    Line::from(spans)
}

fn wrap_styled_table_cell(line: &Line<'static>, width: usize) -> Vec<Line<'static>> {
    if width == 0 {
        return vec![Line::from("")];
    }

    let text = line_plain_text(line);
    if text.is_empty() {
        return vec![Line::from("")];
    }

    let ranges = text_layout::wrap_text_ranges_wordwise(&text, width);
    let mut wrapped = Vec::with_capacity(ranges.len().max(1));
    for (start, end) in ranges {
        wrapped.push(Line::from(text_layout::slice_line_spans(line, start, end)));
    }
    if wrapped.is_empty() {
        vec![Line::from("")]
    } else {
        wrapped
    }
}

fn wrap_styled_cell_lines(lines: &[Line<'static>], width: usize) -> Vec<Line<'static>> {
    let mut wrapped = Vec::new();
    for line in lines {
        wrapped.extend(wrap_styled_table_cell(line, width));
    }
    if wrapped.is_empty() {
        vec![Line::from("")]
    } else {
        wrapped
    }
}

fn apply_base_style(line: &Line<'static>, style: Style) -> Line<'static> {
    Line::from(
        line.spans
            .iter()
            .map(|span| {
                let mut merged = span.style;
                if style.fg.is_some()
                    && (merged.fg.is_none() || merged.fg == theme::assistant_text().fg)
                {
                    merged.fg = style.fg;
                }
                if style.bg.is_some() && merged.bg.is_none() {
                    merged.bg = style.bg;
                }
                if style.bold {
                    merged.bold = true;
                }
                if style.dim {
                    merged.dim = true;
                }
                if style.italic {
                    merged.italic = true;
                }
                if style.underlined {
                    merged.underlined = true;
                }
                Span::styled(span.content.to_string(), merged)
            })
            .collect::<Vec<_>>(),
    )
}

fn expand_column_widths(widths: &[usize], target_content_width: usize) -> Vec<usize> {
    let mut expanded = widths.to_vec();
    let total_base_width: usize = expanded.iter().sum();
    if total_base_width >= target_content_width || expanded.is_empty() {
        return expanded;
    }

    let columns = expanded.len();
    let extra = target_content_width - total_base_width;
    let shared = extra / columns;
    let remainder = extra % columns;

    for (idx, width) in expanded.iter_mut().enumerate() {
        *width += shared;
        if idx < remainder {
            *width += 1;
        }
    }

    expanded
}

fn allocate_shrink_by_weight(shrinkable: &[usize], target_shrink: usize) -> Vec<usize> {
    let mut shrink = vec![0usize; shrinkable.len()];
    if target_shrink == 0 {
        return shrink;
    }

    let weights: Vec<f64> = shrinkable
        .iter()
        .map(|value| {
            if *value == 0 {
                0.0
            } else {
                (*value as f64).sqrt()
            }
        })
        .collect();
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= f64::EPSILON {
        return shrink;
    }

    let mut fractions = vec![0.0f64; shrinkable.len()];
    let mut used = 0usize;

    for idx in 0..shrinkable.len() {
        if shrinkable[idx] == 0 || weights[idx] <= f64::EPSILON {
            continue;
        }
        let exact = weights[idx] / total_weight * target_shrink as f64;
        let whole = shrinkable[idx].min(exact.floor() as usize);
        shrink[idx] = whole;
        fractions[idx] = exact - whole as f64;
        used += whole;
    }

    let mut remaining = target_shrink.saturating_sub(used);
    while remaining > 0 {
        let mut best_idx = None;
        let mut best_fraction = -1.0f64;

        for idx in 0..shrinkable.len() {
            if shrinkable[idx].saturating_sub(shrink[idx]) == 0 {
                continue;
            }
            let fraction = fractions[idx];
            let better = best_idx.is_none()
                || fraction > best_fraction
                || (fraction == best_fraction
                    && shrinkable[idx]
                        > shrinkable[best_idx.expect("best idx should exist when compared")]);
            if better {
                best_idx = Some(idx);
                best_fraction = fraction;
            }
        }

        let Some(idx) = best_idx else {
            break;
        };
        shrink[idx] += 1;
        fractions[idx] = 0.0;
        remaining -= 1;
    }

    shrink
}

fn fit_column_widths_balanced(widths: &[usize], target_content_width: usize) -> Vec<usize> {
    if widths.is_empty() {
        return Vec::new();
    }

    let base_widths: Vec<usize> = widths.iter().map(|width| (*width).max(1)).collect();
    let total_base_width: usize = base_widths.iter().sum();
    if total_base_width <= target_content_width {
        return base_widths;
    }

    let columns = base_widths.len();
    let hard_min_widths = vec![1usize; columns];
    let even_share = (target_content_width / columns).max(1);
    let preferred_min_widths: Vec<usize> = base_widths
        .iter()
        .map(|width| (*width).min(even_share.max(3)))
        .collect();
    let preferred_min_total: usize = preferred_min_widths.iter().sum();
    let floor_widths = if preferred_min_total <= target_content_width {
        preferred_min_widths
    } else {
        hard_min_widths
    };
    let floor_total: usize = floor_widths.iter().sum();
    let clamped_target = floor_total.max(target_content_width);

    if total_base_width <= clamped_target {
        return base_widths;
    }

    let shrinkable: Vec<usize> = base_widths
        .iter()
        .zip(floor_widths.iter())
        .map(|(width, floor)| width.saturating_sub(*floor))
        .collect();
    let total_shrinkable: usize = shrinkable.iter().sum();
    if total_shrinkable == 0 {
        return floor_widths;
    }

    let target_shrink = total_base_width - clamped_target;
    let shrink = allocate_shrink_by_weight(&shrinkable, target_shrink);

    base_widths
        .iter()
        .zip(floor_widths.iter())
        .zip(shrink.iter())
        .map(|((width, floor), shrink)| (*width - *shrink).max(*floor))
        .collect()
}

fn fit_table_column_widths(widths: &[usize], max_width: usize, column_gap: usize) -> Vec<usize> {
    if widths.is_empty() || max_width == 0 {
        return widths.to_vec();
    }

    let total_gap_width = column_gap * widths.len().saturating_sub(1);
    let target_content_width = max_width.saturating_sub(total_gap_width).max(1);
    let current_width: usize = widths.iter().sum();

    if current_width == target_content_width {
        return widths.to_vec();
    }

    if current_width < target_content_width {
        return expand_column_widths(widths, target_content_width);
    }

    fit_column_widths_balanced(widths, target_content_width)
}

/// Parse markdown text and return styled terminal lines.
/// `max_width` constrains table rendering so borders are never wider than the viewport.
pub fn render_markdown(text: &str, max_width: usize) -> Vec<Line<'static>> {
    let mut renderer = MdRenderer::new(max_width);
    let opts = Options::ENABLE_TABLES;
    let parser = Parser::new_ext(text, opts);
    for event in parser {
        renderer.process(event);
    }
    renderer.flush_line();
    wrap_rendered_lines(renderer.lines, max_width)
}

fn is_blank_line(line: &Line<'_>) -> bool {
    line.spans.iter().all(|s| s.content.trim().is_empty())
}

/// Normalize rendered markdown lines for chat-style blocks:
/// - trim leading/trailing blank lines
/// - collapse multiple blank lines to a single blank line
pub fn compact_lines(lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::with_capacity(lines.len());
    let mut prev_blank = true;

    for line in lines {
        let blank = is_blank_line(&line);
        if blank {
            if !prev_blank {
                out.push(Line::from(""));
            }
        } else {
            out.push(line);
        }
        prev_blank = blank;
    }

    while out.last().is_some_and(is_blank_line) {
        out.pop();
    }

    out
}

/// Render markdown and apply chat-style line compaction.
pub fn render_markdown_compact(text: &str, max_width: usize) -> Vec<Line<'static>> {
    compact_lines(render_markdown(text, max_width))
}

struct MdRenderer {
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,
    style_stack: Vec<Style>,
    max_width: usize,
    in_code_block: bool,
    in_item: bool,
    list_stack: Vec<ListContext>,
    // Table buffering: collect all rows, then render with aligned columns
    in_table: bool,
    in_table_head: bool,
    table_rows: Vec<Vec<Vec<Line<'static>>>>,
    table_head: Vec<Vec<Line<'static>>>,
    current_cell_lines: Vec<Line<'static>>,
    current_cell_spans: Vec<Span<'static>>,
}

struct ListContext {
    next_index: Option<usize>,
}

fn wrap_rendered_lines(lines: Vec<Line<'static>>, max_width: usize) -> Vec<Line<'static>> {
    if max_width == 0 {
        return lines;
    }

    let mut wrapped = Vec::with_capacity(lines.len());
    for line in lines {
        if line.width() <= max_width || is_blank_line(&line) {
            wrapped.push(line);
        } else {
            wrapped.extend(text_layout::wrap_styled_line_with_prefix(
                &line,
                max_width,
                markdown_continuation_prefix,
            ));
        }
    }
    wrapped
}

fn markdown_continuation_prefix(line: &Line<'static>) -> Vec<Span<'static>> {
    let text = text_layout::line_text(line);
    let default_style = line
        .spans
        .first()
        .map(|span| span.style)
        .unwrap_or_else(theme::assistant_text);

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

impl MdRenderer {
    fn new(max_width: usize) -> Self {
        Self {
            lines: Vec::new(),
            spans: Vec::new(),
            style_stack: vec![theme::assistant_text()],
            max_width,
            in_code_block: false,
            in_item: false,
            list_stack: Vec::new(),
            in_table: false,
            in_table_head: false,
            table_rows: Vec::new(),
            table_head: Vec::new(),
            current_cell_lines: Vec::new(),
            current_cell_spans: Vec::new(),
        }
    }

    fn current_style(&self) -> Style {
        self.style_stack
            .last()
            .copied()
            .unwrap_or(theme::assistant_text())
    }

    fn push_style(&mut self, style: Style) {
        self.style_stack.push(style);
    }

    fn pop_style(&mut self) {
        if self.style_stack.len() > 1 {
            self.style_stack.pop();
        }
    }

    fn flush_line(&mut self) {
        if !self.spans.is_empty() {
            let spans = std::mem::take(&mut self.spans);
            self.lines.push(Line::from(spans));
        }
    }

    fn blank_line(&mut self) {
        self.lines.push(Line::from(""));
    }

    fn flush_current_cell_line(&mut self) {
        let spans = std::mem::take(&mut self.current_cell_spans);
        self.current_cell_lines.push(Line::from(spans));
    }

    /// Render the buffered table as a wrapped text-table with stable borders.
    fn flush_table(&mut self) {
        let head = std::mem::take(&mut self.table_head);
        let rows = std::mem::take(&mut self.table_rows);

        if head.is_empty() && rows.is_empty() {
            return;
        }

        let col_count = head
            .len()
            .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
        if col_count == 0 {
            return;
        }

        let column_gap = table_column_gap(self.max_width, col_count);
        let mut widths = vec![1usize; col_count];
        for (idx, cell) in head.iter().enumerate() {
            widths[idx] = widths[idx].max(cell_intrinsic_width(cell));
        }
        for row in &rows {
            for (idx, cell) in row.iter().enumerate() {
                widths[idx] = widths[idx].max(cell_intrinsic_width(cell));
            }
        }
        if self.max_width > 0 {
            widths = fit_table_column_widths(&widths, self.max_width, column_gap);
        }

        let header_style = Style::default()
            .fg(theme::brand())
            .add_modifier(Modifier::Bold);
        let rule_style = Style::default().fg(theme::border_dim());
        let cell_style = theme::assistant_text();
        let gap = " ".repeat(column_gap);

        let render_row = |cells: &[Vec<Line<'static>>],
                          style: Style,
                          lines: &mut Vec<Line<'static>>| {
            let wrapped_cells: Vec<Vec<Line<'static>>> = widths
                .iter()
                .enumerate()
                .map(|(idx, width)| {
                    let cell = cells.get(idx).map(Vec::as_slice).unwrap_or(&[]);
                    wrap_styled_cell_lines(cell, *width)
                })
                .collect();
            let row_height = wrapped_cells.iter().map(Vec::len).max().unwrap_or(1);

            for line_idx in 0..row_height {
                let mut spans: Vec<Span<'static>> = Vec::new();
                for (col_idx, width) in widths.iter().enumerate() {
                    let content = wrapped_cells[col_idx]
                        .get(line_idx)
                        .cloned()
                        .unwrap_or_else(|| Line::from(""));
                    let padded = pad_styled_line(&apply_base_style(&content, style), *width, style);
                    spans.extend(padded.spans);
                    if col_idx < widths.len() - 1 {
                        spans.push(Span::raw(gap.clone()));
                    }
                }
                lines.push(Line::from(spans));
            }
        };

        if !head.is_empty() {
            render_row(&head, header_style, &mut self.lines);
            let rule_width =
                widths.iter().sum::<usize>() + column_gap * widths.len().saturating_sub(1);
            self.lines.push(Line::from(Span::styled(
                "\u{2500}".repeat(rule_width),
                rule_style,
            )));
        }

        for row in &rows {
            render_row(row, cell_style, &mut self.lines);
        }
        self.blank_line();
    }

    fn process(&mut self, event: Event<'_>) {
        match event {
            // ── Table events ──
            Event::Start(Tag::Table(_)) => {
                self.flush_line();
                self.in_table = true;
                self.table_head.clear();
                self.table_rows.clear();
            }
            Event::End(TagEnd::Table) => {
                self.in_table = false;
                self.flush_table();
            }
            Event::Start(Tag::TableHead) => {
                self.in_table_head = true;
            }
            Event::End(TagEnd::TableHead) => {
                self.in_table_head = false;
            }
            Event::Start(Tag::TableRow) => {
                if !self.in_table_head {
                    self.table_rows.push(Vec::new());
                }
            }
            Event::End(TagEnd::TableRow) => {}
            Event::Start(Tag::TableCell) => {
                self.current_cell_lines.clear();
                self.current_cell_spans.clear();
            }
            Event::End(TagEnd::TableCell) => {
                if !self.current_cell_spans.is_empty() || self.current_cell_lines.is_empty() {
                    self.flush_current_cell_line();
                }
                let cell = std::mem::take(&mut self.current_cell_lines);
                if self.in_table_head {
                    self.table_head.push(cell);
                } else if let Some(row) = self.table_rows.last_mut() {
                    row.push(cell);
                }
            }

            // ── Heading ──
            // Since headings no longer use brand color, their hierarchy
            // comes from weight + the blank line above and below. compact_lines()
            // collapses runs of blank lines and trims leading blanks, so this
            // stays safe at document boundaries and adjacent to other blocks.
            Event::Start(Tag::Heading { level, .. }) => {
                self.flush_line();
                self.blank_line();
                let style = match level {
                    HeadingLevel::H1 | HeadingLevel::H2 => theme::heading(),
                    _ => theme::subheading(),
                };
                self.push_style(style);
            }
            Event::End(TagEnd::Heading(_)) => {
                self.flush_line();
                self.pop_style();
                self.blank_line();
            }

            // ── Paragraph ──
            Event::Start(Tag::Paragraph) => {}
            Event::End(TagEnd::Paragraph) => {
                if !self.in_table {
                    self.flush_line();
                    if !self.in_item {
                        self.blank_line();
                    }
                }
            }

            // ── Inline formatting ──
            Event::Start(Tag::Strong) => {
                let base = self.current_style();
                self.push_style(base.add_modifier(Modifier::Bold));
            }
            Event::End(TagEnd::Strong) => {
                self.pop_style();
            }
            Event::Start(Tag::Emphasis) => {
                let base = self.current_style();
                self.push_style(base.add_modifier(Modifier::Italic));
            }
            Event::End(TagEnd::Emphasis) => {
                self.pop_style();
            }
            Event::Code(code) => {
                if self.in_table {
                    self.current_cell_spans
                        .push(Span::styled(code.to_string(), theme::inline_code()));
                } else {
                    self.spans
                        .push(Span::styled(code.to_string(), theme::inline_code()));
                }
            }

            // ── Code blocks ──
            Event::Start(Tag::CodeBlock(_)) => {
                self.flush_line();
                self.in_code_block = true;
            }
            Event::End(TagEnd::CodeBlock) => {
                self.in_code_block = false;
                self.blank_line();
            }

            // ── Lists ──
            Event::Start(Tag::List(start)) => {
                self.list_stack.push(ListContext {
                    next_index: start.map(|value| value as usize),
                });
            }
            Event::End(TagEnd::List(_)) => {
                self.list_stack.pop();
                self.blank_line();
            }
            Event::Start(Tag::Item) => {
                self.flush_line();
                self.in_item = true;
                let depth = self.list_stack.len().saturating_sub(1);
                let indent = "  ".repeat(depth);
                let bullet = if depth > 0 { "·" } else { "•" };
                let prefix = match self
                    .list_stack
                    .last_mut()
                    .and_then(|list| list.next_index.as_mut())
                {
                    Some(next_index) => {
                        let current = *next_index;
                        *next_index += 1;
                        format!("{indent}{current}. ")
                    }
                    None => format!("{indent}{bullet} "),
                };
                if depth > 0 {
                    self.push_style(theme::nested_list_item());
                }
                // Emit the list marker eagerly so it survives regardless of
                // which inline event comes first (text, code, bold+code, …).
                // Deferring the prefix to the first Text event used to drop
                // it whenever an item started with inline code or similar.
                self.spans.push(Span::styled(prefix, self.current_style()));
            }
            Event::End(TagEnd::Item) => {
                self.flush_line();
                let depth = self.list_stack.len().saturating_sub(1);
                if depth > 0 {
                    self.pop_style();
                }
                self.in_item = false;
            }

            // ── Text ──
            Event::Text(text) => {
                if self.in_table {
                    self.current_cell_spans
                        .push(Span::styled(text.to_string(), self.current_style()));
                } else if self.in_code_block {
                    for line in text.lines() {
                        self.lines.push(Line::from(vec![
                            Span::styled("\u{2502} ", theme::code_chrome()),
                            Span::styled(line.to_string(), theme::code_content()),
                        ]));
                    }
                } else {
                    self.spans
                        .push(Span::styled(text.to_string(), self.current_style()));
                }
            }
            Event::SoftBreak => {
                if self.in_table {
                    self.flush_current_cell_line();
                } else {
                    // Preserve model-emitted line breaks in chat output instead of
                    // collapsing them into spaces.
                    self.flush_line();
                }
            }
            Event::HardBreak => {
                if self.in_table {
                    self.flush_current_cell_line();
                } else {
                    self.flush_line();
                }
            }

            // ── Horizontal rule (---) ──
            Event::Rule => {
                self.flush_line();
                self.lines.push(Line::from(Span::styled(
                    "\u{2500}".repeat(40),
                    theme::code_chrome(),
                )));
            }

            // ── Links ──
            Event::Start(Tag::Link { .. }) => {}
            Event::End(TagEnd::Link) => {}

            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_width::UnicodeWidthStr;

    fn display_width(text: &str) -> usize {
        UnicodeWidthStr::width(text)
    }

    #[test]
    fn render_plain_text() {
        let lines = render_markdown("hello world", 80);
        // Plain text → paragraph with text + blank line after
        assert!(!lines.is_empty());
        let text: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("hello world"));
    }

    #[test]
    fn render_heading() {
        let lines = render_markdown_compact("# Title", 80);
        // Headings render with surrounding blank lines (space is what gives
        // hierarchy now that brand color is off); compact trims leading and
        // trailing blanks, so a standalone "# Title" collapses to one line.
        assert_eq!(lines.len(), 1);
        let text: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("Title"));
    }

    #[test]
    fn render_bullet_list() {
        let lines = render_markdown("- item one\n- item two", 80);
        let all_text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(all_text.contains("\u{2022}"));
        assert!(all_text.contains("item one"));
        assert!(all_text.contains("item two"));
    }

    #[test]
    fn render_ordered_list_with_nested_bullets_preserves_item_boundaries() {
        let text = "Concise, ordered by likely value for lash:\n\n1. **Session picker / branch navigation as a first-class TUI**\n   - pi has a dedicated `session-picker` and explicit `/tree`, `/resume`, `/fork`, `/session`.\n   - Lash already has resume/fork-ish surfaces, but pi’s approach suggests making session/branch navigation feel like a native browser instead of a command you have to remember.\n   - Worth adapting if you want long-running conversations to feel easier to traverse.\n\n2. **Git-aware footer/status plumbing**\n   - pi has a dedicated `FooterDataProvider` that watches `.git/HEAD` and worktrees cleanly, instead of treating branch info as incidental.\n   - Lash already has a status bar, so the useful adaptation is not \"more chrome,\" it’s **better repo-state fidelity**: branch/worktree awareness, fewer shell-outs, cleaner live updates.\n\n3. **Reusable TUI primitives with explicit overlay/focus model**\n   - pi’s `packages/tui` has a clean model for `Component`, `Focusable`, overlays, focus handoff, cursor markers, and differential rendering.\n   - Lash’s TUI works, but some UX features look more hand-built/special-cased.\n   - Worth adapting as internal architecture, especially if you plan more pickers, popups, menus, or layered tools.\n\n4. **A stronger input editor abstraction**\n   - pi’s editor has explicit undo stack, kill ring, autocomplete plumbing, paste-marker handling, bracketed paste buffering, wrapped layout tracking.\n   - Lash already supports history/paste/images, but if the input box is going to keep growing features, a more editor-like core would pay off.\n   - High leverage if you want fewer ad hoc input-path bugs.";
        let lines = render_markdown_compact(text, 120);
        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        assert!(
            rendered
                .iter()
                .any(|line| line.contains("4.") || line.contains("• A stronger")),
            "rendered lines should preserve the fourth item boundary: {rendered:#?}"
        );
        assert!(
            rendered
                .iter()
                .any(|line| line.contains("3.") || line.contains("• Reusable")),
            "rendered lines should preserve the third item boundary: {rendered:#?}"
        );
        assert!(
            rendered.iter().all(|line| !line.contains("tools.4.")),
            "ordered item boundary collapsed into prior text: {rendered:#?}"
        );
        assert!(
            rendered.iter().all(|line| !line.contains("bugs.5.")),
            "ordered item boundary collapsed into prior text: {rendered:#?}"
        );
    }

    fn rendered_strings(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    /// Input → exact plain-text output golden. Asserts the whole rendered
    /// shape, not just a property — use this when you want to lock in the
    /// visual structure of a markdown fragment.
    #[track_caller]
    fn assert_render_eq(text: &str, max_width: usize, expected: &str) {
        let rendered = render_markdown_compact(text, max_width);
        let actual = rendered_strings(&rendered).join("\n");
        assert_eq!(
            actual, expected,
            "\nINPUT:\n{text}\n\nLINES:\n{rendered:#?}"
        );
    }

    #[test]
    fn golden_ordered_list_with_mixed_inline_spans() {
        // Reproduces the shape from the screenshot bug report: ordered list
        // where some items start with bold text, one starts with bold+code,
        // and one starts with plain text. All five markers must survive.
        let text = "\
1. **Cold parse.** `parse_execute` is slow.
2. **Snapshot restore** costs time.
3. **State churn** shows up too.
4. **`parallel` has a scaling problem.** Branches clone state.
5. The outer profiler is broken.";
        let expected = "\
1. Cold parse. parse_execute is slow.
2. Snapshot restore costs time.
3. State churn shows up too.
4. parallel has a scaling problem. Branches clone state.
5. The outer profiler is broken.";
        assert_render_eq(text, 120, expected);
    }

    #[test]
    fn golden_bullets_with_inline_code_and_bold() {
        let text = "\
- plain bullet
- **bold** bullet
- `code` bullet
- *em* bullet";
        let expected = "\
\u{2022} plain bullet
\u{2022} bold bullet
\u{2022} code bullet
\u{2022} em bullet";
        assert_render_eq(text, 120, expected);
    }

    #[test]
    fn golden_nested_list_preserves_both_levels() {
        let text = "\
1. outer one
   - nested a
   - nested b
2. outer two";
        // `End(List)` always emits a blank line, so the nested sublist
        // ends with a paragraph break before the next top-level item. This
        // is current behavior, not a bug — re-examine if sub-lists start
        // feeling detached from their parent list.
        let expected = "\
1. outer one
  \u{00b7} nested a
  \u{00b7} nested b

2. outer two";
        assert_render_eq(text, 120, expected);
    }

    #[test]
    fn golden_headings_bracket_blank_lines() {
        let text = "Intro\n\n## Heading\n\nBody";
        let expected = "Intro\n\nHeading\n\nBody";
        assert_render_eq(text, 80, expected);
    }

    #[test]
    fn golden_paragraph_then_fenced_code() {
        let text = "Before\n\n```\nfn main() {}\n```\n\nAfter";
        let expected = "Before\n\n\u{2502} fn main() {}\n\nAfter";
        assert_render_eq(text, 80, expected);
    }

    #[test]
    fn ordered_item_starting_with_inline_code_keeps_number_prefix() {
        let text = "1. **`parallel` has a problem.** Each branch clones state.\n\n2. Another item.";
        let rendered = rendered_strings(&render_markdown_compact(text, 120));
        assert!(
            rendered.iter().any(|line| line.starts_with("1. parallel")),
            "ordered item starting with inline code should keep `1. ` prefix: {rendered:#?}"
        );
        assert!(
            rendered.iter().any(|line| line.starts_with("2. Another")),
            "subsequent items should keep their prefix: {rendered:#?}"
        );
    }

    #[test]
    fn ordered_item_starting_with_bare_inline_code_keeps_number_prefix() {
        let text = "1. `foo` bar\n2. `baz` qux";
        let rendered = rendered_strings(&render_markdown_compact(text, 120));
        assert!(
            rendered.iter().any(|line| line.starts_with("1. foo")),
            "item starting with bare inline code should keep `1. ` prefix: {rendered:#?}"
        );
        assert!(
            rendered.iter().any(|line| line.starts_with("2. baz")),
            "second item: {rendered:#?}"
        );
    }

    #[test]
    fn bullet_item_starting_with_inline_code_keeps_bullet_prefix() {
        let text = "- `foo` bar\n- `baz` qux";
        let rendered = rendered_strings(&render_markdown_compact(text, 120));
        assert!(
            rendered.iter().any(|line| line.starts_with("\u{2022} foo")),
            "bulleted item starting with inline code should keep bullet prefix: {rendered:#?}"
        );
        assert!(
            rendered.iter().any(|line| line.starts_with("\u{2022} baz")),
            "second bullet: {rendered:#?}"
        );
    }

    #[test]
    fn render_code_block() {
        let lines = render_markdown("```\nfn main() {}\n```", 80);
        // Code block lines have "│ " prefix
        let has_code_chrome = lines
            .iter()
            .any(|l| l.spans.iter().any(|s| s.content.contains('\u{2502}')));
        assert!(has_code_chrome);
        let all_text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(all_text.contains("fn main()"));
    }

    #[test]
    fn render_bold_and_italic() {
        let lines = render_markdown("**bold** and *italic*", 80);
        // Just verify it doesn't crash and produces output
        assert!(!lines.is_empty());
        let all_text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(all_text.contains("bold"));
        assert!(all_text.contains("italic"));
    }

    #[test]
    fn render_soft_break_as_newline() {
        let lines = render_markdown("line one\nline two", 80);
        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(rendered.iter().any(|line| line.contains("line one")));
        assert!(rendered.iter().any(|line| line.contains("line two")));
        assert!(
            !rendered
                .iter()
                .any(|line| line.contains("line one") && line.contains("line two"))
        );
    }

    #[test]
    fn render_table() {
        let lines = render_markdown("| A | B |\n|---|---|\n| 1 | 2 |", 80);
        let all_text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(all_text.contains("A"));
        assert!(all_text.contains("1"));
        assert!(all_text.contains('\u{2500}'));
    }

    #[test]
    fn render_table_preserves_inline_markdown_styles_in_cells() {
        let lines = render_markdown(
            "| Name | Notes |\n|---|---|\n| **bold** and *italic* | use `inline` code |",
            80,
        );

        let bold_span = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .find(|span| span.content.contains("bold"))
            .expect("bold span in table cell");
        assert!(bold_span.style.bold, "expected bold style: {bold_span:?}");

        let italic_span = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .find(|span| span.content.contains("italic"))
            .expect("italic span in table cell");
        assert!(
            italic_span.style.italic,
            "expected italic style: {italic_span:?}"
        );

        let inline_code_span = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .find(|span| span.content.contains("inline"))
            .expect("inline code span in table cell");
        assert_eq!(inline_code_span.style, theme::inline_code());
    }

    #[test]
    fn render_table_preserves_hard_breaks_inside_cells() {
        let lines = render_markdown(
            "| Col | Value |\n|---|---|\n| key | first line\\
second line |",
            40,
        );

        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        let first_idx = rendered
            .iter()
            .position(|line| line.contains("first line"))
            .expect("first table line");
        let second_idx = rendered
            .iter()
            .position(|line| line.contains("second line"))
            .expect("second table line");
        assert!(
            second_idx > first_idx,
            "cell break should render as a later visual line"
        );
    }

    #[test]
    fn render_table_wraps_long_cells_without_truncating() {
        let lines = render_markdown(
            "| Name | Description |\n|---|---|\n| short | A very long description that should be truncated |",
            30,
        );
        let all_text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(all_text.contains("A very long"));
        assert!(!all_text.contains('\u{2026}'));
        for line in &lines {
            assert!(
                line.width() <= 30,
                "line width {} > 30: {:?}",
                line.width(),
                line
            );
        }
        assert!(
            lines.len() > 5,
            "wrapped table should span multiple visual rows"
        );
    }

    #[test]
    fn render_table_char_wraps_unbroken_tokens() {
        let lines = render_markdown(
            "| Key | Value |\n|---|---|\n| payload | ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 |",
            24,
        );

        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        assert!(rendered.iter().any(|line| line.contains("ABCDEFG")));
        assert!(rendered.iter().any(|line| line.contains("UVWXYZ")));
        assert!(rendered.iter().all(|line| display_width(line) <= 24));
    }

    #[test]
    fn render_table_expands_to_fill_available_width() {
        let lines = render_markdown("| A | B |\n|---|---|\n| 1 | 2 |", 24);
        assert_eq!(lines[0].width(), 24);
        assert_eq!(lines[1].width(), 24);
        assert_eq!(lines[2].width(), 24);
        assert_eq!(lines[3].width(), 0);
    }

    #[test]
    fn render_table_wraps_unicode_cells_with_stable_widths() {
        let lines = render_markdown(
            "| Key | Value |\n|---|---|\n| emoji | こんにちは世界 😀😃😄 mixed with long prose for wrapping |",
            28,
        );
        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        assert!(rendered.iter().any(|line| line.contains("こんにちは")));
        assert!(rendered.iter().all(|line| display_width(line) <= 28));
    }

    #[test]
    fn compact_lines_trims_outer_blank_rows() {
        let lines = vec![
            Line::from(""),
            Line::from("hi"),
            Line::from(""),
            Line::from(""),
        ];
        let compact = compact_lines(lines);
        assert_eq!(compact.len(), 1);
        let text: String = compact[0]
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(text, "hi");
    }

    #[test]
    fn compact_lines_collapses_blank_runs() {
        let lines = vec![
            Line::from("a"),
            Line::from(""),
            Line::from(""),
            Line::from("b"),
        ];
        let compact = compact_lines(lines);
        assert_eq!(compact.len(), 3);
        let text: Vec<String> = compact
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(text, vec!["a".to_string(), "".to_string(), "b".to_string()]);
    }

    #[test]
    fn render_markdown_compact_spaces_headings() {
        let lines = render_markdown_compact("Intro\n\n## Heading\n\nBody", 80);
        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        // Headings must have a blank line above and below. Without brand
        // color, space + weight is the only thing that makes a heading
        // read as a section boundary rather than a slightly bold
        // paragraph line.
        assert_eq!(
            rendered,
            vec![
                "Intro".to_string(),
                "".to_string(),
                "Heading".to_string(),
                "".to_string(),
                "Body".to_string(),
            ]
        );
    }

    #[test]
    fn render_markdown_compact_spaces_rule_and_heading() {
        let lines = render_markdown_compact("Above\n\n---\n\n## Heading\n\nBelow", 80);
        let rendered: Vec<String> = lines
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        // A horizontal rule and a following heading are both section
        // boundaries and each one claims its own breathing room: rule,
        // blank, heading, blank, body. The rule does not steal the
        // heading's leading blank.
        let rule_idx = rendered
            .iter()
            .position(|line| line.contains('\u{2500}'))
            .expect("rule line");
        let heading_idx = rendered
            .iter()
            .position(|line| line == "Heading")
            .expect("heading line");

        assert!(rule_idx < heading_idx);
        assert!(
            rendered[rule_idx + 1..heading_idx]
                .iter()
                .all(|line| line.is_empty()),
            "rule should be followed by blanks before the heading: {rendered:#?}"
        );
        assert_eq!(rendered[heading_idx + 1], "");
        assert_eq!(rendered[heading_idx + 2], "Below");
    }
}
