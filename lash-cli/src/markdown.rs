use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::theme;

/// Pad a string with trailing spaces to reach a target display width.
fn pad_display(s: &str, target: usize) -> String {
    let w = UnicodeWidthStr::width(s);
    let padding = target.saturating_sub(w);
    format!("{}{}", s, " ".repeat(padding))
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn table_column_gap(max_width: usize, col_count: usize) -> usize {
    if col_count <= 1 || max_width < 48 {
        2
    } else {
        3
    }
}

fn cell_intrinsic_width(text: &str) -> usize {
    let width = text.lines().map(display_width).max().unwrap_or(0);
    width.max(1)
}

fn wrap_token_chars(token: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0;

    for ch in token.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if ch_width > width {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
                current_width = 0;
            }
            continue;
        }
        if current_width + ch_width > width && !current.is_empty() {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(ch);
        current_width += ch_width;
    }

    if !current.is_empty() {
        lines.push(current);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

fn wrap_text_to_width(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    let mut out = Vec::new();
    for raw_line in text.lines() {
        if raw_line.trim().is_empty() {
            out.push(String::new());
            continue;
        }

        let mut current = String::new();
        let mut current_width = 0;

        for token in raw_line.split_whitespace() {
            let token_width = display_width(token);
            if current.is_empty() {
                if token_width <= width {
                    current.push_str(token);
                    current_width = token_width;
                } else {
                    out.extend(wrap_token_chars(token, width));
                }
                continue;
            }

            if current_width + 1 + token_width <= width {
                current.push(' ');
                current.push_str(token);
                current_width += 1 + token_width;
                continue;
            }

            out.push(std::mem::take(&mut current));
            current_width = 0;

            if token_width <= width {
                current.push_str(token);
                current_width = token_width;
            } else {
                out.extend(wrap_token_chars(token, width));
            }
        }

        if !current.is_empty() {
            out.push(current);
        }
    }

    if out.is_empty() {
        out.push(String::new());
    }

    out
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

/// Parse markdown text and return styled ratatui Lines.
/// `max_width` constrains table rendering so borders are never wider than the viewport.
pub fn render_markdown(text: &str, max_width: usize) -> Vec<Line<'static>> {
    let mut renderer = MdRenderer::new(max_width);
    let opts = Options::ENABLE_TABLES;
    let parser = Parser::new_ext(text, opts);
    for event in parser {
        renderer.process(event);
    }
    renderer.flush_line();
    renderer.lines
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
    pending_item_prefix: Option<String>,
    // Table buffering: collect all rows, then render with aligned columns
    in_table: bool,
    in_table_head: bool,
    table_rows: Vec<Vec<String>>, // rows of cells (text content)
    table_head: Vec<String>,      // header row
    current_cell: String,         // accumulator for current cell text
}

struct ListContext {
    next_index: Option<usize>,
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
            pending_item_prefix: None,
            in_table: false,
            in_table_head: false,
            table_rows: Vec::new(),
            table_head: Vec::new(),
            current_cell: String::new(),
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

    fn push_pending_item_prefix(&mut self) {
        if !self.in_item || !self.spans.is_empty() {
            return;
        }
        let Some(prefix) = self.pending_item_prefix.take() else {
            return;
        };
        self.spans.push(Span::styled(prefix, self.current_style()));
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
            .fg(theme::SODIUM)
            .add_modifier(Modifier::BOLD);
        let rule_style = Style::default().fg(theme::ASH_MID);
        let cell_style = theme::assistant_text();
        let gap = " ".repeat(column_gap);

        let render_row = |cells: &[String], style: Style, lines: &mut Vec<Line<'static>>| {
            let wrapped_cells: Vec<Vec<String>> = widths
                .iter()
                .enumerate()
                .map(|(idx, width)| {
                    let text = cells.get(idx).map(|value| value.as_str()).unwrap_or("");
                    wrap_text_to_width(text, *width)
                })
                .collect();
            let row_height = wrapped_cells.iter().map(Vec::len).max().unwrap_or(1);

            for line_idx in 0..row_height {
                let mut spans: Vec<Span<'static>> = Vec::new();
                for (col_idx, width) in widths.iter().enumerate() {
                    let content = wrapped_cells[col_idx]
                        .get(line_idx)
                        .map(String::as_str)
                        .unwrap_or("");
                    spans.push(Span::styled(pad_display(content, *width), style));
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
                self.current_cell.clear();
            }
            Event::End(TagEnd::TableCell) => {
                let cell = std::mem::take(&mut self.current_cell);
                if self.in_table_head {
                    self.table_head.push(cell);
                } else if let Some(row) = self.table_rows.last_mut() {
                    row.push(cell);
                }
            }

            // ── Heading ──
            Event::Start(Tag::Heading { .. }) => {
                self.flush_line();
                // Breathing room above headings so they don't stack against prior content
                if !self.lines.is_empty() && !self.lines.last().is_some_and(is_blank_line) {
                    self.blank_line();
                }
                self.push_style(theme::heading());
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
                self.push_style(base.add_modifier(Modifier::BOLD));
            }
            Event::End(TagEnd::Strong) => {
                self.pop_style();
            }
            Event::Start(Tag::Emphasis) => {
                let base = self.current_style();
                self.push_style(base.add_modifier(Modifier::ITALIC));
            }
            Event::End(TagEnd::Emphasis) => {
                self.pop_style();
            }
            Event::Code(code) => {
                if self.in_table {
                    self.current_cell.push_str(&code);
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
                    None => format!("{indent}• "),
                };
                self.pending_item_prefix = Some(prefix);
            }
            Event::End(TagEnd::Item) => {
                self.flush_line();
                self.in_item = false;
                self.pending_item_prefix = None;
            }

            // ── Text ──
            Event::Text(text) => {
                if self.in_table {
                    self.current_cell.push_str(&text);
                } else if self.in_code_block {
                    for line in text.lines() {
                        self.lines.push(Line::from(vec![
                            Span::styled("\u{2502} ", theme::code_chrome()),
                            Span::styled(line.to_string(), theme::code_content()),
                        ]));
                    }
                } else {
                    self.push_pending_item_prefix();
                    self.spans
                        .push(Span::styled(text.to_string(), self.current_style()));
                }
            }
            Event::SoftBreak => {
                if !self.in_table {
                    // Preserve model-emitted line breaks in chat output instead of
                    // collapsing them into spaces.
                    self.flush_line();
                }
            }
            Event::HardBreak => {
                self.flush_line();
            }

            // ── Horizontal rule (---) ──
            Event::Rule => {
                self.flush_line();
                self.lines.push(Line::from(Span::styled(
                    "\u{2500}".repeat(40),
                    theme::code_chrome(),
                )));
                self.blank_line();
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
        let lines = render_markdown("# Title", 80);
        // Should have a bold styled line and a blank line
        assert!(lines.len() >= 2);
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
}
