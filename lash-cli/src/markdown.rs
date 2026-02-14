use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};
use unicode_width::UnicodeWidthStr;

use crate::theme;

/// Pad a string with trailing spaces to reach a target display width.
fn pad_display(s: &str, target: usize) -> String {
    let w = UnicodeWidthStr::width(s);
    let padding = target.saturating_sub(w);
    format!("{}{}", s, " ".repeat(padding))
}

/// Parse markdown text and return styled ratatui Lines.
/// Used for finalized AssistantText blocks only (not streaming).
pub fn render_markdown(text: &str) -> Vec<Line<'static>> {
    let mut renderer = MdRenderer::new();
    let opts = Options::ENABLE_TABLES;
    let parser = Parser::new_ext(text, opts);
    for event in parser {
        renderer.process(event);
    }
    renderer.flush_line();
    renderer.lines
}

struct MdRenderer {
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,
    style_stack: Vec<Style>,
    in_code_block: bool,
    in_list: bool,
    in_item: bool,
    // Table buffering: collect all rows, then render with aligned columns
    in_table: bool,
    in_table_head: bool,
    table_rows: Vec<Vec<String>>, // rows of cells (text content)
    table_head: Vec<String>,      // header row
    current_cell: String,         // accumulator for current cell text
}

impl MdRenderer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            spans: Vec::new(),
            style_stack: vec![theme::assistant_text()],
            in_code_block: false,
            in_list: false,
            in_item: false,
            in_table: false,
            in_table_head: false,
            table_rows: Vec::new(),
            table_head: Vec::new(),
            current_cell: String::new(),
        }
    }

    fn current_style(&self) -> Style {
        self.style_stack.last().copied().unwrap_or(theme::assistant_text())
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

    /// Render the buffered table as aligned columns with box-drawing borders.
    fn flush_table(&mut self) {
        let head = std::mem::take(&mut self.table_head);
        let rows = std::mem::take(&mut self.table_rows);

        if head.is_empty() && rows.is_empty() {
            return;
        }

        // Compute column count and display widths (using Unicode width, not byte len)
        let col_count = head.len().max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
        let mut widths = vec![0usize; col_count];
        for (i, cell) in head.iter().enumerate() {
            widths[i] = widths[i].max(UnicodeWidthStr::width(cell.as_str()));
        }
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(UnicodeWidthStr::width(cell.as_str()));
            }
        }

        let chrome = theme::code_chrome();
        let head_style = theme::assistant_text().add_modifier(Modifier::BOLD);
        let cell_style = theme::assistant_text();

        // Helper: build a row line with display-width-padded cells
        let build_row = |cells: &[String], style: Style, widths: &[usize]| -> Line<'static> {
            let mut spans: Vec<Span<'static>> = Vec::new();
            spans.push(Span::styled("\u{2502}", chrome)); // │
            for (i, w) in widths.iter().enumerate() {
                let text = cells.get(i).map(|s| s.as_str()).unwrap_or("");
                spans.push(Span::styled(format!(" {} ", pad_display(text, *w)), style));
                spans.push(Span::styled("\u{2502}", chrome)); // │
            }
            Line::from(spans)
        };

        // Helper: build a separator line (─ is 1 display column wide)
        let build_sep = |left: &str, mid: &str, right: &str, widths: &[usize]| -> Line<'static> {
            let mut s = left.to_string();
            for (i, w) in widths.iter().enumerate() {
                // +2 for the space padding on each side of the cell
                for _ in 0..(w + 2) {
                    s.push('\u{2500}'); // ─
                }
                if i < widths.len() - 1 {
                    s.push_str(mid);
                }
            }
            s.push_str(right);
            Line::from(Span::styled(s, chrome))
        };

        // Top border: ┌──┬──┐
        self.lines.push(build_sep("\u{250c}", "\u{252c}", "\u{2510}", &widths));

        // Header row
        if !head.is_empty() {
            self.lines.push(build_row(&head, head_style, &widths));
            // Header separator: ├──┼──┤
            self.lines.push(build_sep("\u{251c}", "\u{253c}", "\u{2524}", &widths));
        }

        // Data rows
        for row in &rows {
            self.lines.push(build_row(row, cell_style, &widths));
        }

        // Bottom border: └──┴──┘
        self.lines.push(build_sep("\u{2514}", "\u{2534}", "\u{2518}", &widths));
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
                    self.blank_line();
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
                    self.spans.push(Span::styled(code.to_string(), theme::inline_code()));
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
            Event::Start(Tag::List(_)) => {
                self.in_list = true;
            }
            Event::End(TagEnd::List(_)) => {
                self.in_list = false;
                self.blank_line();
            }
            Event::Start(Tag::Item) => {
                self.flush_line();
                self.in_item = true;
            }
            Event::End(TagEnd::Item) => {
                self.flush_line();
                self.in_item = false;
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
                } else if self.in_item && self.spans.is_empty() {
                    self.spans.push(Span::styled("  \u{2022} ", self.current_style()));
                    self.spans.push(Span::styled(text.to_string(), self.current_style()));
                } else {
                    self.spans.push(Span::styled(text.to_string(), self.current_style()));
                }
            }
            Event::SoftBreak => {
                if !self.in_table {
                    self.spans.push(Span::raw(" "));
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

/// Count the visual height of rendered markdown, accounting for line wrapping at `width`.
pub fn markdown_height(text: &str, width: usize) -> usize {
    let lines = render_markdown(text);
    if width == 0 {
        return lines.len();
    }
    lines
        .iter()
        .map(|line| {
            let w = line.width();
            if w == 0 {
                1
            } else {
                (w + width - 1) / width
            }
        })
        .sum::<usize>()
        .max(1)
}
