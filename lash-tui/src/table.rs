use unicode_width::UnicodeWidthChar;

use crate::{Line, Rect, ScrollState, SelectionState, Span, Style, Viewport};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnWidth {
    Length(u16),
    Min(u16),
    Fill(u16),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Column<'a> {
    pub header: Line<'a>,
    pub width: ColumnWidth,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableCell<'a> {
    pub content: Line<'a>,
    pub style: Style,
    pub wrap: bool,
}

impl<'a> TableCell<'a> {
    pub fn wrapped(mut self) -> Self {
        self.wrap = true;
        self
    }
}

impl<'a> From<Line<'a>> for TableCell<'a> {
    fn from(content: Line<'a>) -> Self {
        Self {
            content,
            style: Style::default(),
            wrap: false,
        }
    }
}

impl<'a> From<&'a str> for TableCell<'a> {
    fn from(content: &'a str) -> Self {
        Self::from(Line::from(content))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableRow<'a> {
    pub cells: Vec<TableCell<'a>>,
    pub style: Style,
}

impl<'a> TableRow<'a> {
    pub fn new(cells: Vec<TableCell<'a>>) -> Self {
        Self {
            cells,
            style: Style::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table<'a> {
    pub columns: Vec<Column<'a>>,
    pub rows: Vec<TableRow<'a>>,
    pub header_style: Style,
    pub row_style: Style,
    pub selected_row_style: Style,
    pub focused_selected_row_style: Style,
}

impl<'a> Table<'a> {
    pub fn render(&self, viewport: &mut Viewport<'_>, state: &mut TableState) {
        let area = viewport.area();
        if area.width == 0 || area.height == 0 || self.columns.is_empty() {
            return;
        }

        let widths = self.column_widths(area.width);
        let body_height = area.height.saturating_sub(1) as usize;
        state.selection.clamp(self.rows.len());
        if self.rows.is_empty() {
            state.scroll.offset = 0;
        } else {
            state.scroll.offset = state.scroll.offset.min(self.rows.len() - 1);
        }
        if body_height > 0 {
            if let Some(selected) = state.selection.selected {
                self.ensure_row_visible(&widths, body_height, selected, &mut state.scroll);
            }
        }

        self.render_header(viewport, &widths);
        if body_height == 0 {
            return;
        }

        let mut row_index = state.scroll.offset;
        let mut y = 1u16;
        while row_index < self.rows.len() && y < area.height {
            let selected_style = if state.selection.selected == Some(row_index) {
                Some(if state.focused {
                    self.focused_selected_row_style
                } else {
                    self.selected_row_style
                })
            } else {
                None
            };
            let prepared = self.prepare_row(&self.rows[row_index], &widths, selected_style);
            let visible_height = prepared.height.min(area.height.saturating_sub(y));
            if visible_height == 0 {
                break;
            }
            self.render_prepared_row(viewport, y, &prepared, &widths, visible_height);
            y = y.saturating_add(visible_height);
            row_index += 1;
        }
    }

    pub fn column_widths(&self, total_width: u16) -> Vec<u16> {
        let mut widths = vec![0u16; self.columns.len()];
        let mut remaining = total_width;

        for (index, column) in self.columns.iter().enumerate() {
            if let ColumnWidth::Length(length) = column.width {
                let assigned = length.min(remaining);
                widths[index] = assigned;
                remaining = remaining.saturating_sub(assigned);
            }
        }

        for (index, column) in self.columns.iter().enumerate() {
            if let ColumnWidth::Min(minimum) = column.width {
                let assigned = minimum.min(remaining);
                widths[index] = assigned;
                remaining = remaining.saturating_sub(assigned);
            }
        }

        let fill_columns = self
            .columns
            .iter()
            .enumerate()
            .filter_map(|(index, column)| match column.width {
                ColumnWidth::Fill(weight) if weight > 0 => Some((index, weight)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if remaining > 0 && !fill_columns.is_empty() {
            let total_weight = fill_columns
                .iter()
                .map(|(_, weight)| *weight as usize)
                .sum::<usize>()
                .max(1);
            let mut assigned = 0u16;
            for (index, weight) in &fill_columns {
                let share = ((remaining as usize) * *weight as usize / total_weight) as u16;
                widths[*index] = widths[*index].saturating_add(share);
                assigned = assigned.saturating_add(share);
            }
            let mut remainder = remaining.saturating_sub(assigned);
            for (index, _) in &fill_columns {
                if remainder == 0 {
                    break;
                }
                widths[*index] = widths[*index].saturating_add(1);
                remainder -= 1;
            }
        }

        widths
    }

    fn render_header(&self, viewport: &mut Viewport<'_>, widths: &[u16]) {
        let area = viewport.area();
        if style_needs_fill(self.header_style) {
            viewport.fill(Rect::new(0, 0, area.width, 1), ' ', self.header_style);
        }
        let mut x = 0u16;
        for (index, width) in widths.iter().copied().enumerate() {
            if width == 0 {
                continue;
            }
            viewport.write_line_styled(x, 0, &self.columns[index].header, self.header_style, width);
            x = x.saturating_add(width);
        }
    }

    fn ensure_row_visible(
        &self,
        widths: &[u16],
        body_height: usize,
        selected: usize,
        scroll: &mut ScrollState,
    ) {
        if self.rows.is_empty() {
            scroll.offset = 0;
            return;
        }
        if selected < scroll.offset {
            scroll.offset = selected;
            return;
        }
        let mut used = 0usize;
        let mut row_index = scroll.offset;
        while row_index < self.rows.len() && used < body_height {
            let row_height = self.row_height(&self.rows[row_index], widths).max(1) as usize;
            if used != 0 && used + row_height > body_height {
                break;
            }
            used = used.saturating_add(row_height);
            row_index += 1;
            if used >= body_height {
                break;
            }
        }
        if selected >= row_index {
            scroll.offset = selected;
            let mut used = self.row_height(&self.rows[selected], widths).max(1) as usize;
            while scroll.offset > 0 {
                let previous_height = self
                    .row_height(&self.rows[scroll.offset - 1], widths)
                    .max(1) as usize;
                if used + previous_height > body_height {
                    break;
                }
                scroll.offset -= 1;
                used += previous_height;
            }
        }
    }

    fn row_height(&self, row: &TableRow<'_>, widths: &[u16]) -> u16 {
        widths
            .iter()
            .enumerate()
            .filter(|(_, width)| **width > 0)
            .map(|(index, width)| {
                row.cells
                    .get(index)
                    .map(|cell| wrap_cell_lines(cell, *width).len().max(1) as u16)
                    .unwrap_or(1)
            })
            .max()
            .unwrap_or(1)
    }

    fn prepare_row(
        &self,
        row: &TableRow<'_>,
        widths: &[u16],
        selected_style: Option<Style>,
    ) -> PreparedRow {
        let row_style = self.row_style.merge(row.style);
        let row_style = selected_style.map_or(row_style, |selected| row_style.merge(selected));
        let cells = widths
            .iter()
            .enumerate()
            .map(|(index, width)| match row.cells.get(index) {
                Some(cell) => PreparedCell {
                    lines: wrap_cell_lines(cell, *width),
                    style: cell.style,
                },
                None => PreparedCell {
                    lines: Vec::new(),
                    style: Style::default(),
                },
            })
            .collect::<Vec<_>>();
        let height = cells
            .iter()
            .map(|cell| cell.lines.len().max(1) as u16)
            .max()
            .unwrap_or(1);
        PreparedRow {
            cells,
            style: row_style,
            height,
        }
    }

    fn render_prepared_row(
        &self,
        viewport: &mut Viewport<'_>,
        y: u16,
        row: &PreparedRow,
        widths: &[u16],
        visible_height: u16,
    ) {
        let area = viewport.area();
        if style_needs_fill(row.style) {
            viewport.fill(Rect::new(0, y, area.width, visible_height), ' ', row.style);
        }
        let mut x = 0u16;
        for (index, width) in widths.iter().copied().enumerate() {
            if width == 0 {
                continue;
            }
            if let Some(cell) = row.cells.get(index) {
                let cell_style = row.style.merge(cell.style);
                if style_needs_fill(cell_style) {
                    viewport.fill(Rect::new(x, y, width, visible_height), ' ', cell_style);
                }
                for (line_index, line) in
                    cell.lines.iter().take(visible_height as usize).enumerate()
                {
                    viewport.write_line_styled(x, y + line_index as u16, line, cell_style, width);
                }
            }
            x = x.saturating_add(width);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TableState {
    pub scroll: ScrollState,
    pub selection: SelectionState,
    pub focused: bool,
}

fn style_needs_fill(style: Style) -> bool {
    style.bg.is_some() || style.bold || style.dim || style.italic || style.underlined
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedCell {
    lines: Vec<Line<'static>>,
    style: Style,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedRow {
    cells: Vec<PreparedCell>,
    style: Style,
    height: u16,
}

fn wrap_cell_lines(cell: &TableCell<'_>, width: u16) -> Vec<Line<'static>> {
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

#[cfg(test)]
mod tests {
    use crate::{Color, Line, Style, render_snapshot};

    use super::{Column, ColumnWidth, Table, TableCell, TableRow, TableState};

    fn sample_table<'a>() -> Table<'a> {
        Table {
            columns: vec![
                Column {
                    header: Line::from("name"),
                    width: ColumnWidth::Length(6),
                },
                Column {
                    header: Line::from("status"),
                    width: ColumnWidth::Fill(1),
                },
            ],
            rows: vec![
                TableRow::new(vec!["alpha".into(), "ok".into()]),
                TableRow::new(vec!["beta".into(), "warn".into()]),
                TableRow::new(vec!["gamma".into(), "busy".into()]),
            ],
            header_style: Style::default().bg(Color::rgb(1, 2, 3)),
            row_style: Style::default(),
            selected_row_style: Style::default().bg(Color::rgb(9, 9, 9)),
            focused_selected_row_style: Style::default().bg(Color::rgb(4, 5, 6)),
        }
    }

    #[test]
    fn table_allocates_fixed_and_fill_columns_predictably() {
        let table = sample_table();
        assert_eq!(table.column_widths(16), vec![6, 10]);
    }

    #[test]
    fn table_renders_header() {
        let table = sample_table();
        let mut state = TableState::default();
        let snapshot = render_snapshot(16, 4, |frame| {
            let mut viewport = frame.viewport(crate::Rect::new(0, 0, 16, 4));
            table.render(&mut viewport, &mut state);
        });
        assert_eq!(snapshot.visible_line_trimmed(0), "name  status");
    }

    #[test]
    fn table_renders_selected_row_style() {
        let table = sample_table();
        let mut state = TableState::default();
        state.selection.select(Some(1));
        state.focused = true;
        let snapshot = render_snapshot(16, 4, |frame| {
            let mut viewport = frame.viewport(crate::Rect::new(0, 0, 16, 4));
            table.render(&mut viewport, &mut state);
        });
        assert_eq!(
            snapshot.cell(0, 2).and_then(|cell| cell.style.bg),
            Some(Color::rgb(4, 5, 6))
        );
    }

    #[test]
    fn table_keeps_header_visible_while_scrolling() {
        let table = sample_table();
        let mut state = TableState::default();
        state.scroll.offset = 1;
        let snapshot = render_snapshot(16, 3, |frame| {
            let mut viewport = frame.viewport(crate::Rect::new(0, 0, 16, 3));
            table.render(&mut viewport, &mut state);
        });
        assert_eq!(snapshot.visible_line_trimmed(0), "name  status");
        assert_eq!(snapshot.visible_line_trimmed(1), "beta  warn");
    }

    #[test]
    fn table_selection_scrolls_into_view() {
        let table = sample_table();
        let mut state = TableState::default();
        state.selection.select(Some(2));
        let _snapshot = render_snapshot(16, 3, |frame| {
            let mut viewport = frame.viewport(crate::Rect::new(0, 0, 16, 3));
            table.render(&mut viewport, &mut state);
        });
        assert_eq!(state.scroll.offset, 1);
    }

    #[test]
    fn table_truncates_cell_content_to_column_width() {
        let table = Table {
            columns: vec![Column {
                header: Line::from("very long"),
                width: ColumnWidth::Length(4),
            }],
            rows: vec![TableRow::new(vec!["abcdefgh".into()])],
            header_style: Style::default(),
            row_style: Style::default(),
            selected_row_style: Style::default(),
            focused_selected_row_style: Style::default(),
        };
        let mut state = TableState::default();
        let snapshot = render_snapshot(4, 2, |frame| {
            let mut viewport = frame.viewport(crate::Rect::new(0, 0, 4, 2));
            table.render(&mut viewport, &mut state);
        });
        assert_eq!(snapshot.visible_line_trimmed(0), "very");
        assert_eq!(snapshot.visible_line_trimmed(1), "abcd");
    }

    #[test]
    fn table_wraps_opt_in_cell_content_to_multiple_lines() {
        let table = Table {
            columns: vec![Column {
                header: Line::from("desc"),
                width: ColumnWidth::Length(4),
            }],
            rows: vec![
                TableRow::new(vec![TableCell::from(Line::from("abcdefgh")).wrapped()]),
                TableRow::new(vec!["done".into()]),
            ],
            header_style: Style::default(),
            row_style: Style::default(),
            selected_row_style: Style::default(),
            focused_selected_row_style: Style::default(),
        };
        let mut state = TableState::default();
        let snapshot = render_snapshot(4, 4, |frame| {
            let mut viewport = frame.viewport(crate::Rect::new(0, 0, 4, 4));
            table.render(&mut viewport, &mut state);
        });
        assert_eq!(snapshot.visible_line_trimmed(0), "desc");
        assert_eq!(snapshot.visible_line_trimmed(1), "abcd");
        assert_eq!(snapshot.visible_line_trimmed(2), "efgh");
        assert_eq!(snapshot.visible_line_trimmed(3), "done");
    }

    #[test]
    fn table_handles_empty_body() {
        let table = Table {
            columns: vec![Column {
                header: Line::from("name"),
                width: ColumnWidth::Fill(1),
            }],
            rows: Vec::new(),
            header_style: Style::default(),
            row_style: Style::default(),
            selected_row_style: Style::default(),
            focused_selected_row_style: Style::default(),
        };
        let mut state = TableState::default();
        let snapshot = render_snapshot(8, 3, |frame| {
            let mut viewport = frame.viewport(crate::Rect::new(0, 0, 8, 3));
            table.render(&mut viewport, &mut state);
        });
        assert_eq!(snapshot.visible_line_trimmed(0), "name");
        assert_eq!(snapshot.visible_line_trimmed(1), "");
    }
}
