use crate::{Line, Rect, ScrollState, SelectionState, Style, Viewport};

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
}

impl<'a> From<Line<'a>> for TableCell<'a> {
    fn from(content: Line<'a>) -> Self {
        Self {
            content,
            style: Style::default(),
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
        state.scroll.clamp(self.rows.len(), body_height);
        if let Some(selected) = state.selection.selected {
            state
                .scroll
                .ensure_visible(selected, selected + 1, body_height, self.rows.len());
        }

        self.render_header(viewport, &widths);
        if body_height == 0 {
            return;
        }

        for visible_row in 0..body_height {
            let row_index = state.scroll.offset + visible_row;
            if row_index >= self.rows.len() {
                break;
            }
            let y = visible_row as u16 + 1;
            let selected_style = if state.selection.selected == Some(row_index) {
                Some(if state.focused {
                    self.focused_selected_row_style
                } else {
                    self.selected_row_style
                })
            } else {
                None
            };
            self.render_row(viewport, y, &self.rows[row_index], &widths, selected_style);
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

    fn render_row(
        &self,
        viewport: &mut Viewport<'_>,
        y: u16,
        row: &TableRow<'_>,
        widths: &[u16],
        selected_style: Option<Style>,
    ) {
        let area = viewport.area();
        let row_style = self.row_style.merge(row.style);
        let row_style = selected_style.map_or(row_style, |selected| row_style.merge(selected));
        if style_needs_fill(row_style) {
            viewport.fill(Rect::new(0, y, area.width, 1), ' ', row_style);
        }
        let mut x = 0u16;
        for (index, width) in widths.iter().copied().enumerate() {
            if width == 0 {
                continue;
            }
            if let Some(cell) = row.cells.get(index) {
                viewport.write_line_styled(x, y, &cell.content, row_style.merge(cell.style), width);
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

#[cfg(test)]
mod tests {
    use crate::{Color, Line, Style, render_snapshot};

    use super::{Column, ColumnWidth, Table, TableRow, TableState};

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
