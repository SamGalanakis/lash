use std::borrow::Cow;
use std::io::{Stdout, Write};

use anyhow::Context;
use crossterm::cursor::{Hide, MoveTo, SetCursorStyle, Show};
use crossterm::event::{
    DisableBracketedPaste, DisableFocusChange, DisableMouseCapture, EnableBracketedPaste,
    EnableFocusChange, EnableMouseCapture, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::style::{
    Attribute, Color as TermColor, Print, ResetColor, SetAttribute, SetBackgroundColor,
    SetForegroundColor,
};
use crossterm::terminal::{
    self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
    enable_raw_mode,
};
use crossterm::{ExecutableCommand, QueueableCommand};
use unicode_width::UnicodeWidthChar;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    fn to_term(self) -> TermColor {
        TermColor::Rgb {
            r: self.r,
            g: self.g,
            b: self.b,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Style {
    pub fg: Option<Color>,
    pub bg: Option<Color>,
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub underlined: bool,
}

impl Style {
    pub const fn reset() -> Self {
        Self {
            fg: None,
            bg: None,
            bold: false,
            dim: false,
            italic: false,
            underlined: false,
        }
    }

    pub const fn fg(mut self, color: Color) -> Self {
        self.fg = Some(color);
        self
    }

    pub const fn bg(mut self, color: Color) -> Self {
        self.bg = Some(color);
        self
    }

    pub const fn add_modifier(mut self, modifier: Modifier) -> Self {
        match modifier {
            Modifier::Bold => self.bold = true,
            Modifier::Dim => self.dim = true,
            Modifier::Italic => self.italic = true,
            Modifier::Underlined => self.underlined = true,
        }
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modifier {
    Bold,
    Dim,
    Italic,
    Underlined,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span<'a> {
    pub content: Cow<'a, str>,
    pub style: Style,
}

impl<'a> Span<'a> {
    pub fn raw(content: impl Into<Cow<'a, str>>) -> Self {
        Self {
            content: content.into(),
            style: Style::default(),
        }
    }

    pub fn styled(content: impl Into<Cow<'a, str>>, style: Style) -> Self {
        Self {
            content: content.into(),
            style,
        }
    }

    pub fn into_owned(self) -> Span<'static> {
        Span {
            content: Cow::Owned(self.content.into_owned()),
            style: self.style,
        }
    }

    pub fn width(&self) -> usize {
        self.content
            .chars()
            .map(|ch| UnicodeWidthChar::width(ch).unwrap_or(0))
            .sum()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Line<'a> {
    pub spans: Vec<Span<'a>>,
}

impl<'a> Line<'a> {
    pub fn into_owned(self) -> Line<'static> {
        Line {
            spans: self.spans.into_iter().map(Span::into_owned).collect(),
        }
    }

    pub fn width(&self) -> usize {
        self.spans.iter().map(Span::width).sum()
    }
}

impl<'a> From<&'a str> for Line<'a> {
    fn from(value: &'a str) -> Self {
        Self {
            spans: vec![Span::raw(value)],
        }
    }
}

impl From<String> for Line<'static> {
    fn from(value: String) -> Self {
        Self {
            spans: vec![Span::raw(value)],
        }
    }
}

impl<'a> From<Span<'a>> for Line<'a> {
    fn from(value: Span<'a>) -> Self {
        Self { spans: vec![value] }
    }
}

impl<'a> From<Vec<Span<'a>>> for Line<'a> {
    fn from(value: Vec<Span<'a>>) -> Self {
        Self { spans: value }
    }
}

impl<'a> std::iter::FromIterator<Span<'a>> for Line<'a> {
    fn from_iter<T: IntoIterator<Item = Span<'a>>>(iter: T) -> Self {
        Self {
            spans: iter.into_iter().collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
}

impl Rect {
    pub const fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub const fn right(self) -> u16 {
        self.x.saturating_add(self.width)
    }

    pub const fn bottom(self) -> u16 {
        self.y.saturating_add(self.height)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    ch: char,
    style: Style,
    continuation: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenCell {
    pub ch: char,
    pub style: Style,
    pub continuation: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            style: Style::default(),
            continuation: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buffer {
    width: u16,
    height: u16,
    cells: Vec<Cell>,
}

impl Buffer {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell::default(); width as usize * height as usize],
        }
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        if self.width == width && self.height == height {
            return;
        }
        *self = Self::new(width, height);
    }

    pub const fn area(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }

    pub fn fill(&mut self, rect: Rect, ch: char, style: Style) {
        let x_end = rect.right().min(self.width);
        let y_end = rect.bottom().min(self.height);
        for y in rect.y..y_end {
            for x in rect.x..x_end {
                self.set_cell(x, y, ch, style);
            }
        }
    }

    pub fn write_span_line(&mut self, x: u16, y: u16, line: &Line<'_>, max_width: u16) {
        let mut cursor_x = x;
        for span in &line.spans {
            for ch in span.content.chars() {
                if cursor_x >= x.saturating_add(max_width) {
                    return;
                }
                cursor_x = self.write_char(cursor_x, y, ch, span.style, x, max_width);
            }
        }
    }

    pub fn write_text(&mut self, x: u16, y: u16, text: &str, style: Style, max_width: u16) {
        let mut cursor_x = x;
        for ch in text.chars() {
            if cursor_x >= x.saturating_add(max_width) {
                break;
            }
            cursor_x = self.write_char(cursor_x, y, ch, style, x, max_width);
        }
    }

    fn write_char(
        &mut self,
        x: u16,
        y: u16,
        ch: char,
        style: Style,
        line_start: u16,
        max_width: u16,
    ) -> u16 {
        let width = UnicodeWidthChar::width(ch).unwrap_or(0) as u16;
        if width == 0 || y >= self.height || x >= self.width {
            return x;
        }
        if width == 1 {
            self.set_cell(x, y, ch, style);
            return x.saturating_add(1);
        }
        if x.saturating_add(width) > line_start.saturating_add(max_width) || x + 1 >= self.width {
            return x;
        }
        self.set_cell(x, y, ch, style);
        if let Some(idx) = self.index(x + 1, y) {
            self.cells[idx] = Cell {
                ch: ' ',
                style,
                continuation: true,
            };
        }
        x.saturating_add(width)
    }

    fn set_cell(&mut self, x: u16, y: u16, ch: char, style: Style) {
        let Some(idx) = self.index(x, y) else {
            return;
        };
        self.cells[idx] = Cell {
            ch,
            style,
            continuation: false,
        };
    }

    fn index(&self, x: u16, y: u16) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(y as usize * self.width as usize + x as usize)
    }

    fn row(&self, y: u16) -> &[Cell] {
        let start = y as usize * self.width as usize;
        let end = start + self.width as usize;
        &self.cells[start..end]
    }

    fn patch_row_style_range<F>(&mut self, x: u16, y: u16, width: u16, mut f: F)
    where
        F: FnMut(Style) -> Style,
    {
        if width == 0 || y >= self.height || x >= self.width {
            return;
        }
        let x_end = x.saturating_add(width).min(self.width);
        let row_start = y as usize * self.width as usize;
        for col in x..x_end {
            let idx = row_start + col as usize;
            let style = self.cells[idx].style;
            self.cells[idx].style = f(style);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScreenSnapshot {
    pub width: u16,
    pub height: u16,
    pub cursor: Option<(u16, u16)>,
    cells: Vec<ScreenCell>,
}

impl ScreenSnapshot {
    pub fn cell(&self, x: u16, y: u16) -> Option<&ScreenCell> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.cells
            .get(y as usize * self.width as usize + x as usize)
    }

    pub fn raw_line(&self, y: u16) -> String {
        (0..self.width)
            .map(|x| self.cell(x, y).map(|cell| cell.ch).unwrap_or(' '))
            .collect()
    }

    pub fn raw_line_trimmed(&self, y: u16) -> String {
        self.raw_line(y).trim_end().to_string()
    }

    pub fn visible_line(&self, y: u16) -> String {
        let mut line = String::new();
        for x in 0..self.width {
            if let Some(cell) = self.cell(x, y)
                && !cell.continuation
            {
                line.push(cell.ch);
            }
        }
        line
    }

    pub fn visible_line_trimmed(&self, y: u16) -> String {
        self.visible_line(y).trim_end().to_string()
    }

    pub fn visible_lines_trimmed(&self) -> Vec<String> {
        (0..self.height)
            .map(|y| self.visible_line_trimmed(y))
            .collect()
    }

    pub fn non_empty_visible_lines(&self) -> Vec<String> {
        self.visible_lines_trimmed()
            .into_iter()
            .filter(|line| !line.is_empty())
            .collect()
    }
}

pub struct Frame<'a> {
    area: Rect,
    buffer: &'a mut Buffer,
    cursor: &'a mut Option<(u16, u16)>,
}

impl<'a> Frame<'a> {
    pub const fn area(&self) -> Rect {
        self.area
    }

    pub fn fill(&mut self, rect: Rect, ch: char, style: Style) {
        self.buffer.fill(rect, ch, style);
    }

    pub fn clear(&mut self, style: Style) {
        self.buffer.fill(self.area, ' ', style);
    }

    pub fn write_line(&mut self, x: u16, y: u16, line: &Line<'_>, max_width: u16) {
        if y >= self.area.height {
            return;
        }
        self.buffer.write_span_line(
            self.area.x.saturating_add(x),
            self.area.y.saturating_add(y),
            line,
            max_width.min(self.area.width.saturating_sub(x)),
        );
    }

    pub fn write_text(&mut self, x: u16, y: u16, text: &str, style: Style, max_width: u16) {
        if y >= self.area.height {
            return;
        }
        self.buffer.write_text(
            self.area.x.saturating_add(x),
            self.area.y.saturating_add(y),
            text,
            style,
            max_width.min(self.area.width.saturating_sub(x)),
        );
    }

    pub fn patch_cell_style<F>(&mut self, x: u16, y: u16, f: F)
    where
        F: FnOnce(Style) -> Style,
    {
        let x = self.area.x.saturating_add(x);
        let y = self.area.y.saturating_add(y);
        let Some(idx) = self.buffer.index(x, y) else {
            return;
        };
        let style = self.buffer.cells[idx].style;
        self.buffer.cells[idx].style = f(style);
    }

    pub fn patch_row_style_range<F>(&mut self, x: u16, y: u16, width: u16, f: F)
    where
        F: FnMut(Style) -> Style,
    {
        if y >= self.area.height {
            return;
        }
        let width = width.min(self.area.width.saturating_sub(x));
        self.buffer.patch_row_style_range(
            self.area.x.saturating_add(x),
            self.area.y.saturating_add(y),
            width,
            f,
        );
    }

    pub fn draw_box(&mut self, rect: Rect, border: Style, fill: Option<Style>) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        if let Some(fill) = fill {
            self.fill(rect, ' ', fill);
        }
        if rect.width == 1 || rect.height == 1 {
            return;
        }
        self.write_text(rect.x, rect.y, "┌", border, 1);
        self.write_text(rect.right() - 1, rect.y, "┐", border, 1);
        self.write_text(rect.x, rect.bottom() - 1, "└", border, 1);
        self.write_text(rect.right() - 1, rect.bottom() - 1, "┘", border, 1);
        for x in rect.x + 1..rect.right() - 1 {
            self.write_text(x, rect.y, "─", border, 1);
            self.write_text(x, rect.bottom() - 1, "─", border, 1);
        }
        for y in rect.y + 1..rect.bottom() - 1 {
            self.write_text(rect.x, y, "│", border, 1);
            self.write_text(rect.right() - 1, y, "│", border, 1);
        }
    }

    pub fn set_cursor_position(&mut self, position: (u16, u16)) {
        *self.cursor = Some((
            self.area.x.saturating_add(position.0),
            self.area.y.saturating_add(position.1),
        ));
    }
}

pub struct Terminal {
    stdout: Stdout,
    front: Buffer,
    back: Buffer,
    cursor: Option<(u16, u16)>,
    entered: bool,
}

impl Terminal {
    pub fn enter() -> anyhow::Result<Self> {
        enable_raw_mode().context("enable raw mode")?;
        let mut stdout = std::io::stdout();
        stdout
            .execute(EnterAlternateScreen)
            .context("enter alternate screen")?;
        stdout.execute(Hide).context("hide cursor")?;
        stdout
            .execute(PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
                    | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES,
            ))
            .ok();
        stdout.execute(EnableBracketedPaste).ok();
        stdout.execute(EnableFocusChange).ok();
        stdout.execute(EnableMouseCapture).ok();
        stdout
            .execute(Print("\x1b]11;rgb:0e/0d/0b\x1b\\"))
            .context("set background")?;
        stdout
            .execute(SetCursorStyle::SteadyBar)
            .context("set cursor style")?;
        let (width, height) = terminal::size().context("read terminal size")?;
        Ok(Self {
            stdout,
            front: Buffer::new(width, height),
            back: Buffer::new(width, height),
            cursor: None,
            entered: true,
        })
    }

    pub fn restore(&mut self) {
        if !self.entered {
            return;
        }
        self.entered = false;
        let _ = self.stdout.execute(PopKeyboardEnhancementFlags);
        let _ = self.stdout.execute(DisableMouseCapture);
        let _ = self.stdout.execute(DisableBracketedPaste);
        let _ = self.stdout.execute(DisableFocusChange);
        let _ = self.stdout.execute(Print("\x1b]111\x1b\\"));
        let _ = self.stdout.execute(SetCursorStyle::DefaultUserShape);
        let _ = self.stdout.execute(Show);
        let _ = self.stdout.execute(LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }

    pub fn size(&self) -> anyhow::Result<(u16, u16)> {
        terminal::size().context("read terminal size")
    }

    pub fn draw<F>(&mut self, render: F) -> anyhow::Result<()>
    where
        F: FnOnce(&mut Frame<'_>),
    {
        let (width, height) = self.size()?;
        let resized = self.front.width != width
            || self.front.height != height
            || self.back.width != width
            || self.back.height != height;
        self.front.resize(width, height);
        if resized {
            self.back.resize(width, height);
        }
        self.cursor = None;
        {
            let area = self.front.area();
            let mut frame = Frame {
                area,
                buffer: &mut self.front,
                cursor: &mut self.cursor,
            };
            frame.clear(Style::default());
            render(&mut frame);
        }
        self.flush_diff(resized)?;
        std::mem::swap(&mut self.front, &mut self.back);
        Ok(())
    }

    fn flush_diff(&mut self, resized: bool) -> anyhow::Result<()> {
        let width = self.front.width;
        let height = self.front.height;
        let mut current_style = None;

        if resized {
            self.stdout.queue(Clear(ClearType::All))?;
        }

        for y in 0..height {
            if self.front.row(y) == self.back.row(y) {
                continue;
            }

            self.stdout.queue(MoveTo(0, y))?;
            for cell in self.front.row(y) {
                if cell.continuation {
                    continue;
                }
                if current_style != Some(cell.style) {
                    queue_style(&mut self.stdout, cell.style)?;
                    current_style = Some(cell.style);
                }
                self.stdout.queue(Print(cell.ch))?;
            }
            if width > 0 {
                self.stdout.queue(ResetColor)?;
                self.stdout.queue(SetAttribute(Attribute::Reset))?;
                current_style = None;
            }
        }

        if let Some((x, y)) = self.cursor {
            self.stdout.queue(MoveTo(x, y))?;
            self.stdout.queue(Show)?;
        } else {
            self.stdout.queue(Hide)?;
        }
        self.stdout.flush()?;
        Ok(())
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        self.restore();
    }
}

pub fn render_snapshot<F>(width: u16, height: u16, render: F) -> ScreenSnapshot
where
    F: FnOnce(&mut Frame<'_>),
{
    let mut buffer = Buffer::new(width, height);
    let mut cursor = None;
    {
        let area = buffer.area();
        let mut frame = Frame {
            area,
            buffer: &mut buffer,
            cursor: &mut cursor,
        };
        frame.clear(Style::default());
        render(&mut frame);
    }

    let cells = buffer
        .cells
        .iter()
        .map(|cell| ScreenCell {
            ch: cell.ch,
            style: cell.style,
            continuation: cell.continuation,
        })
        .collect();

    ScreenSnapshot {
        width,
        height,
        cursor,
        cells,
    }
}

fn queue_style(stdout: &mut Stdout, style: Style) -> anyhow::Result<()> {
    stdout.queue(ResetColor)?;
    stdout.queue(SetAttribute(Attribute::Reset))?;
    if let Some(fg) = style.fg {
        stdout.queue(SetForegroundColor(fg.to_term()))?;
    }
    if let Some(bg) = style.bg {
        stdout.queue(SetBackgroundColor(bg.to_term()))?;
    }
    if style.bold {
        stdout.queue(SetAttribute(Attribute::Bold))?;
    }
    if style.dim {
        stdout.queue(SetAttribute(Attribute::Dim))?;
    }
    if style.italic {
        stdout.queue(SetAttribute(Attribute::Italic))?;
    }
    if style.underlined {
        stdout.queue(SetAttribute(Attribute::Underlined))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_wide_glyphs_without_overflowing_row() {
        let mut buf = Buffer::new(4, 1);
        buf.write_text(0, 0, "好x", Style::default(), 4);
        assert_eq!(buf.row(0)[0].ch, '好');
        assert!(buf.row(0)[1].continuation);
        assert_eq!(buf.row(0)[2].ch, 'x');
    }

    #[test]
    fn style_builder_sets_fields() {
        let style = Style::default()
            .fg(Color::rgb(1, 2, 3))
            .bg(Color::rgb(4, 5, 6))
            .add_modifier(Modifier::Bold)
            .add_modifier(Modifier::Italic);
        assert_eq!(style.fg, Some(Color::rgb(1, 2, 3)));
        assert_eq!(style.bg, Some(Color::rgb(4, 5, 6)));
        assert!(style.bold);
        assert!(style.italic);
    }

    #[test]
    fn patch_row_style_range_updates_contiguous_cells() {
        let snapshot = render_snapshot(6, 1, |frame| {
            frame.write_text(0, 0, "abcdef", Style::default(), 6);
            frame.patch_row_style_range(1, 0, 3, |style| style.bg(Color::rgb(1, 2, 3)));
        });

        assert_eq!(snapshot.cell(0, 0).and_then(|cell| cell.style.bg), None);
        for x in 1..4 {
            assert_eq!(
                snapshot.cell(x, 0).and_then(|cell| cell.style.bg),
                Some(Color::rgb(1, 2, 3))
            );
        }
        assert_eq!(snapshot.cell(4, 0).and_then(|cell| cell.style.bg), None);
    }
}
