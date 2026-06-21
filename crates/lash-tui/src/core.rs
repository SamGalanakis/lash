use std::borrow::Cow;
use std::io::{Stdout, Write};
use std::time::Instant;

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
    self, BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate, EnterAlternateScreen,
    LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use crossterm::{ExecutableCommand, QueueableCommand};
use unicode_width::UnicodeWidthChar;

use crate::input::TermCapabilities;
use crate::prof::{PerfCounters, PerfPhase};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    Rgb { r: u8, g: u8, b: u8 },
    AnsiValue(u8),
    DefaultForeground,
    DefaultBackground,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::Rgb { r, g, b }
    }

    pub const fn ansi(value: u8) -> Self {
        Self::AnsiValue(value)
    }

    pub const fn default_foreground() -> Self {
        Self::DefaultForeground
    }

    pub const fn default_background() -> Self {
        Self::DefaultBackground
    }

    fn to_term(self) -> TermColor {
        match self {
            Self::Rgb { r, g, b } => TermColor::Rgb { r, g, b },
            Self::AnsiValue(value) => TermColor::AnsiValue(value),
            Self::DefaultForeground | Self::DefaultBackground => TermColor::Reset,
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

    pub fn merge(self, overlay: Style) -> Self {
        Self {
            fg: overlay.fg.or(self.fg),
            bg: overlay.bg.or(self.bg),
            bold: self.bold || overlay.bold,
            dim: self.dim || overlay.dim,
            italic: self.italic || overlay.italic,
            underlined: self.underlined || overlay.underlined,
        }
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

    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub fn intersection(self, other: Rect) -> Rect {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if right <= x || bottom <= y {
            return Rect::default();
        }
        Rect::new(x, y, right - x, bottom - y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cell {
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

    pub const fn width(&self) -> u16 {
        self.width
    }

    pub const fn height(&self) -> u16 {
        self.height
    }

    pub fn fill(&mut self, rect: Rect, ch: char, style: Style) {
        let x_end = rect.right().min(self.width);
        let y_end = rect.bottom().min(self.height);
        if rect.x >= x_end || rect.y >= y_end {
            return;
        }
        let fill = Cell {
            ch,
            style,
            continuation: false,
        };
        if rect.x == 0 && x_end == self.width {
            let row_width = self.width as usize;
            let start = rect.y as usize * row_width;
            let end = y_end as usize * row_width;
            self.cells[start..end].fill(fill);
            return;
        }
        let row_width = self.width as usize;
        let x_start = rect.x as usize;
        let x_end = x_end as usize;
        for y in rect.y..y_end {
            let row_start = y as usize * row_width;
            self.cells[row_start + x_start..row_start + x_end].fill(fill);
        }
    }

    pub fn write_span_line(&mut self, x: u16, y: u16, line: &Line<'_>, max_width: u16) {
        self.write_span_line_styled(x, y, line, Style::default(), max_width);
    }

    pub fn write_span_line_styled(
        &mut self,
        x: u16,
        y: u16,
        line: &Line<'_>,
        base_style: Style,
        max_width: u16,
    ) {
        if y >= self.height || x >= self.width || max_width == 0 {
            return;
        }
        let mut cursor_x = x;
        let limit = x.saturating_add(max_width).min(self.width);
        let row_start = y as usize * self.width as usize;
        let row = &mut self.cells[row_start..row_start + self.width as usize];
        for span in &line.spans {
            let style = base_style.merge(span.style);
            for ch in span.content.chars() {
                if cursor_x >= limit {
                    return;
                }
                cursor_x = write_char_to_row(row, self.width, cursor_x, ch, style, limit);
            }
        }
    }

    pub fn write_text(&mut self, x: u16, y: u16, text: &str, style: Style, max_width: u16) {
        if y >= self.height || x >= self.width || max_width == 0 {
            return;
        }
        let mut cursor_x = x;
        let limit = x.saturating_add(max_width).min(self.width);
        let row_start = y as usize * self.width as usize;
        let row = &mut self.cells[row_start..row_start + self.width as usize];
        for ch in text.chars() {
            if cursor_x >= limit {
                break;
            }
            cursor_x = write_char_to_row(row, self.width, cursor_x, ch, style, limit);
        }
    }

    fn index(&self, x: u16, y: u16) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(y as usize * self.width as usize + x as usize)
    }

    pub(crate) fn row(&self, y: u16) -> &[Cell] {
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
        for cell in &mut self.cells[row_start + x as usize..row_start + x_end as usize] {
            cell.style = f(cell.style);
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

    fn row(&self, y: u16) -> &[ScreenCell] {
        let start = y as usize * self.width as usize;
        let end = start + self.width as usize;
        &self.cells[start..end]
    }
}

pub struct Viewport<'a> {
    area: Rect,
    buffer: &'a mut Buffer,
    cursor: &'a mut Option<(u16, u16)>,
}

impl<'a> Viewport<'a> {
    pub const fn area(&self) -> Rect {
        self.area
    }

    pub fn fill(&mut self, rect: Rect, ch: char, style: Style) {
        if rect.is_empty() || self.area.is_empty() {
            return;
        }
        let translated = Rect::new(
            self.area.x.saturating_add(rect.x),
            self.area.y.saturating_add(rect.y),
            rect.width.min(self.area.width.saturating_sub(rect.x)),
            rect.height.min(self.area.height.saturating_sub(rect.y)),
        );
        self.buffer.fill(translated, ch, style);
    }

    pub fn clear(&mut self, style: Style) {
        self.buffer.fill(self.area, ' ', style);
    }

    pub fn write_line(&mut self, x: u16, y: u16, line: &Line<'_>, max_width: u16) {
        self.write_line_styled(x, y, line, Style::default(), max_width);
    }

    pub fn write_line_styled(
        &mut self,
        x: u16,
        y: u16,
        line: &Line<'_>,
        base_style: Style,
        max_width: u16,
    ) {
        if y >= self.area.height {
            return;
        }
        self.buffer.write_span_line_styled(
            self.area.x.saturating_add(x),
            self.area.y.saturating_add(y),
            line,
            base_style,
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

    pub fn viewport(&mut self, rect: Rect) -> Viewport<'_> {
        let relative = Rect::new(
            self.area.x.saturating_add(rect.x),
            self.area.y.saturating_add(rect.y),
            rect.width,
            rect.height,
        );
        let area = self.area.intersection(relative);
        Viewport {
            area,
            buffer: self.buffer,
            cursor: self.cursor,
        }
    }
}

pub type Frame<'a> = Viewport<'a>;

pub struct Terminal {
    stdout: Stdout,
    front: Buffer,
    back: Buffer,
    cursor: Option<(u16, u16)>,
    capabilities: TermCapabilities,
    last_perf: PerfCounters,
    entered: bool,
}

impl Terminal {
    pub fn enter() -> anyhow::Result<Self> {
        enable_raw_mode().context("enable raw mode")?;
        let mut stdout = std::io::stdout();
        stdout
            .execute(EnterAlternateScreen)
            .context("enter alternate screen")?;
        stdout.execute(Print("\x1b]111\x1b\\")).ok();
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
            .execute(SetCursorStyle::SteadyBar)
            .context("set cursor style")?;
        let (width, height) = terminal::size().context("read terminal size")?;
        let keyboard_enhancement = terminal::supports_keyboard_enhancement().unwrap_or(false);
        let synchronized_update = std::env::var("LASH_TUI_SYNC_UPDATE")
            .ok()
            .map(|value| value != "0")
            .unwrap_or_else(|| {
                std::env::var("TERM")
                    .map(|term| term != "dumb")
                    .unwrap_or(true)
            });
        Ok(Self {
            stdout,
            front: Buffer::new(width, height),
            back: Buffer::new(width, height),
            cursor: None,
            capabilities: TermCapabilities {
                bracketed_paste: true,
                focus_change: true,
                mouse_capture: true,
                keyboard_enhancement,
                synchronized_update,
            },
            last_perf: PerfCounters::default(),
            entered: true,
        })
    }

    pub fn set_default_background(&mut self, background: Option<Color>) -> anyhow::Result<()> {
        match background {
            Some(Color::Rgb { r, g, b }) => {
                self.stdout
                    .execute(Print(format!("\x1b]11;rgb:{r:02x}/{g:02x}/{b:02x}\x1b\\")))
                    .context("set terminal default background")?;
            }
            Some(Color::AnsiValue(_) | Color::DefaultForeground | Color::DefaultBackground)
            | None => {
                self.stdout
                    .execute(Print("\x1b]111\x1b\\"))
                    .context("reset terminal default background")?;
            }
        }
        self.stdout
            .flush()
            .context("flush terminal default background")?;
        Ok(())
    }

    pub fn restore(&mut self) {
        if !self.entered {
            return;
        }
        self.entered = false;
        let _ = self.stdout.execute(MoveTo(0, 0));
        let _ = self.stdout.execute(Clear(ClearType::All));
        let _ = self.stdout.execute(ResetColor);
        let _ = self.stdout.execute(SetAttribute(Attribute::Reset));
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

    pub const fn capabilities(&self) -> TermCapabilities {
        self.capabilities
    }

    pub fn last_perf(&self) -> &PerfCounters {
        &self.last_perf
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
        let mut perf = PerfCounters::default();
        {
            let area = self.front.area();
            let mut frame = Viewport {
                area,
                buffer: &mut self.front,
                cursor: &mut self.cursor,
            };
            frame.clear(Style::default());
            let started = Instant::now();
            render(&mut frame);
            perf.record_phase_nanos(PerfPhase::RenderBuild, started.elapsed().as_nanos() as u64);
        }
        self.flush_diff(resized, &mut perf)?;
        self.last_perf = perf;
        std::mem::swap(&mut self.front, &mut self.back);
        Ok(())
    }

    fn flush_diff(&mut self, resized: bool, perf: &mut PerfCounters) -> anyhow::Result<()> {
        let width = self.front.width;
        let height = self.front.height;
        let mut current_style = None;
        let mut queued_bytes = 0u64;
        let mut changed_rows = 0u64;
        let mut changed_cells = 0u64;
        let mut continuation_cells = 0u64;
        let mut wide_glyph_updates = 0u64;

        if self.capabilities.synchronized_update {
            self.stdout.queue(BeginSynchronizedUpdate)?;
            perf.frame.sync_frames = perf.frame.sync_frames.saturating_add(1);
        } else {
            perf.frame.sync_fallback_frames = perf.frame.sync_fallback_frames.saturating_add(1);
        }

        if resized {
            self.stdout.queue(Clear(ClearType::All))?;
        }

        let diff_started = Instant::now();
        for y in 0..height {
            if self.front.row(y) == self.back.row(y) {
                continue;
            }
            changed_rows = changed_rows.saturating_add(1);

            self.stdout.queue(MoveTo(0, y))?;
            for cell in self.front.row(y) {
                if cell.continuation {
                    continuation_cells = continuation_cells.saturating_add(1);
                    continue;
                }
                if current_style != Some(cell.style) {
                    queue_style(&mut self.stdout, cell.style)?;
                    current_style = Some(cell.style);
                }
                self.stdout.queue(Print(cell.ch))?;
                queued_bytes = queued_bytes.saturating_add(cell.ch.len_utf8() as u64);
                changed_cells = changed_cells.saturating_add(1);
                if cell.ch.len_utf8() > 1 {
                    wide_glyph_updates = wide_glyph_updates.saturating_add(1);
                }
            }
            if width > 0 {
                self.stdout.queue(ResetColor)?;
                self.stdout.queue(SetAttribute(Attribute::Reset))?;
                current_style = None;
            }
        }
        perf.record_phase_nanos(
            PerfPhase::DiffScan,
            diff_started.elapsed().as_nanos() as u64,
        );

        let ansi_started = Instant::now();
        if let Some((x, y)) = self.cursor {
            self.stdout.queue(MoveTo(x, y))?;
            self.stdout.queue(Show)?;
        } else {
            self.stdout.queue(Hide)?;
        }
        if self.capabilities.synchronized_update {
            self.stdout.queue(EndSynchronizedUpdate)?;
        }
        perf.record_phase_nanos(
            PerfPhase::AnsiQueue,
            ansi_started.elapsed().as_nanos() as u64,
        );

        let flush_started = Instant::now();
        self.stdout.flush()?;
        perf.record_phase_nanos(
            PerfPhase::FlushSyscall,
            flush_started.elapsed().as_nanos() as u64,
        );
        perf.frame.changed_rows = changed_rows;
        perf.frame.changed_cells = changed_cells;
        perf.frame.bytes_queued = queued_bytes;
        perf.frame.continuation_cells = continuation_cells;
        perf.frame.wide_glyph_updates = wide_glyph_updates;
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
    render_snapshot_with_perf(width, height, None, render).0
}

pub fn render_snapshot_with_perf<F>(
    width: u16,
    height: u16,
    previous: Option<&ScreenSnapshot>,
    render: F,
) -> (ScreenSnapshot, PerfCounters)
where
    F: FnOnce(&mut Frame<'_>),
{
    let mut buffer = Buffer::new(width, height);
    let mut cursor = None;
    let mut perf = PerfCounters::default();
    {
        let area = buffer.area();
        let mut frame = Viewport {
            area,
            buffer: &mut buffer,
            cursor: &mut cursor,
        };
        let started = Instant::now();
        render(&mut frame);
        perf.record_phase_nanos(PerfPhase::RenderBuild, started.elapsed().as_nanos() as u64);
    }
    measure_diff_stats(&buffer, previous, &mut perf);
    (snapshot_from_buffer(buffer, cursor), perf)
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

fn snapshot_from_buffer(buffer: Buffer, cursor: Option<(u16, u16)>) -> ScreenSnapshot {
    let Buffer {
        width,
        height,
        cells,
    } = buffer;
    let cells = cells
        .into_iter()
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

fn measure_diff_stats(front: &Buffer, back: Option<&ScreenSnapshot>, perf: &mut PerfCounters) {
    let diff_started = Instant::now();
    let mut changed_rows = 0u64;
    let mut changed_cells = 0u64;
    let mut continuation_cells = 0u64;
    let mut wide_glyph_updates = 0u64;

    for y in 0..front.height {
        let front_row = front.row(y);
        if row_matches_snapshot(front_row, back, y) {
            continue;
        }
        changed_rows = changed_rows.saturating_add(1);
        for cell in front_row {
            if cell.continuation {
                continuation_cells = continuation_cells.saturating_add(1);
                continue;
            }
            changed_cells = changed_cells.saturating_add(1);
            if cell.ch.len_utf8() > 1 {
                wide_glyph_updates = wide_glyph_updates.saturating_add(1);
            }
        }
    }
    perf.record_phase_nanos(
        PerfPhase::DiffScan,
        diff_started.elapsed().as_nanos() as u64,
    );
    perf.frame.changed_rows = changed_rows;
    perf.frame.changed_cells = changed_cells;
    perf.frame.continuation_cells = continuation_cells;
    perf.frame.wide_glyph_updates = wide_glyph_updates;
}

fn row_matches_snapshot(front_row: &[Cell], back: Option<&ScreenSnapshot>, y: u16) -> bool {
    let Some(snapshot) = back else {
        return false;
    };
    if snapshot.width != front_row.len() as u16 || y >= snapshot.height {
        return false;
    }
    front_row.iter().zip(snapshot.row(y)).all(|(front, back)| {
        front.ch == back.ch && front.style == back.style && front.continuation == back.continuation
    })
}

fn write_char_to_row(
    row: &mut [Cell],
    row_width: u16,
    x: u16,
    ch: char,
    style: Style,
    limit: u16,
) -> u16 {
    if x >= row_width {
        return x;
    }

    if ch.is_ascii() {
        let style = style_preserving_background(row[x as usize].style, style);
        row[x as usize] = Cell {
            ch,
            style,
            continuation: false,
        };
        return x.saturating_add(1);
    }

    let width = UnicodeWidthChar::width(ch).unwrap_or(0) as u16;
    if width == 0 {
        return x;
    }
    if width == 1 {
        let style = style_preserving_background(row[x as usize].style, style);
        row[x as usize] = Cell {
            ch,
            style,
            continuation: false,
        };
        return x.saturating_add(1);
    }
    if x.saturating_add(width) > limit || x + 1 >= row_width {
        return x;
    }

    let primary_style = style_preserving_background(row[x as usize].style, style);
    let continuation_style = style_preserving_background(row[x as usize + 1].style, style);
    row[x as usize] = Cell {
        ch,
        style: primary_style,
        continuation: false,
    };
    row[x as usize + 1] = Cell {
        ch: ' ',
        style: continuation_style,
        continuation: true,
    };
    x.saturating_add(width)
}

fn style_preserving_background(existing: Style, mut next: Style) -> Style {
    if next.bg.is_none() {
        next.bg = existing.bg;
    }
    next
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

    #[test]
    fn write_line_styled_merges_base_and_span_style() {
        let snapshot = render_snapshot(4, 1, |frame| {
            let line = Line::from(vec![Span::styled(
                "x",
                Style::default().fg(Color::rgb(9, 8, 7)),
            )]);
            frame.write_line_styled(0, 0, &line, Style::default().bg(Color::rgb(1, 2, 3)), 4);
        });

        let cell = snapshot.cell(0, 0).expect("cell");
        assert_eq!(cell.ch, 'x');
        assert_eq!(cell.style.fg, Some(Color::rgb(9, 8, 7)));
        assert_eq!(cell.style.bg, Some(Color::rgb(1, 2, 3)));
    }

    #[test]
    fn write_text_preserves_existing_cell_background_when_style_has_none() {
        let snapshot = render_snapshot(4, 1, |frame| {
            frame.fill(frame.area(), ' ', Style::default().bg(Color::rgb(1, 2, 3)));
            frame.write_text(0, 0, "ab", Style::default().fg(Color::rgb(9, 8, 7)), 4);
        });

        let cell = snapshot.cell(0, 0).expect("cell");
        assert_eq!(cell.ch, 'a');
        assert_eq!(cell.style.fg, Some(Color::rgb(9, 8, 7)));
        assert_eq!(cell.style.bg, Some(Color::rgb(1, 2, 3)));
    }

    #[test]
    fn write_text_explicit_background_overrides_existing_cell_background() {
        let snapshot = render_snapshot(4, 1, |frame| {
            frame.fill(frame.area(), ' ', Style::default().bg(Color::rgb(1, 2, 3)));
            frame.write_text(
                0,
                0,
                "ab",
                Style::default()
                    .fg(Color::rgb(9, 8, 7))
                    .bg(Color::rgb(4, 5, 6)),
                4,
            );
        });

        let cell = snapshot.cell(0, 0).expect("cell");
        assert_eq!(cell.ch, 'a');
        assert_eq!(cell.style.fg, Some(Color::rgb(9, 8, 7)));
        assert_eq!(cell.style.bg, Some(Color::rgb(4, 5, 6)));
    }

    #[test]
    fn render_snapshot_with_perf_uses_previous_snapshot_for_diff() {
        let first = render_snapshot(4, 1, |frame| {
            frame.write_text(0, 0, "same", Style::default(), 4);
        });
        let (_, perf) = render_snapshot_with_perf(4, 1, Some(&first), |frame| {
            frame.write_text(0, 0, "same", Style::default(), 4);
        });

        assert_eq!(perf.frame.changed_rows, 0);
        assert_eq!(perf.frame.changed_cells, 0);
    }
}
