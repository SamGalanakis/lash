use std::mem;

use lash_tui::Line;

use crate::assistant_text::{MarkdownLane, render_live_markdown_documents};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenBlockMode {
    Paragraph,
    List,
    BlockQuote,
    CodeFence { fence_char: char, fence_len: usize },
    Table,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OpenBlock {
    mode: OpenBlockMode,
    lines: Vec<String>,
    pending_blank: bool,
}

impl OpenBlock {
    fn new(mode: OpenBlockMode, first_line: String) -> Self {
        Self {
            mode,
            lines: vec![first_line],
            pending_blank: false,
        }
    }

    fn push_line(&mut self, line: String) {
        if self.pending_blank {
            self.lines.push(String::new());
            self.pending_blank = false;
        }
        self.lines.push(line);
    }

    fn note_blank_separator(&mut self) {
        self.pending_blank = true;
    }

    fn text_with_partial(&self, partial_line: &str) -> String {
        let mut parts = self.lines.clone();
        if !partial_line.is_empty() {
            if self.pending_blank {
                parts.push(String::new());
            }
            parts.push(partial_line.to_string());
        }
        parts.join("\n")
    }

    fn finish(self) -> String {
        self.lines.join("\n")
    }
}

#[derive(Clone, Debug)]
pub struct LiveMarkdown {
    lane: MarkdownLane,
    committed: Vec<String>,
    open: Option<OpenBlock>,
    partial_line: String,
    render_width: usize,
    rendered_lines: Vec<Line<'static>>,
    dirty: bool,
    has_visible_text: bool,
}

impl LiveMarkdown {
    pub fn new(lane: MarkdownLane) -> Self {
        Self {
            lane,
            committed: Vec::new(),
            open: None,
            partial_line: String::new(),
            render_width: 0,
            rendered_lines: Vec::new(),
            dirty: true,
            has_visible_text: false,
        }
    }

    pub fn clear(&mut self) {
        self.committed.clear();
        self.open = None;
        self.partial_line.clear();
        self.render_width = 0;
        self.rendered_lines.clear();
        self.dirty = true;
        self.has_visible_text = false;
    }

    pub fn append(&mut self, chunk: &str) {
        if chunk.is_empty() {
            return;
        }
        self.has_visible_text |= chunk.chars().any(|ch| !ch.is_whitespace());
        let normalized = normalize_newlines(chunk);
        for segment in normalized.split_inclusive('\n') {
            if let Some(line) = segment.strip_suffix('\n') {
                self.partial_line.push_str(line);
                let complete = mem::take(&mut self.partial_line);
                self.push_complete_line(complete);
            } else {
                self.partial_line.push_str(segment);
            }
        }
        self.dirty = true;
    }

    pub fn ensure_rendered(&mut self, viewport_width: usize) {
        if !self.dirty && self.render_width == viewport_width {
            return;
        }

        let mut docs = self.committed.clone();
        if let Some(pending) = self.pending_display_text() {
            docs.push(pending);
        }
        self.rendered_lines = render_live_markdown_documents(
            self.lane,
            docs.iter().map(String::as_str),
            viewport_width,
        );
        self.render_width = viewport_width;
        self.dirty = false;
    }

    pub fn lines(&self) -> &[Line<'static>] {
        &self.rendered_lines
    }

    pub fn has_renderable_output(&self) -> bool {
        self.has_visible_text
    }

    pub fn normalized_text(&self) -> Option<String> {
        let mut docs = self.committed.clone();
        if let Some(pending) = self.pending_raw_text() {
            docs.push(pending);
        }
        if docs.is_empty() {
            None
        } else {
            Some(docs.join("\n\n"))
        }
    }

    pub fn take_normalized_text(&mut self) -> Option<String> {
        let text = self.normalized_text();
        self.clear();
        text
    }

    #[cfg(test)]
    pub(crate) fn committed_blocks(&self) -> &[String] {
        &self.committed
    }

    #[cfg(test)]
    pub(crate) fn pending_raw_text_for_test(&self) -> Option<String> {
        self.pending_raw_text()
    }

    fn push_complete_line(&mut self, line: String) {
        if line.trim().is_empty() {
            match self.open.as_mut() {
                Some(
                    open @ OpenBlock {
                        mode: OpenBlockMode::List | OpenBlockMode::BlockQuote,
                        ..
                    },
                ) => {
                    open.note_blank_separator();
                }
                Some(_) => {
                    self.commit_open_block();
                }
                None => {}
            }
            return;
        }

        match self.open.as_mut() {
            Some(open) => match open.mode {
                OpenBlockMode::CodeFence {
                    fence_char,
                    fence_len,
                } => {
                    open.push_line(line.clone());
                    if fence_end(&line, fence_char, fence_len) {
                        self.commit_open_block();
                    }
                }
                OpenBlockMode::Paragraph => {
                    if open.lines.len() == 1
                        && looks_like_table_row(&open.lines[0])
                        && is_table_separator_line(&line)
                    {
                        open.mode = OpenBlockMode::Table;
                        open.push_line(line);
                    } else if starts_new_block(&line) {
                        self.commit_open_block();
                        self.start_or_commit_block(line);
                    } else {
                        open.push_line(line);
                    }
                }
                OpenBlockMode::List => {
                    if is_list_continuation(&line) || is_list_item_start(&line) {
                        open.push_line(line);
                    } else {
                        self.commit_open_block();
                        self.start_or_commit_block(line);
                    }
                }
                OpenBlockMode::BlockQuote => {
                    if is_blockquote_start(&line) {
                        open.push_line(line);
                    } else {
                        self.commit_open_block();
                        self.start_or_commit_block(line);
                    }
                }
                OpenBlockMode::Table => {
                    if looks_like_table_row(&line) || is_table_separator_line(&line) {
                        open.push_line(line);
                    } else {
                        self.commit_open_block();
                        self.start_or_commit_block(line);
                    }
                }
            },
            None => self.start_or_commit_block(line),
        }
    }

    fn start_or_commit_block(&mut self, line: String) {
        if let Some((fence_char, fence_len)) = fence_start(&line) {
            self.open = Some(OpenBlock::new(
                OpenBlockMode::CodeFence {
                    fence_char,
                    fence_len,
                },
                line,
            ));
            return;
        }
        if is_heading(&line) || is_thematic_break(&line) {
            self.committed.push(line);
            return;
        }
        if is_list_item_start(&line) {
            self.open = Some(OpenBlock::new(OpenBlockMode::List, line));
            return;
        }
        if is_blockquote_start(&line) {
            self.open = Some(OpenBlock::new(OpenBlockMode::BlockQuote, line));
            return;
        }
        self.open = Some(OpenBlock::new(OpenBlockMode::Paragraph, line));
    }

    fn commit_open_block(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        let text = open.finish();
        if !text.trim().is_empty() {
            self.committed.push(text);
        }
    }

    fn pending_raw_text(&self) -> Option<String> {
        match (&self.open, self.partial_line.trim()) {
            (Some(open), partial) => {
                let text = open.text_with_partial(partial);
                (!text.trim().is_empty()).then_some(text)
            }
            (None, "") => None,
            (None, partial) => Some(partial.to_string()),
        }
    }

    fn pending_display_text(&self) -> Option<String> {
        let raw = self.pending_raw_text()?;
        let mode = self.pending_mode()?;
        Some(match mode {
            OpenBlockMode::CodeFence {
                fence_char,
                fence_len,
            } => {
                let mut display = raw;
                if !display.ends_with('\n') {
                    display.push('\n');
                }
                for _ in 0..fence_len {
                    display.push(fence_char);
                }
                display
            }
            _ => raw,
        })
    }

    fn pending_mode(&self) -> Option<OpenBlockMode> {
        if let Some(open) = &self.open {
            return Some(open.mode);
        }
        infer_partial_mode(self.partial_line.trim())
    }
}

fn normalize_newlines(chunk: &str) -> String {
    let mut out = String::with_capacity(chunk.len());
    let mut chars = chunk.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\r' if matches!(chars.peek(), Some('\n')) => {
                chars.next();
                out.push('\n');
            }
            '\r' => out.push('\n'),
            _ => out.push(ch),
        }
    }
    out
}

fn infer_partial_mode(line: &str) -> Option<OpenBlockMode> {
    if line.is_empty() {
        return None;
    }
    if let Some((fence_char, fence_len)) = fence_start(line) {
        return Some(OpenBlockMode::CodeFence {
            fence_char,
            fence_len,
        });
    }
    if is_list_item_start(line) || is_list_item_start_prefix(line) {
        return Some(OpenBlockMode::List);
    }
    if is_blockquote_start(line) {
        return Some(OpenBlockMode::BlockQuote);
    }
    Some(OpenBlockMode::Paragraph)
}

fn starts_new_block(line: &str) -> bool {
    fence_start(line).is_some()
        || is_heading(line)
        || is_thematic_break(line)
        || is_list_item_start(line)
        || is_blockquote_start(line)
}

fn is_heading(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with('#') && trimmed[1..].starts_with([' ', '\t', '#'])
}

fn thematic_break_char(line: &str) -> Option<char> {
    let mut s = line;
    let mut spaces = 0usize;
    while spaces < 3 && s.starts_with(' ') {
        s = &s[1..];
        spaces += 1;
    }
    let s = s.trim_end_matches([' ', '\t']);
    let mut it = s.chars();
    let first = it.next()?;
    if first != '-' && first != '*' && first != '_' {
        return None;
    }
    let mut count = 1usize;
    for c in it {
        if c == first {
            count += 1;
            continue;
        }
        if c == ' ' || c == '\t' {
            continue;
        }
        return None;
    }
    (count >= 3).then_some(first)
}

fn is_thematic_break(line: &str) -> bool {
    thematic_break_char(line).is_some()
}

fn fence_start(line: &str) -> Option<(char, usize)> {
    let mut s = line;
    let mut spaces = 0usize;
    while spaces < 3 && s.starts_with(' ') {
        s = &s[1..];
        spaces += 1;
    }
    let bytes = s.as_bytes();
    if bytes.len() < 3 {
        return None;
    }
    let ch = bytes[0] as char;
    if ch != '`' && ch != '~' {
        return None;
    }
    let mut len = 0usize;
    while len < bytes.len() && bytes[len] == bytes[0] {
        len += 1;
    }
    (len >= 3).then_some((ch, len))
}

fn fence_end(line: &str, fence_char: char, fence_len: usize) -> bool {
    let mut s = line;
    let mut spaces = 0usize;
    while spaces < 3 && s.starts_with(' ') {
        s = &s[1..];
        spaces += 1;
    }
    let trimmed = s.trim_end();
    trimmed.chars().all(|ch| ch == fence_char) && trimmed.chars().count() >= fence_len
}

fn is_blockquote_start(line: &str) -> bool {
    line.trim_start().starts_with('>')
}

fn is_list_item_start(line: &str) -> bool {
    let s = line.trim_start();
    if s.len() < 2 {
        return false;
    }
    let bytes = s.as_bytes();
    match bytes[0] {
        b'-' | b'+' | b'*' => bytes[1] == b' ' || bytes[1] == b'\t',
        b'0'..=b'9' => {
            let mut i = 0usize;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            i > 0
                && i + 1 < bytes.len()
                && (bytes[i] == b'.' || bytes[i] == b')')
                && (bytes[i + 1] == b' ' || bytes[i + 1] == b'\t')
        }
        _ => false,
    }
}

fn is_list_item_start_prefix(line: &str) -> bool {
    let s = line.trim_start();
    if s.is_empty() {
        return false;
    }
    let bytes = s.as_bytes();
    match bytes[0] {
        b'-' | b'+' | b'*' => s.len() == 1,
        b'0'..=b'9' => {
            let mut i = 0usize;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i == 0 {
                return false;
            }
            if i == bytes.len() {
                return true;
            }
            (bytes[i] == b'.' || bytes[i] == b')') && i + 1 == bytes.len()
        }
        _ => false,
    }
}

fn is_list_continuation(line: &str) -> bool {
    if is_list_item_start(line) {
        return true;
    }
    let bytes = line.as_bytes();
    if bytes.first() == Some(&b'\t') {
        return true;
    }
    let mut spaces = 0usize;
    for &byte in bytes {
        if byte == b' ' {
            spaces += 1;
            if spaces >= 2 {
                return true;
            }
            continue;
        }
        break;
    }
    false
}

fn looks_like_table_row(line: &str) -> bool {
    line.contains('|') && !line.trim_matches('|').trim().is_empty()
}

fn is_table_separator_line(line: &str) -> bool {
    if !line.contains('|') {
        return false;
    }
    let mut saw_cell = false;
    for cell in line.trim().trim_matches('|').split('|') {
        let trimmed = cell.trim();
        if trimmed.is_empty() {
            continue;
        }
        saw_cell = true;
        let core = trimmed.trim_matches(':');
        if core.len() < 3 || !core.chars().all(|ch| ch == '-') {
            return false;
        }
    }
    saw_cell
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paragraph_and_heading_split_into_stable_blocks() {
        let mut live = LiveMarkdown::new(MarkdownLane::Assistant);
        live.append("Intro line\n\n## Heading\n");

        assert_eq!(
            live.committed_blocks(),
            &["Intro line".to_string(), "## Heading".to_string()]
        );
        assert_eq!(live.pending_raw_text_for_test(), None);
        assert_eq!(
            live.normalized_text().as_deref(),
            Some("Intro line\n\n## Heading")
        );
    }

    #[test]
    fn line_split_across_chunks_stays_pending_until_newline() {
        let mut live = LiveMarkdown::new(MarkdownLane::Assistant);
        live.append("hello");
        assert_eq!(live.committed_blocks(), &[] as &[String]);
        assert_eq!(live.pending_raw_text_for_test().as_deref(), Some("hello"));

        live.append(" world\n\nnext");
        assert_eq!(live.committed_blocks(), &["hello world".to_string()]);
        assert_eq!(live.pending_raw_text_for_test().as_deref(), Some("next"));
    }

    #[test]
    fn unclosed_code_fence_gets_closed_for_pending_display_only() {
        let mut live = LiveMarkdown::new(MarkdownLane::Assistant);
        live.append("```rust\nfn main() {}\n");

        assert_eq!(
            live.pending_raw_text_for_test().as_deref(),
            Some("```rust\nfn main() {}")
        );
        assert_eq!(
            live.pending_display_text().as_deref(),
            Some("```rust\nfn main() {}\n```")
        );
    }

    #[test]
    fn list_blank_separator_is_not_kept_when_list_ends() {
        let mut live = LiveMarkdown::new(MarkdownLane::Assistant);
        live.append("- one\n- two\n\nNext paragraph\n");

        assert_eq!(live.committed_blocks(), &["- one\n- two".to_string()]);
        assert_eq!(
            live.pending_raw_text_for_test().as_deref(),
            Some("Next paragraph")
        );
        assert_eq!(
            live.normalized_text().as_deref(),
            Some("- one\n- two\n\nNext paragraph")
        );
    }
}
