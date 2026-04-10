use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use lash::SkillCatalog;
use lash_ui::UiExtensions;

use crate::command;
use crate::input_items::image_marker_ranges;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LargePaste {
    pub placeholder: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingImage {
    pub id: usize,
    pub png_bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuggestionKind {
    None,
    Command,
    Path,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputSelection {
    pub anchor: usize,
    pub end: usize,
    pub active: bool,
    pub visible: bool,
}

#[derive(Clone, Debug)]
pub struct EditorState {
    pub input: String,
    pub cursor_pos: usize,
    pub selection: InputSelection,
    pub input_history: Vec<String>,
    pub input_history_idx: Option<usize>,
    pub suggestions: Vec<(String, String)>,
    pub suggestion_idx: usize,
    pub suggestion_kind: SuggestionKind,
    pub pending_images: Vec<PendingImage>,
    pub inflight_image_ids: HashSet<usize>,
    pub pending_large_pastes: Vec<LargePaste>,
    pub large_paste_counters: HashMap<usize, usize>,
}

pub(crate) const LARGE_PASTE_CHAR_THRESHOLD: usize = 1000;

impl Default for EditorState {
    fn default() -> Self {
        Self {
            input: String::new(),
            cursor_pos: 0,
            selection: InputSelection::default(),
            input_history: Vec::new(),
            input_history_idx: None,
            suggestions: Vec::new(),
            suggestion_idx: 0,
            suggestion_kind: SuggestionKind::None,
            pending_images: Vec::new(),
            inflight_image_ids: HashSet::new(),
            pending_large_pastes: Vec::new(),
            large_paste_counters: HashMap::new(),
        }
    }
}

impl EditorState {
    /// Find the byte offset within `line` that corresponds to a given display column.
    /// If the target column exceeds the line's display width, returns line.len().
    pub(crate) fn byte_pos_at_display_col(line: &str, target_col: usize) -> usize {
        let mut col = 0;
        for (byte_idx, ch) in line.char_indices() {
            if col >= target_col {
                return byte_idx;
            }
            col += unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        }
        line.len()
    }

    pub fn take_input(&mut self) -> String {
        let text = self.input.clone();
        if !text.is_empty() {
            self.input_history.push(text.clone());
        }
        self.input.clear();
        self.cursor_pos = 0;
        self.clear_selection();
        self.input_history_idx = None;
        text
    }

    pub fn take_pending_images(&mut self) -> Vec<PendingImage> {
        self.inflight_image_ids.clear();
        std::mem::take(&mut self.pending_images)
    }

    pub fn take_large_pastes(&mut self) -> Vec<LargePaste> {
        std::mem::take(&mut self.pending_large_pastes)
    }

    pub fn restore_turn(
        &mut self,
        text: String,
        images: Vec<PendingImage>,
        large_pastes: Vec<LargePaste>,
    ) {
        self.input = text;
        self.cursor_pos = self.input.len();
        self.input_history_idx = None;
        self.clear_selection();
        self.pending_images = images;
        self.inflight_image_ids.clear();
        self.pending_large_pastes = large_pastes;
        self.suggestions.clear();
        self.suggestion_idx = 0;
        self.suggestion_kind = SuggestionKind::None;
    }

    fn ordered_selection_bounds(&self) -> (usize, usize) {
        if self.selection.anchor <= self.selection.end {
            (self.selection.anchor, self.selection.end)
        } else {
            (self.selection.end, self.selection.anchor)
        }
    }

    pub fn selected_range(&self) -> Option<std::ops::Range<usize>> {
        let (start, end) = self.ordered_selection_bounds();
        ((self.selection.visible || self.selection.active) && start < end).then_some(start..end)
    }

    pub fn has_selection(&self) -> bool {
        self.selected_range().is_some()
    }

    pub fn selection_is_active(&self) -> bool {
        self.selection.active
    }

    pub fn clear_selection(&mut self) {
        self.selection = InputSelection::default();
    }

    pub fn start_selection(&mut self, offset: usize) {
        let clamped = offset.min(self.input.len());
        self.selection = InputSelection {
            anchor: clamped,
            end: clamped,
            active: true,
            visible: false,
        };
        self.cursor_pos = clamped;
    }

    pub fn update_selection(&mut self, offset: usize) {
        let clamped = offset.min(self.input.len());
        self.selection.end = clamped;
        self.selection.visible = self.selection.anchor != clamped;
        self.cursor_pos = clamped;
    }

    pub fn finish_selection(&mut self) {
        self.selection.active = false;
        if self.selection.anchor == self.selection.end {
            self.selection.visible = false;
        }
    }

    pub fn selected_text(&self) -> Option<String> {
        self.selected_range()
            .map(|range| self.input[range].to_string())
    }

    pub fn cursor_line(&self) -> usize {
        self.input[..self.cursor_pos].matches('\n').count()
    }

    pub fn line_count(&self) -> usize {
        self.input.split('\n').count()
    }

    pub fn history_up(&mut self) {
        if self.cursor_line() > 0 {
            self.move_cursor_up_line();
            return;
        }
        if self.input_history.is_empty() {
            return;
        }
        let idx = match self.input_history_idx {
            None => self.input_history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.input_history_idx = Some(idx);
        self.input = self.input_history[idx].clone();
        self.cursor_pos = self.input.len();
        self.clear_selection();
    }

    pub fn history_down(&mut self) {
        if self.cursor_line() < self.line_count().saturating_sub(1) {
            self.move_cursor_down_line();
            return;
        }
        match self.input_history_idx {
            None => {}
            Some(i) if i + 1 >= self.input_history.len() => {
                self.input_history_idx = None;
                self.input.clear();
                self.cursor_pos = 0;
                self.clear_selection();
            }
            Some(i) => {
                let idx = i + 1;
                self.input_history_idx = Some(idx);
                self.input = self.input_history[idx].clone();
                self.cursor_pos = self.input.len();
                self.clear_selection();
            }
        }
    }

    fn move_cursor_up_line(&mut self) {
        let before = &self.input[..self.cursor_pos];
        let cur_line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        if cur_line_start == 0 {
            return;
        }
        let cur_text = &self.input[cur_line_start..self.cursor_pos];
        let display_col = unicode_width::UnicodeWidthStr::width(cur_text);
        let prev_content = &self.input[..cur_line_start - 1];
        let prev_line_start = prev_content.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let prev_line = &self.input[prev_line_start..cur_line_start - 1];
        self.cursor_pos = Self::byte_pos_at_display_col(prev_line, display_col) + prev_line_start;
    }

    fn move_cursor_down_line(&mut self) {
        let before = &self.input[..self.cursor_pos];
        let cur_line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let cur_text = &self.input[cur_line_start..self.cursor_pos];
        let display_col = unicode_width::UnicodeWidthStr::width(cur_text);
        let after = &self.input[self.cursor_pos..];
        let newline_offset = match after.find('\n') {
            Some(i) => i,
            None => return,
        };
        let next_line_start = self.cursor_pos + newline_offset + 1;
        let next_after = &self.input[next_line_start..];
        let next_line = match next_after.find('\n') {
            Some(end) => &next_after[..end],
            None => next_after,
        };
        self.cursor_pos = Self::byte_pos_at_display_col(next_line, display_col) + next_line_start;
    }

    fn large_paste_ranges(&self) -> Vec<(usize, std::ops::Range<usize>)> {
        self.pending_large_pastes
            .iter()
            .enumerate()
            .filter_map(|(idx, paste)| {
                self.input
                    .match_indices(&paste.placeholder)
                    .next()
                    .map(|(start, _)| (idx, start..start + paste.placeholder.len()))
            })
            .collect()
    }

    fn ranges_overlap(a: &std::ops::Range<usize>, b: &std::ops::Range<usize>) -> bool {
        a.start < b.end && b.start < a.end
    }

    fn expand_range_to_attachment_boundaries(
        &self,
        mut range: std::ops::Range<usize>,
    ) -> std::ops::Range<usize> {
        loop {
            let mut expanded = false;
            for (marker_range, _) in image_marker_ranges(&self.input) {
                if Self::ranges_overlap(&range, &marker_range)
                    && (marker_range.start < range.start || marker_range.end > range.end)
                {
                    range.start = range.start.min(marker_range.start);
                    range.end = range.end.max(marker_range.end);
                    expanded = true;
                }
            }
            for (_, marker_range) in self.large_paste_ranges() {
                if Self::ranges_overlap(&range, &marker_range)
                    && (marker_range.start < range.start || marker_range.end > range.end)
                {
                    range.start = range.start.min(marker_range.start);
                    range.end = range.end.max(marker_range.end);
                    expanded = true;
                }
            }
            if !expanded {
                return range;
            }
        }
    }

    fn remove_range_internal(&mut self, range: std::ops::Range<usize>) {
        let range = self.expand_range_to_attachment_boundaries(range);
        let removed_len = range.end.saturating_sub(range.start);
        let removed_image_ids: HashSet<usize> = image_marker_ranges(&self.input)
            .into_iter()
            .filter(|(marker_range, _)| Self::ranges_overlap(&range, marker_range))
            .map(|(_, id)| id)
            .collect();
        let removed_large_paste_idxs: HashSet<usize> = self
            .large_paste_ranges()
            .into_iter()
            .filter(|(_, marker_range)| Self::ranges_overlap(&range, marker_range))
            .map(|(idx, _)| idx)
            .collect();

        self.input.drain(range.clone());
        if self.cursor_pos > range.end {
            self.cursor_pos = self.cursor_pos.saturating_sub(removed_len);
        } else if self.cursor_pos > range.start {
            self.cursor_pos = range.start;
        }
        self.pending_images
            .retain(|image| !removed_image_ids.contains(&image.id));
        for image_id in removed_image_ids {
            self.inflight_image_ids.remove(&image_id);
        }
        self.pending_large_pastes = self
            .pending_large_pastes
            .drain(..)
            .enumerate()
            .filter_map(|(idx, paste)| (!removed_large_paste_idxs.contains(&idx)).then_some(paste))
            .collect();
        self.clear_selection();
    }

    fn delete_selection(&mut self) -> bool {
        let Some(range) = self.selected_range() else {
            return false;
        };
        self.remove_range_internal(range);
        true
    }

    pub fn insert_char(&mut self, c: char) {
        if self.delete_selection() {
            self.input.insert(self.cursor_pos, c);
            self.cursor_pos += c.len_utf8();
            return;
        }
        self.input.insert(self.cursor_pos, c);
        self.cursor_pos += c.len_utf8();
    }

    pub fn insert_text(&mut self, text: &str) {
        if self.delete_selection() {
            self.input.insert_str(self.cursor_pos, text);
            self.cursor_pos += text.len();
            return;
        }
        self.input.insert_str(self.cursor_pos, text);
        self.cursor_pos += text.len();
    }

    fn next_large_paste_placeholder(&mut self, char_count: usize) -> String {
        let base = format!("[Pasted Content {char_count} chars]");
        let next_suffix = self.large_paste_counters.entry(char_count).or_insert(0);
        *next_suffix += 1;
        if *next_suffix == 1 {
            base
        } else {
            format!("{base} #{}", *next_suffix)
        }
    }

    pub fn insert_pasted_text(&mut self, text: &str) {
        let char_count = text.chars().count();
        if char_count > LARGE_PASTE_CHAR_THRESHOLD {
            let placeholder = self.next_large_paste_placeholder(char_count);
            self.pending_large_pastes.push(LargePaste {
                placeholder: placeholder.clone(),
                content: text.to_string(),
            });
            self.insert_text(&placeholder);
        } else {
            self.insert_text(text);
        }
    }

    fn large_paste_range_containing(
        &self,
        probe_pos: usize,
    ) -> Option<(usize, std::ops::Range<usize>)> {
        for (idx, paste) in self.pending_large_pastes.iter().enumerate() {
            if let Some((start, _)) = self.input.match_indices(&paste.placeholder).next() {
                let end = start + paste.placeholder.len();
                if probe_pos >= start && probe_pos < end {
                    return Some((idx, start..end));
                }
            }
        }
        None
    }

    fn remove_large_paste_at_probe(&mut self, probe_pos: usize) -> bool {
        let Some((idx, range)) = self.large_paste_range_containing(probe_pos) else {
            return false;
        };
        self.input.drain(range.clone());
        self.cursor_pos = range.start;
        self.pending_large_pastes.remove(idx);
        true
    }

    fn image_marker_range_containing(
        &self,
        probe_pos: usize,
    ) -> Option<(usize, std::ops::Range<usize>)> {
        image_marker_ranges(&self.input)
            .into_iter()
            .find(|(range, _)| probe_pos >= range.start && probe_pos < range.end)
            .map(|(range, idx)| (idx, range))
    }

    fn image_marker_range_for_id(&self, id: usize) -> Option<std::ops::Range<usize>> {
        image_marker_ranges(&self.input)
            .into_iter()
            .find_map(|(range, idx)| (idx == id).then_some(range))
    }

    fn remove_image_marker_range(
        &mut self,
        image_id: usize,
        range: std::ops::Range<usize>,
    ) -> bool {
        let removed_len = range.end.saturating_sub(range.start);
        self.input.drain(range.clone());
        if self.cursor_pos > range.end {
            self.cursor_pos = self.cursor_pos.saturating_sub(removed_len);
        } else if self.cursor_pos > range.start {
            self.cursor_pos = range.start;
        }
        self.pending_images.retain(|image| image.id != image_id);
        self.inflight_image_ids.remove(&image_id);
        true
    }

    fn remove_image_marker_at_probe(&mut self, probe_pos: usize) -> bool {
        let Some((image_id, range)) = self.image_marker_range_containing(probe_pos) else {
            return false;
        };
        self.remove_image_marker_range(image_id, range)
    }

    pub fn remove_image_marker_by_id(&mut self, image_id: usize) -> bool {
        let Some(range) = self.image_marker_range_for_id(image_id) else {
            return false;
        };
        self.remove_image_marker_range(image_id, range)
    }

    pub fn backspace(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor_pos > 0 && self.remove_image_marker_at_probe(self.cursor_pos - 1) {
            return;
        }
        if self.cursor_pos > 0 && self.remove_large_paste_at_probe(self.cursor_pos - 1) {
            return;
        }
        if self.cursor_pos > 0 {
            let prev = self.input[..self.cursor_pos]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.drain(prev..self.cursor_pos);
            self.cursor_pos = prev;
        }
    }

    pub fn delete(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor_pos < self.input.len() && self.remove_image_marker_at_probe(self.cursor_pos)
        {
            return;
        }
        if self.cursor_pos < self.input.len() && self.remove_large_paste_at_probe(self.cursor_pos) {
            return;
        }
        if self.cursor_pos < self.input.len() {
            let next = self.input[self.cursor_pos..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor_pos + i)
                .unwrap_or(self.input.len());
            self.input.drain(self.cursor_pos..next);
        }
    }

    pub fn move_cursor_left(&mut self) {
        if let Some(range) = self.selected_range() {
            self.cursor_pos = range.start;
            self.clear_selection();
            return;
        }
        if self.cursor_pos > 0 {
            self.cursor_pos = self.input[..self.cursor_pos]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    pub fn move_cursor_right(&mut self) {
        if let Some(range) = self.selected_range() {
            self.cursor_pos = range.end;
            self.clear_selection();
            return;
        }
        if self.cursor_pos < self.input.len() {
            self.cursor_pos = self.input[self.cursor_pos..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor_pos + i)
                .unwrap_or(self.input.len());
        }
    }

    pub fn move_cursor_home(&mut self) {
        if let Some(range) = self.selected_range() {
            self.cursor_pos = range.start;
            self.clear_selection();
        }
        let before = &self.input[..self.cursor_pos];
        self.cursor_pos = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    }

    pub fn move_cursor_end(&mut self) {
        if let Some(range) = self.selected_range() {
            self.cursor_pos = range.end;
            self.clear_selection();
        }
        let after = &self.input[self.cursor_pos..];
        if let Some(pos) = after.find('\n') {
            self.cursor_pos += pos;
        } else {
            self.cursor_pos = self.input.len();
        }
    }

    pub fn load_history(&mut self) {
        let path = lash::lash_home().join("history");
        if let Ok(content) = std::fs::read_to_string(&path) {
            self.input_history = content
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect();
        }
    }

    pub fn save_history(&self) {
        let dir = lash::lash_home();
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("history");
        let start = self.input_history.len().saturating_sub(500);
        let lines: Vec<&str> = self.input_history[start..]
            .iter()
            .map(|s| s.as_str())
            .collect();
        let _ = std::fs::write(&path, lines.join("\n"));
    }

    pub fn update_suggestions(&mut self, skills: &SkillCatalog, ui_extensions: &UiExtensions) {
        if let Some((_slash_pos, prefix)) = self.slash_token_at_cursor() {
            self.suggestions = command::completions(&prefix, skills);
            for completion in ui_extensions.completions(&prefix) {
                if !self
                    .suggestions
                    .iter()
                    .any(|(existing, _)| existing == &completion.0)
                {
                    self.suggestions.push(completion);
                }
            }
            self.suggestion_kind = SuggestionKind::Command;
            if self.suggestions.is_empty() {
                self.suggestion_idx = 0;
            } else {
                self.suggestion_idx = self.suggestion_idx.min(self.suggestions.len() - 1);
            }
            return;
        }
        if let Some((_at_pos, partial)) = self.at_token_at_cursor() {
            self.suggestions = complete_path(&partial);
            self.suggestion_kind = SuggestionKind::Path;
            if self.suggestions.is_empty() {
                self.suggestion_idx = 0;
            } else {
                self.suggestion_idx = self.suggestion_idx.min(self.suggestions.len() - 1);
            }
            return;
        }
        self.suggestions.clear();
        self.suggestion_idx = 0;
        self.suggestion_kind = SuggestionKind::None;
    }

    fn at_token_at_cursor(&self) -> Option<(usize, String)> {
        let before = &self.input[..self.cursor_pos];
        let at_byte = before.rfind('@')?;
        if at_byte > 0 {
            let prev_byte = self.input.as_bytes()[at_byte - 1];
            if !prev_byte.is_ascii_whitespace() {
                return None;
            }
        }
        let partial = &self.input[at_byte + 1..self.cursor_pos];
        if partial.contains(' ') || partial.contains('\n') {
            return None;
        }
        Some((at_byte, partial.to_string()))
    }

    fn slash_token_at_cursor(&self) -> Option<(usize, String)> {
        let before = &self.input[..self.cursor_pos];
        let slash_byte = before.rfind('/')?;
        if slash_byte > 0 {
            let prev_byte = self.input.as_bytes()[slash_byte - 1];
            if !prev_byte.is_ascii_whitespace() {
                return None;
            }
        }
        let prefix = &self.input[slash_byte..self.cursor_pos];
        if prefix.contains(' ') || prefix.contains('\n') {
            return None;
        }
        Some((slash_byte, prefix.to_string()))
    }

    pub fn has_suggestions(&self) -> bool {
        !self.suggestions.is_empty()
    }

    pub fn suggestion_up(&mut self) {
        if !self.suggestions.is_empty() {
            self.suggestion_idx = self.suggestion_idx.saturating_sub(1);
        }
    }

    pub fn suggestion_down(&mut self) {
        if !self.suggestions.is_empty() {
            self.suggestion_idx = (self.suggestion_idx + 1).min(self.suggestions.len() - 1);
        }
    }

    pub fn complete_suggestion(&mut self, skills: &SkillCatalog, ui_extensions: &UiExtensions) {
        match self.suggestion_kind {
            SuggestionKind::Command => {
                if let Some((slash_pos, _prefix)) = self.slash_token_at_cursor()
                    && let Some((cmd, _)) = self.suggestions.get(self.suggestion_idx).cloned()
                {
                    let needs_arg = ui_extensions
                        .command_takes_argument(&cmd)
                        .unwrap_or_else(|| command::completion_inserts_space(&cmd, skills));
                    let replacement = if needs_arg { format!("{} ", cmd) } else { cmd };
                    let before = self.input[..slash_pos].to_string();
                    let after = self.input[self.cursor_pos..].to_string();
                    self.input = format!("{}{}{}", before, replacement, after);
                    self.cursor_pos = slash_pos + replacement.len();
                }
                self.suggestions.clear();
                self.suggestion_idx = 0;
                self.suggestion_kind = SuggestionKind::None;
            }
            SuggestionKind::Path => {
                if let Some((at_pos, _partial)) = self.at_token_at_cursor()
                    && let Some((path, _)) = self.suggestions.get(self.suggestion_idx).cloned()
                {
                    let before = self.input[..at_pos].to_string();
                    let after = self.input[self.cursor_pos..].to_string();
                    let is_dir = path.ends_with('/');
                    self.input = format!("{}@{}{}", before, path, after);
                    self.cursor_pos = at_pos + 1 + path.len();
                    if is_dir {
                        return;
                    }
                }
                self.suggestions.clear();
                self.suggestion_idx = 0;
                self.suggestion_kind = SuggestionKind::None;
            }
            SuggestionKind::None => {}
        }
    }

    pub fn next_image_marker_id(&self) -> usize {
        let max_pending = self
            .pending_images
            .iter()
            .map(|image| image.id)
            .max()
            .unwrap_or(0);
        let max_inline = image_marker_ranges(&self.input)
            .into_iter()
            .map(|(_, idx)| idx)
            .max()
            .unwrap_or(0);
        max_pending.max(max_inline) + 1
    }

    pub fn begin_pending_image(&mut self, id: usize) {
        self.inflight_image_ids.insert(id);
    }

    pub fn has_pending_image_jobs(&self) -> bool {
        image_marker_ranges(&self.input)
            .into_iter()
            .any(|(_, idx)| self.inflight_image_ids.contains(&idx))
    }

    pub fn complete_pending_image(&mut self, id: usize, png_bytes: Vec<u8>) -> bool {
        self.inflight_image_ids.remove(&id);
        if !image_marker_ranges(&self.input)
            .into_iter()
            .any(|(_, idx)| idx == id)
        {
            return false;
        }
        if self.pending_images.iter().all(|image| image.id != id) {
            self.pending_images.push(PendingImage { id, png_bytes });
        }
        true
    }

    pub fn fail_pending_image(&mut self, id: usize) -> bool {
        self.inflight_image_ids.remove(&id);
        self.remove_image_marker_by_id(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_text_replaces_visible_selection() {
        let mut editor = EditorState::default();
        editor.input = "alpha beta".to_string();
        editor.cursor_pos = editor.input.len();
        editor.start_selection(6);
        editor.update_selection(10);
        editor.finish_selection();

        editor.insert_text("gamma");

        assert_eq!(editor.input, "alpha gamma");
        assert_eq!(editor.cursor_pos, "alpha gamma".len());
        assert!(!editor.has_selection());
    }

    #[test]
    fn backspace_removes_selected_marker_placeholder() {
        let mut editor = EditorState::default();
        editor.input = "before [Image #2] after".to_string();
        editor.pending_images.push(PendingImage {
            id: 2,
            png_bytes: vec![1, 2, 3],
        });
        editor.cursor_pos = editor.input.len();
        editor.start_selection(8);
        editor.update_selection(12);
        editor.finish_selection();

        editor.backspace();

        assert_eq!(editor.input, "before  after");
        assert!(editor.pending_images.is_empty());
        assert!(!editor.has_selection());
    }
}

fn complete_path(partial: &str) -> Vec<(String, String)> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let (dir, prefix) = if partial.is_empty() {
        (cwd.clone(), String::new())
    } else if partial.ends_with('/') {
        let dir = if partial.starts_with('/') {
            PathBuf::from(partial)
        } else {
            cwd.join(partial)
        };
        (dir, String::new())
    } else {
        let path = if partial.starts_with('/') {
            PathBuf::from(partial)
        } else {
            cwd.join(partial)
        };
        let parent = path.parent().unwrap_or(&cwd).to_path_buf();
        let prefix = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        (parent, prefix)
    };

    let entries = match std::fs::read_dir(&dir) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };

    let show_hidden = prefix.starts_with('.');
    let mut dirs: Vec<(String, String)> = Vec::new();
    let mut files: Vec<(String, String)> = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !show_hidden && name.starts_with('.') {
            continue;
        }
        if !prefix.is_empty() && !name.starts_with(&prefix) {
            continue;
        }
        let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
        let dir_part = if partial.is_empty() {
            String::new()
        } else if partial.ends_with('/') {
            partial.to_string()
        } else if let Some(slash) = partial.rfind('/') {
            partial[..=slash].to_string()
        } else {
            String::new()
        };

        if is_dir {
            dirs.push((format!("{}{}/", dir_part, name), "dir".to_string()));
        } else {
            files.push((format!("{}{}", dir_part, name), "file".to_string()));
        }
    }

    dirs.sort_by(|a, b| a.0.cmp(&b.0));
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut result = dirs;
    result.extend(files);
    result.truncate(20);
    result
}
