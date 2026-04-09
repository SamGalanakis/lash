use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use lash::{PromptPanel, PromptRequest, PromptResponse, PromptSelectionMode};

use crate::session_log::SessionInfo;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PromptFocus {
    #[default]
    Options,
    Text,
}

#[derive(Debug)]
pub struct PromptState {
    pub request: PromptRequest,
    pub focus: PromptFocus,
    pub cursor: usize,
    pub scroll_offset: usize,
    pub selected: BTreeSet<usize>,
    pub reply_text: String,
    pub reply_cursor: usize,
    pub response_tx: std::sync::mpsc::Sender<PromptResponse>,
}

impl PromptState {
    pub fn has_options(&self) -> bool {
        !self.request.options.is_empty()
    }

    pub fn is_freeform(&self) -> bool {
        self.request.is_freeform()
    }

    pub fn supports_note(&self) -> bool {
        self.request.allows_note()
    }

    pub fn is_multi(&self) -> bool {
        self.has_options() && matches!(self.request.selection_mode, PromptSelectionMode::Multi)
    }

    pub fn is_text_entry(&self) -> bool {
        self.is_freeform() || (self.supports_note() && matches!(self.focus, PromptFocus::Text))
    }

    pub fn shows_text_input(&self) -> bool {
        self.is_freeform() || self.supports_note()
    }

    pub fn selected_option_idx(&self) -> Option<usize> {
        if self.has_options() && self.cursor < self.request.options.len() {
            Some(self.cursor)
        } else {
            None
        }
    }

    pub fn option_label(&self, idx: usize) -> Option<&str> {
        self.request.options.get(idx).map(String::as_str)
    }

    pub fn option_marked(&self, idx: usize) -> bool {
        if self.is_multi() {
            self.selected.contains(&idx)
        } else {
            self.selected_option_idx() == Some(idx)
        }
    }

    pub fn move_up(&mut self) {
        if self.has_options() {
            self.cursor = self.cursor.saturating_sub(1);
        }
    }

    pub fn move_down(&mut self) {
        if self.has_options() {
            self.cursor = (self.cursor + 1).min(self.request.options.len().saturating_sub(1));
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }

    pub fn scroll_down(&mut self, amount: usize, max_scroll: usize) {
        self.scroll_offset = self.scroll_offset.saturating_add(amount).min(max_scroll);
    }

    pub fn toggle_current(&mut self) {
        if !self.is_multi() {
            return;
        }
        if !self.selected.insert(self.cursor) {
            self.selected.remove(&self.cursor);
        }
    }

    pub fn toggle_text_focus(&mut self) {
        if !self.supports_note() {
            return;
        }
        self.focus = match self.focus {
            PromptFocus::Options => PromptFocus::Text,
            PromptFocus::Text => PromptFocus::Options,
        };
    }

    pub fn insert_text(&mut self, text: &str) {
        self.reply_text.insert_str(self.reply_cursor, text);
        self.reply_cursor += text.len();
    }

    pub fn backspace(&mut self) {
        if self.reply_cursor == 0 {
            return;
        }
        let prev = self.reply_text[..self.reply_cursor]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.reply_text.drain(prev..self.reply_cursor);
        self.reply_cursor = prev;
    }

    fn response_note(&self) -> Option<String> {
        if !self.supports_note() || self.reply_text.trim().is_empty() {
            return None;
        }
        Some(self.reply_text.clone())
    }

    fn format_note_display(base: String, note: Option<&str>) -> String {
        let Some(note) = note.filter(|note| !note.trim().is_empty()) else {
            return base;
        };
        if base.trim().is_empty() {
            format!("Note: {note}")
        } else {
            format!("{base}\n\nNote: {note}")
        }
    }

    pub fn submitted_response(&self) -> PromptResponse {
        if self.is_freeform() {
            return PromptResponse::Text {
                text: self.reply_text.clone(),
            };
        }

        let note = self.response_note();
        if self.is_multi() {
            let selections = self
                .selected
                .iter()
                .filter_map(|idx| self.option_label(*idx).map(str::to_string))
                .collect();
            return PromptResponse::Multi { selections, note };
        }

        let selection = self
            .selected_option_idx()
            .and_then(|idx| self.option_label(idx))
            .unwrap_or_default()
            .to_string();
        PromptResponse::Single { selection, note }
    }

    pub fn dismissed_response(&self) -> PromptResponse {
        self.request.empty_response()
    }

    pub fn display_response(&self, response: &PromptResponse) -> String {
        match response {
            PromptResponse::Text { text } => text.clone(),
            PromptResponse::Single { selection, note } => {
                let base = self
                    .request
                    .options
                    .iter()
                    .position(|option| option == selection)
                    .map(|idx| format!("{}. {}", idx + 1, selection))
                    .unwrap_or_else(|| selection.clone());
                Self::format_note_display(base, note.as_deref())
            }
            PromptResponse::Multi { selections, note } => {
                let base = if selections.is_empty() {
                    String::new()
                } else {
                    selections
                        .iter()
                        .filter_map(|selection| {
                            self.request
                                .options
                                .iter()
                                .position(|option| option == selection)
                                .map(|idx| format!("{}. {}", idx + 1, selection))
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Self::format_note_display(base, note.as_deref())
            }
        }
    }
}

#[derive(Debug)]
pub struct WaitState {
    pub question: String,
    pub panel: Option<PromptPanel>,
    pub seconds: u64,
    deadline: Instant,
    pub response_tx: std::sync::mpsc::Sender<PromptResponse>,
}

impl WaitState {
    pub fn from_request(
        request: PromptRequest,
        response_tx: std::sync::mpsc::Sender<PromptResponse>,
    ) -> Self {
        let seconds = request
            .wait
            .as_ref()
            .map(|wait| wait.seconds)
            .expect("wait state requires PromptRequest::wait");
        let now = Instant::now();
        Self {
            question: request.question,
            panel: request.panel,
            seconds,
            deadline: now + Duration::from_secs(seconds),
            response_tx,
        }
    }

    pub fn remaining_seconds(&self) -> u64 {
        let now = Instant::now();
        if self.deadline <= now {
            return 0;
        }
        self.deadline.duration_since(now).as_millis().div_ceil(1000) as u64
    }

    pub fn timed_out(&self) -> bool {
        self.deadline <= Instant::now()
    }

    pub fn resume_response(&self) -> PromptResponse {
        PromptResponse::Text {
            text: lash::WAIT_PROMPT_RESUME_EARLY_TOKEN.to_string(),
        }
    }

    pub fn timeout_response(&self) -> PromptResponse {
        PromptResponse::Text {
            text: lash::WAIT_PROMPT_TIMEOUT_TOKEN.to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PickerState<T> {
    pub items: Vec<T>,
    pub selected: usize,
}

impl<T> PickerState<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self { items, selected: 0 }
    }

    pub fn up(&mut self) {
        if !self.items.is_empty() {
            self.selected = self.selected.saturating_sub(1);
        }
    }

    pub fn down(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1).min(self.items.len() - 1);
        }
    }

    pub fn take_selected(&mut self) -> Option<T> {
        if self.items.is_empty() {
            return None;
        }
        let idx = self.selected.min(self.items.len() - 1);
        Some(self.items.remove(idx))
    }
}

#[derive(Debug)]
pub enum OverlayState {
    SessionPicker(PickerState<SessionInfo>),
    SkillPicker(PickerState<(String, String)>),
    Prompt(PromptState),
    Wait(WaitState),
}
