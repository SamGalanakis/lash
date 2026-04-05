use crate::session_log::SessionInfo;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptSelection {
    Option(usize),
    CustomReply,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptFocus {
    Selection,
    ReplyEditor,
}

#[derive(Debug)]
pub struct PromptState {
    pub question: String,
    pub options: Vec<String>,
    pub selection: PromptSelection,
    pub focus: PromptFocus,
    pub reply_text: String,
    pub reply_cursor: usize,
    pub response_tx: std::sync::mpsc::Sender<String>,
}

impl PromptState {
    pub fn has_options(&self) -> bool {
        !self.options.is_empty()
    }

    pub fn is_freeform(&self) -> bool {
        self.options.is_empty()
    }

    pub fn is_editing_reply(&self) -> bool {
        self.is_freeform() || self.focus == PromptFocus::ReplyEditor
    }

    pub fn selected_option_idx(&self) -> Option<usize> {
        match self.selection {
            PromptSelection::Option(idx) if idx < self.options.len() => Some(idx),
            _ => None,
        }
    }

    pub fn selects_custom_reply(&self) -> bool {
        self.selection == PromptSelection::CustomReply
    }

    pub fn move_up(&mut self) {
        if !self.has_options() {
            return;
        }
        self.selection = match self.selection {
            PromptSelection::Option(idx) => PromptSelection::Option(idx.saturating_sub(1)),
            PromptSelection::CustomReply => {
                PromptSelection::Option(self.options.len().saturating_sub(1))
            }
        };
    }

    pub fn move_down(&mut self) {
        if !self.has_options() {
            return;
        }
        self.selection = match self.selection {
            PromptSelection::Option(idx) if idx + 1 < self.options.len() => {
                PromptSelection::Option(idx + 1)
            }
            _ => PromptSelection::CustomReply,
        };
    }

    pub fn toggle_focus(&mut self) {
        if !self.has_options() {
            self.focus = PromptFocus::ReplyEditor;
            return;
        }
        self.focus = match self.focus {
            PromptFocus::Selection => PromptFocus::ReplyEditor,
            PromptFocus::ReplyEditor => PromptFocus::Selection,
        };
    }

    pub fn insert_char(&mut self, c: char) {
        self.reply_text.insert(self.reply_cursor, c);
        self.reply_cursor += c.len_utf8();
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

    pub fn submitted_response(&self) -> String {
        if self.is_freeform() || self.selects_custom_reply() {
            return self.reply_text.clone();
        }

        let Some(idx) = self.selected_option_idx() else {
            return self.reply_text.clone();
        };
        let label = &self.options[idx];
        let truncated: String = label.chars().take(40).collect();
        let suffix = if label.chars().count() > 40 {
            "..."
        } else {
            ""
        };
        let base = format!("{}. {}{}", idx + 1, truncated, suffix);
        if self.reply_text.is_empty() {
            base
        } else {
            format!("{}\n\n{}", base, self.reply_text)
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
}
